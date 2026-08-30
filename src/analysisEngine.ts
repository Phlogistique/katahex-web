// One KataHex engine in a worker, as a request/response service.
//
// The worker holds the wasm engine and the net together (src/engineWorker.ts),
// and `half` picks the precision the net runs at, which is what a match here
// varies. Several queries can be in flight: they are what fills the net's
// batches, and a lone evaluation costs many times its share of a full one.

import type { FromEngine, ToEngine } from './engineWorker';

export type Precision = 'fp16' | 'fp32';

/** How much thinking a move gets. */
export type Condition =
  | { kind: 'policy' }
  | { kind: 'visits'; visits: number }
  | { kind: 'time'; seconds: number };

export type Reply = {
  id: string;
  error?: string;
  rootInfo?: { winrate: number; visits: number };
  moveInfos?: { move: string; order: number; visits: number; winrate: number }[];
  policy?: number[];
};

export class AnalysisEngine {
  private worker: Worker;
  private pending = new Map<string, (reply: Reply) => void>();
  private seq = 0;
  private started: Promise<void>;

  constructor(
    readonly precision: Precision,
    private size: number,
    private onLog: (line: string) => void = () => {},
  ) {
    this.worker = new Worker(new URL('./engineWorker.ts', import.meta.url), { type: 'module' });
    let ready!: () => void;
    this.started = new Promise((resolve) => { ready = resolve; });

    this.worker.onmessage = ({ data }: MessageEvent<FromEngine>) => {
      if (data.kind === 'line') {
        const reply = JSON.parse(data.line) as Reply;
        this.pending.get(reply.id)?.(reply);
        this.pending.delete(reply.id);
      } else if (data.kind === 'log') {
        if (data.line.includes('ready to begin handling requests')) ready();
      } else if (data.kind === 'error') {
        this.onLog(`${precision} engine error: ${data.message}`);
      }
    };

    const start: ToEngine = {
      kind: 'start', boardSize: size, half: precision === 'fp16',
      enginePath: '/katahex.js', modelPath: '/hex27x3.bin.gz',
    };
    this.worker.postMessage(start);
  }

  /** Resolves when the engine has logged that it is up; queries before it are queued anyway. */
  ready(): Promise<void> { return this.started; }

  analyse(moves: string[], condition: Condition): Promise<Reply> {
    const id = `${this.precision}-${++this.seq}`;
    const search =
      condition.kind === 'policy' ? { maxVisits: 1, includePolicy: true } :
      condition.kind === 'visits' ? { maxVisits: condition.visits } :
      // A time limit needs a visit limit it will never reach: whichever comes
      // first ends the search.
      { maxVisits: 1e9, overrideSettings: { maxTime: condition.seconds } };

    const query: ToEngine = {
      kind: 'query',
      json: JSON.stringify({
        id,
        boardXSize: this.size,
        boardYSize: this.size,
        moves: moves.map((move, turn) => [turn % 2 ? 'W' : 'B', move]),
        rules: 'tromp-taylor',
        komi: 0,
        analyzeTurns: [moves.length],
        ...search,
      }),
    };

    return new Promise((resolve) => {
      this.pending.set(id, resolve);
      this.worker.postMessage(query);
    });
  }
}
