// One KataHex engine in a worker, as a request/response service.
//
// The worker holds the wasm engine and the net together (src/engineWorker.ts),
// and `half` picks the precision the net runs at, which is what a match here
// varies. Several queries can be in flight: they are what fills the net's
// batches, and a lone evaluation costs many times its share of a full one.

import type { FromEngine, ToEngine } from './engineWorker';
import type { EvalStats } from './netRunner';

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
  private stats: EvalStats = { evals: 0, rows: 0, nanos: 0 };

  constructor(
    readonly precision: Precision,
    private size: number,
    private onLog: (line: string) => void = () => {},
    // Named rather than left to the worker's default: thread count sets the
    // batch size the search produces, and half precision is worth much more at
    // a small batch than a large one, so the two arms of a match have to agree
    // on it and a result has to record it.
    readonly searchThreads = 16,
    // Waiting a moment before serving a batch is what keeps it full: without it
    // a returning batch's threads re-queue behind the ones already waiting and
    // the two groups alternate at half the thread count forever. It changes the
    // batch size, and so what half precision is worth, so a match names it.
    readonly batchWaitMicros = 3000,
  ) {
    this.worker = new Worker(new URL('./engineWorker.ts', import.meta.url), { type: 'module' });
    let ready!: () => void;
    this.started = new Promise((resolve) => { ready = resolve; });

    this.worker.onmessage = ({ data }: MessageEvent<FromEngine>) => {
      if (data.kind === 'line') {
        const reply = JSON.parse(data.line) as Reply;
        this.pending.get(reply.id)?.(reply);
        this.pending.delete(reply.id);
      } else if (data.kind === 'stats') {
        this.stats = data.stats;
      } else if (data.kind === 'log') {
        if (data.line.includes('ready to begin handling requests')) ready();
      } else if (data.kind === 'error') {
        this.onLog(`${precision} engine error: ${data.message}`);
      }
    };

    const start: ToEngine = {
      kind: 'start', boardSize: size, half: precision === 'fp16',
      searchThreads: this.searchThreads,
      batchWaitMicros: this.batchWaitMicros,
      profile: false,
      enginePath: '/katahex.js', modelPath: '/hex27x3.bin.gz',
    };
    this.worker.postMessage(start);
  }

  /** Resolves when the engine has logged that it is up; queries before it are queued anyway. */
  ready(): Promise<void> { return this.started; }

  /**
   * Net evaluations served so far, which is the honest measure of how much work
   * a search did. Visits are not: a move can take thousands of them walking a
   * subtree the nnCache already holds, and those never reach the GPU. Two
   * identical engines on the same clock came out at 220 and 141 visits a move
   * on that alone.
   */
  get rows(): number { return this.stats.rows; }

  /**
   * Drops the engine and the net with it. Closing the browser on top of live
   * workers leaves the whole process tree behind, still holding a WebGPU device,
   * for the next run to measure itself against.
   */
  stop(): void { this.worker.terminate(); }

  analyse(
    moves: string[],
    condition: Condition,
    overrides: Record<string, unknown> = {},
  ): Promise<Reply> {
    const id = `${this.precision}-${++this.seq}`;
    const limits =
      condition.kind === 'policy' ? { maxVisits: 1, includePolicy: true } :
      condition.kind === 'visits' ? { maxVisits: condition.visits } :
      // A time limit needs a visit limit it will never reach: whichever comes
      // first ends the search.
      { maxVisits: 1e9 };
    const overrideSettings = {
      ...(condition.kind === 'time' ? { maxTime: condition.seconds } : {}),
      ...overrides,
    };

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
        ...limits,
        ...(Object.keys(overrideSettings).length ? { overrideSettings } : {}),
      }),
    };

    return new Promise((resolve) => {
      this.pending.set(id, resolve);
      this.worker.postMessage(query);
    });
  }
}
