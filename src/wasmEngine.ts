// The hexplorer ui talks to the engine through a small bridge on `window`, which
// on Android is implemented in Java over a child process. This is the same bridge
// backed by the engine compiled to WebAssembly, so the page needs nothing else.

import type { EvalStats } from './netRunner';
import type { FromEngine, ToEngine } from './engineWorker';

export type NativeBridge = {
  start(boardSize: number): void;
  query(json: string): void;
  running(): boolean;
  stop(): void;
  save(filename: string, mimeType: string, content: string): void;
};

type Host = {
  Native?: NativeBridge;
  onEngineLine?: (line: string) => void;
  onEngineLog?: (line: string) => void;
  /** Only the wasm engine has this: the phone's takes no measurable time to start. */
  onEngineProgress?: (text: string) => void;
};

export type WasmEngineOptions = {
  /** Where the engine and the net are served from. */
  enginePath?: string;
  /** The weights, half precision, for the GPU. */
  modelPath?: string;
  /** The same net with the weights zeroed, which is all the engine reads. */
  shapePath?: string;
  /** Half precision: twice the speed, and the error the native engine also has. */
  half?: boolean;
  /**
   * Threads searching one position. One: threads were only ever a way of
   * holding evals in flight, a blocked thread holds exactly one, and
   * leafEvals now sets the in-flight count directly. One thread with 64 in
   * flight measures faster than sixteen threads with two each.
   */
  searchThreads?: number;
  /**
   * How long the net's serving thread lets a partial batch keep filling, in
   * microseconds. Without it batches average half the thread count: the serve
   * loop takes what is queued the instant it wakes, and a batch's stragglers
   * are a few milliseconds behind it. Waiting for them is worth ~12%.
   */
  batchWaitMicros?: number;
  /**
   * Engine threads serving the net, each with an evaluation in flight. Two
   * keep the GPU fed back to back: it runs one batch while the other is
   * packed, submitted and read back.
   */
  serverThreads?: number;
  /**
   * Leaf evaluations each search thread may have in flight at once. A thread
   * that reaches a new leaf queues the eval and starts another playout
   * instead of blocking, so one thread alone keeps the GPU in ~50-row
   * batches. Sweeping one thread's in-flight count: 32 gives 168 visits/s,
   * 48 gives 180, 64 gives 188, 96 falls back to 182 -- every pending
   * playout holds virtual losses along its path, and past 64 that distorts
   * move selection by more than the rows buy.
   */
  leafEvals?: number;
  /** Time every GPU dispatch (slower; see webgpuModel.profile). */
  profile?: boolean;
  onStats?: (stats: EvalStats) => void;
};

/** Resolved against the page, so a build serves from a subpath as well as a root. */
const asset = (path: string) => new URL(path, document.baseURI).href;

export function installWasmEngine(options: WasmEngineOptions = {}): void {
  const {
    enginePath = asset('katahex.js'),
    modelPath = asset('net-fp16.bin.gz'),
    shapePath = asset('net-shape.bin.gz'),
    half = true,
    searchThreads = 1,
    batchWaitMicros = 3000,
    serverThreads = 2,
    leafEvals = 64,
    profile = false,
    onStats,
  } = options;

  const host = globalThis as Host;
  let worker: Worker | null = null;

  const send = (message: ToEngine) => worker?.postMessage(message);

  host.Native = {
    /** The net buffers are sized from the board size, so a change is a restart. */
    start(boardSize) {
      worker?.terminate();
      worker = new Worker(new URL('./engineWorker.ts', import.meta.url), { type: 'module' });
      worker.onmessage = ({ data }: MessageEvent<FromEngine>) => {
        if (data.kind === 'line') host.onEngineLine?.(data.line);
        else if (data.kind === 'log') host.onEngineLog?.(data.line);
        else if (data.kind === 'progress') host.onEngineProgress?.(data.text);
        else if (data.kind === 'stats') onStats?.(data.stats);
        else host.onEngineLog?.(`engine error: ${data.message}`);
      };
      send({ kind: 'start', boardSize, half, searchThreads, batchWaitMicros, serverThreads, leafEvals, profile, enginePath, modelPath, shapePath });
    },

    query(json) {
      send({ kind: 'query', json });
    },

    running() {
      return worker !== null;
    },

    stop() {
      worker?.terminate();
      worker = null;
    },

    /** Only reached because the bridge exists; it is the plain browser download. */
    save(filename, mimeType, content) {
      const link = document.createElement('a');
      link.href = URL.createObjectURL(new Blob([content], { type: mimeType }));
      link.download = filename;
      link.click();
      URL.revokeObjectURL(link.href);
    },
  };
}
