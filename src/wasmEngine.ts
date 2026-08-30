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
};

export type WasmEngineOptions = {
  /** Where the engine and the net are served from. */
  enginePath?: string;
  modelPath?: string;
  /** Half precision: twice the speed, and the error the native engine also has. */
  half?: boolean;
  /**
   * Threads searching one position. They are what fills the net's batches: 8
   * threads search at 55 visits/s, 16 at 65, and 32 no faster than 16. Each one
   * is a worker.
   */
  searchThreads?: number;
  onStats?: (stats: EvalStats) => void;
};

export function installWasmEngine(options: WasmEngineOptions = {}): void {
  const {
    enginePath = '/katahex.js',
    modelPath = '/hex27x3.bin.gz',
    half = true,
    searchThreads = 16,
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
        else if (data.kind === 'stats') onStats?.(data.stats);
        else host.onEngineLog?.(`engine error: ${data.message}`);
      };
      send({ kind: 'start', boardSize, half, searchThreads, enginePath, modelPath });
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
