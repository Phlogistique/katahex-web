/// <reference lib="webworker" />

// Runs the KataHex analysis engine, compiled to WebAssembly, and the net it asks
// for, both in this worker. Queries arrive as messages and every line the engine
// writes goes back as one.
//
// The engine's main() waits for queries on a thread of its own, so this thread
// stays free: it is the one the engine's other threads route their output
// through, and the one that answers their net evaluations.

import { parseKataGoModelV8 } from '../vendor/loadModelV8';
import { serveEvals, type EvalStats } from './netRunner';
import { KataGoWebGpuModel } from './webgpuModel';

export type ToEngine =
  | { kind: 'start'; boardSize: number; half: boolean; searchThreads: number;
      batchWaitMicros: number; enginePath: string; modelPath: string }
  | { kind: 'query'; json: string };

export type FromEngine =
  | { kind: 'line'; line: string }   // what the engine writes on stdout: analysis results
  | { kind: 'log'; line: string }    // and on stderr: its own log
  | { kind: 'stats'; stats: EvalStats }
  | { kind: 'error'; message: string };

/** Lets the evaluation graph fill in while a position is searched. */
const NUM_ANALYSIS_THREADS = 2;

const post = (message: FromEngine) => self.postMessage(message);

/** Mostly the config the Android app writes, minus what is about a phone. */
const config = (boardSize: number, searchThreads: number, batchWaitMicros: number) => `
numAnalysisThreads = ${NUM_ANALYSIS_THREADS}
numSearchThreadsPerAnalysisThread = ${searchThreads}
nnMaxBatchSize = ${NUM_ANALYSIS_THREADS * searchThreads}
nnServeBatchWaitMicros = ${batchWaitMicros}
maxBoardXSizeForNNBuffer = ${boardSize}
maxBoardYSizeForNNBuffer = ${boardSize}
nnCacheSizePowerOfTwo = 20
nnMutexPoolSizePowerOfTwo = 16
reportAnalysisWinratesAs = SIDETOMOVE
useGraphSearch = true
cpuctExploration = 0.9
cpuctExplorationLog = 0.6
cpuctExplorationBase = 500
useNoisePruning = true
nnPolicyTemperature = 1.1
subtreeValueBiasFactor = 0.0
noResultUtilityForWhite = 0.0
analysisPVLen = 100
logToStderr = true
`;

type EmscriptenModule = {
  FS: { writeFile(path: string, data: Uint8Array | string): void };
  wasmMemory: WebAssembly.Memory;
  callMain(args: string[]): void;
  ccall(name: string, returns: null, argTypes: string[], args: unknown[]): void;
  _katahexControlBlockAddress(): number;
};

let engine: EmscriptenModule | null = null;
const queued: string[] = [];

async function start(options: Extract<ToEngine, { kind: 'start' }>): Promise<void> {
  const { boardSize, half, searchThreads, batchWaitMicros, enginePath, modelPath } = options;

  const gpu = await navigator.gpu?.requestAdapter();
  const device = await gpu?.requestDevice({
    requiredFeatures: half && gpu.features.has('shader-f16') ? ['shader-f16'] : [],
  });
  if (!device) throw new Error('no WebGPU device');

  // The file is fetched once and used twice: the engine reads it to learn the
  // shape of the net it is searching with, and the net is built from it here.
  // Servers hand a .gz over with Content-Encoding set as often as not, in which
  // case the browser has already inflated it, so go by what the bytes are.
  const downloaded = new Uint8Array(await (await fetch(modelPath)).arrayBuffer());
  const raw = downloaded[0] === 0x1f && downloaded[1] === 0x8b
    ? new Uint8Array(await new Response(new Blob([downloaded]).stream()
        .pipeThrough(new DecompressionStream('gzip'))).arrayBuffer())
    : downloaded;

  // The engine is served verbatim, which is also how it has to be loaded: a
  // plain dynamic import would be rewritten into a request for a module of the
  // bundler's own, and it refuses to make one for a file it does not transform.
  const load = new Function('path', 'return import(path)') as
    (path: string) => Promise<{ default: (options: object) => Promise<EmscriptenModule> }>;
  const module = await (await load(enginePath)).default({
    noInitialRun: true,
    print: (line: string) => post({ kind: 'line', line }),
    printErr: (line: string) => post({ kind: 'log', line }),
  });

  module.FS.writeFile('/model.bin', raw);
  module.FS.writeFile('/analysis.cfg', config(boardSize, searchThreads, batchWaitMicros));

  const model = new KataGoWebGpuModel(device, parseKataGoModelV8(raw), boardSize, half);

  // Answers evaluations for as long as the worker lives; the engine is stopped
  // by dropping the whole worker.
  void serveEvals(module.wasmMemory, module._katahexControlBlockAddress(), model, {
    onEval: (stats) => post({ kind: 'stats', stats }),
  }).catch((error) => post({ kind: 'error', message: String(error?.stack ?? error) }));

  // Returns at once: main() is on a thread of its own.
  module.callMain(['analysis', '-model', '/model.bin', '-config', '/analysis.cfg']);

  engine = module;
  for (const json of queued) module.ccall('katahexPushQuery', null, ['string'], [json]);
  queued.length = 0;
}

self.onmessage = (event: MessageEvent<ToEngine>) => {
  const message = event.data;

  if (message.kind === 'start') {
    start(message).catch((error) => post({ kind: 'error', message: String(error?.stack ?? error) }));
    return;
  }

  // Queries sent while the net is still loading wait for it rather than being lost.
  if (engine) engine.ccall('katahexPushQuery', null, ['string'], [message.json]);
  else queued.push(message.json);
};
