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
      batchWaitMicros?: number; serverThreads?: number; leafEvals?: number; profile?: boolean;
      enginePath: string; modelPath: string; shapePath?: string }
  | { kind: 'query'; json: string };

export type FromEngine =
  | { kind: 'line'; line: string }   // what the engine writes on stdout: analysis results
  | { kind: 'log'; line: string }    // and on stderr: its own log
  | { kind: 'progress'; text: string }  // what it is doing in the ten seconds before that
  | { kind: 'stats'; stats: EvalStats }
  | { kind: 'error'; message: string };

/** Lets the evaluation graph fill in while a position is searched. */
const NUM_ANALYSIS_THREADS = 2;


const post = (message: FromEngine) => self.postMessage(message);

/**
 * The message goes on the page, where 120 characters of it are read by someone
 * who cannot open a console; the stack goes to the log, where it is minified
 * and only useful next to a source map.
 */
const fail = (error: unknown) => {
  const stack = (error as Error)?.stack;
  if (stack) post({ kind: 'log', line: stack });
  post({ kind: 'error', message: String((error as Error)?.message ?? error) });
};

/** Its UA is the only one with a real `Gecko/<date>`; Chrome's says "like Gecko". */
const FIREFOX = /Gecko\/\d/.test(navigator.userAgent);

/**
 * Reads the body a piece at a time so the wait can be reported: this is 49 MB,
 * and on a phone it is most of the time between opening the page and a first
 * move being analysed.
 *
 * Servers hand a .gz over with Content-Encoding set as often as not, in which
 * case the browser has already inflated it, so go by what the bytes are.
 */
async function fetchWeights(path: string, onProgress?: (fraction: number) => void): Promise<Uint8Array> {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText} fetching ${path}`);

  // Set, this is the size on the wire; a server that also sets Content-Encoding
  // is inflating as it goes and the body outgrows it, hence the clamp.
  const total = Number(response.headers.get('content-length')) || 0;
  const reader = response.body!.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    if (total) onProgress?.(Math.min(1, loaded / total));
  }

  const downloaded = new Uint8Array(loaded);
  let at = 0;
  for (const chunk of chunks) { downloaded.set(chunk, at); at += chunk.length; }

  if (downloaded[0] !== 0x1f || downloaded[1] !== 0x8b) return downloaded;
  const stream = new Blob([downloaded]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Mostly the config the Android app writes, minus what is about a phone. */
const config = (boardSize: number, searchThreads: number, batchWaitMicros: number, serverThreads: number,
                leafEvals: number) => `
numAnalysisThreads = ${NUM_ANALYSIS_THREADS}
numSearchThreadsPerAnalysisThread = ${searchThreads}
numLeafEvalsPerThread = ${leafEvals}
numNNServerThreadsPerModel = ${serverThreads}
nnMaxBatchSize = ${NUM_ANALYSIS_THREADS * searchThreads * leafEvals}
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
  _katahexControlBlockAddress(serverThreadIdx: number): number;
};

let engine: EmscriptenModule | null = null;
const queued: string[] = [];

async function start(options: Extract<ToEngine, { kind: 'start' }>): Promise<void> {
  // The newer knobs default here: this is a postMessage boundary, and a
  // driver built against an older shape must keep working.
  const { boardSize, half, searchThreads, batchWaitMicros = 3000, serverThreads = 2,
          leafEvals = 1, profile = false, enginePath, modelPath, shapePath = modelPath } = options;

  const say = (text: string) => post({ kind: 'progress', text });

  say('looking for a GPU');
  // Firefox ships WebGPU on Linux, but off: there is nothing the page can do
  // about that except say which switch, because nobody guesses `about:config`.
  if (!navigator.gpu) throw new Error(FIREFOX
    ? 'Firefox keeps WebGPU behind a switch: set dom.webgpu.enabled in about:config'
    : 'this browser has no WebGPU');
  const gpu = await navigator.gpu.requestAdapter();
  if (!gpu) throw new Error('WebGPU found no adapter to run on');

  // Half precision is the model's, not just the device's: its shaders declare
  // `enable f16`, so on an adapter without the feature they would not compile.
  const useHalf = half && gpu.features.has('shader-f16');
  const wanted: GPUFeatureName[] = [];
  if (useHalf) wanted.push('shader-f16');
  if (profile && gpu.features.has('timestamp-query')) wanted.push('timestamp-query');
  const device = await gpu.requestDevice({ requiredFeatures: wanted });
  if (!device) throw new Error('WebGPU gave no device');
  if (half && !useHalf) say('no shader-f16 on this GPU, running single precision');

  // Two weight files, because the two readers want different things out of one
  // net: the engine only ever reads its name, version and channel counts -- this
  // backend evaluates nothing -- so it is given a copy with the weights zeroed,
  // 100 KB against 49 MB, and the real weights go straight to the GPU. Both are
  // written by scripts/export_net.py; passing the same path twice is the
  // full precision net doing both jobs, which is what the node driver does.
  let percent = -1;
  const [raw, shape] = await Promise.all([
    fetchWeights(modelPath, (fraction) => {
      const now = Math.floor(fraction * 100);
      if (now === percent) return;   // one message per percent, not one per chunk
      percent = now;
      say(`downloading the net, ${now}%`);
    }),
    fetchWeights(shapePath),
  ]);

  // The engine is served verbatim, which is also how it has to be loaded: a
  // plain dynamic import would be rewritten into a request for a module of the
  // bundler's own, and it refuses to make one for a file it does not transform.
  const load = new Function('path', 'return import(path)') as
    (path: string) => Promise<{ default: (options: object) => Promise<EmscriptenModule> }>;
  say('loading the engine');
  const module = await (await load(enginePath)).default({
    noInitialRun: true,
    print: (line: string) => post({ kind: 'line', line }),
    printErr: (line: string) => post({ kind: 'log', line }),
  });

  module.FS.writeFile('/model.bin', shape);
  module.FS.writeFile('/analysis.cfg', config(boardSize, searchThreads, batchWaitMicros, serverThreads, leafEvals));

  say('reading the net');
  const parsed = parseKataGoModelV8(raw);
  say('sending the net to the GPU');
  const model = new KataGoWebGpuModel(device, parsed, boardSize, useHalf);
  model.profile = profile;

  // Answer evaluations for as long as the worker lives; the engine is stopped
  // by dropping the whole worker. One loop per engine server thread, so one
  // batch is packed and read back while the GPU runs the other.
  const stats: EvalStats = { evals: 0, rows: 0, nanos: 0 };
  const onEval = () => post({ kind: 'stats', stats: {
    ...stats,
    stages: { ...model.stages },
    kernels: profile ? [...model.gpuByLabel] : undefined,
  } });
  for (let idx = 0; idx < serverThreads; idx++) {
    void serveEvals(module.wasmMemory, module._katahexControlBlockAddress(idx), model, { stats, onEval })
      .catch(fail);
  }

  // Returns at once: main() is on a thread of its own.
  module.callMain(['analysis', '-model', '/model.bin', '-config', '/analysis.cfg']);

  engine = module;
  for (const json of queued) module.ccall('katahexPushQuery', null, ['string'], [json]);
  queued.length = 0;
}

self.onmessage = (event: MessageEvent<ToEngine>) => {
  const message = event.data;

  if (message.kind === 'start') {
    start(message).catch(fail);
    return;
  }

  // Queries sent while the net is still loading wait for it rather than being lost.
  if (engine) engine.ccall('katahexPushQuery', null, ['string'], [message.json]);
  else queued.push(message.json);
};
