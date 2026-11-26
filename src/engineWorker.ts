/// <reference lib="webworker" />

// Runs the KataHex analysis engine, compiled to WebAssembly, and the net it asks
// for, both in this worker. Queries arrive as messages and every line the engine
// writes goes back as one.
//
// The engine's main() waits for queries on a thread of its own, so this thread
// stays free: it is the one the engine's other threads route their output
// through, and the one that answers their net evaluations.

import { parseKataGoModelV8 } from '../vendor/loadModelV8';
import { serveEvals, type EvalStats, type NetEvaluator } from './netRunner';
import { KataGoWebGpuModel } from './webgpuModel';

export type ToEngine =
  | { kind: 'start'; boardSize: number; half: boolean; searchThreads: number;
      batchWaitMicros?: number; serverThreads?: number; leafEvals?: number; profile?: boolean;
      enginePath: string; modelPath: string; shapePath?: string;
      /** Skip the GPU and run the net on the CPU, as when WebGPU fails. */
      cpu?: boolean }
  | { kind: 'query'; json: string };

export type FromEngine =
  | { kind: 'line'; line: string }   // what the engine writes on stdout: analysis results
  | { kind: 'log'; line: string }    // and on stderr: its own log
  | { kind: 'progress'; text: string }  // what it is doing in the ten seconds before that
  | { kind: 'stats'; stats: EvalStats }
  | { kind: 'backend'; backend: 'gpu' | 'cpu'; detail: string }  // what the net runs on, once it does
  | { kind: 'error'; message: string };

/** Lets the evaluation graph fill in while a position is searched. */
const NUM_ANALYSIS_THREADS = 2;


const post = (message: FromEngine) => self.postMessage(message);

/**
 * Everything in flight when the device dies fails too, each naming whichever
 * buffer it happened to touch, and the banner keeps only the last message.
 */
let silenced = false;

/**
 * The message goes on the page, where 120 characters of it are read by someone
 * who cannot open a console; the stack goes to the log, where it is minified
 * and only useful next to a source map.
 */
const fail = (error: unknown) => {
  if (silenced) return;
  const stack = (error as Error)?.stack;
  if (stack) post({ kind: 'log', line: stack });
  post({ kind: 'error', message: String((error as Error)?.message ?? error) });
};

/** Its UA is the only one with a real `Gecko/<date>`; Chrome's says "like Gecko". */
const FIREFOX = /Gecko\/\d/.test(navigator.userAgent);

/**
 * Names the GPU, for the banner and for the error that follows it. A phone
 * report is a photograph of the screen, so whatever the adapter will say has to
 * be in the text. `info` is the current spelling, `requestAdapterInfo` the one
 * Chrome shipped first; browsers mask the fields they consider fingerprinting,
 * so any of them may be empty.
 */
async function describeAdapter(gpu: GPUAdapter): Promise<string> {
  type Legacy = { info?: GPUAdapterInfo; requestAdapterInfo?: () => Promise<GPUAdapterInfo> };
  const legacy = gpu as GPUAdapter & Legacy;
  const info = legacy.info ?? await legacy.requestAdapterInfo?.();
  const named = [info?.vendor, info?.architecture].filter(Boolean).join(' ');
  return named || info?.description || 'an unnamed GPU';
}

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

/** The GPU, ready to be given the net; throws whatever stops it being usable. */
async function openGpu(half: boolean, profile: boolean, say: (text: string) => void) {
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

  const adapter = await describeAdapter(gpu);
  if (half && !useHalf) say(`no shader-f16 on ${adapter}, running single precision`);

  return { device, adapter, useHalf };
}

/**
 * Runs the net once on an empty board before the engine is given it, and throws
 * if the GPU will not: a browser can hand out a device that then fails on the
 * net's shaders or buffers -- Firefox, whose WebGPU is newer than the kernels'
 * testing, is the one to expect it from. Past this point a failure is fatal, so
 * this is where it can still fall back to the CPU.
 *
 * Inside error scopes, so what goes wrong is thrown here rather than raised as an
 * uncaptured error that ends the page; uploading the weights goes in them too.
 */
async function tryGpuModel(device: GPUDevice, parsed: ReturnType<typeof parseKataGoModelV8>,
                           boardSize: number, half: boolean): Promise<KataGoWebGpuModel> {
  const scopes = ['out-of-memory', 'internal', 'validation'] as const;
  for (const scope of scopes) device.pushErrorScope(scope);

  let model: KataGoWebGpuModel | null = null;
  let thrown: unknown = null;
  let out: Awaited<ReturnType<KataGoWebGpuModel['evaluate']>> | null = null;
  try {
    model = new KataGoWebGpuModel(device, parsed, boardSize, half);
    // Channel 0 is the on-board mask; the rest of an empty board is zeros.
    const area = boardSize * boardSize;
    const spatial = new Float32Array(area * parsed.numInputChannels);
    for (let i = 0; i < area; i++) spatial[i * parsed.numInputChannels] = 1;
    out = await model.evaluate(spatial, new Float32Array(parsed.numInputGlobalChannels), 1);
  } catch (error) {
    thrown = error;
  }

  // Popped whatever happened, innermost first, so the scopes never outlive the trial.
  const errors: (GPUError | null)[] = [];
  for (let i = 0; i < scopes.length; i++) errors.push(await device.popErrorScope());
  const reported = errors.find(Boolean);

  const finite = (values: Float32Array | undefined) => !!values && values.every(Number.isFinite);
  let problem: string | null = null;
  if (reported) problem = reported.message;
  else if (thrown) problem = String((thrown as Error)?.message ?? thrown);
  else if (!out || ![out.policy[0], out.policyPass, out.value[0], out.scoreValue[0]].every(finite)) {
    problem = 'it gave numbers that are not finite';
  }

  if (problem !== null) {
    model?.dispose();
    // A shader's error goes on to quote the source with a caret under it; the
    // banner has room for the first line only.
    throw new Error(`WebGPU could not run the net: ${problem.split('\n')[0]}`);
  }
  return model!;
}

/**
 * Engine settings for a net on the CPU. The GPU's are tuned to keep it in wide
 * batches, sixty-four leaves in flight a thread; on a CPU a batch costs in
 * proportion to its rows, and every leaf in flight past the first is an
 * evaluation the search may never use.
 */
const CPU_SETTINGS = { searchThreads: 1, leafEvals: 2, serverThreads: 1, batchWaitMicros: 0 };

async function start(options: Extract<ToEngine, { kind: 'start' }>): Promise<void> {
  // The newer knobs default here: this is a postMessage boundary, and a
  // driver built against an older shape must keep working.
  const { boardSize, half, profile = false, enginePath, modelPath, shapePath = modelPath, cpu = false } = options;
  let { searchThreads, batchWaitMicros = 3000, serverThreads = 2, leafEvals = 1 } = options;

  const say = (text: string) => post({ kind: 'progress', text });

  // Without a GPU that works, the net runs on the CPU rather than not at all:
  // several times slower, but the same net, so the same moves.
  let gpu: Awaited<ReturnType<typeof openGpu>> | null = null;
  let whyCpu = 'asked for with ?cpu';
  if (!cpu) {
    try {
      gpu = await openGpu(half, profile, say);
    } catch (error) {
      whyCpu = String((error as Error)?.message ?? error);
      post({ kind: 'log', line: `no usable GPU (${whyCpu}): running the net on the CPU` });
    }
  }

  // A GPU process that goes down takes the device with it, and the page hears
  // about it as some later call failing on an object whose instance is gone --
  // "A valid external Instance reference no longer exists", which names a buffer
  // that was never the problem.
  //
  // What the device says for itself is only "GPU connection lost", the same
  // whether the process was killed for its memory or reset after a hang, so the
  // message carries how much had been asked for and how wide a batch was running
  // when it went. The banner truncates at 120 characters: keep it short, and
  // keep the fixed part first.
  let gpuModel: KataGoWebGpuModel | null = null;
  if (gpu) {
    const { device, adapter } = gpu;
    const memory = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
    const state = () => [
      gpuModel ? `${(gpuModel.allocated.bytes / 2 ** 20).toFixed(0)} MB` : 'net not uploaded',
      gpuModel?.lastBatch ? `batch ${gpuModel.lastBatch}` : null,
      memory ? `${memory} GB RAM` : null,
      adapter,
    ].filter(Boolean).join(', ');

    // Nothing is reported after this: the evaluations still in flight are about to
    // fail as well, each naming whichever buffer it happened to touch.
    void device.lost.then((lost) => {
      // Destroyed by us, when the trial run failed and the net went to the CPU.
      if (lost.reason === 'destroyed') return;
      fail(new Error(`GPU device lost (${state()}): ${lost.message}`));
      silenced = true;
    });

    device.addEventListener('uncapturederror', (event) => {
      fail(new Error(`GPU rejected the net (${state()}): ${(event as GPUUncapturedErrorEvent).error.message}`));
    });
  }

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

  say('reading the net');
  const parsed = parseKataGoModelV8(raw);

  if (gpu) {
    say('sending the net to the GPU');
    try {
      gpuModel = await tryGpuModel(gpu.device, parsed, boardSize, gpu.useHalf);
    } catch (error) {
      whyCpu = String((error as Error)?.message ?? error);
      post({ kind: 'log', line: `${whyCpu}: running the net on the CPU` });
      gpu.device.destroy();
      gpu = null;
    }
  }

  let model: NetEvaluator;
  if (gpu && gpuModel) {
    gpuModel.profile = profile;
    model = gpuModel;
    post({ kind: 'backend', backend: 'gpu', detail: gpu.adapter });
  } else {
    ({ searchThreads, leafEvals, serverThreads, batchWaitMicros } = CPU_SETTINGS);
    say('no usable GPU: loading the net on the CPU, which is far slower');
    // Fetched only now: a page with a GPU never downloads TensorFlow.js.
    const { createCpuModel } = await import('./cpuModel');
    const cpuModel = await createCpuModel(parsed, boardSize);
    model = cpuModel;
    post({ kind: 'backend', backend: 'cpu', detail: `${cpuModel.backend}, ${cpuModel.threads} threads; ${whyCpu}` });
  }

  // Written only now: the engine's threads and batches depend on where the net runs.
  module.FS.writeFile('/model.bin', shape);
  module.FS.writeFile('/analysis.cfg', config(boardSize, searchThreads, batchWaitMicros, serverThreads, leafEvals));

  // Answer evaluations for as long as the worker lives; the engine is stopped
  // by dropping the whole worker. One loop per engine server thread, so one
  // batch is packed and read back while the GPU runs the other.
  const stats: EvalStats = { evals: 0, rows: 0, nanos: 0 };
  const onEval = () => post({ kind: 'stats', stats: {
    ...stats,
    stages: gpuModel ? { ...gpuModel.stages } : undefined,
    kernels: profile && gpuModel ? [...gpuModel.gpuByLabel] : undefined,
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
