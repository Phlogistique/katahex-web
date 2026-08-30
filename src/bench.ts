// Times forward passes of the net in the browser, across runtimes and batch
// sizes. What matters is the shape of the batch curve: KataGo's search produces
// a batch of about half the search thread count, and how much throughput that
// buys decides whether the page needs wasm threads at all.

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import '@tensorflow/tfjs-backend-wasm';
import * as ort from 'onnxruntime-web/webgpu';
import { parseKataGoModelV8 } from '../vendor/loadModelV8';
import { KataGoModelV8Tf } from '../vendor/modelV8';
import { KataGoWebGpuModel } from './webgpuModel';

const logEl = document.getElementById('log')!;
const log = (s: string) => {
  logEl.textContent += s + '\n';
  console.log(s);
};

const BATCHES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128];
const ITERS = 12;
const NUM_SPATIAL_FEATURES = 22;
const NUM_GLOBAL_FEATURES = 19;

/** One evaluation of `batchSize` positions, with the results back on the CPU. */
type Runner = { evaluate: (batchSize: number) => Promise<number>; dispose: () => void };

async function fetchModel(url: string): Promise<Uint8Array> {
  const t0 = performance.now();
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${resp.status} fetching ${url}`);
  let bytes = new Uint8Array(await resp.arrayBuffer());

  // A server that labels the file `Content-Encoding: gzip` -- vite's dev server
  // does, for any .gz -- has the browser inflate it on the way in, and inflating
  // it again fails. Go by what actually arrived.
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    bytes = new Uint8Array(await new Response(stream).arrayBuffer());
  }
  log(`model ready, ${(bytes.length / 1e6).toFixed(1)} MB in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  return bytes;
}

async function makeTfRunner(backend: string, size: number): Promise<Runner> {
  await tf.setBackend(backend);
  await tf.ready();
  log(`tensorflow.js backend: ${tf.getBackend()}`);

  const parsed = parseKataGoModelV8(await fetchModel('/hex27x3.bin.gz'));
  log(`parsed: ${parsed.trunk.blocks.length} blocks, ${parsed.trunk.trunkNumChannels} channels, ` +
      `version ${parsed.modelVersion}`);
  const model = new KataGoModelV8Tf(parsed);

  return {
    async evaluate(batchSize) {
      const spatial = tf.zeros([batchSize, size, size, NUM_SPATIAL_FEATURES]) as tf.Tensor4D;
      const global = tf.zeros([batchSize, NUM_GLOBAL_FEATURES]) as tf.Tensor2D;
      const out = model.forwardPolicyValue(spatial, global);
      const policy = await out.policy.data();
      tf.dispose([out, spatial, global]);
      return policy[0];
    },
    dispose: () => model.dispose(),
  };
}

/**
 * The hand-written WebGPU backend. Before timing, checks one random position
 * against the TensorFlow.js implementation of the same net on the CPU: two
 * independent codepaths from the same weights.
 */
async function makeHandRunner(size: number, half: boolean): Promise<Runner> {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error('no WebGPU adapter');
  if (half && !adapter.features.has('shader-f16')) throw new Error('no shader-f16');
  const device = await adapter.requestDevice({ requiredFeatures: half ? ['shader-f16'] : [] });
  device.addEventListener('uncapturederror',
    (e) => log(`GPU ERROR: ${(e as GPUUncapturedErrorEvent).error.message}`));

  const bytes = await fetchModel('/hex27x3.bin.gz');
  const parsed = parseKataGoModelV8(bytes);
  const t0 = performance.now();
  const model = new KataGoWebGpuModel(device, parsed, size, half);
  log(`hand webgpu ${half ? 'fp16' : 'fp32'} weights ready in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

  const hw = size * size;
  // Deterministic pseudo-random position, so error numbers compare across runs.
  let seed = 12345;
  const rand = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x80000000;
  const spatial = new Float32Array(hw * NUM_SPATIAL_FEATURES);
  for (let i = 0; i < spatial.length; i++) {
    spatial[i] = i % NUM_SPATIAL_FEATURES === 0 ? 1 : (rand() < 0.2 ? 1 : 0);
  }
  const global = new Float32Array(NUM_GLOBAL_FEATURES);
  const got = await model.evaluate(spatial, global, 1);

  await tf.setBackend('cpu');
  const ref = tf.tidy(() => {
    const m = new KataGoModelV8Tf(parsed);
    const out = m.forwardPolicyValue(
      tf.tensor4d(spatial, [1, size, size, NUM_SPATIAL_FEATURES]),
      tf.tensor2d(global, [1, NUM_GLOBAL_FEATURES]));
    return {
      policy: out.policy.dataSync() as Float32Array,
      policyPass: out.policyPass.dataSync() as Float32Array,
      value: out.value.dataSync() as Float32Array,
    };
  });
  let maxPolicyErr = 0, maxLogit = 0, sumErr = 0;
  for (let i = 0; i < hw; i++) {
    const err = Math.abs(got.policy[0][i] - ref.policy[i]);
    maxPolicyErr = Math.max(maxPolicyErr, err);
    sumErr += err;
    maxLogit = Math.max(maxLogit, Math.abs(ref.policy[i]));
  }
  const valueErr = Math.max(...[0, 1, 2].map((i) => Math.abs(got.value[0][i] - ref.value[i])));
  log(`check vs tensorflow.js cpu: policy logit err ${maxPolicyErr.toFixed(4)} max, ` +
      `${(sumErr / hw).toFixed(4)} mean (logits reach ${maxLogit.toFixed(1)}), pass err ` +
      `${Math.abs(got.policyPass[0] - ref.policyPass[0]).toFixed(4)}, value err ${valueErr.toFixed(4)}`);

  return {
    async evaluate(batchSize) {
      const out = await model.evaluate(
        new Float32Array(batchSize * hw * NUM_SPATIAL_FEATURES),
        new Float32Array(batchSize * NUM_GLOBAL_FEATURES), batchSize);
      return out.policy[0][0];
    },
    dispose: () => { model.dispose(); device.destroy(); },
  };
}

async function makeOrtRunner(provider: 'webgpu' | 'wasm', size: number, half: boolean): Promise<Runner> {
  // onnxruntime loads its own wasm at runtime. Serve those files as plain static
  // assets: vite's dev server rewrites the dynamic import of the .mjs and hands
  // back HTML, so this path wants `npm run build && npm run preview`.
  ort.env.wasm.wasmPaths = '/ort/';
  const t0 = performance.now();
  const session = await ort.InferenceSession.create(`/hex27x3-${size}${half ? '-fp16' : ''}.onnx`, {
    executionProviders: [provider],
    graphOptimizationLevel: 'all',
  });
  log(`onnxruntime session on ${provider} in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

  return {
    async evaluate(batchSize) {
      const spatial = new ort.Tensor(
        'float32', new Float32Array(batchSize * size * size * NUM_SPATIAL_FEATURES),
        [batchSize, size, size, NUM_SPATIAL_FEATURES]);
      const global = new ort.Tensor(
        'float32', new Float32Array(batchSize * NUM_GLOBAL_FEATURES), [batchSize, NUM_GLOBAL_FEATURES]);
      const out = await session.run({ spatial, global });
      return (out.policy.data as Float32Array)[0];
    },
    dispose: () => void session.release(),
  };
}

async function run() {
  const runtime = (document.getElementById('runtime') as HTMLSelectElement).value;
  const size = Number((document.getElementById('size') as HTMLSelectElement).value);
  (document.getElementById('go') as HTMLButtonElement).disabled = true;
  logEl.textContent = '';

  let runner: Runner | undefined;
  try {
    if (runtime.includes('webgpu')) {
      const adapter = await (navigator as any).gpu?.requestAdapter();
      if (!adapter) throw new Error('no WebGPU adapter (needs a secure context: https or localhost)');
      log(`webgpu adapter, shader-f16: ${adapter.features?.has?.('shader-f16')}`);
    }

    runner = runtime.startsWith('hand')
      ? await makeHandRunner(size, runtime.includes('fp16'))
      : runtime.startsWith('onnx')
        ? await makeOrtRunner(runtime.includes('wasm') ? 'wasm' : 'webgpu', size, runtime.includes('fp16'))
        : await makeTfRunner(runtime.includes('wasm') ? 'wasm' : 'webgpu', size);

    // Up then down, because the GPU throttles as it heats: a batch's two
    // readings straddle the drift, so their average mostly cancels it.
    for (const batchSize of [...BATCHES, ...[...BATCHES].reverse()]) {
      const probe = await runner.evaluate(batchSize); // warm up: compiles every shader
      if (!Number.isFinite(probe)) log(`WARNING: policy[0] is ${probe}`);

      const t0 = performance.now();
      for (let i = 0; i < ITERS; i++) await runner.evaluate(batchSize);
      const ms = (performance.now() - t0) / ITERS;
      log(`${size}x${size} batch ${String(batchSize).padStart(2)}: ` +
          `${ms.toFixed(1)} ms/call = ${(1000 * batchSize / ms).toFixed(1)} evals/s`);
    }
    log('done');
  } catch (e) {
    log(`ERROR: ${e}`);
  } finally {
    runner?.dispose();
    (document.getElementById('go') as HTMLButtonElement).disabled = false;
  }
}

document.getElementById('go')!.addEventListener('click', run);
