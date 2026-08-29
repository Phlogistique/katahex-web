import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import '@tensorflow/tfjs-backend-wasm';
import { parseKataGoModelV8 } from '../vendor/loadModelV8';
import { KataGoModelV8Tf } from '../vendor/modelV8';

const logEl = document.getElementById('log')!;
const log = (s: string) => {
  logEl.textContent += s + '\n';
  console.log(s);
};

const BATCHES = [1, 2, 4, 8, 16];
const ITERS = 12;

async function fetchModel(url: string): Promise<Uint8Array> {
  const t0 = performance.now();
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${resp.status} fetching ${url}`);
  const stream = resp.body!.pipeThrough(new DecompressionStream('gzip'));
  const chunks: Uint8Array[] = [];
  let total = 0;
  const reader = stream.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    total += value.length;
  }
  const out = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  log(`downloaded + gunzipped ${(total / 1e6).toFixed(1)} MB in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  return out;
}

async function run() {
  const backend = (document.getElementById('backend') as HTMLSelectElement).value;
  const size = Number((document.getElementById('size') as HTMLSelectElement).value);
  (document.getElementById('go') as HTMLButtonElement).disabled = true;
  logEl.textContent = '';

  try {
    if (backend === 'webgpu') {
      const adapter = await (navigator as any).gpu?.requestAdapter();
      if (!adapter) throw new Error('no WebGPU adapter (needs a secure context: https or localhost)');
      const info = await (adapter.requestAdapterInfo?.() ?? adapter.info);
      log(`adapter: ${JSON.stringify(info ?? {})}`);
      log(`shader-f16: ${adapter.features?.has?.('shader-f16')}`);
    }
    await tf.setBackend(backend);
    await tf.ready();
    log(`tf backend: ${tf.getBackend()}`);

    const raw = await fetchModel('/hex27x3.bin.gz');
    let t0 = performance.now();
    const parsed = parseKataGoModelV8(raw);
    log(`parsed in ${((performance.now() - t0) / 1000).toFixed(1)}s: ` +
        `${parsed.trunk.blocks.length} blocks, ${parsed.trunk.trunkNumChannels} channels, ` +
        `version ${parsed.modelVersion}, ${parsed.numInputChannels}+${parsed.numInputGlobalChannels} inputs`);

    t0 = performance.now();
    const model = new KataGoModelV8Tf(parsed);
    log(`weights uploaded in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

    for (const n of BATCHES) {
      const spatial = tf.zeros([n, size, size, parsed.numInputChannels]) as tf.Tensor4D;
      const global = tf.zeros([n, parsed.numInputGlobalChannels]) as tf.Tensor2D;

      // Warm up: first call compiles every shader.
      let out = model.forward(spatial, global);
      const probe = await out.policy.data();
      if (!Number.isFinite(probe[0])) log(`WARNING: policy[0] is ${probe[0]}`);
      tf.dispose(out);

      t0 = performance.now();
      for (let i = 0; i < ITERS; i++) {
        out = model.forward(spatial, global);
        await out.policy.data();
        tf.dispose(out);
      }
      const ms = (performance.now() - t0) / ITERS;
      log(`${size}x${size} batch ${String(n).padStart(2)}: ${ms.toFixed(1)} ms/call = ${(1000 * n / ms).toFixed(1)} evals/s`);
      tf.dispose([spatial, global]);
    }
    log(`gpu memory: ${JSON.stringify(tf.memory())}`);
  } catch (e) {
    log(`ERROR: ${e}`);
  } finally {
    (document.getElementById('go') as HTMLButtonElement).disabled = false;
  }
}

document.getElementById('go')!.addEventListener('click', run);
