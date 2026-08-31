// Replays stored feature tensors through the hand-written WebGPU backend and
// measures how far the outputs are from where they should be. No engine, no
// search: this is the net alone, driven by scripts/check-headless.mjs, which
// judges the numbers this page reports.
//
// Tier 1: fp32 against TensorFlow.js goldens frozen to disk by
// scripts/make-positions.mjs (the implementation that agrees with native to
// 1e-6), on 18 positions chosen to hit the numerically distinct paths, plus
// batch-shape invariance. Tier 2: fp16 against the live fp32 the goldens have
// just pinned, over a 512-position bank, so fp32 stands in for the oracle at
// fp16's 0.05 error scale.
//
// `?mode=perf&half=1` instead loads one model and exposes runSlice() for the
// tier-3 driver; nothing runs until the driver asks.

import { parseKataGoModelV8 } from '../vendor/loadModelV8';
import { KataGoWebGpuModel, type NetOutputs } from './webgpuModel';
import { bridge } from './harness';

const SPATIAL = 22;
const GLOBAL = 19;

const params = new URLSearchParams(location.search);
const { driver, log } = bridge<never, unknown>();

async function fetchBytes(url: string): Promise<Uint8Array> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${resp.status} fetching ${url}`);
  let bytes = new Uint8Array(await resp.arrayBuffer());
  // A dev server may have inflated the .gz on the way in; go by the bytes.
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    bytes = new Uint8Array(await new Response(stream).arrayBuffer());
  }
  return bytes;
}

async function makeDevice(): Promise<GPUDevice> {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error('no WebGPU adapter');
  if (!adapter.features.has('shader-f16')) throw new Error('no shader-f16');
  const device = await adapter.requestDevice({ requiredFeatures: ['shader-f16'] });
  device.addEventListener('uncapturederror',
    (e) => log(`GPU ERROR: ${(e as GPUUncapturedErrorEvent).error.message}`));
  return device;
}

/** One stored position: features the way evaluate() takes them. */
type Position = { spatial: Float32Array; global: Float32Array };

function slicePositions(data: Float32Array, count: number, size: number): Position[] {
  const hw = size * size;
  const out: Position[] = [];
  let at = 0;
  for (let i = 0; i < count; i++) {
    out.push({ spatial: data.subarray(at, at += hw * SPATIAL),
               global: data.subarray(at, at += GLOBAL) });
  }
  return out;
}

function evaluateBatch(model: KataGoWebGpuModel, positions: Position[], size: number): Promise<NetOutputs> {
  const hw = size * size;
  const spatial = new Float32Array(positions.length * hw * SPATIAL);
  const global = new Float32Array(positions.length * GLOBAL);
  positions.forEach((p, i) => {
    spatial.set(p.spatial, i * hw * SPATIAL);
    global.set(p.global, i * GLOBAL);
  });
  return model.evaluate(spatial, global, positions.length);
}

/** The four heads of one position, flattened for comparison. */
const heads = (out: NetOutputs, i: number): Record<string, ArrayLike<number>> => ({
  policy: out.policy[i],
  policyPass: [out.policyPass[i]],
  value: out.value[i],
  scoreValue: out.scoreValue[i],
});

function compare(a: Record<string, ArrayLike<number>>, b: Record<string, ArrayLike<number>>) {
  const worst: Record<string, number> = {};
  let finite = true;
  for (const name of Object.keys(a)) {
    let max = 0;
    for (let i = 0; i < a[name].length; i++) {
      const d = Math.abs(a[name][i] - b[name][i]);
      if (!Number.isFinite(a[name][i])) finite = false;
      max = Math.max(max, d);
    }
    worst[name] = max;
  }
  return { worst, finite };
}

// ---------------------------------------------------------------------------

async function tier1(device: GPUDevice, parsed: ReturnType<typeof parseKataGoModelV8>) {
  const manifest = JSON.parse(new TextDecoder().decode(await fetchBytes('/check/tier1.json'))) as {
    positions: { id: string; size: number; golden: Record<string, number[]> }[];
  };
  const data = new Float32Array((await fetchBytes('/check/tier1-features.bin.gz')).buffer);

  // The features file is manifest order: all of one size, then the other.
  const positions: (Position & { id: string; size: number; golden: Record<string, number[]> })[] = [];
  let at = 0;
  for (const p of manifest.positions) {
    const hw = p.size * p.size;
    positions.push({ ...p, spatial: data.subarray(at, at += hw * SPATIAL),
                     global: data.subarray(at, at += GLOBAL) });
  }
  if (at !== data.length) throw new Error('tier1 features do not match manifest');

  const results: { id: string; worst: Record<string, number>; finite: boolean }[] = [];
  for (const size of [11, 13]) {
    const ofSize = positions.filter((p) => p.size === size);
    if (!ofSize.length) continue;
    const model = new KataGoWebGpuModel(device, parsed, size, false);
    const out = await evaluateBatch(model, ofSize, size);
    ofSize.forEach((p, i) => results.push({ id: p.id, ...compare(heads(out, i), p.golden) }));

    // Batch-shape invariance, on the 11x11 set: the same positions as ragged
    // partial tiles (5+11) and alone (batch 1) must answer the same.
    if (size === 11 && ofSize.length >= 16) {
      const again: NetOutputs[] = [
        await evaluateBatch(model, ofSize.slice(0, 5), size),
        await evaluateBatch(model, ofSize.slice(5), size),
        await evaluateBatch(model, [ofSize[0]], size),
        await evaluateBatch(model, [ofSize[7]], size),
      ];
      const pieces = [
        ...ofSize.slice(0, 5).map((_, i) => heads(again[0], i)),
        ...ofSize.slice(5).map((_, i) => heads(again[1], i)),
      ];
      let batchDiff = 0;
      ofSize.forEach((_, i) => {
        const { worst } = compare(heads(out, i), pieces[i]);
        batchDiff = Math.max(batchDiff, ...Object.values(worst));
      });
      for (const [slot, k] of [[2, 0], [3, 7]] as const) {
        const { worst } = compare(heads(out, k), heads(again[slot], 0));
        batchDiff = Math.max(batchDiff, ...Object.values(worst));
      }
      results.push({ id: 'batch-invariance', worst: { policy: batchDiff }, finite: true });
    }
    model.dispose();
  }
  const overall = Math.max(...results.flatMap((r) => Object.values(r.worst)));
  log(`tier 1: worst error vs goldens ${overall.toExponential(2)} over ${results.length} checks`);
  return results;
}

async function tier2(device: GPUDevice, parsed: ReturnType<typeof parseKataGoModelV8>) {
  const raw = new Float32Array((await fetchBytes('/check/bank-11.bin.gz')).buffer);
  const count = raw.length / (11 * 11 * SPATIAL + GLOBAL);
  const positions = slicePositions(raw, count, 11);
  const models = {
    fp32: new KataGoWebGpuModel(device, parsed, 11, false),
    fp16: new KataGoWebGpuModel(device, parsed, 11, true),
  };

  const BATCH = 32;
  let absSum = 0, absCount = 0, max = 0, flips = 0, klSum = 0, dwSum = 0;
  const cellBias = new Float64Array(121);
  const allErrs: number[] = [];

  const softmax = (logits: number[]) => {
    const m = Math.max(...logits);
    const e = logits.map((v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  };

  for (let start = 0; start < count; start += BATCH) {
    const chunk = positions.slice(start, start + BATCH);
    const a = await evaluateBatch(models.fp32, chunk, 11);
    const b = await evaluateBatch(models.fp16, chunk, 11);
    for (let i = 0; i < chunk.length; i++) {
      const la = [...a.policy[i], a.policyPass[i]];
      const lb = [...b.policy[i], b.policyPass[i]];
      for (let c = 0; c < 121; c++) {
        const err = lb[c] - la[c];
        const abs = Math.abs(err);
        absSum += abs; absCount++; allErrs.push(abs);
        max = Math.max(max, abs);
        cellBias[c] += err;
      }
      if (la.indexOf(Math.max(...la)) !== lb.indexOf(Math.max(...lb))) flips++;
      const pa = softmax(la), pb = softmax(lb);
      klSum += pa.reduce((s, p, c) => s + (p > 0 ? p * Math.log(p / Math.max(pb[c], 1e-20)) : 0), 0);
      const wa = softmax([...a.value[i]])[0], wb = softmax([...b.value[i]])[0];
      dwSum += Math.abs(wb - wa);
    }
  }

  allErrs.sort((x, y) => x - y);
  const metrics = {
    positions: count,
    meanAbsErr: absSum / absCount,
    p95: allErrs[Math.floor(allErrs.length * 0.95)],
    max,
    worstCellBias: Math.max(...[...cellBias].map((v) => Math.abs(v / count))),
    top1Flips: flips,
    meanKL: klSum / count,
    meanAbsDWinrate: dwSum / count,
  };
  log(`tier 2: mean |logit err| ${metrics.meanAbsErr.toFixed(4)}, p95 ${metrics.p95.toFixed(4)}, ` +
      `max ${metrics.max.toFixed(3)}, worst cell bias ${metrics.worstCellBias.toFixed(4)}, ` +
      `${flips}/${count} top-1 flips, KL ${metrics.meanKL.toExponential(2)}, ` +
      `dWinrate ${metrics.meanAbsDWinrate.toExponential(2)}`);
  return metrics;
}

// ---------------------------------------------------------------------------

async function perf(device: GPUDevice, parsed: ReturnType<typeof parseKataGoModelV8>) {
  const half = params.get('half') !== '0';
  const model = new KataGoWebGpuModel(device, parsed, 11, half);
  const raw = new Float32Array((await fetchBytes('/check/bank-11.bin.gz')).buffer);
  const positions = slicePositions(raw, raw.length / (11 * 11 * SPATIAL + GLOBAL), 11);

  (globalThis as Record<string, unknown>).runSlice = async (batch: number, evals: number) => {
    const chunk = Array.from({ length: batch }, (_, i) => positions[i % positions.length]);
    await evaluateBatch(model, chunk, 11); // compiles the plan; not timed
    const t0 = performance.now();
    for (let i = 0; i < evals; i++) await evaluateBatch(model, chunk, 11);
    return performance.now() - t0;
  };
  (globalThis as Record<string, unknown>).perfReady = true;
  log(`perf mode ready (${half ? 'fp16' : 'fp32'})`);
}

async function main() {
  const device = await makeDevice();
  const parsed = parseKataGoModelV8(await fetchBytes('/hex27x3.bin.gz'));
  log('model parsed');

  if (params.get('mode') === 'perf') return perf(device, parsed);

  const report = { tier1: await tier1(device, parsed), tier2: await tier2(device, parsed) };
  await driver.report?.(report);
  log('done');
}

main().catch((error) => log(`ERROR: ${error?.stack ?? error}`));
