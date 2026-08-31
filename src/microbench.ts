// How fast can this GPU go from a browser at all, and which shape of matmul
// tile gets it there? The kernel under test is the model's own
// (`matmulShader`), driven here on the shapes the net spends its time in, with
// no transforms, activations or readback around it.
//
// Every variant of every shape is timed once per round, several rounds deep, so
// the thermal drift over a run lands on all of them alike; the score is the best
// round, which is the one that ran coolest.

import { matmulShader, type MatmulTile } from './webgpuModel';

const logEl = document.getElementById('log')!;
const log = (s: string) => { logEl.textContent += s + '\n'; console.log(s); };

type Shape = { m: number; n: number; k: number; z: number; label: string };

// The two shapes the net's arithmetic is almost entirely in, at the batch sizes
// the gate measures, plus a square one for the unobstructed peak.
const SHAPES: Shape[] = [
  { m: 448, n: 192, k: 192, z: 36, label: 'wino trunk, batch 48' },
  { m: 5824, n: 192, k: 384, z: 1, label: '1x1 trunk, batch 48' },
  { m: 64, n: 192, k: 192, z: 36, label: 'wino trunk, batch 1' },
  { m: 32, n: 192, k: 192, z: 36, label: 'wino trunk, batch 1, half the padding' },
  { m: 2048, n: 2048, k: 2048, z: 1, label: 'square 2048' },
];

const TILES: { label: string; tile: MatmulTile }[] = [
  { label: '64x64 4x4 k16', tile: { wgX: 16, wgY: 16, rows: 4, cols4: 1, tileK: 16 } },
  { label: '64x64 4x4 k32', tile: { wgX: 16, wgY: 16, rows: 4, cols4: 1, tileK: 32 } },
  { label: '64x64 4x8 k32', tile: { wgX: 8, wgY: 16, rows: 4, cols4: 2, tileK: 32 } },
  { label: '64x64 8x4 k32', tile: { wgX: 16, wgY: 8, rows: 8, cols4: 1, tileK: 32 } },
  { label: '32x64 4x4 k32', tile: { wgX: 16, wgY: 8, rows: 4, cols4: 1, tileK: 32 } },
  { label: '32x64 2x4 k32', tile: { wgX: 16, wgY: 16, rows: 2, cols4: 1, tileK: 32 } },
];

const ROUNDS = 5;
const fits = (s: Shape, t: MatmulTile) => s.m % (t.wgY * t.rows) === 0 && s.k % t.tileK === 0;

function pipeline(device: GPUDevice, half: boolean, s: Shape, tile: MatmulTile): GPUComputePipeline {
  const code = matmulShader(half ? 'f16' : 'f32', s.m, s.n, s.k, tile);
  return device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
  });
}

/** All-ones inputs, where every output must equal K. Guarded columns included. */
async function verify(device: GPUDevice, half: boolean, tile: MatmulTile) {
  const s: Shape = { m: 2 * tile.wgY * tile.rows, n: 80, k: 96, z: 2, label: 'check' };
  const width = half ? 2 : 4;
  const counts = [s.z * s.m * s.k, s.z * s.k * s.n, s.z * s.m * s.n];
  const buffers = counts.map((count, i) => device.createBuffer({
    size: count * width,
    usage: GPUBufferUsage.STORAGE | (i < 2 ? GPUBufferUsage.COPY_DST : GPUBufferUsage.COPY_SRC),
  }));
  for (const i of [0, 1]) {
    device.queue.writeBuffer(buffers[i], 0, half
      ? new Uint16Array(counts[i]).fill(0x3c00)   // 1.0 as float16
      : new Float32Array(counts[i]).fill(1));
  }

  const p = pipeline(device, half, s, tile);
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(p);
  pass.setBindGroup(0, device.createBindGroup({
    layout: p.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  }));
  pass.dispatchWorkgroups(Math.ceil(s.n / 64), s.m / (tile.wgY * tile.rows), s.z);
  pass.end();
  const bytes = counts[2] * width;
  const readback = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  encoder.copyBufferToBuffer(buffers[2], 0, readback, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const raw = readback.getMappedRange().slice(0);
  readback.unmap();

  const values = half
    ? Array.from(new Uint16Array(raw)).map((h) => {
        const exponent = (h >> 10) & 0x1f, mantissa = h & 0x3ff;
        return (h >> 15 ? -1 : 1) * (exponent ? 2 ** (exponent - 15) * (1 + mantissa / 1024) : 0);
      })
    : Array.from(new Float32Array(raw));
  const wrong = values.filter((v) => v !== s.k).length;
  for (const buffer of [...buffers, readback]) buffer.destroy();
  return wrong;
}

async function run() {
  (document.getElementById('go') as HTMLButtonElement).disabled = true;
  logEl.textContent = '';
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('no WebGPU adapter (needs a secure context)');
    const half = adapter.features.has('shader-f16');
    log(`adapter: ${adapter.info?.vendor} ${adapter.info?.architecture}, shader-f16: ${half}, ` +
        `subgroups: ${adapter.features.has('subgroups')}`);
    const device = await adapter.requestDevice({ requiredFeatures: half ? ['shader-f16'] : [] });
    device.addEventListener('uncapturederror',
      (e) => log(`ERROR: ${(e as GPUUncapturedErrorEvent).error.message}`));

    for (const f of half ? [false, true] : [false]) {
      for (const { label, tile } of TILES) {
        const wrong = await verify(device, f, tile);
        if (wrong) log(`  ${label} ${f ? 'fp16' : 'fp32'}: ${wrong} wrong outputs`);
      }
    }
    log('all tiles check out on all-ones inputs');

    // One set of buffers and pipelines per shape, built before anything is timed.
    const usage = GPUBufferUsage.STORAGE;
    const jobs = (half ? [false, true] : [false]).flatMap((f) => SHAPES.map((s) => {
      const buffers = [s.z * s.m * s.k, s.z * s.k * s.n, s.z * s.m * s.n]
        .map((count) => device.createBuffer({ size: count * (f ? 2 : 4), usage }));
      const runs = TILES.filter(({ tile }) => fits(s, tile)).map(({ label, tile }) => {
        const p = pipeline(device, f, s, tile);
        const bindGroup = device.createBindGroup({
          layout: p.getBindGroupLayout(0),
          entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
        });
        const grid: [number, number, number] =
          [Math.ceil(s.n / 64), s.m / (tile.wgY * tile.rows), s.z];
        return { label, pipeline: p, bindGroup, grid, best: Infinity };
      });
      // Enough repeats that a timed slice runs tens of milliseconds.
      const iters = Math.max(2, Math.round(4e10 / (2 * s.m * s.n * s.k * s.z)));
      return { shape: s, half: f, runs, iters };
    }));

    for (let round = 0; round < ROUNDS; round++) {
      for (const { runs, iters } of jobs) {
        for (const r of runs) {
          const dispatch = (count: number) => {
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(r.pipeline);
            pass.setBindGroup(0, r.bindGroup);
            for (let i = 0; i < count; i++) pass.dispatchWorkgroups(...r.grid);
            pass.end();
            device.queue.submit([encoder.finish()]);
            return device.queue.onSubmittedWorkDone();
          };
          await dispatch(1);
          const t0 = performance.now();
          await dispatch(iters);
          r.best = Math.min(r.best, (performance.now() - t0) / iters);
        }
      }
    }

    for (const { shape, half: f, runs } of jobs) {
      log(`${shape.label}, ${f ? 'fp16' : 'fp32'}: ${shape.z}x ${shape.m}x${shape.k} times ${shape.k}x${shape.n}`);
      for (const r of runs) {
        const gflops = (2 * shape.m * shape.n * shape.k * shape.z) / (r.best / 1000) / 1e9;
        log(`  ${r.label.padEnd(16)} ${r.best.toFixed(3).padStart(8)} ms` +
            `, ${gflops.toFixed(0).padStart(5)} GFLOP/s`);
      }
    }
    log('done');
  } catch (e) {
    log(`ERROR: ${e}`);
  } finally {
    (document.getElementById('go') as HTMLButtonElement).disabled = false;
  }
}

document.getElementById('go')!.addEventListener('click', run);
