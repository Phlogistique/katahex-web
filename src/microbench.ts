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
  { m: 32, n: 192, k: 192, z: 36, label: 'wino trunk, batch 1' },
  { m: 64, n: 192, k: 192, z: 36, label: 'wino trunk, batch 1, padded to 64 rows' },
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

// Every input depends on all of its indices, so a swapped k lane or a wrong
// column stride moves the answer -- which all-ones inputs would not. The values
// are small integers whose products sum exactly in fp16 over these k.
const aAt = (z: number, r: number, k: number) => ((r + 2 * k + z) % 5) - 2;
const bAt = (z: number, k: number, c: number) => ((k + 3 * c + 2 * z) % 3) - 1;

/** 1.0-scale integers only: no rounding, no subnormals. */
const toF16 = (v: number) => {
  if (v === 0) return 0;
  const u = new Uint32Array(new Float32Array([v]).buffer)[0];
  return ((u >>> 16) & 0x8000) | ((((u >>> 23) & 0xff) - 112) << 10) | ((u >>> 13) & 0x3ff);
};

/** The tile against a CPU reference, on a shape whose columns exercise the guard. */
async function verify(device: GPUDevice, half: boolean, tile: MatmulTile) {
  const s: Shape = { m: 2 * tile.wgY * tile.rows, n: 80, k: 96, z: 2, label: 'check' };
  const width = half ? 2 : 4;
  const counts = [s.z * s.m * s.k, s.z * s.k * s.n, s.z * s.m * s.n];
  const buffers = counts.map((count, i) => device.createBuffer({
    size: count * width,
    usage: GPUBufferUsage.STORAGE | (i < 2 ? GPUBufferUsage.COPY_DST : GPUBufferUsage.COPY_SRC),
  }));
  const fill = (count: number, at: (i: number) => number) => {
    const values = Array.from({ length: count }, (_, i) => at(i));
    return half ? new Uint16Array(values.map(toF16)) : new Float32Array(values);
  };
  device.queue.writeBuffer(buffers[0], 0, fill(counts[0],
    (i) => aAt(Math.floor(i / (s.m * s.k)), Math.floor(i / s.k) % s.m, i % s.k)));
  device.queue.writeBuffer(buffers[1], 0, fill(counts[1],
    (i) => bAt(Math.floor(i / (s.k * s.n)), Math.floor(i / s.n) % s.k, i % s.n)));

  const want = Array.from({ length: counts[2] }, (_, i) => {
    const z = Math.floor(i / (s.m * s.n)), r = Math.floor(i / s.n) % s.m, c = i % s.n;
    let sum = 0;
    for (let k = 0; k < s.k; k++) sum += aAt(z, r, k) * bAt(z, k, c);
    return sum;
  });

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
  const wrong = values.filter((v, i) => v !== want[i]).length;
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
        // A wrong kernel must not be timed: its number would be quoted.
        if (wrong) throw new Error(`${label} ${f ? 'fp16' : 'fp32'}: ${wrong} wrong outputs`);
      }
    }
    log('every tile matches the CPU reference');

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
