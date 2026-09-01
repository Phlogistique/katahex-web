// Does Winograd pay on WebGPU? A 3x3 convolution as F(4x4, 3x3) -- input
// transform, one batched matmul over the 36 tile positions, output transform --
// written by hand in WGSL, next to the same convolution as one direct matmul.
// The transform matrices and the (36, channels, tiles) matmul layout are
// KataGo's, from cpp/neuralnet/openclkernels.cpp; the filter transform happens
// on the CPU at load time, as there.

const logEl = document.getElementById('log')!;
const log = (s: string) => { logEl.textContent += s + '\n'; console.log(s); };

const TILE = 64;            // output tile per matmul workgroup
const TILE_K = 16;
const PER_THREAD = 4;

type Conv = { batch: number; board: number; cin: number; cout: number };

// The trunk conv of the b18c384nbt net at the two batch sizes that matter:
// where the onnxruntime curve flattens, and where it ends.
const SHAPES: Conv[] = [
  { batch: 16, board: 11, cin: 192, cout: 192 },
  { batch: 64, board: 11, cin: 192, cout: 192 },
  { batch: 64, board: 13, cin: 192, cout: 192 },
];

const tilesPerDim = (board: number) => Math.ceil(board / 4);
const roundUp = (v: number, m: number) => Math.ceil(v / m) * m;

// Real outputs, as the direct convolution would count them.
const nominalFlops = (c: Conv) =>
  2 * c.batch * c.board * c.board * c.cout * c.cin * 9;

/** M rows of the batched matmul: one per 4x4 output tile, padded for the kernel. */
function gemmM(c: Conv): { real: number; padded: number } {
  const real = c.batch * tilesPerDim(c.board) ** 2;
  return { real, padded: roundUp(real, TILE) };
}

// ---------------------------------------------------------------------------
// Shaders

/** The scalar tiled matmul the numbers below were taken with, batched over GROUPS along z. */
function matmulShader(half: boolean, m: number, n: number, k: number): string {
  const t = half ? 'f16' : 'f32';
  return `${half ? 'enable f16;\n' : ''}
const M = ${m}u; const N = ${n}u; const K = ${k}u;
@group(0) @binding(0) var<storage, read> a : array<${t}>;
@group(0) @binding(1) var<storage, read> b : array<${t}>;
@group(0) @binding(2) var<storage, read_write> c : array<${t}>;
var<workgroup> aTile : array<${t}, ${TILE * TILE_K}>;
var<workgroup> bTile : array<${t}, ${TILE_K * TILE}>;

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid : vec3<u32>,
        @builtin(workgroup_id) wid : vec3<u32>) {
  let aOff = wid.z * M * K;
  let bOff = wid.z * K * N;
  let cOff = wid.z * M * N;
  let rowBase = wid.y * ${TILE}u + lid.y * ${PER_THREAD}u;
  let colBase = wid.x * ${TILE}u + lid.x * ${PER_THREAD}u;
  let tid = lid.y * 16u + lid.x;
  var acc : array<array<${t}, ${PER_THREAD}>, ${PER_THREAD}>;

  for (var k0 = 0u; k0 < K; k0 += ${TILE_K}u) {
    for (var i = 0u; i < 4u; i++) {
      let idx = tid + i * 256u;
      aTile[idx] = a[aOff + (wid.y * ${TILE}u + idx / ${TILE_K}u) * K + k0 + idx % ${TILE_K}u];
      bTile[idx] = b[bOff + (k0 + idx / ${TILE}u) * N + wid.x * ${TILE}u + idx % ${TILE}u];
    }
    workgroupBarrier();
    for (var kk = 0u; kk < ${TILE_K}u; kk++) {
      for (var i = 0u; i < ${PER_THREAD}u; i++) {
        let av = aTile[(lid.y * ${PER_THREAD}u + i) * ${TILE_K}u + kk];
        for (var j = 0u; j < ${PER_THREAD}u; j++) {
          acc[i][j] += av * bTile[kk * ${TILE}u + lid.x * ${PER_THREAD}u + j];
        }
      }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < ${PER_THREAD}u; i++) {
    for (var j = 0u; j < ${PER_THREAD}u; j++) {
      c[cOff + (rowBase + i) * N + colBase + j] = acc[i][j];
    }
  }
}`;
}

/**
 * Input transform (B^T d B per 6x6 tile) and output transform (A^T m A).
 * One thread per (tile, channel). Layouts around the matmul:
 *   v    : [36][mPad][cin]   -- matmul A
 *   mm   : [36][mPad][cout]  -- matmul C
 * with images NCHW. Rows of v beyond the real tiles are never written and the
 * buffer starts zeroed, so the padded matmul rows compute on zeros.
 */
function transformShader(half: boolean, c: Conv, mPad: number): string {
  const t = half ? 'f16' : 'f32';
  const tiles = tilesPerDim(c.board);
  return `${half ? 'enable f16;\n' : ''}
const B = ${c.board}; const TX = ${tiles}u;
const MREAL = ${c.batch * tiles * tiles}u; const MPAD = ${mPad}u;
const CIN = ${c.cin}u; const COUT = ${c.cout}u;
@group(0) @binding(0) var<storage, read> input : array<${t}>;
@group(0) @binding(1) var<storage, read_write> v : array<${t}>;

@compute @workgroup_size(64)
fn transform(@builtin(global_invocation_id) gid : vec3<u32>) {
  let m = gid.x;
  let ic = gid.y;
  if (m >= MREAL) { return; }
  let img = m / (TX * TX);
  let ty = (m / TX) % TX;
  let tx = m % TX;

  var w : array<array<${t}, 6>, 6>;
  for (var sy = 0u; sy < 6u; sy++) {
    let y = i32(ty * 4u + sy) - 1;
    for (var sx = 0u; sx < 6u; sx++) {
      let x = i32(tx * 4u + sx) - 1;
      var value = ${t}(0);
      if (y >= 0 && y < B && x >= 0 && x < B) {
        value = input[(img * CIN + ic) * u32(B * B) + u32(y * B + x)];
      }
      w[sy][sx] = value;
    }
  }
  for (var sy = 0u; sy < 6u; sy++) {
    let z0 = w[sy][0]; let z1 = w[sy][1]; let z2 = w[sy][2];
    let z3 = w[sy][3]; let z4 = w[sy][4]; let z5 = w[sy][5];
    w[sy][0] = 4*z0 - 5*z2 + z4;
    w[sy][1] = -4*z1 - 4*z2 + z3 + z4;
    w[sy][2] = 4*z1 - 4*z2 - z3 + z4;
    w[sy][3] = -2*z1 - z2 + 2*z3 + z4;
    w[sy][4] = 2*z1 - z2 - 2*z3 + z4;
    w[sy][5] = 4*z1 - 5*z3 + z5;
  }
  for (var sx = 0u; sx < 6u; sx++) {
    let z0 = w[0][sx]; let z1 = w[1][sx]; let z2 = w[2][sx];
    let z3 = w[3][sx]; let z4 = w[4][sx]; let z5 = w[5][sx];
    w[0][sx] = 4*z0 - 5*z2 + z4;
    w[1][sx] = -4*z1 - 4*z2 + z3 + z4;
    w[2][sx] = 4*z1 - 4*z2 - z3 + z4;
    w[3][sx] = -2*z1 - z2 + 2*z3 + z4;
    w[4][sx] = 2*z1 - z2 - 2*z3 + z4;
    w[5][sx] = 4*z1 - 5*z3 + z5;
  }
  for (var sy = 0u; sy < 6u; sy++) {
    for (var sx = 0u; sx < 6u; sx++) {
      v[(sy * 6u + sx) * MPAD * CIN + m * CIN + ic] = w[sy][sx];
    }
  }
}

@group(0) @binding(0) var<storage, read> mm : array<${t}>;
@group(0) @binding(1) var<storage, read_write> output : array<${t}>;

@compute @workgroup_size(64)
fn untransform(@builtin(global_invocation_id) gid : vec3<u32>) {
  let m = gid.x;
  let oc = gid.y;
  if (m >= MREAL) { return; }
  let img = m / (TX * TX);
  let ty = (m / TX) % TX;
  let tx = m % TX;

  var w : array<array<${t}, 6>, 6>;
  for (var sy = 0u; sy < 6u; sy++) {
    for (var sx = 0u; sx < 6u; sx++) {
      w[sy][sx] = mm[(sy * 6u + sx) * MPAD * COUT + m * COUT + oc];
    }
  }
  for (var sy = 0u; sy < 6u; sy++) {
    let z0 = w[sy][0]; let z1 = w[sy][1]; let z2 = w[sy][2];
    let z3 = w[sy][3]; let z4 = w[sy][4]; let z5 = w[sy][5];
    w[sy][0] = z0 + z1 + z2 + z3 + z4;
    w[sy][1] = (z1 - z2) + 2*(z3 - z4);
    w[sy][2] = (z1 + z2) + 4*(z3 + z4);
    w[sy][3] = (z1 - z2) + 8*(z3 - z4) + z5;
  }
  for (var sx = 0u; sx < 4u; sx++) {
    let z0 = w[0][sx]; let z1 = w[1][sx]; let z2 = w[2][sx];
    let z3 = w[3][sx]; let z4 = w[4][sx]; let z5 = w[5][sx];
    w[0][sx] = z0 + z1 + z2 + z3 + z4;
    w[1][sx] = (z1 - z2) + 2*(z3 - z4);
    w[2][sx] = (z1 + z2) + 4*(z3 + z4);
    w[3][sx] = (z1 - z2) + 8*(z3 - z4) + z5;
  }
  for (var sy = 0u; sy < 4u; sy++) {
    let y = ty * 4u + sy;
    if (y >= u32(B)) { continue; }
    for (var sx = 0u; sx < 4u; sx++) {
      let x = tx * 4u + sx;
      if (x >= u32(B)) { continue; }
      output[(img * COUT + oc) * u32(B * B) + y * u32(B) + x] = w[sy][sx];
    }
  }
}`;
}

// ---------------------------------------------------------------------------
// Host-side data

const f32Scratch = new Float32Array(1);
const u32Scratch = new Uint32Array(f32Scratch.buffer);

function toHalf(value: number): number {
  f32Scratch[0] = value;
  const bits = u32Scratch[0];
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7fffff;
  if (exponent >= 143) return sign | 0x7c00;
  if (exponent <= 112) {
    if (exponent < 103) return sign;
    return sign | ((mantissa | 0x800000) >> (126 - exponent));
  }
  return sign | ((exponent - 112) << 10) | (mantissa >> 13);
}

function fromHalf(h: number): number {
  const exponent = (h >> 10) & 0x1f, mantissa = h & 0x3ff;
  return (h >> 15 ? -1 : 1) * (exponent ? 2 ** (exponent - 15) * (1 + mantissa / 1024) : 2 ** -14 * (mantissa / 1024));
}

const asDeviceData = (data: Float32Array, half: boolean): Uint16Array<ArrayBuffer> | Float32Array<ArrayBuffer> =>
  half ? Uint16Array.from(data, toHalf) : new Float32Array(data);

function randomArray(length: number, scale: number): Float32Array {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) out[i] = (Math.random() * 2 - 1) * scale;
  return out;
}

/** G g G^T per (oc, ic), from [cout][cin][3][3] to the matmul's [36][cin][cout]. */
function transformFilter(weights: Float32Array, cin: number, cout: number): Float32Array {
  const out = new Float32Array(36 * cin * cout);
  const g3to6 = (z0: number, z1: number, z2: number) => [
    0.25 * z0,
    (-z0 - z1 - z2) / 6,
    (-z0 + z1 - z2) / 6,
    (z0 + 2 * z1 + 4 * z2) / 24,
    (z0 - 2 * z1 + 4 * z2) / 24,
    z2,
  ];
  for (let oc = 0; oc < cout; oc++) {
    for (let ic = 0; ic < cin; ic++) {
      const base = (oc * cin + ic) * 9;
      const rows = [0, 1, 2].map((r) =>
        g3to6(weights[base + r * 3], weights[base + r * 3 + 1], weights[base + r * 3 + 2]));
      for (let sx = 0; sx < 6; sx++) {
        const col = g3to6(rows[0][sx], rows[1][sx], rows[2][sx]);
        for (let sy = 0; sy < 6; sy++) {
          out[(sy * 6 + sx) * cin * cout + ic * cout + oc] = col[sy];
        }
      }
    }
  }
  return out;
}

/** [36][cin][cout], where entry (sy*6+sx, ic, oc) already holds the transformed filter. */
function directConvReference(c: Conv, input: Float32Array, weights: Float32Array): Float32Array {
  const { batch, board, cin, cout } = c;
  const out = new Float32Array(batch * cout * board * board);
  for (let img = 0; img < batch; img++) {
    for (let oc = 0; oc < cout; oc++) {
      for (let y = 0; y < board; y++) {
        for (let x = 0; x < board; x++) {
          let acc = 0;
          for (let ic = 0; ic < cin; ic++) {
            for (let dy = -1; dy <= 1; dy++) {
              const sy = y + dy;
              if (sy < 0 || sy >= board) continue;
              for (let dx = -1; dx <= 1; dx++) {
                const sx = x + dx;
                if (sx < 0 || sx >= board) continue;
                acc += input[(img * cin + ic) * board * board + sy * board + sx] *
                  weights[((oc * cin + ic) * 3 + dy + 1) * 3 + dx + 1];
              }
            }
          }
          out[(img * cout + oc) * board * board + y * board + x] = acc;
        }
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// GPU plumbing

function makePipeline(device: GPUDevice, code: string, entryPoint: string): GPUComputePipeline {
  return device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code }), entryPoint },
  });
}

function bind(device: GPUDevice, pipeline: GPUComputePipeline, buffers: GPUBuffer[]): GPUBindGroup {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
}

type Pass = { pipeline: GPUComputePipeline; bindGroup: GPUBindGroup; dispatch: [number, number, number] };

/** The whole convolution, ready to encode: transform, batched matmul, untransform. */
function buildWinograd(device: GPUDevice, half: boolean, c: Conv,
                       input: GPUBuffer, filter: GPUBuffer, output: GPUBuffer):
    { passes: Pass[]; scratch: GPUBuffer[] } {
  const width = half ? 2 : 4;
  const mPad = gemmM(c).padded;
  const v = device.createBuffer({ size: 36 * mPad * c.cin * width, usage: GPUBufferUsage.STORAGE });
  const mm = device.createBuffer({ size: 36 * mPad * c.cout * width, usage: GPUBufferUsage.STORAGE });

  const trans = transformShader(half, c, mPad);
  const transform = makePipeline(device, trans, 'transform');
  const untransform = makePipeline(device, trans, 'untransform');
  const matmul = makePipeline(device, matmulShader(half, mPad, c.cout, c.cin), 'main');
  const mDispatch = Math.ceil(gemmM(c).real / 64);
  return {
    passes: [
      { pipeline: transform, bindGroup: bind(device, transform, [input, v]), dispatch: [mPad / 64, c.cin, 1] },
      { pipeline: matmul, bindGroup: bind(device, matmul, [v, filter, mm]), dispatch: [c.cout / TILE, mPad / TILE, 36] },
      { pipeline: untransform, bindGroup: bind(device, untransform, [mm, output]), dispatch: [mDispatch, c.cout, 1] },
    ],
    scratch: [v, mm],
  };
}

function encode(device: GPUDevice, passes: Pass[], iters: number): Promise<undefined> {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  for (let i = 0; i < iters; i++) {
    for (const p of passes) {
      pass.setPipeline(p.pipeline);
      pass.setBindGroup(0, p.bindGroup);
      pass.dispatchWorkgroups(...p.dispatch);
    }
  }
  pass.end();
  device.queue.submit([encoder.finish()]);
  return device.queue.onSubmittedWorkDone();
}

async function timePasses(device: GPUDevice, passes: Pass[], iters: number): Promise<number> {
  await encode(device, passes, 1);
  const t0 = performance.now();
  await encode(device, passes, iters);
  return (performance.now() - t0) / 1000 / iters;
}

// ---------------------------------------------------------------------------
// Correctness

async function verify(device: GPUDevice, half: boolean) {
  const c: Conv = { batch: 2, board: 11, cin: 64, cout: 64 };
  const width = half ? 2 : 4;
  const input = randomArray(c.batch * c.cin * c.board * c.board, 1);
  const weights = randomArray(c.cout * c.cin * 9, 0.25);
  const expected = directConvReference(c, input, weights);

  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const outBytes = expected.length * width;
  const inputBuffer = device.createBuffer({ size: input.length * width, usage });
  const filterBuffer = device.createBuffer({ size: 36 * c.cin * c.cout * width, usage });
  const outputBuffer = device.createBuffer({ size: outBytes, usage: usage | GPUBufferUsage.COPY_SRC });
  device.queue.writeBuffer(inputBuffer, 0, asDeviceData(input, half));
  device.queue.writeBuffer(filterBuffer, 0, asDeviceData(transformFilter(weights, c.cin, c.cout), half));

  const { passes } = buildWinograd(device, half, c, inputBuffer, filterBuffer, outputBuffer);
  await encode(device, passes, 1);

  const readback = device.createBuffer({
    size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(outputBuffer, 0, readback, 0, outBytes);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const raw = readback.getMappedRange().slice(0);
  readback.unmap();
  const got = half
    ? Float32Array.from(new Uint16Array(raw), fromHalf)
    : new Float32Array(raw);

  let maxErr = 0, maxRef = 0;
  for (let i = 0; i < expected.length; i++) {
    maxErr = Math.max(maxErr, Math.abs(got[i] - expected[i]));
    maxRef = Math.max(maxRef, Math.abs(expected[i]));
  }
  const bad = !half && maxErr > 1e-3;
  log(`  ${half ? 'fp16' : 'fp32'} check vs direct conv on CPU: ` +
      `max |err| ${maxErr.toExponential(1)} on outputs reaching ${maxRef.toFixed(1)}` +
      (bad ? '  WRONG' : ''));
  if (bad) throw new Error('winograd disagrees with the direct convolution');
}

// ---------------------------------------------------------------------------
// Benchmark

async function bench(device: GPUDevice, half: boolean, c: Conv) {
  const width = half ? 2 : 4;
  const usage = GPUBufferUsage.STORAGE;
  const flops = nominalFlops(c);
  const m = gemmM(c);
  const iters = Math.max(40, Math.round(1e11 / flops));

  const input = device.createBuffer({ size: c.batch * c.cin * c.board ** 2 * width, usage });
  const filter = device.createBuffer({ size: 36 * c.cin * c.cout * width, usage });
  const output = device.createBuffer({ size: c.batch * c.cout * c.board ** 2 * width, usage });
  const { passes, scratch } = buildWinograd(device, half, c, input, filter, output);

  const whole = await timePasses(device, passes, iters);
  const stages = [];
  for (const p of passes) stages.push(await timePasses(device, [p], iters));

  // The same convolution as one direct matmul over an already-built im2col
  // input, which flatters it: batch*board^2 rows by cin*9, timed in the same
  // sitting so the comparison survives this laptop's load swings.
  const directM = roundUp(c.batch * c.board ** 2, TILE);
  const directK = c.cin * 9;
  const directBuffers = [
    device.createBuffer({ size: directM * directK * width, usage }),
    device.createBuffer({ size: directK * c.cout * width, usage }),
    device.createBuffer({ size: directM * c.cout * width, usage }),
  ];
  const directPipeline = makePipeline(device, matmulShader(half, directM, c.cout, directK), 'main');
  const direct: Pass = {
    pipeline: directPipeline,
    bindGroup: bind(device, directPipeline, directBuffers),
    dispatch: [c.cout / TILE, directM / TILE, 1],
  };
  const directSeconds = await timePasses(device, [direct], iters);

  const label = half ? 'fp16' : 'fp32';
  const gflops = (s: number) => (flops / s / 1e9).toFixed(0).padStart(4);
  log(`  ${label} winograd ${(1000 * whole).toFixed(2).padStart(6)} ms, ${gflops(whole)} eff GFLOP/s` +
      ` (transform ${(1000 * stages[0]).toFixed(2)}, matmul ${(1000 * stages[1]).toFixed(2)},` +
      ` untransform ${(1000 * stages[2]).toFixed(2)})`);
  log(`  ${label} direct   ${(1000 * directSeconds).toFixed(2).padStart(6)} ms, ${gflops(directSeconds)} GFLOP/s` +
      `  -> winograd is ${(directSeconds / whole).toFixed(2)}x`);

  for (const b of [input, filter, output, ...scratch, ...directBuffers]) b.destroy();
  return { whole, direct: directSeconds, mPad: m.padded, mReal: m.real };
}

// ---------------------------------------------------------------------------

async function run() {
  (document.getElementById('go') as HTMLButtonElement).disabled = true;
  logEl.textContent = '';
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('no WebGPU adapter (needs a secure context)');
    const half = adapter.features.has('shader-f16');
    log(`adapter: ${adapter.info?.vendor} ${adapter.info?.architecture}, shader-f16: ${half}`);
    const device = await adapter.requestDevice({ requiredFeatures: half ? ['shader-f16'] : [] });
    device.addEventListener('uncapturederror',
      (e) => log(`ERROR: ${(e as GPUUncapturedErrorEvent).error.message}`));

    await verify(device, false);
    if (half) await verify(device, true);

    for (const shape of SHAPES) {
      const m = gemmM(shape);
      log(`${shape.board}x${shape.board} trunk conv, batch ${shape.batch}` +
          ` (${m.real} tiles, matmul rows padded to ${m.padded}):`);
      await bench(device, false, shape);
      if (half) await bench(device, true, shape);
    }
    log('done');
  } catch (e) {
    log(`ERROR: ${e}`);
  } finally {
    (document.getElementById('go') as HTMLButtonElement).disabled = false;
  }
}

document.getElementById('go')!.addEventListener('click', run);
