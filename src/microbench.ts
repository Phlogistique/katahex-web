// How fast can this GPU go from a browser at all? A tiled matrix multiply,
// written by hand in WGSL, sets the ceiling that an inference runtime's
// convolutions are measured against: it is the same arithmetic a convolution
// turns into, with none of a framework's dispatch or layout decisions in the way.

const logEl = document.getElementById('log')!;
const log = (s: string) => { logEl.textContent += s + '\n'; console.log(s); };

const TILE = 64;            // output tile per workgroup
const TILE_K = 16;
const PER_THREAD = 4;       // each of the 16x16 threads computes 4x4 outputs

type Shape = { m: number; n: number; k: number; label: string };

// A square shape for the unobstructed peak, and the shape one 192-channel 3x3
// convolution of the trunk becomes at batch 64: 64*11*11 rows, 192*3*3 deep.
const SHAPES: Shape[] = [
  { m: 2048, n: 2048, k: 2048, label: 'square 2048' },
  { m: 7744, n: 192, k: 1728, label: 'trunk conv, batch 64' },
];

function shader(half: boolean, { m, n, k }: Shape): string {
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
  let rowBase = wid.y * ${TILE}u + lid.y * ${PER_THREAD}u;
  let colBase = wid.x * ${TILE}u + lid.x * ${PER_THREAD}u;
  let tid = lid.y * 16u + lid.x;
  var acc : array<array<${t}, ${PER_THREAD}>, ${PER_THREAD}>;

  for (var k0 = 0u; k0 < K; k0 += ${TILE_K}u) {
    for (var i = 0u; i < 4u; i++) {
      let idx = tid + i * 256u;
      aTile[idx] = a[(wid.y * ${TILE}u + idx / ${TILE_K}u) * K + k0 + idx % ${TILE_K}u];
      bTile[idx] = b[(k0 + idx / ${TILE}u) * N + wid.x * ${TILE}u + idx % ${TILE}u];
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
      c[(rowBase + i) * N + colBase + j] = acc[i][j];
    }
  }
}`;
}

/** Runs the shader on all-ones matrices, where every output must equal K. */
async function verify(device: GPUDevice, half: boolean) {
  const shape: Shape = { m: 64, n: 64, k: 64, label: 'check' };
  const width = half ? 2 : 4;
  const bytes = 64 * 64 * width;
  const ones = half
    ? new Uint16Array(64 * 64).fill(0x3c00)   // 1.0 as float16
    : new Float32Array(64 * 64).fill(1);
  const make = (extra: number) =>
    device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | extra });
  const buffers = [make(GPUBufferUsage.COPY_DST), make(GPUBufferUsage.COPY_DST),
                   make(GPUBufferUsage.COPY_SRC)];
  device.queue.writeBuffer(buffers[0], 0, ones);
  device.queue.writeBuffer(buffers[1], 0, ones);

  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: shader(half, shape) }), entryPoint: 'main' },
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  }));
  pass.dispatchWorkgroups(1, 1);
  pass.end();
  const readback = device.createBuffer({
    size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
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
  const wrong = values.filter((v) => v !== 64).length;
  log(`  ${half ? 'fp16' : 'fp32'} check: ${wrong ? `${wrong} of ${values.length} wrong` : 'all outputs = K'}`);
}

async function bench(device: GPUDevice, half: boolean, shape: Shape, iters: number) {
  const width = half ? 2 : 4;
  const usage = GPUBufferUsage.STORAGE;
  const buffers = [
    device.createBuffer({ size: shape.m * shape.k * width, usage }),
    device.createBuffer({ size: shape.k * shape.n * width, usage }),
    device.createBuffer({ size: shape.m * shape.n * width, usage }),
  ];
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: shader(half, shape) }), entryPoint: 'main' },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });

  const dispatch = (count: number) => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    for (let i = 0; i < count; i++) pass.dispatchWorkgroups(shape.n / TILE, shape.m / TILE);
    pass.end();
    device.queue.submit([encoder.finish()]);
    return device.queue.onSubmittedWorkDone();
  };

  await dispatch(1);
  const t0 = performance.now();
  await dispatch(iters);
  const seconds = (performance.now() - t0) / 1000;

  const gflops = (2 * shape.m * shape.n * shape.k * iters) / seconds / 1e9;
  log(`  ${(half ? 'fp16' : 'fp32').padEnd(5)} ${(1000 * seconds / iters).toFixed(2).padStart(7)} ms each` +
      `, ${gflops.toFixed(0).padStart(4)} GFLOP/s`);
  for (const buffer of buffers) buffer.destroy();
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

    await verify(device, false);
    if (half) await verify(device, true);

    for (const shape of SHAPES) {
      log(`${shape.label}: ${shape.m}x${shape.k} times ${shape.k}x${shape.n}`);
      const iters = Math.max(4, Math.round(4e10 / (2 * shape.m * shape.n * shape.k)));
      await bench(device, false, shape, iters);
      if (half) await bench(device, true, shape, iters);
    }
    log('done');
  } catch (e) {
    log(`ERROR: ${e}`);
  } finally {
    (document.getElementById('go') as HTMLButtonElement).disabled = false;
  }
}

document.getElementById('go')!.addEventListener('click', run);
