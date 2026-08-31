// The KataGo net on raw WebGPU, written by hand because onnxruntime's
// convolution leaves half the speed on the table (see README "Winograd,
// measured"). Activations are [rows, channels] matrices: NHWC flattened, with
// channels padded to a multiple of 16 (the matmul's K tile) and rows to 64 (its
// M tile). Every kernel writes zeros into the channel padding, so a padded
// column entering a matmul contributes nothing; padded rows can hold garbage,
// which stays in padded rows because every op either preserves rows or reduces
// over the real positions of one image.
//
// 1x1 convolutions and dense layers are one matmul each. 3x3 convolutions are
// Winograd F(4x4, 3x3): input transform, a matmul batched over the 36 tile
// positions, output transform, with the filter transformed on the CPU at load
// time. The transform matrices are KataGo's (cpp/neuralnet/openclkernels.cpp).
//
// The whole forward pass for one batch size is precompiled into a Plan: a flat
// list of dispatches with every shape baked into its shader, plus the buffers
// they run on. evaluate() writes the inputs, replays the plan, reads back the
// four outputs.

import type {
  ActivationKind,
  ParsedBatchNorm,
  ParsedConv2d,
  ParsedMatMul,
} from '../vendor/binModelParser';
import type { ParsedKataGoModelV8, ParsedTrunkBlock } from '../vendor/modelV8';

const TILE = 64;

const roundUp = (v: number, m: number) => Math.ceil(v / m) * m;
const padC = (c: number) => roundUp(c, 16);

// ---------------------------------------------------------------------------
// fp16 conversion

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
  return (h >> 15 ? -1 : 1) *
    (exponent ? 2 ** (exponent - 15) * (1 + mantissa / 1024) : 2 ** -14 * (mantissa / 1024));
}

// ---------------------------------------------------------------------------
// Shaders

/**
 * How one matmul pipeline is shaped. A workgroup of wgX by wgY threads computes
 * `wgY * rows` output rows by 64 columns; each thread holds `rows * cols4` vec4
 * accumulators and stages `tileK` of the reduction at a time.
 */
export type MatmulTile = { wgX: number; wgY: number; rows: number; cols4: number; tileK: number };

/** 64 rows x 64 columns per workgroup, 4x4 outputs per thread. */
const WIDE: Omit<MatmulTile, 'tileK'> = { wgX: 16, wgY: 16, rows: 4, cols4: 1 };
/** Half the rows, for an m that a 64-row tile would pad heavily. */
const NARROW: Omit<MatmulTile, 'tileK'> = { wgX: 16, wgY: 8, rows: 4, cols4: 1 };

/**
 * The pipeline shape for a given matmul. A wider k tile halves the barriers but
 * needs k to divide by it, which the head layers' 48 and 144 channels do not.
 */
export function matmulTile(m: number, n: number, k: number): MatmulTile {
  return { ...(m % TILE === 0 ? WIDE : NARROW), tileK: k % 32 === 0 ? 32 : 16 };
}

/**
 * Tiled matmul, C[m,n] = A[m,k] B[k,n], batched over workgroup z when
 * dispatched with depth > 1 (the Winograd tile positions). Everything moves in
 * vec4: four columns of B and four steps of the reduction of A per load, so the
 * inner loop issues a quarter of the shared-memory reads a scalar one would and
 * every lane reads a whole 8-byte bank. m must be a multiple of the tile's row
 * count and k of its k tile; n only has to be a multiple of 16, at the price of
 * a guarded store (`colGuard`) when it is not a multiple of 64.
 *
 * The reduction is emitted fully unrolled so the accumulators stay in registers:
 * a dynamically indexed array of them spills to scratch memory.
 */
export function matmulShader(t: string, m: number, n: number, k: number,
                             tile: MatmulTile = matmulTile(m, n, k)): string {
  const { wgX, wgY, rows, cols4, tileK } = tile;
  const tileM = wgY * rows, threads = wgX * wgY;
  const nq = wgX * cols4;      // vec4 columns a workgroup covers, TILE / 4
  const kq = tileK / 4;        // vec4s of the reduction staged per step
  const n4 = n / 4, k4 = k / 4;
  const aCount = tileM * kq, bCount = tileK * nq;
  if (m % tileM || k % tileK || n % 16 || nq * 4 !== TILE ||
      aCount % threads || bCount % threads) throw new Error(`matmul ${m}x${n}x${k} does not tile`);

  const v4 = `vec4<${t}>`;
  const list = <T,>(count: number, f: (i: number) => T) => Array.from({ length: count }, (_, i) => f(i));
  const acc = (i: number, p: number) => `acc${i}_${p}`;
  const stage = (name: string, count: number, expr: string) =>
    list(count / threads, (s) => `    { let idx = tid + ${s * threads}u; ${name}[idx] = ${expr}; }`);

  // One reduction step: each thread reads a vec4 of A per row and a vec4 of B
  // per column group, then fans them out over the four k lanes of the vec4.
  const step = (q: number) => [
    '    {',
    ...list(rows, (i) => `      let av${i} = aTile[aRead + ${i * kq + q}u];`),
    ...list(4, (c) => list(cols4, (p) =>
      `      let bv${c}_${p} = bTile[bRead + ${(q * 4 + c) * nq + p}u];`)).flat(),
    ...list(rows, (i) => list(4, (c) => list(cols4, (p) =>
      `      ${acc(i, p)} += av${i}.${'xyzw'[c]} * bv${c}_${p};`)).flat()).flat(),
    '    }',
  ];

  return `${t === 'f16' ? 'enable f16;\n' : ''}
@group(0) @binding(0) var<storage, read> a : array<${v4}>;
@group(0) @binding(1) var<storage, read> b : array<${v4}>;
@group(0) @binding(2) var<storage, read_write> c : array<${v4}>;
var<workgroup> aTile : array<${v4}, ${aCount}>;
var<workgroup> bTile : array<${v4}, ${bCount}>;

@compute @workgroup_size(${wgX}, ${wgY})
fn main(@builtin(local_invocation_id) lid : vec3<u32>,
        @builtin(workgroup_id) wid : vec3<u32>) {
  let tid = lid.y * ${wgX}u + lid.x;
  let aOff = wid.z * ${m * k4}u + wid.y * ${tileM * k4}u;
  let bOff = wid.z * ${k * n4}u + wid.x * ${nq}u;
  let colq = wid.x * ${nq}u + lid.x * ${cols4}u;
  let cOff = wid.z * ${m * n4}u + (wid.y * ${tileM}u + lid.y * ${rows}u) * ${n4}u + colq;
  let aRead = lid.y * ${rows * kq}u;
  let bRead = lid.x * ${cols4}u;
  // Explicitly zeroed: this driver does not re-initialize a var per iteration,
  // and these have to survive the k loop anyway.
${list(rows, (i) => list(cols4, (p) => `  var ${acc(i, p)} = ${v4}();`).join('')).join('\n')}

  for (var kb = 0u; kb < ${k4}u; kb += ${kq}u) {
${stage('aTile', aCount, `a[aOff + (idx / ${kq}u) * ${k4}u + kb + idx % ${kq}u]`).join('\n')}
${stage('bTile', bCount, `b[bOff + (kb * 4u + idx / ${nq}u) * ${n4}u + idx % ${nq}u]`).join('\n')}
    workgroupBarrier();
${list(kq, step).flat().join('\n')}
    workgroupBarrier();
  }
${list(rows, (i) => list(cols4, (p) => n % TILE === 0
    ? `  c[cOff + ${i * n4 + p}u] = ${acc(i, p)};`
    : `  if (colq + ${p}u < ${n4}u) { c[cOff + ${i * n4 + p}u] = ${acc(i, p)}; }`).join('\n')).join('\n')}
}`;
}

/**
 * The Winograd transforms, one thread per (tile, channel). The arithmetic is
 * f32 whatever the storage: the transform coefficients stretch fp16 badly and
 * both kernels are memory-bound anyway.
 */
function winogradPrelude(t: string, board: number, batch: number, mPad: number,
                         cin: number, cout: number): string {
  const tiles = Math.ceil(board / 4);
  return `${t === 'f16' ? 'enable f16;\n' : ''}
const B = ${board}; const TX = ${tiles}u;
const MREAL = ${batch * tiles * tiles}u; const MPAD = ${mPad}u;
const CIN = ${cin}u; const COUT = ${cout}u;
`;
}

/** B^T d B per 6x6 input tile: [rows, cinPad] NHWC to the matmul's [36][mPad][cinPad]. */
function winogradTransformShader(t: string, board: number, batch: number,
                                 cinPad: number, mPad: number): string {
  return winogradPrelude(t, board, batch, mPad, cinPad, 0) + `
@group(0) @binding(0) var<storage, read> src : array<${t}>;
@group(0) @binding(1) var<storage, read_write> dst : array<${t}>;

@compute @workgroup_size(64)
fn transform(@builtin(global_invocation_id) gid : vec3<u32>) {
  // Threads run along the channels, which sit next to each other in memory,
  // so the 36 loads and stores each cover whole cache lines per workgroup.
  let ic = gid.x;
  let m = gid.y;
  if (ic >= CIN) { return; }
  let img = m / (TX * TX);
  let ty = (m / TX) % TX;
  let tx = m % TX;

  var w : array<array<f32, 6>, 6>;
  for (var sy = 0u; sy < 6u; sy++) {
    let y = i32(ty * 4u + sy) - 1;
    for (var sx = 0u; sx < 6u; sx++) {
      let x = i32(tx * 4u + sx) - 1;
      var value = 0.0;
      if (y >= 0 && y < B && x >= 0 && x < B) {
        value = f32(src[(img * u32(B * B) + u32(y * B + x)) * CIN + ic]);
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
      dst[(sy * 6u + sx) * MPAD * CIN + m * CIN + ic] = ${t}(w[sy][sx]);
    }
  }
}`;
}

/** A^T m A per output tile, back to [rows, coutPad] NHWC. */
function winogradUntransformShader(t: string, board: number, batch: number,
                                   coutPad: number, mPad: number): string {
  return winogradPrelude(t, board, batch, mPad, 0, coutPad) + `
@group(0) @binding(0) var<storage, read> src : array<${t}>;
@group(0) @binding(1) var<storage, read_write> dst : array<${t}>;

@compute @workgroup_size(64)
fn untransform(@builtin(global_invocation_id) gid : vec3<u32>) {
  let oc = gid.x;
  let m = gid.y;
  if (oc >= COUT) { return; }
  let img = m / (TX * TX);
  let ty = (m / TX) % TX;
  let tx = m % TX;

  var w : array<array<f32, 6>, 6>;
  for (var sy = 0u; sy < 6u; sy++) {
    for (var sx = 0u; sx < 6u; sx++) {
      w[sy][sx] = f32(src[(sy * 6u + sx) * MPAD * COUT + m * COUT + oc]);
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
      dst[(img * u32(B * B) + y * u32(B) + x) * COUT + oc] = ${t}(w[sy][sx]);
    }
  }
}`;
}

const ACT_WGSL: Record<ActivationKind, string> = {
  identity: 'let y = v;',
  relu: 'let y = max(v, 0.0);',
  // mish(x) = x * tanh(softplus(x)), in f32 so fp16 storage cannot overflow exp.
  mish: 'let sp = select(log(1.0 + exp(v)), v, v > 20.0); let y = v * tanh(sp);',
};

/** y = act(x * scale[c] + bias[c]) over a whole [rows, cpad] buffer, math in f32. */
function bnActShader(t: string, elems: number, cpad: number, act: ActivationKind): string {
  return `${t === 'f16' ? 'enable f16;\n' : ''}
@group(0) @binding(0) var<storage, read> x : array<${t}>;
@group(0) @binding(1) var<storage, read> scale : array<${t}>;
@group(0) @binding(2) var<storage, read> bias : array<${t}>;
@group(0) @binding(3) var<storage, read_write> out : array<${t}>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= ${elems}u) { return; }
  let c = i % ${cpad}u;
  let v = f32(x[i]) * f32(scale[c]) + f32(bias[c]);
  ${ACT_WGSL[act]}
  out[i] = ${t}(y);
}`;
}

/** out = a + b, elementwise. */
function addShader(t: string, elems: number): string {
  return `${t === 'f16' ? 'enable f16;\n' : ''}
@group(0) @binding(0) var<storage, read> a : array<${t}>;
@group(0) @binding(1) var<storage, read> b : array<${t}>;
@group(0) @binding(2) var<storage, read_write> out : array<${t}>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= ${elems}u) { return; }
  out[i] = a[i] + b[i];
}`;
}

/** out = x + bias[image, c]: a per-image channel bias broadcast over positions. */
function addBiasShader(t: string, batch: number, hw: number, cpad: number): string {
  return `${t === 'f16' ? 'enable f16;\n' : ''}
@group(0) @binding(0) var<storage, read> x : array<${t}>;
@group(0) @binding(1) var<storage, read> bias : array<${t}>;
@group(0) @binding(2) var<storage, read_write> out : array<${t}>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= ${batch * hw * cpad}u) { return; }
  let c = i % ${cpad}u;
  let img = i / ${hw * cpad}u;
  out[i] = x[i] + bias[img * ${cpad}u + c];
}`;
}

/**
 * KataGo global pooling over one image's positions, one thread per
 * (image, channel). Writes [batch, 3*cpad]: mean, mean*f1, then max (gpool)
 * or mean*f2 (the value head's pooling).
 */
function gpoolShader(t: string, kind: 'gpool' | 'value', batch: number, hw: number,
                     cpad: number, f1: number, f2: number): string {
  const third = kind === 'gpool' ? 'mx' : `mean * ${f2}`;
  return `${t === 'f16' ? 'enable f16;\n' : ''}
@group(0) @binding(0) var<storage, read> x : array<${t}>;
@group(0) @binding(1) var<storage, read_write> out : array<${t}>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let img = gid.y;
  let c = gid.x;
  if (c >= ${cpad}u || img >= ${batch}u) { return; }
  var sum = 0.0;
  var mx = f32(x[img * ${hw * cpad}u + c]);
  for (var p = 0u; p < ${hw}u; p++) {
    let v = f32(x[(img * ${hw}u + p) * ${cpad}u + c]);
    sum += v;
    mx = max(mx, v);
  }
  let mean = sum / ${hw}.0;
  let base = img * ${3 * cpad}u + c;
  out[base] = ${t}(mean);
  out[base + ${cpad}u] = ${t}(mean * ${f1});
  out[base + ${2 * cpad}u] = ${t}(${third});
}`;
}

// ---------------------------------------------------------------------------
// Weight preparation

/** [cinPad, coutPad] with zero padding, from the file's [kY,kX,inC,outC] (1x1) or [inC,outC]. */
function padMatrix(weights: Float32Array, cin: number, cout: number): Float32Array {
  const kp = padC(cin), np = padC(cout);
  const out = new Float32Array(kp * np);
  for (let ic = 0; ic < cin; ic++) {
    out.set(weights.subarray(ic * cout, (ic + 1) * cout), ic * np);
  }
  return out;
}

/** G g G^T per filter, from [kY,kX,inC,outC] to the batched matmul's [36][cinPad][coutPad]. */
function winogradFilter(conv: ParsedConv2d): Float32Array {
  const { inChannels: cin, outChannels: cout, weights } = conv;
  const kp = padC(cin), np = padC(cout);
  const out = new Float32Array(36 * kp * np);
  const g3to6 = (z0: number, z1: number, z2: number) => [
    0.25 * z0,
    (-z0 - z1 - z2) / 6,
    (-z0 + z1 - z2) / 6,
    (z0 + 2 * z1 + 4 * z2) / 24,
    (z0 - 2 * z1 + 4 * z2) / 24,
    z2,
  ];
  const rows: number[][] = [[], [], []];
  for (let ic = 0; ic < cin; ic++) {
    for (let oc = 0; oc < cout; oc++) {
      const at = (ky: number, kx: number) => weights[((ky * 3 + kx) * cin + ic) * cout + oc];
      for (let r = 0; r < 3; r++) rows[r] = g3to6(at(r, 0), at(r, 1), at(r, 2));
      for (let sx = 0; sx < 6; sx++) {
        const col = g3to6(rows[0][sx], rows[1][sx], rows[2][sx]);
        for (let sy = 0; sy < 6; sy++) {
          out[(sy * 6 + sx) * kp * np + ic * np + oc] = col[sy];
        }
      }
    }
  }
  return out;
}

function padVector(values: Float32Array, c: number): Float32Array {
  const out = new Float32Array(padC(c));
  out.set(values);
  return out;
}

// ---------------------------------------------------------------------------
// The model

type Pass = { label: string; pipeline: GPUComputePipeline; bindGroup: GPUBindGroup; dispatch: [number, number, number] };

/** An activation matrix: `rows` real rows in a buffer of rowsPad x cpad. */
type Tensor = { buffer: GPUBuffer; rows: number; rowsPad: number; cpad: number };

type OutputName = 'policy' | 'policyPass' | 'value' | 'scoreValue';

type Plan = {
  passes: Pass[];
  spatialIn: GPUBuffer;
  globalIn: GPUBuffer;
  readbackSize: number;
  readbackLayout: { name: OutputName; offset: number; tensor: Tensor }[];
  /** One per evaluation in flight; evaluate takes one and puts it back. */
  readouts: Readout[];
};

/** The buffers one in-flight evaluation reads its results through. */
type Readout = {
  readback: GPUBuffer;
  querySet?: GPUQuerySet;
  queryResolve?: GPUBuffer;
  queryReadback?: GPUBuffer;
};

export type NetOutputs = {
  /** [batch][boardSize*boardSize] policy logits. */
  policy: Float32Array[];
  policyPass: Float32Array;      // [batch]
  value: Float32Array[];         // [batch][3]
  scoreValue: Float32Array[];    // [batch][scoreValueChannels]
};

export class KataGoWebGpuModel {
  private readonly device: GPUDevice;
  private readonly parsed: ParsedKataGoModelV8;
  private readonly size: number;
  private readonly half: boolean;
  private readonly t: string;
  private readonly width: number;
  private readonly weights = new Map<string, GPUBuffer>();
  private readonly pipelines = new Map<string, GPUComputePipeline>();
  private readonly plans = new Map<number, Plan>();
  private readonly activations: GPUBuffer[] = [];

  /** Wall time spent in each stage of evaluate, cumulative nanoseconds. With
   * two evaluations in flight the gpu spans overlap, so they can sum past the
   * wall clock; read them from a run with one. */
  readonly stages = { pack: 0, gpu: 0, convert: 0 };
  /** Per-kernel GPU time, filled only while `profile` is on. */
  readonly gpuByLabel = new Map<string, { nanos: number; count: number }>();
  /** Time every dispatch with timestamp queries. Needs the device created with
   * 'timestamp-query', costs one compute pass per dispatch, and reads garbage
   * timings unless Chrome runs with --enable-webgpu-developer-features
   * (timestamps are quantized to uselessness otherwise). */
  profile = false;

  constructor(device: GPUDevice, parsed: ParsedKataGoModelV8, size: number, half: boolean) {
    if (parsed.metaEncoderVersion !== 0) throw new Error('meta encoder not supported');
    this.device = device;
    this.parsed = parsed;
    this.size = size;
    this.half = half;
    this.t = half ? 'f16' : 'f32';
    this.width = half ? 2 : 4;
    this.uploadWeights();
  }

  // -- weights --------------------------------------------------------------

  private upload(name: string, data: Float32Array): void {
    const device = this.half ? Uint16Array.from(data, toHalf) : data;
    const buffer = this.device.createBuffer({
      label: name,
      size: device.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buffer, 0, device as Uint16Array<ArrayBuffer> | Float32Array<ArrayBuffer>);
    this.weights.set(name, buffer);
  }

  private uploadConv(name: string, conv: ParsedConv2d): void {
    if (conv.kernelX === 1 && conv.kernelY === 1) {
      this.upload(name, padMatrix(conv.weights, conv.inChannels, conv.outChannels));
    } else if (conv.kernelX === 3 && conv.kernelY === 3) {
      this.upload(name, winogradFilter(conv));
    } else {
      throw new Error(`unsupported kernel ${conv.kernelY}x${conv.kernelX}`);
    }
  }

  private uploadMatMul(name: string, mm: ParsedMatMul): void {
    this.upload(name, padMatrix(mm.weights, mm.inChannels, mm.outChannels));
  }

  private uploadBn(name: string, bn: ParsedBatchNorm): void {
    this.upload(`${name}.scale`, padVector(bn.mergedScale, bn.channels));
    this.upload(`${name}.bias`, padVector(bn.mergedBias, bn.channels));
  }

  private uploadBlock(name: string, block: ParsedTrunkBlock): void {
    this.uploadBn(`${name}.pre`, block.preBN);
    if (block.kind === 'ordinary') {
      this.uploadConv(`${name}.w1`, block.w1);
      this.uploadBn(`${name}.mid`, block.midBN);
      this.uploadConv(`${name}.w2`, block.w2);
    } else if (block.kind === 'gpool') {
      this.uploadConv(`${name}.w1a`, block.w1a);
      this.uploadConv(`${name}.w1b`, block.w1b);
      this.uploadBn(`${name}.gpool`, block.gpoolBN);
      this.uploadMatMul(`${name}.w1r`, block.w1r);
      this.uploadBn(`${name}.mid`, block.midBN);
      this.uploadConv(`${name}.w2`, block.w2);
    } else {
      this.uploadConv(`${name}.preConv`, block.preConv);
      block.blocks.forEach((b, i) => this.uploadBlock(`${name}.${i}`, b));
      this.uploadBn(`${name}.post`, block.postBN);
      this.uploadConv(`${name}.postConv`, block.postConv);
    }
  }

  private uploadWeights(): void {
    const { trunk, policy, value } = this.parsed;
    this.uploadConv('conv1', trunk.conv1);
    this.uploadMatMul('ginput', trunk.ginput);
    trunk.blocks.forEach((b, i) => this.uploadBlock(`b${i}`, b));
    this.uploadBn('tip', trunk.tipBN);
    this.uploadConv('p1', policy.p1);
    this.uploadConv('g1', policy.g1);
    this.uploadBn('g1bn', policy.g1BN);
    this.uploadMatMul('gpoolToBias', policy.gpoolToBias);
    this.uploadBn('p1bn', policy.p1BN);
    this.uploadConv('p2', policy.p2);
    this.uploadMatMul('passMul', policy.passMul);
    this.uploadConv('v1', value.v1);
    this.uploadBn('v1bn', value.v1BN);
    this.uploadMatMul('v2', value.v2);
    this.upload('v2.bias', padVector(value.v2Bias.weights, value.v2Bias.channels));
    this.uploadMatMul('v3', value.v3);
    this.upload('v3.bias', padVector(value.v3Bias.weights, value.v3Bias.channels));
    this.uploadMatMul('sv3', value.sv3);
    this.upload('sv3.bias', padVector(value.sv3Bias.weights, value.sv3Bias.channels));
    // Identity scale for plain bias+activation steps, wide enough for any head.
    this.upload('ones', new Float32Array(padC(value.v2Bias.channels)).fill(1));
  }

  private weight(name: string): GPUBuffer {
    const buffer = this.weights.get(name);
    if (!buffer) throw new Error(`no weight ${name}`);
    return buffer;
  }

  // -- plan building --------------------------------------------------------

  private pipeline(code: string, entryPoint: string): GPUComputePipeline {
    const key = entryPoint + code;
    let pipeline = this.pipelines.get(key);
    if (!pipeline) {
      pipeline = this.device.createComputePipeline({
        layout: 'auto',
        compute: { module: this.device.createShaderModule({ code }), entryPoint },
      });
      this.pipelines.set(key, pipeline);
    }
    return pipeline;
  }

  private buildPlan(batch: number): Plan {
    const device = this.device;
    const t = this.t;
    const size = this.size;
    const hw = size * size;
    const rows = batch * hw;
    const rowsPad = roundUp(rows, TILE);
    const tiles = Math.ceil(size / 4) ** 2;
    const wRows = batch * tiles;
    const wRowsPad = roundUp(wRows, TILE);
    const nPad = roundUp(batch, TILE);

    const passes: Pass[] = [];

    // Buffers pooled by shape; a released buffer is only reused at the same
    // shape, which keeps the channel-padding zeros valid.
    const pool = new Map<string, GPUBuffer[]>();
    const acquire = (r: number, rPad: number, cpad: number): Tensor => {
      const key = `${rPad}x${cpad}`;
      const free = pool.get(key) ?? [];
      let buffer = free.pop();
      if (!buffer) {
        buffer = device.createBuffer({
          label: `act ${key}`,
          size: rPad * cpad * this.width,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        this.activations.push(buffer);
      }
      pool.set(key, free);
      return { buffer, rows: r, rowsPad: rPad, cpad };
    };
    const release = (...tensors: Tensor[]) => {
      for (const t of tensors) pool.get(`${t.rowsPad}x${t.cpad}`)!.push(t.buffer);
    };
    const act = (cpad: number) => acquire(rows, rowsPad, cpad);
    const vec = (cpad: number) => acquire(batch, nPad, cpad);

    const run = (label: string, pipeline: GPUComputePipeline, buffers: GPUBuffer[], dispatch: [number, number, number]) => {
      passes.push({
        label,
        pipeline,
        bindGroup: device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
        }),
        dispatch,
      });
    };

    /** Workgroups a [m, n] output needs from the tile its shader picked. */
    const grid = (m: number, n: number, k: number): [number, number] => {
      const tile = matmulTile(m, n, k);
      return [Math.ceil(n / TILE), m / (tile.wgY * tile.rows)];
    };

    /** out = x @ weightName, [x.rowsPad, kpad] x [kpad, npad]. */
    const matmul = (x: Tensor, weightName: string, nOut: number): Tensor => {
      const npad = padC(nOut);
      const out = acquire(x.rows, x.rowsPad, npad);
      run(`matmul ${x.rowsPad}x${npad}x${x.cpad}`,
          this.pipeline(matmulShader(t, x.rowsPad, npad, x.cpad), 'main'),
          [x.buffer, this.weight(weightName), out.buffer],
          [...grid(x.rowsPad, npad, x.cpad), 1]);
      return out;
    };

    /** Winograd 3x3 convolution over the board. */
    const conv3x3 = (x: Tensor, weightName: string, nOut: number): Tensor => {
      const npad = padC(nOut);
      const v = acquire(36 * wRowsPad, 36 * wRowsPad, x.cpad);
      run(`wino-transform c${x.cpad}`,
          this.pipeline(winogradTransformShader(t, size, batch, x.cpad, wRowsPad), 'transform'),
          [x.buffer, v.buffer], [Math.ceil(x.cpad / 64), wRows, 1]);
      const mm = acquire(36 * wRowsPad, 36 * wRowsPad, npad);
      run(`wino-matmul 36x${wRowsPad}x${npad}x${x.cpad}`,
          this.pipeline(matmulShader(t, wRowsPad, npad, x.cpad), 'main'),
          [v.buffer, this.weight(weightName), mm.buffer],
          [...grid(wRowsPad, npad, x.cpad), 36]);
      release(v);
      const out = act(npad);
      run(`wino-untransform c${npad}`,
          this.pipeline(winogradUntransformShader(t, size, batch, npad, wRowsPad), 'untransform'),
          [mm.buffer, out.buffer], [Math.ceil(npad / 64), wRows, 1]);
      release(mm);
      return out;
    };

    const conv = (x: Tensor, weightName: string, desc: ParsedConv2d): Tensor =>
      desc.kernelX === 1 ? matmul(x, weightName, desc.outChannels) : conv3x3(x, weightName, desc.outChannels);

    const bnAct = (x: Tensor, bnName: string, kind: ActivationKind): Tensor => {
      const out = acquire(x.rows, x.rowsPad, x.cpad);
      const elems = x.rowsPad * x.cpad;
      run(`bnAct c${x.cpad}`, this.pipeline(bnActShader(t, elems, x.cpad, kind), 'main'),
          [x.buffer, this.weight(`${bnName}.scale`), this.weight(`${bnName}.bias`), out.buffer],
          [Math.ceil(elems / 256), 1, 1]);
      return out;
    };

    /** x + bias with scale 1: the head layers' bias + activation. */
    const biasAct = (x: Tensor, biasName: string, kind: ActivationKind): Tensor => {
      const out = acquire(x.rows, x.rowsPad, x.cpad);
      const elems = x.rowsPad * x.cpad;
      run(`bnAct c${x.cpad}`, this.pipeline(bnActShader(t, elems, x.cpad, kind), 'main'),
          [x.buffer, this.weight('ones'), this.weight(biasName), out.buffer],
          [Math.ceil(elems / 256), 1, 1]);
      return out;
    };

    /** out = a + b, into a fresh tensor of a's shape. */
    const add = (a: Tensor, b: Tensor): Tensor => {
      const out = acquire(a.rows, a.rowsPad, a.cpad);
      const elems = a.rowsPad * a.cpad;
      run('add', this.pipeline(addShader(t, elems), 'main'),
          [a.buffer, b.buffer, out.buffer], [Math.ceil(elems / 256), 1, 1]);
      return out;
    };

    /** x [rows, cpad] plus a per-image [batch, cpad] channel bias. */
    const addBias = (x: Tensor, bias: Tensor): Tensor => {
      const out = acquire(x.rows, x.rowsPad, x.cpad);
      run('addBias', this.pipeline(addBiasShader(t, batch, hw, x.cpad), 'main'),
          [x.buffer, bias.buffer, out.buffer], [Math.ceil(batch * hw * x.cpad / 256), 1, 1]);
      return out;
    };

    const factor1 = (size - 14) * 0.1;
    const factor2 = (size - 14) ** 2 * 0.01 - 0.1;
    const gpool = (x: Tensor, kind: 'gpool' | 'value'): Tensor => {
      const out = acquire(batch, nPad, 3 * x.cpad);
      run(`gpool-${kind}`, this.pipeline(gpoolShader(t, kind, batch, hw, x.cpad, factor1, factor2), 'main'),
          [x.buffer, out.buffer], [Math.ceil(x.cpad / 64), batch, 1]);
      return out;
    };

    const applyBlocks = (trunk: Tensor, blocks: ParsedTrunkBlock[], prefix: string): Tensor => {
      blocks.forEach((block, i) => {
        const name = prefix ? `${prefix}.${i}` : `b${i}`;
        const a = bnAct(trunk, `${name}.pre`, block.preActivation);
        let d: Tensor;
        if (block.kind === 'ordinary') {
          const b = conv(a, `${name}.w1`, block.w1);
          release(a);
          const c = bnAct(b, `${name}.mid`, block.midActivation);
          release(b);
          d = conv(c, `${name}.w2`, block.w2);
          release(c);
        } else if (block.kind === 'gpool') {
          const regular = conv(a, `${name}.w1a`, block.w1a);
          const gp = conv(a, `${name}.w1b`, block.w1b);
          release(a);
          const gp2 = bnAct(gp, `${name}.gpool`, block.gpoolActivation);
          release(gp);
          const pooled = gpool(gp2, 'gpool');
          release(gp2);
          const bias = matmul(pooled, `${name}.w1r`, block.w1r.outChannels);
          release(pooled);
          const biased = addBias(regular, bias);
          release(regular, bias);
          const c = bnAct(biased, `${name}.mid`, block.midActivation);
          release(biased);
          d = conv(c, `${name}.w2`, block.w2);
          release(c);
        } else {
          const mid = conv(a, `${name}.preConv`, block.preConv);
          release(a);
          const inner = applyBlocks(mid, block.blocks, name);
          const c = bnAct(inner, `${name}.post`, block.postActivation);
          release(inner);
          d = conv(c, `${name}.postConv`, block.postConv);
          release(c);
        }
        const next = add(trunk, d);
        release(trunk, d);
        trunk = next;
      });
      return trunk;
    };

    // -- inputs and trunk --
    const { trunk: trunkDesc, policy: policyDesc, value: valueDesc } = this.parsed;
    const spatial = act(padC(trunkDesc.conv1.inChannels));
    const global = vec(padC(trunkDesc.ginput.inChannels));

    let trunk = conv(spatial, 'conv1', trunkDesc.conv1);
    const gbias = matmul(global, 'ginput', trunkDesc.ginput.outChannels);
    const trunkBiased = addBias(trunk, gbias);
    release(trunk, gbias);
    trunk = applyBlocks(trunkBiased, trunkDesc.blocks, '');
    const tip = bnAct(trunk, 'tip', trunkDesc.tipActivation);
    release(trunk);

    // -- policy head --
    const p1 = conv(tip, 'p1', policyDesc.p1);
    const g1 = conv(tip, 'g1', policyDesc.g1);
    const g1b = bnAct(g1, 'g1bn', policyDesc.g1Activation);
    release(g1);
    const g1pool = gpool(g1b, 'gpool');
    release(g1b);
    const g1bias = matmul(g1pool, 'gpoolToBias', policyDesc.gpoolToBias.outChannels);
    const p1biased = addBias(p1, g1bias);
    release(p1, g1bias);
    const p1out = bnAct(p1biased, 'p1bn', policyDesc.p1Activation);
    release(p1biased);
    const policy = conv(p1out, 'p2', policyDesc.p2);
    release(p1out);
    const policyPass = matmul(g1pool, 'passMul', policyDesc.passMul.outChannels);
    release(g1pool);

    // -- value head --
    const v1 = conv(tip, 'v1', valueDesc.v1);
    release(tip);
    const v1out = bnAct(v1, 'v1bn', valueDesc.v1Activation);
    release(v1);
    const v1pool = gpool(v1out, 'value');
    release(v1out);
    const v2 = matmul(v1pool, 'v2', valueDesc.v2.outChannels);
    release(v1pool);
    const v2out = biasAct(v2, 'v2.bias', valueDesc.v2Activation);
    release(v2);
    const value = biasAct(matmul(v2out, 'v3', valueDesc.v3.outChannels), 'v3.bias', 'identity');
    const scoreValue = biasAct(matmul(v2out, 'sv3', valueDesc.sv3.outChannels), 'sv3.bias', 'identity');

    // -- readback --
    const outputs = { policy, policyPass, value, scoreValue };
    let offset = 0;
    const readbackLayout: Plan['readbackLayout'] = [];
    for (const name of ['policy', 'policyPass', 'value', 'scoreValue'] as const) {
      const tensor = outputs[name];
      readbackLayout.push({ name, offset, tensor });
      offset += roundUp(tensor.rowsPad * tensor.cpad * this.width, 4);
    }
    return {
      passes,
      spatialIn: spatial.buffer,
      globalIn: global.buffer,
      readbackSize: offset,
      readbackLayout,
      readouts: [],
    };
  }

  private plan(batch: number): Plan {
    let plan = this.plans.get(batch);
    if (!plan) {
      plan = this.buildPlan(batch);
      this.plans.set(batch, plan);
    }
    return plan;
  }

  // -- running --------------------------------------------------------------

  /**
   * spatial is NHWC [batch, size*size, numInputChannels], global
   * [batch, numInputGlobalChannels], both float32 whatever the compute type.
   */
  async evaluate(spatial: Float32Array, global: Float32Array, batch: number): Promise<NetOutputs> {
    const plan = this.plan(batch);
    const hw = this.size * this.size;
    const cin = this.parsed.numInputChannels;
    const cinPad = padC(cin);
    const gin = this.parsed.numInputGlobalChannels;
    const ginPad = padC(gin);

    // Repack into the padded-channel layout, converting to fp16 if that is
    // what the buffers hold. Padding stays zero.
    const spatialPacked = new Float32Array(roundUp(batch * hw, TILE) * cinPad);
    for (let r = 0; r < batch * hw; r++) {
      spatialPacked.set(spatial.subarray(r * cin, (r + 1) * cin), r * cinPad);
    }
    const globalPacked = new Float32Array(roundUp(batch, TILE) * ginPad);
    for (let r = 0; r < batch; r++) {
      globalPacked.set(global.subarray(r * gin, (r + 1) * gin), r * ginPad);
    }
    const cast = (data: Float32Array) =>
      (this.half ? Uint16Array.from(data, toHalf) : data) as Uint16Array<ArrayBuffer> | Float32Array<ArrayBuffer>;
    const started = performance.now();
    this.device.queue.writeBuffer(plan.spatialIn, 0, cast(spatialPacked));
    this.device.queue.writeBuffer(plan.globalIn, 0, cast(globalPacked));

    // Input and activation buffers are shared between evaluations in flight;
    // that is safe because writeBuffer and submit execute in queue order. Only
    // the readout is taken out of the pool for the whole call.
    const readout = plan.readouts.pop() ?? this.makeReadout(plan);

    const encoder = this.device.createCommandEncoder();
    if (this.profile && readout.querySet) {
      plan.passes.forEach((p, i) => {
        const pass = encoder.beginComputePass({ timestampWrites: {
          querySet: readout.querySet!, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
        pass.setPipeline(p.pipeline);
        pass.setBindGroup(0, p.bindGroup);
        pass.dispatchWorkgroups(...p.dispatch);
        pass.end();
      });
    } else {
      const pass = encoder.beginComputePass();
      for (const p of plan.passes) {
        pass.setPipeline(p.pipeline);
        pass.setBindGroup(0, p.bindGroup);
        pass.dispatchWorkgroups(...p.dispatch);
      }
      pass.end();
    }
    for (const { offset, tensor } of plan.readbackLayout) {
      encoder.copyBufferToBuffer(tensor.buffer, 0, readout.readback, offset,
        tensor.rowsPad * tensor.cpad * this.width);
    }
    if (this.profile && readout.querySet) {
      encoder.resolveQuerySet(readout.querySet, 0, 2 * plan.passes.length, readout.queryResolve!, 0);
      encoder.copyBufferToBuffer(readout.queryResolve!, 0, readout.queryReadback!, 0,
        readout.queryReadback!.size);
    }
    this.device.queue.submit([encoder.finish()]);
    const submitted = performance.now();

    await readout.readback.mapAsync(GPUMapMode.READ);
    const raw = readout.readback.getMappedRange().slice(0);
    readout.readback.unmap();
    const mapped = performance.now();

    if (this.profile && readout.queryReadback) {
      await readout.queryReadback.mapAsync(GPUMapMode.READ);
      const stamps = new BigInt64Array(readout.queryReadback.getMappedRange().slice(0));
      readout.queryReadback.unmap();
      plan.passes.forEach((p, i) => {
        const entry = this.gpuByLabel.get(p.label) ?? { nanos: 0, count: 0 };
        entry.nanos += Number(stamps[2 * i + 1] - stamps[2 * i]);
        entry.count += 1;
        this.gpuByLabel.set(p.label, entry);
      });
    }
    const asFloats = (offset: number, elems: number): Float32Array => {
      return this.half
        ? Float32Array.from(new Uint16Array(raw, offset, elems), fromHalf)
        : new Float32Array(raw, offset, elems);
    };

    const out: Partial<NetOutputs> = {};
    for (const { name, offset, tensor } of plan.readbackLayout) {
      const data = asFloats(offset, tensor.rowsPad * tensor.cpad);
      const column = (r: number, c: number) => data[r * tensor.cpad + c];
      if (name === 'policy') {
        out.policy = Array.from({ length: batch }, (_, image) =>
          Float32Array.from({ length: hw }, (_, p) => column(image * hw + p, 0)));
      } else if (name === 'policyPass') {
        out.policyPass = Float32Array.from({ length: batch }, (_, image) => column(image, 0));
      } else {
        const channels = name === 'value' ? 3 : this.parsed.scoreValueChannels;
        out[name] = Array.from({ length: batch }, (_, image) =>
          Float32Array.from({ length: channels }, (_, c) => column(image, c)));
      }
    }
    plan.readouts.push(readout);
    this.stages.pack += (submitted - started) * 1e6;
    this.stages.gpu += (mapped - submitted) * 1e6;
    this.stages.convert += (performance.now() - mapped) * 1e6;
    return out as NetOutputs;
  }

  private makeReadout(plan: Plan): Readout {
    const readback = this.device.createBuffer({
      size: plan.readbackSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    if (!this.profile || !this.device.features.has('timestamp-query')) return { readback };
    const stamps = 2 * plan.passes.length;
    return {
      readback,
      querySet: this.device.createQuerySet({ type: 'timestamp', count: stamps }),
      queryResolve: this.device.createBuffer({
        size: 8 * stamps, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }),
      queryReadback: this.device.createBuffer({
        size: 8 * stamps, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
    };
  }

  dispose(): void {
    for (const buffer of this.weights.values()) buffer.destroy();
    for (const plan of this.plans.values())
      for (const r of plan.readouts) {
        r.readback.destroy();
        r.querySet?.destroy();
        r.queryResolve?.destroy();
        r.queryReadback?.destroy();
      }
    for (const buffer of this.activations) buffer.destroy();
  }
}
