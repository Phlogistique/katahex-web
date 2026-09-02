/**
 * Every float16 bit pattern widened, built once: a net holds 26 million of them
 * and a table lookup is the cheapest way to widen that many.
 */
const HALF_TO_FLOAT = /* @__PURE__ */ (() => {
  const table = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const sign = h >> 15 ? -1 : 1;
    const exponent = (h >> 10) & 0x1f;
    const mantissa = h & 0x3ff;
    table[h] = exponent === 0x1f
      ? (mantissa ? NaN : sign * Infinity)
      : sign * (exponent
        ? 2 ** (exponent - 15) * (1 + mantissa / 1024)
        : 2 ** -14 * (mantissa / 1024));
  }
  return table;
})();

export class KataGoBinModelParser {
  private readonly data: Uint8Array;
  private idx = 0;
  private readonly decoder = new TextDecoder('utf-8');

  constructor(data: Uint8Array) {
    this.data = data;
  }

  private skipWhitespace(): void {
    while (this.idx < this.data.length) {
      const b = this.data[this.idx];
      if (b === 0x20 || b === 0x0a || b === 0x0d || b === 0x09) {
        this.idx++;
        continue;
      }
      break;
    }
  }

  readToken(): string {
    this.skipWhitespace();
    const start = this.idx;
    while (this.idx < this.data.length) {
      const b = this.data[this.idx];
      if (b === 0x20 || b === 0x0a || b === 0x0d || b === 0x09) break;
      this.idx++;
    }
    if (this.idx <= start) {
      throw new Error('Unexpected EOF while reading token');
    }
    return this.decoder.decode(this.data.subarray(start, this.idx));
  }

  readInt(): number {
    const tok = this.readToken();
    const value = Number.parseInt(tok, 10);
    if (!Number.isFinite(value)) throw new Error(`Invalid int token: ${tok}`);
    return value;
  }

  readFloatAscii(): number {
    const tok = this.readToken();
    const value = Number.parseFloat(tok);
    if (!Number.isFinite(value)) throw new Error(`Invalid float token: ${tok}`);
    return value;
  }

  private atMarker(marker: string): boolean {
    for (let i = 0; i < marker.length; i++) {
      if (this.data[this.idx + i] !== marker.charCodeAt(i)) return false;
    }
    return true;
  }

  /**
   * The array KataGo writes is float32; scripts/export_net.py rewrites the bulk
   * of it as float16 under a marker of its own, which is what the page is served.
   */
  readBinaryFloats(count: number): Float32Array {
    this.skipWhitespace();
    const half = this.atMarker('@BINF16@');
    if (!half && !this.atMarker('@BIN@')) throw new Error('Expected @BIN@ marker');
    this.idx += half ? 8 : 5;

    const byteLen = count * (half ? 2 : 4);
    const absStart = this.data.byteOffset + this.idx;
    const absEnd = absStart + byteLen;
    if (absEnd > this.data.buffer.byteLength) throw new Error('Unexpected EOF while reading binary floats');

    const buf = this.data.buffer.slice(absStart, absEnd);
    this.idx += byteLen;
    this.skipWhitespace();

    if (!half) return new Float32Array(buf);
    const packed = new Uint16Array(buf);
    const out = new Float32Array(count);
    for (let i = 0; i < count; i++) out[i] = HALF_TO_FLOAT[packed[i]];
    return out;
  }
}

export interface ParsedBatchNorm {
  readonly channels: number;
  readonly mergedScale: Float32Array;
  readonly mergedBias: Float32Array;
}

export interface ParsedConv2d {
  readonly name: string;
  readonly kernelY: number;
  readonly kernelX: number;
  readonly inChannels: number;
  readonly outChannels: number;
  readonly dilationY: number;
  readonly dilationX: number;
  readonly weights: Float32Array; // [kY,kX,inC,outC] (NHWC filter order)
}

export interface ParsedMatMul {
  readonly name: string;
  readonly inChannels: number;
  readonly outChannels: number;
  readonly weights: Float32Array; // [inC,outC]
}

export interface ParsedMatBias {
  readonly name: string;
  readonly channels: number;
  readonly weights: Float32Array; // [channels]
}

export type ActivationKind = 'identity' | 'relu' | 'mish';

export function parseBatchNormV8(p: KataGoBinModelParser): ParsedBatchNorm {
  p.readToken();
  const channels = p.readInt();
  const epsilon = p.readFloatAscii();
  const hasScale = p.readInt() !== 0;
  const hasBias = p.readInt() !== 0;

  const mean = p.readBinaryFloats(channels);
  const variance = p.readBinaryFloats(channels);
  const scale = hasScale ? p.readBinaryFloats(channels) : new Float32Array(channels).fill(1);
  const bias = hasBias ? p.readBinaryFloats(channels) : new Float32Array(channels).fill(0);

  const mergedScale = new Float32Array(channels);
  const mergedBias = new Float32Array(channels);
  for (let i = 0; i < channels; i++) {
    const ms = scale[i] / Math.sqrt(variance[i] + epsilon);
    mergedScale[i] = ms;
    mergedBias[i] = bias[i] - ms * mean[i];
  }

  return { channels, mergedScale, mergedBias };
}

export function parseActivationNameV8(p: KataGoBinModelParser): string {
  // Model version < 11 only stores the activation layer name.
  return p.readToken();
}

export function parseActivationKind(p: KataGoBinModelParser, modelVersion: number): ActivationKind {
  // ActivationLayerDesc: always stores a name token. For modelVersion >= 11, stores an additional "kind" token.
  p.readToken(); // name
  if (modelVersion < 11) return 'relu';

  const kindTok = p.readToken();
  if (kindTok === 'ACTIVATION_IDENTITY') return 'identity';
  if (kindTok === 'ACTIVATION_RELU') return 'relu';
  if (kindTok === 'ACTIVATION_MISH') return 'mish';
  throw new Error(`Unsupported activation kind token: ${kindTok}`);
}

export function parseConv2d(p: KataGoBinModelParser): ParsedConv2d {
  const name = p.readToken();
  const kernelY = p.readInt();
  const kernelX = p.readInt();
  const inChannels = p.readInt();
  const outChannels = p.readInt();
  const dilationY = p.readInt();
  const dilationX = p.readInt();
  const weights = p.readBinaryFloats(kernelY * kernelX * inChannels * outChannels);
  return { name, kernelY, kernelX, inChannels, outChannels, dilationY, dilationX, weights };
}

export function parseMatMul(p: KataGoBinModelParser): ParsedMatMul {
  const name = p.readToken();
  const inChannels = p.readInt();
  const outChannels = p.readInt();
  const weights = p.readBinaryFloats(inChannels * outChannels);
  return { name, inChannels, outChannels, weights };
}

export function parseMatBias(p: KataGoBinModelParser): ParsedMatBias {
  const name = p.readToken();
  const channels = p.readInt();
  const weights = p.readBinaryFloats(channels);
  return { name, channels, weights };
}
