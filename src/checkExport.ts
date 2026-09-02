// Certifies what scripts/export_net.py wrote: that serving the half precision
// net gives the GPU the same weights the full precision one would have.
//
//   npm run check:export -- ../hex27x3.bin.gz public-web/net-fp16.bin.gz
//
// Every weight the WebGPU backend stores goes through toHalf on its way to the
// buffer, so each array must come back either unchanged -- the batch norm
// statistics, which the parser folds in float32 before anything is rounded --
// or rounded to exactly the half the full precision file would have produced.
//
// That the weights match does not make the outputs match: a 3x3 kernel is
// Winograd transformed before it is rounded, and the transform of a rounded
// kernel is not the rounded transform. What that costs is what check.html
// measures, against the same frozen thresholds as any other numeric change.

import { gunzipSync } from 'node:zlib';
import { readFileSync } from 'node:fs';

import { parseKataGoModelV8 } from '../vendor/loadModelV8';
import { toHalf } from './webgpuModel';

function load(path: string) {
  const raw = new Uint8Array(readFileSync(path));
  return parseKataGoModelV8(raw[0] === 0x1f && raw[1] === 0x8b ? gunzipSync(raw) : raw);
}

/** Every float array in the parsed model, in a fixed order. */
function arrays(node: unknown, into: Float32Array[] = []): Float32Array[] {
  if (node instanceof Float32Array) into.push(node);
  else if (Array.isArray(node)) for (const item of node) arrays(item, into);
  else if (node && typeof node === 'object') {
    for (const key of Object.keys(node).sort()) arrays((node as Record<string, unknown>)[key], into);
  }
  return into;
}

const [full, half] = process.argv.slice(2);
if (!half) throw new Error('usage: check:export -- <full.bin.gz> <fp16.bin.gz>');

const a = arrays(load(full));
const b = arrays(load(half));
if (a.length !== b.length) throw new Error(`${a.length} arrays against ${b.length}`);

let identical = 0, rounded = 0, wrong = 0, worst = 0;
for (let i = 0; i < a.length; i++) {
  if (a[i].length !== b[i].length) throw new Error(`array ${i}: ${a[i].length} against ${b[i].length}`);
  let same = true, isRounding = true;
  for (let j = 0; j < a[i].length; j++) {
    if (a[i][j] !== b[i][j]) same = false;
    if (toHalf(a[i][j]) !== toHalf(b[i][j])) isRounding = false;
    worst = Math.max(worst, Math.abs(a[i][j] - b[i][j]));
  }
  if (same) identical++;
  else if (isRounding) rounded++;
  else { wrong++; console.error(`array ${i} (${a[i].length}) is neither unchanged nor a rounding`); }
}

const total = a.reduce((n, x) => n + x.length, 0);
console.log(`${a.length} arrays, ${(total / 1e6).toFixed(2)}M weights: ` +
  `${identical} unchanged, ${rounded} rounded to what the GPU stores, ${wrong} wrong`);
console.log(`largest weight change ${worst.toExponential(2)}`);
process.exit(wrong ? 1 : 0);
