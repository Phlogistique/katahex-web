// Second half of the ONNX check: runs the input `check_onnx.py` wrote through
// the TensorFlow.js implementation the converter was written from, and reports
// how far the two implementations are apart.
//
//   npx tsx scripts/check-onnx-tfjs.ts build/check

import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';

import * as tf from '@tensorflow/tfjs';
import { parseKataGoModelV8 } from '../vendor/loadModelV8';
import { KataGoModelV8Tf } from '../vendor/modelV8';

const dir = process.argv[2] ?? 'build/check';
const modelPath = process.argv[3] ?? '../hex27x3.bin.gz';

async function main() {
  const input = JSON.parse(readFileSync(`${dir}/input.json`, 'utf8'));
  const onnx = JSON.parse(readFileSync(`${dir}/onnx.json`, 'utf8'));

  await tf.setBackend('cpu');
  await tf.ready();
  const model = new KataGoModelV8Tf(
    parseKataGoModelV8(new Uint8Array(gunzipSync(readFileSync(modelPath)))));

  const out = model.forwardPolicyValue(
    tf.tensor4d(Float32Array.from(input.spatial), input.shape),
    tf.tensor2d(Float32Array.from(input.global), input.globalShape));

  const got: Record<string, Float32Array> = {
    policy: (await out.policy.data()) as Float32Array,
    policyPass: (await out.policyPass.data()) as Float32Array,
    value: (await out.value.data()) as Float32Array,
    scoreValue: (await out.scoreValue.data()) as Float32Array,
  };

  for (const name of Object.keys(got)) {
    const a = got[name];
    const b = onnx[name] as number[];
    let worst = 0;
    for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
    console.log(`${name.padEnd(11)} onnx differs from tensorflow.js by at most ${worst.toExponential(2)}`);
  }
}

main();
