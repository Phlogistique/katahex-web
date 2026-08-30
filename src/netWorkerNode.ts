// Node worker that evaluates the net for the WebAssembly engine, so the bridge
// can be tested without a browser. The browser worker differs only in where the
// net comes from and which TensorFlow.js backend it selects.

import { appendFileSync, readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { parentPort, workerData } from 'node:worker_threads';

import * as tf from '@tensorflow/tfjs';
import { parseKataGoModelV8 } from '../vendor/loadModelV8';
import { KataGoModelV8Tf } from '../vendor/modelV8';
import { serveEvals, type NetEvaluator } from './netRunner';

type WorkerData = {
  memory: WebAssembly.Memory;
  controlAddress: number;
  modelPath: string;
  boardSize: number;
};

// The engine holds node's event loop while it searches, so console output from
// this worker never reaches the terminal. Trace to a file instead.
const trace = process.env.NET_WORKER_LOG;
const log = (message: string) => {
  if (trace) appendFileSync(trace, message + '\n');
};

async function main() {
  log('booted');
  // Atomics.waitAsync does not keep node's event loop alive on its own, so
  // without a handle the worker goes idle and the promise never settles. A
  // browser has no such notion and needs nothing.
  setInterval(() => {}, 1 << 30);
  const { memory, controlAddress, modelPath, boardSize } = workerData as WorkerData;

  await tf.setBackend('cpu');
  await tf.ready();

  const parsed = parseKataGoModelV8(new Uint8Array(gunzipSync(readFileSync(modelPath))));
  const model = new KataGoModelV8Tf(parsed);
  log(`model ready on ${tf.getBackend()}`);
  parentPort!.postMessage({ kind: 'ready', backend: tf.getBackend() });

  // TensorFlow.js hands back one flat array per head, where the runner wants a
  // row per position in the batch.
  const rows = (flat: Float32Array, stride: number, batch: number): Float32Array[] =>
    Array.from({ length: batch }, (_, row) => flat.subarray(row * stride, (row + 1) * stride));

  const evaluator: NetEvaluator = {
    async evaluate(spatial, global, batch) {
      const out = model.forwardPolicyValue(
        tf.tensor4d(spatial.slice(), [batch, boardSize, boardSize, parsed.numInputChannels]),
        tf.tensor2d(global.slice(), [batch, parsed.numInputGlobalChannels]),
      );
      const [policy, policyPass, value, scoreValue] = await Promise.all([
        out.policy.data() as Promise<Float32Array>,
        out.policyPass.data() as Promise<Float32Array>,
        out.value.data() as Promise<Float32Array>,
        out.scoreValue.data() as Promise<Float32Array>,
      ]);
      tf.dispose(out);
      return {
        policy: rows(policy, boardSize * boardSize, batch),
        policyPass,
        value: rows(value, 3, batch),
        scoreValue: rows(scoreValue, parsed.scoreValueChannels, batch),
      };
    },
  };

  await serveEvals(memory, controlAddress, evaluator, {
    onEval: (stats) => parentPort!.postMessage({ kind: 'stats', stats }),
  });
}

main().catch((err) => {
  log(`error: ${err?.stack ?? err}`);
  parentPort!.postMessage({ kind: 'error', message: String(err?.stack ?? err) });
  process.exit(1);
});
