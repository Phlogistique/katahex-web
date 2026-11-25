/// <reference types="vite/client" />
// The net on the CPU, for a browser whose WebGPU does not work: TensorFlow.js
// running KataGo's graph (vendor/modelV8.ts) on its WebAssembly backend, with
// SIMD and threads where the browser has them.
//
// Several times slower than the GPU, but it is the same net reading the same
// weights, so the engine plays the same moves, only later. Imported only when
// the GPU fails, so a page with one never downloads any of it.

import * as tf from '@tensorflow/tfjs';
// Not registered by tfjs-core's entry point, which the page imports in place
// of the whole of TensorFlow.js; the vendored graph calls x.add(), x.mul().
import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';
import { setThreadsCount, setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import plain from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm?url';
import simd from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-simd.wasm?url';
import threadedSimd from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-threaded-simd.wasm?url';
import type { ParsedKataGoModelV8 } from '../vendor/modelV8';
import { KataGoModelV8Tf } from '../vendor/modelV8';
import type { NetEvaluator } from './netRunner';
import type { NetOutputs } from './webgpuModel';

export type CpuModel = NetEvaluator & { backend: string; threads: number };

export async function createCpuModel(parsed: ParsedKataGoModelV8, boardSize: number): Promise<CpuModel> {
  setWasmPaths({
    'tfjs-backend-wasm.wasm': plain,
    'tfjs-backend-wasm-simd.wasm': simd,
    'tfjs-backend-wasm-threaded-simd.wasm': threadedSimd,
  });

  // All but one core: the engine's own threads need one to search on.
  const threads = Math.max(1, (navigator.hardwareConcurrency || 2) - 1);
  setThreadsCount(threads);

  // Plain JavaScript if even WebAssembly fails, which is slower again but runs.
  if (!await tf.setBackend('wasm').catch(() => false)) {
    await tf.setBackend('cpu');
  }
  await tf.ready();

  const model = new KataGoModelV8Tf(parsed);
  const area = boardSize * boardSize;

  // TensorFlow.js hands back one flat array per head, where the runner wants a
  // row per position in the batch.
  const rows = (flat: Float32Array, stride: number, batch: number): Float32Array[] =>
    Array.from({ length: batch }, (_, row) => flat.slice(row * stride, (row + 1) * stride));

  return {
    backend: tf.getBackend(),
    threads,
    async evaluate(spatial, global, batch): Promise<NetOutputs> {
      // Copied: the views are into the engine's memory, which the next request reuses.
      const spatialTensor = tf.tensor4d(spatial.slice(), [batch, boardSize, boardSize, parsed.numInputChannels]);
      const globalTensor = tf.tensor2d(global.slice(), [batch, parsed.numInputGlobalChannels]);
      const out = model.forwardPolicyValue(spatialTensor, globalTensor);

      try {
        const [policy, policyPass, value, scoreValue] = await Promise.all([
          out.policy.data() as Promise<Float32Array>,
          out.policyPass.data() as Promise<Float32Array>,
          out.value.data() as Promise<Float32Array>,
          out.scoreValue.data() as Promise<Float32Array>,
        ]);

        return {
          policy: rows(policy, area, batch),
          policyPass,
          value: rows(value, 3, batch),
          scoreValue: rows(scoreValue, parsed.scoreValueChannels, batch),
        };
      } finally {
        tf.dispose([spatialTensor, globalTensor]);
        tf.dispose(out);
      }
    },
  };
}
