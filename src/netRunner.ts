// Serves neural net evaluations to the WebAssembly engine.
//
// The engine's JS backend (katahex/cpp/neuralnet/jsbackend.cpp) writes its inputs
// into the shared wasm memory and blocks on one atomic word. This reads them,
// evaluates the net, writes the outputs back and wakes it. Field offsets and
// states are the two halves of the same protocol; keep them in step with the C++.

import type { NetOutputs } from './webgpuModel';

export const FIELD = {
  STATE: 0,
  BATCH_SIZE: 1,
  NN_X_LEN: 2,
  NN_Y_LEN: 3,
  NUM_SPATIAL_FEATURES: 4,
  NUM_GLOBAL_FEATURES: 5,
  SPATIAL_INPUT: 6,
  GLOBAL_INPUT: 7,
  POLICY: 8,
  POLICY_PASS: 9,
  VALUE: 10,
  SCORE_VALUE: 11,
} as const;

export const STATE = { IDLE: 0, REQUEST: 1, RESPONSE: 2 } as const;

const NUM_VALUE_CHANNELS = 3;
const NUM_SCORE_VALUE_CHANNELS = 6;

/** Resolves once the word stops being `expected`, without blocking the thread. */
async function waitWhile(control: Int32Array, index: number, expected: number): Promise<void> {
  for (;;) {
    if (Atomics.load(control, index) !== expected) return;
    if ('waitAsync' in Atomics) {
      const result = (Atomics as any).waitAsync(control, index, expected) as
        { async: boolean; value: string | Promise<string> };
      if (result.async) await result.value;
    } else {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }
}

export type EvalStats = { evals: number; rows: number; nanos: number };

/** WebGPU in a page, TensorFlow.js under node. Spatial input is NHWC. */
export type NetEvaluator = {
  evaluate(spatial: Float32Array, global: Float32Array, batch: number): Promise<NetOutputs>;
};

/**
 * Runs until `stopped()` returns true, which is only checked between requests.
 * The board must fill the net's input tensor: the port of KataGo's net does not
 * implement the mask, it takes the board size from the tensor shape, so an
 * engine whose nnXLen exceeds the board would pool over the padding and quietly
 * return wrong numbers. Configure the engine with `maxBoardSizeForNNBuffer` set
 * to the board size.
 */
export async function serveEvals(
  memory: WebAssembly.Memory,
  controlAddress: number,
  model: NetEvaluator,
  options: { stopped?: () => boolean; onEval?: (stats: EvalStats) => void } = {},
): Promise<void> {
  const stats: EvalStats = { evals: 0, rows: 0, nanos: 0 };
  let checkedMask = false;

  while (!options.stopped?.()) {
    const control = new Int32Array(memory.buffer, controlAddress, 12);
    await waitWhile(control, FIELD.STATE, STATE.IDLE);
    if (Atomics.load(control, FIELD.STATE) !== STATE.REQUEST) continue;

    const batchSize = control[FIELD.BATCH_SIZE];
    const xLen = control[FIELD.NN_X_LEN];
    const yLen = control[FIELD.NN_Y_LEN];
    const numSpatial = control[FIELD.NUM_SPATIAL_FEATURES];
    const numGlobal = control[FIELD.NUM_GLOBAL_FEATURES];

    const spatial = new Float32Array(
      memory.buffer, control[FIELD.SPATIAL_INPUT], batchSize * yLen * xLen * numSpatial);
    const global = new Float32Array(memory.buffer, control[FIELD.GLOBAL_INPUT], batchSize * numGlobal);

    if (!checkedMask) {
      checkedMask = true;
      for (let i = 0; i < yLen * xLen; i++) {
        if (spatial[i * numSpatial] !== 1) {
          throw new Error(
            `board does not fill the ${xLen}x${yLen} input tensor; ` +
            'set maxBoardSizeForNNBuffer to the board size');
        }
      }
    }

    const started = performance.now();
    const out = await model.evaluate(spatial, global, batchSize);

    // The buffer is fetched again because growing the wasm heap replaces it.
    const buffer = memory.buffer;
    const write = (address: number, rows: Float32Array[], stride: number) => {
      const all = new Float32Array(buffer, address, batchSize * stride);
      for (let row = 0; row < batchSize; row++) all.set(rows[row], row * stride);
    };
    write(control[FIELD.POLICY], out.policy, yLen * xLen);
    new Float32Array(buffer, control[FIELD.POLICY_PASS], batchSize).set(out.policyPass);
    write(control[FIELD.VALUE], out.value, NUM_VALUE_CHANNELS);
    write(control[FIELD.SCORE_VALUE], out.scoreValue, NUM_SCORE_VALUE_CHANNELS);

    stats.evals += 1;
    stats.rows += batchSize;
    stats.nanos += (performance.now() - started) * 1e6;
    options.onEval?.(stats);

    Atomics.store(control, FIELD.STATE, STATE.RESPONSE);
    Atomics.notify(control, FIELD.STATE);
    await waitWhile(control, FIELD.STATE, STATE.RESPONSE);
  }
}
