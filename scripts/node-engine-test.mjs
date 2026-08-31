// Drives the WebAssembly engine with the net evaluated in a worker, under node.
//
//   node scripts/node-engine-test.mjs < queries.json
//
// Reads analysis queries on stdin and prints the engine's answers. Compare them
// against the native engine on the same queries to check the bridge: the numbers
// should agree to float error.

import { writeSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Worker } from 'node:worker_threads';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../..');
const require = createRequire(import.meta.url);

const ENGINE = process.env.ENGINE ?? resolve(root, 'build-wasm-js-node/katahex.js');
const MODEL = process.env.MODEL ?? resolve(root, 'hex27x3.bin.gz');
const CONFIG = process.env.CONFIG ?? resolve(root, 'katahex/cpp/configs/analysis_example.cfg');
const NET_WORKER = resolve(here, '../build/netWorkerNode.cjs');
const BOARD_SIZE = Number(process.env.BOARD_SIZE ?? 11);
const SEARCH_THREADS = Number(process.env.SEARCH_THREADS ?? 1);

const createModule = require(ENGINE);

// NODERAWFS puts the engine on the real file descriptors, so queries arrive on
// this process's own stdin rather than through a JavaScript hook.
const Module = await createModule({
  noInitialRun: true,
  // Synchronous, because callMain blocks this thread on stdin: an async
  // console.log to a pipe would sit in the event loop's queue until EOF, and a
  // driver waiting on the reply before sending the next query would deadlock.
  print: (line) => writeSync(1, line + '\n'),
  printErr: (line) => {
    if (process.env.VERBOSE) console.error('[engine]', line);
  },
});

const controlAddress = Module._katahexControlBlockAddress();
const worker = new Worker(NET_WORKER, {
  workerData: { memory: Module.wasmMemory, controlAddress, modelPath: MODEL, boardSize: BOARD_SIZE },
});

let lastStats = null;
worker.on('message', (msg) => {
  if (msg.kind === 'stats') lastStats = msg.stats;
  else if (msg.kind === 'error') console.error('[net] error:', msg.message);
});
worker.on('error', (err) => console.error('[net] threw:', err));
await new Promise((ready) => {
  worker.on('message', (msg) => { if (msg.kind === 'ready') ready(msg); });
});

const started = Date.now();
Module.callMain([
  'analysis',
  '-model', MODEL,
  '-config', CONFIG,
  '-override-config',
  [
    'numAnalysisThreads=1',
    `numSearchThreadsPerAnalysisThread=${SEARCH_THREADS}`,
    'numNNServerThreadsPerModel=1',
    `maxBoardSizeForNNBuffer=${BOARD_SIZE}`,
    'nnCacheSizePowerOfTwo=16',
    ...(process.env.EXTRA_OVERRIDE ? [process.env.EXTRA_OVERRIDE] : []),
  ].join(','),
]);

const seconds = (Date.now() - started) / 1000;
if (lastStats) {
  console.error(
    `${lastStats.evals} evals, ${lastStats.rows} rows, ` +
    `${(lastStats.nanos / 1e6 / lastStats.evals).toFixed(0)} ms/eval, ${seconds.toFixed(1)}s total`);
}
await worker.terminate();
