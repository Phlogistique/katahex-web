# KataHex in the browser

Running the KataHex engine as a web page: KataHex's C++ board, input features and
search compiled to WebAssembly, with the neural net evaluated on the GPU through
WebGPU.

`hex27x3.bin.gz` is a KataGo v11 `b18c384nbt` net -- 18 nested-bottleneck blocks,
384 channels, 26.4M parameters -- in KataGo's own weight format. Nothing in the file
is hex-specific, so an existing browser reader for KataGo nets loads it unchanged;
see `vendor/VENDORED.md`. The hex-specific parts stay in C++.

## Net benchmark

`src/bench.ts` loads the net and times forward passes at several batch sizes, to
find out how fast the GPU half can go and how much batching matters:

    npm install
    ln -s ../../hex27x3.bin.gz public/hex27x3.bin.gz
    npm run dev

The dev server sets `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy`,
which `SharedArrayBuffer` -- and so wasm pthreads -- requires.

WebGPU is only exposed in a secure context, so a phone must not reach the dev server
by LAN address. Forward the port instead and open `http://localhost:5173` on the
phone:

    adb reverse tcp:5173 tcp:5173

## Measured for comparison (11x11, native, Iris Xe laptop, 2026-08-29)

| build                   | 1 thread | 4 threads | 16 threads |
| ----------------------- | -------- | --------- | ---------- |
| Eigen CPU, AVX2         | 3.3 v/s  | 7.1 v/s   |            |
| OpenCL, fp16            | 11.7 v/s | 18.0 v/s  | 30.3 v/s   |

Threads buy throughput by growing the eval batch, which is why the batch sweep in
the benchmark decides whether the port needs wasm pthreads at all.

## Engine

`scripts/build-engine.sh` cross-compiles the KataHex engine to WebAssembly. It
needs a katahex checkout on the `wasm-build` branch, which carries the cmake
fixes for targeting Emscripten and the JS backend below. 1.6 MB of wasm.

`BACKEND=EIGEN` builds the engine with the net on the wasm CPU, which works and
answers analysis queries at **1.2 visits/s** -- 2.7x under native Eigen, itself
5x under the GPU. Useless as a product, but it establishes that the board, the
input features and the search need no porting.

## Bridge

`BACKEND=JS`, the default, builds the engine against
`katahex/cpp/neuralnet/jsbackend.cpp`, which evaluates nothing. It writes the net
inputs into the shared wasm memory, wakes whoever is listening and blocks on one
atomic word; `src/netRunner.ts` reads them, runs the net with TensorFlow.js,
writes the outputs back and wakes the engine. The two files are halves of one
protocol.

`npm run test:engine` drives it under node with the TensorFlow.js CPU backend:

    echo '{"id":"a","moves":[["B","f6"],["W","d4"]],"rules":"tromp-taylor",
           "komi":0,"boardXSize":11,"boardYSize":11,"maxVisits":5,
           "analyzeTurns":[2]}' | npm run test:engine

Against the native Eigen engine on the same query with `nnRandomize=false`, the
answers agree to within 1e-6: same principal variation, same move order, same
visit distribution. Two independent implementations of the same net.

Two things to know:

- The board must fill the net's input tensor. The TensorFlow.js port does not
  implement KataGo's mask and takes the board size from the tensor shape, so an
  engine whose `nnXLen` exceeds the board would pool over the padding and quietly
  return wrong numbers. Pass `maxBoardSizeForNNBuffer` equal to the board size.
  `netRunner.ts` checks the mask channel on the first evaluation and throws.
- `Atomics.waitAsync` does not keep node's event loop alive, so the net worker
  needs a handle of its own or its wakeup promise never settles. Browsers have no
  such notion.
