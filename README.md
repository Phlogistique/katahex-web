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

## Measured, Chrome 11x11 on the Iris Xe laptop (TensorFlow.js WebGPU, fp32)

| batch | ms/call | evals/s |     | 13x13   | evals/s |
| ----- | ------- | ------- | --- | ------- | ------- |
| 1     | 86.1    | 11.6    |     | batch 1 | 12.2    |
| 2     | 116.0   | 17.2    |     | batch 2 | 15.1    |
| 4     | 158.5   | 25.2    |     | batch 4 | 15.9    |
| 8     | 305.8   | 26.2    |     | batch 8 | 19.5    |
| 16    | 621.1   | 25.8    |     | batch 16| 17.5    |

It flattens at batch 4. A single evaluation costs the same as it does natively
(86 ms against 83), so what the browser gives up is the batching win, not raw
throughput -- native keeps going down to 7.8 ms per row. Batching is still worth
2.2x here, so the page does want threads, just not many: eight search threads put
the batch at four, and nothing past that helps.

TensorFlow.js does not use fp16 even where the adapter offers `shader-f16`, which
native does use and which is a good part of the remaining gap. Worth trying ONNX
Runtime Web before accepting these numbers as the ceiling.

## What it has to beat

Native OpenCL on this laptop's Iris Xe does 37.5 visits/s at 4 search threads, 83
at 16 and 105 at 32, on 11x11. Threads are the lever because KataGo's average
batch comes out at half the thread count and the GPU is latency-bound: 83 ms per
row at batch 1, 7.8 ms at batch 60.

That is also why the batch sweep in the net benchmark matters. If WebGPU shows
the same curve, the port needs wasm pthreads, and so needs the page to be
cross-origin isolated for `SharedArrayBuffer`.

## Engine

`scripts/build-engine.sh` cross-compiles the KataHex engine to WebAssembly. It
needs a katahex checkout on the `wasm-build` branch, which carries the cmake
fixes for targeting Emscripten and the JS backend below. 1.6 MB of wasm.

`BACKEND=EIGEN` builds the engine with the net on the wasm CPU. It works and
answers analysis queries, at roughly a third of the speed of the same engine
built natively -- itself an order of magnitude under the GPU. Useless as a
product, but it establishes that the board, the input features and the search
need no porting. (Both figures were measured on a loaded machine and are only
good to a factor of two.)

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
