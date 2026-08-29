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

## Measured, Chrome 11x11 on the Iris Xe laptop

Evaluations per second on WebGPU, by batch size. Take these to one significant
figure: this laptop is usually loaded, and repeating the fp16 batch 16 cell on
one afternoon gave 45.9, 48.7 and 63.3. Compare runtimes within a sitting, not
against the table.

| batch | TensorFlow.js fp32 | onnxruntime fp32 | onnxruntime fp16 |
| ----- | ------------------ | ---------------- | ---------------- |
| 1     | 11.6               | 9.8              | 9.9              |
| 2     | 17.2               | 18.4             | 20.3             |
| 4     | 25.2               | 25.0             | 29.7             |
| 8     | 26.2               | 29.4             | 40.2             |
| 16    | 25.8               | 31.0             | 47.4             |
| 32    |                    |                  | 50.1             |
| 64    |                    |                  | 51.6             |

Half precision is what makes the difference, and it is what native uses too.
TensorFlow.js computes in fp32 whatever the adapter offers and flattens at batch
4; onnxruntime on a float16 graph keeps scaling to about 50 evaluations a second,
roughly half of what native OpenCL does on this GPU, and the file is 53 MB rather
than 105.

A single evaluation costs the same everywhere, including natively (around 90 ms
against 83), so what the browser gives up is entirely the batching win. Batch 8
is where most of it has arrived, which is 16 search threads.

13x13 on the same fp16 graph: 10.3 / 17.1 / 24.6 / 31.0 / 34.3 / 35.8 / 36.2
evaluations a second over the same batch sizes.

## Where the remaining gap is

Not in the web platform. `micro.html` runs a tiled matrix multiply written by
hand in WGSL, which is what a convolution becomes:

| shape                            | fp32 | fp16 |
| -------------------------------- | ---- | ---- |
| 2048 square                      | 473  | 698  |
| one trunk conv, batch 64         | 460  | 678  |

GFLOP/s, and the skinny convolution shape costs nothing against the square one.

The net needs 6.33 GFLOP per evaluation, 90% of it in the 62 3x3 convolutions of
the trunk. At 52 evaluations a second onnxruntime is getting 329 GFLOP/s out of
this GPU, less than half of what the hand-written kernel above reaches on the
same shape in the same browser. Feed the net through a kernel that good and it
would run at about 110 evaluations a second, next to native's 128.

So the gap is onnxruntime's convolution, not WebGPU. Two things it lacks:

- **Winograd.** KataGo's OpenCL backend transforms every 3x3 convolution to
  F(4x4, 3x3), which issues 36 multiplies where the direct form issues 144, so
  it does about a third of the arithmetic over the whole net. onnxruntime's
  WebGPU backend has no Winograd path at all: `strings` on its wasm finds
  `conv2d_mm.cc`, an implicit GEMM, and no Winograd anywhere. Worth rechecking
  on a later onnxruntime.
- **Tuning.** KataGo ships a tuner and keeps the result per GPU;
  `~/.katago/opencltuning/` holds the workgroup and tile sizes it picked for this
  Iris Xe, and the fp16 storage and compute flags it turned on. onnxruntime ships
  one set of generic shaders.

Native's 811 GFLOP/s of nominal work is therefore only about 266 GFLOP/s issued.
It wins by doing less arithmetic rather than by driving the GPU harder: the
browser already issues more FLOP/s than native does.

Two things ruled out along the way. Unfused elementwise work is not the problem:
replacing all 118 mish activations with the identity, which drops 354 dispatches
of the 828, changes the time by under 1%. And the platform exposes what a good
kernel would want, `shader-f16` and `subgroups` both, with fp16 worth 1.5x on
this GPU in the hand-written kernel and 1.7x in onnxruntime.

## What it has to beat

Native OpenCL on this laptop's Iris Xe does 37.5 visits/s at 4 search threads, 83
at 16 and 105 at 32, on 11x11. Threads are the lever because KataGo's average
batch comes out at half the thread count and the GPU is latency-bound: 83 ms per
row at batch 1, 7.8 ms at batch 60.

That is also why the batch sweep in the net benchmark matters. If WebGPU shows
the same curve, the port needs wasm pthreads, and so needs the page to be
cross-origin isolated for `SharedArrayBuffer`.

`micro.html`, alongside the net benchmark, is the WGSL matmul. It checks itself
against all-ones inputs before timing.

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

## ONNX

`scripts/bin_to_onnx.py` converts the net to ONNX without going through PyTorch,
which the existing converters need and which we have no checkpoint for. It reads
KataGo's weight format directly and builds the graph to match
`vendor/modelV8.ts`. `--fp16` computes in half precision, with float32 in and out
so callers do not have to care.

    uv run --with onnx --with numpy python scripts/bin_to_onnx.py \
      ../hex27x3.bin.gz public/hex27x3-11-fp16.onnx --size 11 --fp16

The board size is baked in, for the same reason the TensorFlow.js path needs it
passed: KataGo's global pooling scales by the board size and the net carries no
mask here.

`scripts/check_onnx.py` then `scripts/check-onnx-tfjs.ts` check the result. The
float32 graph agrees with the TensorFlow.js implementation to 3e-4, on policy
logits that reach 55. Half precision costs more, 0.22 on those same logits, on a
random input that pushes them further than a real position would.

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
