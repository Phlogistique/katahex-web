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
needs a katahex checkout on the `wasm-build` branch, which carries two cmake
fixes for targeting Emscripten.

The result runs and answers analysis queries:

    node ../build-wasm/katahex.js analysis -model ../hex27x3.bin.gz \
      -config ../katahex/cpp/configs/analysis_example.cfg

1.6 MB of wasm, and **1.2 visits/s** on the laptop -- 2.7x under native Eigen,
which is itself 5x under the GPU. That is the point of the exercise: the search,
the board and the input features are all in place and correct, and the neural net
now has to move off the wasm CPU and onto WebGPU.
