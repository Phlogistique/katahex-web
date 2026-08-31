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

## Winograd, measured

`winograd.html` implements the trunk convolution as F(4x4, 3x3) in WGSL --
input transform, batched matmul over the 36 tile positions, output transform,
with KataGo's transform matrices and matmul layout -- and times it against the
same convolution as one direct matmul in the same sitting. It first checks
itself against a direct convolution on the CPU; fp32 agrees to 4e-5.

Milliseconds per 192-channel trunk convolution, one run:

| shape           | direct fp32 | winograd fp32 | direct fp16 | winograd fp16 |
| --------------- | ----------- | ------------- | ----------- | ------------- |
| 11x11, batch 16 | 3.1         | 1.6           | 2.1         | 1.4           |
| 11x11, batch 64 | 11.7        | 6.1           | 8.3         | 5.2           |
| 13x13, batch 64 | 18.3        | 14.3          | 11.9        | 10.1          |

So on 11x11 winograd is 1.5x over the direct matmul in fp16 and 1.9x in fp32,
about 990 effective GFLOP/s at fp16 batch 64 where onnxruntime's whole net gets
329. If the trunk convolutions (90% of the net's arithmetic) ran at that rate
and the rest stayed at onnxruntime's pace, an evaluation would cost about
7.7 ms at batch 64: some 130 evaluations a second, next to native's 128. That
is the case for writing the net's WebGPU backend by hand.

Caveats:

- Half the winograd time is the two transforms, not the matmul, so there is
  room left; KataGo fuses batchnorm and activation into its transform kernels.
- fp16 winograd is much less exact: max error 0.4 on outputs reaching 8, on
  uniform random inputs, against 4e-5 for fp32. The transform coefficients
  (4, 5, 8) stretch the values fp16 has to accumulate. Whether it matters is a
  question for the whole net's policy outputs, measured the way the ONNX
  section above measures quantization.
- 13x13 gains much less: 4x4 output tiles cover a 13-wide board as 16, so a
  third of the winograd arithmetic is padding. F(2x2, 3x3) or rectangular
  tiles would fit it better.

## The hand-written backend

`src/webgpuModel.ts` runs the whole net on raw WebGPU: activations as
[positions, channels] matrices, 1x1 convolutions and dense layers as one
matmul each, 3x3 convolutions through the Winograd pipeline above, and small
kernels for batchnorm+mish, residual adds, and KataGo's global pooling. The
forward pass for a batch size compiles once into a flat list of dispatches
with every shape baked into its shader.

Evaluations per second on 11x11 fp16, hand-written against onnxruntime in the
same sitting:

| batch | hand-written | onnxruntime |
| ----- | ------------ | ----------- |
| 1     | 25.8         | 11.0        |
| 2     | 45.4         | 21.3        |
| 4     | 72.2         | 32.3        |
| 8     | 82.8         | 41.7        |
| 16    | 106.1        | 49.0        |
| 32    | 120.0        | 51.4        |
| 64    | 109.0        | 53.0        |

2.1-2.4x at every batch size, and the top of the curve sits next to native
OpenCL's 128 visits/s. A single evaluation takes 39 ms where onnxruntime and
native OpenCL both take about 90, which shortens the latency-bound early
search. fp32 reaches 68 evaluations a second, still 2.2x onnxruntime's fp32;
13x13 fp16 about 60, where the 4x4 Winograd tiles fit the board worst.

Correctness is checked on every load against the TensorFlow.js implementation
of the same net, two codepaths sharing only the weight file: fp32 agrees to
1e-4 on the policy logits. fp16 sits at 0.05 mean / 0.6 max on logits reaching
17, and that error is the fp16 storage of the trunk activations themselves,
not the arithmetic: computing the Winograd transforms in f32 cut it by 3x
(their coefficients reach 8, which fp16 amplifies), but accumulating the
matmul in f32 and keeping the Winograd product matrix in f32 each moved it by
less than the run-to-run chaos, so neither is kept. Native KataGo's fp16 mode
quantizes the same tensors the same way.

One WGSL lesson: a `var` declared inside a loop is supposed to be
re-initialized on every iteration, but this Intel/Mesa driver does not, which
turned a per-tile accumulator into a prefix sum and the whole net into NaN.
Zero explicitly.

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

## The page

`katahex-android/ui` is the hexplorer, vendored from PlayHex for the Android app.
It reaches its engine through a small bridge on `window` -- `Native.start(size)`,
`Native.query(json)`, and the engine's two output streams back as
`onEngineLine` and `onEngineLog` -- which the app implements in Java over a child
process. `src/wasmEngine.ts` implements the same bridge with the engine compiled
to WebAssembly, so the same ui becomes a page that needs no server:

    cd ../katahex-android/ui
    ln -s ../../../build-wasm-js-web/katahex.js public/katahex.js
    ln -s ../../../build-wasm-js-web/katahex.wasm public/katahex.wasm
    ln -s ../../../hex27x3.bin.gz public/hex27x3.bin.gz
    npx vite --mode web        # then open /web.html

`src/engineWorker.ts` is the whole page-side engine: one worker holding the wasm
engine, the net on WebGPU, and `src/netRunner.ts` between them. It takes about ten
seconds to start, nearly all of it the 97 MB net. `?fp32` runs the net in single
precision.

Two things had to change for a page, both because a browser's main thread is not
optional:

- **The engine's output is proxied to the thread that loaded the module**, so that
  thread must stay free. `-sPROXY_TO_PTHREAD` puts `main()` on a thread of its
  own, which is what lets it block.
- **A page has no stdin.** Blocking on it deadlocks: the read is proxied to that
  same thread. So the browser build takes its queries by a call instead --
  `katahexPushQuery`, guarded by `KATAHEX_QUERY_QUEUE` in `analysis.cpp`, ten
  lines of queue. The node build keeps reading stdin.

Also worth knowing: the model download arrives *already inflated*, because servers
set `Content-Encoding: gzip` on a `.gz`. Sniff the magic bytes rather than
trusting the name.

### Search speed, 11x11 on the Iris Xe laptop

Sweeping in a reversed order -- 8, 16, 32, 32, 16, 8 -- with nothing else on the
machine, three openings each, cache cleared, median of the six samples:

| search threads | page, fp16 | + 3 ms batch wait | + coalesced Winograd | native OpenCL |
| --- | --- | --- | --- | --- |
| 8 | 65 | | | ~60 (between its 4 and its 16) |
| 16 | 79-83 | 90 | | 83 |
| 32 | 86 | 96 | ~140 | 105 |
| 64 | ~61, and unstable | | ~136, a wash against 32 | ~115 |

**The big one was found by the per-kernel profile** (`?profile`, timestamp
queries on every dispatch): the Winograd transform and untransform were 41% of
all GPU time while the matmuls they feed ran at the expected ~650 GFLOP/s. Their
threads walked the tiles, putting neighbors ~400 bytes apart in memory -- one
f16 per cache line on kernels that are nothing but memory traffic. Walking the
channels instead cut them 8x, to 8% of GPU time, and took the search from ~97
to ~143 visits/s on the same throttled GPU. The page now beats the native
OpenCL build by a third on the same laptop.

**Two server threads keep two evaluations in flight** (`?servers=N`, 2 by
default): the GPU runs one batch while the other is packed, submitted and read
back. Paired A/B: +5%, which is the pack-and-readback share of the wall clock,
exactly what the stage timers say evaluate spends outside the GPU.

**The batch wait exists because batches averaged exactly half their size.**
Instrumenting the batches (`window.engineStats`) showed the GPU 97-99% busy in
every config while the average batch sat at half the thread count, in
complementary pairs (10 and 22, 7 and 25): when a batch of K returns, its K
threads re-queue a few milliseconds later, but the serving thread grabs the
other threads immediately and the two groups never merge. `nnServeBatchWaitMicros`
lets the serving thread wait that long for a partial batch to keep filling (a
full one is run at once). At 3 ms nearly every batch is exactly full, the GPU
does ~110 rows/s at batch 32 instead of ~92 at batch 16, and the search is worth
12% more. So the page is GPU-bound throughout -- the search itself never was the
limit at 16 or 32 threads -- and reaches 91% of native at 32. Native is still
gaining at 64 where the page has fallen apart, because every thread here is a
worker and sixty of them cost more than they bring.

**Sixteen is what it ships with, not thirty-two.** A lone search is fastest at 32,
but the ui asks for two positions at once -- the evaluation graph fills in while a
position is searched -- and the threads of both searches queue against the same
net, so what sets the batch is the total. Two analysis threads of 16 stay at the
32 that measures best; two of 32 would spend that time in the 64-thread regime,
which is worse than either. The batch wait ships at 3 ms (`?batchwait=N` to
override).

**Each thread keeps two leaf evals in flight** (`?leaves=N`, 2 by default),
because the batch a search could form was capped by its thread count: every
thread sat blocked on the one eval it had submitted. Now a thread that reaches
a new leaf queues the eval and goes back to start another playout, collecting
results once two are pending, so 16 threads fill 32-row batches and a lone
search gains 18% (132 to 156 visits/s). Bigger batches keep paying on the net
alone -- its ladder climbs to ~200 rows/s at 96-128 rows -- but three evals per
thread measures 10-20% *worse* than two: a pending playout holds virtual losses
along its whole path, and past two per thread that distorts move selection by
more than the rows buy.

That the curve keeps climbing past 16 at all is a late correction. Measured
earlier the same day it flattened at 66 from 16 threads on, which reads exactly
like a search too slow in WebAssembly to feed the net. It was not: four
`python3 -c "x=0 while True: x+=1"` processes, twelve days old and belonging to
nobody, were holding a third of the machine. The search is the part that is
CPU-bound by construction, so it degraded first and looked like a wasm ceiling.
Killing them moved every point on the curve up and restored the climb.

Those numbers are one significant digit. **Measuring anything here is harder than
it looks, and every figure in this section survived only because it was measured
in an order that runs against its own bias.** Three traps, all hit:

- **A closed browser is not a dead browser.** Playwright's `browser.close()`
  leaves the whole process tree alive when the page holds a wasm engine: it has
  tens of worker threads and a WebGPU device, and the renderer never tears down.
  One left over from a finished run was still holding 817 MB and 13% of a core
  seven minutes later. Runs after it are then racing a whole live engine, so a
  session decays as they pile up: the same 8-thread config gave 56-65 visits/s
  early in one session and 22-25 twenty minutes later. That decay has the shape
  of whatever you happen to be sweeping. Stop the engine before closing, kill
  anything whose `comm` is `chrome-headless` between runs, and refuse to start a
  run while one is alive.
- **Sweep in a reversed order** -- 8, 16, 32, 32, 16, 8 -- so drift cancels
  instead of being read as an effect. This is what caught the leak: the same
  thread count measured three times slower at the end of a session than at the
  start.
- **The GPU throttles from 950 MHz down to ~400-600 under sustained load**
  (`/sys/class/drm/card1/gt_cur_freq_mhz`), so absolute numbers depend on how
  hot the machine already is, and block-ordered comparisons read heat as an
  effect. Any A/B here has to alternate whole runs between the two configs
  (`bench2.py` style) and compare pairwise.

The fp16-against-fp32 ratio is deliberately not quoted here. It was measured at
1.6x, in a block of fp16 runs followed by a block of fp32 runs, which is exactly
the design the leak defeats. A paired measurement over 189 moves put it at 1.2x
instead. Precision is worth something, but not what this page claimed.

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
atomic word; `src/netRunner.ts` reads them, evaluates the net, writes the outputs
back and wakes the engine. The two files are halves of one protocol. What
evaluates is a parameter: the hand-written WebGPU model in a page, TensorFlow.js
on the CPU under node.

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

## fp16 against fp32

Half precision costs accuracy and buys speed. MCTS makes the trade non-obvious,
because a search can absorb a noisier evaluation but cannot get back the visits
it did not have time for, so the question is strength at a fixed time budget
rather than error on one evaluation.

`match.html` plays the two precisions against each other. Both engines live in
one page, one worker each, only one searching at a time, so they see the same
GPU. `scripts/match-headless.mjs` hands the page one game at a time and records
the result, which makes a run resumable:

    node scripts/match-headless.mjs --threads 16 --a fp16:time:1 --b fp32:time:1 \
      --openings openings/balanced-11.json --repeats 6 --out results/time-1s.jsonl
    node scripts/score.mjs results/time-1s.jsonl

A side is `<precision>:<condition>`, the condition being `policy`, `visits:N` or
`time:S`. Naming both sides is what lets one harness answer more than one
question: the same precision at two visit counts prices what a doubling of
search is worth, which is how a speed ratio becomes elo without borrowing a rule
of thumb from go.

### Games have to be paired, and openings have to be close

Hex is a first-player win, so an unpaired game measures the opening. Every
opening is played twice, once with each side as black, and the pair is scored
together. That cancels the first-player advantage exactly, and it removes the
reason to implement the pie rule: an engine that does not know about swapping
would swap every good first move, which moves the bias to the other side rather
than removing it.

Pairing is not enough on its own, because a pair from a decided opening ties.
**No first move on an 11x11 board is anywhere near even.** All 61
rotation-distinct ones, searched at 800 visits, come out as black winrates of

    row 1   a1 0.6  b1 0.6  c1 0.6  d1 0.4  e1 0.5  f1 0.5 ... k1 89.1
    row 2   a2 16.5  b2 17.0  e2 2.4  f2 3.4  h2 3.9  i2 12.4  k2 98.4
    row 3   a3 8.1  d3 97  e3 97  f3 95  g3 93  h3 93  j3 92  k3 6.4
    row 5   a5 96.4  b5 99.4  c5 99.4          centre  f6 99.7

The best of the 121 is `b2` at 17%, and nothing sits between 17% and 84%. From
the centre, white is under 2% within four plies. A bigger board does not help:
twelve sampled first moves on 13x13 give the same bimodal shape with a harder
edge, `e2` at 3.8% and `e3` at 85.9%. With no draws the value head saturates
against a game whose value is 0 or 1, so the fix is granularity, not size.

Two moves give it. From a first move black is losing, white's replies run from
the best, which keeps black losing, to the worst, which hands black the game, so
one of them lands near even. `scripts/book-headless.mjs` searches each parent
with a widened root -- the default analysis search spends almost nothing on a
reply it dislikes, and a reply with two visits has no value worth reading -- then
plays the most promising replies out and searches each on its own before
believing it. `openings/balanced-11.json` is 48 positions between 38.4% and
61.7% across 20 first moves.

### Two things that corrupt the measurement

**A leaked engine.** `browser.close()` does not take the page's process tree with
it: the engine's pthread workers and its WebGPU device keep the renderer alive,
so the tree survives, still burning CPU and still holding a device, and the next
run measures itself against it. One leaked tree was 6:45 old and 817 MB. The same
config measured 56-65 visits/s early in a session and 22-25 late, monotonically
downhill -- which imitates a thread-count effect or a precision effect depending
on the order the runs go in. `scripts/browser.mjs` drops the engines before
closing, waits for the tree to be gone rather than trusting `close()`, and
refuses to start while a `chrome-headless` process is alive.

**Thread count.** It sets the batch size the search produces, and half precision
is worth much more at a small batch than a large one, so it is not a detail the
two arms can inherit. Every game records the thread count it was played at.

### Statistics

Most pairs tie by construction, so a normal approximation over all pairs is
generous. `score.mjs` prints a sign test over the decisive pairs alongside it,
and they disagree: the intuition-only arm reads as -73 elo with a 95% interval
excluding zero, and as 1 of 7 decisive pairs, p = 0.125.

### What it found

11x11, 48 paired two-move openings, 16 search threads, one second a move where a
clock is involved.

| arm | matchup | result |
| --- | ------- | ------ |
| intuition | fp16 vs fp32, one visit a move | -22 elo [-82, +37], 7 of 17 decisive pairs, p = 0.63 |
| fixed time | fp16 vs fp32, 1s a move | **0 elo [-73, +73]**, 12 of 24 decisive pairs, p = 1.000 |
| search | fp16 at 400 visits vs fp16 at 100 | +232 elo [178, 298], 59 of 62 decisive pairs, p = 0.000 |

**Half precision is worth nothing measurable at a fixed time budget, and the
weights take 53 MB of device memory instead of 105.** That is the whole result:
choose fp16 for the footprint, not for the strength. The download is the same
either way -- `hex27x3.bin.gz` stores fp32, and `?fp32` only changes what the
loader casts it to.

The three arms agree. fp16 serves 117 evaluations a move against fp32's 99, so
1.18x the work, which is 0.24 doublings. The search arm prices a doubling at 143
elo (over evaluations; 125 over visits, which are inflated differently in the two
arms). So fp16's speed is worth about **34 elo** -- real, but a quarter of what
the fixed-time arm could resolve. Separating 34 elo from zero needs some ten
hours of games, against the two hours these took.

The search arm is also the positive control the other two need. The same
harness, the same openings, the same scoring: when a real difference exists it
comes out at 59 of 62 decisive pairs and p = 0.000 on 96 pairs. So the fixed-time
null is a null, not a broken rig.

### Two results that were wrong, and how they were caught

An earlier fixed-time arm gave fp16 **+129 elo at p = 0.002** over 48 pairs. It
did not replicate: rerun with the precisions swapped between the two engine
instances, on an engine that had meanwhile got 1.5x faster, it came out at
exactly 0. Both things changed at once and the cause cannot be separated. What
flagged it was not the statistics -- p = 0.002 looked fine -- but that the effect
disagreed with its own mechanism: 129 elo needs 0.78 doublings of search and fp16
had 0.24. **An effect that does not match the size of its cause is worth
distrusting however good its p-value.**

The intuition arm first read -73 elo on 24 openings, 1 of 7 decisive pairs. On 48
openings it reads -22 with 7 of 17. Both were the same null; the first was small
numbers.

Two measurement traps behind them, both worth avoiding:

- **A leaked browser.** `browser.close()` leaves the page's process tree alive
  when the page holds a wasm engine, and a leaked engine still holds a WebGPU
  device. One config measured 56-65 visits/s early in a session and 22-25 late,
  monotonically downhill, purely from the leaks piling up. See
  `scripts/browser.mjs`, which also refuses to start while any headless chromium
  is alive, since this machine is shared.
- **Visits are not work.** A move can spend thousands of visits walking a subtree
  the nnCache already holds, and none of those reach the GPU. Two byte-identical
  engines came out at 220 and 141 visits/move in aggregate on that alone, with a
  per-game median ratio of 1.01 and one game at 25.8x. Which arm gets the freak
  games flips between runs. Evaluations per move is stable; visits per move is
  not.
