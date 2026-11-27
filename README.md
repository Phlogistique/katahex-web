# KataHex in the browser

Running the KataHex engine as a web page: KataHex's C++ board, input features and
search compiled to WebAssembly, with the neural net evaluated on the GPU through
WebGPU. Nothing is sent anywhere; the search runs in the tab.

**https://phlogistique.github.io/katahex-web/** -- needs a browser with WebGPU.

Licences are in `NOTICE.md`: the site is AGPL-3.0-only because the ui is
PlayHex's, the engine stays MIT, and the net is HZY's, redistributed with his
permission.

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

Not in the web platform. `micro.html` runs the matmul kernel written by hand in
WGSL, which is what a convolution becomes, over the shapes the net spends its
time in:

| shape                                 | fp32 | fp16 |
| ------------------------------------- | ---- | ---- |
| 2048 square                           | 565  | 974  |
| the trunk's Winograd matmul, batch 48 | 548  | 958  |
| its 1x1 convolutions, batch 48        | 546  | 955  |

GFLOP/s at the tile the backend ships, out of the several the page sweeps. The
net's skinny shapes cost nothing against the square one.

The net needs 6.33 GFLOP per evaluation, 90% of it in the 62 3x3 convolutions of
the trunk. At 52 evaluations a second onnxruntime is getting 329 GFLOP/s out of
this GPU, a third of what the hand-written kernel above reaches in the same
browser. Feed the net through a kernel that good and it would run at about 150
evaluations a second, past native's 128 -- and that still holds onnxruntime's
direct convolution fixed; the hand-written backend puts Winograd on top of the
tuned kernel and measures 224 below.

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

Weights sit in their buffers as f16 whatever the arithmetic reading them. The
net the page is served is already an fp16 net, so for everything but the
Winograd transformed filters those are the bits the file holds, and the f32
kernels read them back a pair at a time out of a `u32` with `unpack2x16float`,
core WGSL that asks for no feature. At board 11 that is 185 MB of weight
buffers. Holding them at the compute width would be 369 MB on the single
precision path, which is the path a device without `shader-f16` takes and so
the one least likely to have the memory for them.

The transformed filters are 97% of those bytes and the whole of what the
rounding costs: the transform multiplies its taps by 1/6 and 1/24, so what
comes out is not an f16 number whatever went in. The folded batch norm is
under a thousandth of the bytes and stays f32, the precision `export_net.py`
keeps in the file for it. What is left is 0.005 mean policy logit error over
the 512-position bank against the same net at full weight precision, and no
top-1 move changed in 512; half precision, for scale, is 0.045.

Evaluations per second on 11x11, one sitting, each batch size measured on the
way up the ladder and again on the way down and the two averaged, because the
GPU throttles as the sweep heats it. One significant figure:

| batch | fp16 | fp32 |
| ----- | ---- | ---- |
| 1     | 38   | 31   |
| 2     | 74   | 60   |
| 4     | 110  | 87   |
| 8     | 148  | 106  |
| 16    | 185  | 107  |
| 32    | 207  | 119  |
| 48    | 212  | 121  |
| 64    | 224  | 116  |
| 128   | 224  | 106  |

The two columns are not comparable to each other: the fp32 sweep ran at a
median 550 MHz against the fp16 one's 700. A single evaluation takes 26 ms
where onnxruntime and native OpenCL both take about 90, which shortens the
latency-bound early search, and the top of the curve is well past the 128
evaluations a second native OpenCL reaches on this GPU. onnxruntime was
2.1-2.4x behind at every batch size when the two were last measured side by
side, before the Winograd transforms and the matmul were tuned.

The matmul carries nearly all of the arithmetic, so it is the kernel that has
been tuned. A workgroup computes 64 output columns by 64 rows, or 32 rows where
rows are scarce -- the Winograd matmul has 9 real rows per image on 11x11 --
and each thread holds 4x4 of them in vec4 registers. Both tiles move through
shared memory as vec4, 32 steps of the reduction at a time (16 for the head
layers, whose channel counts do not divide by 32), with the reduction emitted
unrolled. Against the scalar version that is 1.207 [1.200, 1.213] over
the whole net at batch 48 and 1.525 [1.485, 1.566] at batch 1 (tier 3), and 735
to 958 GFLOP/s on the trunk's Winograd matmul alone.

`micro.html` sweeps the tile parameters over those shapes. Two that look better
on paper are not: 4 rows by 8 columns per thread wins 2% at batch 48 and loses
8% at batch 1, and both it and 8 rows by 4 fall to 15-100 GFLOP/s in fp32,
where the accumulators stop fitting in registers.

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

### The non-regression gate

A change to `webgpuModel.ts` is checked in three tiers, without playing a
single game. `scripts/make-positions.mjs` built the fixtures once: it replayed
positions from recorded games (plus constructed ones -- empty boards, lone
corner stones, stones packed into the board edge the Winograd tiles pad, a
saturated win, the same position with each side to move) through the node
engine at one visit, capturing the 22-plane feature tensors where `serveEvals`
sees them, and froze the TensorFlow.js fp32 outputs -- the implementation that
agrees with native to 1e-6 -- as goldens under `public/check/`.

    node scripts/check-headless.mjs   # tiers 1+2, ~1 min against a running dev server
    node scripts/check-perf.mjs       # tier 3, ~3 min, working tree against HEAD

- **Tier 1, correctness**: `check.html` runs the hand fp32 backend over 18
  stored positions and compares all four heads against the goldens, tolerance
  1e-3. Observed agreement is 1e-5 to 3e-5; real bugs land orders of
  magnitude higher; nothing plausible lives in between. The same positions go
  through as one batch of 16, as 5+11, and alone -- batch shape must not
  change an answer (observed: bit-exact) -- and tiled to batch 48 against the
  goldens, since each batch size compiles its own plan. Tier 1 runs the
  backend on f32 weights, which no page runs: the f16 pairs the page holds
  move the outputs by 5e-2, tens of times the tolerance, so the goldens would
  be judging that rounding and nothing else. The rounding is judged on its
  own, the model the page runs against the exact one over the same positions,
  limit 0.15.
- **Tier 2, fp16 accuracy**: fp16 against the live fp32 that tier 1 just
  pinned, over 512 stored positions, so fp32 stands in for the oracle at
  fp16's 0.05 error scale. Mean, p95 and max of the policy logit error, the
  worst per-cell signed bias (which is what an edge-tile bug looks like),
  top-1 flips, policy KL and the value head's winrate shift, each against a
  constant frozen at calibration in `public/check/frozen-11.json` -- never
  against the previous run, so drift cannot ratchet. The mean sits ~7
  standard errors under its threshold and a doubling of error ~16 above; the
  whole pipeline is deterministic, so a passing run repeats to the last digit.
- **Tier 3, speed**: the working tree against a baseline commit, the check
  page in both versions bundled the same way and served side by side, one
  browser, two tabs, never both timing at once. Timed slices alternate
  between the tabs, each pair alternates its order, and a pair whose two
  clock bands (`gt_act_freq_mhz`, sampled at 250 ms) disagree by over 15% is
  thrown out. Verdict is a paired t on log-ratios plus a sign test: fail
  under 0.95 with the 95% interval excluding 1.0, warn under 0.98. An A/A
  run across a 950-to-650 MHz throttle reads 0.996 [0.966, 1.027] at batch
  48, so a drift has to reach roughly 5% before this convicts it.

The correctness tiers were proven on seeded bugs: fp16 weights quantized 16x coarser
fails every tier-2 metric (mean 0.15 against the 0.065 limit), and an
off-by-one in the Winograd untransform's edge clip -- which writes one row
into the next image -- fails tier 1 at errors around 10 and breaks batch
invariance, while leaving the first image of every batch clean.

What this cannot catch: anything outside the net evaluation (the search, the
bridge, the batching logic), an error of the same size and shape as fp16
storage noise, and a speed change under a few percent. A change to those
still wants `match-headless` and `score.mjs`, which is hours rather than
minutes.

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

`ui/` is the hexplorer, vendored from PlayHex. It reaches its engine through a
small bridge on `window` -- `Native.start(size)`, `Native.query(json)`, and the
engine's two output streams back as `onEngineLine` and `onEngineLog` -- which the
Android app implements in Java over a child process. `src/wasmEngine.ts`
implements the same bridge with the engine compiled to WebAssembly, so the same
ui becomes a page that needs no server:

    npm --prefix ui install
    npm --prefix ui run dev -- --mode web       # then open /web.html

`src/engineWorker.ts` is the whole page-side engine: one worker holding the wasm
engine, the net on WebGPU, and `src/netRunner.ts` between them. It takes about ten
seconds to start, nearly all of it the net. `?fp32` runs the net in single
precision.

`--mode web` is not optional: without it vite serves no public directory, and
`/katahex.js` comes back as the index page with a 200, which looks like an engine
that never appears and reports nothing anywhere.

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

### Browsers

Chromium runs it. Firefox does too, but not as shipped: WebGPU is off by default
on Linux and a page cannot ask for it, so the worker logs which switch when
`navigator.gpu` is missing. Set `dom.webgpu.enabled` in `about:config` and
Firefox 154 has `shader-f16` in a worker and runs the page end to end, at about
half the throughput: 118 net positions a second against Chromium's 236, same
machine and same search.

### Without a GPU

When WebGPU is missing or fails to give a device, the worker runs the net on the
CPU instead (`src/cpuModel.ts`): the vendored TensorFlow.js graph on its
WebAssembly backend, with SIMD and threads, all cores but one. Same net, same
weights, so the same moves, only slower. `?cpu` forces it on a machine that has
a GPU.

A device is not yet a GPU that runs the net: a browser can hand one out and then
reject the shaders or the buffers, and once the engine is running that is fatal.
So the worker first evaluates an empty board inside WebGPU error scopes, and
anything it throws, reports or returns that is not finite sends the net to the CPU
as if there had been no GPU. Tested by feeding the trial a broken shader on
Chromium's SwiftShader adapter: it logs the WGSL error's first line and the engine
comes up on the CPU.

Only TensorFlow.js's core and wasm backend are bundled, aliased in
`ui/vite.config.ts`, and only into a chunk the worker imports when the GPU has
failed, so a page with a GPU downloads none of it. That needs module workers,
which is why the ui builds its workers as ES modules.

The GPU's engine settings would waste a CPU: sixty-four leaves in flight a
thread keep a GPU in wide batches, where on a CPU a batch costs in proportion to
its rows and every leaf past the first may never be used. On the CPU the engine
runs one search thread with two leaves in flight, and one serving thread.

Measured in a headless Chromium with no GPU, four shared cores: ready 3 seconds
after the page loads, about 4 net evaluations a second on 11x11, so 16 visits in
4 seconds. The survival game at one visit is fully playable there on 14x14,
KataHex answering a move in 0.3 to 0.4 seconds; its setup says the net is on the
CPU and offers to set both searches to one visit.

Firefox 153 (Playwright's build, headless, the same four cores) runs the page on
the CPU end to end -- the service worker's reload, the engine's pthreads, the
TensorFlow.js chunk in the module worker -- but about eight times slower: 2 to 3
seconds a net evaluation, against Chromium's 0.3. That is Firefox's WebAssembly,
not this page: TensorFlow.js alone, twenty 384-channel 3x3 convolutions on
11x11, single-threaded, does 25 GFLOP/s in Chromium and 2.3 in Firefox (4.3 on
three threads), and 0.5 with SIMD off. The survival game at one visit still
plays: self-play on 7x7 hands over in 12 seconds.

### The two weight files

The page is not served `hex27x3.bin.gz`. `scripts/export_net.py` writes what its
two readers each need, into `public-web/`:

    net-fp16.bin.gz    49 MB   convolution and matmul weights as float16
    net-shape.bin.gz  100 KB   the same file with every float array zeroed

Both are still KataGo weight files -- the half precision arrays sit under an
`@BINF16@` marker, which `vendor/binModelParser.ts` reads and `desc.cpp` never
sees -- so the engine loads the second one unchanged. It only ever reads the
name, the version and the channel counts: the JS backend evaluates nothing, so
the weights it parses are dead the moment they are parsed.

The half precision file is not an approximation of what the page used to run.
The WebGPU backend converts every weight to float16 on its way to the buffer, so
for the arrays it uploads unchanged the exported bits are the bits it would have
computed -- `npm run check:export` pins exactly that, which needs `export_net.py`
to round the way `toHalf` does, by truncating. Two exceptions, both deliberate:
batch norm statistics stay float32, because the GPU stores the batch norm folded
into a scale and a bias and rounding the inputs of that division would be a
second, avoidable loss; and 3x3 kernels are Winograd transformed before they are
rounded, so those outputs do move. `check.html` measures that -- mean policy
logit error over the 512-position bank goes from 0.0457 to 0.0460, against a
frozen limit of 0.065.

### Publishing

`public-web/` is the whole site apart from the ui bundle: the engine, the two
weight files, and coi-serviceworker. The rest of `public/` is a few hundred
megabytes the benchmark and the gate use, and is not served.

A static host sends no `Cross-Origin-Opener-Policy` or
`Cross-Origin-Embedder-Policy`, and without them there is no `SharedArrayBuffer`
and so no engine threads. coi-serviceworker registers a service worker that adds
the headers to every response and reloads the page once. `ui/src/web.ts` starts
nothing while `crossOriginIsolated` is false, so that first load does not pull
49 MB of net to have the reload throw it away.

Asset urls are relative and the engine and net paths resolve against
`document.baseURI`, so a build serves from a project page's subpath as happily as
from a root.

`.github/workflows/pages.yml` builds the ui and deploys on a push to `main`. The
engine and the nets are committed, because building them needs emsdk and a 97 MB
file that is not in this repository:

    scripts/build-engine.sh                                    # needs emsdk sourced
    uv run --with numpy python scripts/export_net.py \
      path/to/hex27x3.bin.gz public-web/net

### Survive KataHex

The sidebar's Survive tab opens a game over the board (`ui/src/survival.ts` for the rules,
`ui/src/SurvivalGame.vue` for the game). The url names an open game, `?game=survival` here and
`?game=find-move` for the game below, so a link or a reload opens it directly. It opens as a game with the swap rule does, from
PlayHex's swap maps (`ui/src/shared/swap-maps`, computed with this same net at 1000 playouts a
cell): red plays a fair first move, likelier the closer KataHex rates it to even, and blue
swaps it or not at random, likelier the stronger it is, as PlayHex's own AIs do. Either side
is then fair to take, and the player's is drawn at random unless chosen. KataHex plays itself,
the player's side taking the move its own search prefers and the other a random move whose
network policy is between 5% and 30% of its best move's, until that search gives the
player's side at least 90%, or for 20 moves at most, after which the player takes over
wherever it got; if the game ends first, it stops there and says to change the settings. Then that side is the player's, on a Fischer clock of 20 seconds
plus 5 a move that increments never take past 25, and KataHex answers with its own search.
After each of the player's moves KataHex answers, and the position after its answer is rated,
by that same search, as the winrate of the move it chose: one search a move. A game that is
lost so ends on the move that won it for KataHex; the score is how many moves
the player makes before their side is rated under 10%.

The swap can be turned off. Without it red is winning from the start, KataHex rating the
empty board 99.8% for red on 11x11 and 99.6% on 14x14. From a fair opening, at one visit on
14x14, self-play got there within 20 moves in four games out of five; the fifth was handed
over at 76%.

The thresholds can be set to a millionth of a percent, and the odds option shows the bar and
the graph in log-odds, for playing on 11x11 between 99.99% and 0.01%. Whether KataHex's
winrates mean anything to a human that far out is an open question. The net alone, at one
visit, does not get there: in self-play on 11x11 it rates the winning side at most around
4000:1, near move 30, and then less and less sure as the board fills, down to a few hundred to
one when that side connects. A search may do better, since it sees connections coming; that is
untested.

It is played on 14x14 by default: on 11x11 the winrates saturate so fast that there is
hardly a game. Every number there can be changed before a game, and the best score is kept
for each combination of them, since a score only compares with games played the same way. The
clock stops while KataHex is thinking. The analysis board's searches are suspended while the
game is open, so its searches have the engine to themselves.

Take back, on unless turned off, takes back the player's last move and KataHex's answer, with
the score they had, during a turn or once a move has lost; best scores are kept apart for games
with and without it. It gives no time back: the clock runs on, and a game lost on time stays
lost. The clock can be turned off, for all the time in the world.

Again replays the same game: the position the last one was handed over at, the same side, a
full clock; KataHex answers much as it did, so a line that lost can be tried another way. New
game self-plays a new one. The randomness is in the opening, the swap, the side drawn and the
weak side's every self-play move; the strong side and KataHex's answers are its best moves.

A finished game can be loaded into the analysis board, to see where it was lost, without being
asked to confirm: the tree it replaces is most often the last game's. Every search
the game ran goes into the board's analysis store, so the evaluation graph draws the positions
the game already rated without searching them again, wherever the game's search was as deep
as the board's analyzer asks for: 200 visits by default, as are self-play's searches, where
KataHex's answers, at 100, are searched again.

### Find KataHex's move

The sidebar's Find the move tab opens a second game over the board (`ui/src/findMove.ts` for the
rules and the decoys, `ui/src/FindMoveGame.vue` for the game). The player is red against
KataHex, but does not place stones freely: each turn three cells are marked A, B and C, one the
move KataHex's search plays and two decoys, moves that look natural and after which a search
rates red under 30%. A decoy ends the game, with blue's answer to it played and KataHex's move
circled; the score is how many of KataHex's moves the player found. Blue answers each move with
the net's favourite, at one visit.

Red cannot simply start on the empty board without the swap: KataHex rates it 99.8% for red on
11x11, and in self-play from there nothing natural lost for red, every candidate kept red over
99%. So red opens on a fair first move from PlayHex's swap maps, which red keeps, and the net
plays itself, red its favourite move and blue a middling one as in Survive, until the net rates
red at least 85% after at least a random number of moves up to ten. Red is then ahead, but the
position is still a fight, and from there KataHex's searches rated red between 96% and 99%
while most natural alternatives lost.

The decoys are drawn from moves a human might play: the net's eight favourites, the cells
around KataHex's move (adjacent, or a bridge away), blue's three favourites, red's favourite as
if KataHex's move were already on the board, and, for each stone, the move whose policy gains
most once that stone is taken away, as if it were not there. They are tried nearest the fight
first, within two cells of KataHex's move or of blue's last, then by policy: a far-away decoy is
dismissed at a glance, where a neighbour of the right move has to be read out. The net alone
rates each candidate first, one batch of single evaluations, and only those it does not rate
over 80% for red are searched, until two lose. When none does, KataHex plays its move for the
player and the game goes on.

Measured with the native engine on the CPU, 100 visits for KataHex's move and 32 to check a
decoy, on 11x11, one game of 11 questions so far: every question found its two decoys, ten
of them among the first two candidates searched, and red playing KataHex's move stayed between
96% and 99%. KataHex's move was the net's favourite 8 times out of 11 (where the net alone gives
it away, at a policy of 0.99, 4 times), its second the other 3. Decoys were often a neighbour of
the right move: e7 against d7 and f7 beside it, both under 18%. A question took 20 to 30 seconds
on 4 CPU cores; the page on a GPU searches several times faster.

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

**One search thread with 64 leaf evals in flight is what it ships with**
(`?threads=N`, `?leaves=N`). The batch a search can form is set by how many
evals it keeps in flight, and threads were only ever a way of holding evals: a
blocked thread holds exactly one. Above one leaf eval per thread, a thread
that reaches a new leaf queues the eval and goes back to start another
playout, collecting results as they return -- so in-flight count and thread
count come apart, and one thread feeding the queue keeps the GPU in ~50-row
batches by itself, with none of the cost of sixteen workers. The ui still
asks for two positions at once (the evaluation graph fills in while a
position is searched), now one thread each. The batch wait ships at 3 ms
(`?batchwait=N`).

**64 in flight is a measured knee, and the currency it is paid in is move
selection.** Every pending playout holds virtual losses along its whole path
so the next descent goes somewhere new; the more there are, the more the
search is aimed by fake losses. Sweeping one thread's in-flight count: 32
gives 168 visits/s, 48 gives 180, 64 gives 188, 96 falls back to 182 -- where
sixteen threads of two in flight, the previous default, did 157, and the
original page with every thread blocked on the one eval it had submitted did
132. Whether a 64-in-flight visit is worth as much as a 32-in-flight one is a
strength question these wall-clock numbers do not answer.

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

Alternating whole sweeps is still not enough. Four sweeps in each order, minutes
apart on an idle machine, read 11x11 batch 32 at 140 and 100 evals/s in fp32 and
at 145 and 188 in fp16 -- a ratio of 1.00 one way round and 1.76 the other. The
reversed sweep detects the drift (the same run's up and down legs disagree by up
to 50%) but does not remove it. A precision ratio off this page needs the two
models alternating measurement by measurement in one process, not sweep by
sweep. The numbers above also predate the coalesced Winograd transforms and now
read low.

## Engine

`engine/` is the KataHex fork, carrying the cmake fixes for targeting Emscripten
and the JS backend below; it is a git subtree, squashed, of
https://github.com/selinger/katahex.

`scripts/build-engine.sh` cross-compiles it to WebAssembly and stages the result
in `public-web/`. 1.8 MB of wasm.

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

**Half precision is worth nothing measurable at a fixed time budget.** That is
the whole result: choose fp16 for the speed, not for the strength. Both paths
store the same f16 weights, so what fp16 buys on top of them is the
activations, half the size, and the throughput below. The download is the same
either way, and so is what the weight buffers hold: `?fp32` changes the
arithmetic and the activations, not the weights.

Those games ran at 16 search threads with one leaf evaluation in flight each,
which the driver took by accident: it posts to `engineWorker.ts` directly and so
inherited that module's default rather than the page's. Batches of ~16 rows is
the smallest the config space reaches, and the smallest batch is where half
precision has least to gain, so the null above is the least favourable reading
of fp16 available. At the 1 thread by 64 leaves the page now ships, a paired run
over 8 games puts fp16 at 163 evaluations a move against fp32's 127 -- 1.28x
rather than 1.18x, so about 52 predicted elo rather than 34. Still inside the
interval, so the conclusion carries; it has less room than it did.

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

## On a rented GPU

The same ui can run its engine on a GPU rented by the hour from vast.ai instead of on the
phone. On an RTX 3060 at about $0.06 an hour the b28 net searches some 900 visits a second on
11x11, four times what the laptop's Iris Xe gets out of a net a third its size.

    server/        the engine on CUDA behind a websocket, as a Docker image
    android/       the app: the ui in a WebView, renting and talking to that image

### The engine

Upstream KataHex, not the fork under `engine/`: `hzyhhzy/KataGomo`, branch `Hex2024`, pinned
in `server/Dockerfile` (4bbf953, July 2026). The fork's changes are for WebAssembly;
upstream has moved on since (ONNX and TensorRT updates, move-limit openings) and none of it
touches analysis.

One patch, `server/hexgui-coords.patch`. Upstream names cells the way a Go board would print
the Hex board for Sabaki -- doubled x, y counted from the top -- so `f6` is not a cell to it and
the analysis engine answers the ui's first query with "Could not parse board location". The
patch puts back `Location::toString` and `tryOfString` from Selinger's fork, `a1`..`s19` as
HexGUI writes them, which is what the ui sends and expects. The internal board is the same in
both, so nothing else moves.

The net is HZY's newest, `hex3_27x_b28.bin.gz` from the KataHex 20250131 release: a
b28c512nbt retrained on the 2024 data, +100 Elo on 15x15 and +200 on 19x19 over the 2024 net,
the strongest released for boards up to 27x27. Its sibling `hex3_mm19x_b28` supports the
move-limit mode and is slightly weaker; it is not used. The image pins its sha256.

Built with `-DMAX_BOARD_LEN=19`, matching the ui's limit. The CUDA backend needs no tuning,
unlike OpenCL, which would spend minutes tuning on every fresh machine.

### The server

`server/server.py` carries the ui's engine bridge -- `Native.start(size)`, `Native.query(json)`,
and the engine's stdout and stderr back -- over one websocket, as JSON text frames:

    client -> server   {"t": "start", "size": 11}      (re)start the engine for a board size
                       {"t": "q", "q": "<query json>"} one line to the engine's stdin
                       {"t": "stop"}                    stop the engine
                       {"t": "ping"}                    a sign of life
                       {"t": "bye"}                     destroy this machine now
    server -> client   {"t": "line", "l": "..."}        a line of the engine's stdout
                       {"t": "log", "l": "..."}         a line of its stderr
                       {"t": "pong", "left": s, ...}    seconds before the idle switch fires

`ws://host:port/ws?token=...`, the token random per rental and passed in its environment.
`GET /health` answers `ok` without one, which is how the app knows the machine is up before it
has anything to say. Plain ws, not wss: the machine is reached by bare IP and a port vast maps
at random, so there is no certificate to have, and the token is the only secret on the wire.

- The engine starts at boot on 11x11 (`WARMUP_SIZE`), so the first board is ready when the
  app connects. A client that asks for the size already running keeps the engine and its
  cache: the server sends the engine a `terminate_all` for whatever the last client left
  searching and replays the "ready to begin handling requests" line, which is what makes
  the ui resend its pending and live queries. Its own `terminate_all` reply is not forwarded.
- 32 search threads whatever the core count (`SEARCH_THREADS`), batch 64, two analysis
  threads. Threads are what make batches, and they spend their time waiting on the GPU, not
  on a core: sized from the cpu count, an RTX 3060 on a 4-core slice got 8 and ran at half
  speed (457 against 928 visits a second).
- `BENCH_VISITS` runs one search at boot and logs its speed. A machine nobody can reach yet
  -- this repository's CI, say -- says through vast's log API whether its GPU works.

### The dead man's switch

Every message from a client is a sign of life. With none for `IDLE_TIMEOUT` (600 s), none at
all for `FIRST_CONNECT_TIMEOUT` (900 s) after boot, or `MAX_LIFETIME` (12 h) after boot, the
server destroys its own instance -- `DELETE /api/v0/instances/$CONTAINER_ID/` with
`CONTAINER_API_KEY`, the per-instance key vast puts in every container, which can start, stop
or destroy that instance and nothing else, so the user's own key never leaves the phone -- and
then exits whatever the API answered, which stops the container and the GPU's meter with it.
websocket-level pings do not count: only messages the app chose to send.

The app pings every 30 s while it is on screen and not otherwise, so a phone put away, killed
or out of signal stops paying within ten minutes. Tested on vast: an instance left with no
client destroyed itself at 900 s, to the second.

What it cannot cover: a host whose docker never starts the container, since there is then no
server to run the switch. The app handles those (below).

### The image

`ghcr.io/phlogistique/katahex-server`, public, built by `.github/workflows/server-image.yml`
on every push touching `server/`, tagged `latest` and the commit sha.

1.22 GB compressed. It was 2.39 GB on `nvidia/cuda:12.2.2-cudnn8-runtime`, of which 1.27 GB
was one layer of every CUDA library (cuFFT, cuSPARSE, NPP, cuSOLVER...) and 710 MB all of
cuDNN 8, training included. The engine loads cuBLAS, cuBLASLt and cuDNN's inference half
(`libcudnn`, `libcudnn_ops_infer`, `libcudnn_cnn_infer`), so the image is now `cuda:base` plus
those, each library its own layer so that a host pulls them in parallel -- the biggest is
349 MB -- and websockets vendored rather than pip and its dependencies.

CUDA 12.2 because every host offered runs a driver of 535 or later; the app asks for
`cuda_max_good >= 12.2`. `CMAKE_CUDA_ARCHITECTURES` up to 90 includes PTX, which the driver
compiles for newer cards.

### Renting

`android/.../Vast.java` and `RemoteEngine.java`, against vast's REST API:

- Search `GET /api/v0/bundles/?q=`: on demand, verified, one GPU, `compute_cap >= 750`,
  `cuda_max_good >= 12.2`, a direct port, reliability 0.95, 100 Mbit/s down, 20 GB of disk,
  4 GB of GPU memory, under `KATAHEX_MAX_DPH` ($0.50/h), cheapest first. GTX cards are skipped
  by name: a GTX 16xx is compute capability 7.5 without tensor cores.
- Rent `PUT /api/v0/asks/{offer}/` with `runtype: args` (the image's own entrypoint),
  `env: "-e KATAHEX_TOKEN=... -p 8765:8765"`, label `katahex-app`, `cancel_unavail`. The
  cheapest offer is often gone by the time it is asked for; the app walks down five.
- Poll `GET /api/v0/instances/{id}/` until `actual_status` is `running` and `ports["8765/tcp"]`
  is mapped, then `/health`, then the websocket. `status_msg` meanwhile is docker's pull
  progress, which the page shows.
- A host whose own docker fails -- seen: `unknown flag: --runtime`, with the instance left
  `loading` forever -- is recognised by an error in `status_msg` while not running: the
  instance is destroyed, the machine id kept in a blacklist on the phone, and the next offer
  rented. Anything not up in ten minutes is given up on the same way.
- A machine running and mapped that still does not answer `/health` after a minute is most
  likely a network blocking high ports, and the page says so.
- On start the app reconnects to the instance it left running, if it is still there, and
  destroys any other `katahex-app` instance it lost track of.

Back asks, and "Quit and destroy" deletes the instance. Any other way out leaves it to the
switch.

### Which GPU

11x11, the b28 net, one 20000-visit search from the empty board, one machine each, November
2026 prices:

| GPU      | $/h   | search threads | visits/s | visits per cent |
| -------- | ----- | -------------- | -------- | --------------- |
| GTX 1070 | 0.049 | 16             | 232      | 170k            |
| RTX 2060 | 0.062 | 24             | 740      | 430k            |
| RTX 3060 | 0.054 | 8              | 457      | 300k            |
| RTX 3060 | 0.056 | 32             | 928      | 600k            |
| RTX 3070 | 0.056 | 32             | 917 (a 3000-visit search) | 590k |
| RTX 4070 | 0.116 | 32             | 1331     | 410k            |
| RTX 4090 | 0.374 | 32             | 2488     | 240k            |

The GPU is the bottleneck: on the 1070, 232 visits a second at ~17.5 GFLOP an evaluation is
about 4 TFLOPS of the card's 6.5. fp16 is the whole story at the cheap end. Pascal has no fast
half precision, so the cheapest card by the hour is the dearest by the visit; from Turing on
KataGo runs fp16 on tensor cores and a 3060 does four times a 1070 for the same price. Past
the 3060 money buys speed at a worse rate. The 4090's search was short enough for its ramp-up
to show, so it may be undersold.

### Startup

From the rental to an engine ready to answer:

| case                                         | container running | engine ready |
| -------------------------------------------- | ----------------- | ------------ |
| 2.39 GB image, RTX 3070 in Nevada            | 76 s              | ~85 s        |
| 1.22 GB image, same machine an hour later    | 135 s             | ~145 s       |
| 2.39 GB image, GTX 1070s in South Korea      | 135-150 s         | ~2.5 min     |
| stopped and restarted, same machine          | 21 s              | ~30 s        |

Nearly all of it is docker pulling the image, and a host's bandwidth to ghcr.io varies more
from one hour to the next than halving the image saves: the same Nevada machine pulled at
about 60 MB/s, then at about 10. The figure vast advertises (`inet_down`) is measured to the
internet at large and predicts little. The engine itself is up 6-9 s after the container:
CUDA initialisation and the net onto the GPU.

Stopping instead of destroying keeps the disk, image included, on the host, and a restart
takes about 30 s whatever its bandwidth. A stopped instance is billed for its disk only: hosts
ask $0.13-0.40 per GB per month, typically $0.20, so the 12 GB the image needs is $2.40-3.20 a
month. The catch is that the GPU can be rented by someone else while it is stopped, and the
restart then waits. Not implemented.

### The app

A WebView around the ui built as one inlined page (`npm --prefix ui run build`, the default
mode), whose `window.Native` is `RemoteEngine`. Progress goes to the page through
`onEngineProgress`, which it shows while the engine is not ready, and to a bar under the
board, which also shows the GPU, its price and the time left before the idle switch. Tapping
the bar opens a menu: quit and destroy, or change the vast.ai key. Android 15 draws apps under
the system bars; the root view pads for them.

Two builds:

- `android/build.sh` asks for a vast.ai key on first start and keeps it in the app's
  preferences. This is what `.github/workflows/android-release.yml` builds and attaches to
  every GitHub release; pushing a `v*` tag, or running it by hand with a tag, creates the release. Signed with the `ANDROID_KEYSTORE` secret (a base64 `.jks`, with
  `ANDROID_KEYSTORE_PASSWORD` and `ANDROID_KEY_ALIAS`) when the repository has it, so that
  releases install over each other; with a throwaway debug key otherwise.
- `VASTAI_API_KEY=... android/build.sh` builds the key in. That apk is as secret as the key.

## A PlayHex AI worker on a rented GPU

`worker/` is an image that runs PlayHex's AI workers ([playhex/hex-ai-distributed]) on a
vast.ai GPU, and `android/worker/` is a small app that rents one. The workers pull bot moves,
game analyses and Hexplorer positions from playhex.org over HTTPS, with a PlayHex AI worker
key; PlayHex admins hand those out, and they show in the PlayHex settings, "AI workers".

[playhex/hex-ai-distributed]: https://github.com/playhex/hex-ai-distributed

### The image

`ghcr.io/phlogistique/playhex-worker`, built by the same workflow as the server image.

- The engine is PlayHex's KataHex fork (`playhex/katahex`, branch `Hex2024`), pinned, built
  for CUDA. Game analyses need its `kata-raw-nn-batch` command, which upstream does not have.
- The net is `katahex_model_20220618.bin.gz`, the one PlayHex's own workers run, so that a
  PlayHex bot plays the same whichever worker answers. It comes from the Google Drive link in
  hex-ai-distributed's README; the image pins its sha256.
- The worker is hex-ai-distributed at a pinned commit. It takes every katahex job type,
  tree search included (`AI_JOB_TYPES`), and runs KataHex with 8 search threads: PlayHex's
  tree searches are 400 playouts, too few for 32 threads.
- `WORKERS` (2) worker processes, each with its own KataHex. A worker does one job at a time.

`worker/supervisor.mjs` starts the workers and answers the app over plain HTTP:

    GET  /health              "ok", without a token
    GET  /ping?token=...      a sign of life; answers {left, workers, done, log}
    POST /bye?token=...       destroy this machine now

`log` holds the workers' last 40 lines at info level and above.

### The dead man's switch

The same as the server's, 30 minutes long: with no ping for `IDLE_TIMEOUT` (1800 s), none at
all for `FIRST_CONNECT_TIMEOUT` (1800 s) after boot, or `MAX_LIFETIME` (12 h) after boot, the
supervisor sends each worker two SIGTERMs, so that it gives its current job back to PlayHex
and disconnects, then destroys the instance with `CONTAINER_API_KEY`. Tested on a CPU build
against a stand-in for the PlayHex API: the running tree search came back to PlayHex as a
retryable failure, both workers disconnected, then the supervisor called `DELETE`.

### The app

The app asks for a vast.ai key and a PlayHex worker key, and keeps them on the phone. "Start a
worker" rents the cheapest machine under `WORKER_MAX_DPH` ($0.20/h) with the same search as
the KataHex app, passes the PlayHex key in the container's environment, and waits for
`/health`; if a machine fails to start, it is destroyed and the next one rented. While the
app is on screen it pings every 30 s and shows the workers' log, the jobs done and the time
left. Once the app is left, the machine runs 30 more minutes and destroys itself. "Stop and
destroy" ends it at once.

The host of a vast.ai machine can read its environment, PlayHex key included.

`android/build.sh` builds both apps; the release workflow attaches `playhex-worker-*.apk`
next to `katahex-gpu-*.apk`. `Vast.java`, under `android/vast/`, is shared by both.
