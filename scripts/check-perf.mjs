// Tier 3: is the working tree's backend as fast as the committed one?
//
//   node scripts/check-perf.mjs [--baseline <sha>]
//
// Needs a vite server already serving the repo (npm run dev), or set BASE.
//
// Both sides are the check page in perf mode, bundled the same way with
// esbuild -- the candidate from the working tree, the baseline from a git
// worktree at the baseline commit (HEAD by default) -- and served by the same
// dev server, one tab each, never both
// timing at once. Timing on this GPU is dominated by thermal throttling, so
// slices alternate between the tabs, each pair alternates its order, every
// slice records the clock band it ran in, and a pair whose two bands disagree
// by more than 15% is thrown out.
//
// Run on a clean tree this is an A/A: both sides are HEAD, so what it prints is
// the harness's own floor.

import { execFileSync } from 'node:child_process';
import { cpSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { close, finished, open, watchFrequency } from './browser.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const BASE = process.env.BASE ?? 'http://localhost:5173';
const argv = process.argv.slice(2);
const baselineSha = argv.includes('--baseline')
  ? argv[argv.indexOf('--baseline') + 1]
  : execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim();

// Enough evals that a slice runs several seconds: the 250 ms frequency
// sampling needs something to sample, and per-eval noise averages out.
const SWEEPS = [
  { batch: 48, evals: 12, pairs: 12 },
  { batch: 1, evals: 96, pairs: 6 },
];

// ---------------------------------------------------------------------------
// Two bundles of the same page, built the same way

const PAGE = `<!doctype html><meta charset="utf-8"><title>perf</title>
<pre id="log"></pre><script type="module" src="./check.js"></script>`;

function bundle(sourceRoot, outDir) {
  mkdirSync(outDir, { recursive: true });
  execFileSync('npx', ['esbuild', resolve(sourceRoot, 'src/check.ts'),
    '--bundle', '--format=esm', `--outfile=${resolve(outDir, 'check.js')}`],
    { cwd: root, stdio: ['ignore', 'ignore', 'inherit'] });
  writeFileSync(resolve(outDir, 'index.html'), PAGE);
}

console.error('bundling candidate (working tree)');
bundle(root, resolve(root, 'build/perf/candidate'));

const baselineDir = resolve(root, `build/perf/${baselineSha}`);
console.error(`bundling baseline ${baselineSha}`);
const worktree = resolve(root, 'build/perf/worktree');
rmSync(worktree, { recursive: true, force: true });
execFileSync('git', ['worktree', 'add', '--detach', worktree, baselineSha], { cwd: root });
try {
  // The page itself is measurement harness, not measured code: both sides run
  // the candidate's copy, over the baseline's model sources.
  cpSync(resolve(root, 'src/check.ts'), resolve(worktree, 'src/check.ts'));
  bundle(worktree, baselineDir);
} finally {
  execFileSync('git', ['worktree', 'remove', '--force', worktree], { cwd: root });
}

// ---------------------------------------------------------------------------
// Alternating timed slices

const url = (which) => `${BASE}/build/perf/${which}/index.html?mode=perf`;
const { browser, page: candidate } = await open(url('candidate'), {
  onError: (message) => console.error('[candidate]', message),
});
async function slice(page, batch, evals) {
  const frequency = watchFrequency();
  const ms = await page.evaluate(
    ([b, n]) => globalThis.runSlice(b, n), [batch, evals]);
  return { ms, mhz: frequency().median };
}

const results = [];
try {
  const baseline = await browser.newPage();
  baseline.on('pageerror', (error) => console.error('[baseline]', error.message));
  await baseline.goto(url(baselineSha));
  for (const page of [candidate, baseline]) {
    await page.waitForFunction(() => globalThis.perfReady === true, null, { timeout: 120000 });
  }

  for (const { batch, evals, pairs } of SWEEPS) {
    // Warm both plans outside the timing.
    await slice(candidate, batch, 2);
    await slice(baseline, batch, 2);
    for (let i = 0; i < pairs; i++) {
      // Order alternates so throttling drift lands on both sides equally.
      const candidateFirst = i % 2 === 0;
      const first = await slice(candidateFirst ? candidate : baseline, batch, evals);
      const second = await slice(candidateFirst ? baseline : candidate, batch, evals);
      const [c, b] = candidateFirst ? [first, second] : [second, first];
      // NaN medians mean the box has no frequency counter: keep the pair,
      // there is no band evidence against it.
      const kept = !(Math.abs(c.mhz - b.mhz) / Math.min(c.mhz, b.mhz) > 0.15);
      results.push({ batch, c, b, kept });
      console.error(`batch ${batch} pair ${i}: candidate ${c.ms.toFixed(0)} ms @ ${c.mhz} MHz, ` +
        `baseline ${b.ms.toFixed(0)} ms @ ${b.mhz} MHz` + (kept ? '' : ' (rejected: bands differ)'));
    }
  }
} finally {
  await close(browser, candidate);
}

// ---------------------------------------------------------------------------
// Judgment: paired t on log-ratios plus a sign test, per batch size

// Two-sided 97.5% points of Student's t for small df.
const T975 = [NaN, 12.71, 4.30, 3.18, 2.78, 2.57, 2.45, 2.36, 2.31, 2.26, 2.23, 2.20, 2.18];
const choose = (n, k) => (k < 0 || k > n ? 0 : Array.from({ length: k }, (_, i) => (n - i) / (i + 1))
  .reduce((a, b) => a * b, 1));
const signTest = (wins, n) => {
  const smaller = Math.min(wins, n - wins);
  const tail = Array.from({ length: smaller + 1 }, (_, i) => choose(n, i)).reduce((a, b) => a + b, 0);
  return n ? Math.min(1, 2 * tail / 2 ** n) : 1;
};

let failed = false;
for (const { batch } of SWEEPS) {
  const kept = results.filter((r) => r.batch === batch && r.kept);
  if (kept.length < 3) {
    console.log(`batch ${batch}: only ${kept.length} clean pairs, no verdict -- rerun on a cooler machine`);
    failed = true;
    continue;
  }
  // Throughput ratio candidate/baseline is baseline ms over candidate ms.
  const logs = kept.map((r) => Math.log(r.b.ms / r.c.ms));
  const perMhz = kept.map((r) => Math.log((r.b.ms * r.b.mhz) / (r.c.ms * r.c.mhz)));
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  const m = mean(logs);
  const sd = Math.sqrt(logs.reduce((a, x) => a + (x - m) ** 2, 0) / (logs.length - 1));
  const half = (T975[logs.length - 1] ?? 1.96) * sd / Math.sqrt(logs.length);
  const [lo, hi] = [Math.exp(m - half), Math.exp(m + half)];
  const geomean = Math.exp(m);
  const wins = logs.filter((x) => x > 0).length;
  const p = signTest(wins, logs.length);

  const verdict =
    geomean < 0.95 && hi < 1.0 ? 'FAIL' :
    geomean < 0.98 ? 'warn' : 'ok  ';
  if (verdict === 'FAIL') failed = true;
  console.log(`${verdict} batch ${batch}: candidate/baseline ${geomean.toFixed(3)} ` +
    `[${lo.toFixed(3)}, ${hi.toFixed(3)}] over ${kept.length} pairs ` +
    `(${Math.exp(mean(perMhz)).toFixed(3)} per MHz; sign test ${wins}/${logs.length}, p = ${p.toFixed(3)})`);
}
process.exit(failed ? 1 : 0);
