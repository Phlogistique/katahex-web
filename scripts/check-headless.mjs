// Runs the check page (tiers 1 and 2) and judges what it reports.
//
//   node scripts/check-headless.mjs             # gate: exit 1 on any failure
//   node scripts/check-headless.mjs --calibrate # freeze this run's tier-2 numbers
//
// Needs a vite server already serving the page (npm run dev), or set BASE.
//
// Tier 1 is judged against fixed tolerances: the hand fp32 backend agrees with
// the TensorFlow.js goldens to ~1e-4, real bugs land orders of magnitude
// higher, and nothing plausible lives in between. Tier 2 is judged against
// constants frozen at calibration -- never against the previous run, so drift
// cannot ratchet.

import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import { close, finished, open } from './browser.mjs';

const BASE = process.env.BASE ?? 'http://localhost:5173';
const FROZEN = new URL('../public/check/frozen-11.json', import.meta.url);
const calibrate = process.argv.includes('--calibrate');

const TIER1_TOLERANCE = 1e-3;

let report = null;
const { browser, page } = await open(`${BASE}/check.html`, {
  onError: (message) => console.error('[page]', message),
  expose: {
    note: (line) => void console.error(line),
    report: (r) => { report = r; },
  },
});
await finished(page);
await close(browser, page);
if (!report) {
  console.error('FAIL: page finished without reporting');
  process.exit(1);
}

let failed = false;
const judge = (ok, line) => {
  if (!ok) failed = true;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${line}`);
};

// -- tier 1 --
// Pin the check counts: a page that quietly ran fewer checks must not pass.
const manifest = JSON.parse(readFileSync(new URL('../public/check/tier1.json', import.meta.url), 'utf8'));
const expected = manifest.positions.length + 2; // + batch-invariance + batch-48
judge(report.tier1.length === expected, `tier1 ran ${report.tier1.length} checks (expected ${expected})`);
for (const { id, worst, finite } of report.tier1) {
  const worstOfAll = Math.max(...Object.values(worst));
  judge(finite && worstOfAll <= TIER1_TOLERANCE,
    `tier1 ${id}: ${worstOfAll.toExponential(2)}${finite ? '' : ' (non-finite output)'}`);
}

// -- tier 2 --
const m = report.tier2;
if (calibrate) {
  const sha = execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
  writeFileSync(FROZEN, JSON.stringify({ sha, date: new Date().toISOString().slice(0, 10), ...m }, null, 1));
  console.log(`calibrated: froze tier-2 constants to public/check/frozen-11.json`);
  console.log(JSON.stringify(m));
} else {
  let frozen;
  try { frozen = JSON.parse(readFileSync(FROZEN, 'utf8')); } catch {
    console.error('FAIL: no frozen constants; run with --calibrate first');
    process.exit(1);
  }
  // Absolute thresholds where the null is known (mean 0.05 -> 1.3x; max 0.6
  // observed -> 2x); relative to the frozen calibration elsewhere. SE of the
  // mean over 512 positions is ~0.0022, so the null sits ~7 SE under its
  // threshold and a doubling of error ~16 SE above.
  judge(m.positions === frozen.positions,
    `tier2 bank has ${m.positions} positions (frozen at ${frozen.positions})`);
  judge(m.meanAbsErr <= 0.065, `tier2 mean |logit err| ${m.meanAbsErr.toFixed(4)} (limit 0.065)`);
  judge(m.p95 <= 1.5 * frozen.p95, `tier2 p95 ${m.p95.toFixed(4)} (limit ${(1.5 * frozen.p95).toFixed(4)})`);
  judge(m.max <= 1.2, `tier2 max ${m.max.toFixed(3)} (limit 1.2)`);
  judge(m.worstCellBias <= 0.02, `tier2 worst cell bias ${m.worstCellBias.toFixed(4)} (limit 0.02)`);
  judge(m.top1Flips <= Math.max(2, 1.75 * frozen.top1Flips),
    `tier2 top-1 flips ${m.top1Flips} (limit ${Math.max(2, 1.75 * frozen.top1Flips).toFixed(1)})`);
  judge(m.meanKL <= 2.5 * frozen.meanKL,
    `tier2 mean KL ${m.meanKL.toExponential(2)} (limit ${(2.5 * frozen.meanKL).toExponential(2)})`);
  judge(m.meanAbsDWinrate <= 1.5 * frozen.meanAbsDWinrate,
    `tier2 mean |dWinrate| ${m.meanAbsDWinrate.toExponential(2)} (limit ${(1.5 * frozen.meanAbsDWinrate).toExponential(2)})`);
}

process.exit(failed ? 1 : 0);
