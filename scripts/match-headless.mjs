// Runs the fp16-against-fp32 match in headless Chrome and records the games.
//
//   node scripts/match-headless.mjs --condition time:1 --repeats 2 \
//     --openings openings/balanced-11.json --out results/time-1s.jsonl
//
// Conditions: policy | visits:N | time:S. Every opening is played twice per
// repeat, once with each precision as black, so the pair cancels the
// first-player advantage without needing a swap rule.
//
// Needs a vite server on the katahex-web page (npm run dev) and the symlinks in
// public/ for katahex.js, katahex.wasm and the net. Results are appended one
// game per line and a rerun skips the games already there, so a run can be
// stopped and picked up.

import { appendFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { chromium } from 'playwright';

const argv = process.argv.slice(2);
const flag = (name, fallback) => {
  const at = argv.indexOf(`--${name}`);
  if (at < 0) {
    if (fallback === undefined) throw new Error(`missing --${name}`);
    return fallback;
  }
  return argv[at + 1];
};

const BASE = process.env.BASE ?? 'http://localhost:5173';
const size = Number(flag('size', '11'));
const repeats = Number(flag('repeats', '1'));
const out = flag('out');
const openings = JSON.parse(readFileSync(flag('openings'), 'utf8'));

const condition = (() => {
  const [kind, value] = flag('condition').split(':');
  if (kind === 'policy') return { kind };
  if (kind === 'visits') return { kind, visits: Number(value) };
  if (kind === 'time') return { kind, seconds: Number(value) };
  throw new Error(`unknown condition: ${kind}`);
})();

// Playing each opening from both sides is what makes a pair scorable: hex is a
// first-player win, so only the games where one precision converts an opening
// the other does not carry any information.
const jobs = [];
for (let repeat = 0; repeat < repeats; repeat++) {
  for (const opening of openings) {
    for (const black of ['fp16', 'fp32']) {
      jobs.push({ id: `${opening.join('-')}/${black}/${repeat}`, opening, black, condition });
    }
  }
}

mkdirSync(dirname(out), { recursive: true });
const done = new Set(
  existsSync(out)
    ? readFileSync(out, 'utf8').split('\n').filter(Boolean).map((line) => JSON.parse(line).id)
    : []);
const queue = jobs.filter((job) => !done.has(job.id));
console.error(`${queue.length} games to play, ${done.size} already recorded`);
if (!queue.length) process.exit(0);

const browser = await chromium.launch({
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan'],
});
const page = await browser.newPage();
page.on('pageerror', (error) => console.error('[page]', error.message));

let played = 0;
await page.exposeFunction('nextJob', () => queue.shift() ?? null);
await page.exposeFunction('note', (line) => void console.error(line));
await page.exposeFunction('report', (result) => {
  appendFileSync(out, JSON.stringify(result) + '\n');
  played++;
});

await page.goto(`${BASE}/match.html?size=${size}`);
await page.waitForFunction(
  () => /\n(done|ERROR:)/.test(document.getElementById('log').textContent),
  null, { timeout: 0 });

console.error(`${played} games written to ${out}`);
await browser.close();
