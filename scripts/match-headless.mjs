// Runs the fp16-against-fp32 match in headless Chrome and records the games.
//
//   node scripts/match-headless.mjs --a fp16:time:1 --b fp32:time:1 --repeats 2 \
//     --openings openings/balanced-11.json --out results/time-1s.jsonl
//
// A side is <precision>:<condition>, where the condition is policy, visits:N or
// time:S. Naming both sides is what lets the same harness answer more than one
// question: fp16 against fp32 at the same time budget is the comparison, at the
// same visit count is the precision penalty on its own, and fp32 against itself
// at N and 2N visits is what a doubling of search is worth, which is how a
// speed ratio becomes elo.
//
// Every opening is played twice per repeat, once with each side as black, and
// scored as a pair. That cancels the first-player advantage without needing a
// swap rule.
//
// Needs a vite server on the katahex-web page (npm run dev) and the symlinks in
// public/ for katahex.js, katahex.wasm and the net. Results are appended one
// game per line and a rerun skips the games already there, so a run can be
// stopped and picked up.

import { appendFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { close, finished, open } from './browser.mjs';

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
const threads = Number(flag('threads', '1'));
const batchWait = Number(flag('batchwait', '3000'));
const leaves = Number(flag('leaves', '64'));
const repeats = Number(flag('repeats', '1'));
const out = flag('out');
const openings = JSON.parse(readFileSync(flag('openings'), 'utf8'));

const side = (spec) => {
  const [precision, kind, value] = spec.split(':');
  if (precision !== 'fp16' && precision !== 'fp32') throw new Error(`unknown precision: ${precision}`);
  if (kind === 'policy') return { precision, condition: { kind } };
  if (kind === 'visits') return { precision, condition: { kind, visits: Number(value) } };
  if (kind === 'time') return { precision, condition: { kind, seconds: Number(value) } };
  throw new Error(`unknown condition: ${kind}`);
};

const a = side(flag('a'));
const b = side(flag('b'));

// Playing each opening from both sides is what makes a pair scorable: hex is a
// first-player win, so only the games where one precision converts an opening
// the other does not carry any information.
const jobs = [];
for (let repeat = 0; repeat < repeats; repeat++) {
  for (const opening of openings) {
    jobs.push({ id: `${opening.join('-')}/${repeat}/a`, opening, black: a, white: b });
    jobs.push({ id: `${opening.join('-')}/${repeat}/b`, opening, black: b, white: a });
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

let played = 0;
const { browser, page } = await open(`${BASE}/match.html?size=${size}&threads=${threads}&batchwait=${batchWait}&leaves=${leaves}`, {
  onError: (message) => console.error('[page]', message),
  expose: {
    nextJob: () => queue.shift() ?? null,
    note: (line) => void console.error(line),
    report: (result) => {
      appendFileSync(out, JSON.stringify(result) + '\n');
      played++;
    },
  },
});

await finished(page);
console.error(`${played} games written to ${out}`);
await close(browser, page);
