// Runs both precisions over the same positions and records where they differ.
//
//   node scripts/agree-headless.mjs --condition visits:400 --games results/time-1s.jsonl \
//     --every 3 --out results/agree-400.jsonl
//
// Positions come from games already played, sampled every `--every` plies, which
// is what makes them positions the engines actually reach.

import { appendFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { chromium } from 'playwright';

const argv = process.argv.slice(2);
const flag = (name, fallback) => {
  const at = argv.indexOf(`--${name}`);
  if (at < 0 && fallback === undefined) throw new Error(`missing --${name}`);
  return at < 0 ? fallback : argv[at + 1];
};

const BASE = process.env.BASE ?? 'http://localhost:5173';
const size = Number(flag('size', '11'));
const every = Number(flag('every', '3'));
const out = flag('out');

const condition = (() => {
  const [kind, value] = flag('condition').split(':');
  if (kind === 'visits') return { kind, visits: Number(value) };
  if (kind === 'time') return { kind, seconds: Number(value) };
  throw new Error(`unknown condition: ${kind}`);
})();

const jobs = [];
for (const path of flag('games').split(',')) {
  for (const line of readFileSync(path, 'utf8').split('\n').filter(Boolean)) {
    const game = JSON.parse(line);
    const moves = [...game.opening];
    game.turns.forEach((turn, ply) => {
      // Sampled rather than every ply: neighbouring positions are nearly the
      // same question, and each one costs two searches.
      if (ply % every === 0 && moves.length > 0) {
        jobs.push({ id: `${game.id}@${moves.length}`, moves: [...moves], condition });
      }
      moves.push(turn.move);
    });
  }
}

mkdirSync(dirname(out), { recursive: true });
const done = new Set(
  existsSync(out)
    ? readFileSync(out, 'utf8').split('\n').filter(Boolean).map((line) => JSON.parse(line).id)
    : []);
const queue = jobs.filter((job) => !done.has(job.id));
console.error(`${queue.length} positions to ask, ${done.size} already recorded`);
if (!queue.length) process.exit(0);

const browser = await chromium.launch({
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan'],
});
const page = await browser.newPage();
page.on('pageerror', (error) => console.error('[page]', error.message));
await page.exposeFunction('nextJob', () => queue.shift() ?? null);
await page.exposeFunction('note', (line) => void console.error(line));
await page.exposeFunction('report', (result) => appendFileSync(out, JSON.stringify(result) + '\n'));
await page.goto(`${BASE}/agree.html?size=${size}`);
await page.waitForFunction(
  () => /\n(done|ERROR:)/.test(document.getElementById('log').textContent),
  null, { timeout: 0 });
await browser.close();
