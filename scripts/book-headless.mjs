// Builds an opening book of positions that are actually close.
//
//   node scripts/book-headless.mjs --parents results/openings-11.jsonl \
//     --cache results/book-11.jsonl --out openings/balanced-11.json
//
// No first move on an 11x11 board is anywhere near even -- katahex rates the
// best of the 121 at 17% and most of the rest above 90% or below 10% -- so a
// pair of games from one carries no information: both sides win it from the
// same colour and the pair ties. A second move gives the granularity. From a
// first move that leaves black behind, white's replies run from the best (black
// stays behind) to the worst (black is winning), so one of them lands near even.
//
// Two passes. The first searches each parent and reads every reply's value off
// that one search. The second plays the most promising replies out into their
// own position and searches that, because a reply the parent's search barely
// visited has a value worth very little.

import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
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
const threads = Number(flag('threads', '16'));
const parentVisits = Number(flag('parent-visits', '1200'));
const checkVisits = Number(flag('check-visits', '800'));
const perParent = Number(flag('per-parent', '2'));
const keep = Number(flag('keep', '24'));
const band = Number(flag('band', '0.15'));
const wideRootNoise = Number(flag('wide', '1.0'));
const cachePath = flag('cache');
const out = flag('out');

// A parent has to be a first move black is losing, or white has no reply bad
// enough to bring the position back to even.
const parents = JSON.parse(`[${readFileSync(flag('parents'), 'utf8').trim().split('\n').join(',')}]`)
  .filter((row) => row.blackWinrate < 0.5)
  .sort((a, b) => b.blackWinrate - a.blackWinrate)
  .map((row) => row.move);
console.error(`${parents.length} parent first moves`);

mkdirSync(dirname(cachePath), { recursive: true });
const cache = new Map();
if (existsSync(cachePath)) {
  for (const line of readFileSync(cachePath, 'utf8').split('\n').filter(Boolean)) {
    const row = JSON.parse(line);
    cache.set(row.id, row);
  }
}

// The page pulls one job at a time, so the second pass can be decided from the
// first pass's answers without reloading anything.
const firstPass = parents
  .filter((move) => !cache.has(move))
  .map((move) => ({ id: move, moves: [move], visits: parentVisits, wideRootNoise }));
let secondPass = null;

const candidates = () => parents.flatMap((move) => {
  const parent = cache.get(move);
  if (!parent) return [];
  // White is to move, so a reply that leaves it at 50% leaves the game even.
  return parent.replies
    .filter((reply) => reply.visits >= 5)
    .sort((a, b) => Math.abs(a.winrate - 0.5) - Math.abs(b.winrate - 0.5))
    .slice(0, perParent)
    .map((reply) => ({ id: `${move}-${reply.move}`, moves: [move, reply.move], visits: checkVisits }));
});

const nextJob = () => {
  if (firstPass.length) return firstPass.shift();
  if (!secondPass) secondPass = candidates().filter((job) => !cache.has(job.id));
  return secondPass.shift() ?? null;
};

const browser = await chromium.launch({
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan'],
});
const page = await browser.newPage();
page.on('pageerror', (error) => console.error('[page]', error.message));
await page.exposeFunction('nextJob', nextJob);
await page.exposeFunction('note', (line) => void console.error(line));
await page.exposeFunction('report', (result) => {
  appendFileSync(cachePath, JSON.stringify(result) + '\n');
  cache.set(result.id, result);
});
await page.goto(`${BASE}/positions.html?size=${size}&threads=${threads}`);
await page.waitForFunction(
  () => /\n(done|ERROR:)/.test(document.getElementById('log').textContent),
  null, { timeout: 0 });
await browser.close();

// Black is to move in a two move position, so its winrate is reported directly.
const book = [...cache.values()]
  .filter((row) => row.moves.length === 2 && Math.abs(row.winrate - 0.5) <= band)
  .sort((a, b) => Math.abs(a.winrate - 0.5) - Math.abs(b.winrate - 0.5))
  .slice(0, keep);

mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, JSON.stringify(book.map((row) => row.moves), null, 1) + '\n');
console.error(`${out}: ${book.length} openings`);
for (const row of book) console.error(`  ${row.moves.join(' ')}  black ${(row.winrate * 100).toFixed(1)}%`);
