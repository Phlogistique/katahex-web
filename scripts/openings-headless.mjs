// Rates every first move with a search, and writes the ones closest to even as
// an opening set for the match.
//
//   node scripts/openings-headless.mjs --visits 1000 \
//     --sweep results/openings-11.jsonl --out openings/balanced-11.json --keep 20
//
// Needs a vite server on the katahex-web page (npm run dev). The sweep is
// appended a move at a time, so a rerun only rates what is missing.

import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { close, finished, open } from './browser.mjs';

const argv = process.argv.slice(2);
const flag = (name, fallback) => {
  const at = argv.indexOf(`--${name}`);
  if (at < 0 && fallback === undefined) throw new Error(`missing --${name}`);
  return at < 0 ? fallback : argv[at + 1];
};

const BASE = process.env.BASE ?? 'http://localhost:5173';
const size = Number(flag('size', '11'));
const threads = Number(flag('threads', '16'));
const batchWait = Number(flag('batchwait', '3000'));
const visits = Number(flag('visits', '1000'));
const keep = Number(flag('keep', '20'));
const sweepPath = flag('sweep');
const out = flag('out');

const moveName = (index) => String.fromCharCode(97 + (index % size)) + (Math.floor(index / size) + 1);

// A 180 degree rotation of a hex board is the same game, so a first move and its
// opposite are one opening. `--moves` rates only the ones named, which is enough
// to see the shape of the distribution on a board too big to sweep whole.
const named = flag('moves', '');
const distinct = [];
if (named) {
  distinct.push(...named.split(','));
} else {
  const seen = new Set();
  for (let i = 0; i < size * size; i++) {
    if (seen.has(i)) continue;
    seen.add(size * size - 1 - i);
    distinct.push(moveName(i));
  }
}

mkdirSync(dirname(sweepPath), { recursive: true });
const rated = existsSync(sweepPath)
  ? readFileSync(sweepPath, 'utf8').split('\n').filter(Boolean).map((line) => JSON.parse(line))
  : [];
const queue = distinct.filter((move) => !rated.some((row) => row.move === move))
  .map((move) => ({ move, visits }));
console.error(`${queue.length} first moves to rate, ${rated.length} already done`);

if (queue.length) {
  const { browser, page } = await open(`${BASE}/positions.html?size=${size}&threads=${threads}&batchwait=${batchWait}`, {
    onError: (message) => console.error('[page]', message),
    expose: {
      nextJob: () => {
        const job = queue.shift();
        return job ? { id: job.move, moves: [job.move], visits: job.visits } : null;
      },
      note: (line) => void console.error(line),
      report: (position) => {
        // The side to move after black's first move is white, so black's share
        // is what is left over.
        const row = { move: position.id, blackWinrate: 1 - position.winrate, visits: position.visits };
        appendFileSync(sweepPath, JSON.stringify(row) + '\n');
        rated.push(row);
      },
    },
  });

  await finished(page);
  await close(browser, page);
}

const balanced = [...rated]
  .sort((a, b) => Math.abs(a.blackWinrate - 0.5) - Math.abs(b.blackWinrate - 0.5))
  .slice(0, keep);
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, JSON.stringify(balanced.map((row) => [row.move]), null, 1) + '\n');
console.error(`${out}: ${balanced.map((r) => `${r.move} ${(r.blackWinrate * 100).toFixed(0)}%`).join(', ')}`);
