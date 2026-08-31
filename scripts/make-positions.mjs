// Builds the check page's assets: stored feature tensors for positions the
// engine actually reaches, plus TensorFlow.js fp32 golden outputs for the
// tier-1 positions. One-time (per net); the results are checked in.
//
//   npm run build:net-worker && node scripts/make-positions.mjs
//
// Drives the node engine (TensorFlow.js CPU, the implementation that agrees
// with native to 1e-6) at maxVisits=1 over each position, with NET_CAPTURE
// recording the (spatial, global) tensors and the outputs at the serveEvals
// boundary -- the one place the inputs exist as raw tensors. All queries go in
// one engine run (the wasm engine's replies only flush at stdin EOF, so
// nothing can be interleaved); capture lines are matched to queries in order,
// with each reply's position hash telling apart a query the nnCache absorbed
// (an earlier query hashed the same, so no capture line appeared).

import { appendFileSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createInterface } from 'node:readline';
import { gzipSync } from 'node:zlib';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const OUT = resolve(root, 'public/check');
const CAPTURE = resolve(root, 'build/capture.jsonl');
const BANK_SIZE = Number(process.env.BANK_SIZE ?? 512);

// ---------------------------------------------------------------------------
// Positions

/** Deterministic shuffle, so reruns sample the same bank. */
const shuffled = (items) => {
  let seed = 0x2545f491;
  const rand = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x80000000);
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
};

/** Every game as its full move list, from a match results file. */
const games = [];
for (const file of ['arm-e-slope', 'control-fp16-fp16', 'arm-b-time1', 'arm-b-swapped']) {
  for (const line of readFileSync(resolve(root, `results/${file}.jsonl`), 'utf8').split('\n').filter(Boolean)) {
    const game = JSON.parse(line);
    games.push({ id: `${file}:${game.id}`, moves: [...game.opening, ...game.turns.map((t) => t.move)] });
  }
}

/** Positions sampled every 3 plies, deduplicated by move list. */
const seen = new Set();
const sampled = [];
for (const game of games) {
  for (let ply = 2; ply < game.moves.length; ply += 3) {
    const moves = game.moves.slice(0, ply);
    const key = moves.join(',');
    if (seen.has(key)) continue;
    seen.add(key);
    sampled.push({ id: `${game.id}@${ply}`, size: 11, moves });
  }
}
console.error(`${sampled.length} distinct sampled positions`);
const bankPool = shuffled(sampled);

// Tier-1 positions: chosen to hit the numerically distinct paths -- padding
// tiles (rows/cols 9-11 are the third Winograd tile row at 11x11), saturated
// value, both sides to move on one position, and real positions at a spread of
// plies. `stones`/`player` use the analysis query's initialStones form.
const colFor = (i, size) => 'abcdefghijklm'[i % size];
const stripe = (cells) => cells.map((cell, i) => [i % 2 ? 'W' : 'B', cell]);
const rows911 = [];
for (let y = 9; y <= 11; y++) for (let x = 0; x < 11; x++) rows911.push(`${colFor(x, 11)}${y}`);
const colsIK = [];
for (let x = 8; x <= 10; x++) for (let y = 1; y <= 11; y++) colsIK.push(`${colFor(x, 11)}${y}`);

const byLength = [...games].sort((a, b) => b.moves.length - a.moves.length);
const longest = byLength[0];
const atPly = (which, ply) => byLength[which].moves.slice(0, Math.min(ply, byLength[which].moves.length - 1));
const midgame = atPly(3, 20);
const midgameStones = midgame.map((move, i) => [i % 2 ? 'W' : 'B', move]);
const swapColor = (stones) => stones.map(([color, move]) => [color === 'B' ? 'W' : 'B', move]);
const samePos = atPly(5, 12).map((move, i) => [i % 2 ? 'W' : 'B', move]);

const tier1 = [
  { id: 'empty-b', size: 11, moves: [] },
  { id: 'empty-w', size: 11, moves: [], player: 'W' },
  { id: 'corner-a1', size: 11, stones: [['B', 'a1']], player: 'W' },
  { id: 'corner-k11', size: 11, stones: [['W', 'k11']], player: 'B' },
  { id: 'rows-9-11', size: 11, stones: stripe(rows911), player: 'B' },
  { id: 'cols-i-k', size: 11, stones: stripe(colsIK), player: 'W' },
  { id: 'midgame', size: 11, stones: midgameStones, player: 'B' },
  { id: 'midgame-swapped', size: 11, stones: swapColor(midgameStones), player: 'W' },
  { id: 'ply-5', size: 11, moves: atPly(4, 5) },
  { id: 'ply-15', size: 11, moves: atPly(6, 15) },
  { id: 'ply-25', size: 11, moves: atPly(7, 25) },
  { id: 'ply-40', size: 11, moves: atPly(2, 40) },
  { id: 'near-full', size: 11, moves: longest.moves.slice(0, -1) },
  { id: 'won', size: 11, moves: byLength[1].moves.slice(0, -1) },
  { id: 'same-pos-b', size: 11, stones: samePos, player: 'B' },
  { id: 'same-pos-w', size: 11, stones: samePos, player: 'W' },
  { id: '13-five-stones', size: 13, moves: ['e2', 'f5', 'h8', 'g7', 'c10'] },
  { id: '13-edge-rows', size: 13, stones: stripe(['b12', 'f12', 'k12', 'd13', 'h13', 'm13', 'g3', 'g8']), player: 'W' },
];

// ---------------------------------------------------------------------------
// Driving the engine

const query = (position) => JSON.stringify({
  id: position.id,
  boardXSize: position.size,
  boardYSize: position.size,
  moves: (position.moves ?? []).map((move, turn) => [turn % 2 ? 'W' : 'B', move]),
  ...(position.stones ? { initialStones: position.stones } : {}),
  ...(position.player ? { initialPlayer: position.player } : {}),
  rules: 'tromp-taylor',
  komi: 0,
  maxVisits: 1,
  analyzeTurns: [(position.moves ?? []).length],
});

const captureLines = () => {
  try { return readFileSync(CAPTURE, 'utf8').split('\n').filter(Boolean); } catch { return []; }
};

/**
 * Runs one engine (one board size) over all of `positions`, and returns one
 * capture record per position that actually reached the net -- null where the
 * nnCache absorbed it (a position an earlier query already hashed to).
 */
async function capture(size, positions) {
  const before = captureLines().length;
  const child = spawn('node', [resolve(here, 'node-engine-test.mjs')], {
    env: { ...process.env, BOARD_SIZE: String(size), NET_CAPTURE: CAPTURE,
           EXTRA_OVERRIDE: 'nnRandomize=false' },
    stdio: ['pipe', 'pipe', 'inherit'],
  });
  const replies = [];
  createInterface({ input: child.stdout }).on('line', (line) => {
    if (line.startsWith('{')) replies.push(JSON.parse(line));
  });
  for (const position of positions) child.stdin.write(query(position) + '\n');
  child.stdin.end();
  await new Promise((exited) => child.on('exit', exited));

  if (replies.length !== positions.length) {
    throw new Error(`${replies.length} replies to ${positions.length} queries`);
  }
  const lines = captureLines().slice(before);
  const seen = new Set();
  const records = replies.map((reply) => {
    if (seen.has(reply.rootInfo.thisHash)) return null;
    seen.add(reply.rootInfo.thisHash);
    return JSON.parse(lines[seen.size - 1]);
  });
  if (lines.length !== seen.size) {
    throw new Error(`${lines.length} captures for ${seen.size} distinct positions; ` +
      'the nnCache and the position hash disagree, so ids cannot be trusted');
  }
  return records;
}

// ---------------------------------------------------------------------------

mkdirSync(OUT, { recursive: true });
rmSync(CAPTURE, { force: true });

// Tier 1: features plus goldens. Every position must land.
const tier1by11 = tier1.filter((p) => p.size === 11);
const tier1by13 = tier1.filter((p) => p.size === 13);
const got11 = await capture(11, tier1by11);
const got13 = await capture(13, tier1by13);
const tier1Records = [...got11, ...got13];
const tier1Positions = [...tier1by11, ...tier1by13];
tier1Records.forEach((record, i) => {
  if (!record || record.batch !== 1) throw new Error(`tier-1 ${tier1Positions[i].id} not captured`);
});

const features = [];
const manifest = tier1Positions.map((position, i) => {
  const record = tier1Records[i];
  features.push(Float32Array.from(record.spatial), Float32Array.from(record.global));
  return {
    id: position.id, size: position.size,
    golden: { policy: record.policy[0], policyPass: [record.policyPass[0]],
              value: record.value[0], scoreValue: record.scoreValue[0] },
  };
});
const concat = (arrays) => {
  const out = new Float32Array(arrays.reduce((n, a) => n + a.length, 0));
  let at = 0;
  for (const a of arrays) { out.set(a, at); at += a.length; }
  return out;
};
writeFileSync(resolve(OUT, 'tier1.json'), JSON.stringify({
  model: 'hex27x3.bin.gz', date: new Date().toISOString().slice(0, 10),
  reference: 'tensorflow.js fp32 cpu, nnRandomize=false', positions: manifest,
}, null, 1));
writeFileSync(resolve(OUT, 'tier1-features.bin.gz'), gzipSync(Buffer.from(concat(features).buffer)));

// Tier 2: the bank. Cache hits (transpositions across games) are replaced from
// the surplus of the pool.
rmSync(CAPTURE, { force: true });
const bankIds = [];
const bankFeatures = [];
const pool = [...bankPool];
while (bankIds.length < BANK_SIZE) {
  if (!pool.length) throw new Error('sampled pool exhausted');
  const batch = pool.splice(0, BANK_SIZE - bankIds.length);
  const records = await capture(11, batch);
  records.forEach((record, i) => {
    if (!record) return;
    bankIds.push(batch[i].id);
    bankFeatures.push(Float32Array.from(record.spatial), Float32Array.from(record.global));
  });
  console.error(`bank: ${bankIds.length}/${BANK_SIZE}`);
}
writeFileSync(resolve(OUT, 'bank-11.bin.gz'), gzipSync(Buffer.from(concat(bankFeatures).buffer)));
writeFileSync(resolve(OUT, 'bank-11.json'), JSON.stringify({ count: bankIds.length, ids: bankIds }, null, 1));
console.error('done');
