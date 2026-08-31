// Scores a match recorded by scripts/match-headless.mjs.
//
//   node scripts/score.mjs results/time-1s.jsonl
//
// Games are counted in pairs -- the same opening played from both sides -- so
// what is measured is the two sides rather than the opening. A pair that both
// sides won as black says nothing and shows up as a tie.

import { readFileSync } from 'node:fs';

const games = readFileSync(process.argv[2], 'utf8').split('\n').filter(Boolean).map(JSON.parse);
const name = (side) => `${side.precision}@${
  side.condition.kind === 'policy' ? 'policy'
  : side.condition.kind === 'visits' ? `${side.condition.visits}v`
  : `${side.condition.seconds}s`}`;

// The id ends in which side had black, so the two games of a pair are known
// apart without having to compare the side descriptions.
const pairs = new Map();
for (const game of games) {
  const [opening, repeat, which] = game.id.split('/');
  const key = `${opening}/${repeat}`;
  const pair = pairs.get(key) ?? {};
  pair[which] = which === 'a' ? game.winner === 'B' : game.winner === 'W';
  pairs.set(key, pair);
}

const complete = [...pairs.values()].filter((pair) => pair.a !== undefined && pair.b !== undefined);
const differentials = complete.map((pair) => Number(pair.a) + Number(pair.b) - 1);
const n = differentials.length;
const mean = differentials.reduce((a, b) => a + b, 0) / n;
const variance = differentials.reduce((a, d) => a + (d - mean) ** 2, 0) / Math.max(1, n - 1);
const stderr = Math.sqrt(variance / n);
const score = 0.5 + mean / 2;

// A bound at 0 or 100% is unbounded elo; say so rather than printing NaN.
const elo = (p) => (p <= 0 ? '-inf' : p >= 1 ? '+inf' : (-400 * Math.log10(1 / p - 1)).toFixed(0));

const first = games[0];
const [a, b] = first.id.endsWith('/a') ? [first.black, first.white] : [first.white, first.black];
console.log(`${name(a)} against ${name(b)}`);
console.log(`${games.length} games, ${n} complete pairs, ` +
  `${differentials.filter((d) => d !== 0).length} decisive`);
console.log(`${name(a)} pair score ${(score * 100).toFixed(1)}% +- ${(stderr * 50).toFixed(1)} (1 sigma)`);
console.log(`${name(a)} elo ${elo(score)} ` +
  `[${elo(score - 1.96 * stderr / 2)}, ${elo(score + 1.96 * stderr / 2)}] (95%)`);

// The normal approximation is generous when few pairs are decisive, and most
// pairs are ties by construction: both sides win the same opening from the same
// colour. A sign test over the decisive pairs is the honest one.
const decisive = differentials.filter((d) => d !== 0);
const wins = decisive.filter((d) => d > 0).length;
const choose = (n, k) => (k < 0 || k > n ? 0 : Array.from({ length: k }, (_, i) => (n - i) / (i + 1))
  .reduce((a, b) => a * b, 1));
const tail = (n, k) => Array.from({ length: k + 1 }, (_, i) => choose(n, i)).reduce((a, b) => a + b, 0);
const smaller = Math.min(wins, decisive.length - wins);
const p = decisive.length ? Math.min(1, 2 * tail(decisive.length, smaller) / 2 ** decisive.length) : 1;
console.log(`sign test: ${wins} of ${decisive.length} decisive pairs to ${name(a)}, p = ${p.toFixed(3)}`);

// Speed per side, which is the ratio the comparison turns on, measured under
// the match's own conditions rather than on a bench.
const speed = new Map();
for (const game of games) {
  const aIsBlack = game.id.endsWith('/a');
  game.turns.forEach((turn, ply) => {
    const black = (game.opening.length + ply) % 2 === 0;
    // By seat, not by name: in a control both seats have the same name, and the
    // whole point is to tell the two engine instances apart.
    const seat = black === aIsBlack ? 'a' : 'b';
    const key = `${seat} ${name(black ? game.black : game.white)}`;
    const total = speed.get(key) ?? { visits: 0, rows: 0, ms: 0, moves: 0 };
    total.visits += turn.visits;
    total.rows += turn.rows ?? 0;
    total.ms += turn.ms;
    total.moves += 1;
    speed.set(key, total);
  });
}
// Evaluations, not visits: a move can spend thousands of visits walking a
// subtree the nnCache already holds, and those never reach the GPU at all.
for (const [key, { visits, rows, ms, moves }] of speed) {
  console.log(`${key}: ${(1000 * rows / ms).toFixed(1)} evals/s, ` +
    `${Math.round(rows / moves)} evals/move, ${Math.round(visits / moves)} visits/move ` +
    `(${(visits / Math.max(1, rows)).toFixed(2)} visits/eval) over ${moves} moves`);
}

// When one side just searches more than the other, what the extra search is
// worth per doubling. Off the visits actually reached, not the cap asked for:
// a search overshoots maxVisits by up to threads x leaf evals, because the
// evaluations already in flight drain before it stops.
const [sideA, sideB] = [...speed.values()];
if (sideA && sideB && typeof score === 'number') {
  const doublings = Math.log2((sideA.visits / sideA.moves) / (sideB.visits / sideB.moves));
  if (Math.abs(doublings) > 0.2 && score > 0 && score < 1) {
    const point = -400 * Math.log10(1 / score - 1);
    console.log(`${doublings.toFixed(2)} doublings of search, ` +
      `so ${(point / doublings).toFixed(0)} elo per doubling ` +
      `(${elo(score - 1.96 * stderr / 2)} to ${elo(score + 1.96 * stderr / 2)} elo, ` +
      `divided by ${doublings.toFixed(2)})`);
  }
}
