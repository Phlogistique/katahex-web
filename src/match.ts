// Plays the net against itself at two precisions, to find out what half
// precision is worth over a game rather than over one evaluation.
//
// Both engines live in this page, one worker each, and only one of them is
// searching at any moment, so they see the same GPU. The page takes its work
// from the driver (scripts/match-headless.mjs) one game at a time and hands
// back the result, which is what makes a run resumable.

import { AnalysisEngine, type Condition, type Precision } from './analysisEngine';
import { bridge } from './harness';
import { connected, moveName, other, stones, type Player } from './hex';

/** A side of the match: which net, and how much thinking each move gets. */
export type Side = { precision: Precision; condition: Condition };

export type Job = { id: string; opening: string[]; black: Side; white: Side };

type Turn = { move: string; by: Precision; visits: number; winrate: number; ms: number };

export type Result = {
  id: string;
  opening: string[];
  black: Side;
  white: Side;
  winner: Player;
  /** Both set the batch size, and so what half precision is worth. */
  threads: number;
  batchWaitMicros: number;
  reason: 'connection' | 'resign' | 'no-moves';
  turns: Turn[];
  seconds: number;
};

const params = new URLSearchParams(location.search);
const SIZE = Number(params.get('size') ?? 11);
const THREADS = Number(params.get('threads') ?? 16);
const BATCH_WAIT = Number(params.get('batchwait') ?? 3000);

/** A side that thinks it is this far behind, for this many of its own turns, gives up. */
const RESIGN_WINRATE = 0.02;
const RESIGN_TURNS = 3;

const { driver, log } = bridge<Job, Result>();

const describe = (side: Side) => `${side.precision}@${
  side.condition.kind === 'policy' ? 'policy'
  : side.condition.kind === 'visits' ? `${side.condition.visits}v`
  : `${side.condition.seconds}s`}`;

/** The move the raw net would play: its best policy over the empty cells. */
function policyMove(policy: number[], moves: string[], size: number): string {
  const board = stones(moves, size);
  let best = -1;
  let bestAt = -1;
  for (let i = 0; i < size * size; i++) {
    if (board[i] === null && policy[i] > best) { best = policy[i]; bestAt = i; }
  }
  if (bestAt < 0) throw new Error('no legal move in the policy');
  return moveName(bestAt, size);
}

async function playGame(job: Job, engines: Record<Precision, AnalysisEngine>): Promise<Result> {
  const sides: Record<Player, Side> = { B: job.black, W: job.white };

  const moves = [...job.opening];
  const turns: Turn[] = [];
  const losing: Record<Player, number> = { B: 0, W: 0 };
  const startedAt = performance.now();

  for (;;) {
    const player: Player = moves.length % 2 ? 'W' : 'B';
    const { precision, condition } = sides[player];
    const at = performance.now();
    const reply = await engines[precision].analyse(moves, condition);
    const ms = performance.now() - at;

    if (reply.error) throw new Error(`${precision}: ${reply.error}`);

    // The engine ends a game on a bridge to the edge, not only on a solid
    // chain, so its own silence is the authority on when a game is over. Only
    // the side that just moved can have completed a connection.
    const chosen = condition.kind === 'policy'
      ? (reply.policy ? policyMove(reply.policy, moves, SIZE) : null)
      : (reply.moveInfos?.length
          ? reply.moveInfos.reduce((a, b) => (a.order <= b.order ? a : b)).move
          : null);
    if (!chosen) {
      return {
        ...job, winner: other(player), threads: THREADS, batchWaitMicros: BATCH_WAIT, reason: 'no-moves',
        turns, seconds: (performance.now() - startedAt) / 1000,
      };
    }

    const winrate = reply.rootInfo?.winrate ?? 0.5;
    losing[player] = winrate < RESIGN_WINRATE ? losing[player] + 1 : 0;
    if (losing[player] >= RESIGN_TURNS) {
      return {
        ...job, winner: other(player), threads: THREADS, batchWaitMicros: BATCH_WAIT, reason: 'resign',
        turns, seconds: (performance.now() - startedAt) / 1000,
      };
    }

    moves.push(chosen);
    turns.push({ move: chosen, by: precision, visits: reply.rootInfo?.visits ?? 0, winrate, ms });

    if (connected(stones(moves, SIZE), SIZE, player)) {
      return {
        ...job, winner: player, threads: THREADS, batchWaitMicros: BATCH_WAIT, reason: 'connection',
        turns, seconds: (performance.now() - startedAt) / 1000,
      };
    }
    if (moves.length >= SIZE * SIZE) throw new Error('board full with no winner');
  }
}

async function main() {
  const engines: Record<Precision, AnalysisEngine> = {
    fp16: new AnalysisEngine('fp16', SIZE, log, THREADS, BATCH_WAIT),
    fp32: new AnalysisEngine('fp32', SIZE, log, THREADS, BATCH_WAIT),
  };
  log(`loading both engines at ${SIZE}x${SIZE}`);
  (globalThis as { stopEngines?: () => void }).stopEngines =
    () => { engines.fp16.stop(); engines.fp32.stop(); };
  await Promise.all([engines.fp16.ready(), engines.fp32.ready()]);
  log('both engines ready');

  for (;;) {
    const job = await driver.nextJob();
    if (!job) break;
    const result = await playGame(job, engines);
    const visits = Math.round(
      result.turns.reduce((n, t) => n + t.visits, 0) / Math.max(1, result.turns.length));
    const winner = result.winner === 'B' ? result.black : result.white;
    log(`${job.id}: ${describe(winner)} wins as ${result.winner} ` +
        `in ${result.turns.length} (${result.reason}, ${result.seconds.toFixed(0)}s, ${visits} visits/move)`);
    await driver.report(result);
  }
  log('done');
}

main().catch((error) => log(`ERROR: ${error?.stack ?? error}`));
