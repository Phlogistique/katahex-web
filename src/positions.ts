// Evaluates whatever positions the driver asks about, with the moves available
// in each. Building an opening book needs both: the value of a position, and
// the value of every reply to it.

import { AnalysisEngine } from './analysisEngine';
import { bridge } from './harness';

export type Job = { id: string; moves: string[]; visits: number; wideRootNoise?: number };
export type Reply = { move: string; winrate: number; visits: number };
export type Result = {
  id: string;
  moves: string[];
  /** For the side to move, which is what `reportAnalysisWinratesAs = SIDETOMOVE` gives. */
  winrate: number;
  visits: number;
  replies: Reply[];
};

const params = new URLSearchParams(location.search);
const SIZE = Number(params.get('size') ?? 11);
const THREADS = Number(params.get('threads') ?? 16);
const { driver, log } = bridge<Job, Result>();

async function main() {
  const engine = new AnalysisEngine('fp32', SIZE, log, THREADS);
  (globalThis as { stopEngines?: () => void }).stopEngines = () => engine.stop();
  log(`evaluating positions at ${SIZE}x${SIZE}`);
  await engine.ready();

  for (;;) {
    const job = await driver.nextJob();
    if (!job) break;
    const at = performance.now();
    // The search spends almost nothing on a reply it thinks is bad, and a reply
    // with two visits has no value worth reading. Widening the root is what
    // makes the whole spread of replies legible from one search.
    const reply = await engine.analyse(job.moves, { kind: 'visits', visits: job.visits },
      job.wideRootNoise === undefined ? {} : { wideRootNoise: job.wideRootNoise });
    if (reply.error) throw new Error(`${job.id}: ${reply.error}`);

    const result: Result = {
      id: job.id,
      moves: job.moves,
      winrate: reply.rootInfo?.winrate ?? 0.5,
      visits: reply.rootInfo?.visits ?? 0,
      replies: (reply.moveInfos ?? []).map((info) => ({
        move: info.move, winrate: info.winrate, visits: info.visits,
      })),
    };
    log(`${job.id} [${job.moves.join(' ')}]: ${(result.winrate * 100).toFixed(1)}% to move ` +
        `over ${result.visits} visits, ${result.replies.length} replies, ` +
        `${((performance.now() - at) / 1000).toFixed(0)}s`);
    await driver.report(result);
  }
  log('done');
}

main().catch((error) => log(`ERROR: ${error?.stack ?? error}`));
