// Rates every first move, so a match can be played from openings that are
// actually close.
//
// Hex is a first-player win, so a pair of games from a first move that both
// engines convert scores 1-1 and says nothing. What carries information is an
// opening near 50%, which is also why no pie rule is needed here: a swap-unaware
// engine would swap every good first move and the bias would only change sides.
//
// A 180 degree rotation of the board is the same game, so only half the first
// moves are distinct.

import { AnalysisEngine } from './analysisEngine';
import { bridge } from './harness';

export type Job = { move: string; visits: number };
export type Result = { move: string; blackWinrate: number; visits: number; seconds: number };

const params = new URLSearchParams(location.search);
const SIZE = Number(params.get('size') ?? 11);

const { driver, log } = bridge<Job, Result>();

async function main() {
  const engine = new AnalysisEngine('fp32', SIZE, log);
  log(`rating first moves at ${SIZE}x${SIZE}`);
  await engine.ready();

  for (;;) {
    const job = await driver.nextJob();
    if (!job) break;
    const at = performance.now();
    const reply = await engine.analyse([job.move], { kind: 'visits', visits: job.visits });
    if (reply.error) throw new Error(reply.error);

    // Winrates are reported for the side to move, which after black's first
    // move is white.
    const result: Result = {
      move: job.move,
      blackWinrate: 1 - (reply.rootInfo?.winrate ?? 0.5),
      visits: reply.rootInfo?.visits ?? 0,
      seconds: (performance.now() - at) / 1000,
    };
    log(`${job.move}: black ${(result.blackWinrate * 100).toFixed(1)}% ` +
        `over ${result.visits} visits in ${result.seconds.toFixed(0)}s`);
    await driver.report(result);
  }
  log('done');
}

main().catch((error) => log(`ERROR: ${error?.stack ?? error}`));
