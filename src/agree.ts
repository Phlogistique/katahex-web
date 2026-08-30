// Asks both precisions the same question and records both answers.
//
// This is the half of the fixed-time comparison that does not need games. What
// half precision costs is a different move chosen now and then; how often, and
// how much the search itself thinks the difference is worth, bounds the penalty
// far more tightly than a few hundred games could.

import { AnalysisEngine, type Condition, type Precision } from './analysisEngine';
import { bridge } from './harness';

export type Job = { id: string; moves: string[]; condition: Condition };

type Answer = { move: string; winrate: number; visits: number; ms: number };
export type Result = { id: string; moves: string[]; fp16: Answer; fp32: Answer; agreed: boolean };

const params = new URLSearchParams(location.search);
const SIZE = Number(params.get('size') ?? 11);
const THREADS = Number(params.get('threads') ?? 16);
const { driver, log } = bridge<Job, Result>();

async function ask(engine: AnalysisEngine, job: Job): Promise<Answer> {
  const at = performance.now();
  const reply = await engine.analyse(job.moves, job.condition);
  if (reply.error) throw new Error(reply.error);
  const best = reply.moveInfos?.reduce((a, b) => (a.order <= b.order ? a : b));
  if (!best) throw new Error(`no moves for ${job.id}`);
  return {
    move: best.move,
    winrate: reply.rootInfo?.winrate ?? 0.5,
    visits: reply.rootInfo?.visits ?? 0,
    ms: performance.now() - at,
  };
}

async function main() {
  const engines: Record<Precision, AnalysisEngine> = {
    fp16: new AnalysisEngine('fp16', SIZE, log, THREADS),
    fp32: new AnalysisEngine('fp32', SIZE, log, THREADS),
  };
  (globalThis as { stopEngines?: () => void }).stopEngines =
    () => { engines.fp16.stop(); engines.fp32.stop(); };
  await Promise.all([engines.fp16.ready(), engines.fp32.ready()]);
  log('both engines ready');

  let asked = 0;
  let agreed = 0;
  for (;;) {
    const job = await driver.nextJob();
    if (!job) break;
    // One at a time, so neither engine is timed while the other holds the GPU.
    const fp16 = await ask(engines.fp16, job);
    const fp32 = await ask(engines.fp32, job);
    asked++;
    if (fp16.move === fp32.move) agreed++;
    else log(`${job.id} (${job.moves.length} stones): fp16 ${fp16.move} ` +
             `${(fp16.winrate * 100).toFixed(1)}%, fp32 ${fp32.move} ${(fp32.winrate * 100).toFixed(1)}%`);
    await driver.report({ id: job.id, moves: job.moves, fp16, fp32, agreed: fp16.move === fp32.move });
    if (asked % 25 === 0) log(`${agreed}/${asked} agreed`);
  }
  log(`${agreed}/${asked} agreed`);
  log('done');
}

main().catch((error) => log(`ERROR: ${error?.stack ?? error}`));
