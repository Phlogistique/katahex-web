/*
 * How fast the engine is going, for the sidebar readout.
 *
 * Both counters are cumulative and are sampled rather than watched: a rate over
 * the last few seconds is what a reader wants, and it needs nothing to say when
 * a search has stopped.
 *
 * What is shown only changes while a search is running. Most of the analyzers
 * search in bursts of a few hundred visits, and a figure that appeared for four
 * seconds and then disappeared again would be unreadable; this keeps the last
 * one, and `searching` is how the ui says it is no longer current.
 *
 * Written for this app; not part of PlayHex.
 */
import { ref } from 'vue';
import { engine } from './engine.js';
import type { EvalStats } from '@engine/netRunner';

/**
 * Null until a search has actually run. `netEvals` counts positions through the
 * net, which is what the benchmark means by evals/s, and `batch` is how many of
 * them go through in one dispatch. Both are null where the net is not ours to
 * count.
 */
export const speed = ref<{ visits: number; netEvals: number | null; batch: number | null } | null>(null);

/** Whether that is a search now, or the last one there was. */
export const searching = ref(false);

/** The web build's net counters. The Android app runs its net inside the engine and reports none. */
let net: EvalStats | null = null;
export const noteNetStats = (stats: EvalStats): void => { net = stats; };

const WINDOW_MS = 4000;
const PERIOD_MS = 500;

/** Below this a rate is rounding noise around a search that has stopped. */
const SEARCHING_VISITS_PER_SECOND = 1;

type Sample = { at: number; visits: number; evals: number; rows: number };
const samples: Sample[] = [];

setInterval(() => {
    const at = performance.now();
    samples.push({ at, visits: engine.totalVisits, evals: net?.evals ?? 0, rows: net?.rows ?? 0 });
    while (samples.length > 2 && at - samples[0].at > WINDOW_MS) {
        samples.shift();
    }

    const first = samples[0];
    const last = samples[samples.length - 1];
    const seconds = (last.at - first.at) / 1000;
    if (seconds <= 0) {
        return;
    }

    const visits = (last.visits - first.visits) / seconds;
    searching.value = visits >= SEARCHING_VISITS_PER_SECOND;
    if (!searching.value) {
        return;
    }

    const evals = last.evals - first.evals;
    const rows = last.rows - first.rows;
    speed.value = {
        visits,
        netEvals: net ? rows / seconds : null,
        batch: net && evals > 0 ? rows / evals : null,
    };
}, PERIOD_MS);
