import { KatahexAnalyzer } from './client/vue/hexplorer/analyzers/KatahexAnalyzer.js';

/** Selected on load: a few seconds of search, and it stops on its own. */
export const defaultAnalyzer = new KatahexAnalyzer(200);

/**
 * What the analyzer picker offers, best first: one engine, stopping at different depths. It
 * runs about 64 visits per second on a Pixel 7, so the fixed depths are a few seconds and
 * around fifteen per position; a single visit is one network evaluation, and answers whether
 * or not a search would.
 */
export const analyzers = [
    new KatahexAnalyzer(Infinity),
    defaultAnalyzer,
    new KatahexAnalyzer(1000),
    new KatahexAnalyzer(1),
    new KatahexAnalyzer(0),
];

/** Hexplorer's updateAnalysis: how a search that has grown gets painted. */
export const setAnalyzersRefresh = (refresh: () => void): void => {
    for (const analyzer of analyzers) {
        analyzer.setRefresh(refresh);
    }
};

export const setAnalyzersAwake = (awake: boolean): void => {
    for (const analyzer of analyzers) {
        analyzer.setAwake(awake);
    }
};
