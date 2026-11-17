/*
 * Written here, at PlayHex's module path. PlayHex's analyzers ask its server, which says
 * every 30 seconds which engines have a worker online. Here there is one engine, KataHex,
 * on the device or on a rented GPU behind the same bridge, offered at several depths.
 */
import { ref, type Ref } from 'vue';
import type { AnalyzerInterface } from '../analyzers/AnalyzerInterface.js';
import type { AnalysisEngine } from '../../../../shared/app/hexplorer.js';
import { analyzers, defaultAnalyzer } from '../../../../analyzers.js';

/** The analyzers the AI tab offers, the one selected on load first. */
export const createKatahexAnalyzers = (): AnalyzerInterface[] => [
    defaultAnalyzer,
    ...analyzers.filter(analyzer => analyzer !== defaultAnalyzer),
];

export const analysisEngineName = (engine: AnalysisEngine): string => engine;

/** Always available: the engine says itself when it is not, in the status line. */
export const useAnalysisEngines = (_analysisError: Ref<null | 'engine_unavailable' | 'failed'>) => ({
    aiAvailability: ref(null),
    refreshEnginesStatus: async (): Promise<void> => {},
    isAnalyzerAvailable: (_analyzer: AnalyzerInterface): boolean => true,
    isEngineAvailable: (_engine: AnalysisEngine): boolean => true,
});
