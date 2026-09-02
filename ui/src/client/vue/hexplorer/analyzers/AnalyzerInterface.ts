import type { AnalysisInput, AnalysisOutput } from '../../../../shared/app/hexplorer.js';

export interface AnalyzerInterface
{
    /**
     * Returns analysis for a position.
     * Given the engine features, can fill multiple optional analysis elements.
     */
    analyzePosition(input: AnalysisInput): Promise<AnalysisOutput>;

    /**
     * How it should be displayed to players in UI.
     */
    getName(): string;

    /**
     * Called when leaving the page, to persist whatever caching the analyzer uses.
     */
    persistCache?(): void;

    /**
     * Called with the position now displayed, or null when nothing is analyzed.
     * An analyzer that keeps searching uses it to know which position to search, and to tell
     * it apart from the ancestors it is also asked about.
     */
    setDisplayedPosition?(input: AnalysisInput | null): void;
}
