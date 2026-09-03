import { ref } from 'vue';
import { analysisCacheKey, type AnalysisInput, type AnalysisOutput } from '../../../../shared/app/hexplorer.js';
import { engine, type EngineAnalysis } from '../../../../engine.js';
import { analysisStore, FLOOR } from '../../../../analysisStore.js';
import { AnalyzerInterface } from './AnalyzerInterface.js';

/**
 * The on-device engine. Every entry in the analyzer picker is one of these, differing only in
 * when it stops searching the position on screen: at no visits at all, at one (the network's
 * raw policy, which is what "intuition" is), at a fixed depth, or never, in the manner of
 * Lizzie.
 *
 * Hexplorer reads an analysis in updateAnalysis and paints it, so a growing search is shown by
 * asking it to read again after every partial result, and answering with the search so far.
 * Every other position it asks about (the ancestors of the evaluation graph) gets an ordinary
 * bounded search instead, which is also what the position on screen falls back to for callers
 * that need a move rather than a picture.
 *
 * Written for this app; not part of PlayHex.
 */
export class KatahexAnalyzer implements AnalyzerInterface
{
    /** Paused by the user. Search stops, whatever it had found stays on screen. */
    readonly paused = ref(false);

    /** Visits in the search on screen, shown on the pause button. */
    readonly visits = ref(0);

    private refresh: () => void = () => {};

    /** True while we are the one asking hexplorer to read the analysis again. */
    private refreshing = false;

    /** Stopped while the app is in the background, where a search would only cost battery. */
    private awake = true;

    private displayed: AnalysisInput | null = null;
    private searched: EngineAnalysis | null = null;
    private waiting: { minVisits: number, resolve: (analysis: EngineAnalysis | null) => void }[] = [];
    private idle: (() => void)[] = [];

    constructor(
        private maxVisits: number,
    ) {}

    getName(): string
    {
        if (this.maxVisits < 1) {
            return '--';
        }

        if (this.maxVisits === 1) {
            return 'KataHex intuition';
        }

        if (!Number.isFinite(this.maxVisits)) {
            return 'KataHex live';
        }

        return `KataHex ${this.maxVisits} visits`;
    }

    /** Whether it runs a search worth watching grow, and so worth a pause button. */
    get searches(): boolean
    {
        return this.maxVisits > 1;
    }

    setRefresh(refresh: () => void): void
    {
        this.refresh = refresh;
    }

    /**
     * The position hexplorer is showing. Anything else it asks about is a position it is
     * filling the evaluation graph with, and must not steal the search.
     */
    setDisplayedPosition(input: AnalysisInput | null): void
    {
        if (input && this.displayed && analysisCacheKey(input) === analysisCacheKey(this.displayed)) {
            return;
        }

        this.displayed = input;
        this.searched = null;
        this.visits.value = 0;
        this.sync();
    }

    /** Called when another analyzer is picked: nothing of ours should keep searching. */
    stop(): void
    {
        this.displayed = null;
        this.sync();
    }

    togglePause(): void
    {
        this.paused.value = !this.paused.value;
        this.sync();
    }

    setAwake(awake: boolean): void
    {
        this.awake = awake;
        this.sync();
    }

    async analyzePosition(input: AnalysisInput): Promise<AnalysisOutput>
    {
        if (this.maxVisits < 1) {
            return {};
        }

        if (this.displayed && analysisCacheKey(input) === analysisCacheKey(this.displayed)) {
            // Our own re-read after a partial result: answer with the search as it stands.
            const searched = await this.searchedSoFar(this.refreshing ? 1 : this.floor);

            if (searched) {
                analysisStore.write(input, searched);

                // isSearching() reads the store: the search that just landed may be the one
                // that reached the depth asked for, and so the end of it.
                this.release();
            }

            // The store may hold a deeper search than the one running: this one starts from
            // nothing even on a position already studied, and the overlay must not go
            // backwards on landing there.
            this.visits.value = analysisStore.policyVisits(input);

            return analysisStore.read(input) ?? {};
        }

        return await analysisStore.fill(input, this.floor, () => engine.analyse(input, this.floor));
    }

    /**
     * Resolves once the engine has nothing to do for the position on screen: pausing, switching
     * away and reaching the depth asked for all count, being in the background does not. The
     * evaluation graph's ancestors wait on it, so that the position the user is looking at has
     * the engine to itself.
     *
     * With `KataHex live` there is no such moment while it is running, so that graph fills
     * while the search is paused.
     */
    whenIdle(): Promise<void>
    {
        if (this.free()) {
            return Promise.resolve();
        }

        return new Promise(resolve => this.idle.push(resolve));
    }

    persistCache(): void
    {
        analysisStore.persist();
    }

    /** What a position that is not the one on screen is searched to, at most. */
    private get floor(): number
    {
        return Math.min(this.maxVisits, FLOOR);
    }

    private isSearching(): boolean
    {
        if (!this.displayed || this.maxVisits < 1 || !engine.available) {
            return false;
        }

        // Nothing to add to a position the store already holds this deep, here or in a past
        // session.
        if (analysisStore.policyVisits(this.displayed) >= this.maxVisits) {
            return false;
        }

        return this.awake && !this.paused.value;
    }

    private sync(): void
    {
        engine.unwatch(this);

        if (this.isSearching()) {
            engine.watch(this.displayed!, this.maxVisits, analysis => this.onReport(analysis), this);
        }

        this.release();
    }

    private onReport(analysis: EngineAnalysis): void
    {
        this.searched = analysis;
        this.release();

        // analyzePosition runs before updateAnalysis awaits anything, so this flag marks
        // exactly the read this call provokes, and no other.
        this.refreshing = true;
        try {
            this.refresh();
        } finally {
            this.refreshing = false;
        }
    }

    private searchedSoFar(minVisits: number): Promise<EngineAnalysis | null> {
        if (this.searched && this.searched.visits >= minVisits) {
            return Promise.resolve(this.searched);
        }

        // Nothing is going to grow: answer with whatever there is, rather than leaving
        // hexplorer waiting on an analysis that will never arrive.
        if (!this.isSearching()) {
            return Promise.resolve(this.searched);
        }

        return new Promise(resolve => this.waiting.push({ minVisits, resolve }));
    }

    /** Whether the engine is there to be spent on something other than the position on screen. */
    private free(): boolean
    {
        return this.awake && !this.isSearching();
    }

    private release(): void
    {
        if (this.free()) {
            for (const resolve of this.idle.splice(0)) {
                resolve();
            }
        }

        this.waiting = this.waiting.filter(waiter => {
            if (this.searched && this.searched.visits >= waiter.minVisits) {
                waiter.resolve(this.searched);
                return false;
            }

            if (!this.isSearching()) {
                waiter.resolve(this.searched);
                return false;
            }

            return true;
        });
    }
}
