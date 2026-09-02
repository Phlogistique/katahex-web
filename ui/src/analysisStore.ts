import { coordsToMove } from './shared/move-notation/move-notation.js';
import { analysisCacheKey, type AnalysisInput, type AnalysisOutput } from './shared/app/hexplorer.js';
import type { EngineAnalysis } from './engine.js';

/**
 * What a position's numbers have to be worth to be kept and acted on. Below it a winrate is
 * noise: the evaluation graph is drawn at this depth, and so is the move auto-play makes,
 * which must not come from a search that has barely started.
 */
export const FLOOR = 200;

/**
 * The net the numbers were computed with, which has to be the one Engine.java loads. A different
 * net answers differently, so it does not get to read this.
 */
const STORE = 'analysisCache-hex27x3';

/** The stores from before the depths were merged, which carried theirs in the name. */
const OLD_STORES: Record<string, number> = {
    'analysisCache': 1,
    'analysisCache-1v': 1,
    'analysisCache-200v': 200,
    'analysisCache-1000v': 1000,
    'analysisCache-live': 200,
};

/** What is known about one position, each half at the deepest search that produced it. */
type Entry = {
    whiteWin: number;

    /** Visits behind the winrate. */
    visits: number;

    policy?: number[][];

    /** Visits behind the policy, which need not be the search the winrate came from. */
    policyVisits?: number;
};

/**
 * Every analysis the engine has produced, by position, kept in localStorage.
 *
 * A deeper search of a position says everything a shallower one did, only better: improvedPolicy
 * makes every depth the same quantity at a different resolution. So there is one store rather
 * than one per depth, a read asks for the depth it needs and is happy with more, and a write
 * only lands when it beats what is there.
 *
 * Written for this app; not part of PlayHex.
 */
class AnalysisStore
{
    private entries: Record<string, Entry> = {};

    // Searches in flight, so concurrent calls for the same position share one instead of
    // racing past the check in fill() and asking the engine twice.
    private pending: Record<string, Promise<EngineAnalysis>> = {};

    constructor()
    {
        this.load();
    }

    /** What is stored for a position, as long as its winrate is worth `minVisits`. */
    read(input: AnalysisInput, minVisits = 0): AnalysisOutput | null
    {
        const entry = this.entries[analysisCacheKey(input)];

        if (!entry || entry.visits < minVisits) {
            return null;
        }

        return {
            whiteWin: entry.whiteWin,
            policy: entry.policy,
            recommendedMove: mostSearchedMove(entry.policy),
        };
    }

    /** How deep the stored policy is: what a search of this position would have to beat. */
    policyVisits(input: AnalysisInput): number
    {
        return this.entries[analysisCacheKey(input)]?.policyVisits ?? 0;
    }

    /** Keeps each half of an analysis that is deeper than the one stored, its children too. */
    write(input: AnalysisInput, analysis: EngineAnalysis): void
    {
        const key = analysisCacheKey(input);

        this.writeEval(key, analysis.whiteWin, analysis.visits);
        this.writePolicy(key, analysis.policy, analysis.visits);

        // A search also evaluates the candidates it gives visits to, and a candidate's winrate
        // is the number the evaluation graph reads for the position it leads to. So a minute
        // spent on one position is a free reading of the lines out of it.
        for (const child of analysis.children) {
            if (child.visits >= FLOOR) {
                this.writeEval(analysisCacheKey(afterMove(input, child.move)), child.whiteWin, child.visits);
            }
        }
    }

    /** The stored numbers for a position, searching it first unless they are deep enough. */
    async fill(input: AnalysisInput, minVisits: number, search: () => Promise<EngineAnalysis>): Promise<AnalysisOutput>
    {
        const stored = this.read(input, minVisits);

        if (stored) {
            return stored;
        }

        const key = analysisCacheKey(input);
        const pending = this.pending[key] ??= search().finally(() => {
            delete this.pending[key];
        });

        this.write(input, await pending);

        return this.read(input) ?? {};
    }

    persist(): void
    {
        try {
            localStorage?.setItem(STORE, JSON.stringify(this.entries));
        } catch (e) {
            // Past the quota, most likely. Nothing to do about it here, but the searches of
            // this session are still worth showing.
            console.error('Could not save the analysis cache', e);
        }
    }

    private writeEval(key: string, whiteWin: number | undefined, visits: number): void
    {
        const entry = this.entries[key];

        if (whiteWin === undefined || visits < 1 || (entry && entry.visits >= visits)) {
            return;
        }

        this.entries[key] = { ...entry, whiteWin, visits };
    }

    private writePolicy(key: string, policy: number[][] | undefined, visits: number): void
    {
        const entry = this.entries[key];

        if (!policy || !entry || (entry.policyVisits ?? 0) >= visits) {
            return;
        }

        entry.policy = policy;
        entry.policyVisits = visits;
    }

    private load(): void
    {
        this.entries = JSON.parse(localStorage?.getItem(STORE) ?? 'null') ?? {};

        for (const [name, visits] of Object.entries(OLD_STORES)) {
            const raw = localStorage?.getItem(name);

            if (raw === null || raw === undefined) {
                continue;
            }

            for (const [key, output] of Object.entries(JSON.parse(raw) as Record<string, AnalysisOutput>)) {
                this.writeEval(key, output.whiteWin, visits);
                this.writePolicy(key, output.policy, visits);
            }

            localStorage?.removeItem(name);
        }
    }
}

/** The position a move leads to: it is the other player's turn, one stone heavier. */
const afterMove = (input: AnalysisInput, move: string): AnalysisInput => ({
    size: input.size,
    color: input.color === 'black' ? 'white' : 'black',
    black: input.color === 'black' ? [...input.black, move] : input.black,
    white: input.color === 'white' ? [...input.white, move] : input.white,
});

/** Cells the search never visited are 0, so the argmax is the engine's own best move. */
const mostSearchedMove = (policy: undefined | number[][]) => {
    if (!policy) {
        return null;
    }

    let best = null;
    let bestShare = 0;

    for (let row = 0; row < policy.length; ++row) {
        for (let col = 0; col < policy[row].length; ++col) {
            if (policy[row][col] > bestShare) {
                bestShare = policy[row][col];
                best = coordsToMove({ row, col });
            }
        }
    }

    return best;
};

export const analysisStore = new AnalysisStore();
