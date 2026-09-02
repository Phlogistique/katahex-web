/*
 * The search's opinion of where to play, at the resolution of the network's own policy.
 *
 * Visit counts are the usual answer -- they are what the policy head is trained on -- but they
 * are a sample: at 200 visits a candidate's share moves in steps of half a percent and the whole
 * tail is exact zeros, which is coarser than the policy the search started from.
 *
 * Grill et al., "Monte-Carlo tree search as regularized policy optimization" (ICML 2020), show
 * the visit counts approximate the solution of a regularized policy optimization, and give that
 * solution in closed form:
 *
 *     pi(a) = lambda * prior(a) / (alpha - Q(a))
 *
 * with alpha the one scalar that makes it sum to 1. It is continuous in the priors, it is defined
 * for moves the search never visited, and it is closest to the visit counts exactly where those
 * are trustworthy, which is at high visit counts.
 */

/** The search parameters this reads have to be the ones Engine.java writes into analysis.cfg. */
const CPUCT_EXPLORATION = 0.9;
const CPUCT_EXPLORATION_LOG = 0.6;
const CPUCT_EXPLORATION_BASE = 500;

/** KataGo's own constants: the puct numerator offset, and the root first-play-urgency reduction. */
const WEIGHT_PUCT_OFFSET = 0.01;
const ROOT_FPU_REDUCTION_MAX = 0.2;

export type SearchedMove = {
    /** Value of the move for the player to move, in [-1, 1]. */
    utility: number;
};

type Cell = { row: number, col: number, prior: number, q: number };

/**
 * `prior` is the network's policy, row-major, negative on cells that cannot be played.
 * `searched` holds the moves the search gave a child to, `rootUtility` and `rootWeight` the
 * position's own value and the search's total weight. Utilities are all from the perspective
 * of the player to move, which is how the engine reports them.
 */
export const improvedPolicy = (
    prior: number[][],
    searched: (row: number, col: number) => SearchedMove | undefined,
    rootUtility: number,
    rootWeight: number,
): number[][] => {
    const cells: Cell[] = [];
    let priorSum = 0;
    let searchedPriorMass = 0;

    for (let row = 0; row < prior.length; ++row) {
        for (let col = 0; col < prior[row].length; ++col) {
            if (prior[row][col] < 0) {
                continue;
            }

            cells.push({ row, col, prior: prior[row][col], q: NaN });
            priorSum += prior[row][col];

            if (searched(row, col)) {
                searchedPriorMass += prior[row][col];
            }
        }
    }

    if (cells.length === 0 || priorSum <= 0) {
        return prior.map(row => row.map(value => Math.max(0, value)));
    }

    // What the search assumes about a move it has not tried yet: the position's own value, cut
    // by how much of the policy it has already accounted for.
    const unsearchedQ = rootUtility - ROOT_FPU_REDUCTION_MAX * Math.sqrt(searchedPriorMass);

    for (const cell of cells) {
        cell.q = searched(cell.row, cell.col)?.utility ?? unsearchedQ;
    }

    const cpuct = CPUCT_EXPLORATION
        + CPUCT_EXPLORATION_LOG * Math.log((rootWeight + CPUCT_EXPLORATION_BASE) / CPUCT_EXPLORATION_BASE);
    const lambda = cpuct * Math.sqrt(rootWeight + WEIGHT_PUCT_OFFSET) / (cells.length + rootWeight);

    const alpha = solveAlpha(cells, lambda, priorSum);
    const policy = prior.map(row => row.map(() => 0));

    for (const cell of cells) {
        policy[cell.row][cell.col] = lambda * cell.prior / (alpha - cell.q);
    }

    return policy;
};

/**
 * The alpha that makes the values sum to 1. The sum falls monotonically as alpha grows, and is
 * above 1 just past the best move's value and at most 1 by lambda * priorSum past it, so plain
 * bisection between those brackets it.
 */
const solveAlpha = (cells: Cell[], lambda: number, priorSum: number): number => {
    const best = Math.max(...cells.map(cell => cell.q));
    const total = (alpha: number) =>
        cells.reduce((sum, cell) => sum + lambda * cell.prior / (alpha - cell.q), 0);

    let low = best + 1e-9;
    let high = best + lambda * priorSum + 1e-9;

    for (let i = 0; i < 60; ++i) {
        const middle = (low + high) / 2;

        if (total(middle) > 1) {
            low = middle;
        } else {
            high = middle;
        }
    }

    return (low + high) / 2;
};
