/*
 * Winrates as odds, for the option that shows them so.
 *
 * Hex winrates saturate: as soon as one side is ahead every candidate reads 0% or 100%, and a
 * graph of them sits flat against its edge for the rest of the game. Odds keep the difference
 * visible -- 1:500 is twice as bad as 1:250 where both are "0.2%" -- and log-odds turn it into
 * a distance. Two values are exact: a move that connects is a won position, 100% and odds of
 * infinity, which is the most useful thing the board can say about it. Short of that the net
 * cannot go past ~1e-7 in float32, so finite odds are capped for display at a million to one.
 *
 * Written for this app; not part of PlayHex.
 */

/** Odds past which the board says ">1M" and the graph and bar touch their edge. */
export const ODDS_CAP = 1e6;

const LOGIT_CAP = Math.log(ODDS_CAP);

const logit = (p: number): number => Math.log(p / (1 - p));

/**
 * A winrate placed on a scale from 0 to 1 that is linear in log-odds, a million to one at the
 * edges and even at the middle. 0 and 1 land on the edges.
 */
export const logitScale = (p: number): number =>
    0.5 + Math.max(-LOGIT_CAP, Math.min(LOGIT_CAP, logit(p))) / (2 * LOGIT_CAP);

const formatRatio = (ratio: number): string => {
    if (!Number.isFinite(ratio)) {
        return '∞';
    }

    if (ratio >= ODDS_CAP) {
        return '>1M';
    }

    if (ratio < 10) {
        return String(Math.round(ratio * 10) / 10);
    }

    if (ratio < 1000) {
        return String(Math.round(ratio));
    }

    return ratio < 10000 ? `${(ratio / 1000).toFixed(1)}k` : `${Math.round(ratio / 1000)}k`;
};

/** A winrate as odds for the side it belongs to: "3:1", "1:330", "∞:1" for a won position. */
export const formatOdds = (p: number): string => p >= 0.5
    ? `${formatRatio(p / (1 - p))}:1`
    : `1:${formatRatio((1 - p) / p)}`;
