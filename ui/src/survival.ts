/*
 * Survival: how long can you hold a won game against KataHex?
 *
 * KataHex plays itself from a fair opening and the second player's swap decision, both from
 * PlayHex's swap maps, the player's side well and the other badly, until it rates the
 * player's side at least `startWinrate`, or for `maxSelfPlayMoves` at most. Then
 * the player takes that side, KataHex plays the other properly, and the score is how many moves
 * the player makes before KataHex rates their side under `loseWinrate`.
 *
 * This file is the rules, the opponent's bad moves and the clock, with no engine and no ui, so
 * that they can be read and checked on their own. SurvivalGame.vue plays the game with them.
 *
 * Written for this app; not part of PlayHex.
 */
import { coordsToMove, mirrorMove, parseMove, validateMove } from '@playhex/move-notation';

/** Which side the player takes over: with the swap either is fair, so by default it is drawn. */
export type SurvivalColor = 'random' | 'red' | 'blue';

export type SurvivalSettings = {
    boardSize: number;

    color: SurvivalColor;

    /** Visits of the search KataHex answers the player with. */
    opponentVisits: number;

    /**
     * Visits of the search that rates the player's side after each of KataHex's answers, and
     * that picks that side's moves while KataHex plays itself.
     */
    evalVisits: number;

    /**
     * How badly the player's opponent plays while KataHex plays itself: a move picked at random among those whose
     * network policy is between these two fractions of its best move's, so neither a blunder
     * nor a good move.
     */
    weakMin: number;
    weakMax: number;

    /**
     * Self-play opens with a fair first move, which blue may swap, as a game with the swap rule
     * does, and either side is fair to take. Without it red is winning on the empty board
     * already, KataHex rating it 99.8% on 11x11 and 99.6% on 14x14.
     */
    swap: boolean;

    /** Self-play hands the player their side once it is rated at least this. */
    startWinrate: number;

    /**
     * Moves self-play may take to get there, the opening's included: past them the player takes
     * over wherever it got. If the game ends first, it is over before it began, and the settings
     * ask for too much.
     */
    maxSelfPlayMoves: number;

    /** The game ends as soon as KataHex answers a move and the player's side is rated under this. */
    loseWinrate: number;

    /** The player may take back a move and KataHex's answer, even the one that lost. */
    takeBack: boolean;

    /**
     * Fischer clock, in seconds: the player starts with `initial` (0: no clock), gains
     * `increment` a move, never past `cap` (0: no cap).
     */
    initial: number;
    increment: number;
    cap: number;
};

/** 14x14 by default: on 11x11 winrates saturate too fast for there to be a game. */
export const DEFAULT_SURVIVAL_SETTINGS: SurvivalSettings = {
    boardSize: 14,
    color: 'random',
    opponentVisits: 100,
    evalVisits: 200,
    weakMin: 0.05,
    weakMax: 0.3,
    swap: true,
    takeBack: true,
    startWinrate: 0.9,
    maxSelfPlayMoves: 20,
    loseWinrate: 0.1,
    initial: 20,
    increment: 5,
    cap: 25,
};

/** Settings as stored, which may predate a field or hold nonsense, made safe to play with. */
export const sanitizeSettings = (stored: Partial<SurvivalSettings> | undefined): SurvivalSettings => {
    const merged = { ...DEFAULT_SURVIVAL_SETTINGS, ...(stored ?? {}) };
    const number = (value: unknown, fallback: number, min: number, max: number): number => {
        const n = Number(value);

        return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : fallback;
    };
    const d = DEFAULT_SURVIVAL_SETTINGS;

    const weakMin = number(merged.weakMin, d.weakMin, 0, 1);

    return {
        boardSize: Math.round(number(merged.boardSize, d.boardSize, 3, 19)),
        color: (['random', 'red', 'blue'] as const).find(color => color === merged.color) ?? d.color,
        opponentVisits: Math.round(number(merged.opponentVisits, d.opponentVisits, 1, 100000)),
        evalVisits: Math.round(number(merged.evalVisits, d.evalVisits, 1, 100000)),
        weakMin,
        weakMax: number(merged.weakMax, d.weakMax, weakMin, 1),
        swap: typeof merged.swap === 'boolean' ? merged.swap : d.swap,
        takeBack: typeof merged.takeBack === 'boolean' ? merged.takeBack : d.takeBack,
        startWinrate: number(merged.startWinrate, d.startWinrate, 0, 1),
        maxSelfPlayMoves: Math.round(number(merged.maxSelfPlayMoves, d.maxSelfPlayMoves, 0, 361)),
        loseWinrate: number(merged.loseWinrate, d.loseWinrate, 0, 1),
        initial: number(merged.initial, d.initial, 0, 36000),
        increment: number(merged.increment, d.increment, 0, 36000),
        cap: number(merged.cap, d.cap, 0, 36000),
    };
};

/** What a best score is filed under: a score only compares with games played the same way. */
export const settingsKey = (s: SurvivalSettings): string => [
    s.boardSize, s.color, s.opponentVisits, s.evalVisits, s.weakMin, s.weakMax,
    s.swap ? 'swap' : 'noswap', s.takeBack ? 'takeback' : 'notakeback', s.startWinrate, s.maxSelfPlayMoves, s.loseWinrate, s.initial, s.increment, s.cap,
].join('/');

export type Stone = 0 | 1;

/** PlayHex's name for the second move that takes the first: its stone, mirrored, becomes blue's. */
export const SWAP = 'swap-pieces';

/**
 * A board as the game plays it: red (0) joins top to bottom, blue (1) left to right, and a cell
 * is named as the engine and the board view name it, column letter then row number. The second
 * move may be the swap.
 */
export class SurvivalBoard
{
    readonly cells: (Stone | null)[][];
    readonly moves: string[] = [];

    constructor(readonly size: number) {
        this.cells = Array.from({ length: size }, () => Array<Stone | null>(size).fill(null));
    }

    /** Whose turn it is: red moves first, and there is no swap. */
    get toMove(): Stone {
        return (this.moves.length % 2) as Stone;
    }

    isEmpty(move: string): boolean {
        const cell = this.parse(move);

        return cell !== null && this.cells[cell.row][cell.col] === null;
    }

    play(move: string): void {
        if (move === SWAP) {
            this.swap();
            return;
        }

        const cell = this.parse(move);

        if (cell === null || this.cells[cell.row][cell.col] !== null) {
            throw new Error(`cannot play ${move}`);
        }

        this.cells[cell.row][cell.col] = this.toMove;
        this.moves.push(move);
    }

    /** Takes the last move back, the swap included, and returns it. */
    undo(): string {
        const move = this.moves.pop();

        if (move === undefined) {
            throw new Error('no move to take back');
        }

        if (move === SWAP) {
            const first = this.parse(this.moves[0])!;
            const mirrored = this.parse(mirrorMove(this.moves[0] as Parameters<typeof mirrorMove>[0]))!;

            this.cells[mirrored.row][mirrored.col] = null;
            this.cells[first.row][first.col] = 0;
        } else {
            const cell = this.parse(move)!;

            this.cells[cell.row][cell.col] = null;
        }

        return move;
    }

    /**
     * Blue takes red's first move: the stone goes over to the cell mirrored through the long
     * diagonal, as blue's, and it is red's turn again, as in PlayHex.
     */
    private swap(): void {
        const first = this.moves[0];

        if (this.moves.length !== 1 || first === undefined) {
            throw new Error('the swap can only be the second move');
        }

        const from = this.parse(first)!;
        const to = this.parse(mirrorMove(first as Parameters<typeof mirrorMove>[0]))!;

        this.cells[from.row][from.col] = null;
        this.cells[to.row][to.col] = 1;
        this.moves.push(SWAP);
    }

    /** The stones of one colour, in the engine's notation. */
    stones(player: Stone): string[] {
        const stones: string[] = [];

        this.cells.forEach((line, row) => line.forEach((cell, col) => {
            if (cell === player) {
                stones.push(coordsToMove({ row, col }));
            }
        }));

        return stones;
    }

    full(): boolean {
        return this.cells.every(line => line.every(cell => cell !== null));
    }

    /** Who has joined their two sides, if anyone. */
    winner(): Stone | null {
        for (const player of [0, 1] as const) {
            if (this.connects(player)) {
                return player;
            }
        }

        return null;
    }

    private connects(player: Stone): boolean {
        const n = this.size;
        const seen = new Set<number>();
        const stack: { row: number, col: number }[] = [];
        const at = (row: number, col: number) => this.cells[row][col];

        for (let i = 0; i < n; ++i) {
            const start = player === 0 ? { row: 0, col: i } : { row: i, col: 0 };

            if (at(start.row, start.col) === player) {
                stack.push(start);
                seen.add(start.row * n + start.col);
            }
        }

        while (stack.length) {
            const { row, col } = stack.pop()!;

            if (player === 0 ? row === n - 1 : col === n - 1) {
                return true;
            }

            for (const [dr, dc] of NEIGHBOURS) {
                const r = row + dr;
                const c = col + dc;

                if (r < 0 || r >= n || c < 0 || c >= n || at(r, c) !== player || seen.has(r * n + c)) {
                    continue;
                }

                seen.add(r * n + c);
                stack.push({ row: r, col: c });
            }
        }

        return false;
    }

    private parse(move: string): { row: number, col: number } | null {
        if (!validateMove(move)) {
            return null;
        }

        const cell = parseMove(move);
        const { row, col } = cell;

        return row >= 0 && row < this.size && col >= 0 && col < this.size ? cell : null;
    }
}

/** A cell's six neighbours, in the layout where a1 b1 sit above a2 b2, shifted half a cell right. */
const NEIGHBOURS = [[-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0]] as const;

/**
 * Blue's move while KataHex plays itself: not its best, not a blunder. Among the empty cells
 * whose prior is between `min` and `max` times the best one's, uniformly; failing that, the
 * best cell under that band, then simply the best cell.
 *
 * `prior` is the network policy, row-major, negative where a move cannot be played.
 */
export const pickMediocreMove = (
    prior: number[][],
    isEmpty: (move: string) => boolean,
    min: number,
    max: number,
    random: () => number = Math.random,
): string | null => {
    const cells: { move: string, p: number }[] = [];

    prior.forEach((line, row) => line.forEach((p, col) => {
        const move = coordsToMove({ row, col });

        if (p >= 0 && isEmpty(move)) {
            cells.push({ move, p });
        }
    }));

    if (cells.length === 0) {
        return null;
    }

    cells.sort((a, b) => b.p - a.p);

    const best = cells[0];
    const rest = cells.slice(1);
    const band = rest.filter(({ p }) => p >= min * best.p && p <= max * best.p);

    if (band.length > 0) {
        return band[Math.min(band.length - 1, Math.floor(random() * band.length))].move;
    }

    return rest.find(({ p }) => p <= max * best.p)?.move ?? best.move;
};

/**
 * The player's clock: Fischer, with a cap on what increments can bring it up to. Runs only while
 * `start` has been called and `stop` has not; times are in milliseconds, from `now`.
 */
export class FischerClock
{
    private remaining: number;
    private since: number | null = null;

    constructor(
        initialSeconds: number,
        private incrementSeconds: number,
        private capSeconds: number,
        private now: () => number = () => performance.now(),
    ) {
        this.remaining = initialSeconds * 1000;
    }

    get running(): boolean {
        return this.since !== null;
    }

    start(): void {
        if (this.since === null) {
            this.since = this.now();
        }
    }

    /** Stops the clock, for the engine's turn or the end of the game. */
    stop(): void {
        this.remaining = this.left();
        this.since = null;
    }

    /** Stops the clock on a move played in time, and adds the increment, up to the cap. */
    moved(): void {
        this.stop();

        const raised = this.remaining + this.incrementSeconds * 1000;
        const cap = this.capSeconds * 1000;

        // The cap limits what an increment brings, and never takes back time already held.
        this.remaining = cap > 0 ? Math.max(this.remaining, Math.min(raised, cap)) : raised;
    }

    /** Milliseconds left, never below zero. */
    left(): number {
        const spent = this.since === null ? 0 : this.now() - this.since;

        return Math.max(0, this.remaining - spent);
    }

    flagged(): boolean {
        return this.left() <= 0;
    }
}

/** "0:07.4" under ten seconds, "1:05" above. */
export const formatClock = (ms: number): string => {
    const tenths = Math.floor(ms / 100);
    const seconds = Math.floor(tenths / 10);
    const minutes = Math.floor(seconds / 60);
    const ss = String(seconds % 60).padStart(2, '0');

    return seconds < 10 ? `${minutes}:${ss}.${tenths % 10}` : `${minutes}:${ss}`;
};

/** The empty cell the network likes best, for when a search has no candidate to offer. */
export const bestPriorMove = (prior: number[][], isEmpty: (move: string) => boolean): string | null => {
    let best: { move: string, p: number } | null = null;

    prior.forEach((line, row) => line.forEach((p, col) => {
        const move = coordsToMove({ row, col });

        if (p >= 0 && isEmpty(move) && (best === null || p > best.p)) {
            best = { move, p };
        }
    }));

    return (best as { move: string } | null)?.move ?? null;
};

/** The searched candidate the engine would play: the one it visited most. */
export const bestChild = <T extends { move: string, visits: number }>(children: T[], isEmpty: (move: string) => boolean): T | null => {
    let best: T | null = null;

    for (const child of children) {
        if (isEmpty(child.move) && (best === null || child.visits > best.visits)) {
            best = child;
        }
    }

    return best;
};
