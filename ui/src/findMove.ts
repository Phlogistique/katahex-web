/*
 * Find KataHex's move: the player is red against KataHex, but each turn they pick one of three
 * moves, the one KataHex's search plays and two decoys that look natural and lose.
 *
 * Red without the swap is so far ahead on the empty board that every natural move wins, and
 * there would be nothing to find. So red opens on a fair first move from PlayHex's swap maps,
 * and the net plays itself from there, red its favourite move and blue a middling one that
 * leaves blue at most 65%, until the net rates red winning: a position where red is ahead but
 * mistakes cost. Were red far behind, its best move and a blunder would lose alike.
 *
 * This file is the rules and the decoys, with the engine passed in, and no ui.
 * FindMoveGame.vue plays the game with them.
 *
 * Written for this app; not part of PlayHex.
 */
import { coordsToMove, parseMove as parseCell, type Move } from '@playhex/move-notation';
import type { EngineAnalysis } from './engine.js';
import { bestChild, bestPriorMove, pickMediocreMove, SurvivalBoard, type Stone } from './survival.js';
import { pickFairFirstMove } from './shared/swap-maps/swapDecision.js';

export type FindMoveSettings = {
    boardSize: number;

    /** Visits of the search that finds KataHex's move. */
    visits: number;

    /** Visits of the search that checks a decoy loses. */
    checkVisits: number;
};

export const DEFAULT_FIND_MOVE_SETTINGS: FindMoveSettings = {
    boardSize: 11,
    visits: 200,
    checkVisits: 50,
};

export const sanitizeFindMoveSettings = (stored: Partial<FindMoveSettings> | undefined): FindMoveSettings => {
    const merged = { ...DEFAULT_FIND_MOVE_SETTINGS, ...(stored ?? {}) };
    const integer = (value: unknown, fallback: number, min: number, max: number): number => {
        const n = Math.round(Number(value));

        return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : fallback;
    };
    const d = DEFAULT_FIND_MOVE_SETTINGS;

    return {
        boardSize: integer(merged.boardSize, d.boardSize, 5, 19),
        visits: integer(merged.visits, d.visits, 1, 100000),
        checkVisits: integer(merged.checkVisits, d.checkVisits, 1, 100000),
    };
};

/** What a best score is filed under. */
export const findMoveKey = (s: FindMoveSettings): string => [s.boardSize, s.visits, s.checkVisits].join('/');

/** A position to search, which may be the board's with a stone added or taken away. */
export type Position = { red: string[], blue: string[], toMove: Stone };

export type Analyse = (position: Position, visits: number) => Promise<EngineAnalysis>;

export const positionOf = (board: SurvivalBoard): Position => ({ red: board.stones(0), blue: board.stones(1), toMove: board.toMove });

const redWin = (analysis: EngineAnalysis): number => 1 - (analysis.whiteWin ?? 0.5);

/** The net rates red at least this when self-play hands the position over. */
const OPENING_RED_WIN = 0.85;

/** Self-play hands over after this many moves even if red is not there yet. */
const OPENING_MAX_MOVES = 30;

/** Blue's middling move leaves blue at most this, by the net... */
const MIDDLING_BLUE_WIN = 0.65;

/** ...if one of this many tries does. */
const MIDDLING_BLUE_TRIES = 10;

/**
 * Plays the opening on `board` through `play`: red's fair first move, then the net against
 * itself, red its favourite move and blue a middling one that leaves blue at most
 * MIDDLING_BLUE_WIN, until the net rates red winning and the board holds at least a number of
 * moves drawn up to ten, so that positions vary. Ends with red to move.
 */
export const playOpening = async (
    board: SurvivalBoard,
    analyse: Analyse,
    play: (move: string) => void,
    random: () => number = Math.random,
): Promise<void> => {
    const isEmpty = (move: string) => board.isEmpty(move);
    const first = pickFairFirstMove(board.size, random);

    if (first !== null) {
        play(first);
    }

    const least = 2 * Math.floor(random() * 6);

    for (;;) {
        if (board.winner() !== null || board.full()) {
            throw new Error('the opening ended the game');
        }

        if (board.toMove === 0) {
            const net = await analyse(positionOf(board), 1);

            if (board.moves.length >= OPENING_MAX_MOVES || (board.moves.length >= least && redWin(net) >= OPENING_RED_WIN)) {
                return;
            }

            play(bestPriorMove(net.prior, isEmpty)!);
        } else {
            play(await middlingBlueMove(board, analyse, random));
        }
    }
};

/**
 * Blue's middling move: the first of its middling moves, tried in random order, after which the
 * net rates blue at most MIDDLING_BLUE_WIN, or failing that the move tried that leaves blue the
 * least.
 */
export const middlingBlueMove = async (board: SurvivalBoard, analyse: Analyse, random: () => number = Math.random): Promise<string> => {
    const net = await analyse(positionOf(board), 1);
    const tried: { move: string, blueWin: number }[] = [];

    for (let i = 0; i < MIDDLING_BLUE_TRIES; ++i) {
        const move = pickMediocreMove(net.prior, m => board.isEmpty(m) && !tried.some(t => t.move === m), 0.05, 0.3, random);

        if (move === null) {
            break;
        }

        const blueWin = 1 - redWin(await analyse({ red: board.stones(0), blue: [...board.stones(1), move], toMove: 0 }, 1));

        tried.push({ move, blueWin });

        if (blueWin <= MIDDLING_BLUE_WIN) {
            break;
        }
    }

    return tried.sort((a, b) => a.blueWin - b.blueWin)[0].move;
};

/** A decoy loses if a search after it rates red under this. */
const DECOY_RED_WIN = 0.3;

/** And at least this under KataHex's move, so that the two do not lose alike. */
const DECOY_GAP = 0.3;

/** A candidate the net alone rates red above this is taken to win, and not searched. */
const SURELY_WINS = 0.8;

/** A move to choose, how a search rates red after it and, for a decoy, blue's answer. */
export type Choice = { move: string, redWin: number, answer?: string };

export type Question = {
    /** The move KataHex's search plays, and how it rates red after it. */
    kataHex: Choice;

    /** Up to two moves that lose. */
    decoys: Choice[];
};

const parseMove = (move: string) => parseCell(move as Move);

const NEIGHBOURS = [[-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0]] as const;
const BRIDGES = [[-2, 1], [-1, 2], [1, 1], [2, -1], [1, -2], [-1, -1]] as const;

/** The cells around a move, adjacent or a bridge away. */
const around = (move: string, size: number): string[] => {
    const { row, col } = parseMove(move);

    return [...NEIGHBOURS, ...BRIDGES]
        .map(([dr, dc]) => ({ row: row + dr, col: col + dc }))
        .filter(({ row, col }) => row >= 0 && row < size && col >= 0 && col < size)
        .map(coordsToMove);
};

/** Cells between two cells, on this board's layout. */
const distance = (a: string, b: string): number => {
    const p = parseMove(a);
    const q = parseMove(b);
    const dr = p.row - q.row;
    const dc = p.col - q.col;

    return (Math.abs(dr) + Math.abs(dc) + Math.abs(dr + dc)) / 2;
};

const ranked = (prior: number[][], isEmpty: (move: string) => boolean): { move: string, p: number }[] => {
    const cells: { move: string, p: number }[] = [];

    prior.forEach((line, row) => line.forEach((p, col) => {
        const move = coordsToMove({ row, col });

        if (p >= 0 && isEmpty(move)) {
            cells.push({ move, p });
        }
    }));

    return cells.sort((a, b) => b.p - a.p);
};

/**
 * KataHex's move in the position on the board, red to move, and two decoys.
 *
 * The decoys are drawn from moves a human might play: the net's other favourites, the cells
 * around KataHex's move, where blue would like to play, red's favourite as if KataHex's move
 * were already played, and for each stone, the move the net likes more for red once that stone is
 * taken away, as if it were not there. They are tried in order of how natural they look, and
 * the first two after which a search rates red losing are the decoys.
 */
export const prepareQuestion = async (board: SurvivalBoard, analyse: Analyse, settings: FindMoveSettings): Promise<Question> => {
    const isEmpty = (move: string) => board.isEmpty(move);
    const here = positionOf(board);
    const root = await analyse(here, settings.visits);
    const chosen = bestChild(root.children, isEmpty);
    const kataHex = chosen?.move ?? bestPriorMove(root.prior, isEmpty);

    if (kataHex === null) {
        throw new Error('the engine has no move to suggest');
    }

    const kataHexRedWin = chosen && chosen.visits > 0 ? 1 - chosen.whiteWin : redWin(root);

    const prior = new Map(ranked(root.prior, isEmpty).map(({ move, p }) => [move, p]));
    const candidates = new Set<string>();
    const add = (move: string | undefined | null) => {
        if (move && move !== kataHex && isEmpty(move)) {
            candidates.add(move);
        }
    };

    ranked(root.prior, isEmpty).slice(0, 8).forEach(({ move }) => add(move));
    around(kataHex, board.size).forEach(add);

    const blue = await analyse({ ...here, toMove: 1 }, 1);

    ranked(blue.prior, isEmpty).slice(0, 3).forEach(({ move }) => add(move));

    const afterKataHex = await analyse({ ...here, red: [...here.red, kataHex] }, 1);

    add(ranked(afterKataHex.prior, move => isEmpty(move) && move !== kataHex)[0]?.move);

    const stones = [...here.red, ...here.blue];
    const withouts = await Promise.all(stones.map(stone =>
        analyse({ ...here, red: here.red.filter(m => m !== stone), blue: here.blue.filter(m => m !== stone) }, 1)));

    stones.forEach((stone, i) => {
        const gained = ranked(withouts[i].prior, move => isEmpty(move) && move !== stone)
            .map(({ move, p }) => ({ move, gain: p - (prior.get(move) ?? 0) }))
            .sort((a, b) => b.gain - a.gain)[0];

        if (gained && gained.gain > 0.05) {
            add(gained.move);
        }
    });

    // Nearest the fight first, KataHex's move or blue's last, where a decoy has to be read out
    // rather than dismissed at a glance; then those the net likes better.
    const last = board.moves.at(-1);
    const near = (move: string) => Math.min(distance(move, kataHex), last ? distance(move, last) : Infinity) <= 2;
    const ordered = [...candidates].sort((a, b) =>
        Number(near(b)) - Number(near(a)) || (prior.get(b) ?? 0) - (prior.get(a) ?? 0));
    const afterEach = ordered.map((move): Position => ({ red: [...here.red, move], blue: here.blue, toMove: 1 }));
    const netSays = await Promise.all(afterEach.map(after => analyse(after, 1)));
    const decoys: Choice[] = [];

    for (let i = 0; i < ordered.length && decoys.length < 2; ++i) {
        if (redWin(netSays[i]) > SURELY_WINS) {
            continue;
        }

        const move = ordered[i];
        const checked = await analyse(afterEach[i], settings.checkVisits);
        const free = (cell: string) => isEmpty(cell) && cell !== move;

        if (redWin(checked) < Math.min(DECOY_RED_WIN, kataHexRedWin - DECOY_GAP)) {
            decoys.push({
                move,
                redWin: redWin(checked),
                answer: bestChild(checked.children, free)?.move ?? bestPriorMove(checked.prior, free) ?? undefined,
            });
        }
    }

    return {
        kataHex: { move: kataHex, redWin: kataHexRedWin },
        decoys,
    };
};

/** The three choices in random order, or fewer when there were not two decoys. */
export const shuffled = <T>(items: T[], random: () => number = Math.random): T[] => {
    const out = [...items];

    for (let i = out.length - 1; i > 0; --i) {
        const j = Math.floor(random() * (i + 1));

        [out[i], out[j]] = [out[j], out[i]];
    }

    return out;
};
