/*
 * Written here, at PlayHex's module path: PlayHex's hexplorerLive.ts is its live sessions
 * (streamed analysis, matches from a position), served by its server and built on its player
 * model. This app has neither. useHexplorer still takes and makes the two shapes below, for
 * the live features, which nothing here calls.
 */
import type { HexplorerState, MarkPlacement, SetupStone } from './hexplorerStudy.js';

export type HexplorerLiveSnapshot = {
    state: HexplorerState;
    scratch: null | {
        stones: SetupStone[];
        marks: MarkPlacement[];
    };
    analysis: null | {
        whiteWin?: number;
        policy?: number[][];
        color: 'black' | 'white';
    };
};

export type HexplorerMatchStartPosition = {
    boardsize: number;
    stones: SetupStone[];
    nextPlayer: 0 | 1;
    swapAllowed: boolean;
};
