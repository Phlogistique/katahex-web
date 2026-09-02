/*
 * PlayHex's hexworld.ts, without the link builders: those take a HostedGame, which drags in
 * the whole model layer (typeorm, class-transformer) for something this app has no use for.
 */
import type { HexMove } from '../move-notation/hex-move-notation.js';

/**
 * Encodes moves as a Hexworld moves string, 'e6:sf7i8g10j10'.
 * Special moves have their own token: ':s' for swap, ':p' for pass.
 */
export const movesToHexworldString = (moves: HexMove[]): string => moves
    .map(move => {
        if (move === 'swap-pieces') {
            return ':s';
        }
        if (move === 'pass') {
            return ':p';
        }
        return move;
    })
    .join('')
;

/**
 * Parses 14r1c1,d3:sa3... to { size: 14, moves: ["d3", "swap-pieces",  "a3", ...]' }
 *
 * Supports a cursor comma: "11c1,h4e7f9,b9c3" means the full line is h4 e7 f9 b9 c3,
 * with the cursor at f9 (cursorMoveCount = 3). When present, cursorMoveCount is returned.
 */
export const parseHexworldString = (hexworldString: string): { size: number, moves: string[], cursorMoveCount?: number } => {
    const match = hexworldString.match(/^(\d+)(?:x(\d+))?[^,]*,(.*)$/);

    if (!match) {
        throw new Error('Invalid Hexworld string');
    }

    const [, sizeStr, heightStr, movesStr] = match;

    if (heightStr !== undefined && heightStr !== sizeStr) {
        throw new Error('Non-square boards are not supported');
    }

    // Cursor comma: "played_moves,continuation" — join segments to parse uniformly
    let cursorMoveCount: number | undefined;
    let fullMovesStr = movesStr;
    const commaIndex = movesStr.indexOf(',');
    if (commaIndex !== -1) {
        cursorMoveCount = (movesStr.slice(0, commaIndex).match(/([a-z]+\d+)|(:s|:p)/g) ?? []).length;
        fullMovesStr = movesStr.slice(0, commaIndex) + movesStr.slice(commaIndex + 1);
    }

    const moves = fullMovesStr.match(/([a-z]+\d+)|(:s|:p|:r.|:f.)/g);

    if (!moves) {
        throw new Error('Error while parsing moves');
    }

    for (let i = 0; i < moves.length; ++i) {
        if (!moves[i].startsWith(':')) {
            continue;
        }

        if (moves[i] === ':s') {
            moves[i] = 'swap-pieces';
        } else if (moves[i] === ':p') {
            moves[i] = 'pass';
        } else if (moves[i].match(/^:(r|f).$/)) {
            delete moves[i];
        } else {
            throw new Error('Unexpected special move "' + moves[i] + '"');
        }
    }

    return {
        size: parseInt(sizeStr),
        moves: moves.filter(m => m),
        cursorMoveCount,
    };
};
