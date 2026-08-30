// Hex board bookkeeping for the match harness: coordinates, legality, and the
// win condition.
//
// Black connects the top edge to the bottom, white the left to the right --
// Board::checkConnection transposes the board for white. The engine also ends
// the game on a bridge to the edge (`includeJumpConnection`, on whenever
// rules.maxMoves is 0), which this does not implement: the engine is asked
// instead, by watching for a position it reports no moves for.

export type Player = 'B' | 'W';

export const other = (player: Player): Player => (player === 'B' ? 'W' : 'B');

export const moveName = (index: number, size: number): string =>
  String.fromCharCode(97 + (index % size)) + (Math.floor(index / size) + 1);

export const moveIndex = (name: string, size: number): number => {
  const match = /^([a-z])(\d{1,2})$/.exec(name.toLowerCase());
  if (!match) throw new Error(`not a move: ${name}`);
  return (Number(match[2]) - 1) * size + (match[1].charCodeAt(0) - 97);
};

/** The board as a flat array of `size * size`, row-major, from a move list. */
export function stones(moves: string[], size: number): (Player | null)[] {
  const board: (Player | null)[] = new Array(size * size).fill(null);
  moves.forEach((move, turn) => { board[moveIndex(move, size)] = turn % 2 ? 'W' : 'B'; });
  return board;
}

/** The six hex neighbours of (x, y), on the board. */
function neighbours(index: number, size: number): number[] {
  const x = index % size;
  const y = Math.floor(index / size);
  const out: number[] = [];
  for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1], [1, -1], [-1, 1]]) {
    const nx = x + dx;
    const ny = y + dy;
    if (nx >= 0 && nx < size && ny >= 0 && ny < size) out.push(ny * size + nx);
  }
  return out;
}

/**
 * Whether `player` has a solid chain across the board. The engine's own win
 * condition is looser, so a game can be over while this is still false; it is
 * here to cross-check the engine rather than to drive the game.
 */
export function connected(board: (Player | null)[], size: number, player: Player): boolean {
  const startsAt = (index: number) => (player === 'B' ? index < size : index % size === 0);
  const endsAt = (index: number) =>
    player === 'B' ? index >= size * (size - 1) : index % size === size - 1;

  const seen = new Uint8Array(size * size);
  const stack: number[] = [];
  for (let i = 0; i < size * size; i++) {
    if (board[i] === player && startsAt(i)) { stack.push(i); seen[i] = 1; }
  }
  while (stack.length) {
    const index = stack.pop()!;
    if (endsAt(index)) return true;
    for (const next of neighbours(index, size)) {
      if (!seen[next] && board[next] === player) { seen[next] = 1; stack.push(next); }
    }
  }
  return false;
}
