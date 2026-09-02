import { coordsToMove } from './shared/move-notation/move-notation.js';
import { improvedPolicy } from './improvedPolicy.js';
import type { AnalysisInput, AnalysisOutput } from './shared/app/hexplorer.js';

/** What the search found about one candidate move, which is an analysis of its position. */
export type ChildAnalysis = { move: string; whiteWin: number; visits: number };

/** An analysis plus how much search it is worth, which the live search reports as it grows. */
export type EngineAnalysis = AnalysisOutput & { visits: number; children: ChildAnalysis[] };

type Native = {
    start(boardSize: number): void;
    query(json: string): void;
    running(): boolean;
    stop(): void;
    save(filename: string, mimeType: string, content: string): void;
};

declare global {
    interface Window {
        Native?: Native;
        onEngineLine?: (line: string) => void;
        onEngineLog?: (line: string) => void;
        onEngineProgress?: (line: string) => void;
    }
}

type Reply = {
    id: string;
    error?: string;
    action?: string;
    noResults?: boolean;
    rootInfo?: { currentPlayer: 'B' | 'W'; winrate: number; visits: number; utility: number; weight: number };
    moveInfos?: { move: string; utility: number; winrate: number; visits: number }[];
    policy?: number[];
};

type Pending = {
    json: string;
    resolve: (analysis: EngineAnalysis) => void;
    reject: (error: Error) => void;
};

type Live = {
    id: string;
    json: string;
    report: (analysis: EngineAnalysis) => void;

    /** Who asked for it, so that only they can end it. */
    owner: unknown;
};

/** What an unbounded search asks for: enough that it is ended by terminating it. */
const ENDLESS = 100000000;

/** Row-major, one row per y, matching hexplorer's policy[row][col] and the engine's own order. */
const grid = (size: number, valueAt: (row: number, col: number) => number): number[][] =>
    Array.from({ length: size }, (_, row) =>
        Array.from({ length: size }, (_, col) => valueAt(row, col)));

/**
 * The KataHex analysis engine, as a request/response service plus one live search.
 *
 * The engine sizes its neural net buffers from the board size it was started with, so a size
 * change is a restart; queries submitted meanwhile are held and sent once it is ready again.
 *
 * Several queries can be in flight at once, which is what makes reviewing a whole line bearable:
 * a lone evaluation on the phone gpu costs about a second, twenty at once cost 25ms each.
 */
class Engine {
    /** Called with engine progress and errors; the ui shows it while the engine is not usable. */
    onStatus: (text: string, ready: boolean) => void = () => {};

    /** False in a browser preview, where there is no engine to talk to. */
    get available(): boolean {
        return !!window.Native;
    }

    private size = 0;
    private ready = false;
    private seq = 0;
    private pending = new Map<string, Pending>();
    private live: Live | null = null;

    constructor() {
        window.onEngineLine = line => this.receive(line);
        window.onEngineLog = line => this.log(line);
        // The engine in the page spends about ten seconds downloading its net and
        // handing it to the GPU, and says nothing on its own until it is up.
        window.onEngineProgress = line => { if (!this.ready) this.onStatus(line, false); };
    }

    /**
     * Analyses one position, once. `maxVisits` of 1 returns the raw network policy, more runs a
     * search and reports each candidate's share of it.
     */
    analyse(input: AnalysisInput, maxVisits: number): Promise<EngineAnalysis> {
        if (!window.Native) {
            return Promise.reject(new Error('no engine (browser preview)'));
        }

        this.ensureSize(input.size);

        const id = 'q' + ++this.seq;
        const json = this.query(id, input, { maxVisits });

        return new Promise((resolve, reject) => {
            this.pending.set(id, { json, resolve, reject });
            this.send(json);
        });
    }

    /**
     * Searches one position, reporting the search as it grows, until it reaches `maxVisits`,
     * another position is watched, or unwatch() is called. Only one such search runs at a time.
     */
    watch(input: AnalysisInput, maxVisits: number, report: (analysis: EngineAnalysis) => void, owner: unknown): void {
        if (!window.Native) {
            return;
        }

        this.unwatch();
        this.ensureSize(input.size);

        const id = 'live' + ++this.seq;
        const json = this.query(id, input, {
            maxVisits: Number.isFinite(maxVisits) ? maxVisits : ENDLESS,
            reportDuringSearchEvery: 0.5,
        });

        this.live = { id, json, report, owner };
        this.send(json);
    }

    /**
     * Ends the live search. Its last partial result stays on screen. Passing an owner ends it
     * only if it is theirs: an analyzer being switched away from is told to stop after the one
     * replacing it has started searching, and must not end that search instead of its own.
     */
    unwatch(owner?: unknown): void {
        if (this.live && owner !== undefined && this.live.owner !== owner) {
            return;
        }

        if (!this.live || !window.Native) {
            this.live = null;
            return;
        }

        // Terminating by id rather than terminate_all, which would also drop the one-shot
        // queries filling the evaluation graph.
        window.Native.query(JSON.stringify({ id: 'x', action: 'terminate', terminateId: this.live.id }));
        this.live = null;
    }

    private query(id: string, input: AnalysisInput, options: object): string {
        return JSON.stringify({
            id,
            boardXSize: input.size,
            boardYSize: input.size,
            initialStones: [
                ...input.black.map(move => ['B', move]),
                ...input.white.map(move => ['W', move]),
            ],
            initialPlayer: input.color === 'black' ? 'B' : 'W',
            moves: [],
            rules: 'tromp-taylor',
            komi: 0,
            analyzeTurns: [0],
            // The policy is what the overlay is drawn from, and the docs say asking for it
            // costs nothing measurable.
            includePolicy: true,
            ...options,
        });
    }

    private send(json: string): void {
        if (this.ready) {
            window.Native!.query(json);
        }
    }

    private ensureSize(size: number): void {
        if (size === this.size) {
            return;
        }

        this.size = size;
        this.ready = false;
        this.onStatus(`starting engine for ${size}×${size}…`, false);
        window.Native!.start(size);
    }

    private log(line: string): void {
        if (line.includes('ready to begin handling requests')) {
            this.ready = true;
            this.onStatus('engine ready', true);

            for (const { json } of this.pending.values()) {
                this.send(json);
            }

            if (this.live) {
                this.send(this.live.json);
            }
        } else if (/Tuning|tuning/.test(line)) {
            // A board size with no cached tuning takes about twenty minutes to tune, and the
            // engine says nothing else while it does, so pass its progress through.
            this.onStatus(line.slice(0, 120), false);
        } else if (/error|Error|terminating/.test(line)) {
            this.onStatus(line.slice(0, 120), false);
        }
    }

    private receive(line: string): void {
        let reply: Reply;

        try {
            reply = JSON.parse(line);
        } catch {
            return;
        }

        if (reply.action) {
            return;
        }

        if (this.live && reply.id === this.live.id) {
            // A terminated search that had not visited anything reports no results at all.
            if (reply.rootInfo) {
                this.live.report(this.toAnalysis(reply));
            }
            return;
        }

        const request = this.pending.get(reply.id);

        if (!request) {
            return;
        }

        this.pending.delete(reply.id);

        if (reply.error || !reply.rootInfo) {
            request.reject(new Error(reply.error ?? 'engine returned no analysis'));
            return;
        }

        request.resolve(this.toAnalysis(reply));
    }

    private toAnalysis(reply: Reply): EngineAnalysis {
        const root = reply.rootInfo!;

        // Winrates are reported for the side to move (reportAnalysisWinratesAs = SIDETOMOVE),
        // the root's and each candidate's alike.
        const forWhite = (winrate: number) => root.currentPlayer === 'W' ? winrate : 1 - winrate;

        return {
            whiteWin: forWhite(root.winrate),
            policy: this.toPolicyGrid(reply),
            visits: root.visits,
            children: (reply.moveInfos ?? []).map(info => ({
                move: info.move,
                whiteWin: forWhite(info.winrate),
                visits: info.visits,
            })),
        };
    }

    /**
     * What the overlay draws on each cell: how much the search wants each move played, at the
     * resolution of the network's policy rather than of the visit counts (see improvedPolicy).
     * Not the winrate, which in hex saturates -- whoever is winning, every candidate reads
     * ~100 or ~0.
     */
    private toPolicyGrid(reply: Reply): number[][] {
        const size = this.size;

        if (!reply.policy) {
            return grid(size, () => 0);
        }

        // Row-major, negative on cells that cannot be played.
        const prior = grid(size, (row, col) => reply.policy![row * size + col]);

        // The engine names a cell letter(x) + (y+1), which is exactly coordsToMove's notation.
        const searched = new Map(reply.moveInfos?.map(info => [info.move, info]));
        const root = reply.rootInfo!;

        return improvedPolicy(
            prior,
            (row, col) => searched.get(coordsToMove({ row, col })),
            root.utility,
            root.weight,
        );
    }
}

export const engine = new Engine();
