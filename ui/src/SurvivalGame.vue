<script setup lang="ts">
/*
 * Survival against KataHex, over the analysis board: see survival.ts for the rules.
 *
 * KataHex plays itself until red is clearly winning, then hands red to the player, on a clock,
 * and answers as blue. The score is how many moves the player makes before KataHex rates red
 * under the losing threshold.
 *
 * The analysis board is suspended while this is open (PageHexplorer does it), so every search
 * the engine runs is this game's, one at a time.
 *
 * Written for this app; not part of PlayHex.
 */
import { computed, onBeforeUnmount, onMounted, reactive, ref, useTemplateRef } from 'vue';
import { storeToRefs } from 'pinia';
import { GameView, PlayingGameFacade } from '@playhex/pixi-board';
import type { HexMove } from '@playhex/move-notation';
import { engine, netBackend, SearchAborted, type EngineAnalysis } from './engine.js';
import { analysisStore } from './analysisStore.js';
import usePlayerLocalSettingsStore from './client/stores/playerLocalSettingsStore.js';
import { PlayerSettingsFacade } from './client/services/board-view-facades/PlayerSettingsFacade.js';
import { formatOdds, logitScale } from './winrateScale.js';
import EvaluationGraph from './client/vue/hexplorer/components/EvaluationGraph.vue';
import {
    bestChild,
    bestPriorMove,
    FischerClock,
    formatClock,
    pickMediocreMove,
    sanitizeSettings,
    settingsKey,
    SurvivalBoard,
    SWAP,
    type Stone,
    type SurvivalSettings,
} from './survival.js';
import { pickFairFirstMove, shouldSwap } from './shared/swap-maps/swapDecision.js';

const emit = defineEmits<{
    close: [];
    analyse: [game: { size: number, moves: HexMove[], handover: number, you: Stone }];
}>();

const { localSettings } = storeToRefs(usePlayerLocalSettingsStore());

type Phase = 'setup' | 'selfplay' | 'player' | 'thinking' | 'over';

const phase = ref<Phase>('setup');

/**
 * A fraction as the percentage the form shows, to a millionth of a percent: on 11x11 the
 * thresholds worth playing at are 99.99% and 0.01%, which rounding to a tenth would turn into
 * 100% and 0%, a game that never starts and one that is never lost.
 */
const toPercent = (fraction: number): number => Math.round(fraction * 1e8) / 1e6;

/** The settings form, in the units it shows: percentages and seconds. */
const toForm = (s: SurvivalSettings) => ({
    boardSize: s.boardSize,
    color: s.color,
    opponentVisits: s.opponentVisits,
    evalVisits: s.evalVisits,
    weakMin: toPercent(s.weakMin),
    weakMax: toPercent(s.weakMax),
    swap: s.swap,
    takeBack: s.takeBack,
    maxSelfPlayMoves: s.maxSelfPlayMoves,
    startWinrate: toPercent(s.startWinrate),
    loseWinrate: toPercent(s.loseWinrate),
    timed: s.timed,
    initial: s.initial,
    increment: s.increment,
    cap: s.cap,
});

const fromForm = (f: ReturnType<typeof toForm>): SurvivalSettings => sanitizeSettings({
    ...f,
    weakMin: f.weakMin / 100,
    weakMax: f.weakMax / 100,
    startWinrate: f.startWinrate / 100,
    loseWinrate: f.loseWinrate / 100,
});

const form = reactive(toForm(sanitizeSettings(localSettings.value.survival)));

/** The settings of the game being played, fixed when it starts. */
let settings: SurvivalSettings = fromForm(form);

/** On the CPU a search of a few hundred visits takes minutes; one visit is the net's own judgement. */
const onCpu = computed(() => netBackend.value?.backend === 'cpu');

const useOneVisit = (): void => {
    form.opponentVisits = 1;
    form.evalVisits = 1;
};

const resetForm = (): void => {
    Object.assign(form, toForm(sanitizeSettings({})));
};

/** Blue's winrate after each evaluation, for the graph. */
const evalHistory = ref<number[]>([]);

/** The side the player takes over: red is 0, blue 1. */
const you = ref<Stone>(0);
const colorName = (side: Stone): string => side === 0 ? 'red' : 'blue';

/** Blue's winrate by the last evaluation. */
const whiteWin = ref<number | null>(null);

/** The player's side's winrate by the last evaluation. */
const yourWin = computed(() => whiteWin.value === null ? null : you.value === 1 ? whiteWin.value : 1 - whiteWin.value);

/** Moves the player has made and stayed above the threshold after. */
const survived = ref(0);

/** Moves on the board when the player took over. */
const handover = ref(0);

const clockMs = ref(0);

/** Whether this game got as far as the player's turn. */
const handedOver = ref(false);
const endReason = ref('');
const error = ref<string | null>(null);
const status = ref('');
const bestBefore = ref<number | null>(null);

const best = computed(() => localSettings.value.survivalBest?.[settingsKey(fromForm(form))] ?? null);

const oddsScale = computed(() => !!localSettings.value.hexplorerOddsScale);
const barValue = computed(() => {
    const blue = whiteWin.value ?? 0.5;

    return oddsScale.value ? logitScale(blue) : blue;
});
const redBarHeightCss = computed(() => `${100 - barValue.value * 100}%`);
const blueBarHeightCss = computed(() => `${barValue.value * 100}%`);

/**
 * A winrate as a percentage, with as many decimals as it takes to show two digits of its
 * distance to the nearer edge: 0.012%, 99.97%, 55%.
 */
const percent = (p: number): string => {
    // Rounded first: 1 - 0.9999 is 0.00009999999999998899, a digit short in log10.
    const edge = Number((Math.min(p, 1 - p) * 100).toPrecision(12));
    const decimals = edge >= 10 || edge <= 0 ? 0 : Math.min(8, Math.max(0, 1 - Math.floor(Math.log10(edge))));

    return `${(p * 100).toFixed(decimals).replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '')}%`;
};

/** A threshold from the form, which is in percent, as the odds the board would show. */
const formOdds = (percentValue: number): string => {
    const p = percentValue / 100;

    return Number.isFinite(p) && p > 0 && p < 1 ? formatOdds(p) : '';
};
/** A winrate, or odds: 999:1 when the side is all but won. */
const formatWin = (p: number | null): string => p === null ? '–' : (oddsScale.value ? formatOdds(p) : percent(p));

const boardElement = useTemplateRef<HTMLElement>('board-element');

let board = new SurvivalBoard(settings.boardSize);
let gameView: GameView | null = null;
let facade: PlayingGameFacade | null = null;
let clock: FischerClock | null = null;
let ticker: ReturnType<typeof setInterval> | null = null;

/**
 * Bumped whenever the game in progress is abandoned. Every step of a game checks it after each
 * await, so that a search answering for a game nobody is playing any more does nothing.
 */
let generation = 0;

class Abandoned extends Error {}

/** Self-play that cannot get the player's side to the threshold: a setting to change, not a failure. */
class SelfPlayStuck extends Error {}

const alive = (gen: number): void => {
    if (gen !== generation) {
        throw new Abandoned();
    }
};

const isEmpty = (move: string): boolean => board.isEmpty(move);

/** Searches the position on the board, to `visits`. */
const ask = async (gen: number, visits: number): Promise<EngineAnalysis> => {
    const input = {
        size: board.size,
        color: board.toMove === 0 ? 'black' as const : 'white' as const,
        black: board.stones(0),
        white: board.stones(1),
    };

    // A search can be taken back by the engine restarting or the analysis board waking up;
    // ask again, unless the game itself is what went away.
    for (let attempt = 0; ; ++attempt) {
        try {
            const analysis = await engine.analyse(input, visits);

            // Into the analysis board's store, whether or not the game is still on: loaded into
            // the board, a game is then drawn on its graph from the searches it was played
            // with, wherever they were as deep as the board's own analyzer asks for, rather than
            // searched all over again.
            analysisStore.write(input, analysis);

            alive(gen);
            return analysis;
        } catch (e) {
            alive(gen);

            if (!(e instanceof SearchAborted) || attempt >= 5) {
                throw e;
            }
        }
    }
};

const play = (move: string): void => {
    board.play(move);
    facade?.addMove(move as HexMove);
};

/** The move a search would play, falling back on the network's favourite. */
const searchedMove = (analysis: EngineAnalysis): string => {
    const move = bestChild(analysis.children, isEmpty)?.move ?? bestPriorMove(analysis.prior, isEmpty);

    if (move === null) {
        throw new Error('the engine has no move to suggest');
    }

    return move;
};

const stopTicker = (): void => {
    if (ticker !== null) {
        clearInterval(ticker);
        ticker = null;
    }
};

const destroyBoard = (): void => {
    facade?.destroy();
    gameView?.destroy();
    facade = null;
    gameView = null;
};

const newBoard = async (): Promise<void> => {
    destroyBoard();

    board = new SurvivalBoard(settings.boardSize);
    gameView = new GameView(settings.boardSize);
    facade = new PlayingGameFacade(gameView, false, [], true);

    // The analysis board's look: coordinates, and a flat board turned upright on a phone.
    new PlayerSettingsFacade(gameView);

    gameView.on('hexClicked', move => void onClick(move));

    if (boardElement.value) {
        await gameView.mount(boardElement.value);
    }
};

/**
 * The position the player was handed, to play it again: the moves that led to it, whose side
 * they took, and what had been rated on the way.
 */
type Handover = { moves: string[], you: Stone, whiteWin: number | null, evalHistory: number[] };

const lastHandover = ref<Handover | null>(null);

/** Clears what a game shows, for the next one, and returns the new game's generation. */
const resetGame = async (): Promise<number> => {
    const gen = ++generation;

    stopTicker();
    clock = null;
    turns.length = 0;
    canTakeBack.value = false;
    takeBackOn.value = settings.takeBack;
    timed.value = settings.timed;
    evalHistory.value = [];
    whiteWin.value = null;
    survived.value = 0;
    handover.value = 0;
    handedOver.value = false;
    endReason.value = '';
    error.value = null;
    bestBefore.value = best.value;

    await newBoard();

    return gen;
};

/** A new game: the settings from the form, a side drawn if it is to be, and self-play from scratch. */
const start = async (): Promise<void> => {
    settings = fromForm(form);
    Object.assign(form, toForm(settings));
    localSettings.value.survival = { ...settings };
    lastHandover.value = null;
    you.value = settings.color === 'red' ? 0 : settings.color === 'blue' ? 1 : (Math.random() < 0.5 ? 0 : 1);

    const gen = await resetGame();

    try {
        await selfPlay(gen);
    } catch (e) {
        fail(gen, e);
    }
};

const fail = (gen: number, e: unknown): void => {
    if (e instanceof Abandoned || gen !== generation) {
        return;
    }

    stopTicker();
    clock?.stop();
    error.value = e instanceof Error ? e.message : String(e);

    if (e instanceof SelfPlayStuck) {
        phase.value = 'setup';
        return;
    }

    phase.value = 'over';
    endReason.value = 'The engine failed.';
};

/**
 * The opening of a game with the swap rule, from PlayHex's swap maps (computed with this same
 * net): red opens on a fair cell, one KataHex rates near even, and blue swaps it or not at
 * random, the likelier the stronger the move, as PlayHex's own AIs do.
 */
const openWithSwap = (): boolean => {
    const first = pickFairFirstMove(board.size);

    if (first === null) {
        return false;
    }

    play(first);

    if (shouldSwap(board.size, first)) {
        play(SWAP);
    }

    return true;
};

/**
 * The player's side at its best against the other playing middling moves, until the player's
 * side is clearly winning. Its search is the evaluation's: one search both rates the position
 * and picks its move.
 */
const selfPlay = async (gen: number): Promise<void> => {
    phase.value = 'selfplay';

    const good = you.value;
    const target = `${colorName(good)} at ${percent(settings.startWinrate)}`;

    for (;;) {
        if (board.winner() !== null || board.full()) {
            throw new SelfPlayStuck(`Sorry, the game ended before self-play got ${target}. Change your settings.`);
        }

        if (board.moves.length === 0 && settings.swap && openWithSwap()) {
            continue;
        }

        status.value = `KataHex plays itself, move ${board.moves.length + 1}…`;

        if (board.toMove !== good) {
            const policy = await ask(gen, 1);
            const move = pickMediocreMove(policy.prior, isEmpty, settings.weakMin, settings.weakMax);

            if (move === null) {
                continue;
            }

            play(move);
            continue;
        }

        const rated = await ask(gen, settings.evalVisits);

        whiteWin.value = rated.whiteWin ?? 0.5;
        evalHistory.value = [...evalHistory.value, whiteWin.value];

        // At the threshold, or out of self-play moves: either way it is the player's game now.
        if (yourWin.value! >= settings.startWinrate || board.moves.length >= settings.maxSelfPlayMoves) {
            handOver();
            return;
        }

        play(searchedMove(rated));
    }
};

/**
 * The same game again: the position the last one was handed over at, same side, same settings,
 * a full clock. KataHex answers much as it did, so a line that lost can be tried another way.
 */
const again = async (): Promise<void> => {
    const last = lastHandover.value;

    if (!last) {
        await start();
        return;
    }

    you.value = last.you;
    await resetGame();

    for (const move of last.moves) {
        play(move);
    }

    whiteWin.value = last.whiteWin;
    evalHistory.value = [...last.evalHistory];
    handOver();
};

const handOver = (): void => {
    lastHandover.value = { moves: [...board.moves], you: you.value, whiteWin: whiteWin.value, evalHistory: [...evalHistory.value] };
    handover.value = board.moves.length;
    handedOver.value = true;
    clock = new FischerClock(settings.initial, settings.increment, settings.cap);
    clockMs.value = clock.left();
    yourTurn();
};

/**
 * Where each of the player's turns started, to take a move back to: the moves on the board then,
 * what was rated and the score. Not the time: a take back gives none back, the clock runs on.
 */
type Turn = { moves: number, whiteWin: number | null, evals: number, survived: number };

const turns: Turn[] = [];

/**
 * The turn a take back returns to: the last one before the position on the board. During a
 * turn that is the one before it; once a move has lost, the turn it was played in.
 */
const takeBackTarget = (): Turn | undefined => {
    for (let i = turns.length - 1; i >= 0; --i) {
        if (turns[i].moves < board.moves.length) {
            return turns[i];
        }
    }

    return undefined;
};

/** Reactive stand-in for whether takeBackTarget() has a turn: `turns` and `board` are not reactive. */
const canTakeBack = ref(false);

/** The game's settings allow take backs; `settings` is not reactive either. */
const takeBackOn = ref(false);

/** The game is on a clock; without one the player has all the time in the world. */
const timed = ref(true);

/** Out of time, and so out of the game: a take back would give no time back to play it with. */
const flagged = (): boolean => timed.value && !!clock?.flagged();

const syncTakeBack = (): void => {
    canTakeBack.value = settings.takeBack && (phase.value === 'player' || phase.value === 'over')
        && handedOver.value && !flagged() && takeBackTarget() !== undefined;
};

/** The player's last move and KataHex's answer back, with the score they had; the clock runs on. */
const takeBack = (): void => {
    const target = takeBackTarget();

    if (!settings.takeBack || !target || flagged() || !(phase.value === 'player' || phase.value === 'over')) {
        return;
    }

    ++generation;
    stopTicker();

    while (board.moves.length > target.moves) {
        board.undo();
        facade?.undoLastMove();
    }

    // yourTurn() records this turn again.
    turns.splice(turns.indexOf(target));

    whiteWin.value = target.whiteWin;
    evalHistory.value = evalHistory.value.slice(0, target.evals);
    survived.value = target.survived;
    endReason.value = '';
    yourTurn();
};

const yourTurn = (): void => {
    turns.push({
        moves: board.moves.length,
        whiteWin: whiteWin.value,
        evals: evalHistory.value.length,
        survived: survived.value,
    });
    phase.value = 'player';
    status.value = `Your move, as ${colorName(you.value)}.`;
    stopTicker();
    syncTakeBack();

    if (!timed.value) {
        return;
    }

    clock!.start();
    ticker = setInterval(() => {
        clockMs.value = clock!.left();

        if (clock!.flagged()) {
            finish('Out of time.');
        }
    }, 100);
};

const onClick = async (move: string): Promise<void> => {
    if (phase.value !== 'player' || !board.isEmpty(move) || !clock) {
        return;
    }

    const gen = generation;

    stopTicker();

    if (flagged()) {
        finish('Out of time.');
        return;
    }

    clock.moved();
    clockMs.value = clock.left();
    play(move);

    if (board.winner() === you.value) {
        survived.value++;
        finish('You connected. KataHex never got you under the line.');
        return;
    }

    phase.value = 'thinking';
    syncTakeBack();
    status.value = 'KataHex answers…';

    try {
        // KataHex answers first, and what is rated is the position after its answer, the one
        // the player would move in: losing, they see the move that did it.
        const reply = await ask(gen, settings.opponentVisits);

        play(searchedMove(reply));

        if (board.winner() !== null) {
            finish(`${colorName(board.winner()!)[0].toUpperCase()}${colorName(board.winner()!).slice(1)} connected.`);
            return;
        }

        status.value = 'KataHex rates the position…';

        const rated = await ask(gen, settings.evalVisits);

        whiteWin.value = rated.whiteWin ?? 0.5;
        evalHistory.value = [...evalHistory.value, whiteWin.value];

        if (yourWin.value! < settings.loseWinrate) {
            finish(`After KataHex's answer you are under ${oddsScale.value ? formatOdds(settings.loseWinrate) : percent(settings.loseWinrate)}.`);
            return;
        }

        survived.value++;
        yourTurn();
    } catch (e) {
        fail(gen, e);
    }
};

const finish = (reason: string): void => {
    stopTicker();
    clock?.stop();

    // The ticker may not have run for a while: a browser slows the timers of a page it hides.
    if (clock) {
        clockMs.value = clock.left();
    }

    phase.value = 'over';
    endReason.value = reason;
    status.value = '';
    syncTakeBack();

    const key = settingsKey(settings);
    const previous = localSettings.value.survivalBest?.[key] ?? 0;

    if (survived.value > previous) {
        localSettings.value.survivalBest = { ...localSettings.value.survivalBest, [key]: survived.value };
    }
};

/** Stops whatever the engine is doing for us: the game is being left. */
const abandon = (): void => {
    ++generation;
    stopTicker();
    clock?.stop();
    engine.abortPending();
};

const giveUp = (): void => {
    const selfPlaying = phase.value === 'selfplay';

    abandon();

    if (selfPlaying) {
        phase.value = 'setup';
    } else {
        finish('You gave up.');
    }
};

const backToSettings = (): void => {
    abandon();
    phase.value = 'setup';
};

const analyse = (): void => {
    abandon();
    emit('analyse', { size: board.size, moves: [...board.moves] as HexMove[], handover: handover.value, you: you.value });
};

const close = (): void => {
    abandon();
    emit('close');
};

onMounted(async () => {
    // An empty board under the settings, so the page does not open on a blank space.
    await newBoard();
});

onBeforeUnmount(() => {
    abandon();
    destroyBoard();
});

const playing = computed(() => phase.value === 'player' || phase.value === 'thinking');
const lowTime = computed(() => playing.value && clockMs.value < 5000);
const clockShown = computed(() => timed.value && (playing.value || (phase.value === 'over' && handedOver.value)));
</script>

<template>
    <div class="survival bg-body">
        <div class="top-bar border-bottom">
            <strong class="me-auto">Survive KataHex</strong>
            <span v-if="phase !== 'setup'" class="small text-body-secondary">
                you play <strong :class="you === 0 ? 'text-danger' : 'text-primary'" class="you-color">{{ colorName(you) }}</strong> ·
                survived <strong class="text-body survived">{{ survived }}</strong>
                <template v-if="bestBefore !== null"> · best {{ bestBefore }}</template>
            </span>
            <span
                v-if="clockShown"
                class="clock font-monospace"
                :class="{ 'text-danger': lowTime, 'text-body-secondary': !playing || phase === 'thinking' }"
            >{{ formatClock(clockMs) }}</span>
            <button type="button" class="btn-close" aria-label="Back to the analysis board" @click="close()"></button>
        </div>

        <div class="game-inner">
            <div class="left-analysis-bar" :class="{ 'analysis-unavailable': whiteWin === null }">
                <div class="advantage advantage-blue bg-primary"></div>
                <div class="advantage advantage-red bg-danger"></div>
            </div>
            <div ref="board-element" class="game-view" :class="{ 'your-turn': phase === 'player' }"></div>

            <div v-if="phase === 'setup'" class="panel">
                <div class="card shadow">
                    <div class="card-body">
                        <p class="small">
                            KataHex plays itself from an even opening, your side well and the other badly,
                            until your side is winning. Then it is yours, on a clock, and KataHex plays the
                            other for real. Make as many moves as you can before KataHex rates you under the line.
                        </p>
                        <div v-if="onCpu" class="alert alert-warning small py-2 cpu-note">
                            This device's GPU could not run KataHex, so the net runs on its CPU,
                            many times slower. A search of a few hundred visits can take minutes;
                            at one visit KataHex answers with its net alone, which already plays well.
                            <button
                                v-if="form.opponentVisits > 1 || form.evalVisits > 1"
                                type="button"
                                class="btn btn-sm btn-warning mt-1 d-block"
                                @click="useOneVisit()"
                            >Use 1 visit</button>
                        </div>
                        <p v-if="error" class="text-danger small self-play-error">{{ error }}</p>
                        <form class="settings" @submit.prevent="start()">
                            <label>Board size <input v-model.number="form.boardSize" type="number" min="3" max="19" class="form-control form-control-sm"></label>
                            <label>You play <select v-model="form.color" class="form-select form-select-sm">
                                <option value="random">random</option>
                                <option value="red">red</option>
                                <option value="blue">blue</option>
                            </select></label>
                            <label>KataHex's visits <input v-model.number="form.opponentVisits" type="number" min="1" class="form-control form-control-sm"></label>
                            <label>Rating visits <input v-model.number="form.evalVisits" type="number" min="1" class="form-control form-control-sm"></label>
                            <fieldset>
                                <legend>Your opponent's self-play moves, as % of its best move's policy</legend>
                                <label>from <input v-model.number="form.weakMin" type="number" min="0" max="100" step="any" class="form-control form-control-sm"></label>
                                <label>to <input v-model.number="form.weakMax" type="number" min="0" max="100" step="any" class="form-control form-control-sm"></label>
                            </fieldset>
                            <div class="form-check">
                                <input id="survival-swap" v-model="form.swap" type="checkbox" class="form-check-input">
                                <label for="survival-swap" class="form-check-label">
                                    Open with the swap: a fair first move from
                                    <a href="https://playhex.org/swap-maps" target="_blank" rel="noopener">PlayHex's swap maps</a>,
                                    which blue may take
                                </label>
                            </div>
                            <div class="form-check">
                                <input id="survival-take-back" v-model="form.takeBack" type="checkbox" class="form-check-input">
                                <label for="survival-take-back" class="form-check-label">Allow take backs</label>
                            </div>
                            <fieldset>
                                <legend>Self-play</legend>
                                <label>hands over at your winrate, %<span v-if="oddsScale" class="text-body-secondary"> ({{ formOdds(form.startWinrate) }})</span> <input v-model.number="form.startWinrate" type="number" min="0" max="100" step="any" class="form-control form-control-sm"></label>
                                <label>or after this many moves <input v-model.number="form.maxSelfPlayMoves" type="number" min="0" class="form-control form-control-sm"></label>
                            </fieldset>
                            <fieldset>
                                <legend>You lose when your winrate drops</legend>
                                <label>under, %<span v-if="oddsScale" class="text-body-secondary"> ({{ formOdds(form.loseWinrate) }})</span> <input v-model.number="form.loseWinrate" type="number" min="0" max="100" step="any" class="form-control form-control-sm"></label>
                                <div class="form-check">
                                    <input id="survival-odds" v-model="localSettings.hexplorerOddsScale" type="checkbox" class="form-check-input">
                                    <label for="survival-odds" class="form-check-label">Show winrates as odds (999:1)</label>
                                </div>
                            </fieldset>
                            <fieldset :disabled="!form.timed">
                                <legend class="form-check">
                                    <input id="survival-timed" v-model="form.timed" type="checkbox" class="form-check-input">
                                    <label for="survival-timed" class="form-check-label">Clock, seconds</label>
                                </legend>
                                <label>start <input v-model.number="form.initial" type="number" min="1" step="any" class="form-control form-control-sm"></label>
                                <label>+ a move <input v-model.number="form.increment" type="number" min="0" step="any" class="form-control form-control-sm"></label>
                                <label>at most <input v-model.number="form.cap" type="number" min="0" step="any" class="form-control form-control-sm" title="0 for no cap"></label>
                            </fieldset>
                            <div class="d-flex gap-2 align-items-center mt-2">
                                <button type="submit" class="btn btn-success start-btn">Start</button>
                                <button type="button" class="btn btn-sm btn-outline-secondary" @click="resetForm()">Defaults</button>
                                <span v-if="best !== null" class="small text-body-secondary ms-auto">best with these settings: {{ best }}</span>
                            </div>
                        </form>
                    </div>
                </div>
            </div>

            <div v-else-if="phase === 'over'" class="panel panel-bottom">
                <div class="card shadow result">
                    <div class="card-body">
                        <h2 class="h5">You survived {{ survived }} move{{ survived === 1 ? '' : 's' }}</h2>
                        <p class="mb-1">{{ endReason }}</p>
                        <p v-if="error" class="text-danger small mb-1">{{ error }}</p>
                        <p v-if="bestBefore !== null && survived > bestBefore" class="small text-success mb-1">New best, up from {{ bestBefore }}.</p>
                        <p v-else-if="bestBefore === null && survived > 0" class="small text-body-secondary mb-1">Your first score with these settings.</p>
                        <p v-else-if="bestBefore !== null" class="small text-body-secondary mb-1">Best with these settings: {{ bestBefore }}.</p>
                        <div class="d-flex flex-wrap gap-2 mt-2">
                            <button
                                v-if="canTakeBack"
                                type="button"
                                class="btn btn-success take-back-btn"
                                title="Your last move and KataHex's answer, and play on; the clock gives no time back"
                                @click="takeBack()"
                            >Take back</button>
                            <button
                                v-if="lastHandover"
                                type="button"
                                class="btn btn-success again-btn"
                                title="From the same position, as the same side"
                                @click="again()"
                            >Again</button>
                            <button type="button" class="btn btn-outline-success new-game-btn" @click="start()">New game</button>
                            <button type="button" class="btn btn-outline-secondary" @click="backToSettings()">Settings</button>
                            <button type="button" class="btn btn-outline-primary" @click="analyse()">Analyse this game</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="bottom-bar border-top">
            <EvaluationGraph
                v-if="evalHistory.length"
                :evalHistory
                :cursorIndex="evalHistory.length - 1"
                :scale="oddsScale ? logitScale : undefined"
                :format="oddsScale ? formatOdds : undefined"
                class="graph bg-body"
            />
            <div class="d-flex align-items-center gap-2">
                <span class="status me-auto small">
                    <span v-if="phase === 'selfplay' || phase === 'thinking'" class="spinner-border spinner-border-sm me-1" aria-hidden="true"></span>
                    {{ status }}
                </span>
                <span class="small">you <strong class="your-winrate">{{ formatWin(yourWin) }}</strong></span>
                <button
                    v-if="playing && takeBackOn"
                    type="button"
                    class="btn btn-sm btn-outline-secondary take-back-btn"
                    :disabled="!canTakeBack"
                    @click="takeBack()"
                >Take back</button>
                <button
                    v-if="phase === 'selfplay' || playing"
                    type="button"
                    class="btn btn-sm btn-outline-danger"
                    @click="giveUp()"
                >{{ phase === 'selfplay' ? 'Stop' : 'Give up' }}</button>
            </div>
        </div>
    </div>
</template>

<style lang="stylus" scoped>
.survival
    position fixed
    inset 0
    z-index 1060
    display flex
    flex-direction column
    height 100vh
    height 100dvh

.top-bar
    display flex
    align-items center
    gap 0.75rem
    padding 0.35rem 0.75rem

.clock
    font-size 1.4rem
    min-width 4.5ch
    text-align right

.game-inner
    display flex
    flex 1
    min-height 0
    position relative

.left-analysis-bar
    width 1rem
    height 100%

    .advantage
        width 1rem
        transition-property height
        transition-duration 200ms

    .advantage-red
        height v-bind(redBarHeightCss)

    .advantage-blue
        height v-bind(blueBarHeightCss)

    &.analysis-unavailable .advantage
        background-color #888 !important

.game-view
    width calc(100% - 1rem)
    height 100%

.panel
    position absolute
    inset 0 0 0 1rem
    display flex
    align-items center
    justify-content center
    padding 0.75rem
    overflow auto
    background rgba(0, 0, 0, 0.35)

    .card
        max-width 30rem
        width 100%
        max-height 100%
        overflow auto

    &.panel-bottom
        align-items flex-end
        background none
        pointer-events none

        .card
            pointer-events auto

.settings
    display flex
    flex-wrap wrap
    gap 0.5rem 0.75rem

    label
        display flex
        flex-direction column
        font-size 0.8rem
        flex 1 1 7rem

    .form-check
        width 100%
        font-size 0.8rem

        label
            display inline

    fieldset
        display flex
        flex-wrap wrap
        gap 0.5rem 0.75rem
        width 100%

    legend
        font-size 0.8rem
        font-weight 600
        margin-bottom 0.1rem

    > div
        width 100%

.bottom-bar
    flex-shrink 0
    padding 0.35rem 0.75rem

    .graph
        height 2.5rem
        margin-bottom 0.35rem
</style>
