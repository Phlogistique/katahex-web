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
import { engine, SearchAborted, type EngineAnalysis } from './engine.js';
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
    type SurvivalSettings,
} from './survival.js';

const emit = defineEmits<{
    close: [];
    analyse: [game: { size: number, moves: HexMove[], handover: number }];
}>();

const { localSettings } = storeToRefs(usePlayerLocalSettingsStore());

type Phase = 'setup' | 'selfplay' | 'player' | 'thinking' | 'over';

const phase = ref<Phase>('setup');

/** The settings form, in the units it shows: percentages and seconds. */
const toForm = (s: SurvivalSettings) => ({
    boardSize: s.boardSize,
    opponentVisits: s.opponentVisits,
    evalVisits: s.evalVisits,
    weakMin: Math.round(s.weakMin * 1000) / 10,
    weakMax: Math.round(s.weakMax * 1000) / 10,
    minSelfPlayMoves: s.minSelfPlayMoves,
    startWinrate: Math.round(s.startWinrate * 1000) / 10,
    loseWinrate: Math.round(s.loseWinrate * 1000) / 10,
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

const resetForm = (): void => {
    Object.assign(form, toForm(sanitizeSettings({})));
};

/** Blue's winrate after each evaluation, for the graph. */
const evalHistory = ref<number[]>([]);

/** Red's winrate by the last evaluation. */
const redWin = ref<number | null>(null);

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
    const whiteWin = redWin.value === null ? 0.5 : 1 - redWin.value;

    return oddsScale.value ? logitScale(whiteWin) : whiteWin;
});
const redBarHeightCss = computed(() => `${100 - barValue.value * 100}%`);
const blueBarHeightCss = computed(() => `${barValue.value * 100}%`);

const percent = (p: number): string => `${(p * 100).toFixed(p > 0.99 || p < 0.01 ? 1 : 0)}%`;
const formatRed = (p: number | null): string => p === null ? '–' : (oddsScale.value ? formatOdds(1 - p) : percent(p));

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

const start = async (): Promise<void> => {
    settings = fromForm(form);
    Object.assign(form, toForm(settings));
    localSettings.value.survival = { ...settings };

    const gen = ++generation;

    stopTicker();
    clock = null;
    evalHistory.value = [];
    redWin.value = null;
    survived.value = 0;
    handover.value = 0;
    handedOver.value = false;
    endReason.value = '';
    error.value = null;
    bestBefore.value = best.value;

    await newBoard();

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
    phase.value = 'over';
    endReason.value = 'The engine failed.';
};

/**
 * Red at its best against a blue playing middling moves, until red is clearly winning. Red's
 * search is the evaluation's: one search both rates the position and picks red's move.
 */
const selfPlay = async (gen: number): Promise<void> => {
    phase.value = 'selfplay';

    for (;;) {
        if (board.winner() !== null || board.full()) {
            // Red should have been rated winning long before it won: start over.
            await newBoard();
            evalHistory.value = [];
        }

        status.value = `KataHex plays itself, move ${board.moves.length + 1}…`;

        const rated = await ask(gen, settings.evalVisits);
        const whiteWin = rated.whiteWin ?? 0.5;

        redWin.value = 1 - whiteWin;
        evalHistory.value = [...evalHistory.value, whiteWin];

        if (board.moves.length >= settings.minSelfPlayMoves && redWin.value >= settings.startWinrate) {
            handOver();
            return;
        }

        play(searchedMove(rated));

        status.value = `KataHex plays itself, move ${board.moves.length + 1}…`;

        const policy = await ask(gen, 1);
        const blue = pickMediocreMove(policy.prior, isEmpty, settings.weakMin, settings.weakMax);

        if (blue === null) {
            continue;
        }

        play(blue);
    }
};

const handOver = (): void => {
    handover.value = board.moves.length;
    handedOver.value = true;
    clock = new FischerClock(settings.initial, settings.increment, settings.cap);
    clockMs.value = clock.left();
    yourTurn();
};

const yourTurn = (): void => {
    phase.value = 'player';
    status.value = 'Your move, as red.';
    clock!.start();
    stopTicker();
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

    if (clock.flagged()) {
        finish('Out of time.');
        return;
    }

    clock.moved();
    clockMs.value = clock.left();
    play(move);

    if (board.winner() === 0) {
        survived.value++;
        finish('You connected. KataHex never got you under the line.');
        return;
    }

    phase.value = 'thinking';
    status.value = 'KataHex rates your move…';

    try {
        const rated = await ask(gen, settings.evalVisits);
        const whiteWin = rated.whiteWin ?? 0.5;

        redWin.value = 1 - whiteWin;
        evalHistory.value = [...evalHistory.value, whiteWin];

        if (redWin.value < settings.loseWinrate) {
            finish(`That move took red under ${percent(settings.loseWinrate)}.`);
            return;
        }

        survived.value++;
        status.value = 'KataHex answers…';

        const reply = settings.opponentVisits === settings.evalVisits ? rated : await ask(gen, settings.opponentVisits);

        play(searchedMove(reply));

        if (board.winner() === 1) {
            finish('Blue connected.');
            return;
        }

        yourTurn();
    } catch (e) {
        fail(gen, e);
    }
};

const finish = (reason: string): void => {
    stopTicker();
    clock?.stop();
    phase.value = 'over';
    endReason.value = reason;
    status.value = '';

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
    emit('analyse', { size: board.size, moves: [...board.moves] as HexMove[], handover: handover.value });
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
const clockShown = computed(() => playing.value || (phase.value === 'over' && handedOver.value));
</script>

<template>
    <div class="survival bg-body">
        <div class="top-bar border-bottom">
            <strong class="me-auto">Survive KataHex</strong>
            <span v-if="phase !== 'setup'" class="small text-body-secondary">
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
            <div class="left-analysis-bar" :class="{ 'analysis-unavailable': redWin === null }">
                <div class="advantage advantage-blue bg-primary"></div>
                <div class="advantage advantage-red bg-danger"></div>
            </div>
            <div ref="board-element" class="game-view" :class="{ 'your-turn': phase === 'player' }"></div>

            <div v-if="phase === 'setup'" class="panel">
                <div class="card shadow">
                    <div class="card-body">
                        <p class="small">
                            KataHex plays itself, red well and blue badly, until red is winning.
                            Then red is yours, on a clock, and KataHex plays blue for real.
                            Make as many moves as you can before KataHex rates red under the line.
                        </p>
                        <form class="settings" @submit.prevent="start()">
                            <label>Board size <input v-model.number="form.boardSize" type="number" min="3" max="19" class="form-control form-control-sm"></label>
                            <label>Blue's visits <input v-model.number="form.opponentVisits" type="number" min="1" class="form-control form-control-sm"></label>
                            <label>Rating visits <input v-model.number="form.evalVisits" type="number" min="1" class="form-control form-control-sm"></label>
                            <fieldset>
                                <legend>Blue's self-play moves, as % of its best move's policy</legend>
                                <label>from <input v-model.number="form.weakMin" type="number" min="0" max="100" step="any" class="form-control form-control-sm"></label>
                                <label>to <input v-model.number="form.weakMax" type="number" min="0" max="100" step="any" class="form-control form-control-sm"></label>
                            </fieldset>
                            <fieldset>
                                <legend>You take over red after</legend>
                                <label>at least this many moves <input v-model.number="form.minSelfPlayMoves" type="number" min="0" class="form-control form-control-sm"></label>
                                <label>and at red's winrate, % <input v-model.number="form.startWinrate" type="number" min="0" max="100" step="any" class="form-control form-control-sm"></label>
                            </fieldset>
                            <fieldset>
                                <legend>You lose when red's winrate drops</legend>
                                <label>under, % <input v-model.number="form.loseWinrate" type="number" min="0" max="100" step="any" class="form-control form-control-sm"></label>
                            </fieldset>
                            <fieldset>
                                <legend>Clock, seconds</legend>
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
                            <button type="button" class="btn btn-success again-btn" @click="start()">Again</button>
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
                <span class="small">red <strong class="red-winrate">{{ formatRed(redWin) }}</strong></span>
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
