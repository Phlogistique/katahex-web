<script setup lang="ts">
/*
 * Find KataHex's move, over the analysis board: see findMove.ts for the rules.
 *
 * The analysis board is suspended while this is open (PageHexplorer does it), so every search
 * the engine runs is this game's.
 *
 * Written for this app; not part of PlayHex.
 */
import { computed, onBeforeUnmount, onMounted, reactive, ref, useTemplateRef } from 'vue';
import { storeToRefs } from 'pinia';
import { CircleMark, GameView, PlayingGameFacade, TextMark } from '@playhex/pixi-board';
import { parseMove, type HexMove, type Move } from '@playhex/move-notation';
import { engine, netBackend, SearchAborted, type EngineAnalysis } from './engine.js';
import usePlayerLocalSettingsStore from './client/stores/playerLocalSettingsStore.js';
import { PlayerSettingsFacade } from './client/services/board-view-facades/PlayerSettingsFacade.js';
import { bestPriorMove, SurvivalBoard } from './survival.js';
import {
    findMoveKey,
    middlingBlueMove,
    playOpening,
    positionOf,
    prepareQuestion,
    sanitizeFindMoveSettings,
    shuffled,
    type Choice,
    type FindMoveSettings,
    type Position,
    type Question,
} from './findMove.js';

const emit = defineEmits<{
    close: [];
    analyse: [game: { size: number, moves: HexMove[], handover: number, you: 0 | 1 }];
}>();

const { localSettings } = storeToRefs(usePlayerLocalSettingsStore());

type Phase = 'setup' | 'opening' | 'preparing' | 'choosing' | 'answering' | 'over';

const phase = ref<Phase>('setup');

const form = reactive(sanitizeFindMoveSettings(localSettings.value.findMove));

/** The settings of the game being played, fixed when it starts. */
let settings: FindMoveSettings = { ...form };

const onCpu = computed(() => netBackend.value?.backend === 'cpu');

const found = ref(0);
const handover = ref(0);
const status = ref('');
const endReason = ref('');
const error = ref<string | null>(null);
const bestBefore = ref<number | null>(null);
const best = computed(() => localSettings.value.findMoveBest?.[findMoveKey(sanitizeFindMoveSettings(form))] ?? null);

/** Red's winrate after KataHex's move, by its search. */
const redWin = ref<number | null>(null);

const percent = (p: number): string => `${(p * 100).toFixed(p > 0.99 || p < 0.01 ? 1 : 0)}%`;

const boardElement = useTemplateRef<HTMLElement>('board-element');

let board = new SurvivalBoard(settings.boardSize);
let gameView: GameView | null = null;
let facade: PlayingGameFacade | null = null;
let question: Question | null = null;

/** The moves on offer, by the letter they are marked with. */
const choices = ref<Choice[]>([]);

const MARKS = 'find-move-marks';

/** Bumped whenever the game in progress is abandoned, so that a late search does nothing. */
let generation = 0;

class Abandoned extends Error {}

const alive = (gen: number): void => {
    if (gen !== generation) {
        throw new Abandoned();
    }
};

const ask = (gen: number) => async (position: Position, visits: number): Promise<EngineAnalysis> => {
    const input = {
        size: board.size,
        color: position.toMove === 0 ? 'black' as const : 'white' as const,
        black: position.red,
        white: position.blue,
    };

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

const clearMarks = (): void => {
    gameView?.removeEntitiesGroup(MARKS);
};

const mark = (move: string, text: string): void => {
    gameView?.addEntity(new TextMark(text).setCoords(parseMove(move as Move)), MARKS);
};

const circle = (move: string): void => {
    gameView?.addEntity(new CircleMark(0x198754).setCoords(parseMove(move as Move)), MARKS);
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
    new PlayerSettingsFacade(gameView);
    gameView.on('hexClicked', move => void choose(move));

    if (boardElement.value) {
        await gameView.mount(boardElement.value);
    }
};

const fail = (gen: number, e: unknown): void => {
    if (e instanceof Abandoned || gen !== generation) {
        return;
    }

    error.value = e instanceof Error ? e.message : String(e);
    finish('The engine failed.');
};

const start = async (): Promise<void> => {
    settings = sanitizeFindMoveSettings(form);
    Object.assign(form, settings);
    localSettings.value.findMove = { ...settings };

    const gen = ++generation;

    found.value = 0;
    redWin.value = null;
    endReason.value = '';
    error.value = null;
    choices.value = [];
    bestBefore.value = best.value;
    await newBoard();

    try {
        phase.value = 'opening';
        status.value = 'KataHex plays itself an opening…';
        await playOpening(board, ask(gen), play);
        handover.value = board.moves.length;
        await nextQuestion(gen);
    } catch (e) {
        fail(gen, e);
    }
};

/** Searches the position for KataHex's move and its decoys, and offers them; red is to move. */
const nextQuestion = async (gen: number): Promise<void> => {
    for (;;) {
        phase.value = 'preparing';
        status.value = 'KataHex looks for its move, and for moves that lose…';
        question = await prepareQuestion(board, ask(gen), settings);
        redWin.value = question.kataHex.redWin;

        if (question.decoys.length > 0) {
            break;
        }

        // Nothing natural loses here, so there is nothing to find: KataHex plays its move. If red
        // is behind, which is why nothing loses by much more, blue answers with a middling move.
        play(question.kataHex.move);

        if (await answer(gen, question.kataHex.redWin < 0.5)) {
            return;
        }
    }

    choices.value = shuffled([question.kataHex, ...question.decoys]);
    choices.value.forEach((choice, i) => mark(choice.move, 'ABC'[i]));
    phase.value = 'choosing';
    status.value = 'Which one is KataHex\'s move?';
};

/** Blue answers with the net's favourite move, or a middling one. True if the game is over. */
const answer = async (gen: number, middling = false): Promise<boolean> => {
    if (board.winner() === 0) {
        finish('You connected.');
        return true;
    }

    phase.value = 'answering';
    status.value = 'KataHex answers…';

    play(middling
        ? await middlingBlueMove(board, ask(gen))
        : bestPriorMove((await ask(gen)(positionOf(board), 1)).prior, move => board.isEmpty(move))!);

    if (board.winner() === 1) {
        finish('Blue connected.');
        return true;
    }

    return false;
};

const choose = async (move: string): Promise<void> => {
    if (phase.value !== 'choosing' || !question || !choices.value.some(choice => choice.move === move)) {
        return;
    }

    const gen = generation;
    const decoy = question.decoys.find(d => d.move === move);

    clearMarks();
    choices.value = [];
    play(move);

    if (decoy) {
        redWin.value = decoy.redWin;

        if (decoy.answer) {
            play(decoy.answer);
        }

        circle(question.kataHex.move);
        finish(`That move loses: KataHex rates red ${percent(decoy.redWin)} after it. `
            + (decoy.answer === question.kataHex.move
                ? 'Blue answers on the circled cell, which was KataHex\'s move.'
                : 'Blue answers as shown, and KataHex\'s move is circled.'));
        return;
    }

    found.value++;

    try {
        if (!await answer(gen)) {
            await nextQuestion(gen);
        }
    } catch (e) {
        fail(gen, e);
    }
};

const finish = (reason: string): void => {
    phase.value = 'over';
    endReason.value = reason;
    status.value = '';

    const key = findMoveKey(settings);

    if (found.value > (localSettings.value.findMoveBest?.[key] ?? 0)) {
        localSettings.value.findMoveBest = { ...localSettings.value.findMoveBest, [key]: found.value };
    }
};

const abandon = (): void => {
    ++generation;
    engine.abortPending();
};

const giveUp = (): void => {
    const was = phase.value;

    abandon();

    if (was === 'opening') {
        phase.value = 'setup';
        return;
    }

    clearMarks();

    if (was === 'choosing' && question) {
        circle(question.kataHex.move);
    }

    finish('You gave up.');
};

const backToSettings = (): void => {
    abandon();
    phase.value = 'setup';
};

const analyse = (): void => {
    abandon();
    emit('analyse', { size: board.size, moves: [...board.moves] as HexMove[], handover: handover.value, you: 0 });
};

const close = (): void => {
    abandon();
    emit('close');
};

onMounted(newBoard);

onBeforeUnmount(() => {
    abandon();
    destroyBoard();
});

const busy = computed(() => phase.value === 'opening' || phase.value === 'preparing' || phase.value === 'answering');
</script>

<template>
    <div class="find-move bg-body">
        <div class="top-bar border-bottom">
            <strong class="title">Find KataHex's move</strong>
            <span class="info small text-body-secondary">
                <template v-if="phase !== 'setup'">
                    you play <strong class="text-danger">red</strong> ·
                    found <strong class="text-body found">{{ found }}</strong>
                    <template v-if="bestBefore !== null"> · best {{ bestBefore }}</template>
                </template>
            </span>
            <button type="button" class="btn-close" aria-label="Back to the analysis board" @click="close()"></button>
        </div>

        <div class="game-inner">
            <div ref="board-element" class="game-view"></div>

            <div v-if="phase === 'setup'" class="panel">
                <div class="card shadow">
                    <div class="card-body">
                        <p class="small">
                            You play red against KataHex. Each turn, three moves are marked A, B and C:
                            one is the move KataHex's search plays, the other two look natural but lose.
                            Find KataHex's move as many times in a row as you can.
                        </p>
                        <div v-if="onCpu" class="alert alert-warning small py-2">
                            This device's GPU could not run KataHex, so the net runs on its CPU, many
                            times slower, and each turn takes a search per move checked.
                            <button
                                v-if="form.visits > 1 || form.checkVisits > 1"
                                type="button"
                                class="btn btn-sm btn-warning mt-1 d-block"
                                @click="form.visits = 1; form.checkVisits = 1"
                            >Use 1 visit</button>
                        </div>
                        <form class="settings" @submit.prevent="start()">
                            <label>Board size <input v-model.number="form.boardSize" type="number" min="5" max="19" class="form-control form-control-sm"></label>
                            <label>KataHex's visits <input v-model.number="form.visits" type="number" min="1" class="form-control form-control-sm"></label>
                            <label>Visits to check a decoy loses <input v-model.number="form.checkVisits" type="number" min="1" class="form-control form-control-sm"></label>
                            <div class="d-flex gap-2 align-items-center mt-2">
                                <button type="submit" class="btn btn-success start-btn">Start</button>
                                <span v-if="best !== null" class="small text-body-secondary ms-auto">best with these settings: {{ best }}</span>
                            </div>
                        </form>
                    </div>
                </div>
            </div>

            <div v-else-if="phase === 'over'" class="panel panel-bottom">
                <div class="card shadow result">
                    <div class="card-body">
                        <h2 class="h5">You found {{ found }} move{{ found === 1 ? '' : 's' }}</h2>
                        <p class="mb-1">{{ endReason }}</p>
                        <p v-if="error" class="text-danger small mb-1">{{ error }}</p>
                        <p v-if="bestBefore !== null && found > bestBefore" class="small text-success mb-1">New best, up from {{ bestBefore }}.</p>
                        <div class="d-flex flex-wrap gap-2 mt-2">
                            <button type="button" class="btn btn-success new-game-btn" @click="start()">New game</button>
                            <button type="button" class="btn btn-outline-secondary" @click="backToSettings()">Settings</button>
                            <button type="button" class="btn btn-outline-primary" @click="analyse()">Analyse this game</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="bottom-bar border-top d-flex align-items-center gap-2">
            <span class="status me-auto small">
                <span v-if="busy" class="spinner-border spinner-border-sm me-1" aria-hidden="true"></span>
                {{ status }}
            </span>
            <strong v-if="redWin !== null" class="small text-danger" title="Red's winrate, by KataHex">{{ percent(redWin) }}</strong>
            <button
                v-if="phase !== 'setup' && phase !== 'over'"
                type="button"
                class="btn btn-sm btn-outline-danger"
                @click="giveUp()"
            >{{ phase === 'opening' ? 'Stop' : 'Give up' }}</button>
        </div>
    </div>
</template>

<style lang="stylus" scoped>
.find-move
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
    min-height 2.75rem
    flex-shrink 0

    .title
        white-space nowrap

        @media (max-width: 420px)
            display none

    .info
        flex 1 1 auto
        min-width 0
        white-space nowrap
        overflow hidden
        text-overflow ellipsis
        text-align right

.game-inner
    display flex
    flex 1
    min-height 0
    position relative

.game-view
    width 100%
    height 100%

.panel
    position absolute
    inset 0
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

    > div
        width 100%

.bottom-bar
    flex-shrink 0
    padding 0.35rem 0.75rem
    min-height 2.75rem

    .status
        flex 1 1 auto
        min-width 0
        white-space nowrap
        overflow hidden
        text-overflow ellipsis

    .status ~ *
        flex-shrink 0
        white-space nowrap
</style>
