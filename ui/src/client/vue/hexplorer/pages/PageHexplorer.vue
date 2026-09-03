<script setup lang="ts">
import { whenever } from '@vueuse/core';
import { computed, ref, useTemplateRef, watch } from 'vue';
import { defineOverlay } from '@overlastic/vue';
import { t } from 'i18next';
import { PlaceStoneTool } from '../tools/PlaceStoneTool';
import { PlaceStonesAlternatelyTool } from '../tools/PlaceStonesAlternatelyTool';
import { RemoveStoneTool } from '../tools/RemoveStoneTool';
import { useHead } from '@unhead/vue';
import { useHexplorer } from '../composables/useHexplorer.js';
import {
    IconAlphabet,
    IconArrowBarLeft,
    IconArrowDownUp,
    IconArrowBarRight,
    IconArrowLeft,
    IconArrowRight,
    IconCheck,
    IconChevronLeft,
    IconChevronRight,
    IconCircle,
    IconCircleFill,
    IconCircleHalf,
    IconDiagram2,
    IconDownload,
    IconFileEarmarkPlus,
    IconFolder2Open,
    IconGraphUp,
    IconHexagonFill,
    IconMagic,
    IconPencilSquare,
    IconSave2,
    IconSquare,
    IconTrash,
    IconTriangle,
    IconX,
    IconEraser,
    IconPercent,
    IconStarFill,
    IconXLg,
} from '../../icons.js';
import { hexGameImporter } from '../../../../shared/app/hex-game-importer/index.js';
import { SourceNotSupportedError } from '../../../../shared/app/hex-game-importer/errors.js';
import EvaluationGraph from '../components/EvaluationGraph.vue';
import MoveTree from '../components/MoveTree.vue';
import ImportFormatsHelpOverlay from '../overlays/ImportFormatsHelpOverlay.vue';
import { PlaceMarkTool } from '../tools/PlaceMarkTool.js';
import { analyzers, defaultAnalyzer } from '../../../../analyzers.js'; // katahex-android: on-device engines
import LiveAnalysisButton from '../components/LiveAnalysisButton.vue';
import { searching, speed } from '../../../../engineSpeed.js';
import { MAX_BOARD_SIZE, MIN_BOARD_SIZE } from '../../../../engine.js';

useHead({
    title: t('hexplorer.title'),
});

const {
    gameView,
    tree,
    currentNodeId,
    actionsStack,
    whiteWin,
    state,
    currentTool,
    analysisLoading,
    evalHistory,
    evalCursorIndex,
    lineProgress,
    goToEvalIndex,
    userGoToNode,
    goToParent,
    goToFirstChild,
    pass,
    canDeleteCurrentNode,
    deleteCurrentNode,
    goSetupMode,
    validateSetup,
    cancelSetup,
    updateAnalysis,
    bindKeys,
    mount,
    importGame,
    resetState,
    createAlternatingTool,
    createRemoveStoneTool,
    createMarkTool,
    exportSgf,
    exportAnalysis,
    importAnalysis,
    currentAnalyzer,
    setAnalyzer,
} = useHexplorer(
    document.location.hash,
    defaultAnalyzer,
);

const newBoardsize = ref(state.value.boardsize);
const labelText = ref('A');

// An import, a loaded file or a new board changes the size under the field, which would
// otherwise still offer the size of the game before it.
watch(() => state.value.boardsize, size => { newBoardsize.value = size; });

/** Throws the tree away for an empty board, asking first when there is something to lose. */
const startNewAnalysis = (): void => {
    if (tree.nodes.value.length > 1 && !window.confirm(t('hexplorer.new_analysis_confirm'))) {
        return;
    }

    void resetState(Math.min(MAX_BOARD_SIZE, Math.max(MIN_BOARD_SIZE, newBoardsize.value)));
};

const playerLabel = (player: 0 | 1): string => player === 0 ? t('game.red') : t('game.blue');

const selectedAnalyzerName = computed({
    get: () => currentAnalyzer.value?.getName() ?? '',
    set: name => {
        const analyzer = analyzers.find(a => a.getName() === name) ?? null;
        setAnalyzer(analyzer);
    },
});

const isAlternatingToolSelected = computed(() => currentTool.value instanceof PlaceStonesAlternatelyTool);
const isRemoveToolSelected = computed(() => currentTool.value instanceof RemoveStoneTool);
const selectedStoneColor = computed(() => currentTool.value instanceof PlaceStoneTool ? currentTool.value.color : null);
const selectedMarkType = computed(() => currentTool.value instanceof PlaceMarkTool ? currentTool.value.markType : null);

const selectLabelTool = (): void => {
    currentTool.value = createMarkTool('label', labelText);
};

/*
 * Player names, editable inline in the settings table.
 */
const playerName = (player: 0 | 1): string =>
    (player === 0 ? state.value.playerBlackName : state.value.playerWhiteName) ?? '';

const setPlayerName = (player: 0 | 1, event: Event): void => {
    const name = (event.target as HTMLInputElement).value || undefined;

    if (player === 0) {
        state.value.playerBlackName = name;
    } else {
        state.value.playerWhiteName = name;
    }
};

const gameViewElement = useTemplateRef('game-view-element');

whenever(gameViewElement, async element => {
    await mount(element);
    await updateAnalysis();
}, {
    once: true,
});

const redBarHeightCss = computed(() => `${100 - (whiteWin.value ?? 0.5) * 100}%`);
const blueBarHeightCss = computed(() => `${(whiteWin.value ?? 0.5) * 100}%`);
const whiteWinAvailable = computed(() => typeof whiteWin.value === 'number');

bindKeys();

const sidebarOpen = ref(true);

/*
 * Import
 */
const importFormatsHelpOverlay = defineOverlay(ImportFormatsHelpOverlay);

const openImportOverlay = () => {
    void importFormatsHelpOverlay({
        onImport: async (source: string) => {
            try {
                const importedGame = await hexGameImporter.import(source);
                await importGame(importedGame);
                return { ok: true } as const;
            } catch (e) {
                if (e instanceof SourceNotSupportedError) {
                    return { ok: false, error: t('hexplorer.format_not_supported') } as const;
                }

                return { ok: false, error: e instanceof Error ? e.message : String(e) } as const;
            }
        },
    });
};

/*
 * Analysis JSON import (whole HexplorerState + move tree)
 */
const analysisFileInput = useTemplateRef<HTMLInputElement>('analysisFileInput');
const analysisImportError = ref<string | null>(null);

const onAnalysisFileSelected = async (event: Event) => {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;

    analysisImportError.value = null;

    try {
        const content = await file.text();

        // A saved analysis is json; anything else the importer knows, an sgf above all, is
        // loaded as a game.
        if (content.trimStart().startsWith('{')) {
            await importAnalysis(content);
        } else {
            await importGame(await hexGameImporter.import(content));
        }
    } catch (e) {
        analysisImportError.value = e instanceof Error ? e.message : String(e);
    } finally {
        input.value = ''; // allow re-selecting the same file
    }
};
</script>

<template>
    <div class="layout bg-body" :class="sidebarOpen ? 'sidebar-open' : 'sidebar-closed'">
        <div class="game">
            <div class="game-inner">
                <div class="left-analysis-bar" :class="{
                    'analysis-loading': analysisLoading,
                    'analysis-unavailable': !whiteWinAvailable,
                }">
                    <div class="advantage advantage-blue bg-primary"></div>
                    <div class="advantage advantage-red bg-danger"></div>
                </div>
                <div ref="game-view-element" class="game-view"></div>
            </div>

            <div class="bottom-menu">
                <div class="btn-group">
                    <button
                        @click="goSetupMode"
                        class="btn"
                        :class="state.setupMode ? 'btn-warning' : 'btn-outline-warning'"
                        :disabled="state.setupMode"
                        :title="$t('hexplorer.setup_mode')"
                    >
                        <IconPencilSquare /> <span v-if="state.setupMode">{{ $t('hexplorer.setup_mode') }}</span>
                    </button>

                    <template v-if="state.setupMode">
                        <button
                            @click="validateSetup()"
                            class="btn btn-outline-success"
                            :title="$t('hexplorer.validate')"
                        ><IconCheck /></button>

                        <button
                            @click="cancelSetup()"
                            class="btn btn-outline-danger"
                            :title="$t('cancel')"
                        ><IconX /></button>
                    </template>
                </div>

                <div class="btn-group" v-if="state.setupMode">
                    <button
                        @click="actionsStack.undo()"
                        class="btn btn-outline-warning"
                        :disabled="!actionsStack.canUndo()"
                        :title="$t('hexplorer.undo_placement')"
                    ><IconArrowLeft /></button>

                    <button
                        @click="actionsStack.redo()"
                        class="btn btn-outline-warning"
                        :disabled="!actionsStack.canRedo()"
                        :title="$t('hexplorer.redo_placement')"
                    ><IconArrowRight /></button>
                </div>

                <div class="btn-group" v-if="state.setupMode">
                    <button
                        @click="currentTool = createAlternatingTool()"
                        class="btn"
                        :class="isAlternatingToolSelected ? 'btn-success' : 'btn-outline-success'"
                        :title="$t('hexplorer.alternating_placement')"
                    ><IconCircleHalf /></button>

                    <button
                        @click="currentTool = new PlaceStoneTool(gameView, 0)"
                        class="btn"
                        :class="selectedStoneColor === 0 ? 'btn-danger' : 'btn-outline-danger'"
                        :title="$t('hexplorer.place_red_stone')"
                    ><IconCircleFill /></button>

                    <button
                        @click="currentTool = new PlaceStoneTool(gameView, 1)"
                        class="btn"
                        :class="selectedStoneColor === 1 ? 'btn-primary' : 'btn-outline-primary'"
                        :title="$t('hexplorer.place_blue_stone')"
                    ><IconCircleFill /></button>

                    <button
                        @click="currentTool = createRemoveStoneTool()"
                        class="btn"
                        :class="isRemoveToolSelected ? 'btn-warning' : 'btn-outline-warning'"
                        :title="$t('hexplorer.remove_stone')"
                    ><IconEraser /></button>
                </div>

                <div class="btn-group" v-if="state.setupMode">
                    <button
                        @click="currentTool = createMarkTool('cross')"
                        class="btn"
                        :class="selectedMarkType === 'cross' ? 'btn-secondary' : 'btn-outline-secondary'"
                        :title="$t('hexplorer.mark_cross')"
                    ><IconXLg /></button>

                    <button
                        @click="currentTool = createMarkTool('triangle')"
                        class="btn"
                        :class="selectedMarkType === 'triangle' ? 'btn-secondary' : 'btn-outline-secondary'"
                        :title="$t('hexplorer.mark_triangle')"
                    ><IconTriangle /></button>

                    <button
                        @click="currentTool = createMarkTool('square')"
                        class="btn"
                        :class="selectedMarkType === 'square' ? 'btn-secondary' : 'btn-outline-secondary'"
                        :title="$t('hexplorer.mark_square')"
                    ><IconSquare /></button>

                    <button
                        @click="currentTool = createMarkTool('circle')"
                        class="btn"
                        :class="selectedMarkType === 'circle' ? 'btn-secondary' : 'btn-outline-secondary'"
                        :title="$t('hexplorer.mark_circle')"
                    ><IconCircle /></button>

                    <button
                        @click="currentTool = createMarkTool('select')"
                        class="btn"
                        :class="selectedMarkType === 'select' ? 'btn-secondary' : 'btn-outline-secondary'"
                        :title="$t('hexplorer.highlight_cell')"
                    ><IconHexagonFill class="text-warning" /></button>
                </div>

                <div class="input-group label-input" v-if="state.setupMode">
                    <input
                        type="text"
                        v-model="labelText"
                        maxlength="3"
                        class="form-control"
                        :title="$t('hexplorer.label_text')"
                    >
                    <button
                        @click="selectLabelTool()"
                        class="btn"
                        :class="selectedMarkType === 'label' ? 'btn-info' : 'btn-outline-secondary'"
                        :title="$t('hexplorer.add_label')"
                    ><IconAlphabet /></button>
                </div>

                <div class="btn-group" v-else>
                    <button
                        @click="goToParent()"
                        class="btn btn-outline-primary"
                        :title="$t('hexplorer.go_back')"
                    ><IconChevronLeft /></button>

                    <button
                        @click="goToFirstChild()"
                        class="btn btn-outline-primary"
                        :title="$t('hexplorer.go_forward')"
                    ><IconChevronRight /></button>

                    <button
                        @click="deleteCurrentNode()"
                        class="btn btn-outline-danger"
                        :disabled="!canDeleteCurrentNode"
                        :title="$t('hexplorer.delete_branch')"
                    ><IconTrash /></button>

                    <button
                        @click="pass()"
                        class="btn btn-outline-warning d-flex align-items-center gap-1"
                        :title="$t('hexplorer.pass_turn')"
                    ><IconArrowDownUp /> {{ $t('pass') }}</button>
                </div>

                <button
                    @click="state.currentPlayer = state.currentPlayer === 0 ? 1 : 0; updateAnalysis()"
                    class="btn d-flex align-items-center gap-2"
                    :class="state.currentPlayer === 0 ? 'btn-danger' : 'btn-primary'"
                    :title="$t('hexplorer.change_current_player')"
                >
                    <IconCircleFill />
                    {{ state.currentPlayer === 0 ? $t('game.red') : $t('game.blue') }}
                </button>

                <LiveAnalysisButton :analyzer="currentAnalyzer" :update="updateAnalysis" />

                <button type="button" class="btn btn-outline-primary open-sidebar-btn" @click="sidebarOpen = !sidebarOpen">
                    <IconArrowBarLeft v-if="!sidebarOpen" />
                    <IconArrowBarRight v-else />
                </button>
            </div>
        </div>

        <div class="sidebar bg-body-tertiary" v-if="sidebarOpen">
            <div class="sidebar-content p-3">
                <div class="table-responsive border rounded bg-body mb-2">
                    <table class="table table-sm table-borderless align-middle mb-0 color-settings">
                        <thead>
                            <tr class="text-muted small">
                                <th></th>
                                <th class="text-center">{{ $t('hexplorer.win_rate') }}</th>
                                <th class="text-center">{{ $t('hexplorer.policy') }}</th>
                                <th class="text-center">{{ $t('hexplorer.autoplay') }}</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr v-for="player in [0, 1] as const" :key="player">
                                <td>
                                    <div class="d-flex align-items-center gap-1" :class="player === 0 ? 'text-danger' : 'text-primary'">
                                        <IconCircleFill />
                                        <input
                                            type="text"
                                            class="player-name-input fw-semibold"
                                            :value="playerName(player)"
                                            :placeholder="player === 0 ? $t('game.red') : $t('game.blue')"
                                            @input="setPlayerName(player, $event)"
                                        >
                                    </div>
                                </td>
                                <td class="text-center">
                                    <input
                                        type="checkbox"
                                        class="form-check-input"
                                        v-model="state.winrateEnabled[player]"
                                        :aria-label="`${$t('hexplorer.win_rate')}, ${playerLabel(player)}`"
                                    >
                                </td>
                                <td class="text-center">
                                    <input
                                        type="checkbox"
                                        class="form-check-input"
                                        v-model="state.policyEnabled[player]"
                                        :aria-label="`${$t('hexplorer.policy')}, ${playerLabel(player)}`"
                                    >
                                </td>
                                <td class="text-center">
                                    <input
                                        type="checkbox"
                                        class="form-check-input"
                                        v-model="state.autoPlay[player]"
                                        :disabled="state.autoPlay[1 - player]"
                                        :aria-label="`${$t('hexplorer.autoplay')}, ${playerLabel(player)}`"
                                    >
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <div class="d-flex gap-3 px-1 mb-3">
                    <div class="form-check form-check-inline mb-0">
                        <input
                            id="policy-show-numbers"
                            type="checkbox"
                            class="form-check-input"
                            v-model="state.policyShowNumbers"
                        >
                        <label class="form-check-label small d-flex align-items-center gap-1" for="policy-show-numbers"><IconPercent /> {{ $t('hexplorer.policy_show_numbers') }}</label>
                    </div>
                    <div class="form-check form-check-inline mb-0">
                        <input
                            id="policy-show-best-mark"
                            type="checkbox"
                            class="form-check-input"
                            v-model="state.policyShowBestMark"
                        >
                        <label class="form-check-label small d-flex align-items-center gap-1" for="policy-show-best-mark"><IconStarFill /> {{ $t('hexplorer.policy_show_best_mark') }}</label>
                    </div>
                </div>

                <h6 class="d-flex align-items-center gap-2 mb-2">
                    <IconGraphUp /> {{ $t('hexplorer.evaluation') }}
                </h6>
                <div class="d-flex align-items-center gap-2 mb-3">
                    <label class="text-muted small text-nowrap" for="analyzer-select">{{ $t('hexplorer.engine') }}</label>
                    <select
                        id="analyzer-select"
                        v-model="selectedAnalyzerName"
                        class="form-select form-select-sm"
                    >
                        <option v-for="a in analyzers" :key="a.getName()" :value="a.getName()">{{ a.getName() }}</option>
                    </select>
                </div>

                <p v-if="speed" class="speed text-body-secondary mb-2" :class="{ stale: !searching }">
                    {{ Math.round(speed.visits) }} visits/s<template v-if="speed.netEvals !== null">,
                    net {{ Math.round(speed.netEvals) }} evals/s<template v-if="speed.batch !== null">
                    at batch {{ Math.round(speed.batch) }}</template></template>
                </p>

                <div class="mb-3">
                    <EvaluationGraph
                        :evalHistory
                        :cursorIndex="evalCursorIndex"
                        class="bg-body"
                        @select="goToEvalIndex"
                    />

                    <p v-if="lineProgress" class="graph-progress text-body-secondary mt-1 mb-0">
                        {{ $t(lineProgress.waiting ? 'hexplorer.filling_graph_waiting' : 'hexplorer.filling_graph', lineProgress) }}
                    </p>
                </div>

                <h6 class="d-flex align-items-center gap-2 mb-2">
                    <IconDiagram2 /> {{ $t('hexplorer.move_history') }}
                </h6>
                <div class="move-tree-wrapper bg-body mb-2">
                    <MoveTree
                        :tree
                        :nodeId="0"
                        :currentNodeId
                        @select="userGoToNode"
                    />
                </div>

                <h6 class="d-flex align-items-center gap-2 mb-2">
                    {{ $t('hexplorer.current_analysis') }}
                </h6>
                <div class="d-flex gap-2 mb-2">
                    <div class="input-group input-group-sm" style="max-width: 9rem" :title="$t('hexplorer.new_analysis')">
                        <input
                            type="number"
                            :min="MIN_BOARD_SIZE"
                            :max="MAX_BOARD_SIZE"
                            v-model.number="newBoardsize"
                            class="form-control"
                            :aria-label="$t('hexplorer.board_size')"
                        >
                        <button
                            class="btn btn-outline-warning"
                            @click="startNewAnalysis()"
                        ><IconFileEarmarkPlus /> {{ $t('hexplorer.new') }}</button>
                    </div>

                    <div class="btn-group btn-group-sm flex-grow-1" role="group">
                        <button
                            type="button"
                            class="btn btn-outline-warning"
                            :title="$t('hexplorer.analysis_json_load')"
                            @click="analysisFileInput?.click()"
                        ><IconFolder2Open /> {{ $t('hexplorer.load') }}</button>
                        <button
                            type="button"
                            class="btn btn-outline-success"
                            :title="$t('hexplorer.analysis_json_save')"
                            @click="exportAnalysis()"
                        ><IconSave2 /> {{ $t('hexplorer.save') }}</button>
                    </div>

                    <button
                        class="btn btn-outline-primary btn-sm"
                        @click="exportSgf()"
                        :title="$t('hexplorer.download_sgf')"
                    ><IconDownload /> SGF</button>
                </div>
                <input
                    ref="analysisFileInput"
                    type="file"
                    accept="application/json,.json,.sgf"
                    class="d-none"
                    @change="onAnalysisFileSelected"
                >
                <div v-if="analysisImportError" class="text-danger small mb-2">{{ analysisImportError }}</div>

                <button
                    class="btn btn-outline-info w-100 mb-2 d-flex align-items-center justify-content-center gap-2"
                    @click="openImportOverlay()"
                ><IconMagic /> {{ $t('hexplorer.import_a_game') }}</button>
            </div>

            <div class="sidebar-footer p-2">
                <p class="credits text-body-secondary text-center">
                    <a href="https://github.com/Phlogistique/katahex-web" target="_blank" rel="noopener">Source</a>, AGPL-3.0.
                    Board and analysis ui from <a href="https://github.com/playhex/playhex" target="_blank" rel="noopener">PlayHex</a>,
                    engine <a href="https://github.com/selinger/katahex" target="_blank" rel="noopener">KataHex</a>,
                    net by <a href="https://github.com/hzyhhzy" target="_blank" rel="noopener">HZY</a>.
                </p>
                <button
                    type="button"
                    class="btn btn-outline-secondary btn-sm w-100 d-flex align-items-center justify-content-center gap-2"
                    @click="sidebarOpen = false"
                ><IconArrowBarRight /> {{ $t('close') }}</button>
            </div>
        </div>
    </div>
</template>

<style lang="stylus" scoped>
.layout
    display flex
    position relative
    height calc(100vh - 3rem) // (fallback if dvh is not supported)
    height calc(100dvh - 3rem) // 3rem = header height

    .game
        width 100%
        height 100%
        display flex
        flex-direction column

        .game-inner
            display flex
            flex 1
            min-height 0

        .left-analysis-bar
            display block
            width 1rem
            height 100%
            opacity 1
            transition-property opacity
            transition-duration 200ms

            &.analysis-loading
                opacity 0.3

            .advantage
                display flex
                width 1rem
                transition-property height
                transition-duration 200ms

            .advantage-red
                height v-bind(redBarHeightCss)

            .advantage-blue
                height v-bind(blueBarHeightCss)

            &.analysis-unavailable
                .advantage
                    background-color #888 !important

        .game-view
            width calc(100vw - 1rem)
            height 100%

    .sidebar
        display none
        height 100%
        border-left 1px solid var(--bs-border-color)

.bottom-menu
    flex-shrink 0
    display flex
    flex-wrap wrap
    justify-content center
    gap 0.25em
    align-items center
    position relative
    padding 0.25rem 3.5rem 0.25rem 0.5rem

    .open-sidebar-btn
        position absolute
        top 0
        bottom 0
        right 0
        margin-inline-end 0.75em

    .label-input
        width auto

        input
            width 3.5rem

sidebarOpen()
    .game
        width 100%

        @media (min-width: 576px)
            width 50%

        @media (min-width: 992px)
            width 55%

        @media (min-width: 1200px)
            width 64%

        @media (min-width: 1400px)
            width 68%

        .game-view
            width calc(100% - 1rem)

    .sidebar
        display flex
        flex-direction column
        position relative
        width 100%

        @media (max-width: 575.5px)
            position absolute
            right 0
            top 0
            bottom 0
            --bs-bg-opacity 0.85

        @media (min-width: 576px)
            width 50%

        @media (min-width: 992px)
            width 45%

        @media (min-width: 1200px)
            width 36%

        @media (min-width: 1400px)
            width 32%

.layout.sidebar-open
    sidebarOpen()

.sidebar-content
    overflow-y auto
    flex 1 1 auto
    min-height 0
    width 100%

.speed, .graph-progress
    font-size 0.75rem
    font-variant-numeric tabular-nums

.speed.stale
    opacity 0.5

.credits
    font-size 0.7rem
    line-height 1.25
    margin-bottom 0.5rem

.sidebar-footer
    flex-shrink 0
    border-top 1px solid var(--bs-border-color)

.move-tree-wrapper
    max-height 12rem
    min-height 5rem
    overflow auto
    border 1px solid var(--bs-border-color)
    border-radius var(--bs-border-radius)
    padding 0.5rem

.color-settings
    th, td
        padding 0.15rem 0.5rem

    th:not(:first-child), td:not(:first-child)
        width 1%
        white-space nowrap

    .player-name-input
        width 100%
        min-width 3rem
        border none
        background transparent
        padding 0
        color inherit
        outline none

        &::placeholder
            color inherit
            opacity 0.65

        &:hover, &:focus
            text-decoration underline dotted

textarea
    font-size 0.8em
</style>
