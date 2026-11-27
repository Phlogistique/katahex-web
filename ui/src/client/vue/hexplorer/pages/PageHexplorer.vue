<script setup lang="ts">
/*
 * Patched here. PlayHex's page also saves and shares studies, streams an analysis live and
 * plays matches from a position, all through its server, which this app has none of: the
 * file tab keeps only the local ways in and out, and the live and play tabs are gone. The
 * bottom bar holds the live analysis button, analysis is on from the start since it is the
 * point of the app, the sidebar starts closed on a phone, where it would cover the board,
 * the page takes the whole height in the absence of PlayHex's header, and the sidebar
 * footer credits where the ui, the engine and the net come from. Two tabs open this app's
 * games over the board, the survival game and finding KataHex's move, which have the engine to
 * themselves while they are open.
 */
import { onKeyDown, useMediaQuery, whenever } from '@vueuse/core';
import { computed, provide, ref, useTemplateRef, watch } from 'vue';
import { storeToRefs } from 'pinia';
import { hexplorerKey, useHexplorer } from '../composables/useHexplorer.js';
import { HEXPLORER_ANALYSIS_STATE_KEY } from '../HexplorerState.js';
import { readHexplorerCopy } from '../services/hexplorerCopy.js';
import {
    IconArrowBarLeft,
    IconArrowBarRight,
    IconDiagram2,
    IconFolder2Open,
    IconQuestionCircle,
    IconRobot,
    IconStopwatch,
    IconBullseye,
} from '../../icons.js';
import HexplorerNavButtons from '../components/HexplorerNavButtons.vue';
import LiveAnalysisButton from '../components/LiveAnalysisButton.vue';
import { logitScale } from '../../../../winrateScale.js';
import AppResizableSidebar from '../../components/AppResizableSidebar.vue';
import HexplorerFileTab from '../components/HexplorerFileTab.vue';
import HexplorerHelpTab from '../components/HexplorerHelpTab.vue';
import HexplorerSetupTools from '../components/HexplorerSetupTools.vue';
import HexplorerTreeTab from '../components/HexplorerTreeTab.vue';
import HexplorerAiTab from '../components/HexplorerAiTab.vue';
import { AnalyzerInterface } from '../analyzers/AnalyzerInterface.js';
import { createKatahexAnalyzers } from '../composables/useAnalysisEngines.js';
import usePlayerLocalSettingsStore from '../../../stores/playerLocalSettingsStore.js';
import type { HexMove } from '@playhex/move-notation';
import SurvivalGame from '../../../../SurvivalGame.vue';
import SurvivalTab from '../../../../SurvivalTab.vue';
import FindMoveGame from '../../../../FindMoveGame.vue';
import FindMoveTab from '../../../../FindMoveTab.vue';
import { setAnalyzersSuspended } from '../../../../analyzers.js';
import { engine } from '../../../../engine.js';

// Available analysis engines, selectable in AI tab. First one is the default. AI is disabled by default.
const analyzers: AnalyzerInterface[] = createKatahexAnalyzers();

const hexplorer = useHexplorer(document.location.hash, analyzers);

provide(hexplorerKey, hexplorer);

const {
    whiteWin,
    state,
    analysisLoading,
    aiEnabled,
    currentAnalyzer,
    updateAnalysis,
    bindKeys,
    mount,
    importAnalysis,
    restoreState,
    exportAnalysis,
    importGame,
} = hexplorer;

aiEnabled.value = true;

const gameViewElement = useTemplateRef('game-view-element');

whenever(gameViewElement, async element => {
    await mount(element);
    await importAnalysisFromHistoryState();
    await importAnalysisFromCopy();
    await updateAnalysis();
}, {
    once: true,
});

// In log-odds when the AI tab's odds option is on: a winrate bar sits flat against its edge
// as soon as one side is ahead, see winrateScale.ts.
const barValue = computed(() => {
    const value = whiteWin.value ?? 0.5;

    return localSettings.value.hexplorerOddsScale ? logitScale(value) : value;
});
const redBarHeightCss = computed(() => `${100 - barValue.value * 100}%`);
const blueBarHeightCss = computed(() => `${barValue.value * 100}%`);
const whiteWinAvailable = computed(() => typeof whiteWin.value === 'number');

bindKeys();

/**
 * On small screens, sidebar covers the board: main setup tools are also under it.
 */
const isLargeScreen = useMediaQuery('(min-width: 576px)');

const sidebarOpen = ref(isLargeScreen.value);

type HexplorerTab = 'file' | 'tree' | 'ai' | 'survival' | 'find-move' | 'help';

const { localSettings } = storeToRefs(usePlayerLocalSettingsStore());

const currentTab = ref<HexplorerTab>(localSettings.value.hexplorerHelpDismissed ? 'tree' : 'help');

const navPanels = computed(() => [
    { key: 'file' as const, icon: IconFolder2Open, labelKey: 'hexplorer.tab_file' },
    { key: 'tree' as const, icon: IconDiagram2, labelKey: 'hexplorer.tab_tree' },
    {
        key: 'ai' as const,
        icon: IconRobot,
        labelKey: 'hexplorer.tab_ai',
        badge: aiEnabled.value ? { variant: 'success', titleKey: 'hexplorer.ai_enabled' } : null,
    },
    { key: 'survival' as const, icon: IconStopwatch, labelKey: 'hexplorer.tab_survival' },
    { key: 'find-move' as const, icon: IconBullseye, labelKey: 'hexplorer.tab_find_move' },
    { key: 'help' as const, icon: IconQuestionCircle, labelKey: 'hexplorer.tab_help' },
]);

type Game = 'survival' | 'find-move';

/*
 * The game open over the board, if any, which the url names as ?game=survival or
 * ?game=find-move. The board's searches stop while one is open, and whatever they had in flight
 * is dropped, so that the game's searches have the engine to themselves.
 */
const gameOpen = ref<Game | null>(([
    'survival', 'find-move',
] as const).find(game => game === new URLSearchParams(location.search).get('game')) ?? null);

if (gameOpen.value !== null) {
    currentTab.value = gameOpen.value;
}

watch(gameOpen, game => {
    const open = game !== null;

    setAnalyzersSuspended(open);

    if (open) {
        engine.abortPending();
    }

    const url = new URL(location.href);

    if (open) {
        url.searchParams.set('game', game);
    } else {
        url.searchParams.delete('game');
    }

    history.replaceState(history.state, '', url);
}, { immediate: true });

const closeGame = async (): Promise<void> => {
    gameOpen.value = null;
    await updateAnalysis();
};

/**
 * Opens the game just played on the board, the player's part of it named. Unlike a load from
 * the file tab, without asking first: asking to see a game is asking to have it on the board,
 * and the tree it replaces is most often the last game's.
 */
const analyseGame = async (game: { size: number, moves: HexMove[], handover: number, you: 0 | 1 }): Promise<void> => {
    // Loaded while the board is still suspended: woken first, it would search the position it
    // had, and restart the engine on that board's size only to restart it again on the game's.
    await importGame({
        boardsize: game.size,
        moves: game.moves,
        playerBlackName: game.you === 0 ? `You, from move ${game.handover + 1}` : 'KataHex',
        playerWhiteName: game.you === 1 ? `You, from move ${game.handover + 1}` : 'KataHex',
    });
    gameOpen.value = null;
    currentTab.value = 'tree';
    await updateAnalysis();
};

const openFileTab = (): void => {
    sidebarOpen.value = true;
    currentTab.value = 'file';
};

/*
 * ctrl+S saves the analysis as a file, there being no study to save it to.
 */
onKeyDown(e => (e.ctrlKey || e.metaKey) && e.key === 's', e => {
    e.preventDefault();
    exportAnalysis();
});

const analysisImportError = ref<string | null>(null);

// An import error is shown in the file tab
watch(analysisImportError, error => {
    if (error) {
        openFileTab();
    }
});

/**
 * Import analysis passed when navigating to Hexplorer, e.g a puzzle.
 * Removed from history state once imported, to not import it again on reload.
 */
const importAnalysisFromHistoryState = async (): Promise<void> => {
    const analysis: unknown = history.state?.[HEXPLORER_ANALYSIS_STATE_KEY];

    if (typeof analysis !== 'string') {
        return;
    }

    history.replaceState({ ...history.state, [HEXPLORER_ANALYSIS_STATE_KEY]: undefined }, '');

    try {
        await importAnalysis(analysis);
    } catch (e) {
        analysisImportError.value = e instanceof Error ? e.message : String(e);
    }
};

/**
 * Import analysis copied from a stream, see hexplorerCopy.ts.
 * Query parameter removed once imported.
 */
const importAnalysisFromCopy = async (): Promise<void> => {
    const url = new URL(window.location.href);
    const copyId = url.searchParams.get('import');

    if (copyId === null) {
        return;
    }

    url.searchParams.delete('import');
    history.replaceState(history.state, '', url.toString());

    const copy = readHexplorerCopy(copyId);

    if (copy === null) {
        return;
    }

    try {
        await restoreState(copy);
    } catch (e) {
        analysisImportError.value = e instanceof Error ? e.message : String(e);
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
                <HexplorerSetupTools v-if="!isLargeScreen" compact />

                <HexplorerNavButtons />

                <LiveAnalysisButton :analyzer="currentAnalyzer" :update="updateAnalysis" />

                <button type="button" class="btn btn-outline-primary open-sidebar-btn" @click="sidebarOpen = !sidebarOpen">
                    <IconArrowBarLeft v-if="!sidebarOpen" />
                    <IconArrowBarRight v-else />
                </button>
            </div>
        </div>

        <AppResizableSidebar
            v-if="sidebarOpen"
            :panels="navPanels"
            :active="currentTab"
            @select="tab => currentTab = tab"
        >
            <template #nav-header="{ collapsed }">
                <template v-if="!collapsed">
                    <h1 class="fs-4 fw-semibold text-truncate px-3 pt-2 mb-0">{{ $t('hexplorer.title') }}</h1>
                </template>
            </template>

            <div class="overflow-auto p-3 flex-grow-1">
                <HexplorerFileTab v-if="currentTab === 'file'" :importError="analysisImportError" />
                <HexplorerHelpTab v-else-if="currentTab === 'help'" @dismiss="currentTab = 'tree'" />
                <HexplorerAiTab v-else-if="currentTab === 'ai'" :analyzers />
                <SurvivalTab v-else-if="currentTab === 'survival'" @play="gameOpen = 'survival'" />
                <FindMoveTab v-else-if="currentTab === 'find-move'" @play="gameOpen = 'find-move'" />
                <HexplorerTreeTab v-else />
            </div>

            <template #footer>
                <p class="credits text-body-secondary text-center">
                    <a href="https://github.com/Phlogistique/katahex-web" target="_blank" rel="noopener">Source</a>, AGPL-3.0.
                    Board and analysis ui from <a href="https://github.com/playhex/playhex" target="_blank" rel="noopener">PlayHex</a>,
                    engine <a href="https://github.com/hzyhhzy/KataGomo" target="_blank" rel="noopener">KataHex</a>,
                    net by <a href="https://github.com/hzyhhzy" target="_blank" rel="noopener">HZY</a>.
                </p>
                <button
                    type="button"
                    class="btn btn-outline-secondary btn-sm w-100 d-flex align-items-center justify-content-center gap-2"
                    @click="sidebarOpen = false"
                ><IconArrowBarRight /> {{ $t('close') }}</button>
            </template>
        </AppResizableSidebar>

        <SurvivalGame v-if="gameOpen === 'survival'" @close="closeGame()" @analyse="analyseGame" />
        <FindMoveGame v-else-if="gameOpen === 'find-move'" @close="closeGame()" @analyse="analyseGame" />
    </div>
</template>

<style lang="stylus" scoped>
.layout
    display flex
    position relative
    height 100vh // (fallback if dvh is not supported)
    height 100dvh

    .game
        flex 1 1 0
        min-width 0
        height 100%
        display flex
        flex-direction column

        .game-inner
            display flex
            flex 1
            min-height 0
            position relative

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
            width calc(100% - 1rem)
            height 100%

.credits
    font-size 0.7rem
    line-height 1.25
    margin-bottom 0.5rem

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
</style>
