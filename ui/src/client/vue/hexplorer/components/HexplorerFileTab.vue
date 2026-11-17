<script setup lang="ts">
/*
 * Saved study, and all ways to start, load or export an analysis.
 *
 * Patched here: PlayHex saves, forks and shares studies on its server, which this app has
 * none of, so only the local ways are left. The new board size is held to what the engine
 * plays, New asks before throwing a tree away, and the file loader also takes an sgf: a
 * phone has files to open where a desktop has a clipboard to paste from.
 */
import { computed, inject, ref, useTemplateRef } from 'vue';
import { defineOverlay } from '@overlastic/vue';
import { hexplorerKey } from '../composables/useHexplorer.js';
import { t } from 'i18next';
import { createHexplorerState, movesToTreeNodes, parseHexplorerState } from '../../../../shared/app/hexplorerStudy.js';
import { MAX_BOARD_SIZE as MAX_BOARDSIZE, MIN_BOARD_SIZE as MIN_BOARDSIZE } from '../../../../engine.js';
import { hexGameImporter } from '../../../../shared/app/hex-game-importer/index.js';
import type { ImportedGame } from '../../../../shared/app/hex-game-importer/types.js';
import ImportGameOverlay from '../../components/overlay/ImportGameOverlay.vue';
import {
    IconDownload,
    IconFileEarmarkPlus,
    IconFolder2Open,
    IconMagic,
    IconSave2,
} from '../../icons.js';

const props = defineProps<{
    /**
     * Error while importing an analysis passed when opening page.
     */
    importError: null | string;
}>();

const {
    gameView,
    state,
    importGame,
    resetState,
    exportSgf,
    exportAnalysis,
    restoreState,
} = inject(hexplorerKey)!;

const isValidBoardsize = (size: number): boolean =>
    Number.isInteger(size) && size >= MIN_BOARDSIZE && size <= MAX_BOARDSIZE;

/** A tree with more than its root in it is work that New or a load would throw away. */
const confirmDiscardChanges = (): boolean =>
    state.value.nodes.length <= 1 || confirm(t('hexplorer.new_analysis_confirm'));

const closeStudy = async (): Promise<void> => {};

const newBoardsize = ref(gameView.value.getBoardsize());

const newAnalysis = async (): Promise<void> => {
    if (!confirmDiscardChanges()) {
        return;
    }

    await closeStudy();
    await resetState(newBoardsize.value);
};

/*
 * Import
 */
const importGameOverlay = defineOverlay(ImportGameOverlay);

const openImportOverlay = () => {
    void importGameOverlay({
        onImport: async (importedGame: ImportedGame) => {
            // Refuses a game that cannot be replayed before discarding anything for it
            parseHexplorerState({
                ...createHexplorerState(importedGame.boardsize),
                nodes: importedGame.tree ?? movesToTreeNodes(importedGame.moves),
            });

            if (!confirmDiscardChanges()) {
                return;
            }

            await closeStudy();
            await importGame(importedGame);
        },
    }).catch(() => {
        // dismissed
    });
};

/*
 * Analysis JSON import (whole HexplorerState + move tree)
 */
const analysisFileInput = useTemplateRef<HTMLInputElement>('analysisFileInput');
const analysisFileError = ref<string | null>(null);
const analysisImportError = computed(() => analysisFileError.value ?? props.importError);

const onAnalysisFileSelected = async (event: Event) => {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;

    analysisFileError.value = null;

    try {
        const content = await file.text();

        // A saved analysis is json; anything else the importer knows, an sgf above all, is
        // loaded as a game. Either is refused before discarding anything for it.
        if (content.trimStart().startsWith('{')) {
            const analysis = parseHexplorerState(JSON.parse(content));

            if (!confirmDiscardChanges()) {
                return;
            }

            await restoreState(analysis);
        } else {
            const importedGame = await hexGameImporter.import(content);

            if (!confirmDiscardChanges()) {
                return;
            }

            await importGame(importedGame);
        }
    } catch (e) {
        analysisFileError.value = e instanceof Error ? e.message : String(e);
    } finally {
        input.value = ''; // allow re-selecting the same file
    }
};
</script>

<template>
    <div>
        <h6 class="mb-2">{{ $t('hexplorer.export') }}</h6>

        <div class="d-flex flex-wrap gap-2 mb-2">
            <button
                type="button"
                class="btn btn-sm btn-outline-primary"
                :title="$t('hexplorer.download_sgf')"
                @click="exportSgf()"
            ><IconDownload /> SGF</button>

            <button
                type="button"
                class="btn btn-sm btn-outline-success"
                :title="$t('hexplorer.analysis_json_save')"
                @click="exportAnalysis()"
            ><IconSave2 /> {{ $t('hexplorer.save') }} JSON</button>
        </div>

        <hr>

        <h6 class="mb-2">{{ $t('hexplorer.new_study') }}</h6>

        <div class="d-flex flex-wrap gap-2 mb-2">
            <div class="input-group input-group-sm" style="max-width: 9rem" :title="$t('hexplorer.new_analysis')">
                <input
                    type="number"
                    :min="MIN_BOARDSIZE"
                    :max="MAX_BOARDSIZE"
                    v-model.number="newBoardsize"
                    class="form-control"
                >
                <button
                    class="btn btn-outline-warning"
                    :disabled="!isValidBoardsize(newBoardsize)"
                    @click="newAnalysis()"
                ><IconFileEarmarkPlus /> {{ $t('hexplorer.new') }}</button>
            </div>

            <button
                type="button"
                class="btn btn-sm btn-outline-warning flex-grow-1"
                :title="$t('hexplorer.analysis_json_load')"
                @click="analysisFileInput?.click()"
            ><IconFolder2Open /> {{ $t('hexplorer.load') }}</button>
        </div>
        <input
            ref="analysisFileInput"
            type="file"
            accept="application/json,.json,.sgf,application/x-go-sgf"
            class="d-none"
            @change="onAnalysisFileSelected"
        >
        <div v-if="analysisImportError" class="text-danger small mb-2">{{ analysisImportError }}</div>

        <button
            class="btn btn-outline-info w-100 mb-3 d-flex align-items-center justify-content-center gap-2"
            @click="openImportOverlay()"
        ><IconMagic /> {{ $t('hexplorer.import_a_game') }}</button>

    </div>
</template>
