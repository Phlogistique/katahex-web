<script setup lang="ts">
/*
 * Players, evaluation and tree of the analysis, with what edits current node.
 * AI settings are in AI tab.
 *
 * Patched here: the engine's speed above the graph, and how far filling the graph has got
 * under it, since on one engine that is minutes of work for a long game; and the graph in
 * log-odds when the AI tab's odds option is on.
 */
import { inject } from 'vue';
import { hexplorerKey } from '../composables/useHexplorer.js';
import { HEXPLORER_COMMENT_MAX_LENGTH } from '../../../../shared/app/hexplorerStudy.js';
import EvaluationGraph from './EvaluationGraph.vue';
import { searching, speed } from '../../../../engineSpeed.js';
import { storeToRefs } from 'pinia';
import usePlayerLocalSettingsStore from '../../../stores/playerLocalSettingsStore.js';
import { formatOdds, logitScale } from '../../../../winrateScale.js';
import HexplorerMoveTreePanel from './HexplorerMoveTreePanel.vue';
import HexplorerSetupTools from './HexplorerSetupTools.vue';
import HexplorerPlayerNameInput from './HexplorerPlayerNameInput.vue';
import {
    IconChatRightText,
    IconDiagram2,
    IconGraphUp,
} from '../../icons.js';

const {
    aiEnabled,
    evalHistory,
    evalCursorIndex,
    lineProgress,
    goToEvalIndex,
    userGoToNode,
    currentComment,
} = inject(hexplorerKey)!;

const { localSettings } = storeToRefs(usePlayerLocalSettingsStore());

/** Red's odds when red is ahead, blue's when blue is: a bar's title, on the odds scale. */
const formatEval = (whiteWin: number): string => whiteWin >= 0.5
    ? `blue ${formatOdds(whiteWin)}`
    : `red ${formatOdds(1 - whiteWin)}`;
</script>

<template>
    <div>
        <div class="d-flex gap-3 border rounded bg-body px-2 py-1 mb-3">
            <HexplorerPlayerNameInput :player="0" class="flex-grow-1" />
            <HexplorerPlayerNameInput :player="1" class="flex-grow-1" />
        </div>

        <template v-if="aiEnabled">
            <h6 class="d-flex align-items-center gap-2 mb-2">
                <IconGraphUp /> {{ $t('hexplorer.evaluation') }}
            </h6>
            <p v-if="speed" class="speed text-body-secondary mb-2" :class="{ stale: !searching }">
                {{ Math.round(speed.visits) }} visits/s<template v-if="speed.netEvals !== null">,
                net {{ Math.round(speed.netEvals) }} evals/s<template v-if="speed.batch !== null">
                at batch {{ Math.round(speed.batch) }}</template></template>
            </p>
            <div class="mb-3">
                <EvaluationGraph
                    :evalHistory
                    :cursorIndex="evalCursorIndex"
                    :scale="localSettings.hexplorerOddsScale ? logitScale : undefined"
                    :format="localSettings.hexplorerOddsScale ? formatEval : undefined"
                    class="bg-body"
                    @select="goToEvalIndex"
                />
                <p v-if="lineProgress" class="graph-progress text-body-secondary mt-1 mb-0">
                    {{ $t(lineProgress.waiting ? 'hexplorer.filling_graph_waiting' : 'hexplorer.filling_graph', lineProgress) }}
                </p>
            </div>
        </template>

        <h6 class="d-flex align-items-center gap-2 mb-2">
            <IconDiagram2 /> {{ $t('hexplorer.move_history') }}
        </h6>
        <HexplorerMoveTreePanel class="mb-2" @select="userGoToNode" />

        <HexplorerSetupTools />

        <h6 class="d-flex align-items-center gap-2 mt-3 mb-2">
            <IconChatRightText /> {{ $t('hexplorer.comment') }}
        </h6>
        <textarea
            v-model="currentComment"
            class="form-control form-control-sm mb-3"
            rows="3"
            :maxlength="HEXPLORER_COMMENT_MAX_LENGTH"
            :placeholder="$t('hexplorer.comment_placeholder')"
            :aria-label="$t('hexplorer.comment')"
        ></textarea>
    </div>
</template>

<style lang="stylus" scoped>
.speed, .graph-progress
    font-size 0.75rem
    font-variant-numeric tabular-nums

.speed.stale
    opacity 0.5
</style>
