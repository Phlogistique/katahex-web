<script setup lang="ts">
/*
 * Pauses and resumes the search on the position displayed, and shows how many visits it has
 * done. Only shown while the selected analyzer is one that searches.
 *
 * Written for this app; not part of PlayHex.
 */
import { computed, watch } from 'vue';
import { setAnalyzersRefresh } from '../../../../analyzers.js';
import { KatahexAnalyzer } from '../analyzers/KatahexAnalyzer.js';
import { IconPauseFill, IconPlayFill } from '../../icons.js';
import type { AnalyzerInterface } from '../analyzers/AnalyzerInterface.js';

const props = defineProps<{
    analyzer: AnalyzerInterface | null;

    /** hexplorer's updateAnalysis, which is how a grown search gets painted. */
    update: () => void;
}>();

setAnalyzersRefresh(() => props.update());

const searching = computed(() =>
    props.analyzer instanceof KatahexAnalyzer && props.analyzer.searches ? props.analyzer : null);

const paused = computed(() => searching.value?.paused.value ?? false);
const visits = computed(() => searching.value?.visits.value ?? 0);

// hexplorer points the analyzer it is switching to at the displayed position, but says nothing
// to the one it is leaving, which would go on searching.
watch(() => props.analyzer, (_current, previous) => {
    if (previous instanceof KatahexAnalyzer) {
        previous.stop();
    }
});
</script>

<template>
    <button
        v-if="searching"
        type="button"
        class="btn"
        :class="paused ? 'btn-secondary' : 'btn-success'"
        :title="paused ? 'Resume analysis' : 'Pause analysis'"
        @click="searching.togglePause()"
    >
        <IconPlayFill v-if="paused" /><IconPauseFill v-else />
        <span class="visits">{{ visits }}</span>
    </button>
</template>

<style scoped>
.visits {
    font-variant-numeric: tabular-nums;
    margin-inline-start: 0.35em;
}
</style>
