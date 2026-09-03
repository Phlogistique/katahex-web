<script setup lang="ts">
/*
 * Pauses and resumes the search on the position displayed, and shows how many visits it has
 * done. Only shown while the selected analyzer is one that searches.
 *
 * Its colour is the state of that search: orange while it runs, green once it has reached the
 * depth the analyzer stops at, grey while the user holds it paused. `KataHex live` stops at no
 * depth, so it is orange until paused.
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

const pausable = computed(() =>
    props.analyzer instanceof KatahexAnalyzer && props.analyzer.searches ? props.analyzer : null);

const paused = computed(() => pausable.value?.paused.value ?? false);
const searching = computed(() => pausable.value?.searching.value ?? false);
const visits = computed(() => pausable.value?.visits.value ?? 0);

const color = computed(() => {
    if (paused.value) {
        return 'btn-secondary';
    }

    return searching.value ? 'btn-warning' : 'btn-success';
});

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
        v-if="pausable"
        type="button"
        class="btn"
        :class="color"
        :title="paused ? 'Resume analysis' : 'Pause analysis'"
        @click="pausable.togglePause()"
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
