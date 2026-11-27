<script setup lang="ts">
/*
 * The analysis board's way into the find-the-move game (FindMoveGame.vue).
 *
 * Written for this app; not part of PlayHex.
 */
import { computed } from 'vue';
import { storeToRefs } from 'pinia';
import usePlayerLocalSettingsStore from './client/stores/playerLocalSettingsStore.js';
import { findMoveKey, sanitizeFindMoveSettings } from './findMove.js';
import { IconBullseye } from './client/vue/icons.js';

const emit = defineEmits<{
    play: [];
}>();

const { localSettings } = storeToRefs(usePlayerLocalSettingsStore());

const best = computed(() => localSettings.value.findMoveBest?.[findMoveKey(sanitizeFindMoveSettings(localSettings.value.findMove))] ?? null);
</script>

<template>
    <div>
        <h2>Find KataHex's move</h2>
        <p>
            You play red against KataHex, from a position its net played itself into, with red
            winning but not by much. Each turn you get three moves to choose from: the one
            KataHex's search plays, and two that look natural but lose.
        </p>
        <p>Your score is how many of KataHex's moves you find before you pick a wrong one.</p>
        <p v-if="best !== null" class="text-body-secondary">Your best with the current settings: {{ best }}.</p>
        <button type="button" class="btn btn-success w-100" @click="emit('play')"><IconBullseye /> Play</button>
    </div>
</template>
