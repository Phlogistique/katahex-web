<script setup lang="ts">
/*
 * The analysis board's way into the survival game (SurvivalGame.vue).
 *
 * Written for this app; not part of PlayHex.
 */
import { computed } from 'vue';
import { storeToRefs } from 'pinia';
import usePlayerLocalSettingsStore from './client/stores/playerLocalSettingsStore.js';
import { sanitizeSettings, settingsKey } from './survival.js';
import { IconStopwatch } from './client/vue/icons.js';

const emit = defineEmits<{
    play: [];
}>();

const { localSettings } = storeToRefs(usePlayerLocalSettingsStore());

const best = computed(() => localSettings.value.survivalBest?.[settingsKey(sanitizeSettings(localSettings.value.survival))] ?? null);
</script>

<template>
    <div>
        <h2>Survive KataHex</h2>
        <p>
            KataHex plays itself from an even opening, a first move from PlayHex's swap maps that
            blue may swap, your side well and the other badly, until your side is winning
            comfortably. Then you take it, red or blue, on a short clock, and KataHex plays the
            other for real.
        </p>
        <p>
            Your score is how many moves you make before KataHex rates you under the line.
            The opponent's strength, the search that rates you, how badly blue plays itself into
            a loss and the clock can all be set before a game.
        </p>
        <p v-if="best !== null" class="text-body-secondary">Your best with the current settings: {{ best }}.</p>
        <button type="button" class="btn btn-success w-100 survival-play-btn" @click="emit('play')"><IconStopwatch /> Play</button>
    </div>
</template>
