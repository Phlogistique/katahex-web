/*
 * Written here, at PlayHex's module path: PlayHex's store holds a dozen settings for its whole
 * site, through class-transformer. The hexplorer reads one of them, and one is this app's own.
 */
import { defineStore } from 'pinia';
import { ref, watch } from 'vue';

const LOCAL_SETTINGS_KEY = 'hex-local-settings';

export type LocalSettings = {
    /** The help tab is opened first until it is dismissed. */
    hexplorerHelpDismissed?: boolean;

    /** The board shows each candidate's winrate and visits instead of the policy. */
    hexplorerShowMoveInfos?: boolean;
};

const load = (): LocalSettings => {
    try {
        return JSON.parse(localStorage.getItem(LOCAL_SETTINGS_KEY) ?? '{}') as LocalSettings;
    } catch {
        return {};
    }
};

const usePlayerLocalSettingsStore = defineStore('playerLocalSettingsStore', () => {
    const localSettings = ref<LocalSettings>(load());

    watch(localSettings, settings => {
        try {
            localStorage.setItem(LOCAL_SETTINGS_KEY, JSON.stringify(settings));
        } catch {
            // A WebView without storage: the setting lasts the session.
        }
    }, { deep: true });

    return { localSettings };
});

export default usePlayerLocalSettingsStore;
