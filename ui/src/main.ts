import 'bootstrap/dist/css/bootstrap.min.css';
import './app.css';
import { createApp } from 'vue';
import i18next, { t } from 'i18next';
import unoverlay from '@overlastic/vue';
import en from './shared/app/en.json';
import PageHexplorer from './client/vue/hexplorer/pages/PageHexplorer.vue';
import { setAnalyzersAwake } from './analyzers.js';
import { analysisStore } from './analysisStore.js';
import { engine } from './engine.js';

void i18next.init({
    lng: 'en',
    resources: { en: { translation: en } },
    interpolation: { escapeValue: false },
});

const app = createApp(PageHexplorer);

// PlayHex's templates call $t, which its i18next-vue plugin provides.
app.config.globalProperties.$t = t;

// Overlays are mounted in an app of their own, which only inherits $t from this one once
// overlastic has been told which app is the parent.
app.use(unoverlay);

app.mount('#app');

/*
 * Engine status. The engine takes a few seconds to load the net, and a board size with no
 * cached tuning takes about twenty minutes on first use, so it says so rather than looking dead.
 */
const status = document.getElementById('engine-status') as HTMLElement;

engine.onStatus = (text, ready) => {
    status.textContent = text;
    status.hidden = ready;
};

/*
 * Going to the background is this app's "leaving the page": PlayHex persists its analysis cache
 * there, and a live search left running would go on spending the gpu behind another app.
 */
document.addEventListener('visibilitychange', () => {
    const hidden = document.visibilityState === 'hidden';

    setAnalyzersAwake(!hidden);

    if (hidden) {
        analysisStore.persist();
    }
});
