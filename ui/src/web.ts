// The hexplorer as a page. Same ui as the Android app, but the engine is the one
// compiled to WebAssembly, with its net on WebGPU, rather than a process on a
// phone -- it answers on the same `window.Native` bridge, so nothing else differs.

import { installWasmEngine } from '@engine/wasmEngine';
import { noteNetStats } from './engineSpeed.js';

// ?fp32 runs the net in single precision, the reference the half-precision
// numbers are checked against; ?threads=N searches with N threads a position;
// ?batchwait=N lets the net wait N microseconds for a partial batch to fill;
// ?servers=N runs N serving threads, each with an evaluation in flight;
// ?leaves=N lets each search thread keep N leaf evals in flight;
// ?profile times every GPU dispatch into window.engineStats.kernels.
const options = new URLSearchParams(location.search);

const status = (text: string) => {
    const line = document.getElementById('engine-status');
    if (line) { line.textContent = text; line.hidden = false; }
};

/** Reloads this session has spent chasing isolation, so a page that cannot get it says so. */
const RELOADS = 'coiReloads';
const RELOAD_LIMIT = 2;

/** How long a service worker is given to become active before we stop waiting for it. */
const ACTIVATION_TIMEOUT_MS = 10000;

/**
 * Why the page is not isolated, for someone holding a phone with no console to ask.
 */
const explain = async () => {
    const worker = navigator.serviceWorker;
    const registered = worker ? (await worker.getRegistrations()).length : 0;
    status('This page needs cross-origin isolation and did not get it. Reloading may ' +
        `fix it. https ${isSecureContext}, service workers ${!!worker}, ` +
        `registered ${registered}, controlling ${!!worker?.controller}, ` +
        `SharedArrayBuffer ${typeof SharedArrayBuffer !== 'undefined'}.`);
};

/**
 * Reloads once the worker coi-serviceworker.js registered is active, which is the point from
 * which a load of this page is controlled and gets the isolation headers. Waiting for it is
 * the whole difference from the reload the script does itself, which fires while the worker is
 * still installing and, if it wins that race, leaves a page that never retries.
 */
const reloadWhenIsolationIsAvailable = async () => {
    const reloads = Number(sessionStorage.getItem(RELOADS) ?? 0);

    if (!navigator.serviceWorker || !isSecureContext || reloads >= RELOAD_LIMIT) {
        await explain();
        return;
    }

    const activated = await Promise.race([
        navigator.serviceWorker.ready.then(() => true),
        new Promise<false>(resolve => setTimeout(() => resolve(false), ACTIVATION_TIMEOUT_MS)),
    ]);

    if (!activated) {
        await explain();
        return;
    }

    sessionStorage.setItem(RELOADS, String(reloads + 1));
    location.reload();
};

// The engine's threads need SharedArrayBuffer, which needs the page to be
// cross-origin isolated. A static host sets no headers, so coi-serviceworker
// installs a service worker that adds them -- and until the page is reloaded
// under it, starting anything means downloading 49 MB of net to throw away.
if (!crossOriginIsolated) {
    status('Setting up cross-origin isolation, this reloads once...');
    void reloadWhenIsolationIsAvailable();
} else {
    sessionStorage.removeItem(RELOADS);

    installWasmEngine({
        half: !options.has('fp32'),
        searchThreads: Number(options.get('threads')) || undefined,
        batchWaitMicros: Number(options.get('batchwait')) || undefined,
        serverThreads: Number(options.get('servers')) || undefined,
        leafEvals: Number(options.get('leaves')) || undefined,
        profile: options.has('profile'),
        // Cumulative net counters: the sidebar's speed readout, and readable from
        // the console or a benchmark harness.
        onStats: (stats) => {
            noteNetStats(stats);
            Object.assign(window, { engineStats: stats });
        },
    });

    await import('./main.js');
}
