// The hexplorer as a page. Same ui as the Android app, but the engine is the one
// compiled to WebAssembly, with its net on WebGPU, rather than a process on a
// phone -- it answers on the same `window.Native` bridge, so nothing else differs.

import { installWasmEngine } from '@engine/wasmEngine';

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

// The engine's threads need SharedArrayBuffer, which needs the page to be
// cross-origin isolated. A static host sets no headers, so coi-serviceworker
// installs a service worker that adds them and reloads -- and until that reload
// lands, starting anything means downloading 49 MB of net to throw away.
if (!crossOriginIsolated) {
    status('Setting up cross-origin isolation, this reloads once...');
    setTimeout(() => status(
        'This page needs cross-origin isolation and did not get it. It has to be ' +
        'served over https, with service workers allowed.'), 5000);
} else {
    installWasmEngine({
        half: !options.has('fp32'),
        searchThreads: Number(options.get('threads')) || undefined,
        batchWaitMicros: Number(options.get('batchwait')) || undefined,
        serverThreads: Number(options.get('servers')) || undefined,
        leafEvals: Number(options.get('leaves')) || undefined,
        profile: options.has('profile'),
        // Cumulative net counters, readable from the console or a benchmark harness.
        onStats: (stats) => Object.assign(window, { engineStats: stats }),
    });

    await import('./main.js');
}
