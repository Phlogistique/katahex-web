// PlayHex AI workers on a rented GPU that rents itself back.
//
// Runs WORKERS processes of hex-ai-distributed's worker, which pull jobs from HEX_URL, and serves:
//
//   GET  /health              "ok", without a token
//   GET  /ping?token=TOKEN    a sign of life; answers {left, workers, done, log}
//   POST /bye?token=TOKEN     destroy this machine now
//
// The dead man's switch is server/server.py's: with no ping for IDLE_TIMEOUT seconds
// (FIRST_CONNECT_TIMEOUT before the first one), or once the machine has been up MAX_LIFETIME
// seconds, the workers give their jobs back to PlayHex, then the instance destroys itself
// through the vast.ai API with the per-instance key vast puts in the container, and exits
// whatever the API said, which stops the container and with it the GPU's meter.
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { createServer } from 'node:http';
import { timingSafeEqual } from 'node:crypto';

const env = process.env;
const TOKEN = env.WORKER_TOKEN ?? '';
const PORT = Number(env.PORT ?? 8765);
const WORKERS = Number(env.WORKERS ?? 2);
const IDLE_TIMEOUT = Number(env.IDLE_TIMEOUT ?? 1800);
const FIRST_CONNECT_TIMEOUT = Number(env.FIRST_CONNECT_TIMEOUT ?? 1800);
const MAX_LIFETIME = Number(env.MAX_LIFETIME ?? 12 * 3600);
const VAST_API = env.VAST_API ?? 'https://console.vast.ai/api/v0';

if (!TOKEN) {
    console.error('WORKER_TOKEN must be set');
    process.exit(1);
}

const started = Date.now();
let lastLife = null; // Date.now() of the last ping, null before any
let done = 0;
let quitting = false;
const workers = [];
const recent = []; // the last lines the workers logged, for the app

const log = (line, forApp = true) => {
    console.log(line);
    if (forApp) {
        recent.push(line);
        recent.splice(0, recent.length - 40);
    }
};

for (let i = 0; i < WORKERS; i++) {
    const child = spawn('node', ['/app/dist/src/worker'], { cwd: '/app', stdio: ['ignore', 'pipe', 'pipe'] });
    for (const stream of [child.stdout, child.stderr]) {
        createInterface({ input: stream }).on('line', line => {
            line = line.replace(/\x1b\[[0-9;]*m/g, ''); // winston colours its levels
            if (line.includes('Job done')) {
                done++;
            }
            // The app gets info and above, not the engine's debug output: whole boards of policy.
            log(`[${i}] ${line}`, /^\S+ \S+ (info|notice|warning|error|crit|alert|emerg):/.test(line));
        });
    }
    // Not restarted: a worker exits on its own only for a refused key or a broken engine,
    // which a restart would not fix. The app shows why from the log.
    child.on('exit', code => log(`[${i}] worker exited with code ${code}`));
    workers.push(child);
}

const running = () => workers.filter(child => child.exitCode === null && child.signalCode === null);

/** Workers give back their jobs on a second SIGTERM, so that PlayHex reassigns them at once. */
const stopWorkers = async () => {
    const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
    for (let i = 0; i < 2; i++) {
        running().forEach(child => child.kill('SIGTERM'));
        await sleep(500); // two signals sent at once can arrive as one
    }
    const deadline = Date.now() + 10_000;
    while (running().length > 0 && Date.now() < deadline) {
        await sleep(200);
    }
};

const destroySelf = async reason => {
    if (quitting) {
        return;
    }
    quitting = true;
    log(`dead man's switch: ${reason}`);
    await stopWorkers();
    const instance = env.CONTAINER_ID;
    const keys = [env.CONTAINER_API_KEY, env.VAST_API_KEY].filter(Boolean);
    if (!instance) {
        log('no CONTAINER_ID: not on vast.ai, just exiting');
    }
    for (const key of instance ? keys : []) {
        for (const [method, body] of [['DELETE', undefined], ['PUT', '{"state": "stopped"}']]) {
            try {
                const response = await fetch(`${VAST_API}/instances/${instance}/`, {
                    method, body, signal: AbortSignal.timeout(20_000),
                    headers: { 'Authorization': `Bearer ${key}`, 'Content-Type': 'application/json' },
                });
                log(`${method} instance ${instance} -> ${response.status} ${(await response.text()).slice(0, 200)}`);
                if (response.ok) {
                    process.exit(0);
                }
            } catch (error) {
                log(`${method} instance ${instance} failed: ${error}`);
            }
        }
    }
    process.exit(0);
};

const left = () => {
    const now = Date.now();
    const idle = lastLife === null ? FIRST_CONNECT_TIMEOUT - (now - started) / 1000 : IDLE_TIMEOUT - (now - lastLife) / 1000;
    const life = MAX_LIFETIME > 0 ? MAX_LIFETIME - (now - started) / 1000 : idle;
    return Math.round(Math.min(idle, life));
};

setInterval(() => {
    if (MAX_LIFETIME > 0 && Date.now() - started > MAX_LIFETIME * 1000) {
        destroySelf(`up for ${MAX_LIFETIME}s, the most it may be`);
    } else if (left() <= 0) {
        destroySelf(lastLife === null ? `no client in the first ${FIRST_CONNECT_TIMEOUT}s` : `no sign of life for ${IDLE_TIMEOUT}s`);
    }
}, 10_000);

const authorized = given => {
    const a = Buffer.from(given ?? ''), b = Buffer.from(TOKEN);
    return a.length === b.length && timingSafeEqual(a, b);
};

createServer((request, response) => {
    const url = new URL(request.url, 'http://localhost');
    const reply = (status, body, type = 'text/plain') => {
        response.writeHead(status, { 'Content-Type': type });
        response.end(body);
    };
    if (url.pathname === '/health') {
        return reply(200, 'ok\n');
    }
    if (!authorized(url.searchParams.get('token'))) {
        return reply(403, 'bad token\n');
    }
    if (url.pathname === '/ping') {
        lastLife = Date.now();
        return reply(200, JSON.stringify({ left: left(), workers: running().length, done, log: recent }), 'application/json');
    }
    if (url.pathname === '/bye' && request.method === 'POST') {
        reply(200, 'bye\n');
        return destroySelf('the client said bye');
    }
    reply(404, 'not found\n');
}).listen(PORT, () => log(`listening on ${PORT}; ${WORKERS} workers; idle ${IDLE_TIMEOUT}s, `
    + `first connect ${FIRST_CONNECT_TIMEOUT}s, lifetime ${MAX_LIFETIME}s`));

process.on('SIGTERM', async () => {
    await stopWorkers();
    process.exit(0);
});
