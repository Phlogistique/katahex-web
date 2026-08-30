// Opening and, more importantly, actually closing the browser a run needs.
//
// `browser.close()` does not take the page's process tree with it when the page
// holds a wasm engine: the engine's pthread workers and its WebGPU device keep
// the renderer alive, and the whole tree survives the call. A leaked engine
// still burns CPU and holds a device, so the next run measures itself against
// it -- which looks exactly like a thread-count effect or a precision effect,
// depending on the order the runs happen to go in. Measured three times slower
// twenty minutes into a session than at its start, purely from leaks.
//
// So a run refuses to start while one is alive, and waits for the tree to be
// gone before it returns.

import { execFileSync } from 'node:child_process';
import { chromium } from 'playwright';

/** Chromium needs the Vulkan flags to expose this laptop's Iris Xe to WebGPU. */
const ARGS = ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan'];

const alive = () => {
  const out = execFileSync('ps', ['-eo', 'pid,comm'], { encoding: 'utf8' });
  return out.split('\n')
    .filter((line) => /\bchrome-headless/.test(line))
    .map((line) => Number(line.trim().split(/\s+/)[0]));
};

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export async function open(url, { expose = {}, onError = () => {} } = {}) {
  const leaked = alive();
  if (leaked.length) {
    throw new Error(
      `${leaked.length} chrome-headless process(es) still alive (${leaked.join(', ')}). ` +
      'A previous run leaked its engine; kill them before measuring anything.');
  }

  const browser = await chromium.launch({ args: ARGS });
  const page = await browser.newPage();
  page.on('pageerror', (error) => onError(error.message));
  // Everything the page calls has to exist before it is loaded: it asks for its
  // first job as soon as its module runs.
  for (const [name, fn] of Object.entries(expose)) await page.exposeFunction(name, fn);
  await page.goto(url);
  return { browser, page };
}

/** Resolves when the log says the page is finished, whether it worked or not. */
export const finished = (page) => page.waitForFunction(
  () => /\n(done|ERROR:)/.test(document.getElementById('log').textContent),
  null, { timeout: 0 });

export async function close(browser, page) {
  // The page drops its engines first; closing on top of live workers is what
  // leaves the tree behind.
  await page.evaluate(() => (globalThis).stopEngines?.()).catch(() => {});
  await browser.close().catch(() => {});

  for (let waited = 0; waited < 30000; waited += 250) {
    if (!alive().length) return;
    await sleep(250);
  }
  const stuck = alive();
  for (const pid of stuck) {
    try { process.kill(pid, 'SIGKILL'); } catch { /* already gone */ }
  }
  console.error(`killed ${stuck.length} leaked chrome-headless process(es)`);
}
