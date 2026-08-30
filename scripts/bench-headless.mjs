// Runs the net benchmark page in headless Chrome and prints its log.
//
//   node scripts/bench-headless.mjs hand-webgpu-fp16 hand-webgpu-fp32 -- 11
//
// Needs a vite server already serving the page (npm run dev), or set BASE.
// Chromium needs Vulkan flags to expose this laptop's Iris Xe to WebGPU.

import { chromium } from 'playwright';

const BASE = process.env.BASE ?? 'http://localhost:5173';
const args = process.argv.slice(2);
const sep = args.indexOf('--');
const runtimes = (sep < 0 ? args : args.slice(0, sep));
const sizes = (sep < 0 ? ['11'] : args.slice(sep + 1));

const browser = await chromium.launch({
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan'],
});
const page = await browser.newPage();
page.on('console', (m) => { if (process.env.VERBOSE) console.error('[page]', m.text()); });
await page.goto(BASE + '/index.html');

for (const size of sizes) {
  for (const runtime of runtimes) {
    await page.selectOption('#runtime', runtime);
    await page.selectOption('#size', size);
    await page.click('#go');
    await page.waitForFunction(
      () => /\n(done|ERROR:)/.test(document.getElementById('log').textContent),
      null, { timeout: 20 * 60 * 1000 });
    console.log(`### ${runtime} ${size}x${size}`);
    console.log(await page.textContent('#log'));
  }
}
await browser.close();
