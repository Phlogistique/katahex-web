// Runs the net benchmark page in headless Chrome and prints its log.
//
//   node scripts/bench-headless.mjs hand-webgpu-fp16 hand-webgpu-fp32 -- 11
//
// Needs a vite server already serving the page (npm run dev), or set BASE.
// Chromium needs Vulkan flags to expose this laptop's Iris Xe to WebGPU.

import { close, finished, open, watchFrequency } from './browser.mjs';

const BASE = process.env.BASE ?? 'http://localhost:5173';
const args = process.argv.slice(2);
const sep = args.indexOf('--');
const runtimes = (sep < 0 ? args : args.slice(0, sep));
const sizes = (sep < 0 ? ['11'] : args.slice(sep + 1));

const { browser, page } = await open(BASE + '/index.html', {
  onError: (message) => console.error('[page]', message),
});

for (const size of sizes) {
  for (const runtime of runtimes) {
    await page.selectOption('#runtime', runtime);
    await page.selectOption('#size', size);
    const frequency = watchFrequency();
    await page.click('#go');
    await finished(page);
    console.log(`### ${runtime} ${size}x${size} at ${frequency().band}`);
    console.log(await page.textContent('#log'));
  }
}
await close(browser, page);
