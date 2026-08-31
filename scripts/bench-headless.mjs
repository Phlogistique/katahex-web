// Runs the net benchmark page in headless Chrome and prints its log.
//
//   node scripts/bench-headless.mjs hand-webgpu-fp16 hand-webgpu-fp32 -- 11
//
// Needs a vite server already serving the page (npm run dev), or set BASE.
// Chromium needs Vulkan flags to expose this laptop's Iris Xe to WebGPU.

import { readFileSync } from 'node:fs';

import { close, finished, open } from './browser.mjs';

// This GPU throttles from 950 MHz to 400-600 under load, and a sweep taken at
// 450 against one taken at 900 is not a comparison. Wall clock cannot tell the
// two apart; the achieved frequency can, so every sweep carries the band it ran
// in and a reader can throw out the ones that do not match.
const FREQ = '/sys/class/drm/card1/gt_act_freq_mhz';

function watchFrequency() {
  const seen = [];
  const read = () => {
    try { seen.push(Number(readFileSync(FREQ, 'utf8'))); } catch { /* no counter here */ }
  };
  read();
  const timer = setInterval(read, 250);
  return () => {
    clearInterval(timer);
    if (!seen.length) return 'frequency unknown';
    seen.sort((a, b) => a - b);
    return `${seen[0]}-${seen[seen.length - 1]} MHz, median ${seen[seen.length >> 1]}`;
  };
}

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
    console.log(`### ${runtime} ${size}x${size} at ${frequency()}`);
    console.log(await page.textContent('#log'));
  }
}
await close(browser, page);
