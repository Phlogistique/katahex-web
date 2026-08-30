import { defineConfig } from 'vite';

const headers = {
  // Cross-origin isolation, needed later for wasm pthreads (SharedArrayBuffer).
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  build: { rollupOptions: { input: ['index.html', 'micro.html', 'winograd.html', 'match.html', 'openings.html'] } },
  preview: { headers },
  server: {
    host: true,
    headers,
  },
});
