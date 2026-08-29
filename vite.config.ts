import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    host: true,
    headers: {
      // Cross-origin isolation, needed later for wasm pthreads (SharedArrayBuffer).
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
