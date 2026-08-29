import { defineConfig } from 'vite';

const headers = {
  // Cross-origin isolation, needed later for wasm pthreads (SharedArrayBuffer).
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  preview: { headers },
  server: {
    host: true,
    headers,
  },
});
