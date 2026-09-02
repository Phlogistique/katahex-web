import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import Icons from 'unplugin-icons/vite';
import { viteSingleFile } from 'vite-plugin-singlefile';

const here = (path: string) => fileURLToPath(new URL(path, import.meta.url));
const src = (path: string) => here(`./src/${path}`);

// Where the engine compiled to WebAssembly and the net that runs on WebGPU live.
const katahexWeb = here('..');

// The engine is built with pthreads, so the page must be cross-origin isolated
// for SharedArrayBuffer to exist.
const headers = {
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
};

/**
 * One ui, two shells. By default it builds the app's `index.html`: loaded from a
 * WebView, where module scripts and separate asset requests are blocked, so
 * everything is inlined into one file, and the engine is a native process.
 * `--mode web` builds `web.html` instead, which runs the engine itself.
 */
export default defineConfig(({ mode }) => {
    const page = mode === 'web';

    return {
        // Relative, so a build serves from a project page's subpath as
        // happily as from a root.
        base: page ? './' : undefined,
        plugins: [
            vue(),
            Icons({ compiler: 'vue3', scale: 1 }),
            ...(page ? [] : [viteSingleFile()]),
        ],
        // Only what the page is served: the rest of public/ is the net
        // benchmark's, a few hundred megabytes of nets it never asks for.
        publicDir: page ? here('../public-web') : false,
        resolve: {
            alias: {
                // The board renderer is a yarn workspace inside the playhex repo, vendored here.
                '@playhex/pixi-board': src('shared/pixi-board/index.ts'),
                'vue-router': src('stubs/vue-router.ts'),
                '@unhead/vue': src('stubs/unhead.ts'),
                '@engine': `${katahexWeb}/src`,
            },
        },
        server: { headers, fs: { allow: ['.', katahexWeb] } },
        preview: { headers },
        build: {
            target: 'es2022',
            outDir: page ? 'dist-web' : 'dist',
            assetsInlineLimit: page ? 4096 : 100000000,
            cssCodeSplit: false,
            rollupOptions: page ? { input: here('./web.html') } : undefined,
        },
    };
});
