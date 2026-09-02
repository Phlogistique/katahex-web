# Licences

The site as a whole is **AGPL-3.0-only** (`LICENSE`), because the ui is
PlayHex's hexplorer and that is AGPL-3.0-only. Serving it to a browser hands
the visitor a copy of the program, so its source is owed either way; it is this
repository.

Not everything here is AGPL, and none of it has to be:

| | |
| --- | --- |
| `ui/src` | mostly copied from [PlayHex](https://github.com/playhex/playhex), AGPL-3.0-only. See `ui/VENDORED.md` for what is copied, what is written here, and what is patched. |
| `engine/` | the KataHex fork of KataGo, MIT (`engine/LICENSE`). Combining it with AGPL code does not relicense it, and it stays MIT on its own. |
| `src/`, `scripts/` | the WebGPU backend, the bridge to the engine, and the tools around them. Written here. `src/webgpuModel.ts` and `src/netRunner.ts` touch no PlayHex code and are usable on their own. |
| `vendor/` | the KataGo weight-file parsers from web-katrain (MIT) and coi-serviceworker (MIT). See `vendor/VENDORED.md`. |

## The net

`public-web/net-fp16.bin.gz` is exported from `hex27x3.bin.gz`, the Hex network
trained by HZY (hzyhhzy), redistributed here with his permission. It is not
covered by the licences above.
