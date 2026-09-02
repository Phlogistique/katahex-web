`binModelParser.ts`, `loadModelV8.ts` and `modelV8.ts` are copied from
web-katrain (https://github.com/Sir-Teo/web-katrain), MIT licensed.

They parse KataGo's `.bin.gz` weight format (model versions 8..16) and build
a TensorFlow.js graph from it. KataHex's `hex27x3.bin.gz` is a version 11
`b18c384nbt` net in that same format, so it loads unchanged.

`binModelParser.ts` is modified: `readBinaryFloats` also accepts a float16 array
under an `@BINF16@` marker, which is what `scripts/export_net.py` writes and what
the page downloads. Nothing else in these files knows the difference.

`public-web/coi-serviceworker.js` is coi-serviceworker v0.1.7
(https://github.com/gzuidhof/coi-serviceworker), MIT licensed, copied verbatim.
GitHub Pages serves no response headers of its own, so the page cannot be sent
`Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy`; this registers a
service worker that adds them to every response and reloads the page once, which
is what makes `SharedArrayBuffer` -- and so the engine's threads -- exist.
