`binModelParser.ts`, `loadModelV8.ts` and `modelV8.ts` are copied from
web-katrain (https://github.com/Sir-Teo/web-katrain), MIT licensed.

They parse KataGo's `.bin.gz` weight format (model versions 8..16) and build
a TensorFlow.js graph from it. KataHex's `hex27x3.bin.gz` is a version 11
`b18c384nbt` net in that same format, so it loads unchanged.
