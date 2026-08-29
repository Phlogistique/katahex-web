#!/bin/sh
# Cross-compile the KataHex engine to WebAssembly.
#
# Needs emsdk (https://github.com/emscripten-core/emsdk) sourced, and a katahex
# checkout on a branch carrying the Emscripten cmake fixes and the JS backend.
#
#   BACKEND=JS     the net is evaluated from JavaScript (see src/netRunner.ts)
#   BACKEND=EIGEN  the net is evaluated on the wasm CPU, for comparison; also
#                  needs Eigen headers under $EIGEN
set -eu

: "${EMSDK:?source emsdk_env.sh first}"
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
KATAHEX=${KATAHEX:-$ROOT/katahex}
EIGEN=${EIGEN:-$ROOT/third_party/eigen-prefix}
BACKEND=${BACKEND:-JS}
BUILD=${BUILD:-$ROOT/build-wasm-$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')}
SYSROOT=$EMSDK/upstream/emscripten/cache/sysroot

# -include endian.h: numpywrite.cpp and sha2.cpp expect BYTE_ORDER from
# <sys/types.h>, which musl does not define there.
#
# MODULARIZE and INVOKE_RUN=0 hand the caller a factory instead of running main
# on load, so the net worker can be started before the engine does anything.
#
# NODERAWFS puts the engine on the real file descriptors, which is how it is
# tested under node without a page; a browser build drops it.
emcmake cmake -S "$KATAHEX/cpp" -B "$BUILD" \
  -DUSE_BACKEND="$BACKEND" -DNO_GIT_REVISION=1 -DCMAKE_BUILD_TYPE=Release \
  -DMAX_BOARD_LEN=19 \
  -DCMAKE_PREFIX_PATH="$EIGEN" -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
  -DZLIB_INCLUDE_DIR="$SYSROOT/include" \
  -DZLIB_LIBRARY="$SYSROOT/lib/wasm32-emscripten/libz.a" \
  -DCMAKE_CXX_FLAGS="-pthread -msimd128 -include endian.h -g0" \
  -DCMAKE_EXE_LINKER_FLAGS="-pthread -O3 -g0 \
    -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB -sSTACK_SIZE=8MB \
    -sPTHREAD_POOL_SIZE=8 -sEXIT_RUNTIME=1 \
    -sMODULARIZE=1 -sINVOKE_RUN=0 \
    -sEXPORTED_RUNTIME_METHODS=callMain,wasmMemory,HEAPF32,HEAP32 \
    -sNODERAWFS=1 -sENVIRONMENT=node,worker"

cmake --build "$BUILD" -j"$(nproc)"
