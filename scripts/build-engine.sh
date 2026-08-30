#!/bin/sh
# Cross-compile the KataHex engine to WebAssembly.
#
# Needs emsdk (https://github.com/emscripten-core/emsdk) sourced, and a katahex
# checkout on a branch carrying the Emscripten cmake fixes and the JS backend.
#
#   BACKEND=JS     the net is evaluated from JavaScript (see src/netRunner.ts)
#   BACKEND=EIGEN  the net is evaluated on the wasm CPU, for comparison; also
#                  needs Eigen headers under $EIGEN
#
#   TARGET=web     for the page: MEMFS, and stdin/stdout through the Module hooks
#   TARGET=node    for scripts/node-engine-test.mjs: the real file descriptors
set -eu

: "${EMSDK:?source emsdk_env.sh first}"
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
KATAHEX=${KATAHEX:-$ROOT/katahex}
EIGEN=${EIGEN:-$ROOT/third_party/eigen-prefix}
BACKEND=${BACKEND:-JS}
TARGET=${TARGET:-web}
BUILD=${BUILD:-$ROOT/build-wasm-$(echo "$BACKEND-$TARGET" | tr '[:upper:]' '[:lower:]')}
SYSROOT=$EMSDK/upstream/emscripten/cache/sysroot

# NODERAWFS puts the engine on the real file descriptors. A page has none, so it
# gets MEMFS and writes the model and the config into it.
if [ "$TARGET" = node ]; then
  ENVFLAGS="-sNODERAWFS=1 -sENVIRONMENT=node,worker"
  CXXEXTRA=
  POOL=16
  PROXY=
else
  # EXPORT_ES6 keeps the name katahex.js, which the file itself uses to start
  # its pthread workers, so it must keep it where it is served from too.
  ENVFLAGS="-sFORCE_FILESYSTEM=1 -sENVIRONMENT=web,worker -sEXPORT_ES6=1"
  CXXEXTRA=-DKATAHEX_QUERY_QUEUE
  # Enough workers for 2 x 32 search threads plus the serving and main threads;
  # a thread past the pool is created lazily through the busiest thread there is.
  POOL=80
  PROXY=-sPROXY_TO_PTHREAD=1
fi

# -include endian.h: numpywrite.cpp and sha2.cpp expect BYTE_ORDER from
# <sys/types.h>, which musl does not define there.
#
# MODULARIZE and INVOKE_RUN=0 hand the caller a factory instead of running main
# on load, so the net can be set up before the engine does anything.
#
# The page's build runs main() on a thread of its own (PROXY_TO_PTHREAD) and
# takes its queries by a call (KATAHEX_QUERY_QUEUE, see analysis.cpp). Output
# written by any of the engine's threads is proxied to the thread that loaded
# the module, so that one has to stay free, and main() spends its life blocked
# waiting for the next query.
emcmake cmake -S "$KATAHEX/cpp" -B "$BUILD" \
  -DUSE_BACKEND="$BACKEND" -DNO_GIT_REVISION=1 -DCMAKE_BUILD_TYPE=Release \
  -DMAX_BOARD_LEN=19 \
  -DCMAKE_PREFIX_PATH="$EIGEN" -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
  -DZLIB_INCLUDE_DIR="$SYSROOT/include" \
  -DZLIB_LIBRARY="$SYSROOT/lib/wasm32-emscripten/libz.a" \
  -DCMAKE_CXX_FLAGS="-pthread -msimd128 -include endian.h -g0 $CXXEXTRA" \
  -DCMAKE_EXE_LINKER_FLAGS="-pthread -O3 -g0 \
    -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB -sSTACK_SIZE=8MB \
    -sPTHREAD_POOL_SIZE=$POOL -sEXIT_RUNTIME=1 \
    -sMODULARIZE=1 -sINVOKE_RUN=0 $PROXY \
    -sEXPORTED_RUNTIME_METHODS=callMain,ccall,wasmMemory,HEAPF32,HEAP32,FS \
    $ENVFLAGS"

cmake --build "$BUILD" -j"$(nproc)"
