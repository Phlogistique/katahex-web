#!/bin/sh
# Cross-compile the KataHex engine to WebAssembly.
#
# Needs emsdk (https://github.com/emscripten-core/emsdk) and the Eigen headers,
# and a katahex checkout on a branch that carries the Emscripten cmake fixes.
set -eu

: "${EMSDK:?source emsdk_env.sh first}"
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
KATAHEX=${KATAHEX:-$ROOT/katahex}
EIGEN=${EIGEN:-$ROOT/third_party/eigen-prefix}
BUILD=${BUILD:-$ROOT/build-wasm}

# Emscripten's SSE compatibility headers do not cover everything Eigen reaches
# for, so this is a scalar build. It is a bring-up crutch: the engine is here for
# the board, the input features and the search, and the net is evaluated on the
# GPU from JavaScript.
emcmake cmake -S "$KATAHEX/cpp" -B "$BUILD" \
  -DUSE_BACKEND=EIGEN -DNO_GIT_REVISION=1 -DCMAKE_BUILD_TYPE=Release \
  -DMAX_BOARD_LEN=19 \
  -DCMAKE_PREFIX_PATH="$EIGEN" -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
  -DZLIB_INCLUDE_DIR="$EMSDK/upstream/emscripten/cache/sysroot/include" \
  -DZLIB_LIBRARY="$EMSDK/upstream/emscripten/cache/sysroot/lib/wasm32-emscripten/libz.a" \
  -DCMAKE_CXX_FLAGS="-pthread -msimd128 -include endian.h -g0" \
  -DCMAKE_EXE_LINKER_FLAGS="-pthread -O3 -g0 \
    -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB -sSTACK_SIZE=8MB \
    -sPTHREAD_POOL_SIZE=8 -sEXIT_RUNTIME=1 \
    -sNODERAWFS=1 -sENVIRONMENT=node,worker"

cmake --build "$BUILD" -j"$(nproc)"

# -include endian.h: numpywrite.cpp and sha2.cpp expect BYTE_ORDER from
# <sys/types.h>, which musl does not define there.
#
# NODERAWFS and ENVIRONMENT=node are for running the engine under node, which is
# how it is tested without a page. A browser build drops both and reads the net
# from a fetch instead.
