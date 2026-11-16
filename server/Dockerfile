# KataHex analysis engine behind a websocket, for renting a GPU by the hour.
#
#   docker build -t katahex-server server/
#   docker build --build-arg BACKEND=EIGEN -t katahex-server:cpu server/   # no GPU, for testing
#
# The engine is upstream KataHex (hzyhhzy/KataGomo, branch Hex2024) rather than
# the fork under engine/: the fork's changes are for WebAssembly, and upstream
# has moved on since. The net is the strongest one HZY has released.
ARG CUDA_IMAGE=nvidia/cuda:12.2.2-cudnn8

FROM ${CUDA_IMAGE}-devel-ubuntu22.04 AS build
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git cmake make g++ zlib1g-dev libeigen3-dev ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

ARG KATAHEX_REPO=https://github.com/hzyhhzy/KataGomo
ARG KATAHEX_BRANCH=Hex2024
ARG KATAHEX_COMMIT=4bbf953596e96a78daa2008811187a61381a886f
# Upstream writes Hex cells as a Go board would print them (for Sabaki); the ui speaks
# HexGUI's a1..k11, as Peter Selinger's fork under engine/ does. The patch is that fork's.
COPY hexgui-coords.patch /hexgui-coords.patch
RUN git clone --branch "$KATAHEX_BRANCH" --single-branch "$KATAHEX_REPO" /src \
    && git -C /src checkout "$KATAHEX_COMMIT" \
    && git -C /src apply /hexgui-coords.patch

ARG BACKEND=CUDA
RUN cmake -S /src/cpp -B /build -DUSE_BACKEND=$BACKEND -DMAX_BOARD_LEN=19 \
      -DNO_GIT_REVISION=1 -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /build -j"$(nproc)"

# KataHex 20250131: retrained b28, the strongest net up to 27x27 (no move limit mode).
ARG NET_URL=https://github.com/hzyhhzy/KataGomo/releases/download/Hex_20250131/hex3_27x_b28.bin.gz
ARG NET_SHA256=d8592b43713d08dd68f8fdac70f0450e9089ac663edd221af2591bc1dc3c8f86
RUN curl -fsSL -o /net.bin.gz "$NET_URL" && echo "$NET_SHA256  /net.bin.gz" | sha256sum -c -

FROM ${CUDA_IMAGE}-runtime-ubuntu22.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3 python3-pip zlib1g libgomp1 ca-certificates \
    && pip3 install --no-cache-dir websockets==13.1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /build/katago /opt/katahex/katahex
COPY --from=build /net.bin.gz /opt/katahex/net.bin.gz
COPY server.py /opt/katahex/server.py

ENV PORT=8765 PYTHONUNBUFFERED=1
EXPOSE 8765
WORKDIR /tmp
ENTRYPOINT ["python3", "/opt/katahex/server.py"]
