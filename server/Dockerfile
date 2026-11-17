# KataHex analysis engine behind a websocket, for renting a GPU by the hour.
#
#   docker build -t katahex-server server/
#   docker build --build-arg BACKEND=EIGEN -t katahex-server:cpu server/   # no GPU, for testing
#
# The engine is upstream KataHex (hzyhhzy/KataGomo, branch Hex2024) rather than
# the fork under engine/: the fork's changes are for WebAssembly, and upstream
# has moved on since. The net is the strongest one HZY has released.
ARG CUDA=12.2.2

FROM nvidia/cuda:${CUDA}-cudnn8-devel-ubuntu22.04 AS build
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git cmake make g++ zlib1g-dev libeigen3-dev ca-certificates curl python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir --target /py websockets==14.2

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

# Only what the engine loads, each library its own layer so a host pulls them in parallel.
# The cudnn8-runtime image carries every CUDA library and all of cuDNN, training included:
# 2.4 GB compressed, 1.3 GB of it one layer, which is most of a slow host's start.
RUN mkdir -p /libs/cudart /libs/cublaslt /libs/cublas /libs/cudnn-cnn /libs/cudnn \
    && cp -L /usr/local/cuda/lib64/libcudart.so.12 /libs/cudart/ \
    && cp -L /usr/local/cuda/lib64/libcublasLt.so.12 /libs/cublaslt/ \
    && cp -L /usr/local/cuda/lib64/libcublas.so.12 /libs/cublas/ \
    && cp -L /usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.so.8 /libs/cudnn-cnn/ \
    && cp -L /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/lib/x86_64-linux-gnu/libcudnn_ops_infer.so.8 /libs/cudnn/

FROM nvidia/cuda:${CUDA}-base-ubuntu22.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3 zlib1g libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /libs/cublaslt/ /opt/katahex/lib/
COPY --from=build /libs/cudnn-cnn/ /opt/katahex/lib/
COPY --from=build /net.bin.gz /opt/katahex/net.bin.gz
COPY --from=build /libs/cublas/ /opt/katahex/lib/
COPY --from=build /libs/cudnn/ /opt/katahex/lib/
COPY --from=build /libs/cudart/ /opt/katahex/lib/
COPY --from=build /build/katago /opt/katahex/katahex
COPY --from=build /py /opt/katahex/py
COPY server.py /opt/katahex/server.py

ENV PORT=8765 PYTHONUNBUFFERED=1 PYTHONPATH=/opt/katahex/py \
    LD_LIBRARY_PATH=/opt/katahex/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
EXPOSE 8765
WORKDIR /tmp
ENTRYPOINT ["python3", "/opt/katahex/server.py"]
