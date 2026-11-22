# PlayHex AI workers (playhex/hex-ai-distributed) with KataHex on CUDA, for renting a GPU by
# the hour, with the same dead man's switch as server/.
#
#   docker build -t playhex-worker worker/
#   docker build --build-arg BACKEND=EIGEN -t playhex-worker:cpu worker/   # no GPU, for testing
ARG CUDA=12.2.2

FROM node:22-slim AS app
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ARG WORKER_COMMIT=087316d018fd0a0941001068e1b3cb2b4072b438
WORKDIR /app
RUN git init -q . && git fetch -q --depth 1 https://github.com/playhex/hex-ai-distributed "$WORKER_COMMIT" \
    && git checkout -q FETCH_HEAD \
    && yarn install --frozen-lockfile && yarn tsc && cp .env.dist dist/ \
    && yarn install --frozen-lockfile --production --ignore-scripts && yarn cache clean

FROM nvidia/cuda:${CUDA}-cudnn8-devel-ubuntu22.04 AS build
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git cmake make g++ zlib1g-dev libeigen3-dev ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# PlayHex's fork: Selinger's HexGUI coordinates and the kata-raw-nn-batch command game analyzes need.
ARG KATAHEX_COMMIT=5d3773913e79240fd2f36c45f1acd58d7eb68860
RUN git init -q /src && git -C /src fetch -q --depth 1 https://github.com/playhex/katahex "$KATAHEX_COMMIT" \
    && git -C /src checkout -q FETCH_HEAD

ARG BACKEND=CUDA
RUN cmake -S /src/cpp -B /build -DUSE_BACKEND=$BACKEND -DMAX_BOARD_LEN=32 \
      -DNO_GIT_REVISION=1 -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /build -j"$(nproc)"

# The net PlayHex's own workers run, so that its bots play the same whichever worker answers.
ARG NET_URL=https://drive.usercontent.google.com/download?id=1xMvP_75xgo0271nQbmlAJ40rvpKiFTgP&export=download&confirm=t
ARG NET_SHA256=0d6bcbbb4529778f6e18b234b7b7edc568e3cdf8b08ea7eb0f35e65e87ef1705
RUN curl -fsSL -o /net.bin.gz "$NET_URL" && echo "$NET_SHA256  /net.bin.gz" | sha256sum -c -

# Same libraries as server/Dockerfile, each its own layer.
RUN mkdir -p /libs/cudart /libs/cublaslt /libs/cublas /libs/cudnn-cnn /libs/cudnn \
    && cp -L /usr/local/cuda/lib64/libcudart.so.12 /libs/cudart/ \
    && cp -L /usr/local/cuda/lib64/libcublasLt.so.12 /libs/cublaslt/ \
    && cp -L /usr/local/cuda/lib64/libcublas.so.12 /libs/cublas/ \
    && cp -L /usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.so.8 /libs/cudnn-cnn/ \
    && cp -L /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/lib/x86_64-linux-gnu/libcudnn_ops_infer.so.8 /libs/cudnn/

FROM nvidia/cuda:${CUDA}-base-ubuntu22.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      zlib1g libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /libs/cublaslt/ /opt/katahex/lib/
COPY --from=build /libs/cudnn-cnn/ /opt/katahex/lib/
COPY --from=build /net.bin.gz /app/katahex/model.bin.gz
COPY --from=build /libs/cublas/ /opt/katahex/lib/
COPY --from=build /libs/cudnn/ /opt/katahex/lib/
COPY --from=build /libs/cudart/ /opt/katahex/lib/
COPY --from=build /build/katahex /usr/local/bin/katahex
COPY --from=app /usr/local/bin/node /usr/local/bin/node
COPY --from=app /app/node_modules /app/node_modules
COPY --from=app /app/package.json /app/package.json
COPY --from=app /app/katahex/config.cfg /app/katahex/config.cfg
COPY --from=app /app/dist /app/dist
COPY supervisor.mjs /app/supervisor.mjs

# Every katahex job type, tree search included: that is what a GPU is for.
ENV PORT=8765 WORKERS=2 ENGINE=katahex HEX_URL=https://playhex.org \
    AI_JOB_TYPES=katahex-intuition-move,katahex-mcts-move,katahex-intuition-analyze-position,katahex-mcts-analyze-position,katahex-intuition-analyze-game,katahex-mcts-analyze-move \
    KATAHEX_BIN="/usr/local/bin/katahex gtp -config /app/katahex/config.cfg -model /app/katahex/model.bin.gz -override-config numSearchThreads=8,nnMaxBatchSize=64,nnCacheSizePowerOfTwo=21,gtpMinNNSize=14" \
    LD_LIBRARY_PATH=/opt/katahex/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
EXPOSE 8765
WORKDIR /app
ENTRYPOINT ["node", "/app/supervisor.mjs"]
