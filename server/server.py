#!/usr/bin/env python3
"""The KataHex analysis engine behind one websocket, on a rented GPU that rents itself back.

The page's engine bridge is `Native.start(size)`, `Native.query(json)` and the engine's two
output streams; this carries exactly that over a websocket, as JSON text frames:

    client -> server   {"t": "start", "size": 11}     (re)start the engine for a board size
                       {"t": "q", "q": "<query json>"} one line to the engine's stdin
                       {"t": "stop"}                   stop the engine
                       {"t": "ping"}                   I am still here
                       {"t": "bye"}                    I am done: destroy this machine now
    server -> client   {"t": "line", "l": "..."}       a line of the engine's stdout
                       {"t": "log", "l": "..."}        a line of the engine's stderr
                       {"t": "pong", "idle": s, "left": s}

Connect to ws://host:port/ws?token=TOKEN. GET /health answers "ok" without a token, so a
client can tell the machine is up before it has anything to say.

The dead man's switch: every message from a client counts as a sign of life. With none for
IDLE_TIMEOUT seconds (FIRST_CONNECT_TIMEOUT before the first one), or once the machine has been
up MAX_LIFETIME seconds, it destroys its own instance through the vast.ai API, with the
per-instance key vast puts in the container, and exits whatever the API said, which stops the
container and with it the GPU's meter. The client pings only while the app is on screen, so a
phone put away, killed or out of signal stops paying within IDLE_TIMEOUT.
"""

import asyncio
import hmac
import http
import json
import os
import signal
import sys
import time
import urllib.request
from urllib.parse import parse_qs, urlparse

from websockets.asyncio.server import serve

BIN = os.environ.get("KATAHEX_BIN", "/opt/katahex/katahex")
NET = os.environ.get("KATAHEX_NET", "/opt/katahex/net.bin.gz")
PORT = int(os.environ.get("PORT", "8765"))
TOKEN = os.environ.get("KATAHEX_TOKEN", "")
IDLE_TIMEOUT = float(os.environ.get("IDLE_TIMEOUT", "600"))
FIRST_CONNECT_TIMEOUT = float(os.environ.get("FIRST_CONNECT_TIMEOUT", "900"))
MAX_LIFETIME = float(os.environ.get("MAX_LIFETIME", str(12 * 3600)))
SEARCH_THREADS = int(os.environ.get("SEARCH_THREADS", "0"))  # 0: from the cpu count
# Started at boot, so that the client's first board is ready by the time it connects.
WARMUP_SIZE = int(os.environ.get("WARMUP_SIZE", "11"))
# Visits of a search run once the warmup engine is up, logged with its speed. 0: none.
BENCH_VISITS = int(os.environ.get("BENCH_VISITS", "0"))
VAST_API = os.environ.get("VAST_API", "https://console.vast.ai/api/v0")

STARTED = time.monotonic()
READY_LINE = "Started, ready to begin handling requests"


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def search_threads():
    if SEARCH_THREADS > 0:
        return SEARCH_THREADS
    # The GPU wants batches, which come from threads; past the cpu count they only contend.
    return max(4, min(32, 2 * (os.cpu_count() or 4)))


def config(size):
    threads = search_threads()
    return f"""
numAnalysisThreads = 2
numSearchThreadsPerAnalysisThread = {threads}
nnMaxBatchSize = {2 * threads}
numNNServerThreadsPerModel = 1
maxBoardXSizeForNNBuffer = {size}
maxBoardYSizeForNNBuffer = {size}
nnCacheSizePowerOfTwo = 21
nnMutexPoolSizePowerOfTwo = 16
reportAnalysisWinratesAs = SIDETOMOVE
useGraphSearch = true
cpuctExploration = 0.9
cpuctExplorationLog = 0.6
cpuctExplorationBase = 500
useNoisePruning = true
nnPolicyTemperature = 1.1
subtreeValueBiasFactor = 0.0
noResultUtilityForWhite = 0.0
analysisPVLen = 100
logToStderr = true
logDir = /tmp/katahex-logs
"""


class Hub:
    def __init__(self):
        self.clients = set()
        self.proc = None
        self.size = 0
        self.ready = False
        self.last_life = None  # monotonic time of the last client message, None before any
        self.benched = 0.0
        self.lock = asyncio.Lock()

    async def broadcast(self, kind, line):
        message = json.dumps({"t": kind, "l": line})
        for ws in list(self.clients):
            try:
                await ws.send(message)
            except Exception:
                self.clients.discard(ws)

    async def pump(self, proc, stream, kind):
        while True:
            raw = await stream.readline()
            if not raw:
                break
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if kind == "log":
                if READY_LINE in line and proc is self.proc:
                    self.ready = True
                    if BENCH_VISITS and not self.benched:
                        self.benched = time.monotonic()
                        self.write(json.dumps({
                            "id": "bench", "boardXSize": self.size, "boardYSize": self.size,
                            "initialStones": [], "initialPlayer": "B", "moves": [],
                            "rules": "tromp-taylor", "komi": 0, "maxVisits": BENCH_VISITS}))
                log("engine:", line[:300])
            elif '"id":"reconnect"' in line:
                continue  # the server's own terminate_all, nobody's business
            elif '"id":"bench"' in line:
                reply = json.loads(line)
                visits = reply.get("rootInfo", {}).get("visits", 0)
                seconds = time.monotonic() - self.benched
                best = [(m["move"], m["visits"]) for m in reply.get("moveInfos", [])[:3]]
                log(f"bench: {visits} visits in {seconds:.1f}s, {visits / seconds:.0f}/s, best {best}")
                continue
            await self.broadcast(kind, line)

    async def watch_exit(self, proc):
        code = await proc.wait()
        if proc is self.proc:
            self.proc = None
            self.ready = False
            await self.broadcast("log", f"Error: engine terminating, exit code {code}")
        log("engine exited", code)

    async def start(self, size):
        async with self.lock:
            if self.proc and self.size == size and self.proc.returncode is None:
                # A client that reconnects asks for the board it was on: keep the engine and
                # its cache, drop whatever the client before it left searching.
                self.write(json.dumps({"id": "reconnect", "action": "terminate_all"}))
                if self.ready:
                    await self.broadcast("log", READY_LINE)
                return
            await self.stop_locked()
            path = f"/tmp/analysis-{size}.cfg"
            with open(path, "w") as f:
                f.write(config(size))
            log("starting engine for", size, "with", search_threads(), "search threads")
            self.size = size
            self.ready = False
            self.proc = await asyncio.create_subprocess_exec(
                BIN, "analysis", "-config", path, "-model", NET,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, limit=64 * 1024 * 1024)
            await self.broadcast("log", f"starting engine for {size}x{size}, loading the net on the GPU")
            asyncio.create_task(self.pump(self.proc, self.proc.stdout, "line"))
            asyncio.create_task(self.pump(self.proc, self.proc.stderr, "log"))
            asyncio.create_task(self.watch_exit(self.proc))

    def write(self, line):
        if self.proc and self.proc.returncode is None:
            self.proc.stdin.write(line.encode() + b"\n")

    async def stop_locked(self):
        proc, self.proc = self.proc, None
        self.ready = False
        if proc and proc.returncode is None:
            proc.kill()
            await proc.wait()

    async def stop(self):
        async with self.lock:
            await self.stop_locked()


hub = Hub()


def destroy_self(reason):
    """Destroys this vast.ai instance, then exits, which stops the container if that failed."""
    log("dead man's switch:", reason)
    instance = os.environ.get("CONTAINER_ID")
    keys = [k for k in (os.environ.get("CONTAINER_API_KEY"), os.environ.get("VAST_API_KEY")) if k]
    if instance:
        for key in keys:
            for method, body in (("DELETE", None), ("PUT", b'{"state": "stopped"}')):
                request = urllib.request.Request(
                    f"{VAST_API}/instances/{instance}/", method=method, data=body,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(request, timeout=20) as response:
                        log(method, "instance", instance, "->", response.status, response.read()[:200])
                    os._exit(0)
                except Exception as error:
                    log(method, "instance", instance, "failed:", error)
    else:
        log("no CONTAINER_ID: not on vast.ai, just exiting")
    os._exit(0)


async def dead_mans_switch():
    while True:
        await asyncio.sleep(10)
        now = time.monotonic()
        if MAX_LIFETIME > 0 and now - STARTED > MAX_LIFETIME:
            destroy_self(f"up for {MAX_LIFETIME:.0f}s, the most it may be")
        if hub.last_life is None:
            if now - STARTED > FIRST_CONNECT_TIMEOUT:
                destroy_self(f"no client in the first {FIRST_CONNECT_TIMEOUT:.0f}s")
        elif now - hub.last_life > IDLE_TIMEOUT:
            destroy_self(f"no sign of life for {IDLE_TIMEOUT:.0f}s")


def authorized(path):
    given = parse_qs(urlparse(path).query).get("token", [""])[0]
    return bool(TOKEN) and hmac.compare_digest(given.encode(), TOKEN.encode())


def process_request(connection, request):
    path = urlparse(request.path).path
    if path == "/health":
        return connection.respond(http.HTTPStatus.OK, "ok\n")
    if path != "/ws":
        return connection.respond(http.HTTPStatus.NOT_FOUND, "not found\n")
    if not authorized(request.path):
        return connection.respond(http.HTTPStatus.FORBIDDEN, "bad token\n")
    return None


def left():
    now = time.monotonic()
    idle_left = IDLE_TIMEOUT - (now - (hub.last_life or now))
    life_left = MAX_LIFETIME - (now - STARTED) if MAX_LIFETIME > 0 else idle_left
    return round(min(idle_left, life_left))


async def handle(ws):
    hub.clients.add(ws)
    hub.last_life = time.monotonic()
    log("client connected", ws.remote_address)
    try:
        async for raw in ws:
            hub.last_life = time.monotonic()
            try:
                message = json.loads(raw)
                kind = message.get("t")
            except Exception:
                continue
            if kind == "q":
                hub.write(message["q"])
            elif kind == "start":
                size = int(message["size"])
                if 2 <= size <= 19:
                    await hub.start(size)
                else:
                    await ws.send(json.dumps({"t": "log", "l": f"Error: board size {size} is out of range"}))
            elif kind == "stop":
                await hub.stop()
            elif kind == "ping":
                await ws.send(json.dumps({"t": "pong", "left": left(), "ready": hub.ready, "size": hub.size}))
            elif kind == "bye":
                await hub.stop()
                destroy_self("the client said bye")
    except Exception as error:
        log("client error", error)
    finally:
        hub.clients.discard(ws)
        log("client gone", ws.remote_address)


async def main():
    if not TOKEN:
        sys.exit("KATAHEX_TOKEN must be set")
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, lambda: os._exit(0))
    asyncio.create_task(dead_mans_switch())
    if 2 <= WARMUP_SIZE <= 19:
        await hub.start(WARMUP_SIZE)
    async with serve(handle, "0.0.0.0", PORT, process_request=process_request,
                     max_size=16 * 1024 * 1024, ping_interval=20, ping_timeout=60):
        log(f"listening on {PORT}; idle {IDLE_TIMEOUT:.0f}s, first connect "
            f"{FIRST_CONNECT_TIMEOUT:.0f}s, lifetime {MAX_LIFETIME:.0f}s")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
