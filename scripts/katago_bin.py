"""Reads KataGo's .bin weight format, the one cpp/neuralnet/desc.cpp writes.

Whitespace-separated tokens, with float arrays inline after an @BIN@ marker.
`parse` walks the whole file; what it returns is shaped for the ONNX converter,
but the reader also records where every float array sits, which is what
`export_net.py` rewrites the file from.
"""

import gzip
import pathlib

import numpy as np

WHITESPACE = b" \n\r\t"

#: Marker introducing a float32 array, and the one export_net.py writes instead
#: for a float16 array. Both are understood by vendor/binModelParser.ts.
BIN32 = b"@BIN@"
BIN16 = b"@BINF16@"


class Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.idx = 0
        #: Every float array read, as (start, count, kind), in file order. The
        #: start is the first byte after the marker. `kind` is the layer that
        #: asked for it, which is how the exporter tells weights from the batch
        #: norm statistics it has to leave alone.
        self.arrays: list[tuple[int, int, str]] = []
        self.kind = "?"

    def _skip_whitespace(self) -> None:
        while self.idx < len(self.data) and self.data[self.idx] in WHITESPACE:
            self.idx += 1

    def token(self) -> str:
        self._skip_whitespace()
        start = self.idx
        while self.idx < len(self.data) and self.data[self.idx] not in WHITESPACE:
            self.idx += 1
        if self.idx <= start:
            raise EOFError("token")
        return self.data[start : self.idx].decode()

    def int(self) -> int:
        return int(self.token())

    def float(self) -> float:
        return float(self.token())

    def floats(self, count: int) -> np.ndarray:
        self._skip_whitespace()
        if self.data[self.idx : self.idx + len(BIN32)] != BIN32:
            raise ValueError(f"expected @BIN@ at {self.idx}")
        self.idx += len(BIN32)
        end = self.idx + count * 4
        out = np.frombuffer(self.data[self.idx : end], dtype="<f4").copy()
        if out.size != count:
            raise EOFError("binary floats")
        self.arrays.append((self.idx, count, self.kind))
        self.idx = end
        return out

    def at_end(self) -> bool:
        self._skip_whitespace()
        return self.idx >= len(self.data)


def read_batch_norm(r: Reader) -> tuple[np.ndarray, np.ndarray]:
    """Returns the batch norm folded into a per-channel scale and bias."""
    r.token()
    channels = r.int()
    epsilon = r.float()
    has_scale = r.int() != 0
    has_bias = r.int() != 0
    r.kind = "batchnorm"
    mean = r.floats(channels)
    variance = r.floats(channels)
    scale = r.floats(channels) if has_scale else np.ones(channels, dtype=np.float32)
    bias = r.floats(channels) if has_bias else np.zeros(channels, dtype=np.float32)
    merged_scale = scale / np.sqrt(variance + epsilon)
    return merged_scale.astype(np.float32), (bias - merged_scale * mean).astype(np.float32)


def read_activation(r: Reader, model_version: int) -> str:
    r.token()  # name
    if model_version < 11:
        return "relu"
    kind = r.token()
    return {"ACTIVATION_IDENTITY": "identity", "ACTIVATION_RELU": "relu", "ACTIVATION_MISH": "mish"}[kind]


def read_conv(r: Reader) -> np.ndarray:
    """Returns the kernel already in ONNX's [outC, inC, kY, kX] order."""
    r.token()  # name
    ky, kx, in_c, out_c = r.int(), r.int(), r.int(), r.int()
    dilation_y, dilation_x = r.int(), r.int()
    if dilation_y != 1 or dilation_x != 1:
        raise ValueError("dilated convolutions are not handled")
    r.kind = "conv"
    weights = r.floats(ky * kx * in_c * out_c).reshape(ky, kx, in_c, out_c)
    return np.ascontiguousarray(weights.transpose(3, 2, 0, 1))


def read_matmul(r: Reader) -> np.ndarray:
    r.token()
    in_c, out_c = r.int(), r.int()
    r.kind = "matmul"
    return r.floats(in_c * out_c).reshape(in_c, out_c)


def read_mat_bias(r: Reader) -> np.ndarray:
    r.token()
    channels = r.int()
    r.kind = "matbias"
    return r.floats(channels)


def read_block(r: Reader, model_version: int) -> dict:
    kind = r.token()
    if kind == "ordinary_block":
        r.token()
        return {
            "kind": "ordinary",
            "pre_bn": read_batch_norm(r), "pre_act": read_activation(r, model_version),
            "w1": read_conv(r),
            "mid_bn": read_batch_norm(r), "mid_act": read_activation(r, model_version),
            "w2": read_conv(r),
        }
    if kind == "gpool_block":
        r.token()
        return {
            "kind": "gpool",
            "pre_bn": read_batch_norm(r), "pre_act": read_activation(r, model_version),
            "w1a": read_conv(r), "w1b": read_conv(r),
            "gpool_bn": read_batch_norm(r), "gpool_act": read_activation(r, model_version),
            "w1r": read_matmul(r),
            "mid_bn": read_batch_norm(r), "mid_act": read_activation(r, model_version),
            "w2": read_conv(r),
        }
    if kind == "nested_bottleneck_block":
        r.token()
        count = r.int()
        pre_bn = read_batch_norm(r)
        pre_act = read_activation(r, model_version)
        pre_conv = read_conv(r)
        inner = [read_block(r, model_version) for _ in range(count)]
        post_bn = read_batch_norm(r)
        post_act = read_activation(r, model_version)
        post_conv = read_conv(r)
        return {
            "kind": "nested", "pre_bn": pre_bn, "pre_act": pre_act, "pre_conv": pre_conv,
            "blocks": inner, "post_bn": post_bn, "post_act": post_act, "post_conv": post_conv,
        }
    raise ValueError(f"unsupported block kind {kind}")


def parse(data: bytes, reader: Reader | None = None) -> dict:
    r = reader if reader is not None else Reader(data)
    name = r.token()
    model_version = r.int()
    if not 8 <= model_version <= 14:
        raise ValueError(f"unsupported model version {model_version}")
    num_input_channels = r.int()
    num_input_global_channels = r.int()
    if model_version >= 13:
        for _ in range(7):
            r.float()
    r.token()  # trunk name
    num_blocks = r.int()
    trunk_channels = r.int()
    r.int()  # mid channels
    r.int()  # regular channels
    r.int()
    r.int()  # gpool channels

    model = {
        "name": name,
        "version": model_version,
        "num_input_channels": num_input_channels,
        "num_input_global_channels": num_input_global_channels,
        "trunk_channels": trunk_channels,
        "conv1": read_conv(r),
        "ginput": read_matmul(r),
    }
    model["blocks"] = [read_block(r, model_version) for _ in range(num_blocks)]
    model["tip_bn"] = read_batch_norm(r)
    model["tip_act"] = read_activation(r, model_version)

    r.token()  # policy head name
    model["p1"] = read_conv(r)
    model["g1"] = read_conv(r)
    model["g1_bn"] = read_batch_norm(r)
    model["g1_act"] = read_activation(r, model_version)
    model["gpool_to_bias"] = read_matmul(r)
    model["p1_bn"] = read_batch_norm(r)
    model["p1_act"] = read_activation(r, model_version)
    model["p2"] = read_conv(r)
    model["pass_mul"] = read_matmul(r)

    r.token()  # value head name
    model["v1"] = read_conv(r)
    model["v1_bn"] = read_batch_norm(r)
    model["v1_act"] = read_activation(r, model_version)
    model["v2"] = read_matmul(r)
    model["v2_bias"] = read_mat_bias(r)
    model["v2_act"] = read_activation(r, model_version)
    model["v3"] = read_matmul(r)
    model["v3_bias"] = read_mat_bias(r)
    model["sv3"] = read_matmul(r)
    model["sv3_bias"] = read_mat_bias(r)
    model["ownership"] = read_conv(r)
    return model


def read_maybe_gzipped(path: str | pathlib.Path) -> bytes:
    raw = pathlib.Path(path).read_bytes()
    return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
