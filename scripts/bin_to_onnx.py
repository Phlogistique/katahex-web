"""Convert a KataGo/KataHex .bin.gz net to ONNX.

The existing converters want a PyTorch checkpoint and predate nested bottleneck
blocks, so this reads the weight file itself. The format is KataGo's
cpp/neuralnet/desc.cpp: whitespace-separated tokens, with float arrays inline
after an @BIN@ marker. The graph mirrors vendor/modelV8.ts, which is the
implementation the outputs are checked against.

The board size is baked in. KataGo's global pooling scales by the board size and
the net has no mask here, so a graph is only correct for the size it was exported
for -- the same constraint the TensorFlow.js path has.

    python scripts/bin_to_onnx.py ../hex27x3.bin.gz build/hex27x3-11.onnx --size 11
"""

import argparse
import gzip
import math
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

WHITESPACE = b" \n\r\t"


class Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.idx = 0

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
        if self.data[self.idx : self.idx + 5] != b"@BIN@":
            raise ValueError(f"expected @BIN@ at {self.idx}")
        self.idx += 5
        end = self.idx + count * 4
        out = np.frombuffer(self.data[self.idx : end], dtype="<f4").copy()
        if out.size != count:
            raise EOFError("binary floats")
        self.idx = end
        return out


def read_batch_norm(r: Reader) -> tuple[np.ndarray, np.ndarray]:
    """Returns the batch norm folded into a per-channel scale and bias."""
    r.token()
    channels = r.int()
    epsilon = r.float()
    has_scale = r.int() != 0
    has_bias = r.int() != 0
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
    weights = r.floats(ky * kx * in_c * out_c).reshape(ky, kx, in_c, out_c)
    return np.ascontiguousarray(weights.transpose(3, 2, 0, 1))


def read_matmul(r: Reader) -> np.ndarray:
    r.token()
    in_c, out_c = r.int(), r.int()
    return r.floats(in_c * out_c).reshape(in_c, out_c)


def read_mat_bias(r: Reader) -> np.ndarray:
    r.token()
    return r.floats(r.int())


class Graph:
    """Accumulates ONNX nodes and the constants they refer to."""

    def __init__(self, dtype=np.float32):
        self.nodes: list = []
        self.initializers: list = []
        self.counter = 0
        self.dtype = dtype

    def name(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def constant(self, array: np.ndarray, prefix: str = "const") -> str:
        name = self.name(prefix)
        self.initializers.append(numpy_helper.from_array(array.astype(self.dtype), name))
        return name

    def cast(self, x: str, to) -> str:
        return self.op("Cast", [x], "cast", to=to)

    def rename(self, x: str, name: str) -> str:
        self.nodes.append(helper.make_node("Identity", [x], [name], name=self.name("node")))
        return name

    def op(self, kind: str, inputs: list[str], prefix: str | None = None, **attrs) -> str:
        out = self.name(prefix or kind.lower())
        self.nodes.append(helper.make_node(kind, inputs, [out], name=self.name("node"), **attrs))
        return out

    def conv(self, x: str, kernel: np.ndarray) -> str:
        pad = (kernel.shape[2] // 2, kernel.shape[3] // 2)
        return self.op("Conv", [x, self.constant(kernel, "kernel")], "conv",
                       kernel_shape=[kernel.shape[2], kernel.shape[3]],
                       pads=[pad[0], pad[1], pad[0], pad[1]])

    def channel_affine(self, x: str, scale: np.ndarray, bias: np.ndarray) -> str:
        """Per-channel scale and bias over an NCHW tensor."""
        shaped = lambda a: a.reshape(1, -1, 1, 1)
        scaled = self.op("Mul", [x, self.constant(shaped(scale), "bnscale")])
        return self.op("Add", [scaled, self.constant(shaped(bias), "bnbias")])

    def activation(self, x: str, kind: str) -> str:
        if kind == "identity":
            return x
        if kind == "relu":
            return self.op("Relu", [x])
        # mish, spelled out: onnxruntime's web backends are patchier about the
        # fused op than about these three.
        return self.op("Mul", [x, self.op("Tanh", [self.op("Softplus", [x])])])

    def bn_act(self, x: str, bn: tuple[np.ndarray, np.ndarray], kind: str) -> str:
        return self.activation(self.channel_affine(x, *bn), kind)

    def broadcast_channels(self, x: str, rows: str) -> str:
        """Adds a per-row, per-channel vector [N,C] to an NCHW tensor."""
        shape = self.name("shape")
        self.initializers.append(numpy_helper.from_array(np.array([0, -1, 1, 1], dtype=np.int64), shape))
        return self.op("Add", [x, self.op("Reshape", [rows, shape])])


def pool_gpool(g: Graph, x: str, board_size: int) -> str:
    """KataGo's gpool: concat(mean, mean * (boardSize - 14) * 0.1, max)."""
    factor = (board_size - 14) * 0.1
    mean = g.op("ReduceMean", [x], axes=[2, 3], keepdims=0)
    biggest = g.op("ReduceMax", [x], axes=[2, 3], keepdims=0)
    scaled = g.op("Mul", [mean, g.constant(np.array([factor], dtype=np.float32), "gpoolfactor")])
    return g.op("Concat", [mean, scaled, biggest], axis=1)


def pool_value(g: Graph, x: str, board_size: int) -> str:
    """KataGo's value pooling: concat(mean, mean * f1, mean * f2)."""
    base = board_size - 14
    f1 = base * 0.1
    f2 = base * base * 0.01 - 0.1
    mean = g.op("ReduceMean", [x], axes=[2, 3], keepdims=0)
    a = g.op("Mul", [mean, g.constant(np.array([f1], dtype=np.float32), "valuefactor")])
    b = g.op("Mul", [mean, g.constant(np.array([f2], dtype=np.float32), "valuefactor")])
    return g.op("Concat", [mean, a, b], axis=1)


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


def parse(data: bytes) -> dict:
    r = Reader(data)
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


def build_blocks(g: Graph, trunk: str, blocks: list[dict], board_size: int) -> str:
    for block in blocks:
        a = g.bn_act(trunk, block["pre_bn"], block["pre_act"])
        if block["kind"] == "ordinary":
            b = g.conv(a, block["w1"])
            c = g.bn_act(b, block["mid_bn"], block["mid_act"])
            residual = g.conv(c, block["w2"])
        elif block["kind"] == "gpool":
            regular = g.conv(a, block["w1a"])
            pooled = g.bn_act(g.conv(a, block["w1b"]), block["gpool_bn"], block["gpool_act"])
            bias = g.op("MatMul", [pool_gpool(g, pooled, board_size), g.constant(block["w1r"], "w1r")])
            regular = g.broadcast_channels(regular, bias)
            c = g.bn_act(regular, block["mid_bn"], block["mid_act"])
            residual = g.conv(c, block["w2"])
        else:
            mid = g.conv(a, block["pre_conv"])
            mid = build_blocks(g, mid, block["blocks"], board_size)
            c = g.bn_act(mid, block["post_bn"], block["post_act"])
            residual = g.conv(c, block["post_conv"])
        trunk = g.op("Add", [trunk, residual], "trunk")
    return trunk


def build(model: dict, board_size: int, half: bool = False) -> onnx.ModelProto:
    g = Graph(np.float16 if half else np.float32)
    channels = model["num_input_channels"]

    # The engine fills NHWC buffers, so the graph takes them that way and
    # transposes once; onnxruntime folds this into the first convolution.
    # Inputs and outputs stay float32 whatever the graph computes in, so the
    # caller does not have to know.
    spatial_nhwc = "spatial"
    global_input = "global"
    spatial = g.op("Transpose", [spatial_nhwc], "spatial_nchw", perm=[0, 3, 1, 2])
    global_rows = global_input
    if half:
        spatial = g.cast(spatial, TensorProto.FLOAT16)
        global_rows = g.cast(global_input, TensorProto.FLOAT16)

    trunk = g.conv(spatial, model["conv1"])
    ginput = g.op("MatMul", [global_rows, g.constant(model["ginput"], "ginput")])
    trunk = g.broadcast_channels(trunk, ginput)
    trunk = build_blocks(g, trunk, model["blocks"], board_size)
    trunk = g.bn_act(trunk, model["tip_bn"], model["tip_act"])

    p1 = g.conv(trunk, model["p1"])
    g1 = g.bn_act(g.conv(trunk, model["g1"]), model["g1_bn"], model["g1_act"])
    g1_concat = pool_gpool(g, g1, board_size)
    p1 = g.broadcast_channels(p1, g.op("MatMul", [g1_concat, g.constant(model["gpool_to_bias"], "gpooltobias")]))
    p1 = g.bn_act(p1, model["p1_bn"], model["p1_act"])
    policy_map = g.conv(p1, model["p2"])

    flat = g.name("policyshape")
    g.initializers.append(numpy_helper.from_array(np.array([0, board_size * board_size], dtype=np.int64), flat))
    policy = g.op("Reshape", [policy_map, flat], "policy")
    policy_pass = g.op("MatMul", [g1_concat, g.constant(model["pass_mul"], "passmul")], "policy_pass")

    v1 = g.bn_act(g.conv(trunk, model["v1"]), model["v1_bn"], model["v1_act"])
    v2 = g.op("Add", [g.op("MatMul", [pool_value(g, v1, board_size), g.constant(model["v2"], "v2")]),
                      g.constant(model["v2_bias"].reshape(1, -1), "v2bias")])
    v2 = g.activation(v2, model["v2_act"])
    value = g.op("Add", [g.op("MatMul", [v2, g.constant(model["v3"], "v3")]),
                         g.constant(model["v3_bias"].reshape(1, -1), "v3bias")], "value")
    score_value = g.op("Add", [g.op("MatMul", [v2, g.constant(model["sv3"], "sv3")]),
                               g.constant(model["sv3_bias"].reshape(1, -1), "sv3bias")], "score_value")

    def tensor(name, dims):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)

    if half:
        policy, policy_pass, value, score_value = (
            g.cast(t, TensorProto.FLOAT) for t in (policy, policy_pass, value, score_value))
    policy = g.rename(policy, "policy")
    policy_pass = g.rename(policy_pass, "policyPass")
    value = g.rename(value, "value")
    score_value = g.rename(score_value, "scoreValue")

    graph = helper.make_graph(
        g.nodes,
        model["name"],
        [tensor(spatial_nhwc, ["N", board_size, board_size, channels]),
         tensor(global_input, ["N", model["num_input_global_channels"]])],
        [tensor(policy, ["N", board_size * board_size]),
         tensor(policy_pass, ["N", 1]),
         tensor(value, ["N", 3]),
         tensor(score_value, ["N", 6])],
        g.initializers,
    )
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    proto.ir_version = 9
    onnx.checker.check_model(proto)
    return proto


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--fp16", action="store_true", help="compute in half precision")
    args = ap.parse_args()

    raw = pathlib.Path(args.model).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    model = parse(raw)
    print(f"{model['name']}: version {model['version']}, {len(model['blocks'])} blocks, "
          f"{model['trunk_channels']} channels")
    proto = build(model, args.size, args.fp16)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(proto, str(out))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
