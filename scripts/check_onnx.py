"""Check the ONNX graphs the converter builds.

Runs the float32 and float16 graphs in onnxruntime on one fixed input, prints how
far apart they are, and writes the input and the float32 outputs to a directory
for `check-onnx-tfjs.ts`, which runs the same input through the TensorFlow.js
implementation the converter was written from.

    uv run --with onnx --with onnxruntime --with numpy \
      python scripts/check_onnx.py ../hex27x3.bin.gz --size 11 --out build/check
"""

import argparse
import gzip
import json
import pathlib

import numpy as np
import onnxruntime as ort

from bin_to_onnx import build, parse

NUM_SPATIAL_FEATURES = 22
NUM_GLOBAL_FEATURES = 19
OUTPUTS = ["policy", "policyPass", "value", "scoreValue"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--size", type=int, default=11)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", default="build/check")
    args = ap.parse_args()

    raw = pathlib.Path(args.model).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    model = parse(raw)

    rng = np.random.default_rng(7)
    spatial = rng.standard_normal(
        (args.batch, args.size, args.size, NUM_SPATIAL_FEATURES)).astype(np.float32)
    spatial[:, :, :, 0] = 1.0  # the on-board mask channel
    global_input = rng.standard_normal((args.batch, NUM_GLOBAL_FEATURES)).astype(np.float32)
    feed = {"spatial": spatial, "global": global_input}

    def run(half: bool) -> dict[str, np.ndarray]:
        proto = build(model, args.size, half)
        session = ort.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])
        return dict(zip(OUTPUTS, session.run(OUTPUTS, feed)))

    single = run(False)
    half = run(True)
    for name in OUTPUTS:
        a, b = np.ravel(single[name]), np.ravel(half[name])
        print(f"{name:11s} float16 differs by at most {np.max(np.abs(a - b)):.3e}, "
              f"over values up to {np.max(np.abs(a)):.1f}")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "input.json").write_text(json.dumps({
        "spatial": spatial.ravel().tolist(), "shape": list(spatial.shape),
        "global": global_input.ravel().tolist(), "globalShape": list(global_input.shape),
    }))
    (out / "onnx.json").write_text(json.dumps({n: np.ravel(single[n]).tolist() for n in OUTPUTS}))
    print(f"wrote {out}/input.json and {out}/onnx.json")


if __name__ == "__main__":
    main()
