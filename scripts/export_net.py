"""Writes the two weight files the page is served, from the full-precision net.

    uv run --with numpy python scripts/export_net.py ../hex27x3.bin.gz public-web/net

Half of what the page downloads today is thrown away twice over: the engine
parses the whole net only to read its name, version and channel counts -- the
JS backend evaluates nothing -- and the WebGPU backend converts every weight to
float16 on its way to the GPU. So write those two readers what each of them
actually uses:

    net-fp16.bin.gz    conv and matmul weights as float16, for the GPU
    net-shape.bin.gz   the same file with every float array zeroed, for the engine

Both are still KataGo weight files: the float16 arrays are marked @BINF16@
instead of @BIN@, which vendor/binModelParser.ts understands and desc.cpp never
sees. Everything outside the arrays is copied byte for byte.

Batch norm statistics stay float32. The GPU stores the batch norm folded into a
scale and a bias, computed from the mean and variance in float32 and rounded
once -- rounding the inputs of that division instead would be a second,
avoidable, loss, and the arrays are four thousandths of the file.
"""

import argparse
import gzip
import hashlib
import pathlib

import numpy as np

from katago_bin import BIN16, BIN32, Reader, parse, read_maybe_gzipped

#: Which layers' arrays are stored half precision: the ones the WebGPU backend
#: converts to f16 anyway, so that for 1x1 convolutions and matmuls -- padded,
#: never arithmetic -- the GPU ends up with the identical bits either way.
HALF_KINDS = {"conv", "matmul"}


def to_half(values: np.ndarray) -> np.ndarray:
    """float32 to float16 the way `toHalf` in src/webgpuModel.ts does it.

    Which is by dropping the low mantissa bits, not by rounding to nearest. The
    point of exporting is to hand the GPU the bits it would have computed, so
    this has to agree with that function element for element; a better rounding
    would be a change to what the page evaluates, and belongs on its own.
    """
    bits = values.view("<u4")
    sign = (bits >> 16) & 0x8000
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    if (exponent >= 143).any():
        raise ValueError("weight too large for float16")
    subnormal = np.right_shift(mantissa | 0x800000, np.minimum(126 - exponent, 31),
                               where=exponent > 102, out=np.zeros_like(mantissa))
    return np.where(exponent <= 112, sign | subnormal,
                    sign | ((exponent - 112) << 10) | (mantissa >> 13)).astype("<u2")


def rewrite(raw: bytes, arrays: list[tuple[int, int, str]], half: bool) -> bytes:
    """Copies `raw`, replacing each recorded float array with `half` or zeros."""
    out = bytearray()
    at = 0
    for start, count, kind in arrays:
        marker = start - len(BIN32)
        assert raw[marker:start] == BIN32, f"no marker before the array at {start}"
        out += raw[at:marker]
        values = np.frombuffer(raw, dtype="<f4", count=count, offset=start)
        if not half:
            out += BIN32 + bytes(count * 4)
        elif kind in HALF_KINDS:
            out += BIN16 + to_half(values).tobytes()
        else:
            out += BIN32 + values.tobytes()
        at = start + count * 4
    return bytes(out + raw[at:])


def write(path: pathlib.Path, data: bytes) -> None:
    # mtime 0 so that re-exporting the same net gives the same bytes.
    path.write_bytes(gzip.compress(data, mtime=0))
    print(f"{path}  {len(data) / 1e6:.1f} MB, {path.stat().st_size / 1e6:.1f} MB gzipped, "
          f"sha256 {hashlib.sha256(path.read_bytes()).hexdigest()[:16]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("prefix", help="output path prefix, e.g. public-web/net")
    args = ap.parse_args()

    raw = read_maybe_gzipped(args.model)
    reader = Reader(raw)
    model = parse(raw, reader)
    if not reader.at_end():
        raise ValueError(f"{len(raw) - reader.idx} bytes left after the model")

    counts: dict[str, int] = {}
    for _, count, kind in reader.arrays:
        counts[kind] = counts.get(kind, 0) + count
    print(f"{model['name']}: version {model['version']}, {len(model['blocks'])} blocks, "
          f"{model['trunk_channels']} channels")
    print("  " + ", ".join(f"{kind} {n / 1e6:.2f}M" for kind, n in sorted(counts.items())))

    prefix = pathlib.Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write(prefix.with_name(prefix.name + "-fp16.bin.gz"), rewrite(raw, reader.arrays, half=True))
    write(prefix.with_name(prefix.name + "-shape.bin.gz"), rewrite(raw, reader.arrays, half=False))


if __name__ == "__main__":
    main()
