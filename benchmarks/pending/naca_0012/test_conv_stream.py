"""Test conv-based streaming (1 launch) vs 9x torch.roll, and optimized collision."""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
for _p in (_REPO / "src", _REPO / "benchmarks", _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import json  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from fast_kernel import _mats, fast_collide_mrt, fast_stream  # noqa: E402
from speed_probe import build, timeit  # noqa: E402


def make_stream_weight(dev, dtype):
    from tensorlbm.d2q9 import C
    w = torch.zeros((9, 1, 3, 3), device=dev, dtype=dtype)
    for i in range(9):
        cx, cy = int(C[i, 0]), int(C[i, 1])
        w[i, 0, 1 - cy, 1 - cx] = 1.0
    return w


def conv_stream(f, w):
    p = F.pad(f, (1, 1, 1, 1), mode="circular")
    return F.conv2d(p, w, groups=9)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("C", type=int, nargs="?", default=64)
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--steps", type=int, default=40)
    a = ap.parse_args()
    dev, f, tau, tau_field, solid, nx, ny = build(a.C, a.device)
    w = make_stream_weight(f.device, f.dtype)

    ref = fast_stream(f)
    got = conv_stream(f, w)
    torch.sdaa.synchronize()
    d = (got - ref).abs().max().item()

    r = {"C": a.C, "conv_stream_diff": d}
    r["roll_stream_ms"] = timeit(lambda ff: fast_stream(ff), f, a.steps, dev)
    r["conv_stream_ms"] = timeit(lambda ff: conv_stream(ff, w), f, a.steps, dev)
    # sanity: conv weight on CPU build for different alpha direction
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}, indent=2))


if __name__ == "__main__":
    main()