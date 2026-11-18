"""Time fixed-geometry D2Q9 tank streaming with and without precomputed indices."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from tensorlbm.boundary.tank import TankStreamingPlan, stream_tank


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    ny, nx = 160, 240
    wall = torch.zeros(ny, nx, dtype=torch.bool)
    wall[:, [0, -1]] = True
    wall[-1] = True
    f, g = torch.randn(
        2, 9, ny, nx, dtype=torch.float64, generator=torch.Generator().manual_seed(123)
    )
    start = time.perf_counter()
    plan = TankStreamingPlan(wall)
    setup_ms = (time.perf_counter() - start) * 1000
    methods = {"original": lambda: stream_tank(f, g, wall), "prepared": lambda: plan.stream(f, g)}
    assert all(torch.equal(a, b) for a, b in zip(methods["original"](), methods["prepared"]()))
    times = {name: [] for name in methods}
    for run in methods.values():
        for _ in range(10):
            run()
    for i in range(8):
        for name in list(methods) if i % 2 == 0 else list(reversed(methods)):
            start = time.perf_counter()
            for _ in range(30):
                methods[name]()
            times[name].append((time.perf_counter() - start) * 1000 / 30)
    med = {k: statistics.median(v) for k, v in times.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                scope="stream f and g only",
                unit="ms",
                grid=[nx, ny],
                device="cpu",
                dtype="float64",
                threads=1,
                torch=str(torch.__version__),
                python=platform.python_version(),
                samples=times,
                median=med,
                speedup=med["original"] / med["prepared"],
                setup_ms=setup_ms,
                index_bytes=plan.indices.numel() * plan.indices.element_size(),
                bitwise_equal=True,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
