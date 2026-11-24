"""Benchmark for the compressed fluid-node neighbour table (porous-media LBM).

Runs a series of tests to validate the compressed layout against the dense
reference implementation and measures the performance gain.

Usage:
    python run_bench.py [--output results.json]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensorlbm.porous import (  # noqa: E402
    build_fluid_node_table,
    compressed_step,
    dense_step,
    macroscopic_with_forcing,
    permeability,
    intrinsic_permeability,
)

torch.set_default_dtype(torch.float64)

TAU = 0.8
NU = (TAU - 0.5) / 3.0


def random_porous(n: int = 24, seed: int = 7, p_solid: float = 0.3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    solid = torch.rand((n, n), generator=g) < p_solid
    solid[0, 0] = False
    return solid


def square_grain(n: int, porosity: float) -> torch.Tensor:
    import math
    solid = torch.zeros((n, n), dtype=torch.bool)
    side = int(round(n * math.sqrt((1.0 - porosity) / 4.0)))
    off = [n // 4, 3 * n // 4]
    for y0 in off:
        for x0 in off:
            solid[y0 - side // 2 : y0 - side // 2 + side, x0 - side // 2 : x0 - side // 2 + side] = True
    return solid


def run_accuracy_bench():
    """Run accuracy tests and return results."""
    results = {}
    for seed in [1, 5, 13]:
        solid = random_porous(n=20, seed=seed, p_solid=0.3)
        table = build_fluid_node_table(solid)
        ny = nx = 20
        fluid = ~solid
        g = 1e-5
        f_c = torch.zeros((9, table.n_fluid))
        f_d = torch.zeros((9, ny, nx))
        for f in (f_c, f_d):
            f[0] = 4.0 / 9.0
            f[1] = f[2] = f[3] = f[4] = 1.0 / 9.0
            f[5] = f[6] = f[7] = f[8] = 1.0 / 36.0
        for _ in range(25):
            f_c = compressed_step(f_c, table, TAU, g, 0.0)
            f_d = dense_step(f_d, solid, TAU, g, 0.0)
        rho_c, ux_c, uy_c = macroscopic_with_forcing(f_c, g, 0.0)
        rho_d, ux_d, uy_d = macroscopic_with_forcing(f_d, g, 0.0)
        ux_err = (ux_c - ux_d[fluid]).abs().max().item()
        uy_err = (uy_c - uy_d[fluid]).abs().max().item()
        rho_err = (rho_c - rho_d[fluid]).abs().max().item()
        results[f"seed_{seed}"] = {
            "ux_max_err": ux_err,
            "uy_max_err": uy_err,
            "rho_max_err": rho_err,
            "pass": ux_err < 1e-13 and uy_err < 1e-13 and rho_err < 1e-13,
        }
    return results


def run_permeability_bench():
    """Run permeability test and return results."""
    ny, nx, H = 40, 16, 16
    solid = torch.ones((ny, nx), dtype=torch.bool)
    y0 = (ny - H) // 2
    solid[y0 : y0 + H, :] = False
    table = build_fluid_node_table(solid)
    g = 1e-5
    f = torch.zeros((9, table.n_fluid))
    for d, w in enumerate((4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36)):
        f[d] = w
    for _ in range(8000):
        f = compressed_step(f, table, TAU, g, 0.0)
    _, ux, _ = macroscopic_with_forcing(f, g, 0.0)
    u_sup = float(ux.sum()) / table.n_total
    k_super = permeability(u_sup, NU, g)
    k_intrinsic = intrinsic_permeability(k_super, table.porosity)
    k_ref = H * H / 12.0
    err = abs(k_intrinsic - k_ref) / k_ref
    return {
        "k_intrinsic": k_intrinsic,
        "k_ref": k_ref,
        "err": err,
        "pass": err < 0.15,
    }


def run_performance_bench():
    """Run performance benchmark and return results."""
    solid = square_grain(128, 0.60)
    table = build_fluid_node_table(solid)
    ny = nx = 128
    fluid = ~solid
    g = 1e-5
    f_c = torch.zeros((9, table.n_fluid))
    f_d = torch.zeros((9, ny, nx))
    for f in (f_c, f_d):
        f[0] = 4.0 / 9.0
        f[1] = f[2] = f[3] = f[4] = 1.0 / 9.0
        f[5] = f[6] = f[7] = f[8] = 1.0 / 36.0

    # Warmup
    for _ in range(10):
        f_c = compressed_step(f_c, table, TAU, g, 0.0)
        f_d = dense_step(f_d, solid, TAU, g, 0.0)

    # Benchmark compressed
    start = time.time()
    for _ in range(100):
        f_c = compressed_step(f_c, table, TAU, g, 0.0)
    t_c = time.time() - start

    # Benchmark dense
    start = time.time()
    for _ in range(100):
        f_d = dense_step(f_d, solid, TAU, g, 0.0)
    t_d = time.time() - start

    return {
        "compressed_time": t_c,
        "dense_time": t_d,
        "speedup": t_d / t_c,
    }


def main():
    output = Path(__file__).parent / "results.json"
    results = {
        "accuracy": run_accuracy_bench(),
        "permeability": run_permeability_bench(),
        "performance": run_performance_bench(),
    }
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
