#!/usr/bin/env python
"""CPU smoke test for sphere_re100 — minimal verification.

Runs a tiny-domain (D=8) BFL simulation on CPU for 200 steps to
confirm the algorithm is correct without needing GPU/SDAA hardware.
This is a *smoke test* only — not a full verification run.

Usage:
    PYTHONPATH=src:benchmarks python benchmarks/verified/sphere_re100/smoke_cpu.py
"""
from __future__ import annotations

import functools
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tensorlbm.bfl_common import (
    bfl_bounce_back_sparse,
    bfl_boundary_link_indices,
    bfl_force_ledger_sparse,
    compute_q_sphere_common,
)
from tensorlbm.bfl_d3q19 import bouzidi_bounce_back_d3q19
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.solver3d import collide_mrt3d_low_memory, correct_mass3d, stream3d_roll
from tensorlbm.d3q19 import C, W


def cd_ref(re: float) -> float:
    return 24.0 / re * (1.0 + 0.15 * re**0.687)


def make_solid_mask(nz, ny, nx, cx, cy, cz, R, device):
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float64),
        torch.arange(ny, device=device, dtype=torch.float64),
        torch.arange(nx, device=device, dtype=torch.float64),
        indexing="ij",
    )
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
    return dist2 <= R**2


def init_equilibrium(nz, ny, nx, ulb, device, dtype=torch.float64):
    c = C.to(device).to(dtype)
    w = W.to(device).to(dtype)
    rho0 = torch.ones((nz, ny, nx), device=device, dtype=dtype)
    f = torch.zeros((19, nz, ny, nx), device=device, dtype=dtype)
    u2 = ulb ** 2
    for q in range(19):
        cu = c[q, 0] * ulb
        f[q] = rho0 * w[q] * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u2)
    return f


def run_smoke(D=8, steps=200, sample=20, kernel="sparse", device_str="cpu"):
    device = torch.device(device_str)
    print(f"Smoke test: D={D}, steps={steps}, kernel={kernel}, device={device}")

    lat = 2.0
    up = 3.0
    down = 6.0
    nz = int(D * lat)
    ny = int(D * lat)
    nx = int(D * (up + 1.0 + down))
    cx, cy, cz = nx / 2.0, ny / 2.0, nz / 2.0
    R = D / 2.0

    tau = 0.56
    ulb = 0.05
    nu_lb = (tau - 0.5) / 3.0
    re_eff = ulb * D / nu_lb
    dpS = 0.5 * ulb**2 * math.pi * R**2

    print(f"  Domain: {nz}x{ny}x{nx}, R={R}, tau={tau}, Re_eff={re_eff:.2f}")

    solid = make_solid_mask(nz, ny, nx, cx, cy, cz, R, device)
    n_solid = int(solid.sum().item())
    print(f"  Solid cells: {n_solid}")

    bfl_mask, bfl_q = compute_q_sphere_common(nx, ny, nz, cx, cy, cz, R, device)
    n_links = int(bfl_mask[1:].sum().item())
    print(f"  BFL boundary links: {n_links}")

    f = init_equilibrium(nz, ny, nx, ulb, device)
    sm = solid.unsqueeze(0).expand_as(f)
    w = W.to(device).to(f.dtype)
    for q in range(19):
        f[q] = torch.where(sm[q], w[q], f[q])

    if kernel == "sparse":
        links = bfl_boundary_link_indices(bfl_mask, bfl_q, lattice="D3Q19")
        print(f"  Sparse links: n_links={links.n_links}")
    else:
        links = None

    collide = lambda f: collide_mrt3d_low_memory(f, tau=tau)
    stream_fn = stream3d_roll
    bc_config = {"far_field_faces": ["y-", "y+", "z-", "z+"], "periodic_faces": []}
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=bc_config)

    target_mass = float(f.sum().item())
    hist = []
    sample_set = set(range(sample, steps + 1, sample))
    t0 = time.time()

    for step in range(1, steps + 1):
        f_pre = f.clone()
        f = collide(f)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        del f_pre

        f = bounce_back_cells_3d(f, solid)
        f_pre_stream = f.clone()
        f = stream_fn(f)
        f = far_field_fn(f, ulb)

        if kernel == "sparse":
            force3, _ = bfl_force_ledger_sparse(f, f_pre_stream, links)
            f = bfl_bounce_back_sparse(f, f_pre_stream, links)
        else:
            f, force3 = bouzidi_bounce_back_d3q19(
                f, f_pre_stream, bfl_mask, bfl_q, return_force=True
            )

        if step in sample_set:
            cd = float(force3[0].item()) / dpS
            cl = float(force3[1].item()) / dpS
            cs = float(force3[2].item()) / dpS
            entry = {"step": step, "cd": cd, "cl": cl, "cs": cs, "mass": float(f.sum().item())}
            hist.append(entry)

        if step % 50 == 0:
            f = correct_mass3d(f, target_mass)

        if step % 100 == 0 and not bool(torch.isfinite(f).all()):
            print(f"  DIVERGED at step {step}!")
            return None

        if step % 50 == 0:
            el = time.time() - t0
            print(f"  step {step}/{steps} ({el:.1f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    if hist:
        tail = [h["cd"] for h in hist[-4:]]
        cd_mean = sum(tail) / len(tail)
        cd_ref_val = cd_ref(re_eff)
        err_pct = (cd_mean - cd_ref_val) / cd_ref_val * 100.0
        print(f"\n  Results:")
        print(f"    Cd (last 4 mean): {cd_mean:.6f}")
        print(f"    Cd_ref (Re={re_eff:.1f}): {cd_ref_val:.6f}")
        print(f"    Error: {err_pct:+.2f}%")
        print(f"    Mass drift: {(hist[-1]['mass'] - hist[0]['mass']) / hist[0]['mass'] * 1e6:.4f} ppm")

        return {
            "D": D, "steps": steps, "kernel": kernel, "re_eff": re_eff,
            "cd_mean": cd_mean, "cd_ref": cd_ref_val, "err_pct": err_pct,
            "n_links": n_links, "elapsed_s": elapsed, "history": hist,
        }
    return None


def main():
    print("=" * 60)
    print("sphere_re100 CPU Smoke Test (minimal)")
    print("=" * 60)

    result_sparse = run_smoke(D=8, steps=200, sample=20, kernel="sparse", device_str="cpu")
    print("\n" + "=" * 60)
    result_dense = run_smoke(D=8, steps=200, sample=20, kernel="dense", device_str="cpu")

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    if result_sparse and result_dense:
        diff = abs(result_sparse["cd_mean"] - result_dense["cd_mean"])
        print(f"  Sparse Cd: {result_sparse['cd_mean']:.6f}")
        print(f"  Dense  Cd: {result_dense['cd_mean']:.6f}")
        print(f"  Difference: {diff:.8f}")
        print(f"  Bit-identical: {diff < 1e-12}")

        out = {
            "smoke_test": "sphere_re100_cpu_minimal",
            "sparse": {"cd_mean": result_sparse["cd_mean"], "cd_ref": result_sparse["cd_ref"], "err_pct": result_sparse["err_pct"]},
            "dense": {"cd_mean": result_dense["cd_mean"], "cd_ref": result_dense["cd_ref"], "err_pct": result_dense["err_pct"]},
            "bit_identical": diff < 1e-12,
        }
        out_path = os.path.join(os.path.dirname(__file__), "smoke_cpu_result.json")
        with open(out_path, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
