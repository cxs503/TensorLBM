#!/usr/bin/env python
"""Reproduce the collision-family evidence recorded in ``tensorlbm.registry``.

Two measurements that the registry rows for the D3Q19 collision family cite:

1. *analytic band* — Guo-forced plane Poiseuille flow against the exact
   parabolic profile ``u(y) = G/(2 nu) (y-1/2)(h+1/2-y)``.  This is the same
   harness as ``tests/test_lbm_verification.py::poiseuille_case`` (kept
   independent here so the number in the registry can be re-derived without
   pytest).

2. *anti-mock probe* — MRT must not be a BGK alias.  The Poiseuille case is a
   steady shear flow, so it cannot separate MRT from BGK by itself; the probe
   instead perturbs the non-hydrodynamic relaxation rate ``s_e`` on a
   non-equilibrium field and requires the output to move.  A stand-in kernel
   that forwards to ``collide_bgk3d`` would return a difference of exactly 0.

Usage::

    PYTHONPATH=src python scripts/m23186_collision_family_verification.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.d3q19 import C, W  # noqa: E402
from tensorlbm.solver3d import (  # noqa: E402
    collide_bgk3d,
    collide_mrt3d,
    collide_rlbm3d,
    collide_trt3d,
    equilibrium3d,
    macroscopic3d,
    stream3d,
)

COLLIDE = {
    "bgk": collide_bgk3d,
    "trt": collide_trt3d,
    "rlbm": collide_rlbm3d,
    "mrt": collide_mrt3d,
}


def _no_dynamics_and_bounce(f: torch.Tensor, f_pre: torch.Tensor,
                            solid: torch.Tensor) -> torch.Tensor:
    """NoDynamics (solid nodes keep the *pre-collision* state) + half-way bounce-back.

    Vectorised exactly like the pytest harness: solid nodes are overwritten with
    the pre-collision populations and then reflected by the production bounce-back
    kernel.
    """
    from tensorlbm.boundaries3d import bounce_back_cells_3d

    f = torch.where(solid.unsqueeze(0).expand_as(f), f_pre, f)
    return bounce_back_cells_3d(f, solid)


def poiseuille(collision: str = "bgk", *, ny: int = 10, steps: int = 200,
               tau: float = 1.0, u_max: float = 0.05) -> dict:
    d = torch.device("cpu")
    nx, nz = 24, 2
    nu = (tau - 0.5) / 3.0
    height = ny - 2
    g_force = 2 * nu * u_max / height**2
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=d)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    c = C.to(d).to(torch.get_default_dtype())
    w = W.to(d).to(torch.get_default_dtype())
    accel = (w * 3.0 * c[:, 0]).view(19, 1, 1, 1) * g_force
    zero = torch.zeros(nz, ny, nx, device=d)
    f = equilibrium3d(torch.ones(nz, ny, nx, device=d), zero, zero, zero)
    collide = COLLIDE[collision]
    for _ in range(steps):
        f_pre = f.clone()
        f = collide(f, tau)
        f = _no_dynamics_and_bounce(f, f_pre, solid)
        f = f + accel
        f = stream3d(f)
        f[:, :, :, 0] = f[:, :, :, -2]
        f[:, :, :, -1] = f[:, :, :, -2]
    _, ux, _, _ = macroscopic3d(f)
    u = ux[0, 1:-1, nx // 2].cpu().numpy().astype(float)
    y = torch.arange(1, ny - 1).numpy().astype(float)
    u_exact = g_force / (2 * nu) * (y - 0.5) * (height + 0.5 - y)
    err_pct = float(abs(u - u_exact).max() / max(u_exact.max(), 1e-12)) * 100.0
    return {
        "collision": collision, "ny": ny, "steps": steps, "tau": tau,
        "nu": nu, "body_force": g_force,
        "u_profile_numeric": u.tolist(), "u_profile_exact": u_exact.tolist(),
        "y": y.tolist(), "max_error_pct": err_pct,
    }


def anti_mock_probe(seed: int = 7) -> dict:
    """MRT output must respond to ``s_e``; a BGK stand-in cannot."""
    torch.manual_seed(seed)
    zero = torch.zeros(2, 6, 6)
    f = equilibrium3d(torch.ones(2, 6, 6), zero, zero, zero)
    f = f + 0.05 * torch.randn_like(f)
    ref = collide_bgk3d(f, 0.8)
    mrt_a = collide_mrt3d(f, 0.8, s_e=0.5)
    mrt_b = collide_mrt3d(f, 0.8, s_e=1.9)
    mrt_vs_bgk = float((mrt_a - ref).abs().max() / ref.abs().max())
    mrt_s_e = float((mrt_a - mrt_b).abs().max() / ref.abs().max())
    return {
        "mrt_vs_bgk_max_rel": mrt_vs_bgk,
        "mrt_se_sensitivity_max_rel": mrt_s_e,
        "verdict": "not an alias of BGK"
        if (mrt_vs_bgk > 0 or mrt_s_e > 0) else "SUSPECT: MRT is a BGK alias",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "docs" / "m23186"
                                         / "collision_family_verification.json"))
    args = ap.parse_args()
    result = {
        "case": "D3Q19 plane Poiseuille (Guo body force, half-way bounce-back)",
        "protocol": {"nx": 24, "nz": 2, "ny": 10, "tau": 1.0, "u_max": 0.05,
                     "steps": args.steps},
        "families": [],
        "anti_mock": anti_mock_probe(),
    }
    for name in COLLIDE:
        fam = poiseuille(name, steps=args.steps)
        result["families"].append(fam)
        print(f"{name:5s} max_error = {fam['max_error_pct']:.4f} %", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for fam in result["families"]:
        print(f"{fam['collision']:5s} max_error = {fam['max_error_pct']:.4f} %")
    print("anti-mock:", json.dumps(result["anti_mock"], ensure_ascii=False))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())