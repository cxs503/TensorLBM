#!/usr/bin/env python3
"""XVAL driver: TensorLBM regularised D2Q9 BC vs joconnor22/LIFE's regularisedBC.

Protocol: ``predeclared.json`` (frozen before the production runs).
Reference: ``reference.py`` (per-node transcription of ``oss_src/Grid.cpp``).
Candidate: ``src/tensorlbm/regularised_bc_2d.py``.

Arms (identical BGK kernel, D2Q9, float64, initial state = analytic parabola):

* ``reg``   regularised velocity inlet + regularised pressure outlet +
            regularised wet-node walls (LIFE practice);
* ``zouhe`` Zou-He velocity inlet + Zou-He pressure outlet + solid-node
            half-way bounce-back walls (TensorLBM 2D status quo).

Usage::

    PYTHONPATH=src python3 xval/XVAL-LBM-REG-BC-NEQ-STRESS/run_xval.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import reference as life_ref  # noqa: E402
from tensorlbm import regularised_bc_2d as rb  # noqa: E402
from tensorlbm.boundary.d2q9 import C2Q9, OPPOSITE_2Q9  # noqa: E402
from tensorlbm.boundary.solver2d import equilibrium2d, macroscopic2d  # noqa: E402

torch.set_grad_enabled(False)
DTYPE = torch.float64


def build_channel(ny: int, nx: int, umax: float):
    """Shared kind/normal layout for both arms (LIFE type-priority ordering)."""
    H = ny - 1
    kinds = torch.zeros((ny, nx), dtype=torch.long)
    normals = torch.zeros((ny, nx, 2), dtype=torch.long)
    kinds[:, 0] = rb.VELOCITY
    normals[:, 0, 0] = 1
    kinds[:, -1] = rb.PRESSURE
    normals[:, -1, 0] = -1
    kinds[0, :] = rb.WALL          # y-edges override x-edges (Grid.cpp initialiseGrid)
    normals[0, :, 1] = 1
    kinds[-1, :] = rb.WALL
    normals[-1, :, 1] = -1
    normals[0, 0] = torch.tensor([1, 1])
    normals[0, -1] = torch.tensor([-1, 1])
    normals[-1, 0] = torch.tensor([1, -1])
    normals[-1, -1] = torch.tensor([-1, -1])
    u_bc = torch.zeros((ny, nx, 2), dtype=DTYPE)
    prof_wet = 4.0 * umax * np.arange(ny) * (H - np.arange(ny)) / H**2
    u_bc[:, 0, 0] = torch.from_numpy(prof_wet)
    rho_bc = torch.ones((ny, nx), dtype=DTYPE)
    y_bb = np.arange(ny) - 0.5
    prof_bb = 4.0 * umax * y_bb * (H - y_bb) / H**2
    return kinds, normals, u_bc, rho_bc, prof_bb, H


def analytic_profile(ny: int, H: int, umax: float, offset: float) -> np.ndarray:
    y = np.arange(ny) - offset
    return 4.0 * umax * y * (H - y) / H**2


def push_stream(f: torch.Tensor) -> torch.Tensor:
    """Collide-then-stream (push), non-periodic."""
    out = torch.zeros_like(f)
    ny, nx = f.shape[1], f.shape[2]
    for q in range(9):
        cx, cy = int(C2Q9[q, 0]), int(C2Q9[q, 1])
        yd = slice(max(cy, 0), ny + min(cy, 0))
        ys = slice(max(-cy, 0), ny + min(-cy, 0))
        xd = slice(max(cx, 0), nx + min(cx, 0))
        xs = slice(max(-cx, 0), nx + min(-cx, 0))
        out[q, yd, xd] = f[q, ys, xs]
    return out


def zouhe_columns(out: torch.Tensor, prof_bb: np.ndarray, rho_out: float = 1.0):
    """Zou-He inlet (velocity) and pressure outlet on the interior rows.

    Unknowns inlet x=[rho,f1,f5,f7]; outlet x=[ux,f2,f6,f8].  Four linear
    equations per column: mass, x-momentum, y-momentum (uy = 0) and the Zou-He
    non-equilibrium closure of the diagonal pair.  Corner rows stay bounce-back.
    """
    ny = out.shape[1]
    f = out.numpy().copy()
    rows = slice(1, ny - 1)
    ux = prof_bb[rows]
    n = ux.size

    K = f[0, rows, 0] + f[2, rows, 0] + f[3, rows, 0] + f[4, rows, 0] + f[6, rows, 0] + f[8, rows, 0]
    A = np.zeros((n, 4, 4))
    b = np.zeros((n, 4))
    A[:, 0] = np.array([1.0, -1.0, -1.0, -1.0])
    b[:, 0] = K
    A[:, 1] = np.stack([-ux, np.ones(n), np.ones(n), np.ones(n)], axis=1)
    b[:, 1] = f[2, rows, 0] + f[6, rows, 0] + f[8, rows, 0]
    A[:, 2] = np.stack([np.zeros(n), np.zeros(n), np.ones(n), -np.ones(n)], axis=1)
    b[:, 2] = -f[3, rows, 0] + f[4, rows, 0] - f[6, rows, 0] + f[8, rows, 0]
    A[:, 3] = np.stack([-(1.0 / 18.0 + ux / 6.0), np.zeros(n), np.ones(n), np.ones(n)], axis=1)
    b[:, 3] = np.zeros(n)
    sol = np.linalg.solve(A, b)
    f[0, rows, 0] = sol[:, 0]
    f[1, rows, 0] = sol[:, 1]
    f[5, rows, 0] = sol[:, 2]
    f[7, rows, 0] = sol[:, 3]

    K = f[0, rows, -1] + f[1, rows, -1] + f[3, rows, -1] + f[4, rows, -1] + f[5, rows, -1] + f[7, rows, -1]
    A = np.zeros((n, 4, 4))
    b = np.zeros((n, 4))
    A[:, 0] = np.array([0.0, -1.0, -1.0, -1.0])
    b[:, 0] = rho_out - K
    A[:, 1] = np.array([-rho_out, -1.0, -1.0, -1.0])
    b[:, 1] = -(f[1, rows, -1] + f[5, rows, -1] + f[7, rows, -1])
    A[:, 2] = np.array([0.0, 0.0, 1.0, -1.0])
    b[:, 2] = -(f[3, rows, -1] - f[4, rows, -1]) - (f[5, rows, -1] - f[7, rows, -1])
    A[:, 3] = np.array([rho_out / 6.0, 0.0, 1.0, 1.0])
    b[:, 3] = np.full(n, rho_out / 18.0)
    sol = np.linalg.solve(A, b)
    f[2, rows, -1] = sol[:, 1]
    f[6, rows, -1] = sol[:, 2]
    f[8, rows, -1] = sol[:, 3]

    c = C2Q9.numpy()
    rho_c = f[:, rows, -1].sum(axis=0)
    ux_c = (c[:, 0, None] * f[:, rows, -1]).sum(axis=0) / rho_c
    uy_c = (c[:, 1, None] * f[:, rows, -1]).sum(axis=0) / rho_c
    resid = max(
        float(np.max(np.abs(rho_c - rho_out))),
        float(np.max(np.abs(uy_c))),
        float(np.max(np.abs(ux_c - sol[:, 0]))),
    )
    return torch.from_numpy(f), resid


def step(f, tau, kinds, normals, u_bc, rho_bc, arm, prof_bb, rho_out=1.0):
    rho_f, ux_f, uy_f = macroscopic2d(f)
    feq = equilibrium2d(rho_f, ux_f, uy_f)
    fs = push_stream(f - (f - feq) / tau)
    if arm == "reg":
        u = torch.stack([ux_f, uy_f], dim=-1)
        out, _ = rb.apply_regularised_bc_2d(
            fs, kinds, normals, u_bc=u_bc, rho_bc=rho_bc, rho=rho_f, u=u
        )
        return out, 0.0
    out = fs.clone()
    wall = kinds == rb.WALL
    out = torch.where(wall.unsqueeze(0), out[OPPOSITE_2Q9], out)
    out, resid = zouhe_columns(out, prof_bb, rho_out)
    return out, resid


def compare_implementations(fs, kinds, normals, u_bc, rho_bc, rho_f, u_f):
    """K1/K2: reference transcription vs torch port on the same pre-BC state."""
    f_ref, _, _, _ = life_ref.life_regularised_bc(
        fs.numpy(), kinds.numpy(), normals.numpy(), u_bc.numpy(), rho_bc.numpy(),
        rho_f.numpy(), u_f.numpy(),
    )
    f_port, diag = rb.apply_regularised_bc_2d(
        fs, kinds, normals, u_bc=u_bc, rho_bc=rho_bc, rho=rho_f, u=u_f
    )
    iy, ix = np.nonzero((kinds != rb.FLUID).numpy())
    corner = diag["is_corner"].numpy()
    fp = f_port.numpy()[:, iy, ix]
    fr = f_ref[:, iy, ix]
    dev = np.abs(fp - fr)
    denom = np.sqrt((fr**2).mean(axis=0))
    rel = 100.0 * dev.max(axis=0) / denom
    out = {
        "n_nodes": int(iy.size),
        "n_corner": int(corner.sum()),
        "K1_all_pct": float(rel.max()),
        "K1_noncorner_pct": float(rel[~corner].max()) if (~corner).any() else 0.0,
        "K1_corner_pct": float(rel[corner].max()) if corner.any() else 0.0,
        "K1_max_abs_dev": float(dev.max()),
    }
    pre = np.array([diag["pi_neq_xx_pre"].numpy(), diag["pi_neq_yy_pre"].numpy(), diag["pi_neq_xy_pre"].numpy()])
    post = np.array([diag["pi_neq_xx"].numpy(), diag["pi_neq_yy"].numpy(), diag["pi_neq_xy"].numpy()])
    out["K2_pi_neq_preserved"] = float(np.abs(post - pre).max())
    c = C2Q9.numpy()
    mom = (c[:, 0, None] * fp).sum(axis=0)
    out["K2_momentum_after"] = float(np.abs(mom).max())
    # prescribed state must be reproduced exactly at flat velocity-type nodes
    flat = ((kinds != rb.FLUID) & (kinds != rb.PRESSURE)).numpy()[iy, ix] & ~corner
    got = np.stack([diag["ux"].numpy(), diag["uy"].numpy()], axis=1)
    if flat.any():
        out["K2_u_prescribed_err"] = float(np.abs(got[flat] - u_bc.numpy()[iy, ix][flat]).max())
    else:
        out["K2_u_prescribed_err"] = 0.0
    return out


def run_channel(arm, ny, nx, umax, re, max_steps, parity_steps=()):
    kinds, normals, u_bc, rho_bc, prof_bb, H = build_channel(ny, nx, umax)
    nu = umax * H / re
    tau = 0.5 + 3.0 * nu
    offset = 0.0 if arm == "reg" else 0.5
    y_ex = analytic_profile(ny, H, umax, offset)
    ux0 = torch.from_numpy(np.tile(y_ex[:, None], (1, nx)))
    zero = torch.zeros((ny, nx), dtype=DTYPE)
    f = equilibrium2d(torch.ones((ny, nx), dtype=DTYPE), ux0, zero).clone()

    parities, resid_max = [], 0.0
    prev = ux0.clone()
    converged = False
    last_delta = float("nan")
    t0 = time.time()
    for it in range(1, max_steps + 1):
        if it in parity_steps and arm == "reg":
            rho_f, ux_f, uy_f = macroscopic2d(f)
            feq = equilibrium2d(rho_f, ux_f, uy_f)
            fs = push_stream(f - (f - feq) / tau)
            u = torch.stack([ux_f, uy_f], dim=-1)
            parities.append({"step": it, **compare_implementations(fs, kinds, normals, u_bc, rho_bc, rho_f, u)})
        f, resid = step(f, tau, kinds, normals, u_bc, rho_bc, arm, prof_bb)
        resid_max = max(resid_max, resid)
        if it % 500 == 0:
            _, ux_n, _ = macroscopic2d(f)
            if torch.isfinite(ux_n).all():
                delta = float((ux_n - prev).abs().max())
                last_delta = delta
                prev = ux_n.clone()
                if delta < 1e-13:
                    converged = True
                    break
    wall_time = time.time() - t0

    rho_f, ux_f, uy_f = macroscopic2d(f)
    ux, uy, rho = ux_f.numpy(), uy_f.numpy(), rho_f.numpy()
    rows = slice(1, ny - 1)
    cols = slice(2, nx - 2)
    ex = np.tile(y_ex[:, None], (1, nx))
    err = ux[rows, cols] - ex[rows, cols]
    scale = float(np.sqrt((ex[rows, cols] ** 2).mean()))
    mid = nx // 2
    K3 = 100.0 * float(np.abs(ux[rows, mid] - ex[rows, mid]).max()) / umax
    K4 = 100.0 * float(np.sqrt((err**2).mean())) / scale

    def Q(x):
        return float((rho[rows, x] * ux[rows, x]).sum())

    K5 = 100.0 * abs(Q(2) - Q(nx - 3)) / abs(Q(2))
    K6 = 100.0 * float(np.abs(uy[rows, -1]).max()) / umax
    K7 = 100.0 * float(np.abs(ux[0, 2:-2]).max()) / umax if arm == "reg" else None
    return {
        "arm": arm,
        "ny": ny,
        "nx": nx,
        "H": H,
        "u_max": umax,
        "Re": re,
        "nu": nu,
        "tau": tau,
        "steps": it,
        "converged": converged,
        "last_500step_delta": last_delta,
        "wall_time_s": round(wall_time, 2),
        "zouhe_bc_residual": resid_max,
        "K3_profile_err_pct": K3,
        "K4_field_l2_pct": K4,
        "K5_mass_drift_pct": K5,
        "K6_outlet_uy_pct": K6,
        "K7_wall_slip_pct": K7,
        "rho_max_dev": float(np.abs(rho[rows, cols] - 1.0).max()),
        "u_mid_column": ux[rows, mid].tolist(),
        "y_mid_analytic": y_ex[rows].tolist(),
        "parity": parities,
    }


def main():
    cases = [
        dict(ny=33, nx=129, umax=0.05, re=100.0, max_steps=3000),
        dict(ny=17, nx=65, umax=0.05, re=100.0, max_steps=1500),
    ]
    results = {"mission": 18240, "anchor": "XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-RECON", "cases": []}
    for case in cases:
        for arm in ("reg", "zouhe"):
            parity_steps = (400, 1200) if arm == "reg" and case["ny"] == 33 else ()
            try:
                r = run_channel(arm=arm, parity_steps=parity_steps, **case)
            except Exception:
                import traceback

                traceback.print_exc()
                continue
            results["cases"].append(r)
            print(
                f"arm={arm:5s} H={r['H']:3d} Re={r['Re']:5.1f} steps={r['steps']:6d} "
                f"conv={r['converged']} K3={r['K3_profile_err_pct']:.4f}% K4={r['K4_field_l2_pct']:.4f}% "
                f"K5={r['K5_mass_drift_pct']:.3e}% K6={r['K6_outlet_uy_pct']:.4f}%"
            )
            for p in r["parity"]:
                print(
                    f"   parity@{p['step']}: K1_all={p['K1_all_pct']:.3e}% "
                    f"noncorner={p['K1_noncorner_pct']:.3e}% corner={p['K1_corner_pct']:.3e}% "
                    f"K2_pi={p['K2_pi_neq_preserved']:.2e}"
                )
    out_dir = HERE / "out"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print("wrote", out_dir / "results.json")
    return results


if __name__ == "__main__":
    main()