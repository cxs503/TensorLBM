#!/usr/bin/env python3
"""Analytic-cap test of the wall contact-angle convention.

Why this test exists
--------------------
The round-1 evidence for ``XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE`` compared
the ported rotation kernel with a *literal transcription of the reference
kernel* (``reference_rotation_scalar``).  That proves the port is faithful, but
it cannot prove the reference's own convention; and the "analytic interface
readback" test in round 1 read the angle back with ``acos(g_hat . n_s)`` on the
*rotated* gradient, which is precisely the quantity the kernel sets -- also
circular.  Both tests pass no matter which of the two possible conventions
(``theta`` vs ``180 - theta``) the kernel implements.

This test breaks the circularity by using an **independent geometric truth**:

    an equilibrium sessile drop is a spherical cap.  Put the exact cap for
    contact angle ``theta`` (measured *inside* the drop phase) on the lattice,
    set ``phi = tanh(d/xi)`` with ``d`` the signed distance to that cap, and ask
    the kernel to rotate the gradient at the wall-adjacent nodes.  If the
    kernel realises ``theta``, the rotated gradient must coincide with the
    cap's own outward normal there.  If it realises ``180 - theta``, the
    rotated gradient is off by ``|180 - 2 theta|`` everywhere except the
    symmetric point ``theta = 90`` (which is why a 90 deg arm can never detect
    this).

Geometry, with the wall plane at ``y_wall`` and ``phi = +1`` inside the drop
(nodes at unit spacing, ``nx`` periodic):

    centre   (x_c, y_c),  y_c = y_wall - R cos(theta)
    distance r = |(x, y) - (x_c, y_c)|,   d = R - r        (d > 0 inside)
    the cap meets the wall at ``x = x_c +- R sin(theta)``, so the footprint is
    ``2 R sin(theta)`` and the apex ``R (1 - cos(theta))`` as required.

Usage
-----
python benchmarks/xval_openlbmpm_wetting/analytic_cap_convention.py \
       --thetas 30,60,90,120,150 --radius 20 --xi 2 --out cap_convention.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tensorlbm.cg_wetting_geometric import (  # noqa: E402
    isotropic_color_gradient,
    readback_wall_angle,
    rotate_wall_color_gradient,
)


def cap_phase(ny: int, nx: int, theta_deg: float, radius: float, xi: float,
              y_wall: float = 0.5, solid_bottom: bool = True) -> tuple[np.ndarray, dict]:
    """Exact spherical cap as a phase field (``+1`` inside the drop)."""
    th = math.radians(theta_deg)
    x_c = 0.5 * nx
    y_c = y_wall - radius * math.cos(th)
    y = np.arange(ny, dtype=np.float64)[:, None]
    x = np.arange(nx, dtype=np.float64)[None, :]
    r = np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)
    d = radius - r
    if solid_bottom:
        d[0, :] = -np.inf  # wall: no phase inside the solid
    phi = np.tanh(np.clip(d, -50.0, 50.0) / xi)
    return phi, {"x_centre": x_c, "y_centre": y_c, "radius": radius,
                 "footprint": 2.0 * radius * math.sin(th),
                 "apex": radius * (1.0 - math.cos(th)), "y_wall": y_wall}


def angle_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    na = math.hypot(*a)
    nb = math.hypot(*b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    c = (a[0] * b[0] + a[1] * b[1]) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def run_case(theta_deg: float, *, radius: float, xi: float, ny: int, nx: int,
             mag_tol: float = 1e-3) -> dict:
    phi_np, geo = cap_phase(ny, nx, theta_deg, radius, xi)
    phi = torch.as_tensor(phi_np, dtype=torch.float64)
    solid = torch.zeros(ny, nx, dtype=torch.bool)
    solid[0, :] = True

    gx, gy = isotropic_color_gradient(phi, solid, periodic=(True, False))
    rx, ry = rotate_wall_color_gradient(gx, gy, solid, float(theta_deg), periodic=(True, False))

    # wall-adjacent fluid nodes where the interface actually lives
    thr = math.radians(float(theta_deg))
    x_c, y_c, R = geo["x_centre"], geo["y_centre"], geo["radius"]
    contact_xs = (x_c - R * math.sin(thr), x_c + R * math.sin(thr))
    rows = [1]  # first fluid row next to the y=0 wall
    samples = []
    for j in rows:
        for i in range(nx):
            gm = math.hypot(float(gx[j, i]), float(gy[j, i]))
            if gm < mag_tol:
                continue
            # only the interface region, i.e. near the contact line / cap
            if abs(float(phi_np[j, i])) > 0.95:
                continue
            nx_i = -float(gx[j, i]) / gm   # kernel's g_hat = -g/|g|
            ny_i = -float(gy[j, i]) / gm
            rx_i, ry_i = float(rx[j, i]), float(ry[j, i])
            rm = math.hypot(rx_i, ry_i)
            if rm < mag_tol:
                continue
            # true outward normal of the cap at this node
            dx, dy = (i - x_c), (j - y_c)
            samples.append({
                "x": i, "y": j, "phi": float(phi_np[j, i]),
                "g_in": (nx_i, ny_i),
                "g_out": (rx_i / rm, ry_i / rm),
                "n_true": (dx / math.hypot(dx, dy), dy / math.hypot(dx, dy)),
                "n_s": (0.0, -1.0),  # fluid -> solid at the bottom wall
                "mag_in": gm,
            })
    if not samples:
        return {"theta_cmd": float(theta_deg), "n_samples": 0, "error": "no interface nodes on row 1"}

    # errors of the rotated gradient against the analytic cap normal
    err_rot = [angle_deg(s["g_out"], s["n_true"]) for s in samples]
    err_in = [angle_deg(s["g_in"], s["n_true"]) for s in samples]
    # read-back angles (the round-1 circular metric), for reference
    readback = readback_wall_angle(rx, ry, solid, periodic=(True, False))
    rb_vals = [float(v) for v in np.asarray(readback).ravel() if math.isfinite(float(v)) and abs(float(v)) > 0]
    return {
        "theta_cmd": float(theta_deg),
        "n_samples": len(samples),
        "err_rot_vs_cap_normal_deg_mean": float(np.mean(err_rot)),
        "err_rot_vs_cap_normal_deg_max": float(np.max(err_rot)),
        "err_in_vs_cap_normal_deg_mean": float(np.mean(err_in)),
        "mirror_prediction_deg": abs(180.0 - 2.0 * float(theta_deg)),
        "geometry": geo,
        "readback_angle_abs_mean_deg": float(np.mean(np.abs(rb_vals))) if rb_vals else float("nan"),
        "samples": samples,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thetas", default="30,60,90,120,150")
    ap.add_argument("--radius", type=float, default=20.0)
    ap.add_argument("--xi", type=float, default=2.0)
    ap.add_argument("--ny", type=int, default=80)
    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--out", default="cap_convention.json")
    args = ap.parse_args()

    cases = {}
    print(f"{'theta':>6} {'n':>4} {'err(rot vs cap normal)':>22} {'mirror pred':>11} "
          f"{'err(in vs cap)':>14} {'readback':>9}")
    for t in [float(s) for s in args.thetas.split(",") if s.strip()]:
        c = run_case(t, radius=args.radius, xi=args.xi, ny=args.ny, nx=args.nx)
        cases[f"theta{t:g}"] = c
        print(f"{t:6.1f} {c.get('n_samples', 0):4d} "
              f"{c.get('err_rot_vs_cap_normal_deg_mean', float('nan')):10.3f} (max "
              f"{c.get('err_rot_vs_cap_normal_deg_max', float('nan')):6.2f}) "
              f"{c.get('mirror_prediction_deg', float('nan')):11.2f} "
              f"{c.get('err_in_vs_cap_normal_deg_mean', float('nan')):14.3f} "
              f"{c.get('readback_angle_abs_mean_deg', float('nan')):9.2f}", flush=True)
    Path(args.out).write_text(json.dumps({"params": vars(args), "cases": cases}, indent=2))
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())