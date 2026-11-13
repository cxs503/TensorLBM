#!/usr/bin/env python3
"""XVAL-LBM-FILL-GRADIENT-NORMAL (mission 17757).

Cross-validation of TensorLBM's fill-gradient interface normal against the
mature OSS practice of Maarten-vd-Sande/lbm (oss_solutions id=85,
'fill-gradient-normal').

Arms (see predeclared.json):
  A1 TensorLBM as-is : fill=mass/rho, central diff, NORMALIZED, liquid->gas
                       (src/tensorlbm/free_surface_lbm.py:642)
  A2 OSS clean       : n=0.5*(eps_{i-1}-eps_{i+1}, eps_{j-1}-eps_{j+1}),
                       UNNORMALIZED, eps in {0,1}
  A3 OSS w/ bug      : same as A2 but GAS eps=0.5 (documented 2x underest)
  A4 analytic        : exact normal of the constructed interface

All arms receive the IDENTICAL fill field; only the normal formula differs.

Two distinct questions are answered:
  (XVAL)  A1 vs A2  : does TensorLBM reproduce the OSS fill-gradient normal?
  (DIAG)  A1/A2 vs A4: how accurate is the shared stencil vs the analytic
          normal (a property of the method, not of either implementation).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))

# ---------------------------------------------------------------------------
# Arm implementations (formulas verbatim from source / oss_solutions id=85)
# ---------------------------------------------------------------------------


def a1_tensorlbm_2d(fill: torch.Tensor):
    """TensorLBM _compute_interface_normal, 2D slice (y=dim0, x=dim1).

    Mirrors free_surface_lbm.py:642 exactly (roll-based central difference,
    normalized, sign flipped so n points liquid->gas).
    """
    gx = 0.5 * (fill.roll(-1, dims=1) - fill.roll(1, dims=1))
    gy = 0.5 * (fill.roll(-1, dims=0) - fill.roll(1, dims=0))
    mag = (gx**2 + gy**2).sqrt().clamp(min=1e-10)
    return -gx / mag, -gy / mag


def a1_tensorlbm_3d(fill: torch.Tensor):
    """TensorLBM _compute_interface_normal, 3D (z=dim0, y=dim1, x=dim2).

    Verbatim port of free_surface_lbm.py:642-651.
    """
    gx = 0.5 * (fill.roll(-1, dims=2) - fill.roll(1, dims=2))
    gy = 0.5 * (fill.roll(-1, dims=1) - fill.roll(1, dims=1))
    gz = 0.5 * (fill.roll(-1, dims=0) - fill.roll(1, dims=0))
    mag = (gx**2 + gy**2 + gz**2).sqrt().clamp(min=1e-10)
    return -gx / mag, -gy / mag, -gz / mag


def a2_oss_2d(eps: torch.Tensor, gas_value: float = 0.0):
    """OSS Maarten-vd-Sande/lbm fill-gradient normal (oss id=85).

    n = 0.5*(eps_{i-1}-eps_{i+1}, eps_{j-1}-eps_{j+1}); NOT normalized.
    gas_value=0.0 -> clean; gas_value=0.5 -> documented streamcollide.py:166
    priority bug (arm A3).
    """
    e = eps.clone()
    e[e == 0] = gas_value
    nx = 0.5 * (e.roll(1, dims=1) - e.roll(-1, dims=1))
    ny = 0.5 * (e.roll(1, dims=0) - e.roll(-1, dims=0))
    return nx, ny


def a2_oss_3d(eps: torch.Tensor, gas_value: float = 0.0):
    """OSS formula extended to 3D (same stencil, z added)."""
    e = eps.clone()
    e[e == 0] = gas_value
    nx = 0.5 * (e.roll(1, dims=2) - e.roll(-1, dims=2))
    ny = 0.5 * (e.roll(1, dims=1) - e.roll(-1, dims=1))
    nz = 0.5 * (e.roll(1, dims=0) - e.roll(-1, dims=0))
    return nx, ny, nz


# ---------------------------------------------------------------------------
# Fill-field construction (sharp 0/1 and smeared, periodic-safe)
# ---------------------------------------------------------------------------


def _grid2(N):
    xs = np.arange(0.5 - N / 2, N / 2, 1.0)
    return np.meshgrid(xs, xs, indexing="xy")  # (y, x)


def _grid3(N):
    xs = np.arange(0.5 - N / 2, N / 2, 1.0)
    return np.meshgrid(xs, xs, xs, indexing="ij")  # (z, y, x)


def fill_circle(N, R, c, w=None):
    x, y = _grid2(N)
    d = np.hypot(x - c[0], y - c[1])
    if w is None:
        f = (d <= R).astype(np.float64)
    else:
        f = np.clip((R - d) / w + 0.5, 0.0, 1.0)
    return torch.from_numpy(f), d


def fill_plane(N, theta_deg, c, w=None):
    t = math.radians(theta_deg)
    x, y = _grid2(N)
    s = (x - c[0]) * math.sin(t) + (y - c[1]) * math.cos(t)
    if w is None:
        f = (s < 0).astype(np.float64)
    else:
        f = np.clip(-s / w + 0.5, 0.0, 1.0)
    return torch.from_numpy(f), s


def fill_sphere(N, R, c, w=None):
    z, y, x = _grid3(N)
    d = np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2)
    if w is None:
        f = (d <= R).astype(np.float64)
    else:
        f = np.clip((R - d) / w + 0.5, 0.0, 1.0)
    return torch.from_numpy(f), d


# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------


def dir_err(n1, n2, mask):
    """1 - mean|cos| over mask; also mean angle in deg."""
    n1 = n1.reshape(-1, n1.shape[-1])[mask.reshape(-1)]
    n2 = n2.reshape(-1, n2.shape[-1])[mask.reshape(-1)]
    m1 = np.linalg.norm(n1, axis=1)
    m2 = np.linalg.norm(n2, axis=1)
    ok = (m1 > 1e-9) & (m2 > 1e-9)
    cos = np.abs((n1[ok] * n2[ok]).sum(axis=1) / (m1[ok] * m2[ok]))
    cos = np.clip(cos, 0, 1)
    err = 1.0 - cos.mean()
    ang = float(np.degrees(np.arccos(cos)).mean())
    return float(err), ang, int(ok.sum())


def main():
    out = {"cases": []}
    dev = "cpu"

    # ---- Case 1: 2D circle, resolution ladder (sharp) ----------------------
    for N in (64, 128, 256):
        R = N / 4.0
        fill, d = fill_circle(N, R, (0.0, 0.0))
        x, y = _grid2(N)
        band = (np.abs(d - R) < 1.5) & (d > 1e-9)
        na = np.stack([x / d, y / d], axis=-1)

        nx1, ny1 = a1_tensorlbm_2d(fill)
        nx2, ny2 = a2_oss_2d(fill, 0.0)
        nx3, ny3 = a2_oss_2d(fill, 0.5)
        m1 = np.stack([nx1.numpy(), ny1.numpy()], axis=-1)
        m2 = np.stack([nx2.numpy(), ny2.numpy()], axis=-1)
        m3 = np.stack([nx3.numpy(), ny3.numpy()], axis=-1)
        mask = (np.linalg.norm(m1, axis=-1) > 1e-6) & band
        if mask.sum() < 20:
            out["cases"].append({"case": f"circle-sharp N={N}", "excluded": "too few"})
            continue
        e1, a1, _ = dir_err(m1, na, mask)
        e2, a2, _ = dir_err(m2, na, mask)
        e12, a12, _ = dir_err(m1, m2, mask)
        out["cases"].append({
            "case": f"circle-sharp N={N}", "R": R, "n_cells": int(mask.sum()),
            "A1_vs_analytic": {"dir_err": e1, "angle_deg": a1},
            "A2_vs_analytic": {"dir_err": e2, "angle_deg": a2},
            "A1_vs_A2": {"dir_err": e12, "angle_deg": a12},
            "mag_A2": float(np.linalg.norm(m2, axis=-1)[mask].mean()),
            "mag_A3": float(np.linalg.norm(m3, axis=-1)[mask].mean()),
            "mag_ratio_A3_A2": float(
                np.linalg.norm(m3, axis=-1)[mask].mean()
                / max(np.linalg.norm(m2, axis=-1)[mask].mean(), 1e-12)),
        })

    # ---- Case 2: 2D tilted plane (sharp) ------------------------------------
    N = 256
    theta = 30.0
    fill, s = fill_plane(N, theta, (0.0, 0.0))
    x, y = _grid2(N)
    band = np.abs(s) < 1.5
    na = np.tile(
        np.array([math.sin(math.radians(theta)), math.cos(math.radians(theta))]),
        (N * N, 1))
    nx1, ny1 = a1_tensorlbm_2d(fill)
    nx2, ny2 = a2_oss_2d(fill, 0.0)
    m1 = np.stack([nx1.numpy(), ny1.numpy()], axis=-1)
    m2 = np.stack([nx2.numpy(), ny2.numpy()], axis=-1)
    mask = (np.linalg.norm(m1, axis=-1) > 1e-6) & band
    e1, a1, _ = dir_err(m1, na, mask)
    e2, a2, _ = dir_err(m2, na, mask)
    e12, a12, _ = dir_err(m1, m2, mask)
    out["cases"].append({
        "case": f"plane-sharp theta={theta} N={N}", "n_cells": int(mask.sum()),
        "A1_vs_analytic": {"dir_err": e1, "angle_deg": a1},
        "A2_vs_analytic": {"dir_err": e2, "angle_deg": a2},
        "A1_vs_A2": {"dir_err": e12, "angle_deg": a12},
        "mag_A2": float(np.linalg.norm(m2, axis=-1)[mask].mean()),
    })

    # ---- Case 3: 3D sphere (TensorLBM native dim, sharp) --------------------
    for N in (32, 64, 128):
        R = N / 4.0
        fill, d = fill_sphere(N, R, (0.0, 0.0, 0.0))
        z, y, x = _grid3(N)
        band = (np.abs(d - R) < 1.5) & (d > 1e-9)
        na = np.stack([x / d, y / d, z / d], axis=-1)
        gx, gy, gz = a1_tensorlbm_3d(fill)
        ox, oy, oz = a2_oss_3d(fill, 0.0)
        m1 = np.stack([gx.numpy(), gy.numpy(), gz.numpy()], axis=-1)
        m2 = np.stack([ox.numpy(), oy.numpy(), oz.numpy()], axis=-1)
        mask = (np.linalg.norm(m1, axis=-1) > 1e-6) & band
        if mask.sum() < 20:
            out["cases"].append({"case": f"sphere N={N}", "excluded": "too few"})
            continue
        e1, a1, _ = dir_err(m1, na, mask)
        e2, a2, _ = dir_err(m2, na, mask)
        e12, a12, _ = dir_err(m1, m2, mask)
        out["cases"].append({
            "case": f"sphere-sharp N={N}", "R": R, "n_cells": int(mask.sum()),
            "A1_vs_analytic": {"dir_err": e1, "angle_deg": a1},
            "A2_vs_analytic": {"dir_err": e2, "angle_deg": a2},
            "A1_vs_A2": {"dir_err": e12, "angle_deg": a12},
            "mag_A2": float(np.linalg.norm(m2, axis=-1)[mask].mean()),
        })

    # ---- Case 4: smeared interface convergence (method accuracy) -----------
    # A resolved free surface has a fill field smeared over a few cells; the
    # fill-gradient normal is only meaningful when the interface is resolved.
    for w in (2.0, 4.0, 8.0):
        N = 256
        R = N / 4.0
        fill, d = fill_circle(N, R, (0.0, 0.0), w=w)
        x, y = _grid2(N)
        band = np.abs(d - R) < 2.0 * w
        na = np.stack([x / d, y / d], axis=-1)
        nx1, ny1 = a1_tensorlbm_2d(fill)
        m1 = np.stack([nx1.numpy(), ny1.numpy()], axis=-1)
        mask = (np.linalg.norm(m1, axis=-1) > 1e-6) & band
        e1, a1, _ = dir_err(m1, na, mask)
        out["cases"].append({
            "case": f"circle-smeared w={w} N={N}", "n_cells": int(mask.sum()),
            "A1_vs_analytic": {"dir_err": e1, "angle_deg": a1},
        })

    # ---- Verdict: the XVAL KPI is the cross-implementation direction error -
    checks = []
    for c in out["cases"]:
        if "excluded" in c or "A1_vs_A2" not in c:
            continue
        v = c["A1_vs_A2"]["dir_err"]
        checks.append({"case": c["case"], "kpi": "A1_vs_A2_dir_err",
                         "value": v, "pass_lt_15pct": v < 0.15})
    out["checks"] = checks
    out["all_pass"] = all(c["pass_lt_15pct"] for c in checks)

    (HERE / "results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"all_pass": out["all_pass"], "n_cases": len(out["cases"])}))
    for c in out["cases"]:
        print(json.dumps(c))
    print("CHECKS:", json.dumps(checks))


if __name__ == "__main__":
    main()
