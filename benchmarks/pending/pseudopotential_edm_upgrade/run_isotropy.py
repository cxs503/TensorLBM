"""Analytic (no time-marching) characterisation of the SC force discretisations.

Four measurements, all deterministic:

1. ``moment_table``       – isotropy moment conditions of each stencil
                            (Σ W c_x² = 1, Σ W c_x⁴ = 3 Σ W c_x²c_y²,
                            Σ W c_x⁶ = 5 Σ W c_x⁴c_y²).
2. ``circular_anisotropy``– on a smooth circular interface ψ = tanh((R-r)/w) the
                            force must be purely radial; measure the spurious
                            tangential component max|F_t| / max|F_r|
                            (the interface drive of spurious currents).
3. ``straight_anisotropy``– same measure for a straight interface at
                            θ = 0°, 15°, 30°, 45° to the lattice axes.
4. ``smooth_field_error`` – RMS error of the discrete gradient of a smooth
                            periodic field (amplitude accuracy, not isotropy).
5. ``uniform_exactness``  – interaction force of a *uniform* ψ (must vanish).

Run:  python run_isotropy.py [--out result_isotropy.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, save_json  # noqa: E402

from tensorlbm.multiphase_edm import (  # noqa: E402
    psi_gradient,
    _SHELLS,
    _STENCIL_SHELLS,
    sc_interaction_force,
    stencil,
    stencil_names,
)

STENCILS = list(stencil_names())
FORMS = ["product", "difference", "psi2"]


def moment_table() -> dict:
    out = {}
    for name in STENCILS:
        st = stencil(name)
        vec = np.array(st.vectors, dtype=float)
        wt = np.array(st.weights, dtype=float)
        m2 = float((wt * vec[:, 0] ** 2).sum())
        m4x = float((wt * vec[:, 0] ** 4).sum())
        m4xy = float((wt * vec[:, 0] ** 2 * vec[:, 1] ** 2).sum())
        m6x = float((wt * vec[:, 0] ** 6).sum())
        m6xy = float((wt * vec[:, 0] ** 4 * vec[:, 1] ** 2).sum())
        out[name] = {
            "n_neighbours": len(st.vectors),
            "shell_weights": st.shell_weights,
            "sum_W_cx2": m2,
            "sum_W_cx4_minus_3_cx2cy2": m4x - 3 * m4xy,
            "sum_W_cx6_minus_5_cx4cy2": m6x - 5 * m6xy,
            "isotropy_order": st.max_order,
        }
    return out


def _polar_split(Fx: torch.Tensor, Fy: torch.Tensor, er: torch.Tensor, et: torch.Tensor,
                 mask: torch.Tensor) -> tuple[float, float]:
    Fr = Fx * er + Fy * et
    Ft = -Fx * et + Fy * er
    return float(Fr[mask].abs().max()), float(Ft[mask].abs().max())


def circular_anisotropy(L: int = 128, R: float = 24.0, w: float = 2.0) -> dict:
    y, x = torch.meshgrid(torch.arange(L, dtype=torch.float32), torch.arange(L, dtype=torch.float32),
                          indexing="ij")
    c = 0.5 * (L - 1)
    dx, dy = x - c, y - c
    r = torch.sqrt(dx * dx + dy * dy).clamp(min=1e-12)
    er, et = dx / r, dy / r
    psi = torch.where(r > 1e-9, torch.tanh((R - r) / w), torch.ones_like(r))
    band = (r - R).abs() < 2.5 * w
    out = {}
    for name in STENCILS:
        for form in FORMS:
            Fx, Fy = sc_interaction_force(psi, -1.0, name, form, periodic=True)
            fr, ft = _polar_split(Fx, Fy, er, et, band)
            out[f"{name}|{form}"] = {
                "max_Fr": fr,
                "max_Ft": ft,
                "aniso_pct": 100.0 * ft / fr if fr > 0 else float("nan"),
            }
    return out


def straight_anisotropy(L: int = 128, w: float = 2.0,
                        angles=(0.0, 15.0, 30.0, 45.0)) -> dict:
    """Force on a straight tanh interface at angle θ; interior cells only."""
    out = {}
    yy, xx = torch.meshgrid(torch.arange(L, dtype=torch.float32), torch.arange(L, dtype=torch.float32),
                            indexing="ij")
    for ang in angles:
        th = math.radians(ang)
        nx_, ny_ = math.cos(th), math.sin(th)
        xi = (xx - L / 2) * nx_ + (yy - L / 2) * ny_   # signed distance to the interface
        psi = -torch.tanh(xi / w)                      # psi = -1 far side, +1 near side
        er, et = torch.full_like(xx, nx_), torch.full_like(xx, ny_)
        band = (xi.abs() < 2.5 * w) & (xx > 3) & (xx < L - 4) & (yy > 3) & (yy < L - 4)
        for name in STENCILS:
            for form in FORMS:
                Fx, Fy = sc_interaction_force(psi, -1.0, name, form, periodic=False)
                fr, ft = _polar_split(Fx, Fy, er, et, band)
                out.setdefault(f"{name}|{form}", {})[f"{ang:g}deg"] = {
                    "max_Fn": fr, "max_Ft": ft,
                    "aniso_pct": 100.0 * ft / fr if fr > 0 else float("nan"),
                }
    return out


def smooth_field_error(L: int = 128, periods: float = 1.0) -> dict:
    """RMS error of the discrete gradient of a smooth periodic field.

    Uses the gradient operator directly (``psi_gradient``) so that the ψ
    prefactor of the force does not enter.  Fields:
    ψ = sin(kx)cos(ky) (``periods`` = 1) and ψ = sin(kx) (``periods`` = 8,
    an interface-thickness-scale wave, k·h ≈ 0.4).
    """
    x = torch.arange(L, dtype=torch.float32)
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    out = {}
    cases = {
        "long_wave_kh=0.049": (1.0, torch.sin(2 * math.pi / L * xx) * torch.cos(2 * math.pi / L * yy)),
        "short_wave_kh=0.393": (8.0, torch.sin(2 * math.pi * 8 / L * xx) * torch.cos(2 * math.pi * 8 / L * yy)),
    }
    for cname, (k, psi) in cases.items():
        kk = 2 * math.pi * k / L
        ex = kk * torch.cos(kk * xx) * torch.cos(kk * yy)
        ey = -kk * torch.sin(kk * xx) * torch.sin(kk * yy)
        scale = float(torch.sqrt(ex**2 + ey**2).max())
        for name in STENCILS:
            gx, gy = psi_gradient(psi, name, "difference", True)
            err = torch.sqrt((gx - ex) ** 2 + (gy - ey) ** 2)
            out[f"{name}|{cname}"] = {
                "rms_pct": 100.0 * float(err.mean()) / scale,
                "max_pct": 100.0 * float(err.max()) / scale,
            }
    return out


def uniform_exactness(L: int = 32) -> dict:
    """Interaction force of a uniform ψ: must vanish identically (physical)."""
    out = {}
    for periodic in (True, False):
        psi = torch.full((L, L), 0.7)
        for name in STENCILS:
            for form in FORMS:
                Fx, Fy = sc_interaction_force(psi, -1.0, name, form, periodic=periodic)
                mag = torch.sqrt(Fx * Fx + Fy * Fy)
                out[f"{name}|{form}|periodic={periodic}"] = {
                    "max_abs_F": float(mag.max()),
                    "max_abs_F_interior": float(mag[3:-3, 3:-3].max()),
                }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="result_isotropy.json")
    args = ap.parse_args()
    torch.set_num_threads(4)
    res = {
        "moments": moment_table(),
        "circular": circular_anisotropy(),
        "straight": straight_anisotropy(),
        "smooth_field": smooth_field_error(),
        "uniform": uniform_exactness(),
    }
    save_json(Path(__file__).resolve().parent / args.out, res)

    print("== moment conditions ==")
    for k, v in res["moments"].items():
        print(f"{k:9s} n={v['n_neighbours']:2d}  M2={v['sum_W_cx2']:.6f}  "
              f"M4res={v['sum_W_cx4_minus_3_cx2cy2']:+.3e}  "
              f"M6res={v['sum_W_cx6_minus_5_cx4cy2']:+.4f}  order={v['isotropy_order']}")
    print("\n== circular interface anisotropy (max|Ft|/max|Fr|) ==")
    for k, v in res["circular"].items():
        print(f"{k:22s} {v['aniso_pct']:.4f}%")
    print("\n== straight interface anisotropy (max|Ft|/max|Fn|) ==")
    for k, v in res["straight"].items():
        print(f"{k:22s} " + "  ".join(f"{a}:{d['aniso_pct']:.3f}%" for a, d in v.items()))
    print("\n== smooth field gradient error ==")
    for k, v in res["smooth_field"].items():
        print(f"{k:34s} rms={v['rms_pct']:.4f}%  max={v['max_pct']:.4f}%")
    print("\n== uniform-ψ exactness (max|F|) ==")
    for k, v in res["uniform"].items():
        print(f"{k:32s} {v['max_abs_F']:.3e}  interior {v['max_abs_F_interior']:.3e}")


if __name__ == "__main__":
    main()