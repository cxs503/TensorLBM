#!/usr/bin/env python3
"""Provenance script (m3634): grid-refinement study behind the thresholds in
``tests/test_lbm_verification.py`` and the figure in
``docs/lbm_analytic_verification_r1.md``.

Every number is produced by calling the test-module functions directly, so the
published table cannot drift from what the tests assert.

    python scripts/verify_lbm_analytic_convergence.py

Writes ``artifacts/m3634/lbm_analytic_convergence.json`` and
``artifacts/m3634/lbm_analytic_convergence.png``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import test_lbm_verification as tv  # noqa: E402

OUT = ROOT / "artifacts" / "m3634"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data: dict = {
        "device": str(tv.DEVICE),
        "dtype": str(tv.DTYPE).replace("torch.", ""),
        "shear": [],
        "poiseuille": [],
        "couette": [],
    }

    for nx in (16, 32, 64):
        r = tv.shear_wave_viscosity(nx=nx)
        data["shear"].append(
            {
                "nx": nx,
                "kdx": r["k"],
                "nu_exact": r["nu_exact"],
                "nu_measured": r["nu_measured"],
                "err_pct": r["err_pct"],
            }
        )
        print(f"shear nx={nx:3d} k={r['k']:.4f} err={r['err_pct']:.3e}%", flush=True)

    for ny in (8, 12, 16):
        r = tv.poiseuille_profile(ny=ny)
        data["poiseuille"].append(
            {
                "ny": ny,
                "height": r["height"],
                "peak_err_pct": r["peak_err_pct"],
                "shape_err_pct": r["shape_err_pct"],
                "tau_w_err_pct": r["tau_w_err_pct"],
                "u_peak": max(r["u"]),
                "u_peak_exact": max(r["u_exact"]),
            }
        )
        print(
            f"poiseuille ny={ny:2d} peak={r['peak_err_pct']:.4f}% tau_w={r['tau_w_err_pct']:.4f}%",
            flush=True,
        )

    for ny in (8, 12):
        r = tv.couette_profile(ny=ny)
        data["couette"].append(
            {"ny": ny, "profile_err_pct": r["profile_err_pct"], "cf_err_pct": r["cf_err_pct"]}
        )
        print(f"couette ny={ny:2d} profile={r['profile_err_pct']:.3e}%", flush=True)

    (OUT / "lbm_analytic_convergence.json").write_text(json.dumps(data, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ax = axes[0]
    kdx = [p["kdx"] for p in data["shear"]]
    err = [max(p["err_pct"], 1e-12) for p in data["shear"]]
    ax.loglog(kdx, err, "o-", label="measured $|\\nu_{num}-\\nu_{exact}|/\\nu_{exact}$")
    ref = [err[0] * (k / kdx[0]) ** 2 for k in kdx]
    ax.loglog(kdx, ref, "k--", lw=1, label="slope 2 ($\\propto k^2$)")
    ax.set_xlabel("$k\\,\\Delta x$")
    ax.set_ylabel("relative error [%]")
    ax.set_title("(a) shear-wave viscosity, $\\tau=1$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    ny = [p["ny"] for p in data["poiseuille"]]
    ax.semilogy(ny, [p["peak_err_pct"] for p in data["poiseuille"]], "s-", label="peak velocity")
    ax.semilogy(ny, [p["shape_err_pct"] for p in data["poiseuille"]], "^-", label="profile shape")
    ax.semilogy(
        ny, [p["tau_w_err_pct"] for p in data["poiseuille"]], "d-", label="wall shear $\\tau_w$"
    )
    ax.axhline(tv.TOL_POISEUILLE_PROFILE_PCT, color="r", ls=":", lw=1, label="test tolerance")
    ax.set_xlabel("channel height $n_y-2$ [lattice nodes]")
    ax.set_ylabel("relative error [%]")
    ax.set_title("(b) Poiseuille vs analytic parabola")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[2]
    r = tv.poiseuille_profile(ny=12)
    y = list(range(1, r["ny"] - 1))
    ax.plot(y, r["u_exact"], "k-", lw=2, label="analytic $G(y-y_{w1})(y_{w2}-y)/2\\nu$")
    ax.plot(y, r["u"], "o", ms=5, mfc="none", label="LBM (library Guo body force)")
    ax.set_xlabel("$y$ [lattice]")
    ax.set_ylabel("$u_x$ [lattice]")
    ax.set_title(f"(c) Poiseuille profile, $n_y$={r['ny']} (peak err {r['peak_err_pct']:.3f}%)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    fig.suptitle(
        f"TensorLBM D3Q19 BGK analytic verification (float64, CPU, {os.uname().machine})",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "lbm_analytic_convergence.png", dpi=130)
    print("wrote", OUT / "lbm_analytic_convergence.png", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
