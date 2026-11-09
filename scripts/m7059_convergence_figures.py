#!/usr/bin/env python3
"""Figures for mission m7059 (TensorLBM second-order convergence evidence).

Every number plotted here is read from
``artifacts/lbm_convergence_m7059/convergence_evidence.json`` (produced by
``python -m tensorlbm.convergence_mms``) or recomputed by the same module for the
field snapshots -- nothing is hand-typed.

    PYTHONPATH=src python scripts/m7059_convergence_figures.py

Writes PNGs next to the evidence record:

    fig1_mms_order.png        MMS: L2 error vs h, force on / force off, slopes 1 and 2
    fig2_poiseuille_order.png Poiseuille: three wall/reference conventions
    fig3_mms_fields.png       MMS numerical field, analytic field and pointwise error
    fig4_poiseuille_profiles.png  Profile + wall-position offset + error distribution
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.convergence_mms import (  # noqa: E402
    CS2,
    poiseuille_exact,
    run_mms_2d,
    run_poiseuille_2d,
)

ART = ROOT / "artifacts" / "lbm_convergence_m7059"
EV = ART / "convergence_evidence.json"


def _hs(case: dict) -> list[float]:
    if "n" in case["levels"][0]:
        return [1.0 / lv["n"] for lv in case["levels"]]
    return [1.0 / lv["n_f"] for lv in case["levels"]]


def _errs(case: dict) -> list[float]:
    return [lv["l2_rel"] for lv in case["levels"]]


def _slope_ref(ax, h: np.ndarray, e0: float, p: float, label: str, style="--") -> None:
    ax.plot(h, e0 * (h / h[0]) ** p, style, color="0.55", lw=1.0)
    ax.annotate(
        label,
        xy=(h[-1], e0 * (h[-1] / h[0]) ** p),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=9,
        color="0.4",
    )


def fig_mms(ev: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    styles = {
        "mms_force": ("o", "#1f77b4", "MMS + Guo force (second order)"),
        "mms_no_force": ("s", "#d62728", "negative control: force off (no convergence)"),
    }
    for name, (marker, color, label) in styles.items():
        case = ev["cases"][name]
        h = np.array(_hs(case))
        e = np.array(_errs(case))
        p = case["order_fit"]["p"]
        ax.loglog(h, e, marker + "-", color=color, label=f"{label}, p = {p:.3f}")
    case = ev["cases"]["mms_force"]
    h = np.array(_hs(case))
    _slope_ref(ax, h, _errs(case)[0] * 0.6, 2.0, "slope 2")
    _slope_ref(ax, h, _errs(case)[0] * 0.6, 1.0, "slope 1")
    ax.set_xlabel("lattice spacing h (cells, h = 1/ n)")
    ax.set_ylabel(r"relative $L_2$ velocity error")
    ax.set_title("M2: manufactured Taylor-Green solution, D2Q9 BGK + Guo force")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = ART / "fig1_mms_order.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_poiseuille(ev: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    styles = {
        "poiseuille_bouzidi_true_wall": ("o", "#1f77b4", "Bouzidi q=1/4, exact true walls"),
        "poiseuille_halfway_true_wall": ("s", "#d62728", "half-way BB, exact true walls"),
        "poiseuille_halfway_effective_wall": (
            "^",
            "#2ca02c",
            r"half-way BB, reference at $y_w=0.5\Delta$",
        ),
    }
    for name, (marker, color, label) in styles.items():
        case = ev["cases"][name]
        h = np.array(_hs(case))
        e = np.array(_errs(case))
        p = case["order_fit"]["p"]
        ax.loglog(h, e, marker + "-", color=color, label=f"{label}, p = {p:.3f}")
    h = np.array(_hs(ev["cases"]["poiseuille_bouzidi_true_wall"]))
    _slope_ref(ax, h, _errs(ev["cases"]["poiseuille_bouzidi_true_wall"])[0] * 0.5, 2.0, "slope 2")
    _slope_ref(ax, h, _errs(ev["cases"]["poiseuille_halfway_true_wall"])[0] * 0.5, 1.0, "slope 1")
    ax.axhline(1e-2, color="0.7", lw=0.8, ls=":")
    ax.annotate(
        "1 % error line",
        xy=(h[0], 1e-2),
        xytext=(2, 3),
        textcoords="offset points",
        fontsize=8,
        color="0.4",
    )
    ax.set_xlabel("lattice spacing h (cells, h = 1/ n_f)")
    ax.set_ylabel(r"relative $L_2$ velocity error")
    ax.set_title("Plane Poiseuille: the wall convention sets the measured order")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = ART / "fig2_poiseuille_order.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_mms_fields() -> Path:
    proto = json.loads(EV.read_text())["protocol"]
    n = 64
    r = run_mms_2d(n, u0=proto["u0"], force=True, tol=1e-11)
    ux, uy = r["fields"]["ux"], r["fields"]["uy"]
    ex, ey = r["fields"]["ux_exact"], r["fields"]["uy_exact"]
    speed = np.hypot(ux, uy)
    speed_ex = np.hypot(ex, ey)
    scale = float(speed_ex.max())
    err = np.hypot(ux - ex, uy - ey) / scale
    fig, axs = plt.subplots(1, 3, figsize=(12.0, 3.9))
    im0 = axs[0].imshow(speed, origin="lower", cmap="viridis")
    axs[0].set_title(f"numerical |u| (n={n})")
    im1 = axs[1].imshow(speed_ex, origin="lower", cmap="viridis")
    axs[1].set_title("analytic |u| (manufactured)")
    im2 = axs[2].imshow(err, origin="lower", cmap="magma")
    axs[2].set_title(f"|u_num - u_exact| / max|u|  (L2={r['l2_rel']:.2e})")
    for a, im in zip(axs, (im0, im1, im2)):
        a.set_xlabel("x [cells]")
        a.set_ylabel("y [cells]")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
    fig.suptitle(
        r"M2 manufactured solution: steady Taylor-Green vortex, force $F=2\rho\nu k^2 u$",
        fontsize=10,
    )
    fig.tight_layout()
    out = ART / "fig3_mms_fields.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_poiseuille_profiles() -> Path:
    proto = json.loads(EV.read_text())["protocol"]
    q = proto["poiseuille_wall_fraction_q"]
    u_max = proto["u_max"]
    tau = proto["tau"]
    nu = CS2 * (tau - 0.5)
    n_f = 32
    height = (n_f - 1) + 2 * q
    g = 8.0 * nu * u_max / height**2
    bfl = run_poiseuille_2d(n_f, q=q, wall_model="bouzidi", g=g, init="exact", tol=1e-12)
    hbb = run_poiseuille_2d(n_f, q=q, wall_model="halfway", g=g, init="exact", tol=1e-12)
    exact = poiseuille_exact(bfl["y"], g, nu, 0.0, height)
    exact_eff = poiseuille_exact(bfl["y"], g, nu, q - 0.5, q + n_f - 1 + 0.5)

    fig, axs = plt.subplots(1, 3, figsize=(13.0, 3.9))
    ax = axs[0]
    ax.plot(exact / exact.max(), bfl["y"], "k-", lw=1.6, label="exact (true walls)")
    ax.plot(bfl["ux"] / exact.max(), bfl["y"], "o", ms=3, color="#1f77b4", label="Bouzidi q=1/4")
    ax.plot(
        hbb["ux"] / exact.max(),
        hbb["y"],
        "s",
        ms=3,
        mfc="none",
        color="#d62728",
        label="half-way BB",
    )
    ax.axhline(0.0, color="0.6", lw=1)
    ax.axhline(height, color="0.6", lw=1)
    ax.set_xlabel(r"$u_x / u_{max,exact}$")
    ax.set_ylabel("y [cells]")
    ax.set_title(f"profile, n_f={n_f}, q={q}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axs[1]
    ax.plot(
        (bfl["ux"] - exact) / exact.max(),
        bfl["y"],
        "o-",
        ms=3,
        lw=1,
        color="#1f77b4",
        label="Bouzidi, true-wall ref",
    )
    ax.plot(
        (hbb["ux"] - exact) / exact.max(),
        bfl["y"],
        "s-",
        ms=3,
        lw=1,
        color="#d62728",
        label="half-way, true-wall ref",
    )
    ax.plot(
        (hbb["ux"] - exact_eff) / exact.max(),
        bfl["y"],
        "^--",
        ms=3,
        lw=1,
        color="#2ca02c",
        label=r"half-way, ref at $y_w=0.5\Delta$",
    )
    ax.axvline(0.0, color="0.6", lw=1)
    ax.set_xlabel(r"$(u_{num}-u_{ref})/u_{max}$")
    ax.set_ylabel("y [cells]")
    ax.set_title("error distribution across the channel")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    # wall offset of the half-way model, measured from its own profile
    ax = axs[2]
    from tensorlbm.convergence_mms import study_poiseuille

    st = study_poiseuille(
        (12, 24, 48), q=q, wall_model="halfway", reference="true_wall", u_max=u_max, tol=1e-9
    )
    nfs = [lv["n_f"] for lv in st["levels"]]
    ax.plot(nfs, [lv["wall_offset_left"] for lv in st["levels"]], "o-", label="fitted wall, left")
    ax.plot(nfs, [lv["wall_offset_right"] for lv in st["levels"]], "s-", label="fitted wall, right")
    ax.axhline(-(0.5 - q), color="0.5", ls=":", label=r"predicted $-(0.5-q)$")
    ax.axhline(+(0.5 - q), color="0.5", ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("n_f")
    ax.set_ylabel(r"fitted wall - true wall  [cells]")
    ax.set_title(r"half-way BB: wall sits at $0.5\Delta$, geometry says $q\Delta$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle(
        "Plane Poiseuille: the first-order error is a wall-position offset, not a scheme error",
        fontsize=10,
    )
    fig.tight_layout()
    out = ART / "fig4_poiseuille_profiles.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    if not EV.exists():
        raise SystemExit(f"evidence record missing: {EV}")
    ev = json.loads(EV.read_text())
    outs = [fig_mms(ev), fig_poiseuille(ev), fig_mms_fields(), fig_poiseuille_profiles()]
    for p in outs:
        print(p.relative_to(ROOT), p.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
