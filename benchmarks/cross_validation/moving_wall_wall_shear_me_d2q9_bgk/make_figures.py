"""Figures for XVAL-LBM-MOVING-WALL-WALL-SHEAR-ME-D2Q9-BGK (matplotlib -> png)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mwforce as mf  # noqa: E402

ART = HERE / "artifacts"
S = json.loads((ART / "summary.json").read_text())
FIG = Path("outputs/cross_validation/moving_wall_wall_shear_me_d2q9_bgk/figures").resolve()
FIG.mkdir(parents=True, exist_ok=True)


def fig1_couette() -> dict:
    cases = S["couette"]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ref = next(c for c in cases if c["ny"] == 34 and c["tau"] == 0.8)
    # the profile is not kept in summary.json (only the error metric is): recompute
    # the reference case once with a short averaging window for the plot only
    r = mf.run_couette_shear(ny=34, nx=4, tau=0.8, u0=0.05, n_steps=8000, avg_steps=20)
    y = np.arange(34, dtype=float)
    ax[0].plot(r["u_num"], y, "o", ms=3.0, label="LBM (halfway moving wall)")
    ax[0].plot(
        0.05 * np.clip((y - 0.5) / 32.0, 0, 1), y, "-", lw=1.2, label=r"analytic $U_0(y-0.5)/H$"
    )
    ax[0].set_xlabel(r"$u_x$  [lattice]")
    ax[0].set_ylabel(r"$y$  [lattice]")
    ax[0].set_title(
        f"steady Couette, $H$=32, $\\tau$=0.8\n(max |$\\Delta u$| = "
        f"{ref['u_max_err_pct']:.1e} % of $U_0$)"
    )
    ax[0].legend(loc="upper left", fontsize=8)
    ax[0].grid(alpha=0.3)

    labels = [f"H={int(c['ny'] - 2)}, τ={c['tau']}" for c in cases]
    errs = [max(abs(c["tau_err_pct"]), abs(c["tau_bot_err_pct"])) for c in cases]
    ax[1].bar(labels, errs)
    ax[1].set_yscale("log")
    ax[1].axhline(15.0, color="r", ls="--", lw=1.0, label="15 % (owner standard)")
    ax[1].set_ylabel("wall-shear rel. error  [%]")
    ax[1].set_title(r"$\tau_w$ = momentum exchange vs $\rho\nu U_0/H$")
    for i, e in enumerate(errs):
        ax[1].text(i, e, f"{e:.1e}", ha="center", va="bottom", fontsize=8)
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = FIG / "fig1_couette.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return {
        "name": "fig1_couette.png",
        "path": p,
        "caption": "Steady plane Couette: velocity profile (left) and wall shear from "
        "momentum exchange vs the exact N-S value for four grids "
        "(right). Errors are at round-off, i.e. the discrete ME force of the "
        "halfway moving-wall closure equals rho*nu*U0/H_eff to machine precision.",
    }


def fig2_stokes() -> dict:
    ref = next(c for c in S["stokes2"] if c["tag"] == "REF")
    t = np.array(ref["t"])
    fn = np.array(ref["fx_num"])
    fa = np.array(ref["fx_ana"])
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax[0].plot(t, fn, lw=1.6, label="LBM momentum exchange")
    ax[0].plot(t, fa, "--", lw=1.1, label="analytic $-\\tau_{fluid}$ free (derived)")
    ax[0].set_xlabel("time step  $t$  [lattice]")
    ax[0].set_ylabel(r"$F_{wall}/A$  [lattice]")
    ax[0].set_title(
        "Stokes' 2nd problem: wall force over one period\n"
        f"amp err {ref['amp_err_pct']:+.3f} %, phase err "
        f"{ref['phase_err_deg']:+.3f}°"
    )
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    rows = S["stokes2"]
    lbl = [f"τ={r['tau']}, δ={int(r['delta_bl'])}, U0={r['U0']}" for r in rows]
    amp = [r["amp_err_pct"] for r in rows]
    ph = [r["phase_err_deg"] for r in rows]
    x = np.arange(len(rows))
    ax[1].bar(x - 0.2, amp, 0.4, label="amplitude err [%]")
    ax[1].bar(x + 0.2, ph, 0.4, label="phase err [deg]")
    ax[1].axhline(15.0, color="r", ls="--", lw=1.0)
    ax[1].axhline(-15.0, color="r", ls="--", lw=1.0)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(lbl, rotation=30, ha="right", fontsize=7)
    ax[1].set_title("parameter sweep (15 % / 15 deg gates)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = FIG / "fig2_stokes2.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return {
        "name": "fig2_stokes2.png",
        "path": p,
        "caption": "Stokes' second problem (oscillating wall, delta_bl = 8): measured wall "
        "force over one period vs the analytic viscous traction (left); "
        "amplitude/phase error over the parameter sweep (right).",
    }


def fig3_rayleigh() -> dict:
    ref = next(c for c in S["rayleigh"] if c["tag"] == "REF")
    t = np.array(ref["t"])
    fn = np.abs(np.array(ref["fx_num"]))
    fa = np.abs(np.array(ref["fx_ana"]))
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax[0].loglog(t, fa, "--", lw=1.2, label=r"analytic $\rho U_0\sqrt{\nu/\pi t}$")
    ax[0].loglog(t, fn, lw=1.4, label="LBM momentum exchange")
    ax[0].set_xlabel("time step  $t$  [lattice]")
    ax[0].set_ylabel(r"$|F_{wall}|/A$  [lattice]")
    ax[0].set_title("Rayleigh impulsive start-up: wall shear decay")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")
    rel = np.abs(np.array(ref["fx_num"]) - np.array(ref["fx_ana"])) / np.max(fa)
    ax[1].semilogy(t, rel * 100.0, lw=1.2)
    ax[1].axhline(15.0, color="r", ls="--", lw=1.0, label="15 %")
    ax[1].set_xlabel("time step  $t$  [lattice]")
    ax[1].set_ylabel("rel. error of $\\tau_w$  [%]")
    ax[1].set_title(f"max {ref['err_pct_max_in_win']:.2e} % over $t\\in[400,1600]$")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = FIG / "fig3_rayleigh.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return {
        "name": "fig3_rayleigh.png",
        "path": p,
        "caption": "Impulsive (Rayleigh) start-up of the wall: |tau_w| vs the half-space "
        "similarity solution on log-log axes, and the relative error history; "
        "the power-law decay is reproduced without any rescaling.",
    }


def fig4_ablation() -> dict:
    abl = {c["variant"]: c for c in S["ablation_stokes2"]}
    order = ["ladd_pre_local", "ladd_pre_unitrho", "eq_wall_node", "static_bb", "ladd_sign_flip"]
    vals = [
        abs(abl[v]["amp_err_pct"]) if np.isfinite(abl[v]["amp_err_pct"]) else 100.0 for v in order
    ]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax[0].barh(order, vals, color=["C2"] + ["C0"] * (len(order) - 1))
    ax[0].set_xlabel("wall-force amplitude error  [%]")
    ax[0].set_xlim(0, max(120, max(vals) * 1.05))
    ax[0].axvline(15.0, color="r", ls="--", lw=1.0, label="15 %")
    ax[0].set_title("ablation of the wall closure (same Stokes case)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, axis="x")

    d = S["normal_velocity_diagnostic"]["runs"]
    eps = [r["uw_n"] for r in d]
    rate = [-r["rate_per_step"] for r in d]
    ax[1].plot(eps, rate, "o-")
    fit = S["normal_velocity_diagnostic"]["fit_rate_vs_uw_n"]
    ax[1].set_xlabel(r"normal wall speed  $u_w\cdot n$")
    ax[1].set_ylabel("fluid-mass loss rate  [per step]")
    ax[1].set_title(
        "D1: closure is not mass-neutral for $u_w\\cdot n\\neq0$\n"
        f"linear fit slope {fit['slope']:.4f}, $R^2$={fit['r2']:.4f}"
    )
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = FIG / "fig4_ablation_drift.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return {
        "name": "fig4_ablation_drift.png",
        "path": p,
        "caption": "Left: wall-force amplitude error of the reference closure vs four "
        "ablations (the case discriminates). Right: measured fluid-mass loss "
        "rate when the wall carries a velocity component along its own normal "
        "(out of contract; quantified, not corrected).",
    }


def main() -> None:
    figs = [fig1_couette(), fig2_stokes(), fig3_rayleigh(), fig4_ablation()]
    meta = []
    for f in figs:
        meta.append(
            {
                "name": f["name"],
                "caption": f["caption"],
                "png": str(Path(f["path"]).relative_to(FIG.parent)),
            }
        )
        print("wrote", f["path"])
    (FIG.parent / "figures.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
