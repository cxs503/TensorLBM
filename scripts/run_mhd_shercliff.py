#!/usr/bin/env python
"""Run the Shercliff (insulating-wall) branch of the MHD physics-surface campaign.

Writes, under ``--outdir`` (default ``artifacts/mhd_shercliff``):

* ``shercliff_ladder.json``   - every run, grid-convergence table, criteria
* ``shercliff_profiles.csv``  - node-wise profiles vs the closed form
* ``shercliff_summary.csv``   - per-run scalars (errors, wall shear, current)
* ``shercliff_criteria.json`` - machine-checkable verdicts (thresholds vs worst case)
* ``fig1_shercliff_profiles.png``, ``fig2_damping_law.png``,
  ``fig3_current_density.png``, ``fig4_energy_split.png``,
  ``fig5_grid_convergence.png``

Usage::

    python scripts/run_mhd_shercliff.py --outdir artifacts/mhd_shercliff --workers 8
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorlbm.mhd.hartmann import (  # noqa: E402
    HartmannConfig,
    run_hartmann_channel,
)
from tensorlbm.mhd.shercliff import (  # noqa: E402
    DEFAULT_SHERCLIFF_HA,
    SHERCLIFF_THRESHOLDS,
    ShercliffCase,
    evaluate_shercliff_criteria,
    interpolate_profile,
    run_shercliff_channel,
    shercliff_grad_for_mean,
    shercliff_mean_velocity,
    shercliff_plan,
    shercliff_profile,
    shercliff_wall_shear_coefficient,
)

NU_LATTICE = 0.1
U_MEAN_LATTICE = 0.01
STEP_CAP = 400_000


def _run_one_ha(payload: tuple[float, list[int], int]) -> tuple[float, list[dict]]:
    """Run one Hartmann number over its grid sequence (grid-sequenced start-up)."""
    ha, ny_list, step_cap = payload
    items: list[dict] = []
    u_prev: list[float] | None = None
    for ny in ny_list:
        case = ShercliffCase(
            ha=float(ha),
            ny=int(ny),
            nx=1,
            u_mean=U_MEAN_LATTICE,
            nu=NU_LATTICE,
            steps=int(step_cap),
            check_every=1000,
        )
        u_init = None if u_prev is None else interpolate_profile(u_prev, int(ny))
        run = run_shercliff_channel(case, u_init=u_init)
        u_prev = run["u_profile"]
        items.append(run)
        print(
            f"[shercliff] Ha={ha:5.2f} ny={ny:4d} steps={run['steps']:6d} "
            f"conv={run['converged']} rel_l2={run['rel_l2']:.3e} "
            f"rel_mean={run['rel_mean']:.3e} net_j={run['net_current_relative']:.2e}",
            flush=True,
        )
    return float(ha), items


def _prescribed_gradient_check(step_cap: int) -> dict:
    """Independent check: prescribe ``G`` and predict ``u_mean`` from the law alone.

    Everywhere else ``G`` is *derived* from the target ``u_mean`` (which makes the
    comparison partly self-consistent).  Here ``G`` is fixed by the Poiseuille
    scaling of a reference run, and the Shercliff law
    ``u_mean = G H^2/(mu f(Ha))`` predicts the flow rate - a genuine prediction.
    """
    ha, ny, mu = 5.0, 128, NU_LATTICE
    half = 0.5 * ny
    grad = shercliff_grad_for_mean(0.01, 0.0, half, mu)  # Poiseuille-equivalent G
    case = ShercliffCase(
        ha=ha, ny=ny, nx=1, u_mean=0.01, nu=mu, grad=grad, steps=step_cap, check_every=1000
    )
    run = run_shercliff_channel(case)
    predicted = shercliff_mean_velocity(grad, ha, half, mu)
    return {
        "ha": ha,
        "ny": ny,
        "grad": grad,
        "u_mean_predicted": predicted,
        "u_mean_lbm": run["u_mean_lbm"],
        "rel_error": abs(run["u_mean_lbm"] - predicted) / predicted,
        "profile_rel_l2": run["rel_l2"],
        "steps": run["steps"],
        "converged": run["converged"],
    }


def _conducting_reference_profile(ha: float, ny: int, step_cap: int) -> dict:
    """Conducting-wall (electrode) profile at the same ``u_mean`` for the contrast."""
    config = HartmannConfig(
        ny=ny,
        nx=1,
        ha=ha,
        u_mean=U_MEAN_LATTICE,
        nu=NU_LATTICE,
        wall_mode="conducting",
        steps=step_cap,
        check_every=1000,
    )
    res = run_hartmann_channel(config)
    # Conducting walls: E_z is imposed (K = loading), so j_z = sigma B0 (u - K).
    j_cond = res.u_profile / U_MEAN_LATTICE - config.loading / U_MEAN_LATTICE
    return {
        "ha": ha,
        "ny": ny,
        "y_hat": [float(v) for v in res.y_hat],
        "u_profile": [float(v) for v in res.u_profile],
        "j_normalised": [float(v) for v in j_cond],
        "loading_k_over_u_mean": config.loading / U_MEAN_LATTICE,
        "mean_j_normalised": float(np.mean(j_cond)),
    }


def run_campaign(
    outdir: Path,
    workers: int = 8,
    step_cap: int = STEP_CAP,
) -> dict:
    """Run the Shercliff ladder, write every artefact; returns the criteria dict."""
    outdir.mkdir(parents=True, exist_ok=True)
    ha_list = list(DEFAULT_SHERCLIFF_HA)
    plan = shercliff_plan(ha_list)
    t0 = time.time()
    payload = [(float(ha), plan[float(ha)], step_cap) for ha in ha_list]
    ctx = mp.get_context("fork")
    results: dict[float, list[dict]] = {}
    with ctx.Pool(processes=min(workers, len(payload))) as pool:
        for ha, items in pool.imap_unordered(_run_one_ha, payload):
            results[ha] = items
    elapsed = time.time() - t0
    runs = {ha: sorted(items, key=lambda it: it["ny"]) for ha, items in results.items()}

    criteria = evaluate_shercliff_criteria(runs)

    timings = {"ladder_seconds": elapsed}
    t1 = time.time()
    prescribed = _prescribed_gradient_check(step_cap)
    conducting = _conducting_reference_profile(5.0, 128, step_cap)
    timings["checks_seconds"] = time.time() - t1
    criteria["prescribed_gradient_check"] = prescribed | {
        "threshold": 2.0e-2,
        "passed": prescribed["rel_error"] <= 2.0e-2,
    }

    ladder = {
        "campaign": "p422w3-n08-mhd-lbm / Shercliff (insulating-wall) branch",
        "physical_model": {
            "closure": "insulating walls, zero net axial current, low-Rm (quasi-static B)",
            "E_z": "-B0 <u_x>",
            "j_z": "sigma B0 (u_x - <u_x>)",
            "lorentz_force": "sigma B0^2 (<u_x> - u_x)  (zero mean)",
            "analytic_profile": "u = u_p [1 - cosh(Ha y/H)/cosh Ha],  u_p = u_mean/m(Ha)",
            "analytic_mean": "u_mean = G H^2/(mu f(Ha)),  f = Ha^2 (1-m)/m",
            "wall_shear": "tau_w H/(mu u_mean) = Ha tanh(Ha)/m(Ha)",
        },
        "lattice_parameters": {
            "nu": NU_LATTICE,
            "u_mean_target": U_MEAN_LATTICE,
            "step_cap": step_cap,
            "scheme": "BGK + Guo forcing (2nd order), half-way bounce-back walls",
            "nx": 1,
        },
        "plan": {f"{ha:g}": plan[ha] for ha in plan},
        "runs": {
            f"{ha:g}": [{k: v for k, v in it.items() if k not in ("y_hat", "u_profile", "u_profile_analytic")} for it in items]
            for ha, items in runs.items()
        },
        "conducting_reference": conducting,
        "criteria": criteria,
        "timings": timings,
    }
    (outdir / "shercliff_ladder.json").write_text(json.dumps(ladder, indent=2, default=float))
    (outdir / "shercliff_criteria.json").write_text(
        json.dumps(
            {
                "campaign": ladder["campaign"],
                "thresholds": criteria["thresholds"],
                "worst_case": criteria["worst_case"],
                "verdicts": criteria["verdicts"],
                "all_passed": criteria["all_passed"],
                "per_ha": criteria["per_ha"],
                "prescribed_gradient_check": criteria["prescribed_gradient_check"],
                "poiseuille_limit": criteria["poiseuille_limit"],
            },
            indent=2,
            default=float,
        )
    )

    # --- CSV tables ---------------------------------------------------------
    with (outdir / "shercliff_profiles.csv").open("w") as fh:
        fh.write("ha,ny,y_hat,u_lbm,u_analytic,u_normalised,rel_error\n")
        for ha, items in runs.items():
            for it in items:
                u_an = np.asarray(it["u_profile_analytic"])
                for y, ul, ua in zip(it["y_hat"], it["u_profile"], u_an):
                    fh.write(
                        f"{ha:g},{it['ny']},{y:.10f},{ul:.12e},{ua:.12e},"
                        f"{ul / it['u_mean_lbm']:.10f},{abs(ul - ua) / abs(it['u_mean_analytic']):.6e}\n"
                    )
    with (outdir / "shercliff_summary.csv").open("w") as fh:
        fh.write(
            "ha,ny,steps,converged,u_mean_lbm,u_mean_analytic,rel_mean,rel_l2,rel_linf,"
            "tau_coeff_lbm,tau_coeff_analytic,rel_wall_shear,net_current_relative,"
            "net_current_integral_relative,"
            "joule_fraction,viscous_fraction,energy_closure\n"
        )
        for ha, items in runs.items():
            for it in items:
                e = it["energy"]
                fh.write(
                    f"{ha:g},{it['ny']},{it['steps']},{it['converged']},"
                    f"{it['u_mean_lbm']:.12e},{it['u_mean_analytic']:.12e},"
                    f"{it['rel_mean']:.6e},{it['rel_l2']:.6e},{it['rel_linf']:.6e},"
                    f"{it['tau_wall_coefficient_lbm']:.6f},"
                    f"{it['tau_wall_coefficient_analytic']:.6f},"
                    f"{it['rel_wall_shear']:.6e},{it['net_current_relative']:.6e},"
                    f"{it['net_current_integral_relative']:.6e},"
                    f"{e['joule_fraction']:.6f},{e['viscous_fraction']:.6f},"
                    f"{e['closure_relative']:.3e}\n"
                )

    _figures(outdir, runs, criteria, conducting)
    return criteria


def _figures(outdir: Path, runs: dict, criteria: dict, conducting: dict) -> None:
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

    # --- fig 1: profiles vs closed form -------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    for ax, ha in zip(axes, (1.0, 5.0, 20.0)):
        items = runs[ha]
        fine = items[-1]
        y = np.asarray(fine["y_hat"])
        u = np.asarray(fine["u_profile"]) / fine["u_mean_lbm"]
        ua = shercliff_profile(y, ha, 1.0)
        ax.plot(ua, y, "-", color="#264653", lw=2, label="closed form")
        for it, marker in zip(items, ("o", "s", "^")):
            ax.plot(
                np.asarray(it["u_profile"]) / it["u_mean_lbm"],
                np.asarray(it["y_hat"]),
                marker,
                ms=3,
                mfc="none",
                color="#e76f51",
                label=f"LBM ny={it['ny']}",
            )
        ax.set_title(f"Ha={ha:g}  $L_2$={fine['rel_l2']:.1e}")
        ax.set_xlabel(r"$u/u_{mean}$")
        ax.set_ylabel(r"$y/H$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Shercliff (insulating-wall) channel: LBM vs closed form")
    fig.tight_layout()
    fig.savefig(outdir / "fig1_shercliff_profiles.png")
    plt.close(fig)

    # --- fig 2: damping law u_mean(Ha) at fixed driving gradient ------------
    ha_grid = np.logspace(-1, 1.7, 200)
    mu, half = NU_LATTICE, 0.5 * 128
    grad = shercliff_grad_for_mean(U_MEAN_LATTICE, 0.0, half, mu)
    u_sher = np.array([shercliff_mean_velocity(grad, h, half, mu) for h in ha_grid])
    # conducting walls at the same pressure gradient: u_mean = G H^2/(mu Ha^2) * ...
    u_hart = np.array(
        [
            grad * half**2 / (mu * (h * h)) * (1.0 - 1.0 / max(h, 1e-12) * math.tanh(h))
            for h in ha_grid
        ]
    )
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.loglog(ha_grid, u_sher, "-", color="#2a9d8f", lw=2, label="insulating (Shercliff)")
    ax.loglog(ha_grid, u_hart, "--", color="#e76f51", lw=2, label="conducting (electrodes)")
    ha_pts = [ha for ha, items in sorted(runs.items()) if ha > 0]
    ax.loglog(
        ha_pts,
        [runs[ha][-1]["u_mean_lbm"] for ha in ha_pts],
        "o",
        color="#264653",
        ms=5,
        label="LBM, insulating",
    )
    ax.set_xlabel("Ha")
    ax.set_ylabel(r"$u_{mean}$  (lattice, fixed $\Delta p$)")
    ax.set_title(r"Damping law: $Ha^{-1}$ vs $Ha^{-2}$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "fig2_damping_law.png")
    plt.close(fig)

    # --- fig 3: current density closure contrast ----------------------------
    ha = 5.0
    fine = runs[ha][-1]
    y = np.asarray(fine["y_hat"])
    j_ins = np.asarray(fine["u_profile"]) / fine["u_mean_lbm"] - 1.0
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot(j_ins, y, "-", color="#2a9d8f", lw=2, label=r"insulating: $j_z/\sigma B_0 u_{mean}$")
    ax.plot(conducting["j_normalised"], np.asarray(conducting["y_hat"]), "--", color="#e76f51", lw=2,
            label="conducting: $j_z/\\sigma B_0 u_{mean}$")
    ax.axvline(0.0, color="k", lw=0.6)
    ax.set_xlabel(r"$j_z / (\sigma B_0 u_{mean})$")
    ax.set_ylabel(r"$y/H$")
    ax.set_title(f"Current closure at Ha={ha:g}: net current "
                 f"{fine['net_current_relative']:.1e} (insulating)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "fig3_current_density.png")
    plt.close(fig)

    # --- fig 4: energy split ------------------------------------------------
    has = sorted(ha for ha in runs if ha > 0)
    joule = [runs[ha][-1]["energy"]["joule_fraction"] for ha in has]
    visc = [runs[ha][-1]["energy"]["viscous_fraction"] for ha in has]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(has, joule, "o-", color="#e76f51", label="Joule fraction")
    ax.plot(has, visc, "s-", color="#264653", label="viscous fraction")
    ax.plot(has, [j + v for j, v in zip(joule, visc)], "k--", lw=1, label="sum (energy closure)")
    ax.set_xlabel("Ha")
    ax.set_ylabel(r"$P_{diss}/P_{input}$")
    ax.set_title("Insulating channel energy split (no electrical take-off)")
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "fig4_energy_split.png")
    plt.close(fig)

    # --- fig 5: grid convergence -------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for ha in sorted(runs):
        items = runs[ha]
        nys = [it["ny"] for it in items]
        ax.loglog(nys, [max(it["rel_l2"], 1e-12) for it in items], "o-", label=f"Ha={ha:g}")
    ax.axhline(SHERCLIFF_THRESHOLDS["c_shercliff_profile_l2"], color="r", ls="--", lw=1,
               label="2 % gate")
    ax.set_xlabel(r"$n_y$ (fluid rows)")
    ax.set_ylabel(r"$L_2$ profile error")
    ax.set_title("Grid convergence of the Shercliff branch")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "fig5_grid_convergence.png")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="artifacts/mhd_shercliff")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--step-cap", type=int, default=STEP_CAP)
    args = parser.parse_args()
    verdict = run_campaign(Path(args.outdir), workers=args.workers, step_cap=args.step_cap)
    print(json.dumps(verdict["verdicts"], indent=2))
    print("all_passed:", verdict["all_passed"])
    return 0 if verdict["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())