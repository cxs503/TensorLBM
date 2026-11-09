#!/usr/bin/env python
"""P155W1-N01: Stefan phase-change verification campaign (TensorLBM).

Produces the machine-auditable verification chain for phase-change heat
conduction:

  C1  1-D planar freezing  — front position vs Neumann similarity solution (<= 2 %)
  C2  2-D cylindrical solidification — front radius vs an independent
      sharp-interface reference (<= 3 %) and front *shape* (circularity, <= 3 %)
  C3  melting energy closure — wall heat input vs stored enthalpy (<= 1 %)
  C4  grid convergence — observed order of accuracy of front and profile error
  C5  exact-solution self-verification (PDE residual / interface energy balance)
  C6  pytest suite

All numbers in ``artifacts/stefan_phase_change/results.json`` are written by this
script (no hand-copied values); figures are written to
``artifacts/stefan_phase_change/figures``.

Usage
-----
    PYTHONPATH=src python benchmarks/verified/stefan_phase_change/run.py \\
        [--quick] [--out artifacts/stefan_phase_change]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from tensorlbm.phase_change.stefan_exact import (
    CylindricalSimilarity,
    NeumannSimilarity,
    cylindrical_lambda,
    planar_lambda,
)
from tensorlbm.phase_change.stefan_verification import (
    EnthalpyThermalLBM,
    quasi_steady_radial_front,
    run_cylindrical_wall_solidification,
    run_grid_convergence,
    run_planar_stefan,
    sharp_interface_radial_solidification,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def exact_solution_audit(alpha: float = 1.0 / 6.0, ste: float = 0.5) -> dict:
    """Self-verification of the closed-form solutions used as references.

    * PDE residual of the similarity profile (planar solid/liquid branches and the
      cylindrical branch),
    * interface energy balance ``rho*L*v = k*|dT/dn|`` at the front,
    * boundary values ``T(0) == T_wall`` and ``T(front) == T_freeze``.
    """
    import scipy.special as sp

    out: dict = {"alpha": alpha, "stefan_number": ste}

    # --- cylindrical branch -------------------------------------------------
    cyl = CylindricalSimilarity(jacob_number=ste, alpha=alpha)
    t = 3000.0
    rf = cyl.front(t)
    r = np.linspace(rf + 3.0, rf + 60.0, 400)
    T = cyl.temperature(r, t)
    Tr = (cyl.temperature(r + 1.0, t) - cyl.temperature(r - 1.0, t)) / 2.0
    Trr = cyl.temperature(r + 1.0, t) - 2.0 * T + cyl.temperature(r - 1.0, t)
    Tt = (cyl.temperature(r, t + 1.0) - cyl.temperature(r, t - 1.0)) / 2.0
    res = Tt - alpha * (Trr + Tr / r)
    eps = 1e-5
    grad = abs(
        (cyl.temperature(rf + eps, t) - cyl.temperature(rf + 2 * eps, t)) / eps
    )
    out["cylindrical"] = {
        "lambda": cyl.lam,
        "pde_residual_rel": float(np.max(np.abs(res)) / cyl.dT),
        "interface_balance_rel": abs(cyl.latent * cyl.front_velocity(t) - cyl.cp * alpha * grad)
        / (cyl.latent * cyl.front_velocity(t)),
        "T_at_front_rel": abs(float(cyl.temperature(rf, t)) - cyl.t_freeze) / cyl.dT,
        "T_far_rel_error": abs(float(cyl.temperature(rf + 200.0, t)) - cyl.far_field_temperature)
        / cyl.dT,
    }

    # --- planar branches ----------------------------------------------------
    planar = {}
    for side in ("solid", "liquid"):
        ex = NeumannSimilarity(stefan_number=ste, alpha=alpha, side=side)
        t = 20000.0
        s = ex.front(t)
        x = np.linspace(2.0, s - 3.0, 400)
        T = ex.temperature(x, t)
        Txx = ex.temperature(x + 1.0, t) - 2.0 * T + ex.temperature(x - 1.0, t)
        Tt = (ex.temperature(x, t + 1.0) - ex.temperature(x, t - 1.0)) / 2.0
        eps = 1e-5
        grad = abs((ex.temperature(s - eps, t) - ex.temperature(s - 2 * eps, t)) / eps)
        planar[side] = {
            "lambda": ex.lam,
            "pde_residual_rel": float(np.max(np.abs(Tt - alpha * Txx)) / ex.dT),
            "interface_balance_rel": abs(ex.latent * ex.front_velocity(t) - ex.cp * alpha * grad)
            / (ex.latent * ex.front_velocity(t)),
            "T_at_wall_rel": abs(float(ex.temperature(0.0, t)) - ex.boundary_temperature) / ex.dT,
            "T_at_front_rel": abs(float(ex.temperature(s, t)) - ex.t_freeze) / ex.dT,
        }
    out["planar"] = planar

    # --- LBE diffusion operator against the analytic erfc solution ----------
    nx, n_t = 400, 4000
    sim = EnthalpyThermalLBM(
        nx=nx, ny=1, tau=1.0, cp=1.0, latent=1.0, t_freeze=0.0, periodic=(False, True)
    )
    # Diffusion from a Dirichlet step at x=0 -> T = erfc(x/(2 sqrt(alpha t))).
    # The whole domain is kept on the *liquid* branch (f_l = 1) so that no latent
    # heat is exchanged and the thermal LBE operator is tested in isolation.
    sim.set_field(np.zeros((nx, 1)), np.ones((nx, 1)))
    sim.add_dirichlet(np.array([[True]] + [[False]] * (nx - 1)), 1.0, 1.0)
    sim.add_dirichlet(np.array([[False]] * (nx - 1) + [[True]]), 0.0, 1.0)
    sim.step(n_t)
    x = np.arange(nx, dtype=float)
    ana = 1.0 - np.array([float(sp.erf(xi / (2.0 * math.sqrt(alpha * n_t)))) for xi in x])
    band = np.abs(x) < nx - 20
    out["pure_diffusion"] = {
        "steps": n_t,
        "linf_rel": float(np.max(np.abs(sim.T[:, 0] - ana)[band])),
        "l2_rel": float(np.sqrt(np.mean((sim.T[band, 0] - ana[band]) ** 2))),
        "alpha": alpha,
    }

    # --- closed-domain enthalpy conservation --------------------------------
    sim2 = EnthalpyThermalLBM(nx=64, ny=64, tau=1.0, cp=1.0, latent=1.0, t_freeze=0.0,
                              periodic=(True, True))
    rng = np.random.default_rng(0)
    T = -0.2 + 0.4 * rng.random((64, 64))
    sim2.set_field(T)
    H0 = sim2.total_enthalpy()
    sim2.step(500)
    out["closed_domain_conservation"] = {
        "steps": 500,
        "H0": H0,
        "rel_drift": abs(sim2.total_enthalpy() - H0) / abs(H0),
    }
    return out


def criteria(results: dict) -> dict:
    """Machine-checkable acceptance criteria for the node."""
    c = {}

    planar_frz = results["planar_freezing"]["final"]
    c["C1_planar_front_error"] = {
        "value": planar_frz["front_rel_error"],
        "threshold": 0.02,
        "metric": "conservative phase-integral front vs Neumann, nx=400 (s=100 cells)",
        "pass": bool(planar_frz["front_rel_error"] <= 0.02),
    }
    c["C1b_planar_front_error_local"] = {
        "value": planar_frz["front_local_rel_error"],
        "threshold": 0.02,
        "metric": "f_l=0.5 crossing front vs Neumann (independent of energy bookkeeping)",
        "pass": bool(planar_frz["front_local_rel_error"] <= 0.02),
    }
    c["C1c_planar_profile_error"] = {
        "value": planar_frz["profile_error"]["l2_rel"],
        "threshold": 0.02,
        "metric": "L2 temperature profile error (interface band excluded)",
        "pass": bool(planar_frz["profile_error"]["l2_rel"] <= 0.02),
    }

    cyl = results["cylindrical_wall_solidification"]["final"]
    c["C2_cyl_front_vs_reference"] = {
        "value": cyl["r_rel_error_vs_reference"],
        "threshold": 0.03,
        "metric": "2-D front radius vs independent sharp-interface reference",
        "pass": bool(cyl["r_rel_error_vs_reference"] <= 0.03),
    }
    c["C2b_cyl_front_shape"] = {
        "value": cyl["shape_max_dev_rel"],
        "threshold": 0.03,
        "metric": "max |r(theta)-mean(r)|/mean(r) (exact front is a circle)",
        "pass": bool(cyl["shape_max_dev_rel"] <= 0.03),
    }

    melt = results["planar_melting"]["energy"]
    c["C3_melting_energy_closure"] = {
        "value": melt["closure_2nd_order"],
        "threshold": 0.01,
        "metric": "wall heat input (2nd-order flux) vs stored enthalpy change",
        "pass": bool(melt["closure_2nd_order"] <= 0.01),
    }
    c["C3b_wall_input_vs_exact"] = {
        "value": melt["wall_input_2nd_vs_exact_rel_error"],
        "threshold": 0.01,
        "metric": "cumulative wall heat input vs analytic Neumann value",
        "pass": bool(melt["wall_input_2nd_vs_exact_rel_error"] <= 0.01),
    }
    c["C3c_latent_vs_input_minus_sensible"] = {
        "value": melt["latent_vs_input_minus_sensible"],
        "threshold": 0.01,
        "metric": "latent absorbed vs (input - stored sensible)",
        "pass": bool(melt["latent_vs_input_minus_sensible"] <= 0.01),
    }

    conv = results["grid_convergence"]
    orders = conv["observed_order"]
    errs = [r["front_rel_error"] for r in conv["runs"]]
    c["C4_grid_convergence"] = {
        "observed_orders": orders,
        "front_errors": errs,
        "threshold": (
            "monotone decrease in error; front (phase-integral) order in [0.7, 1.6] "
            "(theoretical first order for a diffuse-interface enthalpy method); "
            "temperature-profile order >= 1.0"
        ),
        "pass": bool(
            all(errs[i + 1] < errs[i] for i in range(len(errs) - 1))
            and 0.7 <= orders["front_rel_error"] <= 1.6
            and orders["profile_l2_rel"] >= 1.0
        ),
    }

    audit = results["exact_solution_audit"]
    worst = max(
        audit["cylindrical"]["pde_residual_rel"],
        audit["cylindrical"]["interface_balance_rel"],
        audit["planar"]["solid"]["pde_residual_rel"],
        audit["planar"]["solid"]["interface_balance_rel"],
        audit["planar"]["liquid"]["pde_residual_rel"],
        audit["planar"]["liquid"]["interface_balance_rel"],
    )
    c["C5_exact_solution_selfcheck"] = {
        "value": worst,
        "threshold": 1e-4,
        "metric": "max PDE residual / interface energy balance residual of the references",
        "pass": bool(worst <= 1e-4),
    }
    c["C5b_pure_diffusion"] = {
        "value": audit["pure_diffusion"]["linf_rel"],
        "threshold": 1e-2,
        "metric": "LBE thermal operator vs analytic erfc diffusion",
        "pass": bool(audit["pure_diffusion"]["linf_rel"] <= 1e-2),
    }
    c["C5c_conservation"] = {
        "value": audit["closed_domain_conservation"]["rel_drift"],
        "threshold": 1e-12,
        "metric": "total enthalpy drift in a closed periodic 2-D domain",
        "pass": bool(audit["closed_domain_conservation"]["rel_drift"] <= 1e-12),
    }
    c["all_pass"] = bool(all(v.get("pass", False) for v in c.values() if isinstance(v, dict)))
    return c


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(res: dict, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    figs = {}

    # --- figure 1: planar front history ------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for mode, color in (("freezing", "C0"), ("melting", "C3")):
        data = res[f"planar_{mode}"]
        t = [e["time"] for e in data["series"]]
        s = [e["front_integral"] for e in data["series"]]
        se = [e["front_exact"] for e in data["series"]]
        err = [100 * e["front_rel_error"] for e in data["series"]]
        ax[0].plot(t, se, "-", color="k", lw=1.2,
                   label="Neumann exact" if mode == "freezing" else None)
        ax[0].plot(t, s, "o", color=color, ms=5,
                   label=f"LBM integral ({mode})")
        ax[1].plot(t, err, "o-", color=color, ms=4, label=f"{mode}")
    ax[0].set_xlabel("lattice step t")
    ax[0].set_ylabel("front position s/Δx")
    ax[0].set_title("1-D Stefan front vs Neumann exact solution")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].axhline(2.0, color="r", ls="--", lw=1, label="2% criterion")
    ax[1].set_xlabel("lattice step t")
    ax[1].set_ylabel("front relative error [%]")
    ax[1].set_title("front error history (nx=400)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = outdir / "fig1_planar_front.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs["fig1_planar_front"] = p

    # --- figure 2: profiles ------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, mode in enumerate(("freezing", "melting")):
        prof = res[f"planar_{mode}"]["profile"]
        x = np.array(prof["x"])
        ax[k].plot(x, prof["T_exact"], "-", color="k", lw=1.4, label="exact")
        ax[k].plot(x, prof["T_num"], "o", color="C0", ms=3, label="LBM")
        ax2 = ax[k].twinx()
        ax2.plot(x, prof["f_l"], color="C3", lw=1.2, ls="--", label="liquid fraction")
        ax2.set_ylabel("liquid fraction $f_l$", color="C3")
        ax2.tick_params(axis="y", colors="C3")
        ax2.set_ylim(-0.05, 1.05)
        ax[k].set_xlabel("x/Δx")
        ax[k].set_ylabel("temperature (lattice units)")
        ax[k].set_title(f"{mode}: T and $f_l$ at final time")
        ax[k].legend(fontsize=8, loc="lower right")
        ax[k].grid(alpha=0.3)
    fig.tight_layout()
    p = outdir / "fig2_planar_profiles.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs["fig2_planar_profiles"] = p

    # --- figure 3: convergence --------------------------------------------
    conv = res["grid_convergence"]
    h = np.array([r["n_front"] for r in conv["runs"]], dtype=float)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for key, label, mk in (
        ("front_rel_error", "front (phase integral)", "o"),
        ("front_local_rel_error", "front ($f_l=0.5$ crossing)", "s"),
        ("profile_l2_rel", "temperature profile L2", "^"),
    ):
        y = np.array([r[key] for r in conv["runs"]], dtype=float)
        ax.loglog(h, y, mk + "-", label=f"{label}: p={conv['observed_order'][key]:.2f}")
    ax.loglog(h, 1.0 / h, "k:", lw=1, label="slope -1")
    ax.set_xlabel("front resolution $n_{front}=s(0.25\\,L)/\\Delta x$")
    ax.set_ylabel("relative error at final time")
    ax.set_title("grid convergence (1-D planar freezing)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = outdir / "fig3_convergence.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs["fig3_convergence"] = p

    # --- figure 4: cylindrical case ---------------------------------------
    cyl = res["cylindrical_wall_solidification"]
    fig = plt.figure(figsize=(12, 4.2))
    ax0 = fig.add_subplot(1, 3, 1)
    n = cyl["field"]["n"]
    row = np.array(cyl["field"]["f_l_centre_row"])
    ax0.imshow(np.tile(row, (n, 1)).T, origin="lower", cmap="coolwarm")
    ax0.set_title("$f_l$ (centre column, r →)\n1=liquid, 0=solid")
    ax0.set_xlabel("y/Δx")
    ax0.set_ylabel("x/Δx")

    ax1 = fig.add_subplot(1, 3, 2, projection="polar")
    th = np.array(cyl["polar"]["theta"])
    rad = np.array(cyl["polar"]["radius"])
    ax1.plot(th, rad, "o-", color="C0", ms=3, label="LBM front $r(\\theta)$")
    ax1.plot(th, np.full_like(th, rad.mean()), "k--", lw=1, label="circle (exact)")
    ax1.set_title("2-D front shape")
    ax1.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.28))

    ax2 = fig.add_subplot(1, 3, 3)
    t = [e["time"] for e in cyl["series"]]
    ax2.plot(t, [e["r_reference"] for e in cyl["series"]], "k-", lw=1.6,
             label="sharp-interface reference")
    ax2.plot(t, [e["r_area"] for e in cyl["series"]], "o", color="C0", ms=4,
             label="LBM (liquid area)")
    ax2.plot(t, [e["r_quasi_steady"] for e in cyl["series"]], "s--", color="C1", ms=3,
             label="Planck quasi-steady")
    ax2.set_xlabel("lattice step t")
    ax2.set_ylabel("front radius $r_f$")
    ax2.set_title("cylindrical solidification")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    p = outdir / "fig4_cylindrical.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs["fig4_cylindrical"] = p

    # --- figure 5: melting energy audit -----------------------------------
    melt = res["planar_melting"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    t = [e["time"] for e in melt["series"]]
    ax[0].plot(t, [abs(e["wall_heat_input_second_order"]) for e in melt["series"]], "o-",
               color="C0", label="∫q_wall dt (2nd-order flux)")
    ax[0].plot(t, [abs(e["wall_heat_input_first_order"]) for e in melt["series"]], "+",
               color="C2", label="∫q_wall dt (1st-order, identity)")
    ax[0].plot(t, [abs(e["wall_heat_input_exact"]) for e in melt["series"]], "k--",
               label="analytic Neumann")
    ax[0].plot(t, [abs(e["energy_change"]) for e in melt["series"]], "s", ms=3,
               color="C3", label="ΔH stored (sensible+latent)")
    ax[0].set_xlabel("lattice step t")
    ax[0].set_ylabel("energy per unit depth")
    ax[0].set_title("melting: boundary input vs stored enthalpy")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)
    ax[1].plot(t, [100 * abs(abs(e["wall_heat_input_second_order"]) - abs(e["energy_change"]))
                   / abs(e["energy_change"]) for e in melt["series"]], "o-", color="C0",
               label="closure (2nd-order flux)")
    ax[1].plot(t, [100 * abs(abs(e["wall_heat_input_exact"]) - abs(e["energy_change"]))
                   / abs(e["energy_change"]) for e in melt["series"]], "s-", color="C3",
               label="vs analytic input")
    ax[1].axhline(1.0, color="r", ls="--", lw=1, label="1% criterion")
    ax[1].set_xlabel("lattice step t")
    ax[1].set_ylabel("relative energy imbalance [%]")
    ax[1].set_title("energy closure history")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = outdir / "fig5_energy_closure.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs["fig5_energy_closure"] = p
    return figs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/stefan_phase_change")
    ap.add_argument("--quick", action="store_true", help="coarse grid / short runs")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nx = 200 if args.quick else 400
    resolutions = (25, 50) if args.quick else (25, 50, 100)

    t0 = time.time()
    res: dict = {
        "node": "P155W1-N01",
        "repo": "TensorLBM",
        "module": "tensorlbm.phase_change.stefan_verification",
        "generated_by": "benchmarks/verified/stefan_phase_change/run.py",
        "lattice_units": {"dx": 1.0, "dt": 1.0, "rho": 1.0, "cp": 1.0, "latent": 1.0,
                          "t_freeze": 0.0},
    }

    print("[1/5] exact-solution self-verification ...")
    res["exact_solution_audit"] = exact_solution_audit()

    print("[2/5] 1-D planar freezing (Neumann) ...")
    res["planar_freezing"] = run_planar_stefan(mode="freezing", nx=nx)

    print("[3/5] 1-D planar melting + energy audit ...")
    res["planar_melting"] = run_planar_stefan(mode="melting", nx=nx)

    print("[4/5] 2-D cylindrical solidification ...")
    res["cylindrical_wall_solidification"] = run_cylindrical_wall_solidification(
        r_wall=30 if args.quick else 40, shell0=4, sample_count=8
    )

    print("[5/5] grid convergence study ...")
    res["grid_convergence"] = run_grid_convergence(resolutions=resolutions, mode="freezing")

    res["criteria"] = criteria(res)
    res["wall_clock_s"] = time.time() - t0

    figures = make_figures(res, out / "figures")
    res["figures"] = {k: str(v) for k, v in figures.items()}

    for key, fig in figures.items():
        res.setdefault("figure_paths", {})[key] = str(fig)

    payload = out / "results.json"
    payload.write_text(json.dumps(res, indent=2, default=float))

    summary = {
        k: (round(v["value"], 6) if isinstance(v.get("value"), (int, float)) else v.get("pass"))
        for k, v in res["criteria"].items()
        if isinstance(v, dict) and "value" in v
    }
    print(json.dumps({"criteria": summary, "all_pass": res["criteria"]["all_pass"],
                      "wall_clock_s": round(res["wall_clock_s"], 1)}, indent=2))
    print(f"wrote {payload}")
    return 0 if res["criteria"]["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())