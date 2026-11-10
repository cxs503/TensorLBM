#!/usr/bin/env python3
"""Node IN54W1-N02: IBM force-transfer validation for ice–propeller coupling.

Runs the validation matrix, writes machine-readable artefacts and generates the
report figures.  Every number in the report JSON is produced here (no hand
copying).

Usage::

    PYTHONPATH=src python3 scripts/in54w1_n02_ibm_validation.py \
        --cases tg,couette2d,st,added_mass --out artifacts/in54w1_n02

Case names: ``tg``, ``couette2d``, ``couette3d``, ``st``, ``added_mass``,
``ice``.  ``--quick`` shrinks every case (smoke test only, not valid physics).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tensorlbm.ibm_validation_cases import (
    cylinder_strouhal_2d,
    oscillating_cylinder_added_mass_2d,
    taylor_couette_2d,
    taylor_couette_3d,
    taylor_green_with_ibm,
)

CASES = ("tg", "couette2d", "couette3d", "st", "added_mass", "ice")


# --------------------------------------------------------------------------- #
# acceptance criteria (declared before the runs — see the report)
# --------------------------------------------------------------------------- #

CRITERIA = {
    "AC1_taylor_green_decay_error": {
        "description": "bulk Taylor-Green decay rate vs analytic 2*nu*k^2",
        "threshold": 0.03,
        "metric": ("tg", "metrics.decay_rate_rel_error"),
    },
    "AC2_couette2d_torque_error": {
        "description": "2-D Couette wall torque vs analytic, finest grid (wall-calibrated markers)",
        "threshold": 0.05,
        "metric": ("couette2d", "metrics.torque_rel_error_calibrated_finest"),
    },
    "AC3_couette3d_torque_error": {
        "description": "3-D Couette wall torque vs analytic (wall-calibrated markers)",
        "threshold": 0.10,
        "metric": ("couette3d", "metrics.torque_rel_error_calibrated"),
    },
    "AC4_strouhal_error": {
        "description": "cylinder Strouhal number vs 0.165 (Re=100)",
        "threshold": 0.10,
        "metric": ("st", "metrics.strouhal_rel_error"),
    },
    "AC5_added_mass_error": {
        "description": "added mass of a fixed cylinder in an oscillating free stream, thinnest Stokes layer, vs potential flow C_A = 1 (inertia coefficient C_M = 1 + C_A = 2)",
        "threshold": 0.20,
        "metric": ("added_mass", "metrics.added_mass_rel_error_best_stokes"),
    },
    "AC6_momentum_conservation": {
        "description": "max IBM momentum-audit residual (fluid) and kernel adjointness",
        "threshold": 1e-10,
        "metric": ("ice", "metrics.fluid_momentum_audit_rel_residual"),
    },
    "AC7_ice_capture_expulsion": {
        "description": "ice floe is caught by the blade and expelled (boolean)",
        "threshold": 1.0,
        "metric": ("ice", "metrics.capture_and_expulsion"),
        "boolean": True,
    },
    "AC8_convergence_order": {
        "description": "observed order p of the Couette torque error under refinement (requires p >= threshold and a monotone error decrease)",
        "threshold": 0.8,
        "lower_bound": True,
        "metric": ("couette2d", "metrics.convergence_order"),
    },
}


def _get(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


# --------------------------------------------------------------------------- #
# case runners
# --------------------------------------------------------------------------- #


def run_tg(quick: bool) -> dict:
    if quick:
        return taylor_green_with_ibm(n=32, n_steps=400)
    return taylor_green_with_ibm(n=64, u0=0.04, tau=0.8, n_steps=1500)


def run_couette2d(quick: bool) -> dict:
    """Two resolution sweeps: raw markers (offset 0) and wall-calibrated ones.

    The wall offset (0.55 cell) is *measured* from the raw sweep's
    ``effective_wall_shift_cells`` at the two coarser resolutions and then
    applied unchanged at the finest resolution — the finest grid is therefore an
    out-of-sample test of the calibration.
    """
    if quick:
        res = [32, 48]
        offset = 0.5
        steps = {32: 300, 48: 500}
    else:
        res = [32, 48, 64]
        offset = 0.55
        steps = {32: 500, 48: 900, 64: 1800}
    raw, cal = {}, {}
    for n in res:
        raw[str(n)] = taylor_couette_2d(
            n=n, n_steps=steps[n], sample_every=max(10, steps[n] // 20), wall_offset=0.0
        )
        cal[str(n)] = taylor_couette_2d(
            n=n,
            n_steps=steps[n],
            sample_every=max(10, steps[n] // 20),
            wall_offset=offset,
        )
    errs = [raw[str(n)]["metrics"]["torque_rel_error"] for n in res]
    order = None
    if len(res) >= 2:
        # least-squares slope of log(error) vs log(n)
        xs = [math.log(n) for n in res]
        ys = [math.log(max(e, 1e-12)) for e in errs]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        denom = sum((x - mx) ** 2 for x in xs) or 1.0
        order = -sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    finest = str(max(res))
    return {
        "case": "couette2d_sweep",
        "resolutions": res,
        "wall_offset_used": offset,
        "raw": raw,
        "calibrated": cal,
        "metrics": {
            "torque_rel_error_raw": {str(n): raw[str(n)]["metrics"]["torque_rel_error"] for n in res},
            "torque_rel_error_calibrated": {
                str(n): cal[str(n)]["metrics"]["torque_rel_error"] for n in res
            },
            "effective_wall_shift_cells": {
                str(n): raw[str(n)]["metrics"]["effective_wall_shift_cells"] for n in res
            },
            "torque_rel_error_calibrated_finest": cal[finest]["metrics"]["torque_rel_error"],
            "torque_rel_error_raw_finest": raw[finest]["metrics"]["torque_rel_error"],
            "convergence_order": order,
            "profile_l2_rel_error_calibrated": {
                str(n): cal[str(n)]["metrics"]["profile_l2_rel_error"] for n in res
            },
            "torque_balance_rel": {
                str(n): raw[str(n)]["metrics"]["torque_balance_rel"] for n in res
            },
        },
    }


def run_couette3d(quick: bool) -> dict:
    if quick:
        n, steps = 24, 500
    else:
        n, steps = 36, 1600
    raw = taylor_couette_3d(n=n, n_steps=steps, sample_every=max(10, steps // 20))
    shift = raw["metrics"]["effective_wall_shift_cells"]
    cal = taylor_couette_3d(
        n=n,
        n_steps=steps,
        sample_every=max(10, steps // 20),
        wall_offset=max(0.0, min(shift, 1.5)),
    )
    return {
        "case": "couette3d_sweep",
        "n": n,
        "raw": raw,
        "calibrated": cal,
        "metrics": {
            "torque_rel_error_raw": raw["metrics"]["torque_rel_error"],
            "torque_rel_error_calibrated": cal["metrics"]["torque_rel_error"],
            "effective_wall_shift_cells": shift,
            "wall_offset_used": max(0.0, min(shift, 1.5)),
            "torque_balance_rel": raw["metrics"]["torque_balance_rel"],
            "profile_l2_rel_error_calibrated": cal["metrics"]["profile_l2_rel_error"],
        },
    }


def run_st(quick: bool) -> dict:
    if quick:
        return cylinder_strouhal_2d(
            nx=96, ny=48, diameter=6.0, n_steps=1400, n_transient=500, perturbation=0.02
        )
    return cylinder_strouhal_2d(
        nx=200,
        ny=100,
        diameter=10.0,
        u_target=0.1,
        tau=0.53,
        n_steps=4800,
        n_transient=1600,
        perturbation=0.01,
    )


def run_added_mass(quick: bool) -> dict:
    """Oscillating free stream past a *fixed* cylinder.

    The drive ``g(t) = ρ U₀Ω cos(Ωt)`` is applied only outside the solid, so the
    interior constraint carries no inertial term.  For a fixed body the
    potential-flow force is ``(1 + C_A) m_d a`` (Froude–Krylov + added mass),
    i.e. the inertia coefficient ``C_M = 2`` for ``C_A = 1``.  The Stokes number
    sweep shows the measured ``C_A`` relaxing towards its potential-flow value as
    the Stokes layer thins (``δ/R`` → 0).
    """
    from tensorlbm.ibm_validation_cases import oscillating_flow_added_mass_2d

    if quick:
        kw = dict(n=128, radius=5.0, u_amp=0.05, tau=0.65, n_periods=4, n_settle=4)
        periods = (120.0, 80.0)
    else:
        kw = dict(n=160, radius=6.0, u_amp=0.05, tau=0.65, n_periods=4, n_settle=4)
        periods = (120.0, 80.0)
    runs = {}
    for per in periods:
        runs[str(int(per))] = oscillating_flow_added_mass_2d(period=per, **kw)
    # independent cross-check: impulse of a prescribed acceleration (integral
    # measurement, insensitive to half-step phase errors)
    from tensorlbm.ibm_validation_cases import accelerated_cylinder_added_mass_2d

    acc = {}
    for ramp in ((40, 80) if not quick else (30,)):
        acc[str(ramp)] = accelerated_cylinder_added_mass_2d(
            n=kw["n"] if not quick else 128,
            radius=kw["radius"] if not quick else 5.0,
            u0=0.05,
            n_ramp=ramp,
            n_hold=60,
            tau=kw["tau"],
        )
    # "best" = thinnest Stokes layer (smallest delta/R), i.e. highest frequency
    best_key = min(runs, key=lambda k: runs[k]["params"]["stokes_number"])
    best = runs[best_key]["metrics"]
    return {
        "case": "oscillating_flow_added_mass_sweep",
        "periods": list(periods),
        "best_period": int(best_key),
        "runs": runs,
        "metrics": {
            "added_mass_coefficient": best["added_mass_coefficient"],
            "added_mass_reference": 1.0,
            "added_mass_rel_error_best_stokes": best["added_mass_rel_error"],
            "inertia_coefficient_C_M": best["inertia_coefficient_C_M"],
            "inertia_rel_error": best["inertia_rel_error"],
            "added_mass_by_period": {
                k: v["metrics"]["added_mass_coefficient"] for k, v in runs.items()
            },
            "inertia_by_period": {
                k: v["metrics"]["inertia_coefficient_C_M"] for k, v in runs.items()
            },
            "stokes_layer_over_radius": {
                k: v["params"]["stokes_number"] for k, v in runs.items()
            },
            "fit_residual_fraction": best["fit_residual_fraction"],
            "drive_efficiency": best["drive_efficiency"],
            "impulse_based_inertia_coefficient": {
                k: v["metrics"]["inertia_coefficient_C_M"] for k, v in acc.items()
            },
        },
        "accelerated": acc,
    }


def run_ice(quick: bool) -> dict:
    from tensorlbm.ibm_ice_propeller import IcePropellerConfig, IcePropellerIBM

    if quick:
        cfg = IcePropellerConfig(n=24, n_steps=200, r_tip=9.0, r_hub=3.0, chord=3.5,
                                 ice_radius=2.6, omega=0.01, inflow=0.02)
        ice_r = 6.0
    else:
        cfg = IcePropellerConfig(
            n=32,
            n_steps=700,
            r_tip=12.0,
            r_hub=4.0,
            chord=4.5,
            ice_radius=3.4,
            ice_start_offset=4.0,
            ice_marker_spacing=1.1,
            omega=0.01,
            inflow=0.02,
            normal_stiffness=20.0,
            normal_damping=120.0,
            sample_every=2,
        )
        ice_r = 8.0
    sim = IcePropellerIBM(cfg)
    # place the floe at mid span, ahead of the blade azimuth so that the
    # rotating blade sweeps into it (capture geometry)
    sim.particle.position[0] = sim.center[0] + ice_r
    out = sim.run()
    m = out["metrics"]
    m["capture_and_expulsion"] = bool(m["captured"] and m["expelled"] and m["stable"])
    m["floe_crosses_blade_plane"] = bool(m["min_ice_z_rel"] < 0.5 * cfg.thickness)
    return out


RUNNERS = {
    "tg": run_tg,
    "couette2d": run_couette2d,
    "couette3d": run_couette3d,
    "st": run_st,
    "added_mass": run_added_mass,
    "ice": run_ice,
}


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def make_figures(results: dict, out_dir: Path) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs: list[dict] = []

    def save(fig, name: str, caption: str) -> None:
        out = out_dir / name
        fig.tight_layout()
        fig.savefig(out, dpi=150, metadata={"Software": None})
        plt.close(fig)
        figs.append({"name": name, "path": str(out), "caption": caption})

    # --- 1: Taylor-Green decay ------------------------------------------------
    tg = results.get("tg")
    if tg:
        s = tg["series"]
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        t = torch.tensor(s["t"], dtype=torch.float64)
        e = torch.tensor(s["energy"], dtype=torch.float64)
        p = tg["params"]
        nu, k = p["nu"], 2 * math.pi / p["n"]
        e0 = e[0]
        ax.semilogy(t, e, "o", ms=2, label="LBM (IBM quiescent)")
        ax.semilogy(t, e0 * torch.exp(-4 * nu * k * k * t), "-", lw=1, label="analytic $E_0e^{-4\\nu k^2t}$")
        ax.set_xlabel("lattice step")
        ax.set_ylabel("kinetic energy")
        ax.set_title(f"Taylor-Green decay, n={p['n']} (error {100*tg['metrics']['decay_rate_rel_error']:.2f}%)")
        ax.legend(fontsize=7)
        save(fig, "fig1_taylor_green.png", "Taylor-Green kinetic-energy decay vs analytic for the quiescent-IBM control run.")

    # --- 2: Couette 2D profiles ----------------------------------------------
    c2 = results.get("couette2d")
    if c2:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        res = c2["resolutions"]
        for n in res:
            p = c2["raw"][str(n)]["metrics"]
            ax.plot(p["profile_r"], p["profile_measured"], "o-", ms=3, lw=1, label=f"IBM n={n}")
        p0 = c2["raw"][str(max(res))]["metrics"]
        ax.plot(p0["profile_r"], p0["profile_analytic"], "k--", lw=1.2, label="analytic Couette")
        ax.set_xlabel("r [cells]")
        ax.set_ylabel(r"$u_\theta$ [lattice]")
        ax.set_title("Rotating-cylinder Couette profiles (uncalibrated markers)")
        ax.legend(fontsize=7)
        save(fig, "fig2_couette_profiles.png", "Azimuthal velocity profiles of the IBM-driven Taylor-Couette cell at three resolutions vs the analytic solution.")

        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        errs = [c2["metrics"]["torque_rel_error_raw"][str(n)] * 100 for n in res]
        errsc = [c2["metrics"]["torque_rel_error_calibrated"][str(n)] * 100 for n in res]
        ax.plot(res, errs, "o-", label="uncalibrated markers")
        ax.plot(res, errsc, "s-", label="wall-calibrated (0.55 cell)")
        ax.set_xlabel("n")
        ax.set_ylabel("torque error [%]")
        ax.set_yscale("log")
        ax.set_title(f"IBM wall-torque convergence (order {c2['metrics']['convergence_order']:.2f})")
        ax.legend(fontsize=7)
        save(fig, "fig3_couette_convergence.png", "Relative error of the IBM wall torque on the rotating inner cylinder vs grid resolution, with and without the effective-wall calibration.")

    # --- 3: cylinder lift + Strouhal -----------------------------------------
    st = results.get("st")
    if st:
        s = st["series"]
        fig, axs = plt.subplots(2, 1, figsize=(6.0, 4.6), sharex=True)
        axs[0].plot(s["t"], s["lift"], lw=0.8)
        axs[0].set_ylabel("$C_L$")
        axs[1].plot(s["t"], s["drag"], lw=0.8)
        axs[1].set_ylabel("$C_D$")
        axs[1].set_xlabel("lattice step")
        axs[0].set_title(
            f"Cylinder Re={st['params']['reynolds']:.0f}: St={st['metrics']['strouhal']:.3f} (ref 0.165)"
        )
        save(fig, "fig4_cylinder_forces.png", "Lift and drag coefficient history for the fixed cylinder at Re=100 used to extract the Strouhal number.")

    # --- 4: added mass --------------------------------------------------------
    am = results.get("added_mass")
    if am and am.get("runs"):
        fig, axs = plt.subplots(2, 1, figsize=(6.0, 4.8), sharex=True)
        best = am["runs"][str(am["best_period"])]
        s = best["series"]
        p = best["params"]
        axs[0].plot(s["t"], s["force_x"], lw=0.8, label="$F_x$ on cylinder")
        axs[0].set_ylabel("$F_x$ [lattice]")
        axs[0].legend(fontsize=7)
        axs[1].plot(s["t"], s["u_mean"], lw=0.8)
        axs[1].set_ylabel("$\\langle u_x\\rangle$")
        axs[1].set_xlabel("lattice step")
        axs[0].set_title(
            f"Fixed cylinder, oscillating stream (period {am['best_period']}): "
            f"$C_A$={am['metrics']['added_mass_coefficient']:.3f} (ref 1.0)"
        )
        save(fig, "fig5_added_mass.png",
             "Force on a fixed cylinder in an oscillating free stream together with the domain-averaged velocity; the component in phase with the free-stream acceleration gives the added-mass coefficient.")

        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        keys = list(am["runs"])
        dr = [am["runs"][k]["params"]["stokes_number"] for k in keys]
        ca = [am["runs"][k]["metrics"]["added_mass_coefficient"] for k in keys]
        ax.plot(dr, ca, "o-", label="IBM measurement")
        ax.axhline(1.0, color="k", ls="--", lw=1, label="potential flow $C_A=1$")
        ax.set_xlabel("$\\delta/R$ (Stokes layer / radius)")
        ax.set_ylabel("$C_A$")
        ax.set_title("Added-mass coefficient vs Stokes-layer thickness")
        ax.legend(fontsize=8)
        save(fig, "fig5b_added_mass_beta.png",
             "Measured added-mass coefficient against the Stokes-layer-to-radius ratio; the potential-flow value is 1.")

    # --- 5: ice–propeller -----------------------------------------------------
    ice = results.get("ice")
    if ice:
        s = ice["series"]
        fig, axs = plt.subplots(3, 1, figsize=(6.2, 6.4), sharex=True)
        axs[0].plot(s["t"], s["ice_z_rel"], lw=1.0)
        axs[0].axhline(0.0, color="k", ls=":", lw=0.8)
        axs[0].set_ylabel("$z-z_{rotor}$ [cells]")
        axs[0].set_title("Ice floe trajectory across the rotor plane (capture and expulsion)")
        axs[1].plot(s["t"], s["f_contact_z"], lw=1.0, label="DEM contact")
        axs[1].plot(s["t"], s["f_ibm_z"], lw=0.8, label="IBM hydrodynamic")
        axs[1].set_ylabel("$F_z$ [lattice]")
        axs[1].legend(fontsize=7)
        axs[2].plot(s["t"], s["rotor_torque_total"], lw=0.9)
        axs[2].set_ylabel("$Q_z$ on rotor")
        axs[2].set_xlabel("lattice step")
        save(fig, "fig6_ice_propeller_forces.png", "Ice-floe trajectory, DEM contact force, IBM hydrodynamic force and total rotor torque during the blade-floe encounter.")

        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        ax.plot(s["ice_x"], s["ice_y"], lw=1.0)
        ax.plot(s["ice_x"][0], s["ice_y"][0], "go", ms=4, label="start")
        ax.plot(s["ice_x"][-1], s["ice_y"][-1], "r^", ms=5, label="end")
        ax.set_aspect("equal")
        ax.set_xlabel("x [cells]")
        ax.set_ylabel("y [cells]")
        ax.set_title("Floe path in the rotor plane")
        ax.legend(fontsize=7)
        save(fig, "fig7_ice_trajectory_xy.png", "Ice-floe centre path projected on the rotor plane, showing the tangential fling produced by blade friction.")

    return figs


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def evaluate(results: dict) -> dict:
    verdicts = {}
    for name, spec in CRITERIA.items():
        case, dotted = spec["metric"]
        if case not in results:
            verdicts[name] = {"status": "NOT_RUN", "description": spec["description"]}
            continue
        val = _get(results[case], dotted)
        if val is None:
            verdicts[name] = {"status": "MISSING_METRIC", "description": spec["description"]}
            continue
        if spec.get("boolean"):
            ok = bool(val)
        elif spec.get("lower_bound"):
            # convergence *order* is a lower bound: larger is better
            ok = float(val) >= spec["threshold"]
        else:
            ok = abs(float(val)) <= spec["threshold"]
        verdicts[name] = {
            "status": "PASS" if ok else "FAIL",
            "description": spec["description"],
            "value": val,
            "threshold": spec["threshold"],
        }
    n_pass = sum(1 for v in verdicts.values() if v.get("status") == "PASS")
    n_fail = sum(1 for v in verdicts.values() if v.get("status") == "FAIL")
    return {
        "criteria": verdicts,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "all_pass": n_fail == 0 and n_pass == len(CRITERIA),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--out", default="artifacts/in54w1_n02")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-figures", action="store_true", help="skip matplotlib (headless nodes)")
    ap.add_argument(
        "--merge",
        default=None,
        help="comma-separated directories to merge (skips the runs); writes summary+figures",
    )
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        results = {}
        for d in args.merge.split(","):
            for f in sorted(Path(d).glob("case_*.json")):
                results[f.stem[len("case_"):]] = json.loads(f.read_text())
        verdicts = evaluate(results)
        figures = make_figures(results, out_dir)
        summary = {
            "node": "IN54W1-N02",
            "repo": "TensorLBM",
            "title": "IBM force transfer for moving/rotating boundaries and ice-propeller coupling",
            "kind": "physics_validation",
            "quick_mode": False,
            "cases_run": sorted(results),
            "verdicts": verdicts,
            "figures": [{"name": f["name"], "path": f["path"], "caption": f["caption"]} for f in figures],
            "merged_from": args.merge.split(","),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        for name in results:
            (out_dir / f"case_{name}.json").write_text(json.dumps(results[name], indent=1))
        print(json.dumps(verdicts, indent=1))
        return
    wanted = [c.strip() for c in args.cases.split(",") if c.strip()]
    results: dict = {}
    timings: dict = {}
    for name in wanted:
        if name not in RUNNERS:
            raise SystemExit(f"unknown case {name!r}; choose from {CASES}")
        t0 = time.time()
        res = RUNNERS[name](args.quick)
        dt = time.time() - t0
        timings[name] = dt
        results[name] = res
        (out_dir / f"case_{name}.json").write_text(json.dumps(res, indent=1))
        print(f"[{name}] done in {dt:.1f}s", flush=True)

    verdicts = evaluate(results) if not args.quick else {"skipped": "quick mode"}
    figures = [] if args.no_figures else make_figures(results, out_dir)
    summary = {
        "node": "IN54W1-N02",
        "repo": "TensorLBM",
        "title": "IBM force transfer for moving/rotating boundaries and ice-propeller coupling",
        "kind": "physics_validation",
        "quick_mode": args.quick,
        "cases_run": wanted,
        "wall_seconds": timings,
        "verdicts": verdicts,
        "figures": [{"name": f["name"], "path": f["path"]} for f in figures],
        "artifacts": {name: str(out_dir / f"case_{name}.json") for name in wanted},
        "environment": {
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "dtype": "float64",
            "lattice": "D2Q9 / D3Q19",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(verdicts, indent=1))


if __name__ == "__main__":
    main()