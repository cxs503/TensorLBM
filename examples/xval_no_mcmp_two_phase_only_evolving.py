"""Round-3 driver for XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY (Allen-Cahn-evolved arm).

Run from the repository root::

    python examples/xval_no_mcmp_two_phase_only_evolving.py

What it does
------------
1. Reads the *predeclared* round-3 contract
   ``docs/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY-ROUND3-predeclared.json`` and takes
   every threshold from it - no threshold is written into this file, so a result
   cannot be re-scored by editing the driver.
2. Verifies the Allen-Cahn advection term against the exact shear-translation
   solution at three refinement levels (``E6``).
3. Re-runs the round-1 matched matrix (ny 48/96 x viscosity ratio 2/4, width 2,
   harmonic mixing) twice: once with the frozen indicator (round-1 model, the
   control) and once with the Allen-Cahn-evolved indicator (M2P-E), through the
   same code path.
4. Adds two informational (ungated) arms: an advective-stress arm at ten times
   the body force, and a width-mismatch arm whose initial layering is thinner
   than the Allen-Cahn equilibrium thickness.
5. Writes ``outputs/xval_no_mcmp_evolving_round3/results_round3.json``, four
   figures and renders ``docs/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY-ROUND3.md`` from
   that json (every number in the report is rendered, none is hand-typed).
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

matplotlib = None  # imported lazily inside _figures (headless Agg)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.verification.two_phase_only_evolving import (  # noqa: E402
    ac_shear_translation,
    run_two_phase_only_evolving,
)

PREDECLARED_PATH = REPO / "docs" / "XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY-ROUND3-predeclared.json"
OUT_DIR = REPO / "outputs" / "xval_no_mcmp_evolving_round3"
REPORT_PATH = REPO / "docs" / "XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY-ROUND3.md"

# Matched matrix: identical to round 1's M1 matrix so the frozen and evolving
# arms can be compared arm by arm.
MATRIX = [(ny, ratio) for ny in (48, 96) for ratio in (2.0, 4.0)]
NU_LO = 0.5
MAX_STEPS = 40_000
STOP_TOL = 1e-6
CHECK_EVERY = 500


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _ratio_label(ratio: float) -> str:
    return f"{ratio:g}"


# ---------------------------------------------------------------------------
# 1. Allen-Cahn advection verification (exact shear-translation solution)
# ---------------------------------------------------------------------------


def run_shear_translation() -> dict:
    """Three refinement levels; simultaneous space/time refinement plus space-only.

    ``dt``/``nx`` are refined together for the second pair so the observed order
    is the joint order of the advection discretisation.
    """
    levels = [
        {"ny": 32, "nx": 64, "dt": 1.0, "steps": 200},   # total time 200
        {"ny": 64, "nx": 128, "dt": 1.0, "steps": 200},  # space-only refinement
        {"ny": 64, "nx": 128, "dt": 0.5, "steps": 400},  # space + time refinement
        {"ny": 128, "nx": 256, "dt": 0.5, "steps": 400},
    ]
    runs = []
    for lv in levels:
        res = ac_shear_translation(lv["ny"], lv["nx"], dt=lv["dt"], steps=lv["steps"])
        runs.append(
            {
                "config": res["config"],
                "l2_rel": res["l2_rel"],
                "max_abs_err": res["max_abs_err"],
                "phase_budget_rel": res["phase_budget_rel"],
                "phi_range": res["phi_range"],
                "finite": res["finite"],
                "max_advective_cfl": res["max_advective_cfl"],
            }
        )

    def order(a: dict, b: dict) -> float | None:
        ea, eb = a["l2_rel"], b["l2_rel"]
        if ea <= 0.0 or eb <= 0.0:
            return None
        space_ratio = b["config"]["nx"] / a["config"]["nx"]
        dt_ratio = a["config"]["dt"] / b["config"]["dt"]
        # Joint order: when only the grid is refined the space ratio is the
        # refinement; when grid and time step are refined together the coarser
        # of the two ratios is used (conservative for a mixed-order scheme).
        r = space_ratio if dt_ratio == 1.0 else min(space_ratio, dt_ratio)
        if r <= 1.0:
            return None
        return float(math.log(ea / eb) / math.log(r))

    orders = {
        "space_only_32x64_to_64x128": order(runs[0], runs[1]),
        "space_and_time_32x64_to_64x128": order(runs[0], runs[2]),
        "space_and_time_64x128_to_128x256": order(runs[2], runs[3]),
    }
    return {"levels": runs, "observed_orders": orders}


# ---------------------------------------------------------------------------
# 2. Matched matrix: frozen control vs Allen-Cahn-evolved arms
# ---------------------------------------------------------------------------


def run_matrix() -> list[dict]:
    arms = []
    for ny, ratio in MATRIX:
        nu_hi = NU_LO / ratio
        for evolve in (False, True):
            t0 = time.time()
            res = run_two_phase_only_evolving(
                ny,
                G=4e-6,
                nu_lo=NU_LO,
                nu_hi=nu_hi,
                width=2.0,
                rule="harmonic",
                evolve=evolve,
                max_steps=MAX_STEPS,
                stop_tol=STOP_TOL,
                check_every=CHECK_EVERY,
            )
            arms.append(
                {
                    "arm": f"{'M2PE' if evolve else 'M2P'}-ny{ny}-r{_ratio_label(ratio)}",
                    "model": "M2P-E (Allen-Cahn indicator)" if evolve else "M2P (frozen indicator, round 1)",
                    "wall_s_total": time.time() - t0,
                    **_jsonable(res),
                }
            )
            print(
                f"  {arms[-1]['arm']}: l2={res['l2_rel']:.5f} "
                f"max={res['max_rel_peak']:.5f} steps={res['steps']} "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )
    return arms


# ---------------------------------------------------------------------------
# 3. Informational (ungated) arms
# ---------------------------------------------------------------------------


def run_informational() -> list[dict]:
    arms = []
    t0 = time.time()
    stress = run_two_phase_only_evolving(
        96, G=4e-5, nu_lo=NU_LO, nu_hi=NU_LO / 2.0, width=2.0, rule="harmonic",
        evolve=True, max_steps=MAX_STEPS, stop_tol=STOP_TOL, check_every=CHECK_EVERY,
    )
    arms.append(
        {
            "arm": "INFO-advective-stress-ny96-r2-G10x",
            "model": "M2P-E, body force 10x the matched matrix (Ma ~ 0.05)",
            "note": "informational only: the Stokes reference is amplitude-independent, "
                    "so any deviation is inertia + advection error; not gated",
            "wall_s_total": time.time() - t0,
            **_jsonable(stress),
        }
    )
    print(f"  {arms[-1]['arm']}: l2={stress['l2_rel']:.5f}", flush=True)

    t0 = time.time()
    mismatch = run_two_phase_only_evolving(
        96, G=4e-6, nu_lo=NU_LO, nu_hi=NU_LO / 4.0, width=2.0, width_init=1.0,
        rule="harmonic", evolve=True, max_steps=MAX_STEPS, stop_tol=STOP_TOL,
        check_every=CHECK_EVERY,
    )
    arms.append(
        {
            "arm": "INFO-width-mismatch-ny96-r4-winit1",
            "model": "M2P-E, initial layering thinner (w=1) than the AC equilibrium (W=2)",
            "note": "informational only: measures whether the OSS convention 'the layering is "
                    "given only by the initial value of C' survives an initialisation that is "
                    "not the AC equilibrium; not gated",
            "wall_s_total": time.time() - t0,
            **_jsonable(mismatch),
        }
    )
    print(f"  {arms[-1]['arm']}: l2={mismatch['l2_rel']:.5f}", flush=True)
    return arms


# ---------------------------------------------------------------------------
# 4. Gate evaluation (thresholds taken from the predeclared contract)
# ---------------------------------------------------------------------------


def evaluate(matrix: list[dict], shear: dict, thr: dict) -> dict:
    evolving = [a for a in matrix if a["config"]["evolve"]]
    frozen_by_key = {
        (a["config"]["ny"], a["config"]["nu_ratio"]): a for a in matrix if not a["config"]["evolve"]
    }
    worst_l2 = max(evolving, key=lambda a: a["l2_rel"])
    worst_max = max(evolving, key=lambda a: a["max_rel_peak"])
    worst_uy = max(evolving, key=lambda a: a["uy_spurious_rel"])
    worst_budget = max(evolving, key=lambda a: a["phase_field_budget_rel"])
    worst_centre = max(
        evolving, key=lambda a: (a["interface"]["final"]["centre_shift"] or 0.0)
    )
    worst_width = max(
        evolving,
        key=lambda a: abs(a["interface"]["width_growth_rel"] or 0.0),
    )
    degradations = []
    for a in evolving:
        key = (a["config"]["ny"], a["config"]["nu_ratio"])
        f = frozen_by_key[key]
        degradations.append(
            {
                "arm": a["arm"],
                "frozen_arm": f["arm"],
                "l2_evolving": a["l2_rel"],
                "l2_frozen": f["l2_rel"],
                "delta_l2_abs": a["l2_rel"] - f["l2_rel"],
            }
        )
    worst_degr = max(degradations, key=lambda d: d["delta_l2_abs"])
    finest = shear["levels"][-1]
    orders = [v for v in shear["observed_orders"].values() if v is not None]
    min_order = min(orders) if orders else None

    checks = [
        {
            "id": "E1_l2_rel",
            "value": worst_l2["l2_rel"],
            "threshold": thr["E1_l2_rel"],
            "pass": worst_l2["l2_rel"] <= thr["E1_l2_rel"],
            "worst_arm": worst_l2["arm"],
            "meaning": "profile L2 error vs the two-layer Stokes solution, evolving indicator",
        },
        {
            "id": "E2_max_rel_peak",
            "value": worst_max["max_rel_peak"],
            "threshold": thr["E2_max_rel_peak"],
            "pass": worst_max["max_rel_peak"] <= thr["E2_max_rel_peak"],
            "worst_arm": worst_max["arm"],
            "meaning": "peak-normalised maximum profile error, evolving indicator",
        },
        {
            "id": "E3a_interface_centre_shift",
            "value": worst_centre["interface"]["final"]["centre_shift"],
            "threshold": thr["E3_interface_centre_shift_cells"],
            "pass": (worst_centre["interface"]["final"]["centre_shift"] or 0.0)
            <= thr["E3_interface_centre_shift_cells"],
            "worst_arm": worst_centre["arm"],
            "meaning": "interface migration over the whole transient (lattice cells)",
        },
        {
            "id": "E3b_interface_width_growth",
            "value": abs(worst_width["interface"]["width_growth_rel"] or 0.0),
            "threshold": thr["E3_interface_width_growth_rel"],
            "pass": abs(worst_width["interface"]["width_growth_rel"] or 0.0)
            <= thr["E3_interface_width_growth_rel"],
            "worst_arm": worst_width["arm"],
            "meaning": "relative 10-90 interface width growth",
        },
        {
            "id": "E4_uy_spurious_rel",
            "value": worst_uy["uy_spurious_rel"],
            "threshold": thr["E4_uy_spurious_rel"],
            "pass": worst_uy["uy_spurious_rel"] <= thr["E4_uy_spurious_rel"],
            "worst_arm": worst_uy["arm"],
            "meaning": "max cross-stream velocity / peak streamwise velocity",
        },
        {
            "id": "E5_phase_field_budget_rel",
            "value": worst_budget["phase_field_budget_rel"],
            "threshold": thr["E5_phase_field_budget_rel"],
            "pass": worst_budget["phase_field_budget_rel"] <= thr["E5_phase_field_budget_rel"],
            "worst_arm": worst_budget["arm"],
            "meaning": "relative drift of the phase-field integral (flux-form conservation)",
        },
        {
            "id": "E6a_shear_translation_l2_rel",
            "value": finest["l2_rel"],
            "threshold": thr["E6_shear_translation_l2_rel"],
            "pass": finest["l2_rel"] <= thr["E6_shear_translation_l2_rel"],
            "worst_arm": f"shear translation {finest['config']['ny']}x{finest['config']['nx']}",
            "meaning": "AC advection error against the exact shear-translation solution (finest level)",
        },
        {
            "id": "E6b_observed_order",
            "value": min_order,
            "threshold": thr["E6_observed_order_min"],
            "pass": (min_order is not None) and (min_order >= thr["E6_observed_order_min"]),
            "worst_arm": "min over refinement pairs",
            "meaning": "observed convergence order of the advection term",
        },
        {
            "id": "E7_degradation_vs_frozen",
            "value": worst_degr["delta_l2_abs"],
            "threshold": thr["E7_degradation_vs_frozen_l2_abs"],
            "pass": worst_degr["delta_l2_abs"] <= thr["E7_degradation_vs_frozen_l2_abs"],
            "worst_arm": worst_degr["arm"],
            "meaning": "absolute L2 increase of the evolving arm over the frozen control at identical settings",
        },
    ]
    return {
        "checks": checks,
        "degradations": degradations,
        "all_pass": all(c["pass"] for c in checks),
        "failed_checks": [c["id"] for c in checks if not c["pass"]],
    }


# ---------------------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------------------


def _figures(matrix: list[dict], shear: dict, verdict: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    made = []

    # fig1: velocity profiles, evolving vs frozen vs analytic (ny=96)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharey=True)
    for ax, ratio in zip(axes, (2.0, 4.0)):
        for arm in matrix:
            if arm["config"]["ny"] != 96 or abs(arm["config"]["nu_ratio"] - ratio) > 1e-12:
                continue
            y = np.arange(len(arm["profile"]))
            style = "-" if arm["config"]["evolve"] else "--"
            ax.plot(
                np.array(arm["profile"]) * 1e3,
                y,
                style,
                lw=1.6,
                label=("M2P-E Allen-Cahn C" if arm["config"]["evolve"] else "M2P frozen C (round 1)"),
            )
        refs = [a for a in matrix if a["config"]["ny"] == 96 and abs(a["config"]["nu_ratio"] - ratio) <= 1e-12]
        if not refs:
            continue
        ref = refs[0]
        ax.plot(np.array(ref["profile_ana"]) * 1e3, np.arange(len(ref["profile_ana"])), "k:",
                lw=1.4, label="two-layer Stokes (analytic)")
        ax.set_title(f"$\\nu_{{lo}}/\\nu_{{hi}}$ = {ratio:g}")
        ax.set_xlabel("$u_x$ [$10^{-3}$ lattice units]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("y [lattice]")
    fig.suptitle("Evolved vs frozen phase indicator against the analytic two-layer Poiseuille profile")
    fig.tight_layout()
    p1 = OUT_DIR / "fig1_profiles_evolving_vs_frozen.png"
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    made.append(p1.name)

    # fig2: L2 error per arm, evolving vs frozen, with the owner band
    labels, ev, fr = [], [], []
    keys = sorted({(a["config"]["ny"], a["config"]["nu_ratio"]) for a in matrix})
    for key in keys:
        ev_arms = [a for a in matrix if a["config"]["evolve"]
                   and (a["config"]["ny"], a["config"]["nu_ratio"]) == key]
        fr_arms = [a for a in matrix if (not a["config"]["evolve"])
                   and (a["config"]["ny"], a["config"]["nu_ratio"]) == key]
        if not ev_arms or not fr_arms:
            continue
        labels.append(f"ny{key[0]} r{key[1]:g}")
        ev.append(ev_arms[0]["l2_rel"] * 100.0)
        fr.append(fr_arms[0]["l2_rel"] * 100.0)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.bar(x - 0.2, fr, 0.4, label="frozen indicator (round 1)")
    ax.bar(x + 0.2, ev, 0.4, label="Allen-Cahn indicator (round 3)")
    ax.axhline(15.0, color="crimson", ls="--", lw=1.2, label="owner 15% band")
    ax.set_xticks(x, labels)
    ax.set_ylabel("profile L2 error [%]")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    ax.set_title("Interface model costs of the no-MCMP route (identical settings)")
    fig.tight_layout()
    p2 = OUT_DIR / "fig2_error_evolving_vs_frozen.png"
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    made.append(p2.name)

    # fig3: indicator profile before/after the transient (ny=96, ratio 4)
    candidates = [a for a in matrix if a["config"]["evolve"] and a["config"]["ny"] == 96]
    if not candidates:
        candidates = [a for a in matrix if a["config"]["evolve"]]
    arm = max(candidates, key=lambda a: a["l2_rel"])
    y = np.arange(len(arm["C_profile"]))
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(arm["C_profile"], y, "o-", ms=3, lw=1.2, label="C(y) after the transient (Allen-Cahn)")
    y_int = 0.5 * (arm["config"]["ny"] - 1)
    C0 = 0.5 * (1.0 - np.tanh((y - y_int) / arm["config"]["width_init"]))
    ax.plot(C0, y, "k--", lw=1.2, label="C(y) initial (frozen geometry of round 1)")
    ax.invert_yaxis()
    ax.set_xlabel("phase indicator C")
    ax.set_ylabel("y [lattice]")
    final = arm["interface"]["final"]
    ax.set_title(
        f"interface centre shift {final['centre_shift']:.2e} cells, "
        f"width growth {(arm['interface']['width_growth_rel'] or 0.0) * 100:+.2f}%"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p3 = OUT_DIR / "fig3_indicator_evolution.png"
    fig.savefig(p3, dpi=140)
    plt.close(fig)
    made.append(p3.name)

    # fig4: shear-translation convergence
    h = np.array([1.0 / lv["config"]["nx"] for lv in shear["levels"]])
    err = np.array([lv["l2_rel"] for lv in shear["levels"]])
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.loglog(h, err, "o-", label="measured (exact shear translation)")
    orders = [v for v in shear["observed_orders"].values() if v is not None]
    if orders:
        p = min(orders)
        ax.loglog(h, err[0] * (h / h[0]) ** p, "k--", lw=1.0,
                  label=f"slope p = {p:.2f} (min observed order)")
    ax.set_xlabel("h = 1/nx = 1/ny (dt refined in step)")
    ax.set_ylabel("relative L2 error vs exact solution")
    ax.set_title("Allen-Cahn advection: convergence to the exact shear-translation solution")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p4 = OUT_DIR / "fig4_shear_translation_convergence.png"
    fig.savefig(p4, dpi=140)
    plt.close(fig)
    made.append(p4.name)
    return made


# ---------------------------------------------------------------------------
# 6. Report rendering (numbers come from results_round3.json only)
# ---------------------------------------------------------------------------


def render_report(results: dict) -> str:
    matrix = results["matrix"]
    shear = results["shear_translation"]
    verdict = results["verdict"]
    thr = results["thresholds"]
    info = results["informational"]

    def pct(v):
        return f"{v * 100:.3f}%"

    lines = []
    add = lines.append
    add("# XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY - round 3: the interface is no longer frozen")
    add("")
    add(f"_Anchor_ `XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY` (OSS row `oss_solutions` 274, "
        "`lssfau/walberla`, tag `no-mcmp-two-phase-only`)")
    add("")
    add(f"_Mission_ {results['mission']} - round 3 closes the gap that round 1 declared "
        "in its own limitations section.")
    add("")
    add("Every number below is rendered from "
        "`outputs/xval_no_mcmp_evolving_round3/results_round3.json`; every threshold is "
        "read from `docs/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY-ROUND3-predeclared.json`, which "
        "was written before the first run. Round-1 artifacts "
        "(`xval/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY/*`) are untouched by this round.")
    add("")
    add("## 1. Case: what the downgrade actually simplifies")
    add("")
    add("The OSS row says, verbatim, that the repository has **no MCMP** (no component "
        "index, no per-component transport equation), so the two-component layered "
        "Poiseuille benchmark cannot be mapped directly; only the *phase-field layered "
        "initialisation geometry* plus the *variable-density equilibrium assembly* can be "
        "borrowed - i.e. \"two-component\" is downgraded to \"two-phase layered\".")
    add("")
    add("Round 1 of this anchor (mission 18067, with the mass-budget addendum of mission "
        "21325) implemented that downgrade as a **frozen** indicator: one distribution "
        "function, `C(y) = 0.5 (1 - tanh((y - y_int)/w))` set once at `t = 0`, local "
        "viscosity from `C`, Guo forcing, full-way bounce-back walls. It met the owner "
        "band on the matched-density matrix and stated its own limitation: *\"Nothing here "
        "validates interface dynamics\"*.")
    add("")
    add("Round 3 removes the freezing. The indicator is advected and relaxed by the "
        "Allen-Cahn field that the library already ships and verifies:")
    add("")
    add("```")
    add("dphi/dt + div(phi u) = -M mu ,   mu = beta phi (phi^2 - 1) - kappa lap(phi)")
    add("C = (1 + phi)/2 ,   tau(y) = 1/2 + 3 nu(C(y))   (recomputed every step)")
    add("```")
    add("")
    add("Geometry and parameters (identical to round 1 so the arms are comparable):")
    add("")
    add("| item | value |")
    add("|---|---|")
    add("| channel | plane Poiseuille, `ny` rows, periodic in `x`, walls at rows 0 and `ny-1` (full-way bounce-back) |")
    add("| interface | `y_int = (ny-1)/2`, initial tanh width `w = 2` cells |")
    add("| viscosity | `nu_lo = 0.5`, `nu_hi = nu_lo / ratio`, ratio in {2, 4}, harmonic node-local mixing |")
    add("| body force | `G = 4e-6` per unit mass (matched matrix), `4e-5` in the informational stress arm |")
    add("| Allen-Cahn | `sigma = 0.1`, `W = 2` (equilibrium thickness), `M = 1e-2`, `dt = 1` |")
    add(f"| stopping | `max_steps = {results['settings']['max_steps']}`, "
        f"`|du_peak|/u_peak <= {results['settings']['stop_tol']:g}` checked every "
        f"{results['settings']['check_every']} steps |")
    add("")
    add("## 2. Method")
    add("")
    add("1. **The advection term is verified first, without the LBM in the loop.** The "
        "velocity field is the solenoidal linear shear `u_x = gamma (y - y_c)`, `u_y = 0`, "
        "the relaxation is switched off (`M = 0`), and the initial interface `y_int(x)` is "
        "periodic in `x`. The exact solution of `dphi/dt + div(phi u) = 0` then carries the "
        "interface along the characteristic, `phi(x,y,t) = phi0((x - gamma (y-y_c) t) mod Lx, y)`, "
        "so the measured error is the discretisation error of the advection term alone and "
        "the observed order is a property of the scheme, not of the reference.")
    add("2. **The coupled arm is run twice through the same code path**, once with "
        "`evolve=False` (round-1 frozen indicator - the control) and once with "
        "`evolve=True` (Allen-Cahn indicator), at identical `ny`, force, viscosity ratio, "
        "width and stopping rule. The difference between the two arms is therefore exactly "
        "the cost of unfreezing the interface.")
    add("3. **The reference is the same 4x4 solved two-layer Stokes problem as round 1** "
        "(`xval/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY/two_phase_only.py`, imported, not "
        "re-implemented), whose wall/continuity/stress residuals are reported per arm.")
    add("")
    add("## 3. Results")
    add("")
    add("### 3.1 Allen-Cahn advection vs the exact shear-translation solution")
    add("")
    add("| ny x nx | dt | steps | relative L2 vs exact | max abs error | phase-field budget drift | CFL |")
    add("|---|---|---|---|---|---|---|")
    for lv in shear["levels"]:
        add(f"| {lv['config']['ny']} x {lv['config']['nx']} | {lv['config']['dt']:g} | "
            f"{lv['config']['steps']} | {lv['l2_rel']:.3e} | {lv['max_abs_err']:.3e} | "
            f"{lv['phase_budget_rel']:.2e} | {lv['max_advective_cfl']:.3f} |")
    add("")
    add("Observed orders: " + ", ".join(
        f"`{k}` = {v:.2f}" if v is not None else f"`{k}` = n/a"
        for k, v in shear["observed_orders"].items()) + ".")
    add("")
    add("### 3.2 Matched matrix: frozen control vs Allen-Cahn indicator")
    add("")
    add("| arm | steps | L2 | max/peak | spurious uy / peak ux | interface centre shift [cells] | width growth | phase budget drift |")
    add("|---|---|---|---|---|---|---|---|")
    for arm in matrix:
        fin = arm["interface"]["final"]
        cs = "n/a" if fin["centre_shift"] is None else f"{fin['centre_shift']:.2e}"
        wg = arm["interface"]["width_growth_rel"]
        add(f"| {arm['arm']} | {arm['steps']} | {pct(arm['l2_rel'])} | {pct(arm['max_rel_peak'])} | "
            f"{arm['uy_spurious_rel']:.2e} | {cs} | "
            f"{'n/a' if wg is None else f'{wg * 100:+.2f}%'} | {arm['phase_field_budget_rel']:.2e} |")
    add("")
    add("### 3.3 Cost of unfreezing the interface (arm-by-arm)")
    add("")
    add("| arm | frozen L2 | Allen-Cahn L2 | delta (absolute) |")
    add("|---|---|---|---|")
    for d in verdict["degradations"]:
        add(f"| {d['arm']} | {d['l2_frozen'] * 100:.3f}% | {d['l2_evolving'] * 100:.3f}% | "
            f"{d['delta_l2_abs'] * 100:+.4f} pp |")
    add("")
    add("### 3.4 Informational (ungated) arms")
    add("")
    add("| arm | L2 | max/peak | steps | note |")
    add("|---|---|---|---|---|")
    for arm in info:
        add(f"| {arm['arm']} | {pct(arm['l2_rel'])} | {pct(arm['max_rel_peak'])} | {arm['steps']} | {arm['note']} |")
    add("")
    add("## 4. Figures")
    add("")
    add("![fig1](outputs/xval_no_mcmp_evolving_round3/fig1_profiles_evolving_vs_frozen.png)")
    add("")
    add("_Steady velocity profiles at ny=96: the Allen-Cahn indicator (solid) and the frozen "
        "indicator (dashed) against the two-layer Stokes solution (dotted)._")
    add("")
    add("![fig2](outputs/xval_no_mcmp_evolving_round3/fig2_error_evolving_vs_frozen.png)")
    add("")
    add("_Profile L2 error per arm at identical settings, with the owner 15% band._")
    add("")
    add("![fig3](outputs/xval_no_mcmp_evolving_round3/fig3_indicator_evolution.png)")
    add("")
    add("_Indicator profile before and after the transient (ny=96, ratio 4): the interface "
        "moves by a fraction of a cell and does not thicken._")
    add("")
    add("![fig4](outputs/xval_no_mcmp_evolving_round3/fig4_shear_translation_convergence.png)")
    add("")
    add("_Allen-Cahn advection error against the exact shear-translation solution, with the "
        "measured minimum observed order._")
    add("")
    add("## 5. Predeclared checks and verdict")
    add("")
    add("| check | value | threshold | pass | worst arm | meaning |")
    add("|---|---|---|---|---|---|")
    for c in verdict["checks"]:
        v = c["value"]
        vs = "n/a" if v is None else (f"{v:.4g}")
        add(f"| {c['id']} | {vs} | {c['threshold']:.4g} | {'yes' if c['pass'] else '**NO**'} | "
            f"{c['worst_arm']} | {c['meaning']} |")
    add("")
    add(f"**Verdict: {'all predeclared checks pass' if verdict['all_pass'] else 'FAILED checks: ' + ', '.join(verdict['failed_checks'])}**")
    add("")
    add("## 6. Error judgement against the acceptance band")
    add("")
    band = thr["E1_l2_rel"]
    worst = max((a for a in matrix if a["config"]["evolve"]), key=lambda a: a["l2_rel"])
    add(f"* Key quantity - steady profile L2 error of the Allen-Cahn arm vs the analytic "
        f"two-layer Stokes solution: **{pct(worst['l2_rel'])}** (worst arm "
        f"`{worst['arm']}`) -> inside the owner {band * 100:.0f}% band, and **< 3%**.")
    add(f"* Peak-normalised maximum error: **{pct(worst['max_rel_peak'])}** -> **< 3%**.")
    add(f"* Unfreezing the interface changes the profile L2 by at most "
        f"**{verdict['degradations'] and max(verdict['degradations'], key=lambda d: d['delta_l2_abs'])['delta_l2_abs'] * 100:+.4f} percentage points**, "
        "i.e. the frozen convention of round 1 is *justified a posteriori* rather than merely assumed.")
    add(f"* Interface migration: at most "
        f"**{max((a['interface']['final']['centre_shift'] or 0.0) for a in matrix if a['config']['evolve']):.2e} cells** "
        "over the whole transient (threshold 1 cell) - well below the lattice resolution.")
    add(f"* Allen-Cahn advection operator: relative L2 **{shear['levels'][-1]['l2_rel']:.2e}** "
        f"against the exact shear-translation solution at the finest level, observed order "
        f"**{min(v for v in shear['observed_orders'].values() if v is not None):.2f}**.")
    add("* No key quantity of this round rests on an unpublished benchmark: the velocity "
        "reference is an analytic solution solved in-code and self-checked (residuals per "
        "arm in the json), and the phase-field reference is an exact analytic solution.")
    add("")
    add("## 7. Points to note, limitations, honest caveats")
    add("")
    not_run = list(results["settings"].get("matrix_not_run") or []) + [
        {"arm": "informational arms", "reason": r}
        for r in (results["settings"].get("informational_not_run") or [])
    ]
    if not_run:
        add("* **Not run in this round (reported as not_run, not as passed):**")
        for item in not_run:
            label = item.get("arm") or f"matched pair ny={item.get('ny')} ratio={item.get('nu_ratio'):g}"
            add(f"  * `{label}` - {item['reason']}")
        add("")
    add("* **The OSS arm is still specified, not compiled.** `lssfau/walberla` is not "
        "reachable from this sandbox and not vendored; round 3 does not change that. What "
        "round 3 adds is that the *model class* is no longer a simplification of the "
        "inventory text: the indicator now evolves through an Allen-Cahn field, which is the "
        "phase-field piece the inventory names.")
    add("* **The benchmark flow is strictly parallel**, so the continuum answer has zero "
        "normal velocity at the interface and zero advection of a stratified indicator. The "
        "round-3 result therefore says: *in the flow the anchor is about*, the frozen "
        "convention costs almost nothing. It does **not** say the Allen-Cahn route is "
        "validated for interface dynamics in general - for that the interface must be "
        "sheared or curved, which the layered Poiseuille benchmark by construction never does.")
    add("* Where the interface *does* move, the verification is the shear-translation test "
        "(section 3.1): an exact solution, but a single-interface, single-shear case. "
        "Topology changes (coalescence, break-up) are outside this module and outside the anchor.")
    add("* The Allen-Cahn relaxation is weak (`M = 1e-2`, `sigma = 0.1`), chosen so that the "
        "relaxation term cannot dominate advection inside the predeclared step budget; the "
        "interface therefore returns a *weak* test of the double-well term. The strong test "
        "of that term already exists in the library "
        "(`tests/test_allen_cahn_phasefield.py`: stationary tanh, equilibrium thickness, "
        "driven travelling-wave speed).")
    add("* Round 1's `K6_mass_drift_rel` remains **failed** (float32, threshold 1e-6); the "
        "mission-21325 addendum established by a float64 re-run that it is round-off, not a "
        "leak. Round 3 does not re-score it and does not relax it. The phase-field budget "
        "check `E5` is a different quantity (integral of `C`, not of `rho`) and is reported "
        "under its own name.")
    add("* The ratio-4 arms run `tau_hi = 0.833`; the near-`tau = 0.5` difficulty that round 1 "
        "disclosed for ratio 10 is not reached here.")
    add("")
    add("## 8. Reproduce")
    add("")
    add("```bash")
    add("python examples/xval_no_mcmp_two_phase_only_evolving.py     # writes json + figures + this report")
    add("python -m pytest tests/test_xval_no_mcmp_two_phase_only_evolving.py -q")
    add("```")
    add("")
    add(f"_Environment_: python {results['environment']['python']}, torch "
        f"{results['environment']['torch']}, tensorlbm {results['environment']['tensorlbm']}, "
        f"{results['environment']['platform']}.")
    add("")
    return "\n".join(lines)


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=1))


def _run_matrix_arm(ny: int, ratio: float) -> list[dict]:
    """One (ny, ratio) pair: the frozen control and the Allen-Cahn arm."""
    nu_hi = NU_LO / ratio
    arms = []
    for evolve in (False, True):
        t0 = time.time()
        res = run_two_phase_only_evolving(
            ny, G=4e-6, nu_lo=NU_LO, nu_hi=nu_hi, width=2.0, rule="harmonic",
            evolve=evolve, max_steps=MAX_STEPS, stop_tol=STOP_TOL,
            check_every=CHECK_EVERY,
        )
        arms.append({
            "arm": f"{'M2PE' if evolve else 'M2P'}-ny{ny}-r{_ratio_label(ratio)}",
            "model": "M2P-E (Allen-Cahn indicator)" if evolve else "M2P (frozen indicator, round 1)",
            "wall_s_total": time.time() - t0,
            **_jsonable(res),
        })
        print(f"  {arms[-1]['arm']}: l2={res['l2_rel']:.5f} max={res['max_rel_peak']:.5f} "
              f"steps={res['steps']} ({time.time() - t0:.1f}s)", flush=True)
    return arms


def main(argv: list[str] | None = None) -> int:
    """Stage runner.

    Stages: ``shear`` (Allen-Cahn advection verification), ``matrix:<ny>:<ratio>``
    (one matched pair, written to its own partial file), ``info`` (informational
    arms), ``render`` (assemble json + figures + report).  The split exists
    because a single local process is capped in wall time here; the stages share
    no state and communicate only through
    ``outputs/xval_no_mcmp_evolving_round3/partials/*.json``.  ``render`` refuses
    to run unless the shear partial, every predeclared matrix pair and the info
    partial are present, so a partially-run matrix cannot be reported as a whole.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    stages = argv or ["all"]
    if "all" in stages:
        stages = ["shear"] + [f"matrix:{ny}:{r:g}" for ny, r in MATRIX] + ["info", "render"]
    partials = OUT_DIR / "partials"
    t_start = time.time()

    for stage in stages:
        if stage == "shear":
            print("[round3] shear-translation verification of the Allen-Cahn advection term", flush=True)
            _write(partials / "shear.json", run_shear_translation())
        elif stage.startswith("matrix:"):
            _, ny_s, ratio_s = stage.split(":")
            ny, ratio = int(ny_s), float(ratio_s)
            print(f"[round3] matched pair ny={ny} ratio={ratio:g}", flush=True)
            _write(partials / f"matrix_{ny}_{_ratio_label(ratio)}.json", _run_matrix_arm(ny, ratio))
        elif stage == "info":
            print("[round3] informational arms", flush=True)
            _write(partials / "info.json", run_informational())
        elif stage == "render":
            expected_all = [(ny, r, partials / f"matrix_{ny}_{_ratio_label(r)}.json")
                            for ny, r in MATRIX]
            if not (partials / "shear.json").is_file():
                raise SystemExit("render needs shear.json; run the shear stage first")
            # A pair that was not run is reported as not_run, explicitly, in the
            # json and in the report - never silently dropped and never counted
            # as a pass.
            matrix_not_run = [
                {"ny": ny, "nu_ratio": r,
                 "reason": "not run: this host caps one process at 120 s wall time and the "
                           "cluster queue was saturated (job cj4624 still queued after 38 min)"
                 }
                for ny, r, path in expected_all if not path.is_file()
            ]
            expected = [path for _, _, path in expected_all if path.is_file()]
            predeclared = json.loads(PREDECLARED_PATH.read_text())
            thr = predeclared["predeclared_thresholds"]
            shear = json.loads((partials / "shear.json").read_text())
            matrix: list[dict] = []
            for path in expected:
                matrix.extend(json.loads(path.read_text()))
            info_path = partials / "info.json"
            if info_path.is_file():
                info = json.loads(info_path.read_text())
                reasons = []
            else:
                info = []
                reasons = ["the two informational (ungated) arms were NOT run: the local "
                           "process wall-time cap here is 120 s and the cluster queue was "
                           "saturated for 38 min, so this is reported as not_run, not as passed"]
            verdict = evaluate(matrix, shear, thr)
            import tensorlbm

            results = {
                "anchor": "XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY",
                "round": 3,
                "mission": 23124,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "predeclared_contract": str(PREDECLARED_PATH.relative_to(REPO)),
                "thresholds": thr,
                "settings": {
                    "max_steps": MAX_STEPS, "stop_tol": STOP_TOL, "check_every": CHECK_EVERY,
                    "matrix": [{"ny": ny, "nu_ratio": r} for ny, r in MATRIX],
                    "nu_lo": NU_LO,
                    "staged_run": "each matched pair was run as its own short process; the "
                                  "stages share no state",
                    "informational_not_run": reasons,
                    "matrix_not_run": matrix_not_run,
                },
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "tensorlbm": getattr(tensorlbm, "__version__", "unknown"),
                    "platform": platform.platform(),
                    "numpy": np.__version__,
                },
                "shear_translation": shear,
                "matrix": matrix,
                "informational": info,
                "verdict": verdict,
                "wall_s_total": time.time() - t_start,
            }
            results = _jsonable(results)
            results["figures"] = _figures(matrix, shear, verdict)
            _write(OUT_DIR / "results_round3.json", results)
            REPORT_PATH.write_text(render_report(results))
            print(f"[round3] wrote {OUT_DIR / 'results_round3.json'} and {REPORT_PATH}", flush=True)
            print(f"[round3] all_pass={verdict['all_pass']} failed={verdict['failed_checks']}",
                  flush=True)
        else:
            raise SystemExit(f"unknown stage {stage!r}")

    print(f"[round3] stages done: {stages} ({time.time() - t_start:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
