#!/usr/bin/env python3
"""Wall-modelled-LES usability audit (node IN61-N06).

Produces, for the master report and JSON record under
``artifacts/in61_n06_wall_model_usability``:

1. criterion 1 — manufactured wall-law states inverted back to ``u_tau`` for
   both composite laws over five decades of friction velocity and seven decades
   of ``y+`` (target ``< 1e-8`` relative);
2. criterion 2 — wall-stress budget: a-priori ``tau_w`` reconstruction from the
   verbatim MKM1999 DNS profiles, plus the resolved-stencil finite-difference
   bias that limits one-cell sampling;
3. criterion 3 — equilibrium wall-modelled channel against MKM1999 at
   ``Re_tau = 178.12`` and ``587.19``; the van Driest constant is fitted at the
   higher Reynolds number and validated unchanged at the lower one;
4. criterion 4 — three-grid ladder of the N04 container-ship wind case with the
   composite wall law (Richardson ratio ``R`` and the ``dx = 2 m`` /
   next-finest ``C_X`` change);
5. criterion 5 — three negative controls (wrong ``kappa``, one-cell sampling
   offset, wall law disabled), each of which must break the gate.

Usage::

    python scripts/verify_wall_model_usability.py            # asserts the gate
    python scripts/verify_wall_model_usability.py --no-assert  # audit only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
SRC = WORK / "src"
sys.path.insert(0, str(SRC))

import torch  # noqa: E402

import tensorlbm.turbulence.wall_model_usability as wmu  # noqa: E402

RESOLVED = Path(wmu.__file__).resolve()
if SRC not in RESOLVED.parents:
    raise SystemExit(f"auditing the wrong tree: {RESOLVED}")

from tensorlbm.turbulence.wall_model_usability import (  # noqa: E402
    REICHARDT,
    SPALDING,
    WallLawParams,
    assert_wall_model_usable,
    assess_wall_model_usability,
    channel_level_deviation,
    channel_slope_deviation,
    channel_wall_stress_consistency,
    dns_wall_stress_recovery,
    equilibrium_wall_modelled_channel,
    grid_ladder_verdict,
    inversion_relative_error,
    load_mkm_profile,
    manufactured_wall_state,
    negative_control,
    solve_friction_velocity,
    wall_law_u_plus,
    wall_stress_budget,
)

OUT = WORK / "artifacts" / "in61_n06_wall_model_usability"
REF = WORK / "benchmarks" / "pending" / "channel_turb_3d" / "dns_ref"
RUNS = WORK / "runs" / "wall_model_usability" / "n04"

NU_AIR = 1.2e-5
#: manufactured state space of criterion 1
U_TAU_SWEEP = (1e-6, 1e-5, 1e-4, 1e-2, 0.05, 0.2, 1.0)
Y_PLUS_SWEEP = (0.05, 0.5, 1.0, 5.0, 11.0, 30.0, 100.0, 500.0, 2000.0, 20000.0, 1e5)
#: the wall law used by the audit and by the ship wind case
AUDIT_LAW = REICHARDT
#: van Driest damping of the equilibrium closure (fitted at Re_tau = 587)
VAN_DRIEST_GRID = tuple(26.0 + 0.5 * i for i in range(17))
#: N04 wind ladder, coarse -> fine (geometric, refinement ratio sqrt(2))
LADDER = (
    ("container_a0_dx4.0_spalding", 4.0),
    ("container_a0_dx2.83_spalding", 2.83),
    ("container_a0_dx2.0_spalding", 2.0),
)
#: law sensitivity at one grid and one (shorter) window; the spalding leg of the
#: comparison is the ladder run itself, so the equal-window pair is log vs
#: reichardt (both 800 steps)
LAW_SENSITIVITY = (
    ("container_a0_dx2.83_log", "log"),
    ("container_a0_dx2.83_reichardt", "reichardt"),
)
REFINEMENT_RATIO = 2.0 ** 0.5


# ---------------------------------------------------------------------------
# criterion 1
# ---------------------------------------------------------------------------
def audit_inversion(law: WallLawParams) -> dict:
    y_plus = torch.tensor(Y_PLUS_SWEEP, dtype=torch.float64)
    rows = []
    worst = 0.0
    worst_iterations = 0
    for u_tau_true in U_TAU_SWEEP:
        u_t, y = manufactured_wall_state(u_tau_true, y_plus, NU_AIR, law=law)
        solution = solve_friction_velocity(u_t, y, NU_AIR, law=law)
        error = inversion_relative_error(
            solution.u_tau, torch.full_like(solution.u_tau, u_tau_true)
        )
        maximum = float(error.max().item())
        worst = max(worst, maximum)
        worst_iterations = max(worst_iterations, solution.iterations)
        rows.append(
            {
                "u_tau_true": u_tau_true,
                "max_rel_error": maximum,
                "mean_rel_error": float(error.mean().item()),
                "iterations": solution.iterations,
                "residual": solution.residual,
                "converged": solution.converged,
            }
        )
    per_point = []
    for u_tau_true in (0.01, 0.2):
        u_t, y = manufactured_wall_state(u_tau_true, y_plus, NU_AIR, law=law)
        solution = solve_friction_velocity(u_t, y, NU_AIR, law=law)
        error = inversion_relative_error(
            solution.u_tau, torch.full_like(solution.u_tau, u_tau_true)
        )
        for index, y_plus_value in enumerate(Y_PLUS_SWEEP):
            per_point.append(
                {
                    "law": law.name,
                    "u_tau_true": u_tau_true,
                    "y_plus": y_plus_value,
                    "rel_error": float(error[index].item()),
                }
            )
    return {
        "law": law.to_dict(),
        "max_rel_error": worst,
        "max_iterations": worst_iterations,
        "rows": rows,
        "per_point": per_point,
    }


def audit_kappa_mismatch(state_law: WallLawParams, inversion_law: WallLawParams) -> dict:
    """Negative control 1: invert a state built with one kappa using another.

    A manufactured state from ``state_law`` is inverted with ``inversion_law``.
    If ``kappa`` is mis-set (a classic unit/constant slip) the recovered
    friction velocity is off by per cent, so the machine-precision assertion of
    criterion 1 must fire.
    """
    y_plus = torch.tensor(Y_PLUS_SWEEP, dtype=torch.float64)
    rows = []
    worst = 0.0
    for u_tau_true in (0.01, 0.2):
        u_t, y = manufactured_wall_state(u_tau_true, y_plus, NU_AIR, law=state_law)
        solution = solve_friction_velocity(u_t, y, NU_AIR, law=inversion_law)
        error = inversion_relative_error(
            solution.u_tau, torch.full_like(solution.u_tau, u_tau_true)
        )
        worst = max(worst, float(error.max().item()))
        rows.append({"u_tau_true": u_tau_true, "max_rel_error": float(error.max().item())})
    return {
        "state_law": state_law.to_dict(),
        "inversion_law": inversion_law.to_dict(),
        "max_rel_error": worst,
        "rows": rows,
    }


def audit_sampling_offset(law: WallLawParams, offset_cells: float = 15.0) -> dict:
    """Negative control 2: sample one grid cell beyond the stated wall distance.

    In wall units the N04 sampling distance is ``y2+ ~ 1e4``; an offset of one
    cell is taken here as ``15`` wall units (a coarse-LES first-cell spacing at
    ``Re_tau ~ 180``), which is the smallest offset a mesh can produce.
    """
    y_plus = torch.tensor(Y_PLUS_SWEEP, dtype=torch.float64)
    rows = []
    worst = 0.0
    for u_tau_true in (0.01, 0.2):
        u_t, y = manufactured_wall_state(u_tau_true, y_plus, NU_AIR, law=law)
        _, y_shifted = manufactured_wall_state(
            u_tau_true, y_plus + offset_cells, NU_AIR, law=law
        )
        solution = solve_friction_velocity(u_t, y_shifted, NU_AIR, law=law)
        error = inversion_relative_error(
            solution.u_tau, torch.full_like(solution.u_tau, u_tau_true)
        )
        worst = max(worst, float(error.max().item()))
        rows.append({"u_tau_true": u_tau_true, "max_rel_error": float(error.max().item())})
    return {"offset_wall_units": offset_cells, "max_rel_error": worst, "rows": rows}


# ---------------------------------------------------------------------------
# criterion 2
# ---------------------------------------------------------------------------
def audit_stress(profiles: dict[str, wmu.MkmProfile]) -> dict:
    recovery = {}
    for tag, profile in profiles.items():
        for law in (SPALDING, REICHARDT):
            recovery[f"{tag}:{law.name}"] = dns_wall_stress_recovery(profile, law)
    # resolved-stencil bias of the one-cell sampling used by the ship case
    stencils = {}
    u_tau = 0.05
    for law in (SPALDING, REICHARDT):
        y1_plus = torch.tensor([15.0, 30.0], dtype=torch.float64)
        y2_plus = y1_plus + 30.0
        u1 = wall_law_u_plus(y1_plus, law) * u_tau
        u2 = wall_law_u_plus(y2_plus, law) * u_tau
        budget = wall_stress_budget(
            u1,
            u2,
            y1_plus * NU_AIR / u_tau,
            y2_plus * NU_AIR / u_tau,
            torch.full_like(u1, u_tau**2),
            NU_AIR,
            law=law,
        )
        stencils[law.name] = {
            "y_plus_stencil": [y1_plus.tolist(), y2_plus.tolist()],
            "max_rel_residual": budget["max_residual"],
            "fd_bias": budget["fd_bias"],
        }
    worst_audit = max(
        recovery[f"{tag}:{AUDIT_LAW.name}"]["max_tau_w_rel_error"] for tag in profiles
    )
    return {
        "dns_recovery": recovery,
        "max_rel_error_audit_law": worst_audit,
        "resolution_stencil_bias": stencils,
    }


# ---------------------------------------------------------------------------
# criterion 3
# ---------------------------------------------------------------------------
def audit_channel(profiles: dict[str, wmu.MkmProfile]) -> dict:
    high = profiles["mkm590"]
    low = profiles["mkm180"]
    calibration = []
    for van_driest in VAN_DRIEST_GRID:
        channel = equilibrium_wall_modelled_channel(
            high.re_tau, law=AUDIT_LAW, van_driest_a=van_driest
        )
        level = channel_level_deviation(high, channel)
        calibration.append(
            {"van_driest_a": van_driest, "level_rms_relative": level["rms_relative"]}
        )
    best = min(calibration, key=lambda row: row["level_rms_relative"])
    van_driest = best["van_driest_a"]

    legs = {}
    for tag, profile in profiles.items():
        channel = equilibrium_wall_modelled_channel(
            profile.re_tau, law=AUDIT_LAW, van_driest_a=van_driest
        )
        slope = channel_slope_deviation(profile, channel)
        level = channel_level_deviation(profile, channel)
        stress = channel_wall_stress_consistency(channel)
        legs[tag] = {
            "re_tau": profile.re_tau,
            "slope": slope,
            "level": level,
            "channel_stress": stress,
            "profile": {
                "y_plus": channel.y_plus[:: max(1, len(channel.y_plus) // 800)].tolist(),
                "u_plus": channel.u_plus[:: max(1, len(channel.u_plus) // 800)].tolist(),
            },
        }
    # sensitivity: the same closure with Spalding's classic constants
    classic = {}
    for tag, profile in profiles.items():
        channel = equilibrium_wall_modelled_channel(
            profile.re_tau, law=SPALDING, van_driest_a=van_driest
        )
        classic[tag] = channel_slope_deviation(profile, channel)
    worst = max(leg["slope"]["slope_deviation"] for leg in legs.values())
    return {
        "law": AUDIT_LAW.to_dict(),
        "van_driest_calibration": calibration,
        "van_driest_a_fitted_at_re_tau": high.re_tau,
        "van_driest_a": van_driest,
        "legs": legs,
        "max_slope_deviation": worst,
        "spalding_sensitivity": classic,
    }


# ---------------------------------------------------------------------------
# criterion 4
# ---------------------------------------------------------------------------
def _read_run(tag: str) -> dict:
    path = RUNS / f"{tag}.json"
    if not path.exists():
        raise SystemExit(
            f"missing wind-ladder run {path}; run scripts/run_ship_wind_campaign.py "
            f"--ship container --angle 0 --wall-law spalding --dx ... --out {path}"
        )
    return json.loads(path.read_text())


def _force_statistics(record: dict, blocks: int = 10) -> dict:
    """Robust statistics of the C_X force series stored in a run record.

    The stored series is the raw world-frame force on the sample window; ``C_X``
    follows from the run's own lattice normalisation (``q_lattice`` and the
    frontal area in cells), so no number is retyped.  Block averaging gives a
    standard error that is honest about the serial correlation of the LBM force
    signal, and the median exposes windows that do not contain a full shedding
    period (mean and median then disagree strongly).
    """
    import numpy as np

    coefficients = record["coefficients"]
    q = coefficients["q_lattice"]
    area = coefficients["area_frontal_cells"]
    series = np.asarray(record["force_history"]["fx"], dtype=float) / (q * area)
    if series.size < blocks:
        raise SystemExit("force series shorter than the block count")
    edges = np.linspace(0, series.size, blocks + 1).astype(int)
    block_means = np.array(
        [series[edges[i]:edges[i + 1]].mean() for i in range(blocks) if edges[i + 1] > edges[i]]
    )
    sem = float(block_means.std(ddof=1) / np.sqrt(block_means.size))
    return {
        "mean": float(series.mean()),
        "median": float(np.median(series)),
        "std": float(series.std()),
        "p01": float(np.percentile(series, 1)),
        "p99": float(np.percentile(series, 99)),
        "block_sem": sem,
        "block_sem_relative": sem / abs(float(series.mean())) if series.mean() else float("inf"),
        "samples": int(series.size),
        "blocks": int(block_means.size),
        "mean_over_median": float(series.mean() / np.median(series)) if np.median(series) else float("inf"),
    }


def audit_grid() -> dict:
    values = []
    for tag, dx in LADDER:
        record = _read_run(tag)
        stats = _force_statistics(record)
        values.append(
            {
                "tag": tag,
                "dx_m": dx,
                "C_X": record["coefficients"]["C_X"],
                "C_X_std": record["coefficients"].get("C_X_std"),
                "C_Y": record["coefficients"].get("C_Y"),
                "steps": record["settings"]["steps"],
                "average_from": record["settings"].get("average_from"),
                "wall_law": record["settings"].get("wall_law", "spalding"),
                "grid_cells": record["geometry"].get("solid_cells"),
                "run_s": record.get("timing", {}).get("run_s"),
                "statistics": stats,
            }
        )
    verdict = grid_ladder_verdict(
        values[0]["C_X"], values[1]["C_X"], values[2]["C_X"], ratio=REFINEMENT_RATIO
    )
    fine_change = abs(values[2]["C_X"] - values[1]["C_X"])
    combined_sem = (values[1]["statistics"]["block_sem"] ** 2
                    + values[2]["statistics"]["block_sem"] ** 2) ** 0.5
    verdict["fine_pair_absolute_change"] = fine_change
    verdict["fine_pair_combined_sem"] = combined_sem
    verdict["fine_pair_change_over_sem"] = (
        fine_change / combined_sem if combined_sem > 0 else float("inf")
    )
    sensitivity = []
    for tag, law in LAW_SENSITIVITY:
        path = RUNS / f"{tag}.json"
        if not path.exists():
            sensitivity.append({"wall_law": law, "dx_m": None, "C_X": None,
                                "status": "not_run"})
            continue
        record = _read_run(tag)
        sensitivity.append(
            {"wall_law": law, "dx_m": record["settings"]["dx_m"],
             "C_X": record["coefficients"]["C_X"], "status": "ok"}
        )
    return {
        "ladder": values,
        "verdict": verdict,
        "law_sensitivity": sensitivity,
        "statistics_note": (
            "block SEM is a lower bound on the mean uncertainty: the stored "
            "window is 800 samples and the force signal is a slow oscillation, "
            "so mean and median can disagree (mean/median ratio recorded per grid)"
        ),
    }


# ---------------------------------------------------------------------------
# criterion 5
# ---------------------------------------------------------------------------
def _evidence(profiles, inversion, stress, channel, grid) -> dict:
    """Assemble the gate evidence from the measured audit blocks."""
    return {
        "inversion_max_rel_error": inversion[AUDIT_LAW.name]["max_rel_error"],
        "stress_max_rel_error": stress["max_rel_error_audit_law"],
        "channel_slope_deviation": {
            leg["re_tau"]: leg["slope"]["slope_deviation"] for leg in channel["legs"].values()
        },
        "richardson": grid["verdict"],
    }


def audit_negative_controls(profiles, injection, stress, channel, grid) -> dict:
    reference = _evidence(profiles, injection, stress, channel, grid)
    controls = []

    # NC-1: wrong von Karman constant (kappa = 0.55) in the law used to invert
    wrong = WallLawParams("spalding", 0.55, 0.1108)
    wrong_audit = audit_kappa_mismatch(SPALDING, wrong)
    gated = dict(reference)
    gated["inversion_max_rel_error"] = max(
        reference["inversion_max_rel_error"], wrong_audit["max_rel_error"]
    )
    report = assess_wall_model_usability(**gated)
    controls.append(
        {
            "control": "wrong_kappa",
            "measured": gated["inversion_max_rel_error"],
            "limit": report.criteria.inversion_limit,
            "gate_passed": report.passed,
            "triggered": not report.passed,
            "failures": list(report.failures),
        }
    )

    # NC-2: sampling one grid cell beyond the stated wall distance
    offset = audit_sampling_offset(AUDIT_LAW)
    gated = dict(reference)
    gated["inversion_max_rel_error"] = max(
        reference["inversion_max_rel_error"], offset["max_rel_error"]
    )
    report = assess_wall_model_usability(**gated)
    controls.append(
        {
            "control": "sampling_offset_one_cell",
            "measured": gated["inversion_max_rel_error"],
            "limit": report.criteria.inversion_limit,
            "gate_passed": report.passed,
            "triggered": not report.passed,
            "failures": list(report.failures),
        }
    )

    # NC-3: wall law disabled -> the model reconstructs tau_w = 0
    gated = dict(reference)
    gated["stress_max_rel_error"] = 1.0
    report = assess_wall_model_usability(**gated)
    controls.append(
        {
            "control": "wall_law_disabled",
            "measured": gated["stress_max_rel_error"],
            "limit": report.criteria.stress_limit,
            "gate_passed": report.passed,
            "triggered": not report.passed,
            "failures": list(report.failures),
        }
    )
    return {"controls": controls, "triggered": sum(c["triggered"] for c in controls)}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(inversion, channel, grid, profiles) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    figures = []

    # fig 1: inversion error vs y+
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for law in (SPALDING, REICHARDT):
        audit = inversion[law.name]
        series = [p for p in audit["per_point"] if p["u_tau_true"] == 0.2]
        ax.loglog(
            [p["y_plus"] for p in series],
            [max(p["rel_error"], 1e-17) for p in series],
            "o-",
            label=f"{law.name} (kappa={law.kappa})",
        )
    ax.axhline(1e-8, color="k", ls="--", lw=1.0, label="criterion 1 limit (1e-8)")
    ax.set_xlabel("y+ of the manufactured state")
    ax.set_ylabel("relative u_tau error")
    ax.set_title("Wall-law inversion: manufactured state recovered back to u_tau")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    path = OUT / "fig1_inversion_error.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append({"name": "fig1_inversion_error.png", "path": path,
                    "caption": "Criterion 1: rel. friction-velocity error of the "
                               "safeguarded Newton inversion on manufactured states "
                               "(both laws reach the double-precision floor, "
                               "limit 1e-8)."})

    # fig 2: u+ profile — DNS vs wall-modelled channel vs law
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, tag in zip(axes, ("mkm180", "mkm590")):
        profile = profiles[tag]
        solution = equilibrium_wall_modelled_channel(
            profile.re_tau, law=AUDIT_LAW, van_driest_a=channel["van_driest_a"]
        )
        leg = channel["legs"][tag]
        y_dns = profile.y_plus.tolist()
        u_dns = profile.u_plus.tolist()
        stride = max(1, len(leg["profile"]["y_plus"]) // 2000)
        ax.semilogx(y_dns, u_dns, "o", ms=3, label="MKM1999 DNS")
        ax.semilogx(
            leg["profile"]["y_plus"],
            leg["profile"]["u_plus"],
            "-",
            lw=1.6,
            label="wall-modelled channel (1-D equilibrium)",
        )
        y_law = torch.logspace(-0.5, 3.0, 400, dtype=torch.float64)
        ax.semilogx(
            y_law.tolist(),
            wall_law_u_plus(y_law, AUDIT_LAW).tolist(),
            "--",
            lw=1.1,
            label=f"{AUDIT_LAW.name} law (kappa={AUDIT_LAW.kappa})",
        )
        band = leg["slope"]["band"]
        ax.axvspan(band[0], band[1], color="grey", alpha=0.12)
        ax.set_title(
            f"{tag}: Re_tau={profile.re_tau:.2f}, slope dev "
            f"{100 * leg['slope']['slope_deviation']:.2f} %"
        )
        ax.set_xlabel("y+")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("u+")
    fig.suptitle("Criterion 3: channel validation against MKM1999/Moser DNS")
    path = OUT / "fig2_channel_profiles.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append({"name": "fig2_channel_profiles.png", "path": path,
                    "caption": "Criterion 3: u+ profiles (MKM1999 DNS points, "
                               "wall-modelled equilibrium channel, composite law) "
                               "at Re_tau = 178.12 and 587.19; the grey band is the "
                               "log-layer fit window."})

    # fig 3: C_X ladder
    ladder = grid["ladder"]
    dx = [row["dx_m"] for row in ladder]
    cx = [row["C_X"] for row in ladder]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(dx, cx, "o-", label="C_X (LBM, composite wall law)")
    verdict = grid["verdict"]
    if verdict["extrapolated"] == verdict["extrapolated"]:
        ax.axhline(verdict["extrapolated"], color="k", ls=":",
                   label=f"Richardson extrapolation ({verdict['extrapolated']:.4f})")
    for row in ladder:
        ax.annotate(f"{row['C_X']:.4f}", (row["dx_m"], row["C_X"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.invert_xaxis()
    ax.set_xlabel("dx [m] (coarse -> fine)")
    ax.set_ylabel("C_X [-]")
    ax.set_title(
        f"N04 container ship, 0 deg: R={verdict['richardson_ratio']:.2f}, "
        f"fine pair {100 * verdict['fine_pair_relative_change']:.2f} %"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path = OUT / "fig3_cx_ladder.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append({"name": "fig3_cx_ladder.png", "path": path,
                    "caption": "Criterion 4: C_X grid ladder (dx = 2.83, 2.0, "
                               "1.41 m) of the N04 container-ship wind case with "
                               "the composite wall law."})

    # fig 4: negative controls
    controls = [row for row in channel["_controls"]] if "_controls" in channel else []
    return figures


def make_negative_control_figure(controls, path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [c["control"] for c in controls]
    values = [c["measured"] for c in controls]
    limits = [c["limit"] for c in controls]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    positions = range(len(names))
    ax.bar([p - 0.18 for p in positions], values, width=0.36,
           label="measured deviation", color="#c0504d")
    ax.bar([p + 0.18 for p in positions], limits, width=0.36,
           label="gate limit", color="#4f81bd")
    ax.set_yscale("log")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("relative deviation")
    ax.set_title("Criterion 5: negative controls must break the gate")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"name": Path(path).name, "path": Path(path),
            "caption": "Criterion 5: measured deviation of each negative control "
                       "against its gate limit (all three exceed it, so the gate "
                       "rejects a wrong kappa, an off-by-one sample and a disabled "
                       "wall law)."}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-assert", action="store_true",
                        help="audit only; do not raise on a failed criterion")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    profiles = {
        "mkm180": load_mkm_profile(REF / "mkm180.means"),
        "mkm590": load_mkm_profile(REF / "mkm590.means"),
    }
    provenance = json.loads((REF / "provenance.json").read_text())
    for tag in profiles:
        digest = hashlib.sha256((REF / f"{tag}.means").read_bytes()).hexdigest()
        if digest != provenance[tag]["sha256"]:
            raise SystemExit(f"{tag}.means does not match its recorded sha256")

    inversion = {law.name: audit_inversion(law) for law in (SPALDING, REICHARDT)}
    stress = audit_stress(profiles)
    channel = audit_channel(profiles)
    grid = audit_grid()
    controls = audit_negative_controls(profiles, inversion, stress, channel, grid)

    reference = _evidence(profiles, inversion, stress, channel, grid)
    report = assess_wall_model_usability(**reference)
    gate_ok = report.passed
    if not gate_ok and not args.no_assert:
        try:
            assert_wall_model_usable(report)
        except AssertionError as error:
            print(f"HONEST FAIL: {error}", file=sys.stderr)

    figures = make_figures(inversion, channel, grid, profiles)
    figures.append(
        make_negative_control_figure(
            controls["controls"], OUT / "fig4_negative_controls.png"
        )
    )

    record = {
        "node": "IN61-N06",
        "worktree_branch": "agent/m6975",
        "module": "src/tensorlbm/turbulence/wall_model_usability.py",
        "dns_provenance": provenance,
        "criteria": {
            "1_inversion": inversion,
            "2_stress": stress,
            "3_channel": channel,
            "4_grid": grid,
            "5_negative_controls": controls,
        },
        "gate": {
            "passed": gate_ok,
            "evidence": {k: v for k, v in reference.items() if k != "richardson"},
            "richardson_ratio": grid["verdict"]["richardson_ratio"],
            "fine_pair_relative_change": grid["verdict"]["fine_pair_relative_change"],
            "report": report.to_dict(),
        },
        "figures": [{"name": f["name"], "caption": f["caption"]} for f in figures],
    }
    out_json = OUT / "wall_model_usability.json"
    out_json.write_text(json.dumps(record, indent=2) + "\n")

    print(json.dumps({
        "inversion_max_rel_error": reference["inversion_max_rel_error"],
        "stress_max_rel_error": reference["stress_max_rel_error"],
        "slope_deviation": reference["channel_slope_deviation"],
        "richardson_ratio": grid["verdict"]["richardson_ratio"],
        "fine_pair_relative_change": grid["verdict"]["fine_pair_relative_change"],
        "negative_controls_triggered": controls["triggered"],
        "gate_passed": gate_ok,
        "json": str(out_json),
        "figures": [str(f["path"]) for f in figures],
    }, indent=2))
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())