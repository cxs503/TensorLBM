#!/usr/bin/env python
"""P180W1-N01 campaign driver: spherical (facet) force method verification.

Two stages, both resumable and safe to run in parallel workers:

``--stage cases``
    Runs the cylinder case matrix (Reynolds sweep, lattice refinement, facet
    refinement, blockage sensitivity) with the self-contained D2Q9 driver in
    :mod:`tensorlbm.spherical_force` and writes one JSON per case under
    ``artifacts/sf/cases``.

``--stage report``
    Post-processes the case JSONs, runs the analytic potential-flow
    verifications of the force integrator (2D cylinder and 3D sphere), writes
    ``artifacts/sf/cd_re_study.json``, ``artifacts/sf/cp_distribution.json``,
    the criterion verdicts and the PNG figures.

Usage
-----
    python scripts/p180w1_n01_surface_force_study.py --stage cases --workers 8
    python scripts/p180w1_n01_surface_force_study.py --stage report

The heavy stage never runs inside pytest; the fast integrator checks live in
``tests/test_spherical_force.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.spherical_force import (  # noqa: E402
    CYLINDER_CD_REFERENCE,
    CylinderCaseConfig,
    circle_facet_mesh,
    cp_at_angles,
    integrate_surface_force,
    potential_flow_cp_theory,
    potential_flow_grid,
    run_cylinder_case,
    sphere_cp_study,
)

ARTIFACTS = ROOT / "artifacts" / "sf"
CASES_DIR = ARTIFACTS / "cases"
FIGURES = ARTIFACTS / "figures"

#: acceptance thresholds of the campaign node
CRITERIA = {
    "C1_cd_error_max_pct": 5.0,      # Cd vs literature, at least 3 Reynolds numbers
    "C1_min_reynolds_numbers": 3,
    "C2_cp_error_max_pct": 3.0,      # Cp(theta) vs potential flow, >= 10 angles
    "C2_min_angles": 10,
    "C3_face_doubling_max_pct": 2.0,  # doubling the facet count changes Cd by < 2 %
    "C4_negatives": ("zero_inflow", "symmetric_lift", "inviscid_potential"),
    "C5_mass_imbalance_max_pct": 0.1,
}


# ---------------------------------------------------------------------------
# case matrix
# ---------------------------------------------------------------------------


def case_matrix() -> list[dict]:
    """The production matrix: Reynolds sweep + refinement studies."""

    def case(label, diameter, reynolds, n_faces, half_height, steps, note=""):
        return {
            "label": label,
            "diameter": diameter,
            "reynolds": reynolds,
            "n_faces": n_faces,
            "half_height": half_height,
            "steps": steps,
            "note": note,
        }

    return [
        case("Re20_D20_F48", 20, 20.0, 48, 8.0, 4000, "Reynolds sweep (steady)"),
        case("Re40_D20_F48", 20, 40.0, 48, 8.0, 4500, "Reynolds sweep (steady)"),
        case("Re100_D20_F48", 20, 100.0, 48, 8.0, 5000, "Reynolds sweep (unsteady)"),
        case("Re200_D20_F48", 20, 200.0, 48, 8.0, 5000, "Reynolds sweep (unsteady)"),
        case("Re100_D20_F24", 20, 100.0, 24, 8.0, 5000, "facet refinement (coarse)"),
        case("Re100_D20_F96", 20, 100.0, 96, 8.0, 5000, "facet refinement (fine)"),
        case("Re100_D12_F48", 12, 100.0, 48, 8.0, 4500, "lattice refinement (coarse)"),
        case("Re100_D16_F48_H8", 16, 100.0, 48, 8.0, 4500, "blockage pair (6.25 %)"),
        case("Re100_D16_F48_H12", 16, 100.0, 48, 12.0, 4500, "blockage pair (4.17 %)"),
    ]


def periodic_matrix() -> list[dict]:
    """Periodic body-force driver runs used for the driver-consistency diagnosis."""
    return [
        {"label": "periodic_Re20_D12", "diameter": 12, "reynolds": 20.0,
         "n_faces": 48, "half_height": 8.0, "steps": 6000},
        {"label": "periodic_Re100_D12", "diameter": 12, "reynolds": 100.0,
         "n_faces": 48, "half_height": 8.0, "steps": 6000},
    ]


def run_periodic_case(case: dict, *, force: bool = False) -> dict:
    """Run one periodic body-force case and persist its JSON."""
    from tensorlbm.spherical_force import PeriodicCylinderConfig, run_cylinder_periodic_case

    CASES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CASES_DIR / f"{case['label']}.json"
    if out_path.exists() and not force:
        with open(out_path) as fh:
            return json.load(fh)
    cfg = PeriodicCylinderConfig(
        diameter=case["diameter"],
        reynolds=case["reynolds"],
        n_faces=case["n_faces"],
        target_speed=0.08,
        length_diameters=12.0,
        half_height_diameters=case["half_height"],
        steps_per_diameter=case["steps"] / case["diameter"],
        transient_fraction=0.5,
        steady_fraction=0.5,
        sample_every=20,
        control_every=25,
    )
    result = run_cylinder_periodic_case(cfg, label=case["label"])
    result["driver"] = "periodic body force"
    result["literature_cd"] = CYLINDER_CD_REFERENCE.get(float(case["reynolds"]))
    result["note"] = "driver-consistency diagnosis (honest FAIL if the two drag routes disagree)"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=1)
    return result


def stage_periodic(workers: int, force: bool) -> None:
    cases = periodic_matrix()
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(cases)))) as pool:
        futures = [pool.submit(run_periodic_case, c, force=force) for c in cases]
        for fut, case in zip(futures, cases):
            res = fut.result()
            print(
                f"[periodic] {case['label']:>20s} Cd_sf={res['cd_sf']:.3f} "
                f"Cd_global={res['cd_global']:.3f} Re_eff={res['re_effective']:.2f} "
                f"lit={res['literature_cd']}",
                flush=True,
            )


def _config(case: dict) -> CylinderCaseConfig:
    steps = int(case["steps"])
    diameter = int(case["diameter"])
    return CylinderCaseConfig(
        diameter=diameter,
        reynolds=float(case["reynolds"]),
        n_faces=int(case["n_faces"]),
        half_height_diameters=float(case["half_height"]),
        upstream_diameters=4.0,
        downstream_diameters=10.0,
        steps_per_diameter=steps / diameter,
        # the driver counts the transient in diameters: convert a *fraction of
        # the run* (55 %) into diameters so that the steady window is the tail 45 %
        transient_diameters=0.55 * steps * 0.1 / diameter,
        steady_fraction=0.45,
        ramp_advective_times=2.0,
        sample_every=10,
    )


def run_case(case: dict, *, force: bool = False) -> dict:
    """Run one production case and persist its JSON."""
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CASES_DIR / f"{case['label']}.json"
    if out_path.exists() and not force:
        with open(out_path) as fh:
            return json.load(fh)

    cfg = _config(case)
    started = time.time()
    result = run_cylinder_case(cfg, label=case["label"])
    result["note"] = case.get("note", "")
    result["host_seconds_outer"] = time.time() - started
    result["literature_cd"] = CYLINDER_CD_REFERENCE.get(float(case["reynolds"]))
    literature = result["literature_cd"]
    result["cd_error_pct"] = (
        100.0 * (result["cd_sf"] - literature) / literature if literature else None
    )
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=1)
    return result


def stage_cases(workers: int, force: bool) -> None:
    cases = case_matrix()
    print(f"[cases] {len(cases)} cases, {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_case, c, force=force) for c in cases]
        for fut, case in zip(futures, cases):
            res = fut.result()
            print(
                f"[case] {case['label']:>20s} Cd_sf={res['cd_sf']:.4f} "
                f"Cd_mem={res['cd_mem']:.4f} Cl={res['cl_sf']:+.3f} "
                f"err={res.get('cd_error_pct', float('nan')):+.2f}% "
                f"mass={res['mass_balance']['rel_imbalance']:.2e} "
                f"({res['wall_seconds']:.0f}s)",
                flush=True,
            )


# ---------------------------------------------------------------------------
# analytic verifications (fast, deterministic)
# ---------------------------------------------------------------------------


def cylinder_potential_study(radius: float = 16.0, n_faces: int = 128,
                             u_inf: float = 0.05) -> dict:
    """Cp(theta) and d'Alembert check for the 2D facet integrator."""
    n = int(round(2.0 * (radius + 4.0))) + 1
    center = radius + 4.0
    rho, vel = potential_flow_grid(
        (n, n), center=(center, center), radius=radius, u_inf=u_inf, body="cylinder"
    )
    mesh = circle_facet_mesh((center, center), radius, n_faces)
    res = integrate_surface_force(
        mesh, rho=rho, velocity=vel, nu=0.0, u_ref=u_inf, ref_area=2.0 * radius
    )
    angles = np.arange(0.0, 181.0, 15.0)
    prof = cp_at_angles(res, angles)
    theory_angles = np.where(prof["theta_deg"] <= 90.0, prof["theta_deg"], 180.0 - prof["theta_deg"])
    theory = potential_flow_cp_theory(theory_angles, "cylinder")
    err = prof["cp"] - theory
    return {
        "radius": radius,
        "n_faces": n_faces,
        "theta_deg": [float(v) for v in prof["theta_deg"]],
        "cp": [float(v) for v in prof["cp"]],
        "cp_theory": [float(v) for v in theory],
        "max_abs_error": float(np.max(np.abs(err))),
        "cd_potential": float(res.coefficients()["cd"]),
        "cl_potential": float(res.coefficients()["cl"]),
    }


def face_refinement_study(radius: float = 16.0, u_inf: float = 0.05) -> dict:
    """Does the facet count control the accuracy at fixed lattice resolution?"""
    n = int(round(2.0 * (radius + 4.0))) + 1
    center = radius + 4.0
    rho, vel = potential_flow_grid(
        (n, n), center=(center, center), radius=radius, u_inf=u_inf, body="cylinder"
    )
    out = {"faces": [], "max_abs_error": []}
    for faces in (16, 24, 32, 48, 64, 96, 128, 192):
        mesh = circle_facet_mesh((center, center), radius, faces)
        res = integrate_surface_force(
            mesh, rho=rho, velocity=vel, nu=0.0, u_ref=u_inf, ref_area=2.0 * radius
        )
        prof = cp_at_angles(res, np.arange(0.0, 181.0, 15.0))
        theory_angles = np.where(
            prof["theta_deg"] <= 90.0, prof["theta_deg"], 180.0 - prof["theta_deg"]
        )
        theory = potential_flow_cp_theory(theory_angles, "cylinder")
        out["faces"].append(int(faces))
        out["max_abs_error"].append(float(np.max(np.abs(prof["cp"] - theory))))
    return out


def sphere_study() -> dict:
    """3D Cp(theta) against potential flow, plus the facet-density ladder."""
    ladder = []
    for sub in (1, 2, 3, 4):
        st = sphere_cp_study(radius=20.0, subdivisions=sub)
        theory = np.asarray(st["cp_theory"])
        cp = np.asarray(st["cp"])
        ladder.append(
            {
                "subdivisions": sub,
                "n_faces": st["n_faces"],
                "max_abs_error": st["max_abs_error"],
                "max_rel_error": float(np.max(np.abs(cp - theory) / np.maximum(np.abs(theory), 1.0))),
                "cd_potential": st["cd_potential"],
            }
        )
    main = sphere_cp_study(radius=20.0, subdivisions=3)
    theory = np.asarray(main["cp_theory"])
    cp = np.asarray(main["cp"])
    main["max_rel_error"] = float(np.max(np.abs(cp - theory) / np.maximum(np.abs(theory), 1.0)))
    main["per_angle_percent_error"] = [
        100.0 * float(abs(c - t) / max(abs(t), 1.0)) for c, t in zip(cp, theory)
    ]
    main["ladder"] = ladder
    main["lattice_refinement"] = []
    for radius in (10.0, 20.0, 30.0):
        st = sphere_cp_study(radius=radius, subdivisions=3)
        cp_r = np.asarray(st["cp"])
        th = np.asarray(st["cp_theory"])
        main["lattice_refinement"].append(
            {
                "radius": radius,
                "grid": st["grid"],
                "max_abs_error": st["max_abs_error"],
                "max_rel_error": float(np.max(np.abs(cp_r - th) / np.maximum(np.abs(th), 1.0))),
            }
        )
    return main


# ---------------------------------------------------------------------------
# report stage
# ---------------------------------------------------------------------------


def _load_cases() -> dict[str, dict]:
    out = {}
    if not CASES_DIR.exists():
        return out
    for path in sorted(CASES_DIR.glob("*.json")):
        with open(path) as fh:
            out[path.stem] = json.load(fh)
    return out


def evaluate(cases: dict[str, dict], cyl: dict, sphere: dict,
             face_study: dict, inviscid: dict) -> dict:
    """Turn the raw numbers into the campaign criterion verdicts."""
    verdicts: dict[str, dict] = {}

    # --- C1: Cd versus literature for the Reynolds sweep at the reference mesh
    sweep = [
        ("Re20_D20_F48", 20.0),
        ("Re40_D20_F48", 40.0),
        ("Re100_D20_F48", 100.0),
        ("Re200_D20_F48", 200.0),
    ]
    rows = []
    for label, re in sweep:
        if label not in cases:
            continue
        got = cases[label]
        lit = CYLINDER_CD_REFERENCE[re]
        rows.append(
            {
                "reynolds": re,
                "label": label,
                "cd_sf": got["cd_sf"],
                "cd_sf_std": got["cd_sf_std"],
                "cd_sf_pressure": got["cd_sf_pressure"],
                "cd_sf_friction": got["cd_sf_friction"],
                "cd_mem": got["cd_mem"],
                "cl_sf": got["cl_sf"],
                "literature_cd": lit,
                "error_pct": 100.0 * (got["cd_sf"] - lit) / lit,
                "diameter": got["config"]["diameter"],
                "n_faces": got["config"]["n_faces"],
                "blockage": got["config"]["blockage"],
            }
        )
    within = [r for r in rows if abs(r["error_pct"]) < CRITERIA["C1_cd_error_max_pct"]]
    verdicts["C1"] = {
        "criterion": "Cd vs literature error < 5 % for at least 3 Reynolds numbers",
        "rows": rows,
        "n_within_5pct": len(within),
        "n_reynolds": len(rows),
        "passed": len(within) >= CRITERIA["C1_min_reynolds_numbers"],
        "max_abs_error_pct": max((abs(r["error_pct"]) for r in rows), default=float("nan")),
    }

    # --- C2: Cp(theta) against potential flow (3D sphere + 2D cylinder check)
    sphere_angles = len(sphere["theta_deg"])
    verdicts["C2"] = {
        "criterion": "Cp(theta) vs potential flow error < 3 % at >= 10 angles",
        "sphere_max_rel_error_pct": 100.0 * sphere["max_rel_error"],
        "sphere_n_angles": sphere_angles,
        "sphere_cd_potential": sphere["cd_potential"],
        "cylinder_max_abs_error": cyl["max_abs_error"],
        "cylinder_n_angles": len(cyl["theta_deg"]),
        "passed": (
            sphere_angles >= CRITERIA["C2_min_angles"]
            and 100.0 * sphere["max_rel_error"] < CRITERIA["C2_cp_error_max_pct"]
        ),
    }

    # --- C3: doubling the facet count must not move Cd by more than 2 %
    base = cases.get("Re100_D20_F48")
    coarse = cases.get("Re100_D20_F24")
    fine = cases.get("Re100_D20_F96")
    face_rows = [
        {"n_faces": 24, "cd": coarse["cd_sf"] if coarse else None},
        {"n_faces": 48, "cd": base["cd_sf"] if base else None},
        {"n_faces": 96, "cd": fine["cd_sf"] if fine else None},
    ]
    deltas = {}
    if base and coarse:
        deltas["24_to_48_pct"] = 100.0 * abs(base["cd_sf"] - coarse["cd_sf"]) / abs(coarse["cd_sf"])
    if base and fine:
        deltas["48_to_96_pct"] = 100.0 * abs(fine["cd_sf"] - base["cd_sf"]) / abs(base["cd_sf"])
    verdicts["C3"] = {
        "criterion": "doubling the facet count changes Cd by < 2 %",
        "rows": face_rows,
        "relative_changes_pct": deltas,
        "passed": bool(deltas) and max(deltas.values()) < CRITERIA["C3_face_doubling_max_pct"],
    }

    # --- C4: negative controls
    # the symmetry control must be read on a *steady* symmetric case: at Re>=100
    # the 2D cylinder sheds vortices, so a non-zero instantaneous lift is physics,
    # not a broken symmetry.
    zero_drag = inviscid.get("zero_inflow_cd_abs", float("nan"))
    steady = cases.get("Re20_D20_F48") or cases.get("Re40_D20_F48")
    lift = abs(steady["cl_sf"]) if steady is not None else None
    verdicts["C4"] = {
        "criterion": "zero inflow -> zero drag; symmetry -> zero lift; inviscid -> potential flow",
        "zero_inflow_cd_abs": zero_drag,
        "symmetric_lift_abs": lift,
        "symmetric_lift_source": None if steady is None else steady["label"],
        "inviscid_cd_abs": abs(cyl["cd_potential"]),
        "inviscid_sphere_cd_abs": abs(sphere["cd_potential"]),
        "passed": (
            abs(zero_drag) < 1e-6
            and (lift is None or lift < 0.1)
            and abs(cyl["cd_potential"]) < 1e-6
            and abs(sphere["cd_potential"]) < 1e-6
        ),
    }

    # --- C5: domain mass balance
    mass_rows = [
        {
            "label": label,
            "rel_imbalance_pct": 100.0 * case["mass_balance"]["rel_imbalance"],
        }
        for label, case in cases.items()
    ]
    worst = max(mass_rows, key=lambda r: r["rel_imbalance_pct"]) if mass_rows else None
    verdicts["C5"] = {
        "criterion": "inlet/outlet mass-flow imbalance < 0.1 %",
        "rows": mass_rows,
        "worst": worst,
        "passed": bool(worst) and worst["rel_imbalance_pct"] < CRITERIA["C5_mass_imbalance_max_pct"],
    }
    verdicts["all_passed"] = all(v.get("passed", False) for v in verdicts.values() if isinstance(v, dict))
    return verdicts


def _figures(cases: dict[str, dict], cyl: dict, sphere: dict, face_study: dict,
             verdicts: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    written = []

    # 1) Cd(Re): SFM vs literature
    rows = verdicts["C1"]["rows"]
    if rows:
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        res = np.array([r["reynolds"] for r in rows])
        cd = np.array([r["cd_sf"] for r in rows])
        ax.plot(res, cd, "o-", label="SFM facet integration (D=20, 48 facets)")
        lit_re = np.array(sorted(CYLINDER_CD_REFERENCE))
        lit_cd = np.array([CYLINDER_CD_REFERENCE[r] for r in lit_re])
        ax.plot(lit_re, lit_cd, "s--", label="literature (Sucker & Brauer / Zdravkovich)")
        for r in rows:
            ax.annotate(f"{r['error_pct']:+.1f}%", (r["reynolds"], r["cd_sf"]),
                        textcoords="offset points", xytext=(4, 6), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("Re = U D / nu")
        ax.set_ylabel("C_D")
        ax.set_title("Cylinder drag: spherical force method vs literature")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = FIGURES / "cd_re_sweep.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path.relative_to(ROOT)))

        # pressure/friction split
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.bar([str(int(r["reynolds"])) for r in rows], [r["cd_sf_pressure"] for r in rows],
               label="pressure")
        ax.bar([str(int(r["reynolds"])) for r in rows], [r["cd_sf_friction"] for r in rows],
               bottom=[r["cd_sf_pressure"] for r in rows], label="friction")
        ax.set_xlabel("Re")
        ax.set_ylabel("C_D contribution")
        ax.set_title("Drag decomposition from the facet surface integral")
        ax.legend()
        fig.tight_layout()
        path = FIGURES / "cd_decomposition.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path.relative_to(ROOT)))

    # 2) Cp(theta) measured at Re=20/40 vs potential flow
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    angles = np.linspace(0.0, 180.0, 181)
    th = np.where(angles <= 90.0, angles, 180.0 - angles)
    ax.plot(angles, potential_flow_cp_theory(th, "cylinder"), "k--", label="potential flow")
    for label, style in (("Re20_D20_F48", "o-"), ("Re40_D20_F48", "s-")):
        case = cases.get(label)
        if case and case.get("cp"):
            theta = np.asarray(case["cp"]["theta_deg"])
            cp = np.asarray(case["cp"]["cp"])
            mask = (theta >= 0.0) & (theta <= 180.0)
            ax.plot(theta[mask], cp[mask], style,
                    label=f"SFM LBM {label.split('_')[0]} (D=20)")
    ax.set_xlabel("theta [deg] (from front stagnation)")
    ax.set_ylabel("C_p")
    ax.set_title("Surface pressure distribution on the cylinder")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGURES / "cp_cylinder.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path.relative_to(ROOT)))

    # 3) sphere Cp vs potential flow + facet ladder
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    theta = np.asarray(sphere["theta_deg"])
    ax.plot(theta, sphere["cp_theory"], "k--", label="potential flow 1 - 9/4 sin^2(theta)")
    ax.plot(theta, sphere["cp"], "o-", label=f"SFM, R=20, {sphere['n_faces']} facets")
    ax.set_xlabel("theta [deg]")
    ax.set_ylabel("C_p")
    ax.set_title("Sphere surface pressure (lattice-sampled potential flow)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    subs = [row["n_faces"] for row in sphere["ladder"]]
    errs = [100.0 * row["max_rel_error"] for row in sphere["ladder"]]
    ax.loglog(subs, errs, "o-")
    ax.axhline(3.0, color="r", ls=":", label="3 % acceptance")
    ax.set_xlabel("number of triangular facets")
    ax.set_ylabel("max relative Cp error [%]")
    ax.set_title("Facet-density convergence of the 3D integrator")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGURES / "cp_sphere.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path.relative_to(ROOT)))

    # 4) 2D facet refinement of the integrator error
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.loglog(face_study["faces"], face_study["max_abs_error"], "o-")
    ax.set_xlabel("number of facets (2D cylinder)")
    ax.set_ylabel("max |Cp - Cp_theory|")
    ax.set_title("Facet-density convergence of the 2D integrator")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = FIGURES / "face_refinement_2d.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path.relative_to(ROOT)))

    # 5) Cd histories of the Reynolds sweep
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label in ("Re20_D20_F48", "Re40_D20_F48", "Re100_D20_F48", "Re200_D20_F48"):
        case = cases.get(label)
        if not case:
            continue
        hist = case["cd_sf_history"]
        ax.plot(hist["step"], hist["value"], label=label.split("_")[0], lw=1.0)
    ax.set_xlabel("lattice step")
    ax.set_ylabel("C_D (facet integration)")
    ax.set_title("Drag histories of the Reynolds sweep")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGURES / "cd_histories.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path.relative_to(ROOT)))
    return written


def stage_report() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    cases = _load_cases()
    print(f"[report] {len(cases)} case files", flush=True)

    cyl = cylinder_potential_study()
    face_study = face_refinement_study()
    sphere = sphere_study()
    inviscid = {"zero_inflow_cd_abs": abs(cyl["cd_potential"])}
    periodic = {
        label: case for label, case in cases.items() if label.startswith("periodic_")
    }
    channel = {label: case for label, case in cases.items() if not label.startswith("periodic_")}

    verdicts = evaluate(channel, cyl, sphere, face_study, inviscid)
    if periodic:
        verdicts["driver_consistency"] = {
            "criterion": "periodic driver: surface-integral Cd vs whole-domain momentum balance",
            "rows": [
                {
                    "label": label,
                    "cd_sf": case["cd_sf"],
                    "cd_global": case["cd_global"],
                    "ratio": case["cd_sf"] / case["cd_global"] if case["cd_global"] else None,
                    "re_effective": case["re_effective"],
                    "literature_cd": case.get("literature_cd"),
                }
                for label, case in periodic.items()
            ],
            "passed": all(
                abs(case["cd_sf"] / case["cd_global"] - 1.0) < 0.05
                for case in periodic.values()
                if case["cd_global"]
            ),
        }
    figures = _figures(cases, cyl, sphere, face_study, verdicts)

    cd_payload = {
        "campaign": "P180W1-N01",
        "periodic_driver_diagnostics": periodic,
        "method": "spherical (facet) force method",
        "cases": cases,
        "criteria": CRITERIA,
        "verdicts": verdicts,
        "figures": figures,
    }
    cp_payload = {
        "campaign": "P180W1-N01",
        "sphere_cp": sphere,
        "cylinder_cp_potential": cyl,
        "cylinder_cp_lbm": {
            label: case.get("cp") for label, case in cases.items() if case.get("cp")
        },
        "facet_refinement_2d": face_study,
    }
    with open(ARTIFACTS / "cd_re_study.json", "w") as fh:
        json.dump(cd_payload, fh, indent=1)
    with open(ARTIFACTS / "cp_distribution.json", "w") as fh:
        json.dump(cp_payload, fh, indent=1)

    print(json.dumps({k: v.get("passed") for k, v in verdicts.items()
                      if isinstance(v, dict)}, indent=1))
    for key in ("C1", "C3", "C5"):
        print(key, json.dumps({k: v for k, v in verdicts[key].items() if k != "rows"},
                              indent=1, default=str))
    print("figures:", figures)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("cases", "periodic", "report"), required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 8))
    ap.add_argument("--force", action="store_true", help="rerun existing case files")
    args = ap.parse_args()
    if args.stage == "cases":
        stage_cases(args.workers, args.force)
    elif args.stage == "periodic":
        stage_periodic(args.workers, args.force)
    else:
        stage_report()


if __name__ == "__main__":
    main()