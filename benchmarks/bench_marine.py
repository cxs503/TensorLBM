"""TensorLBM – Marine / Ship & Ocean Engineering Benchmark Suite.

Runs five canonical benchmarks and reports quantitative comparisons against
published reference data.

Quick mode (default): reduced grids / step counts for fast validation.
Full mode (``--full``): production-quality settings for converged results.

Benchmarks
----------
1. **Cylinder flow (Re = 100)** – Strouhal and drag comparison vs Williamson (1988).
2. **Sloshing tank** – measured oscillation frequency vs Faltinsen (1978) model.
3. **Near-bed pipeline flow (Re = 200, e/D = 0.5)** – Strouhal number vs
   Bearman & Zdravkovich (1978).
4. **Turbulent channel (Re_τ = 100)** – log-law velocity profile comparison.
5. **3-D ship workflow (Re = 200)** – CAD block coefficient, force symmetry, and wake analysis.
6. **SUBOFF resistance** – ITTC-1957 friction-drag coefficient with voxel refinement.
7. **Marine geometry library** – consistency checks for multi-hull CAD generators.

Usage::

    PYTHONPATH=src python benchmarks/bench_marine.py
    PYTHONPATH=src python benchmarks/bench_marine.py --full
    PYTHONPATH=src python benchmarks/bench_marine.py --cases cylinder sloshing
    PYTHONPATH=src python benchmarks/bench_marine.py --output-root /tmp/bench

Note on quick-mode convergence
-------------------------------
The quick settings use fewer steps than a production run.  The cylinder
benchmark requires ~20 000 steps for vortex shedding to develop fully; the
sloshing benchmark uses a 50× larger gravity value (g=1e-3) to shorten the
natural oscillation period; the turbulent channel needs ~50 000 steps for
a fully-developed log layer.  Use ``--full`` for publication-quality results.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

# Cylinder flow Re = 100 (isolated, free-stream)
# Williamson & Roshko (1988) / Zdravkovich (1997)
REF_CYLINDER_ST = 0.166  # Strouhal number
REF_CYLINDER_CD = 1.38   # mean drag coefficient

# Sloshing tank: Faltinsen (1978) – analytical, computed per-run

# Near-bed pipeline Re = 200, e/D = 0.5
# Bearman & Zdravkovich (1978), also Price et al. (2002)
REF_PIPELINE_ST = 0.183  # Strouhal number at e/D = 0.5

# Turbulent channel Re_τ = 100
# Log-law: u⁺ = (1/κ) ln(y⁺) + B,  κ = 0.41, B = 5.2  (Moser et al., 1999)
LOG_LAW_KAPPA = 0.41
LOG_LAW_B = 5.2

# Ship hull workflow: analytical Cb is used for CAD comparison; hydrodynamic
# checks use force symmetry and wake metrics because no single canonical Cd is
# available at Re = 200 in the confined LBM channel.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def _row(label: str, value: float, ref: float, unit: str = "") -> None:
    err = abs(value - ref) / (abs(ref) + 1e-20) * 100.0
    status = "✓" if err < 20.0 else "✗"
    unit_str = f" {unit}" if unit else ""
    print(
        f"  {label:<35} simulated={value:8.4f}  ref={ref:8.4f}{unit_str}"
        f"   error={err:6.2f}%  {status}"
    )


def _section(title: str) -> None:
    print(f"\n  --- {title} ---")


# ---------------------------------------------------------------------------
# Benchmark 1 – 2D cylinder flow
# ---------------------------------------------------------------------------

def bench_cylinder(output_root: Path, full: bool) -> dict[str, object]:
    """Run 2D cylinder flow and compare St and Cd with Williamson (1988)."""
    from tensorlbm import CylinderFlowConfig, run_cylinder_flow

    if full:
        cfg = CylinderFlowConfig(
            nx=400, ny=120, radius=15.0, u_in=0.08, re=100.0,
            n_steps=60000, output_interval=500,
            output_root=output_root, run_name="bench_cylinder_re100", overwrite=True,
        )
    else:
        # r=5 (D=10) → 12.5% blockage and period ~753 steps; 20 000 steps
        # covers 26 shedding cycles – sufficient for a clean FFT peak.
        cfg = CylinderFlowConfig(
            nx=200, ny=80, radius=5.0, u_in=0.08, re=100.0,
            n_steps=20000, output_interval=500,
            output_root=output_root, run_name="bench_cylinder_re100_quick", overwrite=True,
        )

    t0 = time.perf_counter()
    run_dir = run_cylinder_flow(cfg)
    elapsed = time.perf_counter() - t0

    meta = json.loads((run_dir / "run_metadata.json").read_text())
    st = meta.get("strouhal") or 0.0
    diag = meta.get("diagnostics", [])
    cd_values = [d["cd"] for d in diag if isinstance(d.get("cd"), float) and math.isfinite(d["cd"])]
    cd_mean = sum(cd_values[-10:]) / len(cd_values[-10:]) if len(cd_values) >= 10 else (
        sum(cd_values) / len(cd_values) if cd_values else float("nan")
    )

    _header("Benchmark 1 – 2D Cylinder Flow (Re = 100)")
    print(f"  Grid: {cfg.nx}×{cfg.ny},  steps: {cfg.n_steps},  elapsed: {elapsed:.1f} s")
    _section("Strouhal number")
    _row("St (shedding frequency × D / U)", float(st), REF_CYLINDER_ST)
    _section("Mean drag coefficient")
    _row("Cd (momentum-exchange)", float(cd_mean), REF_CYLINDER_CD)
    print("\n  Reference: Williamson (1988),  Zdravkovich (1997)")

    return {
        "name": "cylinder_re100",
        "st_sim": float(st),
        "st_ref": REF_CYLINDER_ST,
        "cd_sim": float(cd_mean),
        "cd_ref": REF_CYLINDER_CD,
        "elapsed_s": elapsed,
        "run_dir": str(run_dir),
    }


# ---------------------------------------------------------------------------
# Benchmark 2 – Sloshing tank
# ---------------------------------------------------------------------------

def bench_sloshing(output_root: Path, full: bool) -> dict[str, object]:
    """Run sloshing-tank and compare measured frequency with Faltinsen (1978).

    Quick mode uses g = 1e-3 to shorten the natural oscillation period to
    ~1390 steps, allowing ~7 complete periods in 10 000 steps.
    Full mode uses the oceanographically-motivated default g = 2e-5 with
    proportionally more steps.
    """
    from tensorlbm import SloshingTankConfig, run_sloshing_tank

    if full:
        cfg = SloshingTankConfig(
            nx=200, ny=160, water_level=80,
            g=2e-5, forcing_amp=3e-5,
            n_steps=60000, output_interval=200,
            output_root=output_root, run_name="bench_sloshing_full", overwrite=True,
        )
    else:
        # g=5e-4 gives a natural period of ~1960 steps; 10 000 steps covers
        # 5 full oscillation cycles.  forcing_amp is kept small (10% of g) to
        # stay in the linear sloshing regime and avoid large-amplitude overturning.
        cfg = SloshingTankConfig(
            nx=120, ny=80, water_level=40,
            g=5e-4, forcing_amp=5e-5,
            n_steps=10000, output_interval=50,
            output_root=output_root, run_name="bench_sloshing_quick", overwrite=True,
        )

    t0 = time.perf_counter()
    run_dir = run_sloshing_tank(cfg)
    elapsed = time.perf_counter() - t0

    meta = json.loads((run_dir / "run_metadata.json").read_text())
    omega_theory = float(meta.get("omega_theory", 0.0))
    omega_meas = float(meta.get("omega_measured") or 0.0)
    rel_err = meta.get("relative_frequency_error") or float("nan")

    _header("Benchmark 2 – Sloshing Tank (Faltinsen 1978)")
    print(f"  Grid: {cfg.nx}×{cfg.ny},  h/L = {cfg.water_level/cfg.nx:.2f},  steps: {cfg.n_steps}")
    print(f"  Elapsed: {elapsed:.1f} s")
    _section("Natural sloshing frequency")
    print(f"  {'Faltinsen theory ω₀':<35} {omega_theory:.6e} rad/step")
    print(f"  {'LBM measured ω':<35} {omega_meas:.6e} rad/step")
    if math.isfinite(float(rel_err)):
        _row("Relative frequency error", float(rel_err) * 100.0, 0.0, "%")
        status = "✓" if float(rel_err) < 0.20 else "✗"
        print(f"  Acceptance: error < 20%  {status}")
    else:
        print("  (spectrum peak not resolved – increase n_steps or output_interval)")
    print("\n  Reference: Faltinsen (1978) linear sloshing theory")

    return {
        "name": "sloshing_tank",
        "omega_theory": omega_theory,
        "omega_measured": omega_meas,
        "relative_error": float(rel_err) if math.isfinite(float(rel_err)) else None,
        "elapsed_s": elapsed,
        "run_dir": str(run_dir),
    }


# ---------------------------------------------------------------------------
# Benchmark 3 – Near-bed pipeline flow
# ---------------------------------------------------------------------------

def bench_pipeline(output_root: Path, full: bool) -> dict[str, object]:
    """Run near-bed pipeline flow and compare Strouhal with Bearman & Zdravkovich (1978)."""
    from tensorlbm import PipelineFlowConfig, run_pipeline_flow

    if full:
        cfg = PipelineFlowConfig(
            nx=400, ny=160, diameter=20.0, gap_ratio=0.5, u_in=0.05, re=200.0,
            n_steps=30000, output_interval=1000,
            output_root=output_root, run_name="bench_pipeline_eD05", overwrite=True,
        )
    else:
        # 20 000 steps needed for shedding to develop and resolve St accurately
        cfg = PipelineFlowConfig(
            nx=240, ny=100, diameter=14.0, gap_ratio=0.5, u_in=0.05, re=200.0,
            n_steps=20000, output_interval=1000,
            output_root=output_root, run_name="bench_pipeline_eD05_quick", overwrite=True,
        )

    t0 = time.perf_counter()
    run_dir = run_pipeline_flow(cfg)
    elapsed = time.perf_counter() - t0

    meta = json.loads((run_dir / "run_metadata.json").read_text())
    st = float(meta.get("strouhal") or 0.0)
    diag = meta.get("diagnostics", [])
    cd_values = [d["cd"] for d in diag if isinstance(d.get("cd"), float) and math.isfinite(d["cd"])]
    cd_mean = sum(cd_values[-5:]) / len(cd_values[-5:]) if len(cd_values) >= 5 else (
        sum(cd_values) / len(cd_values) if cd_values else float("nan")
    )

    _header("Benchmark 3 – Near-Bed Pipeline Flow (Re = 200, e/D = 0.5)")
    print(f"  Grid: {cfg.nx}×{cfg.ny},  D = {cfg.diameter},  gap_ratio = {cfg.gap_ratio}")
    print(f"  Steps: {cfg.n_steps},  elapsed: {elapsed:.1f} s")
    _section("Strouhal number (wake shedding)")
    _row("St (dominant CL frequency × D / U)", st, REF_PIPELINE_ST)
    _section("Mean drag coefficient")
    print(f"  {'Cd (momentum-exchange)':<35} {cd_mean:8.4f}  (no single ref at this Re/e/D)")
    print("\n  Reference: Bearman & Zdravkovich (1978), Price et al. (2002)")

    return {
        "name": "pipeline_eD05",
        "st_sim": st,
        "st_ref": REF_PIPELINE_ST,
        "cd_sim": float(cd_mean),
        "elapsed_s": elapsed,
        "run_dir": str(run_dir),
    }


# ---------------------------------------------------------------------------
# Benchmark 4 – Turbulent channel flow
# ---------------------------------------------------------------------------

def bench_turbulent_channel(output_root: Path, full: bool) -> dict[str, object]:
    """Run body-force turbulent channel and compare velocity profile with log-law."""
    import csv

    from tensorlbm import TurbulentChannelConfig, run_turbulent_channel

    if full:
        cfg = TurbulentChannelConfig(
            nx=256, ny=64, re_tau=100.0, u_tau=0.005, smagorinsky_cs=0.1,
            n_steps=50000, averaging_start=20000, output_interval=5000,
            output_root=output_root, run_name="bench_channel_retau100", overwrite=True,
        )
    else:
        # 40 000 steps allows the driven channel to develop past the initial
        # transient; averaging begins at step 15 000 for a 25 000-step window.
        cfg = TurbulentChannelConfig(
            nx=128, ny=40, re_tau=100.0, u_tau=0.005, smagorinsky_cs=0.1,
            n_steps=40000, averaging_start=15000, output_interval=5000,
            output_root=output_root, run_name="bench_channel_retau100_quick", overwrite=True,
        )

    t0 = time.perf_counter()
    run_dir = run_turbulent_channel(cfg)
    elapsed = time.perf_counter() - t0

    # Read velocity profile CSV
    profile_path = run_dir / "velocity_profile.csv"
    y_plus_vals: list[float] = []
    u_plus_sim: list[float] = []
    u_plus_ref: list[float] = []

    with profile_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yp = float(row["y_plus"])
            up = float(row["u_plus"])
            ref_raw = row.get("u_plus_loglaw", "")
            try:
                ref_val = float(ref_raw)
            except (ValueError, TypeError):
                ref_val = float("nan")
            if yp > 11.0 and math.isfinite(ref_val) and math.isfinite(up) and up > 0.0:
                y_plus_vals.append(yp)
                u_plus_sim.append(up)
                u_plus_ref.append(ref_val)

    rms_err = float("nan")
    if y_plus_vals:
        sq_errors = [(s - r) ** 2 for s, r in zip(u_plus_sim, u_plus_ref, strict=False)]
        rms_err = math.sqrt(sum(sq_errors) / len(sq_errors))

    _header("Benchmark 4 – Turbulent Channel (Re_τ = 100, Smagorinsky LES)")
    print(f"  Grid: {cfg.nx}×{cfg.ny},  Re_τ = {cfg.re_tau},  steps: {cfg.n_steps}")
    print(f"  Elapsed: {elapsed:.1f} s")
    _section("Log-law velocity profile (y⁺ > 11)")
    print(f"  {'Log-law: u⁺ = (1/κ) ln(y⁺) + B':<45}")
    print(f"  κ = {LOG_LAW_KAPPA},  B = {LOG_LAW_B}  (Moser et al. 1999)")

    if math.isfinite(rms_err):
        print(f"  RMS error |u⁺_sim − u⁺_log| = {rms_err:.4f}")
        print(f"  Log-region points: {len(y_plus_vals)}")
        status = "✓" if rms_err < 3.0 else "✗"
        print(f"  Acceptance: RMS < 3.0 wall units  {status}")
        # Print a few representative rows
        print(f"\n  {'y⁺':>8}  {'u⁺ (sim)':>12}  {'u⁺ (log-law)':>14}")
        print("  " + "-" * 38)
        step = max(1, len(y_plus_vals) // 6)
        for i in range(0, len(y_plus_vals), step):
            print(f"  {y_plus_vals[i]:8.2f}  {u_plus_sim[i]:12.4f}  {u_plus_ref[i]:14.4f}")
    else:
        print("  (No log-layer points resolved – increase ny or Re_τ)")

    print("\n  Reference: Moser, Kim & Mansour (1999) DNS data; log-law constants κ=0.41, B=5.2")

    return {
        "name": "turbulent_channel_retau100",
        "rms_loglaw_err": rms_err if math.isfinite(rms_err) else None,
        "n_loglaw_pts": len(y_plus_vals),
        "elapsed_s": elapsed,
        "run_dir": str(run_dir),
    }


# ---------------------------------------------------------------------------
# Benchmark 5 – 3D Wigley hull flow
# ---------------------------------------------------------------------------

def bench_ship_hull(output_root: Path, full: bool) -> dict[str, object]:
    """Run the ship CAD→simulation→postprocess workflow and report quantitative checks."""
    from tensorlbm import ShipHullFlowConfig, run_ship_hull_flow

    if full:
        cfg = ShipHullFlowConfig(
            nx=160, ny=60, nz=40,
            u_in=0.05, re=200.0,
            hull_length=80.0, hull_beam=8.0, hull_draft=12.0,
            smagorinsky_cs=0.1,
            n_steps=4000, output_interval=200,
            output_root=output_root, run_name="bench_wigley_re200", overwrite=True,
        )
    else:
        cfg = ShipHullFlowConfig(
            nx=80, ny=40, nz=30,
            u_in=0.05, re=200.0,
            hull_length=40.0, hull_beam=6.0, hull_draft=8.0,
            smagorinsky_cs=0.1,
            n_steps=2000, output_interval=200,
            output_root=output_root, run_name="bench_wigley_re200_quick", overwrite=True,
        )

    t0 = time.perf_counter()
    run_dir = run_ship_hull_flow(cfg)
    elapsed = time.perf_counter() - t0

    meta = json.loads((run_dir / "run_metadata.json").read_text())
    cad = meta.get("cad", {})
    post = meta.get("postprocess", {})
    forces = post.get("forces", {})
    wake = post.get("wake", {})
    acceptance = post.get("acceptance", {})
    cb_theoretical = float(cad.get("Cb_theoretical") or cad.get("Cb") or 0.0)
    cb_numerical = float(cad.get("Cb_numerical") or 0.0)
    cb_error_pct = float(cad.get("Cb_relative_error_pct") or 0.0)
    cd_mean = float(forces.get("cd_mean") or 0.0)
    cs_ratio = float(forces.get("cs_abs_ratio_to_cd") or 0.0)
    cl_ratio = float(forces.get("cl_abs_ratio_to_cd") or 0.0)
    wake_deficit = float(wake.get("velocity_deficit_max") or 0.0)

    _header("Benchmark 5 – 3D Ship Workflow (CAD → Flow → Postprocess)")
    print(f"  Grid: {cfg.nx}×{cfg.ny}×{cfg.nz},  L = {cfg.hull_length},  B = {cfg.hull_beam}")
    print(
        f"  T = {cfg.hull_draft},  Re = {cfg.re},"
        f"  hull = {cfg.hull_type},  steps = {cfg.n_steps}"
    )
    print(f"  Elapsed: {elapsed:.1f} s")
    _section("CAD fidelity")
    _row("Block coefficient Cb", cb_numerical, cb_theoretical)
    _section("Hydrodynamic symmetry")
    drag_mark = "✓" if bool(acceptance.get("drag_positive")) else "✗"
    cs_mark = "✓" if bool(acceptance.get("sideforce_small")) else "✗"
    cl_mark = "✓" if bool(acceptance.get("lift_small")) else "✗"
    print(f"  {'|Cd| mean (expect > 0)':<45} {abs(cd_mean):8.4f}  {drag_mark}")
    print(f"  {'|Cs|/|Cd| (expect < 0.10)':<45} {cs_ratio:8.4f}  {cs_mark}")
    print(f"  {'|Cl|/|Cd| (expect < 0.25)':<45} {cl_ratio:8.4f}  {cl_mark}")
    _section("Wake post-processing")
    print(f"  {'Max wake velocity deficit':<45} {wake_deficit:8.4f}")
    print(
        f"  {'Recirculation length (lu)':<45}"
        f" {float(wake.get('recirculation_length_lu') or 0.0):8.4f}"
    )
    print("\n  References: Wigley (1926) hull parametrization; Michell (1898) thin-ship theory.")

    consistency_ok = bool(acceptance.get("workflow_ok", False))
    return {
        "name": "wigley_hull_re200",
        "cb_sim": cb_numerical,
        "cb_ref": cb_theoretical,
        "cb_error_pct": cb_error_pct,
        "cd_sim": float(cd_mean),
        "cs_ratio": cs_ratio,
        "cl_ratio": cl_ratio,
        "drag_positive": bool(acceptance.get("drag_positive")),
        "lift_small": bool(acceptance.get("lift_small")),
        "sideforce_small": bool(acceptance.get("sideforce_small")),
        "consistency_ok": consistency_ok,
        "elapsed_s": elapsed,
        "run_dir": str(run_dir),
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def bench_suboff_resistance(output_root: Path, full: bool) -> dict[str, object]:
    """Run SUBOFF drag benchmark and iteratively refine until error target is met."""
    from tensorlbm import SuboffResistanceBenchmarkConfig, run_suboff_resistance_benchmark

    cfg = SuboffResistanceBenchmarkConfig(
        hull_type="full",
        base_length_lu=64.0 if full else 48.0,
        max_iterations=4 if full else 3,
        target_error_pct=3.0,
        use_adaptive_mesh=True,
    )
    result = run_suboff_resistance_benchmark(cfg)
    final_error = float(result["final_error_pct"])

    _header("Benchmark 6 – SUBOFF Resistance (ITTC-1957 + Voxel Refinement)")
    print(f"  Hull type: {result['hull_type']}  Re = {float(result['reynolds']):.3e}")
    print(f"  Iterations: {len(result['iterations'])}  target: {result['target_error_pct']}%")
    cd_ref = float(result["reference"]["cd_richardson"] or result["reference"]["cd_analytical"])
    print(
        f"  {'Cd error vs reference':<45} {final_error:8.4f}%"
        f"  {'✓' if bool(result['target_met']) else '✗'}"
    )
    print(f"  {'Cd (sim / ref)':<45} {float(result['simulated']['cd']):8.4f} / {cd_ref:8.4f}")
    mesh = result.get("adaptive_mesh", {})
    if isinstance(mesh, dict) and bool(mesh.get("enabled", False)):
        print(
            f"  {'AMR cell saving vs full 4x uniform':<45}"
            f" {float(mesh.get('cell_saving_pct_mean', 0.0)):8.2f}%"
        )
    return result


def bench_marine_geometry_library(output_root: Path, full: bool) -> dict[str, object]:
    """Check ship/suboff CAD generators and coefficient consistency."""
    from tensorlbm import (
        ShipHullType,
        SuboffHullType,
        build_ship_hull_mask,
        build_suboff_mask,
    )

    # output_root kept for a uniform benchmark signature.
    _ = output_root
    nx, ny, nz = (128, 64, 48) if full else (80, 40, 30)
    ship_results: list[dict[str, object]] = []
    for hull_type in (ShipHullType.WIGLEY, ShipHullType.SERIES60, ShipHullType.KCS):
        _mask, stats = build_ship_hull_mask(
            hull_type=hull_type,
            nx=nx,
            ny=ny,
            nz=nz,
            length=nx * 0.5,
            beam=ny * 0.22,
            draft=nz * 0.25,
            device="cpu",
        )
        cb_sim = float(stats["Cb_numerical"])
        cb_ref = float(stats["Cb"])
        cb_err = abs(cb_sim - cb_ref) / (abs(cb_ref) + 1e-12) * 100.0
        ship_results.append({
            "hull_type": hull_type.value,
            "cb_sim": cb_sim,
            "cb_ref": cb_ref,
            "cb_error_pct": cb_err,
            "pass": cb_err < 35.0,
        })

    suboff_results: list[dict[str, object]] = []
    solid_cells: list[int] = []
    for hull_type in (
        SuboffHullType.BARE_HULL,
        SuboffHullType.WITH_SAIL,
        SuboffHullType.FULL,
    ):
        _mask, stats = build_suboff_mask(
            hull_type=hull_type,
            nx=nx,
            ny=ny,
            nz=nz,
            length=nx * 0.6,
            device="cpu",
        )
        solid = int(stats["solid_cells"])
        solid_cells.append(solid)
        suboff_results.append({
            "hull_type": hull_type.value,
            "solid_cells": solid,
            "l_d_ratio": float(stats["L_D_ratio"]),
        })

    cb_sim_values = [float(item["cb_sim"]) for item in ship_results]
    cb_order_ok = cb_sim_values[0] < cb_sim_values[1] < cb_sim_values[2]
    ship_ok = all(bool(item["pass"]) for item in ship_results) and cb_order_ok
    suboff_ok = solid_cells == sorted(solid_cells) and len(set(solid_cells)) == len(solid_cells)
    all_ok = ship_ok and suboff_ok

    _header("Benchmark 7 – Marine Geometry Library (Ship + SUBOFF)")
    print(f"  Grid: {nx}×{ny}×{nz}")
    _section("Ship hull block-coefficient consistency")
    for item in ship_results:
        _row(
            f"{item['hull_type']} Cb",
            float(item["cb_sim"]),
            float(item["cb_ref"]),
        )
    _section("SUBOFF variant volume ordering")
    print(
        f"  {'Solid-cell monotonicity (bare < sail < full)':<45}"
        f" {'✓' if suboff_ok else '✗'}"
    )
    print(
        f"  {'Cb ordering (wigley < series60 < kcs)':<45}"
        f" {'✓' if cb_order_ok else '✗'}"
    )
    print(
        f"  {'Solid cells [bare, sail, full]':<45}"
        f" {solid_cells[0]}, {solid_cells[1]}, {solid_cells[2]}"
    )

    return {
        "name": "marine_geometry_library",
        "ship": ship_results,
        "suboff": suboff_results,
        "ship_ok": ship_ok,
        "cb_order_ok": cb_order_ok,
        "suboff_ok": suboff_ok,
        "all_ok": all_ok,
    }


def _print_summary(results: list[dict[str, object]]) -> None:
    _header("Benchmark Summary")
    print(f"  {'Case':<40} {'Key metric':<20} {'Sim':>8} {'Ref':>8} {'Err%':>7} {'Pass'}")
    print("  " + "-" * 90)

    def _pass(err_pct: float) -> str:
        return "✓" if err_pct < 20.0 else "✗"

    for r in results:
        name = r.get("name", "?")
        if name == "cylinder_re100":
            st_s = float(r["st_sim"])
            st_r = float(r["st_ref"])
            err = abs(st_s - st_r) / st_r * 100.0
            print(
                f"  {'Cylinder Re=100':<40} {'St number':<20}"
                f" {st_s:8.4f} {st_r:8.4f} {err:7.2f}% {_pass(err)}"
            )
        elif name == "sloshing_tank":
            rel = r.get("relative_error")
            if rel is not None:
                err_pct = float(rel) * 100.0
                print(
                    f"  {'Sloshing tank':<40} {'ω error %':<20}"
                    f" {err_pct:8.2f} {'0.00':>8} {err_pct:7.2f}% {_pass(err_pct)}"
                )
            else:
                print(f"  {'Sloshing tank':<40} {'ω error %':<20} {'N/A':>8}")
        elif name == "pipeline_eD05":
            st_s = float(r["st_sim"])
            st_r = float(r["st_ref"])
            err = abs(st_s - st_r) / st_r * 100.0
            print(
                f"  {'Pipeline Re=200 e/D=0.5':<40} {'St number':<20}"
                f" {st_s:8.4f} {st_r:8.4f} {err:7.2f}% {_pass(err)}"
            )
        elif name == "turbulent_channel_retau100":
            rms = r.get("rms_loglaw_err")
            if rms is not None:
                mark = "✓" if float(rms) < 3.0 else "✗"
                print(
                    f"  {'Turbulent channel Re_τ=100':<40} {'RMS log-law err':<20}"
                    f" {float(rms):8.4f} {'<3.0':>8} {'':>7}  {mark}"
                )
            else:
                print(f"  {'Turbulent channel Re_τ=100':<40} {'RMS log-law err':<20} {'N/A':>8}")
        elif name == "wigley_hull_re200":
            cb_s = float(r["cb_sim"])
            cb_r = float(r["cb_ref"])
            err = abs(cb_s - cb_r) / cb_r * 100.0
            checks = "✓" if bool(r.get("consistency_ok", False)) else "✗"
            print(
                f"  {'Wigley hull Re=200':<40} {'Cb error + symmetry':<20}"
                f" {cb_s:8.4f} {cb_r:8.4f} {err:7.2f}%  {checks}"
            )
        elif name == "suboff_resistance":
            err = float(r.get("final_error_pct") or 0.0)
            mark = "✓" if bool(r.get("target_met", False)) else "✗"
            print(
                f"  {'SUBOFF full appendage':<40} {'Cd error vs ref':<20}"
                f" {err:8.4f} {'<3.0%':>8} {'':>7}  {mark}"
            )
        elif name == "marine_geometry_library":
            mark = "✓" if bool(r.get("all_ok", False)) else "✗"
            print(
                f"  {'Marine geometry library':<40} {'CAD consistency':<20}"
                f" {str(r.get('ship_ok')):>8} {str(r.get('suboff_ok')):>8} {'':>7}  {mark}"
            )

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TensorLBM marine / ship & ocean engineering benchmark suite"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Use production-quality grid sizes and step counts (slower)",
    )
    parser.add_argument(
        "--output-root", default="outputs/benchmarks/marine",
        help="Root directory for benchmark outputs",
    )
    parser.add_argument(
        "--cases", nargs="+",
        choices=[
            "cylinder",
            "sloshing",
            "pipeline",
            "channel",
            "hull",
            "suboff",
            "geometry",
            "all",
        ],
        default=["all"],
        help="Which benchmarks to run (default: all)",
    )
    parser.add_argument(
        "--report", default=None,
        help="Write a JSON summary to this file",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_all = "all" in args.cases
    cases = set(args.cases)

    print("\nTensorLBM – Marine/Ship & Ocean Engineering Benchmark Suite")
    print(f"Mode: {'full' if args.full else 'quick (reduced)'}   Output: {output_root}")

    results: list[dict[str, object]] = []

    if run_all or "cylinder" in cases:
        results.append(bench_cylinder(output_root, args.full))

    if run_all or "sloshing" in cases:
        results.append(bench_sloshing(output_root, args.full))

    if run_all or "pipeline" in cases:
        results.append(bench_pipeline(output_root, args.full))

    if run_all or "channel" in cases:
        results.append(bench_turbulent_channel(output_root, args.full))

    if run_all or "hull" in cases:
        results.append(bench_ship_hull(output_root, args.full))

    if run_all or "suboff" in cases:
        results.append(bench_suboff_resistance(output_root, args.full))

    if run_all or "geometry" in cases:
        results.append(bench_marine_geometry_library(output_root, args.full))

    _print_summary(results)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"  JSON report saved to: {report_path}")

    # Exit with non-zero if any hard failure (only strict physics violations)
    failed = False
    for r in results:
        name = r.get("name", "")
        if name == "wigley_hull_re200" and not bool(r.get("consistency_ok", True)):
            cd_val = r.get("cd_sim", "?")
            print(
                f"  FAIL: hull physical consistency check failed (Cd={cd_val})",
                file=sys.stderr,
            )
            failed = True
        if name == "suboff_resistance" and not bool(r.get("target_met", False)):
            err = r.get("final_error_pct", "?")
            print(
                f"  FAIL: SUBOFF resistance error target not met (error={err}%)",
                file=sys.stderr,
            )
            failed = True
        if name == "marine_geometry_library" and not bool(r.get("all_ok", False)):
            print("  FAIL: marine geometry library consistency check failed", file=sys.stderr)
            failed = True

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
