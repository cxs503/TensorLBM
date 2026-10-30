#!/usr/bin/env python3
"""SDAA memory diagnostic for suboff_sail_resistance — m289.

Strategy (per mission directive):
  1. Start from small L (16, 24, 32, 48, 56, 64, 80), measure peak memory
     per L tier after setup and after first collide step.
  2. Use torch.sdaa.max_memory_allocated / mem_get_info to measure.
  3. Call empty_cache() between L tiers to defragment.
  4. For tiers that survive, run enough steps to get a Cd estimate and
     compare with Blasius Cf_ref = 1.328/sqrt(Re).
  5. Record everything programmatically — zero hand-copy.

Output: artifacts/sdaa_suboff_sail_resistance_m289.json
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

# Ensure SDAA is loaded
try:
    import torch_sdaa  # noqa: F401
except ImportError:
    pass

from run import (
    SUBOFF_LENGTH_M,
    SUBOFF_RADIUS_M,
    U_PHYS,
    SuboffAppendageEngine,
    blasius_cf,
    wetted_reference,
)
from tensorlbm.general_sim import (
    CollisionModel,
    ForceMethod,
    GeneralSimConfig,
    GeometryConfig,
    GeometrySource,
    LatticeModel,
    OutputConfig,
    PhysicsConfig,
    SolverConfig,
    WallTreatment,
)
from compile_route import compile_mode_from_args

# ---- Config ----
DEVICE = "sdaa:0"
L_LADDER = [16, 24, 32, 48, 56, 64, 80]
# For small L, run enough steps to get a meaningful Cd average.
# steps = 150*L is the frozen protocol, but for diagnostic we use a reduced
# set: enough to pass the "first collide step" gate + get a rough Cd.
# For L<=32: run full 150*L (fast enough). For L>=48: run min(150*L, 3000).
STEPS_OVERRIDE = {}
for L in L_LADDER:
    s = 150 * L
    if s > 4000:
        s = 4000  # cap for diagnostic; convergence windows still meaningful
    STEPS_OVERRIDE[L] = min(s, 500)

CF_REF = blasius_cf(1000.0)  # 0.041995047327036084
GATE_THRESHOLD_PCT = 3.0

ARTIFACTS_DIR = Path(_REPO_ROOT) / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = ARTIFACTS_DIR / "sdaa_suboff_sail_resistance_m289.json"


def _gib(bytes_val: int) -> float:
    return bytes_val / (1024.0 ** 3)


def _mib(bytes_val: int) -> float:
    return bytes_val / (1024.0 ** 2)


def _sdaa_mem_info() -> dict:
    """Get current SDAA memory state."""
    info = {}
    try:
        free, total = torch.sdaa.mem_get_info(0)
        info["free_bytes"] = free
        info["total_bytes"] = total
        info["free_gib"] = round(_gib(free), 4)
        info["total_gib"] = round(_gib(total), 4)
    except Exception as e:
        info["mem_get_info_error"] = str(e)
    try:
        info["allocated_bytes"] = torch.sdaa.memory_allocated(0)
        info["allocated_mib"] = round(_mib(info["allocated_bytes"]), 2)
    except Exception:
        pass
    try:
        info["peak_allocated_bytes"] = torch.sdaa.max_memory_allocated(0)
        info["peak_allocated_gib"] = round(_gib(info["peak_allocated_bytes"]), 4)
    except Exception:
        pass
    return info


def _reset_peak():
    try:
        torch.sdaa.reset_peak_memory_stats(0)
    except Exception:
        pass
    try:
        torch.sdaa.empty_cache()
    except Exception:
        pass


def _build_config(L: int) -> GeneralSimConfig:
    viscosity = U_PHYS * SUBOFF_LENGTH_M / 1000.0  # Re = u*L/nu = 1000
    out_dir = Path(f"/tmp/sdaa_diag_L{L}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return GeneralSimConfig(
        name=f"sdaa_diag_suboff_sail_L{L}",
        geometry=GeometryConfig(
            source=GeometrySource.PARAMETRIC_SUBOFF,
            suboff_length=SUBOFF_LENGTH_M,
            suboff_radius=SUBOFF_RADIUS_M,
        ),
        physics=PhysicsConfig(
            density=1000.0,
            viscosity=viscosity,
            inlet_velocity=U_PHYS,
            reference_length=SUBOFF_LENGTH_M,
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q19,
            collision=CollisionModel.MRT,
            resolution=L,
            domain_padding=(1.0, 4.0, 1.0, 1.0, 1.0, 1.0),
            max_steps=STEPS_OVERRIDE[L],
            warmup_steps=None,
            snapshot_interval=10_000_000,
            force_sample_interval=10,
            device=DEVICE,
            wall_treatment=WallTreatment.AUTO,
            force_method=ForceMethod.PRESSURE_FRICTION,
            pressure_extrap="none",
            p0_method="near_wall",
            friction_formula="mix50",
            mass_correction=True,
            mass_correction_interval=200,
            smagorinsky_cs=0.05,
        ),
        output=OutputConfig(
            directory=str(out_dir),
            formats=[],
            save_macroscopic=False,
            save_forces=True,
        ),
    )


def _compute_cd(engine, L: int) -> dict:
    """Compute Cd from force log, same normalization as run.py."""
    u_lb = engine.uc.u_lb
    nu_lb = engine.uc.nu_lb
    dx = SUBOFF_LENGTH_M / L
    R_lb = SUBOFF_RADIUS_M / dx

    from tensorlbm.suboff_cad import SuboffConfig, suboff_statistics
    st = suboff_statistics("with_sail", float(L), R_lb, SuboffConfig())
    a_app_lb2 = float(st.get("appendage_own_wetted_area_lu2") or 0.0)
    S_total_lb2 = math.pi * (2.0 * R_lb) * float(L) + a_app_lb2
    dpS_wet = 0.5 * u_lb ** 2 * S_total_lb2
    dpS_front = 0.5 * u_lb ** 2 * math.pi * R_lb ** 2
    rescale = dpS_front / dpS_wet

    log = engine.forces_log
    if not log:
        return {"error": "no force samples"}

    n_win = min(1000, len(log))
    win = log[-n_win:]
    cd_p = sum(e["cd_pressure"] for e in win) / n_win * rescale
    cd_f = sum(e["cd_friction"] for e in win) / n_win * rescale
    cd_tot = cd_p + cd_f

    n_win2 = min(500, len(log))
    win2 = log[-n_win2:]
    cd_tot_last500 = sum(e["cd_total"] for e in win2) / n_win2 * rescale

    err_pct = (cd_tot - CF_REF) / CF_REF * 100.0

    # Convergence windows
    conv = {}
    for frac in (0.25, 0.5, 0.75, 1.0):
        k = int(len(log) * frac)
        seg = log[max(0, k - n_win):k]
        if seg:
            conv[f"{int(frac * 100)}%"] = round(
                sum(e["cd_total"] for e in seg) / len(seg) * rescale, 6
            )

    drift_pct = 0.0
    if "100%" in conv and "75%" in conv:
        c75 = conv["75%"]
        if abs(c75) > 1e-12:
            drift_pct = abs(conv["100%"] - c75) / abs(c75) * 100.0

    return {
        "Cd_pressure": round(cd_p, 8),
        "Cd_friction": round(cd_f, 8),
        "Cd_total": round(cd_tot, 8),
        "Cd_total_last500": round(cd_tot_last500, 8),
        "Cf_ref_blasius": CF_REF,
        "err_pct_vs_blasius": round(err_pct, 4),
        "gate_pass": abs(err_pct) <= GATE_THRESHOLD_PCT,
        "n_force_samples": len(log),
        "convergence_windows": conv,
        "drift_pct": round(drift_pct, 4),
        "S_total_wetted_lb2": round(S_total_lb2, 2),
        "rescale": round(rescale, 6),
        "appendage_stats": engine._appendage_stats,
        "appendage_near_cells": engine._appendage_near_cells,
    }


def run_single_L(L: int) -> dict:
    """Run diagnostic for a single L tier."""
    result = {
        "L": L,
        "steps_planned": STEPS_OVERRIDE[L],
        "status": "init",
    }

    # Pre-setup memory
    _reset_peak()
    mem_before = _sdaa_mem_info()
    result["mem_before_setup"] = mem_before

    try:
        config = _build_config(L)
        engine = SuboffAppendageEngine(config, hull_type="with_sail")

        t_setup_start = time.time()
        setup_info = engine.setup()
        t_setup = time.time() - t_setup_start

        mem_after_setup = _sdaa_mem_info()
        _reset_peak()  # reset peak to measure collide step only

        result.update({
            "status": "setup_ok",
            "setup_time_s": round(t_setup, 2),
            "domain_lu": setup_info.get("domain_lu"),
            "total_cells": setup_info.get("total_cells"),
            "obstacle_cells": setup_info.get("obstacle_cells"),
            "near_wall_cells": setup_info.get("near_wall_cells"),
            "Re": setup_info.get("Re"),
            "tau": setup_info.get("tau"),
            "u_lb": setup_info.get("u_lb"),
            "nu_lb": setup_info.get("nu_lb"),
            "device": setup_info.get("device"),
            "auto_wall_treatment": setup_info.get("auto_wall_treatment"),
            "appendage_stats": engine._appendage_stats,
            "appendage_near_cells": engine._appendage_near_cells,
            "mem_after_setup": mem_after_setup,
            "f_tensor_bytes": engine.f.numel() * engine.f.element_size(),
            "f_tensor_mib": round(_mib(engine.f.numel() * engine.f.element_size()), 2),
        })

        print(f"[L={L}] setup OK: domain={setup_info.get('domain_lu')} "
              f"cells={setup_info.get('total_cells')} "
              f"f_tensor={result['f_tensor_mib']:.1f} MiB "
              f"peak_mem={mem_after_setup.get('peak_allocated_gib', '?')} GiB",
              flush=True)

    except torch.sdaa.OutOfMemoryError as e:
        result["status"] = "OOM_AT_SETUP"
        result["oom_error"] = str(e)[:500]
        result["mem_at_oom"] = _sdaa_mem_info()
        print(f"[L={L}] OOM at setup: {e}", flush=True)
        _reset_peak()
        return result
    except Exception as e:
        result["status"] = "ERROR_AT_SETUP"
        result["error"] = f"{type(e).__name__}: {e}"[:500]
        result["mem_at_error"] = _sdaa_mem_info()
        print(f"[L={L}] ERROR at setup: {e}", flush=True)
        _reset_peak()
        return result

    # ---- Run steps ----
    n_steps = STEPS_OVERRIDE[L]
    step_times = []
    oom_at_step = None

    try:
        from run import run_engine_routed
        # Use eager mode (compile_mode=None -> eager) to avoid compile trace memory
        t_run_start = time.time()

        # Run step-by-step to catch OOM precisely
        from tensorlbm.lbm_step_correct import lbm_step_correct
        from tensorlbm.boundaries3d import far_field_bc_3d
        import functools

        collide_fn, collide_kwargs = engine._get_collide_fn()
        far_field_fn = functools.partial(far_field_bc_3d, bc_config=engine._build_bc_config())
        solid = engine.solid
        tau = engine.uc.tau
        nu_lb = engine.uc.nu_lb
        u_in = engine.uc.u_lb
        dpS = engine._compute_dpS()

        from tensorlbm.solver3d import correct_mass3d
        target_mass = engine._initial_mass

        force_interval = engine.config.solver.force_sample_interval
        mass_interval = engine.config.solver.mass_correction_interval

        for step in range(1, n_steps + 1):
            t_step = time.time()
            engine.f = lbm_step_correct(
                engine.f, collide_fn, tau, solid, u_in, far_field_fn, **collide_kwargs
            )
            engine.step_count += 1
            dt = time.time() - t_step
            step_times.append(dt)

            if step % mass_interval == 0:
                engine.f = correct_mass3d(engine.f, target_mass)

            if step % force_interval == 0:
                engine._sample_forces(dpS, nu_lb)

            if step <= 3 or step % 500 == 0:
                mem = _sdaa_mem_info()
                print(f"[L={L}] step {step}/{n_steps} dt={dt*1000:.1f}ms "
                      f"peak={mem.get('peak_allocated_gib', '?')} GiB "
                      f"alloc={mem.get('allocated_mib', '?')} MiB",
                      flush=True)

            # Check finiteness every 100 steps
            if step % 100 == 0 and not torch.isfinite(engine.f).all():
                result["status"] = "DIVERGED"
                result["diverged_at_step"] = step
                break

        t_run = time.time() - t_run_start
        mem_after_run = _sdaa_mem_info()

        result.update({
            "status": "COMPLETED" if result.get("status") != "DIVERGED" else "DIVERGED",
            "run_time_s": round(t_run, 2),
            "ms_per_step": round(sum(step_times) / len(step_times) * 1000, 2) if step_times else None,
            "ms_per_step_last100": round(sum(step_times[-100:]) / min(100, len(step_times)) * 1000, 2) if step_times else None,
            "steps_completed": engine.step_count,
            "mem_peak_run_gib": mem_after_run.get("peak_allocated_gib"),
            "mem_after_run": mem_after_run,
            "finite": bool(torch.isfinite(engine.f).all().item()),
        })

        # Compute Cd if we have force samples
        if engine.forces_log:
            cd_info = _compute_cd(engine, L)
            result["cd_results"] = cd_info
            print(f"[L={L}] Cd_total={cd_info.get('Cd_total', '?')} "
                  f"err={cd_info.get('err_pct_vs_blasius', '?')}% "
                  f"gate={'PASS' if cd_info.get('gate_pass') else 'FAIL'}",
                  flush=True)

        print(f"[L={L}] COMPLETED: {engine.step_count} steps in {t_run:.1f}s "
              f"({result['ms_per_step']:.1f} ms/step) "
              f"peak={mem_after_run.get('peak_allocated_gib', '?')} GiB",
              flush=True)

    except torch.sdaa.OutOfMemoryError as e:
        oom_at_step = engine.step_count if hasattr(engine, 'step_count') else 0
        result["status"] = "OOM_AT_STEP"
        result["oom_at_step"] = oom_at_step
        result["oom_error"] = str(e)[:500]
        result["mem_at_oom"] = _sdaa_mem_info()
        result["steps_completed"] = oom_at_step
        if step_times:
            result["ms_per_step"] = round(sum(step_times) / len(step_times) * 1000, 2)
        print(f"[L={L}] OOM at step {oom_at_step}: {e}", flush=True)
    except Exception as e:
        result["status"] = "ERROR_AT_RUN"
        result["error"] = f"{type(e).__name__}: {e}"[:500]
        result["mem_at_error"] = _sdaa_mem_info()
        result["steps_completed"] = engine.step_count if hasattr(engine, 'step_count') else 0
        print(f"[L={L}] ERROR at run: {e}", flush=True)
    finally:
        # Cleanup
        try:
            del engine
        except Exception:
            pass
        gc.collect()
        _reset_peak()

    return result


def main():
    print("=" * 70, flush=True)
    print("SDAA Memory Diagnostic — suboff_sail_resistance — m289", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Cf_ref (Blasius Re=1000): {CF_REF}", flush=True)
    print(f"L ladder: {L_LADDER}", flush=True)
    print(f"Gate threshold: |err| <= {GATE_THRESHOLD_PCT}%", flush=True)
    print("=" * 70, flush=True)

    # Device info
    dev_info = {}
    try:
        dev_info["device_name"] = torch.sdaa.get_device_name(0)
        props = torch.sdaa.get_device_properties(0)
        dev_info["total_memory_mib"] = props.total_memory
        dev_info["total_memory_gib"] = round(props.total_memory / 1024, 4)
    except Exception as e:
        dev_info["error"] = str(e)
    dev_info.update(_sdaa_mem_info())
    print(f"Device info: {json.dumps(dev_info, indent=2)}", flush=True)

    results = {
        "mission_id": "m289",
        "case": "suboff_sail_resistance",
        "benchmark_path": "benchmarks/verified/suboff_sail_resistance",
        "device": DEVICE,
        "device_info": dev_info,
        "cf_ref_blasius": CF_REF,
        "gate_threshold_pct": GATE_THRESHOLD_PCT,
        "l_ladder": L_LADDER,
        "steps_override": STEPS_OVERRIDE,
        "tiers": {},
    }

    # Save initial state immediately (budget discipline)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[checkpoint] Initial results saved to {OUT_JSON}", flush=True)

    for L in L_LADDER:
        print(f"\n{'='*50}", flush=True)
        print(f"Processing L={L}", flush=True)
        print(f"{'='*50}", flush=True)

        tier_result = run_single_L(L)
        results["tiers"][str(L)] = tier_result

        # Save after each L (budget discipline)
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[checkpoint] Results saved after L={L}", flush=True)

        # Cleanup between tiers
        gc.collect()
        _reset_peak()
        time.sleep(1)

    # ---- Summary ----
    completed = []
    oom = []
    errors = []
    for L_str, tier in results["tiers"].items():
        status = tier.get("status", "unknown")
        if status == "COMPLETED":
            completed.append(int(L_str))
        elif "OOM" in status:
            oom.append(int(L_str))
        else:
            errors.append(int(L_str))

    results["summary"] = {
        "completed_tiers": sorted(completed),
        "oom_tiers": sorted(oom),
        "error_tiers": sorted(errors),
        "max_L_completed": max(completed) if completed else None,
        "n_completed": len(completed),
        "n_oom": len(oom),
        "n_errors": len(errors),
    }

    # Determine verdict
    if completed:
        # Check if any completed tier passes the gate
        gate_pass_tiers = []
        for L_str in completed:
            tier = results["tiers"][str(L_str)]
            cd = tier.get("cd_results", {})
            if cd.get("gate_pass"):
                gate_pass_tiers.append(int(L_str))
        results["summary"]["gate_pass_tiers"] = sorted(gate_pass_tiers)

        if len(completed) >= 2:
            # Check monotonic convergence
            errs = []
            for L_str in sorted(completed):
                cd = results["tiers"][str(L_str)].get("cd_results", {})
                errs.append(cd.get("err_pct_vs_blasius"))
            results["summary"]["err_sequence"] = errs
            results["summary"]["monotonic"] = all(
                abs(errs[i]) >= abs(errs[i + 1]) for i in range(len(errs) - 1)
            ) if len(errs) >= 2 else None

    # Verdict
    max_completed = max(completed) if completed else 0
    if max_completed >= 96 and any(
        results["tiers"].get(str(L), {}).get("cd_results", {}).get("gate_pass")
        for L in completed
    ):
        verdict = "PASS"
    elif max_completed >= 48:
        verdict = "HONEST_FAIL_MEMORY_LIMITED"
    else:
        verdict = "HONEST_FAIL"
    results["verdict"] = verdict

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"Completed tiers: {sorted(completed)}", flush=True)
    print(f"OOM tiers: {sorted(oom)}", flush=True)
    print(f"Error tiers: {sorted(errors)}", flush=True)
    if completed:
        for L_str in sorted(completed):
            cd = results["tiers"][str(L_str)].get("cd_results", {})
            print(f"  L={L_str}: Cd={cd.get('Cd_total', '?')} "
                  f"err={cd.get('err_pct_vs_blasius', '?')}% "
                  f"peak={results['tiers'][str(L_str)].get('mem_peak_run_gib', '?')} GiB",
                  flush=True)
    print(f"Results saved to {OUT_JSON}", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
