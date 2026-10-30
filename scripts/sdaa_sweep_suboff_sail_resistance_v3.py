#!/usr/bin/env python3
"""SDAA sweep v3 — suboff_sail_resistance diagnostic + workaround.

Mission m109 / round 3: the frozen ladder L∈{90,100,110} all OOM on the
14.94 GiB SDAA card. This script:
  1) Probes memory at L∈{48,56,64,72,80} to find the max feasible resolution
  2) Runs the highest feasible resolution with reduced steps (if needed)
  3) Records Cd_total vs Blasius reference and convergence windows
  4) Writes artifacts/sdaa_suboff_sail_resistance_v3.json
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CASE_DIR = _REPO_ROOT / "benchmarks" / "verified" / "suboff_sail_resistance"

sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "benchmarks"))
sys.path.insert(0, str(_CASE_DIR))

import torch
import torch_sdaa

# Now import from the case's run.py
from run import (
    SuboffAppendageEngine,
    run_engine_routed,
    SUBOFF_LENGTH_M,
    SUBOFF_RADIUS_M,
    U_PHYS,
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

BLASIUS_CF_REF = 1.328 / math.sqrt(1000.0)
DEVICE = "sdaa:0"


def get_memory_info():
    free, total = torch.sdaa.mem_get_info(0)
    return {"free_gib": round(free / 1024**3, 2), "total_gib": round(total / 1024**3, 2),
            "used_gib": round((total - free) / 1024**3, 2)}


def make_config(L, steps, out_dir, device=DEVICE):
    viscosity = U_PHYS * SUBOFF_LENGTH_M / 1000.0
    return GeneralSimConfig(
        name=f"sdaa_v3_L{L}_s{steps}",
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
            max_steps=steps,
            warmup_steps=None,
            snapshot_interval=10_000_000,
            force_sample_interval=10,
            device=device,
            wall_treatment=WallTreatment.AUTO,
            force_method=ForceMethod.PRESSURE_FRICTION,
            pressure_extrap="none",
            p0_method="near_wall",
            friction_formula="standard",
            mass_correction=True,
            mass_correction_interval=200,
        ),
        output=OutputConfig(
            directory=str(out_dir),
            formats=[],
            save_macroscopic=False,
            save_forces=True,
        ),
    )


def run_one(L, steps, out_dir):
    """Run simulation and return result dict."""
    torch.sdaa.empty_cache()
    torch.sdaa.reset_peak_memory_stats(0)

    config = make_config(L, steps, out_dir)
    t0 = time.time()

    try:
        engine = SuboffAppendageEngine(config, hull_type="with_sail")
        setup_info = engine.setup()
        t_setup = time.time() - t0
        peak_setup = torch.sdaa.max_memory_allocated(0) / 1024**3

        print(f"    setup: domain={setup_info.get('domain_lu')}, "
              f"cells={setup_info.get('total_cells', 0):,}, "
              f"u_lb={setup_info.get('u_lb')}, tau={setup_info.get('tau'):.4f}")

        # Run using the case's run_engine_routed
        run_info, elapsed = run_engine_routed(engine, compile_mode=None)
        t_run = time.time() - t0 - t_setup
        peak_run = torch.sdaa.max_memory_allocated(0) / 1024**3

        # Post-process
        u_lb = engine.uc.u_lb
        dx = SUBOFF_LENGTH_M / L
        R_lb = SUBOFF_RADIUS_M / dx

        from tensorlbm.suboff_cad import SuboffConfig, suboff_statistics
        st = suboff_statistics("with_sail", float(L), R_lb, SuboffConfig())
        a_app = float(st.get("appendage_own_wetted_area_lu2") or 0.0)
        S_total = math.pi * (2.0 * R_lb) * float(L) + a_app
        dpS_wet = 0.5 * u_lb**2 * S_total
        dpS_front = 0.5 * u_lb**2 * math.pi * R_lb**2
        rescale = dpS_front / dpS_wet

        log = engine.forces_log
        n_total = len(log)

        windows = {}
        for pct in [25, 50, 75, 100]:
            n_win = min(1000, max(1, int(n_total * pct / 100)))
            win = log[-n_win:]
            if win:
                cd_p = sum(e["cd_pressure"] for e in win) / len(win) * rescale
                cd_f = sum(e["cd_friction"] for e in win) / len(win) * rescale
                windows[f"{pct}%"] = round(cd_p + cd_f, 6)

        n_final = min(1000, n_total)
        win_final = log[-n_final:]
        cd_p = sum(e["cd_pressure"] for e in win_final) / n_final * rescale
        cd_f = sum(e["cd_friction"] for e in win_final) / n_final * rescale
        cd_tot = cd_p + cd_f
        err_pct = (cd_tot / BLASIUS_CF_REF - 1.0) * 100.0

        # Window drift
        half = n_final // 2
        first_half = log[-n_final:-half]
        second_half = log[-half:]
        cd1 = sum(e["cd_total"] for e in first_half) / len(first_half) * rescale if first_half else cd_tot
        cd2 = sum(e["cd_total"] for e in second_half) / len(second_half) * rescale if second_half else cd_tot
        drift_pct = abs(cd2 - cd1) / BLASIUS_CF_REF * 100.0 if BLASIUS_CF_REF > 0 else 0.0

        result = {
            "L": L, "steps_requested": steps, "steps_done": run_info.get("steps", 0),
            "status": "ok",
            "t_setup_s": round(t_setup, 1), "t_run_s": round(elapsed, 1),
            "ms_per_step": round(elapsed / max(run_info.get("steps", 1), 1) * 1000, 1),
            "mem_peak_gib": round(max(peak_setup, peak_run), 2),
            "domain": list(setup_info.get("domain_lu", [])),
            "total_cells": setup_info.get("total_cells", 0),
            "u_lb": setup_info.get("u_lb"),
            "tau": setup_info.get("tau"),
            "cd_pressure": round(cd_p, 10),
            "cd_friction": round(cd_f, 10),
            "cd_total_window": round(cd_tot, 10),
            "err_vs_blasius_pct": round(err_pct, 4),
            "blasius_cf_ref": BLASIUS_CF_REF,
            "convergence_windows": windows,
            "window_drift_pct": round(drift_pct, 4),
            "n_force_samples": n_total,
            "appendage_solid_cells": engine._appendage_stats.get("appendage_solid_cells", 0),
            "appendage_near_cells": engine._appendage_near_cells,
            "diverged": run_info.get("diverged", False),
        }
        del engine
        torch.sdaa.empty_cache()
        return result

    except Exception as e:
        err_str = str(e).lower()
        is_oom = "out of memory" in err_str or "oom" in err_str
        status = "oom" if is_oom else "error"
        elapsed = time.time() - t0
        print(f"    EXCEPTION ({status}): {str(e)[:200]}")
        traceback.print_exc()
        return {
            "L": L, "steps_requested": steps, "steps_done": 0,
            "status": status, "error": str(e)[:500],
            "wall_time_s": round(elapsed, 1),
        }


def main():
    print("=" * 70)
    print("SDAA Sweep v3 — suboff_sail_resistance diagnostic")
    print("=" * 70)

    mem = get_memory_info()
    print(f"\nHardware: SDAA total={mem['total_gib']} GiB, "
          f"free={mem['free_gib']} GiB, used={mem['used_gib']} GiB")

    # Phase 1: Memory probe — just setup, no stepping
    print("\n--- Phase 1: Memory probe (setup only) ---")
    probe_results = {}
    for L in [48, 56, 64, 72, 80, 90, 100, 110]:
        torch.sdaa.empty_cache()
        torch.sdaa.reset_peak_memory_stats(0)
        out_dir = Path(f"/tmp/sdaa_v3_probe_L{L}")
        out_dir.mkdir(parents=True, exist_ok=True)
        config = make_config(L, 5, out_dir)
        try:
            engine = SuboffAppendageEngine(config, hull_type="with_sail")
            info = engine.setup()
            peak = torch.sdaa.max_memory_allocated(0) / 1024**3
            probe_results[L] = {
                "status": "ok",
                "domain": list(info.get("domain_lu", [])),
                "total_cells": info.get("total_cells", 0),
                "peak_gib": round(peak, 2),
            }
            print(f"  L={L:3d}: domain={info.get('domain_lu')}, "
                  f"cells={info.get('total_cells', 0):,}, peak={peak:.2f} GiB")
            del engine
            torch.sdaa.empty_cache()
        except Exception as e:
            err_str = str(e).lower()
            is_oom = "out of memory" in err_str or "oom" in err_str
            probe_results[L] = {
                "status": "oom" if is_oom else "error",
                "error": str(e)[:200],
            }
            print(f"  L={L:3d}: {'OOM' if is_oom else 'ERROR'} — {str(e)[:100]}")
            torch.sdaa.empty_cache()

    feasible = [L for L, r in probe_results.items() if r["status"] == "ok"]
    print(f"\nFeasible resolutions: {feasible}")

    if not feasible:
        print("\n*** No resolution fits! Hardware bottleneck. ***")
        artifact = {
            "schema": "sdaa-sweep-v3", "mission": "m109", "round": 3,
            "overall": "HONEST-FAIL", "reason": "hardware_oom_all_resolutions",
            "hardware": mem,
            "probe_results": {str(L): r for L, r in probe_results.items()},
        }
        out_path = _REPO_ROOT / "artifacts" / "sdaa_suboff_sail_resistance_v3.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, indent=1))
        print(f"Artifact: {out_path}")
        return

    max_L = max(feasible)
    print(f"Max feasible resolution: L={max_L}")

    # Phase 2: Run at max feasible L
    print(f"\n--- Phase 2: Run at L={max_L} ---")
    run_results = {}
    for steps in [1500, 3000, 6000]:
        out_dir = _REPO_ROOT / "runs" / f"sdaa_v3_L{max_L}_s{steps}"
        print(f"\n  Running L={max_L}, steps={steps}...")
        res = run_one(max_L, steps, out_dir)
        run_results[steps] = res
        print(f"  -> status={res['status']}, steps_done={res.get('steps_done', 0)}, "
              f"time={res.get('t_run_s', res.get('wall_time_s', '?'))}s")
        if res["status"] == "ok":
            print(f"     Cd_total={res.get('cd_total_window')}, err={res.get('err_vs_blasius_pct')}%")
        elif res["status"] == "oom":
            print(f"     OOM! Stopping.")
            break
        else:
            print(f"     Error: {res.get('error', '?')[:100]}")
            break

    # Phase 3: Try frozen ladder L∈{90,100,110}
    frozen_feasible = [L for L in [90, 100, 110] if L in feasible]
    print(f"\n--- Phase 3: Frozen ladder (feasible: {frozen_feasible}) ---")
    frozen_results = {}
    for L in [90, 100, 110]:
        if L not in feasible:
            frozen_results[L] = {"status": "setup_oom", "note": "setup failed in Phase 1"}
            print(f"  L={L}: skipped (setup OOM in Phase 1)")
            continue
        out_dir = _REPO_ROOT / "runs" / f"sdaa_v3_L{L}_s3000"
        print(f"\n  Attempting L={L}, steps=3000...")
        res = run_one(L, 3000, out_dir)
        frozen_results[L] = res
        print(f"  -> status={res['status']}, time={res.get('t_run_s', res.get('wall_time_s', '?'))}s")
        if res["status"] == "ok":
            print(f"     Cd_total={res.get('cd_total_window')}, err={res.get('err_vs_blasius_pct')}%")

    # Build artifact
    artifact = {
        "schema": "sdaa-sweep-v3",
        "mission": "m109",
        "round": 3,
        "overall": "HONEST-FAIL",
        "hardware": mem,
        "probe_results": {str(L): r for L, r in probe_results.items()},
        "run_results": {str(s): r for s, r in run_results.items()},
        "frozen_ladder_results": {str(L): r for L, r in frozen_results.items()},
        "feasible_resolutions": feasible,
        "max_feasible_L": max_L,
        "blasius_cf_ref": BLASIUS_CF_REF,
        "diagnosis": {
            "frozen_ladder_oom": all(
                frozen_results.get(L, {}).get("status") in ("oom", "setup_oom", "error")
                for L in [90, 100, 110]
            ),
            "lower_res_feasible": len(feasible) > 0,
            "conclusion": "",
        },
    }

    # Determine conclusion
    frozen_ok = {L: r for L, r in frozen_results.items() if r.get("status") == "ok"}
    if frozen_ok:
        # Check criteria
        sorted_L = sorted(frozen_ok.keys())
        errs = [abs(frozen_ok[L].get("err_vs_blasius_pct", 999)) for L in sorted_L]
        g1 = errs[-1] <= 3.0 if errs else False
        g2 = all(errs[i] >= errs[i+1] for i in range(len(errs)-1)) if len(errs) > 1 else False
        if g1 and g2:
            artifact["overall"] = "PASS"
            artifact["diagnosis"]["conclusion"] = "Frozen ladder tiers completed and pass G1+G2."
        else:
            artifact["diagnosis"]["conclusion"] = (
                f"Frozen ladder tiers completed but fail criteria: "
                f"G1(finest |err|≤3%)={'PASS' if g1 else 'FAIL'}, "
                f"G2(|err| non-increasing)={'PASS' if g2 else 'FAIL'}."
            )
    else:
        ok_runs = {s: r for s, r in run_results.items() if r.get("status") == "ok"}
        if ok_runs:
            best_s = max(ok_runs.keys())
            best = ok_runs[best_s]
            artifact["best_lower_res_run"] = {
                "L": max_L, "steps": best_s,
                "Cd_total": best.get("cd_total_window"),
                "err_pct": best.get("err_vs_blasius_pct"),
                "drift_pct": best.get("window_drift_pct"),
            }
            artifact["diagnosis"]["conclusion"] = (
                f"Frozen ladder L∈{{90,100,110}} OOM on 14.94 GiB SDAA. "
                f"Lower resolution L={max_L} runs successfully with "
                f"Cd_total={best.get('cd_total_window')}, err={best.get('err_vs_blasius_pct')}%. "
                f"But this does not satisfy the frozen convergence ladder criteria."
            )
        else:
            artifact["diagnosis"]["conclusion"] = (
                "No resolution could complete a run. Hardware bottleneck confirmed."
            )

    out_path = _REPO_ROOT / "artifacts" / "sdaa_suboff_sail_resistance_v3.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=1))
    print(f"\n{'=' * 70}")
    print(f"Artifact: {out_path}")
    print(f"Overall: {artifact['overall']}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
