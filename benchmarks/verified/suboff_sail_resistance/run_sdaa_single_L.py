#!/usr/bin/env python3
"""Run a single L tier of the SDAA diagnostic."""

from __future__ import annotations
import gc, json, math, os, sys, time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
try:
    import torch_sdaa  # noqa: F401
except ImportError:
    pass

from run import (
    SUBOFF_LENGTH_M, SUBOFF_RADIUS_M, U_PHYS,
    SuboffAppendageEngine, blasius_cf, wetted_reference,
)
from tensorlbm.general_sim import (
    CollisionModel, ForceMethod, GeneralSimConfig, GeometryConfig,
    GeometrySource, LatticeModel, OutputConfig, PhysicsConfig,
    SolverConfig, WallTreatment,
)

L = int(sys.argv[1])
N_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 500
DEVICE = "sdaa:0"
CF_REF = blasius_cf(1000.0)

ARTIFACTS_DIR = Path(_REPO_ROOT) / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = ARTIFACTS_DIR / "sdaa_suboff_sail_resistance_m289.json"


def _gib(b): return b / (1024.0**3)
def _mib(b): return b / (1024.0**2)

def _mem():
    info = {}
    try:
        free, total = torch.sdaa.mem_get_info(0)
        info["free_gib"] = round(_gib(free), 4)
        info["total_gib"] = round(_gib(total), 4)
    except: pass
    try:
        info["allocated_mib"] = round(_mib(torch.sdaa.memory_allocated(0)), 2)
    except: pass
    try:
        info["peak_gib"] = round(_gib(torch.sdaa.max_memory_allocated(0)), 4)
    except: pass
    return info

def _reset():
    try: torch.sdaa.reset_peak_memory_stats(0)
    except: pass
    try: torch.sdaa.empty_cache()
    except: pass

def main():
    print(f"=== L={L} steps={N_STEPS} ===", flush=True)
    _reset()
    mem0 = _mem()
    
    viscosity = U_PHYS * SUBOFF_LENGTH_M / 1000.0
    out_dir = Path(f"/tmp/sdaa_diag_L{L}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    config = GeneralSimConfig(
        name=f"sdaa_diag_suboff_sail_L{L}",
        geometry=GeometryConfig(
            source=GeometrySource.PARAMETRIC_SUBOFF,
            suboff_length=SUBOFF_LENGTH_M,
            suboff_radius=SUBOFF_RADIUS_M,
        ),
        physics=PhysicsConfig(
            density=1000.0, viscosity=viscosity,
            inlet_velocity=U_PHYS, reference_length=SUBOFF_LENGTH_M,
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q19, collision=CollisionModel.MRT,
            resolution=L, domain_padding=(1.0, 4.0, 1.0, 1.0, 1.0, 1.0),
            max_steps=N_STEPS, warmup_steps=None,
            snapshot_interval=10_000_000, force_sample_interval=10,
            device=DEVICE, wall_treatment=WallTreatment.AUTO,
            force_method=ForceMethod.PRESSURE_FRICTION,
            pressure_extrap="none", p0_method="near_wall",
            friction_formula="mix50", mass_correction=True,
            mass_correction_interval=200, smagorinsky_cs=0.05,
        ),
        output=OutputConfig(
            directory=str(out_dir), formats=[],
            save_macroscopic=False, save_forces=True,
        ),
    )
    
    result = {"L": L, "steps_planned": N_STEPS, "mem_before": mem0}
    
    try:
        engine = SuboffAppendageEngine(config, hull_type="with_sail")
        t0 = time.time()
        setup_info = engine.setup()
        t_setup = time.time() - t0
        _reset()
        mem_setup = _mem()
        
        f_bytes = engine.f.numel() * engine.f.element_size()
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
            "mem_after_setup": mem_setup,
            "f_tensor_mib": round(_mib(f_bytes), 2),
            "bytes_per_cell_setup": round(f_bytes / max(1, setup_info.get("total_cells", 1)), 2),
        })
        print(f"[L={L}] setup OK: domain={setup_info.get('domain_lu')} "
              f"cells={setup_info.get('total_cells')} f={result['f_tensor_mib']:.1f}MiB "
              f"peak={mem_setup.get('peak_gib','?')}GiB", flush=True)
    except torch.sdaa.OutOfMemoryError as e:
        result["status"] = "OOM_AT_SETUP"
        result["oom_error"] = str(e)[:300]
        result["mem_at_oom"] = _mem()
        print(f"[L={L}] OOM at setup", flush=True)
        _save(result)
        return
    except Exception as e:
        result["status"] = "ERROR_AT_SETUP"
        result["error"] = f"{type(e).__name__}: {e}"[:300]
        result["mem_at_error"] = _mem()
        print(f"[L={L}] ERROR at setup: {e}", flush=True)
        _save(result)
        return
    
    # Run steps
    step_times = []
    try:
        from tensorlbm.lbm_step_correct import lbm_step_correct
        from tensorlbm.boundaries3d import far_field_bc_3d
        import functools
        from tensorlbm.solver3d import correct_mass3d
        
        collide_fn, collide_kwargs = engine._get_collide_fn()
        far_field_fn = functools.partial(far_field_bc_3d, bc_config=engine._build_bc_config())
        solid = engine.solid
        tau = engine.uc.tau
        nu_lb = engine.uc.nu_lb
        u_in = engine.uc.u_lb
        dpS = engine._compute_dpS()
        target_mass = engine._initial_mass
        force_interval = 10
        mass_interval = 200
        
        t_run0 = time.time()
        for step in range(1, N_STEPS + 1):
            ts = time.time()
            engine.f = lbm_step_correct(
                engine.f, collide_fn, tau, solid, u_in, far_field_fn, **collide_kwargs
            )
            engine.step_count += 1
            dt = time.time() - ts
            step_times.append(dt)
            
            if step % mass_interval == 0:
                engine.f = correct_mass3d(engine.f, target_mass)
            if step % force_interval == 0:
                engine._sample_forces(dpS, nu_lb)
            
            if step <= 3 or step % 100 == 0:
                m = _mem()
                print(f"[L={L}] step {step}/{N_STEPS} dt={dt*1000:.1f}ms "
                      f"peak={m.get('peak_gib','?')}GiB alloc={m.get('allocated_mib','?')}MiB",
                      flush=True)
            
            if step % 100 == 0 and not torch.isfinite(engine.f).all():
                result["status"] = "DIVERGED"
                result["diverged_at_step"] = step
                break
        
        t_run = time.time() - t_run0
        mem_run = _mem()
        
        result.update({
            "status": "COMPLETED" if result.get("status") != "DIVERGED" else "DIVERGED",
            "run_time_s": round(t_run, 2),
            "ms_per_step": round(sum(step_times)/len(step_times)*1000, 2) if step_times else None,
            "ms_per_step_last50": round(sum(step_times[-50:])/min(50,len(step_times))*1000, 2) if step_times else None,
            "steps_completed": engine.step_count,
            "mem_peak_run_gib": mem_run.get("peak_gib"),
            "mem_after_run": mem_run,
            "finite": bool(torch.isfinite(engine.f).all().item()),
            "bytes_per_cell_peak": round(mem_run.get("peak_gib",0)*1024**3 / max(1, setup_info.get("total_cells",1)), 2),
        })
        
        # Compute Cd
        if engine.forces_log:
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
            n_win = min(200, len(log))
            win = log[-n_win:]
            cd_p = sum(e["cd_pressure"] for e in win) / n_win * rescale
            cd_f = sum(e["cd_friction"] for e in win) / n_win * rescale
            cd_tot = cd_p + cd_f
            err = (cd_tot - CF_REF) / CF_REF * 100.0
            
            conv = {}
            for frac in (0.25, 0.5, 0.75, 1.0):
                k = int(len(log) * frac)
                seg = log[max(0, k-n_win):k]
                if seg:
                    conv[f"{int(frac*100)}%"] = round(sum(e["cd_total"] for e in seg)/len(seg)*rescale, 6)
            
            drift = 0.0
            if "100%" in conv and "75%" in conv and abs(conv["75%"]) > 1e-12:
                drift = abs(conv["100%"] - conv["75%"]) / abs(conv["75%"]) * 100.0
            
            result["cd_results"] = {
                "Cd_pressure": round(cd_p, 8),
                "Cd_friction": round(cd_f, 8),
                "Cd_total": round(cd_tot, 8),
                "Cf_ref_blasius": CF_REF,
                "err_pct_vs_blasius": round(err, 4),
                "gate_pass": abs(err) <= 3.0,
                "n_force_samples": len(log),
                "convergence_windows": conv,
                "drift_pct": round(drift, 4),
                "S_total_wetted_lb2": round(S_total, 2),
                "rescale": round(rescale, 6),
                "appendage_stats": engine._appendage_stats,
                "appendage_near_cells": engine._appendage_near_cells,
            }
            print(f"[L={L}] Cd={cd_tot:.6f} err={err:.2f}% peak={mem_run.get('peak_gib','?')}GiB", flush=True)
        
        print(f"[L={L}] DONE: {engine.step_count} steps {t_run:.1f}s "
              f"({result['ms_per_step']:.1f}ms/step) peak={mem_run.get('peak_gib','?')}GiB", flush=True)
        
    except torch.sdaa.OutOfMemoryError as e:
        result["status"] = "OOM_AT_STEP"
        result["oom_at_step"] = engine.step_count
        result["oom_error"] = str(e)[:300]
        result["mem_at_oom"] = _mem()
        result["steps_completed"] = engine.step_count
        if step_times:
            result["ms_per_step"] = round(sum(step_times)/len(step_times)*1000, 2)
        print(f"[L={L}] OOM at step {engine.step_count}", flush=True)
    except Exception as e:
        result["status"] = "ERROR_AT_RUN"
        result["error"] = f"{type(e).__name__}: {e}"[:300]
        result["mem_at_error"] = _mem()
        result["steps_completed"] = engine.step_count if hasattr(engine, 'step_count') else 0
        print(f"[L={L}] ERROR: {e}", flush=True)
    finally:
        try: del engine
        except: pass
        gc.collect()
        _reset()
    
    _save(result)

def _save(result):
    # Merge into existing JSON
    try:
        with open(OUT_JSON) as f:
            data = json.load(f)
    except:
        data = {"tiers": {}}
    
    if "tiers" not in data:
        data["tiers"] = {}
    data["tiers"][str(result["L"])] = result
    
    # Update summary
    tiers = data["tiers"]
    completed = sorted([int(k) for k, v in tiers.items() if v.get("status") == "COMPLETED"])
    oom = sorted([int(k) for k, v in tiers.items() if "OOM" in v.get("status", "")])
    errors = sorted([int(k) for k, v in tiers.items() if "ERROR" in v.get("status", "")])
    
    data["summary"] = {
        "completed_tiers": completed,
        "oom_tiers": oom,
        "error_tiers": errors,
        "max_L_completed": max(completed) if completed else None,
    }
    
    # Verdict
    max_c = max(completed) if completed else 0
    if max_c >= 96:
        data["verdict"] = "PASS" if any(
            tiers.get(str(L), {}).get("cd_results", {}).get("gate_pass")
            for L in completed
        ) else "HONEST_FAIL"
    elif max_c >= 48:
        data["verdict"] = "HONEST_FAIL_MEMORY_LIMITED"
    else:
        data["verdict"] = "HONEST_FAIL"
    
    with open(OUT_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[L={result['L']}] saved to {OUT_JSON}", flush=True)

if __name__ == "__main__":
    main()
