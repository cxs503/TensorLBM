#!/usr/bin/env python3
"""Quick run at L=80 on SDAA to check if it fits and produces valid results."""
import json, math, os, sys, time, traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CASE_DIR = _REPO_ROOT / "benchmarks" / "verified" / "suboff_sail_resistance"
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "benchmarks"))
sys.path.insert(0, str(_CASE_DIR))

import torch, torch_sdaa
from run import SuboffAppendageEngine, run_engine_routed, SUBOFF_LENGTH_M, SUBOFF_RADIUS_M, U_PHYS
from tensorlbm.general_sim import *

BLASIUS_CF_REF = 1.328 / math.sqrt(1000.0)
DEVICE = "sdaa:0"

def make_config(L, steps, out_dir):
    viscosity = U_PHYS * SUBOFF_LENGTH_M / 1000.0
    return GeneralSimConfig(
        name=f"sdaa_quick_L{L}",
        geometry=GeometryConfig(source=GeometrySource.PARAMETRIC_SUBOFF,
                                suboff_length=SUBOFF_LENGTH_M, suboff_radius=SUBOFF_RADIUS_M),
        physics=PhysicsConfig(density=1000.0, viscosity=viscosity,
                              inlet_velocity=U_PHYS, reference_length=SUBOFF_LENGTH_M),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q19, collision=CollisionModel.MRT,
            resolution=L, domain_padding=(1.0,4.0,1.0,1.0,1.0,1.0),
            max_steps=steps, warmup_steps=None, snapshot_interval=10_000_000,
            force_sample_interval=10, device=DEVICE,
            wall_treatment=WallTreatment.AUTO,
            force_method=ForceMethod.PRESSURE_FRICTION,
            pressure_extrap="none", p0_method="near_wall",
            friction_formula="standard", mass_correction=True,
            mass_correction_interval=200),
        output=OutputConfig(directory=str(out_dir), formats=[],
                            save_macroscopic=False, save_forces=True),
    )

for L in [64, 72, 80]:
    steps = 12000 if L <= 72 else 6000
    out_dir = _REPO_ROOT / "runs" / f"sdaa_quick_L{L}_s{steps}"
    torch.sdaa.empty_cache()
    torch.sdaa.reset_peak_memory_stats(0)
    
    print(f"\n{'='*60}")
    print(f"Running L={L}, steps={steps}")
    print(f"Memory before: free={torch.sdaa.mem_get_info(0)[0]/1024**3:.2f} GiB")
    
    config = make_config(L, steps, out_dir)
    t0 = time.time()
    try:
        engine = SuboffAppendageEngine(config, hull_type="with_sail")
        info = engine.setup()
        print(f"Setup OK: domain={info.get('domain_lu')}, cells={info.get('total_cells',0):,}")
        print(f"Memory after setup: alloc={torch.sdaa.memory_allocated(0)/1024**3:.2f} GiB, "
              f"peak={torch.sdaa.max_memory_allocated(0)/1024**3:.2f} GiB")
        
        run_info, elapsed = run_engine_routed(engine, compile_mode=None)
        t_total = time.time() - t0
        peak = torch.sdaa.max_memory_allocated(0) / 1024**3
        
        print(f"Run OK: steps={run_info.get('steps')}, time={elapsed:.0f}s, "
              f"ms/step={elapsed/max(run_info.get('steps',1),1)*1000:.1f}, peak={peak:.2f} GiB")
        
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
        n = min(1000, len(log))
        win = log[-n:]
        cd_p = sum(e["cd_pressure"] for e in win) / n * rescale
        cd_f = sum(e["cd_friction"] for e in win) / n * rescale
        cd_tot = cd_p + cd_f
        err = (cd_tot / BLASIUS_CF_REF - 1.0) * 100.0
        
        print(f"Cd_pressure={cd_p:.6f}, Cd_friction={cd_f:.6f}, Cd_total={cd_tot:.6f}")
        print(f"err vs Blasius = {err:.2f}%")
        print(f"force_samples={len(log)}, diverged={run_info.get('diverged')}")
        
        # Save result
        result = {
            "L": L, "steps": steps, "steps_done": run_info.get("steps"),
            "cd_total": cd_tot, "err_pct": round(err, 4),
            "ms_per_step": round(elapsed/max(run_info.get("steps",1),1)*1000, 1),
            "peak_gib": round(peak, 2),
            "status": "ok",
        }
        
    except Exception as e:
        t_total = time.time() - t0
        peak = torch.sdaa.max_memory_allocated(0) / 1024**3
        err_str = str(e)
        is_oom = "out of memory" in err_str.lower()
        print(f"{'OOM' if is_oom else 'ERROR'} after {t_total:.0f}s, peak={peak:.2f} GiB")
        print(f"  {err_str[:300]}")
        result = {"L": L, "steps": steps, "status": "oom" if is_oom else "error",
                  "error": err_str[:300], "peak_gib": round(peak, 2)}
        try:
            torch.sdaa.empty_cache()
        except:
            pass
    
    # Save individual result
    out_path = _REPO_ROOT / "artifacts" / f"sdaa_quick_L{L}.json"
    out_path.write_text(json.dumps(result, indent=1))
    print(f"Saved: {out_path}")
    
    del engine
    try:
        torch.sdaa.empty_cache()
    except:
        pass

print("\nDone.")
