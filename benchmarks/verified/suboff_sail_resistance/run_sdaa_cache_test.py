#!/usr/bin/env python3
"""Try L=72 with empty_cache() before first collide step."""

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

from run import SUBOFF_LENGTH_M, SUBOFF_RADIUS_M, U_PHYS, SuboffAppendageEngine, blasius_cf
from tensorlbm.general_sim import (
    CollisionModel, ForceMethod, GeneralSimConfig, GeometryConfig,
    GeometrySource, LatticeModel, OutputConfig, PhysicsConfig,
    SolverConfig, WallTreatment,
)

L = int(sys.argv[1]) if len(sys.argv) > 1 else 72
N_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
DEVICE = "sdaa:0"

def _gib(b): return b / (1024.0**3)
def _mem():
    try:
        free, total = torch.sdaa.mem_get_info(0)
        alloc = torch.sdaa.memory_allocated(0)
        peak = torch.sdaa.max_memory_allocated(0)
        return f"free={_gib(free):.2f}GiB alloc={_gib(alloc):.2f}GiB peak={_gib(peak):.2f}GiB"
    except: return "?"
def _reset():
    try: torch.sdaa.reset_peak_memory_stats(0)
    except: pass
    try: torch.sdaa.empty_cache()
    except: pass

print(f"=== L={L} with empty_cache test ===", flush=True)
_reset()

viscosity = U_PHYS * SUBOFF_LENGTH_M / 1000.0
config = GeneralSimConfig(
    name=f"sdaa_cache_test_L{L}",
    geometry=GeometryConfig(source=GeometrySource.PARAMETRIC_SUBOFF,
        suboff_length=SUBOFF_LENGTH_M, suboff_radius=SUBOFF_RADIUS_M),
    physics=PhysicsConfig(density=1000.0, viscosity=viscosity,
        inlet_velocity=U_PHYS, reference_length=SUBOFF_LENGTH_M),
    solver=SolverConfig(lattice=LatticeModel.D3Q19, collision=CollisionModel.MRT,
        resolution=L, domain_padding=(1.0, 4.0, 1.0, 1.0, 1.0, 1.0),
        max_steps=N_STEPS, warmup_steps=None, snapshot_interval=10_000_000,
        force_sample_interval=10, device=DEVICE,
        wall_treatment=WallTreatment.AUTO, force_method=ForceMethod.PRESSURE_FRICTION,
        pressure_extrap="none", p0_method="near_wall",
        friction_formula="mix50", mass_correction=True,
        mass_correction_interval=200, smagorinsky_cs=0.05),
    output=OutputConfig(directory=f"/tmp/sdaa_cache_L{L}", formats=[],
        save_macroscopic=False, save_forces=True),
)

try:
    engine = SuboffAppendageEngine(config, hull_type="with_sail")
    engine.setup()
    print(f"setup OK: domain={engine.nx}x{engine.ny}x{engine.nz} cells={engine.nx*engine.ny*engine.nz}", flush=True)
    print(f"after setup: {_mem()}", flush=True)
    
    # CRITICAL: empty_cache before collide to free reserved memory
    torch.sdaa.empty_cache()
    _reset()
    print(f"after empty_cache+reset: {_mem()}", flush=True)
    
    from tensorlbm.lbm_step_correct import lbm_step_correct
    from tensorlbm.boundaries3d import far_field_bc_3d
    import functools
    from tensorlbm.solver3d import correct_mass3d
    
    collide_fn, collide_kwargs = engine._get_collide_fn()
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=engine._build_bc_config())
    solid = engine.solid
    tau = engine.uc.tau
    u_in = engine.uc.u_lb
    dpS = engine._compute_dpS()
    target_mass = engine._initial_mass
    
    for step in range(1, N_STEPS + 1):
        t0 = time.time()
        engine.f = lbm_step_correct(engine.f, collide_fn, tau, solid, u_in, far_field_fn, **collide_kwargs)
        engine.step_count += 1
        dt = time.time() - t0
        if step % 10 == 0:
            engine._sample_forces(dpS, engine.uc.nu_lb)
        print(f"step {step}/{N_STEPS} dt={dt*1000:.1f}ms {_mem()}", flush=True)
    
    print(f"COMPLETED {engine.step_count} steps", flush=True)
    
    # Save result
    result = {
        "L": L, "status": "COMPLETED_WITH_CACHE_FLUSH",
        "steps_completed": engine.step_count,
        "mem_peak": _mem(),
        "domain": [engine.nx, engine.ny, engine.nz],
        "total_cells": engine.nx * engine.ny * engine.nz,
    }
    
    OUT = Path(_REPO_ROOT) / "artifacts" / "sdaa_suboff_sail_resistance_m289.json"
    try:
        with open(OUT) as f: data = json.load(f)
    except: data = {"tiers": {}}
    if "tiers" not in data: data["tiers"] = {}
    if str(L) not in data["tiers"]:
        data["tiers"][str(L)] = {}
    data["tiers"][str(L)].update(result)
    with open(OUT, "w") as f: json.dump(data, f, indent=2)
    print(f"saved to {OUT}", flush=True)

except torch.sdaa.OutOfMemoryError as e:
    print(f"OOM: {e}", flush=True)
    print(f"at OOM: {_mem()}", flush=True)
    result = {"L": L, "status": "OOM_WITH_CACHE_FLUSH", "oom_error": str(e)[:300], "mem_at_oom": _mem()}
    OUT = Path(_REPO_ROOT) / "artifacts" / "sdaa_suboff_sail_resistance_m289.json"
    try:
        with open(OUT) as f: data = json.load(f)
    except: data = {"tiers": {}}
    if "tiers" not in data: data["tiers"] = {}
    if str(L) not in data["tiers"]:
        data["tiers"][str(L)] = {}
    data["tiers"][str(L)].update(result)
    with open(OUT, "w") as f: json.dump(data, f, indent=2)
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}", flush=True)
