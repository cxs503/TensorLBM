#!/usr/bin/env python3
"""Quick test: D=80 with minimal domain to check memory feasibility."""

import sys
sys.path.insert(0, '/root/tp-agent/work/m46/src')

import torch
from tensorlbm.general_sim import (
    CollisionModel, ForceMethod, GeneralSimConfig, GeneralSimEngine,
    GeometryConfig, GeometrySource, LatticeModel, OutputConfig, OutputFormat,
    PhysicsConfig, SolverConfig, WallTreatment,
)

DEV = "sdaa:0"

# Try very small padding for D=80
D = 80
lat = 1.5
up = 1.5
down = 2.0
pad = (up, down, lat, lat, lat, lat)

cfg = GeneralSimConfig(
    name=f"test_D{D}",
    geometry=GeometryConfig(
        source=GeometrySource.PARAMETRIC_SPHERE,
        sphere_radius=0.5,
        sphere_center=(0.0, 0.0, 0.0),
    ),
    physics=PhysicsConfig(
        density=1000.0,
        viscosity=1.0e-6,
        inlet_velocity=1.0e-4,
        reference_length=1.0,
    ),
    solver=SolverConfig(
        lattice=LatticeModel.D3Q19,
        collision=CollisionModel.MRT,
        resolution=D,
        domain_padding=pad,
        max_steps=10,
        snapshot_interval=10**9,
        force_sample_interval=50,
        device=DEV,
        wall_treatment=WallTreatment.BOUNCE_BACK,
        mass_correction=True,
        mass_correction_interval=200,
        force_method=ForceMethod.MOMENTUM_EXCHANGE,
        mem_variant="wet_node",
    ),
    output=OutputConfig(
        directory="out",
        formats=[OutputFormat.NPY],
        save_macroscopic=False,
        save_forces=True,
    ),
)

print(f"Testing D={D} with pad={pad}")
print(f"SDAA memory: {torch.sdaa.get_device_properties(0).total_memory / 1e9:.2f} GB")

try:
    engine = GeneralSimEngine(cfg)
    engine.setup()
    
    solid = engine.solid
    nz, ny, nx = solid.shape
    print(f"Domain size: {nz} x {ny} x {nx} = {nz*ny*nx/1e6:.2f}M cells")
    print(f"Memory per f array: {nz*ny*nx*19*8/1e9:.2f} GB")
    print(f"SUCCESS: D={D} fits in memory!")
    
except RuntimeError as e:
    print(f"FAILED: {e}")
    
    # Try even smaller
    print("\nTrying D=70...")
    D = 70
    cfg.solver.resolution = D
    try:
        engine = GeneralSimEngine(cfg)
        engine.setup()
        solid = engine.solid
        nz, ny, nx = solid.shape
        print(f"Domain size: {nz} x {ny} x {nx} = {nz*ny*nx/1e6:.2f}M cells")
        print(f"Memory per f array: {nz*ny*nx*19*8/1e9:.2f} GB")
        print(f"SUCCESS: D={D} fits in memory!")
    except RuntimeError as e2:
        print(f"FAILED: {e2}")
