"""Memory probe: build the BFL sphere setup at several D and report device peak."""
from __future__ import annotations
import math, os, sys
import torch
import torch_sdaa  # noqa: F401  (registers torch.sdaa)
sys.path.insert(0, "/root/TensorLBM_feat2/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tensorlbm.general_sim import (  # noqa: E402
    CollisionModel, ForceMethod, GeneralSimConfig, GeneralSimEngine,
    GeometryConfig, GeometrySource, LatticeModel, OutputConfig, OutputFormat,
    PhysicsConfig, SolverConfig, WallTreatment,
)
from tensorlbm.interpolated_bc import compute_q_sphere  # noqa: E402

DEV = os.environ.get("W8A_DEV", "sdaa:2")

def build(D, lat, up, down):
    pad = (up, down, lat, lat, lat, lat)
    return GeneralSimConfig(
        name=f"probe_D{D}",
        geometry=GeometryConfig(source=GeometrySource.PARAMETRIC_SPHERE,
                                sphere_radius=0.5, sphere_center=(0.0, 0.0, 0.0)),
        physics=PhysicsConfig(density=1000.0, viscosity=1e-6, inlet_velocity=1e-4,
                              reference_length=1.0),
        solver=SolverConfig(lattice=LatticeModel.D3Q19, collision=CollisionModel.MRT,
                            resolution=D, domain_padding=pad, max_steps=1,
                            snapshot_interval=10**9, force_sample_interval=50,
                            device=DEV, wall_treatment=WallTreatment.BOUNCE_BACK,
                            mass_correction=True, mass_correction_interval=200,
                            force_method=ForceMethod.MOMENTUM_EXCHANGE,
                            mem_variant="wet_node"),
        output=OutputConfig(directory="out", formats=[OutputFormat.NPY],
                            save_macroscopic=False, save_forces=True),
    )

for D, lat, up, down in [(20,3.0,3.0,4.0),(24,3.0,3.0,4.0),(28,3.0,3.0,4.0),
                         (32,3.0,3.0,4.0),(36,3.0,3.0,4.0),(40,3.0,3.0,4.0)]:
    try:
        torch.sdaa.empty_cache()
        torch.sdaa.reset_peak_memory_stats()
        eng = GeneralSimEngine(build(D, lat, up, down)); eng.setup()
        nz, ny, nx = eng.solid.shape
        R = D/2.0
        bfl_mask, bfl_q = compute_q_sphere(nx, ny, nz, up*D+R, ny/2.0, nz/2.0, R, torch.device(DEV))
        nl = int(bfl_mask[1:].sum().item())
        peak = torch.sdaa.max_memory_allocated()/2**30
        print(f"D={D} dom={nx}x{ny}x{nz} cells={nx*ny*nz/1e6:.1f}M links={nl} peak={peak:.2f}GiB")
        del eng, bfl_mask, bfl_q
        torch.sdaa.empty_cache()
    except Exception as e:
        print(f"D={D} FAIL {str(e)[:100]}")
        try: torch.sdaa.empty_cache()
        except Exception: pass
print("PROBE_DONE")