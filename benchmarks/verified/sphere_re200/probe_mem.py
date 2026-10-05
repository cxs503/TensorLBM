"""Memory probe: build the BFL sphere setup at several D and report device peak.

Extended 2026-10-05 to (a) include small D=16 and (b) mimic the ACTUAL run.py
sparse-route allocation footprint (f + bfl mask/q + f_pre_stream clone) so the
reported peak is the real loop peak, not just setup.
"""
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
from tensorlbm.bfl_common import bfl_boundary_link_indices  # noqa: E402
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

CASES = [(16,3.0,3.0,4.0),(20,3.0,3.0,4.0),(24,3.0,3.0,4.0),(28,3.0,3.0,4.0),
         (30,3.0,3.0,4.0),(32,3.0,3.0,4.0),(36,3.0,3.0,4.0),(40,3.0,3.0,4.0)]

for D, lat, up, down in CASES:
    try:
        torch.sdaa.empty_cache()
        torch.sdaa.reset_peak_memory_stats()
        eng = GeneralSimEngine(build(D, lat, up, down)); eng.setup()
        nz, ny, nx = eng.solid.shape
        R = D/2.0
        bfl_mask, bfl_q = compute_q_sphere(nx, ny, nz, up*D+R, ny/2.0, nz/2.0, R, torch.device(DEV))
        nl = int(bfl_mask[1:].sum().item())
        links = bfl_boundary_link_indices(bfl_mask, bfl_q, lattice="D3Q19")
        # mimic the run loop footprint: f + one full clone (f_pre_stream)
        f = eng.f.clone()
        f_pre_stream = f.clone()
        peak = torch.sdaa.max_memory_allocated()/2**30
        reserved = torch.sdaa.max_memory_reserved()/2**30
        print(f"D={D} dom={nx}x{ny}x{nz} cells={nx*ny*nz/1e6:.2f}M links={nl} "
              f"n_dense_links={links.n_links} peak={peak:.2f}GiB reserved={reserved:.2f}GiB", flush=True)
        del eng, bfl_mask, bfl_q, links, f, f_pre_stream
        torch.sdaa.empty_cache()
    except Exception as e:
        print(f"D={D} FAIL {str(e)[:140]}", flush=True)
        try: torch.sdaa.empty_cache()
        except Exception: pass
print("PROBE_DONE")