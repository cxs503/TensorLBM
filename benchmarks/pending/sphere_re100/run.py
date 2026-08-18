#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""B2: sphere Re=100 D60 网格加密 — 验证 GeneralSimEngine 收敛趋势"""
import sys, os, time, json, math

sys.path.insert(0, "/home/wxsc/cxs/TensorLBM/src")

import torch

from tensorlbm.general_sim import (
    GeneralSimConfig, GeneralSimEngine,
    GeometryConfig, PhysicsConfig, SolverConfig, OutputConfig,
    GeometrySource, LatticeModel, CollisionModel, WallTreatment,
    ForceMethod, OutputFormat,
)


def schiller_naumann_cd(re):
    return 24.0 / re * (1.0 + 0.15 * re**0.687)


def clift_gauvin_cd(re):
    return 24.0 / re * (1.0 + 0.1315 * re ** (0.82 - 0.05 * math.log10(re)))


def main():
    resolution = int(sys.argv[1]) if len(sys.argv) > 1 else 60   # D cells
    extrap = sys.argv[2] if len(sys.argv) > 2 else "quadratic"
    n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 4000

    out_dir = f"/home/wxsc/cxs/TensorLBM/results_bench_b2_sphere_re100_d{resolution}_{extrap}"

    config = GeneralSimConfig(
        name=f"bench_b2_sphere_re100_d{resolution}_{extrap}",
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
            collision=CollisionModel.AUTO,       # Re<1000 -> MRT
            resolution=resolution,
            domain_padding=None,
            max_steps=n_steps,
            warmup_steps=None,
            snapshot_interval=1000,
            force_sample_interval=10,
            device="cuda:0",
            wall_treatment=WallTreatment.AUTO,
            force_method=ForceMethod.PRESSURE_FRICTION,
            pressure_extrap=extrap,
            p0_method="near_wall",
            friction_formula="standard",
            mass_correction=True,
            mass_correction_interval=200,
            smagorinsky_cs=0.05,
        ),
        output=OutputConfig(
            directory=out_dir,
            formats=[OutputFormat.NPY],
            save_macroscopic=True,
            save_forces=True,
        ),
    )

    print(f"=== B2: sphere Re=100 D{resolution}, extrap={extrap} ===")
    engine = GeneralSimEngine(config)
    engine.setup()
    t0 = time.time()
    summary = engine.run()
    print(f"run time: {time.time()-t0:.0f}s")

    # last-100-sample mean of cd_total from the forces log
    recent = engine.forces_log[-min(100, len(engine.forces_log)):]
    cd_tot = float(sum(e.get("cd_total", 0.0) for e in recent) / max(len(recent), 1))
    cd_ref = schiller_naumann_cd(config.reynolds_number)
    cd_ref_cg = clift_gauvin_cd(config.reynolds_number)
    err = (cd_tot - cd_ref) / cd_ref * 100.0 if cd_tot else float("nan")
    err_cg = (cd_tot - cd_ref_cg) / cd_ref_cg * 100.0 if cd_tot else float("nan")
    print(f"  Cd_total = {cd_tot:.4f}  (ref SN {cd_ref:.4f}, err {err:+.2f}%; CG {cd_ref_cg:.4f}, err {err_cg:+.2f}%)")
    with open(os.path.join(out_dir, "bench_result.json"), "w") as fh:
        json.dump({"case": "B2", "resolution": resolution, "extrap": extrap,
                   "cd_total": cd_tot, "cd_ref_sn": cd_ref, "err_pct_sn": err,
                   "cd_ref_cg": cd_ref_cg, "err_pct_cg": err_cg}, fh, indent=2)


if __name__ == "__main__":
    main()
