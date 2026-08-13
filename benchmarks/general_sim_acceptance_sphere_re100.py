#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""GeneralSimEngine acceptance baseline — 3D sphere Re=100 (D=40 cells).

Validates the unified solver entry (src/tensorlbm/general_sim.py) as a
product kernel: configure physics/geometry once, engine auto-selects
collision (MRT), wall treatment (bounce-back), domain size, then runs
lbm_step_correct + drag_pressure integration.

Expected: Cd ≈ 1.09 (Schiller-Naumann at Re=100);
SDAA reference: 1.053 (3.4% @ D40 180^3), 1.079 (0.98% quadratic).

Usage:
  python general_sim_acceptance_sphere_re100.py [extrap] [steps] [pad] [collision] [cs]

  extrap:    none|quadratic          (default none)
  steps:     int                     (default 3000)
  pad:       float|0                 (domain_padding in multiples of L,
                                     all 6 faces; 0 = auto domain; 1.75 → 180³)
  collision: auto|mrt|smag_mrt       (default auto; smag_mrt = MRT+Smagorinsky)
  cs:        float                   (Smagorinsky constant, default 0.05)
"""
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
    extrap = sys.argv[1] if len(sys.argv) > 1 else "none"
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    pad = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    collision_name = sys.argv[4] if len(sys.argv) > 4 else "auto"
    cs = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
    assert extrap in ("none", "quadratic"), extrap
    assert collision_name in ("auto", "mrt", "smag_mrt"), collision_name

    if collision_name == "auto":
        collision = CollisionModel.AUTO
    elif collision_name == "mrt":
        collision = CollisionModel.MRT
    else:
        collision = CollisionModel.SMAGORINSKY_MRT

    tag = f"pad{pad:g}_{collision_name}"
    out_dir = f"/home/wxsc/cxs/TensorLBM/results_general_sim_sphere_re100_{extrap}_{tag}"

    # Re = u*L/nu = 100, L = sphere diameter = 1.0 m, D = 40 lattice cells
    config = GeneralSimConfig(
        name=f"sphere_re100_d40_generalsim_{extrap}_{tag}",
        geometry=GeometryConfig(
            source=GeometrySource.PARAMETRIC_SPHERE,
            sphere_radius=0.5,          # m  → D = 1.0 m = reference_length
            sphere_center=(0.0, 0.0, 0.0),
        ),
        physics=PhysicsConfig(
            density=1000.0,             # kg/m^3 (water)
            viscosity=1.0e-6,           # m^2/s (water)
            inlet_velocity=1.0e-4,      # m/s → Re = 1e-4*1.0/1e-6 = 100
            reference_length=1.0,       # m (sphere diameter)
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q19,
            collision=collision,                # Re<1000 → MRT (auto)
            resolution=40,                      # D = 40 cells, R = 20
            domain_padding=tuple([pad] * 6) if pad > 0 else None,
            max_steps=n_steps,
            warmup_steps=None,                  # auto (reported only)
            snapshot_interval=500,
            force_sample_interval=10,
            device="cuda:0",
            wall_treatment=WallTreatment.AUTO,  # Re<10000 → bounce-back
            force_method=ForceMethod.PRESSURE_FRICTION,
            pressure_extrap=extrap,
            p0_method="near_wall",
            friction_formula="standard",
            mass_correction=True,
            mass_correction_interval=200,
            smagorinsky_cs=cs,
        ),
        output=OutputConfig(
            directory=out_dir,
            formats=[OutputFormat.NPY],
            save_macroscopic=True,
            save_forces=True,
        ),
    )

    print(f"=== GeneralSimEngine acceptance: sphere Re=100 D40, extrap={extrap} ===")
    print("Re (config) =", config.reynolds_number)

    engine = GeneralSimEngine(config)

    # ── Phase 1: setup ──────────────────────────────────────────────
    t0 = time.time()
    setup_info = engine.setup()
    t_setup = time.time() - t0
    print("\n[setup] %.1f s" % t_setup)
    for k in ("Re", "tau", "u_lb", "nu_lb", "Ma", "domain_lu", "obstacle_cells",
              "near_wall_cells", "total_cells", "device", "auto_collision",
              "auto_wall_treatment", "auto_warmup", "status"):
        print(f"  {k:22s} = {setup_info[k]}")
    R_lb = config.geometry.sphere_radius / (config.physics.reference_length / config.solver.resolution)
    print(f"  sphere R_lb            = {R_lb}  (D = {2*R_lb:.0f} cells)")
    print(f"  sphere centre (lu)     = "
          f"({(config.geometry.sphere_center[0]-engine.domain_phys[0])/(config.physics.reference_length/config.solver.resolution):.1f}, "
          f"{(config.geometry.sphere_center[1]-engine.domain_phys[2])/(config.physics.reference_length/config.solver.resolution):.1f}, "
          f"{(config.geometry.sphere_center[2]-engine.domain_phys[4])/(config.physics.reference_length/config.solver.resolution):.1f})")

    # ── Phase 2: run ────────────────────────────────────────────────
    t0 = time.time()
    run_info = engine.run(steps=n_steps)
    t_run = time.time() - t0
    print(f"\n[run] {t_run:.1f} s for {run_info['steps']} steps "
          f"({t_run/max(run_info['steps'],1)*1000:.1f} ms/step), diverged={run_info['diverged']}")

    # ── Phase 3: results ────────────────────────────────────────────
    res = engine.results()
    cd = res["Cd_Cl"]
    cd_ref = schiller_naumann_cd(100.0)
    cd_ref_cg = clift_gauvin_cd(100.0)
    cd_tot = cd.get("Cd_total")
    err = (cd_tot - cd_ref) / cd_ref * 100.0 if cd_tot is not None else float("nan")
    print("\n[results]")
    print(f"  Cd_pressure = {cd.get('Cd_pressure'):.4f}")
    print(f"  Cd_friction = {cd.get('Cd_friction'):.4f}")
    print(f"  Cd_total    = {cd.get('Cd_total'):.4f}")
    print(f"  Cl          = {cd.get('Cl'):.4f}")
    print(f"  St          = {cd.get('St')}")
    print(f"  ref Schiller-Naumann Cd = {cd_ref:.4f}  → err = {err:+.2f}%")
    print(f"  ref Clift-Gauvin    Cd = {cd_ref_cg:.4f}  → err = {(cd_tot-cd_ref_cg)/cd_ref_cg*100:+.2f}%")
    print(f"  files: {len(res['saved_files'])} saved to {res['output_dir']}")

    # last-1000-step window mean from force log (steps 2000..3000)
    if engine.forces_log:
        recent = engine.forces_log[-100:]
        cd_win = sum(e["cd_total"] for e in recent) / len(recent)
        print(f"  Cd_total (last {len(recent)} samples, steps "
              f"{recent[0]['step']}..{recent[-1]['step']}) = {cd_win:.4f}")

    summary = {
        "engine": "GeneralSimEngine",
        "case": f"sphere_re100_d40_{extrap}",
        "Re": setup_info["Re"],
        "domain_lu": list(setup_info["domain_lu"]),
        "R_lb": R_lb,
        "u_lb": setup_info["u_lb"],
        "tau": setup_info["tau"],
        "auto_collision": setup_info["auto_collision"],
        "auto_wall_treatment": setup_info["auto_wall_treatment"],
        "steps": run_info["steps"],
        "run_seconds": round(t_run, 1),
        "ms_per_step": round(t_run / max(run_info["steps"], 1) * 1000, 2),
        "diverged": run_info["diverged"],
        "Cd_pressure": cd.get("Cd_pressure"),
        "Cd_friction": cd.get("Cd_friction"),
        "Cd_total": cd.get("Cd_total"),
        "Cd_ref_SN_1.09": cd_ref,
        "err_pct_vs_SN": round(err, 2),
        "Cd_ref_CliftGauvin": cd_ref_cg,
        "err_pct_vs_CG": round((cd_tot - cd_ref_cg) / cd_ref_cg * 100.0, 2) if cd_tot else None,
    }
    with open(f"{out_dir}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nsummary.json written:", json.dumps(summary, indent=2))
    print("=== DONE ===")


if __name__ == "__main__":
    main()
