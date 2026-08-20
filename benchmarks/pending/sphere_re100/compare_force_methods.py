#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""Sphere Re=100 force-method comparison (G12/G15 fix screening, D=40 fast run).

One simulation, many force measurements:
  - Pressure+friction integration with p0_method = <arg> (default far_field)
  - MEM variants (mem_variant='all'): standard / galilean / bg_sub
After run, re-samples the steady-state field with all p0_methods to compare
background-pressure choices side by side.

Usage:
  python compare_force_methods.py [resolution] [steps] [p0_method] [extrap] [friction]
    resolution: D cells (default 40)
    steps:      total steps (default 4000)
    p0_method:  near_wall|far_field|domain_avg|inlet (default far_field)
    extrap:     none|linear|quadratic (default none)
    friction:   standard|lagrange (default standard)
"""
import sys, os, time, json, math

sys.path.insert(0, "/home/wxsc/cxs/TensorLBM/src")

import torch  # noqa: E402

from tensorlbm.general_sim import (  # noqa: E402
    GeneralSimConfig, GeneralSimEngine,
    GeometryConfig, PhysicsConfig, SolverConfig, OutputConfig,
    GeometrySource, LatticeModel, CollisionModel, WallTreatment,
    ForceMethod, OutputFormat,
)


def schiller_naumann_cd(re):
    return 24.0 / re * (1.0 + 0.15 * re ** 0.687)


def clift_gauvin_cd(re):
    return 24.0 / re * (1.0 + 0.1315 * re ** (0.82 - 0.05 * math.log10(re)))


def main():
    resolution = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    p0_method = sys.argv[3] if len(sys.argv) > 3 else "far_field"
    extrap = sys.argv[4] if len(sys.argv) > 4 else "none"
    friction = sys.argv[5] if len(sys.argv) > 5 else "standard"
    device = sys.argv[6] if len(sys.argv) > 6 else "cuda:0"
    assert friction in ("standard", "2nd_order", "central", "lagrange", "faces"), friction

    out_dir = f"/home/wxsc/cxs/TensorLBM/results_fmcompare_sphere_re100_d{resolution}_{p0_method}_{extrap}_{friction}"

    config = GeneralSimConfig(
        name=f"fmcompare_sphere_re100_d{resolution}_{p0_method}_{extrap}",
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
            collision=CollisionModel.AUTO,
            resolution=resolution,
            domain_padding=None,
            max_steps=n_steps,
            warmup_steps=None,
            snapshot_interval=100000,
            force_sample_interval=10,
            device=device,
            wall_treatment=WallTreatment.AUTO,
            force_method=ForceMethod.BOTH,   # pressure+friction AND all MEM variants
            mem_variant="all",
            pressure_extrap=extrap,
            p0_method=p0_method,
            friction_formula=friction,
            mass_correction=True,
            mass_correction_interval=200,
            smagorinsky_cs=0.05,
        ),
        output=OutputConfig(
            directory=out_dir,
            formats=[OutputFormat.NPY],
            save_macroscopic=False,
            save_forces=True,
        ),
    )

    print(f"=== sphere Re=100 D{resolution}, p0={p0_method}, extrap={extrap}, mem=all ===")
    engine = GeneralSimEngine(config)
    setup_info = engine.setup()
    for k in ("Re", "tau", "u_lb", "nu_lb", "Ma", "domain_lu", "obstacle_cells",
              "near_wall_cells", "total_cells"):
        print(f"  {k:16s} = {setup_info[k]}")
    print(f"  dpS = {engine._compute_dpS():.6f}")
    cd_ref_sn = schiller_naumann_cd(100.0)
    cd_ref_cg = clift_gauvin_cd(100.0)
    print(f"  Cd_ref SN = {cd_ref_sn:.4f}  (task sheet ~1.087), CG = {cd_ref_cg:.4f}")

    t0 = time.time()
    run_info = engine.run(steps=n_steps)
    print(f"  run {run_info['steps']} steps in {time.time()-t0:.0f}s, diverged={run_info['diverged']}")

    # ── Time-averaged comparison over last 100 samples ──────────────────
    log = engine.forces_log
    recent = log[-min(100, len(log)):]
    keys = ["cd_pressure", "cd_friction", "cd_total", "cd_mem",
            "cd_mem_standard", "cd_mem_galilean", "cd_mem_bgsub"]
    means = {k: float(sum(e.get(k, 0.0) for e in recent) / len(recent)) for k in keys}

    # ── Post-run single-frame re-sampling: 4 p0_methods × 4 friction formulas ──
    from tensorlbm.drag_pressure import drag_pressure_integration, drag_friction_integration
    dpS = engine._compute_dpS()
    p0_scan = {}
    for pm in ("near_wall", "far_field", "domain_avg", "inlet"):
        px, py, pz = drag_pressure_integration(
            engine.f, engine.mesh, dpS, extrap=extrap, p0_method=pm, solid=engine.solid
        )
        p0_scan[pm] = px
    friction_scan = {}
    for ff in ("standard", "2nd_order", "central", "lagrange", "faces"):
        fxf, fyf, fzf = drag_friction_integration(engine.f, engine.mesh, dpS,
                                                  setup_info["nu_lb"], formula=ff,
                                                  solid=engine.solid)
        friction_scan[ff] = fxf
    p0_scan["friction_scan"] = friction_scan
    fxf, fyf, fzf = drag_friction_integration(engine.f, engine.mesh, dpS, setup_info["nu_lb"],
                                              formula=friction, solid=engine.solid)
    p0_scan["friction"] = fxf
    # rho*U0*sum(n_hat*dA) leakage estimate (task's approximate background)
    from tensorlbm.momentum_exchange import momentum_exchange_standard
    me_std = momentum_exchange_standard(engine.f, engine.solid, engine.near)
    leak_x = float((engine.mesh.nx_n * engine.mesh.dA).sum())
    leak_y = float((engine.mesh.ny_n * engine.mesh.dA).sum())
    leak_z = float((engine.mesh.nz_n * engine.mesh.dA).sum())
    u_lb = setup_info["u_lb"]

    # ── Report ──────────────────────────────────────────────────────────
    print(f"\n[time-avg last {len(recent)} samples]")
    for k in keys:
        v = means[k]
        ref = cd_ref_sn
        print(f"  {k:18s} = {v:.4f}  (err vs SN {ref:.4f}: {(v-ref)/ref*100:+.2f}%)")
    print(f"[single-frame scan, extrap={extrap}, step={engine.step_count}]")
    for pm in ("near_wall", "far_field", "domain_avg", "inlet", "friction"):
        v = p0_scan[pm]
        print(f"  p0={pm:12s} cd = {v:.4f}  (err {(v-cd_ref_sn)/cd_ref_sn*100:+.2f}%)")
    print("  friction scan (single-frame):")
    for ff, v in friction_scan.items():
        print(f"    {ff:10s} cd_f = {v:.4f}  (err {(v-cd_ref_sn)/cd_ref_sn*100:+.2f}% vs SN)")
    print(f"  surface leakage Σn̂dA = ({leak_x:.4f}, {leak_y:.4f}, {leak_z:.4f})")
    print(f"  ρU·Σn̂dA approx = {1.0*u_lb*leak_x:.6f} -> Cd_approx = {1.0*u_lb*leak_x/dpS:.4f}")

    # window means for convergence check (cd_total and cd_mem_bgsub)
    def window_means(key, nw=5):
        nb = min(nw, len(log) // 100)
        if nb < 2:
            return []
        out = []
        for b in range(nb):
            seg = log[-(nb - b) * 100:-(nb - b - 1) * 100] if b < nb - 1 else log[-100:]
            out.append(round(float(sum(e.get(key, 0.0) for e in seg) / len(seg)), 5))
        return out

    result = {
        "case": "sphere_re100_force_method_compare",
        "resolution": resolution, "steps": run_info["steps"], "p0_method": p0_method,
        "extrap": extrap, "mem_variant": "all", "diverged": run_info["diverged"],
        "Cd_ref_SN": round(cd_ref_sn, 6), "Cd_ref_CG": round(cd_ref_cg, 6),
        "time_avg": means,
        "err_pct_vs_SN": {k: round((v - cd_ref_sn) / cd_ref_sn * 100, 2) for k, v in means.items()},
        "p0_single_frame_scan": p0_scan,
        "surface_leakage_nhat_dA": [leak_x, leak_y, leak_z],
        "rho_U_leak_approx_Cd": 1.0 * u_lb * leak_x / dpS,
        "window_means_cd_total_5x1000": window_means("cd_total"),
        "window_means_cd_mem_bgsub_5x1000": window_means("cd_mem_bgsub"),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "force_compare.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
