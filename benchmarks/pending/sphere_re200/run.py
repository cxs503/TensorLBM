#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""B3: sphere Re=200 drag benchmark — GeneralSimEngine common-module path.

Physics: uniform free stream over a sphere (D = 1.0 m), Re = u*L/nu = 200
with u = 1e-4 m/s, L = 1.0 m, nu = 5e-7 m^2/s (same frame as the Re=100
sphere case, only viscosity halved).

Common-module path: GeneralSimConfig -> GeneralSimEngine.setup() -> run()
  - D3Q19, collision AUTO (-> MRT, Re<1000), wall AUTO (-> half-way bounce-back)
  - mass_correction=True (interval 200)
  - ForceMethod.PRESSURE_FRICTION with pressure_extrap='none'  (REAL simulation,
    no extrapolation), p0_method='near_wall', friction_formula=<arg>

References (formula-exact values, used for the verdict):
  - Schiller-Naumann: Cd = 24/Re*(1 + 0.15*Re^0.687) = 0.8056
  - Clift-Gauvin:     Cd = 24/Re*(1 + 0.1315*Re^(0.82-0.05*log10(Re))) = 0.7810
  NOTE: the task sheet quoted SN(200)=0.769 / CG(200)=0.773 — that arithmetic
  is wrong (0.769 corresponds to Re~185). Formula-exact values are used here,
  same convention as the sphere_re100 D3Q27 case (1.087 vs 1.0917 note).

Usage:
  python run.py [resolution] [steps] [device] [friction]
    resolution: 60|80 (D cells, default 60)
    steps:      total steps (default 16000)
    device:     cuda:N (default cuda:1)
    friction:   standard|lagrange (default standard)
"""
import sys, os, time, json, math, threading

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
    resolution = int(sys.argv[1]) if len(sys.argv) > 1 else 60   # D cells
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 16000
    device = sys.argv[3] if len(sys.argv) > 3 else "cuda:1"
    friction = sys.argv[4] if len(sys.argv) > 4 else "standard"
    assert friction in ("standard", "lagrange"), friction

    out_dir = f"/home/wxsc/cxs/TensorLBM/results_b3_sphere_re200_d{resolution}_{friction}"

    # Re = u*L/nu = 1e-4 * 1.0 / 5e-7 = 200
    config = GeneralSimConfig(
        name=f"b3_sphere_re200_d{resolution}_{friction}",
        geometry=GeometryConfig(
            source=GeometrySource.PARAMETRIC_SPHERE,
            sphere_radius=0.5,          # m  -> D = 1.0 m = reference_length
            sphere_center=(0.0, 0.0, 0.0),
        ),
        physics=PhysicsConfig(
            density=1000.0,             # kg/m^3
            viscosity=5.0e-7,           # m^2/s  -> Re = 200
            inlet_velocity=1.0e-4,      # m/s
            reference_length=1.0,       # m (sphere diameter)
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q19,
            collision=CollisionModel.AUTO,       # Re<1000 -> MRT
            resolution=resolution,
            domain_padding=None,                 # auto domain (3.5D x 3D x 3D)
            max_steps=n_steps,
            warmup_steps=None,
            snapshot_interval=100000,            # no field snapshots (save_macroscopic=False)
            force_sample_interval=10,
            device=device,
            wall_treatment=WallTreatment.AUTO,   # Re<10000 -> bounce-back
            force_method=ForceMethod.MOMENTUM_EXCHANGE,
            pressure_extrap="none",              # REAL simulation, no extrapolation
            p0_method="near_wall",
            friction_formula=friction,
            mass_correction=True,
            mass_correction_interval=200,
            smagorinsky_cs=0.05,
        ),
        output=OutputConfig(
            directory=out_dir,
            formats=[OutputFormat.NPY],
            save_macroscopic=False,              # keep RAM low; forces.csv is the record
            save_forces=True,
        ),
    )

    print(f"=== B3: sphere Re=200 D{resolution}, friction={friction}, extrap=none ===")
    engine = GeneralSimEngine(config)

    t0 = time.time()
    setup_info = engine.setup()
    print(f"[setup] {time.time()-t0:.1f}s")
    for k in ("Re", "tau", "u_lb", "nu_lb", "Ma", "domain_lu", "obstacle_cells",
              "near_wall_cells", "total_cells", "device", "auto_collision",
              "auto_wall_treatment"):
        print(f"  {k:18s} = {setup_info[k]}")
    print(f"  Re config         = {config.reynolds_number}")
    print(f"  Cd_ref SN (exact) = {schiller_naumann_cd(200.0):.4f}")
    print(f"  Cd_ref CG (exact) = {clift_gauvin_cd(200.0):.4f}")

    # Monitor thread: periodic last-100-sample window mean while running
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            stop.wait(90)
            if engine.forces_log:
                recent = engine.forces_log[-100:]
                cd = sum(e["cd_total"] for e in recent) / len(recent)
                cd_p = sum(e["cd_pressure"] for e in recent) / len(recent)
                cd_f = sum(e["cd_friction"] for e in recent) / len(recent)
                print(f"  [mon] step={engine.step_count:6d}  "
                      f"cd_p={cd_p:.4f} cd_f={cd_f:.4f} cd_tot(last100)={cd:.4f}",
                      flush=True)

    th = threading.Thread(target=monitor, daemon=True)
    th.start()

    t0 = time.time()
    run_info = engine.run(steps=n_steps)
    stop.set()
    t_run = time.time() - t0
    print(f"[run] {t_run:.1f}s for {run_info['steps']} steps "
          f"({t_run/max(run_info['steps'],1)*1000:.1f} ms/step), "
          f"diverged={run_info['diverged']}")

    # ── Analysis: last-100-sample mean (task criterion) + convergence windows ──
    log = engine.forces_log
    n_samples = len(log)
    recent = log[-min(100, n_samples):]
    cd_tot = float(sum(e["cd_total"] for e in recent) / max(len(recent), 1))
    cd_p = float(sum(e["cd_pressure"] for e in recent) / max(len(recent), 1))
    cd_f = float(sum(e["cd_friction"] for e in recent) / max(len(recent), 1))

    # block means: 5 consecutive windows of 100 samples (1000 steps each), oldest first
    blocks = []
    nb = min(5, n_samples // 100)
    for b in range(nb):
        seg = log[-(nb - b) * 100:-(nb - b - 1) * 100] if b < nb - 1 else log[-100:]
        blocks.append(round(float(sum(e["cd_total"] for e in seg) / len(seg)), 5))
    drift = None
    if len(blocks) >= 2:
        drift = (blocks[-1] - blocks[-2]) / blocks[-1] * 100.0

    cd_ref_sn = schiller_naumann_cd(200.0)
    cd_ref_cg = clift_gauvin_cd(200.0)
    err_sn = (cd_tot - cd_ref_sn) / cd_ref_sn * 100.0
    err_cg = (cd_tot - cd_ref_cg) / cd_ref_cg * 100.0

    print(f"\n[results] D={resolution}, steps={run_info['steps']}, "
          f"force samples={n_samples}")
    print(f"  Cd_pressure = {cd_p:.4f}")
    print(f"  Cd_friction = {cd_f:.4f}")
    print(f"  Cd_total    = {cd_tot:.4f}  (last {len(recent)} samples)")
    print(f"  window means (5x1000 steps): {blocks}")
    print(f"  last-window drift: {drift if drift is not None else float('nan'):+.3f}%")
    print(f"  ref SN {cd_ref_sn:.4f}  -> err {err_sn:+.2f}%   (task sheet quoted 0.769: arithmetic error)")
    print(f"  ref CG {cd_ref_cg:.4f}  -> err {err_cg:+.2f}%   (task sheet quoted 0.773: arithmetic error)")

    # ── Save: forces.csv (via engine.results) + bench_result.json ──
    os.makedirs(out_dir, exist_ok=True)
    try:
        res = engine.results()
        print(f"  saved {len(res['saved_files'])} files -> {res['output_dir']}")
    except Exception as e:  # results() failure must not lose the JSON
        print(f"  [warn] engine.results() failed: {e}")

    result = {
        "case": "B3",
        "benchmark": "sphere_re200",
        "engine": "GeneralSimEngine",
        "extrap": "none",                  # REAL simulation, no extrapolation
        "physics": {"u_mps": 1e-4, "L_m": 1.0, "nu_m2ps": 5e-7, "Re": 200.0},
        "resolution_D_cells": resolution,
        "domain_lu": list(setup_info["domain_lu"]),
        "tau": setup_info["tau"],
        "u_lb": setup_info["u_lb"],
        "nu_lb": setup_info["nu_lb"],
        "collision": setup_info["auto_collision"],
        "wall_treatment": setup_info["auto_wall_treatment"],
        "friction_formula": friction,
        "mass_correction": True,
        "mass_correction_interval": 200,
        "steps": run_info["steps"],
        "diverged": run_info["diverged"],
        "ms_per_step": round(t_run / max(run_info["steps"], 1) * 1000, 2),
        "Cd_pressure": cd_p,
        "Cd_friction": cd_f,
        "Cd_total": cd_tot,
        "Cd_window_means_5x1000": blocks,
        "last_window_drift_pct": drift,
        "Cd_ref_SN_exact": round(cd_ref_sn, 6),
        "err_pct_vs_SN": round(err_sn, 2),
        "Cd_ref_CG_exact": round(cd_ref_cg, 6),
        "err_pct_vs_CG": round(err_cg, 2),
        "note_task_sheet_refs": ("task sheet quoted SN(200)=0.769 / CG(200)=0.773 "
                                 "but formula-exact values are SN=0.8056 / CG=0.7810 "
                                 "(0.769 corresponds to Re~185); verdict uses formula-exact"),
    }
    with open(os.path.join(out_dir, "bench_result.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print("\nbench_result.json written:")
    print(json.dumps(result, indent=2))
    print("=== DONE ===")


if __name__ == "__main__":
    main()
