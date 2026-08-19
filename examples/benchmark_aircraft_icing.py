#!/usr/bin/env python
"""Aircraft icing benchmark — NACA 0012 rime ice (NASA Glenn IRT).

Phase 2a: minimal physically credible rime model, implemented in
``tensorlbm.aircraft_icing``:

* D3Q19 BGK flow around a NACA 0012 at AoA (bounce-back on airfoil + ice),
* Lagrangian droplets seeded at the *physical* LWC/MVD mass flux
  (seeding rate derived from unit mapping, see module docstring),
* Stokes relaxation ``du_d/dt = (u_f - u_d)/tau_d`` (no velocity mixing),
* mass-conserving freezing (cell fills with rho_rime*dx^3 of water),
* ice feedback on the flow (cd/cl drift),
* beta (collection efficiency) distribution along the airfoil surface,
* full mass audit (seeded = frozen + exited + airborne + pending, <1%).

Reference case (Ruff & Wright / NASA Glenn IRT NACA 0012 rime):
chord 0.5334 m, V=67 m/s, Re=2.5e6, LWC=0.5 g/m^3, MVD=20 um, T=-10 C,
t=360 s, AoA=4 deg.  Phase 2a runs a moderate lattice Re (BGK-stable) and
accelerates time via an effective LWC (exact for rime); the real-time
step count for 360 s is printed.  Re=2.5e6 with cumulant+LES is Phase 2b.

Usage
-----
    CUDA_VISIBLE_DEVICES=4 python examples/benchmark_aircraft_icing.py
    python examples/benchmark_aircraft_icing.py --nx 160 --ny 96 \
        --steps 800 --warmup 400 --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tensorlbm.aircraft_icing import (  # noqa: E402
    IcingConfig,
    eulerian_mass_audit_report,
    mass_audit_report,
    run_rime_icing,
    save_icing_artifacts,
)


def main() -> None:
    p = argparse.ArgumentParser(description="NACA 0012 rime icing (Phase 2a)")
    p.add_argument("--nx", type=int, default=320)
    p.add_argument("--ny", type=int, default=160)
    p.add_argument("--chord-frac", type=float, default=0.4)
    p.add_argument("--u-in", type=float, default=0.05)
    p.add_argument("--tau", type=float, default=0.55)
    p.add_argument("--aoa", type=float, default=4.0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--t-exposure", type=float, default=360.0,
                   help="cloud exposure to represent [s] (drives LWC accel k)")
    p.add_argument("--lwc", type=float, default=5.0e-4)
    p.add_argument("--mvd", type=float, default=20.0e-6)
    p.add_argument("--rho-rime-mode", default="macklin",
                   choices=["const", "macklin", "jones"],
                   help="rime density model (Macklin 1962 / Jones 1990 / fixed)")
    p.add_argument("--rho-rime", type=float, default=917.0,
                   help="fixed density [kg/m3] when --rho-rime-mode const")
    p.add_argument("--t-static", type=float, default=-10.0,
                   help="static temperature [C] (feeds the density correlations)")
    p.add_argument("--compile-mode", default=None,
                   choices=[None, "default", "max-autotune-no-cudagraphs"],
                   help="torch.compile mode for the flow step (shared compile_utils)")
    p.add_argument("--droplet-phase", default="lagrangian",
                   choices=["lagrangian", "eulerian", "both"],
                   help="Phase 2b: droplet formulation (2a Lagrangian default; "
                        "'both' runs both phases on the same flow trajectory "
                        "for beta cross-validation)")
    p.add_argument("--collision", default="bgk", choices=["bgk", "cumulant"],
                   help="flow collision operator (cumulant enables high Re)")
    p.add_argument("--c-s", type=float, default=0.0,
                   help="Smagorinsky constant (cumulant only; 0 disables LES)")
    p.add_argument("--re-lu", type=float, default=None,
                   help="target lattice Reynolds number (derives the knife-edge "
                        "tau; requires --collision cumulant for Re >~ 1e3)")
    p.add_argument("--drag-law", default="stokes",
                   choices=["stokes", "schiller-naumann"],
                   help="droplet drag law (applies to BOTH phases)")
    p.add_argument("--shadow-frac", type=float, default=1e-3,
                   help="Eulerian shadow-region threshold (fraction of alpha_in)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--no-prefill", action="store_true",
                   help="disable the t=0 steady-cloud inventory seeding")
    p.add_argument("--ab-clean", action="store_true",
                   help="also run the clean-airfoil twin (no droplets) to "
                        "isolate the ice feedback on cd/cl from startup drift")
    p.add_argument("--output", default="outputs/aircraft_icing")
    args = p.parse_args()

    cfg = IcingConfig(
        nx=args.nx,
        ny=args.ny,
        chord_frac=args.chord_frac,
        u_in=args.u_in,
        tau=args.tau,
        aoa_deg=args.aoa,
        steps=args.steps,
        warmup_steps=args.warmup,
        lwc=args.lwc,
        mvd=args.mvd,
        rime_density_mode=args.rho_rime_mode,
        rho_rime=args.rho_rime,
        t_static_c=args.t_static,
        t_exposure=args.t_exposure,
        prefill_cloud=not args.no_prefill,
        compile_mode=args.compile_mode,
        seed=args.seed,
        device=args.device,
        log_every=args.log_every,
        droplet_phase=args.droplet_phase,
        collision=args.collision,
        c_s=args.c_s,
        re_lu_target=args.re_lu,
        drag_law=args.drag_law,
        shadow_alpha_frac=args.shadow_frac,
    )

    print("=" * 72)
    print("  AIRCRAFT ICING — NACA 0012 rime (Phase 2a/2b: LWC/MVD-calibrated)")
    print("=" * 72)
    r = run_rime_icing(cfg)
    files = save_icing_artifacts(r, args.output)

    print("-" * 72)
    print("  MASS AUDIT [kg]")
    print(mass_audit_report(r))
    if r.get("eulerian") is not None:
        print("  EULERIAN ALPHA-FIELD AUDIT [kg]")
        print(eulerian_mass_audit_report(r))
    print(f"  rho_rime = {cfg.rho_rime_eff:.0f} kg/m^3 "
          f"(mode={cfg.rime_density_mode}, T_s_eff={cfg.t_surface_eff_c:.1f} C, "
          f"R={cfg.rime_R_macklin:.1f})")
    m = r["metrics"]
    print("-" * 72)
    print("  ICE SHAPE METRICS")
    print(f"    ice cells      : {m['n_ice_cells']}  "
          f"(area {m['ice_area_m2']:.4e} m^2 = {m['ice_area_pct_chord2']:.2f} % chord^2)")
    if "upper_horn_pct_chord" in m:
        print(f"    upper horn     : {m['upper_horn_cells']} cells "
              f"({m['upper_horn_pct_chord']:.2f} % chord = {m['upper_horn_m'] * 1e3:.1f} mm) "
              f"at {m['upper_horn_xy']}")
    if "lower_horn_pct_chord" in m:
        print(f"    lower horn     : {m['lower_horn_cells']} cells "
              f"({m['lower_horn_pct_chord']:.2f} % chord = {m['lower_horn_m'] * 1e3:.1f} mm) "
              f"at {m['lower_horn_xy']}")
    if "horn_symmetry_pct" in m:
        print(f"    horn symmetry  : {m['horn_symmetry_pct']:+.1f} % (U-L)/(U+L)")
    if "max_impact_x_frac" in m:
        print(f"    impingement    : max impact x = LE + "
              f"{m['max_impact_x_frac'] * 100:.1f} % chord")
    print("-" * 72)
    b = r["beta"]
    if len(b["beta"]):
        imax = b["beta"].argmax()
        print(f"  BETA: max={b['beta'].max():.3f} at s/c="
              f"{b['s_over_c'][imax] * 100:+.2f} % chord")
    e = r.get("eulerian")
    if e is not None and len(e["beta"]["beta"]):
        be = e["beta"]
        imax = be["beta"].argmax()
        print(f"  BETA-E (eulerian): max={be['beta'].max():.3f} at s/c="
              f"{be['s_over_c'][imax] * 100:+.2f} % chord, "
              f"capture height={float(e['beta_grid'].sum()):.2f} cells")
        if len(b["beta"]):
            print(f"  BETA   (lagrangian): capture height="
                  f"{float(r['beta_grid'].sum()):.2f} cells")
    if r["cd0"] is not None and r["cd_end"] is not None:
        print(f"  AERO: cd {r['cd0']:.5f} -> {r['cd_end']:.5f} "
              f"({r['cd_drift_pct']:+.1f} %), cl {r['cl0']:.5f} -> {r['cl_end']:.5f}")
        print(f"        reference cl (thin airfoil, AoA={args.aoa:.0f} deg): "
              f"{2 * 3.14159265 * (args.aoa * 3.14159265 / 180):.3f}")
    if args.ab_clean and r["cd0"] is not None:
        from dataclasses import replace
        print("  running clean-airfoil twin (same seed, no droplets) ...")
        twin = run_rime_icing(replace(cfg, disable_droplets=True))
        dcd = 100.0 * (r["cd_end"] - twin["cd_end"]) / twin["cd_end"]
        print(f"  A/B : cd_clean(end)={twin['cd_end']:.5f} "
              f"cl_clean(end)={twin['cl_end']:.5f}")
        print(f"        ice feedback on cd: {dcd:+.1f} %  "
              f"(iced {r['cd_end']:.5f} vs clean {twin['cd_end']:.5f})")
    print("-" * 72)
    for k, v in files.items():
        print(f"  artifact [{k:11s}] {v}")
    print("=" * 72)


if __name__ == "__main__":
    main()
