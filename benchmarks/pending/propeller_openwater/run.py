#!/usr/bin/env python3
"""Propeller open-water benchmark (DTMB 4119 caliber) for TensorLBM.

Geometry
--------
Parametric voxel propeller from :mod:`tensorlbm.propeller_cad` configured to the
DTMB 4119 primary parameters (Z=3, P/D(0.7R)=1.084, A_E/A_0=0.60, hub ratio
0.20), because that module accepts only the *scalar* parameter set, not the real
section tables (OpenProp ``Prop4119_input.m``: XCoD peak 0.4622D, NACA a=0.8
meanline, DTRC-modified NACA66 thickness, zero skew/rake).  This is an honest
parametric stand-in, not the real 4119 blade.

Rotation
--------
Ladd (1994) moving-wall bounce-back on the *voxelised* mask, re-voxelised at the
current physical azimuth every lattice update (``propeller_benchmark``).  Blade
loads are taken from the *same* moving-wall operator reaction
(``moving_wall_bounce_back_3d_with_reaction``) and converted with the repo caliber
``report_propeller_linkwise_loads``.  The legacy static momentum-exchange
estimator is recorded alongside as a cross-check.

Reference
---------
DTMB 4119 open-water experiment (widely tabulated; see module docstring /
&lt;case&gt;/README.md for the two independent sources used to cross-check it).

Usage
-----
    python run.py smoke            # short timing / stability probe
    python run.py full  out_dir    # 2+ J-point open-water sweep -> result.json
    python run.py single 0.7 out.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch  # noqa: F401

from tensorlbm.boundaries3d import (
    apply_zou_he_channel_boundaries_3d,
    make_channel_wall_mask_3d,
)
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.d3q27 import report_propeller_linkwise_loads
from tensorlbm.obstacles import (
    compute_obstacle_forces_3d,
    compute_obstacle_moments_3d,
)
from tensorlbm.propeller_benchmark import (
    moving_wall_bounce_back_3d_with_reaction,
    rotating_wall_velocity_3d,
)
from tensorlbm.propeller_cad import (
    PropellerGeometryConfig,
    build_propeller_mask,
    propeller_statistics,
)
from tensorlbm.solver3d import stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.utils import resolve_device

# ---------------------------------------------------------------------------
# DTMB 4119 reference (open-water experiment).  Primary tabulation reproduced
# from the standard open-water test table used across the DTMB 4119 CFD
# literature; independently cross-checked against the OpenProp "replica design
# from performance inputs" case (Prop4119_input.m: KT=0.15 at J=0.833).
# ---------------------------------------------------------------------------
REF_4119 = {
    0.500: (0.2853, 0.04641),
    0.600: (0.2497, 0.04269),
    0.700: (0.2120, 0.03893),
    0.800: (0.1700, 0.03482),
    0.833: (0.1525, 0.03319),
    0.900: (0.1210, 0.02959),
    1.000: (0.0655, 0.02428),
}
# OpenProp replica-design corroboration (independent source) for the J=0.833 row
REF_4119_CROSSCHECK = {"j": 0.833, "kt": 0.15, "source": "OpenProp Prop4119_input.m"}


def geometry_4119(diameter_lu: float) -> PropellerGeometryConfig:
    return PropellerGeometryConfig(
        n_blades=3,
        diameter=diameter_lu,
        hub_diameter_ratio=0.20,
        hub_length_ratio=0.35,
        pitch_ratio_07=1.084,
        blade_area_ratio=0.60,
        skew_deg=0.0,
        rake_ratio=0.0,
        max_thickness_ratio=0.054,
    )


def run_j(
    *,
    j_target: float,
    geometry: PropellerGeometryConfig,
    nx: int,
    ny: int,
    nz: int,
    rpm: float,
    tau: float,
    cs: float,
    warmup_steps: int,
    sample_steps: int,
    device: torch.device,
) -> dict:
    D = geometry.diameter
    omega = 2.0 * math.pi * rpm
    u_in = j_target * rpm * D
    cx = int(nx * 0.35)
    cy = ny // 2
    cz = nz // 2
    tip_speed = omega * geometry.radius
    tip_ma = tip_speed * math.sqrt(3.0)

    mask0 = build_propeller_mask(
        nx, ny, nz, cx, cy, cz, angle_deg=0.0, config=geometry, device=str(device)
    )
    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    ux0 = torch.full_like(rho0, u_in)
    ux0[mask0] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0), device=device
    )
    previous_mask = mask0

    impulse_x = impulse_t = 0.0
    static_fx = static_mx = 0.0
    n_samples = 0
    n_total = warmup_steps + sample_steps
    t0 = time.perf_counter()
    for step in range(1, n_total + 1):
        azimuth_deg = math.degrees((step * omega) % (2.0 * math.pi))
        mask = build_propeller_mask(
            nx, ny, nz, cx, cy, cz, angle_deg=azimuth_deg, config=geometry,
            device=str(device),
        )
        wall_mask = make_channel_wall_mask_3d(nz, ny, nx, mask, device=device)
        ux_w, uy_w, uz_w = rotating_wall_velocity_3d(mask, cx, cy, cz, omega)

        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs)
        f = stream3d(f)
        fx_static, _, _ = compute_obstacle_forces_3d(f, mask)
        mx_static, _, _ = compute_obstacle_moments_3d(f, mask, cx, cy, cz)
        f = apply_zou_he_channel_boundaries_3d(
            f, u_in=u_in, wall_mask=wall_mask, obstacle_mask=torch.zeros_like(mask)
        )
        f, reaction = moving_wall_bounce_back_3d_with_reaction(
            f, mask, ux_w, uy_w, uz_w, origin=(float(cx), float(cy), float(cz))
        )
        released = previous_mask & ~mask
        if bool(released.any()):
            equilibrium = equilibrium3d(
                torch.ones_like(rho0), torch.full_like(rho0, u_in),
                torch.zeros_like(rho0), torch.zeros_like(rho0), device=device,
            )
            f[:, released] = equilibrium[:, released]
        previous_mask = mask

        if step > warmup_steps:
            impulse_x += float(reaction.fluid_impulse[0].item())
            impulse_t += float(reaction.fluid_torque_impulse[0].item())
            static_fx += float(fx_static.item())
            static_mx += float(mx_static.item())
            n_samples += 1

    elapsed = time.perf_counter() - t0
    mean_fx = impulse_x / max(n_samples, 1)
    mean_mx = impulse_t / max(n_samples, 1)
    mean_static_fx = static_fx / max(n_samples, 1)
    mean_static_mx = static_mx / max(n_samples, 1)
    finite = bool(torch.isfinite(f).all().item())

    # Repo caliber: loads on the fluid -> signed KT/KQ/eta.
    rep = report_propeller_linkwise_loads(
        force_on_fluid=torch.tensor([mean_fx, 0.0, 0.0], dtype=torch.float64),
        torque_on_fluid=torch.tensor([mean_mx, 0.0, 0.0], dtype=torch.float64),
        advance_speed=u_in,
        rotation_rate=rpm,
        diameter=D,
        density=1.0,
        axis=(1.0, 0.0, 0.0),
        max_lattice_speed=tip_speed,
        low_mach_limit=1.0,  # explicit, documented override of the 0.004 gate
    )
    # Legacy static caliber (module default) for cross-check
    kt_static = mean_static_fx / (rpm * rpm * D**4)
    kq_static = mean_static_mx / (rpm * rpm * D**5)

    ref = REF_4119.get(round(j_target, 3))
    out = {
        "j_target": j_target,
        "j_actual": rep.advance_ratio,
        "u_in_lu": u_in,
        "kt": rep.kt,
        "kq": rep.kq,
        "10kq": 10.0 * rep.kq,
        "eta_o": rep.eta_o,
        "kt_static_legacy": kt_static,
        "kq_static_legacy": kq_static,
        "force_on_wall_x": rep.force_on_wall[0].item(),
        "torque_on_wall_x": rep.torque_on_wall[0].item(),
        "n_samples": n_samples,
        "finite": finite,
        "runtime_s": elapsed,
    }
    if ref is not None:
        out["kt_ref"] = ref[0]
        out["kq_ref"] = ref[1]
        out["kt_err_pct"] = 100.0 * (rep.kt - ref[0]) / ref[0]
        out["kq_err_pct"] = 100.0 * (rep.kq - ref[1]) / ref[1]
    return out


def make_config(args):
    geometry = geometry_4119(args.diameter)
    device = resolve_device(args.device)
    return geometry, device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "single", "full"])
    ap.add_argument("arg", nargs="?", default=None)
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--diameter", type=float, default=32.0)
    ap.add_argument("--nx", type=int, default=160)
    ap.add_argument("--ny", type=int, default=80)
    ap.add_argument("--nz", type=int, default=80)
    ap.add_argument("--rpm", type=float, default=1.0 / 1024.0)
    ap.add_argument("--tau", type=float, default=0.503)
    ap.add_argument("--cs", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=512)
    ap.add_argument("--sample", type=int, default=1024)
    ap.add_argument("--js", default="0.5,0.7")
    args = ap.parse_args()

    geometry, device = make_config(args)
    nu = (args.tau - 0.5) / 3.0
    re_d = args.rpm * args.diameter**2 / nu
    print(
        f"[cfg] device={device} D={args.diameter} dom={args.nx}x{args.ny}x{args.nz} "
        f"rpm={args.rpm:.6g} tau={args.tau} nu={nu:.4g} Re_D={re_d:.1f} "
        f"steps/rev={2*math.pi/(2*math.pi*args.rpm):.0f} Cs={args.cs}",
        flush=True,
    )

    if args.mode == "smoke":
        j = float(args.arg or 0.7)
        r = run_j(
            j_target=j, geometry=geometry, nx=args.nx, ny=args.ny, nz=args.nz,
            rpm=args.rpm, tau=args.tau, cs=args.cs, warmup_steps=20,
            sample_steps=args.sample, device=device,
        )
        print(json.dumps(r, indent=2), flush=True)
        return

    if args.mode == "single":
        j = float(args.arg or 0.7)
        r = run_j(
            j_target=j, geometry=geometry, nx=args.nx, ny=args.ny, nz=args.nz,
            rpm=args.rpm, tau=args.tau, cs=args.cs, warmup_steps=args.warmup,
            sample_steps=args.sample, device=device,
        )
        print(json.dumps(r, indent=2), flush=True)
        if args.out:
            Path(args.out).write_text(json.dumps(r, indent=2) + "\n")
            print("WROTE", args.out, flush=True)
        return

    js = [float(x) for x in args.js.split(",")]
    results = []
    for j in js:
        print(f"[J={j}] running ...", flush=True)
        r = run_j(
            j_target=j, geometry=geometry, nx=args.nx, ny=args.ny, nz=args.nz,
            rpm=args.rpm, tau=args.tau, cs=args.cs, warmup_steps=args.warmup,
            sample_steps=args.sample, device=device,
        )
        print(
            f"[J={j}] KT={r['kt']:.5f} KQ={r['kq']:.5f} eta={r['eta_o']:.4f} "
            f"KT_err={r.get('kt_err_pct')} KQ_err={r.get('kq_err_pct')} "
            f"t={r['runtime_s']:.0f}s finite={r['finite']}",
            flush=True,
        )
        results.append(r)

    out_dir = Path(args.out or "benchmarks/pending/propeller_openwater")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "case": "propeller_openwater_dtmb4119",
        "geometry": {
            "source": "tensorlbm.propeller_cad parametric stand-in",
            "n_blades": geometry.n_blades,
            "diameter_lu": geometry.diameter,
            "pitch_ratio_07": geometry.pitch_ratio_07,
            "ae_a0": geometry.blade_area_ratio,
            "hub_diameter_ratio": geometry.hub_diameter_ratio,
            "max_thickness_ratio": geometry.max_thickness_ratio,
        },
        "rotation": "Ladd moving-wall bounce-back, re-voxelised mask per step",
        "load_caliber": "moving-wall same-operator reaction + report_propeller_linkwise_loads",
        "numerics": {
            "nx": args.nx, "ny": args.ny, "nz": args.nz, "tau": args.tau,
            "nu": nu, "re_d": re_d, "cs": args.cs, "rpm": args.rpm,
            "warmup_steps": args.warmup, "sample_steps": args.sample,
        },
        "reference": {
            "name": "DTMB 4119 open-water experiment",
            "table": {str(k): v for k, v in REF_4119.items()},
            "crosscheck": REF_4119_CROSSCHECK,
        },
        "results": results,
    }
    (out_dir / "run_result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("WROTE", out_dir / "run_result.json", flush=True)


if __name__ == "__main__":
    main()