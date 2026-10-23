"""Ship BFL Agent: Test BFL vs staircase on KVLCC2 ship hull.

SDAA:30 = staircase (original wall_function)
SDAA:31 = BFL (ellipsoid q-field approximation, a=40.0, b=7.5)

200³ grid, Re=2e6, 3000 steps, warmup=1000, running-average.
Results → /tmp/ship_bfl_results.json
"""
import json
import math
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C as C19, OPPOSITE as OPP19
from tensorlbm.ship_cad import ShipHullType, build_hull_mask
from tensorlbm.suboff_resistance import _ittc57_friction_coefficient, _voxel_wetted_area
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d

# ---------------------------------------------------------------------------
# Wall function (inline, matches ship_bench_worker)
# ---------------------------------------------------------------------------
KAPPA = 0.41
B_CONST = 5.0


def wallfn(f, solid, nu, y_val=0.5):
    """Log-law wall function, returns (f_updated, df, dp)."""
    fluid = ~solid
    near = torch.zeros_like(solid)
    for ax, sgn in [(2, 1), (2, -1), (1, 1), (1, -1), (0, 1), (0, -1)]:
        near |= torch.roll(solid, sgn, dims=ax) & fluid

    rho, ux, uy, uz = macroscopic3d(f)
    um = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)
    ut = torch.sqrt(nu * um / y_val).clamp(min=1e-12)
    yp = y_val * ut / nu
    turb = (yp > 11.6) & near

    if turb.any():
        uu = ut[turb].clone()
        vm = um[turb]
        for _ in range(8):
            ly = torch.log(y_val * uu / nu)
            fv = uu * (ly / KAPPA + B_CONST) - vm
            fp = (ly / KAPPA + B_CONST) + 1.0 / KAPPA
            uu = (uu - fv / fp.clamp(min=1e-10)).clamp(min=1e-12)
        ut[turb] = uu

    tw = ut * ut
    ium = 1.0 / um
    coef = -(tw / y_val) * near.to(f.dtype)
    fx = coef * (ux * ium)
    fy = coef * (uy * ium)
    fz = coef * (uz * ium)

    device = f.device
    c19 = C19.to(device).float()
    cx19 = c19[:, 0].view(19, 1, 1, 1)
    cy19 = c19[:, 1].view(19, 1, 1, 1)
    cz19 = c19[:, 2].view(19, 1, 1, 1)
    w19 = torch.tensor(
        [1 / 3] + [1 / 18] * 6 + [1 / 36] * 12,
        dtype=f.dtype, device=device,
    ).view(19, 1, 1, 1)
    cs2 = 1.0 / 3.0
    cu = cx19 * ux + cy19 * uy + cz19 * uz
    forcing = w19 * (1.0 + cu / cs2) * (cx19 * fx + cy19 * fy + cz19 * fz) / cs2
    f = f + forcing

    df = (tw * (ux * ium) * near.to(f.dtype)).sum().item()
    p = (rho - 1.0) / 3.0
    sp = torch.roll(solid, 1, dims=2)
    sm = torch.roll(solid, -1, dims=2)
    dp = (p * (sp.to(f.dtype) - sm.to(f.dtype)) * fluid.to(f.dtype)).sum().item()
    return f, df, dp


# ---------------------------------------------------------------------------
# Hull form factor
# ---------------------------------------------------------------------------
_FORM_FACTORS = {
    ShipHullType.WIGLEY: 1.15,
    ShipHullType.SERIES60: 1.18,
    ShipHullType.KCS: 1.20,
    ShipHullType.KVLCC2: 1.25,
    ShipHullType.NPL: 1.10,
}


# ---------------------------------------------------------------------------
# Staircase benchmark (original wall_function approach)
# ---------------------------------------------------------------------------
def run_staircase(device_str, nx, ny, nz, hull_length, u_in, re, n_steps, warmup, cs):
    device = torch.device(device_str)
    torch.sdaa.set_device(device)

    hull = ShipHullType.KVLCC2
    ff = _FORM_FACTORS[hull]

    # Lattice parameters
    nu_lat = u_in * hull_length / re
    tau = 3.0 * nu_lat + 0.5

    # Hull placement (same as ship_bench)
    cx = nx * 0.3
    cy = ny * 0.5
    cz_keel = nz * 0.5

    # Build hull mask
    solid, stats = build_hull_mask(
        hull, nx, ny, nz,
        cx=cx, cy=cy,
        cz_keel=cz_keel,
        length=hull_length,
        device="cpu",
    )
    solid = solid.to(device)
    S = _voxel_wetted_area(solid, 1.0)
    dyn_p_S = 0.5 * 1.0 * u_in ** 2 * S

    # ITTC reference
    cf_ittc = _ittc57_friction_coefficient(re)
    ct_ref = cf_ittc * ff

    # Initialize
    rho0 = torch.ones((nz, ny, nx))
    ux0 = torch.full((nz, ny, nx), u_in)
    uy0 = torch.zeros(nz, ny, nx)
    uz0 = torch.zeros(nz, ny, nx)
    ux0[solid.cpu()] = 0.0
    f = equilibrium3d(rho0, ux0, uy0, uz0)
    f = f.to(device)
    initial_mass = float(f.sum().item())

    fric_vals, pres_vals = [], []
    start_time = time.time()
    final_step = 0

    for step in range(1, n_steps + 1):
        # Collision
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs)

        # Stream
        f = stream3d(f)

        # Wall function
        f, df, dp = wallfn(f, solid, nu_lat, y_val=0.5)

        # Far-field
        f = far_field_bc_3d(f, u_in=u_in)

        # Mass correction
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        if step > warmup and math.isfinite(df):
            fric_vals.append(df)
            pres_vals.append(dp)

        final_step = step

        if not torch.isfinite(f).all():
            print(f"  [staircase] DIVERGED at step {step}", file=sys.stderr)
            break

        if step % 500 == 0 or step == n_steps:
            n_samp = len(fric_vals)
            if n_samp > 0:
                cf_avg = sum(fric_vals) / n_samp / dyn_p_S
                cp_avg = sum(pres_vals) / n_samp / dyn_p_S
                ct_avg = cf_avg + cp_avg
            else:
                cf_avg = cp_avg = ct_avg = 0.0
            elapsed = time.time() - start_time
            msg = (f"  [staircase sdaa:30] step {step:4d}: Ct_fric={cf_avg:.5f} "
                   f"Ct_pres={cp_avg:.5f} Ct_tot={ct_avg:.5f} (ref {ct_ref:.5f}) "
                   f"elapsed={elapsed:.0f}s")
            print(msg, flush=True)
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()

    elapsed = time.time() - start_time
    n_samp = len(fric_vals)
    cf = sum(fric_vals) / max(n_samp, 1) / dyn_p_S if n_samp > 0 else 0.0
    cp = sum(pres_vals) / max(n_samp, 1) / dyn_p_S if n_samp > 0 else 0.0
    ct = cf + cp
    err_pct = abs(ct - ct_ref) / ct_ref * 100 if ct_ref > 0 else float("inf")

    return {
        "label": "KVLCC2_staircase_sdaa30",
        "boundary": "staircase_wallfn",
        "device": device_str,
        "Ct_fric": cf, "Ct_pres": cp, "Ct_total": ct,
        "Ct_reference": ct_ref, "error_pct": err_pct,
        "steps_completed": final_step,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
        "drag_samples": n_samp,
        "wetted_area": S,
        "nu_lattice": nu_lat, "tau": tau,
    }


# ---------------------------------------------------------------------------
# BFL benchmark (ellipsoid q-field + wall function for drag)
# ---------------------------------------------------------------------------
def run_bfl(device_str, nx, ny, nz, hull_length, u_in, re, n_steps, warmup, cs):
    device = torch.device(device_str)
    torch.sdaa.set_device(device)

    from tensorlbm.interpolated_bc import bouzidi_bounce_back_3d
    from tensorlbm.interpolated_bc_ellipsoid import compute_q_ellipsoid

    hull = ShipHullType.KVLCC2
    ff = _FORM_FACTORS[hull]

    # Lattice parameters
    nu_lat = u_in * hull_length / re
    tau = 3.0 * nu_lat + 0.5

    # Hull placement
    cx = nx * 0.3
    cy = ny * 0.5
    cz_keel = nz * 0.5

    # Build hull mask (for wetted area + wall function drag)
    solid, stats = build_hull_mask(
        hull, nx, ny, nz,
        cx=cx, cy=cy,
        cz_keel=cz_keel,
        length=hull_length,
        device="cpu",
    )
    solid = solid.to(device)
    S = _voxel_wetted_area(solid, 1.0)
    dyn_p_S = 0.5 * 1.0 * u_in ** 2 * S

    # ITTC reference
    cf_ittc = _ittc57_friction_coefficient(re)
    ct_ref = cf_ittc * ff

    # Compute BFL q-field using ellipsoid approximation
    # KVLCC2: L=80, beam≈15 → a=40.0 (semi-major), b=7.5 (semi-minor)
    a_semi = 40.0
    b_semi = 7.5
    # Ellipsoid center matches hull center
    cx_ell = cx  # same x-center as hull
    cy_ell = cy  # same y-center as hull
    # Vertical center: keel + draft/2. Default draft = nz*0.3 = 18, so center ≈ cz_keel+9 = 39
    draft_default = nz * 0.3
    cz_ell = cz_keel + draft_default / 2.0

    print(f"  [BFL sdaa:31] Computing ellipsoid q-field "
          f"(a={a_semi:.1f}, b={b_semi:.2f}, center=({cx_ell:.0f},{cy_ell:.0f},{cz_ell:.0f}))...",
          flush=True)
    sys.stderr.write(f"  [BFL sdaa:31] Computing ellipsoid q-field "
                     f"(a={a_semi:.1f}, b={b_semi:.2f}, center=({cx_ell:.0f},{cy_ell:.0f},{cz_ell:.0f}))...\n")
    sys.stderr.flush()
    t_q = time.time()
    bfl_mask, bfl_q = compute_q_ellipsoid(
        nx, ny, nz,
        cx_ell, cy_ell, cz_ell,
        a_semi, b_semi,
        alpha_deg=0.0,
        device=device,
    )
    n_links = int(bfl_mask.sum().item())
    print(f"  [BFL sdaa:31] Q-field ready: {n_links} boundary links "
          f"({time.time() - t_q:.1f}s)", flush=True)
    sys.stderr.write(f"  [BFL sdaa:31] Q-field ready: {n_links} boundary links "
                     f"({time.time() - t_q:.1f}s)\n")
    sys.stderr.flush()

    # Initialize
    rho0 = torch.ones((nz, ny, nx))
    ux0 = torch.full((nz, ny, nx), u_in)
    uy0 = torch.zeros(nz, ny, nx)
    uz0 = torch.zeros(nz, ny, nx)
    ux0[solid.cpu()] = 0.0
    f = equilibrium3d(rho0, ux0, uy0, uz0)
    f = f.to(device)
    initial_mass = float(f.sum().item())

    fric_vals, pres_vals = [], []
    start_time = time.time()
    final_step = 0

    for step in range(1, n_steps + 1):
        # Collision
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs)

        # Save pre-stream for BFL
        f_pre = f.clone()

        # Stream
        f = stream3d(f)

        # Far-field
        f = far_field_bc_3d(f, u_in=u_in)

        # BFL interpolated bounce-back on all boundary links
        for d in range(1, 19):
            if bfl_mask[d].any():
                f = bouzidi_bounce_back_3d(f, f_pre, bfl_mask[d], bfl_q[d], d)

        # Wall function for drag computation
        f, df, dp = wallfn(f, solid, nu_lat, y_val=0.5)

        # Mass correction
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        if step > warmup and math.isfinite(df):
            fric_vals.append(df)
            pres_vals.append(dp)

        final_step = step

        if not torch.isfinite(f).all():
            print(f"  [BFL sdaa:31] DIVERGED at step {step}", file=sys.stderr)
            break

        if step % 500 == 0 or step == n_steps:
            n_samp = len(fric_vals)
            if n_samp > 0:
                cf_avg = sum(fric_vals) / n_samp / dyn_p_S
                cp_avg = sum(pres_vals) / n_samp / dyn_p_S
                ct_avg = cf_avg + cp_avg
            else:
                cf_avg = cp_avg = ct_avg = 0.0
            elapsed = time.time() - start_time
            msg = (f"  [BFL sdaa:31] step {step:4d}: Ct_fric={cf_avg:.5f} "
                   f"Ct_pres={cp_avg:.5f} Ct_tot={ct_avg:.5f} (ref {ct_ref:.5f}) "
                   f"elapsed={elapsed:.0f}s")
            print(msg, flush=True)
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()

    elapsed = time.time() - start_time
    n_samp = len(fric_vals)
    cf = sum(fric_vals) / max(n_samp, 1) / dyn_p_S if n_samp > 0 else 0.0
    cp = sum(pres_vals) / max(n_samp, 1) / dyn_p_S if n_samp > 0 else 0.0
    ct = cf + cp
    err_pct = abs(ct - ct_ref) / ct_ref * 100 if ct_ref > 0 else float("inf")

    return {
        "label": "KVLCC2_BFL_sdaa31",
        "boundary": "BFL_ellipsoid+wallfn",
        "device": device_str,
        "ellipsoid_a": a_semi, "ellipsoid_b": b_semi,
        "ellipsoid_center": [cx_ell, cy_ell, cz_ell],
        "bfl_links": n_links,
        "Ct_fric": cf, "Ct_pres": cp, "Ct_total": ct,
        "Ct_reference": ct_ref, "error_pct": err_pct,
        "steps_completed": final_step,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
        "drag_samples": n_samp,
        "wetted_area": S,
        "nu_lattice": nu_lat, "tau": tau,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    nx, ny, nz = 200, 60, 60
    hull_length = 80.0
    u_in, re = 0.06, 2_000_000
    n_steps = 3000
    warmup = 1000
    cs = 0.05

    print("=" * 70, flush=True)
    print("Ship BFL Agent: KVLCC2 Staircase vs BFL", flush=True)
    print(f"Grid: {nx}×{ny}×{nz}  Re={re:.0e}  hull_length={hull_length}  "
          f"{n_steps} steps (warmup={warmup})", flush=True)
    print("SDAA:30 = staircase   SDAA:31 = BFL (ellipsoid a=40, b=7.5)", flush=True)
    print("=" * 70, flush=True)

    results = []

    # Test 1: Staircase on SDAA:30
    print("\n>>> Running STAIRCASE on SDAA:30 <<<", file=sys.stderr, flush=True)
    try:
        r1 = run_staircase("sdaa:30", nx, ny, nz, hull_length, u_in, re, n_steps, warmup, cs)
        results.append(r1)
        print(f"  [staircase] DONE Ct={r1['Ct_total']:.5f} err={r1['error_pct']:.1f}% "
              f"finite={r1['finite']} ({r1['elapsed_s']:.0f}s)", file=sys.stderr)
    except Exception as e:
        print(f"  [staircase] FAILED: {e}", file=sys.stderr)
        traceback.print_exc()
        results.append({"label": "KVLCC2_staircase_sdaa30", "boundary": "staircase_wallfn",
                        "status": "FAILED", "error": str(e)})

    # Test 2: BFL on SDAA:31
    print("\n>>> Running BFL on SDAA:31 <<<", file=sys.stderr, flush=True)
    try:
        r2 = run_bfl("sdaa:31", nx, ny, nz, hull_length, u_in, re, n_steps, warmup, cs)
        results.append(r2)
        print(f"  [BFL] DONE Ct={r2['Ct_total']:.5f} err={r2['error_pct']:.1f}% "
              f"finite={r2['finite']} ({r2['elapsed_s']:.0f}s)", file=sys.stderr)
    except Exception as e:
        print(f"  [BFL] FAILED: {e}", file=sys.stderr)
        traceback.print_exc()
        results.append({"label": "KVLCC2_BFL_sdaa31", "boundary": "BFL_ellipsoid+wallfn",
                        "status": "FAILED", "error": str(e)})

    # Print comparison
    print("\n" + "=" * 90, file=sys.stderr)
    print("KVLCC2 SHIP HULL: Staircase vs BFL Comparison", file=sys.stderr)
    print(f"Grid: {nx}×{ny}×{nz}  Re={re:.0e}  hull_length={hull_length}  "
          f"{n_steps} steps  warmup={warmup}", file=sys.stderr)
    print("=" * 90, file=sys.stderr)
    print(f"{'Label':<30} {'BC':<25} {'Ct_fric':<10} {'Ct_pres':<10} "
          f"{'Ct_total':<10} {'Err%':<8} {'OK':<5}", file=sys.stderr)
    print("-" * 90, file=sys.stderr)
    for r in results:
        if "error" in r and "Ct_total" not in r:
            print(f"{r['label']:<30} ERROR: {r['error']}", file=sys.stderr)
        else:
            ok = "✓" if r.get("finite") else "✗"
            print(f"{r['label']:<30} {r['boundary']:<25} "
                  f"{r.get('Ct_fric', 0):<10.5f} {r.get('Ct_pres', 0):<10.5f} "
                  f"{r.get('Ct_total', 0):<10.5f} {r.get('error_pct', 0):<8.1f} "
                  f"{ok:<5}", file=sys.stderr)

    # Determine which is better
    print(file=sys.stderr)
    stair = next((r for r in results if "staircase" in r.get("label", "")), None)
    bfl_r = next((r for r in results if "BFL" in r.get("label", "")), None)

    if stair and bfl_r and stair.get("error_pct") is not None and bfl_r.get("error_pct") is not None:
        if bfl_r["error_pct"] < stair["error_pct"]:
            improvement = stair["error_pct"] - bfl_r["error_pct"]
            print(f"✓ BFL IMPROVES KVLCC2 drag prediction: "
                  f"error {stair['error_pct']:.1f}% → {bfl_r['error_pct']:.1f}% "
                  f"({improvement:.1f}pp reduction)", file=sys.stderr)
        else:
            print(f"✗ BFL does NOT improve KVLCC2: "
                  f"error {stair['error_pct']:.1f}% → {bfl_r['error_pct']:.1f}%",
                  file=sys.stderr)

    # Build summary
    summary = {
        "config": {
            "hull": "KVLCC2",
            "lattice": "D3Q19",
            "collision": "MRT+Smagorinsky",
            "C_s": cs,
            "grid": f"{nx}×{ny}×{nz}",
            "Re": re,
            "hull_length": hull_length,
            "u_in": u_in,
            "n_steps": n_steps,
            "warmup": warmup,
        },
        "results": results,
        "comparison": {
            "staircase_error_pct": stair.get("error_pct") if stair else None,
            "bfl_error_pct": bfl_r.get("error_pct") if bfl_r else None,
            "bfl_improves": (
                bfl_r.get("error_pct", float("inf")) < stair.get("error_pct", float("inf"))
                if stair and bfl_r else None
            ),
        },
    }

    # Write results
    out_path = Path("/tmp/ship_bfl_results.json")
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nResults written to {out_path}", file=sys.stderr)

    # Also print to stdout for capture
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
