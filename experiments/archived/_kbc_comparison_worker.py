"""Agent 11: KBC vs MRT+Smag on bare_hull 200³ 5000 steps — RUNNING-AVERAGE drag.

Tests wall_function approach.  Skip heavy BFL q-field (too slow for 200³ CPU).
Uses SDAA, 200³ grid, Re=2e6, 5000 steps, running-average drag.
"""
import json, math, sys, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
from tensorlbm.d3q19 import C as C19, equilibrium3d, macroscopic3d
from tensorlbm.suboff_cad import SuboffHullType, build_suboff_mask
from tensorlbm.suboff_resistance import _voxel_wetted_area
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.entropic_kbc import collide_kbc_d3q19

KAPPA = 0.41; B_CONST = 5.0
REF_CT = 0.00406  # MRT+Smag baseline at 200³

def wallfn(f, solid, nu, y_val=0.5):
    """Log-law wall function with pressure-face drag — matches _collision_worker.py."""
    fluid = ~solid; near = torch.zeros_like(solid)
    for ax, sgn in [(2,1),(2,-1),(1,1),(1,-1),(0,1),(0,-1)]:
        near |= torch.roll(solid, sgn, dims=ax) & fluid
    rho, ux, uy, uz = macroscopic3d(f)
    um = torch.sqrt(ux*ux + uy*uy + uz*uz).clamp(min=1e-12)
    ut = torch.sqrt(nu * um / y_val).clamp(min=1e-12)
    yp = y_val * ut / nu; turb = (yp > 11.6) & near
    if turb.any():
        uu = ut[turb].clone(); vm = um[turb]
        for _ in range(8):
            ly = torch.log(y_val * uu / nu)
            fv = uu * (ly/KAPPA + B_CONST) - vm
            fp = (ly/KAPPA + B_CONST) + 1.0/KAPPA
            uu = (uu - fv/fp.clamp(min=1e-10)).clamp(min=1e-12)
        ut[turb] = uu
    tw = ut * ut; ium = 1.0 / um
    coef = -(tw / y_val) * near.to(f.dtype)
    fx = coef*(ux*ium); fy = coef*(uy*ium); fz = coef*(uz*ium)
    device = f.device
    c19 = C19.to(device).float()
    cx19 = c19[:, 0].view(19, 1, 1, 1)
    cy19 = c19[:, 1].view(19, 1, 1, 1)
    cz19 = c19[:, 2].view(19, 1, 1, 1)
    w19 = torch.tensor([1/3] + [1/18]*6 + [1/36]*12, dtype=f.dtype, device=device).view(19, 1, 1, 1)
    cs2 = 1.0 / 3.0
    cu = cx19*ux + cy19*uy + cz19*uz
    forcing = w19 * (1.0 + cu / cs2) * (cx19*fx + cy19*fy + cz19*fz) / cs2
    f = f + forcing
    df = (tw * (ux*ium) * near.to(f.dtype)).sum().item()
    p = (rho - 1.0) / 3.0
    sp = torch.roll(solid, 1, dims=2); sm = torch.roll(solid, -1, dims=2)
    dp = (p * (sp.to(f.dtype) - sm.to(f.dtype)) * fluid.to(f.dtype)).sum().item()
    return f, df, dp


def run_wallfn(label, collision, f, solid, tau, nu, u_in, dpS, n_steps, device):
    """Run with wall_function approach (same as collision_worker)."""
    print(f"\n{'='*70}")
    print(f"  [{label}] START")
    print(f"{'='*70}")
    
    warmup = n_steps // 3
    fric, pres = [], []
    t0 = time.time()
    im = float(torch.ones_like(solid, dtype=f.dtype).sum().item())
    final_step = 0
    step_times = []
    
    for step in range(1, n_steps + 1):
        t_step = time.time()
        try:
            if collision == "KBC":
                f = collide_kbc_d3q19(f, tau)
            elif collision == "MRT+Smag":
                f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=0.05)
            else:
                raise ValueError(collision)
        except Exception as e:
            print(f"  [{label}] COLLISION ERROR at step {step}: {e}")
            break
        
        f = stream3d(f)
        f, df, dp = wallfn(f, solid, nu, y_val=0.5)
        f = far_field_bc_3d(f, u_in=u_in)
        if step % 100 == 0:
            f = correct_mass3d(f, im)
        
        if step > warmup and math.isfinite(df):
            fric.append(df); pres.append(dp)
        final_step = step
        
        if not torch.isfinite(f).all():
            print(f"  [{label}] DIVERGED at step {step}")
            break
        
        step_times.append(time.time() - t_step)
        
        if step % 500 == 0:
            n_rec = len(fric)
            cf = (sum(fric) / max(n_rec, 1)) / dpS if n_rec > 0 else 0
            cp = (sum(pres) / max(n_rec, 1)) / dpS if n_rec > 0 else 0
            ct = cf + cp
            elapsed = time.time() - t0
            avg_ms = sum(step_times[-200:]) / max(len(step_times[-200:]), 1) * 1000
            print(f"  [{label}] step={step:5d} Ct={ct:.5f} (Cf={cf:.5f} Cp={cp:.5f}) "
                  f"({elapsed:.0f}s, {avg_ms:.0f}ms/step)")
    
    n_rec = len(fric)
    cf = (sum(fric) / max(n_rec, 1)) / dpS if n_rec > 0 else 0
    cp = (sum(pres) / max(n_rec, 1)) / dpS if n_rec > 0 else 0
    ct = cf + cp
    err = abs(ct - REF_CT) / REF_CT * 100
    
    elapsed = time.time() - t0
    finite = bool(torch.isfinite(f).all().item())
    
    result = {
        "label": label,
        "collision": collision,
        "boundary": "wall_function",
        "Ct_fric": cf, "Ct_pres": cp, "Ct_total": ct,
        "error_vs_REF_pct": err,
        "steps_completed": final_step,
        "finite": finite,
        "elapsed_s": elapsed,
        "avg_ms_per_step": elapsed / max(final_step, 1) * 1000,
        "drag_samples": n_rec,
        "grid": f"{f.shape[3]}x{f.shape[2]}x{f.shape[1]}",
    }
    print(f"  [{label}] DONE Ct={ct:.5f} err={err:.1f}% finite={finite} ({elapsed:.0f}s)")
    return result


def run_bfl_fast(label, collision, f, solid, tau, nu, u_in, dpS, n_steps, device):
    """Run with BFL ellipsoid approximation (faster than SUBOFF q-field)."""
    print(f"\n{'='*70}")
    print(f"  [{label}] START (BFL ellipsoid)")
    print(f"{'='*70}")
    
    from tensorlbm.interpolated_bc import bouzidi_bounce_back_3d
    from tensorlbm.interpolated_bc_ellipsoid import compute_q_ellipsoid
    from tensorlbm.d3q19 import OPPOSITE as OPP19
    from tensorlbm.suboff_cad import SuboffConfig
    
    OPP_LIST = [int(x) for x in OPP19.tolist()]
    
    nz, ny, nx = solid.shape
    
    # Compute BFL q-field using fast ellipsoid approximation
    config = SuboffConfig()
    a_semi = 80.0 / 2.0  # hull_length / 2
    b_semi = config.r_over_l * 80.0
    cx_g = nx * 0.35; cy_g = ny / 2.0; cz_g = nz / 2.0
    
    print(f"  [{label}] Computing BFL ellipsoid q-field (a={a_semi:.1f}, b={b_semi:.2f})...")
    t_q = time.time()
    bfl_mask, bfl_q = compute_q_ellipsoid(
        nx, ny, nz, cx_g, cy_g, cz_g,
        a_semi, b_semi, alpha_deg=0.0, device=device,
    )
    n_links = int(bfl_mask.sum().item())
    print(f"  [{label}] Q-field ready: {n_links} boundary links ({time.time()-t_q:.1f}s)")
    
    warmup = n_steps // 3
    fric_bfl, pres_bfl = [], []
    t0 = time.time()
    im = float(torch.ones_like(solid, dtype=f.dtype).sum().item())
    final_step = 0
    
    for step in range(1, n_steps + 1):
        try:
            if collision == "KBC":
                f = collide_kbc_d3q19(f, tau)
            elif collision == "MRT+Smag":
                f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=0.05)
            else:
                raise ValueError(collision)
        except Exception as e:
            print(f"  [{label}] COLLISION ERROR at step {step}: {e}")
            break
        
        f_pre = f.clone()
        f = stream3d(f)
        f = far_field_bc_3d(f, u_in=u_in)
        
        # BFL interpolated bounce-back
        for d in range(1, 19):
            if bfl_mask[d].any():
                f = bouzidi_bounce_back_3d(f, f_pre, bfl_mask[d], bfl_q[d], d)
        
        # Wall function for drag computation + near-wall forcing
        f, df, dp = wallfn(f, solid, nu, y_val=0.5)
        
        if step % 100 == 0:
            f = correct_mass3d(f, im)
        
        if step > warmup and math.isfinite(df):
            fric_bfl.append(df); pres_bfl.append(dp)
        final_step = step
        
        if not torch.isfinite(f).all():
            print(f"  [{label}] DIVERGED at step {step}")
            break
        
        if step % 500 == 0:
            n_rec = len(fric_bfl)
            cf = (sum(fric_bfl) / max(n_rec, 1)) / dpS if n_rec > 0 else 0
            cp = (sum(pres_bfl) / max(n_rec, 1)) / dpS if n_rec > 0 else 0
            ct = cf + cp
            elapsed = time.time() - t0
            print(f"  [{label}] step={step:5d} Ct={ct:.5f} (Cf={cf:.5f} Cp={cp:.5f}) "
                  f"({elapsed:.0f}s)")
    
    n_rec = len(fric_bfl)
    cf = (sum(fric_bfl) / max(n_rec, 1)) / dpS if n_rec > 0 else 0
    cp = (sum(pres_bfl) / max(n_rec, 1)) / dpS if n_rec > 0 else 0
    ct = cf + cp
    err = abs(ct - REF_CT) / REF_CT * 100
    
    elapsed = time.time() - t0
    finite = bool(torch.isfinite(f).all().item())
    
    result = {
        "label": label,
        "collision": collision,
        "boundary": "BFL_ellipsoid",
        "Ct_fric": cf, "Ct_pres": cp, "Ct_total": ct,
        "error_vs_REF_pct": err,
        "steps_completed": final_step,
        "finite": finite,
        "elapsed_s": elapsed,
        "avg_ms_per_step": elapsed / max(final_step, 1) * 1000,
        "drag_samples": n_rec,
        "grid": f"{f.shape[3]}x{f.shape[2]}x{f.shape[1]}",
        "bfl_links": n_links,
    }
    print(f"  [{label}] DONE Ct={ct:.5f} err={err:.1f}% finite={finite} ({elapsed:.0f}s)")
    return result


def main():
    nx = ny = nz = 200
    hull_length = 80.0
    u_in, re = 0.06, 2e6
    n_steps = 5000
    
    nu = u_in * hull_length / re
    tau = 3.0 * nu + 0.5
    
    device_str = "sdaa:4"
    device = torch.device(device_str)
    torch.sdaa.set_device(device)
    
    print(f"Agent 11: KBC vs MRT+Smag on bare_hull {nx}³, Re={re:.0e}, {n_steps} steps")
    print(f"tau={tau:.6f} nu={nu:.2e} hull_length={hull_length}")
    
    # Build geometry on CPU, move to SDAA
    cx, cy, cz = nx * 0.35, ny / 2.0, nz / 2.0
    solid_cpu, _ = build_suboff_mask(
        hull_type=SuboffHullType.BARE_HULL,
        nx=nx, ny=ny, nz=nz, cx=cx, cy=cy, cz=cz,
        length=hull_length, device="cpu",
    )
    solid = solid_cpu.to(device)
    del solid_cpu  # free CPU memory
    
    S = _voxel_wetted_area(solid, 1.0)
    dpS = 0.5 * 1.0 * u_in ** 2 * S
    print(f"S={S:.0f} dpS={dpS:.6f}")
    
    results = []
    
    # Initialise common f
    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device); ux0[solid] = 0
    f0 = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    del rho0, ux0  # free memory
    
    # Test 1: MRT+Smag wall_function (BASELINE)
    r1 = run_wallfn("MRT+Smag_WF", "MRT+Smag",
                    f0.clone(), solid, tau, nu, u_in, dpS, n_steps, device)
    results.append(r1)
    
    # Test 2: KBC wall_function
    r2 = run_wallfn("KBC_WF", "KBC",
                    f0.clone(), solid, tau, nu, u_in, dpS, n_steps, device)
    results.append(r2)
    
    # Test 3: KBC BFL (ellipsoid approximation, faster)
    try:
        r3 = run_bfl_fast("KBC_BFL", "KBC",
                         f0.clone(), solid, tau, nu, u_in, dpS, n_steps, device)
        results.append(r3)
    except Exception as e:
        print(f"KBC_BFL FAILED: {e}")
        traceback.print_exc()
        results.append({"label": "KBC_BFL", "collision": "KBC", "boundary": "BFL_ellipsoid",
                       "error": str(e)})
    
    # Test 4: MRT+Smag BFL
    try:
        r4 = run_bfl_fast("MRT+Smag_BFL", "MRT+Smag",
                         f0.clone(), solid, tau, nu, u_in, dpS, n_steps, device)
        results.append(r4)
    except Exception as e:
        print(f"MRT+Smag_BFL FAILED: {e}")
        traceback.print_exc()
        results.append({"label": "MRT+Smag_BFL", "collision": "MRT+Smag", "boundary": "BFL_ellipsoid",
                       "error": str(e)})
    
    # Print comparison
    print("\n" + "=" * 90)
    print("KBC vs MRT+Smag COMPARISON — bare_hull 200³ 5000 steps running-average")
    print(f"REF Ct = {REF_CT}")
    print("=" * 90)
    print(f"{'Label':<20} {'Collision':<12} {'BC':<14} {'Ct_fric':<10} {'Ct_pres':<10} {'Ct_total':<10} {'Err%':<8} {'OK':<5}")
    print("-" * 90)
    for r in results:
        if "error" in r:
            print(f"{r['label']:<20} ERROR: {r['error']}")
        else:
            ok = "✓" if r["finite"] else "✗"
            print(f"{r['label']:<20} {r['collision']:<12} {r['boundary']:<14} "
                  f"{r['Ct_fric']:<10.5f} {r['Ct_pres']:<10.5f} {r['Ct_total']:<10.5f} "
                  f"{r['error_vs_REF_pct']:<8.1f} {ok:<5}")
    
    # Write JSON
    output_path = Path("/tmp/kbc_comparison.json")
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
