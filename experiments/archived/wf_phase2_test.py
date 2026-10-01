#!/usr/bin/env python3
"""Wall function timing + BB interaction test — Phase 2.

Key improvements over Phase 1:
1. y+ threshold: WF only active when y+ > 11.6 (turbulent regime)
   - For low-Re Poiseuille/Couette, y+ < 11.6 → WF inactive → BB alone
2. No-WF baseline: BB only (confirm BB alone gives <5%)
3. Reduced force: WF provides CORRECTION, not total shear
   - F = -(τ_w - τ_BB) * û where τ_BB = ν*u/y_val (BB-provided shear)
   - When WF shear = BB shear, correction = 0 (no over-damping)
4. Guo forcing variant (with velocity correction term)
"""
import sys, os, json, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import torch
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C, W, OPPOSITE
from tensorlbm.solver3d import collide_bgk3d, stream3d
from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.ibm import ibm_apply_body_force_3d

# ---------------------------------------------------------------------------
# Grid / physics constants
# ---------------------------------------------------------------------------
NX, NY, NZ = 80, 12, 4
TAU = 1.0
NU = (TAU - 0.5) / 3.0
U_MAX_TARGET = 0.05
U_TOP = 0.05
H = NY - 2
G_DRIVE = 8.0 * NU * U_MAX_TARGET / (H * H)
Y_VAL = 0.5
CS2 = 1.0 / 3.0
Y_PLUS_THRESH = 11.6  # viscous-log transition

TAU_W_ANALYTICAL = G_DRIVE * H / 2.0
CF_ANALYTICAL_UMAX = 2.0 * TAU_W_ANALYTICAL / (1.0 * U_MAX_TARGET ** 2)


def build_masks(device):
    solid = torch.zeros((NZ, NY, NX), dtype=torch.bool, device=device)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    fluid = ~solid
    near = torch.zeros_like(solid)
    near[:, 1, :] = True
    near[:, -2, :] = True
    return solid, fluid, near


# ---------------------------------------------------------------------------
# Wall function variants
# ---------------------------------------------------------------------------

def compute_tau_w_gradient(ux, uz, near, nu, y_val=Y_VAL):
    """Gradient wall law: τ_w = ν * u_tan / y_val."""
    u_tan_mag = torch.sqrt(ux * ux + uz * uz).clamp(min=1e-12)
    tau_w = nu * u_tan_mag / y_val
    return torch.where(near, tau_w, torch.zeros_like(tau_w))


def compute_y_plus(tau_w, nu, y_val=Y_VAL):
    """y+ = y_val * u_tau / ν, where u_tau = sqrt(τ_w)."""
    u_tau = torch.sqrt(tau_w.clamp(min=1e-30))
    return y_val * u_tau / nu


def apply_wf_force(f, solid, near, nu, y_val=Y_VAL, yplus_thresh=0.0):
    """WF as body force: F = -τ_w * û. Optional y+ threshold."""
    rho, ux, uy, uz = macroscopic3d(f)
    tau_w = compute_tau_w_gradient(ux, uz, near, nu, y_val)
    if yplus_thresh > 0:
        yp = compute_y_plus(tau_w, nu, y_val)
        active = (yp > yplus_thresh) & near
    else:
        active = near
    u_tan_mag = torch.sqrt(ux * ux + uz * uz).clamp(min=1e-12)
    fx = -tau_w * (ux / u_tan_mag) * active.to(f.dtype)
    fz = -tau_w * (uz / u_tan_mag) * active.to(f.dtype)
    fy = torch.zeros_like(fx)
    return ibm_apply_body_force_3d(f, fx, fy, fz)


def apply_wf_force_guo(f, solid, near, nu, y_val=Y_VAL, yplus_thresh=0.0):
    """WF as Guo body force (with velocity correction term)."""
    rho, ux, uy, uz = macroscopic3d(f)
    tau_w = compute_tau_w_gradient(ux, uz, near, nu, y_val)
    if yplus_thresh > 0:
        yp = compute_y_plus(tau_w, nu, y_val)
        active = (yp > yplus_thresh) & near
    else:
        active = near
    u_tan_mag = torch.sqrt(ux * ux + uz * uz).clamp(min=1e-12)
    fx = -tau_w * (ux / u_tan_mag) * active.to(f.dtype)
    fz = -tau_w * (uz / u_tan_mag) * active.to(f.dtype)
    fy = torch.zeros_like(fx)
    # Guo forcing: w_i * (1 + c·u/cs²) * (c·F) / cs²
    c = C.to(f.device).float()
    w = W.to(f.device).float()
    cx = c[:, 0].view(19, 1, 1, 1)
    cy = c[:, 1].view(19, 1, 1, 1)
    cz = c[:, 2].view(19, 1, 1, 1)
    w_view = w.view(19, 1, 1, 1)
    cu_f = cx * fx.unsqueeze(0) + cy * fy.unsqueeze(0) + cz * fz.unsqueeze(0)
    cu_u = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0) + cz * uz.unsqueeze(0)
    forcing = w_view * (1.0 + cu_u / CS2) * cu_f / CS2
    return f + forcing


def apply_wf_force_correction(f, solid, near, nu, y_val=Y_VAL, yplus_thresh=0.0):
    """WF as CORRECTION force: F = -(τ_w - τ_BB) * û.

    τ_BB = ν * u_tan / y_val (same as gradient law → correction = 0).
    This variant uses the LOG-LAW τ_w instead, so the correction is
    τ_w_log - τ_w_gradient. For low Re (y+ < 11.6), τ_w_log ≈ τ_w_gradient
    → correction ≈ 0. For high Re, correction > 0.
    """
    rho, ux, uy, uz = macroscopic3d(f)
    u_tan_mag = torch.sqrt(ux * ux + uz * uz).clamp(min=1e-12)

    # Gradient law (what BB provides)
    tau_grad = nu * u_tan_mag / y_val
    # Log-law τ_w (Newton iteration)
    u_tau = torch.sqrt(nu * u_tan_mag / y_val).clamp(min=1e-12)
    y_plus = y_val * u_tau / nu
    turb = (y_plus > 11.6) & near
    if bool(turb.any()):
        ut = u_tau[turb].clone()
        um = u_tan_mag[turb]
        kappa, b_log = 0.41, 5.0
        for _ in range(8):
            lyp = torch.log(y_val * ut / nu)
            fv = ut * (lyp / kappa + b_log) - um
            fp = (lyp / kappa + b_log) + 1.0 / kappa
            ut = (ut - fv / fp.clamp(min=1e-10)).clamp(min=1e-12)
        u_tau[turb] = ut
    tau_log = u_tau * u_tau

    # Correction = τ_log - τ_grad (only where log-law is active)
    correction = torch.where(turb, tau_log - tau_grad, torch.zeros_like(tau_grad))
    if yplus_thresh > 0:
        yp = compute_y_plus(tau_grad, nu, y_val)
        correction = torch.where(yp > yplus_thresh, correction, torch.zeros_like(correction))

    fx = -correction * (ux / u_tan_mag)
    fz = -correction * (uz / u_tan_mag)
    fy = torch.zeros_like(fx)
    return ibm_apply_body_force_3d(f, fx, fy, fz)


def apply_wf_shifted(f, solid, near, nu, y_val=Y_VAL, shift_factor=1.0,
                     yplus_thresh=0.0):
    """WF as shifted velocity: u_shifted = u - shift_factor*τ_w/ρ * û."""
    rho, ux, uy, uz = macroscopic3d(f)
    tau_w = compute_tau_w_gradient(ux, uz, near, nu, y_val)
    if yplus_thresh > 0:
        yp = compute_y_plus(tau_w, nu, y_val)
        active = (yp > yplus_thresh) & near
    else:
        active = near
    u_tan_mag = torch.sqrt(ux * ux + uz * uz).clamp(min=1e-12)
    factor = shift_factor * tau_w / rho.clamp(min=1e-12) * active.to(f.dtype)
    ux_s = ux - factor * (ux / u_tan_mag)
    uz_s = uz - factor * (uz / u_tan_mag)
    feq_s = equilibrium3d(rho, ux_s, uy, uz_s)
    feq_c = equilibrium3d(rho, ux, uy, uz)
    delta = (feq_s - feq_c) * near.unsqueeze(0).to(f.dtype)
    return f + delta


# ---------------------------------------------------------------------------
# Bounce-back
# ---------------------------------------------------------------------------

def apply_bb_solid(f, solid):
    return bounce_back_cells_3d(f, solid)


def apply_moving_bb(f, solid_bottom, solid_top, u_top):
    opp = OPPOSITE.to(f.device)
    rho = f.sum(dim=0)
    c = C.to(f.device).float()
    w = W.to(f.device).float()
    cx = c[:, 0].view(19, 1, 1, 1)
    w_view = w.view(19, 1, 1, 1)
    f = torch.where(solid_bottom.unsqueeze(0), f[opp], f)
    cu_w = cx * u_top
    correction = 2.0 * w_view * rho.unsqueeze(0) * cu_w * 3.0
    f = torch.where(solid_top.unsqueeze(0), f[opp] + correction, f)
    return f


# ---------------------------------------------------------------------------
# Poiseuille runner
# ---------------------------------------------------------------------------

def run_poiseuille(device, wf_timing='post', use_bb=True, force_type='force',
                   n_steps=3000, shift_factor=1.0, yplus_thresh=0.0,
                   verbose=True):
    solid, fluid, near = build_masks(device)
    rho0 = torch.ones((NZ, NY, NX), device=device)
    u0 = torch.zeros((NZ, NY, NX), device=device)
    f = equilibrium3d(rho0, u0, u0.clone(), u0.clone())
    fx_drive = torch.zeros((NZ, NY, NX), device=device)
    fx_drive[fluid] = G_DRIVE

    for step in range(n_steps):
        f = collide_bgk3d(f, TAU)
        if use_bb:
            f = apply_bb_solid(f, solid)
        if wf_timing == 'pre':
            if force_type == 'force':
                f = apply_wf_force(f, solid, near, NU, yplus_thresh=yplus_thresh)
            elif force_type == 'force_guo':
                f = apply_wf_force_guo(f, solid, near, NU, yplus_thresh=yplus_thresh)
            elif force_type == 'correction':
                f = apply_wf_force_correction(f, solid, near, NU, yplus_thresh=yplus_thresh)
            elif force_type == 'shifted':
                f = apply_wf_shifted(f, solid, near, NU, shift_factor=shift_factor,
                                      yplus_thresh=yplus_thresh)
            elif force_type == 'none':
                pass
        f = stream3d(f)
        if wf_timing == 'post':
            if force_type == 'force':
                f = apply_wf_force(f, solid, near, NU, yplus_thresh=yplus_thresh)
            elif force_type == 'force_guo':
                f = apply_wf_force_guo(f, solid, near, NU, yplus_thresh=yplus_thresh)
            elif force_type == 'correction':
                f = apply_wf_force_correction(f, solid, near, NU, yplus_thresh=yplus_thresh)
            elif force_type == 'shifted':
                f = apply_wf_shifted(f, solid, near, NU, shift_factor=shift_factor,
                                      yplus_thresh=yplus_thresh)
            elif force_type == 'none':
                pass
        f = ibm_apply_body_force_3d(f, fx_drive, torch.zeros_like(fx_drive),
                                     torch.zeros_like(fx_drive))

    rho, ux, uy, uz = macroscopic3d(f)
    profile = ux[:, :, :].mean(dim=(0, 2)).cpu().numpy()
    u_max = float(profile[1:-1].max())
    u_err = abs(u_max - U_MAX_TARGET) / U_MAX_TARGET * 100.0
    du_dy = (profile[2] - profile[1])
    tau_w_measured = NU * abs(du_dy)
    cf_measured = 2.0 * tau_w_measured / (1.0 * u_max ** 2) if u_max > 1e-10 else 0.0

    # Also compute y+ at near-wall
    u_near = float(profile[1])
    u_tau_est = (NU * u_near / Y_VAL) ** 0.5
    y_plus_est = Y_VAL * u_tau_est / NU

    if verbose:
        print(f"  u_max={u_max:.6f} (target={U_MAX_TARGET}), u_err={u_err:.2f}%, "
              f"Cf={cf_measured:.4f} (anal={CF_ANALYTICAL_UMAX:.4f}), y+={y_plus_est:.4f}")
    return u_max, u_err, cf_measured, profile, y_plus_est


# ---------------------------------------------------------------------------
# Couette runner
# ---------------------------------------------------------------------------

def run_couette(device, wf_timing='post', use_bb=True, force_type='force',
                n_steps=3000, shift_factor=1.0, yplus_thresh=0.0,
                verbose=True):
    solid, fluid, near = build_masks(device)
    solid_bottom = torch.zeros_like(solid)
    solid_bottom[:, 0, :] = True
    solid_top = torch.zeros_like(solid)
    solid_top[:, -1, :] = True

    rho0 = torch.ones((NZ, NY, NX), device=device)
    u0 = torch.zeros((NZ, NY, NX), device=device)
    f = equilibrium3d(rho0, u0, u0.clone(), u0.clone())

    for step in range(n_steps):
        f = collide_bgk3d(f, TAU)
        if use_bb:
            f = apply_moving_bb(f, solid_bottom, solid_top, U_TOP)
        else:
            # Moving BB at top only (to drive flow)
            opp = OPPOSITE.to(device)
            rho = f.sum(dim=0)
            c = C.to(device).float()
            w = W.to(device).float()
            cx = c[:, 0].view(19, 1, 1, 1)
            w_view = w.view(19, 1, 1, 1)
            cu_w = cx * U_TOP
            correction = 2.0 * w_view * rho.unsqueeze(0) * cu_w * 3.0
            f = torch.where(solid_top.unsqueeze(0), f[opp] + correction, f)
        if wf_timing == 'pre':
            if force_type == 'force':
                f = apply_wf_force(f, solid, near, NU, yplus_thresh=yplus_thresh)
            elif force_type == 'none':
                pass
            elif force_type == 'shifted':
                f = apply_wf_shifted(f, solid, near, NU, shift_factor=shift_factor,
                                      yplus_thresh=yplus_thresh)
        f = stream3d(f)
        if wf_timing == 'post':
            if force_type == 'force':
                f = apply_wf_force(f, solid, near, NU, yplus_thresh=yplus_thresh)
            elif force_type == 'none':
                pass
            elif force_type == 'shifted':
                f = apply_wf_shifted(f, solid, near, NU, shift_factor=shift_factor,
                                      yplus_thresh=yplus_thresh)

    rho, ux, uy, uz = macroscopic3d(f)
    profile = ux[:, :, :].mean(dim=(0, 2)).cpu().numpy()
    y_centers = torch.arange(1, NY - 1, dtype=torch.float32)
    u_analytical = U_TOP * (y_centers - 0.5) / H
    u_measured = torch.from_numpy(profile[1:-1])
    u_max = float(u_measured.max().item())
    u_max_anal = float(u_analytical.max().item())
    u_err = abs(u_max - u_max_anal) / u_max_anal * 100.0
    tau_w_anal = NU * U_TOP / H
    du_dy = (profile[2] - profile[1])
    tau_w_meas = NU * abs(du_dy)
    cf_measured = 2.0 * tau_w_meas / (1.0 * u_max ** 2) if u_max > 1e-10 else 0.0
    cf_anal = 2.0 * tau_w_anal / (1.0 * u_max_anal ** 2)

    u_near = float(profile[1])
    u_tau_est = (NU * u_near / Y_VAL) ** 0.5
    y_plus_est = Y_VAL * u_tau_est / NU

    if verbose:
        print(f"  u_max={u_max:.6f} (anal={u_max_anal:.6f}), u_err={u_err:.2f}%, "
              f"Cf={cf_measured:.4f} (anal={cf_anal:.4f}), y+={y_plus_est:.4f}")
    return u_max, u_err, cf_measured, profile, y_plus_est


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='sdaa:4')
    parser.add_argument('--test', type=str, default='all')
    parser.add_argument('--n-steps', type=int, default=3000)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(42)
    print(f"Device: {device}")
    print(f"Grid: nx={NX}, ny={NY}, nz={NZ}, tau={TAU}, nu={NU:.6f}")
    print(f"U_max_target={U_MAX_TARGET}, G_drive={G_DRIVE:.8f}")
    print(f"Analytical: tau_w={TAU_W_ANALYTICAL:.6f}, Cf(umax)={CF_ANALYTICAL_UMAX:.4f}")
    print()

    results = {}

    if args.test in ('all', '1'):
        print("=" * 70)
        print("PHASE 2 TEST 1: Poiseuille — BB only (no WF) baseline")
        print("=" * 70)
        u_max, u_err, cf, _, yp = run_poiseuille(
            device, wf_timing='post', use_bb=True, force_type='none',
            n_steps=args.n_steps)
        results['poiseuille_bb_only'] = {'u_max': u_max, 'u_err': u_err, 'Cf': cf,
                                         'y_plus': yp, 'config': 'post-BB-none'}

        print("\nPHASE 2 TEST 1b: Poiseuille — BB + WF force (no y+ threshold)")
        u_max, u_err, cf, _, yp = run_poiseuille(
            device, wf_timing='post', use_bb=True, force_type='force',
            n_steps=args.n_steps, yplus_thresh=0.0)
        results['poiseuille_bb_force'] = {'u_max': u_max, 'u_err': u_err, 'Cf': cf,
                                          'y_plus': yp, 'config': 'post-BB-force'}

        print("\nPHASE 2 TEST 1c: Poiseuille — BB + WF force (y+ > 11.6 threshold)")
        u_max, u_err, cf, _, yp = run_poiseuille(
            device, wf_timing='post', use_bb=True, force_type='force',
            n_steps=args.n_steps, yplus_thresh=Y_PLUS_THRESH)
        results['poiseuille_bb_force_yp'] = {'u_max': u_max, 'u_err': u_err, 'Cf': cf,
                                             'y_plus': yp, 'config': 'post-BB-force-yp'}

        print("\nPHASE 2 TEST 1d: Poiseuille — BB + WF correction (log-grad)")
        u_max, u_err, cf, _, yp = run_poiseuille(
            device, wf_timing='post', use_bb=True, force_type='correction',
            n_steps=args.n_steps)
        results['poiseuille_bb_correction'] = {'u_max': u_max, 'u_err': u_err, 'Cf': cf,
                                               'y_plus': yp, 'config': 'post-BB-correction'}

        print("\nPHASE 2 TEST 1e: Poiseuille — BB + WF Guo force")
        u_max, u_err, cf, _, yp = run_poiseuille(
            device, wf_timing='post', use_bb=True, force_type='force_guo',
            n_steps=args.n_steps)
        results['poiseuille_bb_guo'] = {'u_max': u_max, 'u_err': u_err, 'Cf': cf,
                                        'y_plus': yp, 'config': 'post-BB-guo'}

    if args.test in ('all', '2'):
        print("\n" + "=" * 70)
        print("PHASE 2 TEST 2: Couette — BB only (no WF) baseline")
        print("=" * 70)
        u_max, u_err, cf, _, yp = run_couette(
            device, wf_timing='post', use_bb=True, force_type='none',
            n_steps=args.n_steps)
        results['couette_bb_only'] = {'u_max': u_max, 'u_err': u_err, 'Cf': cf,
                                      'y_plus': yp, 'config': 'post-BB-none'}

        print("\nPHASE 2 TEST 2b: Couette — BB + WF force (y+ > 11.6)")
        u_max, u_err, cf, _, yp = run_couette(
            device, wf_timing='post', use_bb=True, force_type='force',
            n_steps=args.n_steps, yplus_thresh=Y_PLUS_THRESH)
        results['couette_bb_force_yp'] = {'u_max': u_max, 'u_err': u_err, 'Cf': cf,
                                          'y_plus': yp, 'config': 'post-BB-force-yp'}

        print("\nPHASE 2 TEST 2c: Couette — BB + WF correction (log-grad)")
        u_max, u_err, cf, _, yp = run_couette(
            device, wf_timing='post', use_bb=True, force_type='correction',
            n_steps=args.n_steps)
        results['couette_bb_correction'] = {'u_max': u_max, 'u_err': u_err, 'Cf': cf,
                                            'y_plus': yp, 'config': 'post-BB-correction'}

    if args.test in ('all', '3'):
        print("\n" + "=" * 70)
        print("PHASE 2 TEST 3: Poiseuille — BB + shifted velocity (1x, y+ threshold)")
        print("=" * 70)
        u_max, u_err, cf, _, yp = run_poiseuille(
            device, wf_timing='post', use_bb=True, force_type='shifted',
            n_steps=args.n_steps, shift_factor=1.0, yplus_thresh=Y_PLUS_THRESH)
        results['poiseuille_bb_shifted1_yp'] = {'u_max': u_max, 'u_err': u_err, 'Cf': cf,
                                                'y_plus': yp, 'config': 'post-BB-shifted1-yp'}

    if args.test in ('all', '4'):
        print("\n" + "=" * 70)
        print("PHASE 2 TEST 4: Full config comparison")
        print("=" * 70)
        configs = [
            ('post', True,  'none',       0.0, 0.0,  'BB only (baseline)'),
            ('post', True,  'force',      0.0, 0.0,  'BB + force (no y+)'),
            ('post', True,  'force',      0.0, 11.6, 'BB + force (y+>11.6)'),
            ('post', True,  'force_guo',  0.0, 0.0,  'BB + Guo force'),
            ('post', True,  'force_guo',  0.0, 11.6, 'BB + Guo force (y+>11.6)'),
            ('post', True,  'correction', 0.0, 0.0,  'BB + correction (log-grad)'),
            ('post', True,  'shifted',    1.0, 0.0,  'BB + shifted 1x'),
            ('post', True,  'shifted',    1.0, 11.6, 'BB + shifted 1x (y+>11.6)'),
            ('post', True,  'shifted',    3.0, 0.0,  'BB + shifted 3x (task)'),
            ('post', False, 'none',       0.0, 0.0,  'no BB, no WF'),
            ('post', False, 'force',      0.0, 0.0,  'no BB + force'),
            ('pre',  True,  'force',      0.0, 0.0,  'pre-BB + force'),
            ('pre',  True,  'none',       0.0, 0.0,  'pre-BB only'),
        ]
        for i, (timing, bb, ftype, sf, yp_th, desc) in enumerate(configs, 1):
            print(f"\n  Config {i}/{len(configs)}: {desc}")
            u_max, u_err, cf, _, yp = run_poiseuille(
                device, wf_timing=timing, use_bb=bb, force_type=ftype,
                n_steps=args.n_steps, shift_factor=sf, yplus_thresh=yp_th)
            results[f'cfg_{i}'] = {'u_max': u_max, 'u_err': u_err, 'Cf': cf,
                                  'y_plus': yp, 'config': desc}

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY (sorted by u_err)")
        print("=" * 70)
        sr = sorted(results.items(), key=lambda x: x[1].get('u_err', 999))
        print(f"{'Config':<40} {'u_max':>10} {'u_err%':>10} {'Cf':>10} {'y+':>8}")
        print("-" * 80)
        for k, r in sr:
            print(f"  {r['config']:<38} {r['u_max']:>10.6f} {r['u_err']:>10.2f} "
                  f"{r['Cf']:>10.4f} {r.get('y_plus', 0):>8.4f}")
        print("-" * 80)
        print(f"  {'Analytical':<38} {U_MAX_TARGET:>10.6f} {'0.00':>10} "
              f"{CF_ANALYTICAL_UMAX:>10.4f}")

    # Save
    def _to_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return v
    results_clean = {k: {kk: _to_float(vv) for kk, vv in v.items()}
                     for k, v in results.items()}
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f'wf_phase2_results_{args.device.replace(":", "_")}.json')
    with open(outpath, 'w') as fout:
        json.dump(results_clean, fout, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == '__main__':
    main()
