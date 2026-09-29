#!/usr/bin/env python3
"""Wall function timing + BB interaction test — Phase 3.

ROOT CAUSE FOUND: BB used post-collision f (Method 3) instead of pre-collision
f (Method 1/2). This caused 16.66% over-prediction on Poiseuille.

FIX: Save pre-collision f at solid cells, collide all, then BB with pre-collision f.
    f_pre_solid = f[:, solid].clone()
    f = collide_bgk3d(f, tau)
    f[:, solid] = f_pre_solid[opp]   # BB using PRE-collision f

Also: y+ threshold deactivates WF for low-Re (where BB is already correct).
"""
import sys, os, json, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import torch
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C, W, OPPOSITE
from tensorlbm.solver3d import collide_bgk3d, stream3d
from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.ibm import ibm_apply_body_force_3d

NX, NY, NZ = 80, 12, 4
TAU = 1.0
NU = (TAU - 0.5) / 3.0
U_MAX_TARGET = 0.05
U_TOP = 0.05
H = NY - 2
G_DRIVE = 8.0 * NU * U_MAX_TARGET / (H * H)
Y_VAL = 0.5
CS2 = 1.0 / 3.0
Y_PLUS_THRESH = 11.6
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
# CORRECTED bounce-back (uses pre-collision f)
# ---------------------------------------------------------------------------

def apply_bb_corrected(f, solid, opp):
    """Corrected half-way BB: f[i, solid] = f_PRE[opp[i], solid].

    Must be called AFTER collision but uses the PRE-collision f saved
    before collision. The caller must save f_pre_solid before colliding.
    """
    # f already has post-collision values at fluid, pre-collision at solid
    # (because we restore solid cells after collision)
    # Actually: caller saves f_pre_solid, collides, then calls this
    # We need f_pre_solid passed in
    raise NotImplementedError("Use apply_bb_corrected_step instead")


def collide_with_bb(f, solid, tau, opp):
    """Collide all cells, then apply BB at solid using PRE-collision f.

    This is the CORRECT half-way BB:
    1. Save f at solid cells (pre-collision)
    2. Collide all cells
    3. At solid cells: f[i] = f_pre[opp[i]]  (BB using pre-collision)
    """
    # 1. Save pre-collision f at solid cells
    f_pre_solid = f[:, solid].clone()  # (19, n_solid)
    # 2. Collide all cells
    f = collide_bgk3d(f, tau)
    # 3. BB at solid cells using pre-collision f
    f[:, solid] = f_pre_solid[opp]
    return f


def collide_with_moving_bb(f, solid_bottom, solid_top, u_top, tau, opp):
    """Collide all cells, then apply moving BB at top + stationary BB at bottom.

    Uses PRE-collision f for BB (corrected).
    """
    # 1. Save pre-collision f at solid cells
    f_pre_bottom = f[:, solid_bottom].clone()
    f_pre_top = f[:, solid_top].clone()
    # 2. Collide all cells
    f = collide_bgk3d(f, tau)
    # 3. Stationary BB at bottom (pre-collision)
    f[:, solid_bottom] = f_pre_bottom[opp]
    # 4. Moving BB at top (pre-collision + correction)
    rho = f.sum(dim=0)
    c = C.to(f.device).float()
    w = W.to(f.device).float()
    cx = c[:, 0].view(19, 1, 1, 1)
    w_view = w.view(19, 1, 1, 1)
    cu_w = cx * u_top
    correction = 2.0 * w_view * rho.unsqueeze(0) * cu_w * 3.0
    # f[i, top] = f_pre[opp[i], top] + correction[i, top]
    f[:, solid_top] = f_pre_top[opp] + correction[:, solid_top]
    return f


# ---------------------------------------------------------------------------
# Wall function variants
# ---------------------------------------------------------------------------

def compute_tau_w_gradient(ux, uz, near, nu, y_val=Y_VAL):
    u_tan_mag = torch.sqrt(ux * ux + uz * uz).clamp(min=1e-12)
    tau_w = nu * u_tan_mag / y_val
    return torch.where(near, tau_w, torch.zeros_like(tau_w))


def compute_y_plus(tau_w, nu, y_val=Y_VAL):
    u_tau = torch.sqrt(tau_w.clamp(min=1e-30))
    return y_val * u_tau / nu


def apply_wf_force(f, near, nu, y_val=Y_VAL, yplus_thresh=0.0):
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


def apply_wf_force_guo(f, near, nu, y_val=Y_VAL, yplus_thresh=0.0):
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


def apply_wf_shifted(f, near, nu, y_val=Y_VAL, shift_factor=1.0, yplus_thresh=0.0):
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
# Old (incorrect) BB for comparison
# ---------------------------------------------------------------------------

def apply_bb_old(f, solid):
    """Old BB: f[i, solid] = f_POST[opp[i], solid] (incorrect)."""
    return bounce_back_cells_3d(f, solid)


# ---------------------------------------------------------------------------
# Poiseuille runner
# ---------------------------------------------------------------------------

def run_poiseuille(device, bb_mode='corrected', wf_timing='post',
                   force_type='none', n_steps=3000, shift_factor=1.0,
                   yplus_thresh=0.0, verbose=True):
    """Run Poiseuille. bb_mode: 'corrected', 'old', 'none'."""
    solid, fluid, near = build_masks(device)
    opp = OPPOSITE.to(device)
    rho0 = torch.ones((NZ, NY, NX), device=device)
    u0 = torch.zeros((NZ, NY, NX), device=device)
    f = equilibrium3d(rho0, u0, u0.clone(), u0.clone())
    fx_drive = torch.zeros((NZ, NY, NX), device=device)
    fx_drive[fluid] = G_DRIVE

    for step in range(n_steps):
        # 1. Collide + BB
        if bb_mode == 'corrected':
            f = collide_with_bb(f, solid, TAU, opp)
        elif bb_mode == 'old':
            f = collide_bgk3d(f, TAU)
            f = apply_bb_old(f, solid)
        else:  # none
            f = collide_bgk3d(f, TAU)

        # 2. Pre-stream WF
        if wf_timing == 'pre' and force_type != 'none':
            if force_type == 'force':
                f = apply_wf_force(f, near, NU, yplus_thresh=yplus_thresh)
            elif force_type == 'force_guo':
                f = apply_wf_force_guo(f, near, NU, yplus_thresh=yplus_thresh)
            elif force_type == 'shifted':
                f = apply_wf_shifted(f, near, NU, shift_factor=shift_factor,
                                      yplus_thresh=yplus_thresh)

        # 3. Stream
        f = stream3d(f)

        # 4. Post-stream WF
        if wf_timing == 'post' and force_type != 'none':
            if force_type == 'force':
                f = apply_wf_force(f, near, NU, yplus_thresh=yplus_thresh)
            elif force_type == 'force_guo':
                f = apply_wf_force_guo(f, near, NU, yplus_thresh=yplus_thresh)
            elif force_type == 'shifted':
                f = apply_wf_shifted(f, near, NU, shift_factor=shift_factor,
                                      yplus_thresh=yplus_thresh)

        # 5. Driving force
        f = ibm_apply_body_force_3d(f, fx_drive, torch.zeros_like(fx_drive),
                                     torch.zeros_like(fx_drive))

    rho, ux, uy, uz = macroscopic3d(f)
    profile = ux[:, :, :].mean(dim=(0, 2)).cpu().numpy()
    u_max = float(profile[1:-1].max())
    u_err = abs(u_max - U_MAX_TARGET) / U_MAX_TARGET * 100.0
    du_dy = (profile[2] - profile[1])
    tau_w_measured = NU * abs(du_dy)
    cf_measured = 2.0 * tau_w_measured / (1.0 * u_max ** 2) if u_max > 1e-10 else 0.0
    u_near = float(profile[1])
    u_tau_est = (max(NU * u_near / Y_VAL, 0)) ** 0.5
    y_plus_est = Y_VAL * u_tau_est / NU

    if verbose:
        print(f"  u_max={u_max:.6f} (target={U_MAX_TARGET}), u_err={u_err:.2f}%, "
              f"Cf={cf_measured:.4f} (anal={CF_ANALYTICAL_UMAX:.4f}), y+={y_plus_est:.4f}")
    return u_max, u_err, cf_measured, y_plus_est


# ---------------------------------------------------------------------------
# Couette runner
# ---------------------------------------------------------------------------

def run_couette(device, bb_mode='corrected', wf_timing='post',
                force_type='none', n_steps=3000, yplus_thresh=0.0,
                verbose=True):
    solid, fluid, near = build_masks(device)
    opp = OPPOSITE.to(device)
    solid_bottom = torch.zeros_like(solid)
    solid_bottom[:, 0, :] = True
    solid_top = torch.zeros_like(solid)
    solid_top[:, -1, :] = True

    rho0 = torch.ones((NZ, NY, NX), device=device)
    u0 = torch.zeros((NZ, NY, NX), device=device)
    f = equilibrium3d(rho0, u0, u0.clone(), u0.clone())

    for step in range(n_steps):
        if bb_mode == 'corrected':
            f = collide_with_moving_bb(f, solid_bottom, solid_top, U_TOP, TAU, opp)
        elif bb_mode == 'old':
            f = collide_bgk3d(f, TAU)
            # Old moving BB
            rho = f.sum(dim=0)
            c = C.to(device).float()
            w = W.to(device).float()
            cx = c[:, 0].view(19, 1, 1, 1)
            w_view = w.view(19, 1, 1, 1)
            f = torch.where(solid_bottom.unsqueeze(0), f[opp], f)
            cu_w = cx * U_TOP
            correction = 2.0 * w_view * rho.unsqueeze(0) * cu_w * 3.0
            f = torch.where(solid_top.unsqueeze(0), f[opp] + correction, f)
        else:
            f = collide_bgk3d(f, TAU)
            # Moving BB at top only
            rho = f.sum(dim=0)
            c = C.to(device).float()
            w = W.to(device).float()
            cx = c[:, 0].view(19, 1, 1, 1)
            w_view = w.view(19, 1, 1, 1)
            cu_w = cx * U_TOP
            correction = 2.0 * w_view * rho.unsqueeze(0) * cu_w * 3.0
            f = torch.where(solid_top.unsqueeze(0), f[opp] + correction, f)

        if wf_timing == 'pre' and force_type != 'none':
            if force_type == 'force':
                f = apply_wf_force(f, near, NU, yplus_thresh=yplus_thresh)
        f = stream3d(f)
        if wf_timing == 'post' and force_type != 'none':
            if force_type == 'force':
                f = apply_wf_force(f, near, NU, yplus_thresh=yplus_thresh)

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
    u_tau_est = (max(NU * u_near / Y_VAL, 0)) ** 0.5
    y_plus_est = Y_VAL * u_tau_est / NU

    if verbose:
        print(f"  u_max={u_max:.6f} (anal={u_max_anal:.6f}), u_err={u_err:.2f}%, "
              f"Cf={cf_measured:.4f} (anal={cf_anal:.4f}), y+={y_plus_est:.4f}")
    return u_max, u_err, cf_measured, y_plus_est


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
        print("TEST 1: Poiseuille — Corrected BB only (no WF)")
        print("=" * 70)
        u_max, u_err, cf, yp = run_poiseuille(
            device, bb_mode='corrected', force_type='none', n_steps=args.n_steps)
        results['poiseuille_corrected_bb_only'] = {
            'u_max': u_max, 'u_err': u_err, 'Cf': cf, 'y_plus': yp,
            'config': 'corrected-BB-only'}

        print("\nTEST 1b: Poiseuille — Old BB only (for comparison)")
        u_max, u_err, cf, yp = run_poiseuille(
            device, bb_mode='old', force_type='none', n_steps=args.n_steps)
        results['poiseuille_old_bb_only'] = {
            'u_max': u_max, 'u_err': u_err, 'Cf': cf, 'y_plus': yp,
            'config': 'old-BB-only'}

        print("\nTEST 1c: Poiseuille — Corrected BB + WF force (y+>11.6)")
        u_max, u_err, cf, yp = run_poiseuille(
            device, bb_mode='corrected', force_type='force',
            n_steps=args.n_steps, yplus_thresh=Y_PLUS_THRESH)
        results['poiseuille_corrected_bb_force_yp'] = {
            'u_max': u_max, 'u_err': u_err, 'Cf': cf, 'y_plus': yp,
            'config': 'corrected-BB-force-yp'}

        print("\nTEST 1d: Poiseuille — Corrected BB + WF force (no y+ threshold)")
        u_max, u_err, cf, yp = run_poiseuille(
            device, bb_mode='corrected', force_type='force',
            n_steps=args.n_steps, yplus_thresh=0.0)
        results['poiseuille_corrected_bb_force'] = {
            'u_max': u_max, 'u_err': u_err, 'Cf': cf, 'y_plus': yp,
            'config': 'corrected-BB-force'}

        print("\nTEST 1e: Poiseuille — Corrected BB + WF Guo force (y+>11.6)")
        u_max, u_err, cf, yp = run_poiseuille(
            device, bb_mode='corrected', force_type='force_guo',
            n_steps=args.n_steps, yplus_thresh=Y_PLUS_THRESH)
        results['poiseuille_corrected_bb_guo_yp'] = {
            'u_max': u_max, 'u_err': u_err, 'Cf': cf, 'y_plus': yp,
            'config': 'corrected-BB-guo-yp'}

        print("\nTEST 1f: Poiseuille — No BB + WF force (best from Phase 1)")
        u_max, u_err, cf, yp = run_poiseuille(
            device, bb_mode='none', force_type='force',
            n_steps=args.n_steps, yplus_thresh=0.0)
        results['poiseuille_noBB_force'] = {
            'u_max': u_max, 'u_err': u_err, 'Cf': cf, 'y_plus': yp,
            'config': 'noBB-force'}

        print("\nTEST 1g: Poiseuille — Corrected BB + shifted 1x (y+>11.6)")
        u_max, u_err, cf, yp = run_poiseuille(
            device, bb_mode='corrected', force_type='shifted',
            n_steps=args.n_steps, shift_factor=1.0, yplus_thresh=Y_PLUS_THRESH)
        results['poiseuille_corrected_bb_shifted1_yp'] = {
            'u_max': u_max, 'u_err': u_err, 'Cf': cf, 'y_plus': yp,
            'config': 'corrected-BB-shifted1-yp'}

    if args.test in ('all', '2'):
        print("\n" + "=" * 70)
        print("TEST 2: Couette — Corrected BB only (no WF)")
        print("=" * 70)
        u_max, u_err, cf, yp = run_couette(
            device, bb_mode='corrected', force_type='none', n_steps=args.n_steps)
        results['couette_corrected_bb_only'] = {
            'u_max': u_max, 'u_err': u_err, 'Cf': cf, 'y_plus': yp,
            'config': 'corrected-BB-only'}

        print("\nTEST 2b: Couette — Old BB only (for comparison)")
        u_max, u_err, cf, yp = run_couette(
            device, bb_mode='old', force_type='none', n_steps=args.n_steps)
        results['couette_old_bb_only'] = {
            'u_max': u_max, 'u_err': u_err, 'Cf': cf, 'y_plus': yp,
            'config': 'old-BB-only'}

        print("\nTEST 2c: Couette — Corrected BB + WF force (y+>11.6)")
        u_max, u_err, cf, yp = run_couette(
            device, bb_mode='corrected', force_type='force',
            n_steps=args.n_steps, yplus_thresh=Y_PLUS_THRESH)
        results['couette_corrected_bb_force_yp'] = {
            'u_max': u_max, 'u_err': u_err, 'Cf': cf, 'y_plus': yp,
            'config': 'corrected-BB-force-yp'}

    if args.test in ('all', '3'):
        print("\n" + "=" * 70)
        print("TEST 3: Poiseuille — Corrected BB + shifted velocity variants")
        print("=" * 70)
        for sf, label in [(1.0, '1x'), (3.0, '3x')]:
            print(f"\n  Shifted {label}, y+>11.6:")
            u_max, u_err, cf, yp = run_poiseuille(
                device, bb_mode='corrected', force_type='shifted',
                n_steps=args.n_steps, shift_factor=sf, yplus_thresh=Y_PLUS_THRESH)
            results[f'poiseuille_corrected_bb_shifted{label}_yp'] = {
                'u_max': u_max, 'u_err': u_err, 'Cf': cf, 'y_plus': yp,
                'config': f'corrected-BB-shifted{label}-yp'}

    if args.test in ('all', '4'):
        print("\n" + "=" * 70)
        print("TEST 4: Full config comparison (corrected BB)")
        print("=" * 70)
        configs = [
            ('corrected', 'post', 'none',      0.0, 0.0,  'Corrected BB only'),
            ('corrected', 'post', 'force',      0.0, 0.0,  'Corrected BB + force'),
            ('corrected', 'post', 'force',      0.0, 11.6, 'Corrected BB + force (y+>11.6)'),
            ('corrected', 'post', 'force_guo',  0.0, 0.0,  'Corrected BB + Guo force'),
            ('corrected', 'post', 'force_guo',  0.0, 11.6, 'Corrected BB + Guo (y+>11.6)'),
            ('corrected', 'post', 'shifted',    1.0, 0.0,  'Corrected BB + shifted 1x'),
            ('corrected', 'post', 'shifted',    1.0, 11.6, 'Corrected BB + shifted 1x (y+>11.6)'),
            ('corrected', 'pre',  'force',      0.0, 0.0,  'Pre: Corrected BB + force'),
            ('corrected', 'pre',  'force',      0.0, 11.6, 'Pre: Corrected BB + force (y+>11.6)'),
            ('old',       'post', 'none',      0.0, 0.0,  'Old BB only'),
            ('old',       'post', 'force',      0.0, 0.0,  'Old BB + force'),
            ('none',      'post', 'force',      0.0, 0.0,  'No BB + force'),
            ('none',      'post', 'none',      0.0, 0.0,  'No BB, no WF'),
        ]
        for i, (bb, timing, ftype, sf, yp_th, desc) in enumerate(configs, 1):
            print(f"\n  Config {i}/{len(configs)}: {desc}")
            u_max, u_err, cf, yp = run_poiseuille(
                device, bb_mode=bb, wf_timing=timing, force_type=ftype,
                n_steps=args.n_steps, shift_factor=sf, yplus_thresh=yp_th)
            results[f'cfg_{i}'] = {
                'u_max': u_max, 'u_err': u_err, 'Cf': cf, 'y_plus': yp,
                'config': desc}

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY (sorted by u_err)")
        print("=" * 70)
        sr = sorted(results.items(), key=lambda x: x[1].get('u_err', 999))
        print(f"{'Config':<45} {'u_max':>10} {'u_err%':>10} {'Cf':>10} {'y+':>8}")
        print("-" * 85)
        for k, r in sr:
            print(f"  {r['config']:<43} {r['u_max']:>10.6f} {r['u_err']:>10.2f} "
                  f"{r['Cf']:>10.4f} {r.get('y_plus', 0):>8.4f}")
        print("-" * 85)
        print(f"  {'Analytical':<43} {U_MAX_TARGET:>10.6f} {'0.00':>10} "
              f"{CF_ANALYTICAL_UMAX:>10.4f}")

    # Save
    def _to_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return str(v)
    results_clean = {k: {kk: _to_float(vv) for kk, vv in v.items()}
                     for k, v in results.items()}
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f'wf_phase3_results_{args.device.replace(":", "_")}.json')
    with open(outpath, 'w') as fout:
        json.dump(results_clean, fout, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == '__main__':
    main()
