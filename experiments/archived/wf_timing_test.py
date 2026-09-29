#!/usr/bin/env python3
"""Wall function timing + BB interaction test.

Tests configurations on Poiseuille/Couette flow:
- pre/post stream timing
- BB/no-BB on solid cells
- force/shifted velocity approach

SDAA cards 4-7.
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
NU = (TAU - 0.5) / 3.0          # 1/6
U_MAX_TARGET = 0.05
U_TOP = 0.05
H = NY - 2                       # channel height = 10 (half-way BB walls at 0.5, 10.5)
G_DRIVE = 8.0 * NU * U_MAX_TARGET / (H * H)   # body force per unit mass
Y_VAL = 0.5                      # near-wall cell centre to wall distance
CS2 = 1.0 / 3.0

# Analytical values
TAU_W_ANALYTICAL = G_DRIVE * H / 2.0           # wall shear stress
CF_ANALYTICAL_UMAX = 2.0 * TAU_W_ANALYTICAL / (1.0 * U_MAX_TARGET ** 2)


def build_masks(device):
    """Build solid, fluid, near-wall masks for channel."""
    solid = torch.zeros((NZ, NY, NX), dtype=torch.bool, device=device)
    solid[:, 0, :] = True    # bottom wall
    solid[:, -1, :] = True   # top wall
    fluid = ~solid
    # near-wall fluid cells (adjacent to solid)
    near = torch.zeros_like(solid)
    near[:, 1, :] = True     # first fluid row above bottom
    near[:, -2, :] = True    # first fluid row below top
    return solid, fluid, near


# ---------------------------------------------------------------------------
# Wall function implementations
# ---------------------------------------------------------------------------

def compute_tau_w(ux, uz, near, nu, y_val=Y_VAL):
    """Gradient wall law: τ_w = ν * u_tan / y_val at near-wall cells."""
    u_tan_mag = torch.sqrt(ux * ux + uz * uz).clamp(min=1e-12)
    tau_w = nu * u_tan_mag / y_val
    return torch.where(near, tau_w, torch.zeros_like(tau_w))


def apply_wf_force(f, solid, near, nu, y_val=Y_VAL):
    """Apply WF as body force: F = -τ_w * û_tan (simple forcing)."""
    rho, ux, uy, uz = macroscopic3d(f)
    tau_w = compute_tau_w(ux, uz, near, nu, y_val)
    u_tan_mag = torch.sqrt(ux * ux + uz * uz).clamp(min=1e-12)
    fx = -tau_w * (ux / u_tan_mag)
    fz = -tau_w * (uz / u_tan_mag)
    fy = torch.zeros_like(fx)
    return ibm_apply_body_force_3d(f, fx, fy, fz)


def apply_wf_shifted(f, solid, near, nu, y_val=Y_VAL, shift_factor=3.0):
    """Apply WF as shifted velocity: u_shifted = u - shift_factor*τ_w/ρ * û.

    shift_factor=3.0 → task formula (u - τ_w/(ρ*cs²) = u - 3*τ_w/ρ)
    shift_factor=1.0 → corrected (same momentum as force approach)

    Applied as: f += (feq(u_shifted) - feq(u)) at near-wall cells.
    Equivalent to simple force F = -shift_factor*τ_w * û.
    """
    rho, ux, uy, uz = macroscopic3d(f)
    tau_w = compute_tau_w(ux, uz, near, nu, y_val)
    u_tan_mag = torch.sqrt(ux * ux + uz * uz).clamp(min=1e-12)
    factor = shift_factor * tau_w / rho.clamp(min=1e-12)
    ux_s = ux - factor * (ux / u_tan_mag)
    uz_s = uz - factor * (uz / u_tan_mag)
    feq_s = equilibrium3d(rho, ux_s, uy, uz_s)
    feq_c = equilibrium3d(rho, ux, uy, uz)
    delta = (feq_s - feq_c) * near.unsqueeze(0).to(f.dtype)
    return f + delta


def apply_wf_shifted_eq(f, solid, near, nu, y_val=Y_VAL, shift_factor=3.0):
    """Apply WF as full equilibrium reset (Dirichlet-like) at near-wall cells."""
    rho, ux, uy, uz = macroscopic3d(f)
    tau_w = compute_tau_w(ux, uz, near, nu, y_val)
    u_tan_mag = torch.sqrt(ux * ux + uz * uz).clamp(min=1e-12)
    factor = shift_factor * tau_w / rho.clamp(min=1e-12)
    ux_s = ux - factor * (ux / u_tan_mag)
    uz_s = uz - factor * (uz / u_tan_mag)
    feq_s = equilibrium3d(rho, ux_s, uy, uz_s)
    near_3d = near.unsqueeze(0).to(f.dtype)
    return torch.where(near_3d, feq_s, f)


# ---------------------------------------------------------------------------
# Bounce-back (NoDynamics on solid cells)
# ---------------------------------------------------------------------------

def apply_bb_solid(f, solid):
    """Half-way bounce-back at solid cells: f[i] = f[opp[i]]."""
    return bounce_back_cells_3d(f, solid)


def apply_moving_bb(f, solid_bottom, solid_top, u_top):
    """Moving-wall BB at top, stationary BB at bottom."""
    opp = OPPOSITE.to(f.device)
    rho = f.sum(dim=0)
    c = C.to(f.device).float()
    w = W.to(f.device).float()
    cx = c[:, 0].view(19, 1, 1, 1)
    w_view = w.view(19, 1, 1, 1)
    # Stationary BB at bottom
    f = torch.where(solid_bottom.unsqueeze(0), f[opp], f)
    # Moving BB at top: f[i] = f[opp[i]] + 6*w_i*rho*(c_i·u_w)
    cu_w = cx * u_top   # u_w = (u_top, 0, 0)
    correction = 2.0 * w_view * rho.unsqueeze(0) * cu_w * 3.0  # 6*w*rho*c·u_w
    f_bb_top = f[opp]
    f = torch.where(solid_top.unsqueeze(0), f_bb_top + correction, f)
    return f


# ---------------------------------------------------------------------------
# Poiseuille flow runner
# ---------------------------------------------------------------------------

def run_poiseuille(device, wf_timing='post', use_bb=True, force_type='force',
                   n_steps=3000, shift_factor=3.0, verbose=True):
    """Run Poiseuille channel flow and return (u_max, u_err, Cf, profile)."""
    solid, fluid, near = build_masks(device)

    # Initial condition: zero velocity
    rho0 = torch.ones((NZ, NY, NX), device=device)
    u0 = torch.zeros((NZ, NY, NX), device=device)
    f = equilibrium3d(rho0, u0, u0.clone(), u0.clone())

    # Driving body force on fluid cells
    fx_drive = torch.zeros((NZ, NY, NX), device=device)
    fx_drive[fluid] = G_DRIVE

    for step in range(n_steps):
        # 1. Collide (BGK on all cells)
        f = collide_bgk3d(f, TAU)

        # 2. NoDynamics / BB on solid cells
        if use_bb:
            f = apply_bb_solid(f, solid)

        # 3. Pre-stream WF
        if wf_timing == 'pre':
            if force_type == 'force':
                f = apply_wf_force(f, solid, near, NU)
            elif force_type == 'shifted':
                f = apply_wf_shifted(f, solid, near, NU, shift_factor=shift_factor)
            elif force_type == 'shifted_eq':
                f = apply_wf_shifted_eq(f, solid, near, NU, shift_factor=shift_factor)

        # 4. Stream
        f = stream3d(f)

        # 5. Post-stream WF
        if wf_timing == 'post':
            if force_type == 'force':
                f = apply_wf_force(f, solid, near, NU)
            elif force_type == 'shifted':
                f = apply_wf_shifted(f, solid, near, NU, shift_factor=shift_factor)
            elif force_type == 'shifted_eq':
                f = apply_wf_shifted_eq(f, solid, near, NU, shift_factor=shift_factor)

        # 6. Driving body force
        f = ibm_apply_body_force_3d(f, fx_drive, torch.zeros_like(fx_drive),
                                     torch.zeros_like(fx_drive))

    # Measure
    rho, ux, uy, uz = macroscopic3d(f)
    # Average over x and z → profile in y
    profile = ux[:, :, :].mean(dim=(0, 2)).cpu().numpy()
    u_max = float(profile[1:-1].max())
    u_err = abs(u_max - U_MAX_TARGET) / U_MAX_TARGET * 100.0

    # Cf from velocity gradient at wall
    # du/dy at bottom wall ≈ (u[2] - u[1]) / 1.0
    du_dy = (profile[2] - profile[1])
    tau_w_measured = NU * abs(du_dy)
    cf_measured = 2.0 * tau_w_measured / (1.0 * u_max ** 2) if u_max > 1e-10 else 0.0

    if verbose:
        print(f"  Poiseuille: u_max={u_max:.6f} (target={U_MAX_TARGET}), "
              f"u_err={u_err:.2f}%, Cf={cf_measured:.4f} (analytical={CF_ANALYTICAL_UMAX:.4f})")
    return u_max, u_err, cf_measured, profile


# ---------------------------------------------------------------------------
# Couette flow runner
# ---------------------------------------------------------------------------

def run_couette(device, wf_timing='post', use_bb=True, force_type='force',
                n_steps=3000, shift_factor=3.0, verbose=True):
    """Run Couette flow (top wall moving) and return (u_max, u_err, Cf, profile)."""
    solid, fluid, near = build_masks(device)
    solid_bottom = torch.zeros_like(solid)
    solid_bottom[:, 0, :] = True
    solid_top = torch.zeros_like(solid)
    solid_top[:, -1, :] = True

    # Initial condition: zero velocity
    rho0 = torch.ones((NZ, NY, NX), device=device)
    u0 = torch.zeros((NZ, NY, NX), device=device)
    f = equilibrium3d(rho0, u0, u0.clone(), u0.clone())

    for step in range(n_steps):
        # 1. Collide
        f = collide_bgk3d(f, TAU)

        # 2. NoDynamics / BB
        if use_bb:
            # Moving BB at top, stationary BB at bottom
            f = apply_moving_bb(f, solid_bottom, solid_top, U_TOP)
        else:
            # No BB — but we still need to drive the top wall somehow
            # Use moving BB at top only (to drive the flow), no BB at bottom
            opp = OPPOSITE.to(device)
            rho = f.sum(dim=0)
            c = C.to(device).float()
            w = W.to(device).float()
            cx = c[:, 0].view(19, 1, 1, 1)
            w_view = w.view(19, 1, 1, 1)
            cu_w = cx * U_TOP
            correction = 2.0 * w_view * rho.unsqueeze(0) * cu_w * 3.0
            f_bb_top = f[opp]
            f = torch.where(solid_top.unsqueeze(0), f_bb_top + correction, f)

        # 3. Pre-stream WF
        if wf_timing == 'pre':
            if force_type == 'force':
                f = apply_wf_force(f, solid, near, NU)
            elif force_type == 'shifted':
                f = apply_wf_shifted(f, solid, near, NU, shift_factor=shift_factor)
            elif force_type == 'shifted_eq':
                f = apply_wf_shifted_eq(f, solid, near, NU, shift_factor=shift_factor)

        # 4. Stream
        f = stream3d(f)

        # 5. Post-stream WF
        if wf_timing == 'post':
            if force_type == 'force':
                f = apply_wf_force(f, solid, near, NU)
            elif force_type == 'shifted':
                f = apply_wf_shifted(f, solid, near, NU, shift_factor=shift_factor)
            elif force_type == 'shifted_eq':
                f = apply_wf_shifted_eq(f, solid, near, NU, shift_factor=shift_factor)

    # Measure
    rho, ux, uy, uz = macroscopic3d(f)
    profile = ux[:, :, :].mean(dim=(0, 2)).cpu().numpy()

    # Analytical: u(y) = u_top * (y - 0.5) / H, at cell centers y=1..10
    y_centers = torch.arange(1, NY - 1, dtype=torch.float32)
    u_analytical = U_TOP * (y_centers - 0.5) / H
    u_measured = torch.from_numpy(profile[1:-1])
    u_max = float(u_measured.max().item())
    u_max_anal = float(u_analytical.max().item())
    u_err = abs(u_max - u_max_anal) / u_max_anal * 100.0

    # Cf (linear shear: τ_w = ν * u_top / H)
    tau_w_anal = NU * U_TOP / H
    du_dy = (profile[2] - profile[1])
    tau_w_meas = NU * abs(du_dy)
    cf_measured = 2.0 * tau_w_meas / (1.0 * u_max ** 2) if u_max > 1e-10 else 0.0
    cf_anal = 2.0 * tau_w_anal / (1.0 * u_max_anal ** 2)

    if verbose:
        print(f"  Couette: u_max={u_max:.6f} (analytical={u_max_anal:.6f}), "
              f"u_err={u_err:.2f}%, Cf={cf_measured:.4f} (analytical={cf_anal:.4f})")
    return u_max, u_err, cf_measured, profile


# ---------------------------------------------------------------------------
# Analytical Poiseuille profile
# ---------------------------------------------------------------------------

def analytical_poiseuille():
    """Analytical Poiseuille profile at cell centers."""
    import numpy as np
    y = np.arange(1, NY - 1, dtype=float)
    y_c = (NY - 1) / 2.0   # 5.5
    h = H / 2.0             # 5.0
    return U_MAX_TARGET * (1.0 - ((y - y_c) / h) ** 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Wall function timing test')
    parser.add_argument('--device', type=str, default='sdaa:4')
    parser.add_argument('--test', type=str, default='all',
                        choices=['all', '1', '2', '3', '4'])
    parser.add_argument('--n-steps', type=int, default=3000)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(42)
    print(f"Device: {device}")
    print(f"Grid: nx={NX}, ny={NY}, nz={NZ}, tau={TAU}, nu={NU:.6f}")
    print(f"U_max_target={U_MAX_TARGET}, G_drive={G_DRIVE:.8f}")
    print(f"Analytical: tau_w={TAU_W_ANALYTICAL:.6f}, Cf(umax)={CF_ANALYTICAL_UMAX:.4f}")
    print(f"Channel height H={H}, y_val={Y_VAL}")
    print()

    results = {}

    # ---- TEST 1: Poiseuille with WF post-stream (SDAA:4) ----
    if args.test in ('all', '1'):
        print("=" * 70)
        print("TEST 1: Poiseuille with WF post-stream, BB on solid, force approach")
        print("=" * 70)
        t0 = time.time()
        u_max, u_err, cf, prof = run_poiseuille(
            device, wf_timing='post', use_bb=True, force_type='force',
            n_steps=args.n_steps)
        results['test1'] = {'u_max': u_max, 'u_err': u_err, 'Cf': cf,
                            'config': 'post-BB-force'}
        print(f"  Time: {time.time()-t0:.1f}s")
        print()

    # ---- TEST 2: Couette with WF post-stream (SDAA:5) ----
    if args.test in ('all', '2'):
        print("=" * 70)
        print("TEST 2: Couette with WF post-stream, BB on solid, force approach")
        print("=" * 70)
        t0 = time.time()
        u_max, u_err, cf, prof = run_couette(
            device, wf_timing='post', use_bb=True, force_type='force',
            n_steps=args.n_steps)
        results['test2'] = {'u_max': u_max, 'u_err': u_err, 'Cf': cf,
                            'config': 'post-BB-force'}
        print(f"  Time: {time.time()-t0:.1f}s")
        print()

    # ---- TEST 3: Poiseuille with shifted velocity (SDAA:6) ----
    if args.test in ('all', '3'):
        print("=" * 70)
        print("TEST 3: Poiseuille with shifted velocity (task formula: 3x)")
        print("=" * 70)
        t0 = time.time()
        u_max, u_err, cf, prof = run_poiseuille(
            device, wf_timing='post', use_bb=True, force_type='shifted',
            n_steps=args.n_steps, shift_factor=3.0)
        results['test3_shifted3x'] = {'u_max': u_max, 'u_err': u_err, 'Cf': cf,
                                      'config': 'post-BB-shifted-3x'}
        print(f"  Time: {time.time()-t0:.1f}s")

        # Also try corrected shift factor (1x = same as force)
        print("  --- Also testing corrected shift (1x) ---")
        u_max1, u_err1, cf1, _ = run_poiseuille(
            device, wf_timing='post', use_bb=True, force_type='shifted',
            n_steps=args.n_steps, shift_factor=1.0)
        results['test3_shifted1x'] = {'u_max': u_max1, 'u_err': u_err1, 'Cf': cf1,
                                      'config': 'post-BB-shifted-1x'}
        print(f"  Time: {time.time()-t0:.1f}s")
        print()

    # ---- TEST 4: Config comparison (SDAA:7) ----
    if args.test in ('all', '4'):
        print("=" * 70)
        print("TEST 4: 8-configuration comparison (Poiseuille)")
        print("=" * 70)
        configs = [
            ('pre',  False, 'force'),
            ('pre',  False, 'shifted'),
            ('pre',  True,  'force'),
            ('pre',  True,  'shifted'),
            ('post', False, 'force'),
            ('post', False, 'shifted'),
            ('post', True,  'force'),
            ('post', True,  'shifted'),
        ]
        t0 = time.time()
        for i, (timing, bb, ftype) in enumerate(configs, 1):
            label = f"{timing}-{'BB' if bb else 'noBB'}-{ftype}"
            print(f"\n  Config {i}/8: {label}")
            sf = 3.0 if ftype == 'shifted' else 3.0  # task formula
            u_max, u_err, cf, _ = run_poiseuille(
                device, wf_timing=timing, use_bb=bb, force_type=ftype,
                n_steps=args.n_steps, shift_factor=sf)
            results[f'config_{i}'] = {'u_max': u_max, 'u_err': u_err, 'Cf': cf,
                                      'config': label}

        # Also test shifted with 1x factor for the best post-BB config
        print(f"\n  Config 9 (bonus): post-BB-shifted-1x (corrected)")
        u_max, u_err, cf, _ = run_poiseuille(
            device, wf_timing='post', use_bb=True, force_type='shifted',
            n_steps=args.n_steps, shift_factor=1.0)
        results['config_9'] = {'u_max': u_max, 'u_err': u_err, 'Cf': cf,
                               'config': 'post-BB-shifted-1x'}

        # And shifted_eq (Dirichlet) with 1x
        print(f"\n  Config 10 (bonus): post-BB-shifted_eq-1x (Dirichlet)")
        u_max, u_err, cf, _ = run_poiseuille(
            device, wf_timing='post', use_bb=True, force_type='shifted_eq',
            n_steps=args.n_steps, shift_factor=1.0)
        results['config_10'] = {'u_max': u_max, 'u_err': u_err, 'Cf': cf,
                                'config': 'post-BB-shifted_eq-1x'}

        print(f"\n  Total config test time: {time.time()-t0:.1f}s")

        # Summary table
        print("\n" + "=" * 70)
        print("CONFIGURATION SUMMARY (sorted by u_err)")
        print("=" * 70)
        sorted_results = sorted(results.items(),
                                key=lambda x: x[1].get('u_err', 999))
        print(f"{'Config':<35} {'u_max':>10} {'u_err%':>10} {'Cf':>10}")
        print("-" * 70)
        for key, r in sorted_results:
            if key.startswith('config_'):
                print(f"  {r['config']:<33} {r['u_max']:>10.6f} {r['u_err']:>10.2f} {r['Cf']:>10.4f}")
        print("-" * 70)
        print(f"  {'Analytical':<33} {U_MAX_TARGET:>10.6f} {'0.00':>10} {CF_ANALYTICAL_UMAX:>10.4f}")

    # Save results (convert numpy/tensor types to Python float)
    def _to_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return v
    results_clean = {}
    for k, v in results.items():
        results_clean[k] = {kk: _to_float(vv) for kk, vv in v.items()}
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f'wf_test_results_{args.device.replace(":", "_")}.json')
    with open(outpath, 'w') as fout:
        json.dump(results_clean, fout, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == '__main__':
    main()
