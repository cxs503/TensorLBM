"""Bubble expand/contract: Phase-Field + CSF hybrid vs RP equation.

Key design (per user's optimal approach):
  1. Uniform density ρ=1.0 everywhere (single-phase LBM, no density-ratio instability)
  2. Cahn-Hilliard equation for interface tracking (conserves ∫φ dV)
  3. CSF (Continuum Surface Force) for pressure jump: F = Δp·n·δ(φ)
  4. Gas mass conservation: ρ_gas = M₀/V_gas, p_gas = ρ_gas·cs²

Physics:
  - φ ∈ [-1,+1]: -1=gas, +1=liquid
  - V_gas = ∫(1-φ)/2 dV
  - p_gas = (M_gas0/V_gas)·cs²  (adiabatic: p_gas = p_gas0·(V0/V)^γ)
  - Δp = p_gas - p_inf
  - F_CSF = Δp · n̂ · δ(φ)  (pressure jump as surface force)
  - n̂ = ∇φ/|∇φ|, δ(φ) = (3/4W)·(1-φ²/W²) for |φ|<W

RP equation (spherical, no σ, no μ):
  ρ_l (R·R̈ + 3/2·Ṙ²) = p_gas0·(R0/R)^(3γ) - p_inf
"""
import sys, math, torch, numpy as np
sys.path.insert(0, '/root/TensorLBM_test/src')
from tensorlbm.d3q19 import equilibrium3d, C as C3D, W as W3D, macroscopic3d
from tensorlbm.solver3d import stream3d
from tensorlbm.boundaries3d import bounce_back_cells_3d


# ============================================================
# RP equation solver (RK4)
# ============================================================
def solve_rp(R0, p_ratio, rho_liq, gamma, n_steps, dt=1.0):
    cs2 = 1.0 / 3.0
    p_inf = rho_liq * cs2
    p_gas0 = p_ratio * cs2
    R, Rdot = R0, 0.0
    ts, Rs = [], []
    for step in range(n_steps + 1):
        ts.append(step * dt)
        Rs.append(R)
        def accel(Rv, Rdv):
            p_gas = p_gas0 * (R0 / max(Rv, 0.1)) ** (3 * gamma)
            return (p_gas - p_inf) / (rho_liq * max(Rv, 0.1)) - 1.5 * Rdv**2 / max(Rv, 0.1)
        k1v, k1x = accel(R, Rdot), Rdot
        k2v = accel(R + 0.5*dt*k1x, Rdot + 0.5*dt*k1v); k2x = Rdot + 0.5*dt*k1v
        k3v = accel(R + 0.5*dt*k2x, Rdot + 0.5*dt*k2v); k3x = Rdot + 0.5*dt*k2v
        k4v = accel(R + dt*k3x, Rdot + dt*k3v); k4x = Rdot + dt*k3v
        R += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6.0
        Rdot += dt * (k1v + 2*k2v + 2*k3v + k4v) / 6.0
    return np.array(ts), np.array(Rs)


# ============================================================
# Phase-Field + CSF LBM
# ============================================================
def run_pf_csf_bubble(nx=48, ny=48, nz=48, n_steps=2000,
                      device='sdaa:0', p_ratio=1.5, gamma=1.4,
                      tau_f=0.8, tau_g=0.55, A_coef=0.2, B_coef=0.2,
                      kappa=0.1, csf_factor=0.5, collision='bgk',
                      gz=0.0):
    dev = torch.device(device)
    cs2 = 1.0 / 3.0
    rho_liq = 1.0          # UNIFORM density everywhere
    rho_gas0 = p_ratio     # initial gas density (only for pressure calc)
    R0 = 8.0
    tau_f = tau_f           # flow relaxation (ν = cs²(τ-0.5))
    tau_g = tau_g           # phase field mobility (M = cs²(τ_g-0.5))

    # Cahn-Hilliard parameters
    A_coef = A_coef         # double-well: μ = -Aφ + Bφ³ - κ∇²φ
    B_coef = B_coef         # quartic
    kappa = kappa           # gradient penalty

    # CSF parameters
    W = 2.0                # interface width (lattice cells)

    # Gas mass conservation
    V0 = (4.0/3.0) * math.pi * R0**3
    M_gas0 = rho_gas0 * V0
    p_inf = rho_liq * cs2
    p_gas0 = rho_gas0 * cs2
    R_eq = R0 * (p_gas0 / p_inf) ** (1.0 / (3.0 * gamma))

    cx_b, cy_b, cz_b = nx//2, ny//2, nz//2
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev), torch.arange(ny, device=dev),
        torch.arange(nx, device=dev), indexing='ij')

    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=dev)
    solid[:, 0, :] = True; solid[:, -1, :] = True
    solid[:, :, 0] = True; solid[:, :, -1] = True
    solid[0, :, :] = True; solid[-1, :, :] = True
    fluid_mask = ~solid

    opp = torch.tensor([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17], device=dev)
    c = C3D.to(dev).float()
    cx3d = c[:, 0].view(19, 1, 1, 1)
    cy3d = c[:, 1].view(19, 1, 1, 1)
    cz3d = c[:, 2].view(19, 1, 1, 1)
    w = W3D.to(dev).float().view(19, 1, 1, 1)

    # Initialize phase field: -1 inside (gas), +1 outside (liquid)
    dist = torch.sqrt((xx - cx_b)**2 + (yy - cy_b)**2 + (zz - cz_b)**2)
    phi = torch.tanh((dist - R0) / (W * 0.5))

    # f: uniform density ρ=1.0, zero velocity
    rho_init = torch.ones(nz, ny, nx, device=dev)
    f = equilibrium3d(rho_init, torch.zeros_like(rho_init),
                       torch.zeros_like(rho_init), torch.zeros_like(rho_init), device=dev)
    # g: phase field equilibrium
    g = _init_g_equilibrium(phi, torch.zeros_like(phi), torch.zeros_like(phi),
                             torch.zeros_like(phi), c, w)

    print(f'=== Phase-Field + CSF Bubble ===', flush=True)
    print(f'Grid: {nx}x{ny}x{nz} R0={R0} p_ratio={p_ratio} gamma={gamma}', flush=True)
    print(f'Uniform ρ={rho_liq} (no density contrast)', flush=True)
    print(f'p_gas0={p_gas0:.4f} p_inf={p_inf:.4f} R_eq={R_eq:.3f} (ratio={R_eq/R0:.4f})', flush=True)
    print(f'Cahn-Hilliard: A={A_coef} B={B_coef} κ={kappa} τ_g={tau_g}', flush=True)
    print(f'CSF: W={W} factor={csf_factor} F={csf_factor}·Δp·∇φ', flush=True)
    print(f'Collision: {collision} (τ_f={tau_f})', flush=True)
    print(flush=True)

    ts_lbm, Rs_lbm = [], []

    for step in range(1, n_steps + 1):
        # === 1. Macroscopic from f (uniform density) ===
        rho = f.sum(0)  # should be ~1.0 everywhere
        ux = (f * cx3d).sum(0) / rho.clamp(min=1e-6)
        uy = (f * cy3d).sum(0) / rho.clamp(min=1e-6)
        uz = (f * cz3d).sum(0) / rho.clamp(min=1e-6)

        # === 2. Phase field from g ===
        phi = g.sum(0).clamp(-1.0, 1.0)

        # === 3. Gas volume & pressure (mass conservation) ===
        # Use threshold count (φ<0 = gas) instead of integral — robust to interface sharpening
        V_gas = float((phi < 0).float().mul(fluid_mask.float()).sum())
        V_gas = max(V_gas, 1.0)
        # Adiabatic: p_gas = p_gas0 * (V0/V_gas)^gamma
        p_gas = p_gas0 * (V0 / V_gas) ** gamma
        dp = p_gas - p_inf  # pressure jump

        # === 4. Density from phase field + gas mass conservation + hydrostatic gradient ===
        rho_gas = p_gas / cs2
        # Liquid density includes hydrostatic gradient: ρ_liq(y) = ρ_liq - ρ_liq*g*y/cs²
        rho_liq_h = rho_liq - rho_liq * gz * yy.float() / cs2
        rho_field = rho_gas + (rho_liq_h - rho_gas) * (phi + 1) / 2

        # No CSF force — density contrast drives the flow
        fx_csf = torch.zeros_like(rho)
        fy_csf = torch.zeros_like(rho)
        fz_csf = torch.zeros_like(rho)

        # === 5. Flow collision (f) — selectable collision with density from phase field ===
        # Replace density in f with rho_field (keep non-equilibrium)
        rho_post = f.sum(0)
        ux_post = (f * cx3d).sum(0) / rho_post.clamp(min=1e-6)
        uy_post = (f * cy3d).sum(0) / rho_post.clamp(min=1e-6)
        uz_post = (f * cz3d).sum(0) / rho_post.clamp(min=1e-6)
        feq_new = equilibrium3d(rho_field.clamp(min=1e-6, max=3.0),
                                ux_post.clamp(-0.5, 0.5),
                                uy_post.clamp(-0.5, 0.5),
                                uz_post.clamp(-0.5, 0.5), device=dev)
        feq_old = equilibrium3d(rho_post.clamp(min=1e-6, max=3.0),
                                ux_post.clamp(-0.5, 0.5),
                                uy_post.clamp(-0.5, 0.5),
                                uz_post.clamp(-0.5, 0.5), device=dev)
        f = f - feq_old + feq_new  # replace density, keep non-equilibrium

        # Collision operator (selectable)
        if collision == 'mrt':
            from tensorlbm.solver3d import collide_mrt3d
            f = collide_mrt3d(f, tau_f)
        elif collision == 'trt':
            from tensorlbm.solver3d import collide_trt3d
            f = collide_trt3d(f, tau_f)
        elif collision == 'rlbm':
            from tensorlbm.solver3d import collide_rlbm3d
            f = collide_rlbm3d(f, tau_f)
        else:  # bgk
            feq = equilibrium3d(rho_field.clamp(min=1e-6, max=3.0),
                                ux_post.clamp(-0.5, 0.5),
                                uy_post.clamp(-0.5, 0.5),
                                uz_post.clamp(-0.5, 0.5), device=dev)
            f = f - (f - feq) / tau_f
        # Gravity on water phase only (Guo body force)
        if gz > 0:
            Fy = -rho_liq * gz * (phi + 1.0) / 2.0  # downward, only where φ≈+1 (liquid)
            f = f + (1.0 - 0.5/tau_f) * w * cy3d * Fy.unsqueeze(0) / cs2
        f = f.clamp(min=0.0, max=3.0)

        # === 6. Phase field update (FD Cahn-Hilliard, NOT LBM) ===
        # ∂φ/∂t + u·∇φ = M∇²μ,  μ = -Aφ + Bφ³ - κ∇²φ
        # Upwind advection (low numerical diffusion) + Cahn-Hilliard diffusion
        phi = g.sum(0).clamp(-1.0, 1.0)

        # Upwind advection: φ -= dt * u·∇φ
        ux_s = ux.clamp(-0.5, 0.5)
        uy_s = uy.clamp(-0.5, 0.5)
        uz_s = uz.clamp(-0.5, 0.5)
        # Upwind gradients (first-order, stable)
        dphi_dx = torch.where(ux_s > 0,
            phi - torch.roll(phi, 1, dims=2),
            torch.roll(phi, -1, dims=2) - phi) * ux_s
        dphi_dy = torch.where(uy_s > 0,
            phi - torch.roll(phi, 1, dims=1),
            torch.roll(phi, -1, dims=1) - phi) * uy_s
        dphi_dz = torch.where(uz_s > 0,
            phi - torch.roll(phi, 1, dims=0),
            torch.roll(phi, -1, dims=0) - phi) * uz_s
        phi_adv = phi - (dphi_dx + dphi_dy + dphi_dz)

        # Cahn-Hilliard: μ = -Aφ + Bφ³ - κ∇²φ, then φ += M·∇²μ
        lap_phi = _laplacian_3d(phi_adv)
        mu = -A_coef * phi_adv + B_coef * phi_adv**3 - kappa * lap_phi
        lap_mu = _laplacian_3d(mu)
        M_mob = cs2 * (tau_g - 0.5)  # mobility
        phi_new = phi_adv + M_mob * lap_mu

        # Fakhari anti-diffusion: drive φ towards ±1 in bulk
        # eF = (1-φ²)/W * sign(φ), active only at interface (|φ|<1)
        W_ac = 4.0  # interface width for anti-diffusion
        alpha_ac = 0.02  # anti-diffusion strength
        ac_source = alpha_ac * (1.0 - phi_new**2) / W_ac * torch.sign(phi_new)
        phi_new = phi_new + ac_source

        phi_new = phi_new.clamp(-1.0, 1.0)
        # Fix solid cells to liquid (φ=+1)
        phi_new[solid] = 1.0

        # Reconstruct g from updated φ (for next step's macroscopic)
        g = _init_g_equilibrium(phi_new, ux_s, uy_s, uz_s, c, w)

        # === 7. Streaming (f only — g is FD-updated, not LBM) ===
        f = stream3d(f)

        # === 8. Bounce-back at walls (f only) ===
        f = bounce_back_cells_3d(f, solid)

        # === 9. Wall BC: pressure p_inf (ρ=1.0, u=0) ===
        feq_wall = equilibrium3d(
            torch.full_like(rho, rho_liq),
            torch.zeros_like(rho), torch.zeros_like(rho),
            torch.zeros_like(rho), device=dev)
        f[:, solid] = feq_wall[:, solid]

        # === 10. Measurement ===
        if step % 100 == 0 or step == n_steps:
            phi_now = g.sum(0).clamp(-1.0, 1.0)
            V_now = float((phi_now < 0).float().mul(fluid_mask.float()).sum())
            R_now = (3.0 * V_now / (4.0 * math.pi)) ** (1.0/3.0) if V_now > 0 else 0
            R_ratio = R_now / R0

            p_gas_now = p_gas0 * (V0 / max(V_now, 1.0)) ** gamma
            rho_max = float(rho[fluid_mask].max())
            rho_min = float(rho[fluid_mask].min())
            phi_min = float(phi_now[fluid_mask].min())
            phi_max = float(phi_now[fluid_mask].max())

            ts_lbm.append(step)
            Rs_lbm.append(R_now)

            status = "EXPAND" if R_ratio > R_eq/R0 + 0.01 else ("CONTRACT" if R_ratio < R_eq/R0 - 0.01 else "STABLE")
            gas_mask = (phi_now < 0) & fluid_mask
            cy_b = float(yy[gas_mask].float().mean()) if gas_mask.any() else 0
            v_b = float(uy[gas_mask].mean()) if gas_mask.any() else 0
            print(f'step {step:4d}: R={R_now:.3f} ratio={R_ratio:.4f} R_eq={R_eq/R0:.4f} '
                  f'V={V_now:.0f} p_gas={p_gas_now:.4f} dp={p_gas_now-p_inf:+.4f} '
                  f'cy={cy_b:.1f} v={v_b:+.4f} '
                  f'ρ=[{rho_min:.3f},{rho_max:.3f}] φ=[{phi_min:.2f},{phi_max:.2f}] {status}',
                  flush=True)

    return np.array(ts_lbm), np.array(Rs_lbm)


def _init_g_equilibrium(phi, ux, uy, uz, c, w):
    """Initialize g distribution from phase field φ and velocity u."""
    cx = c[:, 0].float().view(19, 1, 1, 1)
    cy = c[:, 1].float().view(19, 1, 1, 1)
    cz = c[:, 2].float().view(19, 1, 1, 1)
    wv = w.view(19, 1, 1, 1)
    cu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0) + cz * uz.unsqueeze(0)
    u_sq = (ux**2 + uy**2 + uz**2).unsqueeze(0)
    return wv * phi.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u_sq)


def _laplacian_3d(field):
    """3D Laplacian via central differences (periodic)."""
    return (torch.roll(field, 1, dims=0) + torch.roll(field, -1, dims=0)
            + torch.roll(field, 1, dims=1) + torch.roll(field, -1, dims=1)
            + torch.roll(field, 1, dims=2) + torch.roll(field, -1, dims=2)
            - 6.0 * field)


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Phase-Field + CSF bubble vs RP equation')
    p.add_argument('--nx', type=int, default=48)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--device', default='sdaa:0')
    p.add_argument('--p-ratio', type=float, default=1.5, help='p_gas0/p_liq (>1 expand, <1 contract)')
    p.add_argument('--tau-f', type=float, default=0.8, help='flow relaxation time')
    p.add_argument('--tau-g', type=float, default=0.55, help='phase field relaxation')
    p.add_argument('--A', type=float, default=0.2, help='Cahn-Hilliard double-well A')
    p.add_argument('--kappa', type=float, default=0.1, help='Cahn-Hilliard gradient penalty')
    p.add_argument('--csf-factor', type=float, default=0.5, help='CSF force factor')
    p.add_argument('--collision', default='bgk', choices=['bgk', 'mrt', 'trt', 'rlbm'],
                   help='Collision operator: bgk, mrt, trt, rlbm')
    p.add_argument('--gz', type=float, default=0.0, help='Gravity on liquid phase')
    p.add_argument('--rise', action='store_true', help='Bubble rise mode')
    g = p.parse_args()

    R0 = 8.0
    rho_liq = 1.0
    gamma = 1.4

    if g.rise:
        g.nx = 48; g.ny = 128; g.nz = 48
        g.gz = g.gz if g.gz > 0 else 0.001
        g.p_ratio = 1.0  # pressure equilibrium, only buoyancy
        g.steps = 4000

    # 1. RP equation
    print('='*60)
    print('Solving RP equation...')
    ts_rp, Rs_rp = solve_rp(R0, g.p_ratio, rho_liq, gamma, g.steps)
    R_eq = R0 * (g.p_ratio / 1.0) ** (1.0 / (3.0 * gamma))
    print(f'R0={R0} R_eq={R_eq:.3f} R_eq/R0={R_eq/R0:.4f}')
    print(f'RP: R_min={Rs_rp.min():.3f} R_max={Rs_rp.max():.3f} R_final={Rs_rp[-1]:.3f}')
    print()

    # 2. Phase-Field + CSF
    print('='*60)
    print('Running Phase-Field + CSF LBM...')
    ts_lbm, Rs_lbm = run_pf_csf_bubble(
        g.nx, g.ny, g.nz, g.steps, g.device, g.p_ratio, gamma,
        tau_f=g.tau_f, tau_g=g.tau_g, A_coef=g.A, B_coef=g.A,
        kappa=g.kappa, csf_factor=g.csf_factor, collision=g.collision,
        gz=g.gz)
    print()

    # 3. Summary
    print('='*60)
    print('=== COMPARISON SUMMARY ===')
    print(f'{"":16s} {"R0":>8s} {"R_eq":>8s} {"R_min":>8s} {"R_max":>8s} {"R_final":>8s}')
    print(f'{"RP":16s} {R0:8.3f} {R_eq:8.3f} {Rs_rp.min():8.3f} {Rs_rp.max():8.3f} {Rs_rp[-1]:8.3f}')
    print(f'{"PF+CSF":16s} {R0:8.3f} {R_eq:8.3f} {Rs_lbm.min():8.3f} {Rs_lbm.max():8.3f} {Rs_lbm[-1]:8.3f}')
    print()

    # Error
    rp_final = Rs_rp[-1]
    lbm_err = abs(Rs_lbm[-1] - rp_final) / rp_final * 100
    print(f'Final R error vs RP: {lbm_err:.1f}%')

    # MAPE over all common timesteps
    n_common = min(len(Rs_rp), len(Rs_lbm))
    if n_common > 0:
        mape = np.mean(np.abs(Rs_lbm[:n_common] - Rs_rp[:n_common]) / Rs_rp[:n_common]) * 100
        print(f'MAPE over {n_common} points: {mape:.1f}%')
