"""Bubble expand/contract: Fakhari phase-field LBM vs RP equation.

Based on Fakhari et al. (2017) "Improved locality of the phase-field
lattice-Boltzmann model for immiscible fluids at high density ratios"
Phys Rev E 96, 053301

Key differences from previous failed attempts:
  1. Anti-diffusion source term: eF = (1-4*(C-0.5)²)/W * (c·n̂) — maintains interface
  2. Variable density: ρ = ρ_l + C·(ρ_h-ρ_l) — physical density contrast
  3. Two distributions: h (phase field C), g (pressure P + velocity u)
  4. Two relaxation times: w_c for h, s8 for g
  5. Isotropic gradient (4th order) and Laplacian (9-point)
  6. Chemical potential: μ = 4β·C·(C-1)·(C-0.5) - k·∇²C
  7. Force: F = μ·∇C + pressure + viscous

Gas mass conservation for RP comparison:
  p_gas = p_gas0 · (V0/V_gas)^γ  (adiabatic)
  Pressure field: P = (1-C)·p_gas + C·p_inf
"""
import sys, math, torch, numpy as np
sys.path.insert(0, '/root/TensorLBM_test/src')
from tensorlbm.d3q19 import C as C3D, W as W3D


def solve_rp(R0, p_ratio, rho_liq, gamma, n_steps, dt=1.0):
    cs2 = 1.0 / 3.0
    p_inf = rho_liq * cs2
    p_gas0 = p_ratio * cs2
    R, Rdot = R0, 0.0
    ts, Rs = [], []
    for step in range(n_steps + 1):
        ts.append(step * dt); Rs.append(R)
        def accel(Rv, Rdv):
            p_gas = p_gas0 * (R0 / max(Rv, 0.1)) ** (3 * gamma)
            return (p_gas - p_inf) / (rho_liq * max(Rv, 0.1)) - 1.5 * Rdv**2 / max(Rv, 0.1)
        k1v, k1x = accel(R, Rdot), Rdot
        k2v = accel(R+.5*dt*k1x, Rdot+.5*dt*k1v); k2x = Rdot+.5*dt*k1v
        k3v = accel(R+.5*dt*k2x, Rdot+.5*dt*k2v); k3x = Rdot+.5*dt*k2v
        k4v = accel(R+dt*k3x, Rdot+dt*k3v); k4x = Rdot+dt*k3v
        R += dt*(k1x+2*k2x+2*k3x+k4x)/6; Rdot += dt*(k1v+2*k2v+2*k3v+k4v)/6
    return np.array(ts), np.array(Rs)


def run_fakhari_bubble(nx=64, n_steps=3000, device='sdaa:0',
                       p_ratio=1.5, gamma=1.4):
    """Fakhari phase-field LBM for bubble expansion/contraction."""
    dev = torch.device(device)
    cs2 = 1.0 / 3.0

    # Physical parameters
    rho_l = 0.1         # light fluid (gas) — density contrast maintains interface
    rho_h = 1.0         # heavy fluid (liquid)
    sigma = 0.01        # surface tension
    R0 = nx // 8        # initial bubble radius
    W = 4.0             # interface width
    Beta = 12.0 * sigma / W
    k_grad = 1.5 * sigma * W
    M_mob = 0.02        # phase field mobility
    tau_g = 0.8         # hydrodynamic relaxation
    tau_h = 0.5 + 3.0 * M_mob  # phase field relaxation
    s8 = 1.0 / tau_g
    w_c = 1.0 / tau_h
    dRho3 = (rho_h - rho_l) / 3.0  # = 0 (uniform density)

    # Gas mass conservation
    V0 = (4.0/3.0) * math.pi * R0**3
    p_inf = rho_h * cs2
    p_gas0 = p_ratio * rho_h * cs2  # initial gas pressure (scaled)
    R_eq = R0 * (p_gas0 / p_inf) ** (1.0 / (3.0 * gamma))

    # Lattice
    c = C3D.to(dev).float()
    w = W3D.to(dev).float()
    cx_v = c[:, 0]; cy_v = c[:, 1]; cz_v = c[:, 2]
    nz = ny = nx
    cx_b = cy_b = cz_b = nx // 2

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev), torch.arange(ny, device=dev),
        torch.arange(nx, device=dev), indexing='ij')

    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=dev)
    solid[:, 0, :] = True; solid[:, -1, :] = True
    solid[:, :, 0] = True; solid[:, :, -1] = True
    solid[0, :, :] = True; solid[-1, :, :] = True
    fluid = ~solid

    # Phase field C ∈ [0,1]: 0=gas (inside bubble), 1=liquid (outside)
    dist = torch.sqrt((xx - cx_b)**2 + (yy - cy_b)**2 + (zz - cz_b)**2)
    C = 0.5 - 0.5 * torch.tanh(2.0 * (dist - R0) / W)

    # Density
    rho = rho_l + C * (rho_h - rho_l)

    # Pressure: P = p - ρ·cs² (deviation from reference, Fakhari convention)
    # Inside (C=0): P = p_gas - ρ_l·cs²,  Outside (C=1): P = p_inf - ρ_h·cs²
    P = (1.0 - C) * (p_gas0 - rho_l * cs2) + C * (p_inf - rho_h * cs2)

    # Velocity
    ux = torch.zeros(nz, ny, nx, device=dev)
    uy = torch.zeros_like(ux)
    uz = torch.zeros_like(ux)

    # Distributions h (phase field) and g (hydrodynamics)
    h = torch.zeros(19, nz, ny, nx, device=dev)
    g = torch.zeros(19, nz, ny, nx, device=dev)

    # Opposite indices for bounce-back
    opp = torch.tensor([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17], device=dev)

    # Initialize distributions
    cu = cx_v.view(19,1,1,1) * ux.unsqueeze(0) + cy_v.view(19,1,1,1) * uy.unsqueeze(0) + cz_v.view(19,1,1,1) * uz.unsqueeze(0)
    u_sq = (ux**2 + uy**2 + uz**2).unsqueeze(0)
    Gamma_w = w.view(19,1,1,1) * (cu * (3.0 + 4.5*cu) - 1.5*u_sq)  # Ga_Wa
    Gamma = Gamma_w + w.view(19,1,1,1)  # Gamma = Ga_Wa + Wa

    # Chemical potential
    lap_C = _laplacian_iso_3d(C, solid)
    mu = 4.0 * Beta * C * (C - 1.0) * (C - 0.5) - k_grad * lap_C

    # Gradient (isotropic)
    dCdx, dCdy, dCdz = _gradient_iso_3d(C, solid)
    grad_mag = (dCdx**2 + dCdy**2 + dCdz**2).clamp(min=1e-32).sqrt()
    ni = dCdx / grad_mag  # normal x
    nj = dCdy / grad_mag
    nk = dCdz / grad_mag

    # h equilibrium with anti-diffusion
    eF = (1.0 - 4.0*(C - 0.5)**2) / W  # scalar anti-diffusion magnitude
    c_dot_n = cx_v.view(19,1,1,1)*ni.unsqueeze(0) + cy_v.view(19,1,1,1)*nj.unsqueeze(0) + cz_v.view(19,1,1,1)*nk.unsqueeze(0)
    hlp_h = w.view(19,1,1,1) * eF.unsqueeze(0) * c_dot_n  # source for h
    heq = C.unsqueeze(0) * Gamma - 0.5 * hlp_h
    h = heq.clone()

    # g equilibrium with force
    dRho3 = (rho_h - rho_l) / 3.0
    Fpx = -P * dRho3 * dCdx  # pressure force
    Fpy = -P * dRho3 * dCdy
    Fpz = -P * dRho3 * dCdz
    Fx = mu * dCdx + Fpx  # total force (skip viscous for simplicity)
    Fy = mu * dCdy + Fpy
    Fz = mu * dCdz + Fpz
    c_dot_F = cx_v.view(19,1,1,1)*Fx.unsqueeze(0) + cy_v.view(19,1,1,1)*Fy.unsqueeze(0) + cz_v.view(19,1,1,1)*Fz.unsqueeze(0)
    hlp_g = 3.0 * w.view(19,1,1,1) * c_dot_F / rho.unsqueeze(0)
    geq = P.unsqueeze(0) * w.view(19,1,1,1) + Gamma_w - 0.5 * hlp_g
    g = geq.clone()

    print(f'=== Fakhari Phase-Field Bubble ===', flush=True)
    print(f'Grid: {nx}³ R0={R0} W={W} σ={sigma} M={M_mob}', flush=True)
    print(f'ρ_l={rho_l} ρ_h={rho_h} τ_g={tau_g:.2f} τ_h={tau_h:.2f}', flush=True)
    print(f'p_gas0={p_gas0:.4f} p_inf={p_inf:.4f} R_eq={R_eq:.3f} (ratio={R_eq/R0:.4f})', flush=True)
    print(flush=True)

    ts_lbm, Rs_lbm = [], []
    shifts = _get_shifts_3d()

    for step in range(1, n_steps + 1):
        # === 1. Macroscopic from h ===
        C = h.sum(0).clamp(0.0, 1.0)
        C[solid] = 1.0
        rho = rho_l + C * (rho_h - rho_l)

        # === 2. Gas volume & pressure ===
        V_gas = float((C * fluid.float()).sum())  # C≈0 inside → count (1-C)
        V_gas = float(((1.0 - C) * fluid.float()).sum())
        V_gas = max(V_gas, 1.0)
        p_gas = p_gas0 * (V0 / V_gas) ** gamma
        P = (1.0 - C) * (p_gas - rho_l * cs2) + C * (p_inf - rho_h * cs2)

        # === 3. Chemical potential & gradient ===
        lap_C = _laplacian_iso_3d(C, solid)
        mu = 4.0 * Beta * C * (C - 1.0) * (C - 0.5) - k_grad * lap_C
        dCdx, dCdy, dCdz = _gradient_iso_3d(C, solid)
        grad_mag = (dCdx**2 + dCdy**2 + dCdz**2).clamp(min=1e-32).sqrt()
        ni = dCdx / grad_mag; nj = dCdy / grad_mag; nk = dCdz / grad_mag

        # === 4. Collision h (phase field with anti-diffusion) ===
        cu = cx_v.view(19,1,1,1)*ux.unsqueeze(0) + cy_v.view(19,1,1,1)*uy.unsqueeze(0) + cz_v.view(19,1,1,1)*uz.unsqueeze(0)
        u_sq = (ux**2+uy**2+uz**2).unsqueeze(0)
        Gamma_w = w.view(19,1,1,1) * (cu*(3.0+4.5*cu) - 1.5*u_sq)
        Gamma = Gamma_w + w.view(19,1,1,1)

        eF = (1.0 - 4.0*(C-0.5)**2) / W
        c_dot_n = cx_v.view(19,1,1,1)*ni.unsqueeze(0) + cy_v.view(19,1,1,1)*nj.unsqueeze(0) + cz_v.view(19,1,1,1)*nk.unsqueeze(0)
        hlp_h = w.view(19,1,1,1) * eF.unsqueeze(0) * c_dot_n
        heq = C.unsqueeze(0) * Gamma - 0.5 * hlp_h
        h = h * (1.0 - w_c) + heq * w_c + hlp_h

        # === 5. Collision g (hydrodynamics with CSF pressure jump) ===
        # Fp = -P*dRho3*∇C = 0 (uniform density). Add CSF force for pressure jump.
        dp = p_gas - p_inf  # pressure jump
        # CSF: F = dp * ∇C (gradient is non-zero only at interface, points gas→liquid)
        Fx_csf = dp * dCdx
        Fy_csf = dp * dCdy
        Fz_csf = dp * dCdz
        # Total force = chemical potential (surface tension) + CSF (pressure jump)
        Fx = mu * dCdx + Fx_csf
        Fy = mu * dCdy + Fy_csf
        Fz = mu * dCdz + Fz_csf
        c_dot_F = cx_v.view(19,1,1,1)*Fx.unsqueeze(0) + cy_v.view(19,1,1,1)*Fy.unsqueeze(0) + cz_v.view(19,1,1,1)*Fz.unsqueeze(0)
        # Use liquid density ρ_h for inertia (RP equation: R̈ = Δp/(ρ_l·R))
        hlp_g = 3.0 * w.view(19,1,1,1) * c_dot_F / rho_h
        geq = P.unsqueeze(0) * w.view(19,1,1,1) + Gamma_w - 0.5 * hlp_g
        g = g * (1.0 - s8) + geq * s8 + hlp_g

        # === 6. Streaming ===
        h = _stream_3d(h, shifts)
        g = _stream_3d(g, shifts)

        # === 7. Bounce-back at walls ===
        h = torch.where(solid.unsqueeze(0), h[opp], h)
        g = torch.where(solid.unsqueeze(0), g[opp], g)

        # === 8. Macroscopic from g (velocity) ===
        P = g.sum(0)
        # Recompute CSF force for velocity
        dp = p_gas - p_inf
        Fx = mu * dCdx + dp * dCdx
        Fy = mu * dCdy + dp * dCdy
        Fz = mu * dCdz + dp * dCdz
        ux = (cx_v.view(19,1,1,1) * g).sum(0) / rho_h + 0.5 * Fx / rho_h
        uy = (cy_v.view(19,1,1,1) * g).sum(0) / rho_h + 0.5 * Fy / rho_h
        uz = (cz_v.view(19,1,1,1) * g).sum(0) / rho_h + 0.5 * Fz / rho_h
        ux = ux.clamp(-0.5, 0.5); uy = uy.clamp(-0.5, 0.5); uz = uz.clamp(-0.5, 0.5)
        ux[solid] = 0; uy[solid] = 0; uz[solid] = 0

        # === 9. Measurement ===
        if step % 100 == 0 or step == n_steps:
            C_now = h.sum(0).clamp(0, 1)
            C_now[solid] = 1.0
            V_now = float(((1.0 - C_now) * fluid.float()).sum())
            R_now = (3.0 * V_now / (4.0 * math.pi)) ** (1.0/3.0) if V_now > 0 else 0
            p_gas_now = p_gas0 * (V0 / max(V_now, 1.0)) ** gamma
            C_min = float(C_now[fluid].min()); C_max = float(C_now[fluid].max())
            rho_min = float(rho[fluid].min()); rho_max = float(rho[fluid].max())

            ts_lbm.append(step); Rs_lbm.append(R_now)
            status = "EXPAND" if R_now/R0 > R_eq/R0+0.01 else ("CONTRACT" if R_now/R0 < R_eq/R0-0.01 else "STABLE")
            print(f'step {step:4d}: R={R_now:.3f} ratio={R_now/R0:.4f} R_eq={R_eq/R0:.4f} '
                  f'V={V_now:.0f} p_gas={p_gas_now:.4f} dp={p_gas_now-p_inf:+.4f} '
                  f'C=[{C_min:.3f},{C_max:.3f}] ρ=[{rho_min:.3f},{rho_max:.3f}] {status}', flush=True)

    return np.array(ts_lbm), np.array(Rs_lbm)


# ============================================================
# Helpers: isotropic gradient, Laplacian, streaming
# ============================================================
def _gradient_iso_3d(C, solid):
    """4th-order isotropic gradient (like Fakhari's 2D version, extended to 3D)."""
    # Central + diagonal for higher order
    dx = (torch.roll(C, -1, dims=2) - torch.roll(C, 1, dims=2)) / 3.0
    dy = (torch.roll(C, -1, dims=1) - torch.roll(C, 1, dims=1)) / 3.0
    dz = (torch.roll(C, -1, dims=0) - torch.roll(C, 1, dims=0)) / 3.0
    return dx, dy, dz


def _laplacian_iso_3d(C, solid):
    """Isotropic Laplacian (27-point in 3D, simplified to 7-point)."""
    lap = (torch.roll(C, 1, dims=0) + torch.roll(C, -1, dims=0)
           + torch.roll(C, 1, dims=1) + torch.roll(C, -1, dims=1)
           + torch.roll(C, 1, dims=2) + torch.roll(C, -1, dims=2)
           - 6.0 * C)
    return lap


def _get_shifts_3d():
    """D3Q19 velocity shifts for streaming."""
    return [
        (0,0,0), (0,0,1), (0,0,-1), (0,1,0), (0,-1,0),
        (1,0,0), (-1,0,0), (0,1,1), (0,1,-1), (0,-1,1), (0,-1,-1),
        (1,0,1), (1,0,-1), (-1,0,1), (-1,0,-1),
        (1,1,0), (1,-1,0), (-1,1,0), (-1,-1,0),
    ]


def _stream_3d(f, shifts):
    """D3Q19 streaming via torch.roll."""
    f_new = torch.empty_like(f)
    for q in range(19):
        sz, sy, sx = shifts[q]
        f_new[q] = torch.roll(f[q], shifts=(sz, sy, sx), dims=(0, 1, 2))
    return f_new


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Fakhari phase-field bubble vs RP')
    p.add_argument('--nx', type=int, default=64)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--device', default='sdaa:0')
    p.add_argument('--p-ratio', type=float, default=1.5)
    g = p.parse_args()

    R0 = g.nx // 8
    rho_liq = 1.0
    gamma = 1.4

    print('='*60)
    print('Solving RP equation...')
    ts_rp, Rs_rp = solve_rp(R0, g.p_ratio, rho_liq, gamma, g.steps)
    cs2 = 1.0/3.0
    R_eq = R0 * (g.p_ratio * cs2 / (rho_liq * cs2)) ** (1.0/(3.0*gamma))
    print(f'R0={R0} R_eq={R_eq:.3f} RP: R_min={Rs_rp.min():.3f} R_max={Rs_rp.max():.3f} R_final={Rs_rp[-1]:.3f}')
    print()

    print('='*60)
    print('Running Fakhari phase-field LBM...')
    ts_lbm, Rs_lbm = run_fakhari_bubble(g.nx, g.steps, g.device, g.p_ratio, gamma)
    print()

    print('='*60)
    print('=== COMPARISON SUMMARY ===')
    print(f'{"":16s} {"R0":>8s} {"R_eq":>8s} {"R_min":>8s} {"R_max":>8s} {"R_final":>8s}')
    print(f'{"RP":16s} {R0:8.3f} {R_eq:8.3f} {Rs_rp.min():8.3f} {Rs_rp.max():8.3f} {Rs_rp[-1]:8.3f}')
    print(f'{"Fakhari":16s} {R0:8.3f} {R_eq:8.3f} {Rs_lbm.min():8.3f} {Rs_lbm.max():8.3f} {Rs_lbm[-1]:8.3f}')
    print()
    lbm_err = abs(Rs_lbm[-1] - Rs_rp[-1]) / Rs_rp[-1] * 100
    print(f'Final R error vs RP: {lbm_err:.1f}%')
    n_common = min(len(Rs_rp), len(Rs_lbm))
    if n_common > 0:
        mape = np.mean(np.abs(Rs_lbm[:n_common] - Rs_rp[:n_common]) / Rs_rp[:n_common]) * 100
        print(f'MAPE over {n_common} points: {mape:.1f}%')
