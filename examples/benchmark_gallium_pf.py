#!/usr/bin/env python
"""Gallium melting — phase-field (Cahn-Hilliard + Fakhari anti-diffusion) + thermal LBM.

Combines:
  - PF interface tracking from hull_fs_pf_mrt (Cahn-Hilliard + anti-diffusion,
    maintains sharp φ=±1 interface, no f_l diffusion)
  - Phase-change source from benchmark_stefan_freezing (enthalpy-consistent:
    Δφ = 2·cp·(T−T_m)/L in superheated solid, latent heat g+=w·L·(−Δφ)/2·cp)
  - D2Q5 thermal LBM transports T (not H), so no mushy-zone f_l diffusion

φ = +1 liquid, −1 solid.  Solid (φ<0): bounce-back (u=0).
Phase change: solid (φ<0) with T>T_m → melts (φ↑), latent heat absorbed.
"""
from __future__ import annotations
import argparse, math, os, sys
from pathlib import Path
import numpy as np
import torch

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tensorlbm.d3q19 import C, W, OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d
from benchmark_gallium_melting import (
    C_D2Q5, W_D2Q5, equilibrium_thermal, collide_thermal_bgk, stream_thermal,
    macroscopic_thermal, apply_temperature_bc, bounce_back_solid, apply_buoyancy,
    compute_interface_velocity, rebuild_phi_from_interface,
    _GV_FO, _GV_FLIQ,
)

W_D2Q5_DEV = W_D2Q5.float()  # (5,)


def _laplacian(phi):
    """Three-dimensional Laplacian with no-flux outer faces.

    ``torch.roll`` is periodic and therefore made the liquid hot wall
    communicate with the cold wall (and the top with the bottom) through the
    phase-field chemical potential.  The Gallium cavity has physical walls:
    its horizontal faces are adiabatic/no-flux and the vertical wall values
    are imposed separately below.  Replicated ghost values give the required
    zero normal derivative without a wrap-around phase flux.
    """
    p_zm = torch.cat((phi[:1], phi[:-1]), dim=0)
    p_zp = torch.cat((phi[1:], phi[-1:]), dim=0)
    p_ym = torch.cat((phi[:, :1], phi[:, :-1]), dim=1)
    p_yp = torch.cat((phi[:, 1:], phi[:, -1:]), dim=1)
    p_xm = torch.cat((phi[:, :, :1], phi[:, :, :-1]), dim=2)
    p_xp = torch.cat((phi[:, :, 1:], phi[:, :, -1:]), dim=2)
    return p_zm + p_zp + p_ym + p_yp + p_xm + p_xp - 6.0 * phi


def stefan_phase_source(phi, temperature, *, cp, latent_heat,
                        melting_temperature, rate=1.0, active_mask=None):
    """Return a locally enthalpy-conservative Stefan phase increment.

    ``delta_temperature`` is the sensible increment to add to the thermal
    field. Consequently ``cp*T + L*(phi+1)/2`` is unchanged before transport.
    Fixed walls must be excluded because their imposed phase cannot accept a
    latent-heat source.
    """
    active = torch.ones_like(phi, dtype=torch.bool) if active_mask is None else active_mask
    superheat = torch.clamp(temperature - melting_temperature, min=0.0)
    subcool = torch.clamp(melting_temperature - temperature, min=0.0)
    delta_phi = (2.0 * rate * cp / latent_heat) * torch.where(
        phi < 0, superheat, -subcool)
    delta_phi = torch.where(active, delta_phi.clamp(min=-1.0 - phi, max=1.0 - phi),
                            torch.zeros_like(delta_phi))
    return delta_phi, phase_increment_to_temperature(
        delta_phi, cp=cp, latent_heat=latent_heat)


def phase_increment_to_temperature(delta_phi, *, cp, latent_heat):
    """Temperature increment that balances a phase-fraction increment.

    The PF flux changes the local latent enthalpy just as the Stefan source
    does.  Applying this increment to *every* phase update preserves
    ``cp*T + L*(phi+1)/2`` cell-by-cell before thermal transport.
    """
    return -latent_heat * delta_phi / (2.0 * cp)


def conservative_phase_field_update(phi, *, ux, uy, mobility, interface_mobility,
                                    interface_width, active_mask=None):
    """Conservative CH transport plus conservative interface compression.

    The former Fakhari ``sign(phi)`` increment was a non-conservative source
    that could undo Stefan melting after its latent heat had been removed.
    Every update here is a face-flux divergence; phase volume can therefore
    change only through the Stefan closure.
    """
    active = torch.ones_like(phi, dtype=torch.bool) if active_mask is None else active_mask

    def divergence(fx, fy):
        out = torch.zeros_like(phi)
        out[:, :, 1:] += fx[:, :, 1:]; out[:, :, :-1] -= fx[:, :, 1:]
        out[:, 1:, :] += fy[:, 1:, :]; out[:, :-1, :] -= fy[:, 1:, :]
        return out

    face_x = active[:, :, 1:] & active[:, :, :-1]
    face_y = active[:, 1:, :] & active[:, :-1, :]
    ux_face = 0.5 * (ux[:, :, 1:] + ux[:, :, :-1])
    uy_face = 0.5 * (uy[:, 1:, :] + uy[:, :-1, :])
    phi_x = torch.where(ux_face >= 0, phi[:, :, :-1], phi[:, :, 1:])
    phi_y = torch.where(uy_face >= 0, phi[:, :-1, :], phi[:, 1:, :])
    fx = torch.zeros_like(phi); fy = torch.zeros_like(phi)
    fx[:, :, 1:] = torch.where(face_x, ux_face * phi_x, torch.zeros_like(phi_x))
    fy[:, 1:, :] = torch.where(face_y, uy_face * phi_y, torch.zeros_like(phi_y))
    phi_next = phi - divergence(fx, fy)

    lap_phi = _laplacian(phi_next)
    mu = -0.2 * phi_next + 0.2 * phi_next ** 3 - 0.1 * lap_phi
    mux = mu[:, :, 1:] - mu[:, :, :-1]; muy = mu[:, 1:, :] - mu[:, :-1, :]
    fx.zero_(); fy.zero_()
    fx[:, :, 1:] = torch.where(face_x, -mobility * mux, torch.zeros_like(mux))
    fy[:, 1:, :] = torch.where(face_y, -mobility * muy, torch.zeros_like(muy))
    phi_next = phi_next - divergence(fx, fy)

    gx = torch.zeros_like(phi_next); gy = torch.zeros_like(phi_next)
    gx[:, :, 1:-1] = 0.5 * (phi_next[:, :, 2:] - phi_next[:, :, :-2])
    gy[:, 1:-1, :] = 0.5 * (phi_next[:, 2:, :] - phi_next[:, :-2, :])
    norm = torch.sqrt(gx * gx + gy * gy + 1e-12)
    qx = interface_mobility * (1.0 - phi_next * phi_next) * gx / norm / interface_width
    qy = interface_mobility * (1.0 - phi_next * phi_next) * gy / norm / interface_width
    fx.zero_(); fy.zero_()
    fx[:, :, 1:] = torch.where(face_x, 0.5 * (qx[:, :, 1:] + qx[:, :, :-1]), torch.zeros_like(mux))
    fy[:, 1:, :] = torch.where(face_y, 0.5 * (qy[:, 1:, :] + qy[:, :-1, :]), torch.zeros_like(muy))
    phi_next = phi_next - divergence(fx, fy)
    # Clipping would constitute another non-conservative phase source.
    return torch.where(active, phi_next, phi)


def phase_field_update_with_energy_closure(
        phi, *, ux, uy, mobility, interface_mobility, interface_width,
        cp, latent_heat, active_mask=None):
    """Apply PF fluxes with their matching local latent-energy transfer.

    A conservative Cahn--Hilliard update conserves *global* phase volume, but
    it transports latent enthalpy between cells. Because the thermal LBM
    transports sensible temperature, the corresponding local sensible-energy
    increment preserves ``cp*T + L*(phi+1)/2`` before thermal transport.
    """
    phi_next = conservative_phase_field_update(
        phi, ux=ux, uy=uy, mobility=mobility,
        interface_mobility=interface_mobility, interface_width=interface_width,
        active_mask=active_mask)
    return phi_next, phase_increment_to_temperature(
        phi_next - phi, cp=cp, latent_heat=latent_heat)


def run_gallium_pf(nx=40, ny=56, nz=1, tau=0.506, tau_T=0.8,
                   T_hot=1.0, T_cold=0.0, T_melt=0.148, T_init=None,
                   cp=1.0, L_latent=18.52, beta=0.1, gy=-0.001875,
                   u_clamp=0.15, k_melt=1.0, steps=8000, device="cpu",
                   log_every=1000, quiet=False):
    dev = torch.device(device)
    nu = (tau - 0.5) / 3.0
    alpha = (tau_T - 0.5) / 3.0
    Pr = nu / alpha
    Ste = cp * (T_hot - T_melt) / L_latent
    deltaT = T_hot - T_cold
    T_ref = T_melt
    g_mag = abs(gy)
    Ra = g_mag * beta * deltaT * nx ** 3 / (nu * alpha)
    Fo_factor = alpha / (nx * nx)
    if T_init is None:
        T_init = T_cold

    # PF parameters (from hull_fs_pf_mrt)
    A_coef, B_coef, kappa_ch = 0.2, 0.2, 0.1
    W_ac, alpha_ac = 4.0, 0.02
    M_mob = (1.0 / 3.0) * (tau_T - 0.5)
    w_d2q5_view = W_D2Q5.to(dev).float().view(5, 1, 1, 1)

    # Wall mask
    wall_mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=dev)
    wall_mask[:, :, 0] = True; wall_mask[:, :, -1] = True
    wall_mask[:, 0, :] = True; wall_mask[:, -1, :] = True
    # Only vertical wall values are prescribed phase values.  Horizontal faces
    # are zero-flux PF faces, not reset ghost cells: resetting them would be a
    # hidden phase source.  Face fluxes below close at both kinds of wall.
    phase_active = torch.ones_like(wall_mask)
    phase_active[:, :, 0] = False
    phase_active[:, :, -1] = False

    _, j_idx, i_idx = torch.meshgrid(
        torch.arange(nz, device=dev, dtype=torch.float32),
        torch.arange(ny, device=dev, dtype=torch.float32),
        torch.arange(nx, device=dev, dtype=torch.float32), indexing="ij")

    # Initial: all solid (φ=-1), hot wall liquid (φ=+1)
    phi = -torch.ones((nz, ny, nx), device=dev, dtype=torch.float32)
    phi[:, :, 0] = 1.0
    s = torch.full((ny,), 1.0, device=dev, dtype=torch.float32)
    # Temperature: subcooled solid, hot wall
    T_field = torch.full((nz, ny, nx), float(T_init), device=dev, dtype=torch.float32)
    T_field[:, :, 0] = T_hot
    T_field = T_field + 0.002 * torch.sin(math.pi * j_idx / max(ny - 1, 1)) * \
              torch.sin(math.pi * i_idx / max(nx - 1, 1))

    rho0 = torch.ones((nz, ny, nx), device=dev)
    u0 = torch.zeros_like(rho0)
    f = equilibrium3d(rho0, u0, u0.clone(), u0.clone(), device=dev)
    g = equilibrium_thermal(T_field, u0, u0.clone())
    g = apply_temperature_bc(g, T_hot, T_cold)

    if not quiet:
        print(f"\n{'─' * 72}")
        print(f"  Gallium melting — PF (Cahn-Hilliard + anti-diffusion) + thermal")
        print(f"  Grid: {nx} × {ny} × {nz}  Fo_final ≈ {Fo_factor*steps:.4f}")
        print(f"  Pr={Pr:.4f}  Ra={Ra:.2f}  Ste={Ste:.4f}  (physical Ste≈0.046)")
        print(f"  T_hot={T_hot} T_cold={T_cold} T_m={T_melt} cp={cp} L={L_latent}")
        print(f"  PF: A={A_coef} B={B_coef} κ={kappa_ch} W={W_ac} α_ac={alpha_ac} M={M_mob:.4f}")
        print(f"{'─' * 72}")
        print(f"  {'step':>6s} {'Fo':>8s} {'f_liq':>7s} {'s_top':>6s} {'s_mid':>6s} {'s_bot':>6s} {'u_max':>8s} {'T_min':>6s} {'T_max':>6s}")

    history = []
    with torch.no_grad():
        for step in range(1, steps + 1):
            # === 1. Macroscopic ===
            rho, ux, uy, uz = macroscopic3d(f)
            T = macroscopic_thermal(g)

            # === 2. Momentum: collide → stream → buoyancy → bounce-back ===
            f = collide_bgk3d(f, tau)
            f = stream3d(f)
            f = apply_buoyancy(f, rho, T, T_ref=T_ref, beta=beta, gy=gy)
            solid_mask = wall_mask | (phi < 0)
            f = bounce_back_solid(f, solid_mask)
            f = f.clamp(min=0.0, max=5.0)

            # === 3. Temperature: collide → stream → BC ===
            rho, ux, uy, uz = macroscopic3d(f)
            ux = ux.masked_fill(solid_mask, 0.0)
            uy = uy.masked_fill(solid_mask, 0.0)
            if u_clamp > 0:
                ux = ux.clamp(-u_clamp, u_clamp)
                uy = uy.clamp(-u_clamp, u_clamp)
            T = macroscopic_thermal(g)
            g = collide_thermal_bgk(g, T, ux, uy, tau_T=tau_T)
            g = stream_thermal(g)
            g = apply_temperature_bc(g, T_hot, T_cold)

            # === 4. Stefan phase source and its matching latent-heat sink ===
            # The source is applied once.  Its exact realized increment is used
            # for thermal energy so clipping cannot create an energy mismatch.
            T_cur = macroscopic_thermal(g)
            delta_phi, latent_temperature = stefan_phase_source(
                phi, T_cur, cp=cp, latent_heat=L_latent,
                melting_temperature=T_melt, rate=k_melt, active_mask=phase_active)
            phi = phi + delta_phi
            g = g + w_d2q5_view * latent_temperature.unsqueeze(0)
            g = apply_temperature_bc(g, T_hot, T_cold)

            # === 5. Conservative PF transport/sharpening ===
            # It redistributes φ but has no net source, protecting the Stefan
            # phase increment and its associated latent-energy bookkeeping.
            phi, pf_temperature = phase_field_update_with_energy_closure(
                phi, ux=ux.clamp(-0.5, 0.5), uy=uy.clamp(-0.5, 0.5),
                mobility=M_mob, interface_mobility=alpha_ac,
                interface_width=W_ac, cp=cp, latent_heat=L_latent,
                active_mask=phase_active)
            g = g + w_d2q5_view * pf_temperature.unsqueeze(0)
            g = apply_temperature_bc(g, T_hot, T_cold)
            phi[:, :, 0] = 1.0    # hot wall = liquid
            phi[:, :, -1] = -1.0  # cold wall = solid

            # === 6. NaN guard ===
            if step % 200 == 0:
                if torch.isnan(f).any() or torch.isnan(g).any() or torch.isnan(phi).any():
                    print(f"  WARNING: NaN at step {step} — stopping.")
                    break

            # === 7. Diagnostics ===
            if step % log_every == 0 or step == steps:
                T_d = macroscopic_thermal(g)
                f_l = float(((1.0 + phi) / 2.0).mean().item())
                s_per = ((1.0 + phi) / 2.0)[0].sum(dim=1)
                n20 = max(ny // 5, 1)
                s_top = float(s_per[-n20:].mean().item())
                s_mid = float(s_per[ny // 2 - n20 // 2: ny // 2 + n20 // 2].mean().item())
                s_bot = float(s_per[:n20].mean().item())
                rho_d, ux_d, uy_d, _ = macroscopic3d(f)
                u_mag = torch.sqrt(ux_d ** 2 + uy_d ** 2)
                liq_mask = ~solid_mask
                u_max = float(u_mag[liq_mask].max().item()) if liq_mask.any() else 0.0
                Fo = Fo_factor * step
                history.append({"step": step, "Fo": Fo, "f_liq": f_l,
                                "s_top": s_top, "s_mid": s_mid, "s_bot": s_bot,
                                "u_max": u_max, "T_min": float(T_d.min().item()),
                                "T_max": float(T_d.max().item())})
                if not quiet:
                    print(f"  {step:6d} {Fo:8.4f} {f_l:7.4f} {s_top:6.2f} {s_mid:6.2f} {s_bot:6.2f} {u_max:8.5f} {float(T_d.min().item()):6.3f} {float(T_d.max().item()):6.3f}", flush=True)

    # Final
    T_final = macroscopic_thermal(g)
    rho_f, ux_f, uy_f, _ = macroscopic3d(f)
    f_l_final = float(((1.0 + phi) / 2.0).mean().item())
    s_per_f = ((1.0 + phi) / 2.0)[0].sum(dim=1)
    n20 = max(ny // 5, 1)
    s_top_f = float(s_per_f[-n20:].mean().item())
    s_mid_f = float(s_per_f[ny // 2 - n20 // 2: ny // 2 + n20 // 2].mean().item())
    s_bot_f = float(s_per_f[:n20].mean().item())
    u_mag_f = torch.sqrt(ux_f ** 2 + uy_f ** 2)
    liq_mask = ~(wall_mask | (phi < 0))
    u_max_final = float(u_mag_f[liq_mask].max().item()) if liq_mask.any() else 0.0
    Fo_final = Fo_factor * steps
    if not quiet:
        print(f"\n{'─' * 72}")
        print(f"  Final f_l={f_l_final:.4f}  s_top={s_top_f:.2f} s_mid={s_mid_f:.2f} s_bot={s_bot_f:.2f}")
        print(f"  Deformation={s_top_f - s_bot_f:.2f}  u_max={u_max_final:.6f}  Fo={Fo_final:.4f}")
        print(f"  Pr={Pr:.4f} Ra={Ra:.2f} Ste={Ste:.4f}")
        print(f"{'─' * 72}")
    return {"step": steps, "f_liq": f_l_final, "s_top": s_top_f, "s_mid": s_mid_f,
            "s_bot": s_bot_f, "deformation": s_top_f - s_bot_f, "u_max": u_max_final,
            "Fo": Fo_final, "T_field": T_final.detach().cpu().numpy(),
            "phi_field": phi.detach().cpu().numpy(),
            "ux_field": ux_f.detach().cpu().numpy(), "uy_field": uy_f.detach().cpu().numpy(),
            "history": history, "nu": nu, "alpha": alpha, "Pr": Pr, "Ra": Ra, "Ste": Ste,
            "nx": nx, "ny": ny, "nz": nz, "Fo_factor": Fo_factor}


def main():
    p = argparse.ArgumentParser(description="Gallium melting — PF + thermal LBM")
    p.add_argument("--nx", type=int, default=40)
    p.add_argument("--ny", type=int, default=56)
    p.add_argument("--nz", type=int, default=1)
    p.add_argument("--tau", type=float, default=0.506)
    p.add_argument("--tau-T", type=float, default=0.8)
    p.add_argument("--T-hot", type=float, default=1.0)
    p.add_argument("--T-cold", type=float, default=0.0)
    p.add_argument("--T-melt", type=float, default=0.148)
    p.add_argument("--L-latent", type=float, default=18.52)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--gy", type=float, default=-0.001875)
    p.add_argument("--u-clamp", type=float, default=0.15)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-every", type=int, default=1000)
    args = p.parse_args()
    r = run_gallium_pf(nx=args.nx, ny=args.ny, nz=args.nz, tau=args.tau, tau_T=args.tau_T,
                       T_hot=args.T_hot, T_cold=args.T_cold, T_melt=args.T_melt,
                       L_latent=args.L_latent, beta=args.beta, gy=args.gy,
                       u_clamp=args.u_clamp, steps=args.steps, device=args.device,
                       log_every=args.log_every)
    ok = True
    s_mean = (r["s_top"] + r["s_mid"] + r["s_bot"]) / 3.0
    if s_mean > 2.0:
        print(f"\n  ✓ PASS  melt front advanced  (s_mean={s_mean:.2f})")
    else:
        print(f"\n  ✗ FAIL  melt front did not advance  (s_mean={s_mean:.2f})")
        ok = False
    if r["deformation"] > 0.3:
        print(f"  ✓ PASS  interface deformed  (top−bottom={r['deformation']:.2f})")
    else:
        print(f"  ✗ FAIL  interface not deformed  (top−bottom={r['deformation']:.2f})")
        ok = False
    if r["f_liq"] > 0.01:
        print(f"  ✓ PASS  liquid fraction grew  (f_l={r['f_liq']:.4f})")
    else:
        print(f"  ✗ FAIL  liquid fraction too small  (f_l={r['f_liq']:.4f})")
        ok = False
    if r["u_max"] > 1e-4:
        print(f"  ✓ PASS  convection present  (u_max={r['u_max']:.6f})")
    else:
        print(f"  ✗ FAIL  no convection  (u_max={r['u_max']:.6f})")
        ok = False
    # Quantitative vs Gau-Viskanta
    hist = r.get("history", [])
    if len(hist) > 1:
        lbm_fo = np.array([h["Fo"] for h in hist])
        lbm_fl = np.array([h["f_liq"] for h in hist])
        mask = (_GV_FO >= lbm_fo.min()) & (_GV_FO <= lbm_fo.max())
        if int(mask.sum()) >= 2:
            lbm_at = np.interp(_GV_FO[mask], lbm_fo, lbm_fl)
            mape = float(np.mean(np.abs(lbm_at - _GV_FLIQ[mask]) / _GV_FLIQ[mask]) * 100)
            print(f"\n  Gau-Viskanta MAPE = {mape:.1f}% ({int(mask.sum())} pts, Fo≤{lbm_fo.max():.3f})")
            if mape < 20.0:
                print(f"  ✓ PASS  quantitative match  (MAPE < 20%)")
            else:
                print(f"  ✗ FAIL  quantitative mismatch  (MAPE ≥ 20%)")
                ok = False
    print(f"\n  Pr={r['Pr']:.4f} Ra={r['Ra']:.2f} Ste={r['Ste']:.4f} Fo={r['Fo']:.4f}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
