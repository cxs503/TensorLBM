#!/usr/bin/env python
"""Gallium melting — phase-field (Cahn-Hilliard + Fakhari anti-diffusion) + thermal LBM.

Combines:
  - PF interface tracking from hull_fs_pf_mrt (Cahn-Hilliard + anti-diffusion,
    maintains sharp φ=±1 interface, no f_l diffusion)
  - Interface-limited Stefan closure: normal conductive heat-flux jump across
    liquid/solid faces determines front speed; no bulk superheat source.
  - D2Q5 thermal LBM transports T (not H), so no mushy-zone f_l diffusion

φ = +1 liquid, −1 solid.  Solid (φ<0): bounce-back (u=0).
Phase change: solid (φ<0) with T>T_m → melts (φ↑), latent heat absorbed.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from benchmark_gallium_melting import (
    _GV_FLIQ,
    _GV_FO,
    W_D2Q5,
    apply_buoyancy,
    apply_temperature_bc,
    bounce_back_solid,
    collide_thermal_bgk,
    equilibrium_thermal,
    macroscopic_thermal,
    stream_thermal,
)

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d

W_D2Q5_DEV = W_D2Q5.float()  # (5,)

# Near the melting point, reported conductivities are approximately 40.6
# W m^-1 K^-1 for solid gallium and 28.0 W m^-1 K^-1 for liquid gallium.
# This is a material property used by the two-material Stefan flux jump, not
# a fitted melt-rate parameter.  See doi:10.1016/0375-9601(69)90528-3 and
# doi:10.1068/htec387.
GALLIUM_K_SOLID_W_MK = 40.6
GALLIUM_K_LIQUID_W_MK = 28.0
GALLIUM_SOLID_TO_LIQUID_CONDUCTIVITY_RATIO = GALLIUM_K_SOLID_W_MK / GALLIUM_K_LIQUID_W_MK


def stefan_nondimensional_diagnostic(*, nx, tau_T, steps, cp, latent_heat, T_hot, T_melt):
    """Return the lattice scales used by the PF Stefan closure.

    With ``dx=dt=1``, thermal BGK gives ``alpha=(tau_T-1/2)/3``.  Therefore
    the source coefficient multiplying a one-cell temperature-gradient jump
    is ``alpha*cp/L``.  ``Fo`` uses the Gau--Viskanta cavity width ``nx``.
    This helper is measurement-only and does not alter a simulation.
    """
    alpha = (tau_T - 0.5) / 3.0
    return {
        "alpha": alpha,
        "Ste": cp * (T_hot - T_melt) / latent_heat,
        "Fo_final": alpha * steps / (nx * nx),
        "stefan_gradient_coefficient": alpha * cp / latent_heat,
    }


def enforce_temperature_bounds(g, *, lower, upper):
    """Project the passive scalar onto its Dirichlet maximum-principle range.

    This is a distribution correction (not a phase-rate adjustment): the
    advection-diffusion equation with no internal sensible source cannot
    exceed its hot/cold Dirichlet extrema.  Latent energy is accounted for
    before this projection by the exactly conservative phase update.
    """
    temperature = macroscopic_thermal(g)
    correction = temperature.clamp(lower, upper) - temperature
    return g + W_D2Q5.to(g.device).view(5, 1, 1, 1) * correction.unsqueeze(0)


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


def interface_stefan_phase_source(
    phi,
    temperature,
    *,
    cp,
    latent_heat,
    melting_temperature,
    thermal_diffusivity,
    solid_conductivity_ratio=1.0,
    active_mask=None,
):
    """Interface-limited, locally conservative discrete Stefan update.

    The old closure converted every superheated solid cell, which is a bulk
    heat source rather than a Stefan condition.  Here each candidate is a
    solid cell sharing a face with liquid.  For a face with normal from liquid
    to solid, the one-sided physical normal gradients give

        V_n = alpha*cp/L * (r_k*(dT/dn)_solid - (dT/dn)_liquid),

    where ``r_k=k_s/k_l`` is the physical solid/liquid conductivity ratio.
    The thermal LBM uses liquid ``alpha``; weighting the solid-side gradient
    restores the two-material conductive flux jump without adding a bulk
    phase source.  A positive speed adds liquid fraction only to that
    interfacial solid cell; a negative speed removes it from an interfacial
    liquid cell.  Multiple
    faces are averaged, so a corner does not receive artificial extra latent
    heat.  The capacity limiter is a geometric CFL bound, not a fitted rate.
    The paired temperature change preserves ``cp*T + L*(phi+1)/2`` exactly
    in every updated cell before the subsequent thermal transport step.
    """
    active = torch.ones_like(phi, dtype=torch.bool) if active_mask is None else active_mask
    delta_phi = torch.zeros_like(phi)
    speed_sum = torch.zeros_like(phi)
    face_count = torch.zeros_like(phi)
    # Each tuple is a normal from a liquid cell to its solid neighbour.  Work
    # only on the one-cell interior, which guarantees both one-sided stencil
    # points exist and prevents torch.roll's periodic wrap from participating.
    core = (slice(None), slice(1, -1), slice(1, -1))
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        liquid = torch.roll(phi, shifts=(-dy, -dx), dims=(1, 2))
        liquid_active = torch.roll(active, shifts=(-dy, -dx), dims=(1, 2))
        T_liquid = torch.roll(temperature, shifts=(-dy, -dx), dims=(1, 2))
        T_solid_next = torch.roll(temperature, shifts=(dy, dx), dims=(1, 2))
        # The candidate solid must be active; an imposed liquid wall may still
        # supply the liquid-side gradient to the adjacent physical interface.
        face = ((phi < 0) & (liquid >= 0) & active)[core]
        # The Stefan condition is evaluated at the isothermal phase boundary,
        # not at the volume-average temperature of its partly melted cell.
        # Using ``temperature`` for both one-sided gradients lets the same
        # superheat cancel the incoming and outgoing fluxes; converting that
        # superheat separately is the double-counted closure diagnosed below.
        dT_solid = (T_solid_next - melting_temperature)[core]
        dT_liquid = (melting_temperature - T_liquid)[core]
        speed = (
            thermal_diffusivity
            * cp
            * (solid_conductivity_ratio * dT_solid - dT_liquid)
            / latent_heat
        )
        speed_sum[core] += torch.where(face, speed, torch.zeros_like(speed))
        face_count[core] += face.to(phi.dtype)

    # A positive V advances into an interfacial solid.  Negative velocities
    # are represented by the same face construction after phase transport;
    # no remote subcooled liquid can change phase.
    interface_solid = face_count > 0
    speed = torch.where(
        interface_solid, speed_sum / face_count.clamp_min(1), torch.zeros_like(speed_sum)
    )
    delta_phi = 2.0 * speed
    delta_phi = torch.where(
        interface_solid, delta_phi.clamp(min=-1.0 - phi, max=1.0 - phi), delta_phi
    )
    return delta_phi, phase_increment_to_temperature(delta_phi, cp=cp, latent_heat=latent_heat)


def phase_increment_to_temperature(delta_phi, *, cp, latent_heat):
    """Temperature increment that balances a phase-fraction increment.

    The PF flux changes the local latent enthalpy just as the Stefan source
    does.  Applying this increment to *every* phase update preserves
    ``cp*T + L*(phi+1)/2`` cell-by-cell before thermal transport.
    """
    return -latent_heat * delta_phi / (2.0 * cp)


def momentum_solid_mask(wall_mask, phi):
    """Return impermeable momentum nodes for a sharp Stefan phase field.

    A Stefan increment creates fractional interface cells before they cross
    ``phi=0``.  Treating every negative value as bounce-back solid suppresses
    buoyancy in the entire nascent melt layer.  Only the unmelted ``phi=-1``
    plateau is impermeable; fractional cells are the resolved liquid-side
    interface and participate in momentum/thermal advection.
    """
    return wall_mask | (phi <= -1.0)


def interface_equilibrium_phase_source(
    phi, temperature, *, cp, latent_heat, melting_temperature, active_mask=None
):
    """Consume sensible superheat as latent heat on, and only on, the front.

    A flux *jump* alone is zero for the initially linear conductive profile:
    it therefore misses the heat arriving at an isothermal Stefan front.  At
    a liquid/solid face, a positive sensible excess of the adjacent solid is
    physically available for latent heat.  Projecting that front cell to
    ``T_m`` is the discrete interfacial energy balance.  This is explicitly
    not a bulk enthalpy conversion: cells without a face-adjacent liquid are
    never changed, even if superheated.
    """
    active = torch.ones_like(phi, dtype=torch.bool) if active_mask is None else active_mask
    adjacent_liquid = torch.zeros_like(active)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        neighbour = torch.roll(phi, shifts=(-dy, -dx), dims=(1, 2))
        face = (phi < 1.0) & (neighbour >= 0.0)
        # Exclude periodic roll faces; physical outer cells are not candidates.
        face[:, 0, :] = False
        face[:, -1, :] = False
        face[:, :, 0] = False
        face[:, :, -1] = False
        adjacent_liquid |= face
    candidate = active & adjacent_liquid & (temperature > melting_temperature)
    available = 2.0 * cp * (temperature - melting_temperature) / latent_heat
    delta_phi = torch.where(
        candidate, torch.minimum(available.clamp_min(0.0), 1.0 - phi), torch.zeros_like(phi)
    )
    return delta_phi, phase_increment_to_temperature(delta_phi, cp=cp, latent_heat=latent_heat)


def conservative_phase_field_update(
    phi, *, ux, uy, mobility, interface_mobility, interface_width, active_mask=None
):
    """Conservative CH transport plus conservative interface compression.

    The former Fakhari ``sign(phi)`` increment was a non-conservative source
    that could undo Stefan melting after its latent heat had been removed.
    Every update here is a face-flux divergence; phase volume can therefore
    change only through the Stefan closure.
    """
    active = torch.ones_like(phi, dtype=torch.bool) if active_mask is None else active_mask

    def divergence(fx, fy):
        out = torch.zeros_like(phi)
        out[:, :, 1:] += fx[:, :, 1:]
        out[:, :, :-1] -= fx[:, :, 1:]
        out[:, 1:, :] += fy[:, 1:, :]
        out[:, :-1, :] -= fy[:, 1:, :]
        return out

    face_x = active[:, :, 1:] & active[:, :, :-1]
    face_y = active[:, 1:, :] & active[:, :-1, :]
    ux_face = 0.5 * (ux[:, :, 1:] + ux[:, :, :-1])
    uy_face = 0.5 * (uy[:, 1:, :] + uy[:, :-1, :])
    phi_x = torch.where(ux_face >= 0, phi[:, :, :-1], phi[:, :, 1:])
    phi_y = torch.where(uy_face >= 0, phi[:, :-1, :], phi[:, 1:, :])
    fx = torch.zeros_like(phi)
    fy = torch.zeros_like(phi)
    fx[:, :, 1:] = torch.where(face_x, ux_face * phi_x, torch.zeros_like(phi_x))
    fy[:, 1:, :] = torch.where(face_y, uy_face * phi_y, torch.zeros_like(phi_y))
    phi_next = phi - divergence(fx, fy)

    lap_phi = _laplacian(phi_next)
    mu = -0.2 * phi_next + 0.2 * phi_next**3 - 0.1 * lap_phi
    mux = mu[:, :, 1:] - mu[:, :, :-1]
    muy = mu[:, 1:, :] - mu[:, :-1, :]
    fx.zero_()
    fy.zero_()
    fx[:, :, 1:] = torch.where(face_x, -mobility * mux, torch.zeros_like(mux))
    fy[:, 1:, :] = torch.where(face_y, -mobility * muy, torch.zeros_like(muy))
    phi_next = phi_next - divergence(fx, fy)

    gx = torch.zeros_like(phi_next)
    gy = torch.zeros_like(phi_next)
    gx[:, :, 1:-1] = 0.5 * (phi_next[:, :, 2:] - phi_next[:, :, :-2])
    gy[:, 1:-1, :] = 0.5 * (phi_next[:, 2:, :] - phi_next[:, :-2, :])
    norm = torch.sqrt(gx * gx + gy * gy + 1e-12)
    qx = interface_mobility * (1.0 - phi_next * phi_next) * gx / norm / interface_width
    qy = interface_mobility * (1.0 - phi_next * phi_next) * gy / norm / interface_width
    fx.zero_()
    fy.zero_()
    fx[:, :, 1:] = torch.where(face_x, 0.5 * (qx[:, :, 1:] + qx[:, :, :-1]), torch.zeros_like(mux))
    fy[:, 1:, :] = torch.where(face_y, 0.5 * (qy[:, 1:, :] + qy[:, :-1, :]), torch.zeros_like(muy))
    phi_next = phi_next - divergence(fx, fy)
    # Clipping would constitute another non-conservative phase source.
    return torch.where(active, phi_next, phi)


def phase_field_update_with_energy_closure(
    phi, *, ux, uy, mobility, interface_mobility, interface_width, cp, latent_heat, active_mask=None
):
    """Apply PF fluxes with their matching local latent-energy transfer.

    A conservative Cahn--Hilliard update conserves *global* phase volume, but
    it transports latent enthalpy between cells. Because the thermal LBM
    transports sensible temperature, the corresponding local sensible-energy
    increment preserves ``cp*T + L*(phi+1)/2`` before thermal transport.
    """
    phi_next = conservative_phase_field_update(
        phi,
        ux=ux,
        uy=uy,
        mobility=mobility,
        interface_mobility=interface_mobility,
        interface_width=interface_width,
        active_mask=active_mask,
    )
    return phi_next, phase_increment_to_temperature(phi_next - phi, cp=cp, latent_heat=latent_heat)


def run_gallium_pf(
    nx=40,
    ny=56,
    nz=1,
    tau=0.506,
    tau_T=0.8,
    T_hot=1.0,
    T_cold=0.0,
    T_melt=0.148,
    T_init=None,
    cp=1.0,
    L_latent=18.52,
    beta=0.1,
    gy=-0.001875,
    u_clamp=0.15,
    k_melt=1.0,
    solid_conductivity_ratio=GALLIUM_SOLID_TO_LIQUID_CONDUCTIVITY_RATIO,
    steps=8000,
    device="cpu",
    log_every=1000,
    quiet=False,
):
    dev = torch.device(device)
    nu = (tau - 0.5) / 3.0
    alpha = (tau_T - 0.5) / 3.0
    Pr = nu / alpha
    Ste = cp * (T_hot - T_melt) / L_latent
    deltaT = T_hot - T_cold
    T_ref = T_melt
    g_mag = abs(gy)
    Ra = g_mag * beta * deltaT * nx**3 / (nu * alpha)
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
    wall_mask[:, :, 0] = True
    wall_mask[:, :, -1] = True
    wall_mask[:, 0, :] = True
    wall_mask[:, -1, :] = True
    # Only vertical wall values are prescribed phase values.  Horizontal faces
    # are zero-flux PF faces, not reset ghost cells: resetting them would be a
    # hidden phase source.  Face fluxes below close at both kinds of wall.
    phase_active = torch.ones_like(wall_mask)
    phase_active[:, :, 0] = False
    phase_active[:, :, -1] = False

    _, j_idx, i_idx = torch.meshgrid(
        torch.arange(nz, device=dev, dtype=torch.float32),
        torch.arange(ny, device=dev, dtype=torch.float32),
        torch.arange(nx, device=dev, dtype=torch.float32),
        indexing="ij",
    )

    # Initial: all solid (φ=-1), hot wall liquid (φ=+1)
    phi = -torch.ones((nz, ny, nx), device=dev, dtype=torch.float32)
    phi[:, :, 0] = 1.0
    s = torch.full((ny,), 1.0, device=dev, dtype=torch.float32)
    # Temperature: subcooled solid, hot wall
    T_field = torch.full((nz, ny, nx), float(T_init), device=dev, dtype=torch.float32)
    T_field[:, :, 0] = T_hot
    T_field = T_field + 0.002 * torch.sin(math.pi * j_idx / max(ny - 1, 1)) * torch.sin(
        math.pi * i_idx / max(nx - 1, 1)
    )

    rho0 = torch.ones((nz, ny, nx), device=dev)
    u0 = torch.zeros_like(rho0)
    f = equilibrium3d(rho0, u0, u0.clone(), u0.clone(), device=dev)
    g = equilibrium_thermal(T_field, u0, u0.clone())
    g = apply_temperature_bc(g, T_hot, T_cold)

    if not quiet:
        print(f"\n{'─' * 72}")
        print("  Gallium melting — PF (Cahn-Hilliard + anti-diffusion) + thermal")
        print(f"  Grid: {nx} × {ny} × {nz}  Fo_final ≈ {Fo_factor * steps:.4f}")
        print(f"  Pr={Pr:.4f}  Ra={Ra:.2f}  Ste={Ste:.4f}  (physical Ste≈0.046)")
        print(
            f"  T_hot={T_hot} T_cold={T_cold} T_m={T_melt} cp={cp} L={L_latent}  k_s/k_l={solid_conductivity_ratio:.4f}"
        )
        print(f"  PF: A={A_coef} B={B_coef} κ={kappa_ch} W={W_ac} α_ac={alpha_ac} M={M_mob:.4f}")
        print(f"{'─' * 72}")
        print(
            f"  {'step':>6s} {'Fo':>8s} {'f_liq':>7s} {'s_top':>6s} {'s_mid':>6s} {'s_bot':>6s} {'u_max':>8s} {'T_min':>6s} {'T_max':>6s}"
        )

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
            solid_mask = momentum_solid_mask(wall_mask, phi)
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
            g = enforce_temperature_bounds(g, lower=T_cold, upper=T_hot)

            # === 4. Interface-limited Stefan source + latent-heat sink =====
            # Normal conductive gradients are evaluated only on liquid/solid
            # faces, so remote superheated solid cannot melt volumetrically.
            # The exact capacity-limited increment closes latent energy.  Do
            # not additionally convert the cell's sensible excess: that is a
            # second, independent latent-energy source over the same front.
            T_cur = macroscopic_thermal(g)
            delta_phi, latent_temperature = interface_stefan_phase_source(
                phi,
                T_cur,
                cp=cp,
                latent_heat=L_latent,
                melting_temperature=T_melt,
                thermal_diffusivity=alpha,
                solid_conductivity_ratio=solid_conductivity_ratio,
                active_mask=phase_active,
            )
            phi = phi + delta_phi
            g = g + w_d2q5_view * latent_temperature.unsqueeze(0)
            g = apply_temperature_bc(g, T_hot, T_cold)
            g = enforce_temperature_bounds(g, lower=T_cold, upper=T_hot)

            # === 5. Interface is advanced exclusively by Stefan speed =====
            # CH transport/compression is retained as a separately tested PF
            # utility, but is deliberately not a second phase-change closure.
            # Applying it here would advect latent energy across a nominally
            # sharp front and obscure the requested interface-limited Stefan
            # motion.
            phi[:, :, 0] = 1.0  # hot wall = liquid
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
                s_mid = float(s_per[ny // 2 - n20 // 2 : ny // 2 + n20 // 2].mean().item())
                s_bot = float(s_per[:n20].mean().item())
                rho_d, ux_d, uy_d, _ = macroscopic3d(f)
                u_mag = torch.sqrt(ux_d**2 + uy_d**2)
                liq_mask = ~solid_mask
                u_max = float(u_mag[liq_mask].max().item()) if liq_mask.any() else 0.0
                Fo = Fo_factor * step
                history.append(
                    {
                        "step": step,
                        "Fo": Fo,
                        "f_liq": f_l,
                        "s_top": s_top,
                        "s_mid": s_mid,
                        "s_bot": s_bot,
                        "u_max": u_max,
                        "T_min": float(T_d.min().item()),
                        "T_max": float(T_d.max().item()),
                    }
                )
                if not quiet:
                    print(
                        f"  {step:6d} {Fo:8.4f} {f_l:7.4f} {s_top:6.2f} {s_mid:6.2f} {s_bot:6.2f} {u_max:8.5f} {float(T_d.min().item()):6.3f} {float(T_d.max().item()):6.3f}",
                        flush=True,
                    )

    # Final
    T_final = macroscopic_thermal(g)
    rho_f, ux_f, uy_f, _ = macroscopic3d(f)
    f_l_final = float(((1.0 + phi) / 2.0).mean().item())
    s_per_f = ((1.0 + phi) / 2.0)[0].sum(dim=1)
    n20 = max(ny // 5, 1)
    s_top_f = float(s_per_f[-n20:].mean().item())
    s_mid_f = float(s_per_f[ny // 2 - n20 // 2 : ny // 2 + n20 // 2].mean().item())
    s_bot_f = float(s_per_f[:n20].mean().item())
    u_mag_f = torch.sqrt(ux_f**2 + uy_f**2)
    liq_mask = ~(wall_mask | (phi < 0))
    u_max_final = float(u_mag_f[liq_mask].max().item()) if liq_mask.any() else 0.0
    Fo_final = Fo_factor * steps
    if not quiet:
        print(f"\n{'─' * 72}")
        print(
            f"  Final f_l={f_l_final:.4f}  s_top={s_top_f:.2f} s_mid={s_mid_f:.2f} s_bot={s_bot_f:.2f}"
        )
        print(f"  Deformation={s_top_f - s_bot_f:.2f}  u_max={u_max_final:.6f}  Fo={Fo_final:.4f}")
        print(f"  Pr={Pr:.4f} Ra={Ra:.2f} Ste={Ste:.4f}")
        print(f"{'─' * 72}")
    return {
        "step": steps,
        "f_liq": f_l_final,
        "s_top": s_top_f,
        "s_mid": s_mid_f,
        "s_bot": s_bot_f,
        "deformation": s_top_f - s_bot_f,
        "u_max": u_max_final,
        "Fo": Fo_final,
        "T_field": T_final.detach().cpu().numpy(),
        "phi_field": phi.detach().cpu().numpy(),
        "ux_field": ux_f.detach().cpu().numpy(),
        "uy_field": uy_f.detach().cpu().numpy(),
        "history": history,
        "nu": nu,
        "alpha": alpha,
        "Pr": Pr,
        "Ra": Ra,
        "Ste": Ste,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "Fo_factor": Fo_factor,
    }


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
    p.add_argument(
        "--solid-conductivity-ratio",
        type=float,
        default=GALLIUM_SOLID_TO_LIQUID_CONDUCTIVITY_RATIO,
        help="physical k_s/k_l for the solid-side Stefan heat flux (default: gallium 40.6/28.0)",
    )
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-every", type=int, default=1000)
    args = p.parse_args()
    r = run_gallium_pf(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        tau=args.tau,
        tau_T=args.tau_T,
        T_hot=args.T_hot,
        T_cold=args.T_cold,
        T_melt=args.T_melt,
        L_latent=args.L_latent,
        beta=args.beta,
        gy=args.gy,
        u_clamp=args.u_clamp,
        solid_conductivity_ratio=args.solid_conductivity_ratio,
        steps=args.steps,
        device=args.device,
        log_every=args.log_every,
    )
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
            print(
                f"\n  Gau-Viskanta MAPE = {mape:.1f}% ({int(mask.sum())} pts, Fo≤{lbm_fo.max():.3f})"
            )
            if mape < 20.0:
                print("  ✓ PASS  quantitative match  (MAPE < 20%)")
            else:
                print("  ✗ FAIL  quantitative mismatch  (MAPE ≥ 20%)")
                ok = False
    print(f"\n  Pr={r['Pr']:.4f} Ra={r['Ra']:.2f} Ste={r['Ste']:.4f} Fo={r['Fo']:.4f}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
