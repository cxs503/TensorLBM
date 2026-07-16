"""Free-surface LBM (D3Q27) — full Körner model with mass tracking.

Implements the complete Körner et al. (2005) free-surface LBM for the D3Q27
lattice, mirroring the D3Q19 implementation in :mod:`free_surface_lbm` but
using all 27 velocity directions (6 face + 12 edge + 8 corner).

The D3Q27 stencil achieves 4th-order isotropy, which can reduce numerical
artefacts in flows with strong corner-region gradients.

Cell types: GAS / INTERFACE / LIQUID / SOLID
  - Mass tracking: independent mass variable (mass ≠ rho)
  - Mass redistribution: excess mass distributed to interface neighbors
  - Interface gas pressure: anti-bounce-back at interface cells
  - Neighbor flags: prevents isolated cells

References: Körner et al. (2005), waLBerla free_surface/
"""
from __future__ import annotations

import torch

from .d3q27 import C as C27
from .d3q27 import W as W27
from .d3q27 import OPPOSITE as OPP27
from .d3q27 import (
    _get_d3q27_mrt_matrices,
    collide_bgk27,
    collide_mrt27,
    equilibrium27,
    macroscopic27,
    stream27,
)
from .boundaries_d3q27 import bounce_back_cells_27
from .core.d3q27_stencil import (
    D3Q27_MOVING_Q,
    all_moving_neighbor_masks_27,
    assert_no_direct_phase_links_27,
    moving_tensor_shifts_27,
    roll_from_pull_source_27,
    roll_to_neighbor_27,
)
from .turbulence import (
    _neq_stress_norm_27,
    _smagorinsky_tau,
    _wale_nu_t_3d,
    _vreman_nu_t_3d,
    _nu_t_to_tau_eff,
)

GAS = 0
LIQUID = 1
INTERFACE = 2
SOLID = 3

# D3Q27 velocity vectors and weights
_C = C27  # (27, 3)
_W = W27
KAPPA = 0.41
B_CONST = 5.0

# Opposite direction indices for D3Q27 (built from the stencil)
_OPP = OPP27

# Precompute (cx, cy, cz) shifts for all 27 directions
_C27_SHIFTS = [(int(C27[q, 0]), int(C27[q, 1]), int(C27[q, 2])) for q in range(27)]


def _stream27_roll(f: torch.Tensor) -> torch.Tensor:
    """D3Q27 streaming via torch.roll (pull scheme).

    ``out[q](x) = f[q](x - c_q)`` — the standard pull scheme.
    """
    out = torch.empty_like(f)
    for q in range(27):
        sx, sy, sz = _C27_SHIFTS[q]
        out[q] = torch.roll(f[q], shifts=(sz, sy, sx), dims=(0, 1, 2))
    return out


def _assert_no_direct_liquid_gas_links_27(flags: torch.Tensor) -> None:
    """Reject states where a D3Q27 liquid link reaches GAS without INTERFACE."""
    try:
        assert_no_direct_phase_links_27(flags, LIQUID, GAS, "direct LIQUID-GAS D3Q27")
    except ValueError as error:
        if "direct phase link" not in str(error):
            raise
        raise ValueError(
            f"{error}; insert INTERFACE cells between LIQUID and GAS"
        ) from None


# ===========================================================================
# Initialization
# ===========================================================================

def init_fill_rectangular_27(
    nz: int, ny: int, nx: int,
    column_width: int, column_height: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize fill field for rectangular liquid column (dam-break IC)."""
    fill = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    solid[:, :, 0] = True
    solid[:, :, -1] = True
    cw, ch = int(column_width), int(column_height)
    fill[:, :ch, 1:cw] = 1.0
    fx, fy = column_width - cw, column_height - ch
    if fx > 0 and cw < nx - 1:
        fill[:, :ch, cw] = fx
    if fy > 0 and ch < ny - 1:
        fill[:, ch, 1:cw] = fy
    if fx > 0 and fy > 0 and cw < nx - 1 and ch < ny - 1:
        fill[:, ch, cw] = 0.5 * (fx + fy)
    return fill, solid


def init_flags_from_fill_27(
    fill: torch.Tensor, solid_mask: torch.Tensor,
) -> torch.Tensor:
    """Set flags from fill and create the required D3Q27 interface envelope.

    A zero-fill GAS cell directly linked to LIQUID has no Körner boundary
    reconstruction.  Such cells begin as empty INTERFACE cells so every
    D3Q27 liquid/gas link is represented by the interface model.
    """
    flags = torch.full_like(fill, GAS, dtype=torch.int8)
    flags[fill >= 1.0] = LIQUID
    flags[(fill > 0) & (fill < 1)] = INTERFACE
    liquid = flags == LIQUID
    liquid_neighbor = torch.stack(all_moving_neighbor_masks_27(liquid)).any(dim=0)
    flags[(flags == GAS) & liquid_neighbor & ~solid_mask] = INTERFACE
    flags[solid_mask] = SOLID
    return flags


def init_mass_from_fill_27(
    fill: torch.Tensor, flags: torch.Tensor, rho_liquid: float = 1.0,
) -> torch.Tensor:
    """Initialize mass field: mass = fill * rho_liquid for liquid/interface."""
    mass = torch.zeros_like(fill)
    mass[flags == LIQUID] = rho_liquid
    mass[flags == INTERFACE] = fill[flags == INTERFACE] * rho_liquid
    return mass


def total_liquid_inventory_27(
    f: torch.Tensor, fill: torch.Tensor, flags: torch.Tensor,
    rho_liquid: float = 1.0,
) -> torch.Tensor:
    """Return liquid inventory (population density for LIQUID + fill mass for INTERFACE)."""
    rho = f.sum(dim=0)
    return (
        torch.where(flags == LIQUID, rho, torch.zeros_like(rho)).sum()
        + torch.where(flags == INTERFACE, fill * rho_liquid, torch.zeros_like(fill)).sum()
    )


# ===========================================================================
# Helper: init new cells with neighbor-averaged velocity
# ===========================================================================

def _init_new_27(
    f: torch.Tensor, flags: torch.Tensor, mask: torch.Tensor,
    rho_init: float, device: torch.device,
    ux: torch.Tensor | None = None,
    uy: torch.Tensor | None = None,
    uz: torch.Tensor | None = None,
) -> torch.Tensor:
    """Init newly converted cells with neighbor-averaged velocity (vectorized)."""
    if ux is None or uy is None or uz is None:
        _, ux, uy, uz = macroscopic27(f)
    active = (flags == LIQUID) | (flags == INTERFACE)
    sl_stack = torch.stack(all_moving_neighbor_masks_27(active))
    ux_stack = torch.stack([roll_from_pull_source_27(ux, q) for q in D3Q27_MOVING_Q])
    uy_stack = torch.stack([roll_from_pull_source_27(uy, q) for q in D3Q27_MOVING_Q])
    uz_stack = torch.stack([roll_from_pull_source_27(uz, q) for q in D3Q27_MOVING_Q])
    sl_f = sl_stack.float()
    uxa = (ux_stack * sl_f).sum(dim=0)
    uya = (uy_stack * sl_f).sum(dim=0)
    uza = (uz_stack * sl_f).sum(dim=0)
    cnt = sl_f.sum(dim=0).clamp(min=1)
    rho_f = torch.full_like(ux, float(rho_init))
    feq = equilibrium27(rho_f, uxa / cnt, uya / cnt, uza / cnt)
    return torch.where(mask.unsqueeze(0), feq, f)


# ===========================================================================
# Interface normal computation (for mass redistribution)
# ===========================================================================

def _compute_interface_normal_27(
    flags: torch.Tensor, mass: torch.Tensor, rho: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute interface normal n = -∇fill / |∇fill| (points from liquid to gas)."""
    fill = mass / rho.clamp(min=1e-6)
    grad_x = 0.5 * (fill.roll(-1, dims=2) - fill.roll(1, dims=2))
    grad_y = 0.5 * (fill.roll(-1, dims=1) - fill.roll(1, dims=1))
    grad_z = 0.5 * (fill.roll(-1, dims=0) - fill.roll(1, dims=0))
    mag = (grad_x ** 2 + grad_y ** 2 + grad_z ** 2).sqrt().clamp(min=1e-10)
    return -grad_x / mag, -grad_y / mag, -grad_z / mag


# ===========================================================================
# MRT collision with per-cell effective relaxation time (SGS support)
# ===========================================================================

def _collide_mrt27_with_tau_eff(
    f: torch.Tensor,
    feq: torch.Tensor,
    tau_eff: torch.Tensor,
    device: torch.device,
    s_e: float = 1.19,
    s_eps: float = 1.4,
    s_q: float = 1.2,
    s_pi: float | None = None,
) -> torch.Tensor:
    """D3Q27 MRT collision with per-cell effective relaxation time.

    Identical to :func:`~tensorlbm.d3q27.collide_mrt27` except that the
    stress-mode relaxation rates (rows 5–9) use the per-cell ``1/τ_eff(x)``
    instead of a scalar ``1/τ``.  This is the standard pattern for
    coupling MRT with any algebraic SGS model (Smagorinsky, WALE, Vreman).

    Args:
        f: Post-collision-ready distribution, shape ``(27, nz, ny, nx)``.
        feq: Equilibrium distribution, same shape.
        tau_eff: Per-cell effective relaxation time, shape ``(nz, ny, nx)``.
        device: Torch device.
        s_e, s_eps, s_q, s_pi: Fixed MRT relaxation rates for non-stress modes.

    Returns:
        Post-MRT-collision distribution of the same shape as *f*.
    """
    if s_pi is None:
        s_pi = s_e
    M, M_inv = _get_d3q27_mrt_matrices(device)
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    f_flat = f.reshape(27, -1)
    feq_flat = feq.reshape(27, -1)
    s_nu_flat = (1.0 / tau_eff).reshape(-1)  # (N,)

    m = M @ f_flat
    m_eq = M @ feq_flat
    dm = m - m_eq

    s_fixed = torch.tensor(
        [
            0.0,   # 0  mass
            0.0,   # 1  jx
            0.0,   # 2  jy
            0.0,   # 3  jz
            s_e,   # 4  energy
            0.0,   # 5  Nxx  – overridden below
            0.0,   # 6  Nyy  – overridden below
            0.0,   # 7  Pxy  – overridden below
            0.0,   # 8  Pxz  – overridden below
            0.0,   # 9  Pyz  – overridden below
            s_q,   # 10
            s_q,   # 11
            s_q,   # 12
            s_q,   # 13
            s_q,   # 14
            s_q,   # 15
            s_q,   # 16
            s_q,   # 17
            s_q,   # 18
            s_eps, # 19
            s_pi,  # 20
            s_pi,  # 21
            s_pi,  # 22
            s_pi,  # 23
            s_pi,  # 24
            s_pi,  # 25
            s_pi,  # 26
        ],
        dtype=f.dtype,
        device=device,
    )
    m_star = m - s_fixed.unsqueeze(1) * dm
    for k in (5, 6, 7, 8, 9):
        m_star[k] = m[k] - s_nu_flat * dm[k]
    return (M_inv @ m_star).reshape(27, nz, ny, nx)


def _compute_tau_eff_sgs(
    tau: float,
    sgs_model: str,
    f_neq: torch.Tensor,
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    C_s: float,
) -> torch.Tensor:
    """Compute per-cell effective relaxation time for a given SGS model.

    Shared by D3Q19 and D3Q27 free-surface steps.  Uses the common
    turbulence module functions so that both lattices share identical
    SGS physics.
    """
    if sgs_model == 'smagorinsky':
        return _smagorinsky_tau(tau, _neq_stress_norm_27(f_neq), rho, C_s)
    elif sgs_model == 'wale':
        nu_t = _wale_nu_t_3d(ux, uy, uz, C_s)
        return _nu_t_to_tau_eff(tau, nu_t)
    else:  # vreman
        nu_t = _vreman_nu_t_3d(ux, uy, uz, C_s)
        return _nu_t_to_tau_eff(tau, nu_t)


# ===========================================================================
# Core timestep — full Körner model (D3Q27)
# ===========================================================================

def free_surface_step_27(
    f: torch.Tensor,
    fill: torch.Tensor,
    flags: torch.Tensor,
    solid_mask: torch.Tensor,
    mass: torch.Tensor | None = None,
    tau: float = 1.0,
    gx: float = 0.0, gy: float = 0.0, gz: float = 0.0,
    rho_liquid: float = 1.0, rho_gas: float = 1.0,
    surface_tension: float = 0.0, C_s: float = 0.0,
    sgs_model: str = 'smagorinsky',
    free_slip_y: bool = False, y_wall_mask: torch.Tensor | None = None,
    collision: str = 'bgk',
    mass_ledger: dict | None = None,
    freeze_topology: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One free-surface LBM timestep (full Körner model) for D3Q27.

    This mirrors :func:`free_surface_lbm.free_surface_step` but uses the
    D3Q27 lattice with 27 velocity directions.

    Pipeline:
      1. Macroscopic + collision (BGK or MRT, optional SGS)
      2. Guo gravity force
      3. Stream (pull scheme)
      4. Zero gas cells
      5. Anti-bounce-back (ABB) for interface cells (gas pressure boundary)
      6. Wall BCs (bounce-back)
      7. Mass exchange (Körner independent mass variable)
      8. Topology conversion (G→I, I→L, I→G) + mass redistribution
      9. Halo/isolation cleanup

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        fill: Fill field of shape ``(nz, ny, nx)``.
        flags: Cell flag field of shape ``(nz, ny, nx)``.
        solid_mask: Boolean solid mask of shape ``(nz, ny, nx)``.
        mass: Independent mass field (auto-initialized if None).
        tau: Relaxation time τ > 0.5.
        gx, gy, gz: Body force components (Guo forcing).
        rho_liquid: Liquid density (lattice units).
        rho_gas: Gas density (lattice units).
        surface_tension: Surface tension coefficient (0 = disabled).
        C_s: SGS model constant (0 = no SGS).  Interpreted as the
            Smagorinsky constant, WALE constant, or Vreman constant
            depending on *sgs_model*.
        sgs_model: SGS model name — ``'smagorinsky'``, ``'wale'``, or
            ``'vreman'``.  Only used when ``C_s > 0``.
        free_slip_y: Apply free-slip on y-walls.
        y_wall_mask: Wall mask for free-slip (required if free_slip_y).
        collision: 'bgk' or 'mrt'.
        mass_ledger: Optional dict for mass tracking diagnostics.
        freeze_topology: If True, skip topology conversion (diagnostic mode).

    Returns:
        Tuple ``(f, fill, flags, mass, df)`` where ``df`` is the wall
        shear stress diagnostic (0.0 if wall function not used).
    """
    _VALID_SGS = ('smagorinsky', 'wale', 'vreman')
    if sgs_model not in _VALID_SGS:
        raise ValueError(
            f"sgs_model must be one of {_VALID_SGS}, got {sgs_model!r}"
        )
    device = f.device
    _assert_no_direct_liquid_gas_links_27(flags)
    c_dev = _C.to(device).float()
    non_gas = ~(flags == GAS)

    # Initialize mass if not provided
    if mass is None:
        mass = init_mass_from_fill_27(fill, flags, rho_liquid)
    mass_start_value = float(mass.sum())

    if mass_ledger is not None:
        mass_ledger['start'] = mass_start_value
        mass_ledger['interface_start'] = float(mass[flags == INTERFACE].sum())
        mass_ledger['liquid_start'] = float(mass[flags == LIQUID].sum())

    # ---- 1. Macroscopic + collision ----
    rho, ux, uy, uz = macroscopic27(f)
    rho_s = rho.clamp(min=1e-6, max=rho_liquid * 3.0)
    ux_eq = (ux + tau * gx).clamp(-0.5, 0.5)
    uy_eq = (uy + tau * gy).clamp(-0.5, 0.5)
    uz_eq = (uz + tau * gz).clamp(-0.5, 0.5)
    feq = equilibrium27(rho_s, ux_eq, uy_eq, uz_eq)

    # For non-BGK: set gas cells to small equilibrium (prevent NaN)
    if collision != 'bgk':
        feq_gas = equilibrium27(
            torch.full_like(rho_s, rho_gas),
            torch.zeros_like(rho_s), torch.zeros_like(rho_s), torch.zeros_like(rho_s),
        )
        f_collide = torch.where(non_gas.unsqueeze(0), f, feq_gas)
    else:
        f_collide = f

    if collision == 'mrt':
        if C_s > 0:
            tau_eff = _compute_tau_eff_sgs(
                tau, sgs_model, f_collide - feq, rho_s, ux, uy, uz, C_s,
            )
            f = _collide_mrt27_with_tau_eff(f_collide, feq, tau_eff, device)
        else:
            f = collide_mrt27(f_collide, tau)
    else:  # bgk
        if C_s > 0:
            tau_eff = _compute_tau_eff_sgs(
                tau, sgs_model, f_collide - feq, rho_s, ux, uy, uz, C_s,
            )
            f = f_collide - (f_collide - feq) / tau_eff.unsqueeze(0)
        else:
            f = f_collide - (f_collide - feq) / tau

    # Guo gravity force (only when gravity is non-zero)
    if gx != 0.0 or gy != 0.0 or gz != 0.0:
        cs2 = 1.0 / 3.0
        cx = c_dev[:, 0].view(27, 1, 1, 1)
        cy = c_dev[:, 1].view(27, 1, 1, 1)
        cz = c_dev[:, 2].view(27, 1, 1, 1)
        w_dev = _W.to(device).float().view(27, 1, 1, 1)
        ng = non_gas.float()
        Fx = rho_liquid * gx * ng
        Fy = rho_liquid * gy * ng
        Fz = rho_liquid * gz * ng
        cu_force = cx * Fx.unsqueeze(0) + cy * Fy.unsqueeze(0) + cz * Fz.unsqueeze(0)
        f = f + (1.0 - 0.5 / tau) * w_dev * cu_force / cs2

    # Surface tension force (curvature correction, standard Körner)
    if surface_tension > 0:
        cx = c_dev[:, 0].view(27, 1, 1, 1)
        cy = c_dev[:, 1].view(27, 1, 1, 1)
        cz = c_dev[:, 2].view(27, 1, 1, 1)
        w_dev = _W.to(device).float().view(27, 1, 1, 1)
        cs2 = 1.0 / 3.0
        fill_field = mass / rho_s.clamp(min=1e-6)
        grad_x = 0.5 * (fill_field.roll(-1, dims=2) - fill_field.roll(1, dims=2))
        grad_y = 0.5 * (fill_field.roll(-1, dims=1) - fill_field.roll(1, dims=1))
        grad_z = 0.5 * (fill_field.roll(-1, dims=0) - fill_field.roll(1, dims=0))
        mag = (grad_x ** 2 + grad_y ** 2 + grad_z ** 2).sqrt().clamp(min=1e-10)
        nx, ny_n, nz_n = -grad_x / mag, -grad_y / mag, -grad_z / mag
        kappa = 0.5 * (
            (nx.roll(-1, dims=2) - nx.roll(1, dims=2))
            + (ny_n.roll(-1, dims=1) - ny_n.roll(1, dims=1))
            + (nz_n.roll(-1, dims=0) - nz_n.roll(1, dims=0))
        )
        Fx_st = surface_tension * kappa * grad_x
        Fy_st = surface_tension * kappa * grad_y
        Fz_st = surface_tension * kappa * grad_z
        cu_st = cx * Fx_st.unsqueeze(0) + cy * Fy_st.unsqueeze(0) + cz * Fz_st.unsqueeze(0)
        f = f + (1.0 - 0.5 / tau) * w_dev * cu_st / cs2

    # Clamp f for numerical stability
    f = f.clamp(min=0.0, max=rho_liquid * 3.0)

    # Remove NaN for non-BGK
    if collision != 'bgk':
        f = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

    # Preserve post-collision outgoing populations for ABB
    f_post = torch.where(non_gas.unsqueeze(0), f, torch.zeros_like(f))
    f = f_post

    # ---- 2. Stream ----
    f = _stream27_roll(f)

    # ---- 2b. Zero gas cells AFTER streaming ----
    gas_mask_pre = flags == GAS
    f = torch.where(gas_mask_pre.unsqueeze(0), torch.zeros_like(f), f)

    # ---- 2c. Anti-bounce-back for interface cells (gas pressure) ----
    iface_abb = flags == INTERFACE
    # Build neighbor_flags for all 27 directions (pull-source flags)
    neighbor_flags = torch.stack([
        flags.roll(sz, dims=0).roll(sy, dims=1).roll(sx, dims=2)
        for sx, sy, sz in _C27_SHIFTS
    ])  # (27, nz, ny, nx)
    need_abb = iface_abb.unsqueeze(0) & (neighbor_flags == GAS)
    # Standard Körner ABB: f_q(x) = f_eq_gas + f_eq_gas[opp] - f_post[opp]
    rho_g_field = torch.full_like(rho, float(rho_gas))
    f_eq_gas = equilibrium27(rho_g_field, ux, uy, uz)
    f_abb = f_eq_gas + f_eq_gas[_OPP.to(device)] - f_post[_OPP.to(device)]
    abb_delta = torch.where(need_abb, f_abb - f, torch.zeros_like(f))
    if mass_ledger is not None:
        mass_ledger['abb_population_delta'] = float(abb_delta.sum())
    f = torch.where(need_abb, f_abb, f)

    # ---- 3. Wall BCs ----
    f = bounce_back_cells_27(f, solid_mask)

    # ---- 4. Mass exchange (standard Körner, independent mass variable) ----
    rho_new = f.sum(dim=0)
    iface_mask = flags == INTERFACE
    f_opp_nb = f_post[_OPP.to(device)]  # (27, nz, ny, nx)
    iface_27 = iface_mask.unsqueeze(0)
    from_liq = iface_27 & (neighbor_flags == LIQUID)
    from_gas = iface_27 & (neighbor_flags == GAS)
    from_iface = iface_27 & (neighbor_flags == INTERFACE)
    mass_delta_liquid = torch.where(from_liq, f - f_opp_nb, torch.zeros_like(f))
    mass_delta_interface = torch.where(
        from_iface, (f - f_opp_nb) * 0.5, torch.zeros_like(f),
    )
    mass_delta = (mass_delta_liquid + mass_delta_interface).sum(0)
    mass = torch.where(~solid_mask, mass + mass_delta, mass)
    mass_after_exchange_value = float(mass.sum())
    fill = torch.where(~solid_mask, (mass / rho_liquid).clamp(0.0, 1.0), fill)

    if mass_ledger is not None:
        mass_ledger['exchange'] = mass_after_exchange_value
        mass_ledger['exchange_liquid_delta'] = float(mass_delta_liquid.sum())
        mass_ledger['exchange_interface_delta'] = float(mass_delta_interface.sum())

    # Diagnostic mode: skip topology conversion
    df = torch.tensor(0.0, device=device, dtype=f.dtype)
    if freeze_topology:
        if mass_ledger is not None:
            mass_ledger['redistribution'] = mass_after_exchange_value
            mass_ledger['clamp'] = mass_after_exchange_value
            mass_ledger['conversion'] = mass_after_exchange_value
            mass_ledger['isolation'] = mass_after_exchange_value
            mass_ledger['boundary'] = float(mass.sum())
            mass_ledger['fill_mass_final'] = float((fill * rho_liquid).sum())
        return f, fill, flags, mass, df

    gas_mask = flags == GAS
    interface_mask = flags == INTERFACE
    liquid_mask = flags == LIQUID

    # Gas → Interface (received mass from streaming)
    to_iface = gas_mask & (fill > 0.01) & (~solid_mask)
    to_liq = interface_mask & (fill >= 0.999) & (~solid_mask)
    to_gas = (interface_mask | liquid_mask) & (fill <= 0.01) & (~solid_mask)

    # ---- 5a. Körner mass redistribution (excess → interface neighbors) ----
    excess = (
        torch.where(to_liq, mass - rho_liquid, torch.zeros_like(mass))
        + torch.where(to_gas, mass, torch.zeros_like(mass))
    )
    # Existing interface cells receive first
    recv_iface = interface_mask & ~to_liq & ~to_gas
    # Promote adjacent gas halo to receivers if a converting interface has none
    adjacent_converting = torch.stack(all_moving_neighbor_masks_27(to_liq)).any(dim=0)
    recv_new = gas_mask & adjacent_converting & ~solid_mask
    recv_mask = recv_iface | recv_new

    # Count receiving cells per donor over every moving D3Q27 link
    shifted_recv = torch.stack(all_moving_neighbor_masks_27(recv_mask))
    n_recv = shifted_recv.sum(dim=0).float().clamp(min=1.0)
    excess_per_nb = excess / n_recv

    # Aggregate every D3Q27 receiver contribution, then commit once
    redistribution_increment = torch.stack([
        roll_to_neighbor_27(excess_per_nb, q) * recv_mask
        for q in D3Q27_MOVING_Q
    ]).sum(dim=0)
    mass = mass + redistribution_increment
    mass_after_redistribution = float(mass.sum())

    # Clamp mass to [0, rho_liquid]
    mass = mass.clamp(0.0, rho_liquid)
    mass_after_clamp = float(mass.sum())

    # ---- 5b. Topology conversion ----
    # G→I: init newly converted gas cells with equilibrium
    f = _init_new_27(f, flags, to_iface, rho_gas, device, ux, uy, uz)
    flags = torch.where(to_iface, torch.full_like(flags, INTERFACE), flags)

    # I→L
    flags = torch.where(to_liq, torch.full_like(flags, LIQUID), flags)
    fill = torch.where(to_liq, torch.ones_like(fill), fill)
    mass = torch.where(to_liq, torch.full_like(mass, rho_liquid), mass)

    # I→G (or L→G, though L→G shouldn't happen normally)
    flags = torch.where(to_gas, torch.full_like(flags, GAS), flags)
    fill = torch.where(to_gas, torch.zeros_like(fill), fill)
    mass = torch.where(to_gas, torch.zeros_like(mass), mass)
    f = torch.where(to_gas.unsqueeze(0), torch.zeros_like(f), f)
    mass_after_conversion = float(mass.sum())

    # ---- 5c. Halo boundary (gas cells adjacent to liquid/interface → INTERFACE) ----
    shifted_flags = torch.stack(all_moving_neighbor_masks_27(flags))
    is_neighbor = (
        (shifted_flags == LIQUID) | (shifted_flags == INTERFACE)
    ).any(dim=0)
    to_i = ((gas_mask | to_gas) & is_neighbor & ~solid_mask) | recv_new
    f = _init_new_27(f, flags, to_i, rho_gas, device, ux, uy, uz)
    flags = torch.where(to_i, torch.full_like(flags, INTERFACE), flags)
    fill = torch.where(to_i & ~recv_new, torch.zeros_like(fill), fill)
    mass = torch.where(to_i & ~recv_new, torch.zeros_like(mass), mass)

    # ---- 5d. Isolation cleanup (isolated interface → gas) ----
    interface_mask = flags == INTERFACE
    has_neighbor = (
        (torch.stack(all_moving_neighbor_masks_27(flags)) == LIQUID)
        | (torch.stack(all_moving_neighbor_masks_27(flags)) == INTERFACE)
    ).any(dim=0)
    isolated = interface_mask & ~has_neighbor & ~solid_mask
    flags = torch.where(isolated, torch.full_like(flags, GAS), flags)
    fill = torch.where(isolated, torch.zeros_like(fill), fill)
    mass = torch.where(isolated, torch.zeros_like(mass), mass)
    f = torch.where(isolated.unsqueeze(0), torch.zeros_like(f), f)

    # ---- 5e. Solid enforcement ----
    flags = torch.where(solid_mask, torch.full_like(flags, SOLID), flags)

    mass_after_isolation = float(mass.sum())

    if mass_ledger is not None:
        mass_ledger['redistribution'] = mass_after_redistribution
        mass_ledger['clamp'] = mass_after_clamp
        mass_ledger['conversion'] = mass_after_conversion
        mass_ledger['isolation'] = mass_after_isolation
        mass_ledger['boundary'] = float(mass.sum())
        mass_ledger['fill_mass_final'] = float((fill * rho_liquid).sum())
        mass_ledger['end'] = float(mass.sum())
        mass_ledger['mass_drift'] = float(mass.sum()) - mass_start_value

    return f, fill, flags, mass, df
