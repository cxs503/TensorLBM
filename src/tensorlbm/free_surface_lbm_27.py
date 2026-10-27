"""Free-surface LBM (D3Q27) — full Körner model with mass tracking.

Reference mechanics (Körner et al. 2005; Thürey 2007; waLBerla
``free_surface``), mirroring :mod:`free_surface_lbm` (D3Q19).

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

import os

import torch

from .boundaries_d3q27 import bounce_back_cells_27
from .core.d3q27_stencil import (
    D3Q27_MOVING_Q,
    all_moving_neighbor_masks_27,
    assert_no_direct_phase_links_27,
    roll_from_pull_source_27,
    roll_to_neighbor_27,
)
from .d3q27 import OPPOSITE as OPP27
from .d3q27 import C as C27
from .d3q27 import W as W27
from .d3q27 import (
    _get_d3q27_mrt_matrices,
    collide_mrt27,
    equilibrium27,
    macroscopic27,
)
from .turbulence import (
    _neq_stress_norm_27,
    _nu_t_to_tau_eff,
    _smagorinsky_tau,
    _vreman_nu_t_3d,
    _wale_nu_t_3d,
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


# D3Q27 specular mirror tables (built from the stencil: flip one component).
def _build_mirror_27(axis: int) -> "torch.Tensor":
    _key = {}
    for _q in range(27):
        _key[(int(C27[_q, 0]), int(C27[_q, 1]), int(C27[_q, 2]))] = _q
    out = []
    for _q in range(27):
        _c = [int(C27[_q, 0]), int(C27[_q, 1]), int(C27[_q, 2])]
        _c[axis] = -_c[axis]
        out.append(_key[tuple(_c)])
    return torch.tensor(out, dtype=torch.long)


_D3Q27_MIRROR_X = _build_mirror_27(0)
_D3Q27_MIRROR_Y = _build_mirror_27(1)
_D3Q27_MIRROR_Z = _build_mirror_27(2)


def _env_bool_27(name: str, default: bool = False) -> bool:
    """Read an opt-out boolean env gate for A/B diagnosis of the FS fixes."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "")


def _specular_wall_mirror_index_27(slip_mask, device):
    """Per-link specular reflected source indices for D3Q27 slip walls.

    For every link ``q`` at a fluid cell whose pull source lies in a slip
    wall, the reflected link reverses each coordinate component of ``c_q``
    for which the corresponding signed axis neighbour is a slip wall (a face
    wall reverses one component -> specular; an edge/corner reverses two or
    three -> full bounce).  Returns a ``(27, nz, ny, nx)`` int64 tensor of
    source link indices to gather from ``f_post``.
    """
    smx = slip_mask.roll(1, dims=2)
    spx = slip_mask.roll(-1, dims=2)
    smy = slip_mask.roll(1, dims=1)
    spy = slip_mask.roll(-1, dims=1)
    smz = slip_mask.roll(1, dims=0)
    spz = slip_mask.roll(-1, dims=0)
    off2idx = {(sx, sy, sz): q for q, (sx, sy, sz) in enumerate(_C27_SHIFTS)}
    table = torch.zeros((27, 2, 2, 2), dtype=torch.long)
    for q, (cx, cy, cz) in enumerate(_C27_SHIFTS):
        for fx in (0, 1):
            for fy in (0, 1):
                for fz in (0, 1):
                    table[q, fx, fy, fz] = off2idx[
                        (-cx if fx else cx, -cy if fy else cy, -cz if fz else cz)
                    ]
    table = table.to(device)
    idx = torch.empty((27,) + tuple(slip_mask.shape), dtype=torch.long, device=device)
    for q, (cx, cy, cz) in enumerate(_C27_SHIFTS):
        fx = ((cx > 0) & smx) | ((cx < 0) & spx)
        fy = ((cy > 0) & smy) | ((cy < 0) & spy)
        fz = ((cz > 0) & smz) | ((cz < 0) & spz)
        idx[q] = table[q, fx.long(), fy.long(), fz.long()]
    return idx




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
        raise ValueError(f"{error}; insert INTERFACE cells between LIQUID and GAS") from None


# ===========================================================================
# Initialization
# ===========================================================================


def init_fill_rectangular_27(
    nz: int,
    ny: int,
    nx: int,
    column_width: int,
    column_height: int,
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
    fill: torch.Tensor,
    solid_mask: torch.Tensor,
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
    fill: torch.Tensor,
    flags: torch.Tensor,
    rho_liquid: float = 1.0,
) -> torch.Tensor:
    """Initialize mass field: mass = fill * rho_liquid for liquid/interface."""
    mass = torch.zeros_like(fill)
    mass[flags == LIQUID] = rho_liquid
    mass[flags == INTERFACE] = fill[flags == INTERFACE] * rho_liquid
    return mass


def total_liquid_inventory_27(
    f: torch.Tensor,
    fill: torch.Tensor,
    flags: torch.Tensor,
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
    f: torch.Tensor,
    flags: torch.Tensor,
    mask: torch.Tensor,
    rho_init: float,
    device: torch.device,
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
    if rho_init is None:
        # Neighbour-averaged *active* density (waLBerla EquilibriumRefilling).
        rho = f.sum(0)
        rho_stack = torch.stack([roll_from_pull_source_27(rho, q) for q in D3Q27_MOVING_Q])
        rho_nb = (rho_stack * sl_f).sum(dim=0) / cnt
        rho_f = torch.where(sl_f.sum(dim=0) > 0, rho_nb, torch.zeros_like(rho_nb))
    elif isinstance(rho_init, torch.Tensor):
        rho_f = rho_init.to(f.dtype)
    else:
        rho_f = torch.full_like(ux, float(rho_init))
    feq = equilibrium27(rho_f, uxa / cnt, uya / cnt, uza / cnt)
    return torch.where(mask.unsqueeze(0), feq, f)


# ===========================================================================
# Interface normal computation (for mass redistribution)
# ===========================================================================


def _compute_interface_normal_27(
    flags: torch.Tensor,
    mass: torch.Tensor,
    rho: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute interface normal n = -∇fill / |∇fill| (points from liquid to gas)."""
    fill = mass / rho.clamp(min=1e-6)
    grad_x = 0.5 * (fill.roll(-1, dims=2) - fill.roll(1, dims=2))
    grad_y = 0.5 * (fill.roll(-1, dims=1) - fill.roll(1, dims=1))
    grad_z = 0.5 * (fill.roll(-1, dims=0) - fill.roll(1, dims=0))
    mag = (grad_x**2 + grad_y**2 + grad_z**2).sqrt().clamp(min=1e-10)
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
            0.0,  # 0  mass
            0.0,  # 1  jx
            0.0,  # 2  jy
            0.0,  # 3  jz
            s_e,  # 4  energy
            0.0,  # 5  Nxx  – overridden below
            0.0,  # 6  Nyy  – overridden below
            0.0,  # 7  Pxy  – overridden below
            0.0,  # 8  Pxz  – overridden below
            0.0,  # 9  Pyz  – overridden below
            s_q,  # 10
            s_q,  # 11
            s_q,  # 12
            s_q,  # 13
            s_q,  # 14
            s_q,  # 15
            s_q,  # 16
            s_q,  # 17
            s_q,  # 18
            s_eps,  # 19
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
    if sgs_model == "smagorinsky":
        return _smagorinsky_tau(tau, _neq_stress_norm_27(f_neq), rho, C_s)
    elif sgs_model == "wale":
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
    gx: float = 0.0,
    gy: float = 0.0,
    gz: float = 0.0,
    rho_liquid: float = 1.0,
    rho_gas: float = 1.0,
    surface_tension: float = 0.0,
    C_s: float = 0.0,
    sgs_model: str = "smagorinsky",
    free_slip_y: bool = False,
    y_wall_mask: torch.Tensor | None = None,
    collision: str = "bgk",
    mass_ledger: dict | None = None,
    freeze_topology: bool = False,
    bubble_pressure: float | None = None,
    cell_conversion_threshold: float = 1e-2,
    cell_conversion_force_threshold: float = 0.1,
    free_slip_mask: torch.Tensor | None = None,
    free_slip_all: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One free-surface LBM timestep (full Körner model) for D3Q27.

    Reference mechanics (Körner et al. 2005; Thürey 2007; waLBerla
    ``free_surface``), mirroring :func:`free_surface_lbm.free_surface_step`:

      1. Macroscopic + collision (BGK or MRT, optional SGS; skipped in
         SOLID/GAS) + single fill-weighted Guo force (Körner eq. 15)
      2. Pull streaming + gas zeroing
      2c. ABB gas-pressure reconstruction (Thürey eq. 4.5) with Laplace
          pressure (``surface_tension``) and ``bubble_pressure`` support
      3. Fluid-side half-way wall bounce-back, with selective specular
         (free-slip) reflection on ``free_slip_mask`` / y-walls / all walls
      4. Körner mass exchange (eq. 9): full-weight L/I links, fill-averaged
         I/I links with the Thürey table-4.1 selector, no gas flux
      5. Conversion marking: only INTERFACE converts (fill > 1+κ / < -κ plus
         forced neighbourhood rules)
      5a. Even excess redistribution to surviving + conversion-driven new
          receivers (local donor density)
      5b. Conversion (I→L keeps PDFs), conversion-driven envelope + LIQUID
          demotion, no clamp, no isolation stage

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
        surface_tension: Surface tension coefficient for the Laplace pressure
            jump in the ABB gas density (0 = disabled).
        C_s: SGS model constant (0 = no SGS).
        sgs_model: SGS model name — ``'smagorinsky'``, ``'wale'``, or
            ``'vreman'``.  Only used when ``C_s > 0``.
        free_slip_y: Apply free-slip on y-walls.
        y_wall_mask: Wall mask for free-slip (required if free_slip_y).
        collision: 'bgk' or 'mrt'.
        mass_ledger: Optional dict for mass tracking diagnostics.
        freeze_topology: If True, skip topology conversion (diagnostic mode).
        bubble_pressure: Optional override of the ABB gas reference density
            (enclosed, possibly pressurised gas volume).
        cell_conversion_threshold: κ for fill > 1+κ / fill < -κ conversion.
        cell_conversion_force_threshold: κ_f for the forced rules.
        free_slip_mask: Optional boolean mask of free-slip (specular) walls.
        free_slip_all: Free-slip on all solid walls.

    Returns:
        Tuple ``(f, fill, flags, mass, df)`` where ``df`` is the wall
        shear stress diagnostic (0.0 if wall function not used).
    """
    _VALID_SGS = ("smagorinsky", "wale", "vreman")
    if sgs_model not in _VALID_SGS:
        raise ValueError(f"sgs_model must be one of {_VALID_SGS}, got {sgs_model!r}")
    device = f.device
    _assert_no_direct_liquid_gas_links_27(flags)
    c_dev = _C.to(device).float()
    non_gas = ~(flags == GAS)

    # Initialize mass if not provided
    if mass is None:
        mass = init_mass_from_fill_27(fill, flags, rho_liquid)
    mass_start_value = float(mass.sum())

    if mass_ledger is not None:
        mass_ledger["start"] = mass_start_value
        mass_ledger["interface_start"] = float(mass[flags == INTERFACE].sum())
        mass_ledger["liquid_start"] = float(mass[flags == LIQUID].sum())

    # ---- 1. Macroscopic + collision ----
    rho, ux, uy, uz = macroscopic27(f)
    if _env_bool_27("TL_FS_INPUT_GUARD", True):
        ux = ux.clamp(-0.5, 0.5)
        uy = uy.clamp(-0.5, 0.5)
        uz = uz.clamp(-0.5, 0.5)
    rho_s = rho.clamp(min=1e-6, max=rho_liquid * 3.0)
    liquid_pre = flags == LIQUID
    iface_pre = flags == INTERFACE
    active_pre = liquid_pre | iface_pre
    fill_pre = torch.where(
        active_pre, (mass / rho_s.clamp(min=1e-6)).clamp(0.0, 1.0), torch.zeros_like(mass)
    )
    force_weight = torch.where(iface_pre, fill_pre, torch.ones_like(fill_pre))
    force_weight = torch.where(active_pre, force_weight, torch.zeros_like(force_weight))
    ux_eq = (ux + 0.5 * force_weight * gx).clamp(-0.5, 0.5)
    uy_eq = (uy + 0.5 * force_weight * gy).clamp(-0.5, 0.5)
    uz_eq = (uz + 0.5 * force_weight * gz).clamp(-0.5, 0.5)
    feq = equilibrium27(rho_s, ux_eq, uy_eq, uz_eq)
    f_pre_collision = f

    # For non-BGK: set gas cells to small equilibrium (prevent NaN)
    if collision != "bgk":
        feq_gas = equilibrium27(
            torch.full_like(rho_s, rho_gas),
            torch.zeros_like(rho_s),
            torch.zeros_like(rho_s),
            torch.zeros_like(rho_s),
        )
        f_collide = torch.where(non_gas.unsqueeze(0), f, feq_gas)
    else:
        f_collide = f

    if collision == "mrt":
        if C_s > 0:
            tau_eff = _compute_tau_eff_sgs(
                tau,
                sgs_model,
                f_collide - feq,
                rho_s,
                ux,
                uy,
                uz,
                C_s,
            )
            f = _collide_mrt27_with_tau_eff(f_collide, feq, tau_eff, device)
        else:
            f = collide_mrt27(f_collide, tau)
    else:  # bgk
        if C_s > 0:
            tau_eff = _compute_tau_eff_sgs(
                tau,
                sgs_model,
                f_collide - feq,
                rho_s,
                ux,
                uy,
                uz,
                C_s,
            )
            f = f_collide - (f_collide - feq) / tau_eff.unsqueeze(0)
        else:
            f = f_collide - (f_collide - feq) / tau

    # Single Guo body force (Körner eq. 15: F = a * rho * fill at INTERFACE,
    # F = a * rho at LIQUID), with the full second-order Guo source term.
    if gx != 0.0 or gy != 0.0 or gz != 0.0:
        cs2 = 1.0 / 3.0
        cx = c_dev[:, 0].view(27, 1, 1, 1)
        cy = c_dev[:, 1].view(27, 1, 1, 1)
        cz = c_dev[:, 2].view(27, 1, 1, 1)
        w_dev = _W.to(device).float().view(27, 1, 1, 1)
        Fx = rho_s * gx * force_weight
        Fy = rho_s * gy * force_weight
        Fz = rho_s * gz * force_weight
        cu = (cx * ux + cy * uy + cz * uz) / cs2
        forcing = (
            ((cx - ux) * Fx + (cy - uy) * Fy + (cz - uz) * Fz) / cs2
            + cu * (cx * Fx + cy * Fy + cz * Fz) / cs2
        )
        f = f + (1.0 - 0.5 / tau) * w_dev * forcing

    # NOTE: surface tension is carried as a Laplace pressure jump inside the
    # ABB gas density below (Thürey 2007 §4.2), not as a CSF body force.

    # Clamp f for numerical stability
    f = f.clamp(min=0.0, max=rho_liquid * 3.0)

    # Remove NaN for non-BGK
    if collision != "bgk":
        f = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

    # Collision is skipped in SOLID and GAS cells (reference dynamics apply
    # only to liquid/interface cells).
    f = torch.where(solid_mask.unsqueeze(0), f_pre_collision, f)
    f = torch.where((flags == GAS).unsqueeze(0), torch.zeros_like(f), f)

    # Preserve post-collision outgoing populations for ABB
    f_post = torch.where(non_gas.unsqueeze(0), f, torch.zeros_like(f))
    f = f_post

    # ---- 2. Stream ----
    f = _stream27_roll(f)

    # ---- 2b. Zero gas cells AFTER streaming ----
    gas_mask_pre = flags == GAS
    f = torch.where(gas_mask_pre.unsqueeze(0), torch.zeros_like(f), f)
    if _env_bool_27("TL_FS_INPUT_GUARD", True):
        f = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0).clamp(
            min=0.0, max=rho_liquid * 3.0
        )
    # Freeze the pure streamed state before ABB for the mass exchange.
    f_exchange = f

    # ---- 2c. Anti-bounce-back for interface cells (gas pressure) ----
    iface_abb = flags == INTERFACE
    # Build neighbor_flags for all 27 directions (pull-source flags)
    neighbor_flags = torch.stack(
        [flags.roll(sz, dims=0).roll(sy, dims=1).roll(sx, dims=2) for sx, sy, sz in _C27_SHIFTS]
    )  # (27, nz, ny, nx)
    need_abb = iface_abb.unsqueeze(0) & (neighbor_flags == GAS)
    _opp_dev = _OPP.to(device)
    _rho_pc, _ux_pc, _uy_pc, _uz_pc = macroscopic27(f_post)
    if _env_bool_27("TL_FS_INPUT_GUARD", True):
        _ux_pc = _ux_pc.clamp(-0.5, 0.5)
        _uy_pc = _uy_pc.clamp(-0.5, 0.5)
        _uz_pc = _uz_pc.clamp(-0.5, 0.5)
    _rho_gas_base = float(bubble_pressure) if bubble_pressure is not None else float(rho_gas)
    rho_g_field = torch.full_like(rho, _rho_gas_base)
    if surface_tension > 0:
        # Laplace pressure jump inside the ABB gas density: Δρ = 3σκ, ±0.1.
        gx_f = 0.5 * (fill_pre.roll(-1, dims=2) - fill_pre.roll(1, dims=2))
        gy_f = 0.5 * (fill_pre.roll(-1, dims=1) - fill_pre.roll(1, dims=1))
        gz_f = 0.5 * (fill_pre.roll(-1, dims=0) - fill_pre.roll(1, dims=0))
        mag = (gx_f**2 + gy_f**2 + gz_f**2).sqrt().clamp(min=1e-10)
        nx_n, ny_n, nz_n = -gx_f / mag, -gy_f / mag, -gz_f / mag
        kappa = 0.5 * (
            (nx_n.roll(-1, dims=2) - nx_n.roll(1, dims=2))
            + (ny_n.roll(-1, dims=1) - ny_n.roll(1, dims=1))
            + (nz_n.roll(-1, dims=0) - nz_n.roll(1, dims=0))
        )
        delta_rho = (3.0 * float(surface_tension) * kappa).clamp(-0.1, 0.1)
        rho_g_field = rho_g_field + torch.where(
            iface_abb, delta_rho, torch.zeros_like(delta_rho)
        )
    # Standard Körner ABB: f_q(x) = f_eq_gas + f_eq_gas[opp] - f_post[opp].
    # Consistent only for rho_gas O(rho_liquid); use rho_gas = rho_liquid for
    # an open free surface (pressure-matched ambient).
    f_eq_gas = equilibrium27(rho_g_field, _ux_pc, _uy_pc, _uz_pc)
    f_abb = f_eq_gas + f_eq_gas[_opp_dev] - f_post[_opp_dev]
    abb_delta = torch.where(need_abb, f_abb - f, torch.zeros_like(f))
    if mass_ledger is not None:
        mass_ledger["abb_population_delta"] = float(abb_delta.sum())
    f = torch.where(need_abb, f_abb, f)

    # ---- 3. Wall BCs (fluid-side half-way bounce-back + selective slip) ----
    _wall_mode = os.environ.get("TL_FS_WALL_MODE", "halfway").strip().lower()
    if _wall_mode != "legacy":
        _src_solid = torch.stack(
            [
                solid_mask.roll(sz, dims=0).roll(sy, dims=1).roll(sx, dims=2)
                for sx, sy, sz in _C27_SHIFTS
            ]
        )  # (27,nz,ny,nx): pull source x - c_q lies in SOLID
        _fluid_bb = (~solid_mask).unsqueeze(0) & _src_solid
        _f_reflect = f_post[_opp_dev]
        _slip = free_slip_mask
        if _slip is None and free_slip_all:
            _slip = solid_mask
        if _slip is not None:
            _src_slip = torch.stack(
                [
                    _slip.roll(sz, dims=0).roll(sy, dims=1).roll(sx, dims=2)
                    for sx, sy, sz in _C27_SHIFTS
                ]
            )
            _mir_idx = _specular_wall_mirror_index_27(_slip, device)
            _f_reflect = torch.where(
                _fluid_bb & _src_slip, torch.gather(f_post, 0, _mir_idx), _f_reflect
            )
        elif free_slip_y and y_wall_mask is not None:
            _src_y = torch.stack(
                [
                    y_wall_mask.roll(sz, dims=0).roll(sy, dims=1).roll(sx, dims=2)
                    for sx, sy, sz in _C27_SHIFTS
                ]
            )
            _f_reflect = torch.where(
                _fluid_bb & _src_y, f_post[_D3Q27_MIRROR_Y.to(device)], _f_reflect
            )
        f = torch.where(_fluid_bb, _f_reflect, f)
    f = bounce_back_cells_27(f, solid_mask)

    # ---- 4. Mass exchange (Körner eq. 9 / Thürey table 4.1) ----
    iface_mask = flags == INTERFACE
    f_opp_nb = f_post[_opp_dev]  # (27, nz, ny, nx)
    iface_27 = iface_mask.unsqueeze(0)
    from_liq = iface_27 & (neighbor_flags == LIQUID)
    from_iface = iface_27 & (neighbor_flags == INTERFACE)
    diff = f_exchange - f_opp_nb
    mass_delta_liquid = torch.where(from_liq, diff, torch.zeros_like(f))
    fill_nb = torch.stack(
        [fill_pre] + [roll_from_pull_source_27(fill_pre, q) for q in D3Q27_MOVING_Q]
    )
    ii_weight = 0.5 * (fill_pre.unsqueeze(0) + fill_nb)
    liq_nbr_count = (neighbor_flags == LIQUID).sum(dim=0)
    liq_nbr_at_source = torch.stack(
        [liq_nbr_count]
        + [roll_from_pull_source_27(liq_nbr_count, q) for q in D3Q27_MOVING_Q]
    )
    gas_nbr_here = (neighbor_flags == GAS).any(dim=0).unsqueeze(0)
    ii_dead = from_iface & (liq_nbr_at_source == 0) & ~gas_nbr_here
    mass_delta_interface = torch.where(
        from_iface & ~ii_dead, ii_weight * diff, torch.zeros_like(f)
    )
    mass_delta = (mass_delta_liquid + mass_delta_interface).sum(0)
    mass_delta = torch.nan_to_num(mass_delta, nan=0.0, posinf=0.0, neginf=0.0)
    mass = torch.where(iface_mask, mass + mass_delta, mass)
    mass_after_exchange_value = float(mass.sum())
    # Post-boundary local density: fill = mass / rho_local (unclamped).
    rho_new = f.sum(dim=0).clamp(min=1e-6)
    fill = torch.where(iface_mask, mass / rho_new, fill_pre)
    fill = torch.where(liquid_pre, torch.ones_like(fill), fill)
    fill = torch.where(gas_mask_pre | solid_mask, torch.zeros_like(fill), fill)

    if mass_ledger is not None:
        mass_ledger["exchange"] = mass_after_exchange_value
        mass_ledger["exchange_liquid_delta"] = float(mass_delta_liquid.sum())
        mass_ledger["exchange_interface_delta"] = float(mass_delta_interface.sum())

    # Diagnostic mode: skip topology conversion
    df = torch.tensor(0.0, device=device, dtype=f.dtype)
    if freeze_topology:
        if mass_ledger is not None:
            mass_ledger["redistribution"] = mass_after_exchange_value
            mass_ledger["clamp"] = mass_after_exchange_value
            mass_ledger["conversion"] = mass_after_exchange_value
            mass_ledger["isolation"] = mass_after_exchange_value
            mass_ledger["boundary"] = float(mass.sum())
            mass_ledger["fill_mass_final"] = float((fill * rho_liquid).sum())
        return f, fill, flags, mass, df

    gas_mask = flags == GAS
    interface_mask = flags == INTERFACE
    liquid_mask = flags == LIQUID

    # ---- 5. Conversion marking (interface-only + forced rules) ----
    _kappa = float(cell_conversion_threshold)
    _kappa_f = float(cell_conversion_force_threshold)
    to_liq = interface_mask & (fill > 1.0 + _kappa)
    to_gas = interface_mask & (fill < -_kappa)
    _has_gas_nb = (neighbor_flags == GAS).any(dim=0)
    _has_liq_nb = (neighbor_flags == LIQUID).any(dim=0)
    _has_if_nb = (neighbor_flags == INTERFACE).any(dim=0)
    to_liq = to_liq | (interface_mask & ~_has_gas_nb & ((fill > 1.0 - _kappa_f) | ~_has_if_nb))
    to_gas = to_gas | (interface_mask & ~_has_liq_nb & ((fill < _kappa_f) | ~_has_if_nb))
    to_gas = to_gas & ~to_liq  # liquid priority on the double trigger

    # ---- 5a. Even excess redistribution (local donor density) ----
    excess = torch.where(to_liq, mass - rho_new, torch.zeros_like(mass)) + torch.where(
        to_gas, mass, torch.zeros_like(mass)
    )
    recv_iface = interface_mask & ~to_liq & ~to_gas
    adjacent_converting = torch.stack(all_moving_neighbor_masks_27(to_liq)).any(dim=0)
    recv_new = gas_mask & adjacent_converting & ~solid_mask
    recv_mask = recv_iface | recv_new

    # Count receiving cells per donor over every moving D3Q27 link
    shifted_recv = torch.stack(all_moving_neighbor_masks_27(recv_mask))
    n_recv = shifted_recv.sum(dim=0).float().clamp(min=1.0)
    excess_per_nb = excess / n_recv

    # Aggregate every D3Q27 receiver contribution, then commit once
    redistribution_increment = torch.stack(
        [roll_to_neighbor_27(excess_per_nb, q) * recv_mask for q in D3Q27_MOVING_Q]
    ).sum(dim=0)
    mass = mass + redistribution_increment
    mass_after_redistribution = float(mass.sum())

    # No clamp (the reference model has none).
    mass_after_clamp = float(mass.sum())

    # ---- 5b. Topology conversion (I→L keeps PDFs) ----
    flags = torch.where(to_liq, torch.full_like(flags, LIQUID), flags)
    fill = torch.where(to_liq, torch.ones_like(fill), fill)
    mass = torch.where(to_liq, rho_new, mass)

    flags = torch.where(to_gas, torch.full_like(flags, GAS), flags)
    fill = torch.where(to_gas, torch.zeros_like(fill), fill)
    mass = torch.where(to_gas, torch.zeros_like(mass), mass)
    f = torch.where(to_gas.unsqueeze(0), torch.zeros_like(f), f)
    mass_after_conversion = float(mass.sum())

    # ---- 5c. Conversion-driven envelope + LIQUID demotion ----
    to_i = recv_new & (flags == GAS) & ~solid_mask
    f = _init_new_27(f, flags, to_i, None, device, ux, uy, uz)
    flags = torch.where(to_i, torch.full_like(flags, INTERFACE), flags)
    rho_fill = torch.where(to_i, f.sum(dim=0).clamp(min=1e-6), rho_new)
    fill = torch.where(to_i, mass / rho_fill.clamp(min=1e-6), fill)
    liq_adj_new_gas = torch.stack(all_moving_neighbor_masks_27(to_gas)).any(dim=0)
    demote = (flags == LIQUID) & liq_adj_new_gas & ~solid_mask
    flags = torch.where(demote, torch.full_like(flags, INTERFACE), flags)
    fill = torch.where(demote, mass / rho_fill.clamp(min=1e-6), fill)
    fill = torch.where(flags == INTERFACE, mass / rho_fill.clamp(min=1e-6), fill)
    fill = torch.where(flags == LIQUID, torch.ones_like(fill), fill)
    fill = torch.where((flags == GAS) | solid_mask, torch.zeros_like(fill), fill)

    # No isolation stage (forced rules at marking time cover it).
    mass_after_isolation = float(mass.sum())

    # ---- 5e. Solid enforcement ----
    flags = torch.where(solid_mask, torch.full_like(flags, SOLID), flags)

    if mass_ledger is not None:
        mass_ledger["redistribution"] = mass_after_redistribution
        mass_ledger["clamp"] = mass_after_clamp
        mass_ledger["conversion"] = mass_after_conversion
        mass_ledger["isolation"] = mass_after_isolation
        mass_ledger["boundary"] = float(mass.sum())
        mass_ledger["fill_mass_final"] = float((fill * rho_liquid).sum())
        mass_ledger["end"] = float(mass.sum())
        mass_ledger["mass_drift"] = float(mass.sum()) - mass_start_value

    return f, fill, flags, mass, df
