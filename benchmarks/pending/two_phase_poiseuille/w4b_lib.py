"""W4-B shared driver library — Wave-4 two-phase Poiseuille + Washburn.

IRON RULE COMPLIANCE: no physics kernels are defined here. All collide/
stream/equilibrium/bounce-back calls are the library functions from
tensorlbm (multiphase.collide_sc_two_component, solver.stream,
boundaries.bounce_back_cells, d2q9.equilibrium/macroscopic,
porous_media.apply_wall_wettability_sc). This module only provides
initialisation, driving loops, and measurement code.
grep self-check: no 'def collide|stream|equilibrium|bounce|zou_he|far_world' here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))  # <repo>/src

import torch  # noqa: E402

from tensorlbm.boundaries import bounce_back_cells  # noqa: E402
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.multiphase import collide_sc_two_component  # noqa: E402
from tensorlbm.porous_media import apply_wall_wettability_sc  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402

CS2 = 1.0 / 3.0


# ---------------------------------------------------------------- init fields


def tanh_field_1d(ny: int, nx: int, y0: float, width: float, device) -> torch.Tensor:
    """alpha=1 below y0 (water side), 0 above, tanh of given width."""
    ys = torch.arange(ny, dtype=torch.float32, device=device).view(ny, 1)
    ys = ys.expand(ny, nx)
    return 0.5 * (1.0 - torch.tanh((ys - y0) / width))


def init_full_swap(alpha: torch.Tensor, heavy: float, light: float):
    """Full-swap MCMP fields: rho_w = heavy*alpha + light*(1-alpha) etc."""
    rho_w = heavy * alpha + light * (1.0 - alpha)
    rho_g = light * alpha + heavy * (1.0 - alpha)
    return rho_w, rho_g


def init_from_fractions(alpha: torch.Tensor, rho_w_a, rho_g_a, rho_w_b, rho_g_b):
    """Generic init: phase A where alpha=1, phase B where alpha=0 (smooth)."""
    rho_w = rho_w_a * alpha + rho_w_b * (1.0 - alpha)
    rho_g = rho_g_a * alpha + rho_g_b * (1.0 - alpha)
    return rho_w, rho_g


# ---------------------------------------------------------------- driving loops


def sc_mcmp_step(fw, fg, G12, tau1, tau2, gx=0.0, gy=0.0, solid=None):
    """One SC MCMP step with library kernels only."""
    fw, fg = collide_sc_two_component(
        fw, fg, G_12=G12, tau1=tau1, tau2=tau2, gx=gx, gy=gy, solid_mask=solid
    )
    fw = stream(fw)
    fg = stream(fg)
    if solid is not None:
        fw = bounce_back_cells(fw, solid)
        fg = bounce_back_cells(fg, solid)
    return fw, fg


def tube_solid_mask(ny: int, nx: int, tube_width: int, device) -> torch.Tensor:
    """Same geometry rule as porous_media._capillary_wall_mask."""
    solid = torch.ones((ny, nx), dtype=torch.bool, device=device)
    y_center = ny // 2
    y_lo = y_center - tube_width // 2
    y_hi = y_center + tube_width // 2
    solid[y_lo : y_hi + 1, :] = False
    return solid


def apply_adsorption_and_solid_eq(fw, fg, solid, G_ads_water, G_ads_gas, G12, tau1, tau2):
    """Replicates run_capillary_invasion wall-wettability protocol exactly:
    set adsorbed pseudo-densities at solid nodes, then overwrite the solid-node
    distributions with the equilibrium at those densities (zero velocity) so the
    SC force in the next collision sees the adsorbed values."""
    zero = torch.zeros_like(fw[0])
    rho_w, _, _ = macroscopic(fw)
    rho_g, _, _ = macroscopic(fg)
    rho_w, rho_g = apply_wall_wettability_sc(
        rho_w, rho_g, solid, G_ads1=G_ads_water, G_ads2=G_ads_gas
    )
    feq_w = equilibrium(rho_w, zero, zero)
    feq_g = equilibrium(rho_g, zero, zero)
    solid_4d = solid.unsqueeze(0)
    fw = torch.where(solid_4d, feq_w, fw)
    fg = torch.where(solid_4d, feq_g, fg)
    return fw, fg


# ---------------------------------------------------------------- measurements


def mixture_fields(fw, fg):
    rho_w, ux_w, uy_w = macroscopic(fw)
    rho_g, ux_g, uy_g = macroscopic(fg)
    rho_tot = (rho_w + rho_g).clamp(min=1e-12)
    ux = (rho_w * ux_w + rho_g * ux_g) / rho_tot
    uy = (rho_w * ux_w + rho_g * uy_g) / rho_tot
    return rho_w, rho_g, rho_tot, ux, uy


def gas_front(fw, fg, solid) -> float:
    """Same rule as porous_media._measure_invasion_front (phi>0.4 max column)."""
    rho_w, rho_g, _, _, _ = mixture_fields(fw, fg)
    fluid = (~solid).float()
    phi = rho_g / (rho_w + rho_g + 1e-12)
    phi_col = (phi * fluid).sum(dim=0) / fluid.sum(dim=0).clamp(min=1)
    gas_cols = (phi_col > 0.4).nonzero(as_tuple=True)[0]
    if gas_cols.numel() == 0:
        return 0.0
    return float(gas_cols.max().item())
