"""D2Q9 tank streaming with link bounce-back and an optional pressure top.

Input is POST-COLLISION, PRE-STREAMING. No link wraps around the domain.
The order parameter has an impermeable boundary even when the momentum
distribution uses a pressure top. This is not a two-phase outflow model.
"""

import math

import torch

from tensorlbm.d2q9 import OPPOSITE, C


def stream_tank(
    f: torch.Tensor,
    g: torch.Tensor,
    wall_mask: torch.Tensor,
    top_bc: str = "bounce_back",
    rho_top: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stream fluid links, reflect wall links, then reconstruct a pressure top.

    Zou/He reconstruction at y=0 prescribes density and zero tangential
    velocity. y increases downwards, so directions 2, 5, 6 are unknown.
    Solid cells are inactive storage and are excluded from conserved totals.
    """
    if f.shape != g.shape or f.ndim != 3 or f.shape[0] != 9:
        raise ValueError("f and g must have matching (9, ny, nx) shapes")
    if wall_mask.shape != f.shape[1:] or wall_mask.dtype != torch.bool:
        raise ValueError("wall_mask must be a matching boolean (ny, nx) tensor")
    if top_bc not in ("bounce_back", "pressure"):
        raise ValueError("top_bc must be 'bounce_back' or 'pressure'")
    if not math.isfinite(rho_top) or rho_top <= 0:
        raise ValueError("rho_top must be finite and positive")
    f_out, g_out = f.clone(), g.clone()
    ny, nx = wall_mask.shape
    y = torch.arange(ny, device=f.device)[:, None]
    x = torch.arange(nx, device=f.device)[None, :]
    for i, (cx, cy) in enumerate(C.tolist()):
        sy, sx = y - cy, x - cx
        inside = (sy >= 0) & (sy < ny) & (sx >= 0) & (sx < nx)
        sy, sx = sy.clamp(0, ny - 1), sx.clamp(0, nx - 1)
        fluid_link = inside & ~wall_mask[sy, sx]
        opposite = int(OPPOSITE[i])
        for src, dst in ((f, f_out), (g, g_out)):
            incoming = torch.where(fluid_link, src[i, sy, sx], src[opposite])
            dst[i] = torch.where(wall_mask, src[i], incoming)

    if top_bc == "pressure":
        _pressure_top(f_out, ~wall_mask[0], rho_top)
    return f_out, g_out


def _pressure_top(f_out, active, rho_top):
    p = f_out[:, 0, :]
    # rho*uy from the prescribed zeroth moment and known populations.
    jy = rho_top - (p[0] + p[1] + p[3] + 2.0 * (p[4] + p[7] + p[8]))
    replacements = {
        2: p[4] + (2.0 / 3.0) * jy,
        5: p[7] + 0.5 * (p[3] - p[1]) + jy / 6.0,
        6: p[8] + 0.5 * (p[1] - p[3]) + jy / 6.0,
    }
    for i, value in replacements.items():
        f_out[i, 0] = torch.where(active, value, p[i])


class TankStreamingPlan:
    """Precompute the source of every D2Q9 population for a fixed tank.

    The plan owns a snapshot of the geometry. Construct another plan when
    the tank changes. Both distributions use the same integer gather map.
    """

    def __init__(self, wall_mask, top_bc="bounce_back", rho_top=1.0):
        if wall_mask.ndim != 2 or wall_mask.dtype != torch.bool or not wall_mask.numel():
            raise ValueError("expected a nonempty 2-D boolean wall mask")
        if top_bc not in ("bounce_back", "pressure") or not math.isfinite(rho_top) or rho_top <= 0:
            raise ValueError("invalid top boundary or density")
        ny, nx = wall_mask.shape
        self.shape = (9, ny, nx)
        self.top_bc = top_bc
        self.rho_top = rho_top
        self.active_top = ~wall_mask[0]
        cells = torch.arange(ny * nx, device=wall_mask.device).reshape(ny, nx)
        maps = []
        for i, (cx, cy) in enumerate(C.tolist()):
            sy, sx = cells // nx - cy, cells % nx - cx
            inside = (sy >= 0) & (sy < ny) & (sx >= 0) & (sx < nx)
            sy, sx = sy.clamp(0, ny - 1), sx.clamp(0, nx - 1)
            source = torch.where(
                inside & ~wall_mask[sy, sx],
                i * nx * ny + sy * nx + sx,
                int(OPPOSITE[i]) * nx * ny + cells,
            )
            maps.append(torch.where(wall_mask, i * ny * nx + cells, source))
        self.indices = torch.stack(maps).flatten()

    def stream(self, f, g):
        if f.shape != self.shape or g.shape != self.shape:
            raise ValueError("population shapes must match the prepared tank")
        if f.device != self.indices.device or g.device != self.indices.device:
            raise ValueError("populations and the prepared tank must share a device")
        f_out = f.reshape(-1).index_select(0, self.indices).reshape(self.shape)
        g_out = g.reshape(-1).index_select(0, self.indices).reshape(self.shape)
        if self.top_bc == "pressure":
            _pressure_top(f_out, self.active_top, self.rho_top)
        return f_out, g_out
