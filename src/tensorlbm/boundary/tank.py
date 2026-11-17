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
        p = f_out[:, 0, :]
        # rho*uy from the prescribed zeroth moment and known populations.
        jy = rho_top - (p[0] + p[1] + p[3] + 2.0 * (p[4] + p[7] + p[8]))
        active = ~wall_mask[0]
        replacements = {
            2: p[4] + (2.0 / 3.0) * jy,
            5: p[7] + 0.5 * (p[3] - p[1]) + jy / 6.0,
            6: p[8] + 0.5 * (p[1] - p[3]) + jy / 6.0,
        }
        for i, value in replacements.items():
            f_out[i, 0] = torch.where(active, value, p[i])
    return f_out, g_out
