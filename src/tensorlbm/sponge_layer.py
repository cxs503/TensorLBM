"""Low-reflection equilibrium-difference sponge layers for LBM.

Unlike direct blending ``(1-sigma) f + sigma f_eq*``, this damping term
relaxes only the difference between the local and target equilibria,
preserving the local non-equilibrium stress content:

``f <- f + sigma (f_eq(rho*,u*) - f_eq(rho,u))``.

The construction follows the absorbing-layer form used in mature LBM
implementations, with a C2-continuous fifth-order strength ramp.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


def smoothstep5(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return value.pow(3) * (10.0 - 15.0 * value + 6.0 * value.square())


def build_sponge_sigma_3d(
    shape: tuple[int, int, int],
    *,
    width: int,
    max_strength: float = 0.2,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    faces: tuple[str, ...] = ("x-", "x+", "y-", "y+", "z-", "z+"),
) -> torch.Tensor:
    """Return a smooth sponge coefficient field of shape ``(nz,ny,nx)``."""
    if width < 0:
        raise ValueError("width must be non-negative")
    if not 0.0 <= max_strength <= 1.0:
        raise ValueError("max_strength must be in [0,1]")
    allowed = {"x-", "x+", "y-", "y+", "z-", "z+"}
    if not set(faces) <= allowed:
        raise ValueError("unknown sponge face")
    nz, ny, nx = shape
    sigma = torch.zeros(shape, device=device, dtype=dtype)
    if width == 0 or max_strength == 0.0:
        return sigma
    coordinates = {
        "x-": torch.arange(nx, device=device, dtype=dtype).view(1, 1, nx),
        "x+": torch.arange(nx - 1, -1, -1, device=device, dtype=dtype).view(1, 1, nx),
        "y-": torch.arange(ny, device=device, dtype=dtype).view(1, ny, 1),
        "y+": torch.arange(ny - 1, -1, -1, device=device, dtype=dtype).view(1, ny, 1),
        "z-": torch.arange(nz, device=device, dtype=dtype).view(nz, 1, 1),
        "z+": torch.arange(nz - 1, -1, -1, device=device, dtype=dtype).view(nz, 1, 1),
    }
    for face in faces:
        depth = ((width - coordinates[face]) / max(width, 1)).clamp(0.0, 1.0)
        sigma = torch.maximum(sigma, max_strength * smoothstep5(depth))
    return sigma


def build_anisotropic_sponge_sigma_3d(
    shape: tuple[int, int, int],
    *,
    face_widths: Mapping[str, int],
    max_strength: float = 0.2,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a sponge with an independent thickness on each selected face."""
    sigma = torch.zeros(shape, device=device, dtype=dtype)
    for face, width in face_widths.items():
        if width < 0:
            raise ValueError("sponge face widths must be non-negative")
        if width == 0:
            continue
        face_sigma = build_sponge_sigma_3d(
            shape,
            width=width,
            max_strength=max_strength,
            device=device,
            dtype=dtype,
            faces=(face,),
        )
        sigma = torch.maximum(sigma, face_sigma)
    return sigma


def apply_equilibrium_difference_sponge(
    f: torch.Tensor,
    sigma: torch.Tensor,
    *,
    rho_target: float = 1.0,
    velocity_target: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Apply equilibrium-difference damping to D3Q19 or D3Q27 populations."""
    if sigma.shape != f.shape[1:]:
        raise ValueError("sigma must have the spatial population shape")
    if f.shape[0] == 19:
        from .d3q19 import equilibrium3d, macroscopic3d

        equilibrium, macro = equilibrium3d, macroscopic3d
    elif f.shape[0] == 27:
        from .d3q27 import equilibrium27, macroscopic27

        equilibrium, macro = equilibrium27, macroscopic27
    else:
        raise ValueError("only D3Q19 and D3Q27 are supported")
    rho, ux, uy, uz = macro(f)
    local_eq = equilibrium(rho, ux, uy, uz, device=f.device)
    target_rho = torch.full_like(rho, rho_target)
    target_eq = equilibrium(
        target_rho,
        torch.full_like(rho, velocity_target[0]),
        torch.full_like(rho, velocity_target[1]),
        torch.full_like(rho, velocity_target[2]),
        device=f.device,
    )
    return f + sigma.to(device=f.device, dtype=f.dtype).unsqueeze(0) * (target_eq - local_eq)


__all__ = [
    "apply_equilibrium_difference_sponge",
    "build_anisotropic_sponge_sigma_3d",
    "build_sponge_sigma_3d",
    "smoothstep5",
]
