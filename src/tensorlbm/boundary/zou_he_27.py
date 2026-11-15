"""Zou/He channel inlet/outlet reconstruction for D3Q27 (thin lattice bindings).

Same closure as :mod:`tensorlbm.boundary.zou_he_3d` (D3Q19), namely:

* non-equilibrium bounce-back on the **normal** link only,
* remaining unknown-set mass split in proportion to the equilibrium values,
* tangential momentum imposed exactly by a minimum-norm correction.

The D3Q27 unknown set has nine directions (``c_x = +-1`` face, edge and corner
links), so the minimum-norm projector is 8x3 instead of 4x3; everything else is
lattice-independent and therefore shared.  See the D3Q19 module docstring for
the derivation and for the transverse-momentum defect that the
"NEBB for all directions" construction produces.
"""

from __future__ import annotations

import torch

from tensorlbm.d3q27 import C as _C27
from tensorlbm.d3q27 import equilibrium27 as _equilibrium27

from .zou_he_3d import _apply_face


def zou_he_inlet_velocity_27(
    f: torch.Tensor,
    u_in: float | torch.Tensor,
    uy_in: float | torch.Tensor = 0.0,
    uz_in: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Zou/He velocity inlet on the west face ``x = 0`` (D3Q27).

    All three velocity components are reproduced exactly; ``rho`` is implied.
    """
    return _apply_face(
        f, column=0, mode="velocity", lattice_c=_C27, equilibrium=_equilibrium27,
        rho_name="rho_in", u=u_in, uy=uy_in, uz=uz_in, where="zou_he_inlet_velocity_27",
    )


def zou_he_inlet_pressure_27(
    f: torch.Tensor,
    rho_in: float | torch.Tensor = 1.0,
    uy_in: float | torch.Tensor = 0.0,
    uz_in: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Zou/He pressure inlet on the west face ``x = 0`` (D3Q27).

    Prescribes ``rho`` and implies ``ux``; the transverse velocity is the
    documented closure assumption (zero unless supplied).
    """
    return _apply_face(
        f, column=0, mode="pressure", lattice_c=_C27, equilibrium=_equilibrium27,
        rho_name="rho_in", rho=rho_in, uy=uy_in, uz=uz_in, where="zou_he_inlet_pressure_27",
    )


def zou_he_outlet_pressure_27(
    f: torch.Tensor,
    rho_out: float | torch.Tensor = 1.0,
    uy_out: float | torch.Tensor = 0.0,
    uz_out: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Zou/He pressure outlet on the east face ``x = nx-1`` (D3Q27)."""
    return _apply_face(
        f, column=-1, mode="pressure", lattice_c=_C27, equilibrium=_equilibrium27,
        rho_name="rho_out", rho=rho_out, uy=uy_out, uz=uz_out,
        where="zou_he_outlet_pressure_27",
    )


def zou_he_outlet_velocity_27(
    f: torch.Tensor,
    u_out: float | torch.Tensor,
    uy_out: float | torch.Tensor = 0.0,
    uz_out: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Zou/He velocity outlet on the east face ``x = nx-1`` (D3Q27)."""
    return _apply_face(
        f, column=-1, mode="velocity", lattice_c=_C27, equilibrium=_equilibrium27,
        rho_name="rho_out", u=u_out, uy=uy_out, uz=uz_out, where="zou_he_outlet_velocity_27",
    )