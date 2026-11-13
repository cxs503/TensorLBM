"""D3Q19-isotropic finite-difference operators for 3-D scalar phase fields.

Why this module exists
----------------------
The Allen-Cahn / capillary force chain needs a Laplacian and a gradient of the
phase field.  :mod:`tensorlbm.phasefield.operators` provides the *seven-point*
Laplacian and the *two-point* central gradient.  Those are second-order accurate
on a Cartesian grid, but their leading truncation terms are **anisotropic**:
the seven-point Laplacian has a fourth-order angular error, so on a curved
interface the discrete chemical potential ``mu`` and the capillary force
``F = mu grad(phi)`` pick up a spurious *tangential* component that has no
continuum counterpart and drives parasitic currents.

The open-source reference implementation ``lbmpy``
(``lbmpy.phasefield_allen_cahn.derivatives``, i.e. the Allen-Cahn CSF force
chain of Fakhari/van der Graaf, PhysRevE.96.053301) instead derives its
derivatives from the *lattice stencil itself* with
``pystencils.fd.derivation.FiniteDifferenceStencilDerivation`` and the
``isotropic=True`` constraint.  For D3Q19 that yields (verified symbolically in
``scripts/xval_m17410_allen_cahn_csf_force_chain.py``)::

    lap_iso  = (1/3) * sum_{6 axis neighbours} phi  +  (1/6) * sum_{12 face-diagonal neighbours} phi  - 4 phi
    grad_x   = (1/6) * (phi_E - phi_W)
               + (1/12) * [ (phi_BE - phi_BW) + (phi_NE - phi_NW) + (phi_SE - phi_SW) + (phi_TE - phi_TW) ]

This module implements exactly those two operators on the TensorLBM
``(z, y, x)`` tensor layout, with the same ``periodic`` / ``no_flux`` boundary
policy semantics as :mod:`tensorlbm.phasefield.operators` (implemented *by
reusing* ``_neighbor`` from there, so boundary handling cannot drift apart).

Provenance
----------
* Reference source: ``lbmpy/phasefield_allen_cahn/derivatives.py``
  (``laplacian_symbolic``, ``isotropic_gradient_symbolic``), lbmpy 2.0.
* Weights are re-derived here in closed form and unit-tested against lbmpy's
  own symbolic output when lbmpy is importable
  (``tests/test_allen_cahn_isotropic_fd_xval_m17410.py``).
"""

from __future__ import annotations

import torch

from .operators import BoundaryPolicy, _neighbor

#: The 12 face-diagonal neighbour directions of the D3Q19 stencil, written in
#: the ``(z, y, x)`` axis order used by the tensors of this package.
FACE_DIAGONAL_OFFSETS: tuple[tuple[int, int, int], ...] = tuple(
    tuple(
        {axis_a: sign_a, axis_b: sign_b}.get(dim, 0) for dim in range(3)
    )
    for (axis_a, axis_b) in ((0, 1), (0, 2), (1, 2))
    for sign_a in (1, -1)
    for sign_b in (1, -1)
)

#: ``(1, 1)``-norm class of every D3Q19 neighbour, keyed by ``|offset|_1``.
_AXIS_WEIGHT = 1.0 / 3.0
_DIAGONAL_WEIGHT = 1.0 / 6.0
_CENTER_WEIGHT = -4.0


def _validate_like(field: torch.Tensor, boundary: BoundaryPolicy,
                   solid_mask: torch.Tensor | None) -> None:
    if field.ndim != 3:
        raise ValueError("isotropic phase-field operators require a 3-D scalar tensor shaped (z, y, x)")
    if boundary not in ("periodic", "no_flux"):
        raise ValueError("boundary must be either 'periodic' or 'no_flux'")
    if solid_mask is not None and (
        solid_mask.shape != field.shape or solid_mask.dtype != torch.bool
    ):
        raise ValueError("solid_mask must be a bool tensor with the same (z, y, x) shape")


def _diag_neighbor(
    field: torch.Tensor,
    off: tuple[int, int, int],
    boundary: BoundaryPolicy,
    solid_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Fetch the face-diagonal neighbour ``field[z+off[0], y+off[1], x+off[2]]``.

    Composing :func:`tensorlbm.phasefield.operators._neighbor` twice keeps the
    boundary policy identical to the seven-point operators: with ``periodic``
    the two rolls wrap, and with ``no_flux`` / ``solid_mask`` each hop mirrors
    the invalid source, so a diagonal never reaches across a wall.
    """
    out = field
    for dim, step in enumerate(off):
        if step:
            out = _neighbor(out, dim, step, boundary, solid_mask)
    return out


def isotropic_laplacian_3d(
    field: torch.Tensor,
    *,
    boundary: BoundaryPolicy,
    solid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the D3Q19-isotropic nine-point-class Laplacian of ``(z, y, x)`` data.

    ``lap = (1/3) * sum_{6 axis} phi + (1/6) * sum_{12 face diagonal} phi - 4 phi``

    The weights sum to zero (so constants are in the null space) and both the
    second and the fourth isotropic moments are matched, which removes the
    angular error of the seven-point stencil.
    """
    _validate_like(field, boundary, solid_mask)
    result = _CENTER_WEIGHT * field
    for dim in range(3):
        for direction in (1, -1):
            result = result + _AXIS_WEIGHT * _neighbor(field, dim, direction, boundary, solid_mask)
    for off in FACE_DIAGONAL_OFFSETS:
        result = result + _DIAGONAL_WEIGHT * _diag_neighbor(field, off, boundary, solid_mask)
    if solid_mask is not None:
        result = torch.where(
            solid_mask, torch.zeros((), dtype=field.dtype, device=field.device), result
        )
    return result


def isotropic_gradient_3d(
    field: torch.Tensor,
    *,
    boundary: BoundaryPolicy,
    solid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the D3Q19-isotropic gradients ``(d/dx, d/dy, d/dz)``.

    For the ``x`` component::

        (1/6) (phi_E - phi_W)
        + (1/12) [(phi_BE - phi_BW) + (phi_NE - phi_NW) + (phi_SE - phi_SW) + (phi_TE - phi_TW)]

    i.e. the axis difference plus the four face-diagonal differences that carry
    an ``x`` component; the ``y`` and ``z`` components follow by rotation.  Every
    neighbour enters with ``offset_component / (6 if |offset|_1 == 1 else 12)``,
    so the operator is buildable from the stencil alone.
    """
    _validate_like(field, boundary, solid_mask)
    zeros = torch.zeros((), dtype=field.dtype, device=field.device)
    grads: list[torch.Tensor] = []
    # Axis order of the tensors, and of the returned tuple's docstring, is
    # (z, y, x); the public return value keeps the (x, y, z) order of
    # tensorlbm.phasefield.operators.central_gradient_3d.
    for dim, axis_weight in ((2, 1.0 / 6.0), (1, 1.0 / 6.0), (0, 1.0 / 6.0)):
        acc = axis_weight * (
            _neighbor(field, dim, 1, boundary, solid_mask)
            - _neighbor(field, dim, -1, boundary, solid_mask)
        )
        for off in FACE_DIAGONAL_OFFSETS:
            component = off[dim]
            if component == 0:
                continue
            acc = acc + (1.0 / 12.0) * float(component) * _diag_neighbor(field, off, boundary, solid_mask)
        if solid_mask is not None:
            acc = torch.where(solid_mask, zeros, acc)
        grads.append(acc)
    return grads[0], grads[1], grads[2]


__all__ = [
    "FACE_DIAGONAL_OFFSETS",
    "isotropic_gradient_3d",
    "isotropic_laplacian_3d",
]