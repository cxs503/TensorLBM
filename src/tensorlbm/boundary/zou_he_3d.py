"""Zou/He (1997) channel inlet/outlet reconstruction, D3Q19 and D3Q27.

Anchor: ``XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTION`` (the
``apply_boundary_conditions`` inlet/outlet reconstruction of
loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow), re-derived from the
primary source instead of transcribed.

What this module provides
-------------------------
On an ``x``-normal face, ``Q`` populations are split into an unknown set ``U``
(the ones that would stream in from outside the domain, all sharing the same
``c_x = s``, ``s = +-1``) and a known set ``K``.  Two driving modes:

* **velocity mode** -- prescribe ``u_x``, ``u_y``, ``u_z``; the density follows
  from mass + normal momentum::

      rho = (sum_K f - s * sum_K c_x f) / (1 - s * u_x)

* **pressure mode** -- prescribe ``rho``; the normal velocity follows from the
  same balance::

      M  = rho - sum_K f          (unknown-set mass)
      u_x = (sum_K c_x f + s * M) / rho

The closure for the unknown populations is the part that the previous
implementation got wrong, so it is spelled out:

1. **Normal link, non-equilibrium bounce-back** (Zou & He 1997 sec. II)::

       f_n = f_opp(n) + f^eq_n - f^eq_opp(n)

   This is the only direction for which bounce-back is physical; the
   tangential directions are *not* obtained this way.
2. **Remaining mass** ``R = M - f_n`` is split over the other unknown
   directions in proportion to their equilibrium values (the least-assumption
   split: it introduces no non-equilibrium of its own).
3. **Tangential momentum is then imposed exactly** by a minimum-norm
   correction ``delta`` that leaves the mass unchanged::

       delta = B @ (0, rho u_y - sum c_y f, rho u_z - sum c_z f),
       B = A^T (A A^T)^{-1},  A = [1; c_y; c_z] on the free directions.

   ``B`` is a fixed matrix depending only on the lattice, computed once.

Why step 3 matters (this is the defect this module fixes)
--------------------------------------------------------
Rebuilding **every** unknown population from its own opposite ("NEBB for all")
does satisfy mass and the normal momentum exactly, and therefore *looks* right
in a channel where ``u_y = u_z = 0``.  It does **not** impose the prescribed
transverse velocity: for D3Q19 it injects exactly ``rho u_y / 3``, so an
angled inflow arrives with one third of the requested transverse momentum
(measured, see ``tests/test_zou_he_3d_inlet_outlet.py``).  The 2D module
:mod:`tensorlbm.boundary.boundaries` already closes ``u_y`` from mass +
``y``-momentum; step 3 is the 3D generalisation of that.

Field (per-cell) inputs
-----------------------
``u_x``/``u_y``/``u_z`` (velocity mode) and ``rho`` (pressure mode) may be
scalars or ``(nz, ny)`` fields, so a fully developed parabolic inflow can be
imposed row by row.  This is what makes the closure usable as a *channel*
inlet/outlet rather than a plug-flow device.

Closure assumptions that remain (documented, not hidden)
-------------------------------------------------------
* A prescribed *pressure* face leaves the transverse velocity undetermined;
  it defaults to zero unless supplied.  That is a modelling assumption.
* The positivity guard (``_clip_positive``) only activates for extreme
  transverse velocities, where it trades exactness for stability.

References
----------
Zou, Q. & He, X. (1997) *Phys. Fluids* 9(6) 1592-1603, sec. II (velocity mode)
and sec. IV (pressure mode).
Krueger, T. et al. (2017) *The Lattice Boltzmann Method: Principles and
Practice*, sec. 8.2.2-8.2.4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..input_contract import check_positive_scalar

_SPECS: dict[tuple[str, int], "FaceSpec"] = {}


@dataclass(frozen=True)
class FaceSpec:
    """Index bookkeeping for one lattice's ``x``-normal face."""

    name: str
    Q: int
    column: int  # 0 for the west face, -1 for the east face
    s: int  # c_x of the unknown set (+1 west, -1 east)
    unknown: tuple[int, ...]
    normal: int
    normal_opp: int
    free: tuple[int, ...]  # unknown minus the normal direction
    cy: torch.Tensor
    cz: torch.Tensor
    cx: torch.Tensor
    projector: torch.Tensor  # (len(free), 3): minimum-norm mass/momentum correction

    def to(self, dtype: torch.dtype, device: torch.device) -> "FaceSpec":
        if self.cy.dtype == dtype and self.cy.device == device:
            return self
        return FaceSpec(
            name=self.name,
            Q=self.Q,
            column=self.column,
            s=self.s,
            unknown=self.unknown,
            normal=self.normal,
            normal_opp=self.normal_opp,
            free=self.free,
            cy=self.cy.to(dtype=dtype, device=device),
            cz=self.cz.to(dtype=dtype, device=device),
            cx=self.cx.to(dtype=dtype, device=device),
            projector=self.projector.to(dtype=dtype, device=device),
        )


def _build_spec(name: str, column: int, s: int, lattice_c: torch.Tensor) -> FaceSpec:
    C = lattice_c.to(torch.int64).cpu().numpy()
    Q = C.shape[0]
    unknown = tuple(int(k) for k in range(Q) if int(C[k, 0]) == s)
    normal = next(k for k in range(Q) if C[k, 0] == s and C[k, 1] == 0 and C[k, 2] == 0)
    normal_opp = next(k for k in range(Q) if C[k, 0] == -s and C[k, 1] == 0 and C[k, 2] == 0)
    free = tuple(k for k in unknown if k != normal)
    cy_full = torch.tensor([float(C[k, 1]) for k in range(Q)], dtype=torch.float64)
    cz_full = torch.tensor([float(C[k, 2]) for k in range(Q)], dtype=torch.float64)
    cx_full = torch.tensor([float(C[k, 0]) for k in range(Q)], dtype=torch.float64)
    # Minimum-norm correction: keep mass (row of ones) fixed while meeting the
    # two tangential momentum residuals exactly.
    A = np.stack(
        [
            np.ones(len(free)),
            np.array([C[k, 1] for k in free], dtype=np.float64),
            np.array([C[k, 2] for k in free], dtype=np.float64),
        ]
    )  # (3, n_free)
    AAT = A @ A.T
    if abs(np.linalg.det(AAT)) < 1e-12:  # pragma: no cover - lattice invariant
        raise RuntimeError(f"degenerate face closure for {name}")
    projector = torch.tensor(A.T @ np.linalg.inv(AAT), dtype=torch.float64)
    spec = FaceSpec(
        name=name,
        Q=Q,
        column=column,
        s=s,
        unknown=unknown,
        normal=normal,
        normal_opp=normal_opp,
        free=free,
        cy=cy_full,
        cz=cz_full,
        cx=cx_full,
        projector=projector,
    )
    _SPECS[(name, Q)] = spec
    return spec


def west_spec(lattice_c: torch.Tensor) -> FaceSpec:
    """Face spec for the inlet face ``x = 0`` (unknown set has ``c_x = +1``)."""
    key = ("west", int(lattice_c.shape[0]))
    return _SPECS.get(key) or _build_spec("west", 0, +1, lattice_c)


def east_spec(lattice_c: torch.Tensor) -> FaceSpec:
    """Face spec for the outlet face ``x = nx-1`` (unknown set has ``c_x = -1``)."""
    key = ("east", int(lattice_c.shape[0]))
    return _SPECS.get(key) or _build_spec("east", -1, -1, lattice_c)


def plane_field(
    value: float | torch.Tensor,
    ref: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Broadcast a scalar or per-cell face field to ``ref``'s shape.

    Args:
        value: scalar, ``(nz, ny)`` or ``(nz, ny, 1)`` tensor.
        ref: ``(nz, ny)`` reference tensor supplying dtype/device.
        name: label used in error messages.

    Returns:
        ``(nz, ny)`` tensor.
    """
    if isinstance(value, torch.Tensor):
        field = value.to(dtype=ref.dtype, device=ref.device)
        if field.ndim == 3 and field.shape[-1] == 1:
            field = field[..., 0]
        if (
            field.ndim == 2
            and field.shape[0] == ref.shape[0]
            and field.shape[1] == ref.shape[1] - 2
        ):
            # fluid-row profile (the two solid rows are implicit zeros / never used)
            embedded = torch.zeros_like(ref)
            embedded[:, 1:-1] = field
            field = embedded
        if field.shape != ref.shape:
            raise ValueError(
                f"{name} must have shape {tuple(ref.shape)}, (nz, ny, 1) or the "
                f"fluid-row profile (nz, ny-2); got {tuple(value.shape)}"
            )
        if not bool(torch.isfinite(field).all()):
            raise ValueError(f"{name} contains non-finite entries")
        return field
    out = torch.full_like(ref, float(value))
    if not bool(torch.isfinite(out).all()):
        raise ValueError(f"{name} is not finite ({value!r})")
    return out


def _check_density(rho: float | torch.Tensor, ref: torch.Tensor, name: str,
                   where: str) -> torch.Tensor:
    if isinstance(rho, torch.Tensor):
        field = plane_field(rho, ref, name)
        if not bool((field > 0.0).all()):
            raise ValueError(f"{name} field must be strictly positive")
        return field
    return torch.full_like(ref, check_positive_scalar(rho, name, where=where))


def _equilibrium_plane(equilibrium, rho, ux, uy, uz) -> torch.Tensor:
    """Equilibrium on a single ``x``-plane, shaped ``(Q, nz, ny, 1)``."""
    return equilibrium(
        rho.unsqueeze(-1),
        ux.unsqueeze(-1),
        uy.unsqueeze(-1),
        uz.unsqueeze(-1),
        device=rho.device,
    )


def _clip_positive(values: torch.Tensor) -> torch.Tensor:
    """Positivity guard for the reconstructed populations (rarely active)."""
    return torch.clamp(values, min=0.0)


def face_closure(
    f: torch.Tensor,
    *,
    spec: FaceSpec,
    equilibrium,
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
) -> torch.Tensor:
    """Rebuild the unknown populations of one ``x``-face at fixed moments.

    ``rho``/``ux`` must already be consistent with the known set (see
    :func:`opening_moments`); this routine then enforces the full moment set
    exactly, with non-equilibrium bounce-back on the normal link only.

    Args:
        f: populations ``(Q, nz, ny, nx)``.
        spec: face index bookkeeping.
        equilibrium: ``equilibrium3d``/``equilibrium27``-style callable.
        rho, ux, uy, uz: target moments, each ``(nz, ny)``.

    Returns:
        New population tensor (input untouched).
    """
    spec = spec.to(f.dtype, f.device)
    plane = f[:, :, :, spec.column]
    unknown = list(spec.unknown)
    known = [k for k in range(spec.Q) if k not in spec.unknown]
    known_sum = plane[known].sum(dim=0)
    known_cx = (plane[known] * spec.cx[known].view(-1, 1, 1)).sum(dim=0)
    known_cy = (plane[known] * spec.cy[known].view(-1, 1, 1)).sum(dim=0)
    known_cz = (plane[known] * spec.cz[known].view(-1, 1, 1)).sum(dim=0)

    feq = _equilibrium_plane(equilibrium, rho, ux, uy, uz)[:, :, :, 0].to(f.dtype)
    # (1) non-equilibrium bounce-back on the normal link
    f_normal = plane[spec.normal_opp] + feq[spec.normal] - feq[spec.normal_opp]
    # (2) remaining unknown-set mass split in proportion to equilibrium values
    free = list(spec.free)
    remaining = rho - known_sum - f_normal
    feq_free = feq[free].clamp_min(0.0)
    denom = feq_free.sum(dim=0, keepdim=True)
    weights = torch.where(denom > 0.0, feq_free / denom, torch.full_like(feq_free, 1.0 / len(free)))
    base = remaining.unsqueeze(0) * weights
    # (3) minimum-norm correction that imposes the tangential momentum exactly
    residual_y = rho * uy - known_cy - (base * spec.cy[free].view(-1, 1, 1)).sum(dim=0)
    residual_z = rho * uz - known_cz - (base * spec.cz[free].view(-1, 1, 1)).sum(dim=0)
    zero = torch.zeros_like(residual_y)
    r = torch.stack([zero, residual_y, residual_z])  # (3, nz, ny)
    delta = torch.einsum(
        "ij,j...->i...", spec.projector, r.to(spec.projector.dtype)
    ).to(f.dtype)
    values = _clip_positive(base + delta)

    f_new = f.clone()
    f_new[spec.normal, :, :, spec.column] = f_normal
    for idx, k in enumerate(free):
        f_new[k, :, :, spec.column] = values[idx]
    return f_new


def opening_moments(
    f: torch.Tensor,
    *,
    spec: FaceSpec,
    mode: str,
    u: torch.Tensor | None = None,
    rho: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Moments implied by the mass + normal-momentum balance on the face.

    Args:
        f: populations ``(Q, nz, ny, nx)``.
        spec: face bookkeeping.
        mode: ``"velocity"`` (``u`` = u_x given, density implied) or
            ``"pressure"`` (``rho`` given, u_x implied).
        u: prescribed normal velocity (velocity mode).
        rho: prescribed density (pressure mode).

    Returns:
        ``(rho, ux)``.
    """
    plane = f[:, :, :, spec.column]
    known = [k for k in range(spec.Q) if k not in spec.unknown]
    known_sum = plane[known].sum(dim=0)
    known_cx = (plane[known] * spec.cx[known].view(-1, 1, 1)).sum(dim=0)
    if mode == "velocity":
        assert u is not None
        return (known_sum - spec.s * known_cx) / (1.0 - spec.s * u), u
    if mode == "pressure":
        assert rho is not None
        mass_unknown = rho - known_sum
        return rho, (known_cx + spec.s * mass_unknown) / rho
    raise ValueError(f"unknown mode {mode!r}")

# ---------------------------------------------------------------------------
# Public closures (D3Q19; the D3Q27 twins live in ``zou_he_3d27``)
# ---------------------------------------------------------------------------

from tensorlbm.d3q19 import C as _C19  # noqa: E402
from tensorlbm.d3q19 import equilibrium3d as _equilibrium3d  # noqa: E402


def _apply_face(
    f: torch.Tensor,
    *,
    column: int,
    mode: str,
    lattice_c: torch.Tensor,
    equilibrium,
    rho_name: str,
    u: float | torch.Tensor | None = None,
    rho: float | torch.Tensor | None = None,
    uy: float | torch.Tensor = 0.0,
    uz: float | torch.Tensor = 0.0,
    where: str,
) -> torch.Tensor:
    """Shared entry point for the four public closures."""
    if f.ndim != 4 or f.shape[0] != int(lattice_c.shape[0]):
        raise ValueError(
            f"f must have shape ({int(lattice_c.shape[0])}, nz, ny, nx); got {tuple(f.shape)}"
        )
    spec = west_spec(lattice_c) if column == 0 else east_spec(lattice_c)
    ref = f[0, :, :, column]
    uy_f = plane_field(uy, ref, "uy")
    uz_f = plane_field(uz, ref, "uz")
    if mode == "velocity":
        assert u is not None
        u_f = plane_field(u, ref, "u_in")
        if not bool((u_f.abs() < 1.0).all()):
            raise ValueError("u_in must satisfy |u_in| < 1 for a Zou/He velocity face")
        rho_f, ux_f = opening_moments(f, spec=spec, mode="velocity", u=u_f)
    else:
        assert rho is not None
        rho_f = _check_density(rho, ref, rho_name, where)
        rho_f, ux_f = opening_moments(f, spec=spec, mode="pressure", rho=rho_f)
    return face_closure(
        f, spec=spec, equilibrium=equilibrium, rho=rho_f, ux=ux_f, uy=uy_f, uz=uz_f
    )


def zou_he_inlet_velocity_3d(
    f: torch.Tensor,
    u_in: float | torch.Tensor,
    uy_in: float | torch.Tensor = 0.0,
    uz_in: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Zou/He velocity inlet on the west face ``x = 0`` (D3Q19).

    Prescribes ``ux``, ``uy``, ``uz`` on every cell of the plane (scalars or
    ``(nz, ny)`` profiles) and implies ``rho`` from mass + normal momentum.
    All three velocity components are reproduced exactly -- the transverse
    components are *not* approximate (see the module docstring for the defect
    this replaced).

    Returns:
        New population tensor; the input is left untouched.
    """
    return _apply_face(
        f, column=0, mode="velocity", lattice_c=_C19, equilibrium=_equilibrium3d,
        rho_name="rho_in", u=u_in, uy=uy_in, uz=uz_in, where="zou_he_inlet_velocity_3d",
    )


def zou_he_inlet_pressure_3d(
    f: torch.Tensor,
    rho_in: float | torch.Tensor = 1.0,
    uy_in: float | torch.Tensor = 0.0,
    uz_in: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Zou/He pressure inlet on the west face ``x = 0`` (D3Q19).

    Prescribes ``rho`` (scalar or field) and implies ``ux``; the transverse
    velocity is the documented closure assumption (zero unless supplied).
    Needed to drive a channel *by a pressure difference* without a body force.

    Returns:
        New population tensor; the input is left untouched.
    """
    return _apply_face(
        f, column=0, mode="pressure", lattice_c=_C19, equilibrium=_equilibrium3d,
        rho_name="rho_in", rho=rho_in, uy=uy_in, uz=uz_in, where="zou_he_inlet_pressure_3d",
    )


def zou_he_outlet_pressure_3d(
    f: torch.Tensor,
    rho_out: float | torch.Tensor = 1.0,
    uy_out: float | torch.Tensor = 0.0,
    uz_out: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Zou/He pressure outlet on the east face ``x = nx-1`` (D3Q19).

    Mirror of :func:`zou_he_inlet_pressure_3d`.

    Returns:
        New population tensor; the input is left untouched.
    """
    return _apply_face(
        f, column=-1, mode="pressure", lattice_c=_C19, equilibrium=_equilibrium3d,
        rho_name="rho_out", rho=rho_out, uy=uy_out, uz=uz_out,
        where="zou_he_outlet_pressure_3d",
    )


def zou_he_outlet_velocity_3d(
    f: torch.Tensor,
    u_out: float | torch.Tensor,
    uy_out: float | torch.Tensor = 0.0,
    uz_out: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Zou/He velocity outlet on the east face ``x = nx-1`` (D3Q19).

    Included for completeness of the west/east x velocity/pressure square;
    a velocity outlet is rarely the right physical choice (it over-constrains
    an outflow), so prefer :func:`zou_he_outlet_pressure_3d`.

    Returns:
        New population tensor; the input is left untouched.
    """
    return _apply_face(
        f, column=-1, mode="velocity", lattice_c=_C19, equilibrium=_equilibrium3d,
        rho_name="rho_out", u=u_out, uy=uy_out, uz=uz_out, where="zou_he_outlet_velocity_3d",
    )
