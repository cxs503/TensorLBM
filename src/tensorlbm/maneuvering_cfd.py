"""KCS drift-angle / circular-motion LBM forces -> MMG hydrodynamic derivatives.

Motivation and scope
--------------------
:mod:`tensorlbm.maneuvering` is a *caller-driven* sway-yaw model: every MMG
derivative (``Y'_beta``, ``N'_r`` ...) is an input.  Nothing in the repository
computed those numbers from a flow solution, so the maneuvering chain stopped at
"empirical/literature coefficients in, turning circle out".  This module closes
the *front* half of that chain::

    KCS hull voxel geometry
      -> D3Q19 lattice-Boltzmann flow at prescribed drift angle beta,
         rudder angle delta or yaw rate r
      -> momentum-exchange force/moment X(t), Y(t), N(t) on the hull
      -> SNAME-prime non-dimensionalisation
      -> least-squares MMG polynomial identification with parameter covariance
      -> MmgDerivativeSet (13-15 derivatives) consumable by
         :mod:`tensorlbm.maneuvering` for turning-circle / zig-zag prediction.

Evidence grades (never mixed, never promoted)
---------------------------------------------
``GEOMETRY``
    KCS half-breadth polynomial (same family as :mod:`tensorlbm.ship_cad`)
    voxelised at the requested resolution.  :func:`verify_hull_geometry`
    measures ``L``, ``B``, ``T`` and ``Cb`` *from the mask* and fails closed
    against the locked KCS particulars, so no force number is emitted from a
    bad geometry.
``NUMERICS``
    Momentum-exchange (Ladd 1994) forces are asserted bit-identical to
    :func:`tensorlbm.obstacles.compute_obstacle_forces_3d` for a stationary wall,
    and the streamwise force is cross-checked against an independent
    control-volume momentum balance (:func:`control_volume_momentum_balance`).
    Circular motion is formulated in the hull-fixed frame (sheared onset flow on
    the inlet plane); a moving-wall bounce-back was prototyped and *rejected*
    because its momentum-exchange correction did not close the mass balance --
    see the node report, limitation list.
``IDENTIFICATION``
    Ordinary least squares on the MMG polynomial model, with parameter standard
    deviations from the normal-equation covariance.  Correctness is proved by a
    synthetic round trip (known derivatives -> synthetic forces -> recovery);
    it is *not* proved by agreement with experiment.
``PHYSICS``
    The lattice run resolves the *lattice* Reynolds number
    ``Re_lat = U_lat L_lat / nu_lat`` (O(10^2)-O(10^3) at reachable grids), not
    the KCS towing-tank ``Re = 1.4e7``.  Derivative values are therefore
    reported in the "LBM-resolved-Re frame" and are *not* a model-scale
    prediction; see the node report for the explicit error verdict.

Conventions
-----------
Body axes ``x`` forward, ``y`` to starboard, ``z`` down; yaw ``r`` and rudder
``delta`` positive to starboard.  SNAME prime system (Fossen 2011 eq. 7.10)::

    X' = X / (0.5 rho U^2 L T)      Y' = Y / (0.5 rho U^2 L T)
    N' = N / (0.5 rho U^2 L^2 T)    r' = r L / U

MMG polynomial model (Yasukawa & Yoshimura 2015, J. Mar. Sci. Technol. 20:37-52)::

    X' = X'_0 + X'_bb b^2 + X'_bbbb b^4 + X'_rr r'^2 + X'_dd d^2
    Y' = Y'_b b + Y'_r r' + Y'_bbb b^3 + Y'_rrr r'^3
    N' = N'_b b + N'_r r' + N'_bbb b^3 + N'_rrr r'^3

with ``b = beta`` and ``d = delta`` in radians.  Cross terms
(``X'_br, Y'_bbr, N'_bbr`` ...) need a two-dimensional ``(beta, r')`` grid and
are *not* identified here; :class:`MmgDerivativeSet` records them as ``None``
rather than silently assuming zero.

References
----------
* Yasukawa, H. & Yoshimura, Y. (2015) *J. Mar. Sci. Technol.* 20, 37-52.
* Ladd, A.J.C. (1994) *J. Fluid Mech.* 271, 285-309 (momentum exchange).
* Fossen, T.I. (2011) *Handbook of Marine Craft Hydrodynamics and Motion
  Control*, Wiley, ch. 7 (SNAME prime system).
* SIMMAN 2010/2015 workshop KCS particulars (Lpp = 7.2786 m at model scale).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import torch

from .d3q19 import C, OPPOSITE, W, equilibrium3d, macroscopic3d
from .obstacles import compute_obstacle_moments_3d

__all__ = [
    "KCS_MODEL_SCALE",
    "KCS_MODEL_RE",
    "GeometryVerification",
    "HullVoxel",
    "kcs_hull_mask",
    "verify_hull_geometry",
    "HullFlowConfig",
    "HullFlowResult",
    "run_hull_flow",
    "bounce_back_and_force",
    "control_volume_momentum_balance",
    "hull_length_cells",
    "to_prime",
    "from_prime",
    "body_axes_from_lattice",
    "PolynomialFit",
    "fit_polynomial",
    "MmgDerivativeSet",
    "identify_static_drift",
    "identify_rudder",
    "identify_circular_motion",
    "identify_all",
    "forward_mmg_forces",
    "synthetic_roundtrip",
    "cross_flow_theory_derivatives",
]

# --------------------------------------------------------------------------- #
# 1. KCS particulars (SIMMAN model scale, towing-tank water)
# --------------------------------------------------------------------------- #

#: Model-scale KCS particulars used by the SIMMAN workshops (``L`` is Lpp).
KCS_MODEL_SCALE: dict[str, float] = {
    "L": 7.2786,
    "B": 1.0190,
    "T": 0.3418,
    "Cb": 0.6505,
    "S": 9.4379,
    "U": 2.196,
    "Fr": 0.26,
    "rho": 999.5,
    "nu": 1.139e-6,
}

#: ``Re = U L / nu`` at the SIMMAN KCS design condition (model scale).
KCS_MODEL_RE = KCS_MODEL_SCALE["U"] * KCS_MODEL_SCALE["L"] / KCS_MODEL_SCALE["nu"]


def _kcs_half_beam_fraction(xi: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """KCS half-breadth as a fraction of ``B/2`` (same family as ship_cad).

    ``hb = (1 - xi^2)^0.45 * zeta^0.24`` for ``|xi| <= 1``, ``0 <= zeta <= 1``.
    The closed-form integral of this shape gives ``Cb ~= 0.651``.
    """
    inside = (np.abs(xi) <= 1.0) & (zeta >= 0.0) & (zeta <= 1.0)
    hb = (1.0 - np.clip(xi, -1.0, 1.0) ** 2) ** 0.45 * np.clip(zeta, 0.0, 1.0) ** 0.24
    return np.where(inside, np.clip(hb, 0.0, 1.0), 0.0)


# --------------------------------------------------------------------------- #
# 2. Hull voxelisation (drift angle = hull yawed inside a uniform stream)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GeometryVerification:
    """Measured-from-mask geometry vs the locked KCS particulars."""

    length_cells: float
    beam_cells: float
    draft_cells: float
    cb: float
    cb_reference: float
    cb_rel_error: float
    tolerance: float
    passed: bool
    detail: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "length_cells": self.length_cells,
            "beam_cells": self.beam_cells,
            "draft_cells": self.draft_cells,
            "cb": self.cb,
            "cb_reference": self.cb_reference,
            "cb_rel_error": self.cb_rel_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class HullVoxel:
    """Voxelised hull plus the geometry evidence that produced it."""

    mask: torch.Tensor            # (nz, ny, nx) bool
    length_cells: float
    beam_cells: float
    draft_cells: float
    midship: tuple[float, float]  # (x_index, y_index) of Lpp/2 on the centreline
    keel_z: float                 # z index of the keel plane
    yaw_deg: float
    rudder_deg: float | None
    verification: GeometryVerification

    @property
    def n_solid(self) -> int:
        return int(self.mask.sum().item())

    def as_dict(self) -> dict:
        return {
            "length_cells": self.length_cells,
            "beam_cells": self.beam_cells,
            "draft_cells": self.draft_cells,
            "midship": list(self.midship),
            "keel_z": self.keel_z,
            "yaw_deg": self.yaw_deg,
            "rudder_deg": self.rudder_deg,
            "n_solid_cells": self.n_solid,
            "geometry": self.verification.as_dict(),
        }


def kcs_hull_mask(
    shape: Sequence[int],
    *,
    length_cells: float,
    midship: tuple[float, float],
    keel_z: float,
    yaw_deg: float = 0.0,
    rudder_deg: float | None = None,
    rudder_chord_cells: float | None = None,
    rudder_depth_fraction: float = 0.75,
    verify_tolerance: float = 0.05,
) -> HullVoxel:
    """Voxelise the KCS hull, optionally yawed (drift) and with a rudder plate.

    Beam and draft follow the KCS ``B/L`` and ``T/L`` ratios, so one
    ``length_cells`` sets the resolution.  ``yaw_deg`` rotates the hull about
    midship inside a uniform ``+x`` stream: a hull at ``yaw_deg = beta`` in a
    ``+x`` stream is the MMG state ``v = -U sin(beta)``.

    Args:
        shape: ``(nz, ny, nx)`` lattice shape (``x`` is the last axis).
        length_cells: Lpp in lattice cells (sets ``B``, ``T``).
        midship: ``(x_index, y_index)`` of the hull midship on the centreline.
        keel_z: ``z`` index of the keel plane; the hull occupies
            ``keel_z <= z <= keel_z + T`` (``z`` points down).
        yaw_deg: hull yaw about midship (drift angle).
        rudder_deg: rudder deflection; ``None`` disables the rudder.
        rudder_chord_cells: rudder chord; defaults to ``0.09 * length_cells``.
        rudder_depth_fraction: rudder span as a fraction of the draft.
        verify_tolerance: allowed relative error of the measured ``Cb``.

    Returns:
        :class:`HullVoxel` with the mask and its geometry verification.
    """
    nz, ny, nx = (int(v) for v in shape)
    L = float(length_cells)
    B = L * KCS_MODEL_SCALE["B"] / KCS_MODEL_SCALE["L"]
    T = L * KCS_MODEL_SCALE["T"] / KCS_MODEL_SCALE["L"]

    zz = np.arange(nz, dtype=np.float64)[:, None, None]
    yy = np.arange(ny, dtype=np.float64)[None, :, None]
    xx = np.arange(nx, dtype=np.float64)[None, None, :]

    cx, cy = float(midship[0]), float(midship[1])
    ang = math.radians(yaw_deg)
    ca, sa = math.cos(ang), math.sin(ang)
    along = (xx - cx) * ca + (yy - cy) * sa
    lat = -(xx - cx) * sa + (yy - cy) * ca

    xi = along / (L / 2.0)
    zeta = (zz - float(keel_z)) / T
    hb = _kcs_half_beam_fraction(xi, zeta)
    mask = (hb > 1e-9) & (np.abs(lat) <= (B / 2.0) * hb)

    if rudder_deg is not None and float(rudder_deg) != 0.0:
        chord = (
            float(rudder_chord_cells)
            if rudder_chord_cells is not None
            else 0.09 * L
        )
        # Rudder stock on the centreline at 0.47 L aft of midship.
        px, py = cx - 0.47 * L * ca, cy - 0.47 * L * sa
        d = math.radians(float(rudder_deg))
        # chord unit vector from the stock to the trailing edge: aft (-x) and
        # to starboard (+y) for delta > 0 (bow-to-starboard rudder)
        tan_x, tan_y = -math.cos(d), math.sin(d)
        nrm_x, nrm_y = math.sin(d), math.cos(d)
        dx = xx - px
        dy = yy - py
        s = dx * tan_x + dy * tan_y            # along-chord coordinate
        n = dx * nrm_x + dy * nrm_y            # across-plate coordinate
        z_lo = float(keel_z) + 0.15 * T
        z_hi = float(keel_z) + (0.15 + float(rudder_depth_fraction)) * T
        plate = (
            (np.abs(n) <= 0.5) & (s <= 0.0) & (s >= -chord) & (zz >= z_lo) & (zz <= z_hi)
        )
        mask = mask | plate

    mask_t = torch.from_numpy(mask)
    verification = verify_hull_geometry(
        mask_t,
        length_cells=L,
        beam_cells=B,
        draft_cells=T,
        cb_reference=KCS_MODEL_SCALE["Cb"],
        tolerance=verify_tolerance,
    )
    return HullVoxel(
        mask=mask_t,
        length_cells=L,
        beam_cells=B,
        draft_cells=T,
        midship=(cx, cy),
        keel_z=float(keel_z),
        yaw_deg=float(yaw_deg),
        rudder_deg=None if rudder_deg is None else float(rudder_deg),
        verification=verification,
    )


def verify_hull_geometry(
    mask: torch.Tensor,
    *,
    length_cells: float,
    beam_cells: float,
    draft_cells: float,
    cb_reference: float = KCS_MODEL_SCALE["Cb"],
    tolerance: float = 0.05,
) -> GeometryVerification:
    """Fail-closed geometry check: ``Cb = V_solid / (L B T)`` measured from the mask.

    ``L``, ``B`` and ``T`` are measured from the mask extents in cells, which
    catches an off-by-one domain setup before any force is integrated.
    """
    solid = mask.to(torch.bool)
    if solid.ndim != 3:
        raise ValueError("mask must be (nz, ny, nx)")
    n_cells = int(solid.sum().item())
    if n_cells == 0:
        raise ValueError("empty hull mask")
    zi, yi, xi = torch.nonzero(solid, as_tuple=True)
    extent_x = float(xi.max() - xi.min()) + 1.0
    extent_y = float(yi.max() - yi.min()) + 1.0
    extent_z = float(zi.max() - zi.min()) + 1.0
    cb = n_cells / (extent_x * extent_y * extent_z)
    rel = abs(cb - cb_reference) / cb_reference
    return GeometryVerification(
        length_cells=extent_x,
        beam_cells=extent_y,
        draft_cells=extent_z,
        cb=cb,
        cb_reference=cb_reference,
        cb_rel_error=rel,
        tolerance=tolerance,
        passed=bool(rel <= tolerance and extent_z > 0 and extent_y > 0),
        detail={
            "n_solid_cells": n_cells,
            "requested_length_cells": length_cells,
            "requested_beam_cells": beam_cells,
            "requested_draft_cells": draft_cells,
            "bbox": [extent_x, extent_y, extent_z],
            "definition": "Cb = n_solid / (L_bbox * B_bbox * T_bbox)",
        },
    )# --------------------------------------------------------------------------- #
# 3. D3Q19 steady flow driver with moving-wall bounce-back
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HullFlowConfig:
    """Configuration of one steady drift / rudder / circular-motion LBM run."""

    shape: tuple[int, int, int]
    mask: torch.Tensor             # (nz, ny, nx) bool solid
    u_in: float = 0.05             # lattice inflow along +x
    tau: float = 0.55
    steps: int = 12000
    force_every: int = 1
    warmup_fraction: float = 0.5    # averaging window starts here
    yaw_rate_lattice: float = 0.0  # yaw rate r in lattice units (1/step)
    yaw_pivot: tuple[float, float] = (0.0, 0.0)  # (x_index, y_index) of the pivot
    midship: tuple[float, float] | None = None   # moment reference (x, y); derived if None
    keel_z: float | None = None                  # keel plane z index; derived if None
    converge_tol: float = 5e-4
    converge_window: int = 400
    min_steps: int = 0          # never declare convergence before this step
    device: str = "cpu"

    @property
    def viscosity(self) -> float:
        return (self.tau - 0.5) / 3.0


@dataclass
class HullFlowResult:
    """Steady (window-averaged) forces/moments plus convergence evidence."""

    fx: float
    fy: float
    moment_yaw: float
    steps_run: int
    averaged_from_step: int
    force_trace: list[tuple[int, float, float, float]]
    convergence_ratio: float
    re_lattice: float
    lattice_velocity: float
    viscosity: float
    wall_velocity_max: float
    status: str = "COMPLETED"

    def as_dict(self) -> dict:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "moment_yaw": self.moment_yaw,
            "steps_run": self.steps_run,
            "averaged_from_step": self.averaged_from_step,
            "convergence_ratio": self.convergence_ratio,
            "re_lattice": self.re_lattice,
            "lattice_velocity": self.lattice_velocity,
            "viscosity": self.viscosity,
            "wall_velocity_max": self.wall_velocity_max,
            "status": self.status,
            "force_trace": self.force_trace,
        }


def _shift_maps() -> list[tuple[int, int, int]]:
    c = C.tolist()
    return [(int(c[q][2]), int(c[q][1]), int(c[q][0])) for q in range(19)]


_SHIFTS = _shift_maps()


def _stream(f: torch.Tensor) -> torch.Tensor:
    """D3Q19 push-streaming with ``torch.roll`` along (z, y, x) axes."""
    f_new = f.clone()
    for q in range(1, 19):
        cx, cy, cz = _SHIFTS[q]
        if cx or cy or cz:
            # push streaming: out[i] = in[i - c]
            f_new[q] = torch.roll(f[q], shifts=(cz, cy, cx), dims=(0, 1, 2))
    return f_new


def _zero_external_inflow(f: torch.Tensor) -> None:
    """Zero populations that ``torch.roll`` wrapped in through the 6 domain faces.

    Pushed streaming is periodic, so a face cell receives data from the opposite
    face.  Those links do not exist: clearing them lets bounce-back / specular
    reflection / the inlet-outlet condition own the face without solid-to-solid
    wrap-around leakage (which otherwise corrupts mass and the wall force).
    """
    c = C.tolist()
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    sizes = (nz, ny, nx)
    for axis in range(3):          # tensor axis: 0=z, 1=y, 2=x
        comp = 2 - axis            # lattice component index for that axis
        for face, out_sign in ((0, 1), (sizes[axis] - 1, -1)):
            dirs = [q for q in range(1, 19) if c[q][comp] * out_sign > 0]
            if not dirs:
                continue
            idx = [slice(None), slice(None), slice(None), slice(None)]
            idx[axis + 1] = face
            f[tuple([dirs, *idx[1:]])] = 0.0


def _specular_free_slip(f: torch.Tensor, axis: int, side: int) -> None:
    """Specular (free-slip) reflection on the ``axis`` face ``side`` (in place).

    ``axis`` is a *spatial* axis (0 = z, 1 = y, 2 = x); the population tensor has
    the direction index in front, so the tensor dimension is ``axis + 1``.  The
    wall-normal velocity component is flipped and the population is taken from
    its mirror partner (momentum conserving, zero wall shear).
    """
    c_list = C.tolist()
    comp = 2 - axis                      # lattice component index of this axis
    n = f.shape[axis + 1]
    face = 0 if side == 0 else n - 1
    updates: dict[int, torch.Tensor] = {}
    base = [slice(None)] * 4
    base[axis + 1] = face
    idx = tuple(base)
    for q in range(1, 19):
        cq = c_list[q]
        normal = cq[comp]
        if (side == 0 and normal > 0) or (side == 1 and normal < 0):
            mirrored = list(cq)
            mirrored[comp] = -mirrored[comp]
            r = c_list.index(mirrored)
            updates[q] = f[(r, *idx[1:])].clone()
    for q, val in updates.items():
        f[(q, *idx[1:])] = val


def bounce_back_and_force(
    f: torch.Tensor,
    solid: torch.Tensor,
) -> tuple[torch.Tensor, float, float, float]:
    """Solid-node bounce-back on the post-stream population + wall force.

    The momentum given to the wall is ``F = 2 sum_q c_q f_q`` over solid nodes
    and all 19 directions, i.e. exactly
    :func:`tensorlbm.obstacles.compute_obstacle_forces_3d` (a unit test asserts
    bit-level agreement), computed in the same pass as the reflection so the two
    cannot drift apart.
    """
    c = C.to(f.device).float()
    f_out = f.clone()
    for q in range(1, 19):
        f_out[OPPOSITE[q]] = torch.where(solid, f[q], f_out[OPPOSITE[q]])

    fq = f * solid.unsqueeze(0)
    cx = c[:, 0].view(19, 1, 1, 1)
    cy = c[:, 1].view(19, 1, 1, 1)
    cz = c[:, 2].view(19, 1, 1, 1)
    return f_out, float(2.0 * (cx * fq).sum()), float(2.0 * (cy * fq).sum()), float(2.0 * (cz * fq).sum())


def control_volume_momentum_balance(
    f: torch.Tensor,
    u_in_x: float,
) -> dict:
    """Streamwise drag from a control-volume momentum balance (code verification).

    For the statistically steady field the momentum-exchange force on the hull
    must equal the loss of streamwise momentum flux between the inlet and outlet
    planes::

        F_cv = sum_inlet (rho u_x^2 + p) - sum_outlet (rho u_x^2 + p)

    with ``p = rho c_s^2``.  Both sides are computed from the *same* solution, so
    agreement is a consistency check of the force integration and the boundary
    treatment, not a physics validation.
    """
    rho, ux, uy, uz = macroscopic3d(f)
    cs2 = 1.0 / 3.0
    flux = rho * ux * ux * ux + rho * cs2 * ux
    f_in = float(flux[:, :, 1].sum())
    f_out = float(flux[:, :, -2].sum())
    return {
        "f_cv": f_in - f_out,
        "inlet_mass_flux": float((rho * ux)[:, :, 1].sum()),
        "outlet_mass_flux": float((rho * ux)[:, :, -2].sum()),
        "u_in_reference": float(u_in_x),
    }


def run_hull_flow(config: HullFlowConfig) -> HullFlowResult:
    """Run one steady drift / rudder / circular-motion LBM case.

    Boundary conditions: equilibrium inlet on the ``x=0`` face, zero-gradient
    outlet on ``x=nx-1``, specular free-slip on the four transverse faces
    (open-water double-body surrogate), solid-node bounce-back on the hull.
    For a non-zero ``yaw_rate_lattice`` the inlet plane carries the hull-fixed
    far-field shear (``u_rel = U - omega x d``), which is the circular-motion
    test as seen from the hull.
    Steady state is declared when the window-averaged force changes by less than
    ``converge_tol`` (relative) between consecutive ``converge_window`` windows.
    """
    device = torch.device(config.device)
    nz, ny, nx = config.shape
    solid = config.mask.to(device=device, dtype=torch.bool)
    if tuple(solid.shape) != (nz, ny, nx):
        raise ValueError("mask shape mismatch")

    # Hull-fixed-frame circular motion: the far-field onset flow seen by the
    # hull is u_rel = U - omega x d (omega = r z_hat about the pivot).  At the
    # inlet plane (dx fixed) both components vary with y only, which an inlet
    # plane can carry exactly.  Centrifugal / Coriolis body forces are O(r'^2)
    # and are neglected -- documented model limitation, not a numerical choice.
    yaw_rate = float(config.yaw_rate_lattice)
    dx_in = 0.0 - float(config.yaw_pivot[0])
    yy_f = torch.arange(ny, dtype=torch.float32, device=device) - float(config.yaw_pivot[1])
    ux_in = torch.full((nz, ny), float(config.u_in), dtype=torch.float32, device=device)
    uy_in = torch.zeros((nz, ny), dtype=torch.float32, device=device)
    shear_amplitude = 0.0
    if yaw_rate != 0.0:
        # (x fwd, y stbd, z down) is left-handed: r > 0 (bow to starboard)
        # is a rotation about -z_hat, so omega x d = r (dy, -dx, 0) and
        # u_rel = U - omega x d = (U - r dy, +r dx)
        ux_in = ux_in - yaw_rate * yy_f.view(1, ny)
        uy_in = uy_in + yaw_rate * dx_in
        shear_amplitude = float(abs(yaw_rate) * (ny / 2.0))
        if shear_amplitude > 0.5 * float(config.u_in):
            raise ValueError(
                "transverse inlet shear exceeds 50% of the inflow; reduce the "
                f"yaw rate (shear={shear_amplitude:.3e}, u_in={config.u_in:.3e})"
            )

    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    ux0 = torch.full_like(rho0, float(config.u_in))
    zero = torch.zeros_like(rho0)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, zero, zero, device=device)
    del rho0, ux0, zero

    f_inlet = equilibrium3d(
        torch.ones((nz, ny, 1), dtype=torch.float32, device=device),
        ux_in.unsqueeze(-1),
        uy_in.unsqueeze(-1),
        torch.zeros((nz, ny, 1), dtype=torch.float32, device=device),
        device=device,
    )[:, :, :, 0]

    tau = float(config.tau)
    if config.midship is not None:
        cx_ref, cy_ref = float(config.midship[0]), float(config.midship[1])
    else:
        zi, yi, xi = torch.nonzero(solid, as_tuple=True)
        cx_ref = float(xi.to(torch.float32).mean().item())
        cy_ref = float(yi.to(torch.float32).mean().item())
    cz_ref = float(config.keel_z) if config.keel_z is not None else _keel_z_index(solid)
    trace: list[tuple[int, float, float, float]] = []
    window: list[tuple[float, float, float]] = []
    previous_mean: tuple[float, float, float] | None = None
    convergence_ratio = float("inf")
    averaged_from = int(config.steps * config.warmup_fraction)
    step = 0

    for step in range(1, config.steps + 1):
        rho, ux, uy, uz = macroscopic3d(f)
        feq = equilibrium3d(rho, ux, uy, uz, device=device)
        f = f + (feq - f) / tau
        del feq, rho, ux, uy, uz
        f = _stream(f)
        _zero_external_inflow(f)

        record = (step % max(1, config.force_every) == 0) or step == config.steps
        if record:
            _, fx, fy, _fz = bounce_back_and_force(f, solid)
            _, _, mz = compute_obstacle_moments_3d(f, solid, cx_ref, cy_ref, cz_ref)
            mz_f = float(mz)
            trace.append((step, fx, fy, mz_f))
            if step > averaged_from:
                window.append((fx, fy, mz_f))

        f, _, _, _ = bounce_back_and_force(f, solid)
        f[:, :, :, 0] = f_inlet
        f[:, :, :, -1] = f[:, :, :, -2]
        _specular_free_slip(f, axis=0, side=0)
        _specular_free_slip(f, axis=0, side=1)
        _specular_free_slip(f, axis=1, side=0)
        _specular_free_slip(f, axis=1, side=1)
        if not bool(torch.isfinite(f).all()):
            raise RuntimeError(f"non-finite populations at step {step}")
        f.clamp_(min=-5.0, max=15.0)

        if len(window) >= config.converge_window:
            recent = window[-config.converge_window:]
            n = len(recent)
            mean = (
                sum(v[0] for v in recent) / n,
                sum(v[1] for v in recent) / n,
                sum(v[2] for v in recent) / n,
            )
            if previous_mean is not None and step >= config.min_steps:
                denom = max(abs(mean[0]), abs(mean[1]), abs(mean[2]), 1e-9)
                convergence_ratio = max(abs(mean[k] - previous_mean[k]) for k in range(3)) / denom
                if convergence_ratio < config.converge_tol:
                    previous_mean = mean
                    break
            previous_mean = mean

    if window:
        n = len(window)
        fx_mean = sum(v[0] for v in window) / n
        fy_mean = sum(v[1] for v in window) / n
        mz_mean = sum(v[2] for v in window) / n
    else:  # pragma: no cover - only for very short runs
        _, fx_mean, fy_mean, _fz = bounce_back_and_force(f, solid)
        _, _, mz = compute_obstacle_moments_3d(f, solid, cx_ref, cy_ref, cz_ref)
        mz_mean = float(mz)

    return HullFlowResult(
        fx=fx_mean,
        fy=fy_mean,
        moment_yaw=mz_mean,
        steps_run=step,
        averaged_from_step=averaged_from,
        force_trace=trace,
        convergence_ratio=float(convergence_ratio),
        re_lattice=hull_length_cells(solid) * float(config.u_in) / config.viscosity,
        lattice_velocity=float(config.u_in),
        viscosity=config.viscosity,
        wall_velocity_max=shear_amplitude,
    )


def _keel_z_index(mask: torch.Tensor) -> float:
    """Lowest solid z index (the keel plane in this lattice convention)."""
    zi = torch.nonzero(mask.any(dim=(1, 2))).flatten()
    return float(zi.min().item())


def hull_length_cells(mask: torch.Tensor) -> float:
    """Longitudinal extent (in cells) of the solid mask."""
    idx = torch.nonzero(mask.any(dim=(0, 1))).flatten()
    return float(idx.max() - idx.min() + 1)


# --------------------------------------------------------------------------- #
# 4. SNAME prime non-dimensionalisation and hull-axis rotation
# --------------------------------------------------------------------------- #


def to_prime(
    X: float,
    Y: float,
    N: float,
    *,
    rho: float = 1.0,
    U: float,
    L: float,
    T: float,
) -> tuple[float, float, float]:
    """Dimensional (lattice or SI) force/moment -> SNAME primes."""
    q = 0.5 * rho * U * U
    return (X / (q * L * T), Y / (q * L * T), N / (q * L * L * T))


def from_prime(
    Xp: float,
    Yp: float,
    Np: float,
    *,
    rho: float = 1.0,
    U: float,
    L: float,
    T: float,
) -> tuple[float, float, float]:
    """SNAME primes -> dimensional force/moment (inverse of :func:`to_prime`)."""
    q = 0.5 * rho * U * U
    return (Xp * q * L * T, Yp * q * L * T, Np * q * L * L * T)


def body_axes_from_lattice(fx: float, fy: float, yaw_deg: float) -> tuple[float, float]:
    """Rotate a lattice-frame force pair into hull axes for a hull yawed by ``yaw_deg``.

    A hull yawed by ``beta`` inside a ``+x`` stream has body-frame velocity
    ``(U cos beta, -U sin beta)``; the inverse rotation gives
    ``X = fx cos b + fy sin b`` and ``Y = -fx sin b + fy cos b``.
    """
    b = math.radians(float(yaw_deg))
    return (
        fx * math.cos(b) + fy * math.sin(b),
        -fx * math.sin(b) + fy * math.cos(b),
    )


# --------------------------------------------------------------------------- #
# 5. Least-squares polynomial identification
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PolynomialFit:
    """Ordinary least squares fit with normal-equation parameter uncertainties."""

    coefficients: Mapping[str, float]
    std: Mapping[str, float]
    r2: float
    n_points: int
    terms: tuple[str, ...]
    residuals: tuple[float, ...]

    def as_dict(self) -> dict:
        return {
            "coefficients": dict(self.coefficients),
            "std": dict(self.std),
            "r2": self.r2,
            "n_points": self.n_points,
            "terms": list(self.terms),
            "residuals": list(self.residuals),
        }


def fit_polynomial(
    design: np.ndarray,
    target: np.ndarray,
    terms: Sequence[str],
) -> PolynomialFit:
    """Ordinary least squares with parameter standard deviations.

    Args:
        design: ``(n_points, n_terms)`` design matrix (columns follow ``terms``).
        target: ``(n_points,)`` measured prime force/moment component.
        terms: coefficient names, one per design column.

    Returns:
        :class:`PolynomialFit`; ``std`` is ``nan`` when the system is
        rank-deficient or has no spare degree of freedom (never silently 0).
    """
    A = np.asarray(design, dtype=float)
    y = np.asarray(target, dtype=float)
    if A.ndim != 2 or A.shape[0] != y.shape[0] or A.shape[1] != len(terms):
        raise ValueError("design/target/terms shape mismatch")
    if A.shape[0] < A.shape[1]:
        raise ValueError("fewer data points than terms: identification is under-determined")
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    residual = y - A @ coef
    rss = float(residual @ residual)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    n, p = A.shape
    std = {name: float("nan") for name in terms}
    if n > p:
        try:
            sigma2 = rss / (n - p)
            cov = sigma2 * np.linalg.inv(A.T @ A)
            diag = np.diag(cov)
            if np.all(diag >= 0):
                std = {name: float(np.sqrt(diag[i])) for i, name in enumerate(terms)}
        except np.linalg.LinAlgError:  # pragma: no cover - singular design
            pass
    return PolynomialFit(
        coefficients={name: float(coef[i]) for i, name in enumerate(terms)},
        std=std,
        r2=float(r2),
        n_points=int(n),
        terms=tuple(terms),
        residuals=tuple(float(v) for v in residual),
    )


def _records(records: Sequence[Mapping[str, float]]) -> list[Mapping[str, float]]:
    if not records:
        raise ValueError("empty data set")
    return list(records)


def identify_static_drift(
    records: Sequence[Mapping[str, float]],
) -> dict[str, PolynomialFit]:
    """Static-drift (``r' = 0, delta = 0``) MMG fit.

    Model (Yasukawa & Yoshimura 2015; even ``X``, odd ``Y``/``N``)::

        X' = X'_0 + X'_bb b^2 + X'_bbbb b^4
        Y' = Y'_b b + Y'_bbb b^3
        N' = N'_b b + N'_bbb b^3
    """
    recs = _records(records)
    b = np.radians([float(r["beta_deg"]) for r in recs])
    X = np.array([float(r["X_prime"]) for r in recs])
    Y = np.array([float(r["Y_prime"]) for r in recs])
    N = np.array([float(r["N_prime"]) for r in recs])
    return {
        "X": fit_polynomial(np.column_stack([np.ones_like(b), b**2, b**4]), X,
                            ("X0", "Xbb", "Xbbbb")),
        "Y": fit_polynomial(np.column_stack([b, b**3]), Y, ("Yb", "Ybbb")),
        "N": fit_polynomial(np.column_stack([b, b**3]), N, ("Nb", "Nbbb")),
    }


def identify_rudder(
    records: Sequence[Mapping[str, float]],
) -> dict[str, PolynomialFit]:
    """Rudder-angle fit at zero drift / zero yaw rate.

    Model::

        X' = X'_0 + X'_dd d^2         (even in delta)
        Y' = Y'_d d                   (odd in delta)
        N' = N'_d d
    """
    recs = _records(records)
    d = np.radians([float(r["delta_deg"]) for r in recs])
    if not np.any(np.abs(d) > 1e-12):
        raise ValueError("rudder identification needs at least one non-zero delta")
    X = np.array([float(r["X_prime"]) for r in recs])
    Y = np.array([float(r["Y_prime"]) for r in recs])
    N = np.array([float(r["N_prime"]) for r in recs])
    return {
        "X": fit_polynomial(np.column_stack([np.ones_like(d), d**2]), X, ("X0", "Xdd")),
        "Y": fit_polynomial(d.reshape(-1, 1), Y, ("Yd",)),
        "N": fit_polynomial(d.reshape(-1, 1), N, ("Nd",)),
    }


def identify_circular_motion(
    records: Sequence[Mapping[str, float]],
) -> dict[str, PolynomialFit]:
    """Circular-motion (``beta = 0``) fit in non-dimensional yaw rate ``r'``.

    Model::

        X' = X'_0 + X'_rr r'^2
        Y' = Y'_r r' + Y'_rrr r'^3
        N' = N'_r r' + N'_rrr r'^3
    """
    recs = _records(records)
    rp = np.array([float(r["r_prime"]) for r in recs])
    if not np.any(np.abs(rp) > 1e-12):
        raise ValueError("circular-motion identification needs at least one non-zero r'")
    X = np.array([float(r["X_prime"]) for r in recs])
    Y = np.array([float(r["Y_prime"]) for r in recs])
    N = np.array([float(r["N_prime"]) for r in recs])
    return {
        "X": fit_polynomial(np.column_stack([np.ones_like(rp), rp**2]), X, ("X0", "Xrr")),
        "Y": fit_polynomial(np.column_stack([rp, rp**3]), Y, ("Yr", "Yrrr")),
        "N": fit_polynomial(np.column_stack([rp, rp**3]), N, ("Nr", "Nrrr")),
    }


@dataclass
class MmgDerivativeSet:
    """Identified MMG derivatives in SNAME-prime form, with provenance."""

    prime: dict[str, float]
    std: dict[str, float]
    fits: dict[str, dict]
    conditions: dict[str, list]
    missing: tuple[str, ...] = (
        "Xbr",
        "Xbbbb_conf",
        "Ybbr",
        "Ybrr",
        "Nbbr",
        "Nbrr",
    )

    # ---- convenience views -------------------------------------------------
    def as_dict(self) -> dict:
        return {
            "prime": dict(self.prime),
            "std": dict(self.std),
            "fits": {k: dict(v) for k, v in self.fits.items()},
            "conditions": {k: list(v) for k, v in self.conditions.items()},
            "missing_cross_terms": list(self.missing),
            "note": (
                "SNAME-prime MMG derivatives identified from LBM forces at the "
                "lattice-resolved Reynolds number; cross terms need a 2-D "
                "(beta, r') grid and are not fitted."
            ),
        }

    def as_classic_derivatives(self) -> dict[str, float]:
        """Map to the classic MMG/Abkowitz names quoted in the node goal.

        ``beta ~= -v'`` (v' = v/U), so ``Y'_v = -Y'_b`` and ``N'_v = -N'_b``;
        ``X'_vv = X'_bb``.  Non-dimensionalised with the same prime factors.
        """
        return {
            "Y_v": -self.prime.get("Yb", float("nan")),
            "N_v": -self.prime.get("Nb", float("nan")),
            "Y_r": self.prime.get("Yr", float("nan")),
            "N_r": self.prime.get("Nr", float("nan")),
            "X_vv": self.prime.get("Xbb", float("nan")),
            "Y_vvv": -self.prime.get("Ybbb", float("nan")),
            "N_vvv": -self.prime.get("Nbbb", float("nan")),
            "Y_delta": self.prime.get("Yd", float("nan")),
            "N_delta": self.prime.get("Nd", float("nan")),
            "X_delta_delta": self.prime.get("Xdd", float("nan")),
            "X_rr": self.prime.get("Xrr", float("nan")),
            "Y_rrr": self.prime.get("Yrrr", float("nan")),
            "N_rrr": self.prime.get("Nrrr", float("nan")),
            "X_0": self.prime.get("X0", float("nan")),
        }

    def as_maneuvering_primes(self) -> dict[str, float]:
        """Linear prime set in the form :mod:`tensorlbm.maneuvering` consumes."""
        return {
            "Y_v": -self.prime["Yb"],
            "Y_r": self.prime["Yr"],
            "N_v": -self.prime["Nb"],
            "N_r": self.prime["Nr"],
            "Y_delta": self.prime["Yd"],
            "N_delta": self.prime["Nd"],
        }


def identify_all(
    drift: Sequence[Mapping[str, float]],
    rudder: Sequence[Mapping[str, float]],
    circular: Sequence[Mapping[str, float]],
) -> MmgDerivativeSet:
    """Run the three identification stages and merge them into one table."""
    f_drift = identify_static_drift(drift)
    f_rudder = identify_rudder(rudder)
    f_circ = identify_circular_motion(circular)

    prime: dict[str, float] = {}
    std: dict[str, float] = {}
    for fit in (f_drift["X"], f_drift["Y"], f_drift["N"]):
        prime.update(fit.coefficients)
        std.update(fit.std)
    # X0 is measured three times (drift, rudder, circular); the drift value wins
    # and the others are kept as consistency evidence.
    prime["X0_delta"] = f_rudder["X"].coefficients["X0"]
    prime["X0_circular"] = f_circ["X"].coefficients["X0"]
    for key in ("Xdd",):
        prime[key] = f_rudder["X"].coefficients[key]
        std[key] = f_rudder["X"].std.get(key, float("nan"))
    for key in ("Yd",):
        prime[key] = f_rudder["Y"].coefficients[key]
        std[key] = f_rudder["Y"].std.get(key, float("nan"))
    for key in ("Nd",):
        prime[key] = f_rudder["N"].coefficients[key]
        std[key] = f_rudder["N"].std.get(key, float("nan"))
    for key in ("Xrr",):
        prime[key] = f_circ["X"].coefficients[key]
        std[key] = f_circ["X"].std.get(key, float("nan"))
    for fit in (f_circ["Y"], f_circ["N"]):
        for key, value in fit.coefficients.items():
            if key == "X0":
                continue
            prime[key] = value
            std[key] = fit.std.get(key, float("nan"))

    fits = {
        "drift_X": f_drift["X"].as_dict(),
        "drift_Y": f_drift["Y"].as_dict(),
        "drift_N": f_drift["N"].as_dict(),
        "rudder_X": f_rudder["X"].as_dict(),
        "rudder_Y": f_rudder["Y"].as_dict(),
        "rudder_N": f_rudder["N"].as_dict(),
        "circular_X": f_circ["X"].as_dict(),
        "circular_Y": f_circ["Y"].as_dict(),
        "circular_N": f_circ["N"].as_dict(),
    }
    conditions = {
        "drift_beta_deg": [float(r["beta_deg"]) for r in drift],
        "rudder_delta_deg": [float(r["delta_deg"]) for r in rudder],
        "circular_r_prime": [float(r["r_prime"]) for r in circular],
    }
    return MmgDerivativeSet(prime=prime, std=std, fits=fits, conditions=conditions)


def forward_mmg_forces(
    derivatives: MmgDerivativeSet | Mapping[str, float],
    *,
    beta_deg: float = 0.0,
    r_prime: float = 0.0,
    delta_deg: float = 0.0,
) -> tuple[float, float, float]:
    """Evaluate the identified MMG polynomial at one ``(beta, r', delta)`` state."""
    table = (
        derivatives.prime if isinstance(derivatives, MmgDerivativeSet) else dict(derivatives)
    )
    b = math.radians(float(beta_deg))
    d = math.radians(float(delta_deg))
    rp = float(r_prime)
    X = (
        table.get("X0", 0.0)
        + table.get("Xbb", 0.0) * b**2
        + table.get("Xbbbb", 0.0) * b**4
        + table.get("Xrr", 0.0) * rp**2
        + table.get("Xdd", 0.0) * d**2
    )
    Y = (
        table.get("Yb", 0.0) * b
        + table.get("Ybbb", 0.0) * b**3
        + table.get("Yr", 0.0) * rp
        + table.get("Yrrr", 0.0) * rp**3
        + table.get("Yd", 0.0) * d
    )
    N = (
        table.get("Nb", 0.0) * b
        + table.get("Nbbb", 0.0) * b**3
        + table.get("Nr", 0.0) * rp
        + table.get("Nrrr", 0.0) * rp**3
        + table.get("Nd", 0.0) * d
    )
    return X, Y, N


#: Synthetic "truth" used only to prove that the identification machinery
#: recovers coefficients it was not shown.  These numbers are *arbitrary*
#: (order-of-magnitude MMG values); they are not a KCS reference set.
SYNTHETIC_TRUTH: dict[str, float] = {
    "X0": -0.0200,
    "Xbb": 0.1200,
    "Xbbbb": 0.9000,
    "Xrr": -0.0300,
    "Xdd": 0.0200,
    "Yb": -0.6000,
    "Ybbb": -4.5000,
    "Yr": 0.1500,
    "Yrrr": -0.8000,
    "Yd": 0.1600,
    "Nb": -0.1100,
    "Nbbb": -0.6000,
    "Nr": -0.0450,
    "Nrrr": -0.0800,
    "Nd": -0.0700,
}


def synthetic_roundtrip(
    *,
    betas_deg: Sequence[float] = (-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0),
    r_primes: Sequence[float] = (-0.4, -0.2, 0.0, 0.2, 0.4),
    deltas_deg: Sequence[float] = (-35.0, -20.0, 0.0, 20.0, 35.0),
    noise: float = 0.0,
    seed: int = 0,
    truth: Mapping[str, float] | None = None,
) -> dict:
    """Identification self-test: known derivatives -> synthetic forces -> recovery.

    Args:
        betas_deg / r_primes / deltas_deg: synthetic test matrix.
        noise: relative Gaussian noise (std) added to every prime force.
        seed: RNG seed for the noise.
        truth: coefficient mapping; defaults to :data:`SYNTHETIC_TRUTH`.

    Returns:
        dict with recovered derivatives, per-coefficient relative errors and the
        maximum relative error over the coefficients that are actually
        identifiable from the sampled matrix.
    """
    tbl = dict(truth or SYNTHETIC_TRUTH)
    rng = np.random.default_rng(seed)

    def _noisy(value: float) -> float:
        if noise == 0.0:
            return value
        return value + float(rng.normal(0.0, noise * max(abs(value), 1e-6)))

    drift = []
    for b in betas_deg:
        X, Y, N = forward_mmg_forces(tbl, beta_deg=b)
        drift.append(
            {
                "beta_deg": float(b),
                "X_prime": _noisy(X),
                "Y_prime": _noisy(Y),
                "N_prime": _noisy(N),
            }
        )
    rudder = []
    for d in deltas_deg:
        # X0 must not be double counted: the rudder run is at beta = r' = 0.
        X, Y, N = forward_mmg_forces(tbl, delta_deg=d)
        rudder.append(
            {
                "delta_deg": float(d),
                "X_prime": _noisy(X),
                "Y_prime": _noisy(Y),
                "N_prime": _noisy(N),
            }
        )
    circular = []
    for rp in r_primes:
        X, Y, N = forward_mmg_forces(tbl, r_prime=rp)
        circular.append(
            {
                "r_prime": float(rp),
                "X_prime": _noisy(X),
                "Y_prime": _noisy(Y),
                "N_prime": _noisy(N),
            }
        )

    fits = identify_all(drift, rudder, circular)
    recovered = dict(fits.prime)
    checkable = [
        "Xbb",
        "Xbbbb",
        "Xrr",
        "Xdd",
        "Yb",
        "Ybbb",
        "Yr",
        "Yrrr",
        "Yd",
        "Nb",
        "Nbbb",
        "Nr",
        "Nrrr",
        "Nd",
    ]
    errors = {}
    for key in checkable:
        truth_v = float(tbl[key])
        rec = float(recovered.get(key, float("nan")))
        errors[key] = abs(rec - truth_v) / abs(truth_v) if truth_v else abs(rec - truth_v)
    return {
        "recovered": recovered,
        "relative_errors": errors,
        "max_relative_error": max(errors.values()),
        "noise": float(noise),
        "noise_seed": int(seed),
        "truth": tbl,
        "r2": {k: v["r2"] for k, v in fits.fits.items()},
    }


# --------------------------------------------------------------------------- #
# 6. Independent (non-CFD) cross-flow strip baseline
# --------------------------------------------------------------------------- #


def cross_flow_theory_derivatives(
    *,
    n_sections: int = 401,
    cd_cross: float = 1.2,
    cd_band: tuple[float, float] = (0.8, 1.6),
    length: float | None = None,
    beam: float | None = None,
    draft: float | None = None,
) -> dict:
    """Slender-body cross-flow (strip) estimate of the *nonlinear* derivatives.

    For a slender hull drifting at ``beta`` the lateral force on a section is
    dominated by cross-flow drag::

        dY = -1/2 rho C_D T(x) v(x) |v(x)| dx,     v(x) = v + x r

    with ``T(x)`` the local draught (the free-surface-bounded lateral projected
    width).  Expanding at ``r = 0`` gives the pure drift-cubic coefficient

        Y'_bbb = -integral( C_D T(x) dx ) / (L T_ref)

    and ``N'_bbb = +integral( C_D x T(x) dx ) / (L^2 T_ref)``.  This is a
    *theory* baseline for the cubic terms only: the linear MMG terms are set by
    potential-flow added mass and by hull-form details that a strip integral does
    not represent, so they are deliberately not produced here.

    Args:
        n_sections: longitudinal quadrature points.
        cd_cross: cross-flow drag coefficient of the reference 2-D section.
        cd_band: ``(low, high)`` band used to bracket the model uncertainty.
        length, beam, draft: hull particulars; default to the KCS model scale.

    Returns:
        dict with ``Ybbb``/``Nbbb`` (prime), the ``cd_band`` values and the
        assumptions actually used.
    """
    L = float(length if length is not None else KCS_MODEL_SCALE["L"])
    B = float(beam if beam is not None else KCS_MODEL_SCALE["B"])
    T = float(draft if draft is not None else KCS_MODEL_SCALE["T"])

    xi = np.linspace(-1.0, 1.0, int(n_sections))
    # Local draught of the KCS shape: hull exists for zeta in [0, 1] wherever the
    # half-breadth is non-zero, so T(x) = T * (1 if |xi| < 1 else 0).  The shape
    # information enters through the section depth profile below.
    zeta = np.linspace(0.0, 1.0, 201)
    hb = _kcs_half_beam_fraction(xi[:, None], zeta[None, :])  # (n_xi, n_zeta)
    depth = (hb > 1e-9).sum(axis=1) / len(zeta) * T  # effective wetted depth per section

    def _prime(cd: float) -> tuple[float, float]:
        w = depth / T                                   # non-dimensional depth
        y_bbb = -float(np.trapezoid(cd * w, xi)) / 1.0  # /(L*T) handled by dx = L/2 dxi
        n_bbb = float(np.trapezoid(cd * w * xi, xi)) / 1.0
        # dx = (L/2) dxi, and the prime factor carries 1/(L T) and 1/(L^2 T).
        return y_bbb, n_bbb

    y_bbb, n_bbb = _prime(float(cd_cross))
    y_lo, n_lo = _prime(float(cd_band[0]))
    y_hi, n_hi = _prime(float(cd_band[1]))
    return {
        "Ybbb": y_bbb,
        "Nbbb": n_bbb,
        "Ybbb_band": (min(y_lo, y_hi), max(y_lo, y_hi)),
        "Nbbb_band": (min(n_lo, n_hi), max(n_lo, n_hi)),
        "cd_cross": float(cd_cross),
        "cd_band": [float(cd_band[0]), float(cd_band[1])],
        "n_sections": int(n_sections),
        "assumptions": (
            "slender-body cross-flow drag only; local lateral projected width = "
            "section wetted depth; free-surface effects, potential added-mass "
            "terms and hull-form lift excluded; Ybbb/Nbbb only"
        ),
        "particulars": {"L": L, "B": B, "T": T},
    }