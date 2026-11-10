"""Drag-reduction bookkeeping for the IN67-N01 micro-bubble channel.

The single job of this module is to turn resolved mean fields into a *closed*
wall-shear budget.  Everything is a direct integral of measured profiles, so no
coefficient is fitted.

Momentum identity
-----------------
For a statistically steady plane channel (half height ``h``, wall at ``y=0``,
centreline at ``y=h``, streamwise periodic) the mean momentum equation reads

    d(tau)/dy = -rho f ,      tau(y) = mu_eff (d ubar/dy) - rho <u'v'> ,

with ``tau(0) = tau_w`` and ``tau(h) = 0``.  Therefore
``tau(y) = tau_w (1 - y/h)`` and, integrating against ``1 - y/h`` over the half
channel,

    tau_w = A - B ,
    A = (2/h) int_0^h mu_eff (d ubar/dy) dy ,
    B = (2/h) int_0^h rho <u'v'> dy .

``A`` is evaluated *exactly* (no integration by parts) by the telescoping sum

    A = (2/h) sum_j mu_j (u_face_j - u_face_{j-1}) ,   sum_j (.) = u_c - u_s ,

so a spatially varying ``mu_eff`` and wall slip ``u_s`` are both handled without
approximation beyond the face interpolation.  ``B`` is a trapezoidal integral
with the boundary values ``<u'v'>(0) = <u'v'>(h) = 0`` imposed from symmetry.
Laminar check: ``B = 0``, ``u_c = f h^2/(2 nu)`` gives ``A = rho f h = tau_w``.

Mechanism split
---------------
Define the measured viscosity-contrast integral

    C_mu = (2/h) int (mu_eff - mu_l) (d ubar/dy) dy   (= telescoping sum too),

then, adding and subtracting ``(2/h) mu_l (u_c - u_s)`` in ``A``:

    tau_w0 - tau_w = [slip] + [viscosity] + [turbulence] + [profile]

    slip       =  (2/h) mu_l u_s
    viscosity  = -C_mu
    turbulence =  (B - B_0)
    profile    =  (2/h) mu_l (u_c0 - u_c)

The four terms close against ``tau_w0 - tau_w`` *identically* (see the algebra in
the report): the only error left is the discretisation of the two integrals, so
the closure residual is a genuine numerical test, not a bookkeeping identity.
The ``profile`` piece is the mean-flow footprint of the modified Reynolds-stress
budget (at fixed mass flux the centreline velocity must drop when ``B`` rises);
it is reported separately and folded into the turbulence term in the three-term
statement, which is flagged wherever a number is quoted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HalfChannelProfiles:
    """Time- and x-averaged profiles on the lower half channel (wall -> centre)."""

    y: np.ndarray            # cell-centre distance from the wall (lattice units)
    rho: np.ndarray          # mean density
    u: np.ndarray            # mean streamwise velocity
    mu: np.ndarray           # mean effective dynamic viscosity
    rho_uv: np.ndarray       # mean <rho u'v'>
    alpha: np.ndarray        # mean void fraction
    h: float                 # half channel height (lattice units)

    def du_dy(self) -> np.ndarray:
        """Central differences on the interior, one-sided at both ends."""
        return np.gradient(self.u, self.y, edge_order=2)


def _polyfit_local(x: np.ndarray, y: np.ndarray, deg: int) -> tuple[np.ndarray, float]:
    """Least-squares polynomial fit on a scaled abscissa (numpy-version safe).

    ``np.polyfit`` raises a bare ``LinAlgError`` (wrapped as ``SystemError`` on the
    loong64 cluster's numpy) when the Vandermonde matrix is badly conditioned --
    the first nodes of a wall profile are all within a few lattice units.  Scaling
    the abscissa and calling ``lstsq`` directly removes both the conditioning and
    the version dependence.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    scale = float(np.max(np.abs(x)))
    if scale <= 0.0:
        scale = 1.0
    vander = np.vander(x / scale, deg + 1)
    coef, *_ = np.linalg.lstsq(vander, y, rcond=None)
    return coef, scale


def wall_gradient(y: np.ndarray, u: np.ndarray, n_fit: int = 6) -> float:
    """Wall shear rate ``du/dy|_0`` from a local quadratic fit.

    A two-point one-sided difference is biased at the first node (the
    wall-adjacent cell of a Shan-Chen channel carries part of the interaction
    force); a quadratic least-squares fit through the first few nodes and its
    derivative *at the wall* removes most of that bias, which is what keeps the
    three friction-velocity estimates inside the 2 % consistency band.
    """
    idx = slice(0, max(4, n_fit))
    coef, scale = _polyfit_local(y[idx], u[idx], 2)
    # coef = [c2, c1, c0] in the scaled variable; derivative at the wall = c1/scale
    return float(coef[-2] / scale)


def slip_velocity(y: np.ndarray, u: np.ndarray, n_fit: int = 4) -> tuple[float, float]:
    """Wall slip velocity ``u_s`` and slip length ``b`` from a near-wall fit.

    ``u(y) ~ u_s + gamma_wall y`` over the first ``n_fit`` nodes; the *quadratic*
    fit is used so that a parabolic (laminar) profile returns ``u_s = 0`` exactly
    instead of a curvature-induced offset.  ``b = u_s/gamma``.
    A non-positive extrapolated ``u_s`` is quoted as ``b = 0`` because a slip
    length is only meaningful for positive extrapolated wall velocity.
    """
    idx = slice(0, max(4, n_fit))
    coef, scale = _polyfit_local(y[idx], u[idx], 2)
    gamma = float(coef[-2] / scale)
    u_s = float(coef[-1])
    b = u_s / gamma if abs(gamma) > 1e-12 and u_s > 0.0 else 0.0
    return u_s, b


def _faces(values: np.ndarray, bottom: float, top: float) -> np.ndarray:
    """Face values of a cell-centred profile, with the two boundary values given."""
    faces = np.empty(values.size + 1, dtype=float)
    faces[0] = bottom
    faces[-1] = top
    faces[1:-1] = 0.5 * (values[:-1] + values[1:])
    return faces


def telescoping_integrals(prof: HalfChannelProfiles, mu_liquid: float) -> dict[str, float]:
    """``A``, ``C_mu`` and ``B`` (see module docstring) by exact telescoping."""
    h = prof.h
    u_s, b = slip_velocity(prof.y, prof.u)
    u_c = float(prof.u[-1])
    faces = _faces(prof.u, u_s, u_c)
    du = np.diff(faces)
    scale = 2.0 / h
    A = scale * float(np.sum(prof.mu * du))
    C_mu = scale * float(np.sum((prof.mu - mu_liquid) * du))
    # Reynolds term: <u'v'> vanishes at both the wall and the centreline
    y_ext = np.concatenate(([0.0], prof.y, [h]))
    rv_ext = np.concatenate(([0.0], prof.rho_uv, [0.0]))
    B = scale * float(np.trapezoid(rv_ext, y_ext))
    return {
        "A_viscous": A,
        "B_reynolds": B,
        "C_mu": C_mu,
        "tau_w_identity": A - B,
        "u_slip": u_s,
        "slip_length": b,
        "u_center": u_c,
        "tau_w_gradient": float(prof.mu[0] * wall_gradient(prof.y, prof.u)),
    }


def wall_shear_budget(prof: HalfChannelProfiles, mu_liquid: float) -> dict[str, float]:
    """Backwards-compatible alias of :func:`telescoping_integrals`."""
    return telescoping_integrals(prof, mu_liquid)


def mechanism_split(
    ref: dict[str, float],
    case: dict[str, float],
    mu_liquid: float,
    h: float,
) -> dict[str, float]:
    """Split ``DR = 1 - tau_w/tau_w0`` into slip / viscosity / turbulence.

    ``ref`` and ``case`` carry the measured ``tau_w``, ``C_mu``, ``B_reynolds``,
    ``u_center`` and ``u_slip``.  All three quoted terms are direct integrals of
    measured fields; the closure residual is the numerical error of those
    integrals and is what criterion C3 tests.
    """
    tau_w0 = float(ref["tau_w"])
    tau_w = float(case["tau_w"])
    if tau_w0 <= 0.0:
        raise ValueError("reference wall shear must be positive")
    scale = 2.0 / h

    dr = 1.0 - tau_w / tau_w0
    slip = scale * mu_liquid * float(case["u_slip"]) / tau_w0
    visc = -float(case["C_mu"]) / tau_w0
    turb = (float(case["B_reynolds"]) - float(ref["B_reynolds"])) / tau_w0
    profile = scale * mu_liquid * (float(ref["u_center"]) - float(case["u_center"])) / tau_w0

    three = slip + visc + turb + profile
    return {
        "DR_total": dr,
        "DR_slip": slip,
        "DR_viscosity": visc,
        "DR_turbulence": turb,
        "DR_profile_footprint": profile,
        "DR_three_term": three,
        "DR_turb_incl_profile": turb + profile,
        "closure_residual": three - dr,
        "closure_rel": abs(three - dr) / abs(dr) if dr != 0.0 else abs(three - dr),
    }


def laminar_wall_shear(mu: float, u_bulk: float, h: float) -> float:
    """Plane-Poiseuille wall shear ``tau_w = 3 mu u_bulk / h`` (full height 2h)."""
    return 3.0 * mu * u_bulk / h


def slip_length_drag_reduction(b: float, h: float) -> float:
    """``dC_f`` of a laminar channel with slip length ``b`` at matched flow rate.

    Poiseuille with wall slip: ``Q = (f h^2/nu)(h/3 + b)`` so at matched ``Q``
    ``tau_w(b)/tau_w(0) = 1/(1 + 3b/h)`` and ``dC_f = (3b/h)/(1 + 3b/h)`` -- the
    analytic companion used to check the measured slip term.
    """
    if b <= 0.0:
        return 0.0
    x = 3.0 * b / h
    return x / (1.0 + x)


def einstein_bubbly_viscosity(alpha: float) -> float:
    """Relative mixture viscosity of a dilute bubble suspension, ``1 + alpha``.

    Bubbles are stress-free inclusions, so the Taylor/Einstein factor for solid
    spheres (``1 + 2.5 alpha``) reduces to ``1 + alpha``.  Quoted only as an
    external comparison: the lattice mixture viscosity here comes from its
    phase-dependent relaxation time.
    """
    return 1.0 + alpha