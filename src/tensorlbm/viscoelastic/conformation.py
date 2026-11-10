"""Conformation-tensor constitutive models for viscoelastic LBM.

Two constitutive laws are provided, both written in terms of the (symmetric
positive-definite) conformation tensor ``C``:

Oldroyd-B
    DC/Dt = L C + C L^T - (1/lam) (C - I)
    tau_p = (eta_p / lam) (C - I)

FENE-P (Peterlin closure)
    DC/Dt = L C + C L^T - (1/lam) (f(C) C - a I)
    f(C)  = L2 / (L2 - tr C),      a = L2 / (L2 - 3)
    tau_p = (eta_p / lam) (f(C) C - a I)

with ``L = grad u`` the velocity-gradient tensor, ``C_0 = I`` the equilibrium
state (tr C = 3), ``lam`` the relaxation time and ``eta_p`` the polymer
contribution to the zero-shear viscosity.

Every object here is *machine checkable*: :func:`analytic_channel` returns the
exact steady plane-Poiseuille solution of both models (velocity, shear rate,
full conformation tensor and first normal-stress difference), which is the
reference against which the LBM solver is verified.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "OldroydB",
    "FeneP",
    "I3",
    "trace3",
    "project_implementable",
    "min_eigenvalue",
    "analytic_channel",
]

I3 = np.eye(3, dtype=float)

#: Realizability floor for the *spectrum* of C.  C is a covariance-like tensor,
#: hence positive semi-definite.  Note that the individual eigenvalues of C are
#: NOT bounded below by 1: the exact Oldroyd-B steady-shear solution
#: [[1+2Wi^2, Wi], [Wi, 1]] has min eigenvalue ``1+Wi^2 - Wi sqrt(1+Wi^2)``
#: (= 0.586 at Wi=1), while tr C = 3 + 2 Wi^2 >= 3 still holds.  Requiring
#: ``C >= I`` would therefore exclude the exact solution -- the correct
#: implementability statement is ``C >= 0`` together with ``tr C >= 3``.
MIN_EIG = 0.0

#: Trace floor of C (equilibrium = 3, i.e. the isotropic coil size).
TRACE_FLOOR = 3.0


# --------------------------------------------------------------------------- #
# constitutive models
# --------------------------------------------------------------------------- #
class _ConstitutiveModel:
    """Common interface of the supported conformation-tensor models."""

    name = "base"
    lam: float

    # -- Peterlin function and its equilibrium coefficient ------------------ #
    def peterlin(self, tr_c):
        raise NotImplementedError

    def coeff_a(self) -> float:
        raise NotImplementedError

    def stress(self, C, eta_p):
        """Polymer stress ``tau_p`` for a conformation field ``C``."""
        raise NotImplementedError

    def elastic_energy_density(self, tr_c, eta_p):
        """Elastic energy per unit volume at equilibrium entropy (E(tr C=3)=0)."""
        raise NotImplementedError

    def peterlin_dissipation(self, tr_c, eta_p):
        """Peterlin (conformational) dissipation density, >= 0."""
        raise NotImplementedError

    # convenient equality / repr for reporting
    def params(self) -> dict:
        return {"model": self.name, "lam": self.lam}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}({self.params()})"


class OldroydB(_ConstitutiveModel):
    """Oldroyd-B: linear (Hookean) dumbbells, constant polymer viscosity."""

    name = "oldroyd_b"

    def __init__(self, lam: float):
        if lam <= 0:
            raise ValueError("relaxation time lam must be > 0")
        self.lam = float(lam)

    def peterlin(self, tr_c):
        return np.ones_like(np.asarray(tr_c, dtype=float))

    def coeff_a(self) -> float:
        return 1.0

    def stress(self, C, eta_p):
        C = np.asarray(C, dtype=float)
        return (eta_p / self.lam) * (C - I3)

    def elastic_energy_density(self, tr_c, eta_p):
        return (eta_p / (2.0 * self.lam)) * (np.asarray(tr_c, float) - 3.0)

    def peterlin_dissipation(self, tr_c, eta_p):
        return (eta_p / (2.0 * self.lam ** 2)) * np.maximum(
            np.asarray(tr_c, float) - 3.0, 0.0
        )


class FeneP(_ConstitutiveModel):
    """FENE-P: finitely extensible dumbbells, maximum squared stretch ``L2``."""

    name = "fene_p"

    def __init__(self, lam: float, L2: float):
        if lam <= 0:
            raise ValueError("relaxation time lam must be > 0")
        if L2 <= 3.0:
            raise ValueError("FENE-P extensibility L2 must be > 3")
        self.lam = float(lam)
        self.L2 = float(L2)

    # keep L2 accessible under the numpy-ish name used in the docs
    @property
    def l2(self) -> float:
        return self.L2

    def params(self) -> dict:
        return {"model": self.name, "lam": self.lam, "L2": self.L2}

    def peterlin(self, tr_c):
        tr_c = np.asarray(tr_c, dtype=float)
        # the physical branch requires tr C < L2; clip for numerical safety
        denom = np.maximum(self.L2 - tr_c, 1e-12 * self.L2)
        return self.L2 / denom

    def coeff_a(self) -> float:
        return self.L2 / (self.L2 - 3.0)

    def stress(self, C, eta_p):
        C = np.asarray(C, dtype=float)
        f = self.peterlin(trace3(C))
        return (eta_p / self.lam) * (f[..., None, None] * C - self.coeff_a() * I3)

    def elastic_energy_density(self, tr_c, eta_p):
        tr_c = np.asarray(tr_c, dtype=float)
        ratio = (self.L2 - 3.0) / np.maximum(self.L2 - tr_c, 1e-12 * self.L2)
        return (eta_p / (2.0 * self.lam)) * self.L2 * np.log(np.maximum(ratio, 1e-300))

    def peterlin_dissipation(self, tr_c, eta_p):
        tr_c = np.asarray(tr_c, dtype=float)
        f = self.peterlin(tr_c)
        a = self.coeff_a()
        return (eta_p / (2.0 * self.lam ** 2)) * f * (f * tr_c - 3.0 * a)


# --------------------------------------------------------------------------- #
# geometry of the tensor
# --------------------------------------------------------------------------- #
def trace3(C):
    """Trace over the last two (3x3) axes."""
    C = np.asarray(C, dtype=float)
    return C[..., 0, 0] + C[..., 1, 1] + C[..., 2, 2]


def min_eigenvalue(C):
    """Smallest eigenvalue of a (..., 3, 3) symmetric tensor field."""
    C = np.asarray(C, dtype=float)
    C = 0.5 * (C + np.swapaxes(C, -1, -2))
    w = np.linalg.eigvalsh(C)
    return w[..., 0]


def project_implementable(C, floor: float = MIN_EIG, trace_floor: float = TRACE_FLOOR):
    """Project ``C`` onto the physically realizable set.

    Realizability of a conformation tensor means (i) ``C`` symmetric positive
    semi-definite (it is an ensemble average of dyadics ``<R R>``) and (ii)
    ``tr C >= 3`` (the equilibrium coil has tr C = 3 and the upper-convected
    dynamics cannot shrink the trace below it).  The projection therefore

    1. clips the spectrum from below at ``floor`` (0 by default), and
    2. adds an isotropic increment when ``tr C < trace_floor``.

    Both steps are minimal: they change nothing on a field that already
    satisfies the constraint, so the projection is inert for the exact laminar
    solution and only acts as a safety net at high Weissenberg number.

    Returns
    -------
    (C_projected, n_modified) : tuple
        Projected tensor and number of nodes whose spectrum/trace was modified.
    """
    C = np.asarray(C, dtype=float)
    C = 0.5 * (C + np.swapaxes(C, -1, -2))
    w, V = np.linalg.eigh(C)
    bad = w < floor
    n_bad = int(np.count_nonzero(bad))
    if n_bad:
        w = np.maximum(w, floor)
        C = np.einsum("...ik,...k,...jk->...ij", V, w, V)
        C = 0.5 * (C + np.swapaxes(C, -1, -2))
    tr_c = trace3(C)
    deficit = trace_floor - tr_c
    n_low = int(np.count_nonzero(deficit > 0.0))
    if n_low:
        C = C + np.where(deficit > 0.0, deficit / 3.0, 0.0)[..., None, None] * I3
    return C, n_bad + n_low


def _fene_p_trace_from_wi(wi, L2):
    """Self-consistent FENE-P steady-shear trace ``tr C`` for a given Wi.

    Steady simple shear gives ``C_yy = C_zz = a/f``, ``C_xy = a lam gamma_dot / f^2``
    and hence the *implicit* scalar equation

        tr C = a (3 + 2 Wi^2 / f^2) / f,      f = L2 / (L2 - tr C),

    solved by Brent's method on ``tr C in [3, L2)``.  The ``L2 -> infinity``
    limit reproduces ``tr C = 3 + 2 Wi^2`` (Oldroyd-B).  (An earlier version
    used the explicit large-``L2`` approximation ``f = 1 + a(3+2Wi^2)/L2``,
    which is wrong by O(Wi^2/L2) in ``C_xy``: the test
    ``test_fene_p_closed_form_matches_ode`` pins the self-consistent branch.)
    """
    from scipy.optimize import brentq

    wi = np.atleast_1d(np.asarray(wi, dtype=float))
    a = L2 / (L2 - 3.0)
    out = np.empty_like(wi)

    def resid(s, w2):
        f = L2 / (L2 - s)
        return a * (3.0 + 2.0 * w2 / f ** 2) / f - s

    for k, w in enumerate(wi.ravel()):
        w2 = float(w) ** 2
        if w2 == 0.0:
            out.ravel()[k] = 3.0
            continue
        out.ravel()[k] = brentq(
            resid, 3.0, L2 * (1.0 - 1e-14), args=(w2,),
            xtol=1e-15, rtol=1e-15, maxiter=300,
        )
    return out


# --------------------------------------------------------------------------- #
# exact steady plane-Poiseuille solution
# --------------------------------------------------------------------------- #
def _steady_shear_state(gamma_dot, lam, model, L2):
    """Exact steady simple-shear state for a shear rate ``gamma_dot``.

    Returns ``(C, f, a, eta_polymer_factor)`` where ``a/f`` is the factor by
    which FENE-P reduces the polymer viscosity (equal to 1 for Oldroyd-B).
    """
    gd = np.asarray(gamma_dot, dtype=float)
    wi = lam * np.abs(gd)
    if model == "oldroyd_b":
        f = np.ones_like(wi)
        a = 1.0
        tr_c = 3.0 + 2.0 * wi ** 2
        c_xy = lam * gd
    elif model == "fene_p":
        tr_c = _fene_p_trace_from_wi(wi, L2)
        f = L2 / (L2 - tr_c)
        a = L2 / (L2 - 3.0)
        c_xy = a * lam * gd / f ** 2
    else:  # pragma: no cover - guard
        raise ValueError(f"unknown model {model!r}")
    C = np.zeros(gd.shape + (3, 3), dtype=float)
    C[..., 1, 1] = a / f
    C[..., 2, 2] = a / f
    C[..., 0, 1] = C[..., 1, 0] = c_xy
    C[..., 0, 0] = tr_c - 2.0 * a / f
    return C, f, a, a / f


def analytic_channel(
    y,
    H: float,
    G: float,
    nu0: float,
    lam: float,
    beta: float = 0.5,
    model: str = "oldroyd_b",
    L2: float | None = None,
    n_table: int = 8001,
):
    """Exact steady plane-Poiseuille solution of Oldroyd-B / FENE-P.

    Geometry: walls at ``y = +-H``, body force ``G`` (per unit mass, driving +x),
    density 1.  ``nu0`` is the *zero-shear* kinematic viscosity and
    ``beta = eta_s / eta_0`` the solvent fraction, hence ``nu_s = beta nu0`` and
    ``eta_p = (1-beta) nu0``.

    Construction (exact, no approximation beyond the constitutive closure)
    ----------------------------------------------------------------------
    In fully developed flow ``u = u(y) e_x``, so every material point undergoes a
    *steady simple shear* and ``u . grad C = 0`` identically.  The constitutive
    equation therefore reduces to its algebraic steady-shear branch, which gives
    the local polymer shear stress ``tau_p,xy(gamma_dot)``.  The momentum balance

        nu_s gamma_dot + tau_p,xy(gamma_dot) = -G y

    is then a scalar, strictly monotone equation for ``gamma_dot(y)`` (solved
    here by inverting a fine table), and

        u(y) = -int_y^H gamma_dot(s) ds .

    Consequences:

    * **Oldroyd-B** has a shear-rate independent viscosity, so this reproduces
      the classical parabola ``u = G(H^2-y^2)/(2 nu0)`` and
      ``N1 = 2 eta_p lam gamma_dot^2`` exactly.
    * **FENE-P is shear-thinning**: its polymer viscosity is
      ``eta_p a/f`` with ``f = L2/(L2-trC) > 1``, so the exact profile is *not*
      parabolic and the total viscosity is not ``nu0``.  (An earlier version of
      this function wrongly reused the parabolic profile for FENE-P and missed
      the ``1/f^2`` factor of ``C_xy``; both errors are pinned by the tests
      ``test_analytic_shear_fene_p_solves_the_steady_constitutive_equation`` and
      ``test_fene_p_is_shear_thinning``.)

    Returns
    -------
    dict with ``u``, ``gamma_dot`` (signed ``du/dy``), ``C``, ``N1``, ``tau_xy``,
    ``trC``, ``wi`` (local), ``wi_wall``, ``eta_local`` (total viscosity) and
    ``u_max``.
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    nu_s = beta * nu0
    eta_p = (1.0 - beta) * nu0

    # --- table of the (monotone) stress/shear-rate relation ----------------- #
    gd_max = 1.5 * abs(G) * H / max(nu_s, 1e-30)
    gd_tab = np.linspace(0.0, gd_max, n_table)
    _, _, _, ratio_tab = _steady_shear_state(gd_tab, lam, model, L2)
    tau_tab = (nu_s + eta_p * ratio_tab) * gd_tab        # tau_xy > 0

    # --- local shear rate from the momentum balance ------------------------- #
    tau_needed = abs(G) * np.abs(y)
    gd_abs = np.interp(tau_needed, tau_tab, gd_tab)
    gamma_dot = -np.sign(G) * np.sign(y) * gd_abs        # du/dy, signed

    C, f, a, ratio = _steady_shear_state(gamma_dot, lam, model, L2)
    eta_local = nu_s + eta_p * ratio

    # --- velocity: integrate du/dy from the wall inwards -------------------- #
    y_fine = np.linspace(0.0, abs(H), 20001)
    gd_fine = np.interp(abs(G) * y_fine, tau_tab, gd_tab)
    u_fine = np.concatenate(
        ([0.0], np.cumsum(0.5 * (gd_fine[1:] + gd_fine[:-1]) * np.diff(y_fine)))
    )                                        # int_0^s |gamma_dot| ds, s from centre
    # u(y) = int_{|y|}^{H} |gamma_dot| ds  (no-slip at the wall)
    integral = u_fine
    u_of_r = integral[-1] - np.interp(np.abs(y), y_fine, integral)
    u = np.sign(G) * u_of_r

    tr_c = trace3(C)
    N1 = (eta_p / lam) * f * (C[..., 0, 0] - C[..., 1, 1])
    tau_xy = (eta_p / lam) * f * C[..., 0, 1]
    return {
        "u": u,
        "gamma_dot": gamma_dot,
        "C": C,
        "N1": N1,
        "tau_xy": tau_xy,
        "trC": tr_c,
        "wi": lam * np.abs(gamma_dot),
        "wi_wall": lam * float(np.max(gd_abs)),
        "eta_local": eta_local,
        "eta_p": eta_p,
        "nu_s": nu_s,
        "u_max": float(np.max(np.abs(u))),
    }
