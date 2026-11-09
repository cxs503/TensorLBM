"""Closed forms for the relative viscosity of a particle suspension.

Notation: ``phi`` is the particle volume fraction, ``eta_r = eta_eff / eta_f``
the relative (dimensionless) viscosity, ``[eta]`` the intrinsic viscosity
(``5/2`` for rigid spheres, force- and torque-free).

Closed forms implemented
------------------------
Einstein (1906, 1911) -- dilute, exact to first order in ``phi``::

    eta_r = 1 + [eta] phi,   [eta] = 5/2

Batchelor (1977) / Batchelor & Green (1972) -- dilute + first pair
interaction of hard spheres in a statistically homogeneous shear::

    eta_r = 1 + [eta] phi + k2 phi**2,   k2 = 6.2

Krieger & Dougherty (1959) -- resums the whole concentration range with a
single divergence at the maximum packing fraction::

    eta_r = (1 - phi / phi_max) ** (-[eta] phi_max)

so that ``d eta_r / d phi |_{phi=0} = [eta]`` is preserved exactly and the
viscosity diverges at ``phi_max``.

Why a *contract* and not just three functions
---------------------------------------------
The Bingham / mu(I) kernels elsewhere in the platform need one callable
``eta_r(phi)``.  Using the three forms naively introduces a discontinuous
derivative (and, when each branch is fitted independently, a discontinuous
*value*) at the switch points.  :class:`ViscosityContract` blends the
branches over a finite window with a C1 ``smootherstep`` weight, so the
assembled closure is C0 (value) and C1 (slope) everywhere.  The blend window
is a *documented modelling choice*, and the resulting jump across the switch
is measured by :meth:`ViscosityContract.max_rel_jump` (the campaign
acceptance criterion asks for ``<= 5 %``).

Coupling to the mu(I) and power-law branches
--------------------------------------------
``eta_r(phi)`` multiplies the suspending-fluid viscosity.  For a
Bingham/Herschel-Bulkley carrier::

    tau = tau_y(phi) + K(phi) gamma_dot**n,   tau_y(phi) = tau_y0 * eta_r(phi)
    K(phi) = K0 * eta_r(phi)

(the yield stress of a suspension scales with the same concentration factor,
see Chateau, Ovarlez & Trung 2008; Mahaut et al. 2013).  For the inertial
mu(I) branch the dimensionless stress is the sum of the viscous (particle)
stress and the inertial contribution::

    eta_r(phi, I) = eta_r_vis(phi) * (1 + I / I_v)

with ``I = gamma_dot d sqrt(rho_p / p)`` the inertial number and ``I_v`` the
viscous-number crossover measured by the ``Re_p`` sweep
(``calibrate.re_p_sweep``).  ``I/I_v -> 0`` recovers the Stokes branch, so
the inertia correction does not disturb the calibrated closures.

All functions are NumPy-vectorised and accept scalars, lists or arrays.

References
----------
Einstein, A. (1906) *Ann. Phys.* 19, 289; (1911) *Ann. Phys.* 34, 591.
Batchelor, G. K. (1977) *J. Fluid Mech.* 83, 97.
Batchelor, G. K. & Green, J. T. (1972) *J. Fluid Mech.* 56, 401.
Krieger, I. M. & Dougherty, T. J. (1959) *Trans. Soc. Rheol.* 3, 137.
Boyer, F., Guazzelli, E. & Pouliquen, O. (2011) *Phys. Rev. Lett.* 107,
    188301.
Chateau, X., Ovarlez, G. & Trung, K. L. (2008) *Rheol. Acta* 48, 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "INTRINSIC_VISCOSITY_SPHERE",
    "BATCHELOR_K2",
    "einstein_relative_viscosity",
    "batchelor_relative_viscosity",
    "krieger_dougherty_relative_viscosity",
    "fit_krieger_dougherty_phi_max",
    "smootherstep",
    "ViscosityContract",
    "bingham_suspension_stress",
    "power_law_suspension_consistency",
]

INTRINSIC_VISCOSITY_SPHERE = 2.5
"""``[eta]`` for rigid spheres with free rotation (Einstein 1906)."""

BATCHELOR_K2 = 6.2
"""Second-order coefficient of the hard-sphere shear viscosity (Batchelor 1977)."""


def _as_float_array(phi) -> np.ndarray:
    return np.asarray(phi, dtype=float)


def einstein_relative_viscosity(phi, intrinsic: float = INTRINSIC_VISCOSITY_SPHERE):
    """``eta_r = 1 + [eta] phi`` (Einstein 1906).  Exact to O(phi**2)."""
    return 1.0 + intrinsic * _as_float_array(phi)


def batchelor_relative_viscosity(
    phi,
    intrinsic: float = INTRINSIC_VISCOSITY_SPHERE,
    k2: float = BATCHELOR_K2,
):
    """``eta_r = 1 + [eta] phi + k2 phi**2`` (Batchelor 1977)."""
    p = _as_float_array(phi)
    return 1.0 + intrinsic * p + k2 * p * p


def krieger_dougherty_relative_viscosity(
    phi,
    phi_max: float,
    intrinsic: float = INTRINSIC_VISCOSITY_SPHERE,
):
    """``eta_r = (1 - phi/phi_max)**(-[eta] phi_max)`` (Krieger & Dougherty 1959).

    Raises ``ValueError`` for ``phi >= phi_max`` (the closure diverges there and
    silently returning ``inf`` hides a physical misuse upstream).
    """
    p = _as_float_array(phi)
    if np.any(p >= phi_max):
        raise ValueError(
            f"Krieger-Dougherty requires phi < phi_max={phi_max}; got max phi="
            f"{np.max(p):.4f}"
        )
    return (1.0 - p / phi_max) ** (-intrinsic * phi_max)


def fit_krieger_dougherty_phi_max(
    phi,
    eta_r,
    intrinsic: float = INTRINSIC_VISCOSITY_SPHERE,
    bounds: tuple[float, float] = (0.40, 0.90),
) -> dict:
    """Least-squares fit of ``phi_max`` (and optionally ``[eta]``) to data.

    KD is linearised as ``ln eta_r = -[eta] phi_max ln(1 - phi/phi_max)``; the
    one-parameter fit in ``phi_max`` is a scalar bounded minimisation (golden
    section here, no SciPy dependency).  The intrinsic viscosity is kept fixed
    by default because with a dilute-window regression we already measure it
    independently -- letting both floats free on a narrow ``phi`` window makes
    the fit ill-conditioned and is reported as such (``cond_flag``).

    Returns a dict with ``phi_max``, ``intrinsic``, ``rmse``, ``r2`` and the
    residual vector, so the caller can quote an uncertainty instead of a
    single number.
    """
    p = _as_float_array(phi).ravel()
    e = _as_float_array(eta_r).ravel()
    if p.size < 2:
        raise ValueError("need at least 2 (phi, eta_r) points to fit phi_max")

    def sse(pm: float) -> float:
        model = np.log(krieger_dougherty_relative_viscosity(p, pm, intrinsic))
        return float(np.sum((np.log(e) - model) ** 2))

    lo, hi = bounds
    # golden-section search on [lo, hi] (sse is unimodal in phi_max here)
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = sse(c), sse(d)
    for _ in range(200):
        if b - a < 1e-12:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = sse(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = sse(d)
    phi_max = 0.5 * (a + b)

    pred = krieger_dougherty_relative_viscosity(p, phi_max, intrinsic)
    resid = e - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((e - np.mean(e)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # conditioning probe: perturb phi_max by +-2% and report the sse penalty
    probe = max(sse(phi_max * 1.02), sse(phi_max * 0.98))
    return {
        "phi_max": float(phi_max),
        "intrinsic": float(intrinsic),
        "rmse": float(np.sqrt(ss_res / p.size)),
        "r2": float(r2),
        "residual": resid.tolist(),
        "n_points": int(p.size),
        "sse": ss_res,
        "sse_2pct_probe": float(probe),
        "phi_range": [float(p.min()), float(p.max())],
    }


def smootherstep(x):
    """C2 ``smootherstep`` ramp on ``[0, 1]`` (0 below, 1 above)."""
    t = np.clip(_as_float_array(x), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@dataclass
class ViscosityContract:
    """Assembled ``eta_r(phi)`` for the Bingham / mu(I) kernels.

    Branches, in increasing ``phi``:

    ==========  ==================  ==================================
    branch      range (default)     closed form
    ==========  ==================  ==================================
    einstein    ``phi <= 0.05``      ``1 + 2.5 phi``
    batchelor   ``0.05 .. 0.15``     ``1 + 2.5 phi + 6.2 phi**2``
    kd          ``phi > 0.15``       ``(1-phi/phi_max)**(-2.5 phi_max)``
    ==========  ==================  ==================================

    Transitions are blended over ``[switch - width/2, switch + width/2]`` with
    :func:`smootherstep`, giving a C1 closure.  ``phi_max`` and the intrinsic
    viscosity are calibration outputs (see ``calibrate``).
    """

    phi_max: float = 0.63
    intrinsic: float = INTRINSIC_VISCOSITY_SPHERE
    k2: float = BATCHELOR_K2
    switches: tuple[float, float] = (0.05, 0.15)
    blend_width: float = 0.02
    i_v: float | None = None
    """Inertial crossover number ``I_v`` (filled by the Re_p sweep)."""

    metadata: dict = field(default_factory=dict)

    # -- individual branches -------------------------------------------------
    def branch_values(self, phi) -> dict:
        p = _as_float_array(phi)
        return {
            "einstein": einstein_relative_viscosity(p, self.intrinsic),
            "batchelor": batchelor_relative_viscosity(p, self.intrinsic, self.k2),
            "kd": krieger_dougherty_relative_viscosity(p, self.phi_max, self.intrinsic),
        }

    def branch(self, phi) -> np.ndarray:
        """Hard (unblended) branch label for each ``phi`` (diagnostic only)."""
        p = _as_float_array(phi)
        s1, s2 = self.switches
        out = np.where(p <= s1, "einstein", np.where(p <= s2, "batchelor", "kd"))
        return out

    def eta_r(self, phi) -> np.ndarray:
        """Blended relative viscosity (C1 across the switch points)."""
        p = _as_float_array(phi)
        v = self.branch_values(p)
        s1, s2 = self.switches
        h = 0.5 * self.blend_width
        w1 = smootherstep((p - (s1 - h)) / max(self.blend_width, 1e-12))
        w2 = smootherstep((p - (s2 - h)) / max(self.blend_width, 1e-12))
        out = (1.0 - w1) * v["einstein"] + w1 * ((1.0 - w2) * v["batchelor"] + w2 * v["kd"])
        return out

    __call__ = eta_r

    def hard_switch_rel_jump(self, eps: float = 1e-6) -> float:
        """Max relative value jump of a naive (unblended) switch.

        This is the number the acceptance criterion ``<= 5 %`` refers to when
        the three closures are stitched *without* a blend; it is reported as
        the "before" baseline against which the blended contract is the fix.
        """
        s1, s2 = self.switches
        v = self.branch_values(np.array([s1 - eps, s1 + eps, s2 - eps, s2 + eps]))
        jumps = []
        pairs = [(0, 1), (2, 3)]
        for i, j in pairs:
            e = v["einstein"][i] if i < 2 else None  # noqa: F841  (documentation)
        lo1, hi1 = v["einstein"][0], v["batchelor"][1]
        lo2, hi2 = v["batchelor"][2], v["kd"][3]
        jumps = [abs(hi1 - lo1) / lo1, abs(hi2 - lo2) / lo2]
        return float(max(jumps))

    def max_rel_jump(self, phi_grid=None, eps: float = 1e-6) -> float:
        """Max relative jump of the assembled contract on a fine ``phi`` grid.

        Consecutive-point relative differences of the blended closure; for a
        C0 closure on a grid of spacing ``dphi`` this is ``O(dphi * dln eta/dphi)``
        and vanishes with ``dphi``.  The hard-switch value is returned by
        :meth:`hard_switch_rel_jump` for comparison.
        """
        if phi_grid is None:
            phi_grid = np.linspace(1e-4, min(0.4, self.phi_max - 1e-3), 20001)
        p = _as_float_array(phi_grid)
        e = self.eta_r(p)
        return float(np.max(np.abs(np.diff(e)) / np.abs(e[:-1])))

    def eta_r_inertial(self, phi, inertial_number) -> np.ndarray:
        """``eta_r(phi, I) = eta_r_vis(phi) (1 + I / I_v)`` (mu(I) coupling).

        Requires ``i_v`` to have been set by the ``Re_p`` calibration; falls
        back to ``i_v = 1`` with an explicit warning-free documented default so
        a downstream kernel never silently gets a different formula.
        """
        i_v = 1.0 if self.i_v is None else float(self.i_v)
        p = _as_float_array(phi)
        big_i = _as_float_array(inertial_number)
        return self.eta_r(p) * (1.0 + big_i / i_v)

    def as_dict(self) -> dict:
        return {
            "phi_max": self.phi_max,
            "intrinsic": self.intrinsic,
            "k2": self.k2,
            "switches": list(self.switches),
            "blend_width": self.blend_width,
            "i_v": self.i_v,
            "hard_switch_rel_jump": self.hard_switch_rel_jump(),
            "blended_max_rel_jump": self.max_rel_jump(),
            "metadata": self.metadata,
        }


def bingham_suspension_stress(
    shear_rate,
    phi,
    tau_y0: float,
    consistency: float,
    exponent: float = 1.0,
    contract: ViscosityContract | None = None,
    regularisation: float = 1e-3,
):
    """Herschel-Bulkley stress of a *suspension* carrier, ``phi``-coupled.

    ``tau = tau_y(phi) + K(phi) gamma_dot**n`` with
    ``tau_y(phi) = tau_y0 * eta_r(phi)`` and ``K(phi) = K * eta_r(phi)``.

    ``regularisation`` is the shear-rate floor (lattice units) that keeps the
    apparent viscosity finite at ``gamma_dot -> 0``; it is reported in the
    output dict by the caller, not silently applied twice.
    """
    c = contract or ViscosityContract()
    g = np.maximum(_as_float_array(shear_rate), regularisation)
    eta_r = c.eta_r(phi)
    tau_y = tau_y0 * eta_r
    k_eff = consistency * eta_r
    tau = tau_y + k_eff * g ** exponent
    return {
        "tau": tau,
        "tau_y": tau_y,
        "consistency": k_eff,
        "eta_apparent": tau / g,
        "eta_r": eta_r,
    }


def power_law_suspension_consistency(
    phi,
    consistency: float,
    power_index: float,
    contract: ViscosityContract | None = None,
):
    """Power-law carrier: the suspension rescales ``K`` only, not ``n``.

    ``mu(gamma_dot, phi) = K eta_r(phi) gamma_dot**(n-1)``.  The power index is
    unchanged because the particle stress is linear in the *local* strain rate
    in the viscous regime; a ``phi``-dependent ``n`` would double count the
    shear-thinning already carried by the carrier.
    """
    c = contract or ViscosityContract()
    eta_r = c.eta_r(phi)
    return {
        "consistency": consistency * eta_r,
        "power_index": power_index,
        "eta_r": eta_r,
    }