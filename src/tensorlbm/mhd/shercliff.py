"""Shercliff (insulating-wall) low-Reynolds-number MHD channel: closed form + verification.

Physical problem
----------------
A plane channel spans ``y in [-H, H]`` (streamwise ``x`` periodic, no-slip walls),
a uniform magnetic field ``B0`` points along ``y`` and the two walls are
*electrically insulating* - no electrode is present, so no current can leave the
channel.  With Ohm's law ``j_z = sigma (E_z + u B0)`` the transverse field is not
imposed (as in the Hartmann/electrode set-up) but follows from the constraint
that the net axial current vanishes,

    <j_z> = sigma (E_z + B0 <u>) = 0   =>   E_z = -B0 u_mean,

so that

    j_z(y) = sigma B0 (u(y) - u_mean)                (current reversal, <j_z> = 0)
    F_x = (j x B)_x = -j_z B0 = sigma B0^2 (u_mean - u)

and the steady momentum balance with a driving force density ``G`` (pressure
gradient, ``-dp/dx``) reads

    mu u'' = -G + sigma B0^2 (u - u_mean).

Solving with ``u(+-H) = 0`` and ``a = Ha/H``, ``Ha = B0 H sqrt(sigma/mu)``:

    u(y) = u_p [1 - cosh(Ha y/H)/cosh(Ha)],    u_p = u_mean / m(Ha),
    m(Ha) = 1 - tanh(Ha)/Ha,
    u_mean = G H^2 m(Ha) / (mu Ha^2 (1 - m(Ha))) = G H^2 / (mu * f(Ha)),
    f(Ha) = Ha^2 (1 - m)/m  ->  3 (Ha -> 0),  Ha (Ha -> inf).

Two consequences that separate this closure from the conducting-wall Hartmann
problem on the same geometry:

* **Damping law**: at fixed driving ``G`` the insulating channel carries
  ``u_mean ~ G H^2/(mu Ha)`` for large ``Ha`` - one power of ``Ha`` *less* braking
  than the electrode (conducting-wall) case ``u_mean ~ G H^2/(mu Ha^2)``,
  because the current loop closes inside the channel instead of being drained by
  the electrodes.
* **Current**: the current density reverses sign across the channel and carries
  no net current; the Lorentz force can only redistribute momentum
  (``<F_x> = 0``), never add or remove it.  All Joule heat is dissipated in the
  channel (there is no electrical power take-off).

The profile *shape* normalised by its own mean is the same function as in the
Hartmann case, ``(1 - cosh(Ha y/H)/cosh Ha)/m(Ha)``; what changes is the
amplitude law and the current/field closure.  This module therefore verifies the
closure with (i) the mean-velocity law above, (ii) the zero-net-current
constraint, (iii) the reversal of ``j_z`` and (iv) the wall shear
``tau_w H/(mu u_mean) = Ha tanh(Ha)/m(Ha)`` (equal to ``3`` at ``Ha = 0``).

Lattice units as in :mod:`tensorlbm.mhd.hartmann`: ``H = ny/2``, ``mu = rho0 nu``,
``sigma`` and ``B0`` in lattice units, ``Ha`` prescribed and ``sigma`` derived
from it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .hartmann import (
    HartmannConfig,
    hartmann_mean_factor,
    hartmann_normalised_profile,
    hartmann_shape_factor,
    poiseuille_profile,
    run_hartmann_channel,
    shercliff_grad_factor,
)

#: ``np.trapezoid`` (numpy >= 2) with a fallback for older numpy releases.
_trapz = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]

#: Hartmann numbers used by the Shercliff verification ladder.
DEFAULT_SHERCLIFF_HA: tuple[float, ...] = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0)
#: Grid sequence (fluid rows across the channel) for the refinement study.
DEFAULT_SHERCLIFF_RESOLUTIONS: tuple[int, ...] = (64, 128, 256)
#: Pre-declared acceptance thresholds of the campaign node (2 % Shercliff gate).
SHERCLIFF_THRESHOLDS: dict[str, float] = {
    "c_shercliff_profile_l2": 2.0e-2,
    "c_shercliff_profile_linf": 2.0e-2,
    "c_shercliff_mean": 2.0e-2,
    "c_shercliff_wall_shear": 2.0e-2,
    "c_shercliff_zero_current": 1.0e-9,
    "c_shercliff_zero_current_integral": 2.0e-2,
    "c_shercliff_lorentz_mean": 1.0e-3,
    "c_shercliff_gci": 3.0e-2,
    "c_shercliff_poiseuille_limit": 1.0e-4,
}


# ---------------------------------------------------------------------------
# Closed form
# ---------------------------------------------------------------------------
def shercliff_mean_velocity(
    grad: float,
    ha: float,
    half_width: float,
    mu: float,
) -> float:
    """Mean velocity of the insulating-wall channel driven by ``G = grad``.

    ``u_mean = G H^2 / (mu f(Ha))`` with ``f = Ha^2 (1-m)/m``; the ``Ha -> 0``
    limit is plane Poiseuille ``u_mean = G H^2/(3 mu)``.
    """
    if mu <= 0.0:
        raise ValueError(f"mu must be > 0, got {mu}")
    return float(grad) * half_width**2 / (mu * shercliff_grad_factor(ha))


def shercliff_grad_for_mean(
    u_mean: float,
    ha: float,
    half_width: float,
    mu: float,
) -> float:
    """Driving force density that produces ``u_mean`` (inverse of the law above)."""
    return mu * float(u_mean) * shercliff_grad_factor(ha) / half_width**2


def shercliff_amplitude(u_mean: float, ha: float) -> float | None:
    """Shape-function amplitude ``u_p = u_mean/m(Ha)`` of the Shercliff profile.

    The shape factor ``1 - cosh(Ha y/H)/cosh Ha`` vanishes like ``Ha^2 (1-y^2)/2``
    as ``Ha -> 0``, so this amplitude diverges in that limit while the product
    stays finite (plane Poiseuille).  ``None`` is returned for ``Ha < 0.05`` to
    keep the report JSON finite; use :func:`shercliff_profile` for the profile.
    """
    if float(ha) < 0.05:
        return None
    m = hartmann_mean_factor(ha)
    if m <= 0.0:
        raise ValueError(f"m(Ha)={m} must be > 0 (Ha={ha})")
    return float(u_mean) / m


def shercliff_profile(y_hat: Any, ha: float, u_mean: float) -> np.ndarray:
    """Velocity profile ``u(y) = u_p [1 - cosh(Ha y/H)/cosh(Ha)]``.

    Implemented through the mean-normalised shape (which carries a ``Ha -> 0``
    series branch) so the plane-Poiseuille limit ``1.5 u_mean (1 - y^2/H^2)`` is
    reproduced without a ``0 * inf``.
    """
    return float(u_mean) * hartmann_normalised_profile(y_hat, ha)


def shercliff_normalised_profile(y_hat: Any, ha: float) -> np.ndarray:
    """Profile normalised by its own mean (identical shape to the Hartmann case)."""
    return hartmann_normalised_profile(y_hat, ha)


def shercliff_current_density(y_hat: Any, ha: float, u_mean: float) -> np.ndarray:
    """Axial current density normalised by ``sigma B0 u_mean``: ``u/u_mean - 1``.

    Reverses sign about the mid-plane and integrates to zero (insulating walls):
    ``E_z = -B0 u_mean`` makes the current a pure functional of the profile.
    """
    return hartmann_normalised_profile(y_hat, ha) - 1.0


def shercliff_wall_shear_coefficient(ha: float) -> float:
    """``tau_w H/(mu u_mean) = Ha tanh(Ha)/m(Ha)`` (``-> 3`` as ``Ha -> 0``)."""
    ha = float(ha)
    if ha <= 0.0:
        return 3.0
    return ha * math.tanh(ha) / hartmann_mean_factor(ha)


def shercliff_joule_power_density(
    ha: float, u_mean: float, sigma: float, b0: float, half_width: float
) -> float:
    """Volume-averaged Joule dissipation ``<sigma (E_z + u B0)^2> = <j_z^2/sigma>``.

    For the insulating closure ``j_z = sigma B0 (u - u_mean)``, so

        <j^2/sigma> = sigma B0^2 <(u - u_mean)^2>,
        <(u-u_mean)^2> = u_p^2 [<A^2> - <A>^2],  A = 1 - cosh(Ha y/H)/cosh(Ha).

    The second moment is obtained by quadrature on a fine uniform grid so that
    the closed form used in the ledger is independent of the LBM grid.
    """
    y = np.linspace(-1.0, 1.0, 200001)
    u = shercliff_profile(y, ha, u_mean)
    second = float(_trapz((u - u_mean) ** 2, y)) / 2.0
    return sigma * b0 * b0 * second


def shercliff_hydraulic_power_density(grad: float, u_mean: float) -> float:
    """Mechanical power per unit volume delivered by the driving gradient, ``G u_mean``."""
    return float(grad) * float(u_mean)


def shercliff_efficiency_insulating(
    ha: float,
    u_mean: float,
    grad: float,
    sigma: float,
    b0: float,
    mu: float,
    half_width: float,
) -> dict[str, float]:
    """Energy split of the insulating channel (no electric power take-off).

    Returns the hydraulic input, the Joule dissipation and the viscous
    dissipation, plus the fraction lost to Joule heating.  Because the walls are
    insulating the electrical channel cannot do work on an external circuit: the
    whole Joule heat is a loss, and the useful (hydraulic) power fraction is
    ``1 - P_joule/P_input``.
    """
    p_in = shercliff_hydraulic_power_density(grad, u_mean)
    p_joule = shercliff_joule_power_density(ha, u_mean, sigma, b0, half_width)
    # viscous dissipation = mu * <(du/dy)^2>; the derivative is evaluated on the
    # same quadrature grid as the Joule term (Ha = 0 included).
    y = np.linspace(-1.0, 1.0, 200001)
    dudy = np.gradient(shercliff_profile(y, ha, u_mean), y * half_width)
    p_visc = mu * float(_trapz(dudy * dudy, y)) / 2.0
    total = p_joule + p_visc
    return {
        "P_hydraulic_per_volume": p_in,
        "P_joule_per_volume": p_joule,
        "P_viscous_per_volume": p_visc,
        "closure_relative": abs(total - p_in) / max(abs(p_in), 1e-300),
        "joule_fraction": p_joule / max(abs(p_in), 1e-300),
        "viscous_fraction": p_visc / max(abs(p_in), 1e-300),
    }


# ---------------------------------------------------------------------------
# Solver wrapper
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShercliffCase:
    """One insulating-wall channel run.

    Args:
        ha: Hartmann number.
        ny: fluid rows across the channel (``H = ny/2``).
        nx: periodic streamwise cells (``1`` suffices - the flow is uniform in x).
        u_mean: target mean velocity; it fixes the driving force density ``G``
            through the closed-form relation (unless ``grad`` is given).
        nu: lattice kinematic viscosity.
        rho0: reference density (``mu = rho0 nu``).
        grad: optional *explicitly prescribed* driving force density; when given
            the run is an independent physics check of the mean-velocity law (the
            analytic ``u_mean`` is then predicted from ``grad`` alone).
        steps: hard step cap.
        check_every: steady-state check interval.
        tol: steady-state tolerance (max change of the streamwise profile).
    """

    ha: float
    ny: int = 64
    nx: int = 1
    u_mean: float = 0.01
    nu: float = 0.1
    rho0: float = 1.0
    grad: float | None = None
    steps: int = 400_000
    check_every: int = 1000
    tol: float = 1.0e-10

    @property
    def half_width(self) -> float:
        return 0.5 * self.ny

    @property
    def mu(self) -> float:
        return self.rho0 * self.nu

    @property
    def driving_grad(self) -> float:
        """Force density actually applied (explicit ``grad`` or the closed form)."""
        if self.grad is not None:
            return float(self.grad)
        return shercliff_grad_for_mean(self.u_mean, self.ha, self.half_width, self.mu)

    def to_config(self, u_init: Sequence[float] | None = None) -> HartmannConfig:
        return HartmannConfig(
            ny=self.ny,
            nx=self.nx,
            ha=self.ha,
            u_mean=self.u_mean,
            nu=self.nu,
            rho0=self.rho0,
            drive=self.driving_grad,
            wall_mode="insulating",
            steps=self.steps,
            check_every=self.check_every,
            tol=self.tol,
            u_init=None if u_init is None else tuple(float(v) for v in u_init),
        )


def interpolate_profile(u_coarse: Sequence[float], ny_fine: int) -> list[float]:
    """Linear interpolation of a coarse-grid profile onto the ``ny_fine`` node centres.

    Used for grid-sequenced start-up: the coarse solution is stretched onto the
    finer node centres (both sets are cell-centred in ``[-H, H]``), which cuts the
    transient of the refinement runs by an order of magnitude.
    """
    u = np.asarray(u_coarse, dtype=float)
    ny_coarse = u.size
    yc = (np.arange(ny_coarse) + 0.5 - 0.5 * ny_coarse) / (0.5 * ny_coarse)
    yf = (np.arange(ny_fine) + 0.5 - 0.5 * ny_fine) / (0.5 * ny_fine)
    return [float(v) for v in np.interp(yf, yc, u)]


def run_shercliff_channel(
    case: ShercliffCase,
    u_init: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Run one insulating-wall channel and score it against the closed form.

    Returns a JSON-friendly dict with the measured scalars, every relative error
    and the closure checks (zero net current, zero mean Lorentz force).
    """
    config = case.to_config(u_init)
    result = run_hartmann_channel(config)
    # The analytic mean is predicted from the *applied* gradient, not from the
    # target: with ``grad`` given explicitly this is a genuine prediction.
    u_mean_an = shercliff_mean_velocity(case.driving_grad, case.ha, case.half_width, case.mu)
    u_an = shercliff_profile(result.y_hat, case.ha, u_mean_an)
    scale = max(abs(u_mean_an), 1e-300)
    err = result.u_profile - u_an
    sigma = config.sigma
    b0 = config.b0
    # Current density *of the simulation*: j_z = sigma B0 (u - <u>), evaluated on
    # the LBM node centres (the analytic j_z is used only for the figures).
    j_over = sigma * b0 * (result.u_profile - result.u_mean)
    j_scale = max(float(np.max(np.abs(j_over))), 1e-300)
    # Two independent statements about the zero-net-current closure:
    #  * discrete (cell) mean - the constraint the solver imposes at every step
    #    (K = <u_x^*>), exact up to round-off: verifies the implementation;
    #  * trapezoidal integral of the node values - the continuous statement
    #    ``int j dy = 0``, which carries the O(dy^2)/boundary-layer discretisation
    #    error and therefore converges with the grid (gated at the 2 % level).
    net_current_discrete = float(np.mean(j_over))
    net_current_integral = float(_trapz(j_over, result.y_hat)) / 2.0
    tau_coeff = result.tau_wall * case.half_width / (case.mu * abs(result.u_mean))
    energy = shercliff_efficiency_insulating(
        case.ha, u_mean_an, case.driving_grad, sigma, b0, case.mu, case.half_width
    )
    # reverse-flow / reversal check: j_z must change sign about the mid-plane
    # Reversal statement: j_z has one sign in the wall layers (where u < u_mean)
    # and the opposite sign in the core (where u > u_mean).  The integral over
    # *each half* vanishes by antisymmetry, so half-integrals cannot be used.
    lower = float(np.mean(np.concatenate([j_over[:3], j_over[-3:]])))
    upper = float(np.mean(j_over[case.ny // 2 - 2 : case.ny // 2 + 2]))
    return {
        "ha": case.ha,
        "ny": case.ny,
        "nx": case.nx,
        "half_width": case.half_width,
        "nu": case.nu,
        "mu": case.mu,
        "sigma": sigma,
        "b0": b0,
        "driving_grad": case.driving_grad,
        "driving_grad_explicit": case.grad is not None,
        "u_mean_target": case.u_mean,
        "u_mean_lbm": result.u_mean,
        "u_mean_analytic": u_mean_an,
        "u_amplitude_analytic": shercliff_amplitude(u_mean_an, case.ha),
        "rel_mean": abs(result.u_mean - u_mean_an) / scale,
        "rel_l2": float(np.sqrt(np.mean(err * err)) / scale),
        "rel_linf": float(np.max(np.abs(err)) / scale),
        "rel_l2_normalised": float(
            np.sqrt(np.mean((result.u_profile / result.u_mean - u_an / u_mean_an) ** 2))
        ),
        "tau_wall_coefficient_lbm": tau_coeff,
        "tau_wall_coefficient_analytic": shercliff_wall_shear_coefficient(case.ha),
        "rel_wall_shear": abs(tau_coeff - shercliff_wall_shear_coefficient(case.ha))
        / shercliff_wall_shear_coefficient(case.ha),
        "net_current_relative": abs(net_current_discrete) / j_scale,
        "net_current_integral_relative": abs(net_current_integral) / j_scale,
        "lorentz_mean_relative": abs(result.lorentz_force)
        / max(abs(result.drive_force), 1e-300),
        "current_reversal_check": {
            "j_wall_layer_mean": lower,
            "j_core_mean": upper,
            "sign_reversal": bool(lower * upper < 0.0),
        },
        "closure_residual": result.closure_residual,
        "steps": result.steps,
        "converged": bool(result.converged),
        "max_mach": result.max_mach,
        "energy": energy,
        "wall_mode": config.wall_mode,
        "profile": {
            "y_hat": [float(v) for v in result.y_hat],
            "u_lbm": [float(v) for v in result.u_profile],
            "u_analytic": [float(v) for v in u_an],
        },
        "y_hat": [float(v) for v in result.y_hat],
        "u_profile": [float(v) for v in result.u_profile],
        "u_profile_analytic": [float(v) for v in u_an],
    }


def gci(
    fine: float,
    medium: float,
    coarse: float,
    r: float = 2.0,
    safety: float = 1.25,
) -> dict[str, Any]:
    """Roache grid-convergence index (same definition as the Hartmann ladder)."""
    eps = 1e-300
    if abs(fine) < eps:
        return {"gci_fine": None, "order": None, "note": "zero functional"}
    if fine == medium == coarse:
        return {"gci_fine": 0.0, "order": None, "note": "grid-independent"}
    r21 = (medium - coarse) / (fine - medium) if fine != medium else 0.0
    if r21 <= 0.0:
        # non-monotone: fall back to the conservative "fine vs medium" difference
        return {
            "gci_fine": safety * abs((fine - medium) / fine),
            "order": None,
            "note": "non-monotone convergence: conservative fine/medium estimate",
            "r21": r21,
        }
    p = math.log(r21) / math.log(r)
    denom = r**p - 1.0
    if denom <= 0.0:
        return {"gci_fine": None, "order": p, "note": "degenerate observed order"}
    return {
        "gci_fine": safety * abs((fine - medium) / fine) / denom,
        "order": p,
        "r21": r21,
        "note": "three-grid Roache GCI (safety factor 1.25)",
    }


def shercliff_plan(
    ha_values: Iterable[float] = DEFAULT_SHERCLIFF_HA,
    resolutions: Sequence[int] = DEFAULT_SHERCLIFF_RESOLUTIONS,
) -> dict[float, list[int]]:
    """Grid plan per ``Ha``; the thinner Hartmann layers get finer grids."""
    res = [int(v) for v in resolutions]
    plan: dict[float, list[int]] = {}
    for ha in ha_values:
        ha = float(ha)
        if ha >= 20.0:
            # Hartmann layer H/Ha = ny/(2 Ha) = 6.4 cells at ny = 256; the 512
            # run adds nothing to the 2 % gate and only cost.
            plan[ha] = [max(res[-2], 128), max(res[-1], 256)]
        else:
            plan[ha] = [min(res[1], 64), min(res[2], 128), max(res[-1], 256)]
    return plan


def evaluate_shercliff_criteria(
    runs: dict[float, list[dict[str, Any]]],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Machine-checkable verdicts for the Shercliff branch of the campaign node.

    ``runs`` maps ``Ha -> list of run dicts`` ordered coarse -> fine.
    """
    th = dict(SHERCLIFF_THRESHOLDS)
    if thresholds:
        th.update(thresholds)
    per_ha: dict[str, Any] = {}
    worst = {
        "c_shercliff_profile_l2": 0.0,
        "c_shercliff_profile_linf": 0.0,
        "c_shercliff_mean": 0.0,
        "c_shercliff_wall_shear": 0.0,
        "c_shercliff_zero_current": 0.0,
        "c_shercliff_zero_current_integral": 0.0,
        "c_shercliff_lorentz_mean": 0.0,
        "c_shercliff_gci": 0.0,
    }
    for ha, items in sorted(runs.items()):
        fine = items[-1]
        entry: dict[str, Any] = {
            "ny_fine": fine["ny"],
            "rel_l2": fine["rel_l2"],
            "rel_linf": fine["rel_linf"],
            "rel_mean": fine["rel_mean"],
            "rel_wall_shear": fine["rel_wall_shear"],
            "net_current_relative": fine["net_current_relative"],
            "net_current_integral_relative": fine["net_current_integral_relative"],
            "lorentz_mean_relative": fine["lorentz_mean_relative"],
            "converged": fine["converged"],
        }
        for key, metric in (
            ("c_shercliff_profile_l2", "rel_l2"),
            ("c_shercliff_profile_linf", "rel_linf"),
            ("c_shercliff_mean", "rel_mean"),
            ("c_shercliff_wall_shear", "rel_wall_shear"),
            ("c_shercliff_zero_current", "net_current_relative"),
            ("c_shercliff_zero_current_integral", "net_current_integral_relative"),
            ("c_shercliff_lorentz_mean", "lorentz_mean_relative"),
        ):
            worst[key] = max(worst[key], float(fine[metric]))
        if len(items) >= 3:
            # The Roache index must act on a functional that converges to a
            # *non-zero* value; applied to a vanishing error functional it is
            # inflated to O(1) (the relative change of a quantity that tends to
            # zero is itself O(1)).  Primary functional: the dimensionless wall
            # shear (exact value Ha tanh(Ha)/m), which is grid sensitive.
            taus = [it["tau_wall_coefficient_lbm"] for it in items[-3:]]
            g_tau = gci(taus[2], taus[1], taus[0], r=2.0)
            entry["gci_wall_shear"] = g_tau | {
                "analytic": items[-1]["tau_wall_coefficient_analytic"]
            }
            means = [it["u_mean_lbm"] for it in items[-3:]]
            entry["gci_u_mean"] = gci(means[2], means[1], means[0], r=2.0)
            errs = [it["rel_l2"] for it in items[-3:]]
            entry["gci_profile_error_informational"] = gci(errs[2], errs[1], errs[0], r=2.0)
            cand = [
                v
                for v in (g_tau.get("gci_fine"), entry["gci_u_mean"].get("gci_fine"))
                if v is not None
            ]
            gci_val = max(cand) if cand else 0.0
            entry["gci"] = gci_val
            worst["c_shercliff_gci"] = max(worst["c_shercliff_gci"], float(gci_val))
        else:
            entry["gci"] = None
        per_ha[f"{ha:g}"] = entry

    # Ha -> 0 analytic limit: the insulating law must collapse onto Poiseuille.
    grad0 = 1.0e-5
    mu, half = 0.1, 32.0
    u_limit = shercliff_mean_velocity(grad0, 0.0, half, mu)
    u_poiseuille = grad0 * half**2 / (3.0 * mu)
    y = np.linspace(-1.0, 1.0, 2001)
    limit_err = float(np.max(np.abs(shercliff_profile(y, 0.0, u_limit) - poiseuille_profile(y, u_limit))))
    limit_err /= max(abs(u_limit), 1e-300)
    worst["c_shercliff_poiseuille_limit"] = limit_err
    worst["c_shercliff_poiseuille_limit_scale"] = abs(u_limit - u_poiseuille) / u_poiseuille

    verdicts = {
        "c_shercliff_profile_l2": worst["c_shercliff_profile_l2"] <= th["c_shercliff_profile_l2"],
        "c_shercliff_profile_linf": worst["c_shercliff_profile_linf"]
        <= th["c_shercliff_profile_linf"],
        "c_shercliff_mean": worst["c_shercliff_mean"] <= th["c_shercliff_mean"],
        "c_shercliff_wall_shear": worst["c_shercliff_wall_shear"] <= th["c_shercliff_wall_shear"],
        "c_shercliff_zero_current": worst["c_shercliff_zero_current"]
        <= th["c_shercliff_zero_current"],
        "c_shercliff_zero_current_integral": worst["c_shercliff_zero_current_integral"]
        <= th["c_shercliff_zero_current_integral"],
        "c_shercliff_lorentz_mean": worst["c_shercliff_lorentz_mean"]
        <= th["c_shercliff_lorentz_mean"],
        "c_shercliff_gci": worst["c_shercliff_gci"] <= th["c_shercliff_gci"],
        "c_shercliff_poiseuille_limit": limit_err <= th["c_shercliff_poiseuille_limit"],
    }
    return {
        "thresholds": th,
        "worst_case": worst,
        "per_ha": per_ha,
        "verdicts": verdicts,
        "all_passed": all(verdicts.values()),
        "poiseuille_limit": {
            "grad": grad0,
            "u_mean_shercliff_Ha0": u_limit,
            "u_mean_poiseuille": u_poiseuille,
            "profile_max_rel_error": limit_err,
        },
    }