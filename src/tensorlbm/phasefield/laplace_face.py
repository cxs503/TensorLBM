r"""Young-Laplace activation face for the D3Q19 free-energy Cahn-Hilliard path.

Why this module exists
----------------------
``tensorlbm.multiphase3d.free_energy_step_3d`` is the free-energy (Cahn-Hilliard)
counterpart of the Shan-Chen path, and the MCMP migration plan recorded in
``docs/MCMP_STRATIFIED_POISEUILLE_R4.md`` §6 lists three *activation* tests that
must hold before the free-energy model can replace the pseudopotential one:

* (a) a flat interface reproduces the analytic ``tanh`` profile with the width
  implied by the free-energy coefficients,
* (b) the capillary pressure follows Young-Laplace ``dp = sigma / R``,
* (c) the static spurious current stays below a pinned gate.

The library currently marks the free-energy Laplace chain as *withheld*
(``docs/phasefield_static_droplet_r1.md``: ``status="withheld"``,
``physical_acceptance=False``).  Withheld is an honest status but not yet an
*attributable* one: the failure modes are not separated, so the record cannot
say whether a measured ``dp`` is a Laplace measurement at all.  This module
turns the withheld status into an attributable one by measuring, per run,

1. the **equilibrium residual** (spread of the chemical potential ``mu`` over
   the droplet: at Cahn-Hilliard equilibrium ``mu`` must be uniform),
2. the **spurious-current level** ``max|u|`` (a static droplet must be at rest),
3. two *independent* routes to the surface tension of the measured field -
   the analytic coefficient formula and the gradient-energy integral
   ``sigma = integral kappa (dphi/dr)^2 dr`` along a radial cut - so a Laplace
   mismatch can be attributed to the coefficient convention or to the state,
4. the **stress-divergence consistency** of the two force conventions that the
   library ships (``force_minus_phi_grad_mu`` = ``-phi grad mu``, the one used by
   the production step, and ``force_mu_grad_phi`` = ``-mu grad phi``).  Their
   radial integrals across the interface are compared with the measured
   ``cs^2 * d rho`` jump, which is what the lattice momentum balance requires.

Analytic reference (double-well free energy)
--------------------------------------------
With ``f(phi) = -A phi^2/2 + B phi^4/4`` and gradient term ``(kappa/2)|grad phi|^2``
the chemical potential is ``mu = -A phi + B phi^3 - kappa lap(phi)`` and the
coexistence / interface / tension quantities are *derived* (not fitted):

.. math::

    phi_0 = sqrt(A/B), \quad \xi = sqrt(2 kappa / A), \quad
    \sigma = \frac{2 A^2}{3 B}\sqrt{\frac{2\kappa}{A}} = \sqrt{\frac{8\kappa A^3}{9 B^2}}

The two forms of ``sigma`` are algebraically identical; :func:`analytic_reference`
returns both so that a regression test can pin the identity instead of one
hand-copied constant.  ``sigma`` is also recovered independently from the
gradient-energy integral of the analytic profile, ``integral kappa phi'^2 dx =
(2 B / 3) phi_0^4 xi``, which :func:`sigma_from_profile` measures on any field
(analytic or simulated) so the two routes can be compared.

Honest status
-------------
Everything here is *measurement* plus *guardrails*: no function in this module
claims a Laplace PASS.  The verdict of a run is
``{"status": "withheld"|"measured", "codes": [...]}`` where ``"withheld"`` means
the state is not an equilibrium state (see :func:`equilibrium_gate`) and a
``dp`` computed from it must not be reported as a Laplace number.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from ..d3q19 import equilibrium3d
from ..multiphase3d import free_energy_step_3d, init_free_energy_g_3d
from ..solver3d import stream3d
from .free_energy import (
    DoubleWellFreeEnergy,
    force_minus_phi_grad_mu,
    force_mu_grad_phi,
)
from .operators import central_gradient_3d

CS2 = 1.0 / 3.0
DENSITY_RATIO_NOTE = (
    "the density is uniform here (rho = 1), so dp = cs^2 d rho and no "
    "density-ratio contribution enters the measurement"
)


# ---------------------------------------------------------------------------
# Analytic reference
# ---------------------------------------------------------------------------


def analytic_reference(A: float, B: float, kappa: float) -> dict[str, float]:
    """Coexistence amplitude, interface width and surface tension.

    Args:
        A: Double-well quadratic coefficient (must be > 0, ``phi_0 = sqrt(A/B)``).
        B: Double-well quartic coefficient (must be > 0).
        kappa: Gradient-energy coefficient (must be > 0).

    Returns:
        ``{"phi0", "xi", "sigma", "sigma_closed_form", "sigma_identity_residual"}``
        where ``sigma`` is ``2 A^2/(3 B) * sqrt(2 kappa / A)`` and
        ``sigma_closed_form`` is ``sqrt(8 kappa A^3 / (9 B^2))``; the residual is
        their relative difference (algebraically zero, numerically ~1e-16).
    """
    for name, value in (("A", A), ("B", B), ("kappa", kappa)):
        if not (value > 0.0):
            raise ValueError(f"{name} must be positive, got {value!r}")
    phi0 = math.sqrt(A / B)
    xi = math.sqrt(2.0 * kappa / A)
    sigma = (2.0 * A * A / (3.0 * B)) * xi
    sigma_closed = math.sqrt(8.0 * kappa * A**3 / (9.0 * B * B))
    residual = abs(sigma - sigma_closed) / sigma_closed
    return {
        "phi0": phi0,
        "xi": xi,
        "sigma": sigma,
        "sigma_closed_form": sigma_closed,
        "sigma_identity_residual": residual,
    }


def tanh_profile(r: torch.Tensor, *, R: float, phi0: float, xi: float) -> torch.Tensor:
    """Analytic planar interface profile wrapped on a sphere of radius ``R``."""
    return phi0 * torch.tanh((R - r) / xi)


def analytic_sigma_gradient_integral(A: float, B: float, kappa: float) -> float:
    """``integral kappa (dphi/dr)^2 dr`` of the analytic profile (closed form).

    ``= (2 B / 3) phi_0^4 xi`` with ``phi_0^4 = (A/B)^2``, i.e. the same number as
    :func:`analytic_reference`, obtained by a different route.
    """
    phi0 = math.sqrt(A / B)
    xi = math.sqrt(2.0 * kappa / A)
    return (2.0 * B / 3.0) * phi0**4 * xi


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def radial_setup(n: int, *, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, int]:
    """Return ``(r, centre)`` radius field on an ``n^3`` periodic grid."""
    coords = torch.arange(n, dtype=dtype)
    z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")
    centre = n // 2
    r = torch.sqrt((x - centre) ** 2 + (y - centre) ** 2 + (z - centre) ** 2)
    return r, centre


def init_static_droplet(
    n: int,
    *,
    R: float,
    A: float,
    B: float,
    kappa: float,
    rho0: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equilibrium-initialised static droplet: ``(f, g, phi)``.

    ``f`` starts from the isothermal equilibrium at ``rho0`` and zero velocity;
    ``g`` starts from the free-energy equilibrium of the analytic ``tanh`` field,
    so any subsequent drift is dynamics, not an initialisation shock.
    """
    r, _ = radial_setup(n, dtype=dtype)
    ref = analytic_reference(A, B, kappa)
    phi = tanh_profile(r, R=R, phi0=ref["phi0"], xi=ref["xi"])
    zero = torch.zeros_like(phi)
    f = equilibrium3d(torch.full_like(phi, rho0), zero, zero, zero)
    g = init_free_energy_g_3d(phi)
    return f, g, phi


# ---------------------------------------------------------------------------
# Diagnostics on a state
# ---------------------------------------------------------------------------


def chemical_potential_spread(
    phi: torch.Tensor, *, A: float, B: float, kappa: float
) -> dict[str, float]:
    """Range of the chemical potential over the domain (equilibrium residual).

    At Cahn-Hilliard equilibrium ``mu`` is uniform, so ``mu_max - mu_min`` is a
    direct measure of how far the state is from equilibrium.  It is reported
    relative to the natural scale ``A phi_0`` of the double well.
    """
    model = DoubleWellFreeEnergy(A=A, B=B, kappa=kappa)
    mu = model.chemical_potential(phi, boundary="periodic")
    spread = float(mu.max() - mu.min())
    scale = A * math.sqrt(A / B)
    return {
        "mu_min": float(mu.min()),
        "mu_max": float(mu.max()),
        "mu_spread": spread,
        "mu_spread_rel": spread / scale,
    }


def sigma_from_profile(
    phi: torch.Tensor,
    *,
    centre: int,
    kappa: float,
    axis: int = 2,
) -> float:
    """Surface tension of a *measured* field: ``integral kappa (dphi/dr)^2 dr``.

    A central-difference gradient is taken along ``axis`` through ``centre`` and
    the gradient-energy density is integrated over **one** interface, i.e. along
    the outward half-line ``centre -> centre + n/2``; a full axis cut crosses the
    interface twice and returns ``2 sigma`` (a factor that is easy to miss: the
    first version of this function did exactly that, and the bench caught it as a
    +27..+88 % bias against the analytic profile).  This is a route to ``sigma``
    that uses neither the free-energy coefficients nor ``dp``.
    """
    if axis == 2:
        cut = phi[centre, centre, :]
    elif axis == 1:
        cut = phi[centre, :, centre]
    else:
        cut = phi[:, centre, centre]
    n = cut.shape[0]
    half = cut[centre : centre + n // 2]
    grad = 0.5 * (half[2:] - half[:-2])
    return float(kappa * torch.sum(grad * grad))


def bulk_phase_values(phi: torch.Tensor) -> dict[str, float]:
    """Mean phase field inside (``phi > 0.9``) and outside (``phi < -0.9``)."""
    inside = phi > 0.9
    outside = phi < -0.9
    return {
        "phi_inside": float(phi[inside].mean()) if bool(inside.any()) else float("nan"),
        "phi_outside": float(phi[outside].mean()) if bool(outside.any()) else float("nan"),
    }


def measure_delta_p(
    rho: torch.Tensor,
    phi: torch.Tensor,
) -> dict[str, float]:
    """``dp = cs^2 (rho_inside - rho_outside)`` with the interface shell excluded.

    Masks are ``phi > 0.9`` (droplet) and ``phi < -0.9`` (bulk), the same
    convention as :func:`tensorlbm.allen_cahn_lbm.laplace_pressure_jump`, applied
    here to a uniform-density droplet so that ``dp`` cannot be confused with a
    density-ratio effect.
    """
    inside = phi > 0.9
    outside = phi < -0.9
    if not bool(inside.any()) or not bool(outside.any()):
        raise ValueError("droplet and bulk masks must both be non-empty")
    p_in = CS2 * float(rho[inside].mean())
    p_out = CS2 * float(rho[outside].mean())
    d_rho = float(rho[inside].mean() - rho[outside].mean())
    return {
        "p_inside": p_in,
        "p_outside": p_out,
        "dp_measured": p_in - p_out,
        "delta_rho": d_rho,
    }


def stress_divergence_integrals(
    phi: torch.Tensor,
    *,
    A: float,
    B: float,
    kappa: float,
) -> dict[str, float]:
    """Radial integral of each shipped force convention, radially binned.

    For a spherically symmetric state the momentum balance reads
    ``cs^2 d rho = integral F_r dr`` across the interface, so the radial integral
    of ``F_r`` is the ``dp`` that each force convention *predicts*.  Both force
    fields come from the production functions
    (:func:`tensorlbm.phasefield.free_energy.force_minus_phi_grad_mu` and
    :func:`~tensorlbm.phasefield.free_energy.force_mu_grad_phi`), evaluated on the
    same state - this is a post-processing comparison, not a second solver.

    Returns:
        ``{"integrals_phi_grad_mu", "integrals_mu_grad_phi"}`` plus the analytic
        capillary pressure passed in by the caller through ``A/B/kappa`` and the
        per-shell arrays are *not* returned (totals only, see the bench for the
        profiles).
    """
    model = DoubleWellFreeEnergy(A=A, B=B, kappa=kappa)
    mu = model.chemical_potential(phi, boundary="periodic")
    f_phi_mu = force_minus_phi_grad_mu(phi, mu, boundary="periodic")
    f_mu_phi = force_mu_grad_phi(phi, mu, boundary="periodic")
    n = phi.shape[-1]
    r, centre = radial_setup(n, dtype=phi.dtype)
    # radial unit vector, guarded at the centre
    dz = torch.arange(n, dtype=phi.dtype).view(n, 1, 1) - centre
    dy = torch.arange(n, dtype=phi.dtype).view(1, n, 1) - centre
    dx = torch.arange(n, dtype=phi.dtype).view(1, 1, n) - centre
    rs = torch.clamp(r, min=1e-12)
    er = (dx / rs, dy / rs, dz / rs)

    def _total(force: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        fr = force[0] * er[0] + force[1] * er[1] + force[2] * er[2]
        return float(fr.sum())

    return {
        "integral_phi_grad_mu": _total(f_phi_mu),
        "integral_mu_grad_phi": _total(f_mu_phi),
    }


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


def equilibrium_gate(
    *,
    u_max: float,
    mu_spread_rel: float,
    finite: bool,
    u_tol: float = 1.0e-3,
    mu_tol: float = 1.0e-2,
) -> dict[str, Any]:
    """Decide whether a ``dp`` measured from this state may be reported.

    Three independent conditions: the run must be finite, the chemical potential
    must be uniform to ``mu_tol`` (relative to ``A phi_0``), and the droplet must
    be at rest to ``u_tol``.  Any violation *withholds* the Laplace measurement
    and records the code, so a failing case can never be quoted as a Laplace
    number.

    Returns:
        ``{"status": "measured"|"withheld", "codes": [...]}``.
    """
    codes: list[str] = []
    if not finite:
        codes.append("NON_FINITE")
    if mu_spread_rel > mu_tol:
        codes.append("MU_NOT_UNIFORM")
    if u_max > u_tol:
        codes.append("NOT_AT_REST")
    return {"status": "withheld" if codes else "measured", "codes": codes}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_static_droplet(
    *,
    n: int,
    R: float,
    A: float = 0.1,
    B: float = 0.1,
    kappa: float = 0.05,
    Gamma: float = 0.1,
    tau_f: float = 1.0,
    tau_g: float = 0.7,
    steps: int = 400,
    check_every: int = 50,
    freeze_phi: bool = False,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Relax a static droplet with the production free-energy step and measure it.

    ``freeze_phi=True`` keeps the order-parameter distribution fixed at its
    analytic initial value and advances only the momentum equation.  That
    separates *transport* from *force balance*: the Cahn-Hilliard evolution
    minimises interfacial area, so a droplet in a periodic box is not a
    stationary state of the transport equation at all (it is a downhill path
    towards a flat slab), whereas the capillary pressure is a property of the
    force balance at fixed ``phi``.  A quench run therefore answers the Laplace
    question on its own terms and attributes any failure either to the force
    convention or to the transport.

    The step/stream pair is the production one
    (``multiphase3d.free_energy_step_3d`` + ``solver3d.stream3d``); the initial
    field is the analytic ``tanh`` droplet, so the run starts in the state the
    model claims to hold.  ``u_max`` is sampled every ``check_every`` steps - a
    parasitic vortex field is a plateau, not a decay to zero, so the last value is
    reported rather than a convergence flag.

    Returns:
        Dict with ``analytic`` (from :func:`analytic_reference`), ``history``,
        ``metrics`` (``dp``, ``sigma_from_field``, ``mu_spread``, ``u_max``,
        ``phi`` bulks) and ``gate`` (from :func:`equilibrium_gate`).
    """
    from ..d3q19 import macroscopic3d

    ref = analytic_reference(A, B, kappa)
    f, g, phi_init = init_static_droplet(n, R=R, A=A, B=B, kappa=kappa, dtype=dtype)
    history: list[list[float]] = []
    finite = True
    u_max = float("nan")
    for step in range(1, steps + 1):
        f_new, g_new = free_energy_step_3d(
            f, g, tau_f=tau_f, tau_g=tau_g, A=A, B=B, kappa=kappa, Gamma=Gamma
        )
        f = stream3d(f_new)
        if not freeze_phi:
            g = stream3d(g_new)
        if step % check_every == 0 or step == steps:
            rho, ux, uy, uz = macroscopic3d(f)
            u_max = float(torch.sqrt(ux * ux + uy * uy + uz * uz).max())
            finite = finite and math.isfinite(u_max) and bool(torch.isfinite(rho).all())
            history.append([float(step), u_max])
            if not finite:
                break

    result: dict[str, Any] = {
        "config": {
            "n": n,
            "R": R,
            "A": A,
            "B": B,
            "kappa": kappa,
            "Gamma": Gamma,
            "tau_f": tau_f,
            "tau_g": tau_g,
            "steps": steps,
            "freeze_phi": freeze_phi,
            "dtype": str(dtype).split(".")[-1],
            "density_ratio_note": DENSITY_RATIO_NOTE,
        },
        "analytic": ref,
        "history": history,
        "finite": finite,
    }
    if not finite:
        result["metrics"] = None
        result["gate"] = equilibrium_gate(u_max=u_max, mu_spread_rel=float("nan"), finite=False)
        return result

    rho, ux, uy, uz = macroscopic3d(f)
    phi = g.sum(dim=0)
    _, centre = radial_setup(n, dtype=dtype)
    mu = chemical_potential_spread(phi, A=A, B=B, kappa=kappa)
    if bool((phi > 0.9).any()) and bool((phi < -0.9).any()):
        dp = measure_delta_p(rho, phi)
    else:
        # the phase field no longer separates into two bulk phases: there is no
        # droplet to measure a capillary pressure on.
        dp = {
            "p_inside": float("nan"),
            "p_outside": float("nan"),
            "dp_measured": float("nan"),
            "delta_rho": float("nan"),
        }
        result["gate"] = equilibrium_gate(
            u_max=u_max, mu_spread_rel=mu["mu_spread_rel"], finite=True
        )
        result["gate"]["codes"] = list(result["gate"]["codes"]) + ["NO_DROPLET"]
    sigma_field = sigma_from_profile(phi, centre=centre, kappa=kappa)
    dp_expected = ref["sigma"] / R
    metrics = {
        "u_max": u_max,
        "phi_max": float(phi.max()),
        "phi_min": float(phi.min()),
        "phi_max_drift": float(phi.max() - phi_init.max()),
        "phi_total_rel_drift": float(
            (phi.sum() - phi_init.sum()).abs() / phi_init.sum().abs()
        ),
        "dp_measured": dp["dp_measured"],
        "delta_rho": dp["delta_rho"],
        "dp_laplace": dp_expected,
        "dp_rel_error": (
            abs(dp["dp_measured"] - dp_expected) / dp_expected
            if dp["dp_measured"] == dp["dp_measured"]
            else float("nan")
        ),
        "sigma_from_field": sigma_field,
        "sigma_analytic": ref["sigma"],
        "sigma_field_rel_error": abs(sigma_field - ref["sigma"]) / ref["sigma"],
        **{k: v for k, v in mu.items()},
        **bulk_phase_values(phi),
        **stress_divergence_integrals(phi, A=A, B=B, kappa=kappa),
    }
    result["metrics"] = metrics
    result["gate"] = equilibrium_gate(
        u_max=u_max, mu_spread_rel=mu["mu_spread_rel"], finite=True
    )
    return result


def stability_probe(
    *,
    n: int = 24,
    R: float = 6.0,
    A: float = 0.1,
    B: float = 0.1,
    kappa: float,
    Gamma: float,
    tau_g: float = 0.7,
    steps: int = 100,
    u_blowup: float = 1.0e-1,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Classify one parameter point as ``stable`` / ``diverged`` / ``non_finite``.

    Cheap screening used to map the admissible region of ``(kappa, Gamma, tau_g)``
    before spending a Laplace run on it.
    """
    from ..d3q19 import macroscopic3d

    f, g, phi = init_static_droplet(n, R=R, A=A, B=B, kappa=kappa, dtype=dtype)
    for _ in range(steps):
        f, g = free_energy_step_3d(
            f, g, tau_f=1.0, tau_g=tau_g, A=A, B=B, kappa=kappa, Gamma=Gamma
        )
        f, g = stream3d(f), stream3d(g)
    rho, ux, uy, uz = macroscopic3d(f)
    u_max = torch.sqrt(ux * ux + uy * uy + uz * uz).max()
    finite = bool(torch.isfinite(rho).all() and torch.isfinite(u_max))
    # A run can stay finite while exploding (measured: u_max = 1.1e12 while every
    # entry stayed finite), so "finite" alone is not a stability verdict.
    if not finite:
        status = "non_finite"
    elif float(u_max) > u_blowup:
        status = "diverged"
    else:
        status = "stable"
    return {
        "kappa": kappa,
        "Gamma": Gamma,
        "tau_g": tau_g,
        "xi": math.sqrt(2.0 * kappa / A),
        "u_max": float(u_max) if finite else float("nan"),
        "finite": finite,
        "status": status,
    }