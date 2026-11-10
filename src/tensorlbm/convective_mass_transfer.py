"""Convective mass transfer verification surface (laminar forced convection).

Verification ladder for the **mass-transfer** (species) side of the library: the
passive-scalar lattice Boltzmann solver of :mod:`tensorlbm.scalar_transport`
(D2Q5 two-relaxation-time advection-diffusion) is driven by a known momentum
field and its wall flux is compared against closed-form mass-transfer solutions
and established correlations.

Three independent hard anchors are provided:

1. **Leveque / Graetz problem** -- linear shear ``u = a y`` over a wall held at a
   constant concentration.  The similarity solution (Leveque 1928) is exact::

       h_m(x) = C_L D (a / (9 D x))^{1/3},  C_L = 1 / (9^{1/3} Gamma(4/3))
       Sh_x   = h_m x / D = C_L (a x^2 / D)^{1/3}

   i.e. the local transfer coefficient decays as ``h_m ~ x^{-1/3}`` (the
   "-1/3 Leveque law") and ``Sh_x ~ x^{2/3}``.  The exponent fit is a
   one-parameter discrimination test of the advection-diffusion operator: a
   scheme with the wrong numerical diffusivity rotates the exponent away from
   ``2/3`` (equivalently ``-1/3`` for the flux).

2. **Blasius flat plate** -- ``Sh_x = 0.332 Re_x^{1/2} Sc^{1/3}`` and
   ``Sh_L = 0.664 Re_L^{1/2} Sc^{1/3}``.  The velocity field is the exact
   Blasius similarity field (:func:`blasius_velocity_field`), so the momentum
   solution contributes no error and the test isolates the scalar operator.  A
   fully coupled D2Q9-momentum variant is available through
   :func:`solve_channel_plate_momentum` for the three-way analogy check.

3. **Cylinder in cross flow** -- time-mean ``Sh`` against
   ``Sh = 2 + 0.6 Re^{1/2} Sc^{1/3}`` (Ranz & Marshall 1952, derived for a
   *sphere*; kept because it is the campaign reference) and, for a fair
   geometric match, against Hilpert (1933) and Churchill-Bernstein (1977).
   The transfer rate comes from an **exact mass budget**: the cylinder cells are
   held at ``C_w`` and the mass that has to be injected into them per unit time
   *is* the transfer rate, so ``Sh = mdot / (pi D_AB (C_w - C_inf))`` (per unit
   depth) carries no surface reconstruction error.

Chilton-Colburn analogy (``j_D = j_H``)::

    j_D = St_D Sc^{2/3},   j_H = St_H Pr^{2/3},   St = Sh / (Re X)

Both are evaluated from the *same* velocity field and compared with the
skin-friction reference ``C_f / 2``, which is where the Colburn exponents come
from.

Evidence level: REAL -- every number is produced by the lattice update; the
analytic anchors are closed-form / ODE solutions.

References
----------
Leveque, M. A. (1928). Les lois de la transmission de chaleur par convection.
    *Ann. Mines* 13, 201-299.
Blasius, H. (1908). Grenzschichten in Fluessigkeiten mit kleiner Reibung.
    *Z. Math. Phys.* 56, 1-37.
Ranz, W. E., & Marshall, W. R. (1952). Evaporation from drops.
    *Chem. Eng. Prog.* 48, 141-146.
Hilpert, R. (1933). Waermeabgabe von geheizten Draehten und Rohren.
    *Forsch. Geb. Ingenieurwes.* 4, 215-224.
Churchill, S. W., & Bernstein, M. (1977). A correlating equation for forced
    convection from gases and liquids to a circular cylinder in cross flow.
    *J. Heat Transfer* 99, 300-306.
Chilton, T. H., & Colburn, A. P. (1934). Mass transfer (absorption)
    coefficients. *Ind. Eng. Chem.* 26, 1183-1187.
Ginzburg, I. (2005). Equilibrium-type and link-type lattice Boltzmann models
    for generic advection and diffusion. *Adv. Water Resour.* 28, 1171-1195.
Yoshida, H., & Nagaoka, M. (2010). Multiple-relaxation-time lattice Boltzmann
    model for the convection and anisotropic diffusion equation.
    *J. Comput. Phys.* 229, 7774-7795.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from .boundaries import (
    bounce_back_cells,
    compute_obstacle_forces,
    cylinder_mask,
    make_channel_wall_mask,
)
from .d2q9 import equilibrium, macroscopic
from .scalar_transport import (
    W_D2Q5,
    collide_scalar_trt,
    equilibrium_scalar_d2q5,
    scalar_field,
    stream_scalar,
    tau_from_diffusivity,
    tau_symmetric_from_magic,
)
from .solver import collide_bgk, stream as stream_momentum

__all__ = [
    "LEVEQUE_CONST",
    "leveque_local_h",
    "leveque_local_sh",
    "leveque_mean_sh",
    "leveque_similarity_profile",
    "blasius_similarity",
    "blasius_velocity_field",
    "blasius_local_sh",
    "blasius_mean_sh",
    "blasius_local_cf",
    "ranz_marshall_mean_sh",
    "hilpert_mean_sh",
    "churchill_bernstein_mean_sh",
    "stanton_number",
    "chilton_colburn_j",
    "fit_power_law",
    "wall_flux_from_profile",
    "FrozenFieldResult",
    "solve_frozen_field_ad",
    "solve_shear_channel_ad",
    "solve_blasius_plate_ad",
    "solve_channel_plate_momentum",
    "run_cylinder_mass_transfer",
    "effective_diffusivity",
]

_W5 = tuple(float(v) for v in W_D2Q5)  # (2/6, 1/6, 1/6, 1/6, 1/6)

#: Leveque similarity constant ``1 / (9^{1/3} Gamma(4/3)) = 0.5383592...``
LEVEQUE_CONST = 1.0 / (9.0 ** (1.0 / 3.0) * math.gamma(4.0 / 3.0))

torch.set_grad_enabled(False)


# ---------------------------------------------------------------------------
# Closed-form mass-transfer references
# ---------------------------------------------------------------------------


def leveque_local_h(x: Any, shear: float, diffusivity: float) -> np.ndarray:
    """Leveque local mass-transfer coefficient ``h_m(x)`` (decays like ``x^{-1/3}``)."""
    x = np.asarray(x, dtype=np.float64)
    if np.any(x <= 0.0):
        raise ValueError("x must be positive")
    return diffusivity * (shear / (9.0 * diffusivity * x)) ** (1.0 / 3.0) / math.gamma(4.0 / 3.0)


def leveque_local_sh(x: Any, shear: float, diffusivity: float) -> np.ndarray:
    """``Sh_x = C_L (a x^2 / D)^{1/3}`` -- grows like ``x^{2/3}``."""
    x = np.asarray(x, dtype=np.float64)
    if np.any(x <= 0.0):
        raise ValueError("x must be positive")
    return LEVEQUE_CONST * (shear * x * x / diffusivity) ** (1.0 / 3.0)


def leveque_mean_sh(length: float, shear: float, diffusivity: float) -> float:
    """Length-averaged Leveque Sherwood number: ``(3/2) Sh_L``."""
    return 1.5 * float(leveque_local_sh(length, shear, diffusivity))


def leveque_similarity_profile(eta: Any) -> np.ndarray:
    """Leveque similarity profile ``theta(eta) = (1/Gamma(4/3)) int_0^eta e^{-s^3} ds``.

    ``eta = y (a / (9 D x))^{1/3}``; ``theta`` is 1 at the wall and 0 far away.
    Trapezoidal quadrature on a fine grid.
    """
    eta = np.atleast_1d(np.asarray(eta, dtype=np.float64))
    hi = max(float(eta.max()) * 1.2, 8.0)
    s = np.linspace(0.0, hi, 60001)
    integrand = np.exp(-s**3)
    cum = np.concatenate(
        [[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(s))]
    )
    return np.interp(eta, s, cum) / math.gamma(4.0 / 3.0)


_BLAS_ETA = np.linspace(0.0, 12.0, 4801)
_BLAS_TABLE: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


def blasius_similarity() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blasius similarity table ``(eta, f'(eta), f(eta))`` for ``2 f''' + f f'' = 0``.

    Shooting + classical RK4; the converged wall slope is
    ``f''(0) = 0.3320573...`` (asserted in the tests).
    """
    global _BLAS_TABLE
    if _BLAS_TABLE is not None:
        return _BLAS_TABLE

    h = float(_BLAS_ETA[1] - _BLAS_ETA[0])

    def integrate(w0: float) -> np.ndarray:
        y = np.array([0.0, 0.0, w0])
        out = np.empty((_BLAS_ETA.size, 3))
        out[0] = y

        def rhs(v: np.ndarray) -> np.ndarray:
            return np.array([v[1], v[2], -0.5 * v[0] * v[2]])

        for k in range(1, _BLAS_ETA.size):
            k1 = rhs(y)
            k2 = rhs(y + 0.5 * h * k1)
            k3 = rhs(y + 0.5 * h * k2)
            k4 = rhs(y + h * k3)
            y = y + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            out[k] = y
        return out

    lo, hi = 0.2, 0.6
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if integrate(mid)[-1, 1] < 1.0:
            lo = mid
        else:
            hi = mid
    tab = integrate(0.5 * (lo + hi))
    _BLAS_TABLE = (_BLAS_ETA.copy(), tab[:, 1].copy(), tab[:, 0].copy())
    return _BLAS_TABLE


def blasius_velocity_field(
    ny: int,
    nx: int,
    u_inf: float,
    nu: float,
    *,
    x_lead: float = 0.5,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact Blasius similarity velocity field on a lattice of shape ``(ny, nx)``.

    ``u = U f'(eta)``, ``v = (1/2) sqrt(nu U / x) (eta f' - f)`` with
    ``eta = y sqrt(U / (nu x))``; ``x = j - x_lead`` is the distance from the
    leading edge, so column 0 (the inlet) carries the uniform inflow profile.
    """
    eta_t, fp_t, f_t = blasius_similarity()
    i = np.arange(ny, dtype=np.float64)
    x = np.maximum(np.arange(nx, dtype=np.float64) - x_lead, x_lead)
    eta = np.outer(i, np.sqrt(u_inf / (nu * x)))
    fp = np.interp(eta, eta_t, fp_t, right=1.0)
    f = np.where(eta <= eta_t[-1], np.interp(eta, eta_t, f_t), eta - 1.72)
    ux = u_inf * fp
    uy = 0.5 * np.sqrt(nu * u_inf / x)[None, :] * (eta * fp - f)
    ux[:, 0] = u_inf
    uy[:, 0] = 0.0
    return torch.tensor(ux, dtype=dtype), torch.tensor(uy, dtype=dtype)


def blasius_local_sh(x: Any, u_inf: float, nu: float, diffusivity: float) -> np.ndarray:
    """``Sh_x = 0.332 Re_x^{1/2} Sc^{1/3}``."""
    x = np.asarray(x, dtype=np.float64)
    sc = nu / diffusivity
    return 0.332 * np.sqrt(u_inf * x / nu) * sc ** (1.0 / 3.0)


def blasius_mean_sh(length: float, u_inf: float, nu: float, diffusivity: float) -> float:
    """``Sh_L = 0.664 Re_L^{1/2} Sc^{1/3}``."""
    return 2.0 * float(blasius_local_sh(length, u_inf, nu, diffusivity))


def blasius_local_cf(x: Any, u_inf: float, nu: float) -> np.ndarray:
    """``C_f(x) = 0.664 Re_x^{-1/2}``; ``C_f/2`` is Colburn's momentum-side ``j``."""
    x = np.asarray(x, dtype=np.float64)
    return 0.664 / np.sqrt(u_inf * x / nu)


def ranz_marshall_mean_sh(reynolds: float, schmidt: float) -> float:
    """``Sh = 2 + 0.6 Re^{1/2} Sc^{1/3}`` (Ranz & Marshall 1952, sphere)."""
    return 2.0 + 0.6 * math.sqrt(reynolds) * schmidt ** (1.0 / 3.0)


def hilpert_mean_sh(reynolds: float, schmidt: float) -> float:
    """Hilpert (1933) circular-cylinder correlation ``Sh = C Re^m Sc^{1/3}``."""
    if reynolds < 40.0:
        c, m = 0.911, 0.385
    elif reynolds < 4000.0:
        c, m = 0.683, 0.466
    else:
        c, m = 0.193, 0.618
    return c * reynolds**m * schmidt ** (1.0 / 3.0)


def churchill_bernstein_mean_sh(reynolds: float, schmidt: float) -> float:
    """Churchill-Bernstein (1977) cylinder correlation (Chilton-Colburn form)."""
    if reynolds <= 0.0:
        raise ValueError("Re must be positive")
    sc = float(schmidt)
    num = 0.62 * math.sqrt(reynolds) * sc ** (1.0 / 3.0)
    den = (1.0 + (0.4 / sc) ** (2.0 / 3.0)) ** 0.25
    corr = (1.0 + (reynolds / 282000.0) ** (5.0 / 8.0)) ** (4.0 / 5.0)
    return 0.3 + num * corr / den


def stanton_number(sh: float, reynolds: float, diffusivity_ratio: float) -> float:
    """``St = Sh / (Re X)``; ``X`` is ``Pr`` (heat) or ``Sc`` (mass)."""
    return sh / (reynolds * diffusivity_ratio)


def chilton_colburn_j(stanton: float, diffusivity_ratio: float) -> float:
    """Chilton-Colburn ``j = St X^{2/3}``."""
    return stanton * diffusivity_ratio ** (2.0 / 3.0)


def fit_power_law(
    x: Any,
    y: Any,
    *,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
) -> dict:
    """Least-squares fit ``y = A x^p`` -> ``{'p','A','r2','n'}``."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if x_min is not None:
        m &= x >= x_min
    if x_max is not None:
        m &= x <= x_max
    if y_min is not None:
        m &= y >= y_min
    n = int(m.sum())
    if n < 3:
        return {"p": float("nan"), "A": float("nan"), "r2": float("nan"), "n": n}
    lx, ly = np.log(x[m]), np.log(y[m])
    p, ln_a = np.polyfit(lx, ly, 1)
    pred = p * lx + ln_a
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    return {
        "p": float(p),
        "A": float(math.exp(ln_a)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n": n,
    }


def wall_flux_from_profile(profile: Sequence[Any], spacing: float = 1.0, order: int = 3) -> np.ndarray:
    """``-dC/dy`` at the wall from the first rows of a field (positive = into the fluid).

    Row 0 is the wall node; ``order=3`` is the second-order one-sided stencil
    ``(3 c0 - 4 c1 + c2) / (2 dy)``.
    """
    c0 = np.asarray(profile[0], dtype=np.float64)
    if order == 2:
        return (c0 - np.asarray(profile[1], dtype=np.float64)) / spacing
    if order == 3:
        return (
            3.0 * c0
            - 4.0 * np.asarray(profile[1], dtype=np.float64)
            + np.asarray(profile[2], dtype=np.float64)
        ) / (2.0 * spacing)
    if order == 4:
        return (
            11.0 * c0
            - 18.0 * np.asarray(profile[1], dtype=np.float64)
            + 9.0 * np.asarray(profile[2], dtype=np.float64)
            - 2.0 * np.asarray(profile[3], dtype=np.float64)
        ) / (6.0 * spacing)
    raise ValueError("order must be 2, 3 or 4")


# ---------------------------------------------------------------------------
# Frozen-field advection-diffusion solver (plate-like geometry)
# ---------------------------------------------------------------------------


def _resolve_tau_sym(
    tau_odd: float, magic: float | None, cap: float = 100.0
) -> tuple[float | None, str]:
    """Even-mode relaxation time, with the high-Pe guard.

    The TRT magic parameter fixes ``Lambda = (tau_s - 1/2)(tau_o - 1/2)``, so for
    a very small diffusivity (``tau_o -> 1/2``) the even relaxation time blows up
    (``tau_s ~ 1/(tau_o - 1/2)``) and the even mode is effectively frozen, which
    destabilises the Dirichlet wall once ``tau_s`` exceeds ~20 (the Sc = 100
    plate case diverged with ``tau_s = 79``; see the campaign report).  Above
    ``cap`` the collision therefore falls back to BGK, and the fallback is
    reported (``collision``) so it can never be silent.
    """
    if magic is None:
        return None, "bgk"
    tau_s = tau_symmetric_from_magic(tau_odd, magic)
    if tau_s > cap:
        return None, "bgk_tau_s_capped"
    return tau_s, "trt"


@dataclass
class FrozenFieldResult:
    """Result of a frozen-velocity advection-diffusion solve on a wall-bounded box."""

    G: np.ndarray
    flux: np.ndarray
    sh_x: np.ndarray
    x: np.ndarray
    history: list[dict] = field(default_factory=list)
    steps: int = 0
    converged: bool = False
    mass: dict = field(default_factory=dict)
    collision: str = "trt"

    def as_dict(self) -> dict:
        return {
            "steps": self.steps,
            "converged": self.converged,
            "history": self.history,
            "mass": self.mass,
            "collision": self.collision,
        }


def _apply_plate_bcs(
    g: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    c_wall: float,
    c_inf: float,
) -> torch.Tensor:
    """Dirichlet wall (row 0) + freestream top row + inflow column + zero-gradient outlet.

    The wall row uses the anti-bounce-back rule ``g_in = -g_out + 2 w_in C_w``
    (second order for a scalar Dirichlet condition); the top row uses the
    mirrored rule with ``C_inf``.  Writes are in place: callers pass the fresh
    tensor returned by :func:`stream_scalar`.
    """
    ny, nx = int(g.shape[1]), int(g.shape[2])
    w = _W5
    # inflow: equilibrium of the uniform stream (ux[:,0], uy[:,0] are frozen)
    ux0 = float(ux[0, 0])
    uy0 = float(uy[0, 0])
    for idx in range(5):
        g[idx, :, 0] = w[idx] * c_inf * (1.0 + 3.0 * ((1 if idx == 1 else (-1 if idx == 3 else 0)) * ux0
                                                      + (1 if idx == 2 else (-1 if idx == 4 else 0)) * uy0))
    # outlet: zero gradient
    g[:, :, -1] = g[:, :, -2]
    # wall (row 0) Dirichlet C_w with anti-bounce-back on the wall-normal pair
    g[0, 0, :] = w[0] * c_wall
    g[1, 0, :] = w[1] * c_wall
    g[3, 0, :] = w[3] * c_wall
    g[2, 0, :] = -g[4, 0, :] + 2.0 * w[2] * c_wall
    # freestream (row -1) Dirichlet C_inf
    g[0, -1, :] = w[0] * c_inf
    g[1, -1, :] = w[1] * c_inf
    g[3, -1, :] = w[3] * c_inf
    g[4, -1, :] = -g[2, -1, :] + 2.0 * w[4] * c_inf
    return g


def solve_frozen_field_ad(
    ux: torch.Tensor,
    uy: torch.Tensor,
    diffusivity: float,
    *,
    c_wall: float = 1.0,
    c_inf: float = 0.0,
    steps: int = 12000,
    magic: float | None = 0.1875,
    trt: bool = True,
    tau_sym_cap: float = 20.0,
    tol: float = 1e-5,
    check_every: int = 200,
    init: str = "profile",
    init_length: float = 2.0,
    c_init: torch.Tensor | None = None,
    verbose: bool = False,
) -> FrozenFieldResult:
    """Iterate the D2Q5 advection-diffusion lattice to steady state on a frozen field.

    Bottom row = Dirichlet wall (``C_wall``), top row = freestream Dirichlet
    (``C_inf``), column 0 inflow, column ``nx-1`` zero-gradient outlet.
    Convergence is declared when the relative change of the wall-normal flux over
    ``check_every`` steps drops below ``tol``; the flux history is retained so the
    residual can be audited.
    """
    ny, nx = int(ux.shape[0]), int(ux.shape[1])
    dtype, device = ux.dtype, ux.device
    tau_o = tau_from_diffusivity(diffusivity)
    tau_s, collision = _resolve_tau_sym(tau_o, magic if trt else None, cap=tau_sym_cap)

    if c_init is not None:
        c0 = c_init.to(dtype=dtype, device=device)
    elif init == "uniform":
        c0 = torch.full((ny, nx), float(c_inf), dtype=dtype, device=device)
    else:
        y = torch.arange(ny, dtype=dtype, device=device).view(ny, 1)
        c0 = (c_inf + (c_wall - c_inf) * torch.exp(-y / float(init_length))).expand(ny, nx).contiguous()
    g = equilibrium_scalar_d2q5(c0, ux, uy)
    g = _apply_plate_bcs(g, ux, uy, c_wall, c_inf)

    history: list[dict] = []
    flux_prev: float | None = None
    converged = False
    step = 0
    for step in range(1, steps + 1):
        c = scalar_field(g)
        g = collide_scalar_trt(g, c, ux, uy, tau_o, tau_s)
        g, _ = stream_scalar(g, x_mode="open", y_mode="open")
        g = _apply_plate_bcs(g, ux, uy, c_wall, c_inf)
        if step % check_every == 0 or step == steps:
            prof = scalar_field(g).numpy()
            flux = wall_flux_from_profile([prof[r] for r in range(4)], order=4)
            # residual: change of the downstream-half integral of the wall flux
            # (the near-inlet columns are unresolved by construction and would
            # otherwise dominate a pointwise criterion)
            tail = flux[nx // 2 : nx - 3]
            f_int = float(tail.sum())
            if flux_prev is not None:
                rel = abs(f_int - flux_prev) / (abs(f_int) + 1e-30)
            else:
                rel = float("nan")
            flux_prev = f_int
            history.append({"step": step, "flux_change": rel, "tail_flux_integral": f_int})
            if verbose:
                print(f"  step={step:6d} dflux={rel:.3e} tail_int={f_int:.6f}", flush=True)
            if rel < tol and step > 4 * check_every:
                converged = True
                break

    scale = abs(c_wall - c_inf)
    G = (scalar_field(g).numpy() - c_inf) / scale
    flux = wall_flux_from_profile([G[r] for r in range(4)], order=3)
    x = np.maximum(np.arange(nx, dtype=np.float64) - 0.5, 0.5)
    sh_x = x * flux
    sh_x[:4] = np.nan
    sh_x[-3:] = np.nan
    return FrozenFieldResult(
        G=G,
        flux=flux,
        sh_x=sh_x,
        x=x,
        history=history,
        steps=step,
        converged=converged,
        mass={"tail_flux_integral": float(flux[nx // 2 : nx - 3].sum()), "steps": float(step)},
        collision=collision,
    )


def solve_shear_channel_ad(
    nx: int,
    ny: int,
    shear: float,
    diffusivity: float,
    *,
    steps: int = 12000,
    magic: float = 0.1875,
    trt: bool = True,
    dtype: torch.dtype = torch.float32,
    **kwargs: Any,
) -> FrozenFieldResult:
    """Leveque/Graetz configuration: linear shear ``u = a y`` over a wall at ``C_w``.

    ``y`` is in lattice units, so ``a`` has units of 1/time; the concentration
    layer thickness is ``(9 D x / a)^{1/3}`` and must stay inside the box.
    """
    i = np.arange(ny, dtype=np.float64)
    ux = torch.tensor(np.repeat((shear * i)[:, None], nx, axis=1), dtype=dtype)
    uy = torch.zeros((ny, nx), dtype=dtype)
    if kwargs.pop("init", None) == "leveque":
        kwargs["c_init"] = _leveque_initial_field(ny, nx, shear, diffusivity, dtype)
    return solve_frozen_field_ad(ux, uy, diffusivity, steps=steps, magic=magic, trt=trt, **kwargs)


def _leveque_initial_field(
    ny: int, nx: int, shear: float, diffusivity: float, dtype: torch.dtype
) -> torch.Tensor:
    """Leveque similarity field ``G = 1 - theta(eta)`` used as a starting guess.

    ``eta = y (a / (9 D x))^{1/3}`` with ``x`` the distance from the leading edge.
    This is only an initial condition: the converged flux is audited against a
    run started from a uniform field (see the campaign report), and the
    similarity field is the exact steady solution of the linear-shear problem,
    so the lattice update has almost nothing to do except confirm it.
    """
    y = np.arange(ny, dtype=np.float64)[:, None]
    x = np.maximum(np.arange(nx, dtype=np.float64) - 0.5, 0.5)[None, :]
    eta = y * (shear / (9.0 * diffusivity * x)) ** (1.0 / 3.0)
    G = 1.0 - leveque_similarity_profile(eta.ravel()).reshape(ny, nx)
    G[:, 0] = 0.0
    return torch.tensor(G, dtype=dtype)


def solve_blasius_plate_ad(
    nx: int,
    ny: int,
    u_inf: float,
    nu: float,
    diffusivity: float,
    *,
    steps: int = 12000,
    magic: float = 0.1875,
    trt: bool = True,
    dtype: torch.dtype = torch.float32,
    x_lead: float = 0.5,
    **kwargs: Any,
) -> tuple[FrozenFieldResult, torch.Tensor, torch.Tensor]:
    """Blasius flat plate mass transfer driven by the exact similarity field."""
    ux, uy = blasius_velocity_field(ny, nx, u_inf, nu, x_lead=x_lead, dtype=dtype)
    if kwargs.pop("init", None) in ("blasius", "similarity"):
        # local-shear Leveque approximation: a(x) = U sqrt(U/(nu x)) f''(0)
        x = np.maximum(np.arange(nx, dtype=np.float64) - x_lead, x_lead)
        shear_x = u_inf * np.sqrt(u_inf / (nu * x)) * 0.3320573362
        y = np.arange(ny, dtype=np.float64)[:, None]
        eta = y * (shear_x[None, :] / (9.0 * diffusivity * x[None, :])) ** (1.0 / 3.0)
        G = 1.0 - leveque_similarity_profile(eta.ravel()).reshape(ny, nx)
        G[:, 0] = 0.0
        kwargs["c_init"] = torch.tensor(G, dtype=dtype)
    res = solve_frozen_field_ad(ux, uy, diffusivity, steps=steps, magic=magic, trt=trt, **kwargs)
    res.x = np.maximum(np.arange(nx, dtype=np.float64) - x_lead, 0.5)
    res.sh_x = res.x * res.flux
    res.sh_x[:4] = np.nan
    res.sh_x[-3:] = np.nan
    return res, ux, uy

# ---------------------------------------------------------------------------
# Coupled momentum (+ scalar) drivers
# ---------------------------------------------------------------------------


def _momentum_step(
    f: torch.Tensor,
    tau: float,
    u_in: float,
    wall_mask: torch.Tensor,
    obstacle: torch.Tensor,
    *,
    free_slip_top: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One D2Q9 BGK step: collision, streaming, inlet, outlet, walls, obstacle.

    Bottom row / obstacle are no-slip bounce-back; the top row is free-slip
    (specular: ``4<->2``, ``7<->6``, ``8<->5``) when ``free_slip_top`` is set,
    i.e. a flat plate on the bottom wall of a uniform inflow.  Momentum-exchange
    forces on the obstacle are evaluated between streaming and bounce-back.
    """
    ny, nx = int(f.shape[1]), int(f.shape[2])
    f = collide_bgk(f, tau)
    f = stream_momentum(f)
    fx, fy = compute_obstacle_forces(f, obstacle)
    rho, ux, uy = macroscopic(f)
    ux0 = torch.full((ny, 1), float(u_in), dtype=f.dtype)
    uy0 = torch.zeros((ny, 1), dtype=f.dtype)
    f[:, :, 0] = equilibrium(rho[:, 0:1], ux0, uy0)[:, :, 0]
    f[:, :, -1] = f[:, :, -2]
    if free_slip_top:
        f[4, -1, :] = f[2, -1, :]
        f[7, -1, :] = f[6, -1, :]
        f[8, -1, :] = f[5, -1, :]
    f = bounce_back_cells(f, wall_mask | obstacle)
    rho, ux, uy = macroscopic(f)
    ux = torch.where(obstacle, torch.zeros_like(ux), ux)
    uy = torch.where(obstacle, torch.zeros_like(uy), uy)
    return f, ux, uy, fx, fy


def solve_channel_plate_momentum(
    nx: int,
    ny: int,
    u_in: float,
    nu: float,
    n_steps: int,
    *,
    scalar_ds: Sequence[float] | None = None,
    scalar_labels: Sequence[str] | None = None,
    c_wall: float = 1.0,
    c_inf: float = 0.0,
    magic: float = 0.1875,
    dtype: torch.dtype = torch.float32,
    diagnostic_every: int = 1000,
    free_slip_top: bool = True,
    average_from: float = 0.5,
    verbose: bool = False,
) -> dict:
    """Coupled D2Q9 momentum + D2Q5 scalars over a no-slip bottom plate.

    Uniform inflow enters column 0 and a Blasius-type boundary layer develops on
    the bottom wall, so the *same* velocity field feeds the momentum-side
    ``C_f(x)`` and every scalar's ``Sh(x)``: this is the three-way
    Chilton-Colburn check.  Wall shear comes from the lattice velocity with a
    second-order one-sided difference, ``tau_w = rho nu du/dy``.

    Scalars are evolved in lockstep with the flow (two-distribution coupling),
    and the reported fields/fluxes are time-averaged over the last
    ``1 - average_from`` of the run.
    """
    device = torch.device("cpu")
    obstacle = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall = make_channel_wall_mask(ny, nx, obstacle, device).clone()
    if free_slip_top:
        wall[-1, :] = False
    tau = 0.5 + 3.0 * nu
    rho0 = torch.ones((ny, nx), dtype=dtype)
    ux0 = torch.full((ny, nx), u_in, dtype=dtype)
    uy0 = torch.zeros((ny, nx), dtype=dtype)
    f = equilibrium(rho0, ux0, uy0)
    gs: list[torch.Tensor] = []
    taus: list[tuple[float, float | None]] = []
    for d in scalar_ds or ():
        tau_o = tau_from_diffusivity(d)
        taus.append((tau_o, tau_symmetric_from_magic(tau_o, magic)))
        gs.append(
            _apply_plate_bcs(
                equilibrium_scalar_d2q5(torch.full((ny, nx), c_inf, dtype=dtype), ux0, uy0),
                ux0,
                uy0,
                c_wall,
                c_inf,
            )
        )
    ux_sum = torch.zeros((ny, nx), dtype=dtype)
    uy_sum = torch.zeros((ny, nx), dtype=dtype)
    flux_sum = [np.zeros(nx) for _ in gs]
    n_avg = 0
    cd_hist: list[float] = []
    cl_hist: list[float] = []
    start_avg = int(n_steps * average_from)
    labels = list(scalar_labels or [f"scalar{i}" for i in range(len(gs))])
    for step in range(1, n_steps + 1):
        f, ux, uy, fx, fy = _momentum_step(
            f, tau, u_in, wall, obstacle, free_slip_top=free_slip_top
        )
        if not torch.isfinite(ux).all():
            raise FloatingPointError(f"momentum diverged at step {step}")
        for k, g in enumerate(gs):
            c = scalar_field(g)
            g = collide_scalar_trt(g, c, ux, uy, taus[k][0], taus[k][1])
            g, _ = stream_scalar(g, x_mode="open", y_mode="open")
            g = _apply_plate_bcs(g, ux, uy, c_wall, c_inf)
            gs[k] = g
        if step > start_avg:
            ux_sum += ux
            uy_sum += uy
            n_avg += 1
            for k in range(len(gs)):
                prof = scalar_field(gs[k]).numpy()
                flux_sum[k] += wall_flux_from_profile([prof[r] for r in range(3)], order=3)
        if step % diagnostic_every == 0:
            u_row = ux.numpy()
            du = (4.0 * u_row[1] - u_row[2]) / 2.0
            cf = 2.0 * nu * du / (u_in * u_in)
            cd_hist.append(float(np.mean(cf[10:-10])))
            cl_hist.append(float(np.abs(uy.numpy()).max()))
            if verbose:
                print(f"  step={step:6d} <Cf>={cd_hist[-1]:.5f} max|v|={cl_hist[-1]:.4f}", flush=True)
    n_avg = max(n_avg, 1)
    ux_mean = (ux_sum / n_avg).numpy()
    uy_mean = (uy_sum / n_avg).numpy()
    du = (4.0 * ux_mean[1] - ux_mean[2]) / 2.0
    cf_x = 2.0 * nu * du / (u_in * u_in)
    x = np.maximum(np.arange(nx, dtype=np.float64) - 0.5, 0.5)
    scalars: dict[str, dict] = {}
    for k, lab in enumerate(labels):
        flux = flux_sum[k] / n_avg
        sh = x * flux
        sh[:4] = np.nan
        sh[-3:] = np.nan
        scalars[lab] = {"D": float(scalar_ds[k]), "flux": flux, "sh_x": sh}
    return {
        "ux_mean": ux_mean,
        "uy_mean": uy_mean,
        "cf_x": cf_x,
        "x": x,
        "cf_history": cd_hist,
        "max_v_history": cl_hist,
        "scalars": scalars,
        "steps": n_steps,
        "n_avg_steps": n_avg,
    }


def run_cylinder_mass_transfer(
    nx: int,
    ny: int,
    radius: float,
    u_in: float,
    reynolds: float,
    *,
    scalar_sc: Sequence[float] = (1.0,),
    n_steps: int = 24000,
    average_from: float = 0.55,
    c_wall: float = 1.0,
    c_inf: float = 0.0,
    magic: float = 0.1875,
    dtype: torch.dtype = torch.float32,
    diagnostic_every: int = 1000,
    verbose: bool = False,
) -> dict:
    """Cylinder in cross flow: D2Q9 momentum coupled to D2Q5 mass-transfer scalars.

    The cylinder cells are held at ``C_wall`` and the mass injected into them per
    unit time is accumulated -- that *is* the exact mass-transfer rate ``mdot``.
    With ``Sh = mdot / (pi D_AB (C_w - C_inf))`` (per unit depth) the result is
    free of surface-reconstruction error.  Time-mean values are taken over the
    last ``1 - average_from`` fraction of the run, which covers several vortex
    shedding cycles for ``Re > 47``.
    """
    device = torch.device("cpu")
    diameter = 2.0 * radius
    nu = u_in * diameter / reynolds
    tau = 0.5 + 3.0 * nu
    if tau < 0.505:
        raise ValueError(f"tau={tau:.4f} too close to 0.5; increase radius or reduce Re")
    obstacle = cylinder_mask(nx, ny, nx * 0.25, ny * 0.5, radius, device)
    wall = make_channel_wall_mask(ny, nx, obstacle, device)
    rho0 = torch.ones((ny, nx), dtype=dtype)
    ux0 = torch.full((ny, nx), u_in, dtype=dtype)
    uy0 = torch.zeros((ny, nx), dtype=dtype)
    f = equilibrium(rho0, ux0, uy0)
    zero = torch.zeros((ny, nx), dtype=dtype)
    f[:, obstacle] = equilibrium(rho0, zero, zero)[:, obstacle]
    n_solid = int(obstacle.sum())
    taus: list[tuple[float, float, float | None]] = []
    for sc in scalar_sc:
        d = nu / sc
        tau_o = tau_from_diffusivity(d)
        taus.append((d, tau_o, tau_symmetric_from_magic(tau_o, magic)))
    gs = [
        equilibrium_scalar_d2q5(torch.full((ny, nx), c_inf, dtype=dtype), ux0, uy0)
        for _ in taus
    ]
    mdot = np.zeros(len(gs))
    n_avg = 0
    cd_hist: list[float] = []
    cl_hist: list[float] = []
    cl_window: list[float] = []
    dyn = 0.5 * u_in * u_in * diameter
    start_avg = int(n_steps * average_from)
    for step in range(1, n_steps + 1):
        f, ux, uy, fx, fy = _momentum_step(f, tau, u_in, wall, obstacle, free_slip_top=True)
        if not torch.isfinite(ux).all():
            raise FloatingPointError(f"momentum diverged at step {step}")
        for k, g in enumerate(gs):
            _d, tau_o, tau_s = taus[k]
            c = scalar_field(g)
            g = collide_scalar_trt(g, c, ux, uy, tau_o, tau_s)
            g, _ = stream_scalar(g, x_mode="open", y_mode="open")
            g[:, :, 0] = equilibrium_scalar_d2q5(
                torch.full((ny, 1), c_inf, dtype=dtype), ux[:, 0:1], uy[:, 0:1]
            )[:, :, 0]
            g[:, :, -1] = g[:, :, -2]
            g[2, 0, :] = g[4, 0, :]
            g[4, -1, :] = g[2, -1, :]
            held = n_solid * c_wall
            released = held - float(g[:, obstacle].sum())
            for idx in range(5):
                g[idx, obstacle] = _W5[idx] * c_wall
            if step > start_avg:
                mdot[k] += released
            gs[k] = g
        if step > start_avg:
            n_avg += 1
            cl_window.append(float(fy) / dyn)
        if step % diagnostic_every == 0:
            cd = float(fx) / dyn
            cl = float(fy) / dyn
            cd_hist.append(cd)
            cl_hist.append(cl)
            if verbose:
                print(
                    f"  step={step:6d} Cd={cd:.4f} Cl={cl:.4f} "
                    + " ".join(f"mdot{k}={v:.3f}" for k, v in enumerate(mdot)),
                    flush=True,
                )
    d_eff = 2.0 * math.sqrt(n_solid / math.pi)
    out_scalars = []
    for k, sc in enumerate(scalar_sc):
        d = taus[k][0]
        rate = float(mdot[k]) / max(n_avg, 1)
        sh = rate / (math.pi * d * abs(c_wall - c_inf))
        out_scalars.append({"Sc": float(sc), "D": float(d), "mdot": rate, "Sh": sh})
    return {
        "Re": float(reynolds),
        "nu": nu,
        "tau": tau,
        "D_diameter": diameter,
        "D_eff_from_area": d_eff,
        "n_solid_cells": n_solid,
        "Cd_mean": float(np.mean(cd_hist[-max(len(cd_hist) // 2, 1) :])) if cd_hist else float("nan"),
        "Cl_history": cl_hist,
        "Cd_history": cd_hist,
        "Cl_window": cl_window,
        "St": _strouhal_from_series(cl_window, u_in, diameter),
        "blockage": diameter / ny,
        "scalars": out_scalars,
        "n_avg_steps": n_avg,
    }


def _strouhal_from_series(series: Sequence[float], u_in: float, diameter: float) -> float:
    """Dominant Strouhal number of a lift-coefficient series (peak of the FFT)."""
    if len(series) < 32:
        return float("nan")
    arr = np.asarray(series, dtype=np.float64)
    arr = arr - arr.mean()
    if np.allclose(arr, 0.0):
        return 0.0
    spec = np.abs(np.fft.rfft(arr * np.hanning(arr.size)))
    freqs = np.fft.rfftfreq(arr.size)
    spec[0] = 0.0
    f_peak = float(freqs[int(np.argmax(spec))])
    return f_peak * diameter / u_in


# ---------------------------------------------------------------------------
# Numerical-diffusivity calibration
# ---------------------------------------------------------------------------


def effective_diffusivity(
    diffusivity: float,
    u: float = 0.0,
    *,
    nx: int = 64,
    n_waves: float = 1.0,
    steps: int = 4000,
    magic: float | None = 0.1875,
    trt: bool = True,
    tau_sym_cap: float = 20.0,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Measured decay rate of a periodic Fourier mode -> effective diffusivity.

    A sine wave ``sin(k x)`` is advected at constant velocity ``u`` on a periodic
    lattice; the amplitude decays as ``exp(-D_eff k^2 t)``.  The ratio
    ``D_eff / D`` is the numerical-diffusion error of the advection-diffusion
    operator at the cell Peclet number ``Pe_cell = u / D`` -- the quantity that
    governs how fine the grid must be for high-Schmidt-number boundary layers.
    """
    k = 2.0 * math.pi * n_waves / nx
    x = torch.arange(nx, dtype=dtype) * 2.0 * math.pi / nx * n_waves
    ux = torch.full((3, nx), float(u), dtype=dtype)
    uy = torch.zeros((3, nx), dtype=dtype)
    g = equilibrium_scalar_d2q5(torch.sin(k * x).expand(3, nx).contiguous(), ux, uy)
    tau_o = tau_from_diffusivity(diffusivity)
    tau_s, collision = _resolve_tau_sym(tau_o, magic if trt else None, cap=tau_sym_cap)
    samples: list[tuple[int, float]] = []
    every = max(steps // 10, 1)
    for step in range(steps + 1):
        if step % every == 0:
            c = scalar_field(g).reshape(-1)[:nx]
            amp = float(torch.sqrt(2.0 * ((c - c.mean()) ** 2).mean()))
            samples.append((step, amp))
        if step == steps:
            break
        c = scalar_field(g)
        g = collide_scalar_trt(g, c, ux, uy, tau_o, tau_s)
        g, _ = stream_scalar(g, x_mode="periodic", y_mode="periodic")
    arr = np.asarray(samples, dtype=np.float64)
    keep = (arr[:, 0] >= arr[-1, 0] * 0.2) & (arr[:, 1] > 0)
    slope = float(np.polyfit(arr[keep, 0], np.log(arr[keep, 1]), 1)[0])
    d_eff = -slope / (k * k)
    return {
        "D": float(diffusivity),
        "u": float(u),
        "Pe_cell": float(u / diffusivity) if diffusivity > 0 else float("inf"),
        "tau_o": tau_o,
        "tau_s": tau_s,
        "collision": collision,
        "D_eff": d_eff,
        "ratio": d_eff / diffusivity,
        "k": k,
        "steps": steps,
    }
