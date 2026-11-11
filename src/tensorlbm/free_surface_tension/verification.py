"""Machine-checkable acceptance cases for the free-surface CSF surface tension.

Every case returns a plain ``dict`` (JSON-serialisable after conversion) with

* ``params``  -- the full parameter set of the run (for the report table);
* ``metrics`` -- the numbers that the acceptance criteria are judged on;
* ``series``  -- the raw history needed for the figures.

The five physics anchors
-----------------------
============================  ===========================================  ==========
case                          anchor                                       criterion
============================  ===========================================  ==========
``young_laplace_2d``          ``Delta p = sigma/R`` (cylinder)              < 3 %
``young_laplace_3d``          ``Delta p = 2 sigma/R`` (sphere)              < 5 %
``droplet_oscillation``       ``T = 2 pi sqrt(rho R^3/(8 sigma))``          < 5 %
                              (Rayleigh 1879, n = 2)
``marangoni``                 ``u_i = (dsigma/dx) h1 h2 / ((h1+h2) mu)``    linear
                              (flat interface, two walls)                  R^2 > 0.95
``spurious``                  ``|u_spurious| ~ h^2`` (4 grid levels)        slope >= 1.8
``mass_conservation``         ``sum(phi)`` invariant                        loss < 0.1 %
============================  ===========================================  ==========
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from .csf import (
    curvature,
    gradient,
    laplace_pressure,
    rayleigh_period,
    tanh_droplet_phi,
    tanh_ellipse_phi,
    tanh_flat_phi,
    tanh_sphere_phi,
)
from .solver import CapillarySolver, SolverConfig

__all__ = [
    "young_laplace_2d",
    "young_laplace_3d",
    "droplet_oscillation",
    "marangoni_case",
    "spurious_case",
    "mass_conservation",
    "zero_crossings",
    "period_from_crossings",
    "linear_fit",
    "power_law_slope",
    "parabola_at",
]


# --------------------------------------------------------------------------- #
# signal analysis (pure functions, unit-tested)                               #
# --------------------------------------------------------------------------- #
def zero_crossings(t: Sequence[float], y: Sequence[float]) -> list[float]:
    """Linear-interpolated zero crossings of ``y(t)`` (ascending ``t``)."""
    out: list[float] = []
    for i in range(1, len(y)):
        a, b = float(y[i - 1]), float(y[i])
        if a == 0.0:
            out.append(float(t[i - 1]))
        elif a * b < 0.0:
            frac = a / (a - b)
            out.append(float(t[i - 1]) + frac * (float(t[i]) - float(t[i - 1])))
    return out


def period_from_crossings(crossings: Sequence[float]) -> tuple[float, float, int]:
    """Period from successive zero crossings: returns ``(T, std, n_half_periods)``."""
    if len(crossings) < 2:
        return float("nan"), float("nan"), 0
    d = [float(b - a) for a, b in zip(crossings[:-1], crossings[1:])]
    mean_half = sum(d) / len(d)
    var = sum((x - mean_half) ** 2 for x in d) / len(d)
    return 2.0 * mean_half, 2.0 * math.sqrt(var), len(d)


def linear_fit(x: Sequence[float], y: Sequence[float]) -> tuple[float, float, float]:
    """Least-squares ``y = a x + b``; returns ``(a, b, R^2)``."""
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    a = sxy / sxx
    b = my - a * mx
    syy = sum((yi - my) ** 2 for yi in y)
    ss_res = sum((yi - (a * xi + b)) ** 2 for xi, yi in zip(x, y))
    r2 = 1.0 - ss_res / syy if syy > 0 else 1.0
    return a, b, r2


def power_law_slope(x: Sequence[float], y: Sequence[float]) -> tuple[float, float, float]:
    """Slope of ``log y`` vs ``log x``; returns ``(slope, intercept, R^2)``."""
    lx = [math.log(v) for v in x]
    ly = [math.log(v) for v in y]
    return linear_fit(lx, ly)


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _band_mean_kappa(phi: torch.Tensor) -> float:
    """Curvature averaged over the interface band (``|grad phi|`` weighted)."""
    k = curvature(phi)
    delta = torch.sqrt(sum(g * g for g in gradient(phi)) + 1e-30)
    w = delta / delta.sum()
    return float((k * w).sum())


def _viscosity(tau: float) -> float:
    return (tau - 0.5) / 3.0


def parabola_at(prof: Sequence[float], y0: float) -> float:
    """Quadratic interpolation of a 1-D profile at ``y0`` (cell centres ``j+0.5``).

    Used to read the interface velocity at the *continuous* interface position
    instead of a cell average, so the reported value is not biased by the
    discrete peak being split between two rows.
    """
    j = int(round(y0 - 0.5))
    j = max(1, min(len(prof) - 2, j))
    y = [j - 0.5, j + 0.5, j + 1.5]
    u = [float(prof[j - 1]), float(prof[j]), float(prof[j + 1])]
    # Lagrange quadratic through (y, u)
    return float(
        sum(
            u[m] * math.prod([(y0 - y[o]) / (y[m] - y[o]) for o in range(3) if o != m])
            for m in range(3)
        )
    )


# --------------------------------------------------------------------------- #
# case A/B -- Young-Laplace                                                   #
# --------------------------------------------------------------------------- #
def _young_laplace(
    dim: int,
    radius: float,
    sigma: float = 0.05,
    tau: float = 0.9,
    width: float = 3.0,
    steps: Sequence[int] = (500, 1500, 3000, 6000),
    shell: float = 3.0,
    init_laplace: bool = False,
    dtype: torch.dtype = torch.float64,
) -> dict:
    L = int(round(4 * radius))
    shape = (L,) * dim if dim == 3 else (L, L)
    cfg = SolverConfig(
        sigma=sigma,
        tau_liquid=tau,
        tau_gas=tau,
        width=width,
        dtype=dtype,
        init_laplace=init_laplace,
    )
    solver = CapillarySolver(shape, cfg)
    center = [0.5 * n for n in shape]
    phi = (
        tanh_sphere_phi(shape, center, radius, width=width, dtype=dtype)
        if dim == 3
        else tanh_droplet_phi(shape, center, radius, width=width, dtype=dtype)
    )
    solver.set_phi(phi)
    k_band = _band_mean_kappa(solver.phi)
    k_ref = laplace_pressure(1.0, radius, ndim=dim)  # sigma = 1 -> kappa
    ref = laplace_pressure(sigma, radius, ndim=dim)

    series = []
    done = 0
    for n in steps:
        solver.step(int(n) - done)
        done = int(n)
        series.append(
            {
                "step": int(n),
                "dp": solver.pressure_jump(radius, shell=shell),
                "umax": solver.max_speed(),
                "mass": solver.phi_mass(),
            }
        )
    last = series[-1]
    return {
        "case": f"young_laplace_{dim}d",
        "params": {
            "dim": dim,
            "radius": radius,
            "L": L,
            "sigma": sigma,
            "tau": tau,
            "mu": _viscosity(tau),
            "width": width,
            "steps": list(steps),
            "shell": shell,
            "init_laplace": init_laplace,
            "dtype": str(dtype).replace("torch.", ""),
        },
        "metrics": {
            "dp": last["dp"],
            "dp_ref": ref,
            "dp_rel_err": last["dp"] / ref - 1.0,
            "kappa_band": k_band,
            "kappa_ref": k_ref,
            "kappa_rel_err": k_band / k_ref - 1.0,
            "umax": last["umax"],
            "mass_drift": (series[-1]["mass"] - series[0]["mass"]) / series[0]["mass"],
        },
        "series": series,
    }


def young_laplace_2d(radius: float, **kw) -> dict:
    """2-D (cylindrical) droplet: ``p_in - p_out = sigma/R``."""
    return _young_laplace(2, radius, **kw)


def young_laplace_3d(radius: float, **kw) -> dict:
    """3-D (spherical) droplet: ``p_in - p_out = 2 sigma/R``."""
    kw.setdefault("dtype", torch.float32)
    kw.setdefault("steps", (300, 1000, 2000, 3500))
    return _young_laplace(3, radius, **kw)


# --------------------------------------------------------------------------- #
# case C -- droplet oscillation (Rayleigh n = 2)                              #
# --------------------------------------------------------------------------- #
def droplet_oscillation(
    sigma: float,
    radius: float = 12.0,
    L: int = 64,
    eps: float = 0.05,
    tau: float = 0.9,
    width: float = 3.0,
    periods: float = 2.5,
    compression: float = 0.0,
    sharpen: bool = False,
    advection: str = "sl",
    sample: int = 2,
) -> dict:
    """Initial ellipse ``(R(1+eps), R(1-eps))`` relaxing with the mode-2 period.

    The time signal is the ``phi`` second-moment anisotropy
    ``D(t) = (Q_xx - Q_yy)/(Q_xx + Q_yy)``.  Its zero crossings give the period
    (a half period elapses between two successive crossings), which is compared
    with ``T = 2 pi sqrt(rho R^3 / (8 sigma))``.
    """
    cfg = SolverConfig(
        sigma=sigma,
        tau_liquid=tau,
        tau_gas=tau,
        width=width,
        compression=compression,
        sharpen=sharpen,
        advection="sl",
    )
    solver = CapillarySolver((L, L), cfg)
    solver.set_phi(
        tanh_ellipse_phi(
            (L, L),
            (0.5 * L, 0.5 * L),
            (radius * (1.0 + eps), radius * (1.0 - eps)),
            width=width,
        )
    )
    t_ref = rayleigh_period(sigma, radius, rho=cfg.rho0, mode=2)
    t_ref_2d = 2.0 * math.pi * math.sqrt(cfg.rho0 * radius**3 / (6.0 * sigma))
    n_steps = int(periods * t_ref)
    t_hist, d_hist, u_hist, m_hist = [], [], [], []
    done = 0
    while done < n_steps:
        solver.step(min(sample, n_steps - done), advect=True)
        done += min(sample, n_steps - done)
        t_hist.append(done)
        d_hist.append(solver.second_moment_anisotropy())
        u_hist.append(solver.max_speed())
        m_hist.append(solver.phi_mass())
    cr = zero_crossings(t_hist, d_hist)
    t_meas, t_std, n_half = period_from_crossings(cr)
    mass0 = m_hist[0]
    return {
        "case": "droplet_oscillation",
        "params": {
            "sigma": sigma,
            "radius": radius,
            "L": L,
            "eps": eps,
            "tau": tau,
            "mu": _viscosity(tau),
            "width": width,
            "compression": compression,
            "sharpen": sharpen,
            "steps": n_steps,
            "periods": periods,
        },
        "metrics": {
            "period": t_meas,
            "period_ref": t_ref,
            "period_ref_2d": t_ref_2d,
            "period_rel_err": t_meas / t_ref - 1.0,
            "period_std": t_std,
            "n_zero_crossings": len(cr),
            "amp_first": max(abs(v) for v in d_hist[: max(3, len(d_hist) // 10)]),
            "amp_last": max(abs(v) for v in d_hist[-max(3, len(d_hist) // 10) :]),
            "umax": max(u_hist),
            "mass_loss": abs(m_hist[-1] - mass0) / mass0,
        },
        "series": {"t": t_hist, "D": d_hist, "umax": u_hist, "mass": m_hist},
    }


# --------------------------------------------------------------------------- #
# case D -- Marangoni (thermo-capillary) drive                                #
# --------------------------------------------------------------------------- #
def marangoni_case(
    dsdx: float,
    Ly: int = 32,
    Lx: int = 4,
    y0: float | None = None,
    tau: float = 0.9,
    width: float = 3.0,
    steps: int = 6000,
    sample: int = 100,
) -> dict:
    """Flat interface with a constant ``dsigma/dx``: two-layer Couette anchor.

    Liquid fills the box between two no-slip walls; the interface sits at
    ``y0``.  A Marangoni stress ``dsigma/dx`` acts tangentially at the
    interface, and with ``h1 = y0 - y_wall`` / ``h2 = y_wall_top - y0`` the
    steady Stokes solution is a piecewise-linear profile whose interface value
    is ``u_i = (dsigma/dx) h1 h2 / ((h1 + h2) mu)``.
    """
    if y0 is None:
        y0 = 0.5 * Ly
    mu = _viscosity(tau)
    cfg = SolverConfig(
        sigma=0.0,
        tau_liquid=tau,
        tau_gas=tau,
        width=width,
        init_laplace=False,
    )
    solver = CapillarySolver((Ly, Lx), cfg)
    solver.set_phi(tanh_flat_phi((Ly, Lx), y0, width=width))
    solver.add_wall(axis=1, side="both")

    h1 = y0 - 0.5
    h2 = (Ly - 0.5) - y0
    u_ref = dsdx * h1 * h2 / ((h1 + h2) * mu)

    # rows of the interface band, restricted to |y - y0| <= 2.5 so that the
    # periodic-wrap gradient at the y boundary cannot enter the measurement
    rows = [
        j
        for j in range(Ly)
        if abs((j + 0.5) - y0) <= 2.5
    ]

    def interface_velocity() -> float:
        prof_ = solver.u[0].mean(dim=1)
        return parabola_at([float(v) for v in prof_], y0)

    hist = []
    done = 0
    while done < steps:
        solver.step(min(sample, steps - done), advect=False, grad_sigma=(dsdx, 0.0))
        done += min(sample, steps - done)
        hist.append((done, interface_velocity(), solver.max_speed()))
    # mean x-velocity profile as a function of y
    prof = solver.u[0].mean(dim=1)
    y_idx = torch.arange(Ly, dtype=torch.float64) + 0.5
    ui = parabola_at([float(v) for v in prof], y0)
    ui_band = float(prof[rows].mean())
    return {
        "case": "marangoni",
        "params": {
            "dsdx": dsdx,
            "Ly": Ly,
            "Lx": Lx,
            "y0": y0,
            "tau": tau,
            "mu": mu,
            "h1": h1,
            "h2": h2,
            "steps": steps,
            "Ma": abs(u_ref) * h1 / mu,  # Ma = U_mar H / nu, nu = mu/rho0
        },
        "metrics": {
            "u_interface": ui,
            "u_interface_band_mean": ui_band,
            "u_ref": u_ref,
            "u_rel_err": (ui / u_ref - 1.0) if u_ref else float("nan"),
            "u_slope": float(prof[min(int(h1), Ly - 1)] / h1),
            "umax": solver.max_speed(),
            "u_converged_drift": abs(hist[-1][1] - hist[-2][1]) / max(abs(hist[-1][1]), 1e-30),
        },
        "series": {
            "y": [float(v) for v in y_idx],
            "ux_profile": [float(v) for v in prof],
            "t": [h[0] for h in hist],
            "u_interface": [h[1] for h in hist],
        },
    }


# --------------------------------------------------------------------------- #
# case E -- spurious-current convergence                                      #
# --------------------------------------------------------------------------- #
def spurious_case(
    radius: float,
    sigma: float = 0.05,
    tau: float = 0.9,
    width: float = 3.0,
    steps: int = 4000,
    sample: int = 500,
    init_laplace: bool = True,
) -> dict:
    """Static droplet started on the balanced (Laplace) state; measure ``|u|``.

    The balanced start removes the acoustic transient that otherwise masks the
    parasitic current, so ``max |u|`` is the residual of the discrete force /
    pressure-gradient balance, i.e. the quantity that must vanish as ``h^2``.
    """
    L = int(round(4 * radius))
    cfg = SolverConfig(
        sigma=sigma,
        tau_liquid=tau,
        tau_gas=tau,
        width=width,
        init_laplace=init_laplace,
    )
    solver = CapillarySolver((L, L), cfg)
    solver.set_phi(tanh_droplet_phi((L, L), (0.5 * L, 0.5 * L), radius, width=width))
    hist = []
    done = 0
    while done < steps:
        solver.step(min(sample, steps - done))
        done += min(sample, steps - done)
        hist.append((done, solver.max_speed(), solver.pressure_jump(radius, 3.0)))
    dp_err = hist[-1][2] / laplace_pressure(sigma, radius, 2) - 1.0
    return {
        "case": "spurious",
        "params": {
            "radius": radius,
            "L": L,
            "sigma": sigma,
            "tau": tau,
            "mu": _viscosity(tau),
            "width": width,
            "steps": steps,
            "init_laplace": init_laplace,
        },
        "metrics": {
            "umax": hist[-1][1],
            "umax_early": hist[0][1],
            "dp_rel_err": dp_err,
            "h": 1.0 / radius,
        },
        "series": {"t": [h[0] for h in hist], "umax": [h[1] for h in hist]},
    }


# --------------------------------------------------------------------------- #
# case F -- mass conservation of the interface transport                      #
# --------------------------------------------------------------------------- #
def mass_conservation(
    steps: int = 1000,
    sigma: float = 0.05,
    radius: float = 12.0,
    L: int = 64,
    eps: float = 0.05,
    tau: float = 0.9,
    width: float = 3.0,
    compression: float = 0.0,
    sharpen: bool = True,
    sample: int = 50,
) -> dict:
    """``sum(phi)`` drift of the moving interface over ``steps`` lattice steps."""
    cfg = SolverConfig(
        sigma=sigma,
        tau_liquid=tau,
        tau_gas=tau,
        width=width,
        compression=compression,
        sharpen=sharpen,
        advection="sl",
    )
    solver = CapillarySolver((L, L), cfg)
    solver.set_phi(
        tanh_ellipse_phi(
            (L, L),
            (0.5 * L, 0.5 * L),
            (radius * (1.0 + eps), radius * (1.0 - eps)),
            width=width,
        )
    )
    m0 = solver.phi_mass()
    hist = []
    done = 0
    while done < steps:
        solver.step(min(sample, steps - done), advect=True)
        done += min(sample, steps - done)
        hist.append((done, solver.phi_mass(), solver.max_speed()))
    loss = abs(hist[-1][1] - m0) / m0
    return {
        "case": "mass_conservation",
        "params": {
            "steps": steps,
            "sigma": sigma,
            "radius": radius,
            "L": L,
            "eps": eps,
            "tau": tau,
            "compression": compression,
            "sharpen": sharpen,
        },
        "metrics": {"mass0": m0, "mass1": hist[-1][1], "mass_loss": loss},
        "series": {"t": [h[0] for h in hist], "mass": [h[1] for h in hist]},
    }