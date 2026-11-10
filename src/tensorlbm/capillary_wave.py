"""Capillary-wave dispersion verification for the CAC two-phase lattice Boltzmann model.

Why this module exists
----------------------
``tensorlbm.cac_lbm`` takes the surface tension ``sigma`` as an **input** parameter
(``free_energy_params``: ``beta = 12 sigma / W``, ``k = 1.5 sigma W``, Liang et al.
PRE 97, 033309, Eq. 7).  Every static check of that number -- the Laplace pressure
of a droplet, the equilibrium ``tanh`` profile -- tests only the *statics* of the
Korteweg stress.  A free-surface solver used for ship hydrodynamics (wave making,
air layers, ice/water interfaces) lives or dies on the **dynamics** of the same
coefficient: if the numerical interface responds to a curvature perturbation with
the wrong frequency, the whole free-surface force chain is wrong even though every
static benchmark is green.

This module supplies that missing verification rung: a standing capillary wave on a
layered interface, measured against the classical sharp-interface dispersion
relation of a two-fluid system without gravity

    omega^2 = sigma k^3 / (rho_l + rho_g),          k = 2 pi m / Lx        (1)

(Kelvin 1871; Landau & Lifshitz, *Fluid Mechanics*, 2nd ed. §62) and the
two-fluid viscous decay rate of the interface amplitude

    gamma = 2 nu k^2                                                       (2)

(Landau & Lifshitz §62, equal densities and equal viscosities).

Nothing here re-implements the solver: the time stepping is
:class:`tensorlbm.cac_lbm.CACSim`, and the only new code is (i) the perturbed
layered initial condition, (ii) an interface tracker that turns a phase field into
a height profile ``h(x, t)`` and its modal amplitude, and (iii) a damped-sinusoid
estimator for ``(omega, gamma)`` from the amplitude time series.

Usage
-----
    from tensorlbm.capillary_wave import CapillaryWaveConfig, run_capillary_wave
    res = run_capillary_wave(CapillaryWaveConfig(mode=2))
    print(res["omega_meas"] / res["omega_theory"] - 1.0)

References
----------
Liang, H., Xu, J., Chen, J., Wang, H., Chai, Z., Shi, B. (2018). Phase-field-based
    lattice Boltzmann modeling of large-density-ratio two-phase flows.
    Phys. Rev. E 97, 033309.
Landau, L. D. & Lifshitz, E. M. (1987). *Fluid Mechanics*, 2nd ed., §62
    (capillary waves on a fluid interface).
Kendon, V. M., Cates, M. E., Pagonabarraga, I., Desplat, J.-C., Bladon, P. (2001).
    Inertial effects in three-dimensional spinodal decomposition of a symmetric
    binary fluid mixture. J. Fluid Mech. 440, 147 -- capillary-wave validation of a
    free-energy LBM with exactly this dispersion relation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

__all__ = [
    "CapillaryWaveConfig",
    "cylinder_phi",
    "flat_interface_phi",
    "damped_sinusoid_fit",
    "damping_theory",
    "dispersion_theory",
    "interface_height",
    "modal_amplitude",
    "perturbed_layer_phi",
    "run_capillary_wave",
    "run_flat_interface",
    "run_laplace_drop",
    "sigma_from_profile",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapillaryWaveConfig:
    """A single standing-capillary-wave run.

    The interface is ``y = y0 + A cos(k x)`` with ``k = 2 pi mode / nx`` on a
    doubly-periodic domain; phase ``phi = 1`` below (liquid) and ``0`` above (gas).
    """

    ny: int = 32
    nx: int = 64
    nz: int = 2
    mode: int = 2
    amplitude: float = 0.5
    sigma: float = 0.05
    nu: float = 0.05
    width: float = 4.0
    rho_l: float = 1.0
    rho_g: float = 1.0
    n_steps: int = 3000
    sample_every: int = 10
    dtype: str = "float32"

    @property
    def k(self) -> float:
        """Wavenumber of the standing mode, ``2 pi mode / nx``."""
        return 2.0 * math.pi * self.mode / self.nx

    @property
    def steepness(self) -> float:
        """``A k`` -- the linear-theory small parameter."""
        return self.amplitude * self.k

    def validate(self) -> None:
        if self.mode < 1 or self.mode >= self.nx // 4:
            msg = "mode must satisfy 1 <= mode < nx/4 so the wave is resolved"
            raise ValueError(msg)
        if self.nz < 1 or self.ny < 8 or self.nx < 16:
            msg = "domain must be at least (nz>=1, nx>=16, ny>=8)"
            raise ValueError(msg)
        if self.sigma <= 0.0:
            msg = "sigma must be > 0 (this is a capillary-wave test)"
            raise ValueError(msg)
        if self.nu <= 0.0:
            msg = "nu (the CACSim mobility, i.e. the LBE viscosity) must be > 0"
            raise ValueError(msg)
        if not 2.0 <= self.width <= self.ny / 4.0:
            msg = "width must lie in [2, ny/4] (resolved, and away from the layers' far side)"
            raise ValueError(msg)
        if self.amplitude <= 0.0 or self.steepness > 0.15:
            msg = "amplitude must be > 0 and the steepness A*k must stay <= 0.15 (linear regime)"
            raise ValueError(msg)
        if self.rho_l <= 0.0 or self.rho_g <= 0.0:
            msg = "rho_l and rho_g must be > 0"
            raise ValueError(msg)
        if self.n_steps < 100 or self.sample_every < 1:
            msg = "n_steps must be >= 100 and sample_every >= 1"
            raise ValueError(msg)


# ---------------------------------------------------------------------------
# Theory
# ---------------------------------------------------------------------------


def dispersion_theory(sigma: float, k: float, rho_l: float, rho_g: float) -> float:
    """Sharp-interface capillary-wave frequency, ``sqrt(sigma k^3 / (rho_l + rho_g))``."""
    return math.sqrt(sigma * k**3 / (rho_l + rho_g))


def damping_theory(nu: float, k: float) -> float:
    """Two-fluid viscous amplitude decay rate ``gamma = 2 nu k^2`` (LL §62)."""
    return 2.0 * nu * k * k


# ---------------------------------------------------------------------------
# Initial condition and measurement
# ---------------------------------------------------------------------------


def perturbed_layer_phi(config: CapillaryWaveConfig, *, device: str = "cpu") -> torch.Tensor:
    """Phase field of the perturbed layered interface, shape ``(nz, ny, nx)``.

    ``phi = 0.5 + 0.5 tanh(2 (y_i(x) - y) / W)`` with ``y_i = ny/2 + A cos(kx)``,
    i.e. :func:`tensorlbm.cac_lbm.init_phi_layers` with a displaced interface.
    """
    config.validate()
    dt = torch.float64 if config.dtype == "float64" else torch.float32
    x = torch.arange(config.nx, dtype=dt, device=device).view(1, 1, -1)
    y = torch.arange(config.ny, dtype=dt, device=device).view(1, -1, 1)
    yi = 0.5 * config.ny + config.amplitude * torch.cos(config.k * x)
    phi = 0.5 + 0.5 * torch.tanh(2.0 * (yi - y) / config.width)
    return phi.expand(config.nz, config.ny, config.nx).contiguous()


def interface_height(
    phi: torch.Tensor, *, level: float = 0.5, row: int | None = None
) -> torch.Tensor:
    """Height ``h(x)`` of the ``phi = level`` contour, one value per column.

    ``phi`` is ``(nz, ny, nx)``; the contour is located by linear interpolation
    between the two rows that straddle ``level``, scanning upward from the bottom
    layer (``phi = 1``).  Because ``phi`` decreases monotonically with ``y`` here,
    the first crossing from below is the physical interface.
    """
    if phi.dim() != 3:
        msg = f"phi must be (nz, ny, nx), got shape {tuple(phi.shape)}"
        raise ValueError(msg)
    nz, ny, nx = phi.shape
    col = phi[nz // 2 if row is None else row]  # (ny, nx)
    below = col < level
    idx = (below.to(torch.int64).argmax(dim=0) - 1).clamp(0, ny - 2)
    lo = torch.gather(col, 0, idx.unsqueeze(0)).squeeze(0)
    hi = torch.gather(col, 0, (idx + 1).unsqueeze(0)).squeeze(0)
    frac = (lo - level) / torch.clamp(lo - hi, min=1e-12)
    return idx.to(col.dtype) + frac


def modal_amplitude(phi: torch.Tensor, k: float, *, level: float = 0.5) -> float:
    """Cosine-mode amplitude ``2/Lx * sum_x (h(x) - <h>) cos(kx)`` of the interface."""
    nz, ny, nx = phi.shape
    h = interface_height(phi, level=level)
    xs = torch.arange(nx, dtype=h.dtype, device=h.device)
    basis = torch.cos(k * xs)
    return float((2.0 / nx) * ((h - h.mean()) * basis).sum().item())


# ---------------------------------------------------------------------------
# (omega, gamma) estimator
# ---------------------------------------------------------------------------


def damped_sinusoid_fit(
    t: Any,
    a: Any,
    *,
    omega_bounds: tuple[float, float],
    gamma_bounds: tuple[float, float] = (0.0, 0.1),
    omega_points: int = 241,
    gamma_points: int = 61,
    refine: int = 4,
) -> dict[str, float]:
    """Least-squares fit of ``a(t) = e^{-gamma t}(C cos(omega t) + S sin(omega t))``.

    A coarse grid in ``(omega, gamma)`` is scanned and, for each pair, the two
    linear coefficients are solved exactly (variable projection); the best grid
    point is then refined on a shrinking box.  This avoids derivative-based
    optimisers on a residual that is notoriously multi-modal for short records.

    Returns ``omega``, ``gamma``, ``amplitude`` (``hypot(C, S)``), ``phase``,
    ``r2`` and ``rmse``.  Raises ``ValueError`` when fewer than 4 samples or a
    non-finite record is supplied -- a fitting failure must never be silently
    turned into a passing number.
    """
    import numpy as np

    tt = np.asarray(t, dtype=np.float64).ravel()
    aa = np.asarray(a, dtype=np.float64).ravel()
    keep = np.isfinite(tt) & np.isfinite(aa)
    tt, aa = tt[keep], aa[keep]
    if tt.size < 4:
        msg = "damped_sinusoid_fit needs at least 4 finite samples"
        raise ValueError(msg)

    def solve(w: float, g: float) -> tuple[float, np.ndarray]:
        env = np.exp(-g * tt)
        basis = np.stack([env * np.cos(w * tt), env * np.sin(w * tt)], axis=1)
        coef, *_ = np.linalg.lstsq(basis, aa, rcond=None)
        resid = aa - basis @ coef
        return float(resid @ resid), coef

    w_lo, w_hi = omega_bounds
    g_lo, g_hi = gamma_bounds
    best = (math.inf, 0.5 * (w_lo + w_hi), 0.5 * (g_lo + g_hi), np.zeros(2))
    for w in np.linspace(w_lo, w_hi, omega_points):
        for g in np.linspace(g_lo, g_hi, gamma_points):
            sse, coef = solve(w, g)
            if sse < best[0]:
                best = (sse, float(w), float(g), coef)

    sse, w_best, g_best, coef = best
    w_half = 0.5 * (w_hi - w_lo) / (omega_points - 1)
    g_half = 0.5 * (g_hi - g_lo) / (gamma_points - 1)
    for _ in range(refine):
        dw = w_half * 0.5 / max(omega_points - 1, 1) * 2.0
        dg = g_half * 0.5 / max(gamma_points - 1, 1) * 2.0
        for w in np.linspace(max(w_best - dw, 1e-9), w_best + dw, 41):
            for g in np.linspace(max(g_best - dg, 0.0), g_best + dg, 21):
                s2, c2 = solve(w, g)
                if s2 < sse:
                    sse, w_best, g_best, coef = s2, float(w), float(g), c2
        w_half = dw
        g_half = dg

    sst = float(((aa - aa.mean()) ** 2).sum())
    return {
        "omega": w_best,
        "gamma": g_best,
        "amplitude": float(math.hypot(*coef)),
        "phase": float(math.atan2(-coef[1], coef[0])),
        "r2": 1.0 - sse / sst if sst > 0.0 else float("nan"),
        "rmse": math.sqrt(sse / tt.size),
        "n_samples": int(tt.size),
    }


def cylinder_phi(shape: tuple[int, int, int], radius: float, width: float) -> torch.Tensor:
    nz, ny, nx = shape
    y = torch.arange(ny, dtype=torch.float64).view(1, -1, 1)
    x = torch.arange(nx, dtype=torch.float64).view(1, 1, -1)
    r = ((y - ny / 2.0) ** 2 + (x - nx / 2.0) ** 2).sqrt()
    return (0.5 + 0.5 * torch.tanh(2.0 * (radius - r) / width)).expand(nz, ny, nx).contiguous()


def run_laplace_drop(
    radius: float,
    *,
    width: float = 4.0,
    sigma: float = 0.05,
    nu: float = 0.05,
    n_steps: int = 3000,
    nz: int = 2,
    ny: int = 64,
    nx: int = 64,
) -> dict[str, Any]:
    """Relax one cylindrical drop and return the measured surface tension ``dp * R``."""
    from .cac_lbm import CACSim, _macro_fp64

    dev = torch.device("cpu")
    shape = (nz, ny, nx)
    sim = CACSim(
        shape,
        rho_l=1.0,
        rho_g=1.0,
        mu_l=nu,
        mu_g=nu,
        sigma=sigma,
        width=width,
        mobility=nu,
        axes="ppp",
        device=dev,
        dtype=torch.float32,
    )
    sim.initialize(cylinder_phi(shape, radius, width))
    for _ in range(n_steps):
        sim.step()
    phi, _rho, _grad, _mu, _F, u, p, _u_star = _macro_fp64(sim)
    y = torch.arange(ny, dtype=torch.float64).view(1, -1, 1)
    x = torch.arange(nx, dtype=torch.float64).view(1, 1, -1)
    r = ((y - ny / 2.0) ** 2 + (x - nx / 2.0) ** 2).sqrt().expand(shape)
    inside = r <= radius - 1.5 * width
    outside = r >= radius + 2.0 * width
    p_in = float(p[inside].mean().item())
    p_out = float(p[outside].mean().item())
    dp = p_in - p_out
    area = float((phi[nz // 2] >= 0.5).sum().item())
    r_eff = math.sqrt(area / math.pi)
    umax = float((u[0] ** 2 + u[1] ** 2 + u[2] ** 2).sqrt().max().item())
    return {
        "radius_input": radius,
        "radius_eff": r_eff,
        "width": width,
        "sigma_input": sigma,
        "nu": nu,
        "delta_p": dp,
        "sigma_measured": dp * r_eff,
        "sigma_ratio": dp * r_eff / sigma,
        "spurious_u_max": umax,
        "phi_range": [float(phi.min().item()), float(phi.max().item())],
    }


# ---------------------------------------------------------------------------
# Static surface tension from the interface profile
# ---------------------------------------------------------------------------


def flat_interface_phi(
    *, ny: int = 48, nx: int = 8, nz: int = 2, width: float = 4.0
) -> torch.Tensor:
    """Equilibrium layered interface ``phi = 0.5 + 0.5 tanh(2 (y0 - y) / W)``."""
    y = torch.arange(ny, dtype=torch.float64).view(1, -1, 1)
    return (0.5 + 0.5 * torch.tanh(2.0 * (0.5 * ny - y) / width)).expand(nz, ny, nx).contiguous()


def sigma_from_profile(phi: torch.Tensor, *, sigma_input: float, width: float) -> dict[str, float]:
    """Surface tension implied by the interface profile, ``sigma = k * int (dphi/dn)^2 dn``.

    For the CAC free energy the Korteweg stress of a planar interface integrates to
    ``sigma = k_coef * int (dphi/dn)^2 dn`` with ``k_coef = 1.5 sigma W`` (Liang et
    al. Eq. 7 with the ``tanh(2 (y0-y)/W)`` profile).  Evaluating the integral on the
    *simulated* profile therefore measures the surface tension that the discretised
    model actually carries, without any pressure diagnostic and without waiting for a
    droplet's slow viscous relaxation.

    The derivative is a central difference over the **interior** rows only: the
    periodic wrap row pair differs by ~1 in ``phi`` and would otherwise inject a
    spurious ``~0.5`` into the integral (a 3x bias, caught by this function's own test).
    """
    if phi.dim() != 3 or phi.shape[1] < 5:
        msg = "phi must be (nz, ny, nx) with ny >= 5"
        raise ValueError(msg)
    core = phi[:, 1:-1, :]
    d = 0.5 * (phi[:, 2:, :] - phi[:, :-2, :])  # central difference, interior rows
    integral = float((d * d).sum(dim=1).mean().item())
    k_coef = 1.5 * sigma_input * width
    slope_max = float(d.abs().max().item())
    return {
        "integral": integral,
        "sigma_profile": k_coef * integral,
        "sigma_ratio": k_coef * integral / sigma_input,
        "slope_max": slope_max,
        "width_measured": 0.5 / slope_max if slope_max > 0.0 else float("inf"),
        "width_analytic": width / 2.0,
        "phi_range_min": float(phi.min().item()),
        "phi_range_max": float(phi.max().item()),
        "n_rows_used": int(core.shape[1]),
    }


def run_flat_interface(
    *,
    sigma: float = 0.05,
    width: float = 4.0,
    nu: float = 0.05,
    ny: int = 48,
    nx: int = 8,
    nz: int = 2,
    n_steps: int = 4000,
    mobility: float | None = None,
) -> dict[str, Any]:
    """Relax a planar CAC interface and report its measured surface tension."""
    from .cac_lbm import CACSim

    dev = torch.device("cpu")
    phi0 = flat_interface_phi(ny=ny, nx=nx, nz=nz, width=width).to(torch.float32)
    sim = CACSim(
        (nz, ny, nx),
        rho_l=1.0,
        rho_g=1.0,
        mu_l=nu,
        mu_g=nu,
        sigma=sigma,
        width=width,
        mobility=nu if mobility is None else mobility,
        axes="ppp",
        device=dev,
        dtype=torch.float32,
    )
    sim.initialize(phi0)
    for _ in range(n_steps):
        sim.step()
    phi = sim.f.sum(dim=0)
    out = {
        "initial": sigma_from_profile(phi0, sigma_input=sigma, width=width),
        "relaxed": sigma_from_profile(phi, sigma_input=sigma, width=width),
        "sigma_input": sigma,
        "width": width,
        "nu": nu,
        "mobility": nu if mobility is None else mobility,
        "n_steps": n_steps,
        "phi_range_relaxed": [float(phi.min().item()), float(phi.max().item())],
    }
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_capillary_wave(config: CapillaryWaveConfig, *, device: str = "cpu") -> dict[str, Any]:
    """Run one standing capillary wave and return the measured vs theoretical dynamics.

    The returned dict contains the configuration, the amplitude time series, the
    ``(omega, gamma)`` fit, the theory values of Eqs. (1-2), the relative error of
    the frequency, and the effective surface tension implied by the measurement

        sigma_eff = omega_meas^2 (rho_l + rho_g) / k^3 ,

    which is the quantity to compare with the ``sigma`` *input* of the solver.
    """
    from .cac_lbm import CACSim

    config.validate()
    torch_dtype = torch.float64 if config.dtype == "float64" else torch.float32
    dev = torch.device(device)
    shape = (config.nz, config.ny, config.nx)
    phi0 = perturbed_layer_phi(config, device=device)
    sim = CACSim(
        shape,
        rho_l=config.rho_l,
        rho_g=config.rho_g,
        mu_l=config.nu,
        mu_g=config.nu,
        sigma=config.sigma,
        width=config.width,
        mobility=config.nu,
        axes="ppp",
        device=dev,
        dtype=torch_dtype,
    )
    sim.initialize(phi0)

    k = config.k
    omega_th = dispersion_theory(config.sigma, k, config.rho_l, config.rho_g)
    steps: list[int] = []
    amps: list[float] = []
    phi_min: list[float] = []
    phi_max: list[float] = []
    blow_up: int | None = None
    for step in range(1, config.n_steps + 1):
        sim.step()
        if step % config.sample_every == 0 or step == 1:
            phi = sim.f.sum(dim=0)
            a = modal_amplitude(phi, k)
            steps.append(step)
            amps.append(a)
            phi_min.append(float(phi.min().item()))
            phi_max.append(float(phi.max().item()))
            if not math.isfinite(a) or abs(a) > 5.0 * config.amplitude:
                blow_up = step
                break

    import numpy as np

    t = np.asarray(steps, dtype=float)
    a = np.asarray(amps, dtype=float)
    fit: dict[str, float] | None = None
    if blow_up is None and a.size >= 4:
        # Drop the initial acoustic/relaxation transient and fit the decaying mode.
        start = t >= max(5.0 * config.sample_every, 0.1 * t[-1])
        try:
            fit = damped_sinusoid_fit(
                t[start],
                a[start],
                omega_bounds=(0.25 * omega_th, 2.5 * omega_th),
                gamma_bounds=(0.0, max(6.0 * config.nu * k * k, 0.01)),
            )
        except ValueError:
            fit = None

    result: dict[str, Any] = {
        "config": asdict(config),
        "k": k,
        "steepness": config.steepness,
        "k_times_width": k * config.width,
        "omega_theory": omega_th,
        "gamma_theory": damping_theory(config.nu, k),
        "blow_up_step": blow_up,
        "phi_range": [min(phi_min), max(phi_max)],
        "steps": steps,
        "amplitudes": amps,
        "fit": fit,
    }
    if fit is not None:
        result["omega_meas"] = fit["omega"]
        result["gamma_meas"] = fit["gamma"]
        result["rel_err_omega"] = fit["omega"] / omega_th - 1.0
        result["rel_err_gamma"] = fit["gamma"] / result["gamma_theory"] - 1.0
        result["sigma_eff"] = fit["omega"] ** 2 * (config.rho_l + config.rho_g) / k**3
        result["sigma_ratio"] = result["sigma_eff"] / config.sigma
    return result
