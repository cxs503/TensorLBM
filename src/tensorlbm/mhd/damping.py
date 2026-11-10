"""Magnetic damping of a shear wave: exact decay-rate anchor for low-Rm MHD.

Physical problem
----------------
Take an unbounded conducting fluid, a uniform magnetic field
``B = B0 e_y`` and the *shear-wave* velocity field

    u_x(y, t) = U sin(k y) exp(-gamma t),      k = 2 pi m / L_y,

which has ``u . grad u = 0`` and ``grad p = 0``: the incompressible
Navier-Stokes equation reduces to a linear balance between viscous diffusion
and the Lorentz body force.  With ``Rm << 1`` the current density follows
Ohm's law ``j_z = sigma (E_z + u_x B0)``; in a domain with no electrodes the
axial electric field is set by the constraint of **zero net axial current**,
``<j_z> = 0``, which for a zero-mean shear wave gives ``E_z = 0`` and

    j_z = sigma B0 u_x,     F_x = (j x B)_x = -j_z B0 = -sigma B0^2 u_x.

The momentum balance then closes mode by mode,

    du_x/dt = nu d^2u_x/dy^2 - (sigma B0^2 / rho) u_x
    =>  u_x(y,t) = U sin(k y) exp(-gamma t),
        gamma = nu k^2 + sigma B0^2 / rho.                       (exact)

Writing the magnetic rate with the same reference length ``h`` that defines the
Hartmann number, ``Ha = B0 h sqrt(sigma / mu)``, gives the scaling law that the
node asks for,

    gamma_magnetic = sigma B0^2 / rho = nu Ha^2 / h^2   =>   gamma ~ Ha^2.

This is an *exact* solution of the full (nonlinear, incompressible) low-Rm MHD
system, so the measured decay rate is a genuine solver anchor rather than a
linearised approximation.  Two independent numbers are extracted: the total
rate ``gamma`` (compared with ``nu k^2 + nu Ha^2/h^2``) and the slope of the
measured magnetic rate vs ``Ha`` on a log-log ladder (theory: exactly 2).

Numerics
--------
D2Q9, BGK, ``float64``, fully periodic in the wall-normal direction (the
streamwise direction is uniform, so one column of nodes is exact), body force
through the second-order Guo splitting with the implicit half-step correction
(``u = (sum_i c_i f_i + F/2)/rho``; the force is velocity dependent, so the
closure is solved algebraically, see :func:`_force_velocity`).

The scheme is the same as the channel branch of :mod:`tensorlbm.mhd.hartmann`
(same D2Q9 lattice, same Guo forcing, same implicit velocity closure); the only
difference is the boundary treatment (bounce-back walls there, periodicity
here), which is what makes the two verifications independent anchors of the
same collision kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from .hartmann import C_D2Q9, W_D2Q9

# ---------------------------------------------------------------------------
# Exact solution
# ---------------------------------------------------------------------------


def wave_number(ny: int, mode: int = 1, length: float | None = None) -> float:
    """Shear-wave wavenumber ``k = 2 pi mode / L``.

    ``length`` defaults to the box height ``ny`` (one lattice unit per row), so
    the mode is periodic on the domain by construction.
    """
    if ny < 4:
        raise ValueError(f"ny must be >= 4, got {ny}")
    if mode < 1:
        raise ValueError(f"mode must be >= 1, got {mode}")
    box = float(ny if length is None else length)
    return 2.0 * math.pi * mode / box


def magnetic_decay_rate(ha: float, nu: float, half_width: float) -> float:
    """Magnetic damping rate ``sigma B0^2/rho = nu Ha^2 / h^2`` (exact)."""
    if half_width <= 0.0:
        raise ValueError(f"half_width must be > 0, got {half_width}")
    return nu * ha * ha / (half_width * half_width)


def viscous_decay_rate(nu: float, k: float) -> float:
    """Viscous damping rate of the shear mode, ``nu k^2`` (exact)."""
    return nu * k * k


def exact_amplitude(case: "ShearWaveCase", t: float) -> float:
    """Exact shear-wave amplitude ``U exp(-gamma t)`` at time ``t``."""
    return case.amplitude * math.exp(-case.rate * t)


def exact_profile(case: "ShearWaveCase", t: float) -> np.ndarray:
    """Exact ``u_x(y, t)`` on the node rows of ``case``."""
    return (
        case.amplitude
        * np.sin(case.wave_number * case.rows)
        * math.exp(-case.rate * t)
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShearWaveCase:
    """One periodic shear-wave damping run.

    Parameters
    ----------
    ny, nx:
        Rows in the wall-normal direction and (uniform) streamwise columns.
    ha:
        Hartmann number based on the half-height ``h = ny/2``; the conductivity
        that realises it is derived exactly as in the channel branch.
    nu, rho0:
        Kinematic viscosity and density in lattice units (``mu = rho0 nu``).
    amplitude:
        Peak velocity of the initial shear wave (kept inside the low-Mach
        verification window; ``amplitude <= 0.05``).
    mode:
        Integer mode number over the box.
    steps:
        Number of lattice steps; ``None`` = enough for ``decay_periods``
        e-folds at the exact rate.
    decay_periods:
        Target number of e-folds of the exact solution when ``steps`` is None.
    sample_every:
        Sampling stride of the amplitude history.
    """

    ny: int = 32
    nx: int = 1
    ha: float = 10.0
    nu: float = 0.05
    rho0: float = 1.0
    b0: float = 1.0
    amplitude: float = 0.02
    mode: int = 1
    steps: int | None = None
    decay_periods: float = 4.0
    sample_every: int = 5
    dtype: str = "float64"

    def __post_init__(self) -> None:
        if self.ny < 4:
            raise ValueError(f"ny must be >= 4, got {self.ny}")
        if self.nx < 1:
            raise ValueError(f"nx must be >= 1, got {self.nx}")
        if self.ha < 0.0:
            raise ValueError(f"ha must be >= 0, got {self.ha}")
        if self.nu <= 0.0:
            raise ValueError(f"nu must be > 0, got {self.nu}")
        if self.rho0 <= 0.0:
            raise ValueError(f"rho0 must be > 0, got {self.rho0}")
        if not 0.0 < self.amplitude <= 0.05:
            raise ValueError(
                f"amplitude={self.amplitude} leaves the low-Mach window (0, 0.05]"
            )
        if self.mode < 1:
            raise ValueError(f"mode must be >= 1, got {self.mode}")
        if self.sample_every < 1:
            raise ValueError(f"sample_every must be >= 1, got {self.sample_every}")

    # -- geometry ----------------------------------------------------------
    @property
    def half_width(self) -> float:
        """Reference length ``h = ny/2`` (same convention as the channel branch)."""
        return 0.5 * self.ny

    @property
    def rows(self) -> np.ndarray:
        """Node-centre coordinates in ``[0, ny)`` lattice units."""
        return np.arange(self.ny, dtype=float)

    @property
    def wave_number(self) -> float:
        return wave_number(self.ny, self.mode)

    # -- physical coefficients --------------------------------------------
    @property
    def mu(self) -> float:
        return self.rho0 * self.nu

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.nu

    @property
    def sigma(self) -> float:
        """Conductivity that realises ``ha`` at ``b0`` and ``h = ny/2``."""
        if self.b0 == 0.0:
            return 0.0
        h = self.half_width
        return self.mu * self.ha * self.ha / (self.b0 * self.b0 * h * h)

    @property
    def force_coefficient(self) -> float:
        """``sigma B0^2``: Lorentz force density per unit velocity."""
        return self.sigma * self.b0 * self.b0

    @property
    def viscous_rate(self) -> float:
        return viscous_decay_rate(self.nu, self.wave_number)

    @property
    def magnetic_rate(self) -> float:
        return magnetic_decay_rate(self.ha, self.nu, self.half_width)

    @property
    def rate(self) -> float:
        """Exact total decay rate ``nu k^2 + nu Ha^2/h^2``."""
        return self.viscous_rate + self.magnetic_rate

    @property
    def decay_steps(self) -> int:
        if self.steps is not None:
            return int(self.steps)
        return max(20, int(math.ceil(self.decay_periods / self.rate)))

    def initial_profile(self) -> np.ndarray:
        """Initial ``u_x`` on the node rows (zero mean by construction)."""
        return self.amplitude * np.sin(self.wave_number * self.rows)

    def validate(self) -> None:
        if self.tau <= 0.5:
            raise ValueError(f"tau={self.tau} must exceed 0.5")
        if abs(self.initial_profile().mean()) > 1e-10 * self.amplitude:
            raise ValueError("initial shear wave must have zero mean (mode >= 1)")
        if self.decay_steps < 1:
            raise ValueError("steps must be >= 1")


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def _force_velocity(
    m: torch.Tensor, force_coeff: float, rho: torch.Tensor, loading: float
) -> torch.Tensor:
    """Solve ``u = (m + F(u)/2)/rho`` with ``F = -force_coeff (u - loading)``.

    The Lorentz force of the low-Rm closure is *velocity dependent*, so the
    half-step velocity that enters the Guo source must be implicit.  With the
    linear closure the implicit relation is algebraic and exact:

        u (1 + force_coeff/(2 rho)) = m/rho + force_coeff * loading/(2 rho).
    """
    scale = 1.0 + force_coeff / (2.0 * rho)
    return (m + 0.5 * force_coeff * loading) / (rho * scale)


def run_damping_case(case: ShearWaveCase) -> dict[str, Any]:
    """Run one shear-wave damping case and score the measured decay rate.

    Returns a JSON-friendly dict: the sampled amplitude history, the fitted
    decay rate, the exact rate and every relative error.
    """
    case.validate()
    dtype = getattr(torch, case.dtype)
    c = torch.tensor(C_D2Q9, dtype=dtype)
    w = torch.tensor(W_D2Q9, dtype=dtype).reshape(9, 1)
    cx = c[:, 0].reshape(9, 1)
    cy = c[:, 1].reshape(9, 1)
    ny = case.ny
    rho = torch.full((ny,), case.rho0, dtype=dtype)
    ux = torch.as_tensor(case.initial_profile(), dtype=dtype)
    uy = torch.zeros_like(ux)
    cu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0)
    usq = (ux * ux + uy * uy).unsqueeze(0)
    f = w * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
    tau = case.tau
    force_coeff = case.force_coefficient
    sin_k = torch.as_tensor(np.sin(case.wave_number * case.rows), dtype=dtype)
    # zero net-current closure: E_z = -B0 <u_x>, i.e. the loading is the mean
    # velocity (updated each step; exactly zero for an integer-mode shear wave).
    loading = float(ux.mean())

    def project(u: torch.Tensor) -> float:
        """Amplitude of the ``sin(k y)`` mode: ``2 <u sin ky>``."""
        return float(2.0 * (u * sin_k).mean())

    history: list[dict[str, float]] = [
        {"t": 0.0, "amplitude": project(ux), "exact": case.amplitude}
    ]
    energy: list[float] = [0.5 * case.rho0 * float((ux * ux).mean())]
    for step in range(1, case.decay_steps + 1):
        rho = f.sum(dim=0)
        m = (cx * f).sum(dim=0)
        # implicit half-step velocity of the velocity-dependent Lorentz force
        ux = _force_velocity(m, force_coeff, rho, loading)
        uy = torch.zeros_like(ux)
        loading = float(ux.mean())
        fx = -force_coeff * (ux - loading)
        fy = torch.zeros_like(fx)
        cu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0)
        cuf = cx * fx.unsqueeze(0) + cy * fy.unsqueeze(0)
        usq = (ux * ux + uy * uy).unsqueeze(0)
        feq = w * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
        src = (1.0 - 0.5 / tau) * w * (3.0 * cuf + 9.0 * cu * cuf)
        f_post = f - (f - feq) / tau + src
        # periodic streaming along the wall-normal direction
        f = torch.stack(
            [torch.roll(f_post[i], shifts=-int(cy[i]), dims=0) for i in range(9)]
        )
        if step % case.sample_every == 0 or step == case.decay_steps:
            u_row = (cx * f).sum(dim=0) / f.sum(dim=0)
            history.append(
                {
                    "t": float(step),
                    "amplitude": project(u_row),
                    "exact": exact_amplitude(case, float(step)),
                }
            )
            energy.append(0.5 * case.rho0 * float((u_row * u_row).mean()))

    t_arr = np.asarray([h["t"] for h in history], dtype=float)
    a_arr = np.asarray([h["amplitude"] for h in history], dtype=float)
    gamma_fit, intercept, r2 = _fit_rate(t_arr, a_arr)
    # energy decays at twice the amplitude rate: sqrt(E) ~ exp(-gamma t)
    t_e = t_arr[: len(energy)]
    gamma_energy, _, _ = _fit_rate(t_e, np.sqrt(np.maximum(np.asarray(energy), 1e-300)))
    a_end = float(a_arr[-1])
    a_end_exact = exact_amplitude(case, float(case.decay_steps))
    u_final = ((cx * f).sum(dim=0) / f.sum(dim=0)).cpu().numpy()
    return {
        "ny": ny,
        "nx": case.nx,
        "ha": case.ha,
        "nu": case.nu,
        "rho0": case.rho0,
        "b0": case.b0,
        "half_width": case.half_width,
        "mode": case.mode,
        "wave_number": case.wave_number,
        "amplitude0": case.amplitude,
        "steps": case.decay_steps,
        "sigma": case.sigma,
        "force_coefficient": case.force_coefficient,
        "gamma_fit": gamma_fit,
        "gamma_exact": case.rate,
        "gamma_viscous": case.viscous_rate,
        "gamma_magnetic": case.magnetic_rate,
        "gamma_magnetic_fit": gamma_fit - case.viscous_rate,
        "rel_rate_error": abs(gamma_fit - case.rate) / case.rate,
        "rel_magnetic_error": (
            abs(gamma_fit - case.viscous_rate - case.magnetic_rate)
            / case.magnetic_rate
            if case.magnetic_rate > 0.0
            else 0.0
        ),
        "gamma_energy_fit": gamma_energy,
        "fit_intercept": intercept,
        "fit_r2": r2,
        "amplitude_end": a_end,
        "amplitude_end_exact": a_end_exact,
        "rel_amplitude_end": abs(a_end - a_end_exact) / max(abs(a_end_exact), 1e-300),
        "max_mach": float(np.max(np.abs(u_final))) * math.sqrt(3.0),
        "history": history,
        "profile": {
            "y": [float(v) for v in case.rows],
            "u_lbm": [float(v) for v in u_final],
            "u_exact": [float(v) for v in exact_profile(case, float(case.decay_steps))],
        },
    }


def _fit_rate(t: np.ndarray, a: np.ndarray) -> tuple[float, float, float]:
    """Least-squares fit of ``ln a = intercept - gamma t`` -> ``(gamma, c, r2)``."""
    mask = a > 0.0
    if mask.sum() < 3:
        raise ValueError("need at least three positive amplitude samples to fit a rate")
    x = t[mask]
    y = np.log(a[mask])
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return -float(slope), float(intercept), r2


def ha_scaling_exponent(ha_values: Sequence[float], rates: Sequence[float]) -> float:
    """Log-log slope of the magnetic rate vs ``Ha`` (theory: 2)."""
    ha_arr = np.asarray(ha_values, dtype=float)
    r_arr = np.asarray(rates, dtype=float)
    if ha_arr.size < 2:
        raise ValueError("need at least two Hartmann numbers")
    mask = (ha_arr > 0.0) & (r_arr > 0.0)
    if mask.sum() < 2:
        raise ValueError("need at least two positive (Ha, rate) pairs")
    slope, _ = np.polyfit(np.log(ha_arr[mask]), np.log(r_arr[mask]), 1)
    return float(slope)


THRESHOLDS = {
    "decay_rate": 0.05,
    "scaling_exponent": 0.10,
    "viscous_control": 0.05,
}


def evaluate_damping_criteria(
    runs: dict[str, dict[str, Any]],
    exponent: float,
    control: dict[str, Any],
) -> dict[str, Any]:
    """Machine criteria of the magnetic-damping branch (node criteria (d))."""
    worst = max(runs.values(), key=lambda r: r["rel_rate_error"])
    return {
        "c4_damping_rate": {
            "id": "c4_damping_rate",
            "value": worst["rel_rate_error"],
            "threshold": THRESHOLDS["decay_rate"],
            "passed": bool(worst["rel_rate_error"] <= THRESHOLDS["decay_rate"]),
            "worst_ha": worst["ha"],
            "note": (
                "|gamma_fit - (nu k^2 + nu Ha^2/h^2)| / gamma_exact, worst Ha of the "
                "Ha = 5/10/20 ladder"
            ),
        },
        "c4b_scaling_exponent": {
            "id": "c4b_scaling_exponent",
            "value": exponent,
            "target": 2.0,
            "tolerance": THRESHOLDS["scaling_exponent"],
            "passed": bool(abs(exponent - 2.0) <= THRESHOLDS["scaling_exponent"]),
            "note": "log-log slope of the measured magnetic rate vs Ha (theory: exactly 2)",
        },
        "c4c_viscous_control": {
            "id": "c4c_viscous_control",
            "value": control["rel_rate_error"],
            "threshold": THRESHOLDS["viscous_control"],
            "passed": bool(control["rel_rate_error"] <= THRESHOLDS["viscous_control"]),
            "note": "Ha = 0 control: the measured rate must be nu k^2 (pure viscous)",
        },
    }


def damping_refinement(
    ha: float = 10.0,
    ny_list: Sequence[int] = (16, 32, 64, 128),
    nu: float = 0.05,
    amplitude: float = 0.02,
) -> dict[str, Any]:
    """Grid refinement of the measured damping rate (observed order, theory ~2).

    The shear wave is one Fourier mode, so refining ``ny`` at fixed ``k`` resolves
    the same physical decay with a smaller truncation error; the rate error is
    expected to fall like ``dy^2`` (BGK + Guo forcing is second order in space
    and time).  This is the independence check of criterion (f) on the *damping*
    branch: it needs no walls and no steady state.
    """
    runs: dict[str, dict[str, Any]] = {}
    for ny in ny_list:
        run = run_damping_case(ShearWaveCase(ny=int(ny), ha=ha, nu=nu, amplitude=amplitude))
        runs[str(int(ny))] = run
    errs = [runs[str(int(ny))]["rel_rate_error"] for ny in ny_list]
    orders = [
        float(np.log(errs[i] / errs[i + 1]) / np.log(float(ny_list[i + 1]) / float(ny_list[i])))
        for i in range(len(errs) - 1)
        if errs[i] > 0.0 and errs[i + 1] > 0.0
    ]
    return {
        "ha": ha,
        "ny_list": [int(v) for v in ny_list],
        "rel_rate_errors": errs,
        "orders": orders,
        "finest_order": orders[-1] if orders else None,
        "runs": {k: {kk: vv for kk, vv in v.items() if kk not in ("history", "profile")}
                 for k, v in runs.items()},
    }


def damping_ladder(
    ha_values: Sequence[float] = (5.0, 10.0, 20.0),
    ny: int = 32,
    nu: float = 0.05,
    amplitude: float = 0.02,
    progress: bool = True,
) -> dict[str, Any]:
    """Ha ladder of shear-wave damping runs plus the ``gamma ~ Ha^2`` slope."""
    runs: dict[str, dict[str, Any]] = {}
    for ha in ha_values:
        case = ShearWaveCase(ny=ny, ha=float(ha), nu=nu, amplitude=amplitude)
        run = run_damping_case(case)
        runs[f"{float(ha):g}"] = run
        if progress:
            print(
                f"[mhd-damping] Ha={float(ha):5.2f} ny={ny:3d} steps={run['steps']:5d} "
                f"gamma_fit={run['gamma_fit']:.6e} exact={run['gamma_exact']:.6e} "
                f"rel={run['rel_rate_error']:.2e}",
                flush=True,
            )
    mag_rates = [runs[f"{float(ha):g}"]["gamma_magnetic_fit"] for ha in ha_values]
    exponent = ha_scaling_exponent([float(h) for h in ha_values], mag_rates)
    # pure-viscous control: Ha = 0 must reproduce nu k^2 (no magnetic damping)
    control = run_damping_case(
        ShearWaveCase(ny=ny, ha=0.0, nu=nu, amplitude=amplitude)
    )
    return {
        "ny": ny,
        "nu": nu,
        "amplitude": amplitude,
        "ha_values": [float(h) for h in ha_values],
        "runs": runs,
        "magnetic_rates_fit": [float(v) for v in mag_rates],
        "scaling_exponent": exponent,
        "viscous_control": control,
        "criteria": evaluate_damping_criteria(runs, exponent, control),
    }