"""Collision-operator stability domain: prediction / measurement exchange law.

This module turns the linear-stability domain of a D2Q9 collision operator from
"experience with blow-ups" into a quantity that is *predicted* (von Neumann
eigen-analysis of the linearised collision + streaming operator) and
*measured* (independent time integration of the full nonlinear periodic-box
LBM), and it defines the exchange law between the two: the predicted and the
measured divergence boundaries of the same ``(tau, Ma)`` scan must agree.

Why a separate module
---------------------
:mod:`tensorlbm.stability.von_neumann` owns the BGK/MRT amplification matrix
with *fixed* MRT rates (``s_e, s_eps, s_q`` and ``s_pxx = s_pxy = 1/tau``).
The question asked here is different: which of the nine moment relaxation rates
may be released, and how much stability does that buy?  That needs (a) a
per-slot rate vector, (b) pluggable moment bases (so that a non-orthogonal
basis can be audited as a *defect*), and (c) a measured side cheap enough to
run on a 30x30+ ``(tau, Ma)`` plane.

Conventions (declared before any run, see :data:`PREDECLARED`)
-------------------------------------------------------------
* Lattice units ``dx = dt = 1``, ``cs^2 = 1/3``.
* ``Ma`` is the Mach number of the base flow: ``u0 = (Ma * cs, 0)``.
* MRT slot ordering (Lallemand & Luo 2000), rows of ``M`` in this order::

      ("rho", "e", "eps", "jx", "qx", "jy", "qy", "pxx", "pxy")
         s0    s1   s2    s3  s4   s5  s6   s7     s8

  ``s7`` (``pxx``) and ``s8`` (``pxy``) are the shear rates tied to the
  viscosity, ``s7 = s8 = 1/tau``; ``s4`` (``qx``) is the x energy-flux rate.
* **Stability test**: ``lambda_max = max_{k != 0} max_j |mu_j(k)| <= 1``.
  The ``k = 0`` mean mode is excluded: the conserved moments make it exactly
  neutral for every collision model, so including it would make every model
  "marginal" and destroy the discrimination we are after.
* **Measured growth**: a single Fourier mode of the box grid is seeded in the
  full nonlinear float64 solver and its amplitude is fitted; the measured
  per-step amplification is ``lambda_meas``.  ``k`` is taken from the
  prediction (mode *selection* only) -- the time integration and the
  eigen-analysis share nothing else.

References
----------
* Sterling & Chen, J. Comput. Phys. 123 (1996) 196.
* Lallemand & Luo, Phys. Rev. E 61 (2000) 6546.
* D'Humieres, Prog. Astronaut. Aeronaut. 159 (1992) 450.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .von_neumann import (
    CX,
    CY,
    W,
    M_D2Q9,
    _I9,
    equilibrium_jacobian,
)

CS = 1.0 / math.sqrt(3.0)

MOMENT_SLOTS: tuple[str, ...] = ("rho", "e", "eps", "jx", "qx", "jy", "qy", "pxx", "pxy")

#: von Neumann verdict tolerance on ``lambda_max`` (prediction side).
STABILITY_TOL = 1e-9
#: Measurement-side tolerance on the fitted amplification factor.
MEASURED_TOL = 2e-3

#: All thresholds and grids used by the campaign.  Declared *before* the runs
#: and never retuned afterwards; the campaign JSON embeds a copy of this dict.
PREDECLARED: dict[str, Any] = {
    "lattice": "D2Q9",
    "ma_definition": "u0 = (Ma * cs, 0), cs = 1/sqrt(3)",
    "tau_window": (0.505, 0.9),
    "ma_window": (0.02, 0.5),
    "scan_grid": (35, 35),  # >= 30x30 cases per tier as required
    "k_grid_scan": 17,  # k-points per axis for the (tau, Ma) plane
    "k_grid_oracle": 17,  # cheaper grid used inside point-wise rate selection
    "k_grid_design": 17,  # coarse grid for the per-tau rate design rule
    "design_ma": 0.5,  # worst-case Mach number used by the per-tau design rule
    "k_grid_boundary": 81,  # fine grid for boundary localisation
    "stability_tol": STABILITY_TOL,
    "measured_tol": MEASURED_TOL,
    "measure_box": 32,
    "measure_steps": 400,
    "measure_amplitude": 1e-9,
    "measure_seed": 7,
    "broadband_box": 16,
    "broadband_steps": 200,
    "broadband_amplitude": 1e-4,
    "broadband_seed": 11,
    "broadband_ratio_threshold": 1e3,
    "broadband_window": 10,
    "bisection_iters": 9,
    "mrt_a_rates": {"s_e": 1.64, "s_eps": 1.54, "s_qx": 1.7, "s_qy": 1.7},
    "rate_candidates": {
        "s_e": (1.0, 1.2, 1.4, 1.64, 1.8, 1.99),
        "s_eps": (1.0, 1.2, 1.4, 1.54, 1.8, 1.99),
        "s_qx": (0.8, 1.2, 1.5, 1.7, 1.9, 1.99),
    },
    # Foils used by the falsifiability leg (M3).  Both are *legal* MRT settings
    # (every rate inside (0, 2) and an orthogonal moment basis):
    #   s4_under_relaxed -- canonical Lallemand & Luo high-order rates with the
    #       x energy-flux moment under-relaxed (the single knob MRT-B releases),
    #   mismatched -- the worst combination of the declared candidate sets at the
    #       declared probe point (tau = 0.6, Ma = 0.4); deliberately adversarial.
    "foil_rates": {
        "s4_under_relaxed": {"s_e": 1.64, "s_eps": 1.54, "s_qx": 0.8, "s_qy": 1.7},
        "mismatched": {"s_e": 1.99, "s_eps": 1.0, "s_qx": 0.8, "s_qy": 1.7},
    },
    "foil_probe": {"tau": 0.6, "ma": 0.4},
    "negative_controls": {
        "over_relax_shear": {"slot": "pxy", "rate": 2.4},
        "over_relax_energy": {"slot": "e", "rate": 2.6},
        "mixed_basis": {"mix_beta": 0.8, "mix_from": "jx", "mix_into": "qx", "s_qx": 1.95},
        "mixed_basis_over_relax": {"mix_beta": 0.8, "mix_from": "jx", "mix_into": "qx", "s_qx": 2.2},
    },
}


# ---------------------------------------------------------------------------
# Moment bases
# ---------------------------------------------------------------------------


def orthogonal_basis() -> np.ndarray:
    """The Lallemand & Luo D2Q9 moment matrix (rows = moments, columns = f_i)."""
    return np.array(M_D2Q9, dtype=float, copy=True)


def mixed_basis(mix_from: str = "jx", mix_into: str = "qx", beta: float = 0.8) -> np.ndarray:
    """A *non-orthogonal* moment basis used as a negative control.

    Row ``mix_into`` is replaced by ``row_into + beta * row_from``.  The basis
    stays invertible (it is a triangular mixing) but its rows are no longer
    mutually orthogonal, so ``M^-1 diag(s) M`` is no longer the intended MRT
    operator: it leaks a conserved moment into a kinetic one.  This is exactly
    the class of "wrong implementation" a stability diagnostic must reject.
    """
    basis = orthogonal_basis()
    i, j = MOMENT_SLOTS.index(mix_into), MOMENT_SLOTS.index(mix_from)
    basis[i] = basis[i] + float(beta) * basis[j]
    return basis


def basis_orthogonality_defect(basis: np.ndarray) -> float:
    """Frobenius norm of the off-diagonal part of ``M M^T`` (0 = orthogonal basis).

    The Lallemand & Luo D2Q9 moment rows are orthogonal with respect to the
    *unweighted* scalar product ``sum_i m_a(i) m_b(i)``, which is the property
    the MRT construction relies on; a mixed (non-orthogonal) basis has a
    strictly positive defect.
    """
    basis = np.asarray(basis, dtype=float)
    gram = basis @ basis.T
    return float(np.linalg.norm(gram - np.diag(np.diag(gram))))


# ---------------------------------------------------------------------------
# Collision models
# ---------------------------------------------------------------------------


def rates_vector(
    tau: float,
    *,
    s_e: float = 1.64,
    s_eps: float = 1.54,
    s_qx: float = 1.7,
    s_qy: float = 1.7,
    s_pxx: float | None = None,
    s_pxy: float | None = None,
) -> np.ndarray:
    """MRT relaxation-rate vector in :data:`MOMENT_SLOTS` order."""
    shear = 1.0 / float(tau)
    return np.array(
        [
            0.0,
            float(s_e),
            float(s_eps),
            0.0,
            float(s_qx),
            0.0,
            float(s_qy),
            shear if s_pxx is None else float(s_pxx),
            shear if s_pxy is None else float(s_pxy),
        ],
        dtype=float,
    )


@dataclass
class CollisionModel:
    """A collision operator: moment basis + one relaxation rate per moment."""

    name: str
    tau: float
    rates: np.ndarray
    basis: np.ndarray = field(default_factory=orthogonal_basis)
    genus: str = "mrt"
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tau = float(self.tau)
        if not math.isfinite(self.tau) or self.tau <= 0.0:
            raise ValueError(f"tau must be finite and > 0, got {self.tau}")
        self.rates = np.asarray(self.rates, dtype=float).reshape(9)
        self.basis = np.asarray(self.basis, dtype=float).reshape(9, 9)
        self.basis_inv = np.linalg.inv(self.basis)
        if not np.all(np.isfinite(self.basis_inv)):
            raise ValueError("moment basis is singular")

    @property
    def viscosity(self) -> float:
        return (self.tau - 0.5) / 3.0

    @property
    def max_rate(self) -> float:
        return float(np.max(np.abs(self.rates)))

    def operator_matrix(self) -> np.ndarray:
        """Collision matrix ``C`` with ``f^post = f - C (f - f^eq)``."""
        return self.basis_inv @ np.diag(self.rates) @ self.basis

    def map_matrix(self, ma: float) -> np.ndarray:
        """Linearised post-collision map ``P = d f^post / d f`` at the base flow."""
        j_eq = equilibrium_jacobian(1.0, (float(ma) * CS, 0.0))
        return _I9 - self.operator_matrix() @ (_I9 - j_eq)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "genus": self.genus,
            "tau": self.tau,
            "viscosity": self.viscosity,
            "rates": [float(r) for r in self.rates],
            "rates_named": {slot: float(r) for slot, r in zip(MOMENT_SLOTS, self.rates)},
            "max_rate": self.max_rate,
            "basis_orthogonality_defect": basis_orthogonality_defect(self.basis),
            "notes": dict(self.notes),
        }


def bgk_model(tau: float, name: str = "BGK") -> CollisionModel:
    """Single-relaxation-time model: every moment relaxes with ``1/tau``."""
    return CollisionModel(
        name=name,
        tau=tau,
        rates=np.full(9, 1.0 / float(tau)),
        genus="bgk",
        notes={"rule": "all nine rates = 1/tau"},
    )


def mrt_model(
    tau: float,
    *,
    s_e: float = 1.64,
    s_eps: float = 1.54,
    s_qx: float = 1.7,
    s_qy: float = 1.7,
    s_pxx: float | None = None,
    s_pxy: float | None = None,
    name: str = "MRT",
    basis: np.ndarray | None = None,
    notes: dict[str, Any] | None = None,
) -> CollisionModel:
    """MRT model with free high-order rates and shear rate ``1/tau``."""
    return CollisionModel(
        name=name,
        tau=tau,
        rates=rates_vector(
            tau, s_e=s_e, s_eps=s_eps, s_qx=s_qx, s_qy=s_qy, s_pxx=s_pxx, s_pxy=s_pxy
        ),
        basis=orthogonal_basis() if basis is None else np.asarray(basis, dtype=float),
        genus="mrt",
        notes=dict(notes or {}),
    )


# ---------------------------------------------------------------------------
# Wave-vector sets and von Neumann prediction
# ---------------------------------------------------------------------------


def quarter_wave_vectors(n: int, kmin: float = 1e-12) -> np.ndarray:
    """``(n*n - 1, 2)`` first-quadrant wave vectors in ``[0, pi]^2``, ``k != 0``.

    With a base flow along ``x`` on the D2Q9 lattice the stability domain is
    symmetric under ``k -> -k`` and ``ky -> -ky``, so the closed first quadrant
    exhausts the spectrum.  ``k = 0`` is dropped (see module docstring).
    """
    theta = np.linspace(0.0, math.pi, int(n))
    kx, ky = np.meshgrid(theta, theta, indexing="ij")
    kx, ky = kx.ravel(), ky.ravel()
    keep = np.hypot(kx, ky) > kmin
    return np.stack([kx[keep], ky[keep]], axis=1)


def box_wave_vectors(n: int) -> np.ndarray:
    """Wave vectors of a periodic ``n x n`` box that are non-trivial (``k != 0``)."""
    m = np.arange(0, int(n) // 2 + 1)
    kx, ky = np.meshgrid(2.0 * math.pi * m / n, 2.0 * math.pi * m / n, indexing="ij")
    kx, ky = kx.ravel(), ky.ravel()
    keep = ~((kx == 0.0) & (ky == 0.0))
    return np.stack([kx[keep], ky[keep]], axis=1)


def amplification_eigenvalues(maps: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Eigenvalues of ``A_i(k) = diag(exp(-i k.c)) P_i`` for a stack of maps.

    Parameters
    ----------
    maps : ``(M, 9, 9)`` real matrices ``P``.
    k : ``(K, 2)`` wave vectors.

    Returns
    -------
    ``(M, K, 9)`` complex eigenvalues.
    """
    maps = np.asarray(maps, dtype=float)
    if maps.ndim == 2:
        maps = maps[None]
    k = np.asarray(k, dtype=float).reshape(-1, 2)
    phase = np.exp(-1j * (k[:, 0:1] * CX + k[:, 1:2] * CY))  # (K, 9)
    amp = phase[None, :, :, None] * maps[:, None, :, :]  # (M, K, 9, 9)
    eigs = np.linalg.eigvals(amp.reshape(-1, 9, 9))
    return eigs.reshape(maps.shape[0], k.shape[0], 9)


def lambda_max_batch(models: Sequence[CollisionModel], ma: float, k: np.ndarray) -> np.ndarray:
    """``lambda_max`` of several collision models at one ``Ma`` and a wave-vector set."""
    maps = np.stack([m.map_matrix(ma) for m in models], axis=0)
    return np.abs(amplification_eigenvalues(maps, k)).max(axis=(1, 2))


def spectral_radius(
    model: CollisionModel,
    ma: float,
    *,
    n_theta: int = 33,
    k: np.ndarray | None = None,
) -> dict[str, Any]:
    """von Neumann verdict of one model at one ``(tau, Ma)`` point."""
    if k is None:
        k = quarter_wave_vectors(n_theta)
    eigs = amplification_eigenvalues(np.stack([model.map_matrix(ma)]), k)[0]
    absmax = np.abs(eigs).max(axis=1)
    j = int(absmax.argmax())
    lam_max = float(absmax[j])
    return {
        "lambda_max": lam_max,
        "growth_rate": math.log(lam_max) if lam_max > 0 else float("-inf"),
        "k_argmax": (float(k[j, 0]), float(k[j, 1])),
        "stable": lam_max <= 1.0 + STABILITY_TOL,
        "margin": 1.0 - lam_max,
        "n_wave_vectors": int(k.shape[0]),
        "tau": model.tau,
        "ma": float(ma),
        "model": model.name,
    }


def predicted_box_radius(model: CollisionModel, ma: float, *, n_box: int = 32) -> dict[str, Any]:
    """von Neumann radius restricted to the wave vectors of a periodic ``n_box`` box."""
    k = box_wave_vectors(n_box)
    eigs = amplification_eigenvalues(np.stack([model.map_matrix(ma)]), k)[0]
    absmax = np.abs(eigs).max(axis=1)
    j = int(absmax.argmax())
    lam = float(absmax[j])
    return {
        "lambda_max": lam,
        "k_argmax": (float(k[j, 0]), float(k[j, 1])),
        "n_box": int(n_box),
        "stable": lam <= 1.0 + STABILITY_TOL,
    }


# ---------------------------------------------------------------------------
# Measured side: full nonlinear periodic-box integration (independent path)
# ---------------------------------------------------------------------------


def _feq(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    cu = 3.0 * (CX[:, None, None] * ux + CY[:, None, None] * uy)
    u2 = ux * ux + uy * uy
    return W[:, None, None] * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * u2)


def _project_out_conserved(d: np.ndarray) -> np.ndarray:
    """Remove mass and momentum content of a perturbation ``d`` (shape ``(9, n, n)``)."""
    d = d - d.mean(axis=(1, 2), keepdims=True)
    cc = np.stack([CX, CY], axis=1)  # (9, 2)
    j = np.einsum("ia,iyx->ayx", cc, d)
    return d - np.einsum("ia,ayx->iyx", cc * W[:, None] * 3.0, j)


def _seed_single_mode(n: int, kx: float, ky: float, amplitude: float, seed: int) -> np.ndarray:
    """Perturbation carrying one and only one wave vector ``(kx, ky)``."""
    rng = np.random.default_rng(int(seed))
    x = np.arange(n)[:, None]
    y = np.arange(n)[None, :]
    phase = np.exp(1j * (kx * x + ky * y))
    d = np.zeros((9, n, n), dtype=float)
    for i in range(9):
        a = rng.standard_normal() + 1j * rng.standard_normal()
        d[i] = amplitude * math.sqrt(2.0) * (a * phase).real
    return _project_out_conserved(d)


def _fit_growth(amps: np.ndarray, *, tmin: int, lo: float, hi: float) -> dict[str, Any]:
    t = np.arange(amps.size)
    good = np.isfinite(amps) & (amps > lo) & (amps < hi) & (t >= int(tmin))
    if int(good.sum()) < 12:
        return {"lambda": float("nan"), "growth_rate": float("nan"), "r2": float("nan"), "n_fit": int(good.sum())}
    tt, yy = t[good].astype(float), np.log(amps[good])
    slope, intercept = np.polyfit(tt, yy, 1)
    resid = yy - (slope * tt + intercept)
    ss_tot = float(((yy - yy.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return {"lambda": float(math.exp(slope)), "growth_rate": float(slope), "r2": float(r2), "n_fit": int(good.sum())}


def run_mode_box(
    model: CollisionModel,
    ma: float,
    *,
    k: Sequence[float] | None = None,
    n: int | None = None,
    n_steps: int | None = None,
    amplitude: float | None = None,
    seed: int | None = None,
    rho0: float = 1.0,
    fit_hi: float = 1e-3,
    fit_tmin: int = 20,
) -> dict[str, Any]:
    """Integrate the full nonlinear LBM seeded with one Fourier mode and fit growth.

    The collision uses the *same* moment basis and rate vector as the
    prediction, but the time integration (streaming, nonlinear equilibrium,
    moment evaluation) is an independent code path from the eigen-analysis.
    """
    pre = PREDECLARED
    n = int(pre["measure_box"] if n is None else n)
    n_steps = int(pre["measure_steps"] if n_steps is None else n_steps)
    amplitude = float(pre["measure_amplitude"] if amplitude is None else amplitude)
    seed = int(pre["measure_seed"] if seed is None else seed)
    if k is None:
        k = predicted_box_radius(model, ma, n_box=n)["k_argmax"]
    kx, ky = float(k[0]), float(k[1])
    for comp in (kx, ky):
        m = comp * n / (2.0 * math.pi)
        if abs(m - round(m)) > 1e-6:
            raise ValueError(
                f"wave vector {(kx, ky)!r} is not representable on a periodic {n}x{n} box "
                f"({n}*k/(2*pi) = {m!r} is not an integer); the seeded field would not be "
                "periodic and the measured growth would be meaningless"
            )

    u0 = float(ma) * CS
    rho = np.full((n, n), float(rho0))
    ux = np.full((n, n), u0)
    uy = np.zeros((n, n))
    f = _feq(rho, ux, uy) + _seed_single_mode(n, kx, ky, amplitude, seed)

    rates = model.rates[:, None, None]
    basis, basis_inv = model.basis, model.basis_inv
    amps = np.full(n_steps, np.nan)
    blown = False
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for t in range(n_steps):
            rho_t = f.sum(axis=0)
            ux_t = (f * CX[:, None, None]).sum(axis=0) / rho_t
            uy_t = (f * CY[:, None, None]).sum(axis=0) / rho_t
            amps[t] = float(np.sqrt((((ux_t - u0) ** 2 + uy_t**2)).mean()))
            fe = _feq(rho_t, ux_t, uy_t)
            m = np.einsum("ij,jyx->iyx", basis, f - fe)
            f = f - np.einsum("ij,jyx->iyx", basis_inv, rates * m)
            g = np.empty_like(f)
            for i in range(9):
                g[i] = np.roll(f[i], (int(CX[i]), int(CY[i])), axis=(0, 1))
            f = g
            if not np.isfinite(amps[t]) or not np.isfinite(f).all():
                blown = True
                break

    # Adaptive start of the fit window: a fast-growing mode leaves the linear
    # window within a few steps, so a fixed ``fit_tmin`` would throw away every
    # usable sample.  Prefer the largest start that still leaves >= 30 samples.
    fit = _fit_growth(amps[: t + 1], tmin=fit_tmin, lo=1e-14, hi=fit_hi)
    if fit["n_fit"] < 30:
        for tmin_try in (15, 10, 6, 3, 0):
            if tmin_try >= fit_tmin:
                continue
            cand = _fit_growth(amps[: t + 1], tmin=tmin_try, lo=1e-14, hi=fit_hi)
            if cand["n_fit"] > fit["n_fit"]:
                fit, fit_tmin = cand, tmin_try
            if fit["n_fit"] >= 30:
                break
    first = float(amps[0])
    last = float(amps[t]) if np.isfinite(amps[t]) else float("nan")
    return {
        "lambda_meas": fit["lambda"],
        "growth_rate_meas": fit["growth_rate"],
        "fit_r2": fit["r2"],
        "n_fit": fit["n_fit"],
        "fit_tmin_used": int(fit_tmin),
        "k": (kx, ky),
        "n": n,
        "n_steps": n_steps,
        "amplitude_initial": first,
        "amplitude_final": last,
        "total_growth": (last / first) if (first > 0 and last == last) else float("nan"),
        "blown_up": bool(blown),
        "model": model.name,
        "ma": float(ma),
        "tau": model.tau,
    }


def run_broadband_box(
    model: CollisionModel,
    ma: float,
    *,
    n: int | None = None,
    n_steps: int | None = None,
    amplitude: float | None = None,
    seed: int | None = None,
    window: int | None = None,
    ratio_threshold: float | None = None,
    rho0: float = 1.0,
) -> dict[str, Any]:
    """Full nonlinear periodic box, broadband seed, literal divergence detector.

    The detector is the one declared for the campaign: the velocity residual
    ``||u - u0||_2`` must not grow by more than ``ratio_threshold`` (1e3) inside
    ``window`` (10) consecutive steps anywhere in the run.  Because a per-step
    factor ``lambda`` compounds to ``lambda**window``, this detector fires only
    for ``lambda > 1e3**(1/10) = 1.995``: it is a *deep-instability* detector,
    not a marginal one.  It is reported next to the fitted growth rate; the
    marginal boundary is defined on the fitted rate
    (see :func:`measured_critical_tau`).
    """
    pre = PREDECLARED
    n = int(pre["broadband_box"] if n is None else n)
    n_steps = int(pre["broadband_steps"] if n_steps is None else n_steps)
    amplitude = float(pre["broadband_amplitude"] if amplitude is None else amplitude)
    seed = int(pre["broadband_seed"] if seed is None else seed)
    window = int(pre["broadband_window"] if window is None else window)
    threshold = float(pre["broadband_ratio_threshold"] if ratio_threshold is None else ratio_threshold)

    u0 = float(ma) * CS
    rho = np.full((n, n), float(rho0))
    ux = np.full((n, n), u0)
    uy = np.zeros((n, n))
    rng = np.random.default_rng(seed)
    noise = _project_out_conserved(rng.standard_normal((9, n, n)))
    f = _feq(rho, ux, uy) + amplitude * noise

    rates = model.rates[:, None, None]
    basis, basis_inv = model.basis, model.basis_inv
    amps = np.full(n_steps, np.nan)
    blown = False
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for t in range(n_steps):
            rho_t = f.sum(axis=0)
            ux_t = (f * CX[:, None, None]).sum(axis=0) / rho_t
            uy_t = (f * CY[:, None, None]).sum(axis=0) / rho_t
            amps[t] = float(np.sqrt((((ux_t - u0) ** 2 + uy_t**2)).mean()))
            fe = _feq(rho_t, ux_t, uy_t)
            m = np.einsum("ij,jyx->iyx", basis, f - fe)
            f = f - np.einsum("ij,jyx->iyx", basis_inv, rates * m)
            g = np.empty_like(f)
            for i in range(9):
                g[i] = np.roll(f[i], (int(CX[i]), int(CY[i])), axis=(0, 1))
            f = g
            if not np.isfinite(amps[t]) or not np.isfinite(f).all():
                blown = True
                break

    ratios = np.full(max(n_steps - window, 1), np.nan)
    for t in range(0, max(n_steps - window, 1)):
        if t + window < amps.size and np.isfinite(amps[t]) and np.isfinite(amps[t + window]) and amps[t] > 0:
            ratios[t] = amps[t + window] / amps[t]
    max_ratio = float(np.nanmax(ratios)) if np.isfinite(ratios).any() else float("nan")
    fired = bool((np.isfinite(max_ratio) and max_ratio > threshold) or blown)
    return {
        "divergent": fired,
        "max_window_ratio": max_ratio,
        "window": window,
        "ratio_threshold": threshold,
        "blown_up": bool(blown),
        "amplitude_initial": float(amps[0]),
        "amplitude_final": float(amps[t]) if np.isfinite(amps[t]) else float("inf"),
        "max_amplitude": float(np.nanmax(amps)) if np.isfinite(amps).any() else float("inf"),
        "n": n,
        "n_steps": n_steps,
        "model": model.name,
        "ma": float(ma),
        "tau": model.tau,
    }


# ---------------------------------------------------------------------------
# Boundary localisation: prediction vs measurement (the exchange law)
# ---------------------------------------------------------------------------


def predicted_critical_tau(
    factory: Callable[[float], CollisionModel],
    ma: float,
    *,
    lo: float | None = None,
    hi: float | None = None,
    n_theta: int | None = None,
    iters: int | None = None,
) -> dict[str, Any]:
    """Bisect ``tau`` for ``lambda_max(tau, Ma) = 1`` (von Neumann prediction)."""
    pre = PREDECLARED
    lo = float(pre["tau_window"][0] if lo is None else lo)
    hi = float(pre["tau_window"][1] if hi is None else hi)
    n_theta = int(pre["k_grid_boundary"] if n_theta is None else n_theta)
    iters = int(pre["bisection_iters"] if iters is None else iters)
    k = quarter_wave_vectors(n_theta)
    lam_lo = spectral_radius(factory(lo), ma, k=k)["lambda_max"]
    lam_hi = spectral_radius(factory(hi), ma, k=k)["lambda_max"]
    if lam_lo <= 1.0 + STABILITY_TOL:
        return {"tau_crit": None, "lam_at_lo": lam_lo, "lam_at_hi": lam_hi, "interior": False}
    if lam_hi > 1.0 + STABILITY_TOL:
        return {"tau_crit": None, "lam_at_lo": lam_lo, "lam_at_hi": lam_hi, "interior": False}
    a, b = lo, hi
    for _ in range(iters):
        mid = 0.5 * (a + b)
        if spectral_radius(factory(mid), ma, k=k)["lambda_max"] > 1.0 + STABILITY_TOL:
            a = mid
        else:
            b = mid
    return {"tau_crit": 0.5 * (a + b), "lam_at_lo": lam_lo, "lam_at_hi": lam_hi, "interior": True}


def _as_stable(lam: float, tol: float) -> bool:
    """NaN means "decayed below the fit floor" -> stable."""
    return (not math.isfinite(lam)) or lam <= 1.0 + tol


def measured_critical_tau(
    factory: Callable[[float], CollisionModel],
    ma: float,
    *,
    lo: float | None = None,
    hi: float | None = None,
    iters: int | None = None,
    n_box: int | None = None,
    n_steps: int | None = None,
    tol: float | None = None,
) -> dict[str, Any]:
    """Bisect ``tau`` for the measured modal growth ``lambda_meas(tau, Ma) = 1``.

    The seeded wave vector is always a *box-representable* mode of the ``n_box``
    grid (``k = 2*pi*m/n``), selected as the predicted arg-max over that same
    box grid, so the measured and the box-grid predicted radii are compared at
    identical ``k``.  The growth itself comes from the independent nonlinear
    time integration.
    """
    pre = PREDECLARED
    lo = float(pre["tau_window"][0] if lo is None else lo)
    hi = float(pre["tau_window"][1] if hi is None else hi)
    iters = int(pre["bisection_iters"] if iters is None else iters)
    n_box = int(pre["measure_box"] if n_box is None else n_box)
    n_steps = int(pre["measure_steps"] if n_steps is None else n_steps)
    tol = float(pre["measured_tol"] if tol is None else tol)
    k_box = box_wave_vectors(n_box)

    def growth(tau: float) -> dict[str, Any]:
        model = factory(tau)
        k = predicted_box_radius(model, ma, n_box=n_box)["k_argmax"]
        out = run_mode_box(model, ma, k=k, n=n_box, n_steps=n_steps)
        out["k_pred_box"] = k
        out["lambda_pred_box"] = predicted_box_radius(model, ma, n_box=n_box)["lambda_max"]
        return out

    probe_lo, probe_hi = growth(lo), growth(hi)
    if _as_stable(probe_lo["lambda_meas"], tol):
        return {
            "tau_crit": None,
            "lambda_lo": probe_lo["lambda_meas"],
            "lambda_hi": probe_hi["lambda_meas"],
            "interior": False,
            "n_bisection": 0,
            "k_box": probe_lo.get("k_pred_box"),
        }
    if not _as_stable(probe_hi["lambda_meas"], tol):
        return {
            "tau_crit": None,
            "lambda_lo": probe_lo["lambda_meas"],
            "lambda_hi": probe_hi["lambda_meas"],
            "interior": False,
            "n_bisection": 0,
            "k_box": probe_hi.get("k_pred_box"),
        }
    a, b = lo, hi
    last = probe_lo
    for _ in range(iters):
        mid = 0.5 * (a + b)
        last = growth(mid)
        if not _as_stable(last["lambda_meas"], tol):
            a = mid
        else:
            b = mid
    return {
        "tau_crit": 0.5 * (a + b),
        "lambda_lo": probe_lo["lambda_meas"],
        "lambda_hi": probe_hi["lambda_meas"],
        "lambda_at_bracket": last["lambda_meas"],
        "lambda_pred_box_at_bracket": last.get("lambda_pred_box"),
        "fit_r2_at_bracket": last["fit_r2"],
        "k_at_bracket": last.get("k_pred_box"),
        "interior": True,
        "n_bisection": iters,
    }


def box_predicted_critical_tau(
    factory: Callable[[float], CollisionModel],
    ma: float,
    *,
    lo: float | None = None,
    hi: float | None = None,
    iters: int | None = None,
    n_box: int | None = None,
) -> dict[str, Any]:
    """Bisect ``tau`` for the box-grid predicted radius (same ``k`` set as the measurement)."""
    pre = PREDECLARED
    lo = float(pre["tau_window"][0] if lo is None else lo)
    hi = float(pre["tau_window"][1] if hi is None else hi)
    iters = int(pre["bisection_iters"] if iters is None else iters)
    n_box = int(pre["measure_box"] if n_box is None else n_box)
    k_box = box_wave_vectors(n_box)

    def radius(tau: float) -> float:
        return float(lambda_max_batch([factory(tau)], ma, k_box)[0])

    lam_lo, lam_hi = radius(lo), radius(hi)
    if lam_lo <= 1.0 + STABILITY_TOL or lam_hi > 1.0 + STABILITY_TOL:
        return {"tau_crit": None, "lam_at_lo": lam_lo, "lam_at_hi": lam_hi, "interior": False}
    a, b = lo, hi
    for _ in range(iters):
        mid = 0.5 * (a + b)
        if radius(mid) > 1.0 + STABILITY_TOL:
            a = mid
        else:
            b = mid
    return {"tau_crit": 0.5 * (a + b), "lam_at_lo": lam_lo, "lam_at_hi": lam_hi, "interior": True}


# ---------------------------------------------------------------------------
# Tier definitions (the three design tiers of the campaign)
# ---------------------------------------------------------------------------


def tier_factory(tier: str, variant: str = "canonical") -> Callable[[float], CollisionModel]:
    """Return ``tau -> CollisionModel`` for one of the three declared tiers.

    * ``"bgk"``    -- single relaxation time, one knob (``1/tau``).
    * ``"mrt_a"``  -- ``s7 = s8 = 1/tau``; high-order moments (``s1``, ``s2``)
      free, ``s4 = s_qx = s_qy = 1.7`` (Lallemand & Luo default).
    * ``"mrt_b"``  -- as ``mrt_a`` *and* ``s4 = s_qx`` free.

    ``variant="canonical"`` freezes every free rate at its textbook value;
    ``variant="oracle"`` selects the free rates from the predeclared candidate
    sets, per ``(tau, Ma)``, to maximise the von Neumann margin.  The oracle
    variant is an *upper envelope* over a declared discrete design space, not a
    universal optimum, and is labelled as such in every artefact.
    """
    tier = tier.lower()
    cand = PREDECLARED["rate_candidates"]
    base = dict(PREDECLARED["mrt_a_rates"])
    if tier == "bgk":
        return lambda tau: bgk_model(tau, name="BGK")
    if tier not in ("mrt_a", "mrt_b"):
        raise ValueError(f"unknown tier {tier!r}")
    if variant == "canonical":
        if tier == "mrt_a":
            return lambda tau: mrt_model(tau, name="MRT-A", notes={"tier": tier, "variant": "canonical", **base})
        return lambda tau: mrt_model(tau, name="MRT-B", notes={"tier": tier, "variant": "canonical", **base})
    if variant != "oracle":
        raise ValueError(f"unknown variant {variant!r}")
    # oracle variants need the Ma-dependent rate choice: handled by tier_model_at()
    raise ValueError("use tier_model_at() for the oracle variant")


def tier_model_at(
    tier: str,
    tau: float,
    ma: float,
    *,
    variant: str = "canonical",
    k: np.ndarray | None = None,
) -> CollisionModel:
    """Build the model of one tier at one ``(tau, Ma)`` (oracle selection happens here)."""
    tier = tier.lower()
    if tier == "bgk":
        return bgk_model(tau, name="BGK")
    if variant == "canonical":
        return tier_factory(tier, "canonical")(tau)
    if variant != "oracle":
        raise ValueError(f"unknown variant {variant!r}")
    cand = PREDECLARED["rate_candidates"]
    if k is None:
        k = quarter_wave_vectors(PREDECLARED["k_grid_oracle"])
    base = dict(PREDECLARED["mrt_a_rates"])
    best = None
    if tier == "mrt_a":
        space = [(e, ep) for e in cand["s_e"] for ep in cand["s_eps"]]
        models = [
            mrt_model(tau, s_e=e, s_eps=ep, s_qx=base["s_qx"], s_qy=base["s_qy"],
                      name="MRT-A", notes={"tier": tier, "variant": "oracle", "s_e": e, "s_eps": ep})
            for e, ep in space
        ]
    else:
        space = [(e, ep, q) for e in cand["s_e"] for ep in cand["s_eps"] for q in cand["s_qx"]]
        models = [
            mrt_model(tau, s_e=e, s_eps=ep, s_qx=q, s_qy=base["s_qy"],
                      name="MRT-B", notes={"tier": tier, "variant": "oracle", "s_e": e, "s_eps": ep, "s_qx": q})
            for e, ep, q in space
        ]
    lam = lambda_max_batch(models, ma, k)
    idx = int(np.argmin(lam))
    best = models[idx]
    best.notes["lambda_max_at_selection"] = float(lam[idx])
    best.notes["n_candidates"] = len(models)
    return best


def foil_model(family: str, tau: float, name: str | None = None) -> CollisionModel:
    """A legal but badly tuned MRT setting, used to falsify "MRT is more stable".

    ``family`` selects a declared foil from
    :data:`PREDECLARED` ``["foil_rates"]``; every rate stays inside ``(0, 2)`` and
    the moment basis stays orthogonal, so the setting is a legitimate MRT model
    that a user could really run.
    """
    rates = PREDECLARED["foil_rates"][family]
    return mrt_model(
        tau,
        s_e=rates["s_e"],
        s_eps=rates["s_eps"],
        s_qx=rates["s_qx"],
        s_qy=rates["s_qy"],
        name=name or f"MRT-foil:{family}",
        notes={"foil": family, "rule": "legal but badly tuned (all rates in (0, 2))"},
    )


def poor_mrt_model(tau: float, name: str = "MRT-poor") -> CollisionModel:
    """Backwards-compatible alias for the adversarial ``mismatched`` foil."""
    return foil_model("mismatched", tau, name=name)


def negative_control(family: str, tau: float) -> CollisionModel:
    """Declared defective operators that a stability diagnostic must reject."""
    pre = PREDECLARED["negative_controls"]
    if family == "over_relax_shear":
        spec = pre[family]
        return mrt_model(tau, s_pxy=spec["rate"],
                         name=f"NC:over_relax({spec['slot']}={spec['rate']})",
                         notes={"defect": "relaxation rate > 2 (over-relaxation)", **spec})
    if family == "over_relax_energy":
        spec = pre[family]
        return mrt_model(tau, s_e=spec["rate"], name=f"NC:over_relax({spec['slot']}={spec['rate']})",
                         notes={"defect": "relaxation rate > 2 (over-relaxation)", **spec})
    if family == "mixed_basis":
        spec = pre[family]
        return mrt_model(
            tau,
            s_qx=spec["s_qx"],
            basis=mixed_basis(spec["mix_from"], spec["mix_into"], spec["mix_beta"]),
            name="NC:mixed_basis",
            notes={"defect": "non-orthogonal moment basis (jx leaks into qx)", **spec},
        )
    if family == "mixed_basis_over_relax":
        spec = pre[family]
        return mrt_model(
            tau,
            s_qx=spec["s_qx"],
            basis=mixed_basis(spec["mix_from"], spec["mix_into"], spec["mix_beta"]),
            name="NC:mixed_basis+over_relax",
            notes={"defect": "non-orthogonal basis AND rate > 2", **spec},
        )
    raise ValueError(f"unknown negative control {family!r}")


def design_rates(tier: str, tau: float, **_: Any) -> dict[str, Any]:
    """Per-``tau`` rate design used by the ``oracle`` plane scans.

    Design rule (declared before the runs): minimise ``lambda_max`` at the
    highest Mach number of the window (``Ma = design_ma``) on the coarse design
    wave-vector grid ``k_grid_design``.  ``mrt_a`` searches the (s_e, s_eps)
    product; ``mrt_b`` then adds ``s_qx`` greedily on top of the best
    (s_e, s_eps).  Storing the rule per ``tau`` (not per ``(tau, Ma)``) keeps the
    tier an implementable design instead of an unrealisable point-wise optimum.
    """
    tier = tier.lower()
    pre = PREDECLARED
    cand = pre["rate_candidates"]
    base = dict(pre["mrt_a_rates"])
    ma = float(pre["design_ma"])
    k = quarter_wave_vectors(int(pre["k_grid_design"]))
    if tier == "bgk":
        return {"tier": "bgk", "s_e": None, "s_eps": None, "s_qx": None}

    def score(**kw: float) -> float:
        model = mrt_model(tau, s_qy=base["s_qy"], **kw)
        return float(np.abs(amplification_eigenvalues(np.stack([model.map_matrix(ma)]), k)).max())

    best = None
    for e in cand["s_e"]:
        for ep in cand["s_eps"]:
            val = score(s_e=e, s_eps=ep, s_qx=base["s_qx"])
            if best is None or val < best[0]:
                best = (val, {"s_e": float(e), "s_eps": float(ep), "s_qx": float(base["s_qx"])})
    design = dict(best[1])
    if tier == "mrt_b":
        best_q = None
        for q in cand["s_qx"]:
            val = score(s_e=design["s_e"], s_eps=design["s_eps"], s_qx=q)
            if best_q is None or val < best_q[0]:
                best_q = (val, float(q))
        design["s_qx"] = float(best_q[1])
        design["greedy_note"] = "s_qx selected after (s_e, s_eps)"
    design["tier"] = tier
    design["lambda_max_at_design"] = float(best[0] if tier == "mrt_a" else best_q[0])
    design["design_ma"] = ma
    design["k_grid_design"] = int(pre["k_grid_design"])
    return design


def tier_model_designed(tier: str, tau: float, design: dict[str, Any]) -> CollisionModel:
    """Build the model of one tier with an explicit (per-``tau``) design."""
    tier = tier.lower()
    if tier == "bgk":
        return bgk_model(tau, name="BGK")
    return mrt_model(
        tau,
        s_e=float(design["s_e"]),
        s_eps=float(design["s_eps"]),
        s_qx=float(design["s_qx"]),
        s_qy=float(PREDECLARED["mrt_a_rates"]["s_qy"]),
        name="MRT-A" if tier == "mrt_a" else "MRT-B",
        notes={"tier": tier, "variant": "oracle", **{k: float(v) for k, v in design.items()
                                                     if isinstance(v, (int, float))}},
    )


# ---------------------------------------------------------------------------
# Canonical JSON / digest for bitwise-reproducible envelopes
# ---------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, floats rounded to 12 digits."""
    return json.dumps(_round_floats(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _round_floats(obj: Any) -> Any:
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite float in envelope: {obj!r}")
        return round(obj, 12)
    if isinstance(obj, dict):
        return {str(k): _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v) for v in obj]
    if isinstance(obj, np.floating):
        return _round_floats(float(obj))
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _round_floats(obj.tolist())
    return obj


def envelope_digest(envelope: dict[str, Any]) -> str:
    """SHA-256 over the canonical JSON of an envelope (excluding ``digest`` itself)."""
    payload = {k: v for k, v in envelope.items() if k != "digest"}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


__all__ = [
    "CS",
    "MOMENT_SLOTS",
    "PREDECLARED",
    "STABILITY_TOL",
    "MEASURED_TOL",
    "CollisionModel",
    "amplification_eigenvalues",
    "basis_orthogonality_defect",
    "bgk_model",
    "design_rates",
    "box_predicted_critical_tau",
    "box_wave_vectors",
    "canonical_json",
    "envelope_digest",
    "foil_model",
    "lambda_max_batch",
    "measured_critical_tau",
    "mixed_basis",
    "mrt_model",
    "negative_control",
    "orthogonal_basis",
    "poor_mrt_model",
    "predicted_box_radius",
    "predicted_critical_tau",
    "quarter_wave_vectors",
    "rates_vector",
    "run_broadband_box",
    "run_mode_box",
    "spectral_radius",
    "tier_factory",
    "tier_model_at",
    "tier_model_designed",
]
