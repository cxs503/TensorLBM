"""Pulsatile plane-channel (Womersley) verification case: LBM vs exact solution.

Closes the LBM side of anchor ``CASE-LBM-WOMERSLEY-PULSATILE`` (ref: Womersley
1955, *J. Physiol.* 127:553): a plane channel driven by a harmonic body force,
sampled over whole periods and compared with the exact unsteady Stokes
solution.  The **pipe** version of that solution already lives in
:mod:`tensorlbm.turbulence.womersley_exact`; this module adds

1. the closed-form **plane-channel** phasor (the pipe solution is a Bessel
   function, the channel one a cosh — a channel LBM run cannot be gated against
   ``J0/J1``),
2. an in-module D2Q9 driver (Guo forcing + pre-streaming half-way bounce-back),
3. the acceptance gate of the case anchor (``α = 5``: centreline amplitude
   ratio and phase lag vs the analytic solution, whole-period average).

Exact solution
--------------
Fully developed channel flow, walls at ``y = ±H/2``, forcing
``-∂p/∂x = ρ a e^{iωt}``::

    ν Û'' - iω Û = -a ,      Û(±H/2) = 0

whose solution is, with ``κ = (H/2) sqrt(i ω / ν)`` and ``ŷ = 2y/H``,

.. math::

    Û(ŷ) = \\frac{a}{iω}\\left(1 - \\frac{\\cosh(κ ŷ)}{\\cosh κ}\\right).

Normalising by the **cross-sectionally averaged** velocity (what an inlet
condition or a flow meter supplies) gives the unit-bulk shape

.. math::

    \\hat W(ŷ) = \\frac{1 - \\cosh(κ ŷ)/\\cosh κ}{1 - \\tanh(κ)/κ},
    \\qquad \\langle\\hat W\\rangle \\equiv 1,

and the centreline value ``Ŵ(0) = (1 - sech κ)/(1 - tanh κ/κ)`` is the analytic
amplitude ratio / phase (lag relative to the *flow rate*) that this case gates
on.  Note ``α = 2κ`` with ``α = H sqrt(ω/ν)`` in the classical Womersley
notation.

Independent references used by the tests
----------------------------------------
* ``κ → 0``: ``Ŵ → (3/2)(1 - ŷ²)`` — quasi-steady Poiseuille, error ``O(κ²)``.
* ``κ → ∞``: ``Ŵ(0) → 1`` (plug in phase with the flow rate) while the wall
  shear leads it by 45° — the classical Womersley asymptote (the same one the
  pipe module checks).
* ``⟨Ŵ⟩ = 1`` at every ``κ`` (flux identity, quadrature check).
* finite-difference solution of the same ODE (no Bessel/cosh algebra reuse).

Solver conventions (verified, not assumed)
------------------------------------------
* D2Q9 / BGK, Guo (2002) forcing with the shifted velocity
  ``u = (Σ c_i f_i + a/2)/ρ`` and zero-sum source
  ``S_i = (1 - 1/(2τ)) w_i [3 (c_i - u)·a + 9 (c_i·u)(c_i·a)]``.  The forcing
  strength is **measured** against steady Poiseuille
  (:func:`run_steady_poiseuille`), never assumed: an ``a/2`` bookkeeping error
  shows up there as a factor.
* Walls: pre-streaming half-way bounce-back with ghost rows, so the no-slip
  planes sit exactly at ``y = ±H/2`` with ``H = ny`` **fluid rows** (same steady
  run validates it: ``u_c = a H²/(8ν)`` to <1 %).
* Periodic in ``x``; the field is ``x``-uniform, so ``nx = 2`` suffices.

Known limitations (recorded, not hidden)
----------------------------------------
* ``1 - cosh(κŷ)/cosh κ`` cancels near the wall, exactly as in the pipe module;
  analytic profiles are trusted for ``|ŷ| ≤ 0.98`` and the L2 gate excludes the
  wall-adjacent rows (their LBM error is O(h²) regardless).
* BGK only: at ``α = 5`` the Stokes layer ``δ = sqrt(2ν/ω)`` is ≈ 9 nodes at
  ``H = 32``.  Residual error is dominated by compressibility (``O(Ma²)``) and
  the bounce-back wall position; both are reported in
  ``docs/CASE-LBM-WOMERSLEY-PULSATILE.md``.
* ``α`` is recomputed from the actual lattice parameters (``H``, ``ν``, ``ω``)
  and reported as ``alpha_effective`` — a config with a mismatched ``α`` is
  silently a different case, so the gate checks the effective value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "CASE_KEY",
    "GATE",
    "ChannelPulsatileConfig",
    "channel_centreline_phasor",
    "channel_phasor",
    "quasi_steady_shape",
    "run_pulsatile_channel",
    "run_steady_poiseuille",
    "unit_bulk_kappa",
    "evaluate_gate",
]

CASE_KEY = "LBM-WOMERSLEY-PULSATILE"

#: Case-anchor acceptance: < 5 % centreline amplitude error and < 5° centreline
#: phase error at alpha = 5, whole-period phase average.
GATE = {"alpha": 5.0, "amp_rel_pct_max": 5.0, "phase_deg_max": 5.0}

# D2Q9 (same order as tensorlbm.d2q9.C / W)
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
WQ = np.array(
    [
        4.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
    ]
)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)


# ---------------------------------------------------------------------------
# exact solution
# ---------------------------------------------------------------------------
def unit_bulk_kappa(alpha: float) -> complex:
    """``κ = (α/2) e^{iπ/4}`` for the real Womersley number ``α = H sqrt(ω/ν)``.

    ``α`` here is the classical real Womersley number; the complex eigenvalue is
    ``κ = (H/2) sqrt(iω/ν) = (α/2) e^{iπ/4}``.  Forgetting the ``e^{iπ/4}`` makes
    every phase lag vanish while leaving amplitudes almost unchanged -- the same
    conjugation trap the pipe module recorded, in a milder form.
    """
    return 0.5 * complex(alpha) * np.sqrt(1j)


def channel_phasor(y: np.ndarray, H: float, omega: float, nu: float) -> np.ndarray:
    """Unit-bulk complex amplitude ``Ŵ(ŷ)`` at wall-normal positions ``y``.

    ``y`` is measured from the channel centre (walls at ``±H/2``).  The physical
    velocity for unit bulk flow rate at phase ``φ`` is
    ``Re[Ŵ e^{iφ}]``.
    """
    y = np.asarray(y, dtype=float)
    yhat = 2.0 * y / H
    kappa = 0.5 * H * np.sqrt(1j * omega / nu)
    a = kappa * yhat
    b = kappa
    with np.errstate(over="ignore", invalid="ignore"):
        # cosh(a)/cosh(b) = exp(a-b)(1+exp(-2a))/(1+exp(-2b)) -- overflow safe
        ratio = np.exp(a - b) * (1.0 + np.exp(-2.0 * a)) / (1.0 + np.exp(-2.0 * b))
    return (1.0 - ratio) / (1.0 - np.tanh(kappa) / kappa)


def channel_centreline_phasor(alpha: float | complex) -> complex:
    """``Ŵ(0)``: analytic centreline amplitude/phase relative to the flow rate."""
    kappa = unit_bulk_kappa(alpha)
    return (1.0 - 1.0 / np.cosh(kappa)) / (1.0 - np.tanh(kappa) / kappa)


def quasi_steady_shape(y: np.ndarray, H: float) -> np.ndarray:
    """``(3/2)(1 - ŷ²)``: the ``κ → 0`` limit of :func:`channel_phasor`."""
    yhat = 2.0 * np.asarray(y, dtype=float) / H
    return 1.5 * (1.0 - yhat**2)


def bulk_phasor_for_forcing(a0: float, alpha: float, omega: float) -> complex:
    """Complex bulk velocity produced by forcing ``a0 cos(ωt)`` (exact solution)."""
    kappa = unit_bulk_kappa(alpha)
    return a0 / (1j * omega) * (1.0 - np.tanh(kappa) / kappa)


def _node_y(ny: int) -> tuple[np.ndarray, float]:
    """Wall-normal coordinates of the ``ny`` fluid rows; effective gap ``H = ny``.

    Ghost-row half-way bounce-back places the walls half a cell outside the
    first/last fluid row: ``y_j = j + 0.5 - H/2``.
    """
    H = float(ny)
    return np.arange(ny, dtype=float) + 0.5 - H / 2.0, H


# ---------------------------------------------------------------------------
# D2Q9 driver
# ---------------------------------------------------------------------------
@dataclass
class ChannelPulsatileConfig:
    """Lattice parameters for the pulsatile/steady channel case."""

    ny: int = 32
    nx: int = 1
    nu: float = 0.05
    alpha: float = 5.0
    u_bulk: float = 0.03
    skip_periods: float = 3.0
    measure_periods: float = 3.0
    init_from_exact: bool = True
    a0: float | None = None
    tag: str = ""

    def __post_init__(self) -> None:
        if self.ny < 4:
            raise ValueError("ny must be >= 4 fluid rows")
        if not 0.0 < self.nu < 1.0:
            raise ValueError("nu out of range")

    @property
    def H(self) -> float:
        return float(self.ny)

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5

    @property
    def omega_exact(self) -> float:
        """Angular frequency implied by the requested ``α`` (per lattice step)."""
        return self.nu * (self.alpha / self.H) ** 2

    @property
    def period_steps(self) -> int:
        """Integer-step period (avoids frequency/period mismatch in the fit)."""
        return max(8, int(round(2.0 * math.pi / self.omega_exact)))

    @property
    def omega(self) -> float:
        return 2.0 * math.pi / self.period_steps

    @property
    def alpha_effective(self) -> float:
        return self.H * math.sqrt(self.omega / self.nu)

    @property
    def forcing_amplitude(self) -> float:
        """``a0`` realising the requested bulk amplitude (linear response)."""
        if self.a0 is not None:
            return float(self.a0)
        kappa = unit_bulk_kappa(self.alpha_effective)
        return self.u_bulk * self.omega / abs(1.0 - np.tanh(kappa) / kappa)


class _ChannelD2Q9:
    """Minimal D2Q9 channel solver: Guo forcing, half-way bounce-back walls."""

    def __init__(self, cfg: ChannelPulsatileConfig) -> None:
        self.cfg = cfg
        self.tau = cfg.tau
        self.f = np.zeros((9, cfg.ny, cfg.nx), dtype=float)
        self.y, self.H = _node_y(cfg.ny)
        self.yhat = 2.0 * self.y / self.H

    @staticmethod
    def _equilibrium(rho, ux, uy):
        cu = CX[:, None, None] * ux + CY[:, None, None] * uy
        usq = ux * ux + uy * uy
        return WQ[:, None, None] * rho * (
            1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None]
        )

    def _macroscopic(self, ax, ay):
        rho = self.f.sum(axis=0)
        mx = (self.f * CX[:, None, None]).sum(axis=0)
        my = (self.f * CY[:, None, None]).sum(axis=0)
        return rho, mx / rho + 0.5 * ax, my / rho + 0.5 * ay

    def _force_term(self, ux, uy, ax, ay):
        cu = CX[:, None, None] * ux + CY[:, None, None] * uy
        ca = CX[:, None, None] * ax + CY[:, None, None] * ay
        cma = ca - (ux * ax + uy * ay)[None]
        return (1.0 - 0.5 / self.tau) * WQ[:, None, None] * (
            3.0 * cma + 9.0 * cu * ca
        )

    def step(self, ax: float, ay: float = 0.0):
        rho, ux, uy = self._macroscopic(ax, ay)
        feq = self._equilibrium(rho, ux, uy)
        src = self._force_term(ux, uy, ax, ay)
        f_post = self.f - (self.f - feq) / self.tau + src

        ny, nx = self.cfg.ny, self.cfg.nx
        # padded: 0 = bottom ghost row, 1..ny = fluid rows, ny+1 = top ghost
        pad = np.zeros((9, ny + 2, nx), dtype=float)
        pad[:, 1:-1] = f_post
        for i in range(9):
            if CY[i] > 0:  # streams in from below the domain
                pad[i, 0] = f_post[OPP[i], 0]
            elif CY[i] < 0:  # streams in from above the domain
                pad[i, -1] = f_post[OPP[i], -1]
        new = np.empty_like(self.f)
        for i in range(9):
            rolled = np.roll(pad[i], CX[i], axis=1)
            lo = 1 - CY[i]
            new[i] = rolled[lo : lo + ny]
        self.f = new
        return rho, ux, uy

    # -- observables ------------------------------------------------------
    def init_equilibrium(self, ux0) -> None:
        ux = np.zeros((self.cfg.ny, self.cfg.nx))
        ux[:] = np.asarray(ux0, dtype=float)[:, None]
        zeros = np.zeros_like(ux)
        self.f = self._equilibrium(np.ones_like(ux), ux, zeros)

    def row_velocity(self, ax: float) -> np.ndarray:
        return self._macroscopic(ax, 0.0)[1].mean(axis=1)

    def centreline_index(self) -> tuple[int, int]:
        ny = self.cfg.ny
        if ny % 2 == 0:
            return ny // 2 - 1, ny // 2
        return ny // 2, ny // 2

    def density_drift_pct(self) -> float:
        return float((self.f.sum(axis=0).mean() - 1.0) * 100.0)


def _phasor_fit(t: np.ndarray, signal: np.ndarray, omega: float) -> dict:
    """Least-squares phasor: ``signal ≈ Re[Z e^{iωt}]``, ``Z = A - iB``."""
    design = np.column_stack([np.cos(omega * t), np.sin(omega * t)])
    coef, *_ = np.linalg.lstsq(design, signal, rcond=None)
    resid = signal - design @ coef
    scale = max(float(np.abs(signal).max()), 1e-30)
    return {
        "Z": complex(coef[0], -coef[1]),
        "rms_rel_pct": float(np.sqrt(np.mean(resid**2)) / scale * 100.0),
    }


def run_pulsatile_channel(cfg: ChannelPulsatileConfig | None = None) -> dict:
    """Run one pulsatile channel case; return gate metrics, profiles, settings."""
    cfg = cfg or ChannelPulsatileConfig()
    solver = _ChannelD2Q9(cfg)
    omega, a0 = cfg.omega, cfg.forcing_amplitude
    period = cfg.period_steps

    if cfg.init_from_exact:
        z_bulk0 = bulk_phasor_for_forcing(a0, cfg.alpha_effective, omega)
        u0 = np.real(channel_phasor(solver.y, solver.H, omega, cfg.nu) * z_bulk0)
        solver.init_equilibrium(u0)
    else:
        # rest start: equilibrium at rho = 1, u = 0.  (Zero populations would
        # give rho = 0 and NaN on the first macroscopic call.)
        solver.init_equilibrium(np.zeros(cfg.ny))

    skip = int(round(cfg.skip_periods * period))
    n_meas = int(round(cfg.measure_periods * period))
    total = skip + n_meas

    rows = np.empty((n_meas, cfg.ny))
    rho_drift = 0.0
    for step in range(total):
        ax = a0 * math.cos(omega * step)
        solver.step(ax)
        if step >= skip:
            rows[step - skip] = solver.row_velocity(ax)
        rho_drift = solver.density_drift_pct()

    j_a, j_b = solver.centreline_index()
    centre = rows[:, j_a] if j_a == j_b else rows[:, [j_a, j_b]].mean(axis=1)
    bulk = rows.mean(axis=1)
    t_meas = np.arange(skip, total, dtype=float)

    fit_c = _phasor_fit(t_meas, centre, omega)
    fit_b = _phasor_fit(t_meas, bulk, omega)
    z_c, z_bulk = fit_c["Z"], fit_b["Z"]

    # analytic unit-bulk quantities, rescaled by the *measured* bulk phasor so
    # the comparison cannot be polluted by a forcing-amplitude mismatch
    w_hat_c = channel_centreline_phasor(cfg.alpha_effective)
    w_hat_rows = channel_phasor(solver.y, solver.H, omega, cfg.nu)
    z_c_ana = w_hat_c * z_bulk

    amp_ratio = abs(z_c) / abs(z_c_ana)
    phase_diff_deg = math.degrees(np.angle(z_c / z_c_ana))

    # phase-averaged (whole-period) profile and its per-row phasor
    n_bins = 24
    bin_idx = ((omega * t_meas) / (2.0 * math.pi) * n_bins).astype(int) % n_bins
    counts = np.bincount(bin_idx, minlength=n_bins)
    ok = counts > 0
    prof_bins = np.zeros((int(ok.sum()), cfg.ny))
    for k, b in enumerate(np.flatnonzero(ok)):
        prof_bins[k] = rows[bin_idx == b].mean(axis=0)
    # Bin phase: bin b covers [(b)2pi/n, (b+1)2pi/n), so use its centre.  NB the
    # *arithmetic* mean of omega*t over the samples of a bin is NOT the bin
    # phase: samples of one bin appear at phi and phi + 2pi (one per measured
    # period), whose mean is phi + pi for an even number of periods -- that
    # silently flips the sign of every profile phasor while leaving the
    # centreline fit (separate full-series fit) correct.  Caught by comparing
    # the binned profile phasor with the full-series centreline phasor.
    bin_phase = (np.flatnonzero(ok) + 0.5) * 2.0 * math.pi / n_bins
    design = np.column_stack([np.cos(bin_phase), np.sin(bin_phase)])
    coef, *_ = np.linalg.lstsq(design, prof_bins, rcond=None)
    z_rows_num = coef[0] - 1j * coef[1]

    interior = np.abs(solver.yhat) <= 0.9
    rel = np.abs(z_rows_num[interior] / (w_hat_rows * z_bulk)[interior] - 1.0) * 100.0
    z_rows_ana = (w_hat_rows * z_bulk)[interior]
    u_ana_int = np.real(z_rows_ana[:, None] * np.exp(1j * bin_phase)[None, :]).T
    scale = max(abs(z_c_ana), 1e-30) / math.sqrt(2.0)
    wave_l2 = float(
        np.sqrt(np.mean((prof_bins[:, interior] - u_ana_int) ** 2)) / scale * 100.0
    )

    result = {
        "case": CASE_KEY,
        "tag": cfg.tag,
        "ny": cfg.ny,
        "nx": cfg.nx,
        "H": solver.H,
        "nu": cfg.nu,
        "tau": cfg.tau,
        "alpha_requested": cfg.alpha,
        "alpha_effective": cfg.alpha_effective,
        "omega": omega,
        "period_steps": period,
        "stokes_layer_nodes": math.sqrt(2.0 * cfg.nu / omega),
        "a0": a0,
        "u_bulk_target": cfg.u_bulk,
        "u_bulk_measured": abs(z_bulk),
        "u_centre_measured": abs(z_c),
        "u_centre_analytic": abs(z_c_ana),
        "Ma": float(abs(z_c) / math.sqrt(1.0 / 3.0)),
        "amp_ratio_num_over_ana": float(amp_ratio),
        "amp_rel_pct": abs(amp_ratio - 1.0) * 100.0,
        "phase_diff_deg": float(phase_diff_deg),
        "centre_amp_num_over_bulk": float(abs(z_c / z_bulk)),
        "centre_lag_num_deg": -math.degrees(np.angle(z_c / z_bulk)),
        "centre_lag_ana_deg": -math.degrees(np.angle(w_hat_c)),
        "fit_rms_rel_pct": fit_c["rms_rel_pct"],
        "profile_amp_rel_pct_max": float(rel.max()),
        "profile_amp_rel_pct_rms": float(np.sqrt(np.mean(rel**2))),
        "profile_wave_l2_pct": wave_l2,
        "density_drift_pct": rho_drift,
        "skip_steps": skip,
        "measure_steps": n_meas,
        "total_steps": total,
        "wall": "pre-streaming half-way bounce-back, walls at y=+-H/2, H=ny",
        "forcing": "Guo2002, a(t)=a0 cos(omega t), u=(sum c f + a/2)/rho",
        "profile": {
            "y": solver.y.tolist(),
            "yhat": solver.yhat.tolist(),
            "amp_ratio": (np.abs(z_rows_num) / np.abs(w_hat_rows * z_bulk)).tolist(),
            "lag_num_deg": (-np.degrees(np.angle(z_rows_num / z_bulk))).tolist(),
            "lag_ana_deg": (-np.degrees(np.angle(w_hat_rows))).tolist(),
            "phase_centers": bin_phase.tolist(),
            "centreline_wave_num": prof_bins[:, [j_a, j_b]].mean(axis=1).tolist(),
            "centreline_wave_ana": np.real(
                z_c_ana * np.exp(1j * bin_phase)
            ).tolist(),
            "complex_rel_pct": (
                np.abs(z_rows_num / (w_hat_rows * z_bulk) - 1.0) * 100.0
            ).tolist(),
            "profile_phase_averaged": prof_bins.tolist(),
            "num_phasor_re": np.real(z_rows_num).tolist(),
            "num_phasor_im": np.imag(z_rows_num).tolist(),
            "ana_phasor_re": np.real(w_hat_rows * z_bulk).tolist(),
            "ana_phasor_im": np.imag(w_hat_rows * z_bulk).tolist(),
        },
    }
    result["gate"] = evaluate_gate(result)
    return result


def evaluate_gate(result: dict, gate: dict | None = None) -> dict:
    """Apply the case anchor's acceptance (<5 % amplitude, <5° phase at α=5)."""
    gate = gate or GATE
    if abs(result["alpha_effective"] - gate["alpha"]) > 5e-3:
        return {
            "status": "not_applicable",
            "alpha_effective": result["alpha_effective"],
            "gate_alpha": gate["alpha"],
        }
    amp_ok = result["amp_rel_pct"] <= gate["amp_rel_pct_max"]
    ph_ok = abs(result["phase_diff_deg"]) <= gate["phase_deg_max"]
    return {
        "status": "pass" if (amp_ok and ph_ok) else "fail",
        "alpha": result["alpha_effective"],
        "amp_rel_pct": result["amp_rel_pct"],
        "amp_rel_pct_max": gate["amp_rel_pct_max"],
        "phase_diff_deg": result["phase_diff_deg"],
        "phase_deg_max": gate["phase_deg_max"],
        "amp_ok": bool(amp_ok),
        "phase_ok": bool(ph_ok),
    }


# ---------------------------------------------------------------------------
# steady Poiseuille calibration (forcing bookkeeping + wall position)
# ---------------------------------------------------------------------------
def run_steady_poiseuille(
    ny: int = 32, nu: float = 0.05, a0: float = 1e-4, steps: int | None = None
) -> dict:
    """Steady channel under constant body force: ``u_c = a0 H²/(8ν)``.

    Calibration reference for the Guo bookkeeping and for the half-way
    bounce-back wall position: either error appears here as a percent-level
    profile deviation, which is why the pulsatile case is only trusted after
    this passes.
    """
    cfg = ChannelPulsatileConfig(ny=ny, nu=nu, a0=a0, alpha=1.0, u_bulk=1.0)
    solver = _ChannelD2Q9(cfg)
    solver.init_equilibrium(np.zeros(ny))
    H = solver.H
    n_steps = steps or int(max(4000, 20.0 * H**2 / nu))
    for _ in range(n_steps):
        solver.step(a0)
    prof = solver.row_velocity(a0)
    ana = a0 / (2.0 * nu) * ((H / 2.0) ** 2 - solver.y**2)
    interior = np.abs(solver.yhat) <= 0.9
    u_c_num = float(prof[list(solver.centreline_index())].mean())
    u_c_ana = a0 * H**2 / (8.0 * nu)
    return {
        "case": "LBM-POISEUILLE-STEADY",
        "ny": ny,
        "nu": nu,
        "tau": cfg.tau,
        "a0": a0,
        "steps": n_steps,
        "u_centre_num": u_c_num,
        "u_centre_ana": u_c_ana,
        "centre_rel_pct": abs(u_c_num / u_c_ana - 1.0) * 100.0,
        "profile_rel_max_pct": float(
            np.abs(prof[interior] / ana[interior] - 1.0).max() * 100.0
        ),
        "density_drift_pct": solver.density_drift_pct(),
        "y": solver.y.tolist(),
        "u_num": prof.tolist(),
        "u_ana": ana.tolist(),
    }