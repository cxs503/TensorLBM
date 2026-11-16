"""Transient (oscillatory) verification of the Zou-He channel closure (D2Q9).

Anchor
------
``XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI`` -- the streamwise closure
that reconstructs the unknown populations on the inlet/outlet planes.

Why this module exists (the verification debt it closes)
-------------------------------------------------------
The steady debt of this closure is already paid: ``verification/zou_he_channel``
compares the converged ``pressure-pressure`` / ``velocity-pressure`` channel
against plane Poiseuille, 2-D and 3-D.  A *steady* comparison cannot see a
closure that is time-accurate but momentarily out of phase -- and that is
exactly the risk of a reconstruction BC: it rebuilds the unknown populations
from *stale* post-streaming data every step, so it can behave like a lag filter
at the boundary.  The pulsatile (Womersley) case already in the repo
(``verification/womersley_pulsatile``) does **not** exercise this path: it is
driven by a uniform body force with **periodic** streamwise boundaries, so the
Zou-He planes never appear.

Here the driving is applied *through the Zou-He planes themselves*: both ends
are pressure-mode reconstructions with

    rho(0, t)   = 1 + (drho/2) cos(omega t)
    rho(nx-1, t)= 1 - (drho/2) cos(omega t)

(quasi-static ramp over ``ramp_periods``), i.e. the flux is emergent, exactly as
in the steady ``pressure-pressure`` arm.  The flow is fully developed over the
middle of the channel and the periodic response is compared with the exact
oscillatory plane-Poiseuille (Sexl 1930) solution, single-sourced from
``verification.womersley_pulsatile.analytical``::

    u_x(y, t) = Re{ U(y) e^{i omega t} },
    U(y) = (a0 / (i omega)) [ 1 - cosh(lam eta) / cosh(lam) ],
    eta = (y - H/2)/(H/2),  lam = sqrt(i) alpha,  alpha = (H/2) sqrt(omega/nu)

with ``a0 = -(1/3) d rho / dx`` (lattice equation of state, ``a0`` complex so
the *measured* driving amplitude **and phase** are used -- see below).

Two deliberate design choices
-----------------------------
1. **The driving is measured, not assumed.**  The distance between the two
   reconstructed planes carries an O(1) convention ambiguity (soft BC, wall
   mid-plane offsets), so the reference is built from the complex density
   gradient fitted to the *computed* field over the middle third of the
   channel.  The stricter metric built on the nominal gradient
   ``drho / (3 (nx-1))`` is reported alongside, never instead.
2. **The fit is local.**  The pressure field is not strictly uniform in ``x``:
   the acoustic transit time ``L / c_s`` gives a real ``O(omega L / c_s)`` phase
   variation of the local gradient.  Fitting over the middle third keeps that
   systematic out of the reported error, and the residual ``x``-dependence of
   the amplitude is itself reported as ``x_spread``.

The estimator is a single-frequency phasor correlation over a whole number of
periods, so a non-converged transient or a frequency mismatch shows up as
amplitude loss, not as a silent fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from ..boundary.d2q9 import C2Q9
from ..boundary.zou_he_channel import zou_he_wall
from .womersley_pulsatile import analytical as sexl
from .zou_he_channel.solver import _stream, equilibrium_d2q9

__all__ = [
    "UnsteadyCase",
    "run_oscillatory",
    "phasor_metrics",
    "measured_gradient",
    "analytic_reference",
    "nominal_gradient",
]

_CS = 1.0 / math.sqrt(3.0)          # lattice speed of sound
_FIT_BAND = (3.0 / 8.0, 5.0 / 8.0)  # middle third of the channel (x / L)


@dataclass(frozen=True)
class UnsteadyCase:
    """Geometry / physics of one oscillatory channel arm (lattice units).

    The Womersley number fixes the frequency (the steady module fixes the
    gradient); ``length`` is only the number of cells between the two
    reconstructed planes and therefore must be long enough for the middle third
    to be fully developed.
    """

    height: int = 16               # H = fluid rows; ny = H + 2 (two solid rows)
    length: int = 48               # nx, inlet at x=0, outlet at x=nx-1
    tau: float = 0.56              # nu = (tau - 1/2)/3 = 0.02
    drho: float = 4.0e-3           # peak-to-peak end-density amplitude
    womersley: float = 2.0         # alpha = (H/2) sqrt(omega/nu)
    ramp_periods: float = 2.0      # quasi-static ramp of the end densities
    warmup_periods: int = 3        # periods discarded before averaging
    avg_periods: int = 3           # periods used by the phasor correlation
    dtype: torch.dtype = torch.float64

    @property
    def ny(self) -> int:
        return self.height + 2

    @property
    def nx(self) -> int:
        return self.length

    @property
    def nu(self) -> float:
        return (self.tau - 0.5) / 3.0

    @property
    def omega(self) -> float:
        """Angular frequency giving ``womersley`` (channel, gap ``H``)."""
        return sexl.omega_from_alpha("channel", self.womersley, self.nu, float(self.height))

    @property
    def period(self) -> float:
        return 2.0 * math.pi / self.omega

    @property
    def alpha(self) -> float:
        return sexl.alpha_from_omega("channel", self.omega, self.nu, float(self.height))

    @property
    def stokes_cells(self) -> float:
        """Stokes-layer thickness in cells -- must be well resolved."""
        return 0.5 * self.height / self.alpha

    @property
    def acoustic_phase_deg(self) -> float:
        """``omega * L / c_s`` in degrees: how non-uniform the driving is in x."""
        return math.degrees(self.omega * (self.nx - 1) / _CS)

    def n_steps(self) -> int:
        return int(round((self.warmup_periods + self.avg_periods) * self.period))


def nominal_gradient(case: UnsteadyCase) -> complex:
    """Driving amplitude if the planes were ``nx-1`` apart (real, phase 0)."""
    return complex(case.drho / (3.0 * (case.nx - 1)), 0.0)


def measured_gradient(
    rho_phasor: np.ndarray, x: np.ndarray, band=_FIT_BAND
) -> complex:
    """Complex ``-(1/3) d rho / dx`` from a least-squares fit inside ``band``."""
    x = np.asarray(x, dtype=float)
    r = np.asarray(rho_phasor, dtype=complex)
    lo, hi = band[0] * x[-1], band[1] * x[-1]
    sel = (x >= lo) & (x <= hi)
    xm, rm = x[sel].mean(), r[sel].mean()
    slope = np.sum((x[sel] - xm) * (r[sel] - rm)) / np.sum((x[sel] - xm) ** 2)
    return complex(-slope / 3.0)


def analytic_reference(
    case: UnsteadyCase, a0: complex, alpha: Optional[float] = None
) -> np.ndarray:
    """Exact Sexl/oscillatory-Poiseuille phasor ``U(y)`` on the fluid rows."""
    alpha = case.alpha if alpha is None else float(alpha)
    coord = np.arange(1, case.ny - 1, dtype=float) - 0.5   # 0 at lower wall plane
    return sexl.phasor(
        "channel", coord, float(case.height), alpha, complex(a0), case.omega
    )


def phasor_metrics(
    case: UnsteadyCase,
    U: np.ndarray,
    a0: complex,
    alpha: Optional[float] = None,
    core_frac: float = 0.2,
) -> Dict[str, float]:
    """Compare one measured complex velocity profile with the exact reference.

    Amplitude error is normalised by the peak reference amplitude, with the
    peak-row and axis-row values reported on their own; the phase error is taken
    over the rows where the reference amplitude exceeds ``core_frac`` of its
    peak, because the phase of a vanishing amplitude is noise, not evidence.
    """
    U = np.asarray(U, dtype=complex)
    Ur = analytic_reference(case, a0, alpha=alpha)
    scale = float(np.abs(Ur).max())
    amp_ref = np.abs(Ur)
    core = amp_ref >= core_frac * scale

    d_amp = np.abs(U) - amp_ref
    amp_rel_l2 = float(np.sqrt(np.mean(d_amp ** 2)) / scale)
    amp_rel_linf = float(np.abs(d_amp).max() / scale)

    # NOTE: for alpha >~ 3 the reference amplitude peaks *off* the axis (the
    # Sexl / Womersley "annular effect": the Stokes layer overshoots the core),
    # so the peak row and the axis row are reported separately -- calling the
    # peak "centreline" would silently mis-name the physics.
    i_peak = int(np.argmax(amp_ref))
    amp_peak_err = float(abs(abs(U[i_peak]) - scale) / scale)
    H = case.height
    axis_rows = (H // 2 - 1, H // 2) if H % 2 == 0 else ((H - 1) // 2,)
    amp_axis_ref = float(amp_ref[list(axis_rows)].mean())
    amp_axis_err = float(
        abs(float(np.abs(U[list(axis_rows)]).mean()) - amp_axis_ref) / amp_axis_ref
    )

    dphase = np.angle(U * np.conj(Ur))
    phase_rms = float(np.degrees(np.sqrt(np.mean(dphase[core] ** 2))))
    phase_axis_deg = float(
        np.degrees(abs(dphase[list(axis_rows)].mean()))
    )

    return {
        "alpha_ref": float(case.alpha if alpha is None else alpha),
        "a0_abs": float(abs(a0)),
        "a0_phase_deg": float(np.degrees(np.angle(a0))),
        "u_amp_max_sim": float(np.abs(U).max()),
        "u_amp_max_ref": scale,
        "amp_rel_l2": amp_rel_l2,
        "amp_rel_linf": amp_rel_linf,
        "amp_peak_err": amp_peak_err,
        "amp_axis_err": amp_axis_err,
        "i_peak": i_peak,
        "phase_rms_deg": phase_rms,
        "phase_axis_deg": phase_axis_deg,
        "n_core_rows": int(core.sum()),
    }


def _macro(f: torch.Tensor, dtype: torch.dtype):
    rho = f.sum(0)
    cx = C2Q9[:, 0].view(9, 1, 1).to(dtype)
    cy = C2Q9[:, 1].view(9, 1, 1).to(dtype)
    ux = (f * cx).sum(0) / rho
    uy = (f * cy).sum(0) / rho
    return rho, ux, uy


def run_oscillatory(case: UnsteadyCase, verbose: bool = False) -> Dict[str, object]:
    """Run the end-driven oscillatory channel and return the phasor field.

    Returns a dict with the phasors (``U`` on the fluid rows, ``rho`` and the
    streamwise flux ``q`` per column), the fitted / nominal driving amplitudes,
    and diagnostics (prescribed-density residual, ``x``-independence of the
    amplitude and of the flux, centreline time trace).
    """
    ny, nx, H = case.ny, case.nx, case.height
    dtype = case.dtype
    omega = case.omega

    rho0 = torch.ones((ny, nx), dtype=dtype)
    ux0 = torch.zeros((ny, nx), dtype=dtype)
    uy0 = torch.zeros((ny, nx), dtype=dtype)
    f = equilibrium_d2q9(rho0, ux0, uy0)

    fluid = torch.ones((ny, nx), dtype=torch.bool)
    fluid[0] = False
    fluid[-1] = False

    interior = slice(1, ny - 1)
    xs = np.arange(nx, dtype=float)
    cx = C2Q9[:, 0].view(9, 1, 1).to(dtype)

    n_steps = case.n_steps()
    avg_start = int(round(case.warmup_periods * case.period))
    ramp_steps = max(1.0, case.ramp_periods * case.period)

    ux_acc = np.zeros((H, nx), dtype=complex)
    rho_acc = np.zeros(nx, dtype=complex)
    q_acc = np.zeros(nx, dtype=complex)
    n_acc = 0
    presc_res = 0.0
    trace_steps, trace_u = [], []

    rho, ux, uy = _macro(f, dtype)
    for step in range(1, n_steps + 1):
        feq = equilibrium_d2q9(rho, ux, uy)
        f_post = f - (f - feq) / case.tau
        f = _stream(f_post, fluid)

        phase = omega * step
        ramp = min(1.0, step / ramp_steps)
        amp = 0.5 * case.drho * ramp * math.cos(phase)
        zou_he_wall(f, "west", mode="pressure", rho=1.0 + amp)
        zou_he_wall(f, "east", mode="pressure", rho=1.0 - amp)

        rho, ux, uy = _macro(f, dtype)

        if step >= avg_start:
            presc_res = max(
                presc_res,
                abs(rho[interior, 0].mean().item() - (1.0 + amp)),
                abs(rho[interior, -1].mean().item() - (1.0 - amp)),
            )
            w = np.exp(-1j * phase)
            ux_acc += ux[interior].numpy() * w
            rho_acc += rho[interior].mean(dim=0).numpy() * w
            q_acc += (f[:, interior, :] * cx).sum(dim=(0, 1)).numpy() * w
            n_acc += 1
            if verbose and n_acc % int(case.period) == 0:
                print(f"  step {step:7d}  period {n_acc / case.period:.1f} done")

        trace_steps.append(step)
        trace_u.append(ux[interior, nx // 2].mean().item())

    if n_acc == 0:
        raise RuntimeError("no averaging samples; increase avg_periods")

    # u(t) = Re{U e^{i omega t}}  =>  U = 2/N sum_t u(t) e^{-i omega t}
    scaling = 2.0 / float(n_acc)
    U_cols = ux_acc * scaling
    rho_ph = rho_acc * scaling
    q_ph = q_acc * scaling

    lo, hi = int(_FIT_BAND[0] * (nx - 1)), int(_FIT_BAND[1] * (nx - 1))
    U = U_cols[:, lo:hi + 1].mean(axis=1)          # fully developed profile
    g_fit = measured_gradient(rho_ph, xs)
    amp_cols = np.abs(U_cols[:, lo:hi + 1]).mean(axis=0)
    x_spread = float(np.std(amp_cols) / max(np.mean(amp_cols), 1e-30))
    q_core = np.abs(q_ph[lo:hi + 1])
    q_spread = float(np.std(q_core) / max(np.mean(q_core), 1e-30))

    return {
        "case": case,
        "steps": n_steps,
        "omega": omega,
        "alpha": case.alpha,
        "stokes_cells": case.stokes_cells,
        "acoustic_phase_deg": case.acoustic_phase_deg,
        "period": case.period,
        "n_avg": n_acc,
        "U": U,
        "U_cols": U_cols,
        "x": xs,
        "fit_band_indices": (lo, hi),
        "rho_phasor": rho_ph,
        "q_phasor": q_ph,
        "g_fit": g_fit,
        "g_nominal": nominal_gradient(case),
        "x_spread": x_spread,
        "q_spread": q_spread,
        "prescription_residual": presc_res,
        "trace_steps": np.asarray(trace_steps),
        "trace_u": np.asarray(trace_u),
    }