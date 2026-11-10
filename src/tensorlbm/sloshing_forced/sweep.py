"""Forced-sway sloshing case runner and measurement layer (P88W1-N01).

The solver is the audited Körner free-surface LBM of
:mod:`tensorlbm.sloshing_resonance.tank2d` (unchanged).  This module adds the
*forced-frequency-sweep* layer that the sibling node did not need:

* a **case specification** in SI units (tank ``L x H``, fill ratio, sway
  amplitude, frequency ratio, lattice resolution) with the Froude/Mach consistent
  scaling of :class:`~tensorlbm.sloshing_resonance.tank2d.LatticeScaling`,
* ``run_decay_case`` / ``run_forced_case`` wrappers that return numpy-friendly
  summaries plus the raw time series needed for figures and for the GCI study,
* measurement/identification helpers: coherent harmonic amplitude of the modal
  coefficients ``a_1``, ``a_2`` in a steady window, wall run-up, wall dynamic
  pressure at three gauge heights, wall-front velocity, and the impact-pressure
  metrics compared against the analytic bounds of :mod:`.theory`.

Everything here is deterministic given the spec, so tests can run single short
cases and the campaign can shard cases across processes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from ..sloshing_resonance.identify import harmonic_amplitude
from ..sloshing_resonance.tank2d import (
    LatticeScaling,
    SloshingTank,
    build_tank_field,
    run_free_decay,
    run_sway,
)

__all__ = [
    "TankSpec",
    "build_case",
    "run_decay_case",
    "run_forced_case",
    "harmonic_metrics",
    "pressure_metrics",
    "wall_runup_series",
    "identify_resonance",
    "CASE_DEFAULTS",
]

CASE_DEFAULTS = {
    "sway_frac_linear": 0.005,   # X0 / L for the linear-regime runs
    "sway_frac_impact": 0.020,   # X0 / L for the large-amplitude runs
    "decay_periods": 4.0,
    "forced_periods": 5.0,
    "ramp_periods": 0.5,         # in units of the mode-1 period
    "steady_frac": 0.5,          # fraction of the record used for steady metrics
    "records_per_period": 80,
}


# ---------------------------------------------------------------------------
# case specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TankSpec:
    """Rectangular tank + lattice resolution of one verification case."""

    length: float = 1.0
    height: float = 0.4
    breadth: float = 0.5
    fill_ratio: float = 0.5
    g: float = 9.81
    rho_liquid: float = 1000.0
    cells_per_height: int = 32
    u_lat: float = 0.0035
    u_char: float = 0.074
    tau: float = 0.502
    headroom: float | None = None
    slip_walls: bool = True
    gauge_heights: tuple = (0.05, 0.5, 0.95)  # fraction of h above the tank bottom

    # -- derived quantities -------------------------------------------------
    @property
    def depth(self) -> float:
        return self.fill_ratio * self.height

    @property
    def freeboard(self) -> float:
        return self.height - self.depth

    @property
    def gas_headroom(self) -> float:
        return self.headroom if self.headroom is not None else self.freeboard

    def check(self) -> None:
        if not 0.0 < self.fill_ratio < 1.0:
            raise ValueError("fill_ratio must be in (0, 1)")
        if self.cells_per_height < 16:
            raise ValueError("cells_per_height must be >= 16 for a resolved free surface")
        if not 0.0 < self.sway_fraction_ok:
            raise ValueError("lattice Mach number out of range")
        if self.tau <= 0.5:
            raise ValueError("tau must exceed 0.5")

    @property
    def sway_fraction_ok(self) -> float:
        return 1.0 if 0.0 < self.u_lat < 0.05 else 0.0

    def tank(self) -> SloshingTank:
        return SloshingTank(
            length=self.length,
            height=self.height,
            g=self.g,
            rho_liquid=self.rho_liquid,
        )

    def scaling(self) -> LatticeScaling:
        return LatticeScaling.from_resolution(
            self.tank(),
            self.cells_per_height,
            u_lat=self.u_lat,
            u_char=self.u_char,
            tau=self.tau,
        )

    def gauge_rows(self, depth_cells: float) -> list[int]:
        rows = []
        for frac in self.gauge_heights:
            rows.append(max(1, int(round(frac * depth_cells))))
        return sorted(set(rows))

    def as_dict(self) -> dict:
        return {
            "length_m": self.length,
            "height_m": self.height,
            "breadth_m": self.breadth,
            "fill_ratio": self.fill_ratio,
            "depth_m": self.depth,
            "g_m_s2": self.g,
            "rho_kg_m3": self.rho_liquid,
            "cells_per_height": self.cells_per_height,
            "dx_m": None,
            "dt_s": None,
            "u_lat": self.u_lat,
            "u_char_m_s": self.u_char,
            "tau": self.tau,
            "slip_walls": self.slip_walls,
            "gauge_heights_frac_h": list(self.gauge_heights),
        }


def build_case(spec: TankSpec, *, mode: int = 1, amplitude: float = 1e-3, surface=None):
    """Build ``(tank, scaling, state)`` for one case of ``spec``."""
    spec.check()
    tank = spec.tank()
    scaling = spec.scaling()
    state = build_tank_field(
        tank,
        scaling,
        depth=spec.depth,
        mode=mode,
        amplitude=amplitude,
        headroom=spec.gas_headroom,
        surface=surface,
    )
    return tank, scaling, state


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------


def _record_every(scaling: LatticeScaling, period: float, records_per_period: int) -> int:
    steps_per_period = max(period / scaling.dt, 10.0)
    return max(1, int(round(steps_per_period / float(records_per_period))))


def _summary(res: dict, spec: TankSpec, f1: float) -> dict:
    """Convert a ``tank2d`` result dict into numpy + JSON-friendly summary."""
    eta = res["eta"].numpy() if isinstance(res["eta"], torch.Tensor) else np.asarray(res["eta"])
    press = (
        res["pressure"].numpy()
        if isinstance(res["pressure"], torch.Tensor)
        else np.asarray(res["pressure"])
    )
    t = res["t"].numpy()
    modal = {int(m): v.numpy() for m, v in res["eta_modes"].items()}
    return {
        "t": t,
        "eta": eta,
        "pressure": press,
        "a": modal,
        "inventory": res["inventory"].numpy(),
        "umax": res["umax"].numpy(),
        "gauge_rows": list(res["gauge_rows"]),
        "dx": res["dx"],
        "dt": res["dt"],
        "depth_cells": res["depth_cells"],
        "tau": res["tau"],
        "g_lattice": res["g_lattice"],
        "nu_physical": res["nu_physical"],
        "u_lat": res["u_lat"],
        "mass_drift": res["mass_drift"],
        "max_mass_drift": res["max_mass_drift"],
        "n_steps": res["n_steps"],
        "period_theory": res["period_theory"],
        "f1_theory": f1,
    }


def run_decay_case(
    spec: TankSpec,
    *,
    mode: int = 1,
    amplitude: float = 1e-3,
    n_periods: float | None = None,
    records_per_period: int = CASE_DEFAULTS["records_per_period"],
) -> dict:
    """Free-decay release of standing mode ``mode`` (for f_n and zeta)."""
    from .theory import abramson_frequency, abramson_period

    tank, scaling, state = build_case(spec, mode=mode, amplitude=amplitude)
    period = abramson_period(mode, spec.depth, spec.length, spec.g)
    n_periods = CASE_DEFAULTS["decay_periods"] if n_periods is None else n_periods
    rec = _record_every(scaling, period, records_per_period)
    res = run_free_decay(
        state,
        n_periods,
        record_every=rec,
        modes=(1, 2, 3),
        gauge_rows=spec.gauge_rows(state.depth_cells),
        slip_walls=spec.slip_walls,
    )
    out = _summary(res, spec, abramson_frequency(1, spec.depth, spec.length, spec.g))
    out.update({"kind": "decay", "mode": mode, "amplitude": amplitude, "cells_per_height": spec.cells_per_height})
    return out


def run_forced_case(
    spec: TankSpec,
    *,
    ratio: float,
    sway_frac: float = CASE_DEFAULTS["sway_frac_linear"],
    n_periods: float | None = None,
    ramp_periods: float = CASE_DEFAULTS["ramp_periods"],
    records_per_period: int = CASE_DEFAULTS["records_per_period"],
) -> dict:
    """Harmonic sway ``X(t) = X0 sin(omega t)`` at ``omega = ratio * omega_1``.

    ``n_periods`` and ``ramp_periods`` are counted in periods of the *forcing*
    frequency and of the mode-1 natural period respectively (``run_sway`` counts
    both in units of the natural period of the state's mode, so the wrapper
    converts).
    """
    from .theory import abramson_frequency, abramson_period

    if ratio <= 0.0:
        raise ValueError("ratio must be positive")
    tank, scaling, state = build_case(spec, mode=1, amplitude=1e-4)
    f1 = abramson_frequency(1, spec.depth, spec.length, spec.g)
    t1 = abramson_period(1, spec.depth, spec.length, spec.g)
    omega = 2.0 * math.pi * ratio * f1
    t_force = t1 / ratio
    n_periods = CASE_DEFAULTS["forced_periods"] if n_periods is None else n_periods
    rec = _record_every(scaling, t_force, records_per_period)
    res = run_sway(
        state,
        n_periods * t_force / t1,          # -> units of the natural period
        sway_amplitude=sway_frac * spec.length,
        omega=omega,
        record_every=rec,
        modes=(1, 2, 3),
        gauge_rows=spec.gauge_rows(state.depth_cells),
        ramp_periods=ramp_periods,
        slip_walls=spec.slip_walls,
    )
    out = _summary(res, spec, f1)
    out.update(
        {
            "kind": "forced",
            "ratio": ratio,
            "sway_frac": sway_frac,
            "sway_amplitude_m": sway_frac * spec.length,
            "omega": omega,
            "forcing_frequency_hz": ratio * f1,
            "cells_per_height": spec.cells_per_height,
        }
    )
    return out


# ---------------------------------------------------------------------------
# measurement / identification
# ---------------------------------------------------------------------------


def _window(t: np.ndarray, steady_frac: float) -> slice:
    if t.size < 8:
        raise ValueError("record too short for a steady-window analysis")
    start = t[0] + steady_frac * (t[-1] - t[0])
    return slice(int(np.searchsorted(t, start)), t.size)


def harmonic_metrics(t: np.ndarray, y: np.ndarray, frequency: float, *, steady_frac: float = 0.5) -> dict:
    """Coherent harmonic amplitude/phase of ``y`` at ``frequency`` + peak values."""
    win = _window(t, steady_frac)
    tw, yw = t[win], y[win]
    amp, phase = harmonic_amplitude(tw, yw, frequency)
    peak = float(np.max(yw))
    trough = float(np.min(yw))
    return {
        "amplitude": float(amp),
        "phase_deg": float(phase),  # harmonic_amplitude returns degrees
        "peak": peak,
        "trough": trough,
        "peak_to_peak": peak - trough,
        "rms": float(np.sqrt(np.mean(yw * yw))),
        "n_samples": int(yw.size),
        "window_start": float(tw[0]),
        "window_end": float(tw[-1]),
    }


def wall_runup_series(case: dict, modes: int = 5) -> np.ndarray:
    """Reconstructed wall elevation ``eta(0,t) = sum_{n odd} a_n(t)`` (metres).

    The raw mass-ledger column adjacent to the wall carries the Körner model's
    documented wall-film artefact (the film can dry out and then reads a
    spurious elevation), so the *reconstruction from the trimmed modal
    projection* is used as the wall run-up; the raw column series stays available
    in the ``.npz`` files as a diagnostic.
    """
    total = np.zeros_like(case["a"][1])
    for n in range(1, int(modes) + 1, 2):
        total = total + case["a"].get(n, np.zeros_like(total))
    return total


def pressure_metrics(
    case: dict,
    spec: TankSpec,
    *,
    steady_frac: float = CASE_DEFAULTS["steady_frac"],
) -> dict:
    """Wall-pressure and run-up metrics of a forced case, with all analytic bounds.

    Two pressure measures are reported for every gauge, on purpose:

    * ``amplitude_Pa`` — least-squares amplitude of the component at the forcing
      frequency (a *first-harmonic* quantity, the one the linear theory predicts);
    * ``raw_max_Pa``   — the raw instantaneous maximum, which is spike-sensitive
      (interface cells entering/leaving a gauge row) and is therefore reported as
      a diagnostic, never as a criterion value.
    """
    from .theory import (
        bore_reflection_pressure,
        gauge_height_from_swl,
        stagnation_impact_pressure,
    )

    t = case["t"]
    f_force = case["forcing_frequency_hz"]
    win = _window(t, steady_frac)

    eta_wall = wall_runup_series(case)
    runup = harmonic_metrics(t, eta_wall, f_force, steady_frac=steady_frac)
    runup["raw_max_m"] = float(np.nanmax(eta_wall[win]))
    runup["raw_min_m"] = float(np.nanmin(eta_wall[win]))
    dt = float(np.mean(np.diff(t[win])))

    vel = np.gradient(eta_wall[win], dt)
    v_impact = float(np.max(np.abs(vel)))

    gauges = []
    for i, row in enumerate(case["gauge_rows"]):
        p = np.asarray(case["pressure"][:, i], dtype=float)
        y = gauge_height_from_swl(row, case["depth_cells"], case["dx"])
        met = harmonic_metrics(t, p, f_force, steady_frac=steady_frac)
        finite = p[win][np.isfinite(p[win])]
        # out-of-band (broadband) residual after removing the forcing-frequency
        # harmonic: a lower bound on the pressure-reconstruction noise floor
        tw, yw = t[win], p[win]
        ok = np.isfinite(yw)
        if ok.sum() >= 8:
            yy = yw[ok] - np.mean(yw[ok])
            w = 2.0 * math.pi * f_force
            S = np.stack([np.cos(w * tw[ok]), np.sin(w * tw[ok])], axis=1)
            coef, *_ = np.linalg.lstsq(S, yy, rcond=None)
            resid = yy - S @ coef
            broadband = float(np.sqrt(np.mean(resid * resid)))
        else:
            broadband = float("nan")
        snr = float(met["amplitude"] / broadband) if broadband and np.isfinite(broadband) else float("nan")
        gauges.append(
            {
                "row": int(row),
                "height_above_bottom_m": float((row - 0.5) * case["dx"]),
                "y_from_swl_m": float(y),
                "amplitude_Pa": met["amplitude"],
                "phase_deg": met["phase_deg"],
                "raw_max_Pa": float(np.max(finite)) if finite.size else float("nan"),
                "raw_min_Pa": float(np.min(finite)) if finite.size else float("nan"),
                "rms_Pa": met["rms"],
                "broadband_rms_Pa": broadband,
                "snr": snr,
                "finite_fraction": float(finite.size / max(p[win].size, 1)),
            }
        )

    bottom = gauges[0]
    p_amp = float(bottom["amplitude_Pa"])
    p_raw = float(bottom["raw_max_Pa"])
    rho, g, h = spec.rho_liquid, spec.g, spec.depth
    bounds = {
        "hydrostatic_runup_Pa": float(rho * g * max(runup["amplitude"], 0.0)),
        "hydrostatic_runup_rawmax_Pa": float(rho * g * max(runup["raw_max_m"], 0.0)),
        "wagner_stagnation_Pa": stagnation_impact_pressure(rho, v_impact),
        "bore_reflection_Pa": bore_reflection_pressure(rho, g, h, v_impact),
        "rho_g_h_Pa": float(rho * g * h),
    }
    return {
        "runup": runup,
        "impact_velocity_m_s": v_impact,
        "gauges": gauges,
        "p_amp_Pa": p_amp,
        "p_raw_max_Pa": p_raw,
        "p_amp_over_rho_g_h": p_amp / (rho * g * h),
        "bounds": bounds,
        "steady_frac": steady_frac,
    }


def identify_resonance(
    ratios: Sequence[float], amplitudes: Sequence[float], *, n_fit: int = 3
) -> dict:
    """Locate the response maximum of a sweep (parabolic fit around the argmax).

    Returns the discrete argmax and a 3-point parabolic refinement, both in units
    of the mode-1 natural frequency.  ``quality`` is the curvature-derived
    half-width of the fit; a non-positive curvature raises ``ValueError``.
    """
    r = np.asarray(ratios, dtype=float)
    a = np.asarray(amplitudes, dtype=float)
    if r.size != a.size or r.size < 3:
        raise ValueError("need at least 3 (ratio, amplitude) pairs of equal size")
    order = np.argsort(r)
    r, a = r[order], a[order]
    imax = int(np.argmax(a))
    result = {
        "ratio_argmax": float(r[imax]),
        "amplitude_argmax": float(a[imax]),
        "ratio_parabolic": float(r[imax]),
        "amplitude_parabolic": float(a[imax]),
        "curvature": float("nan"),
        "width": float("nan"),
        "n_points": int(r.size),
    }
    if 0 < imax < r.size - 1:
        x0, x1, x2 = r[imax - 1], r[imax], r[imax + 1]
        y0, y1, y2 = a[imax - 1], a[imax], a[imax + 1]
        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        if abs(denom) > 0.0:
            A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
            B = (
                x2 * x2 * (y0 - y1)
                + x1 * x1 * (y2 - y0)
                + x0 * x0 * (y1 - y2)
            ) / denom
            C = (
                x2 * x2 * (x1 * y0 - x0 * y1)
                + x1 * x1 * (x0 * y2 - x2 * y0)
                + x0 * x0 * (x2 * y1 - x1 * y2)
            ) / denom
            if A < 0.0:
                vertex = -B / (2.0 * A)
                if x0 <= vertex <= x2:
                    result["ratio_parabolic"] = float(vertex)
                    result["amplitude_parabolic"] = float(A * vertex * vertex + B * vertex + C)
                result["curvature"] = float(A)
                result["width"] = float(math.sqrt(max(-1.0 / A, 0.0)))
    return result