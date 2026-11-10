"""Acceptance campaign for campaign node P142W3-N01.

LNG membrane-tank partial-fill sloshing: LBM two-phase free-surface runs x
analytic resonance anchor x Wagner/MLM impact anchors x Dirlik/rainflow fatigue
load spectrum.

Cases
-----
``freq``    small-amplitude free decay of the first mode at 15/30/50 % fill
            (plus a 3-grid family at 30 % for the GCI): measures ``f_1`` and the
            damping ratio, judges it against the linear dispersion anchor
            ``f_n = sqrt(g k_n tanh(k_n h)) / 2 pi``.
``impact``  forced harmonic sway at ``omega_1`` with ``X0 = 0.02 L``:
            wall-pressure history -> peak dynamic pressure, impact kinematics
            (near-wall normal velocity, free-surface deadrise), stress history
            -> Dirlik and rainflow damage.
``wedge``   controlled Wagner/MLM benchmark: the liquid body starts with a
            prescribed wedge corner angle ``gamma`` against the wall and a
            uniform normal velocity ``V``; the measured dynamic wall-pressure
            excess is compared with the MLM anchor
            ``p = rho V^2 (pi / (2 tan gamma) - 1/2)``.

Every case writes one JSON (scalars) plus one ``.npz`` (time series) so the
campaign is shardable and resumable, and every number in the report is read
back from those files (no hand copying).

Usage
-----
    python -m tensorlbm.lng_membrane_sloshing.campaign run --case freq:0.30:16
    python -m tensorlbm.lng_membrane_sloshing.campaign summarize --outdir runs/p142w3_n01
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..free_surface_lbm_2d import GAS, INTERFACE, LIQUID, koerner_step_2d
from ..sloshing_resonance.tank2d import (
    LatticeScaling,
    SloshingTank,
    TankField,
    build_tank_field,
    column_liquid_height,
    modal_amplitude,
    surface_elevation,
    wall_pressure_profile,
)
from ..sloshing_resonance.theory import abramson_frequency, abramson_period
from . import audit, fatigue, impact
from .tank import (
    DEFAULT_FILLS,
    LNGProperties,
    MembraneTank174k,
    PanelModel,
    scaling_for_case,
    similarity_report,
)

__all__ = [
    "build_case_state",
    "march",
    "run_frequency_case",
    "run_impact_case",
    "run_wedge_case",
    "summarize",
    "CASE_MATRIX",
]

CS2 = 1.0 / 3.0
NEAR_WALL_COLUMNS = 6


# ---------------------------------------------------------------------------
# state construction
# ---------------------------------------------------------------------------


def build_case_state(
    tank: MembraneTank174k,
    scaling: LatticeScaling,
    *,
    fill: float,
    headroom: float | None = None,
    surface=None,
    initial_velocity: float = 0.0,
    device: str | torch.device = "cpu",
) -> TankField:
    """Build the free-surface state (optionally with a uniform initial velocity)."""
    depth = tank.fill_height(fill)
    if headroom is None:
        headroom = tank.height - depth
    state = build_tank_field(
        tank.sloshing_tank,
        scaling,
        depth=depth,
        headroom=headroom,
        amplitude=0.0,
        surface=surface,
        device=device,
    )
    if initial_velocity != 0.0:
        rho = state.rho
        active = (state.flags != GAS) & (~state.solid)
        u_lat = initial_velocity * scaling.dt / scaling.dx
        ux = torch.where(
            active, torch.full_like(rho, u_lat), torch.zeros_like(rho)
        )
        uy = torch.zeros_like(rho)
        from ..d2q9 import equilibrium

        state.f = torch.where(
            active.unsqueeze(0), equilibrium(rho, ux, uy), torch.zeros_like(state.f)
        )
    return state


def _fluid_mask(state: TankField) -> torch.Tensor:
    return ((state.flags == LIQUID) | (state.flags == INTERFACE)) & (~state.solid)


# ---------------------------------------------------------------------------
# marching loop with energy accounting and impact measurement
# ---------------------------------------------------------------------------


def march(
    state: TankField,
    n_steps: int,
    *,
    gx_of_t=None,
    record_every: int = 5,
    rho_liquid: float = 1.0,
    device: str | torch.device = "cpu",
) -> dict:
    """March the free-surface model, recording pressure/kinematics/energy.

    Returns a dict of time series (SI units for lengths, velocities and
    pressures through ``state.scaling``).
    """
    cfg = state.scaling
    g_lat = cfg.g_lattice
    p_scale = cfg.pressure_scale(state.tank)
    dx, dt = cfg.dx, cfg.dt
    tol = 1e-12

    times: list[float] = []
    press: list[np.ndarray] = []
    etas: list[np.ndarray] = []
    modal1: list[float] = []
    inv: list[float] = []
    v_wall: list[float] = []
    ke: list[float] = []
    pe: list[float] = []
    diss: list[float] = []
    work: list[float] = []
    e_ref = [0.0]

    def snapshot(step: int) -> None:
        prof = wall_pressure_profile(state)
        times.append(step * dt)
        press.append(prof.detach().cpu().numpy())  # NaN kept: dry rows must stay masked
        eta = surface_elevation(state)
        etas.append(eta.detach().cpu().numpy())
        modal1.append(modal_amplitude(state, 1))
        inv.append(float(state.inventory))
        mask = _fluid_mask(state)
        rho = state.rho
        u = audit.macroscopic_velocity(state.f, rho, mask)
        cols = 1 + NEAR_WALL_COLUMNS
        band = mask[:, 1:cols]
        ux_band = u[:, 1:cols, 0]
        v_wall.append(float(ux_band[band].abs().max()) * dx / dt if band.any() else 0.0)
        e = audit.mechanical_energy(state.f, _liquid_content(state), mask, g=g_lat)
        ke.append(e["ke"])
        pe.append(e["pe"])
        if e_ref[0] <= 0.0:
            e_ref[0] = max(e["total"], tol)
        d = audit.viscous_dissipation_2d(state.f, rho, u, cfg.tau)
        diss.append(float(d[mask].sum()))
        gx = 0.0 if gx_of_t is None else float(gx_of_t(step))
        work.append(float((rho * u[..., 0] * gx)[mask].sum()))

    snapshot(0)
    for n in range(1, n_steps + 1):
        gx = 0.0 if gx_of_t is None else float(gx_of_t(n))
        state.f, state.mass, state.flags = koerner_step_2d(
            state.f,
            state.mass,
            state.flags,
            state.solid,
            tau=cfg.tau,
            gx=gx,
            gy=-g_lat,
            rho_liquid=1.0,
            rho_gas=1.0,
        )
        state.step += 1
        if n % record_every == 0 or n == n_steps:
            snapshot(n)

    t = np.asarray(times)
    ke_a, pe_a = np.asarray(ke), np.asarray(pe)
    diss_a, work_a = np.asarray(diss), np.asarray(work)
    # trapezoidal accumulation of the rates (per lattice step)
    diss_total = float(np.trapezoid(diss_a, t / dt))
    work_total = float(np.trapezoid(work_a, t / dt))
    budget = audit.energy_budget(
        {"ke": ke_a[0], "pe": pe_a[0], "total": ke_a[0] + pe_a[0]},
        {"ke": ke_a[-1], "pe": pe_a[-1], "total": ke_a[-1] + pe_a[-1]},
        work_excitation=work_total,
        dissipated=diss_total,
        e_ref=max(ke_a[0] + pe_a[0], e_ref[0]),
    )
    return {
        "t": t,
        "pressure": np.asarray(press).T,  # (rows, time); NaN = dry gauge row
        "eta": np.asarray(etas),
        "modal1": np.asarray(modal1),
        "inventory": np.asarray(inv),
        "v_wall": np.asarray(v_wall),
        "ke": ke_a,
        "pe": pe_a,
        "dissipation_rate": diss_a,
        "work_rate": work_a,
        "energy_budget": budget,
        "n_steps": n_steps,
        "record_every": record_every,
    }


def _liquid_content(state: TankField) -> torch.Tensor:
    """Per-cell liquid inventory (LIQUID = 1, INTERFACE = mass / rho_liquid)."""
    content = torch.zeros_like(state.mass)
    content = torch.where(state.flags == LIQUID, torch.ones_like(content), content)
    interface = state.flags == INTERFACE
    content = torch.where(interface, (state.mass / 1.0).clamp(min=0.0), content)
    return content


# ---------------------------------------------------------------------------
# case runners
# ---------------------------------------------------------------------------


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or y.size < window:
        return y
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.concatenate([np.full(pad, y[0]), y, np.full(window - 1 - pad, y[-1])])
    return np.convolve(padded, kernel, mode="valid")


def _decay_frequency(t: np.ndarray, x: np.ndarray, smooth_fraction: float = 0.05) -> dict:
    """Frequency / damping of a decaying oscillation from its time series.

    The mode-1 amplitude carries interface-conversion jitter, so the signal is
    first smoothed over ``smooth_fraction`` of the record and the period is then
    taken from the *extrema* spacing (2 extrema per period) -- a zero-crossing
    count on the raw signal double counts the noise and over-predicts ``f`` by a
    factor 2-3 (documented defect found in the first campaign run).
    """
    n = x.size
    skip = max(int(0.02 * n), 1)
    y = x[skip:]
    tt = t[skip:]
    if y.size < 8:
        return {"f_damped": float("nan"), "zeta": float("nan"), "f_natural": float("nan")}
    y_s = _smooth(y, max(3, int(smooth_fraction * y.size)))
    ext = [i for i in range(1, y_s.size - 1) if (y_s[i] - y_s[i - 1]) * (y_s[i + 1] - y_s[i]) <= 0]
    if len(ext) >= 3:
        period = 2.0 * float(np.mean(np.diff(tt[ext])))
    else:
        sign = np.sign(y_s)
        cross = np.where(np.diff(sign) != 0)[0]
        period = 2.0 * float(np.mean(np.diff(tt[cross]))) if cross.size >= 2 else float("nan")
    f_d = 1.0 / period if period == period and period > 0 else float("nan")
    # log-envelope damping from the absolute peaks of each half cycle
    peaks = []
    for i in range(1, y.size - 1):
        if (y[i] - y[i - 1]) * (y[i + 1] - y[i]) <= 0 and abs(y[i]) > abs(y.mean()):
            peaks.append((tt[i], abs(y[i] - y.mean())))
    zeta = float("nan")
    if len(peaks) >= 4:
        pt = np.array([p[0] for p in peaks])
        pa = np.array([p[1] for p in peaks])
        good = pa > 0
        if good.sum() >= 4:
            slope = np.polyfit(pt[good], np.log(pa[good]), 1)[0]
            zeta = float(-slope / (2.0 * math.pi * f_d)) if f_d == f_d else float("nan")
    f_n = f_d / math.sqrt(1.0 - zeta**2) if (zeta == zeta and abs(zeta) < 1.0) else f_d
    return {"f_damped": f_d, "zeta": zeta, "f_natural": f_n, "period": period}


def run_frequency_case(
    *,
    fill: float,
    cells_per_depth: int,
    outdir: Path,
    amplitude_over_length: float = 0.005,
    n_periods: float = 5.0,
    record_every: int = 4,
    tau: float = 0.55,
    compressibility_budget: float = 0.02,
    device: str = "cpu",
) -> dict:
    """Free-decay run of the first mode at one fill level."""
    tank = MembraneTank174k()
    tank.check()
    h = tank.fill_height(fill)
    f1 = abramson_frequency(1, h, tank.length, 9.81)
    amplitude = amplitude_over_length * tank.length
    cph = int(round(cells_per_depth / (h / tank.height)))
    scaling = scaling_for_case(
        tank,
        fill=fill,
        cells_per_height=cph,
        amplitude=amplitude,
        frequency=f1,
        tau=tau,
        compressibility_budget=compressibility_budget,
    )
    depth_cells = h / scaling.dx
    headroom = tank.height - h
    state = build_tank_field(
        tank.sloshing_tank,
        scaling,
        depth=h,
        headroom=headroom,
        amplitude=amplitude,
        mode=1,
        device=device,
    )
    period_lat = abramson_period(1, h, tank.length, 9.81) / scaling.dt
    n_steps = max(int(round(n_periods * period_lat)), 50)
    t0 = time.time()
    out = march(state, n_steps, record_every=record_every, device=device)
    wall = time.time() - t0
    freq = _decay_frequency(out["t"], out["modal1"])
    f_anchor = abramson_frequency(1, h, tank.length, 9.81)
    record = {
        "kind": "frequency",
        "fill": fill,
        "depth_m": h,
        "cells_per_depth": depth_cells,
        "amplitude_m": amplitude,
        "f1_anchor_hz": f_anchor,
        "f1_lbm_hz": freq["f_natural"],
        "f1_damped_hz": freq["f_damped"],
        "zeta": freq["zeta"],
        "f1_error_pct": 100.0 * (freq["f_natural"] / f_anchor - 1.0),
        "energy_budget": out["energy_budget"],
        "inventory_drift": float((out["inventory"][-1] - out["inventory"][0]) / out["inventory"][0]),
        "similarity": similarity_report(tank, scaling, fill=fill),
        "n_steps": out["n_steps"],
        "wall_seconds": wall,
    }
    _save(outdir, f"freq_f{fill:.2f}_d{int(round(depth_cells))}", record, out)
    return record


def run_impact_case(
    *,
    fill: float,
    cells_per_depth: int,
    outdir: Path,
    amplitude_over_length: float = 0.02,
    n_periods: float = 8.0,
    ramp_periods: float = 2.0,
    record_every: int = 4,
    tau: float = 0.55,
    compressibility_budget: float = 0.02,
    device: str = "cpu",
) -> dict:
    """Forced harmonic sway at the first natural frequency -> wall load history."""
    tank = MembraneTank174k()
    tank.check()
    h = tank.fill_height(fill)
    f1 = abramson_frequency(1, h, tank.length, 9.81)
    omega = 2.0 * math.pi * f1
    amplitude = amplitude_over_length * tank.length
    cph = int(round(cells_per_depth / (h / tank.height)))
    scaling = scaling_for_case(
        tank,
        fill=fill,
        cells_per_height=cph,
        amplitude=amplitude,
        frequency=f1,
        tau=tau,
        compressibility_budget=compressibility_budget,
    )
    depth_cells = h / scaling.dx
    state = build_tank_field(
        tank.sloshing_tank,
        scaling,
        depth=h,
        headroom=tank.height - h,
        amplitude=1e-3 * h,
        mode=1,
        device=device,
    )
    period_lat = abramson_period(1, h, tank.length, 9.81) / scaling.dt
    n_steps = max(int(round(n_periods * period_lat)), 50)
    n_ramp = max(int(round(ramp_periods * period_lat)), 1)
    amp_lat = amplitude / scaling.dx
    w_lat = omega * scaling.dt

    def gx_of_t(n: int) -> float:
        ramp = min(1.0, n / float(n_ramp))
        return ramp * amp_lat * w_lat * w_lat * math.sin(w_lat * n)

    t0 = time.time()
    out = march(state, n_steps, gx_of_t=gx_of_t, record_every=record_every, device=device)
    wall = time.time() - t0

    rec = _impact_metrics(out, scaling, h, t_skip=ramp_periods * period_lat * scaling.dt)
    stress = fatigue.stress_from_pressure(
        rec["pressure_series"], span=0.8, thickness=0.022
    )
    dt_s = out["t"][1] - out["t"][0]
    f_cut = 6.0 * f1
    stress = fatigue.lowpass(stress, dt_s, f_cut=f_cut)
    d_dirlik = fatigue.dirlik_damage(stress, dt_s, f_max=f_cut)
    d_rf = fatigue.rainflow_damage(stress, threshold=1e-3 * float(np.max(np.abs(stress))))
    moments = fatigue.spectral_moments(stress, dt_s, f_max=f_cut)
    record = {
        "kind": "impact",
        "fill": fill,
        "depth_m": h,
        "cells_per_depth": depth_cells,
        "amplitude_m": amplitude,
        "omega": omega,
        "f1_anchor_hz": f1,
        **rec["scalars"],
        "gamma_mlm_deg": rec["scalars"]["beta_eff_deg"],
        "p_star_lbm": rec["scalars"]["p_max_pa"] / max(rec["scalars"]["rho_v2"], 1e-30),
        "fatigue": {
            "damage_dirlik": d_dirlik,
            "damage_rainflow": d_rf,
            "damage_ratio": (d_dirlik / d_rf) if d_rf > 0 else float("nan"),
            "damage_error_pct": (100.0 * (d_dirlik / d_rf - 1.0)) if d_rf > 0 else float("nan"),
            "spectral_moments": list(map(float, moments)),
            "bandwidth_eps": fatigue.wideband_bandwidth_parameter(moments),
            "stress_rms_mpa": float(np.std(stress)),
            "stress_max_mpa": float(np.max(np.abs(stress))),
            "cycles_rainflow": len(fatigue.rainflow_cycles(stress)),
        },
        "energy_budget": out["energy_budget"],
        "inventory_drift": float(
            (out["inventory"][-1] - out["inventory"][0]) / out["inventory"][0]
        ),
        "similarity": similarity_report(tank, scaling, fill=fill),
        "n_steps": out["n_steps"],
        "wall_seconds": wall,
    }
    _save(outdir, f"impact_f{fill:.2f}_d{int(round(depth_cells))}", record, out)
    return record


def _impact_metrics(out: dict, scaling: LatticeScaling, h: float, t_skip: float = 0.0) -> dict:
    """Peak dynamic wall pressure, impact kinematics and deadrise estimate.

    The wall gauges of :func:`tensorlbm.sloshing_resonance.tank2d.wall_pressure_profile`
    already return the pressure in excess of the *still-water* hydrostatic
    profile, so ``p_max`` is the dynamic wall pressure.  The pure impact excess
    is obtained by removing the *local* run-up hydrostatic column
    ``rho g eta_wall``.
    """
    tank = MembraneTank174k()
    rho = tank.lng.density
    g = 9.81
    dx = scaling.dx
    press = out["pressure"]  # Pa (dynamic w.r.t. still water), rows x time
    eta = out["eta"]         # m, per column (eta = 0 at rest)
    t = out["t"]
    if t_skip > 0.0:
        keep = t >= t_skip
        press, eta, t = press[:, keep], eta[keep], t[keep]
    if press.size == 0:
        return {"scalars": {}, "pressure_series": np.zeros(1)}
    finite = np.isfinite(press)
    pressed = np.where(finite, press, -np.inf)
    iy, it = np.unravel_index(np.argmax(pressed), pressed.shape)
    p_max = float(pressed[iy, it])
    y_cell = (iy - 0.5) * dx              # gauge height above the tank bottom [m]
    eta_wall = float(eta[it, 0])          # wall-column surface elevation [m]
    p_excess = p_max - rho * g * eta_wall
    v_impact = float(out["v_wall"][it])
    ncol = min(8, eta.shape[1] - 1)
    slope = (float(eta[it, ncol]) - eta_wall) / (ncol * dx)
    beta_deg = math.degrees(math.atan(max(abs(slope), 1e-6)))
    beta_deg = min(max(beta_deg, 1.0), 80.0)
    q = rho * v_impact**2
    scalars = {
        "p_max_pa": p_max,
        "p_max_time_s": float(t[it]),
        "p_max_height_m": float(y_cell),
        "p_dyn_excess_pa": p_excess,
        "v_impact_ms": v_impact,
        "rho_v2": q,
        "p_star_lbm": (p_excess / q) if q > 0 else float("nan"),
        "beta_eff_deg": beta_deg,
        "wall_slope": slope,
        "eta_wall_at_peak_m": eta_wall,
        "eta_max_m": float(np.max(eta)),
        "eta_min_m": float(np.min(eta)),
        "v_wall_max_ms": float(np.max(out["v_wall"])),
    }
    # the fatigue load history is taken from the gauge row with the largest
    # pressure fluctuation (the row of the peak can be dry for most of the
    # record, which would give a zero-variance "load")
    finite_press = np.where(finite, press, np.nan)
    row_std = np.nanstd(finite_press, axis=1)
    iy_load = int(np.nanargmax(row_std)) if np.any(row_std > 0) else iy
    pressure_series = np.nan_to_num(finite_press[iy_load, :])
    scalars["load_gauge_height_m"] = float((iy_load - 0.5) * dx)
    return {"scalars": scalars, "pressure_series": pressure_series}


def run_wedge_case(
    *,
    fill: float,
    gamma_deg: float,
    velocity: float,
    outdir: Path,
    cells_per_depth: int = 24,
    ramp_over_length: float = 0.10,
    run_time_s: float = 2.5,
    record_every: int = 1,
    tau: float = 0.55,
    device: str = "cpu",
) -> dict:
    """Controlled Wagner/MLM impact: wedge corner angle ``gamma`` at speed ``V``.

    The liquid surface rises linearly towards the wall over
    ``ramp_over_length * L`` and meets the vertical wall at the corner angle
    ``gamma`` (the Wagner deadrise).  The whole liquid body is given the uniform
    horizontal velocity ``V`` towards the wall.
    """
    tank = MembraneTank174k()
    tank.check()
    h = tank.fill_height(fill)
    f1 = abramson_frequency(1, h, tank.length, 9.81)
    cph = int(round(cells_per_depth / (h / tank.height)))
    scaling = scaling_for_case(
        tank,
        fill=fill,
        cells_per_height=cph,
        amplitude=velocity / (2 * math.pi * f1),
        frequency=f1,
        tau=tau,
        compressibility_budget=0.02,
    )
    ramp = ramp_over_length * tank.length
    slope = math.tan(math.radians(90.0 - gamma_deg))
    x0 = ramp

    def surface(x: float) -> float:
        if x >= x0:
            return h
        return h + (x0 - x) * slope

    state = build_case_state(
        tank,
        scaling,
        fill=fill,
        surface=surface,
        initial_velocity=-velocity,   # liquid moves towards the left wall
        device=device,
    )
    n_steps = max(int(run_time_s / scaling.dt), 20)
    t0 = time.time()
    out = march(state, n_steps, record_every=record_every, device=device)
    wall = time.time() - t0
    rec = _impact_metrics(out, scaling, h)
    rho = tank.lng.density
    v_meas = rec["scalars"]["v_impact_ms"] or velocity
    anchors = impact.impact_pressure_anchors(
        v_impact=v_meas,
        beta_deg=gamma_deg,
        rho=rho,
        c_sound=impact.LNG_SOUND_SPEED,
    )
    p_mlm = (
        rho * v_meas**2 * (0.5 * math.pi / math.tan(math.radians(gamma_deg)) - 0.5)
    )
    p_mlm = max(p_mlm, 0.5 * rho * v_meas**2)
    p_wh = impact.water_hammer_pressure(rho, impact.LNG_SOUND_SPEED, v_meas)
    p_lbm = rec["scalars"]["p_dyn_excess_pa"]
    record = {
        "kind": "wedge",
        "fill": fill,
        "depth_m": h,
        "gamma_deg": gamma_deg,
        "cells_per_depth": h / scaling.dx,
        "v_prescribed_ms": velocity,
        **rec["scalars"],
        "p_mlm_pa": p_mlm,
        "p_water_hammer_pa": p_wh,
        "p_stagnation_pa": 0.5 * rho * v_meas**2,
        "mlm_error_pct": 100.0 * (p_lbm / p_mlm - 1.0),
        "water_hammer_error_pct": 100.0 * (p_lbm / p_wh - 1.0),
        "anchors_dimensionless": anchors,
        "energy_budget": out["energy_budget"],
        "inventory_drift": float(
            (out["inventory"][-1] - out["inventory"][0]) / out["inventory"][0]
        ),
        "similarity": similarity_report(tank, scaling, fill=fill),
        "n_steps": out["n_steps"],
        "wall_seconds": wall,
    }
    _save(outdir, f"wedge_f{fill:.2f}_g{gamma_deg:.0f}", record, out)
    return record


# ---------------------------------------------------------------------------
# persistence and campaign driver
# ---------------------------------------------------------------------------


def _save(outdir: Path, name: str, record: dict, arrays: dict) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{name}.json").write_text(json.dumps(record, indent=2, default=_json_default))
    np.savez_compressed(
        outdir / f"{name}.npz",
        t=arrays["t"],
        pressure=arrays["pressure"],
        eta=arrays["eta"],
        modal1=arrays["modal1"],
        v_wall=arrays["v_wall"],
        ke=arrays["ke"],
        pe=arrays["pe"],
        dissipation_rate=arrays["dissipation_rate"],
        work_rate=arrays["work_rate"],
    )


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


#: Case matrix of the campaign (index -> case).  ``cells_per_depth`` is the
#: number of lattice cells across the still-water depth; the GCI family at the
#: 30 % fill uses the 16/24/36 family (refinement ratio r = 1.5).
CASE_MATRIX: list[dict] = (
    # free decay (resonance anchor) at the three fills + GCI family at 30 %
    [{"kind": "freq", "fill": f, "cells_per_depth": 16} for f in DEFAULT_FILLS]
    + [
        {"kind": "freq", "fill": 0.30, "cells_per_depth": 24},
        {"kind": "freq", "fill": 0.30, "cells_per_depth": 36},
    ]
    # forced resonance (impact load history + fatigue) at the three fills
    + [{"kind": "impact", "fill": f, "cells_per_depth": 16} for f in DEFAULT_FILLS]
    # controlled Wagner/MLM wedge benchmarks
    + [
        {"kind": "wedge", "fill": 0.30, "gamma": g, "velocity": 1.0, "cells_per_depth": 24}
        for g in (30.0, 45.0, 60.0)
    ]
    + [
        {"kind": "wedge", "fill": f, "gamma": 45.0, "velocity": 1.0, "cells_per_depth": 24}
        for f in DEFAULT_FILLS
    ]
)


def _case_tag(case: dict) -> str:
    if case["kind"] == "wedge":
        return f"wedge_f{case['fill']:.2f}_g{case['gamma']:.0f}"
    return f"{case['kind']}_f{case['fill']:.2f}_d{case['cells_per_depth']}"


def run_case(case: dict, outdir: Path, *, device: str = "cpu") -> dict:
    kind = case["kind"]
    outdir = Path(outdir)
    existing = outdir / f"{_case_tag(case)}.json"
    if existing.exists():
        return json.loads(existing.read_text())
    if kind == "freq":
        return run_frequency_case(
            fill=case["fill"], cells_per_depth=case["cells_per_depth"], outdir=outdir, device=device
        )
    if kind == "impact":
        return run_impact_case(
            fill=case["fill"], cells_per_depth=case["cells_per_depth"], outdir=outdir, device=device
        )
    if kind == "wedge":
        return run_wedge_case(
            fill=case["fill"],
            gamma_deg=case["gamma"],
            velocity=case["velocity"],
            cells_per_depth=case.get("cells_per_depth", 24),
            outdir=outdir,
            device=device,
        )
    raise ValueError(f"unknown case kind {kind}")


def summarize(outdir: Path) -> dict:
    """Read every case JSON, apply the acceptance judges, return the report dict."""
    outdir = Path(outdir)
    records = [json.loads(p.read_text()) for p in sorted(outdir.glob("*.json"))]
    freq = [r for r in records if r.get("kind") == "frequency"]
    imp = [r for r in records if r.get("kind") == "impact"]
    wedge = [r for r in records if r.get("kind") == "wedge"]

    def pick(rows, fill, depth=None):
        for r in rows:
            if abs(r["fill"] - fill) < 1e-9 and (
                depth is None or abs(r["cells_per_depth"] - depth) < 1e-6
            ):
                return r
        return None

    judges: dict[str, dict] = {}
    # (1) frequency vs the dispersion anchor
    freq_rows = [
        {
            "fill": r["fill"],
            "cells_per_depth": r["cells_per_depth"],
            "f_anchor": r["f1_anchor_hz"],
            "f_lbm": r["f1_lbm_hz"],
            "error_pct": r["f1_error_pct"],
        }
        for r in freq
    ]
    worst = max(freq_rows, key=lambda r: abs(r["error_pct"])) if freq_rows else None
    judges["frequency"] = {
        "threshold_pct": 5.0,
        "rows": freq_rows,
        "worst": worst,
        "pass": bool(worst and abs(worst["error_pct"]) <= 5.0),
    }
    # (2) impact peak vs MLM / water hammer
    impact_rows = [
        {
            "fill": r["fill"],
            "cells_per_depth": r["cells_per_depth"],
            "p_max_pa": r["p_max_pa"],
            "p_dyn_excess_pa": r["p_dyn_excess_pa"],
            "v_impact_ms": r["v_impact_ms"],
            "beta_eff_deg": r["beta_eff_deg"],
            "p_star_lbm": r.get("p_star_lbm"),
        }
        for r in imp
    ]
    wedge_rows = [
        {
            "fill": r["fill"],
            "gamma_deg": r["gamma_deg"],
            "v_impact_ms": r["v_impact_ms"],
            "p_lbm_pa": r["p_dyn_excess_pa"],
            "p_mlm_pa": r["p_mlm_pa"],
            "p_water_hammer_pa": r["p_water_hammer_pa"],
            "mlm_error_pct": r["mlm_error_pct"],
            "water_hammer_error_pct": r["water_hammer_error_pct"],
        }
        for r in wedge
    ]
    design_wedge = [r for r in wedge_rows if abs(r["gamma_deg"] - 45.0) < 1e-6]
    worst_wedge = (
        max(design_wedge, key=lambda r: abs(r["mlm_error_pct"])) if design_wedge else None
    )
    judges["impact_mlm"] = {
        "threshold_pct": 15.0,
        "rows": wedge_rows,
        "design_rows": design_wedge,
        "worst": worst_wedge,
        "pass": bool(worst_wedge and abs(worst_wedge["mlm_error_pct"]) <= 15.0),
        "sloshing_rows": impact_rows,
    }
    # (3) energy budget
    budgets = [
        {
            "case": r.get("kind"),
            "fill": r["fill"],
            "residual_pct": 100.0 * r["energy_budget"]["residual_relative"],
            "inventory_drift": r.get("inventory_drift"),
        }
        for r in records
        if "energy_budget" in r
    ]
    worst_budget = max(budgets, key=lambda b: abs(b["residual_pct"])) if budgets else None
    judges["energy"] = {
        "threshold_pct": 3.0,
        "rows": budgets,
        "worst": worst_budget,
        "pass": bool(worst_budget and abs(worst_budget["residual_pct"]) <= 3.0),
    }
    # (4) Dirlik vs rainflow
    fatigue_rows = [
        {
            "fill": r["fill"],
            "damage_dirlik": r["fatigue"]["damage_dirlik"],
            "damage_rainflow": r["fatigue"]["damage_rainflow"],
            "error_pct": r["fatigue"]["damage_error_pct"],
            "bandwidth_eps": r["fatigue"]["bandwidth_eps"],
        }
        for r in imp
    ]
    worst_fatigue = (
        max(fatigue_rows, key=lambda r: abs(r["error_pct"])) if fatigue_rows else None
    )
    judges["fatigue"] = {
        "threshold_pct": 15.0,
        "rows": fatigue_rows,
        "worst": worst_fatigue,
        "pass": bool(worst_fatigue and abs(worst_fatigue["error_pct"]) <= 15.0),
    }
    # (5) GCI on the fill-0.30 family
    gci_out = {}
    for label, rows, key in (
        ("frequency", freq, "f1_lbm_hz"),
        ("impact_pmax", imp, "p_dyn_excess_pa"),
    ):
        family = sorted(
            [r for r in rows if abs(r["fill"] - 0.30) < 1e-9],
            key=lambda r: r["cells_per_depth"],
        )
        if len(family) == 3:
            vals = [r[key] for r in family]
            r_ratio = (family[2]["cells_per_depth"] / family[1]["cells_per_depth"])
            gci_out[label] = {
                "values": vals,
                "cells_per_depth": [f["cells_per_depth"] for f in family],
                "ratio": r_ratio,
                **audit.gci_family(vals, r_ratio),
            }
    gci_pass = all(
        abs(v["gci_fine_pct"]) <= 5.0 for v in gci_out.values()
    ) if gci_out else False
    judges["gci"] = {"threshold_pct": 5.0, "families": gci_out, "pass": bool(gci_pass)}

    report = {
        "node": "P142W3-N01",
        "tank": asdict(MembraneTank174k(lng=LNGProperties())),
        "cases": len(records),
        "judges": judges,
        "all_pass": all(j["pass"] for j in judges.values()),
        "records": records,
    }
    (outdir / "campaign_summary.json").write_text(
        json.dumps(report, indent=2, default=_json_default)
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P142W3-N01 sloshing campaign")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--case", required=True, help="index into CASE_MATRIX or tag")
    run.add_argument("--outdir", default="runs/p142w3_n01")
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--device", default="cpu")
    summ = sub.add_parser("summarize")
    summ.add_argument("--outdir", default="runs/p142w3_n01")
    args = parser.parse_args(argv)
    torch.set_num_threads(max(1, args.threads))
    if args.cmd == "run":
        try:
            index = int(args.case)
            case = CASE_MATRIX[index]
        except ValueError:
            case = next(c for c in CASE_MATRIX if _case_tag(c) == args.case)
        rec = run_case(case, Path(args.outdir), device=args.device)
        print(json.dumps({k: v for k, v in rec.items() if k != "fatigue"}, default=_json_default))
        return 0
    report = summarize(Path(args.outdir))
    print(json.dumps({k: v for k, v in report["judges"].items()}, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())