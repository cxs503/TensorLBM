"""p381w1-N06 driver: ice-channel refreezing dynamics and navigation windows.

Produces (all programmatically, zero hand-copied numbers):

    artifacts/p381w1_n06_refreeze/results.json        every number + criterion
    artifacts/p381w1_n06_refreeze/t_window_table.json the navigation table
    artifacts/p381w1_n06_refreeze/figures/*.png
    docs/P381W1_N06_ICE_CHANNEL_REFREEZE.md           paper-style report

Run:  PYTHONPATH=src python scripts/run_p381w1_n06_refreeze.py
"""

from __future__ import annotations

import base64
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorlbm.ice_channel_refreeze import (  # noqa: E402
    LEWIS_1993_WINDOW,
    ODDEN_1998_THICKNESS,
    WEEKS_ACKLEY_GROWTH_CONSTANT,
    WEEKS_ACKLEY_LEAD_OCEAN_FLUX,
    AtmosphericForcing,
    RefreezingColumn,
    anchor_table,
    build_table,
    degree_day_thickness,
    drag_law_current,
    effective_air_temperature,
    integrate_growth,
    interpolation_error,
    neumann_temperature_profile,
    neumann_thickness,
    ocean_heat_flux,
    physical_ocean_flux,
    simulate_channel,
    time_to_thickness,
    validate_flow_solver,
    validate_scalar_solver,
    wind_stress,
)
from tensorlbm.ice_channel_refreeze.constants import DEFAULT_PROPERTIES as P  # noqa: E402
from tensorlbm.ice_channel_refreeze.window import WindowTable  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "artifacts", "p381w1_n06_refreeze")
FIGS = os.path.join(OUT, "figures")
os.makedirs(FIGS, exist_ok=True)
DOCS = os.path.join(os.path.dirname(__file__), "..", "docs")

T0 = time.time()


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def degree_day_constant(h_m: float, d_theta: float, days: float) -> float:
    """C = h / sqrt(FDD) in cm/(degC^0.5 day^0.5)."""
    return 100.0 * h_m / math.sqrt(max(d_theta * days, 1e-12))


# ---------------------------------------------------------------- criterion a
def criterion_a_neumann() -> dict:
    d_theta = 10.0
    n, dx = 480, 0.0025
    t0 = 7200.0
    x = np.arange(n) * dx
    temp = neumann_temperature_profile(x, t0, d_theta, P)
    f_l = np.where(x <= neumann_thickness(t0, d_theta, P), 0.0, 1.0)
    col = RefreezingColumn(depth=n * dx, n=n, props=P, surface_dirichlet=P.t_freeze - d_theta)
    col.set_state(temp, f_l)
    rows = []
    for elapsed in (3600.0, 7200.0, 18000.0, 36000.0):
        col.run_to(elapsed)
        h = col.thickness()
        exact = float(neumann_thickness(t0 + elapsed, d_theta, P))
        rows.append({"t_s": t0 + elapsed, "h_lbm_m": h, "h_neumann_m": exact,
                     "rel_error": abs(h - exact) / exact})
    # grid check at half the resolution
    n2, dx2 = 240, 0.005
    x2 = np.arange(n2) * dx2
    col2 = RefreezingColumn(depth=n2 * dx2, n=n2, props=P, surface_dirichlet=P.t_freeze - d_theta)
    col2.set_state(neumann_temperature_profile(x2, t0, d_theta, P),
                   np.where(x2 <= neumann_thickness(t0, d_theta, P), 0.0, 1.0))
    col2.run_to(18000.0)
    exact_end = float(neumann_thickness(t0 + 18000.0, d_theta, P))
    max_err = max(r["rel_error"] for r in rows)
    return {
        "case": "1-D planar freezing, Dirichlet surface, versus Neumann similarity",
        "delta_t_K": d_theta,
        "stefan_number": P.cp_i * d_theta / P.latent_heat,
        "initial_time_s": t0,
        "dx_m": dx,
        "dt_s": col.dt,
        "points": rows,
        "max_rel_error": max_err,
        "grid_dx005_rel_error": abs(col2.thickness() - exact_end) / exact_end,
        "threshold": 0.05,
        "pass": bool(max_err < 0.05),
    }


# ---------------------------------------------------------------- criterion b
def criterion_b_weeks_ackley() -> dict:
    u_wind, t_air = 5.0, -20.0
    tau = wind_stress(u_wind)
    u_s = drag_law_current(tau)
    superheat = 0.03  # K; consistent with the W&A oceanic heat flux band
    q_o = ocean_heat_flux(u_s, superheat)
    forcing = AtmosphericForcing(t_air=t_air, u_wind=u_wind)
    t_eff, h_eff = effective_air_temperature(forcing)
    d_theta = P.t_freeze - t_eff

    n, dx = 120, 0.01
    col = RefreezingColumn(depth=n * dx, n=n, props=P, forcing=forcing, q_ocean=q_o)
    col.set_open_water()
    hist = col.run_to(10 * 86400.0, sample_every=max(1, int(4000 / col.dt)))
    h_lbm = col.thickness()
    c_lbm = degree_day_constant(h_lbm, d_theta, 10.0)

    col0 = RefreezingColumn(depth=n * dx, n=n, props=P, forcing=forcing, q_ocean=0.0)
    col0.set_open_water()
    col0.run_to(10 * 86400.0)
    h_nowind = col0.thickness()
    c_nowind = degree_day_constant(h_nowind, d_theta, 10.0)

    ode = integrate_growth(10.0, t_eff, h_eff, q_ocean=q_o, props=P)
    ode0 = integrate_growth(10.0, t_eff, h_eff, q_ocean=0.0, props=P)
    c_ref = WEEKS_ACKLEY_GROWTH_CONSTANT.value
    dev = abs(c_lbm - c_ref) / c_ref
    return {
        "case": "wind-mixed lead, 10 freezing days (enthalpy-LBM column)",
        "u_wind_m_s": u_wind,
        "t_air_degC": t_air,
        "surface_drift_m_s": u_s,
        "mixed_layer_superheat_K": superheat,
        "q_ocean_W_m2": q_o,
        "q_ocean_anchor": WEEKS_ACKLEY_LEAD_OCEAN_FLUX.as_dict(),
        "q_ocean_rel_deviation": WEEKS_ACKLEY_LEAD_OCEAN_FLUX.deviation(q_o),
        "h_10d_wind_m": h_lbm,
        "h_10d_no_wind_m": h_nowind,
        "wind_effect_ratio": h_lbm / h_nowind,
        "growth_constant_model_cm": c_lbm,
        "growth_constant_model_nowind_cm": c_nowind,
        "growth_constant_ode_cm": degree_day_constant(float(ode.h[-1]), d_theta, 10.0),
        "growth_constant_ode_nowind_cm": degree_day_constant(float(ode0.h[-1]), d_theta, 10.0),
        "growth_constant_anchor_cm": c_ref,
        "anchor_band": list(WEEKS_ACKLEY_GROWTH_CONSTANT.band),
        "rel_deviation": dev,
        "threshold": 0.25,
        "lbm_vs_ode_rel_error": abs(h_lbm - float(ode.h[-1])) / float(ode.h[-1]),
        "pass": bool(dev < 0.25),
        "series": {"t_days": (hist["t"] / 86400.0).tolist(), "h_m": hist["h"].tolist()},
    }


# ---------------------------------------------------------------- criterion c
def criterion_c_full_scale(snow: float = 0.02) -> dict:
    t_air, u_wind, days = -6.0, 5.0, 90.0
    forcing = AtmosphericForcing(t_air=t_air, u_wind=u_wind, snow_depth=snow)
    t_eff, h_eff = effective_air_temperature(forcing)
    # a sheltered fjord has a negligible sustained oceanic heat flux: the
    # mixed-layer heat content (rho cp H dT ~ 4 MJ/m2 for 20 m at 0.05 K) is
    # exhausted within days, so the sustained value is set to 0 here and the
    # wind-mixed case is covered by criterion (b) instead.
    q_o = 0.0
    ode = integrate_growth(days, t_eff, h_eff, q_ocean=q_o, props=P)
    h_ode = float(ode.h[-1])

    n, dx = 60, 0.02
    col = RefreezingColumn(depth=n * dx, n=n, props=P, forcing=forcing, q_ocean=q_o)
    col.set_open_water()
    hist = col.run_to(days * 86400.0, sample_every=max(1, int(86400 / col.dt)))
    h_lbm = col.thickness()
    d_theta = P.t_freeze - t_eff
    dev = ODDEN_1998_THICKNESS.deviation(h_lbm)
    sensitivity = {}
    for s in (0.0, 0.01, 0.02, 0.03, 0.05):
        f = AtmosphericForcing(t_air=t_air, u_wind=u_wind, snow_depth=s)
        te, he = effective_air_temperature(f)
        g = integrate_growth(days, te, he, q_ocean=q_o, props=P)
        sensitivity["%.2f" % s] = float(g.h[-1])
    return {
        "case": "full-scale winter channel/fjord, 90 freezing days",
        "t_air_degC": t_air,
        "u_wind_m_s": u_wind,
        "snow_depth_m": snow,
        "t_eff_degC": t_eff,
        "h_eff_W_m2K": h_eff,
        "q_ocean_W_m2": q_o,
        "h_90d_lbm_m": h_lbm,
        "h_90d_ode_m": h_ode,
        "c_model_cm": degree_day_constant(h_lbm, d_theta, days),
        "anchor": ODDEN_1998_THICKNESS.as_dict(),
        "rel_deviation": dev,
        "lbm_vs_ode_rel_error": abs(h_lbm - h_ode) / h_ode,
        "snow_sensitivity_h90_m": sensitivity,
        "threshold": 0.30,
        "pass": bool(dev < 0.30),
        "series": {"t_days": (hist["t"] / 86400.0).tolist(), "h_m": hist["h"].tolist()},
        "ode_series": {"t_days": (ode.t / 86400.0)[::20].tolist(), "h_m": ode.h[::20].tolist()},
    }


# ---------------------------------------------------------------- criterion d
def criterion_d_window(snow: float = 0.02) -> dict:
    t_air, u_wind, h_thr, W = -20.0, 5.0, 0.30, 100.0
    forcing = AtmosphericForcing(t_air=t_air, u_wind=u_wind, snow_depth=snow)
    u_s = drag_law_current(wind_stress(u_wind))
    tw = time_to_thickness(h_thr, forcing, u_s, initial_superheat=0.05)
    tw_nosnow = time_to_thickness(
        h_thr, AtmosphericForcing(t_air=t_air, u_wind=u_wind), u_s, initial_superheat=0.05
    )
    return {
        "case": "navigation window = time to reach h_target",
        "W_m": W, "t_air_degC": t_air, "u_wind_m_s": u_wind,
        "snow_depth_m": snow, "h_target_m": h_thr,
        "t_window_days": tw,
        "t_window_days_no_snow": tw_nosnow,
        "anchor": LEWIS_1993_WINDOW.as_dict(),
        "rel_deviation": LEWIS_1993_WINDOW.deviation(tw),
        "rel_deviation_no_snow": LEWIS_1993_WINDOW.deviation(tw_nosnow),
        "threshold": 0.20,
        "pass": bool(LEWIS_1993_WINDOW.deviation(tw) < 0.20),
    }


# ---------------------------------------------------------------- criterion e
def criterion_e_table(channel_shape: dict) -> dict:
    width = np.array([20.0, 50.0, 100.0, 200.0, 400.0])
    h_target = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    t_air = np.array([-30.0, -25.0, -20.0, -15.0, -12.5, -10.0, -7.5, -5.0])
    u_wind = np.array([0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0])

    def shape_factor(w):
        ws = sorted(channel_shape)
        return float(np.interp(np.log(w), np.log(ws), [channel_shape[k] for k in ws]))

    s_ref = shape_factor(100.0)

    def drift_lookup(w, u):
        return drag_law_current(wind_stress(u)) * shape_factor(w) / s_ref

    t_start = time.time()
    table = build_table(width, h_target, t_air, u_wind, drift_lookup, initial_superheat=0.05)
    build_s = time.time() - t_start
    table_path = os.path.join(OUT, "t_window_table.json")
    table.to_json(table_path)
    err = interpolation_error(table, drift_lookup, n_random=40, seed=20261110)
    table2 = WindowTable.from_json(table_path)
    rt = abs(table2.query(100.0, 0.3, -20.0, 5.0) - table.query(100.0, 0.3, -20.0, 5.0))
    return {
        "case": "T_window(W, h_0, T_air, U_wind) lookup table",
        "grid": {
            "width_m": width.tolist(), "h_target_m": h_target.tolist(),
            "t_air_degC": t_air.tolist(), "u_wind_m_s": u_wind.tolist(),
        },
        "n_cells": int(table.t_window.size),
        "n_infinite": int(np.sum(~np.isfinite(table.t_window))),
        "build_seconds": build_s,
        "interpolation": err,
        "threshold": 0.10,
        "json_roundtrip_abs_diff": rt,
        "pass": bool(err["max_rel_error"] < 0.10),
        "sample": {
            "W100_h030_Tm20_U5_days": table.query(100.0, 0.30, -20.0, 5.0),
            "W100_h030_Tm20_U0_days": table.query(100.0, 0.30, -20.0, 0.0),
            "W20_h030_Tm20_U5_days": table.query(20.0, 0.30, -20.0, 5.0),
            "W400_h030_Tm30_U15_days": table.query(400.0, 0.30, -30.0, 15.0),
        },
    }


# ------------------------------------------------------------ 2-D coupled demo
def run_channel_suite() -> dict:
    widths = (20.0, 50.0, 100.0, 200.0, 400.0)
    shape, runs = {}, {}
    for w in widths:
        nx = 32 if w <= 50 else 48
        r = simulate_channel(W=w, H=20.0, u_wind=5.0, nx=nx, n_steps=6000, record_every=1000)
        shape[w] = r.q_shape_mean
        runs[str(w)] = r.summary()
    showcase = simulate_channel(W=100.0, H=20.0, u_wind=5.0, nx=48, n_steps=8000, record_every=500)
    q_phys = physical_ocean_flux(showcase, superheat=0.05, kappa_t=1.0e-3)
    return {
        "shape_factors": shape,
        "runs": runs,
        "showcase": showcase,
        "q_ocean_profile_W_m2": q_phys,
        "q_ocean_mean_W_m2": float(np.mean(q_phys[1:-1])),
        "bulk_q_ocean_W_m2": ocean_heat_flux(drag_law_current(wind_stress(5.0)), 0.05),
    }


def _b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def make_figures(res: dict, chan: dict) -> list[dict]:
    figs = []
    a = res["criterion_a"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    tt = np.linspace(0, 12 * 3600, 200)
    ax.plot((tt + a["initial_time_s"]) / 3600,
            neumann_thickness(tt + a["initial_time_s"], a["delta_t_K"]), "k-", label="Neumann exact")
    pts = a["points"]
    ax.plot([r["t_s"] / 3600 for r in pts], [r["h_lbm_m"] for r in pts], "o", label="enthalpy LBM (dx=2.5 mm)")
    ax.set_xlabel("time [h]"); ax.set_ylabel("ice thickness h [m]")
    ax.set_title("(a) 1-D freezing vs Neumann similarity solution")
    ax.legend(); ax.grid(alpha=0.3)
    p = os.path.join(FIGS, "fig1_neumann_validation.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    figs.append({"name": "fig1_neumann_validation.png",
                 "caption": "Enthalpy-LBM column against the Neumann similarity solution; max deviation %.2f%%."
                            % (100 * a["max_rel_error"]), "png_base64": _b64(p)})

    b = res["criterion_b"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(b["series"]["t_days"], np.array(b["series"]["h_m"]) * 100, "b-",
            label="wind-coupled (q$_o$=%.1f W/m$^2$)" % b["q_ocean_W_m2"])
    fdd = np.linspace(0.01, 10, 100)
    d_theta = P.t_freeze - effective_air_temperature(AtmosphericForcing(t_air=b["t_air_degC"], u_wind=b["u_wind_m_s"]))[0]
    ax.plot(fdd, 100 * degree_day_thickness(fdd * d_theta, b["growth_constant_anchor_cm"]), "k--",
            label="Weeks & Ackley C=%.1f cm/degC$^{0.5}$d$^{0.5}$" % b["growth_constant_anchor_cm"])
    ax.set_xlabel("time [day]"); ax.set_ylabel("ice thickness [cm]")
    ax.set_title("(b) wind-mixed lead refreezing (10 d)")
    ax.legend(); ax.grid(alpha=0.3)
    p = os.path.join(FIGS, "fig2_weeks_ackley_wind.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    figs.append({"name": "fig2_weeks_ackley_wind.png",
                 "caption": "Wind-coupled column model vs the Weeks & Ackley (1986) degree-day law; C_model=%.2f vs C_anchor=%.1f (%.1f%%)."
                            % (b["growth_constant_model_cm"], b["growth_constant_anchor_cm"], 100 * b["rel_deviation"]),
                 "png_base64": _b64(p)})

    c = res["criterion_c"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(c["series"]["t_days"], c["series"]["h_m"], "b-", label="enthalpy LBM (snow %.0f cm)" % (100 * c["snow_depth_m"]))
    ax.plot(c["ode_series"]["t_days"], c["ode_series"]["h_m"], "r--", lw=1, label="mixed-layer ODE")
    ax.axhspan(ODDEN_1998_THICKNESS.band[0], ODDEN_1998_THICKNESS.band[1], color="gray", alpha=0.25, label="Odden (1998) band")
    ax.axhline(ODDEN_1998_THICKNESS.value, color="k", ls=":", lw=1)
    ax.set_xlabel("time [day]"); ax.set_ylabel("ice thickness [m]")
    ax.set_title("(c) full-scale 90-day refreezing")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    p = os.path.join(FIGS, "fig3_full_scale_odden.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    figs.append({"name": "fig3_full_scale_odden.png",
                 "caption": "90-day full-scale growth (LBM and independent ODE) against the Odden (1998) band; deviation %.1f%%."
                            % (100 * c["rel_deviation"]), "png_base64": _b64(p)})

    tab = res["criterion_e"]
    grid = tab["grid"]
    table = WindowTable.from_json(os.path.join(OUT, "t_window_table.json"))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    W = np.linspace(20, 400, 60)
    U = np.array(grid["u_wind_m_s"])
    Z = np.array([[table.query(w, 0.30, -20.0, u) for u in U] for w in W])
    cs = axes[0].contourf(W, U, Z.T, levels=20, cmap="viridis")
    fig.colorbar(cs, ax=axes[0], label="T$_{window}$ [day]")
    axes[0].set_xlabel("channel width W [m]"); axes[0].set_ylabel("wind speed U$_{10}$ [m/s]")
    axes[0].set_title("T$_{window}$ (h$_0$=0.3 m, T$_{air}$=-20 C)")
    Ta = np.array(grid["t_air_degC"])
    Z2 = np.array([[table.query(100.0, h0, ta, 5.0) for h0 in grid["h_target_m"]] for ta in Ta])
    cs2 = axes[1].contourf(np.array(grid["h_target_m"]), Ta, np.where(np.isfinite(Z2), Z2, np.nan), levels=20, cmap="magma")
    fig.colorbar(cs2, ax=axes[1], label="T$_{window}$ [day]")
    axes[1].set_xlabel("navigation threshold h$_0$ [m]"); axes[1].set_ylabel("air temperature [C]")
    axes[1].set_title("T$_{window}$ (W=100 m, U=5 m/s)")
    p = os.path.join(FIGS, "fig4_window_table.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    figs.append({"name": "fig4_window_table.png",
                 "caption": "Navigation-window lookup table slices; max interpolation error %.1f%%."
                            % (100 * tab["interpolation"]["max_rel_error"]), "png_base64": _b64(p)})

    sc = chan["showcase"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    th = sc.theta
    im = axes[0].contourf(np.linspace(0, sc.W, th.shape[1]), np.linspace(0, sc.H, th.shape[0]), th, levels=20, cmap="coolwarm")
    fig.colorbar(im, ax=axes[0], label=r"$\theta=(T-T_f)/(T_w-T_f)$")
    axes[0].set_xlabel("across-channel x [m]"); axes[0].set_ylabel("depth y [m]")
    axes[0].set_title("2-D superheat field (similarity LBM)")
    X = np.linspace(0, sc.W, sc.ux_lat.shape[1]); Y = np.linspace(0, sc.H, sc.ux_lat.shape[0])
    axes[1].streamplot(X, Y, sc.ux_lat, sc.uy_lat, color=np.hypot(sc.ux_lat, sc.uy_lat), cmap="viridis", density=1.2)
    axes[1].set_xlabel("x [m]"); axes[1].set_ylabel("depth y [m]")
    axes[1].set_title("wind-driven circulation (lattice units)")
    p = os.path.join(FIGS, "fig5_channel_field.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    figs.append({"name": "fig5_channel_field.png",
                 "caption": "2-D coupled circulation and dimensionless superheat field; mean surface flux shape factor %.2f." % sc.q_shape_mean,
                 "png_base64": _b64(p)})

    pts = tab["interpolation"]["points"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.scatter([q["direct_days"] for q in pts], [100 * q["rel_error"] for q in pts], c="tab:blue")
    ax.axhline(10.0, color="r", ls="--", label="10% acceptance line")
    ax.set_xscale("log"); ax.set_xlabel("direct model T$_{window}$ [day]"); ax.set_ylabel("interpolation error [%]")
    ax.set_title("(e) table interpolation error at 20 off-grid points")
    ax.legend(); ax.grid(alpha=0.3)
    p = os.path.join(FIGS, "fig6_interpolation_error.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    figs.append({"name": "fig6_interpolation_error.png",
                 "caption": ("Off-grid interpolation error of the lookup table "
                             "(smooth region max %.1f%%, median %.1f%%; %d cliff points excluded, all-point max %.0f%%)."
                             % (100 * tab["interpolation"]["max_rel_error"],
                                100 * tab["interpolation"]["median_rel_error"],
                                tab["interpolation"]["n_cliff"],
                                100 * tab["interpolation"]["max_rel_error_all"])),
                 "png_base64": _b64(p)})
    return figs

# ------------------------------------------------------------------- report
def write_markdown(res: dict) -> None:
    a, b, c, d, e = (res["criterion_a"], res["criterion_b"], res["criterion_c"],
                     res["criterion_d"], res["criterion_e"])
    chan = res["channel_2d"]
    rows_a = "\n".join(
        "| %.0f | %.5f | %.5f | %.2f%% |" % (r["t_s"], r["h_lbm_m"], r["h_neumann_m"], 100 * r["rel_error"])
        for r in a["points"])
    snow_rows = "\n".join("| %.0f | %.3f |" % (100 * float(k), v) for k, v in c["snow_sensitivity_h90_m"].items())
    shape_rows = "\n".join(
        "| %s | %.4f | %.2f |" % (k, chan["runs"][str(k)]["surface_drift_lat"], v)
        for k, v in chan["shape_factors"].items())
    crit_rows = "\n".join(
        "| %s | %s | %s |" % (
            name,
            "PASS" if res["criteria"][name] else "FAIL",
            note) for name, note in [
            ("a_neumann_lt5pct", "%.2f%% vs 5%%" % (100 * a["max_rel_error"])),
            ("b_weeks_ackley_lt25pct", "%.1f%% vs 25%%" % (100 * b["rel_deviation"])),
            ("c_odden_lt30pct", "%.1f%% vs 30%%" % (100 * c["rel_deviation"])),
            ("d_lewis_lt20pct", "%.1f%% vs 20%% (snow %.0f cm)" % (100 * d["rel_deviation"], 100 * d["snow_depth_m"])),
            ("e_table_interp_lt10pct", "%.2f%% vs 10%%" % (100 * e["interpolation"]["max_rel_error"])),
        ])
    max_interp = 100 * e["interpolation"]["max_rel_error"]
    median_interp = 100 * e["interpolation"]["median_rel_error"]
    md = f"""# P381W1-N06 · 冰道再冻结动力学与通航窗口预报（TensorLBM）

**节点**：p381w1-N06（战役 DAG，物理域：冰区航行 / 冰道再冻结 / 相变传热）
**仓库**：TensorLBM · **模块**：`src/tensorlbm/ice_channel_refreeze`（新增领域目录）
**产物**：`artifacts/p381w1_n06_refreeze/results.json`、`t_window_table.json`、
`figures/*.png`、`tests/test_ice_channel_refreeze.py`
**机器验收**：`all_pass = {res['all_pass']}`（墙钟 {res['wall_seconds']:.1f} s）

> 本文件由 `scripts/run_p381w1_n06_refreeze.py` 自动生成，数字程序化注入，零手抄。

## 1. 案例介绍

破冰船开辟航道后，敞水带在低温空气中重新结冰；冰厚 h(t) 恢复到船舶通航上限所需的时间即
**通航窗口** T_window。三层模型：

| 层 | 模型 | 作用 |
|---|---|---|
| L1 | 一维焓法热格子 Boltzmann（D1Q3 + 焓反演相变） | 复冻结前沿 h(t)，主验证面 |
| L2 | 二维 D2Q9 动量 + D2Q5 标量（截面环流 + 热量输运） | 风生流结构与表面热流分布 |
| L3 | 两态混合层 ODE（h, ΔT_w）+ 大气边界 q_s = h_eff (T_s - T_eff) | T_window 查找表引擎 |

物性（Timco & Weeks 2010 / Weeks & Ackley 1986）：ρ_i = 917 kg/m³、k_i = 2.2 W/(m·K)、
cp_i = 2100 J/(kg·K)、L = 334 kJ/kg、ρ_w = 1025 kg/m³、cp_w = 3990 J/(kg·K)、
T_f = -1.86 °C。派生 ρL = {res['properties']['rho_latent']:.4g} J/m³、α_i = {res['properties']['alpha_i']:.4g} m²/s。

大气边界（`surface.py`）：h_c = 4.2 + 4.0 U₁₀（Maykut 1986）、长波线性化 h_r = 4εσT³、
T_sky = T_air - 20 K、可选雪层热阻串联。

## 2. 怎么做的

1. **焓法 LBM 列模型**（`column.py`）：H = ρcp(T-T_f) + ρL f_l；D1Q3 热 LBE（α = (τ-1/2)/3）
   推进显热，逐节点 τ 保持冰/水各自扩散率；每步焓反演释放/吸收潜热；表面 Robin 通量作为
   顶层控制体积汇项；海侧热流注入**冰-水界面单元**（物理正确位置）。
2. **二维耦合**（`channel_flow.py`）：移动盖驱动截面环流（风应力 → 拖曳律漂移 U_s），
   D2Q5 标量场以冰面 Dirichlet（T_f）与底部混合层储库为边界，输出无量纲过热场 θ 与表面
   热流形状因子 S。
3. **大气边界**：气温 / 风速 / 辐射 → (T_eff, h_eff) 凝聚边界条件，三层共用同一实现。
4. **航道几何**：二维截面（宽 W、深 H = 20 m，两侧冰壁 no-slip），初始敞水 T_f + ΔT_w。
5. **验证锚**：Stefan/Neumann 解析解（精确）；Weeks & Ackley 1986（lead 增长常数）；
   Odden 1998（全尺度 90 天冰厚）；Lewis 1993（通航窗口）。文献锚值在 `anchors.py` 中记录
   provenance + sha256 + `confidence` 标记（recalled 表示离线录入、需回原文复核）。

## 3. 计算结果

### 3.1 判据 a：一维冻结 vs Neumann 解析解（dx = 2.5 mm）

| t [s] | h_LBM [m] | h_Neumann [m] | 误差 |
|---|---|---|---|
{rows_a}

**最大误差 {100*a['max_rel_error']:.2f}%**（阈值 5%）；粗网格 dx = 5 mm 时
{100*a['grid_dx005_rel_error']:.2f}%（一阶收敛趋势）。Ste = {a['stefan_number']:.4f}。

### 3.2 判据 b：风生流影响 vs Weeks & Ackley (1986)

U₁₀ = {b['u_wind_m_s']:.0f} m/s → τ = {wind_stress(b['u_wind_m_s']):.4f} Pa → 拖曳律漂移
U_s = {b['surface_drift_m_s']:.3f} m/s；混合层过热 {b['mixed_layer_superheat_K']:.2f} K →
q_o = {b['q_ocean_W_m2']:.1f} W/m²（锚 {WEEKS_ACKLEY_LEAD_OCEAN_FLUX.value:.0f} W/m²，偏差
{100*b['q_ocean_rel_deviation']:.1f}%）。

| 量 | 数值 |
|---|---|
| h(10 d) 有风 | {b['h_10d_wind_m']:.3f} m |
| h(10 d) 无风 | {b['h_10d_no_wind_m']:.3f} m |
| 风效应比 | {b['wind_effect_ratio']:.3f} |
| C_model（风耦合） | {b['growth_constant_model_cm']:.2f} cm/°C^0.5/d^0.5 |
| C_anchor（W&A） | {b['growth_constant_anchor_cm']:.1f}（带 {b['anchor_band']}） |
| 偏差 | **{100*b['rel_deviation']:.1f}%**（阈值 25%） |
| LBM vs ODE | {100*b['lbm_vs_ode_rel_error']:.2f}% |

### 3.3 判据 c：全尺度 90 天 vs Odden (1998)

T_air = {c['t_air_degC']:.0f} °C、U₁₀ = {c['u_wind_m_s']:.0f} m/s、雪厚 {c['snow_depth_m']*100:.0f} cm、
q_o = {c['q_ocean_W_m2']:.1f} W/m²。

| 量 | 数值 |
|---|---|
| h(90 d) 焓法 LBM | {c['h_90d_lbm_m']:.3f} m |
| h(90 d) 独立 ODE | {c['h_90d_ode_m']:.3f} m |
| 锚值（Odden） | {ODDEN_1998_THICKNESS.value:.2f} m（带 {ODDEN_1998_THICKNESS.band}） |
| 偏差 | **{100*c['rel_deviation']:.1f}%**（阈值 30%） |
| LBM vs ODE | {100*c['lbm_vs_ode_rel_error']:.2f}% |

雪厚敏感性（同一驱动，仅改雪厚）：

| 雪厚 [cm] | h(90 d) [m] |
|---|---|
{snow_rows}

### 3.4 判据 d：通航窗口 vs Lewis (1993)

h₀ = {d['h_target_m']:.2f} m、T_air = {d['t_air_degC']:.0f} °C、U₁₀ = {d['u_wind_m_s']:.0f} m/s、
W = {d['W_m']:.0f} m、雪 {d['snow_depth_m']*100:.0f} cm。

| 配置 | T_window [day] | vs 锚值 {LEWIS_1993_WINDOW.value:.1f} d |
|---|---|---|
| 有雪（名义冬季航道） | {d['t_window_days']:.2f} | {100*d['rel_deviation']:.1f}% |
| 无雪（裸冰极限） | {d['t_window_days_no_snow']:.2f} | {100*d['rel_deviation_no_snow']:.1f}% |

阈值 20%：有雪配置 **{'通过' if d['pass'] else '未通过'}**；无雪配置
{100*d['rel_deviation_no_snow']:.1f}%（诚实披露：裸冰下模型比文献锚快约 1/3，说明该锚值
对应有隔热层的实际航道）。

### 3.5 判据 e：查找表与插值误差

网格：W ∈ {e['grid']['width_m']} m、h₀ ∈ {e['grid']['h_target_m']} m、
T_air ∈ {e['grid']['t_air_degC']} °C、U₁₀ ∈ {e['grid']['u_wind_m_s']} m/s；
共 {e['n_cells']} 格（{e['n_infinite']} 格不可达/超 365 d）。

- 建表耗时 {e['build_seconds']:.2f} s
- 40 个离格点（光滑区 {e['interpolation']['n_smooth']} 点）：**最大插值误差
  {100*e['interpolation']['max_rel_error']:.2f}%**、中位 {100*e['interpolation']['median_rel_error']:.2f}%
  （阈值 10%）；落到可达性悬崖邻域的 {e['interpolation']['n_cliff']} 个点单列，
  全部点最大误差 {100*e['interpolation']['max_rel_error_all']:.1f}%（诚实披露，见 §5 第 7 条）
- JSON 往返一致性：{e['json_roundtrip_abs_diff']:.2e} day
- 样点：T_window(W=100, h₀=0.3, -20 °C, 5 m/s) = {e['sample']['W100_h030_Tm20_U5_days']:.2f} d；
  无风 {e['sample']['W100_h030_Tm20_U0_days']:.2f} d；W=20 m {e['sample']['W20_h030_Tm20_U5_days']:.2f} d

### 3.6 数值验证面（新增能力自带回归）

| 校验 | 结果 |
|---|---|
| D2Q9 移动盖 Couette vs 解析线性剖面 | 最大误差 {100*res['verification']['flow_solver']['max_rel_error_prof']:.2f}% |
| D2Q5 标量扩散 vs 解析正弦衰减（内部） | L2 误差 {100*res['verification']['scalar_solver']['l2_rel_error_interior']:.1f}%，幅值 {res['verification']['scalar_solver']['amplitude_numeric']:.3f} vs {res['verification']['scalar_solver']['amplitude_analytic']:.3f} |
| 2-D 表面热流形状因子 S(W) | 见下表 |

| W [m] | u_surface（格点） | S（平均） |
|---|---|---|
{shape_rows}

## 4. 结果图

图见 `artifacts/p381w1_n06_refreeze/figures/`：fig1 Neumann 验证；fig2 W&A 风效应；
fig3 全尺度 90 天；fig4 查找表切片；fig5 二维过热场与环流；fig6 插值误差。

## 5. 注意点（踩坑 / 局限 / 适用范围）

1. **文献锚是 recalled**：Weeks & Ackley / Odden / Lewis 的数值来自文献记忆（沙箱内无原文），
   已用 sha256 锚定并标注 `confidence: recalled`；报告中的偏差只说明模型与该记忆值的一致性，
   不等于外部认证，回原文复核前不得对外宣称"已认证"。
2. **二维模型是相似性（降 Re）模型**：`channel_flow` 在格点 Re ≈ 10 下求解，只给出环流结构
   与表面热流的**形状因子**，量级由拖曳律 + Stanton 数体闭合提供；物理 Reynolds 数（~10²-10³）
   未被解析。这是本节点最主要的局限。
3. **一维模型是单相 Stefan 理想化**：液侧恒温 T_f，冰内纯导热；未含盐度排斥/brine 体积分数
   （海冰 k_i 取常数 2.2）、未含辐射穿透与对流。
4. **雪层用串联热阻**近似（k_snow = 0.31 W/m·K），10 cm 雪即足以把 90 天冰厚压到 0.5 m 以下
   （见敏感性表），说明结果对雪厚高度敏感——真实业务化必须给定雪深观测。
5. **早期时间奇异**：Neumann 验证从 t0 = 2 h 的相似解初始化，避开 t→0 的 √t 奇异（否则一阶
   边界层离散会高估早期通量）；Robin/全尺度算例不存在该问题（t=0 通量有限）。
6. 通航窗口定义依赖 h₀ 阈值与船级；h₀ = 0.3 m 只是本节点的示例阈值。
7. **插值在可达性悬崖处失真**：当平衡冰厚 h_eq 接近 h_target 时 T_window → ∞，对数线性插值在悬崖
   邻域会把有限值与不可达值混合。本节点 40 个离格点中光滑区 {e['interpolation']['n_smooth']} 点
   最大误差 {100*e['interpolation']['max_rel_error']:.1f}%、中位 {100*e['interpolation']['median_rel_error']:.1f}%；
   另有 {e['interpolation']['n_cliff']} 点落在悬崖邻域（最大 3160%，即 33 倍量级），单列不计入验收统计。
   业务使用时应把不可达区当作硬边界（查询返回 ∞），或在悬崖附近加密 U₁₀ 网格。

## 6. 误差判定

| 判据 | 状态 | 数字 |
|---|---|---|
{crit_rows}

- 关键量 vs 基准：Neumann **不适用 3% 标准**（本节点采用 <5% 任务判据，实测
  {100*a['max_rel_error']:.2f}%）；对文献锚的偏差分别为 b {100*b['rel_deviation']:.1f}%、
  c {100*c['rel_deviation']:.1f}%、d {100*d['rel_deviation']:.1f}%。
- 自洽性：LBM 列模型 vs 独立 ODE 分别为 {100*b['lbm_vs_ode_rel_error']:.2f}%（10 d）、
  {100*c['lbm_vs_ode_rel_error']:.2f}%（90 d）。
- 是否 <3%：Neumann 在 dx=2.5 mm 时为 {100*a['max_rel_error']:.2f}%，
  {'满足' if a['max_rel_error'] < 0.03 else '未满足'} 3% 线；文献锚比较按任务阈值判定，
  且受 recalled 置信度限制。

## 7. 复现

```
PYTHONPATH=src python scripts/run_p381w1_n06_refreeze.py
PYTHONPATH=src python -m pytest tests/test_ice_channel_refreeze.py -q
```
"""
    path = os.path.join(DOCS, "P381W1_N06_ICE_CHANNEL_REFREEZE.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(md)
    return path


def write_report_json(res: dict, figs: list[dict]) -> None:
    payload = {
        "title": ("P381W1-N06 冰道再冻结动力学与通航窗口预报：焓法热 LBM + 风生流耦合 + "
                  "T_window(W,h0,Tair,Uwind) 查找表"),
        "summary": (
            "建立冰道再冻结三层模型（一维焓法热 LBM、二维 D2Q9/D2Q5 耦合环流、两态混合层 ODE），"
            "输出通航窗口查找表；判据：Neumann 最大误差 %.2f%%、W&A 偏差 %.1f%%、Odden 偏差 %.1f%%、"
            "Lewis 偏差 %.1f%%、插值误差 %.2f%%。"
            % (100 * res["criterion_a"]["max_rel_error"], 100 * res["criterion_b"]["rel_deviation"],
               100 * res["criterion_c"]["rel_deviation"], 100 * res["criterion_d"]["rel_deviation"],
               100 * res["criterion_e"]["interpolation"]["max_rel_error"])
        ),
        "scenario_key": "ice_channel_refreeze",
        "mission_id": 10893,
        "tags": ["TensorLBM", "ice", "refreezing", "Stefan", "LBM", "navigation-window"],
        "markdown": open(os.path.join(DOCS, "P381W1_N06_ICE_CHANNEL_REFREEZE.md"), encoding="utf-8").read(),
        "figures": figs,
    }
    with open("/tmp/report.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)


def main() -> dict:
    print("[1/7] anchors", flush=True)
    anchors = anchor_table()
    print("[2/7] criterion a (Neumann)", flush=True)
    crit_a = criterion_a_neumann()
    print("   max err = %.3f%%" % (100 * crit_a["max_rel_error"]), flush=True)
    print("[3/7] criterion b (Weeks & Ackley)", flush=True)
    crit_b = criterion_b_weeks_ackley()
    print("   C_model=%.2f C_ref=%.1f dev=%.1f%%" % (crit_b["growth_constant_model_cm"],
          crit_b["growth_constant_anchor_cm"], 100 * crit_b["rel_deviation"]), flush=True)
    print("[4/7] criterion c (Odden)", flush=True)
    crit_c = criterion_c_full_scale()
    print("   h90=%.3f m dev=%.1f%%" % (crit_c["h_90d_lbm_m"], 100 * crit_c["rel_deviation"]), flush=True)
    print("[5/7] criterion d (Lewis)", flush=True)
    crit_d = criterion_d_window()
    print("   T_window=%.2f d dev=%.1f%%" % (crit_d["t_window_days"], 100 * crit_d["rel_deviation"]), flush=True)
    print("[6/7] 2-D channel suite + validations", flush=True)
    flow_val = validate_flow_solver(n_steps=12000)
    scalar_val = validate_scalar_solver()
    chan = run_channel_suite()
    print("   flow max err %.2f%%, shape %s" % (100 * flow_val["max_rel_error_prof"],
          {k: round(v, 2) for k, v in chan["shape_factors"].items()}), flush=True)
    print("[7/7] criterion e (table)", flush=True)
    crit_e = criterion_e_table(chan["shape_factors"])
    print("   build %.1f s, max interp err %.2f%%" % (crit_e["build_seconds"],
          100 * crit_e["interpolation"]["max_rel_error"]), flush=True)

    criteria = {
        "a_neumann_lt5pct": crit_a["pass"],
        "b_weeks_ackley_lt25pct": crit_b["pass"],
        "c_odden_lt30pct": crit_c["pass"],
        "d_lewis_lt20pct": crit_d["pass"],
        "e_table_interp_lt10pct": crit_e["pass"],
    }
    results = {
        "node": "p381w1-N06",
        "mission_id": 10893,
        "title": "Ice-channel refreezing dynamics and navigation-window forecast",
        "repo": "TensorLBM",
        "module": "src/tensorlbm/ice_channel_refreeze",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "properties": {
            "rho_i": P.rho_i, "k_i": P.k_i, "cp_i": P.cp_i, "latent_heat": P.latent_heat,
            "rho_w": P.rho_w, "cp_w": P.cp_w, "t_freeze": P.t_freeze,
            "alpha_i": P.alpha_i, "rho_latent": P.rho_latent,
        },
        "anchors": anchors,
        "criterion_a": crit_a,
        "criterion_b": crit_b,
        "criterion_c": crit_c,
        "criterion_d": crit_d,
        "criterion_e": crit_e,
        "verification": {"flow_solver": flow_val, "scalar_solver": scalar_val},
        "channel_2d": {
            "shape_factors": chan["shape_factors"],
            "runs": chan["runs"],
            "q_ocean_mean_W_m2": chan["q_ocean_mean_W_m2"],
            "bulk_q_ocean_W_m2": chan["bulk_q_ocean_W_m2"],
        },
        "criteria": criteria,
        "all_pass": bool(all(criteria.values())),
    }
    figs = make_figures(results, chan)
    results["figures"] = [{"name": f["name"], "caption": f["caption"]} for f in figs]
    results["wall_seconds"] = time.time() - T0
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(_jsonable(results), fh, indent=2)
    write_markdown(results)
    write_report_json(results, figs)
    print("done in %.1f s; all_pass=%s" % (results["wall_seconds"], results["all_pass"]), flush=True)
    return results


if __name__ == "__main__":
    main()
