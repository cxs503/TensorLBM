"""P21-N05 air-lubrication campaign driver.

Produces every number quoted in ``docs/AIR_LUBRICATION_P21_N05.md``.  Nothing is
typed by hand: the report is generated from ``artifacts/p21_n05/*.json``.

Stages
------
``transport``  bubble-curtain void-fraction transport -> ``alpha(y)``, ``delta_b``,
               species-mass audit (criteria C2, C3-mass).
``re_sweep``   15-case carrier-flow Re sweep with the air layer -> ``Delta C_f``,
               ``Re_theta``, momentum audit (C1, C3-momentum, C7).
``curvature``  hull-bottom curvature sweep -> retention monotonicity (C5).
``energy``     air-flow / compressor balance -> net power (C4).

Usage::

    python experiments/air_lubrication_p21_n05/run_campaign.py --stage all
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.air_lubrication import (  # noqa: E402
    LAMBDA_AIR_WATER,
    NU_WATER,
    RHO_WATER,
    ChannelAirLubrication2D,
    VoidFractionTransport2D,
    alpha_similarity_profile,
    compressor_power,
    curvature_drift_velocity,
    mixture_viscosity,
    stratified_channel_exact,
    stratified_drag_reduction_from_alpha,
    rise_velocity_schiller_naumann,
    stokes_rise_velocity,
)

# ---------------------------------------------------------------- physical set
U_SHIP = 5.14  # m/s (10 kn)
DELTA = 0.10  # boundary-layer thickness at the station of interest [m]
L_SHIP = 100.0
B_SHIP = 20.0
T_SHIP = 6.0
S_WET = 1.7 * L_SHIP * (2.0 * T_SHIP + B_SHIP)
LAMBDA_LBM = 0.05  # numerically realisable viscosity ratio (NOT 1.8e-5)
NY = 96
NX = 160
U_MAX_LAT = 0.08
N_STEPS_TRANSPORT = 7000
TURB_INTENSITY_DISP = 0.02


def out_dir() -> Path:
    d = REPO / "artifacts" / "p21_n05"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fig_dir() -> Path:
    d = REPO / "figures" / "p21_n05"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------- transport
def curtain_case(d_b_um, nx=96, ny=NY, steps=N_STEPS_TRANSPORT, flux=None):
    """Run one developed bubble curtain for bubble diameter ``d_b_um``."""
    d_b = d_b_um * 1e-6
    v_b_phys = rise_velocity_schiller_naumann(d_b)
    H = float(ny) - 0.5
    y = np.arange(ny, dtype=np.float64) + 0.5
    eta = np.clip(y / H, 1e-9, 1.0)
    u_lat = U_MAX_LAT * eta ** (1.0 / 7.0)
    # Disperse-phase turbulent diffusivity: the standard outer-layer closure
    # D_t = c_D U H (c_D = 0.05, the usual boundary-layer scalar eddy
    # diffusivity), reduced by a wall-blocking factor over the inner 5% and
    # floored at 15% so the near-wall layer is not artificially isolated.
    disp = 0.05 * U_MAX_LAT * H * np.clip(eta / 0.05, 0.15, 1.0)
    rise_lat = U_MAX_LAT * v_b_phys / U_SHIP
    if flux is None:
        flux = 0.004
    inj = np.zeros(nx)
    inj[:12] = flux
    tr = VoidFractionTransport2D(
        nx=nx,
        ny=ny,
        u_profile=u_lat,
        rise_velocity=rise_lat,
        disp_y=disp,
        disp_x=None,
        injection=inj,
        dt=0.25,
    )
    cons = 0.0
    U = np.repeat(u_lat[:, None], nx, axis=1)
    for _ in range(steps):
        tr.step(U)
        cons = max(cons, tr.conservation_error())
    x0, x1 = int(0.55 * nx), int(0.95 * nx)
    prof = tr.wall_normal_profile(x0, x1)
    a_wall = float(prof[0])
    delta_b = float(np.sum(prof) / max(a_wall, 1e-30))
    ref = alpha_similarity_profile(y, a_wall, delta_b)
    mask = prof > 0.02 * max(prof.max(), 1e-30)
    shape_err = float(
        np.max(np.abs(prof[mask] - ref[mask]) / np.maximum(ref[mask], 1e-30))
    )
    return {
        "d_b_um": float(d_b_um),
        "v_b_phys_m_s": float(v_b_phys),
        "v_b_over_U": float(v_b_phys / U_SHIP),
        "rise_lattice": float(rise_lat),
        "alpha_wall": a_wall,
        "delta_b_cells": delta_b,
        "H_cells": H,
        "delta_b_over_delta": delta_b / H,
        "profile": prof.tolist(),
        "profile_reference": ref.tolist(),
        "y_over_delta": (y / H).tolist(),
        "shape_error_vs_similarity": shape_err,
        "mass_audit_max_rel": cons,
        "injected": tr.injected_total,
        "outflowed": tr.outflow_total,
        "mass_final": tr.mass(),
        "negativity_clipped": tr.negativity,
        "j_air_m_s": float(a_wall * v_b_phys),
    }


def stage_transport():
    cases = [100.0, 200.0, 500.0, 1000.0, 2000.0, 4000.0, 7000.0]
    rows = [curtain_case(d) for d in cases]
    band = (0.4, 0.6)
    best = min(rows, key=lambda r: abs(r["delta_b_over_delta"] - 0.5))
    ref_err = abs(best["delta_b_over_delta"] - 0.5) / 0.5
    shape_err = best["shape_error_vs_similarity"]
    payload = {
        "stage": "transport",
        "cases": rows,
        "band_delta_b_over_delta": list(band),
        "band_case": {
            "d_b_um": best["d_b_um"],
            "alpha_wall": best["alpha_wall"],
            "delta_b_over_delta": best["delta_b_over_delta"],
        },
        "criterion_C2": {
            "thickness_error_vs_band_centre": ref_err,
            "shape_error_vs_similarity": shape_err,
            "metric": max(ref_err, shape_err),
            "tolerance": 0.30,
            "pass": bool(max(ref_err, shape_err) < 0.30),
            "note": (
                "Reference is the published wall-peaked exponential similarity profile "
                "with delta_b/delta in [0.4,0.6] (Madavan et al. 1985 band); digitised "
                "point data were not available offline, so a point-by-point experiment "
                "overlay is WITHHELD."
            ),
        },
        "criterion_C3_mass": {
            "max_relative_residual": max(r["mass_audit_max_rel"] for r in rows),
            "tolerance": 1e-8,
            "pass": bool(max(r["mass_audit_max_rel"] for r in rows) < 1e-8),
        },
    }
    (out_dir() / "transport.json").write_text(json.dumps(payload, indent=2))
    return payload


# -------------------------------------------------------------------- Re sweep
def _profile_error(u_num, alpha_profile, lam, H, u_mean):
    if alpha_profile is None:
        mu = np.ones(len(np.asarray(u_num)), dtype=float)
    else:
        mu = mixture_viscosity(np.asarray(alpha_profile, dtype=float), 1.0, lam)
    ex = stratified_channel_exact(mu, 1.0, H=H)
    ref = ex["u_profile"] / float(np.mean(ex["u_profile"])) * u_mean
    return float(np.max(np.abs(np.asarray(u_num) - ref)) / max(np.max(ref), 1e-30))


def lbm_case(nu, alpha_profile, lam, ny=32, nx=8, steps=700, u_mean=0.05, cs_smag=0.0):
    sim = ChannelAirLubrication2D(
        nx=nx,
        ny=ny,
        u_mean=u_mean,
        nu_liquid=nu,
        lam=lam,
        alpha_profile=alpha_profile,
        sgs=cs_smag > 0.0,
        cs_smag=cs_smag,
    )
    st = sim.run(steps)
    return {
        "nu": float(nu),
        "tau_w": st.tau_w,
        "force": st.force,
        "u_mean": st.u_mean,
        "re_tau": st.re_tau,
        "re_theta": st.re_theta,
        "re_bulk": st.re_bulk,
        "cf": st.cf,
        "momentum_residual": st.momentum_residual_max,
        "force_balance_error": abs(st.tau_w - sim.force * sim.H) / (sim.force * sim.H),
        "u_profile": st.u_profile.tolist(),
        "H": sim.H,
        "profile_error_vs_exact": _profile_error(st.u_profile, alpha_profile, lam, sim.H, st.u_mean),
    }


def _transport_shape(ny, alpha_wall):
    """Resample the transport-derived void-fraction shape onto ``ny`` rows."""
    tr = json.loads((out_dir() / "transport.json").read_text())
    band_case = tr["band_case"]
    case = min(tr["cases"], key=lambda c: abs(c["d_b_um"] - band_case["d_b_um"]))
    prof = np.array(case["profile"], dtype=float)
    prof = prof / max(prof[0], 1e-30)
    src = (np.arange(prof.size) + 0.5) / prof.size
    dst = (np.arange(ny) + 0.5) / ny
    return np.clip(np.interp(dst, src, prof), 0.0, 1.0) * alpha_wall, case["d_b_um"]


def stage_re_sweep():
    y = np.arange(32, dtype=np.float64) + 0.5
    alpha, d_b_shape = _transport_shape(32, 0.10)
    nus = np.geomspace(2.0e-3, 4.0e-4, 15)
    base = lbm_case(nus[0], None, 1.0)
    rows = []
    for nu in nus:
        base = lbm_case(nu, None, 1.0)
        air = lbm_case(nu, alpha, LAMBDA_LBM)
        exact = stratified_drag_reduction_from_alpha(alpha, air["H"], lam=LAMBDA_LBM)
        dr_lbm = 1.0 - air["tau_w"] / base["tau_w"]
        rows.append(
            {
                "nu": float(nu),
                "re_theta": air["re_theta"],
                "re_tau": air["re_tau"],
                "re_bulk": air["re_bulk"],
                "cf0": base["cf"],
                "tau_w0": base["tau_w"],
                "tau_w_air": air["tau_w"],
                "dr_lbm": float(dr_lbm),
                "dr_exact": float(exact["dr"]),
                "dr_error": float(abs(dr_lbm - exact["dr"]) / max(exact["dr"], 1e-12)),
                "h_air": float(exact["h_air"]),
                "momentum_residual": max(base["momentum_residual"], air["momentum_residual"]),
                "force_balance_error": max(base["force_balance_error"], air["force_balance_error"]),
                "profile_error": air["profile_error_vs_exact"],
                "u_profile_air": air["u_profile"],
            }
        )
    payload = {
        "stage": "re_sweep",
        "lambda_lbm": LAMBDA_LBM,
        "alpha_wall": 0.10,
        "profile_source_d_b_um": d_b_shape,
        "cases": rows,
        "criterion_C1": {
            "max_relative_error": float(max(r["dr_error"] for r in rows)),
            "tolerance": 0.20,
            "pass": bool(max(r["dr_error"] for r in rows) < 0.20),
            "note": (
                "Reference = exact stratified half-channel law at the SAME lambda_lbm; the "
                "true air/water ratio 1.8e-5 is not realisable in a BGK lattice."
            ),
        },
        "criterion_C3_momentum": {
            "max_budget_residual": float(max(r["momentum_residual"] for r in rows)),
            "max_force_balance_error": float(max(r["force_balance_error"] for r in rows)),
            "tolerance": 1e-8,
            "pass": bool(max(r["momentum_residual"] for r in rows) < 1e-8),
        },
    }
    (out_dir() / "re_sweep.json").write_text(json.dumps(payload, indent=2))

    return payload



def stage_re_sweep_hi():
    """High-Re (SGS active) sub-sweep + viscosity-ratio sensitivity."""
    y = np.arange(32, dtype=np.float64) + 0.5
    alpha, _ = _transport_shape(32, 0.10)
    d = json.loads((out_dir() / "re_sweep.json").read_text())
    d["high_re_sgs"] = []
    for nu in np.geomspace(4.0e-4, 1.0e-4, 5):
        b = lbm_case(nu, None, 1.0, cs_smag=0.10, steps=1200)
        air = lbm_case(nu, alpha, LAMBDA_LBM, cs_smag=0.10, steps=1200)
        ex = stratified_drag_reduction_from_alpha(alpha, air["H"], lam=LAMBDA_LBM)
        d["high_re_sgs"].append(
            {
                "nu": float(nu),
                "re_theta": air["re_theta"],
                "re_tau": air["re_tau"],
                "cf0": b["cf"],
                "dr_lbm": float(1.0 - air["tau_w"] / b["tau_w"]),
                "dr_exact": float(ex["dr"]),
                "momentum_residual": max(b["momentum_residual"], air["momentum_residual"]),
                "note": (
                    "Smagorinsky SGS active (Cs=0.10): the sub-grid viscosity erases the "
                    "molecular viscosity contrast, so the purely viscous stratification "
                    "mechanism decays with Re -- the high-Re drag reduction must come from "
                    "bubble-layer suppression of the turbulent shear stress instead."
                ),
            }
        )
    lam_rows = []
    for lam in (0.02, 0.05, 0.10, 0.20):
        b = lbm_case(2.0e-3, None, 1.0)
        air = lbm_case(2.0e-3, alpha, lam)
        ex = stratified_drag_reduction_from_alpha(alpha, air["H"], lam=lam)
        lam_rows.append(
            {
                "lambda": lam,
                "dr_lbm": 1.0 - air["tau_w"] / b["tau_w"],
                "dr_exact": float(ex["dr"]),
            }
        )
    phys = stratified_drag_reduction_from_alpha(alpha, 32.0, lam=LAMBDA_AIR_WATER)
    alphaw_rows = []
    for aw in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
        a, _ = _transport_shape(32, aw)
        e = stratified_drag_reduction_from_alpha(a, 32.0, lam=LAMBDA_AIR_WATER)
        alphaw_rows.append({"alpha_wall": aw, "dr_exact_physical_lambda": float(e["dr"])})
    (out_dir() / "lambda_sensitivity.json").write_text(
        json.dumps(
            {
                "lambda_lbm_rows": lam_rows,
                "lambda_physical": {
                    "lambda": LAMBDA_AIR_WATER,
                    "dr_exact": float(phys["dr"]),
                    "note": "theory only; not realisable in the BGK lattice",
                },
                "alpha_wall_sweep_physical_lambda": alphaw_rows,
            },
            indent=2,
        )
    )
    (out_dir() / "re_sweep.json").write_text(json.dumps(d, indent=2))
    return d


# -------------------------------------------------------------------- curvature
def stage_curvature():
    ny, nx = 48, 64
    H = float(ny) - 0.5
    y = np.arange(ny, dtype=np.float64) + 0.5
    eta = np.clip(y / H, 1e-9, 1.0)
    u_lat = U_MAX_LAT * eta ** (1.0 / 7.0)
    disp = np.minimum(
        TURB_INTENSITY_DISP * U_MAX_LAT * H * eta * np.clip(1.0 - eta, 0.05, 1.0), 0.05
    )
    d_b = 300e-6
    v_b = stokes_rise_velocity(d_b)
    rise_lat = U_MAX_LAT * v_b / U_SHIP
    inj = np.zeros(nx)
    inj[:10] = 0.004
    rows = []
    for kappa_phys in (0.0, 0.25, 0.5, 1.0, 2.0):
        u_drift_phys = curvature_drift_velocity(U_SHIP, kappa_phys, d_b)
        u_drift_lat = U_MAX_LAT * u_drift_phys / U_SHIP
        tr = VoidFractionTransport2D(
            nx=nx,
            ny=ny,
            u_profile=u_lat,
            rise_velocity=rise_lat,
            disp_y=disp,
            disp_x=disp,
            injection=inj,
            curvature=np.full(ny, u_drift_lat),
            dt=0.4,
        )
        U = np.repeat(u_lat[:, None], nx, axis=1)
        for _ in range(2500):
            tr.step(U)
        prof = tr.wall_normal_profile(int(0.55 * nx), int(0.95 * nx))
        near = float(np.sum(prof[: int(0.10 * ny)]))
        total = float(np.sum(prof))
        rows.append(
            {
                "kappa_1_per_m": kappa_phys,
                "radius_m": (1.0 / kappa_phys) if kappa_phys else None,
                "u_drift_phys_m_s": float(u_drift_phys),
                "u_drift_over_rise": float(u_drift_phys / max(v_b, 1e-12)),
                "alpha_wall": float(prof[0]),
                "retention_inner_10pct": near,
                "retention_fraction": near / max(total, 1e-30),
                "profile": prof.tolist(),
            }
        )
    fracs = [r["retention_fraction"] for r in rows]
    mono = all(fracs[i] > fracs[i + 1] for i in range(len(fracs) - 1))
    payload = {
        "stage": "curvature",
        "cases": rows,
        "criterion_C5": {
            "monotone_decreasing": bool(mono),
            "series": fracs,
            "pass": bool(mono),
            "note": (
                "kappa>0 = concave-up hull bottom: a light bubble migrates toward the "
                "centre of curvature, i.e. away from the hull, so near-wall retention must "
                "decrease monotonically with curvature."
            ),
        },
    }
    (out_dir() / "curvature.json").write_text(json.dumps(payload, indent=2))
    return payload


# ----------------------------------------------------------------------- energy
def ittc57_cf(re):
    return 0.075 / (math.log10(re) - 2.0) ** 2


def stage_energy():
    transport = json.loads((out_dir() / "transport.json").read_text())
    re_sweep = json.loads((out_dir() / "re_sweep.json").read_text())
    dr_lbm = float(np.mean([r["dr_lbm"] for r in re_sweep["cases"]]))
    re_ship = U_SHIP * L_SHIP / NU_WATER
    cf0 = ittc57_cf(re_ship)
    rows = []
    for case in transport["cases"]:
        v_b = case["v_b_phys_m_s"]
        a_wall = max(case["alpha_wall"], 1e-12)
        j_air = a_wall * v_b
        q_air = j_air * S_WET
        p_air = compressor_power(q_air, depth=T_SHIP, rho_water=RHO_WATER)
        p_saved = dr_lbm * cf0 * 0.5 * RHO_WATER * U_SHIP**3 * S_WET
        rows.append(
            {
                "d_b_um": case["d_b_um"],
                "alpha_wall": a_wall,
                "j_air_m_s": float(j_air),
                "q_air_m3_s": float(q_air),
                "q_air_over_US": float(q_air / (U_SHIP * S_WET)),
                "power_air_kW": p_air / 1e3,
                "power_saved_kW": p_saved / 1e3,
                "net_power_kW": (p_saved - p_air) / 1e3,
            }
        )
    best = max(rows, key=lambda r: r["net_power_kW"])
    p_atm = 101325.0
    r = (p_atm + RHO_WATER * 9.81 * T_SHIP) / p_atm
    j_be = dr_lbm * cf0 * 0.5 * RHO_WATER * U_SHIP**3 * 0.6 / (p_atm * math.log(r))
    payload = {
        "stage": "energy",
        "u_ship_m_s": U_SHIP,
        "delta_m": DELTA,
        "s_wet_m2": S_WET,
        "re_ship": re_ship,
        "cf0_ittc57": cf0,
        "dr_used": dr_lbm,
        "cases": rows,
        "optimum": best,
        "j_break_even_m_s": float(j_be),
        "criterion_C4": {
            "best_net_power_kW": best["net_power_kW"],
            "pass": bool(best["net_power_kW"] > 0.0),
            "q_air_optimum_m3_s": best["q_air_m3_s"],
            "note": (
                "Air consumption is the escaping flux alpha_wall*V_rise over the wetted "
                "area; compressor power is isothermal at the injection depth with eta=0.6. "
                "Drag reduction is the mean lattice value (lambda=0.05)."
            ),
        },
    }
    (out_dir() / "energy.json").write_text(json.dumps(payload, indent=2))
    return payload


# ---------------------------------------------------------------------- figures
def make_figures():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fd = fig_dir()
    figs = []

    tr = json.loads((out_dir() / "transport.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for case in tr["cases"]:
        ax[0].plot(
            np.array(case["profile"]) / max(case["alpha_wall"], 1e-12),
            np.array(case["y_over_delta"]),
            label="d_b=%.0f um" % case["d_b_um"],
        )
    ref = tr["cases"][3]
    ax[0].plot(
        np.array(ref["profile_reference"]) / max(ref["alpha_wall"], 1e-12),
        np.array(ref["y_over_delta"]),
        "k--",
        lw=1.2,
        label="exp similarity (ref)",
    )
    ax[0].set_xlabel(r"$\alpha/\alpha_{wall}$")
    ax[0].set_ylabel(r"$y/\delta$")
    ax[0].set_ylim(0, 0.6)
    ax[0].legend(fontsize=7)
    ax[0].set_title("(a) void-fraction similarity profiles")
    ax[1].axhspan(0.4, 0.6, color="0.85", label="published band 0.4-0.6")
    ax[1].semilogx([c["d_b_um"] for c in tr["cases"]], [c["delta_b_over_delta"] for c in tr["cases"]], "o-")
    ax[1].set_xlabel(r"$d_b$ [$\mu$m]")
    ax[1].set_ylabel(r"$\delta_b/\delta$")
    ax[1].legend(fontsize=7)
    ax[1].set_title("(b) bubble-layer thickness vs bubble size")
    fig.tight_layout()
    fig.savefig(fd / "fig1_transport.png", dpi=130)
    plt.close(fig)
    figs.append(
        (
            "fig1_transport.png",
            "(a) computed void-fraction profiles against the exponential similarity shape; "
            "(b) bubble-layer thickness against the published 0.4-0.6 band.",
        )
    )

    rs = json.loads((out_dir() / "re_sweep.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    rth = [r["re_theta"] for r in rs["cases"]]
    ax[0].semilogx(rth, [r["dr_lbm"] for r in rs["cases"]], "o-", label=r"lattice $\Delta C_f$")
    ax[0].semilogx(rth, [r["dr_exact"] for r in rs["cases"]], "s--", mfc="none", label="exact stratified law")
    ax[0].set_xlabel(r"$Re_\theta$")
    ax[0].set_ylabel(r"$\Delta C_f / C_{f0}$")
    ax[0].legend(fontsize=8)
    ax[0].set_title("(a) drag reduction vs Re_theta (15 cases)")
    ax[0].grid(alpha=0.3)
    ax[1].semilogx(rth, [r["cf0"] for r in rs["cases"]], "o-")
    ax[1].set_xlabel(r"$Re_\theta$")
    ax[1].set_ylabel(r"$C_{f0}$")
    ax[1].set_title("(b) baseline skin-friction coefficient")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fd / "fig2_re_sweep.png", dpi=130)
    plt.close(fig)
    figs.append(
        (
            "fig2_re_sweep.png",
            "Fifteen-case Re sweep: lattice drag reduction against the exact stratified law, "
            "and the baseline skin-friction coefficient.",
        )
    )

    ls = json.loads((out_dir() / "lambda_sensitivity.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    lams = [r["lambda"] for r in ls["lambda_lbm_rows"]]
    ax[0].loglog(lams, [r["dr_lbm"] for r in ls["lambda_lbm_rows"]], "o-", label="lattice")
    ax[0].loglog(lams, [r["dr_exact"] for r in ls["lambda_lbm_rows"]], "s--", mfc="none", label="exact")
    ax[0].axvline(LAMBDA_AIR_WATER, color="0.6", ls=":", label=r"$\lambda_{air/water}$")
    ax[0].set_xlabel(r"$\lambda=\mu_{air}/\mu_{water}$")
    ax[0].set_ylabel(r"$\Delta C_f/C_{f0}$")
    ax[0].legend(fontsize=8)
    ax[0].set_title("(a) viscosity-ratio sensitivity")
    aw = [r["alpha_wall"] for r in ls["alpha_wall_sweep_physical_lambda"]]
    ax[1].plot(aw, [r["dr_exact_physical_lambda"] for r in ls["alpha_wall_sweep_physical_lambda"]], "o-")
    ax[1].set_xlabel(r"$\alpha_{wall}$")
    ax[1].set_ylabel(r"$\Delta C_f/C_{f0}$")
    ax[1].set_title(r"(b) physical $\lambda$ sandwich law vs $\alpha_{wall}$")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fd / "fig3_lambda.png", dpi=130)
    plt.close(fig)
    figs.append(
        (
            "fig3_lambda.png",
            "Viscosity-ratio sensitivity of the drag reduction and the physical-ratio sandwich "
            "law against wall void fraction.",
        )
    )

    cv = json.loads((out_dir() / "curvature.json").read_text())
    en = json.loads((out_dir() / "energy.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot([r["kappa_1_per_m"] for r in cv["cases"]], [r["retention_fraction"] for r in cv["cases"]], "o-")
    ax[0].set_xlabel(r"hull-bottom curvature $\kappa$ [1/m] (concave up)")
    ax[0].set_ylabel("near-wall air retention fraction")
    ax[0].set_title("(a) curvature -> retention monotonicity")
    ax[0].grid(alpha=0.3)
    q = [max(r["q_air_m3_s"], 1e-9) for r in en["cases"]]
    ax[1].semilogx(q, [r["net_power_kW"] for r in en["cases"]], "o-", label="net power")
    ax[1].semilogx(q, [r["power_saved_kW"] for r in en["cases"]], "s--", mfc="none", label="saved")
    ax[1].semilogx(q, [r["power_air_kW"] for r in en["cases"]], "^:", mfc="none", label="compressor")
    ax[1].axhline(0, color="0.5", lw=0.8)
    ax[1].set_xlabel(r"$Q_{air}$ [m$^3$/s] (atmospheric)")
    ax[1].set_ylabel("power [kW]")
    ax[1].legend(fontsize=8)
    ax[1].set_title("(b) net energy balance")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fd / "fig4_curvature_energy.png", dpi=130)
    plt.close(fig)
    figs.append(
        (
            "fig4_curvature_energy.png",
            "Hull-curvature retention monotonicity and the net power balance against air flow rate.",
        )
    )

    case = tr["cases"][3]
    fig2, ax2 = plt.subplots(figsize=(7.5, 3.0))
    prof = np.array(case["profile"])
    field = np.tile(prof[:, None], (1, 60))
    im = ax2.imshow(field, origin="lower", aspect="auto", cmap="viridis")
    fig2.colorbar(im, ax=ax2, label=r"$\alpha$")
    ax2.set_xlabel("x [cells] (developed section)")
    ax2.set_ylabel("y [cells]")
    ax2.set_title(r"developed void-fraction field $\alpha(x,y)$, $d_b$=%.0f $\mu$m" % case["d_b_um"])
    fig2.tight_layout()
    fig2.savefig(fd / "fig5_alpha_field.png", dpi=130)
    plt.close(fig2)
    figs.append(
        (
            "fig5_alpha_field.png",
            "Wall-normal void-fraction field of the developed bubble curtain reconstructed from "
            "the streamwise-averaged transport solution.",
        )
    )
    return figs


# ------------------------------------------------------------------------- main
def summary():
    lines = []
    for name in (
        "transport",
        "re_sweep",
        "lambda_sensitivity",
        "curvature",
        "energy",
    ):
        p = out_dir() / (name + ".json")
        if not p.exists():
            lines.append("%-20s MISSING" % name)
            continue
        d = json.loads(p.read_text())
        if name == "transport":
            lines.append(
                "transport   C2 metric=%.4f pass=%s | C3_mass=%.2e pass=%s"
                % (
                    d["criterion_C2"]["metric"],
                    d["criterion_C2"]["pass"],
                    d["criterion_C3_mass"]["max_relative_residual"],
                    d["criterion_C3_mass"]["pass"],
                )
            )
        elif name == "re_sweep":
            lines.append(
                "re_sweep    C1 maxerr=%.4f pass=%s | C3_mom=%.2e pass=%s | Re_theta %.0f..%.0f"
                % (
                    d["criterion_C1"]["max_relative_error"],
                    d["criterion_C1"]["pass"],
                    d["criterion_C3_momentum"]["max_budget_residual"],
                    d["criterion_C3_momentum"]["pass"],
                    min(c["re_theta"] for c in d["cases"]),
                    max(c["re_theta"] for c in d["cases"]),
                )
            )
        elif name == "curvature":
            lines.append(
                "curvature   C5 monotone=%s pass=%s"
                % (d["criterion_C5"]["monotone_decreasing"], d["criterion_C5"]["pass"])
            )
        elif name == "energy":
            lines.append(
                "energy      C4 best net=%.1f kW pass=%s | Cf0=%.5f Re=%.2e"
                % (
                    d["criterion_C4"]["best_net_power_kW"],
                    d["criterion_C4"]["pass"],
                    d["cf0_ittc57"],
                    d["re_ship"],
                )
            )
    print("\n".join(lines))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        default="all",
        choices=["all", "transport", "re_sweep", "re_sweep_hi", "curvature", "energy", "figures"],
    )
    args = ap.parse_args()
    if args.stage in ("all", "transport"):
        stage_transport()
    if args.stage in ("all", "re_sweep"):
        stage_re_sweep()
    if args.stage in ("all", "re_sweep_hi"):
        stage_re_sweep_hi()
    if args.stage in ("all", "curvature"):
        stage_curvature()
    if args.stage in ("all", "energy"):
        stage_energy()
    if args.stage in ("all", "figures"):
        make_figures()
    summary()


if __name__ == "__main__":
    main()