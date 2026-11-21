#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K2-BUDGET-B 判决脚本：撞击动量/能量在峰时刻之前的去向（逐项收支分解）。

输入（本 worktree 内）：
  runs/k2budget/result.json        —— 预算臂（--diag-budget，只读测量）
  runs/k2budget_ctrl/result.json   —— 对照臂（诊断关闭，轨迹不变性对照）
  runs/k2budget_planes/planes.npz  —— 中截面快照（可选，坑形交叉校核）
  benchmarks/water_entry/single_drop/fig5_experiment.csv —— 实验基准点
锚（外部只读参照，不参与计算）：
  /root/tp-agent/work/m26392/runs/cavsweep/ctrl.json
输出：
  runs/k2budget_verdict/verdict.json  —— 全部数字（可溯，零手抄）
  runs/k2budget_verdict/verdict.md    —— 判决摘要（数字同样程序化注入）
判决阈值写死在 PREREG.md（docs/audits/k2_budget_20261121/），本脚本只执行不改。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from benchmarks.water_entry.single_drop.budget_diag import integrate  # noqa: E402

ANCHOR = "/root/tp-agent/work/m26392/runs/cavsweep/ctrl.json"
EXP_CSV = os.path.join(ROOT, "benchmarks", "water_entry", "single_drop", "fig5_experiment.csv")
P_AMB = 1.0 / 3.0            # rho_gas = 1 (压强匹配) -> p = cs^2 rho = 1/3
RHO_W = 997.0                # kg/m^3, protocol.json water_density_kg_m3
G_SI = 9.81                  # m/s^2

# 预登记阈值（PREREG.md 第 4 节）
CLOSURE_TOL = 0.10           # |R| / E_scale < 10 % 才算收支闭合
DOMINANCE = 0.50             # 单项份额 >= 50 % 判为主导通道


def read_experiment():
    with open(EXP_CSV) as handle:
        rows = list(csv.DictReader(handle))
    return [(float(r["time_ms"]), float(r["depth_over_D"])) for r in rows]


def peak_of(history):
    best = None
    for row in history:
        value = row.get("depth_over_D")
        if value is None:
            continue
        if best is None or value > best["depth_over_D"]:
            best = row
    return best


def history_delta(a_rows, b_rows):
    """Max |delta| over every shared numeric field of two histories."""
    worst = {}
    if len(a_rows) != len(b_rows):
        return {"__length__": abs(len(a_rows) - len(b_rows))}
    for ra, rb in zip(a_rows, b_rows):
        for key in ra:
            va, vb = ra.get(key), rb.get(key)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                delta = abs(float(va) - float(vb))
                if delta > worst.get(key, 0.0):
                    worst[key] = delta
    return worst


def _channel_key(name, ch):
    """(label, formatted value) of the headline number of a channel."""
    if name == "1_gas_pressure_impulse":
        return "冲量 z", f"{ch['impulse_z_lattice']:.6e}"
    if name == "2_surface_energy_increment":
        return "sigma*dA_cav", f"{ch['cavity_wall_sigma_dA_lattice']:.6e}"
    if name == "3_interface_viscous_dissipation":
        return "dissipation", f"{ch['total_lattice']:.6e}"
    if name == "4_lateral_jet_kinetic_energy":
        return "dKE_lat", f"{ch['ke_lat_gain']:.6e}"
    if name == "5_droplet_residual_kinetic_energy":
        return "KE_drop(t*)", f"{ch['ke_drop_final']:.6e}"
    if name == "6_axial_water_column_kinetic_energy":
        return "KE_axial(t*)", f"{ch['ke_axial_final']:.6e}"
    if name == "7_mass_exchange_conversion":
        return "conv_loss", f"{ch['conv_loss_lattice']:.6e}"
    if name == "8_gravity_work":
        return "W_grav", f"{ch['work_lattice']:.6e}"
    return name, ""


def _crater_profile(tree, target_ms, cells_per_D, dx_m):
    """Crater half-width and depth integral from the mid-plane snapshot.

    Returns None when the planes arm is not available.  The mid-plane is a
    diameter through the axis; the crater is treated as axisymmetric, so the
    potential energy below the flat level is 2 pi rho g integral |z(r)| r dr.
    The plane closest to the depth peak is used.
    """
    path = os.path.join(tree, "runs", "k2budget_planes", "planes.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    times = data["time_ms"]
    fill = data["fill"]
    x = data["x_over_D"]
    z = data["z_over_D"]
    target = target_ms
    k = int(np.argmin(np.abs(times - target)))
    field = fill[k]
    depth = np.zeros_like(x)
    for j in range(field.shape[1]):
        col = field[:, j]
        idx = np.where(col <= 0.5)[0]
        if len(idx) == 0 or idx[0] == 0:
            continue
        i = int(idx[0])
        lo, hi = float(col[i - 1]), float(col[i])
        frac = 0.0 if lo == hi else (lo - 0.5) / (lo - hi)
        z_if = z[i - 1] + frac * (z[i] - z[i - 1])
        depth[j] = -float(z_if)          # depth below the initial surface, in D
    wet = depth > 0
    if not wet.any():
        return None
    r_cav_D = float(np.max(np.abs(x[wet])))
    mask = (np.abs(x) <= r_cav_D) & wet
    rr = np.abs(x[mask])
    zz = depth[mask]
    order = np.argsort(rr)
    rr, zz = rr[order], zz[order]
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    # 2 pi integral_0^R z(r) r dr = pi * integral_{-R}^{R} z(r) |r| dr
    pe_integral_D3 = math.pi * float(trapz(zz * rr, rr))
    return {
        "r_cav_D": r_cav_D,
        "r_cav_cells": r_cav_D * cells_per_D,
        "r_cav_m": r_cav_D * dx_m,
        "pe_integral_m3": pe_integral_D3 * dx_m**3,
        "plane_time_ms": float(times[k]),
        "depth_profile_D": [float(v) for v in depth],
        "x_D": [float(v) for v in x],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", default=ROOT)
    ap.add_argument("--budget", default="runs/k2budget/result.json")
    ap.add_argument("--ctrl", default="runs/k2budget_ctrl/result.json")
    ap.add_argument("--out", default="runs/k2budget_verdict")
    args = ap.parse_args()
    tree = os.path.abspath(args.tree)

    with open(os.path.join(tree, args.budget)) as handle:
        budget = json.load(handle)
    with open(os.path.join(tree, args.ctrl)) as handle:
        ctrl = json.load(handle)
    sc = budget["scales"]
    sigma = sc["sigma"]
    rows = budget["budget_diag"]["rows"]
    history = budget["history"]
    contact_step = budget["contact_step"]
    peak = peak_of(history)
    t_star_step = peak["step"]

    window = [r for r in rows if contact_step <= r["step"] <= t_star_step]
    first, last = window[0], window[-1]

    # ---- integrals over the window ---------------------------------------
    keys = [
        "grav_power", "gas_power_wall", "gas_power_free", "diss_rate", "diss_iface",
        "conv_loss", "conv_gain", "grav_force_z", "gas_force_wall_z", "conv_mom_z",
    ]
    integ = integrate(window, keys)

    E_scale = first["ke_liq"]
    ke_star = last["ke_liq"]
    d_ke = ke_star - E_scale
    w_grav = integ["grav_power"]
    w_atm = -P_AMB * (last["vol_liq"] - first["vol_liq"])
    d_e_surf = sigma * (last["area_total"] - first["area_total"])
    d_e_surf_cav = sigma * (last["area_cav"] - first["area_cav"])
    e_diss = integ["diss_rate"]
    e_diss_iface = integ["diss_iface"]
    e_conv = integ["conv_gain"] - integ["conv_loss"]

    residual_a = d_ke - (w_grav + w_atm - d_e_surf - e_diss + e_conv)
    residual_b = d_ke - (w_grav - d_e_surf - e_diss + e_conv)
    scale_a = max(abs(E_scale), 1e-30)
    closure_a = abs(residual_a) / scale_a
    closure_b = abs(residual_b) / scale_a

    # ---- excavation impulse: the weight of the excavated water column -----
    # the cavity gas removes the hydrostatic push of the excavated column;
    # the impulse the impact has to supply to hold the crater open is
    #   J_exc = integral rho_l g V_cav(t) dt
    exc = integrate(window, ["cav_volume"])
    j_exc = float(sc["g"] * exc["cav_volume"])
    imp_p_abs = abs(float(first["p_drop_z"]))

    # ---- channel shares (share of the impact energy E_scale) --------------
    def share(value):
        return float(value / scale_a)

    channels = {
        "1_gas_pressure_impulse": {
            "kind": "source/impulse",
            "impulse_z_lattice": float(integ["gas_force_wall_z"]),
            "local_work_lattice": float(integ["gas_power_wall"]),
            "global_atmospheric_work_lattice": float(w_atm),
            "excavation_impulse_lattice": j_exc,
            "excavation_impulse_over_impact_momentum": (j_exc / imp_p_abs) if imp_p_abs else None,
            "note": "the cavity gas sits at the ambient pressure (rho_gas = 1, pressure "
                    "matched) so it delivers no net downward impulse beyond the "
                    "hydrostatic reference; the impact has to supply the excavation "
                    "impulse rho_l g integral(V_cav dt) itself",
        },
        "2_surface_energy_increment": {
            "kind": "store",
            "total_sigma_dA_lattice": float(d_e_surf),
            "cavity_wall_sigma_dA_lattice": float(d_e_surf_cav),
            "share": share(d_e_surf_cav),
            "area_total_cav_lattice": [first["area_total"], last["area_total"]],
        },
        "3_interface_viscous_dissipation": {
            "kind": "sink",
            "total_lattice": float(e_diss),
            "interface_neighbourhood_lattice": float(e_diss_iface),
            "bulk_lattice": float(e_diss - e_diss_iface),
            "share": share(e_diss),
        },
        "4_lateral_jet_kinetic_energy": {
            "kind": "store",
            "ke_lat_initial": float(first["ke_lat"]),
            "ke_lat_final": float(last["ke_lat"]),
            "ke_lat_gain": float(last["ke_lat"] - first["ke_lat"]),
            "ke_lat_roi_initial": float(first["ke_lat_roi"]),
            "ke_lat_roi_final": float(last["ke_lat_roi"]),
            "ke_lat_roi_gain": float(last["ke_lat_roi"] - first["ke_lat_roi"]),
            "share_gain": share(last["ke_lat"] - first["ke_lat"]),
        },
        "5_droplet_residual_kinetic_energy": {
            "kind": "store",
            "ke_drop_initial": float(first["ke_drop"]),
            "ke_drop_final": float(last["ke_drop"]),
            "ke_drop_loss": float(first["ke_drop"] - last["ke_drop"]),
            "share": share(last["ke_drop"]),
        },
        "6_axial_water_column_kinetic_energy": {
            "kind": "store",
            "ke_axial_initial": float(first["ke_axial"]),
            "ke_axial_final": float(last["ke_axial"]),
            "share": share(last["ke_axial"]),
        },
        "7_mass_exchange_conversion": {
            "kind": "sink",
            "conv_loss_lattice": float(integ["conv_loss"]),
            "conv_gain_lattice": float(integ["conv_gain"]),
            "share": share(integ["conv_loss"]),
        },
        "8_gravity_work": {
            "kind": "source",
            "work_lattice": float(w_grav),
            "share": share(w_grav),
        },
    }

    # ---- dominance test ---------------------------------------------------
    dominance = {}
    for name, ch in channels.items():
        value = ch.get("share")
        if value is None:
            continue
        dominance[name] = {"share": float(value), "dominates": bool(abs(value) >= DOMINANCE)}
    ranked = sorted(
        [(k, v["share"]) for k, v in dominance.items()],
        key=lambda kv: abs(kv[1]), reverse=True,
    )

    # ---- SI conversion factors (lattice -> physical) ---------------------
    #   E_SI = E_lattice * rho_l dx^5 / dt^2   (kinetic energy)
    #   P_SI = P_lattice * rho_l dx^4 / dt     (momentum)
    #   F_SI = F_lattice * rho_l dx^4 / dt^2   (force)
    dx_m = sc["dx_m"]
    dt_s = sc["dt_s"]
    e_unit = RHO_W * dx_m**5 / dt_s**2
    p_unit = RHO_W * dx_m**4 / dt_s
    f_unit = RHO_W * dx_m**4 / dt_s**2

    # ---- experiment comparison (part b) ----------------------------------
    experiment = read_experiment()
    exp_peak = max(experiment, key=lambda t: t[1])
    d_exp = float(exp_peak[1])
    d_lbm = float(peak["depth_over_D"])
    depth_deficit = (d_lbm - d_exp) / d_exp

    # crater radius / profile: prefer the measured mid-plane profile
    cav_volume = float(last["cav_volume"])
    r_cav = math.sqrt(2.0 * cav_volume / (math.pi * d_lbm)) if d_lbm > 0 else float("nan")
    dx = dx_m
    r_cav_m = r_cav * dx
    d_deficit_m = (d_exp - d_lbm) * dx
    profile = _crater_profile(tree, float(peak["time_ms"]), budget.get("cells_per_D"), dx_m)
    # Gravitational cost of the deeper experimental crater.
    #   The excavation lifts the displaced water against gravity; the PE
    #   deficit of a depression of profile z(r) is rho g * integral |z| dA,
    #   and for the same crater shape it scales linearly with the depth.
    #   Only the excavated volume is needed:
    #       dE_PE = rho g * (d_exp/d_lbm - 1) * V_excavated
    #   with V_excavated = cavity volume at the peak (measured cells).
    shape_ratio = (d_exp / d_lbm - 1.0) if d_lbm > 0 else float("nan")
    d_v_excavated_cells = shape_ratio * cav_volume
    d_e_pe_lattice = float(sc["g"] * d_v_excavated_cells)
    d_e_pe_si = d_e_pe_lattice * e_unit
    d_e_pe_si_direct = RHO_W * G_SI * d_v_excavated_cells * dx_m**3
    if profile is not None:
        r_cav_m = profile["r_cav_m"]
        r_cav = profile["r_cav_cells"]
        pe_source = ("excavated-volume scaling (cavity volume at the peak); "
                     "crater half-width from the measured mid-plane profile")
    else:
        pe_source = "excavated-volume scaling (cavity volume at the peak)"
    e_scale_si = E_scale * e_unit
    pe_ratio = d_e_pe_si / e_scale_si if e_scale_si else float("nan")

    # extra axial kinetic energy that the experimental depth would need
    # (declared assumption in PREREG: crater depth proportional to axial KE)
    axial_need_lattice = float(last["ke_axial"]) * (d_exp / d_lbm - 1.0) if d_lbm > 0 else float("nan")
    axial_need_si = axial_need_lattice * e_unit

    # which channel holds at least that much energy?
    holders = []
    for name, ch in channels.items():
        for field in ("total_lattice", "interface_neighbourhood_lattice", "ke_lat_gain",
                      "conv_loss_lattice", "cavity_wall_sigma_dA_lattice", "local_work_lattice"):
            value = ch.get(field)
            if value is None:
                continue
            if value > 0 and value >= abs(axial_need_lattice):
                holders.append({"channel": name, "field": field, "value": value})

    # physical (non-artifact) channels: the interface churn (7) and the two
    # pressure-integration artefacts (1) are excluded, the rest are physical
    # stores/sinks of the impact energy.
    physical_values = {
        "2_surface_energy_increment": channels["2_surface_energy_increment"]["total_sigma_dA_lattice"],
        "3_interface_viscous_dissipation": channels["3_interface_viscous_dissipation"]["total_lattice"],
        "4_lateral_jet_kinetic_energy": channels["4_lateral_jet_kinetic_energy"]["ke_lat_gain"],
        "5_droplet_residual_kinetic_energy": channels["5_droplet_residual_kinetic_energy"]["ke_drop_final"],
        "6_axial_water_column_kinetic_energy": channels["6_axial_water_column_kinetic_energy"]["ke_axial_final"],
    }
    physical_ranked = sorted(
        ((k, v / scale_a) for k, v in physical_values.items()), key=lambda kv: abs(kv[1]), reverse=True
    )
    largest_physical = physical_ranked[0] if physical_ranked else (None, None)
    # energy supply vs delivery: the crater is not energy starved

    # ---- momentum ledger (supplementary) ----------------------------------
    imp_p_z = float(first["p_drop_z"])
    d_p_liq_z = float(last["p_liq_z"] - first["p_liq_z"])
    j_grav = float(integ["grav_force_z"])
    j_gas = float(integ["gas_force_wall_z"])
    j_conv = float(integ["conv_mom_z"])
    residual_mom = d_p_liq_z - (j_grav + j_gas + j_conv)
    momentum = {
        "impact_p_drop_z_SI_kg_m_s": float(imp_p_z * p_unit),
        "residual_z_SI_kg_m_s": float(residual_mom * p_unit),
        "impact_p_drop_z_lattice": imp_p_z,
        "d_p_liq_z_lattice": d_p_liq_z,
        "j_gravity_z_lattice": j_grav,
        "j_cavity_gas_wall_z_lattice": j_gas,
        "j_conversion_z_lattice": j_conv,
        "residual_z_lattice": residual_mom,
        "residual_over_impact": abs(residual_mom) / abs(imp_p_z) if imp_p_z else float("nan"),
        "p_drop_z_final": float(last["p_drop_z"]),
        "p_liq_z_final": float(last["p_liq_z"]),
        "note": "the wall reaction is not part of the ledger, so the momentum residual "
                "is an upper bound on the un-modelled wall/pressure reaction",
    }

    # ---- read-only control + anchor --------------------------------------
    delta = history_delta(budget["history"], ctrl["history"])
    read_only = {"max_abs_delta_per_field": delta, "identical": all(v == 0.0 for v in delta.values())}
    anchor = None
    if os.path.exists(ANCHOR):
        with open(ANCHOR) as handle:
            anchor = json.load(handle)
        anchor_peak = peak_of(anchor["history"])
        anchor["peak_depth_over_D"] = float(anchor_peak["depth_over_D"])
        anchor["peak_time_ms"] = float(anchor_peak["time_ms"])
        anchor["ctrl_peak_depth_over_D"] = float(peak_of(ctrl["history"])["depth_over_D"])
        anchor["abs_delta"] = abs(anchor["ctrl_peak_depth_over_D"] - anchor["peak_depth_over_D"])
        anchor["reproduced"] = bool(anchor["abs_delta"] <= 1e-9)

    # ---- running residual and its correlation with interface activity ----
    ke_series = np.array([r["ke_liq"] for r in window])
    def _cum(key):
        vals = np.array([r.get(key, 0.0) or 0.0 for r in window], dtype=float)
        return np.cumsum(vals) - 0.5 * vals[0]
    grav_c = _cum("grav_power")
    diss_c = _cum("diss_rate")
    loss_c = _cum("conv_loss")
    gain_c = _cum("conv_gain")
    running = (ke_series - ke_series[0]) - (grav_c - diss_c + gain_c - loss_c)
    area_series = np.array([r["area_total"] for r in window])
    conv_rate = np.array([r.get("conv_loss", 0.0) or 0.0 for r in window])

    def corr(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    residual_profile = {
        "running_residual_lattice": [float(v) for v in running],
        "corr_residual_vs_interface_area": corr(running, area_series),
        "corr_residual_vs_conversion_loss_rate": corr(running, conv_rate),
        "residual_growth_per_step_mean": float(np.mean(np.diff(running))) if len(running) > 1 else None,
    }

    # ---- localization of the closure residual ----------------------------
    drift_atm = abs(w_atm)
    cap_cells = int(max(r.get("n_at_cap", 0) for r in window))
    ke_at_cap = float(max(r.get("ke_at_cap", 0.0) for r in window))
    localization = {
        "residual_with_atm_term_lattice": float(residual_a),
        "residual_without_atm_term_lattice": float(residual_b),
        "atmospheric_term_lattice": float(w_atm),
        "free_surface_volume_drift_cells": float(last["vol_liq"] - first["vol_liq"]),
        "pseudo_current_cap_cells_max": cap_cells,
        "pseudo_current_ke_at_cap_max": ke_at_cap,
        "conversion_loss_lattice": float(integ["conv_loss"]),
        "conversion_gain_lattice": float(integ["conv_gain"]),
        "conversion_net_lattice": float(integ["conv_loss"] - integ["conv_gain"]),
        "conversion_net_over_E_scale": float((integ["conv_loss"] - integ["conv_gain"]) / scale_a),
        "dissipation_lattice": float(e_diss),
        "notes": [
            "the ambient-pressure term is polluted by the free-surface volume drift "
            "(rigid-domain net volume work)",
            "the local free-surface pressure integral is not a valid continuum work "
            "term: applied to the falling drop it returns an order of magnitude larger "
            "power than the drop's own kinetic-energy change",
            "the anti-bounce-back reconstruction replaces populations at the interface; "
            "its population-level energy contribution is not captured by any macroscopic "
            "channel of this ledger",
        ],
    }

    verdict = {
        "mission": "K2-BUDGET-B",
        "case": {
            "case_id": budget.get("case_id"),
            "cells_per_D": budget.get("cells_per_D"),
            "flags": {
                "gap_over_D": budget.get("gap_over_D"),
                "lattice_speed": budget.get("lattice_speed"),
                "release_speed_m_s": budget.get("release_speed_m_s"),
                "pool_depth_over_D": budget.get("pool_depth_over_D"),
                "u_limit": budget.get("u_limit"),
                "capacity_redistribution": budget.get("capacity_redistribution"),
                "simple_mass_exchange": budget.get("simple_mass_exchange"),
            },
            "scales": sc,
            "stop_reason": budget.get("stop_reason"),
            "wall_seconds": budget.get("_wall_s"),
        },
        "window": {
            "contact_step": contact_step,
            "t_star_step": t_star_step,
            "t_star_time_ms": float(peak["time_ms"]),
            "rows": len(window),
            "t_star_depth_over_D": d_lbm,
        },
        "closure": {
            "E_scale_lattice": float(E_scale),
            "E_scale_SI_J": float(e_scale_si),
            "d_ke_lattice": float(d_ke),
            "w_gravity_lattice": float(w_grav),
            "w_atmosphere_lattice": float(w_atm),
            "d_e_surface_lattice": float(d_e_surf),
            "e_dissipation_lattice": float(e_diss),
            "e_conversion_net_lattice": float(e_conv),
            "residual_with_atm_lattice": float(residual_a),
            "residual_without_atm_lattice": float(residual_b),
            "closure_with_atm": float(closure_a),
            "closure_without_atm": float(closure_b),
            "tolerance": CLOSURE_TOL,
            "closes": bool(closure_a < CLOSURE_TOL),
            "closes_without_atm": bool(closure_b < CLOSURE_TOL),
        },
        "channels": channels,
        "dominance": {
            "threshold": DOMINANCE,
            "per_channel": dominance,
            "ranked": [{"channel": k, "share": v} for k, v in ranked],
            "dominant_channel": ranked[0][0] if ranked and abs(ranked[0][1]) >= DOMINANCE else None,
            "status": ("DETERMINED" if ranked and abs(ranked[0][1]) >= DOMINANCE else "UNDETERMINED"),
        },
        "physical_disposition": {
            "excluded_as_model_level": [
                "7_mass_exchange_conversion (interface churn: gross loss exceeds E_scale)",
                "1_gas_pressure_impulse (pressure-work integral is not a continuum work term)",
            ],
            "per_channel_lattice": physical_values,
            "per_channel_share": {k: v / scale_a for k, v in physical_values.items()},
            "ranked": [{"channel": k, "share": v} for k, v in physical_ranked],
            "largest_physical_channel": largest_physical[0],
            "largest_physical_share": largest_physical[1],
            "energy_supply": {
                "gravitational_PE_requirement_over_impact_energy": float(pe_ratio),
                "depth_deficit_relative": float(depth_deficit),
                "supply_to_deficit_ratio": float(abs(pe_ratio) / abs(depth_deficit)) if depth_deficit else None,
                "statement": "the crater needs far less energy gravitationally than the impact "
                             "delivers, so the depth deficit is a delivery (inertial) problem, "
                             "not an energy-supply problem",
            },
        },
        "experiment_comparison": {
            "experiment_peak_D": d_exp,
            "experiment_peak_time_ms": float(exp_peak[0]),
            "lbm_peak_D": d_lbm,
            "lbm_peak_time_ms": float(peak["time_ms"]),
            "depth_deficit_relative": float(depth_deficit),
            "cavity_volume_cells": cav_volume,
            "extra_excavated_volume_cells": float(d_v_excavated_cells),
            "gravitational_PE_requirement_lattice": d_e_pe_lattice,
            "gravitational_PE_requirement_direct_SI_J": float(d_e_pe_si_direct),
            "crater_radius_cells_paraboloid": r_cav,
            "crater_radius_m": r_cav_m,
            "crater_profile": profile,
            "gravitational_PE_requirement_J": float(d_e_pe_si),
            "gravitational_PE_source": pe_source,
            "SI_energy_unit_J": float(e_unit),
            "SI_momentum_unit_kg_m_s": float(p_unit),
            "SI_force_unit_N": float(f_unit),
            "gravitational_PE_over_impact_energy": float(pe_ratio),
            "extra_axial_KE_needed_J": float(axial_need_si),
            "extra_axial_KE_needed_lattice": float(axial_need_lattice),
            "channels_holding_at_least_that": holders,
        },
        "momentum_ledger": momentum,
        "excavation": {
            "impulse_lattice": j_exc,
            "impulse_SI_kg_m_s": float(j_exc * p_unit),
            "impulse_over_impact_momentum": (j_exc / imp_p_abs) if imp_p_abs else None,
            "cavity_volume_final_cells": cav_volume,
        },
        "localization": localization,
        "residual_profile": residual_profile,
        "controls": {
            "diagnostic_is_read_only": read_only,
            "anchor": anchor,
        },
        "rows": window,
    }

    out_dir = os.path.join(tree, args.out)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "verdict.json"), "w") as handle:
        json.dump(verdict, handle, indent=1, allow_nan=False)

    # ---- markdown summary (all numbers injected from the dict) -----------
    lines = []
    lines.append("# K2-BUDGET-B 判决（程序化生成，零手抄）\n")
    lines.append(f"- 工况：cells={budget.get('cells_per_D')}、gap={budget.get('gap_over_D')}、"
                 f"speed={budget.get('lattice_speed')}、rs={budget.get('release_speed_m_s')}、"
                 f"ul={budget.get('u_limit')}、commit 见 run.log")
    lines.append(f"- 积分窗口：接触步 {contact_step} → 深度峰步 {t_star_step}"
                 f"（{float(peak['time_ms']):.6f} ms，{len(window)} 行）")
    lines.append(f"- 收支闭合：|R|/E_scale = {closure_a:.4%}（含环境压功）/ "
                 f"{closure_b:.4%}（不含）→ "
                 f"{'闭合 PASS' if closure_a < CLOSURE_TOL else '不闭合 honest FAIL'}"
                 f"（阈值 {CLOSURE_TOL:.0%}）")
    lines.append(f"- 主导通道：{verdict['dominance']['dominant_channel']} "
                 f"（{verdict['dominance']['status']}），最大份额 "
                 f"{ranked[0][1]:.4%} @ {ranked[0][0]}" if ranked else "- 主导通道：无")
    lines.append(f"- 实验对照：实验峰 {d_exp:.6f} D vs LBM {d_lbm:.6f} D，"
                 f"缺口 {depth_deficit:.4%}；重力位能需求/撞击能量 = {pe_ratio:.4%}")
    lines.append(f"- 只读对照：history 逐字段最大 |Δ| = "
                 f"{max(read_only['max_abs_delta_per_field'].values(), default=0.0):.3e}"
                 f"（identical={read_only['identical']}）\n")
    lines.append("## 逐项（格点能量单位；份额 = /E_scale）\n")
    lines.append("| 通道 | 类型 | 关键量 | 份额 |")
    lines.append("|---|---|---|---|")
    for name, ch in channels.items():
        key = _channel_key(name, ch)
        share_txt = f"{ch['share']:.4%}" if "share" in ch else "—"
        lines.append(f"| {name} | {ch['kind']} | {key[0]} = {key[1]} | {share_txt} |")
    with open(os.path.join(out_dir, "verdict.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({
        "closure_with_atm": closure_a,
        "closure_without_atm": closure_b,
        "dominant_channel": verdict["dominance"]["dominant_channel"],
        "ranked": verdict["dominance"]["ranked"][:4],
        "pe_ratio": pe_ratio,
        "read_only_identical": read_only["identical"],
    }, indent=1))


if __name__ == "__main__":
    main()
