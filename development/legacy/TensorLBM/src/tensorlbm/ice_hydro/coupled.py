"""Coupled LBM-pressure -> DEM-cover breakup study and critical-velocity sweep.

For each ice thickness the module computes

* the analytical criteria
    - prescribed k^2 criterion            U_cr = (4 D g / rho_w)^(1/4)
    - first-principles flexural-gravity   U_cr = min_k c(k)
    - both restricted to the discrete channel modes k_n = n pi / L
* the bonded-particle (DEM) onset velocities
    - ``momentum`` loading (membrane compression N = rho_w U^2, i.e. the loading
      model implied by the prescribed criterion)
    - ``hydro`` loading (spectral pressure feedback calibrated by the LBM run)
* the crack sequence (time + position) from a dynamic relaxation run
* the sub/super-critical detection sweep (J3)

All numbers written to JSON by :func:`run_study` (no hand-copied values).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .analytical import (
    CoverProperties,
    antinode_positions,
    critical_mode,
    critical_velocity_k2,
    crack_moment_capacity,
    mode_wavenumbers,
    steady_divergence_velocity,
    u_cr_flexural_gravity,
    u_cr_k2,
)
from .bonded_cover import BondedCover


def analytic_row(props: CoverProperties, n_modes: int = 12) -> dict:
    """Analytical critical velocities (continuous minimum + discrete channel modes)."""
    D = props.D
    u_k2, k_k2 = u_cr_k2(D, props.rho_w, props.g)
    u_fg, k_fg = u_cr_flexural_gravity(D, props.h, props.rho_w, props.g)
    kn = mode_wavenumbers(n_modes, props.L)
    u_k2_modes = critical_velocity_k2(kn, D, props.rho_w, props.g)
    u_fg_modes = steady_divergence_velocity(kn, D, props.h, props.rho_w, props.g)
    n_k2, k_k2_mode, u_k2_mode = critical_mode(kn, lambda k: critical_velocity_k2(k, D, props.rho_w, props.g))
    n_fg, k_fg_mode, u_fg_mode = critical_mode(kn, lambda k: steady_divergence_velocity(k, D, props.h, props.rho_w, props.g))
    return {
        "h_i": props.h_i,
        "D": D,
        "M_f": props.M_f,
        "characteristic_length": (D / (props.rho_w * props.g)) ** 0.25,
        "u_cr_k2_continuous": u_k2,
        "k_c_k2_continuous": k_k2,
        "u_cr_k2_discrete": u_k2_mode,
        "n_critical_k2": n_k2,
        "k_critical_k2": k_k2_mode,
        "u_cr_fg_continuous": u_fg,
        "k_c_fg_continuous": k_fg,
        "u_cr_fg_discrete": u_fg_mode,
        "n_critical_fg": n_fg,
        "k_critical_fg": k_fg_mode,
        "modes": {
            "n": list(range(1, n_modes + 1)),
            "k": kn.tolist(),
            "u_cr_k2": u_k2_modes.tolist(),
            "u_cr_fg": u_fg_modes.tolist(),
        },
        "ratio_k2_to_fg": u_k2 / u_fg,
    }


def dem_row(props: CoverProperties, gain: float = 1.0, n_nodes: int = 129) -> dict:
    """DEM (bonded-particle) onset velocities for both loading models."""
    cover = BondedCover(props=props, n_nodes=n_nodes)
    u_mom = cover.onset_velocity("momentum")
    u_hyd = cover.onset_velocity("hydro", gain=gain)
    u_hyd_unit = cover.onset_velocity("hydro", gain=1.0)
    return {
        "h_i": props.h_i,
        "n_nodes": n_nodes,
        "spacing": cover.a,
        "u_cr_dem_momentum": u_mom,
        "u_cr_dem_hydro": u_hyd,
        "u_cr_dem_hydro_unit_gain": u_hyd_unit,
        "gain_used": gain,
        "M_f": props.M_f,
    }


def detection_sweep(
    props: CoverProperties,
    load_model: str,
    u_cr: float,
    gain: float = 1.0,
    n_points: int = 13,
    fractions: Sequence[float] | None = None,
    n_nodes: int = 129,
    t_end: float = 30.0,
    seed_mode: int = 1,
) -> dict:
    """Sub/super-critical sweep: crack detected or not for each U/U_cr fraction (J3)."""
    fr = list(fractions) if fractions is not None else [
        0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 1.05, 1.10, 1.15, 1.25, 1.40
    ]
    rows = []
    for f in fr:
        cover = BondedCover(props=props, n_nodes=n_nodes)
        out = cover.transient(load_model=load_model, U=f * u_cr, gain=gain, t_end=t_end,
                              seed_mode=seed_mode)
        rows.append({
            "fraction": f,
            "U": f * u_cr,
            "n_cracks": out["n_cracks"],
            "first_crack_time": out["cracks"][0]["time"] if out["cracks"] else None,
            "first_crack_position": out["cracks"][0]["position"] if out["cracks"] else None,
            "M_max": out["M_max"],
            "M_f": out["M_f"],
            "M_ratio": out["M_max"] / out["M_f"],
        })
    below = [r for r in rows if r["fraction"] < 1.0]
    above = [r for r in rows if r["fraction"] > 1.0]
    return {
        "load_model": load_model,
        "u_cr": u_cr,
        "rows": rows,
        "below_critical_cracks": int(sum(r["n_cracks"] for r in below)),
        "below_critical_points": len(below),
        "above_critical_detected": int(sum(1 for r in above if r["n_cracks"] > 0)),
        "above_critical_points": len(above),
        "detection_rate_above": (sum(1 for r in above if r["n_cracks"] > 0) / len(above))
        if above else float("nan"),
    }


@dataclass
class IceCoverBreakupStudy:
    thicknesses: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0)
    gain_lbm: float = 1.0
    n_modes: int = 12
    n_nodes: int = 129
    reference_thickness: float = 1.0
    t_end: float = 40.0        # reference crack-sequence runs
    t_end_sweep: float = 60.0  # detection sweep

    def run(self) -> dict:
        rows = []
        for h_i in self.thicknesses:
            props = CoverProperties(h_i=h_i)
            a = analytic_row(props, self.n_modes)
            d = dem_row(props, gain=self.gain_lbm, n_nodes=self.n_nodes)
            a.update({
                "u_cr_dem_momentum": d["u_cr_dem_momentum"],
                "u_cr_dem_hydro": d["u_cr_dem_hydro"],
                "u_cr_dem_hydro_unit_gain": d["u_cr_dem_hydro_unit_gain"],
                "dev_dem_momentum_vs_k2": (d["u_cr_dem_momentum"] - a["u_cr_k2_continuous"])
                / a["u_cr_k2_continuous"],
                "dev_dem_momentum_vs_k2_discrete": (d["u_cr_dem_momentum"] - a["u_cr_k2_discrete"])
                / a["u_cr_k2_discrete"],
                "dev_dem_hydro_vs_fg": (d["u_cr_dem_hydro"] - a["u_cr_fg_continuous"])
                / a["u_cr_fg_continuous"],
                "dev_dem_hydro_vs_fg_discrete": (d["u_cr_dem_hydro"] - a["u_cr_fg_discrete"])
                / a["u_cr_fg_discrete"],
            })
            rows.append(a)

        props_ref = CoverProperties(h_i=self.reference_thickness)
        ref = analytic_row(props_ref, self.n_modes)
        dem_ref = dem_row(props_ref, gain=self.gain_lbm, n_nodes=self.n_nodes)

        # crack sequence near the onset of each loading model
        sequences = {}
        seeds = {"momentum": ref["n_critical_k2"], "hydro": ref["n_critical_fg"]}
        for model, u_cr in (("momentum", dem_ref["u_cr_dem_momentum"]),
                            ("hydro", dem_ref["u_cr_dem_hydro"])):
            mode = seeds[model]
            cover = BondedCover(props=props_ref, n_nodes=self.n_nodes)
            out = cover.transient(load_model=model, U=1.10 * u_cr, gain=self.gain_lbm,
                                  t_end=self.t_end, seed_mode=mode)
            antinodes = antinode_positions(props_ref.L, mode)
            first = out["cracks"][0]["position"] if out["cracks"] else None
            dev = (min(abs(first - xa) / xa for xa in antinodes) if first is not None else None)
            sequences[model] = {
                **out,
                "u_used": 1.10 * u_cr,
                "predicted_crack_positions": antinodes,
                "predicted_crack_position": min(antinodes, key=lambda xa: abs(xa - first))
                if first is not None else antinodes[0],
                "dev_crack_position": dev,
                "crack_positions": [c["position"] for c in out["cracks"]],
                "crack_times": [c["time"] for c in out["cracks"]],
            }

        sweeps = {
            "momentum": detection_sweep(props_ref, "momentum", dem_ref["u_cr_dem_momentum"],
                                        gain=self.gain_lbm, n_nodes=self.n_nodes,
                                        t_end=self.t_end_sweep, seed_mode=seeds["momentum"]),
            "hydro": detection_sweep(props_ref, "hydro", dem_ref["u_cr_dem_hydro"],
                                     gain=self.gain_lbm, n_nodes=self.n_nodes,
                                     t_end=self.t_end_sweep, seed_mode=seeds["hydro"]),
        }
        return {
            "case": {
                "h_i_samples": list(self.thicknesses),
                "water_depth_m": props_ref.h,
                "channel_length_m": props_ref.L,
                "E_ice_Pa": props_ref.E,
                "nu": props_ref.nu,
                "sigma_f_Pa": props_ref.sigma_f,
                "rho_w": props_ref.rho_w,
                "rho_i": props_ref.rho_i,
                "g": props_ref.g,
                "dem_nodes": self.n_nodes,
                "gain_lbm": self.gain_lbm,
                "n_modes": self.n_modes,
            },
            "rows": rows,
            "reference": {"analytic": ref, "dem": dem_ref},
            "crack_sequences": sequences,
            "detection_sweeps": sweeps,
        }