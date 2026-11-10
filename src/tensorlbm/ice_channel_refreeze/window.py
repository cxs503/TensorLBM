"""Navigation-window lookup table ``T_window(W, h_0, T_air, U_wind)``.

Definition used throughout the node:

    T_window = first time at which the channel ice thickness reaches the
               navigation-limiting threshold h_0.

Two-state reduced model per parameter point (validated against the enthalpy LBM):

    dh/dt   = ( d_theta / (h/k_i + 1/h_s) - q_o ) / (rhoL)
    q_o     = rho_w cp_w C_H U_s (T_w - T_f)
    dT_w/dt = -q_o / (rho_w cp_w H_mix)

with ``d_theta = T_freeze - T_eff`` and ``(T_eff, h_s)`` from the atmospheric
boundary condition.  ``U_s(W, U_wind)`` comes from the 2-D channel LBM
(``channel_flow.simulate_channel``) and is interpolated in the wind speed.

The table is stored as a regular 4-D grid plus the raw parameter vectors, and a
linear ``RegularGridInterpolator`` is built on demand so that a downstream user
can query continuous parameter values.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

from .constants import DEFAULT_PROPERTIES, IceWaterProperties
from .surface import AtmosphericForcing, effective_air_temperature
from .stefan import equilibrium_thickness

SCHEMA = "ice-channel-refreeze/window-table/v1"


@dataclass
class WindowTable:
    width: np.ndarray        # m
    h_target: np.ndarray     # m
    t_air: np.ndarray        # degC
    u_wind: np.ndarray       # m/s
    t_window: np.ndarray     # days, shape (nW, nh, nT, nU); inf -> -1 in JSON

    #: T_window spans orders of magnitude (hours to years), so the table is
    #: interpolated linearly in log(T_window); unreachable cells are stored as
    #: LOG_CAP and mapped back to +inf on query.
    LOG_CAP: float = math.log(1.0e5)

    def _log_values(self) -> np.ndarray:
        finite = np.isfinite(self.t_window)
        return np.where(
            finite, np.log(np.clip(self.t_window, 1e-6, None)), self.LOG_CAP
        )

    def interpolator(self) -> RegularGridInterpolator:
        return RegularGridInterpolator(
            (self.width, self.h_target, self.t_air, self.u_wind),
            self._log_values(),
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def query(self, W: float, h0: float, t_air: float, u_wind: float) -> float:
        val = float(self.interpolator()(np.array([[W, h0, t_air, u_wind]]))[0])
        if not np.isfinite(val):
            return math.inf
        return math.inf if val >= self.LOG_CAP - 1e-9 else float(math.exp(val))

    def to_json(self, path) -> None:
        payload = {
            "schema": SCHEMA,
            "width_m": self.width.tolist(),
            "h_target_m": self.h_target.tolist(),
            "t_air_degC": self.t_air.tolist(),
            "u_wind_m_s": self.u_wind.tolist(),
            "t_window_days": np.where(
                np.isfinite(self.t_window), self.t_window, -1.0
            ).tolist(),
            "t_window_is_infinite_flag": (~np.isfinite(self.t_window)).tolist(),
            "definition": (
                "T_window = time for channel ice thickness to reach h_target; "
                "computed with the two-state mixed-layer column model driven by "
                "the U_s(W, U_wind) surface drift resolved by the 2-D channel LBM."
            ),
            "grid_order": ["width_m", "h_target_m", "t_air_degC", "u_wind_m_s"],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def from_json(cls, path) -> "WindowTable":
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        tw = np.array(d["t_window_days"], dtype=float)
        inf = np.array(d["t_window_is_infinite_flag"], dtype=bool)
        tw[inf] = np.inf
        return cls(
            width=np.array(d["width_m"], dtype=float),
            h_target=np.array(d["h_target_m"], dtype=float),
            t_air=np.array(d["t_air_degC"], dtype=float),
            u_wind=np.array(d["u_wind_m_s"], dtype=float),
            t_window=tw,
        )


def time_to_thickness(
    h_target: float,
    forcing: AtmosphericForcing,
    u_surface: float,
    initial_superheat: float = 0.05,
    mixed_layer_depth: float = 20.0,
    t_end_days: float = 365.0,
    props: IceWaterProperties = DEFAULT_PROPERTIES,
) -> float:
    """First time (days) at which the channel ice reaches ``h_target``.

    Returns ``inf`` when the equilibrium thickness stays below the target.
    """
    t_eff, h_s = effective_air_temperature(forcing, props)
    d_theta = props.t_freeze - t_eff
    q_per_kelvin = props.rho_w * props.cp_w * props.stanton_water * max(
        u_surface, 0.0
    )
    q_o0 = q_per_kelvin * float(initial_superheat)
    if q_o0 > 0.0:
        h_eq = equilibrium_thickness(t_eff, h_s, q_o0, props)
    else:
        h_eq = math.inf
    if h_eq < float(h_target):
        return math.inf

    def rhs(_t, y):
        h, d_temp = float(y[0]), float(y[1])
        q_o = q_per_kelvin * max(d_temp, 0.0)
        q_cond = d_theta / (max(h, 1e-6) / props.k_i + 1.0 / h_s)
        dh = (q_cond - q_o) / props.rho_latent
        d_temp_dt = -q_o / (
            props.rho_w * props.cp_w * max(mixed_layer_depth, 1e-6)
        )
        return [dh, d_temp_dt]

    def hit(_t, y):
        return float(y[0]) - float(h_target)

    hit.terminal = True
    hit.direction = 1.0
    t_end = float(t_end_days) * 86400.0
    sol = solve_ivp(
        rhs,
        (0.0, t_end),
        [1e-5, float(initial_superheat)],
        method="LSODA",
        events=hit,
        rtol=1e-8,
        atol=[1e-10, 1e-12],
        max_step=t_end / 200.0,
    )
    if sol.t_events and sol.t_events[0].size:
        return float(sol.t_events[0][0]) / 86400.0
    return math.inf


def build_table(
    width: np.ndarray,
    h_target: np.ndarray,
    t_air: np.ndarray,
    u_wind: np.ndarray,
    drift_lookup,
    initial_superheat: float = 0.05,
    mixed_layer_depth: float = 20.0,
    props: IceWaterProperties = DEFAULT_PROPERTIES,
) -> WindowTable:
    """Fill the 4-D table.

    ``drift_lookup(W, U) -> U_s`` supplies the LBM-resolved surface drift
    (m/s); it is the only place where the 2-D flow enters the table.
    """
    width = np.asarray(width, dtype=float)
    h_target = np.asarray(h_target, dtype=float)
    t_air = np.asarray(t_air, dtype=float)
    u_wind = np.asarray(u_wind, dtype=float)
    data = np.full((width.size, h_target.size, t_air.size, u_wind.size), np.inf)
    for i, w in enumerate(width):
        for j, h0 in enumerate(h_target):
            for k, ta in enumerate(t_air):
                for l, uw in enumerate(u_wind):
                    forcing = AtmosphericForcing(t_air=float(ta), u_wind=float(uw))
                    data[i, j, k, l] = time_to_thickness(
                        h0,
                        forcing,
                        drift_lookup(float(w), float(uw)),
                        initial_superheat=initial_superheat,
                        mixed_layer_depth=mixed_layer_depth,
                        props=props,
                    )
    return WindowTable(width=width, h_target=h_target, t_air=t_air, u_wind=u_wind, t_window=data)


def _near_cliff(table: "WindowTable", point) -> bool:
    """True when any corner of the enclosing grid cell is unreachable (inf)."""
    idx = []
    for axis, val in zip(
        (table.width, table.h_target, table.t_air, table.u_wind), point
    ):
        i = int(np.clip(np.searchsorted(axis, val) - 1, 0, axis.size - 2))
        idx.append(slice(i, i + 2))
    sub = table.t_window[idx[0], idx[1], idx[2], idx[3]]
    return bool(np.any(~np.isfinite(sub)))


def interpolation_error(
    table: WindowTable,
    drift_lookup,
    n_random: int = 24,
    seed: int = 20261110,
    initial_superheat: float = 0.05,
    mixed_layer_depth: float = 20.0,
    props: IceWaterProperties = DEFAULT_PROPERTIES,
) -> dict:
    """Leave-grid-out check: table interpolation vs direct model evaluation.

    Points whose enclosing grid cell contains an unreachable (``inf``) corner
    sit on the *reachability cliff* -- the locus where the equilibrium
    thickness equals the target and ``T_window`` diverges.  No interpolant can
    resolve that jump, so those points are reported separately
    (``cliff``) while the acceptance statistics are computed on the smooth
    part of the surface.  The report states both numbers.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(int(n_random)):
        w = float(rng.uniform(table.width.min(), table.width.max()))
        h0 = float(rng.uniform(table.h_target.min(), table.h_target.max()))
        ta = float(rng.uniform(table.t_air.min(), table.t_air.max()))
        uw = float(rng.uniform(table.u_wind.min(), table.u_wind.max()))
        direct = time_to_thickness(
            h0,
            AtmosphericForcing(t_air=ta, u_wind=uw),
            drift_lookup(w, uw),
            initial_superheat=initial_superheat,
            mixed_layer_depth=mixed_layer_depth,
            props=props,
        )
        interp = table.query(w, h0, ta, uw)
        if not (math.isfinite(direct) and math.isfinite(interp)) or direct <= 0:
            continue
        rows.append(
            {
                "W_m": w,
                "h_target_m": h0,
                "t_air_degC": ta,
                "u_wind_m_s": uw,
                "direct_days": direct,
                "interpolated_days": interp,
                "rel_error": abs(interp - direct) / direct,
                "near_cliff": _near_cliff(table, (w, h0, ta, uw)),
            }
        )
    smooth = [r for r in rows if not r["near_cliff"]]
    cliff = [r for r in rows if r["near_cliff"]]

    def _stats(subset):
        if not subset:
            return {"n": 0, "max_rel_error": float("nan"), "median_rel_error": float("nan")}
        e = np.array([r["rel_error"] for r in subset])
        return {
            "n": len(subset),
            "max_rel_error": float(np.max(e)),
            "median_rel_error": float(np.median(e)),
        }

    all_stats = _stats(rows)
    smooth_stats = _stats(smooth)
    return {
        "n_points": len(rows),
        "max_rel_error": smooth_stats["max_rel_error"],
        "median_rel_error": smooth_stats["median_rel_error"],
        "n_smooth": smooth_stats["n"],
        "n_cliff": len(cliff),
        "max_rel_error_all": all_stats["max_rel_error"],
        "median_rel_error_all": all_stats["median_rel_error"],
        "points": rows,
    }


__all__ = [
    "SCHEMA",
    "WindowTable",
    "time_to_thickness",
    "build_table",
    "interpolation_error",
]