"""Wind-load coefficient table: LBM -> ship-performance-platform integration.

The LBM campaign produces aerodynamic coefficients at a handful of discrete
wind angles.  This module

1. assembles the per-angle run records into a single, self-describing
   coefficient table (``schema = tensorlbm.wind_load_table/v1``);
2. interpolates the table at an arbitrary wind angle (periodic, because the
   superstructure aerodynamics repeat with 360 deg);
3. converts a table lookup into the request payload expected by the
   ship-performance-platform environmental-force endpoint, so the platform can
   replace its empirical wind-load build-up by the LBM table while keeping its
   own ship/current/ice coupling;
4. evaluates the acceptance criteria and the grid-convergence uncertainty.

Reference-area conventions (identical on both sides of the interface)
--------------------------------------------------------------------
``q = 0.5 rho_air U^2``, ``A_F`` = projected frontal area, ``A_L`` =
projected lateral area, ``B`` = superstructure breadth, ``H`` = height above
deck.  Forces and moments are expressed in the *ship* frame:

    X_wind  = C_X(theta) q A_L        (longitudinal, +x = opposition to the
    Y_wind  = C_Y(theta) q A_L         apparent wind)
    K_wind  = C_K(theta) q A_L H      (heeling moment about x)
    N_wind  = C_N(theta) q A_L B      (yaw moment about z)

Note the C_X reference area: the table stores ``C_X`` normalised by ``A_F``
(LBM convention, the physically meaningful one for the drag coefficient) and
additionally the equivalent ``C_X_AL = C_X A_F / A_L`` so that a platform
implementation normalised on the lateral area can be driven directly.  Both
are shipped in the JSON to make the contract unambiguous.
"""

from __future__ import annotations

import json
import math
from typing import Iterable, Sequence

SCHEMA = "tensorlbm.wind_load_table/v1"

REQUIRED_KEYS = (
    "schema", "geometry", "reference", "angles_deg",
    "C_X", "C_Y", "C_N", "C_K", "provenance", "validation",
)


# ---------------------------------------------------------------------------
# Table assembly
# ---------------------------------------------------------------------------

def _mean(vals: Sequence[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def build_table(records: Sequence[dict], *, geometry_name: str,
                dx_by_angle: dict[float, float] | None = None,
                validation: dict | None = None,
                provenance: dict | None = None) -> dict:
    """Assemble per-angle run records into the shipped coefficient table.

    ``records`` are the JSON dicts written by ``scripts/run_lng_wind_load.py``;
    one record per (angle, grid) combination.  When several grids are present
    for an angle the *finest* grid is used for the table (the coarse ones feed
    only the convergence study).
    """
    by_angle: dict[float, list[dict]] = {}
    for r in records:
        by_angle.setdefault(float(r["angle_deg"]), []).append(r)
    if not by_angle:
        raise ValueError("no run records supplied")

    angles = sorted(by_angle)
    cx, cy, cn, ck = [], [], [], []
    cx_std, cy_std = [], []
    dx_used = []
    for a in angles:
        recs = sorted(by_angle[a], key=lambda r: r["geometry"]["dx_m"])
        r = recs[0]                                    # finest available
        c = r["coefficients"]
        cx.append(float(c["C_X"]))
        cy.append(float(c["C_Y"]))
        cn.append(float(c["C_N"]))
        ck.append(float(c["C_K"]))
        cx_std.append(float(c.get("C_X_std", 0.0)))
        cy_std.append(float(c.get("C_Y_std", 0.0)))
        dx_used.append(float(r["geometry"]["dx_m"]))

    ref = records[0]["geometry"]
    a_f, a_l = float(ref["area_frontal_m2"]), float(ref["area_lateral_m2"])
    ext = ref["extent_m"]
    b_ref, h_ref = float(ext[1]), float(ext[2])
    units = records[0]["units"]

    table = {
        "schema": SCHEMA,
        "geometry": {
            "name": geometry_name,
            "description": ("LNG carrier (Moss type) above-deck superstructure, "
                            "stepped (voxel) model: deck house, bridge deck and "
                            "wings, funnel casing, two masts, pipe racks, aft "
                            "Moss dome trunks, foredeck house"),
            "length_m": float(ext[0]),
            "breadth_m": b_ref,
            "height_m": h_ref,
            "dx_m_by_angle": dict(zip(map(str, angles), dx_used)),
            "solid_volume_m3": float(ref["solid_volume_m3"]),
        },
        "reference": {
            "q": "0.5 * rho_air * U_wind^2",
            "rho_air_kg_m3": float(units["rho_phys_kgm3"]),
            "A_F_m2": a_f,
            "A_L_m2": a_l,
            "B_m": b_ref,
            "H_m": h_ref,
            "wind_speed_ms": float(units["u_phys_ms"]),
            "C_X_normalisation": "q * A_F (also C_X_AL = C_X*A_F/A_L supplied)",
            "C_Y_normalisation": "q * A_L",
            "C_K_normalisation": "q * A_L * H",
            "C_N_normalisation": "q * A_L * B",
        },
        "angles_deg": angles,
        "C_X": cx,
        "C_Y": cy,
        "C_N": cn,
        "C_K": ck,
        "C_X_AL": [c * a_f / a_l for c in cx],
        "C_X_std": cx_std,
        "C_Y_std": cy_std,
        "provenance": provenance or {},
        "validation": validation or {},
    }
    validate_table(table)
    return table


def validate_table(table: dict) -> None:
    """Structural validation of a coefficient table (contract enforcement)."""
    for k in REQUIRED_KEYS:
        if k not in table:
            raise ValueError(f"table missing required key: {k}")
    if table["schema"] != SCHEMA:
        raise ValueError(f"unexpected schema {table['schema']!r}")
    n = len(table["angles_deg"])
    if n < 2:
        raise ValueError("at least two wind angles are required")
    for k in ("C_X", "C_Y", "C_N", "C_K"):
        if len(table[k]) != n:
            raise ValueError(f"length mismatch for {k}: {len(table[k])} != {n}")
    if any(b <= a for a, b in zip(table["angles_deg"], table["angles_deg"][1:])):
        raise ValueError("angles_deg must be strictly increasing")


def interpolate(table: dict, angle_deg: float) -> dict:
    """Periodic linear interpolation of the coefficients at ``angle_deg``."""
    validate_table(table)
    a = float(angle_deg) % 360.0
    xs = [float(v) % 360.0 for v in table["angles_deg"]]
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    keys = ("C_X", "C_Y", "C_N", "C_K", "C_X_AL")
    ys = {k: [float(table[k][i]) for i in order] for k in keys}
    if a < xs[0]:
        a += 360.0
    if a >= xs[-1]:
        x0, x1 = xs[-1], xs[0] + 360.0
        w = (a - x0) / (x1 - x0) if x1 != x0 else 0.0
        out = {k: ys[k][-1] + w * (ys[k][0] - ys[k][-1]) for k in keys}
    else:
        j = max(i for i in range(len(xs)) if xs[i] <= a)
        x0, x1 = xs[j], xs[j + 1]
        w = (a - x0) / (x1 - x0)
        out = {k: ys[k][j] + w * (ys[k][j + 1] - ys[k][j]) for k in keys}
    out["angle_deg"] = a
    return out


# ---------------------------------------------------------------------------
# Platform interface
# ---------------------------------------------------------------------------

def to_platform_payload(table: dict, *, angle_deg: float, wind_speed: float,
                        ship: dict) -> dict:
    """Build the ship-platform environmental-force request with LBM coefficients.

    The payload mirrors ``POST /api/v1/calculations/environmental-forces``
    (ship + wind + current + ice) and carries the resolved LBM coefficients in
    the ``wind`` object.  Platform implementations that only accept
    ``V_wind``/``psi_wind`` keep working; implementations that honour
    ``coefficients`` use the LBM table instead of their empirical build-up.
    """
    coef = interpolate(table, angle_deg)
    ref = table["reference"]
    rho = float(ref["rho_air_kg_m3"])
    q = 0.5 * rho * wind_speed ** 2
    a_l = float(ref["A_L_m2"])
    a_f = float(ref["A_F_m2"])
    b, h = float(ref["B_m"]), float(ref["H_m"])
    return {
        "ship": ship,
        "wind": {
            "V_wind": float(wind_speed),
            "psi_wind": float(angle_deg) % 360.0,
            "coefficients": {
                "source": SCHEMA,
                "C_X": coef["C_X"], "C_X_AL": coef["C_X_AL"],
                "C_Y": coef["C_Y"], "C_N": coef["C_N"], "C_K": coef["C_K"],
                "reference": {"A_F_m2": a_f, "A_L_m2": a_l, "B_m": b, "H_m": h},
            },
            "forces": {
                "F_x_N": coef["C_X"] * q * a_f,
                "F_y_N": coef["C_Y"] * q * a_l,
                "M_K_Nm": coef["C_K"] * q * a_l * h,
                "M_N_Nm": coef["C_N"] * q * a_l * b,
            },
        },
    }


def dumps(table: dict, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(table, fh, indent=1, sort_keys=False)


def loads(path: str) -> dict:
    with open(path) as fh:
        table = json.load(fh)
    validate_table(table)
    return table


# ---------------------------------------------------------------------------
# Acceptance criteria
# ---------------------------------------------------------------------------

def relative_error(value: float, reference: float) -> float:
    if reference == 0:
        return float("inf") if value != 0 else 0.0
    return abs(value - reference) / abs(reference)


def acceptance_criteria(*, C_X_0: float, C_X_90: float,
                        ref_C_X_0: float, ref_C_X_90: float,
                        U_GCI: float, St: float, St_ref: float,
                        mass_error: float, pytest_passed: int,
                        table_path: str, reference_class: str) -> dict:
    """Evaluate the machine-checkable acceptance criteria of the campaign.

    ``reference_class`` documents *what* the reference values are (wind-tunnel
    digitisation, published correlation, or engineering build-up); the honest
    origin is carried into the report rather than being implied.
    """
    e0 = relative_error(C_X_0, ref_C_X_0)
    e90 = relative_error(C_X_90, ref_C_X_90)
    eSt = relative_error(St, St_ref)
    crit = {
        "a_C_X_0_within_10pct": {"value": C_X_0, "reference": ref_C_X_0,
                                 "rel_error": e0, "limit": 0.10,
                                 "pass": bool(e0 < 0.10)},
        "b_C_X_90_within_15pct": {"value": C_X_90, "reference": ref_C_X_90,
                                  "rel_error": e90, "limit": 0.15,
                                  "pass": bool(e90 < 0.15)},
        "c_GCI_under_5pct": {"U_GCI": U_GCI, "limit": 0.05,
                             "pass": bool(U_GCI < 0.05)},
        "d_Strouhal_within_10pct": {"value": St, "reference": St_ref,
                                    "rel_error": eSt, "limit": 0.10,
                                    "pass": bool(eSt < 0.10)},
        "e_mass_balance_0p1pct": {"rel_error": mass_error, "limit": 1e-3,
                                  "pass": bool(mass_error < 1e-3)},
        "f_pytest_at_least_8": {"passed": int(pytest_passed), "limit": 8,
                                "pass": bool(pytest_passed >= 8)},
        "g_table_delivered": {"path": table_path,
                              "pass": bool(table_path)},
        "reference_class": reference_class,
        "all_pass": False,
    }
    core = ("a_C_X_0_within_10pct", "b_C_X_90_within_15pct",
            "c_GCI_under_5pct", "d_Strouhal_within_10pct",
            "e_mass_balance_0p1pct", "f_pytest_at_least_8",
            "g_table_delivered")
    crit["all_pass"] = all(crit[k]["pass"] for k in core)
    return crit