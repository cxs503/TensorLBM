"""Wageningen B-series propeller performance module for TensorLBM.

Implements the Wageningen B-series open-water propeller performance model
based on the polynomial regression of Oosterveld & van Oossanen (1975).
Also provides a simple propeller disk actuator mask for LBM simulations.

Polynomial form (Rn = 2×10⁶):
    KT = Σ C · J^s · (P/D)^t · (Ae/A0)^u · Z^v
    KQ = Σ C · J^s · (P/D)^t · (Ae/A0)^u · Z^v

References
----------
Oosterveld, M.W.C. and van Oossanen, P. (1975). "Further computer-analyzed
data of the Wageningen B-screw series." *International Shipbuilding
Progress*, 22 (251), 3–14.

Bernitsas, M.M., Ray, D., Kinley, P. (1981). "KT, KQ and Efficiency Curves
for the Wageningen B-Series Propellers." University of Michigan Report 237.

Carlton, J. (2007). *Marine Propellers and Propulsion*. 2nd ed. Table 6.6.

Public API
----------
- :func:`wageningen_b_series`   – compute KT, KQ, η₀ from J, P/D, Ae/A0, Z.
- :func:`optimal_advance_ratio` – J at maximum efficiency for given P/D.
- :func:`propeller_design`      – size a propeller for given thrust/speed.
- :func:`propeller_disk_mask`   – circular disk obstacle mask for LBM.
- :func:`plot_b_series_curves`  – open-water diagram (KT, 10KQ, η₀ vs J).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure

__all__ = [
    "wageningen_b_series",
    "optimal_advance_ratio",
    "propeller_design",
    "propeller_disk_mask",
    "plot_b_series_curves",
]


# ---------------------------------------------------------------------------
# Wageningen B-series polynomial coefficients (Oosterveld & van Oossanen 1975,
# digitisation from Carlton 2007 / Bernitsas 1981, verified against published
# open-water diagrams for B4-40, B4-70, B5-60).
# Each row: (C, s_J, s_PD, s_EAR, s_Z)
# KT = Σ C · J^s_J · (P/D)^s_PD · (Ae/A0)^s_EAR · Z^s_Z
# ---------------------------------------------------------------------------

_KT_COEFFS = np.array([
    ( 0.00880496, 0, 0, 0, 0),
    (-0.204554,   1, 0, 0, 0),
    ( 0.166351,   0, 1, 0, 0),
    ( 0.158114,   0, 2, 0, 0),
    (-0.147581,   2, 0, 1, 0),
    (-0.481497,   1, 1, 1, 0),
    ( 0.415437,   0, 2, 1, 0),
    ( 0.0144043,  0, 0, 0, 1),
    (-0.0530054,  2, 0, 0, 1),
    ( 0.0143481,  0, 1, 0, 1),
    ( 0.0606826,  1, 1, 0, 1),
    (-0.0125894,  0, 0, 1, 1),
    ( 0.0109689,  1, 0, 1, 1),
    (-0.133698,   0, 3, 0, 0),
    ( 0.00638407, 0, 6, 0, 0),
    (-0.00132718, 2, 6, 0, 0),
    ( 0.168496,   3, 0, 1, 0),
    (-0.0507214,  0, 0, 2, 0),
    ( 0.0854559,  2, 0, 2, 0),
    (-0.0504475,  3, 0, 2, 0),
    ( 0.010465,   1, 6, 2, 0),
    (-0.00648272, 2, 6, 2, 0),
    (-0.00841728, 0, 3, 0, 1),
    ( 0.0168424,  1, 3, 0, 1),
    (-0.00102296, 3, 3, 0, 1),
    (-0.0317791,  0, 3, 1, 1),
    ( 0.018604,   1, 0, 2, 1),
    (-0.00410798, 0, 2, 2, 1),
    (-0.000606848, 0, 0, 0, 2),
    (-0.0049819,  1, 0, 0, 2),
    ( 0.0025983,  2, 0, 0, 2),
    (-0.000560528, 3, 0, 0, 2),
    (-0.00163652, 1, 2, 0, 2),
    (-0.000328787, 1, 6, 0, 2),
    ( 0.000116502, 2, 6, 0, 2),
    ( 0.000690904, 0, 0, 1, 2),
    ( 0.00421749, 0, 3, 1, 2),
    ( 0.0000565229, 3, 6, 1, 2),
    (-0.00146564, 0, 3, 2, 2),
], dtype=np.float64)

_KQ_COEFFS = np.array([
    ( 0.00379368,   0, 0, 0, 0),
    ( 0.00886523,   2, 0, 0, 0),
    (-0.032241,     1, 1, 0, 0),
    ( 0.00344778,   0, 2, 0, 0),
    (-0.0408811,    0, 1, 1, 0),
    (-0.108009,     1, 1, 1, 0),
    (-0.0885381,    2, 1, 1, 0),
    ( 0.188561,     0, 2, 1, 0),
    (-0.00370871,   1, 0, 0, 1),
    ( 0.00513696,   0, 1, 0, 1),
    ( 0.0209449,    1, 1, 0, 1),
    ( 0.00474319,   2, 1, 0, 1),
    (-0.00723408,   2, 0, 1, 1),
    ( 0.00438388,   1, 1, 1, 1),
    (-0.0269403,    0, 2, 1, 1),
    ( 0.0558082,    3, 0, 1, 0),
    ( 0.0161886,    0, 3, 1, 0),
    ( 0.00318086,   1, 3, 1, 0),
    ( 0.015896,     0, 0, 2, 0),
    ( 0.0471729,    1, 0, 2, 0),
    ( 0.0196283,    3, 0, 2, 0),
    (-0.0502782,    0, 1, 2, 0),
    (-0.030055,     3, 1, 2, 0),
    ( 0.0417122,    2, 2, 2, 0),
    (-0.0397722,    0, 3, 2, 0),
    (-0.00350024,   0, 6, 2, 0),
    (-0.0106854,    3, 0, 0, 1),
    ( 0.00110903,   3, 3, 0, 1),
    (-0.000313912,  0, 6, 0, 1),
    ( 0.0035985,    3, 0, 1, 1),
    (-0.00142121,   0, 6, 1, 1),
    (-0.00383637,   1, 0, 2, 1),
    ( 0.0126803,    0, 2, 2, 1),
    (-0.00318278,   2, 3, 2, 1),
    ( 0.00334268,   0, 6, 2, 1),
    (-0.00183491,   1, 1, 0, 2),
    ( 0.000112451,  3, 2, 0, 2),
    (-0.0000297228, 3, 6, 0, 2),
    ( 0.000269551,  1, 0, 1, 2),
    ( 0.00083265,   2, 0, 1, 2),
    ( 0.00155334,   0, 2, 1, 2),
    ( 0.000302683,  0, 6, 1, 2),
    (-0.0001843,    0, 0, 2, 2),
    (-0.000425399,  0, 3, 2, 2),
    ( 0.0000869243, 3, 3, 2, 2),
    (-0.000465899,  0, 6, 2, 2),
    ( 0.0000554194, 1, 6, 2, 2),
], dtype=np.float64)


def _b_poly(coeffs: np.ndarray, J: float, P_D: float, Ae_A0: float, Z: int) -> float:
    """Evaluate the B-series polynomial for a single operating point."""
    C = coeffs[:, 0]
    s = coeffs[:, 1]
    t = coeffs[:, 2]
    u = coeffs[:, 3]
    v = coeffs[:, 4]
    return float(np.sum(C * (J ** s) * (P_D ** t) * (Ae_A0 ** u) * (float(Z) ** v)))


def wageningen_b_series(
    J: float,
    P_D: float,
    Ae_A0: float,
    Z: int,
    *,
    rho: float = 1025.0,
    n: float | None = None,
    D: float | None = None,
) -> dict:
    """Compute open-water performance of a Wageningen B-series propeller.

    Uses the polynomial regression of Oosterveld & van Oossanen (1975) at
    Reynolds number Rn = 2×10⁶ (full-scale correction per Lerbs not applied).

    Parameters
    ----------
    J     : Advance ratio J = Va / (n·D).  Typical range: 0.0 – 1.5.
    P_D   : Pitch ratio P/D.  Typical range: 0.5 – 1.4.
    Ae_A0 : Expanded blade area ratio.  Typical range: 0.30 – 1.05.
    Z     : Number of blades.  Integer 2 – 7.
    rho   : Water density [kg/m³] (for dimensional outputs; default 1025).
    n     : Rotational speed [rev/s] (required for dimensional forces).
    D     : Propeller diameter [m] (required for dimensional forces).

    Returns
    -------
    dict with keys:

    - ``J``, ``P_D``, ``Ae_A0``, ``Z`` : inputs
    - ``KT``    : thrust coefficient
    - ``KQ``    : torque coefficient
    - ``eta_0`` : open-water efficiency J·KT / (2π·KQ)
    - ``T_N``   : thrust [N] if ``n`` and ``D`` are provided
    - ``Q_Nm``  : torque [N·m] if ``n`` and ``D`` are provided
    - ``P_kW``  : delivered power [kW] if ``n`` and ``D`` are provided
    """
    J_f  = float(J)
    PD   = float(P_D)
    EAR  = float(Ae_A0)
    Z_i  = int(Z)

    KT = max(_b_poly(_KT_COEFFS, J_f, PD, EAR, Z_i), 0.0)
    KQ = max(_b_poly(_KQ_COEFFS, J_f, PD, EAR, Z_i), 1e-12)

    if J_f > 0.0 and KQ > 0.0:
        eta_0 = J_f * KT / (2.0 * math.pi * KQ)
    else:
        eta_0 = 0.0

    result: dict = {
        "J": J_f,
        "P_D": PD,
        "Ae_A0": EAR,
        "Z": Z_i,
        "KT": round(KT, 6),
        "KQ": round(KQ, 6),
        "eta_0": round(eta_0, 4),
    }

    if n is not None and D is not None:
        n_f = float(n)
        D_f = float(D)
        T_N  = float(rho) * n_f**2 * D_f**4 * KT
        Q_Nm = float(rho) * n_f**2 * D_f**5 * KQ
        P_kW = 2.0 * math.pi * n_f * Q_Nm / 1000.0
        result["T_N"]  = round(T_N,  2)
        result["Q_Nm"] = round(Q_Nm, 4)
        result["P_kW"] = round(P_kW, 4)

    return result


def optimal_advance_ratio(
    P_D: float,
    Ae_A0: float,
    Z: int,
    *,
    J_range: tuple[float, float] = (0.01, 1.4),
    n_points: int = 200,
) -> dict:
    """Find the advance ratio J that maximises open-water efficiency η₀.

    Parameters
    ----------
    P_D, Ae_A0, Z : Propeller parameters.
    J_range   : Search range for J.
    n_points  : Number of evaluation points.

    Returns
    -------
    dict with keys: J_opt, eta_max, KT_at_Jopt, KQ_at_Jopt.
    """
    J_arr = np.linspace(J_range[0], J_range[1], n_points)
    best_eta = -1.0
    best_J   = float(J_arr[0])
    for J_val in J_arr:
        res = wageningen_b_series(float(J_val), P_D, Ae_A0, Z)
        if res["eta_0"] > best_eta:
            best_eta = res["eta_0"]
            best_J   = float(J_val)
    best_res = wageningen_b_series(best_J, P_D, Ae_A0, Z)
    return {
        "J_opt":     round(best_J, 4),
        "eta_max":   round(best_eta, 4),
        "KT_at_Jopt": best_res["KT"],
        "KQ_at_Jopt": best_res["KQ"],
        "P_D":   P_D,
        "Ae_A0": Ae_A0,
        "Z":     Z,
    }


def propeller_design(
    thrust_n: float,
    Va_ms: float,
    P_D: float,
    Ae_A0: float,
    Z: int,
    n_rps: float,
    *,
    rho: float = 1025.0,
) -> dict:
    """Size a propeller for a given required thrust and advance speed.

    Given required thrust, advance speed, and propeller geometry (P/D,
    Ae/A0, Z), finds the diameter delivering the required thrust at the
    given shaft speed and reports the full performance point.

    Parameters
    ----------
    thrust_n  : Required thrust [N].
    Va_ms     : Advance speed [m/s] (ship speed × (1 − wake fraction)).
    P_D       : Design pitch ratio.
    Ae_A0     : Blade area ratio.
    Z         : Number of blades.
    n_rps     : Shaft speed [rev/s].
    rho       : Water density [kg/m³].

    Returns
    -------
    dict with J, KT, KQ, eta_0, D_m, T_N, Q_Nm, P_kW.
    """
    opt   = optimal_advance_ratio(P_D, Ae_A0, Z)
    J_des = opt["J_opt"]
    KT_des = opt["KT_at_Jopt"]

    D_m = Va_ms / (n_rps * J_des) if (n_rps * J_des) > 0.0 else 1.0

    T_actual = rho * n_rps**2 * D_m**4 * KT_des
    if T_actual > 0.0:
        D_m = D_m * (thrust_n / T_actual) ** 0.25
        J_actual = Va_ms / (n_rps * D_m) if (n_rps * D_m) > 0.0 else J_des
        res = wageningen_b_series(J_actual, P_D, Ae_A0, Z, rho=rho, n=n_rps, D=D_m)
    else:
        res = wageningen_b_series(J_des, P_D, Ae_A0, Z, rho=rho, n=n_rps, D=D_m)

    res["D_m"]              = round(D_m, 4)
    res["thrust_required_N"] = thrust_n
    return res


def propeller_disk_mask(
    nx: int,
    ny: int,
    nz: int,
    diameter_lu: float,
    thickness_lu: float = 2.0,
    *,
    cx: float | None = None,
    cy: float | None = None,
    z_centre: float | None = None,
    axis: str = "z",
) -> np.ndarray:
    """Generate a thin circular disk obstacle mask for an LBM actuator disc.

    This approximates a propeller as a thin permeable disc for body-force
    actuator-disc LBM simulations.

    Parameters
    ----------
    nx, ny, nz    : Grid dimensions.
    diameter_lu   : Disc diameter (lattice units).
    thickness_lu  : Disc thickness (lattice units, default 2).
    cx, cy        : Disc centre in x-y plane (default: grid centre).
    z_centre      : Disc centre along z-axis (default: nz/2).
    axis          : Normal axis of the disc: ``"x"``, ``"y"``, or ``"z"``.

    Returns
    -------
    Boolean numpy array of shape (nx, ny, nz).
    """
    cx       = float(cx)       if cx       is not None else nx / 2.0
    cy       = float(cy)       if cy       is not None else ny / 2.0
    z_centre = float(z_centre) if z_centre is not None else nz / 2.0
    r = diameter_lu / 2.0
    t = thickness_lu / 2.0

    x_idx = np.arange(nx, dtype=np.float32)
    y_idx = np.arange(ny, dtype=np.float32)
    z_idx = np.arange(nz, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x_idx, y_idx, z_idx, indexing="ij")

    if axis == "z":
        in_circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        in_thick  = np.abs(zz - z_centre) <= t
    elif axis == "x":
        in_circle = (yy - cy) ** 2 + (zz - z_centre) ** 2 <= r ** 2
        in_thick  = np.abs(xx - cx) <= t
    elif axis == "y":
        in_circle = (xx - cx) ** 2 + (zz - z_centre) ** 2 <= r ** 2
        in_thick  = np.abs(yy - cy) <= t
    else:
        raise ValueError(f"axis must be 'x', 'y' or 'z'; got {axis!r}")

    return in_circle & in_thick


def plot_b_series_curves(
    P_D: float,
    Ae_A0: float,
    Z: int,
    *,
    J_range: tuple[float, float] = (0.01, 1.4),
    n_points: int = 100,
) -> "matplotlib.figure.Figure":
    """Plot KT, 10·KQ and η₀ vs J for a Wageningen B-series propeller.

    Parameters
    ----------
    P_D, Ae_A0, Z : Propeller parameters.
    J_range   : Advance ratio range.
    n_points  : Number of evaluation points.

    Returns
    -------
    matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    J_arr  = np.linspace(J_range[0], J_range[1], n_points)
    KT_arr  = []
    KQ10_arr = []
    eta_arr  = []
    for J_val in J_arr:
        res = wageningen_b_series(float(J_val), P_D, Ae_A0, Z)
        KT_arr.append(res["KT"])
        KQ10_arr.append(res["KQ"] * 10.0)
        eta_arr.append(res["eta_0"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(J_arr, KT_arr,   "b-",  lw=2, label="KT")
    ax.plot(J_arr, KQ10_arr, "g--", lw=2, label="10KQ")
    ax.plot(J_arr, eta_arr,  "r-.", lw=2, label="η₀")

    ax.set_xlabel("Advance ratio J")
    ax.set_ylabel("Coefficient")
    ax.set_title(
        f"Wageningen B{Z}-series  P/D={P_D:.2f}  Ae/A0={Ae_A0:.2f}  Z={Z}"
    )
    ax.legend()
    ax.grid(True, alpha=0.4)
    ax.set_xlim(J_range)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    return fig
