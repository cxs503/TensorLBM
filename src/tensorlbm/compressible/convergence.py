"""Mesh-convergence and shock-signature verification utilities (IN50-W4-N04).

The functions here turn a *family* of solutions, obtained on systematically
refined meshes, into machine-checkable verification quantities.  Nothing in
this module solves a PDE: it is the metric layer used by
``scripts/run_in50_w4_n04.py`` and ``tests/test_compressible_lbm.py``.

Contents
--------
* :func:`mesh_size` -- representative cell size ``h`` of an O-grid, used as the
  refinement parameter of a grid family (monotone in *both* index directions).
* :func:`richardson_3grid` -- observed order of accuracy ``p``, Richardson
  extrapolated value and Roache grid-convergence index (GCI) for a three-grid
  family with a *uniform* refinement ratio; monotonicity is reported explicitly
  (a non-monotone triplet makes the triplet-based order meaningless, which is
  how the IN421-W2-N03 study ended up with "GCI not computable").
* :func:`shock_strength` -- terminal-shock pressure ratio measured from the
  surface pressure distribution, compared with the Rankine-Hugoniot prediction
  built from the *pre-shock plateau* (never from a single cell inside the
  smeared shock band), plus the shock thickness in grid cells.
* :func:`nwave_metrics` -- N-wave source signature descriptors (front
  compression, rear expansion, impulse, asymmetry, zero crossing).

References
----------
* Roache, P.J., "Verification and Validation in Computational Science and
  Engineering", Hermosa, 1998 (GCI, observed order).
* Richardson, L.F., Phil. Trans. R. Soc. A 226, 299-361, 1927.
* Anderson, J.D., "Modern Compressible Flow", 3rd ed., 2003 (oblique shock,
  Ackeret / linearised supersonic theory, N-wave discussion in ch. 9-11).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

GAMMA = 1.4


# ---------------------------------------------------------------------------
# grid family
# ---------------------------------------------------------------------------
def mesh_size(n_cells: int, chord: float = 1.0) -> float:
    """Representative cell size ``h = sqrt(chord**2 / n_cells)``.

    Used as the refinement parameter of a structured grid family, so that a
    family refined by a factor ``r`` in *both* directions has ``h`` reduced by
    ``r`` (and ``n_cells`` grows by ``r**2``).
    """
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")
    return float(math.sqrt(chord * chord / n_cells))


def refinement_ratio(n_cells_coarse: int, n_cells_fine: int) -> float:
    """Uniform refinement ratio implied by two cell counts, ``r = h_c/h_f``."""
    return mesh_size(n_cells_coarse) / mesh_size(n_cells_fine)


# ---------------------------------------------------------------------------
# Richardson extrapolation / GCI
# ---------------------------------------------------------------------------
def richardson_3grid(f_coarse: float, f_medium: float, f_fine: float,
                     r: float, safety: float = 1.25,
                     p_min: float = 0.1, p_max: float = 6.0) -> Dict[str, object]:
    """Observed order, Richardson extrapolation and GCI of a three-grid family.

    Parameters
    ----------
    f_coarse, f_medium, f_fine
        The quantity on the coarse, medium and fine grids (``h``, ``h/r``,
        ``h/r**2``).
    r
        Uniform refinement ratio (``r > 1``).
    safety
        Roache safety factor (1.25 for a three-grid study).

    Returns
    -------
    dict with ``observed_order`` (``None`` if the triplet is not monotone),
    ``extrapolated``, ``rel_change_medium_fine``, ``gci_fine`` (fraction),
    ``monotone`` and ``status`` (``"CONVERGED"`` / ``"NON_MONOTONE"``).
    A non-monotone triplet is reported honestly instead of silently falling
    back to an assumed order: the caller decides whether to accept ``p = 2``.
    """
    f1, f2, f3 = float(f_coarse), float(f_medium), float(f_fine)
    if r <= 1.0:
        raise ValueError("refinement ratio r must exceed 1")
    d12 = f2 - f1
    d23 = f3 - f2
    scale = max(abs(f1), abs(f2), abs(f3), 1e-30)
    monotone = (abs(d12) > 1e-14 * scale) and (abs(d23) > 1e-14 * scale) \
        and (d12 * d23 > 0.0)
    out: Dict[str, object] = {
        "f_coarse": f1, "f_medium": f2, "f_fine": f3, "r": float(r),
        "monotone": bool(monotone),
        "rel_change_coarse_medium": (f2 - f1) / scale,
        "rel_change_medium_fine": (f3 - f2) / scale,
        "observed_order": None, "extrapolated": None,
        "gci_fine": None, "status": "NON_MONOTONE",
    }
    if not monotone:
        return out
    p = math.log(abs(d12 / d23)) / math.log(r)
    out["observed_order_raw"] = float(p)
    p_clipped = float(min(max(p, p_min), p_max))
    out["observed_order"] = float(p_clipped) if abs(p_clipped - p) > 1e-12 else float(p)
    out["order_clipped"] = bool(abs(p_clipped - p) > 1e-12)
    ext = f3 + (f3 - f2) / (r ** p_clipped - 1.0)
    out["extrapolated"] = float(ext)
    if abs(ext) > 1e-30:
        out["rel_error_fine_vs_extrapolated"] = (f3 - ext) / abs(ext)
    out["gci_fine"] = float(safety * abs((f3 - f2) / (f3 if abs(f3) > 1e-30 else 1.0))
                            / (r ** p_clipped - 1.0))
    out["status"] = "CONVERGED"
    return out


# ---------------------------------------------------------------------------
# Rankine-Hugoniot helpers
# ---------------------------------------------------------------------------
def normal_shock_pressure_ratio(mach1: float, gamma: float = GAMMA) -> float:
    """Exact normal-shock ``p2/p1`` for upstream Mach ``mach1`` (gamma=1.4)."""
    m2 = mach1 * mach1
    return 1.0 + 2.0 * gamma / (gamma + 1.0) * (m2 - 1.0)


def normal_shock_downstream_mach(mach1: float, gamma: float = GAMMA) -> float:
    """Exact normal-shock downstream Mach number (``< 1`` for ``mach1 > 1``)."""
    m2 = mach1 * mach1
    return math.sqrt((1.0 + 0.5 * (gamma - 1.0) * m2)
                     / (gamma * m2 - 0.5 * (gamma - 1.0)))


def isentropic_mach_from_pressure(p_over_pinf: float,
                                  mach_inf: float,
                                  gamma: float = GAMMA) -> float:
    """Invert the isentropic relation ``p/p_inf = (1+0.2M^2)^(gamma/(gamma-1))``.

    ``p/p_inf`` is the *total-to-static* ratio along a streamline, i.e. the
    local static pressure is normalised by the freestream **stagnation**
    pressure, so that ``M = mach_inf`` returns ``mach_inf``.
    """
    if p_over_pinf <= 0.0:
        return float("nan")
    stag = (1.0 + 0.5 * (gamma - 1.0) * mach_inf ** 2)
    ratio = stag * p_over_pinf
    if ratio <= 0.0:
        return float("nan")
    val = 2.0 / (gamma - 1.0) * (ratio ** ((gamma - 1.0) / gamma) - 1.0)
    # Cp below -(stagnation pressure ratio) is outside the isentropic map (e.g. an
    # unconverged or post-shock sample); return NaN instead of raising so that a
    # bad station can never abort an entire verification run.
    return float(math.sqrt(val)) if val >= 0.0 else float("nan")


# ---------------------------------------------------------------------------
# shock strength from a surface pressure distribution
# ---------------------------------------------------------------------------
def shock_strength(x_surf: Sequence[float], cp: Sequence[float],
                   mach_inf: float, gamma: float = GAMMA,
                   pre_window: Optional[Tuple[float, float]] = None,
                   post_window: Tuple[float, float] = (0.02, 0.06),
                   ) -> Dict[str, object]:
    """Terminal-shock position and strength on an airfoil surface.

    Workflow (deliberately different from "where is dCp/dx largest"):

    1. locate the supersonic pocket with the isentropic ``Cp -> M`` map of
       :func:`~tensorlbm.compressible.transonic_aero.surface_mach_from_cp`;
    2. take the **pre-shock plateau** as the mean ``Cp`` of the fully supersonic
       stations ahead of pocket closure (a few stations after the pocket starts
       and a quarter of the pocket excluded, so the captured ramp is not
       averaged in);
    3. take the **post-shock plateau** as the mean ``Cp`` of the subsonic
       stations immediately behind the shock (``post_window`` chords);
    4. convert ``Cp`` to static pressure (``p/p_inf = 1 + gamma/2 M_inf^2 Cp``,
       exact, no isentropic assumption) and compare the measured ``p2/p1`` with
       the exact Rankine-Hugoniot value built from the pre-shock plateau Mach
       number.

    Plateau standard deviations are reported with every mean, so an unresolved
    (non-flat) state is visible instead of being hidden behind an average.
    """
    from .transonic_aero import surface_mach_from_cp

    x = np.asarray(x_surf, float)
    c = np.asarray(cp, float)
    order = np.argsort(x)
    xs, cs = x[order], c[order]
    q_inf = 0.5 * gamma * mach_inf ** 2          # p_inf = rho_inf = 1 -> q = g M^2/2
    m = surface_mach_from_cp(cs, mach_inf, gamma)
    ok = np.isfinite(m)
    if not ok.any():
        return {"status": "NO_DATA"}
    sup = np.flatnonzero(ok & (m > 1.0))
    if sup.size == 0:
        return {"status": "SUBCRITICAL",
                "max_surface_mach": float(np.nanmax(m[ok]))}
    i0, i1 = int(sup[0]), int(sup[-1])
    n_pl = max(3, (i1 - i0) // 4)
    a = min(i0 + 2, i1 - 1)
    b = max(a + 1, i1 - n_pl)
    if pre_window is not None:
        sel = (xs >= pre_window[0]) & (xs <= pre_window[1]) & (m > 1.0)
        idx = np.flatnonzero(sel)
        if idx.size >= 3:
            a, b = int(idx[0]), int(idx[-1]) + 1
    pre_cp = float(np.mean(cs[a:b]))
    pre_std = float(np.std(cs[a:b]))
    m1 = float(surface_mach_from_cp(np.array([pre_cp]), mach_inf, gamma)[0])

    x_shock = float(xs[i1])
    sel_post = (xs > x_shock) & (xs <= x_shock + post_window[1]) & ok & (m < 1.0)
    if sel_post.sum() < 3:
        sel_post = (xs > x_shock) & (xs <= x_shock + post_window[1]) & ok
    post_cp = float(np.mean(cs[sel_post])) if sel_post.any() else float("nan")
    post_std = float(np.std(cs[sel_post])) if sel_post.any() else float("nan")

    p1 = 1.0 + q_inf * pre_cp
    p2 = 1.0 + q_inf * post_cp
    measured = p2 / p1
    rh = normal_shock_pressure_ratio(m1, gamma) if m1 > 1.0 else float("nan")
    dcp = np.abs(np.gradient(cs, xs))
    n_band = int(np.count_nonzero(dcp > 0.05 * dcp.max())) if dcp.max() > 0 else 0
    m2_rh = normal_shock_downstream_mach(m1, gamma) if m1 > 1.0 else float("nan")
    return {
        "status": "SUPERCRITICAL",
        "i_pocket_start": i0, "i_pocket_end": i1,
        "x_pocket_start": float(xs[i0]), "x_shock": x_shock,
        "max_surface_mach": float(np.nanmax(m[ok])),
        "pre_shock_mach_plateau": float(m1),
        "pre_plateau_cp": pre_cp, "pre_plateau_cp_std": pre_std,
        "pre_plateau_n": int(b - a),
        "post_plateau_cp": post_cp, "post_plateau_cp_std": post_std,
        "post_plateau_n": int(sel_post.sum()),
        "measured_p_ratio": float(measured),
        "rankine_hugoniot_p_ratio": float(rh),
        "strength_rel_error": float(measured / rh - 1.0)
        if rh == rh and rh > 0 else float("nan"),
        "rh_downstream_mach": float(m2_rh),
        "shock_band_stations": n_band,
    }


def isentropic_mach_from_cp_local(cp: float, mach_inf: float,
                                  gamma: float = GAMMA) -> float:
    """Local Mach from a surface ``Cp`` (delegates to the canonical map)."""
    from .transonic_aero import surface_mach_from_cp
    if not np.isfinite(cp):
        return float("nan")
    return float(surface_mach_from_cp(np.array([cp]), mach_inf, gamma)[0])


# ---------------------------------------------------------------------------
# N-wave signature metrics
# ---------------------------------------------------------------------------
def nwave_metrics(x: Sequence[float], p_ratio: Sequence[float],
                  chord: float = 1.0) -> Dict[str, object]:
    """Describe the N-wave **source** signature along a wedge surface.

    For a double wedge at ``Ma > 1`` the wall pressure is
    ``compression plateau -> expansion plateau``; the resulting far-field
    signature is the classical N wave.  Reported quantities:

    ``front_dp_over_p`` / ``rear_dp_over_p``
        mean compression / expansion plateau, ``p/p_inf - 1``;
    ``impulse``  ``\\int (p/p_inf - 1) dx / chord`` (per unit span, scaled by
        ``p_inf``): the linear-theory N-wave impulse of a single wedge pair is
        proportional to the signed area of the signature, and it is the
        quantity a boom-mitigation study minimises;
    ``asymmetry``  ``|rear_dp| / front_dp`` (``> 1`` for a nonlinear wave,
        textbook result: expansion fans are "stronger" than the equal-angle
        oblique shock at high Mach);
    ``zero_crossing_x``  where the wall pressure returns to ``p_inf``.
    """
    x = np.asarray(x, float)
    pr = np.asarray(p_ratio, float)
    if x.size < 4:
        return {"status": "INSUFFICIENT_DATA"}
    order = np.argsort(x)
    xs, ps = x[order], pr[order]
    span = float(xs.max() - xs.min())
    if span <= 0:
        return {"status": "INSUFFICIENT_DATA"}
    mid = 0.5 * (xs.min() + xs.max())
    front_sel = (xs > xs.min() + 0.15 * span) & (xs < mid)
    rear_sel = (xs > mid) & (xs < xs.min() + 0.85 * span)
    front = float(np.mean(ps[front_sel])) - 1.0 if front_sel.any() else float("nan")
    rear = float(np.mean(ps[rear_sel])) - 1.0 if rear_sel.any() else float("nan")
    impulse = float(np.trapezoid(ps - 1.0, xs) / chord) if hasattr(np, "trapezoid") \
        else float(np.trapz(ps - 1.0, xs) / chord)
    sign = np.sign(ps - 1.0)
    zero = float("nan")
    for i in range(1, xs.size):
        if sign[i] * sign[i - 1] < 0.0:
            w = (ps[i - 1] - 1.0) / ((ps[i - 1] - 1.0) - (ps[i] - 1.0))
            zero = float(xs[i - 1] + w * (xs[i] - xs[i - 1]))
            break
    return {
        "status": "OK",
        "front_dp_over_p": front,
        "rear_dp_over_p": rear,
        "front_plateau_std": float(np.std(ps[front_sel])) if front_sel.any() else float("nan"),
        "rear_plateau_std": float(np.std(ps[rear_sel])) if rear_sel.any() else float("nan"),
        "impulse": impulse,
        "asymmetry": float(abs(rear) / front) if front not in (0.0,) and np.isfinite(front)
        and front != 0.0 else float("nan"),
        "zero_crossing_x": zero,
        "x_min": float(xs.min()), "x_max": float(xs.max()),
    }


def oblique_shock_pressure_ratio(mach: float, theta_deg: float,
                                 gamma: float = GAMMA) -> float:
    """Exact oblique-shock static pressure ratio for wedge angle ``theta``."""
    from .transonic_aero import oblique_shock_angle
    beta = oblique_shock_angle(mach, theta_deg, gamma)
    mn1 = mach * math.sin(beta)
    return 1.0 + 2.0 * gamma / (gamma + 1.0) * (mn1 ** 2 - 1.0)


def ackeret_dp_over_p(mach: float, theta_deg: float, gamma: float = GAMMA) -> float:
    """Ackeret linear theory ``dp/p_inf = gamma M^2 theta / sqrt(M^2 - 1)``."""
    if mach <= 1.0:
        raise ValueError("Ackeret theory requires Ma > 1")
    return gamma * mach ** 2 * math.radians(theta_deg) / math.sqrt(mach ** 2 - 1.0)


__all__ = [
    "mesh_size", "refinement_ratio", "richardson_3grid",
    "normal_shock_pressure_ratio", "normal_shock_downstream_mach",
    "isentropic_mach_from_pressure", "isentropic_mach_from_cp_local",
    "shock_strength", "nwave_metrics",
    "oblique_shock_pressure_ratio", "ackeret_dp_over_p",
]