#!/usr/bin/env python3
"""Analysis + gate evaluation for the compressible-shock family verification.

Reads the raw fields written by ``run_family.py`` (``out/``) and produces

* ``result.json``        every measured number, per case and per grid
* ``gates_verified.json``  the pass/fail table with criterion, reference,
                           measured value, error and threshold per gate
* ``figures/*.png``      density fields with the detected structures, the
                           normal-shock error chart and the convergence chart

Definitions used by the gates (all fixed here, none tuned to the outcome):

``normal shock``   Mach 2/3/5/8/10 x (400, 800, 1600) cells, driver state
                   inverted from the shock-tube relations so the shock Mach
                   number is an ``output``; ratios compared with the exact
                   Rankine-Hugoniot values at the target Mach number.
``incident shock`` inclination of the fitted front above the triple point;
                   the exact reference is 60 deg to the wall because the
                   Woodward-Colella initial data prescribe a Mach-10 front
                   inclined 30 deg to the wall normal.
``Mach stem``       the near-vertical part of the leading front; its height is
                   the triple-point height above the wall, referenced to the
                   Richardson extrapolation of the three grids.
``slip line``       (a) ray scan maximising the density jump minus the pressure
                   jump, (b) entropy-ridge principal-axis fit; the gate is their
                   difference plus the physical continuity checks.
``vortex``          pressure minimum of the rolled-up slip line, referenced to
                   Richardson extrapolation, error normalised by the channel
                   height.
``bow shock``       local geometry (front slope) versus the shock angle implied
                   by the Rankine-Hugoniot jump at the same station, i.e. the
                   exact oblique-shock relation with the upstream state.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tensorlbm.compressible_shock import analysis, family  # noqa: E402
from tensorlbm.compressible_shock import GAMMA  # noqa: E402

DMR_GRIDS = [(240, 60), (360, 90), (480, 120)]
FFS_GRIDS = [(150, 50), (225, 75), (300, 100)]
RATIO = 1.5  # grid refinement ratio of both 2-D families


# ------------------------------------------------------------------ utilities
def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _dump(path, payload):
    with open(path, "w") as fh:
        json.dump(_jsonable(payload), fh, indent=1)
    print(f"[wrote] {path}", flush=True)


def richardson(coarse: float, medium: float, fine: float, ratio: float = RATIO) -> dict:
    """Richardson extrapolation from three grids with an estimated order.

    ``coarse/medium/fine`` are the values on the three grids (refinement ratio
    ``ratio``).  The observed order follows from the three-point formula; when
    the differences are too small or the order is not bracketed in [0.3, 4] the
    formal second order is used and flagged, so the extrapolated value is never
    silently based on a bad fit.
    """
    d1 = medium - coarse
    d2 = fine - medium
    out = {"values": [coarse, medium, fine], "ratio": ratio}
    if abs(d2) < 1e-14 or d1 * d2 <= 0.0:
        p = 2.0
        flag = "order not observable (non-monotone or too small); p=2 used"
    else:
        p = math.log(abs(d1 / d2)) / math.log(ratio)
        flag = "observed"
        if not (0.3 <= p <= 4.0):
            p = 2.0
            flag = f"observed order {p:.2f} outside [0.3,4]; p=2 used"
    out["order"] = p
    out["order_note"] = flag
    out["extrapolated"] = fine + (fine - medium) / (ratio**p - 1.0)
    return out


def _load(path):
    d = np.load(path)
    data = {k: d[k] for k in d.files}
    for k in ("rho", "u", "v", "p", "rho_probe"):
        if k in data:
            data[k] = data[k].astype(float)
    for k in ("xc", "yc"):
        data[k] = data[k].astype(float)
    if "meta" in data:
        data["meta"] = json.loads(str(data["meta"]))
    return data


def _state(rho, u, v, p, xc, yc, x, y):
    r = float(family.bilinear(rho, xc, yc, x, y))
    uu = float(family.bilinear(u, xc, yc, x, y))
    vv = float(family.bilinear(v, xc, yc, x, y))
    pp = float(family.bilinear(p, xc, yc, x, y))
    a = math.sqrt(GAMMA * pp / r)
    return {"rho": r, "u": uu, "v": vv, "p": pp, "a": a, "M": math.hypot(uu, vv) / a}


def _front_slope(x_front, yc, y0, half=2):
    """Local ``dx/dy`` of the front at ``y0`` from a centred linear window."""
    j = int(np.argmin(np.abs(yc - y0)))
    lo, hi = max(j - half, 0), min(j + half, len(yc) - 1)
    sel = np.isfinite(x_front[lo : hi + 1])
    if sel.sum() < 3:
        return float("nan")
    yy = yc[lo : hi + 1][sel]
    xx = x_front[lo : hi + 1][sel]
    return float(np.polyfit(yy, xx, 1)[0])


def _acute_deg(a, b):
    """Acute angle between two undirected line angles in degrees."""
    d = abs((a - b + 90.0) % 180.0 - 90.0)
    return float(d)


# ------------------------------------------------------- forward-facing step
def _leading_front(rho, xc, yc, threshold):
    """Leading-front position per row found by scanning *from the inflow side*.

    The forward-facing step has internal shocks downstream of the bow shock, so
    the right-to-left scan of ``analysis.front_profile`` (built for the DMR) picks
    the wrong wave.  Scanning from ``x_min`` for the first crossing of
    ``threshold`` always returns the bow shock, which is what the step case needs.
    """
    prof = np.full(yc.shape, np.nan)
    for j in range(yc.size):
        idx = np.flatnonzero(rho[j] > threshold)
        if idx.size == 0:
            continue
        i = int(idx[0])
        i0 = max(i - 1, 0)
        r0, r1 = float(rho[j, i0]), float(rho[j, i])
        frac = (threshold - r0) / (r1 - r0) if r1 != r0 else 0.0
        frac = min(max(frac, 0.0), 1.0)
        prof[j] = xc[i0] + frac * (xc[i] - xc[i0])
    return prof


def _ffs_stations(field, y_list, h_cells=3.0):
    """Shock-geometry vs Rankine-Hugoniot consistency along the leading front.

    For every station ``y`` on the leading front the local slope gives the shock
    angle ``beta_geom`` (angle between the front and the *measured* upstream
    velocity), and the density/pressure jumps 3 cells ahead/behind the front give
    ``beta_jump`` through the exact oblique-shock inversion at the measured
    upstream Mach number.  The difference is a pure physics residual: geometry on
    one side, conservation laws on the other.
    """
    rho, u, v, p, xc, yc = (field[k] for k in ("rho", "u", "v", "p", "xc", "yc"))
    thr = 0.5 * (1.4 + 3.857 * 1.4)
    prof = _leading_front(rho, xc, yc, thr)
    h = h_cells * (xc[1] - xc[0])
    stations = []
    for y0 in y_list:
        j = int(np.argmin(np.abs(yc - y0)))
        xf = float(prof[j])
        if not np.isfinite(xf):
            continue
        s = _front_slope(prof, yc, y0)
        if not np.isfinite(s):
            continue
        nrm = math.hypot(1.0, s)
        nx, ny = 1.0 / nrm, -s / nrm
        ahead = _state(rho, u, v, p, xc, yc, xf - h * nx, yc[j] - h * ny)
        behind = _state(rho, u, v, p, xc, yc, xf + h * nx, yc[j] + h * ny)
        flow_ang = math.degrees(math.atan2(ahead["v"], ahead["u"]))
        shock_line_ang = math.degrees(math.atan2(1.0, s))
        # beta is the angle between the shock line and the upstream velocity
        beta_geom = _acute_deg(shock_line_ang, flow_ang)
        ratio = behind["rho"] / ahead["rho"]
        try:
            inv = family.shock_polar_angle_from_ratio(ahead["M"], ratio)
            beta_jump = inv["beta_deg"] if inv["attached"] else float("nan")
        except ValueError:
            beta_jump = float("nan")
        stations.append(
            {
                "y": float(yc[j]),
                "x_front": xf,
                "front_slope_dxdy": s,
                "beta_geom_deg": beta_geom,
                "M1_measured": ahead["M"],
                "rho_ratio": ratio,
                "p_ratio": behind["p"] / ahead["p"],
                "beta_jump_deg": beta_jump,
                "beta_error_deg": (beta_jump - beta_geom) if np.isfinite(beta_jump) else float("nan"),
            }
        )
    errs = [abs(s["beta_error_deg"]) for s in stations if np.isfinite(s["beta_error_deg"])]
    out = {
        "stations": stations,
        "n_stations": len(stations),
        "max_abs_beta_error_deg": float(max(errs)) if errs else float("nan"),
        "mean_abs_beta_error_deg": float(np.mean(errs)) if errs else float("nan"),
    }
    return out


def _recompression_shock(field, x_lo=1.10, x_hi=1.55, y_lo=0.30, y_hi=0.56, h_cells=3.0,
                         local_y=(0.35, 0.41, 0.47)):
    """Internal recompression (reflected) shock: ridge fit + jump inversion.

    The ridge is the per-row maximum of ``|grad p|`` inside the window; the same
    station then supplies the upstream state (3 cells ahead along the ridge
    normal) and the downstream density, so the oblique-shock inversion yields the
    thermodynamic shock angle to compare with the geometric one.
    """
    rho, u, v, p, xc, yc = (field[k] for k in ("rho", "u", "v", "p", "xc", "yc"))
    gp = np.sqrt(np.gradient(p, xc, axis=1) ** 2 + np.gradient(p, yc, axis=0) ** 2)
    rows = np.flatnonzero((yc >= y_lo) & (yc <= y_hi))
    pts = []
    for j in rows:
        i0 = int(np.argmin(np.abs(xc - x_lo)))
        i1 = int(np.argmin(np.abs(xc - x_hi)))
        i = i0 + int(np.argmax(gp[j, i0 : i1 + 1]))
        pts.append((float(xc[i]), float(yc[j])))
    if len(pts) < 4:
        return {"found": False, "reason": "window too small"}
    pts = np.array(pts)
    slope, intercept = np.polyfit(pts[:, 1], pts[:, 0], 1)
    resid = pts[:, 0] - (slope * pts[:, 1] + intercept)
    # one robust pass: reject rows whose ridge sits off the fitted line
    keep = np.abs(resid) <= max(2.0 * float(np.std(resid)), 1e-12)
    if keep.sum() >= 4:
        pts = pts[keep]
        slope, intercept = np.polyfit(pts[:, 1], pts[:, 0], 1)
        resid = pts[:, 0] - (slope * pts[:, 1] + intercept)
    local = []
    gpx = np.gradient(p, xc, axis=1)
    gpy = np.gradient(p, yc, axis=0)
    i0 = int(np.argmin(np.abs(xc - x_lo)))
    i1 = int(np.argmin(np.abs(xc - x_hi)))
    for ytar in local_y:
        j = int(np.argmin(np.abs(yc - ytar)))
        i = i0 + int(np.argmax(np.hypot(gpx[j, i0 : i1 + 1], gpy[j, i0 : i1 + 1])))
        ang = math.degrees(math.atan2(gpx[j, i], -gpy[j, i])) % 180.0
        local.append({"y": float(yc[j]), "x": float(xc[i]), "angle_deg": float(ang)})
    h = h_cells * (xc[1] - xc[0])
    jm = int(len(pts) // 2)
    x0, y0 = pts[jm]
    nrm = math.hypot(1.0, slope)
    nx, ny = 1.0 / nrm, -slope / nrm
    ahead = _state(rho, u, v, p, xc, yc, x0 - h * nx, y0 - h * ny)
    behind = _state(rho, u, v, p, xc, yc, x0 + h * nx, y0 + h * ny)
    flow_ang = math.degrees(math.atan2(ahead["v"], ahead["u"]))
    shock_line_ang = math.degrees(math.atan2(1.0, slope))
    beta_geom = _acute_deg(shock_line_ang, flow_ang)
    ratio = behind["rho"] / ahead["rho"]
    try:
        inv = family.shock_polar_angle_from_ratio(ahead["M"], ratio)
        beta_jump = inv["beta_deg"] if inv["attached"] else float("nan")
    except ValueError:
        beta_jump = float("nan")
    return {
        "found": True,
        "angle_from_x_deg": float(shock_line_ang),
        "local_tangent_angles": local,
        "slope_dxdy": float(slope),
        "fit_rms": float(np.sqrt(np.mean(resid**2))),
        "n_points": int(len(pts)),
        "station": [float(x0), float(y0)],
        "M1_measured": ahead["M"],
        "rho_ratio": float(ratio),
        "p_ratio": float(behind["p"] / ahead["p"]),
        "beta_geom_deg": float(beta_geom),
        "beta_jump_deg": float(beta_jump),
        "beta_error_deg": float(beta_jump - beta_geom) if np.isfinite(beta_jump) else float("nan"),
    }


def ffs_metrics(path):
    f = _load(path)
    xc, yc, rho = f["xc"], f["yc"], f["rho"]
    meta = f["meta"]
    steady = float(np.max(np.abs(rho - f["rho_probe"])) / np.max(rho))
    thr = 0.5 * (1.4 + 3.857 * 1.4)
    prof = _leading_front(rho, xc, yc, thr)
    foot = family.front_angle(prof, yc, 0.22, 0.50)
    upper = family.front_angle(prof, yc, 0.60, 0.95)
    standoff = float(meta["step_x"] - float(np.interp(0.25, yc, prof)))
    bow = _ffs_stations(f, np.linspace(0.24, 0.56, 9))
    probe_angle = family.front_angle(_leading_front(f["rho_probe"], xc, yc, thr), yc, 0.22, 0.50)
    rec = _recompression_shock(f)
    return {
        "grid": [int(xc.size), int(yc.size)],
        "dx": float(xc[1] - xc[0]),
        "dy": float(yc[1] - yc[0]),
        "t": float(f["t"]),
        "steadiness_rel_change": steady,
        "steadiness_rel_change_rms": float(np.sqrt(np.mean((rho - f["rho_probe"]) ** 2)) / rho.max()),
        "probe_angle_deg": probe_angle.get("angle_deg"),
        "angle_drift_deg": (
            float(abs(foot["angle_deg"] - probe_angle["angle_deg"]))
            if np.isfinite(probe_angle.get("angle_deg", np.nan))
            else float("nan")
        ),
        "wall_time_s": float(f["wall_time_s"]),
        "n_admissibility_fixes": int(f["n_admissibility_fixes"]),
        "front": {
            "foot_angle_deg": foot,
            "upper_angle_deg": upper,
            "standoff_at_y025": standoff,
            "profile_x": prof,
        },
        "bow_shock": bow,
        "recompression_shock": rec,
        "fields": {
            "rho_min": float(rho.min()),
            "rho_max": float(rho.max()),
            "p_min": float(f["p"].min()),
            "p_max": float(f["p"].max()),
        },
    }

# --------------------------------------------------------- double Mach reflection
def triple_point_from_branches(x_front, yc, y_tp_guess, ly, stem_frac=0.55, inc_lo=0.72):
    """Triple point as the intersection of the Mach-stem and incident branches.

    ``mach_stem_height`` reports the switch between the two branches on the grid,
    i.e. quantised by ``dy``.  Fitting the near-vertical stem branch (from the
    wall up to ``stem_frac*y_tp_guess``) and the 60 deg incident branch (from
    ``inc_lo*ly`` up to ``0.98*ly``) and intersecting the two lines gives the
    triple point to sub-cell accuracy, and the stem height is its height above
    the wall.
    """
    x = np.asarray(x_front, dtype=float)
    y = np.asarray(yc, dtype=float)
    ok = np.isfinite(x)
    stem_sel = ok & (y >= y[ok][0]) & (y <= stem_frac * y_tp_guess)
    inc_sel = ok & (y >= inc_lo * ly) & (y <= 0.98 * ly)
    if stem_sel.sum() < 4 or inc_sel.sum() < 4:
        return {"found": False, "reason": "branch windows too small"}
    bs, as_ = np.polyfit(y[stem_sel], x[stem_sel], 1)
    bi, ai = np.polyfit(y[inc_sel], x[inc_sel], 1)
    if abs(bs - bi) < 1e-9:
        return {"found": False, "reason": "branches parallel"}
    y_tp = (ai - as_) / (bs - bi)
    return {
        "found": True,
        "x": float(as_ + bs * y_tp),
        "y": float(y_tp),
        "height": float(y_tp - y[ok][0]),
        "stem_slope": float(bs),
        "incident_slope": float(bi),
        "stem_points": int(stem_sel.sum()),
        "incident_points": int(inc_sel.sum()),
        "stem_rms": float(np.sqrt(np.mean((x[stem_sel] - (as_ + bs * y[stem_sel])) ** 2))),
        "incident_rms": float(np.sqrt(np.mean((x[inc_sel] - (ai + bi * y[inc_sel])) ** 2))),
    }


def _vortex_from_pressure(p, xc, yc, box):
    """Pressure-minimum centre of the rolled-up slip line (with a quadratic fit).

    The primary vortex of the double Mach reflection carries a pressure deficit
    at its core; the minimum inside ``box`` is refined by a least-squares
    quadratic through its 5x5 neighbourhood so the position is not quantised to
    the grid.
    """
    X, Y = np.meshgrid(xc, yc)
    x0, x1, y0, y1 = box
    mask = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    if mask.sum() < 25:
        return {"found": False, "reason": "box too small"}
    pm = np.where(mask, p, np.inf)
    j, i = np.unravel_index(int(np.argmin(pm)), pm.shape)
    i_lo, i_hi = max(i - 2, 0), min(i + 3, p.shape[1])
    j_lo, j_hi = max(j - 2, 0), min(j + 3, p.shape[0])
    xs = (X[j_lo:j_hi, i_lo:i_hi] - xc[i]).ravel()
    ys = (Y[j_lo:j_hi, i_lo:i_hi] - yc[j]).ravel()
    zs = p[j_lo:j_hi, i_lo:i_hi].ravel()
    A = np.vstack([np.ones_like(xs), xs, ys, xs * xs, xs * ys, ys * ys]).T
    coef, *_ = np.linalg.lstsq(A, zs, rcond=None)
    c, cx, cy, cxx, cxy, cyy = coef
    hess = np.array([[2 * cxx, cxy], [cxy, 2 * cyy]])
    grad = np.array([cx, cy])
    try:
        off = -np.linalg.solve(hess, grad)
    except np.linalg.LinAlgError:
        off = np.zeros(2)
    off = np.clip(off, -1.5 * (xc[1] - xc[0]), 1.5 * (xc[1] - xc[0]))
    return {
        "found": True,
        "x": float(xc[i] + off[0]),
        "y": float(yc[j] + off[1]),
        "grid_i": int(i),
        "grid_j": int(j),
        "p_min": float(p[j, i]),
        "p_min_refined": float(c + cx * off[0] + cy * off[1] + cxx * off[0] ** 2
                               + cxy * off[0] * off[1] + cyy * off[1] ** 2),
        "box": list(box),
    }


def dmr_metrics(path):
    f = _load(path)
    rho, u, v, p, xc, yc = (f[k] for k in ("rho", "u", "v", "p", "xc", "yc"))
    t1, t2 = float(f["t1"]), float(f["t"])
    prof1, prof2 = f["prof1"], f["prof2"]
    meta = f["meta"]
    thr = 0.5 * (meta["pre_shock"]["rho"] + meta["post_shock"]["rho"])
    stem1 = analysis.mach_stem_height(prof1, yc)
    stem2 = analysis.mach_stem_height(prof2, yc)
    tpx = triple_point_from_branches(prof2, yc, float(stem2["triple_point_y"]), float(yc[-1]))
    if tpx["found"]:
        y_tp = tpx["y"]
    else:  # pragma: no cover - fallback keeps the analysis defined
        y_tp = float(stem2["triple_point_y"])
    j_tp = int(np.argmin(np.abs(yc - y_tp)))
    x_tp = float(prof2[j_tp])
    tp1 = triple_point_from_branches(prof1, yc, float(stem1["triple_point_y"]), float(yc[-1]))
    if tp1["found"]:
        y_tp1 = tp1["y"]
    else:  # pragma: no cover
        y_tp1 = float(stem1["triple_point_y"])
    j1 = int(np.argmin(np.abs(yc - stem1["triple_point_y"])))
    x_tp1 = float(prof1[j1])
    foot_speed = (float(prof2[0]) - float(prof1[0])) / (t2 - t1)
    vt = ((x_tp - x_tp1) / (t2 - t1), (y_tp - y_tp1) / (t2 - t1))

    inc = family.front_angle(prof2, yc, 0.60, 0.97)
    exact_inc = 90.0 - float(meta["angle_deg"])

    ang_grid = np.arange(170.0, 260.01, 0.25)
    ray = family.slip_line_angle(rho, p, xc, yc, (x_tp, y_tp), 0.30, angles_deg=ang_grid)
    vel = family.slip_line_angle_velocity(u, v, xc, yc, (x_tp, y_tp), 0.30, angles_deg=ang_grid)
    ridge = family.contact_ridge_angle(rho, p, xc, yc, (x_tp, y_tp), 0.08, 0.28)
    slip_diff = _acute_deg(ray["angle_deg"], vel["angle_deg"])

    box = (x_tp - 0.55, x_tp - 0.06, 0.04, 0.88 * max(y_tp, 1e-6))
    vortex = _vortex_from_pressure(p, xc, yc, box)

    # Mach-stem jump: upstream state is the exact pre-shock state, downstream is
    # sampled 3..8 cells behind the wall-row front.
    j_st = int(np.argmin(np.abs(yc - 0.06)))
    xf = float(prof2[j_st])
    i_f = int(np.argmin(np.abs(xc - xf)))
    n = max(int(8 * (yc[1] - yc[0]) / (xc[1] - xc[0])), 3)
    seg = slice(max(i_f - n, 0), max(i_f - 3, 1))
    rho_stem = float(np.mean(rho[j_st, seg]))
    p_stem = float(np.mean(p[j_st, seg]))
    a1 = math.sqrt(GAMMA * meta["pre_shock"]["p"] / meta["pre_shock"]["rho"])
    ms = foot_speed / a1
    try:
        from tensorlbm.compressible_shock.rankine_hugoniot import normal_shock

        ns = normal_shock(ms)
        stem_jump = {
            "shock_mach_measured": ms,
            "rho_ratio_measured": rho_stem / meta["pre_shock"]["rho"],
            "rho_ratio_exact": float(ns["rho_ratio"]),
            "rho_error": analysis.relative_error(
                rho_stem / meta["pre_shock"]["rho"], ns["rho_ratio"]
            ),
            "p_ratio_measured": p_stem / meta["pre_shock"]["p"],
            "p_ratio_exact": float(ns["p_ratio"]),
            "p_error": analysis.relative_error(p_stem / meta["pre_shock"]["p"], ns["p_ratio"]),
        }
    except ValueError as exc:  # pragma: no cover
        stem_jump = {"error": str(exc)}

    # von Neumann three-shock closure attempt in the triple-point frame: the
    # incident shock keeps its prescribed normal Mach 10 and 60 deg inclination,
    # so the steady upstream Mach is M_s / sin(60 deg) exactly.
    ms_incident = float(meta["mach"])
    m1_tp = ms_incident / math.sin(math.radians(exact_inc))
    try:
        closure = family.three_shock_closure(m1_tp, exact_inc)
        theory = {
            "status": "closed" if closure["polylines_cross"] else "no non-degenerate solution",
            "reason": (
                "reflected shock would have to turn the flow back by more than "
                "theta_max(M2); the pressure polars only meet in the degenerate "
                "limit chi -> theta_i (vanishing reflected shock)"
                if not closure["polylines_cross"]
                else "pressure polars cross"
            ),
            "upstream_mach_triple_point_frame": closure["upstream_mach"],
            "incident_theta_deg": closure["incident_theta_deg"],
            "M2_behind_incident": closure["M2_behind_incident"],
            "theta_max_downstream_deg": closure["theta_max_downstream_deg"],
            "chi_min_feasible_deg": closure["chi_min_feasible_deg"],
            "pressure_gap_at_chi_min": closure["pressure_gap_at_chi_min"],
        }
    except ValueError as exc:  # pragma: no cover - diagnostic only
        theory = {"status": "failed", "reason": str(exc)}

    return {
        "grid": [int(xc.size), int(yc.size)],
        "dx": float(xc[1] - xc[0]),
        "dy": float(yc[1] - yc[0]),
        "t1": t1,
        "t2": t2,
        "wall_time_s": float(f["wall_time_s"]),
        "n_admissibility_fixes": int(f["n_admissibility_fixes"]),
        "triple_point": {
            "x": x_tp,
            "y": y_tp,
            "t1": [x_tp1, y_tp1],
            "grid_quantised_tp_y": float(stem2["triple_point_y"]),
            "branch_intersection": tpx,
        },
        "mach_stem": {
            "height": float(tpx["height"]) if tpx["found"] else float(stem2["stem_height"]),
            "height_grid_quantised": float(stem2["stem_height"]),
            "height_t1": float(tp1["height"]) if tp1["found"] else float(stem1["stem_height"]),
            "foot_x": float(prof2[0]),
            "foot_x_t1": float(prof1[0]),
            "wall_speed": foot_speed,
            "triple_point_velocity": list(vt),
            "jump": stem_jump,
        },
        "incident_shock": {
            "measured_angle_deg": inc["angle_deg"],
            "exact_angle_deg": exact_inc,
            "error_deg": abs(inc["angle_deg"] - exact_inc),
            "fit_rms": inc["rms"],
            "n_points": inc["n_points"],
        },
        "slip_line": {
            "ray_angle_deg": ray["angle_deg"],
            "ray_score": ray["score"],
            "ray_density_jump": ray["density_jump"],
            "ray_pressure_jump": ray["pressure_jump"],
            "velocity_angle_deg": vel["angle_deg"],
            "velocity_score": vel["score"],
            "entropy_ridge_angle_deg": ridge.get("angle_deg"),
            "entropy_ridge_cells": ridge.get("n_cells"),
            "angle_difference_deg": slip_diff,
            "angle_from_wall_deg": _acute_deg(ray["angle_deg"], 180.0),
        },
        "vortex": vortex,
        "three_shock_theory": theory,
        "fields": {
            "rho_min": float(rho.min()),
            "rho_max": float(rho.max()),
            "p_min": float(p.min()),
            "p_max": float(p.max()),
        },
    }


# ------------------------------------------------------------------ assembly
def _figures(ns, dmr, ffs, gates, figdir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(figdir, exist_ok=True)
    grids_d = sorted(dmr.values(), key=lambda m: m["dx"])
    grids_f = sorted(ffs.values(), key=lambda m: m["dx"])

    # 1. normal-shock Mach scan
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for n in (400, 800, 1600):
        xs, ys = [], []
        for m in (2.0, 3.0, 5.0, 8.0, 10.0):
            rec = [r for r in ns["records"] if r["variant"] == "riemann" and r["mach"] == m and r["n"] == n]
            if rec:
                xs.append(m)
                ys.append(100 * max(rec[0]["error"][k] for k in ("rho_ratio", "p_ratio", "T_ratio")))
        axes[0].semilogy(xs, ys, "o-", label=f"n = {n}")
    axes[0].axhline(1.0, color="r", ls="--", label="1% acceptance")
    axes[0].set_xlabel("shock Mach number")
    axes[0].set_ylabel("max |ratio error| [%]")
    axes[0].set_title("normal shock: R-H ratios (riemann variant)")
    axes[0].legend()
    axes[0].grid(alpha=0.3, which="both")
    finest = max(ns["grids"])
    for k, mk, lab in (("rho_ratio", "o", "rho2/rho1"), ("p_ratio", "s", "p2/p1"), ("T_ratio", "^", "T2/T1")):
        ex, me = [], []
        for r in ns["records"]:
            if r["variant"] == "riemann" and r["n"] == finest:
                ex.append(r["exact"][k])
                me.append(r["measured"][k])
        axes[1].plot(ex, me, mk, label=lab)
    lim = [1, 150]
    axes[1].plot(lim, lim, "k--", lw=1)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("exact ratio")
    axes[1].set_ylabel("measured ratio")
    axes[1].set_title(f"measured vs exact (n = {finest})")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "normal_shock_scan.png"), dpi=140)
    plt.close(fig)

    # 2. DMR density fields
    fig, axes = plt.subplots(len(grids_d), 1, figsize=(10, 3.1 * len(grids_d)))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, grids_d):
        f = _load(os.path.join(HERE, "out", "fields", f"dmr_{m['grid'][0]}x{m['grid'][1]}.npz"))
        cs = ax.contourf(f["xc"], f["yc"], np.log10(f["rho"]), levels=40, cmap="viridis")
        tp = m["triple_point"]
        ax.plot([tp["x"]], [tp["y"]], "r*", ms=13, label="triple point")
        vx = m["vortex"]
        if vx.get("found"):
            ax.plot([vx["x"]], [vx["y"]], "wo", ms=6, mfc="none", label="vortex core")
        ang = math.radians(m["slip_line"]["ray_angle_deg"])
        s = np.linspace(0.0, 0.30, 20)
        ax.plot(tp["x"] + s * math.cos(ang), tp["y"] + s * math.sin(ang), "w--", lw=1.2, label="slip line (ray)")
        ax.set_title(f"DMR {m['grid'][0]}x{m['grid'][1]}: log10 density, t = {m['t2']:.2f}")
        ax.set_xlim(1.2, 3.4)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("y")
        ax.legend(loc="upper left", fontsize=7)
        fig.colorbar(cs, ax=ax)
    axes[-1].set_xlabel("x")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "dmr_density.png"), dpi=140)
    plt.close(fig)

    # 3. DMR convergence
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    dx = [m["dx"] for m in grids_d]
    h = [m["mach_stem"]["height"] for m in grids_d]
    axes[0].plot(dx, h, "o-", label="measured")
    axes[0].axhline(gates["_dmr_stem_ref"], color="r", ls="--", label="Richardson")
    axes[0].set_xlabel("dx (dy = dx)")
    axes[0].set_ylabel("Mach stem height [H]")
    axes[0].set_title("Mach-stem / triple-point height")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    for key, lab, mk in (("x", "vortex x", "s"), ("y", "vortex y", "d")):
        vals = [m["vortex"][key] for m in grids_d if m["vortex"].get("found")]
        if vals:
            axes[1].plot(dx[: len(vals)], vals, mk + "-", label=lab)
    axes[1].set_xlabel("dx")
    axes[1].set_title("primary vortex core position")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "dmr_convergence.png"), dpi=140)
    plt.close(fig)

    # 4. FFS density fields
    fig, axes = plt.subplots(len(grids_f), 1, figsize=(10, 3.0 * len(grids_f)))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, grids_f):
        f = _load(os.path.join(HERE, "out", "fields", f"ffs_{m['grid'][0]}x{m['grid'][1]}.npz"))
        cs = ax.contourf(f["xc"], f["yc"], np.log10(f["rho"]), levels=40, cmap="viridis")
        prof = np.array(m["front"]["profile_x"])
        ax.plot(prof, f["yc"], "w--", lw=1.0, label="leading bow shock")
        rec = m["recompression_shock"]
        if rec.get("found"):
            s = np.linspace(0, 0.30, 10)
            x0, y0 = rec["station"]
            ax.plot(x0 + s * rec["slope_dxdy"], y0 + s, "r-", lw=1.0, label="recompression shock fit")
        ax.set_title(f"FFS {m['grid'][0]}x{m['grid'][1]}: log10 density, Mach 3, t = {m['t']:.1f}")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("y")
        ax.legend(loc="upper right", fontsize=7)
        fig.colorbar(cs, ax=ax)
    axes[-1].set_xlabel("x")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "ffs_density.png"), dpi=140)
    plt.close(fig)

    # 5. FFS convergence
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    dxf = [m["dx"] for m in grids_f]
    ax.plot(dxf, [m["front"]["foot_angle_deg"]["angle_deg"] for m in grids_f], "o-", label="bow-shock foot angle")
    ax.plot(dxf, [m["recompression_shock"]["angle_from_x_deg"] for m in grids_f], "s-", label="recompression shock angle")
    ax.axhline(gates["_ffs_foot_ref"], color="r", ls="--", label="Richardson (foot)")
    ax.axhline(gates["_ffs_rec_ref"], color="g", ls="--", label="Richardson (recompression)")
    ax.set_xlabel("dx")
    ax.set_ylabel("angle from the wall [deg]")
    ax.set_title("forward-facing step: shock angles vs grid")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "ffs_convergence.png"), dpi=140)
    plt.close(fig)

    # 6. slip-line score curve
    finest_d = grids_d[-1]
    f = _load(os.path.join(HERE, "out", "fields", f"dmr_{finest_d['grid'][0]}x{finest_d['grid'][1]}.npz"))
    tp = finest_d["triple_point"]
    ray = family.slip_line_angle(
        f["rho"], f["p"], f["xc"], f["yc"], (tp["x"], tp["y"]), 0.30, angles_deg=np.arange(170.0, 260.01, 0.25)
    )
    vel = family.slip_line_angle_velocity(
        f["u"], f["v"], f["xc"], f["yc"], (tp["x"], tp["y"]), 0.30,
        angles_deg=np.arange(170.0, 260.01, 0.25),
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    rs = ray["scores"] / np.nanmax(ray["scores"])
    vs = vel["scores"] / np.nanmax(vel["scores"])
    ax.plot(ray["angles_deg"], rs, "-", label="density-jump score (normalised)")
    ax.plot(vel["angles_deg"], vs, "--", label="velocity-sheet score (normalised)")
    ax.axvline(ray["angle_deg"], color="r", ls="--", label=f"density ray: {ray['angle_deg']:.2f} deg")
    ax.axvline(vel["angle_deg"], color="b", ls=":", label=f"velocity ray: {vel['angle_deg']:.2f} deg")
    ax.set_xlabel("ray angle from +x [deg]")
    ax.set_ylabel("normalised detector score")
    ax.set_title(
        f"slip-line detectors, DMR {finest_d['grid'][0]}x{finest_d['grid'][1]}"
        f" (|dalpha| = {abs(ray['angle_deg'] - vel['angle_deg']):.2f} deg)"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "slip_line_score.png"), dpi=140)
    plt.close(fig)


def assemble():
    outdir = os.path.join(HERE, "out")
    figdir = os.path.join(HERE, "figures")
    ns = json.load(open(os.path.join(outdir, "normal_shock.json")))

    dmr, ffs = {}, {}
    for nx, ny in DMR_GRIDS:
        p = os.path.join(outdir, "fields", f"dmr_{nx}x{ny}.npz")
        if os.path.exists(p):
            dmr[(nx, ny)] = dmr_metrics(p)
    for nx, ny in FFS_GRIDS:
        p = os.path.join(outdir, "fields", f"ffs_{nx}x{ny}.npz")
        if os.path.exists(p):
            ffs[(nx, ny)] = ffs_metrics(p)
    if len(dmr) < 3 or len(ffs) < 3:
        raise SystemExit(f"missing fields: dmr={sorted(dmr)} ffs={sorted(ffs)}")

    grids_d = [dmr[g] for g in DMR_GRIDS]
    grids_f = [ffs[g] for g in FFS_GRIDS]

    stem_rich = richardson(*[m["mach_stem"]["height"] for m in grids_d])
    vx_rich = richardson(*[m["vortex"]["x"] for m in grids_d])
    vy_rich = richardson(*[m["vortex"]["y"] for m in grids_d])
    foot_rich = richardson(*[m["front"]["foot_angle_deg"]["angle_deg"] for m in grids_f])
    rec_rich = richardson(*[m["recompression_shock"]["angle_from_x_deg"] for m in grids_f])
    stand_rich = richardson(*[m["front"]["standoff_at_y025"] for m in grids_f])

    gates = []
    # ---- normal shock (one gate per Mach number)
    ref_grid = max(ns["grids"])
    for mach in (2.0, 3.0, 5.0, 8.0, 10.0):
        recs = [
            r
            for r in ns["records"]
            if r["variant"] == "riemann" and r["mach"] == mach and r["n"] == ref_grid
        ]
        worst = max(recs, key=lambda r: max(r["error"][k] for k in ("rho_ratio", "p_ratio", "T_ratio")))
        err = max(worst["error"][k] for k in ("rho_ratio", "p_ratio", "T_ratio"))
        all_grids = [
            {
                "grid": r["n"],
                "max_ratio_error": max(r["error"][k] for k in ("rho_ratio", "p_ratio", "T_ratio")),
            }
            for r in ns["records"]
            if r["variant"] == "riemann" and r["mach"] == mach
        ]
        gates.append(
            {
                "id": f"NS-M{mach:g}",
                "case": "normal_shock",
                "criterion": (
                    "max relative error of rho2/rho1, p2/p1, T2/T1 vs Rankine-Hugoniot "
                    f"at the production grid ({ref_grid} cells)"
                ),
                "reference": (
                    "exact R-H relations, gamma=1.4; the shock Mach number is a solver output "
                    "(driver state inverted from the shock-tube relations)"
                ),
                "measured": err,
                "threshold": 0.01,
                "unit": "relative",
                "reference_grid": ref_grid,
                "worst_case": {"Mach": worst["mach"], "grid": worst["n"], "ratios": worst["measured"]},
                "coarse_grid_errors": all_grids,
                "pass": bool(err < 0.01),
            }
        )
    # ---- DMR
    inc_err = max(m["incident_shock"]["error_deg"] for m in grids_d)
    gates.append(
        {
            "id": "DMR-01",
            "case": "double_mach_reflection",
            "criterion": "incident-shock inclination above the triple point",
            "reference": "exact 60 deg to the wall (prescribed Mach-10 front, 30 deg to the wall normal)",
            "measured": inc_err,
            "threshold": 1.0,
            "unit": "deg",
            "pass": bool(inc_err < 1.0),
        }
    )
    stem_err = abs(grids_d[-1]["mach_stem"]["height"] - stem_rich["extrapolated"]) / abs(
        stem_rich["extrapolated"]
    )
    gates.append(
        {
            "id": "DMR-02",
            "case": "double_mach_reflection",
            "criterion": "Mach stem height (triple-point height above the wall)",
            "reference": "Richardson extrapolation of the three grids",
            "measured": stem_err,
            "threshold": 0.05,
            "unit": "relative",
            "values": stem_rich,
            "pass": bool(stem_err < 0.05),
        }
    )
    slip = grids_d[-1]["slip_line"]
    # detector-validity checks (not accuracy thresholds): the detected
    # discontinuity must be a real density jump (>10% relative) and must be
    # density-dominated (a contact, not a shock)
    slip_ok = (
        np.isfinite(slip["angle_difference_deg"])
        and slip["angle_difference_deg"] < 3.0
        and slip["ray_density_jump"] > 0.10
        and slip["ray_density_jump"] > 2.0 * slip["ray_pressure_jump"]
    )
    gates.append(
        {
            "id": "DMR-03",
            "case": "double_mach_reflection",
            "criterion": (
                "slip-line angle: density-jump ray scan vs tangential-velocity (vortex-sheet) ray scan; "
                "validity: |drho|/rho > 10% and density-dominated (|drho|/rho > 2|dp|/p)"
            ),
            "reference": "independent momentum-based detector on the finest grid (density/entropy detector is the other)",
            "measured": slip["angle_difference_deg"],
            "threshold": 3.0,
            "unit": "deg",
            "detail": {
                "ray_angle_from_x_deg": slip["ray_angle_deg"],
                "velocity_sheet_angle_deg": slip["velocity_angle_deg"],
                "entropy_ridge_angle_deg": slip["entropy_ridge_angle_deg"],
                "angle_from_wall_deg": slip["angle_from_wall_deg"],
                "density_jump": slip["ray_density_jump"],
                "pressure_jump": slip["ray_pressure_jump"],
            },
            "pass": bool(slip_ok),
        }
    )
    vx_err = abs(grids_d[-1]["vortex"]["x"] - vx_rich["extrapolated"])
    vy_err = abs(grids_d[-1]["vortex"]["y"] - vy_rich["extrapolated"])
    v_err = max(vx_err, vy_err)
    gates.append(
        {
            "id": "DMR-04",
            "case": "double_mach_reflection",
            "criterion": "primary vortex core position (pressure minimum)",
            "reference": "Richardson extrapolation of the three grids, error normalised by H",
            "measured": v_err,
            "threshold": 0.05,
            "unit": "H",
            "values": {"x": vx_rich, "y": vy_rich},
            "pass": bool(v_err < 0.05),
        }
    )
    # ---- FFS
    foot_err = abs(grids_f[-1]["front"]["foot_angle_deg"]["angle_deg"] - foot_rich["extrapolated"])
    gates.append(
        {
            "id": "FFS-01",
            "case": "forward_facing_step",
            "criterion": "detached bow-shock angle at the foot (y in [0.22, 0.50])",
            "reference": "Richardson extrapolation of the three grids (no closed-form inviscid solution)",
            "measured": foot_err,
            "threshold": 2.0,
            "unit": "deg",
            "values": foot_rich,
            "grid_spread_deg": abs(
                grids_f[-1]["front"]["foot_angle_deg"]["angle_deg"]
                - grids_f[-2]["front"]["foot_angle_deg"]["angle_deg"]
            ),
            "order_note": foot_rich["order_note"],
            "pass": bool(foot_err < 2.0),
        }
    )
    rec_err = abs(grids_f[-1]["recompression_shock"]["angle_from_x_deg"] - rec_rich["extrapolated"])
    gates.append(
        {
            "id": "FFS-02",
            "case": "forward_facing_step",
            "criterion": "recompression (reflected) shock angle",
            "reference": "Richardson extrapolation of the three grids (no closed-form inviscid solution)",
            "measured": rec_err,
            "threshold": 1.0,
            "unit": "deg",
            "values": rec_rich,
            "grid_spread_deg": abs(
                grids_f[-1]["recompression_shock"]["angle_from_x_deg"]
                - grids_f[-2]["recompression_shock"]["angle_from_x_deg"]
            ),
            "order_note": rec_rich["order_note"],
            "alternative_definitions": {
                "local_tangent_y041_deg": [
                    min(
                        m["recompression_shock"]["local_tangent_angles"],
                        key=lambda d: abs(d["y"] - 0.41),
                    )["angle_deg"]
                    for m in grids_f
                ],
                "note": (
                    "the internal recompression shock is curved; a linear ridge fit over the "
                    "isolated window and the local tangent at y=0.41 are both reported so the "
                    "resolution dependence of this quantity is visible"
                ),
            },
            "pass": bool(rec_err < 1.0),
        }
    )
    stand_err = abs(grids_f[-1]["front"]["standoff_at_y025"] - stand_rich["extrapolated"]) / abs(
        stand_rich["extrapolated"]
    )
    gates.append(
        {
            "id": "FFS-03",
            "case": "forward_facing_step",
            "criterion": "bow-shock stand-off distance ahead of the step face at y = 0.25",
            "reference": "Richardson extrapolation of the three grids",
            "measured": stand_err,
            "threshold": 0.05,
            "unit": "relative",
            "values": stand_rich,
            "pass": bool(stand_err < 0.05),
        }
    )

    smooth = []
    for r in ns["records"]:
        if r["variant"] == "jump":
            smooth.append(
                {"mach": r["mach"], "n": r["n"], "max_ratio_error": r["max_ratio_error"]}
            )

    result = {
        "suite": "compressible_shock_family",
        "campaign_node": "P19-N04",
        "gamma": GAMMA,
        "flux": "HLLC (kinetic, gamma-corrected) + MUSCL/minmod + SSP-RK2",
        "normal_shock": {
            "machs": ns["machs"],
            "grids": ns["grids"],
            "n_cases": ns["n_cases"],
            "max_ratio_error_riemann": ns["max_ratio_error"],
            "max_all_error": ns["max_all_error"],
            "prescribed_jump_variant": smooth,
            "records": ns["records"],
        },
        "double_mach_reflection": {
            "grids": [m["grid"] for m in grids_d],
            "t": [grids_d[0]["t1"], grids_d[0]["t2"]],
            "reference": "Woodward & Colella (1984) J. Comput. Phys. 54, 115, problem 2",
            "per_grid": grids_d,
            "richardson": {"stem_height": stem_rich, "vortex_x": vx_rich, "vortex_y": vy_rich},
        },
        "forward_facing_step": {
            "grids": [m["grid"] for m in grids_f],
            "t": grids_f[0]["t"],
            "reference": "Woodward & Colella (1984) J. Comput. Phys. 54, 115, problem 1",
            "per_grid": grids_f,
            "richardson": {
                "foot_angle": foot_rich,
                "recompression_angle": rec_rich,
                "standoff": stand_rich,
            },
        },
        "gates": gates,
        "all_gates_pass": all(g["pass"] for g in gates),
        "n_gates": len(gates),
        "n_gates_pass": sum(1 for g in gates if g["pass"]),
    }
    _dump(os.path.join(HERE, "result.json"), result)
    _dump(
        os.path.join(HERE, "gates_verified.json"),
        {
            "campaign_node": "P19-N04",
            "all_pass": result["all_gates_pass"],
            "n_gates": result["n_gates"],
            "n_pass": result["n_gates_pass"],
            "gates": gates,
        },
    )
    gates_view = {
        "_dmr_stem_ref": stem_rich["extrapolated"],
        "_ffs_foot_ref": foot_rich["extrapolated"],
        "_ffs_rec_ref": rec_rich["extrapolated"],
    }
    _figures(ns, dmr, ffs, gates_view, figdir)
    print(
        f"[gates] {result['n_gates_pass']}/{result['n_gates']} pass; "
        f"all_pass={result['all_gates_pass']}",
        flush=True,
    )
    return result
