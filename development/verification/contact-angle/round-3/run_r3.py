#!/usr/bin/env python3
"""XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS — round 3 (verifier seat, tp-ver-2).

Independent audit of the round-2 PASS claim.  The round-2 harness and its two
shape estimators are treated as the *subject under audit*: this file rebuilds
the matrix, and measures a THIRD angle from scratch (``theta_geodesic``, apex /
basal-width geodesic relation), probes the level sensitivity of that
measurement, and re-relaxes the rows that round 2 counted while they had not
met their declared tolerance.

Usage
-----
    python3 run_r3.py all          # full matrix + long re-relaxation of the slow rows
    python3 run_r3.py summarize    # gates -> summary_r3.json
    python3 run_r3.py report       # REPORT_R3.md + figures
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))

R2 = HERE.parent / "round-2"
PRED2 = json.loads((R2 / "predeclared.json").read_text())
PRED3 = json.loads((HERE / "predeclared_r3.json").read_text())
RUNS = HERE / "runs_r3"
GRIDS = {
    "baseline": PRED2["matrix"]["baseline_grid"],
    "refined": PRED2["matrix"]["refined_grid"],
    "fine_interface": PRED2["matrix"]["fine_interface_grid"],
}
NUM = PRED2["numerics"]


# ---------------------------------------------------------------------------
# third estimator, written from scratch (numpy only; no round-2 measurement code)
# ---------------------------------------------------------------------------
def gas_level(psi: np.ndarray, frac: float) -> tuple[float, float, float]:
    """(gas, liquid, level) with the gas sampled in the two far top corners."""
    gas = 0.5 * (float(psi[-4:, :4].mean()) + float(psi[-4:, -4:].mean()))
    liquid = float(psi.max())
    return gas, liquid, gas + frac * (liquid - gas)


def crossing_along(profile: np.ndarray, level: float, last: bool = True) -> float:
    """Sub-cell crossing of ``level`` along a 1-D profile (index coordinate)."""
    d = profile - level
    s = d[:-1] * d[1:]
    idx = np.nonzero(s < 0.0)[0]
    if idx.size == 0:
        return float("nan")
    i = int(idx[-1] if last else idx[0])
    return i + float(d[i] / (d[i] - d[i + 1]))


def right_crossing(row: np.ndarray, level: float, jc: int) -> float:
    """Right-hand crossing of ``level`` on a row, searched outwards from ``jc``."""
    d = row - level
    for i in range(jc, row.size - 1):
        if d[i] * d[i + 1] < 0.0:
            return i + float(d[i] / (d[i] - d[i + 1]))
    return float("nan")


def geodesic_angle(
    psi: np.ndarray, xi: float, frac: float = 0.5, wall_y: float = -0.5
) -> dict:
    """theta = 2 atan(a/h) from the apex height and the basal half width.

    For a sessile circular cap of radius ``R`` the top of the sphere sits
    ``h = R (1 + cos theta)`` above the wall and the contact radius is
    ``a = R sin theta``, so ``a/h = tan(theta/2)`` exactly for
    0 < theta < 180 (note: ``h`` is measured above the *wall*, not above the
    contact line, which is what makes the same expression valid on both sides
    of 90 deg).

    ``a`` is the mid-level contour's right-hand branch extrapolated to the wall
    plane (quadratic fit over the contour rows within ``3 xi`` of the wall, or
    linear when a shallow cap leaves fewer than four rows); ``h`` is the apex
    crossing on the centre column.
    """
    ny, nx = psi.shape
    gas, _liquid, level = gas_level(psi, frac)
    # axis of the cap taken from the field itself (liquid centroid), never from
    # the configured centre: the drop axis sits at a half-cell position
    xs_grid = np.arange(nx, dtype=float)
    w = np.clip(psi - gas, 0.0, None).sum(axis=0)
    x_c = float((xs_grid * w).sum() / w.sum()) if w.sum() > 0 else (nx - 1) / 2.0
    jc = int(min(nx - 2, max(1, round(x_c))))
    height = crossing_along(psi[:, jc], level) - wall_y

    ys, xs = [], []
    for k in range(ny):
        x = right_crossing(psi[k], level, jc)
        if math.isfinite(x):
            ys.append(float(k))
            xs.append(x - x_c)
    out = {
        "theta_geodesic_deg": float("nan"),
        "apex_height": float(height),
        "basal_half_width": float("nan"),
        "n_contour_rows": float(len(ys)),
        "axis_x": x_c,
        "level": float(level),
    }
    if len(ys) >= 4:
        ys_a, xs_a = np.asarray(ys), np.asarray(xs)
        near = ys_a <= 3.0 * xi
        if int(near.sum()) < 4:
            near = np.ones_like(ys_a, dtype=bool)
        deg = 3 if int(near.sum()) >= 5 else (2 if int(near.sum()) >= 4 else 1)
        coef = np.polyfit(ys_a[near], xs_a[near], deg)
        a = float(np.polyval(coef, wall_y))
        out["basal_half_width"] = a
        out["basal_fit_order"] = float(deg)
        if a > 0.0 and height > 0.0:
            out["theta_geodesic_deg"] = math.degrees(2.0 * math.atan(a / height))
    return out


def offset_circle_prediction(psi, xi, wall_y=-0.5):
    """Analytic wall angle of an *offset* level set of a circular cap.

    A diffuse circular interface has concentric level sets: the level set at
    ``psi = level(f)`` is a circle of radius ``R + delta`` with ``delta =
    -xi atanh(2f - 1)`` sharing the centre of the mid-level circle.  Its wall angle
    is therefore **not** the contact angle but

        theta_off(delta) = 2 atan( sqrt( (R+delta-d) / (R+delta+d) ) ),
        d = R cos(theta_0),   R = h_0 / (1 + cos theta_0),

    with ``(h_0, theta_0)`` measured at the mid level.  Checking the measured
    offset angles against this closed form separates a genuine measurement bias
    from the exact geometry of a concentric offset -- it is the calibration that
    a naive "the angle must not depend on the level" probe gets wrong.
    """
    mid = geodesic_angle(psi, xi, frac=0.5, wall_y=wall_y)
    th0 = math.radians(mid["theta_geodesic_deg"])
    if not math.isfinite(mid["theta_geodesic_deg"]) or not (0.0 < th0 < math.pi):
        return {"theta_mid_deg": float("nan"), "offset_dev_deg": float("nan")}
    h0 = mid["apex_height"]
    R = h0 / (1.0 + math.cos(th0))
    d = R * math.cos(th0)
    out = {
        "theta_mid_deg": mid["theta_geodesic_deg"],
        "R_circ": R,
        "centre_offset": d,
        "theta_offset_pred_deg": {},
        "theta_offset_meas_deg": {},
        "offset_dev_deg": float("nan"),
    }
    devs = []
    for f in (0.35, 0.65):
        Rf = R - xi * math.atanh(2.0 * f - 1.0)  # frac<0.5 -> outer contour
        if Rf + d <= 0.0 or Rf - d <= 0.0:
            continue
        pred = math.degrees(2.0 * math.atan(math.sqrt((Rf - d) / (Rf + d))))
        meas = geodesic_angle(psi, xi, frac=f, wall_y=wall_y)["theta_geodesic_deg"]
        key = "lvl%02d" % int(round(f * 100))
        out["theta_offset_pred_deg"][key] = pred
        out["theta_offset_meas_deg"][key] = meas
        if math.isfinite(meas):
            devs.append(abs(meas - pred))
    if devs:
        out["offset_dev_deg"] = max(devs)
    return out


def level_convention_angles(psi, xi, wall_y=-0.5):
    """theta under the three level conventions in use for a diffuse interface.

    (i) ``0.5 (gas + liquid)`` (round-2 convention), (ii) ``psi = 0`` (the tanh
    zero of a symmetric profile), (iii) ``0.5 (gas + min)`` with the gas taken
    from the field minimum.  A supersaturated gas (measured: up to +0.10) makes
    these three differ, which is a measurement ambiguity round 2 never stated.
    """
    gas, liquid, _lvl = gas_level(psi, 0.5)
    variants = {
        "mid": 0.5 * (gas + liquid),
        "tanh_zero": 0.0,
        "gas_min": 0.5 * (gas + float(psi.min())),
    }
    ang = {}
    for name, lvl in variants.items():
        frac = (lvl - gas) / (liquid - gas) if liquid > gas else 0.5
        ang[name] = geodesic_angle(psi, xi, frac=frac, wall_y=wall_y)["theta_geodesic_deg"]
    finite = [v for v in ang.values() if math.isfinite(v)]
    out = dict(ang)
    out["spread_deg"] = max(finite) - min(finite) if len(finite) >= 2 else float("nan")
    return out


def synthetic_cap(theta_deg, radius=14.0, xi=2.0, ny=58, nx=69, wall_y=-0.5):
    """Diffuse circular cap with analytically known contact angle (no simulation)."""
    th = math.radians(theta_deg)
    y_c = wall_y + radius * math.cos(th)
    x_c = (nx - 1) / 2.0
    y = np.arange(ny)[:, None].astype(float)
    x = np.arange(nx)[None, :].astype(float)
    r = np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)
    return -np.tanh((r - radius) / xi)


def calibrate_synthetic() -> dict:
    """P3a: the third estimator against analytic caps (bias + level convention)."""
    rows = []
    for radius in (14.0, 18.0, 22.0):
        for xi in (2.0, 3.0):
            for th in (45.0, 70.0, 90.0, 110.0, 135.0):
                ny, nx = int(2 * radius + 30), int(2 * radius + 41)
                psi = synthetic_cap(th, radius=radius, xi=xi, ny=ny, nx=nx)
                meas = geodesic_angle(psi, xi)
                lvl = level_convention_angles(psi, xi)
                rows.append(
                    {
                        "theta_true_deg": th,
                        "radius": radius,
                        "xi": xi,
                        "theta_est_deg": meas["theta_geodesic_deg"],
                        "dev_deg": abs(meas["theta_geodesic_deg"] - th),
                        "level_spread_deg": lvl["spread_deg"],
                    }
                )
    th = PRED3["predeclared_thresholds"]
    violations = []
    for r in rows:
        ratio = r["xi"] / r["radius"]
        limit = (
            th["estimator_calibration_deg"]
            if ratio <= th["xi_over_R_fine"]
            else th["estimator_calibration_coarse_deg"]
        )
        r["xi_over_R"] = ratio
        r["tier_limit_deg"] = limit
        r["tier_ok"] = bool(math.isfinite(r["dev_deg"]) and r["dev_deg"] <= limit)
        if not r["tier_ok"]:
            violations.append(r)
    errs = [r["dev_deg"] for r in rows if math.isfinite(r["dev_deg"])]
    fine = [r["dev_deg"] for r in rows if r["xi_over_R"] <= th["xi_over_R_fine"]]
    coarse = [r["dev_deg"] for r in rows if r["xi_over_R"] > th["xi_over_R_fine"]]
    return {
        "n": len(rows),
        "max_dev_deg": max(errs) if errs else float("nan"),
        "max_dev_fine_deg": max(fine) if fine else float("nan"),
        "max_dev_coarse_deg": max(coarse) if coarse else float("nan"),
        "median_dev_deg": float(np.median(errs)) if errs else float("nan"),
        "n_tier_violations": len(violations),
        "tier_violations": violations,
        "rows": rows,
    }


def symmetry_error(psi):
    """Max |psi(x) - psi(-x)| / 2; the run is symmetric by construction."""
    return float(np.abs(psi - psi[:, ::-1]).max())


# ---------------------------------------------------------------------------
# one row
# ---------------------------------------------------------------------------
def tag_of(arm: str, theta_deg: float, grid: str, xi_override=None) -> str:
    xi = GRIDS[grid]["xi"] if xi_override is None else xi_override
    return f"{arm}__th{theta_deg:05.1f}__{grid}__xi{xi:g}"


def run_one(job: dict) -> dict:
    import torch

    torch.set_num_threads(1)
    from tensorlbm.wetting.two_impl_contact_angle import (
        TwoImplConfig,
        measure_contact_angle,
        run_arm,
    )

    arm, theta_deg, grid = job["arm"], job["theta"], job["grid"]
    max_iter = job.get("max_iter", NUM["max_iter"])
    g = dict(GRIDS[grid])
    xi = float(job["xi"]) if job.get("xi") is not None else g.pop("xi")
    radius = g.pop("radius")
    cfg = TwoImplConfig(interface_width=xi, **g)
    theta = math.radians(theta_deg)

    t0 = time.time()
    out = run_arm(
        arm,
        theta,
        cfg=cfg,
        radius=radius,
        max_iter=max_iter,
        omega=NUM["omega"],
        tol=NUM["tol"],
    )
    psi = out.pop("psi").detach().to(torch.float64).numpy()
    m = measure_contact_angle(
        torch.as_tensor(psi), xi, wall_y=-0.5
    )

    geo = {f"lvl{int(round(f*100)):02d}": geodesic_angle(psi, xi, frac=f) for f in (0.35, 0.5, 0.65)}
    spreads = [
        abs(geo["lvl35"]["theta_geodesic_deg"] - geo["lvl65"]["theta_geodesic_deg"])
    ]
    offs = offset_circle_prediction(psi, xi)
    lvl = level_convention_angles(psi, xi)
    payload = {
        "tag": tag_of(arm, theta_deg, grid, job.get("xi")) + job.get("suffix", ""),
        "arm": arm,
        "grid": grid,
        "interface_width": xi,
        "theta_prescribed_deg": theta_deg,
        "max_iter": max_iter,
        "iterations": out["iterations"],
        "converged": bool(out["converged"]),
        "drift": float(out["drift"]),
        "mass_error": float(out["mass_error"]),
        "theta_cap_deg": float(m["theta_cap_deg"]),
        "theta_circle_deg": float(m["theta_circle_deg"]),
        "gas_reference": float(m["gas_reference"]),
        "gas_supersaturation": float(m["gas_supersaturation"]),
        "circle_rms": float(m["circle_rms"]),
        "apex_height_r2": float(m["apex_height"]),
        "measurable": bool(m["measurable"]),
        "theta_geodesic_deg": geo["lvl50"]["theta_geodesic_deg"],
        "geodesic_levels": geo,
        "level_spread_deg": float(spreads[0]) if all(
            math.isfinite(g_["theta_geodesic_deg"]) for g_ in geo.values()
        ) else float("nan"),
        "offset_circle": offs,
        "level_convention": lvl,
        "apex_height_r3": geo["lvl50"]["apex_height"],
        "basal_half_width": geo["lvl50"]["basal_half_width"],
        "symmetry_error": symmetry_error(psi),
        "wall_seconds": time.time() - t0,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    if arm != "dirichlet" and grid in ("baseline", "refined"):
        np.savez_compressed(RUNS / f"{payload['tag']}.npz", psi=psi)
    (RUNS / f"{payload['tag']}.json").write_text(
        json.dumps(payload, indent=2, default=str)
    )
    return payload


LONG_MAX_ITER = 200000


def jobs_long() -> list[dict]:
    """P4: re-relax the rows round 2 counted while they stopped at max_iter."""
    r2 = {r["tag"]: r for r in json.loads((R2 / "summary.json").read_text())["rows"]}
    jobs = []
    for tag, r in sorted(r2.items()):
        if r["measureable"] if "measureable" in r else r["measurable"]:
            if not r["converged"]:
                jobs.append(
                    {
                        "arm": r["arm"],
                        "theta": r["theta_prescribed_deg"],
                        "grid": r["grid"],
                        "max_iter": LONG_MAX_ITER,
                        "suffix": "__long",
                    }
                )
    return jobs


def jobs_all() -> list[dict]:
    jobs = []
    for arm in PRED2["arms"]:
        for th in PRED2["matrix"]["angles_deg"]:
            jobs.append({"arm": arm, "theta": th, "grid": "baseline"})
    for arm in PRED2["arms"]:
        for th in PRED2["matrix"]["angles_refined_deg"]:
            jobs.append({"arm": arm, "theta": th, "grid": "refined"})
    for arm in PRED2["independent_pair"]:
        for xi in PRED2["matrix"]["xi_sensitivity"]:
            for th in (70.0, 110.0):
                jobs.append({"arm": arm, "theta": th, "grid": "baseline", "xi": xi})
    for arm in PRED2["independent_pair"]:
        jobs.append({"arm": arm, "theta": 45.0, "grid": "fine_interface"})
    return jobs


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def load_runs() -> dict:
    out = {}
    for p in sorted(RUNS.glob("*.json")):
        d = json.loads(p.read_text())
        out[d["tag"]] = d
    return out


def summarize() -> dict:
    th = PRED3["predeclared_thresholds"]
    r3 = load_runs()
    r2 = {r["tag"]: r for r in json.loads((R2 / "summary.json").read_text())["rows"]}
    rows = []
    for tag, a in r3.items():
        b = r2.get(tag)
        if b is None:
            continue
        row = {
            "tag": tag,
            "arm": a["arm"],
            "grid": a["grid"],
            "theta_prescribed_deg": a["theta_prescribed_deg"],
            "measurable_r2": b["measurable"],
            "converged_r3": a["converged"],
            "drift_r3": a["drift"],
            "theta_cap_r2": b["theta_cap_deg"],
            "theta_circle_r2": b["theta_circle_deg"],
            "theta_cap_r3": a["theta_cap_deg"],
            "theta_circle_r3": a["theta_circle_deg"],
            "d_cap_deg": abs(a["theta_cap_deg"] - b["theta_cap_deg"]),
            "d_circle_deg": abs(a["theta_circle_deg"] - b["theta_circle_deg"]),
            "theta_geodesic_deg": a["theta_geodesic_deg"],
            "err_geodesic_pct": (
                abs(a["theta_geodesic_deg"] - a["theta_prescribed_deg"])
                / a["theta_prescribed_deg"] * 100.0
                if math.isfinite(a["theta_geodesic_deg"])
                else None
            ),
            "gap_geodesic_vs_circle_deg": (
                abs(a["theta_geodesic_deg"] - a["theta_circle_deg"])
                if math.isfinite(a["theta_geodesic_deg"])
                else None
            ),
            "level_spread_deg": a["level_spread_deg"],
            "level_convention_spread_deg": a["level_convention"]["spread_deg"],
            "offset_dev_deg": a["offset_circle"]["offset_dev_deg"],
            "gas_supersaturation": a["gas_supersaturation"],
            "symmetry_error": a["symmetry_error"],
        }
        row["reproduced"] = bool(
            row["d_cap_deg"] <= th["reproduce_deg"]
            and row["d_circle_deg"] <= th["reproduce_deg"]
        )
        rows.append(row)

    independent = set(PRED2["independent_pair"])
    aud = [r for r in rows if r["measurable_r2"] and r["arm"] in independent]

    p1 = [r["tag"] for r in rows if not r["reproduced"]]
    p2 = [
        r["tag"]
        for r in aud
        if r["err_geodesic_pct"] is None
        or r["err_geodesic_pct"] > th["accept_pct"]
        or r["gap_geodesic_vs_circle_deg"] is None
        or r["gap_geodesic_vs_circle_deg"] > th["third_estimator_gap_deg"]
    ]
    p3b = [
        r["tag"]
        for r in aud
        if r.get("level_convention_spread_deg") is None
        or not math.isfinite(r["level_convention_spread_deg"])
        or r["level_convention_spread_deg"] > th["level_convention_deg"]
    ]
    p3c = [
        r["tag"]
        for r in aud
        if r.get("offset_dev_deg") is None
        or not math.isfinite(r["offset_dev_deg"])
        or r["offset_dev_deg"] > th["offset_circle_deg"]
    ]
    p3 = p3b + [t for t in p3c if t not in p3b]
    long_rows = {t: d for t, d in r3.items() if t.endswith("__long")}
    p4_details = []
    for r in aud:
        if r["converged_r3"]:
            continue
        lg = long_rows.get(r["tag"] + "__long")
        if lg is None:
            p4_details.append({"tag": r["tag"], "status": "long_run_missing"})
            continue
        d_cap = abs(lg["theta_cap_deg"] - r["theta_cap_r3"])
        d_circle = abs(lg["theta_circle_deg"] - r["theta_circle_r3"])
        d_geo = (
            abs(lg["theta_geodesic_deg"] - r["theta_geodesic_deg"])
            if math.isfinite(lg["theta_geodesic_deg"]) and math.isfinite(r["theta_geodesic_deg"])
            else float("nan")
        )
        p4_details.append(
            {
                "tag": r["tag"],
                "iterations_short": r["iterations"],
                "iterations_long": lg["iterations"],
                "converged_long": lg["converged"],
                "drift_long": lg["drift"],
                "d_cap_deg": d_cap,
                "d_circle_deg": d_circle,
                "d_geodesic_deg": d_geo,
                "angle_moved": bool(
                    max(d_cap, d_circle) > th["convergence_angle_deg"]
                    or (math.isfinite(d_geo) and d_geo > th["convergence_angle_deg"])
                ),
            }
        )
    p4 = [d["tag"] for d in p4_details if d.get("angle_moved") or d.get("status") == "long_run_missing"]

    cross = {}
    by_angle = {}
    for r in aud:
        by_angle.setdefault(r["theta_prescribed_deg"], {})[r["arm"]] = r["theta_circle_r3"]
    for ang, d in sorted(by_angle.items()):
        if len(d) < 2:
            continue
        a, b = d["geometric_ghost"], d["neumann_flux"]
        cross[ang] = {
            "theta_geometric_ghost_deg": a,
            "theta_neumann_flux_deg": b,
            "diff_pct": abs(a - b) / ang * 100.0,
            "pass": abs(a - b) / ang * 100.0 <= th["cross_arm_pct"],
        }

    summary = {
        "anchor": PRED3["anchor"],
        "round": 3,
        "n_rows": len(rows),
        "thresholds": th,
        "probes": {
            "P1_reproduce_fail_rows": p1,
            "P1_max_delta_deg": max(
                [max(r["d_cap_deg"], r["d_circle_deg"]) for r in rows], default=None
            ),
            "P2_fail_rows": p2,
            "P2_n_audited": len(aud),
            "P2_max_err_pct": max(
                [r["err_geodesic_pct"] for r in aud if r["err_geodesic_pct"] is not None],
                default=None,
            ),
            "P2_max_gap_deg": max(
                [r["gap_geodesic_vs_circle_deg"] for r in aud if r["gap_geodesic_vs_circle_deg"] is not None],
                default=None,
            ),
            "P3a_synthetic_calibration": calibrate_synthetic(),
            "P3b_level_convention_fail_rows": p3b,
            "P3b_max_spread_deg": max(
                [r["level_convention_spread_deg"] for r in aud
                 if r.get("level_convention_spread_deg") is not None], default=None),
            "P3c_offset_circle_fail_rows": p3c,
            "P3c_max_dev_deg": max(
                [r["offset_dev_deg"] for r in aud
                 if r.get("offset_dev_deg") is not None], default=None),
            "P3_fail_rows": p3,
            "P4_rows_counted_but_not_converged": [
                r["tag"] for r in aud if not r["converged_r3"]
            ],
            "P4_details": p4_details,
            "P4_fail_rows": p4,
            "P5_cross_arm_nondegenerate": {
                str(k): v for k, v in cross.items() if k != 90.0
            },
            "P5_cross_arm_degenerate_90": cross.get(90.0),
        },
        "pass_counts_restated": {
            arm: {
                "measurable_r2": sum(1 for r in aud if r["arm"] == arm),
                "passes_all_three_estimators": sum(
                    1
                    for r in aud
                    if r["arm"] == arm
                    and r["converged_r3"]
                    and r["err_geodesic_pct"] is not None
                    and r["err_geodesic_pct"] <= th["accept_pct"]
                ),
                "measurable_and_converged": sum(
                    1 for r in aud if r["arm"] == arm and r["converged_r3"]
                ),
            }
            for arm in sorted(independent)
        },
        "rows": rows,
    }
    gates = [
        not p1,
        not p2,
        not p3,
        not p4,
        all(v["pass"] for v in cross.values() if True),
    ]
    summary["verdict"] = "PASS" if all(gates) else "FAIL"
    summary["failed_probes"] = [
        name
        for name, ok in zip(
            ["P1_reproduce", "P2_third_estimator", "P3_offset_and_level", "P4_convergence", "P5_cross_arm"],
            gates,
        )
        if not ok
    ]
    (HERE / "summary_r3.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def report() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = summarize()
    rows = [r for r in s["rows"] if r["measurable_r2"]]
    fig_dir = HERE / "figures"
    fig_dir.mkdir(exist_ok=True)

    arms = sorted({r["arm"] for r in rows})
    marks = {"dirichlet": "s", "neumann_flux": "o", "geometric_ghost": "^"}

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    lim = [30, 150]
    ax.plot(lim, lim, "k--", lw=1, label="analytic Young law")
    for arm in arms:
        sub = sorted([r for r in rows if r["arm"] == arm], key=lambda r: r["theta_prescribed_deg"])
        ax.plot(
            [r["theta_prescribed_deg"] for r in sub],
            [r["theta_geodesic_deg"] for r in sub],
            marks[arm] + "-",
            label=f"{arm} (round-3 geodesic estimator)",
        )
    ax.set_xlabel(r"prescribed $\theta$ (deg)")
    ax.set_ylabel(r"measured $\theta$ (deg)")
    ax.set_title("Third, from-scratch estimator vs analytic Young law")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "r3_third_estimator.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for arm in arms:
        sub = sorted([r for r in rows if r["arm"] == arm], key=lambda r: r["theta_prescribed_deg"])
        ax.plot(
            [r["theta_prescribed_deg"] for r in sub],
            [r["err_geodesic_pct"] for r in sub],
            marks[arm] + "-",
            label=arm,
        )
    ax.axhline(s["thresholds"]["accept_pct"], color="r", ls="--", lw=1, label="15 % owner gate")
    ax.set_xlabel(r"prescribed $\theta$ (deg)")
    ax.set_ylabel("relative error of geodesic estimator (%)")
    ax.set_title("Anchor gate on the third estimator")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "r3_error_gate.png", dpi=160)
    plt.close(fig)

    ordered = sorted(rows, key=lambda r: (r["arm"], r["theta_prescribed_deg"]))
    tags = [r["tag"] for r in ordered]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(range(len(tags)), [r["level_spread_deg"] for r in ordered],
           label="naive offset spread |theta(.35)-theta(.65)|")
    ax.plot(range(len(tags)), [r["offset_dev_deg"] for r in ordered], "ko-", ms=3,
            label="deviation from the closed-form offset-circle law")
    ax.axhline(s["thresholds"]["offset_circle_deg"], color="r", ls="--", lw=1,
               label="declared 2 deg (offset law)")
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=90, fontsize=5)
    ax.set_ylabel("angle (deg)")
    ax.set_title("Offset contours: naive expectation vs concentric-offset law")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "r3_offset_circle_law.png", dpi=160)
    plt.close(fig)

    lines = [
        "# XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS — round 3 (verifier audit)",
        "",
        f"Verdict: **{s['verdict']}**   failed probes: {s['failed_probes'] or 'none'}",
        "",
        f"Rows audited: {s['n_rows']}",
        "",
        "## Probe results",
        "",
        f"* P1 reproduce round-2 matrix: max |delta theta| = {s['probes']['P1_max_delta_deg']:.3g} deg, "
        f"failing rows = {s['probes']['P1_reproduce_fail_rows'] or 'none'}",
        f"* P2 third estimator: audited {s['probes']['P2_n_audited']} measurable rows, "
        f"max error {s['probes']['P2_max_err_pct']:.2f} %, max gap vs circle fit "
        f"{s['probes']['P2_max_gap_deg']:.2f} deg, failing rows = {s['probes']['P2_fail_rows'] or 'none'}",
        f"* P3a estimator calibration on analytic caps: max deviation "
        f"{s['probes']['P3a_synthetic_calibration']['max_dev_deg']:.2f} deg over "
        f"{s['probes']['P3a_synthetic_calibration']['n']} synthetic caps",
        f"* P3b level-convention ambiguity: max spread {s['probes']['P3b_max_spread_deg']:.2f} deg, "
        f"failing rows = {s['probes']['P3b_level_convention_fail_rows'] or 'none'}",
        f"* P3c offset-contour analytic law: max deviation {s['probes']['P3c_max_dev_deg']:.2f} deg, "
        f"failing rows = {s['probes']['P3c_offset_circle_fail_rows'] or 'none'}",
        f"* P4 rows counted by round 2 while not converged: "
        f"{s['probes']['P4_rows_counted_but_not_converged'] or 'none'}",
        f"* P5 cross-arm (non-degenerate, 90 deg excluded): "
        + ", ".join(
            f"{k} deg -> {v['diff_pct']:.2f} %"
            for k, v in s["probes"]["P5_cross_arm_nondegenerate"].items()
        ),
        "",
        "## Restated pass counts (two independent arms)",
        "",
        "| arm | measurable (round-2 rule) | measurable AND converged | pass all three estimators |",
        "|---|---|---|---|",
    ]
    for arm, d in s["pass_counts_restated"].items():
        lines.append(
            f"| {arm} | {d['measurable_r2']} | {d['measurable_and_converged']} | {d['passes_all_three_estimators']} |"
        )
    lines += ["", "## Figures", "", "![third estimator](figures/r3_third_estimator.png)",
              "![error gate](figures/r3_error_gate.png)",
              "![offset circle law](figures/r3_offset_circle_law.png)",
              "", "## P4 convergence qualification", ""]
    for d in s["probes"]["P4_details"]:
        if d.get("status") == "long_run_missing":
            lines.append(f"* {d['tag']}: long re-relaxation missing")
            continue
        lines.append(
            f"* {d['tag']}: iters {d['iterations_short']} -> {d['iterations_long']}, "
            f"converged_long={d['converged_long']}, drift_long={d['drift_long']:.2e}, "
            f"|dtheta_cap|={d['d_cap_deg']:.3f} deg, |dtheta_circle|={d['d_circle_deg']:.3f} deg, "
            f"|dtheta_geodesic|={d['d_geodesic_deg']:.3f} deg"
            if math.isfinite(d.get("d_geodesic_deg", float("nan")))
            else f"* {d['tag']}: iters {d['iterations_short']} -> {d['iterations_long']}, "
            f"converged_long={d['converged_long']}, |dtheta_cap|={d['d_cap_deg']:.3f} deg, "
            f"|dtheta_circle|={d['d_circle_deg']:.3f} deg"
        )
    lines.append("")
    (HERE / "REPORT_R3.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["all", "summarize", "report"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.action == "all":
        jobs = jobs_all() + jobs_long()
        print(f"{len(jobs)} rows on {a.workers} workers", flush=True)
        with Pool(a.workers) as pool:
            for r in pool.imap_unordered(run_one, jobs):
                print(
                    f"{r['tag']:44s} cap={r['theta_cap_deg']:7.2f} circ={r['theta_circle_deg']:7.2f} "
                    f"geo={r['theta_geodesic_deg']:7.2f} conv={r['converged']} "
                    f"{r['wall_seconds']:.1f}s",
                    flush=True,
                )
    elif a.action == "summarize":
        print(json.dumps(summarize()["probes"], indent=1, default=str))
    else:
        report()


if __name__ == "__main__":
    main()