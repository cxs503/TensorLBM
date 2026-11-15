#!/usr/bin/env python3
"""XVAL-LBM-AKAI2018-GEOMETRICAL-WETTING, round R5: analytic-field calibration of
the contact-angle *estimators*, and re-adjudication of the theta = 30 deg arm.

Why this study exists
---------------------
Round R4 (``docs/XVAL-LBM-AKAI2018-GEOMETRICAL-WETTING-R4-OPP-FIX.md``) ended with
two open items:

1. the theta = 30 deg arm still read 39.5 deg (+32 %) and was attributed to a
   *resolution floor of the cap geometry*: ``h = R(1-cos theta) = 1.88 cells`` is
   thinner than the diffuse interface ``xi = 2``, so -- the argument went -- "the
   isocontour returned by the circle fit is a remnant swallowed by the interface
   thickness".  The document then *proposed* the criterion

       R (1 - cos theta) / xi  >=  2.5

   but never tested it (the two cluster jobs that would have tested it, cj3864 /
   cj3865, were never scheduled).
2. no independent estimator existed that could read the 30 deg arm at all: on the
   saved fields the circle fit returns ``nan`` (``benchmarks/xval_akai2018_wetting/
   log_gk30.txt``), so "the estimator is wrong" and "the physics is wrong" could not
   be separated.

Both are estimator/geometry questions, and both can be answered *without a solver
run*: take a field whose contact angle is known by construction, hand it to each
estimator, and measure the bias.  This script does exactly that.

Analytic reference field
------------------------
The solver initialises (``run.py:init_drop``) and the analyser assumes the same
object: a circular cap of radius ``R`` above a plate at ``y_wall``, diffused with
the colour-gradient ``tanh`` profile,

    phi(x, y) = tanh( (R - r) / xi ),   r = |x - x_c, y - y_c|,
    y_c = y_wall - R cos(theta)                        (cap centre below the plate)

so the ``phi = 0`` isocontour *is* the circular arc of the cap, exactly, for any
``xi``: this field isolates the estimators from the physics.  It is evaluated on
the same lattice (``ny, nx``), the same two solid rows, and the same ``y_wall=0.5``
as the production case, so discretisation is included.

Estimators under test
---------------------
* ``circle``  : ``tensorlbm.wetting.metrics.contact_angle_circle_fit`` (the R1-R4
                decision metric, ``level=0``, ``trim=0.05``)
* ``cap_area``: ``tensorlbm.cg_contact_angle.contact_angle_cap_inversion``
* ``cap_wm``  : ``tensorlbm.wetting.metrics.spherical_cap_inversion`` (the second
                metric reported by ``run.py``)
* ``normal``  : ``tensorlbm.cg_contact_angle.contact_angle_local_normal`` (the
                near-wall interface-normal read-back)

Part 2 of the script re-adjudicates the *saved solver fields* of the R4 campaign
with the estimators that part 1 validates, and reports each field's cap height and
base width against the value a true cap of the target angle would have -- i.e. it
asks "is the settled shape a 30 deg cap at all?" rather than "what does one
estimator say?".

Outputs (all programmatic, nothing hand-copied)
-----------------------------------------------
``--out DIR`` (default ``artifacts/m22257_akai_r5``):
    ``estimator_calibration.json``  every case, every estimator, bias in degrees
    ``fig_estimator_calibration.png`` bias vs ``h/xi`` per estimator + floor map
    ``fig_shape_adjudication.png``  saved solver fields vs the exact cap contour
    ``summary.json``                aggregate: floors, max bias, verdict per arm

Usage
-----
python benchmarks/xval_akai2018_wetting/estimator_analytic_study.py \
    [--out artifacts/m22257_akai_r5] [--solver-dir benchmarks/xval_akai2018_wetting]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _resolve_src() -> Path:
    import os

    cands = []
    if "--src" in sys.argv:
        i = sys.argv.index("--src")
        if i + 1 < len(sys.argv):
            cands.append(Path(sys.argv[i + 1]))
    if os.environ.get("TENSORLBM_SRC"):
        cands.append(Path(os.environ["TENSORLBM_SRC"]))
    cands.append(HERE.parents[1] / "src")
    for c in cands:
        if (c / "tensorlbm" / "d2q9.py").is_file():
            return c
    raise SystemExit("repo src/ not found: pass --src PATH or set TENSORLBM_SRC")


sys.path.insert(0, str(_resolve_src()))

import numpy as np  # noqa: E402

from tensorlbm.cg_contact_angle import (  # noqa: E402
    contact_angle_cap_inversion,
    contact_angle_local_normal,
    interface_contour,
)
from tensorlbm.wetting.metrics import (  # noqa: E402
    contact_angle_circle_fit,
    spherical_cap_inversion,
)

Y_WALL = 0.5


# ---------------------------------------------------------------------------
# Analytic sessile cap
# ---------------------------------------------------------------------------
def cap_field(
    radius: float,
    theta_deg: float,
    xi: float,
    nx: int = 96,
    ny: int = 64,
    y_wall: float = Y_WALL,
    x_center: float | None = None,
) -> np.ndarray:
    """Exact diffused circular cap: ``phi = tanh((R - r)/xi)``, ``+1`` inside."""
    th = math.radians(theta_deg)
    xc = (nx - 1) / 2.0 if x_center is None else float(x_center)
    yc = y_wall - radius * math.cos(th)
    y = np.arange(ny, dtype=float)[:, None]
    x = np.arange(nx, dtype=float)[None, :]
    r = np.hypot(x - xc, y - yc)
    return np.tanh((radius - r) / xi)


def solid_grid(nx: int, ny: int) -> np.ndarray:
    solid = np.zeros((ny, nx), dtype=bool)
    solid[0, :] = True
    solid[-1, :] = True
    return solid


def grid_for(radius: float, theta_deg: float, pad: float = 24.0) -> tuple[int, int]:
    """Lattice that holds the whole cap with ``pad`` cells of vapour margin."""
    th = math.radians(theta_deg)
    height = radius * (1.0 - math.cos(th))
    width = 2.0 * radius * math.sin(th)
    nx = int(max(96, math.ceil(width + pad)))
    ny = int(max(48, math.ceil(height + pad)))
    return nx, ny


# ---------------------------------------------------------------------------
# Estimator harness
# ---------------------------------------------------------------------------
def _call(fn, *args, **kwargs):
    """Run an estimator; return ``(theta_deg, note)`` and never raise."""
    try:
        out = fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - diagnostic path
        return float("nan"), type(exc).__name__ + ": " + str(exc)[:80]
    val = out.get("theta_deg", float("nan"))
    return float(val), "ok"


def measure_all(phi: np.ndarray, solid: np.ndarray) -> dict:
    ths = {}
    ths["circle"], n1 = _call(
        contact_angle_circle_fit, phi, Y_WALL, trim=0.05, level=0.0
    )
    ths["cap_area"], n2 = _call(contact_angle_cap_inversion, phi, Y_WALL, level=0.0)
    ths["cap_wm"], n3 = _call(spherical_cap_inversion, phi, Y_WALL, level=0.0)
    ths["normal"], n4 = _call(
        contact_angle_local_normal, phi, solid, Y_WALL, level=0.0, band=3.0
    )
    return {"theta": ths, "notes": {"circle": n1, "cap_area": n2, "cap_wm": n3, "normal": n4}}


# ---------------------------------------------------------------------------
# Part 1: calibration sweep
# ---------------------------------------------------------------------------
THETAS = (15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 150.0)
RADII = (8.0, 10.0, 14.0, 20.0, 30.0, 40.0)
XIS = (1.0, 2.0, 3.0)


def calibration_sweep() -> list[dict]:
    rows = []
    for xi in XIS:
        for theta in THETAS:
            for radius in RADII:
                nx, ny = grid_for(radius, theta)
                phi = cap_field(radius, theta, xi, nx=nx, ny=ny)
                solid = solid_grid(nx, ny)
                meas = measure_all(phi, solid)
                h = radius * (1.0 - math.cos(math.radians(theta)))
                row = {
                    "theta_true": theta,
                    "radius": radius,
                    "xi": xi,
                    "h": h,
                    "h_over_xi": h / xi,
                    "apex_above_wall_cells": h,  # apex height in lattice units
                    "nx": nx,
                    "ny": ny,
                    "status": meas["notes"],
                }
                for k, v in meas["theta"].items():
                    row[f"theta_{k}"] = v
                    row[f"bias_{k}"] = (
                        v - theta if math.isfinite(v) else float("nan")
                    )
                rows.append(row)
    return rows


def summarise(rows: list[dict], estimators=("circle", "cap_area", "cap_wm", "normal")) -> dict:
    out: dict = {"n_cases": len(rows), "estimators": {}}
    for est in estimators:
        ok = [r for r in rows if math.isfinite(r[f"theta_{est}"])]
        bias = np.array([r[f"bias_{est}"] for r in ok])
        rel = np.array([r[f"bias_{est}"] / r["theta_true"] for r in ok])
        # resolution floor: smallest h/xi at which |bias| <= 1 deg, scanning up
        floor = None
        for lim in (1.0,):
            cand = [r["h_over_xi"] for r in ok if abs(r[f"bias_{est}"]) <= lim]
            floor = max(cand) if cand else None
        out["estimators"][est] = {
            "n_ok": len(ok),
            "n_fail": len(rows) - len(ok),
            "max_abs_bias_deg": float(np.max(np.abs(bias))) if bias.size else None,
            "median_abs_bias_deg": float(np.median(np.abs(bias))) if bias.size else None,
            "max_abs_rel_err": float(np.max(np.abs(rel))) if rel.size else None,
            "min_h_over_xi_with_abs_bias_le_1deg": floor,
            "fail_below_h_over_xi": (
                None
                if floor is None
                else float(
                    max(
                        [r["h_over_xi"] for r in rows if not math.isfinite(r[f"theta_{est}"])],
                        default=float("nan"),
                    )
                )
            ),
        }
    # does the proposed R4 criterion (h/xi >= 2.5) actually separate the cases
    # the decision metric gets wrong?
    circle = [r for r in rows if math.isfinite(r["bias_circle"])]
    bad = [r for r in circle if abs(r["bias_circle"]) > 1.0]
    out["r4_criterion_test"] = {
        "criterion": "R(1-cos theta)/xi >= 2.5 keeps |circle bias| <= 1 deg",
        "n_cases_above": sum(1 for r in circle if r["h_over_xi"] >= 2.5),
        "n_cases_below": sum(1 for r in circle if r["h_over_xi"] < 2.5),
        "n_bad_above": sum(1 for r in bad if r["h_over_xi"] >= 2.5),
        "n_bad_below": sum(1 for r in bad if r["h_over_xi"] < 2.5),
        "smallest_h_over_xi_of_bad_case": (
            float(min((r["h_over_xi"] for r in bad), default=float("nan")))
            if bad
            else None
        ),
        "largest_h_over_xi_of_bad_case": (
            float(max((r["h_over_xi"] for r in bad), default=float("nan"))) if bad else None
        ),
        "verdict": (
            "falsified" if any(r["h_over_xi"] < 2.5 for r in circle if abs(r["bias_circle"]) <= 1.0)
            else "not_falsified"
        ),
    }
    return out


# ---------------------------------------------------------------------------
# Part 2: re-adjudicate the saved solver fields
# ---------------------------------------------------------------------------
def cap_extents(phi: np.ndarray, y_wall: float = Y_WALL, level: float = 0.0) -> dict:
    """Base width ``b`` and apex height ``h`` of the ``phi = level`` contour.

    Uses the library contour extractor (``cg_contact_angle.interface_contour``:
    lowest crossing per column), so the extents are measured exactly the way the
    cap-inversion estimators measure them -- no second convention in the study.
    """
    try:
        c = interface_contour(phi, level=level, y_min=1)
    except Exception as exc:  # noqa: BLE001
        return {"b": float("nan"), "h": float("nan"), "n_cols": 0, "error": repr(exc)}
    x, y = c["x"], c["y"]
    if x.size < 2:
        return {"b": float("nan"), "h": float("nan"), "n_cols": int(x.size)}
    return {
        "b": float(x.max() - x.min()),
        "h": float(y.max() - y_wall),
        "x_left": float(x.min()),
        "x_right": float(x.max()),
        "n_cols": int(x.size),
    }


def true_cap_extents(radius: float, theta_deg: float) -> tuple[float, float]:
    th = math.radians(theta_deg)
    return 2.0 * radius * math.sin(th), radius * (1.0 - math.cos(th))


def adjudicate_solver_fields(solver_dir: Path, targets: dict) -> list[dict]:
    out = []
    for arm, theta_target in targets.items():
        p = solver_dir / f"_last_phi_{arm}.npy"
        if not p.is_file():
            out.append({"arm": arm, "theta_target": theta_target, "status": "field_missing"})
            continue
        phi = np.load(p)
        solid = solid_grid(phi.shape[1], phi.shape[0])
        meas = measure_all(phi, solid)
        ext = cap_extents(phi)
        rec: dict = {
            "arm": arm,
            "theta_target": theta_target,
            "field": p.name,
            "shape": list(phi.shape),
            "phi_min": float(phi.min()),
            "phi_max": float(phi.max()),
            "base_width_cells": ext["b"],
            "apex_height_cells": ext["h"],
            "status": meas["notes"],
        }
        for k, v in meas["theta"].items():
            rec[f"theta_{k}"] = v
            rec[f"err_pct_{k}"] = (
                100.0 * (v - theta_target) / theta_target if math.isfinite(v) else float("nan")
            )
        # radius implied by the settled shape, and the cap height a *true* cap of
        # the target angle with the same base width would have
        if math.isfinite(ext["b"]) and ext["h"] > 0:
            bb, hh = ext["b"], ext["h"]
            rec["radius_from_extents"] = (bb * bb + 4.0 * hh * hh) / (8.0 * hh) / 1.0
            th_true = math.radians(theta_target)
            h_expected = 0.5 * bb * (1.0 - math.cos(th_true)) / math.sin(th_true)
            rec["apex_height_expected_target"] = h_expected
            rec["apex_height_ratio"] = hh / h_expected if h_expected > 0 else float("nan")
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_calibration(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    colors = {"circle": "C0", "cap_area": "C1", "cap_wm": "C2", "normal": "C3"}
    labels = {
        "circle": "circle fit (R1-R4 decision metric)",
        "cap_area": "cap inversion (cg_contact_angle)",
        "cap_wm": "cap inversion (wetting.metrics)",
        "normal": "near-wall normal read-back",
    }
    for est in colors:
        sub = [r for r in rows if math.isfinite(r[f"bias_{est}"])]
        axes[0].scatter(
            [r["h_over_xi"] for r in sub],
            [r[f"bias_{est}"] for r in sub],
            s=18,
            alpha=0.75,
            color=colors[est],
            label=labels[est],
        )
    axes[0].axhline(0.0, color="k", lw=0.8)
    axes[0].axhline(1.0, color="gray", lw=0.6, ls=":")
    axes[0].axhline(-1.0, color="gray", lw=0.6, ls=":")
    axes[0].axvline(2.5, color="firebrick", lw=0.9, ls="--", label="R4 criterion h/$\\xi$=2.5")
    axes[0].set_xlabel("cap height / interface thickness   $h/\\xi$")
    axes[0].set_ylabel("bias  $\\theta_{est}-\\theta_{true}$  [deg]")
    axes[0].set_xscale("log")
    axes[0].set_ylim(-30, 30)
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    axes[0].set_title("estimator bias on exact diffused caps")

    # recovery map: |bias| of the circle fit over (theta, h/xi)
    sub = [r for r in rows if r["xi"] == 2.0 and math.isfinite(r["bias_circle"])]
    sc = axes[1].scatter(
        [r["theta_true"] for r in sub],
        [r["h_over_xi"] for r in sub],
        c=[abs(r["bias_circle"]) for r in sub],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=70,
        edgecolor="k",
        linewidth=0.4,
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("prescribed contact angle  $\\theta$  [deg]")
    axes[1].set_ylabel("$h/\\xi$   (xi=2)")
    axes[1].set_title("circle-fit |bias| [deg]; blank = estimator failed")
    fig.colorbar(sc, ax=axes[1], label="|bias| [deg]")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_adjudication(fields: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [a for a in fields if "phi_max" in fields[a]]
    n = max(len(arms), 1)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6), squeeze=False)
    for k, arm in enumerate(arms):
        phi = np.load(fields[arm]["path"])
        ax = axes[0][k]
        ax.imshow(phi, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
        th = fields[arm]["theta_target"]
        # exact cap of the target angle through the *measured* base width
        ext = cap_extents(phi)
        if math.isfinite(ext["b"]):
            bb = ext["b"]
            th_r = math.radians(th)
            R = 0.5 * bb / math.sin(th_r)
            xc = 0.5 * (ext["x_left"] + ext["x_right"])
            yc = Y_WALL - R * math.cos(th_r)
            xs = np.linspace(xc - R, xc + R, 400)
            arc = yc + np.sqrt(np.clip(R * R - (xs - xc) ** 2, 0.0, None))
            keep = arc >= Y_WALL
            ax.plot(xs[keep], arc[keep], "k--", lw=1.0, label=f"exact {th:.0f}$^\\circ$ cap")
        ax.set_title(
            f"{arm}\n$\\theta_{{circle}}$={fields[arm].get('theta_circle', float('nan')):.1f}"
            f"$^\\circ$  (target {th:.0f}$^\\circ$)",
            fontsize=8,
        )
        ax.set_xlim(0, phi.shape[1])
        ax.set_ylim(0, min(phi.shape[0], 2.2 * (ext["h"] + 4)))
        ax.legend(fontsize=6, loc="upper right")
        ax.set_xlabel("x [lattice]")
        if k == 0:
            ax.set_ylabel("y [lattice]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parents[1] / "artifacts" / "m22257_akai_r5"))
    ap.add_argument("--solver-dir", default=str(HERE))
    ap.add_argument("--src", default=None)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = calibration_sweep()
    summary = summarise(rows)
    (out / "estimator_calibration.json").write_text(
        json.dumps({"cases": rows, "summary": summary}, indent=1)
    )

    solver_dir = Path(args.solver_dir)
    targets = {"gk30": 30.0, "gk60": 60.0, "gk90": 90.0, "ctl60": 90.0}
    adj = adjudicate_solver_fields(solver_dir, targets)
    (out / "solver_field_adjudication.json").write_text(json.dumps(adj, indent=1))

    if not args.no_figures:
        fig_calibration(rows, out / "fig_estimator_calibration.png")
        fieldmap = {}
        for rec in adj:
            if "phi_max" in rec:
                fieldmap[rec["arm"]] = dict(rec)
                fieldmap[rec["arm"]]["path"] = str(solver_dir / f"_last_phi_{rec['arm']}.npy")
        if fieldmap:
            fig_adjudication(fieldmap, out / "fig_shape_adjudication.png")

    agg = {
        "summary": summary,
        "solver_fields": adj,
        "artifacts": {
            "cases": str(out / "estimator_calibration.json"),
            "adjudication": str(out / "solver_field_adjudication.json"),
            "fig_calibration": str(out / "fig_estimator_calibration.png"),
            "fig_adjudication": str(out / "fig_shape_adjudication.png"),
        },
    }
    (out / "summary.json").write_text(json.dumps(agg, indent=1))

    print("== estimator calibration (exact diffused caps) ==")
    for est, s in summary["estimators"].items():
        print(
            f"  {est:9s} ok={s['n_ok']:3d} fail={s['n_fail']:2d} "
            f"max|bias|={s['max_abs_bias_deg']:.3f} deg  "
            f"median|bias|={s['median_abs_bias_deg']:.3f}  "
            f"max|rel|={100 * s['max_abs_rel_err']:.1f}%"
        )
    print("== R4 criterion", json.dumps(summary["r4_criterion_test"]))
    print("== solver fields ==")
    for rec in adj:
        print(
            f"  {rec['arm']:6s} target={rec['theta_target']:5.1f} "
            f"circle={rec.get('theta_circle', float('nan')):7.2f} "
            f"cap_area={rec.get('theta_cap_area', float('nan')):7.2f} "
            f"cap_wm={rec.get('theta_cap_wm', float('nan')):7.2f} "
            f"h={rec.get('apex_height_cells', float('nan')):6.2f} "
            f"b={rec.get('base_width_cells', float('nan')):6.2f}"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())