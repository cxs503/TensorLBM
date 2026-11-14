#!/usr/bin/env python3
"""XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS — pool case runner.

Anchor
------
``lssfau/walberla``'s *two independent contact-angle implementations* discipline:
the same continuum wall condition (Young's law) is coded **twice, sharing no
code**, and the two implementations are cross-validated.  Here the two (three,
in fact) implementations live in
:mod:`tensorlbm.wetting.two_impl_contact_angle`:

``dirichlet``        wall *value* ``psi_w`` with ``cos(theta) = (3 psi_w - psi_w^3)/2``
``neumann_flux``     wall *flux* ``kappa d_n psi = (3/4) sigma cos(theta)(1-psi^2)``
``geometric_ghost``  purely *geometric* ghost ``psi_g = psi_0 + cot(theta)|d_t psi|``

They are relaxations of the same diffuse-interface free energy with three
different discretisations of the wall term; the reference is the continuum
Young angle ``theta_ref`` the caller prescribes.  A fourth, *independent*
implementation runs in the production solver
(:func:`tensorlbm.wetting.ladder.contact_two_impl` -> ``CAC2D`` D2Q9 CAC/HCH +
bounce-back) and is enabled with ``--solver``.

What this runner does
---------------------
For every (arm, prescribed angle) it relaxes the drop to static equilibrium,
measures the angle with three **implementation-blind** estimators and writes
``summary.json`` / ``angles.csv`` / ``criteria.json`` plus figures:

``theta_cap_deg``     conserved-content Laplace-cap area inversion (module)
``theta_circle_deg``  algebraic circle fit of the mid-level contour (module)
``theta_normal_deg``  contact-line interface-normal direction from field
                      gradients (this file; shares no code with the module's
                      estimators and no code with the arms)

Predeclared gates are read from ``refs.json`` (``G1``..``G7``); nothing is
tuned after the sweep is inspected.

Usage
-----
    PYTHONPATH=src python3 benchmarks/case_pool/two_impl_contact_angle/run_case.py \
        --out runs/two_impl_contact_angle
    ... --quick            # 2 angles, short relaxations (smoke, not acceptance)
    ... --solver           # add the production-solver arm (slow: LBE solve)
    python benchmarks/case_pool/run_pool.py --case two_impl_contact_angle
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensorlbm.wetting.two_impl_contact_angle import (  # noqa: E402
    ARMS,
    TwoImplConfig,
    run_arm,
)

HERE = Path(__file__).resolve().parent
REFS = json.loads((HERE / "refs.json").read_text())

DEFAULT_ANGLES = tuple(float(a) for a in REFS["reference"]["theta_ref_deg"])
#: wall plane of the psi-descent harness (row 0 touches y = -1/2)
WALL_Y = -0.5


# ---------------------------------------------------------------------------
# third, independent estimator (benchmark side, implementation-blind)
# ---------------------------------------------------------------------------
def _gas_reference(psi: torch.Tensor) -> float:
    """Far-field gas value from the two top corners (never inside the drop)."""
    return 0.5 * (float(psi[-4:, :4].mean()) + float(psi[-4:, -4:].mean()))


def _level_crossings(row: torch.Tensor, level: float) -> list[float]:
    """Sub-cell x positions where ``row`` crosses ``level`` (sorted, left to right)."""
    dev = row - level
    prod = dev[:-1] * dev[1:]
    out: list[float] = []
    for i in (prod < 0).flatten().nonzero().flatten().tolist():
        d0, d1 = float(dev[i]), float(dev[i + 1])
        if d1 == d0:
            continue
        out.append(i + d0 / (d0 - d1))
    return out


def theta_from_contact_normal(
    psi: torch.Tensor,
    interface_width: float = 3.0,
    rows=(0, 1, 2),
) -> dict:
    """Contact angle from the local interface trace at the contact line.

    The mid-level contour is located in the two/three rows closest to the wall
    (``rows``); on each side of the drop the *trace* ``x(y)`` of the interface is
    fitted by a straight line through those rows and the contact angle is the
    angle between that trace and the wall direction pointing into the liquid,
    ``theta = atan2(1, -sign(x - x_c) * dx/dy)`` (theta > 90 deg means the
    interface leans outward above the wall, i.e. a non-wetting drop).

    Only local near-wall data enter: no global content, no global least squares,
    so this estimator is independent of ``theta_cap_deg`` (area + apex) and
    ``theta_circle_deg`` (whole-contour circle fit) and of the wall condition
    the arm applied (the wall ghost is never read).
    """
    p = psi.detach().to(torch.float64)
    gas = _gas_reference(p)
    level = 0.5 * (gas + float(p.max()))
    ys = [r for r in rows if r < p.shape[0]]
    if len(ys) < 2:
        return {"theta_normal_deg": float("nan"), "n_normal_points": 0, "normal_points": []}
    xs = {r: _level_crossings(p[r], level) for r in ys}
    if any(len(xs[r]) < 2 for r in ys):
        return {"theta_normal_deg": float("nan"), "n_normal_points": 0, "normal_points": []}
    nx = p.shape[1]
    x_c = 0.5 * (xs[ys[0]][0] + xs[ys[0]][-1])  # drop axis from the wall-row chord
    thetas: list[float] = []
    pts: list[tuple[float, float, float]] = []
    for side in (0, -1):  # left / right crossing of each row
        xv = [xs[r][side] for r in ys]
        yv = [float(r) for r in ys]
        n = len(yv)
        my = sum(yv) / n
        mx = sum(xv) / n
        # slope *at the wall plane* by a quadratic (3-point) trace fit, so the
        # chord bias of a straight-line fit one row up is removed
        if n >= 3:
            (y0, y1, y2) = yv[:3]
            (x0, x1, x2) = xv[:3]
            d1 = (x1 - x0) / (y1 - y0)
            d2 = (x2 - x1) / (y2 - y1)
            c = (d2 - d1) / (y2 - y0)
            b = d1 - c * (y1 + y0)
            slope = b + 2.0 * c * WALL_Y
        else:
            sxx = sum((y - my) ** 2 for y in yv)
            slope = sum((y - my) * (x - mx) for x, y in zip(xv, yv)) / sxx if sxx else 0.0
        sgn = 1.0 if mx > x_c else -1.0
        th = math.degrees(math.atan2(1.0, -sgn * slope))
        thetas.append(th)
        pts.append((mx, my, th))
    if not thetas:
        return {"theta_normal_deg": float("nan"), "n_normal_points": 0, "normal_points": pts}
    return {
        "theta_normal_deg": float(statistics.median(thetas)),
        "n_normal_points": len(thetas),
        "normal_points": pts,
        "near_wall_rows": ys,
        "_nx_used": nx,
    }


# ---------------------------------------------------------------------------
# one row = (arm, prescribed angle)
# ---------------------------------------------------------------------------
def measure_row(psi: torch.Tensor, cfg: TwoImplConfig) -> dict:
    """Three implementation-blind estimators on one final field."""
    from tensorlbm.wetting.two_impl_contact_angle import measure_contact_angle

    base = measure_contact_angle(psi, cfg.interface_width, wall_y=-0.5)
    normal = theta_from_contact_normal(psi, cfg.interface_width)
    est = [base["theta_cap_deg"], base["theta_circle_deg"], normal["theta_normal_deg"]]
    finite = [e for e in est if math.isfinite(e)]
    return {
        **base,
        **normal,
        "theta_med_deg": float(statistics.median(finite)) if finite else float("nan"),
        "estimator_spread_deg": (max(finite) - min(finite)) if len(finite) > 1 else float("nan"),
        "n_estimators": len(finite),
    }


def sweep(
    angles=DEFAULT_ANGLES,
    arms=ARMS,
    cfg: TwoImplConfig | None = None,
    max_iter: int = 30000,
    radius: float = 16.0,
    quick: bool = False,
) -> list[dict]:
    """Relax every (arm, angle) and return one row per relaxation."""
    cfg = cfg or TwoImplConfig()
    rows: list[dict] = []
    for theta_ref in angles:
        theta = math.radians(theta_ref)
        for arm in arms:
            t0 = time.time()
            r = run_arm(
                arm, theta, cfg=cfg, radius=radius,
                max_iter=(6000 if quick else max_iter),
            )
            meas = measure_row(r["psi"], cfg)
            rows.append({
                "arm": arm,
                "theta_ref_deg": theta_ref,
                "theta_cap_deg": meas["theta_cap_deg"],
                "theta_circle_deg": meas["theta_circle_deg"],
                "theta_normal_deg": meas["theta_normal_deg"],
                "theta_med_deg": meas["theta_med_deg"],
                "estimator_spread_deg": meas["estimator_spread_deg"],
                "n_estimators": meas["n_estimators"],
                "err_pct": (meas["theta_med_deg"] - theta_ref) / theta_ref * 100.0,
                "measurable": bool(meas["measurable"]),
                "circle_rms": meas["circle_rms"],
                "apex_height": meas["apex_height"],
                "R_curv": meas["radius"],
                "n_contour_points": meas["n_contour_points"],
                "n_normal_points": meas["n_normal_points"],
                "gas_supersaturation": meas["gas_supersaturation"],
                "wall_angle_readback_deg": r["wall_angle_readback_deg"],
                "wall_value_mean": r["wall_value_mean"],
                "iterations": r["iterations"],
                "converged": bool(r["converged"]),
                "drift": r["drift"],
                "mass_error": r["mass_error"],
                "psi_min": r["psi_min"],
                "psi_max": r["psi_max"],
                "quick": quick,
                "wall_row_psi": [round(float(v), 6) for v in r["psi"][0].tolist()],
                "wall_time_s": time.time() - t0,
            })
    return rows


# ---------------------------------------------------------------------------
# predeclared gates
# ---------------------------------------------------------------------------
def evaluate_gates(rows: list[dict], solver_rows: list[dict] | None = None) -> dict:
    g = REFS["gates"]
    by_angle: dict[float, list[float]] = {}
    for r in rows:
        by_angle.setdefault(float(r["theta_ref_deg"]), []).append(float(r["theta_med_deg"]))

    g1 = all(r["measurable"] for r in rows)
    acc = [{"arm": r["arm"], "theta_ref_deg": r["theta_ref_deg"], "err_pct": r["err_pct"]}
           for r in rows]
    g2 = all(abs(a["err_pct"]) < g["G2_accuracy_vs_young_pct"] for a in acc)
    spread = {}
    for th, vals in by_angle.items():
        spread[th] = (max(vals) - min(vals)) / th * 100.0
    g3 = all(v < g["G3_cross_arm_spread_pct"] for v in spread.values())
    g4_rows = [{"arm": r["arm"], "theta_ref_deg": r["theta_ref_deg"],
                "spread_deg": r["estimator_spread_deg"],
                "estimators": [r["theta_cap_deg"], r["theta_circle_deg"], r["theta_normal_deg"]]}
               for r in rows]
    g4 = all(r["spread_deg"] < g["G4_estimator_spread_deg"] for r in g4_rows)
    g5 = all(r["converged"] and r["mass_error"] < 1e-6 and r["drift"] < 1e-9 for r in rows)

    # G6: the two flux/geometric implementations must collapse onto the same
    # Neumann ghost in their common theta=90 limit (field-level comparison).
    g6_val = None
    for th in by_angle:
        if abs(th - 90.0) < 1e-9:
            a = next((r for r in rows if abs(r["theta_ref_deg"] - th) < 1e-9 and r["arm"] == "neumann_flux"), None)
            b = next((r for r in rows if abs(r["theta_ref_deg"] - th) < 1e-9 and r["arm"] == "geometric_ghost"), None)
            if a and b:
                g6_val = max(abs(a["theta_cap_deg"] - b["theta_cap_deg"]),
                             abs(a["theta_circle_deg"] - b["theta_circle_deg"]),
                             abs(a["theta_normal_deg"] - b["theta_normal_deg"]),
                             abs(a["psi_max"] - b["psi_max"]), abs(a["psi_min"] - b["psi_min"]))
    g6 = g6_val is not None and g6_val < 1e-6

    out = {
        "G1_measurable": {"pass": g1, "value": sum(1 for r in rows if r["measurable"]),
                          "n_rows": len(rows)},
        "G2_accuracy_vs_young": {"pass": g2, "threshold_pct": g["G2_accuracy_vs_young_pct"],
                                 "max_abs_err_pct": max(abs(a["err_pct"]) for a in acc),
                                 "rows": acc},
        "G3_cross_arm_spread": {"pass": g3, "threshold_pct": g["G3_cross_arm_spread_pct"],
                                "by_angle_pct": spread,
                                "max_spread_pct": max(spread.values())},
        "G4_estimator_spread": {"pass": g4, "threshold_deg": g["G4_estimator_spread_deg"],
                                "max_spread_deg": max(r["spread_deg"] for r in g4_rows),
                                "violations": [r for r in g4_rows
                                               if r["spread_deg"] >= g["G4_estimator_spread_deg"]]},
        "G5_conservation": {"pass": g5, "max_mass_error": max(r["mass_error"] for r in rows),
                            "max_drift": max(r["drift"] for r in rows)},
        "G6_neutral_limit_degeneracy": {"pass": g6, "value": g6_val,
                                        "threshold": 1e-6,
                                        "definition": g["G6_neutral_limit_degeneracy"]},
    }
    if solver_rows:
        ok = []
        for s in solver_rows:
            if "err_pct" in s and math.isfinite(s["err_pct"]):
                ok.append(abs(s["err_pct"]) < g["G7_solver_cross_check_pct"])
        out["G7_solver_cross_check"] = {
            "pass": bool(ok) and all(ok),
            "threshold_pct": g["G7_solver_cross_check_pct"],
            "rows": [{"theta_ref_deg": s["theta_ref_deg"], "psi_wall": s["psi_wall_from_angle"],
                      "theta_measured_deg": s.get("theta_measured_deg"),
                      "err_pct": s.get("err_pct"), "steps": s["inputs"]["steps"]} for s in solver_rows],
        }
    out["all_accepted"] = all(v["pass"] for k, v in out.items() if k.startswith("G"))
    return out


# ---------------------------------------------------------------------------
# artefacts
# ---------------------------------------------------------------------------
def write_csv(rows: list[dict], path: Path) -> None:
    cols = ["arm", "theta_ref_deg", "theta_cap_deg", "theta_circle_deg", "theta_normal_deg",
            "theta_med_deg", "err_pct", "estimator_spread_deg", "measurable", "circle_rms",
            "apex_height", "R_curv", "gas_supersaturation", "wall_angle_readback_deg",
            "iterations", "converged", "mass_error", "wall_time_s"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k]) for k in cols})


def figures(rows: list[dict], outdir: Path, solver_rows: list[dict] | None = None,
            profiles: dict | None = None) -> list[Path]:
    """Three figures: ref-vs-got parity, cross-arm spread, final fields."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - figure generation is optional
        print(f"[figures] matplotlib unavailable ({exc}); skipped")
        return []

    outdir.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    marks = {"dirichlet": "o", "neumann_flux": "s", "geometric_ghost": "^"}
    for arm in sorted({r["arm"] for r in rows}):
        sel = [r for r in rows if r["arm"] == arm]
        ax.plot([r["theta_ref_deg"] for r in sel], [r["theta_med_deg"] for r in sel],
                marks.get(arm, "x") + "-", label=f"{arm} (psi-descent)")
    if solver_rows:
        sel = [s for s in solver_rows if "theta_measured_deg" in s]
        if sel:
            ax.plot([s["theta_ref_deg"] for s in sel], [s["theta_measured_deg"] for s in sel],
                    "d--", label="CAC2D production solver")
    lims = [50, 145]
    ax.plot(lims, lims, "k:", lw=1, label="1:1 (theta_got = theta_ref)")
    ax.fill_between(lims, [v * 0.85 for v in lims], [v * 1.15 for v in lims],
                    color="grey", alpha=0.15, label="+/-15% band")
    ax.set_xlabel("prescribed Young angle theta_ref [deg]")
    ax.set_ylabel("measured angle theta_got [deg] (median of 3 estimators)")
    ax.set_title("two independent contact-angle implementations vs Young reference")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    p = outdir / "fig1_ref_vs_got.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); made.append(p)

    angles = sorted({r["theta_ref_deg"] for r in rows})
    arms = sorted({r["arm"] for r in rows})
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    width = 0.8 / max(len(arms), 1)
    for k, arm in enumerate(arms):
        vals = [next(r["theta_med_deg"] for r in rows
                     if r["arm"] == arm and r["theta_ref_deg"] == a) for a in angles]
        ax.bar([i + (k - (len(arms) - 1) / 2) * width for i in range(len(angles))], vals,
               width=width, label=arm)
    ax.plot(range(len(angles)), angles, "k_", ms=22, mew=2, label="theta_ref")
    ax.set_xticks(range(len(angles)))
    ax.set_xticklabels([f"{a:g} deg" for a in angles])
    ax.set_ylabel("measured angle [deg]")
    ax.set_title("cross-arm agreement at each prescribed angle")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    p = outdir / "fig2_cross_arm.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); made.append(p)

    if profiles:
        fig, ax = plt.subplots(figsize=(5.4, 4.2))
        for label, psi in profiles.items():
            ax.plot(range(len(psi)), psi, "-", lw=1.2, label=f"{label}: psi at wall row")
        ax.set_xlabel("x [lattice]")
        ax.set_ylabel("psi (order parameter, wall-adjacent row)")
        ax.set_title("wall-row order parameter: the implementations differ at the wall")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        p = outdir / "fig3_wall_row.png"
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); made.append(p)
    return made


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(REPO / "runs" / "two_impl_contact_angle"),
                    help="artefact directory (runs/...; never the source tree)")
    ap.add_argument("--angles", default="", help="comma list of prescribed angles [deg]")
    ap.add_argument("--arms", default="", help="comma list of arms (default: all three)")
    ap.add_argument("--max-iter", type=int, default=30000)
    ap.add_argument("--radius", type=float, default=16.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="smoke: 2 angles, short relaxation")
    ap.add_argument("--solver", action="store_true",
                    help="also run the production CAC2D arm (slow)")
    ap.add_argument("--solver-steps", type=int, default=2000)
    ap.add_argument("--solver-shape", default="64x64")
    ap.add_argument("--smoke", action="store_true", help="alias of --quick (pool driver)")
    args = ap.parse_args(argv)

    quick = args.quick or args.smoke
    torch.set_num_threads(max(1, args.threads))
    angles = ([float(a) for a in args.angles.split(",") if a.strip()] if args.angles
              else ((90.0, 110.0) if quick else DEFAULT_ANGLES))
    arms = tuple(a for a in args.arms.split(",") if a.strip()) if args.arms else ARMS

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    profiles: dict[str, list] = {}
    rows = []
    for r in sweep(angles, arms, max_iter=args.max_iter, radius=args.radius, quick=quick):
        rows.append(r)
        print(f"[sweep] {r['arm']:16s} theta_ref={r['theta_ref_deg']:6.1f}  "
              f"got={r['theta_med_deg']:7.2f}  err={r['err_pct']:+7.2f}%  "
              f"spread={r['estimator_spread_deg']:5.2f} deg", flush=True)

    # wall-row profiles of the arms at the middle angle (fig3) -- taken from the
    # relaxations already computed above, never re-run
    mid = 90.0 if any(abs(a - 90.0) < 1e-9 for a in angles) else angles[0]
    cfg = TwoImplConfig()
    for r in rows:
        if abs(r["theta_ref_deg"] - mid) < 1e-9:
            profiles[f"{r['arm']} @{mid:g}"] = r["wall_row_psi"]

    solver_rows: list[dict] = []
    if args.solver:
        from tensorlbm.wetting.ladder import contact_two_impl

        ny, nx = (int(v) for v in args.solver_shape.lower().split("x"))
        for th in angles:
            s = contact_two_impl(th, "dirichlet", shape=(ny, nx), radius=min(12.0, 0.16 * ny),
                                 steps=args.solver_steps)
            solver_rows.append(s)
            print(f"[solver] theta_ref={th:6.1f}  psi_w={s['psi_wall_from_angle']:+.5f}  "
                  f"got={s.get('theta_measured_deg', float('nan')):7.2f}  "
                  f"err={s.get('err_pct', float('nan')):+7.2f}%", flush=True)

    gates = evaluate_gates(rows, solver_rows or None)
    figs = figures(rows, outdir / "figures", solver_rows or None, profiles)

    payload = {
        "anchor": REFS["anchor"],
        "case": "two_impl_contact_angle",
        "generated_by": "benchmarks/case_pool/two_impl_contact_angle/run_case.py",
        "quick": quick,
        "config": {"angles_deg": list(angles), "arms": list(arms), "max_iter": args.max_iter,
                   "radius": args.radius, "interface_width": cfg.interface_width,
                   "surface_tension": cfg.surface_tension, "ny": cfg.ny, "nx": cfg.nx,
                   "dtype": str(cfg.dtype), "threads": args.threads,
                   "solver": {"enabled": bool(args.solver), "shape": args.solver_shape,
                              "steps": args.solver_steps}},
        "reference": REFS["reference"],
        "rows": rows,
        "solver_arm": solver_rows,
        "gates": gates,
        "figures": [str(p.relative_to(outdir)) for p in figs],
        "wall_time_s": time.time() - t0,
    }
    (outdir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    (outdir / "criteria.json").write_text(json.dumps(gates, indent=2, sort_keys=True))
    write_csv(rows, outdir / "angles.csv")
    for r in rows:
        r.pop("wall_row_psi", None)

    print(json.dumps({k: v for k, v in gates.items() if k != "G2_accuracy_vs_young"}, indent=2)[:2000])
    print(f"[done] {outdir}  gates_all_accepted={gates['all_accepted']}  "
          f"wall={payload['wall_time_s']:.1f}s")
    return 0 if gates["all_accepted"] or quick else 1


if __name__ == "__main__":
    raise SystemExit(main())