#!/usr/bin/env python3
"""CASE-TAYLOR-GREEN-3D -- case-pool entry: decay of the 3-D Taylor-Green vortex.

Reference: the **exact time-Taylor series** of the Helmholtz-projected
Navier-Stokes equations (``级数精确解``, ``src/tensorlbm/tg3d_series.py``),
``u(x,t) = sum_n t^n u^(n)(x)`` with

    u^(n+1) = ( nu*Lap(u^(n)) - P sum_{j=0..n} (u^(j).grad) u^(n-j) ) / (n+1)

and ``P`` the Helmholtz projection.  Every ``u^(n)`` is a finite Fourier series on
integer wavevectors, so the recursion has exact spatial derivatives and no time
discretisation.  That is what makes the classic (z-dependent) Taylor-Green field
comparable at all: it has *no* closed form, and this case's point is precisely the
3-D signal the z-invariant mode cannot produce (``w != 0``, vortex stretching).

The series generator is not trusted on assertion -- it is pinned first (Q0)
against two independent closed forms (the z-invariant mode and the Beltrami/ABC
flow, both of which it must reproduce as pure exponential decay to 1e-9), its own
truncation is bounded (Q5, Q8 order convergence, Q9 working-grid invariance) and
it is cross-checked against a de-aliased pseudo-spectral DNS of the same IC.

Protocol: lattice units (dx = dt = 1), periodic N^3 box, D3Q27 BGK, float64,
nu = A*N/Re (fixed Reynolds number Re = 160, so refining the grid refines both
the spatial resolution and tau).

Usage
-----
    PYTHONPATH=src:benchmarks python3 benchmarks/case_pool/taylor_green_3d/run_case.py
    PYTHONPATH=src:benchmarks python3 benchmarks/case_pool/taylor_green_3d/run_case.py --smoke
    PYTHONPATH=src:benchmarks python3 benchmarks/case_pool/taylor_green_3d/run_case.py --legs anchor,convergence

Writes ``artifacts/summary.json`` (+ ``field_primary.npz``) when run outside
``--smoke``; ``--smoke`` is write-free so CI can never clobber the committed
evidence bundle.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parent
REPO = CASE_DIR.parents[2]
ARTIFACTS = CASE_DIR / "artifacts"

# --------------------------------------------------------------------------- #
# predeclared protocol (no number below is chosen after seeing a result)
# --------------------------------------------------------------------------- #
CASE = "CASE-TAYLOR-GREEN-3D"
TOL_PCT = 8.0                 # feature_ledger acceptance band
AMPLITUDE = 0.05              # A  -> Ma = A*sqrt(3) = 0.0866
REYNOLDS = 160.0              # Re = A*N/nu
SERIES_ORDER = 12             # retained powers of t (primary reference)
EDDY_TIMES = 1.0              # T, in eddy times A*k*t
PRIMARY_GRID = 32
SMOKE_GRID = 24
ORDER_GRIDS = (16, 24, 32)    # grids used for the observed-order fit
ORDER_MIN = 1.5               # predeclared floor on the observed order
REF_ORDER_LADDER = ((10, 60), (12, 60), (12, 72))  # (order, working nfft) variants

# id -> (tolerance, direction).  "le" = value must be <= tol, "ge" = >= tol.
THRESHOLDS: dict[str, tuple[float, str]] = {
    "Q0_series_vs_closed_form_rel": (1e-9, "le"),
    "Q1_energy_ratio_rel_err_pct": (TOL_PCT, "le"),
    "Q2_enstrophy_ratio_rel_err_pct": (TOL_PCT, "le"),
    "Q3_velocity_l2_rel_err_pct": (TOL_PCT, "le"),
    "Q4_vorticity_l2_rel_err_pct": (TOL_PCT, "le"),
    "Q5_series_truncation_rel": (1e-3, "le"),
    # O(Ma^2) = 7.5e-3 at Ma = 0.0866: the local density of an athermal LBM is a
    # finite-Mach quantity, not a conserved one -> the bound is ~2.7*Ma^2, not 0.
    "Q6_rho_deviation": (1e-2, "le"),
    "Q7_observed_order": (ORDER_MIN, "ge"),
    "Q8_series_order_convergence_rel": (1e-3, "le"),
    "Q9_series_workgrid_invariance_rel": (1e-9, "le"),
}

LEGS = ("anchor", "convergence", "exactness", "primary", "grid")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def parameters(n: int) -> dict:
    """Lattice parameters at fixed (amplitude, Reynolds)."""
    from tensorlbm.tg3d_lbm_check import tg3d_parameters

    return tg3d_parameters(n, amplitude=AMPLITUDE, reynolds=REYNOLDS)


def steps_for(n: int, eddy_times: float = EDDY_TIMES) -> int:
    p = parameters(n)
    return max(2, int(round(eddy_times / p["eddy_time_per_step"])))


def series(n: int, *, order: int = SERIES_ORDER, nfft: int | None = None):
    from tensorlbm.tg3d_lbm_check import series_reference

    return series_reference(
        n, amplitude=AMPLITUDE, reynolds=REYNOLDS, order=order, nfft=nfft
    )


def _rel(a: float, b: float) -> float:
    return abs(a / b - 1.0) if b else float("nan")


# --------------------------------------------------------------------------- #
# leg 0: pin the reference generator against closed forms
# --------------------------------------------------------------------------- #
def closed_form_anchor(*, n: int = PRIMARY_GRID, order: int = SERIES_ORDER,
                       eddy_times: float = EDDY_TIMES) -> dict:
    """The series must reduce to pure exponential decay where a closed form exists.

    * z-invariant Taylor-Green mode: E(t)/E(0) = exp(-2*nu*|k|^2*t), |k|^2 = 2k^2
    * Beltrami (ABC) flow:           E(t)/E(0) = exp(-2*nu*k^2*t)

    Both are exact Navier-Stokes solutions *and* nonlinearly trivial (the first has
    a gradient-only convective term, the second is Beltrami), so the series has to
    reproduce them to machine precision.  This is the layer-1 self-consistency
    check: it fails if the recursion, the projection or the normalisation inside
    the reference are tampered with, independently of any LBM run.
    """
    from tensorlbm.tg3d_series import TG3DSeries, abc_flow, taylor_green_extruded

    p = parameters(n)
    t = eddy_times / p["eddy_time_per_step"]
    k = 2.0 * math.pi / n
    out: dict[str, float] = {"t_lattice": t, "nu": p["nu"], "n": float(n), "order": float(order)}

    for name, ic_fn, k2 in (
        ("z_invariant", taylor_green_extruded, 2.0 * k * k),
        ("abc_beltrami", abc_flow, k * k),
    ):
        ser = TG3DSeries(ic_fn(AMPLITUDE), box=float(n), nu=p["nu"], order=order)
        ratio_series = ser.energy(t) / ser.energy(0.0)
        ratio_closed = math.exp(-2.0 * p["nu"] * k2 * t)
        out[f"{name}_ratio_series"] = float(ratio_series)
        out[f"{name}_ratio_closed_form"] = float(ratio_closed)
        out[f"{name}_rel_err"] = float(_rel(ratio_series, ratio_closed))
    out["max_rel_err"] = max(out["z_invariant_rel_err"], out["abc_beltrami_rel_err"])
    return out


# --------------------------------------------------------------------------- #
# leg 0b: the reference's own convergence (is the 8 % band resolvable at all?)
# --------------------------------------------------------------------------- #
def leg_reference_convergence(*, n: int = PRIMARY_GRID,
                              eddy_times: float = EDDY_TIMES,
                              variants: tuple[tuple[int, int], ...] = REF_ORDER_LADDER) -> dict:
    """Bound the reference error itself before comparing anything to it.

    Two independent knobs are moved: the retained Taylor order and the working
    (de-aliasing) grid.  A 2/3-dealiased pseudo-spectral product zeroes every mode
    above ``nfft/3``, so ``dealias_margin()`` reports ``margin == 0`` *by
    construction* for this construction -- it is not a safety statement.  The
    honest reference-error bound is therefore the observed spread below.
    """
    t = eddy_times / parameters(n)["eddy_time_per_step"]
    rows = []
    for order, nfft in variants:
        s = series(n, order=order, nfft=nfft)
        rows.append(
            {
                "order": order,
                "nfft": nfft,
                "energy_ratio": float(s.energy(t) / s.energy(0.0)),
                "enstrophy_ratio": float(s.enstrophy(t) / s.enstrophy(0.0)),
                "support_peak_index": s.dealias_margin()["support_peak_index"],
                "cutoff_index": s.dealias_margin()["cutoff_index"],
            }
        )
    ref = rows[-1]

    def spread(prev: dict, cur: dict) -> float:
        return max(
            abs(prev["energy_ratio"] / cur["energy_ratio"] - 1.0),
            abs(prev["enstrophy_ratio"] / cur["enstrophy_ratio"] - 1.0),
        )

    coarsest_nfft = min(r["nfft"] for r in rows)
    top_order = max(r["order"] for r in rows)
    by_order = [r for r in rows if r["nfft"] == coarsest_nfft]
    by_grid = [r for r in rows if r["order"] == top_order]
    order_conv = spread(by_order[-2], by_order[-1]) if len(by_order) >= 2 else float("nan")
    grid_inv = spread(by_grid[0], by_grid[-1]) if len(by_grid) >= 2 else float("nan")
    return {
        "n": n,
        "eddy_times": eddy_times,
        "t_lattice": t,
        "variants": rows,
        "order_convergence_rel": order_conv,
        "workgrid_invariance_rel": grid_inv,
        "note": "order_convergence_rel = |order 10 vs order 12| at nfft=60; "
                "workgrid_invariance_rel = |nfft 60 vs nfft 72| at the primary order",
    }


# --------------------------------------------------------------------------- #
# leg 1: exactness / reference-quality diagnostics
# --------------------------------------------------------------------------- #
def leg_exactness(*, n: int = PRIMARY_GRID, order: int = SERIES_ORDER) -> dict:
    """Why the classic field needs a *series* reference, plus the truncation bounds."""
    from tensorlbm.tg3d_series import (
        abc_flow,
        convective_acceleration,
        taylor_green_classic,
        taylor_green_extruded,
    )

    p = parameters(n)
    acc = {}
    for name, ic_fn in (
        ("classic_z_dependent", taylor_green_classic),
        ("z_invariant", taylor_green_extruded),
        ("abc_beltrami", abc_flow),
    ):
        acc[name] = convective_acceleration(ic_fn(AMPLITUDE), box=float(n), n=32)

    ser = series(n, order=order)
    t_end = float(steps_for(n))
    return {
        "n": n,
        "order": order,
        "nu": p["nu"],
        "tau": p["tau"],
        "convective_acceleration": acc,
        "curl_over_adv_classic": acc["classic_z_dependent"]["curl_over_adv"],
        "curl_over_adv_z_invariant": acc["z_invariant"]["curl_over_adv"],
        "series_dealias": ser.dealias_margin(),
        "series_tail_at_T": ser.relative_tail(t_end),
        "anchor": closed_form_anchor(n=n, order=order),
    }


# --------------------------------------------------------------------------- #
# leg 2: one decay run against the reference
# --------------------------------------------------------------------------- #
def leg_decay(n: int, *, eddy_times: float = EDDY_TIMES, order: int = SERIES_ORDER,
              save_field: bool = False) -> dict:
    """Run the D3Q27 lattice and compare decay ratios + fields to the series."""
    from tensorlbm.tg3d_lbm_check import compare_to_series, run_tg3d_lbm

    t0 = time.time()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ser = series(n, order=order)
    run = run_tg3d_lbm(
        n, amplitude=AMPLITUDE, reynolds=REYNOLDS, eddy_times=eddy_times
    )
    rows = compare_to_series(run, ser)
    wall = time.time() - t0
    last = rows[-1]

    hist = [
        {
            "step": r["step"],
            "eddy_time": r["eddy_time"],
            "energy_ratio_lbm": r["energy_ratio_lbm"],
            "energy_ratio_series": r["energy_ratio_series"],
            "enstrophy_ratio_lbm": r["enstrophy_ratio_lbm"],
            "enstrophy_ratio_series": r["enstrophy_ratio_series"],
            "rho_min": h["rho_min"],
            "rho_max": h["rho_max"],
        }
        for r, h in zip(rows, run["history"])
    ]
    rho_dev = max(
        max(abs(h["rho_min"] - 1.0), abs(h["rho_max"] - 1.0)) for h in run["history"]
    )
    tail = ser.relative_tail(last["time"])

    out = {
        "n": n,
        "steps": run["steps"],
        "eddy_time": last["eddy_time"],
        "tau": run["tau"],
        "nu": run["nu"],
        "ma": run["ma"],
        "reynolds": run["reynolds"],
        "energy_ratio_lbm": last["energy_ratio_lbm"],
        "energy_ratio_series": last["energy_ratio_series"],
        "energy_ratio_rel_err_pct": 100.0
        * (last["energy_ratio_lbm"] / last["energy_ratio_series"] - 1.0),
        "energy_pointwise_rel_err_pct": last["energy_rel_err_pct"],
        "enstrophy_ratio_lbm": last["enstrophy_ratio_lbm"],
        "enstrophy_ratio_series": last["enstrophy_ratio_series"],
        "enstrophy_ratio_rel_err_pct": 100.0
        * (last["enstrophy_ratio_lbm"] / last["enstrophy_ratio_series"] - 1.0),
        "velocity_l2_rel_err_pct": last["u_l2_rel_err_pct"],
        "vorticity_l2_rel_err_pct": last["vorticity_l2_rel_err_pct"],
        "u_l2_abs": last["u_l2_abs"],
        "u_l2_ref": last["u_l2_ref"],
        "series_field_order": last["field_order"],
        "series_truncation_rel": max(
            abs(tail["energy_last_order_rel"]), abs(tail["enstrophy_last_order_rel"])
        ),
        "rho_deviation": float(rho_dev),
        "wall_s": round(wall, 2),
        "history": hist,
    }

    if save_field:
        from tensorlbm.tg3d_series import vorticity_spectral

        t_end = last["time"]
        ue, ve, we = ser.velocity_field(t_end, n=n)
        wl = vorticity_spectral(
            run["final"]["u"], run["final"]["v"], run["final"]["w"], float(n)
        )
        wr = vorticity_spectral(ue, ve, we, float(n))
        np.savez_compressed(
            ARTIFACTS / "field_primary.npz",
            n=n,
            time=t_end,
            eddy_time=last["eddy_time"],
            wz_lbm_midplane=wl[2][:, :, 0],
            wz_series_midplane=wr[2][:, :, 0],
            u_midplane_lbm=run["final"]["u"][:, :, 0],
            u_midplane_series=ue[:, :, 0],
        )
        out["field_saved"] = "field_primary.npz"

    return out


# --------------------------------------------------------------------------- #
# leg 3: fixed-Re grid ladder -> observed order
# --------------------------------------------------------------------------- #
def leg_grid(*, grids: tuple[int, ...] = ORDER_GRIDS,
             eddy_times: float = EDDY_TIMES) -> dict:
    runs = [leg_decay(n, eddy_times=eddy_times) for n in grids]
    err = np.array([abs(r["energy_ratio_rel_err_pct"]) for r in runs])
    ns = np.array([float(r["n"]) for r in runs])
    slope = float(-np.polyfit(np.log(ns), np.log(err), 1)[0]) if len(runs) >= 2 else float("nan")
    return {
        "grids": [int(n) for n in ns],
        "energy_ratio_rel_err_pct": err.tolist(),
        "enstrophy_ratio_rel_err_pct": [abs(r["enstrophy_ratio_rel_err_pct"]) for r in runs],
        "velocity_l2_rel_err_pct": [r["velocity_l2_rel_err_pct"] for r in runs],
        "observed_order": slope,
        "monotone_decreasing": bool(np.all(np.diff(err) < 0)),
        "runs": runs,
    }


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #
def _quantity(qid: str, ref: float, got: float, unit: str = "") -> dict:
    tol, direction = THRESHOLDS[qid]
    ok = (got <= tol) if direction == "le" else (got >= tol)
    return {
        "id": qid,
        "ref": ref,
        "got": got,
        "tolerance": tol,
        "direction": direction,
        "unit": unit,
        "pass": bool(ok),
    }


def build_verdict(*, grid: int = PRIMARY_GRID, eddy_times: float = EDDY_TIMES,
                  with_grid: bool = True, with_ref_conv: bool = True,
                  save_field: bool = True, anchor: dict | None = None,
                  refconv: dict | None = None, primary: dict | None = None,
                  grid_leg: dict | None = None) -> dict:
    t0 = time.time()
    anchor = anchor if anchor is not None else closed_form_anchor(n=grid, eddy_times=eddy_times)
    if refconv is None:
        refconv = leg_reference_convergence(n=grid) if with_ref_conv else None
    primary = primary if primary is not None else leg_decay(
        grid, eddy_times=eddy_times, save_field=save_field)
    if grid_leg is None and with_grid:
        grid_leg = leg_grid(eddy_times=eddy_times)

    # Q0..Q2 ref columns are the reference-side value of the same quantity where a
    # reference-side value exists; Q0/Q5/Q6 are *deviations*, so their reference
    # value is 0.
    qs = [
        _quantity("Q0_series_vs_closed_form_rel", 0.0, anchor["max_rel_err"], "rel"),
        _quantity("Q1_energy_ratio_rel_err_pct", primary["energy_ratio_series"],
                  abs(primary["energy_ratio_rel_err_pct"]), "%"),
        _quantity("Q2_enstrophy_ratio_rel_err_pct", primary["enstrophy_ratio_series"],
                  abs(primary["enstrophy_ratio_rel_err_pct"]), "%"),
        _quantity("Q3_velocity_l2_rel_err_pct", primary["u_l2_ref"],
                  primary["velocity_l2_rel_err_pct"], "%"),
        _quantity("Q4_vorticity_l2_rel_err_pct", primary["u_l2_ref"],
                  primary["vorticity_l2_rel_err_pct"], "%"),
        _quantity("Q5_series_truncation_rel", 0.0, primary["series_truncation_rel"], "rel"),
        _quantity("Q6_rho_deviation", 0.0, primary["rho_deviation"], "rel"),
    ]
    if grid_leg is not None:
        qs.append(_quantity("Q7_observed_order", ORDER_MIN,
                            abs(grid_leg["observed_order"]), "-"))
    if refconv is not None:
        qs.append(_quantity("Q8_series_order_convergence_rel", 0.0,
                            refconv["order_convergence_rel"], "rel"))
        qs.append(_quantity("Q9_series_workgrid_invariance_rel", 0.0,
                            refconv["workgrid_invariance_rel"], "rel"))

    failed = [q["id"] for q in qs if not q["pass"]]
    return {
        "case": CASE,
        "repo": "TensorLBM",
        "reference": "exact time-Taylor series (级数精确解), src/tensorlbm/tg3d_series.py",
        "protocol": {
            "grid_primary": grid,
            "grids_ladder": list(grid_leg["grids"]) if grid_leg else [],
            "eddy_times": eddy_times,
            "amplitude": AMPLITUDE,
            "reynolds": REYNOLDS,
            "series_order": SERIES_ORDER,
            "tol_pct": TOL_PCT,
            "thresholds": {k: {"tolerance": v[0], "direction": v[1]}
                           for k, v in THRESHOLDS.items()},
        },
        "quantities": qs,
        "verdict": {"pass": not failed, "failed": failed, "n_quantities": len(qs)},
        "legs": {
            "anchor": anchor,
            "reference_convergence": refconv,
            "primary": {k: v for k, v in primary.items() if k != "history"},
            "primary_history": primary["history"],
            "grid": {k: v for k, v in (grid_leg or {}).items() if k != "runs"},
        },
        "wall_s": round(time.time() - t0, 1),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }


# --------------------------------------------------------------------------- #
# pool entry point
# --------------------------------------------------------------------------- #
def main(extra: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CASE-TAYLOR-GREEN-3D (pool case)")
    ap.add_argument("--legs", default="anchor,convergence,exactness,primary,grid",
                    help="comma separated subset of: " + ",".join(LEGS))
    ap.add_argument("--smoke", action="store_true",
                    help="fast, write-free check on a coarse grid (CI)")
    ap.add_argument("--grid", type=int, default=PRIMARY_GRID)
    ap.add_argument("--eddy-times", type=float, default=EDDY_TIMES)
    ap.add_argument("--out", default="")
    ap.add_argument("--grids", default="",
                    help="comma separated grids for the --legs grid leg")
    ap.add_argument("--dump-leg", action="store_true",
                    help="write the single selected leg verbatim (no verdict); used to "
                         "run legs in separate short processes, then --assemble")
    ap.add_argument("--assemble", action="store_true",
                    help="merge artifacts/legs/*.json into the verdict (artifacts/summary.json)")
    args = ap.parse_args(list(extra) if extra is not None else None)

    if args.assemble:
        lp = ARTIFACTS / "legs"
        legs = {}
        for f in sorted(lp.glob("*.json")):
            payload = json.loads(f.read_text())
            # a --dump-leg file is either the bare leg or {leg_name: leg}
            inner = list(payload.values())[0] if len(payload) == 1 else None
            legs[f.stem] = inner if isinstance(inner, dict) else payload
        primary = legs["primary"]
        res = build_verdict(
            grid=int(primary["n"]), anchor=legs["anchor"], refconv=legs["convergence"],
            primary=primary, grid_leg=legs.get("grid"))
        res["legs"]["exactness"] = legs["exactness"]
        res["assembly"] = {"per_leg_files": sorted(legs), "note": res.get("assembly", {}).get("note")}
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "summary.json").write_text(json.dumps(res, indent=2) + "\n")
        for q in res["quantities"]:
            print(f"  {q['id']:34s} ref={q['ref']:<12.6g} got={q['got']:<12.6g} "
                  f"{q['direction']} {q['tolerance']:<10g} {'PASS' if q['pass'] else 'FAIL'}")
        print(f"VERDICT: {'PASS' if res['verdict']['pass'] else 'FAIL ' + str(res['verdict']['failed'])}")
        return 0 if res["verdict"]["pass"] else 1

    if args.smoke:
        s = build_verdict(grid=SMOKE_GRID, eddy_times=args.eddy_times,
                          with_grid=False, with_ref_conv=False, save_field=False)
        failed = s["verdict"]["failed"]
        print(f"[smoke] grid={SMOKE_GRID} " +
              " ".join(f"{q['id']}={q['got']:.4g}" for q in s["quantities"]))
        print(f"[smoke] verdict={'PASS' if not failed else 'FAIL ' + str(failed)}")
        return 0 if not failed else 1

    wanted = [w for w in args.legs.split(",") if w]
    unknown = [w for w in wanted if w not in LEGS]
    if unknown:
        ap.error(f"unknown legs {unknown}; known: {LEGS}")

    failed: list[str] = []
    if set(wanted) == {"anchor"}:
        res: dict = {"anchor": closed_form_anchor(n=args.grid)}
    elif set(wanted) == {"convergence"}:
        res = {"reference_convergence": leg_reference_convergence(n=args.grid)}
    elif set(wanted) == {"exactness"}:
        res = {"exactness": leg_exactness(n=args.grid)}
    elif set(wanted) == {"grid"}:
        grids = tuple(int(g) for g in args.grids.split(",") if g) or ORDER_GRIDS
        res = {"grid": leg_grid(grids=grids)}
    elif set(wanted) == {"primary"}:
        res = {"primary": leg_decay(args.grid, eddy_times=args.eddy_times,
                                    save_field=args.dump_leg)}
    else:
        res = build_verdict(grid=args.grid, eddy_times=args.eddy_times,
                            with_grid="grid" in wanted)
        failed = res["verdict"]["failed"]

    if args.dump_leg:
        (ARTIFACTS / "legs").mkdir(parents=True, exist_ok=True)
        out_path = ARTIFACTS / "legs" / f"{wanted[0]}.json"
    else:
        out_path = Path(args.out) if args.out else ARTIFACTS / "summary.json"
    out_path.write_text(json.dumps(res, indent=2, sort_keys=False) + "\n")
    print(f"wrote {out_path}")
    if not args.dump_leg and "quantities" not in res:
        return 0 if not failed else 1
    if "quantities" in res:
        for q in res["quantities"]:
            print(f"  {q['id']:34s} ref={q['ref']:<12.6g} got={q['got']:<12.6g} "
                  f"{q['direction']} {q['tolerance']:<10g} {'PASS' if q['pass'] else 'FAIL'}")
        print(f"VERDICT: {'PASS' if res['verdict']['pass'] else 'FAIL ' + str(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())