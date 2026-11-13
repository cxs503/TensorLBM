#!/usr/bin/env python3
"""Lid-driven square cavity, Re = 5000 -- TensorLBM benchmark runner (CASE-CAVITY-RE5000).

Reference benchmark
-------------------
U. Ghia, K. N. Ghia, C. T. Shin, "High-Re solutions for incompressible flow using
the Navier-Stokes equations and a multigrid method", *J. Comput. Phys.* **48**
(1982) 387-411.  Re = 5000 lives in Table III (u on x/L = 0.5) / Table IV
(v on y/L = 0.5); the Re = 100/400/1000 columns are Table I/II of the same paper.

Contract (pinned by ``tests/test_cavity_re5000_runner.py``)
-----------------------------------------------------------
* every invocation writes ``--out`` with a machine-readable payload, even when the
  run ends before the first residual monitor fires (``steps < monitor_every``);
* a finite solution exits ``0`` with ``run_status == "ok"``; a diverged / non-finite
  solution exits ``2`` with ``run_status == "failed"``, ``diverged == True`` and
  ``comparison == {"status": "VOID", ...}`` -- never a silent "pass";
* a missing reference makes the comparison ``WITHHELD`` (never a fake pass);
* with ``--ref-json`` every grid gets a scored block (``u_rows``/``v_rows``/
  ``pass``/``max_abs_err_pct``) plus ``comparison["ref_provenance"]``.

Convergence evidence (added for the CASE-CAVITY-RE5000 verification debt)
------------------------------------------------------------------------
``--monitor-every`` records ``resid_history = [[step, max|du|, max|du|/u_lid,
max_err_pct_vs_ref, u_min, v_max], ...]``; ``--resid-tol`` stops the run as soon
as ``max|du|`` over one monitor interval drops below the tolerance, and the grid
record carries ``converged``.  Without a residual tolerance the number of steps is
an arbitrary truncation, so such a run must *not* be quoted as a steady solution.

Single precision of the picture: ``--center-average`` (default on) samples the two
central lattice planes, i.e. the geometric centre x/L = y/L = 0.5 for an even ``n``
(the walls sit half a lattice unit outside the first/last fluid rows), and
``--coord-convention`` selects how lattice rows map to ``y/L``:

``node``  (legacy)  y = j/(n-1)              -- walls at y = 0 and 1 exactly on rows
``wall``  (physical) y = (j-0.5)/(n-2)        -- half-way bounce-back walls at 0 and 1

Usage
-----
    python benchmarks/verified/cavity/re5000/run.py --n 128 256 --steps 50000
    python benchmarks/verified/cavity/re5000/run.py --n 256 --steps 400000 \
        --monitor-every 5000 --resid-tol 1e-7 \
        --ref-json benchmarks/verified/cavity/re5000/reference/ghia_re5000_reference.json

Exit codes
----------
``0``  every grid reached a finite solution (comparison may still be WITHHELD)
``2``  at least one grid diverged / produced non-finite fields
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "TensorLBM" / "src"))

from tensorlbm.verification.lid_driven_cavity.solver import (  # noqa: E402
    CavityConfig,
    CavitySolver,
)

import tensorlbm  # noqa: E402

TOL_PCT = 8.0        # mission tolerance, % of the lid speed
U_LID = 0.1
RE = 5000.0


# --------------------------------------------------------------------------- #
# reference handling
# --------------------------------------------------------------------------- #
def load_reference(ref_json=None):
    """Return (y, u_ref, x, v_ref, provenance) or None when no reference is given.

    ``ref_json`` format (see ``reference/ghia_re5000_reference.json``):
    ``{"u_re5000": [[y, u], ...], "v_re5000": [[x, v], ...], "source": ...}``.
    No library fallback on purpose: a path that silently substitutes an
    unverified table cannot be told apart from a verified one.
    """
    if not ref_json:
        return None
    path = Path(ref_json)
    if not path.is_file():
        raise FileNotFoundError(f"--ref-json not found: {path}")
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    u = np.asarray(data["u_re5000"], dtype=float)
    v = np.asarray(data["v_re5000"], dtype=float)
    if u.ndim != 2 or u.shape[1] != 2 or v.ndim != 2 or v.shape[1] != 2:
        raise ValueError("u_re5000 / v_re5000 must be [[coord, value], ...]")
    provenance = {
        "source": data.get("source"),
        "table": data.get("table"),
        "page": data.get("page"),
        "declared_sha256": data.get("sha256"),
        "ref_file": str(path),
        "ref_sha256": hashlib.sha256(raw).hexdigest(),
        "n_rows": [int(u.shape[0]), int(v.shape[0])],
    }
    return u[:, 0], u[:, 1], v[:, 0], v[:, 1], provenance


def _coords(n: int, convention: str):
    """Map lattice row index -> normalised wall coordinate y/L (or x/L)."""
    j = np.arange(n, dtype=float)
    if convention == "wall":
        return (j - 0.5) / (n - 2)
    return j / (n - 1)


def _centerline(u: torch.Tensor, v: torch.Tensor, n: int, average: bool):
    """u on the vertical centreline, v on the horizontal one.

    With an even ``n`` the geometric centre of the cavity lies halfway between the
    two central lattice planes, so the two planes are averaged (default).
    """
    mid = n // 2
    if average and n % 2 == 0:
        uc = 0.5 * (u[:, mid - 1] + u[:, mid])
        vc = 0.5 * (v[mid - 1, :] + v[mid, :])
    else:
        uc = u[:, mid]
        vc = v[mid, :]
    return uc.cpu().numpy(), vc.cpu().numpy()


def score_grid(rec, ref, convention="node", average=True):
    """Pointwise comparison of one grid record against the reference table."""
    y_ref, u_ref, x_ref, v_ref, _ = ref
    y = np.asarray(rec[f"y_centerline_{convention}"])
    x = np.asarray(rec[f"x_centerline_{convention}"])
    u_got = np.interp(y_ref, y, np.asarray(rec["u_centerline"]))
    v_got = np.interp(x_ref, x, np.asarray(rec["v_centerline"]))
    du = (u_got - u_ref) * 100.0          # % of lid speed (profiles are normalised)
    dv = (v_got - v_ref) * 100.0
    max_err = float(max(np.abs(du).max(), np.abs(dv).max()))
    rmse = float(np.sqrt(np.mean(np.concatenate([du, dv]) ** 2)))
    return {
        "max_abs_err_pct": max_err,
        "rmse_pct": rmse,
        "pass": bool(max_err <= rec.get("tolerance_pct", TOL_PCT)),
        "u_rows": [[float(a), float(b), float(c), float(d)]
                   for a, b, c, d in zip(y_ref, u_ref, u_got, du)],
        "v_rows": [[float(a), float(b), float(c), float(d)]
                   for a, b, c, d in zip(x_ref, v_ref, v_got, dv)],
    }


# --------------------------------------------------------------------------- #
# single grid
# --------------------------------------------------------------------------- #
def run_case(n, re=RE, u_lid=U_LID, max_steps=50000, monitor_every=5000,
             resid_tol=None, coord="node", center_average=True,
             ref=None, verbose=True):
    """Run one grid; always returns a JSON-serialisable record."""
    rec = {
        "n": n,
        "re": re,
        "u_lid": u_lid,
        "max_steps": max_steps,
        "monitor_every": monitor_every,
        "resid_tol": resid_tol,
        "dtype": "float64",
        "tolerance_pct": TOL_PCT,
    }
    try:
        cfg = CavityConfig(n=n, re=re, u_lid=u_lid, max_steps=max_steps)
        solver = CavitySolver(cfg)
    except Exception as exc:                       # e.g. n < 17 or tau <= 0.5
        rec.update(status="config_error", error=f"{type(exc).__name__}: {exc}",
                   elapsed_s=0.0, resid_history=[], final_resid=None, steps=0,
                   y_centerline_node=_coords(n, "node").tolist(),
                   y_centerline_wall=_coords(n, "wall").tolist(),
                   u_centerline=None, v_centerline=None)
        rec["x_centerline_node"] = rec["y_centerline_node"]
        rec["x_centerline_wall"] = rec["y_centerline_wall"]
        return rec
    rec.update(tau=cfg.tau, nu_lattice=cfg.nu,
               collision="rlbm" if cfg.use_rlbm else "bgk")

    t0 = time.time()
    history = []
    converged = False
    diverged = False
    step = 0
    _, ux_prev, uy_prev = solver._macroscopic(solver.f)

    for step in range(1, max_steps + 1):
        solver.step()
        if step % monitor_every == 0:
            rho, ux, uy = solver._macroscopic(solver.f)
            with torch.no_grad():
                du = float(torch.max(torch.abs(ux[solver.fluid] - ux_prev[solver.fluid])))
                dv = float(torch.max(torch.abs(uy[solver.fluid] - uy_prev[solver.fluid])))
            resid = max(du, dv)
            finite = bool(np.isfinite(resid)) and bool(torch.all(torch.isfinite(ux[solver.fluid])))
            ux_prev, uy_prev = ux.clone(), uy.clone()
            if not finite:
                diverged = True
                history.append([step, None, None, None, None, None])
                break
            u_now, v_now = _centerline(ux, uy, n, center_average)
            u_now = u_now / u_lid
            v_now = v_now / u_lid
            err = None
            if ref is not None:
                y_ref = ref[0]
                x_ref = ref[2]
                uy_got = np.interp(y_ref, _coords(n, coord), u_now)
                vx_got = np.interp(x_ref, _coords(n, coord), v_now)
                err = float(max(np.abs(uy_got - ref[1]).max(),
                                np.abs(vx_got - ref[3]).max()) * 100.0)
            history.append([step, resid, resid / u_lid, err,
                            float(u_now.min()), float(v_now.max())])
            if verbose:
                e = "n/a" if err is None else f"{err:6.2f}%"
                print(f"  n={n} step={step} max|du|={resid:.3e} "
                      f"(rel {resid / u_lid:.2e}) err={e} "
                      f"u_min={u_now.min():+.4f} v_max={v_now.max():+.4f}", flush=True)
            if resid_tol is not None and resid < resid_tol:
                converged = True
                break

    elapsed = time.time() - t0
    rho, ux, uy = solver._macroscopic(solver.f)
    uc, vc = _centerline(ux, uy, n, center_average)
    uc = uc / u_lid
    vc = vc / u_lid
    finite = bool(np.all(np.isfinite(uc)) and np.all(np.isfinite(vc)))
    if diverged or not finite:
        status = "diverged" if diverged else "non_finite"
    elif converged:
        status = "ok"
    elif resid_tol is not None:
        status = "not_converged"
    else:
        status = "ok"

    y = _coords(n, "node")
    y_wall = _coords(n, "wall")
    rec.update(
        status=status,
        diverged=bool(diverged or not finite),
        converged=bool(converged),
        steps=int(step),
        elapsed_s=round(elapsed, 1),
        resid_history=history,
        final_resid=(history[-1][1] if history else None),
        final_resid_rel=(history[-1][2] if history else None),
        u_min=(float(uc.min()) if finite else None),
        u_min_y=float(y[int(np.argmin(uc))]),
        v_max=(float(vc.max()) if finite else None),
        v_max_x=float(y[int(np.argmax(vc))]),
        u_centerline=uc.tolist(),
        v_centerline=vc.tolist(),
        y_centerline_node=y.tolist(),
        x_centerline_node=y.tolist(),
        y_centerline_wall=y_wall.tolist(),
        x_centerline_wall=y_wall.tolist(),
    )
    if not finite:
        rec["u_centerline"] = [None] * n
        rec["v_centerline"] = [None] * n
    return rec


def grid_convergence(runs, coord="node"):
    """Change of u(0.5,y) between the two finest usable grids."""
    usable = {k: r for k, r in runs.items()
              if r.get("status") == "ok" and r.get("u_centerline")}
    if len(usable) < 2:
        return {"status": "single-grid"}
    keys = sorted(usable, key=lambda k: usable[k]["n"])
    fine, coarse = usable[keys[-1]], usable[keys[-2]]
    y = np.asarray(fine[f"y_centerline_{coord}"])
    uc = np.interp(y, np.asarray(coarse[f"y_centerline_{coord}"]),
                   np.asarray(coarse["u_centerline"], dtype=float))
    d = np.abs(np.asarray(fine["u_centerline"], dtype=float) - uc) * 100.0
    return {"pair": [coarse["n"], fine["n"]],
            "max_abs_delta_pct": float(d.max()),
            "rmse_pct": float(np.sqrt(np.mean(d ** 2)))}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cavity Re=5000 benchmark (CASE-CAVITY-RE5000)")
    ap.add_argument("--n", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--u-lid", type=float, default=U_LID)
    ap.add_argument("--re", type=float, default=RE)
    ap.add_argument("--monitor-every", type=int, default=5000)
    ap.add_argument("--resid-tol", type=float, default=None,
                    help="stop when max|du| over one monitor interval < this (lattice units)")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    ap.add_argument("--coord-convention", choices=("node", "wall"), default="node")
    ap.add_argument("--no-center-average", action="store_true")
    ap.add_argument("--ref-json", default=None,
                    help="reference table JSON; omitted -> comparison WITHHELD")
    ap.add_argument("--selfcheck", action="store_true",
                    help="Re=1000 vs Ghia Table I/II on the same code path must pass first")
    ap.add_argument("--out", default=str(Path(__file__).parent / "result.json"))
    args = ap.parse_args(argv)

    if args.threads:
        torch.set_num_threads(int(args.threads))

    ref = load_reference(args.ref_json)
    ref_prov = ref[4] if ref else None
    average = not args.no_center_average
    coord = args.coord_convention

    payload = {
        "case": "cavity_re5000_ghia",
        "description": "Lid-driven cavity Re=5000, D2Q9 RLBM, Ghia(1982) Table III/IV verification",
        "lattice": "D2Q9",
        "collision": "rlbm (regularized BGK, Latt & Chopard 2006)",
        "boundary": "half-way bounce-back (3 stationary walls) + Zou-He velocity (moving lid)",
        "tolerance_pct": TOL_PCT,
        "coordinate_convention": coord,
        "centerline_sampling": "mean of the two central lattice planes" if average
                               else "single central lattice plane",
        "reference": "Ghia 1982 Table III (u, x=0.5) / Table IV (v, y=0.5), Re=5000 column",
        "ref_file": args.ref_json,
        "ref_sha256": ref_prov["ref_sha256"] if ref_prov else None,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "dtype": "float64",
            "tensorlbm": tensorlbm.__file__,
        },
        "acceptance_policy": {
            "pointwise": f"|ref-got| <= {TOL_PCT}% of the lid speed on Table III/IV rows",
            "convergence": "resid (max|du| per monitor interval) must reach --resid-tol; "
                           "a truncated run is reported not_converged",
            "grid_convergence": "coarse->fine change of u(0.5,y) must be below the "
                                "finest-grid error to the reference",
        },
    }

    if args.selfcheck:
        # Pre-declared: Re=1000 on the same code path must pass 8 % first.
        self_ref = None
        try:
            from tensorlbm.verification.lid_driven_cavity.reference_data import GHIA_RE1000
            self_ref = (np.asarray(GHIA_RE1000.y), np.asarray(GHIA_RE1000.u),
                        np.asarray(GHIA_RE1000.x), np.asarray(GHIA_RE1000.v),
                        {"source": "library GHIA_RE1000 (Ghia 1982 Table I/II)"})
        except Exception as exc:                   # pragma: no cover
            print(f"[selfcheck] reference unavailable: {exc}", flush=True)
        if self_ref is not None:
            sc = run_case(128, re=1000.0, u_lid=args.u_lid, max_steps=30000,
                          monitor_every=args.monitor_every, resid_tol=None,
                          coord=coord, center_average=average, ref=self_ref,
                          verbose=False)
            sc_score = score_grid(sc, self_ref, coord, average)
            payload["harness_selfcheck"] = {
                "grid": 128, "re": 1000.0, "steps": sc["steps"],
                "final_resid": sc["final_resid"], **sc_score,
            }
            print(f"[selfcheck] max_err={sc_score['max_abs_err_pct']:.2f}% "
                  f"pass={sc_score['pass']}", flush=True)

    runs = {}
    for n in args.n:
        print(f"[run] n={n} re={args.re} u_lid={args.u_lid} monitor={args.monitor_every} "
              f"resid_tol={args.resid_tol}", flush=True)
        runs[str(n)] = run_case(n, re=args.re, u_lid=args.u_lid, max_steps=args.steps,
                                monitor_every=args.monitor_every, resid_tol=args.resid_tol,
                                coord=coord, center_average=average, ref=ref)
        r = runs[str(n)]
        print(f"[done] n={n} steps={r['steps']} status={r['status']} "
              f"final_resid={r['final_resid']} t={r['elapsed_s']:.0f}s", flush=True)

    payload["grids"] = runs
    payload["grid_convergence"] = grid_convergence(runs, coord)

    bad = {k: v.get("status") for k, v in runs.items()
           if v.get("status") in ("diverged", "non_finite", "config_error")}
    if bad:
        # a non-finite solution voids the case even when no reference was given:
        # WITHHELD would suggest "could not be scored", not "the run blew up"
        payload["comparison"] = {
            "status": "VOID",
            "reason": f"non-finite solution on grid(s) {sorted(bad)} -- "
                      f"no benchmark number is reported",
            "failed_grids": bad,
        }
    elif ref is None:
        payload["comparison"] = {
            "status": "WITHHELD",
            "reason": "no --ref-json reference table supplied -- no benchmark number is reported",
        }
    else:
        comparison = {}
        for k, r in runs.items():
            if r.get("u_centerline"):
                comparison[k] = score_grid({**r, "tolerance_pct": TOL_PCT}, ref, coord, average)
        comparison["ref_provenance"] = ref_prov
        payload["comparison"] = comparison

    # pre-declared verdict, computed from the numbers above (never hand-set)
    verdict = {"pointwise_pass": None, "grid_converged": None, "notes": []}
    if isinstance(payload["comparison"], dict) and "status" not in payload["comparison"]:
        worst = max(comparison[k]["max_abs_err_pct"] for k in comparison
                    if k not in ("ref_provenance",))
        verdict["pointwise_pass"] = bool(worst <= TOL_PCT)
        verdict["worst_max_abs_err_pct"] = worst
        gc = payload["grid_convergence"]
        if "max_abs_delta_pct" in gc:
            verdict["grid_converged"] = bool(gc["max_abs_delta_pct"] < worst)
            verdict["grid_convergence_delta_pct"] = gc["max_abs_delta_pct"]
        not_conv = [k for k, r in runs.items() if not r.get("converged")
                    and args.resid_tol is not None]
        if not_conv:
            verdict["notes"].append(
                f"residual tolerance not reached on grid(s) {not_conv}: "
                f"the numbers are a truncation, not a steady state")
    payload["verdict"] = verdict
    payload["diverged"] = any(bool(r.get("diverged")) for r in runs.values())
    payload["run_status"] = "ok" if not bad else "failed"
    payload["failed_grids"] = bad
    payload["wall_clock_s"] = round(sum(r.get("elapsed_s", 0.0) for r in runs.values()), 1)

    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"DONE {args.out} run_status={payload['run_status']}", flush=True)
    if bad:
        print(f"[FAIL] non-finite/diverged grids: {bad} -> exit 2", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())