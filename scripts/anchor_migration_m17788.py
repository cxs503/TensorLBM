#!/usr/bin/env python
"""External-anchor (Ghia 1982) tau-inversion migration study for
``GAP-PP-2-lid-wall-diff`` (selfstart seat m17788).

Background
----------
The merged differentiable moving-wall BC (``src/tensorlbm/diff_lid_cavity.py``)
was verified by m17147 with two readings of acceptance criterion ①
("Re=100 tau inversion recovers tau to <2 %"):

* **same-chain** (truth = the same differentiable chain): PASS, 3.6e-06;
* **external anchor** (truth = the Ghia et al. 1982 Re=100 centreline tables,
  i.e. data the inverse model did *not* generate): **FAIL at n=33**, Re error
  52.6 % -- the LBM's own discretisation error at that resolution biases the
  recovered tau toward lower Re.

m17147 flagged the *critical open item*: the n=33 single point cannot be
extrapolated.  If the external-anchor Re error drops below 15 % at n>=129
(and Ma<=0.05), the 52.6 % is a resolution limit, not a method defect.

This script answers exactly that, at two resolutions, with Ma controlled to
<=0.05 (u_lid=0.08 -> Ma=0.046).  For each resolution it reports:

* **grid optimum** -- a no-grad sweep of the Ghia-MSE over Re; the Re at which
  the *non-circular* objective actually minimises (the honest diagnostic);
* **root-finding inversion** -- the shipped
  :func:`tensorlbm.diff_lid_cavity.invert_relaxation_time_rootfind` (Illinois
  on dL/ds) against the Ghia target;
* the Ghia-MSE-vs-Re curve, so the report can show where the minimum sits.

Every number is written to JSON by the code path that computed it (no
hand-copying).  Usage::

    PYTHONPATH=src python scripts/anchor_migration_m17788.py \
        --n 129 --steps 16000 --u-lid 0.08 --re 100 \
        --tau-init 0.81 --max-iter 12 --threads 16 \
        --out artifacts/anchor_migration_m17788/anchor_n129.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from tensorlbm.d2q9 import macroscopic  # noqa: F401  (parity with verify script)
from tensorlbm.diff_lid_cavity import (
    GHIA_STATIONS,
    CavityGeometry,
    centreline_probes,
    initial_field,
    invert_relaxation_time_rootfind,
    rollout,
    tau_from_reynolds,
)
from tensorlbm.lid_driven_cavity import GHIA_RE100

DTYPE = torch.float64
RE_TARGET = 100.0


# ---------------------------------------------------------------------------
# Ghia (1982) table -> probe-layout target vector (copied from the m17147
# verifier so this study is self-contained and re-runnable without it).
# ---------------------------------------------------------------------------


def ghia_target(stations: Sequence[float] = GHIA_STATIONS) -> torch.Tensor:
    """Ghia Re=100 centreline data interpolated onto the Ghia probe layout.

    Returns ``(2*len(stations),)`` normalised (u/u_lid) values matching
    :func:`tensorlbm.diff_lid_cavity.centreline_probes`: ``u`` on the vertical
    centreline (bottom->top) then ``v`` on the horizontal centreline
    (left->right).  ``GHIA_RE100["y"]`` is stored descending, so it is reversed
    before interpolation.
    """
    y = list(GHIA_RE100["y"])
    u = list(GHIA_RE100["u"])
    x = list(GHIA_RE100["x"])
    v = list(GHIA_RE100["v"])
    if y[0] > y[-1]:
        y = y[::-1]
        u = u[::-1]
    if x[0] > x[-1]:
        x = x[::-1]
        v = v[::-1]
    uu = [_lin_interp(y, u, s) for s in stations]
    vv = [_lin_interp(x, v, s) for s in stations]
    return torch.tensor(uu + vv, dtype=DTYPE)


def _lin_interp(xs: list[float], ys: list[float], x: float) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            w = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] * (1.0 - w) + ys[i + 1] * w
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# parameterisation helpers
# ---------------------------------------------------------------------------


def _tau_of_re(n: int, re: float, u_lid: float) -> float:
    return tau_from_reynolds(re, u_lid, length=n - 1)


def _re_of_tau(n: int, tau: float, u_lid: float) -> float:
    nu = (tau - 0.5) / 3.0
    if nu <= 0.0:
        return float("inf")
    return u_lid * (n - 1) / nu


def _tau_of_re_direct(n: int, re: float, u_lid: float) -> float:
    """tau for a *given* Re (used to build the sweep grid in Re-space)."""
    nu = u_lid * (n - 1) / re
    return 0.5 + 3.0 * nu


# ---------------------------------------------------------------------------
# stage A: no-grad Ghia-MSE sweep over Re (the honest diagnostic)
# ---------------------------------------------------------------------------


def run_sweep(geom: CavityGeometry, target: torch.Tensor, u_lid: float,
               steps: int, re_grid: Sequence[float]) -> list[dict[str, Any]]:
    f0 = initial_field(geom)
    mask = geom.wall_mask
    curve = []
    for re in re_grid:
        tv = _tau_of_re_direct(geom.nx, re, u_lid)
        with torch.no_grad():
            f = rollout(f0, torch.tensor(tv, dtype=DTYPE), u_lid, steps,
                        wall_mask=mask)
            obs = centreline_probes(f, u_lid)
            loss = float(((obs - target) ** 2).mean())
        curve.append({"re": re, "tau": tv, "mse_vs_ghia": loss,
                      "rmse_vs_ghia": math.sqrt(loss) if loss == loss else None})
    return curve


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict[str, Any]:
    n, steps, u_lid, re = args.n, args.steps, args.u_lid, args.re
    geom = CavityGeometry(ny=n, nx=n)
    tau_true = _tau_of_re(n, re, u_lid)
    ma = u_lid / math.sqrt(3.0)
    target = ghia_target()

    # ---- stage A: no-grad sweep over Re ---------------------------------
    re_grid = list(args.re_grid)
    t0 = time.time()
    curve = run_sweep(geom, target, u_lid, steps, re_grid)
    sweep_wall = time.time() - t0
    finite = [c for c in curve if c["mse_vs_ghia"] is not None and c["mse_vs_ghia"] == c["mse_vs_ghia"]]
    best = min(finite, key=lambda c: c["mse_vs_ghia"]) if finite else None
    re_true = _re_of_tau(n, tau_true, u_lid)
    grid_summary = None
    if best is not None:
        grid_summary = {
            "re_at_min": best["re"],
            "tau_at_min": best["tau"],
            "rmse_vs_ghia_at_min": best["rmse_vs_ghia"],
            "rel_err_re_vs_true": abs(best["re"] - re_true) / re_true,
            "monotonic_decreasing": all(
                curve[i + 1]["mse_vs_ghia"] is not None
                and curve[i]["mse_vs_ghia"] is not None
                and curve[i + 1]["mse_vs_ghia"] < curve[i]["mse_vs_ghia"]
                for i in range(len(curve) - 1)
            ),
        }

    # ---- stage B: differentiable root-finding inversion -----------------
    # wide bracket covering the sweep range so the root-finder can walk to the
    # (biased) minimum instead of being trapped near tau_init.
    tau_lo = _tau_of_re_direct(n, max(re_grid), u_lid)
    tau_hi = _tau_of_re_direct(n, min(re_grid), u_lid)
    bracket = (min(tau_lo, tau_hi), max(tau_lo, tau_hi))
    t1 = time.time()
    res = invert_relaxation_time_rootfind(
        target, geom, u_lid, args.tau_init, steps,
        method="illinois", bracket_tau=bracket, max_iter=args.max_iter,
        step_tol=1e-9, checkpoint_every=200,
    )
    inv_wall = time.time() - t1
    tau_rec = res["tau_recovered"]
    re_rec = _re_of_tau(n, tau_rec, u_lid)
    tau_best = res["tau_best"]
    re_best = _re_of_tau(n, tau_best, u_lid)

    out = {
        "case": "GAP-PP-2-lid-wall-diff / EXTERNAL anchor (Ghia 1982) migration",
        "n": n, "steps": steps, "u_lid": u_lid, "ma": ma,
        "re_true": re_true, "tau_true": tau_true,
        "tau_init": args.tau_init, "bracket_tau": list(bracket),
        "sweep": {
            "re_grid": re_grid,
            "wall_time_s": sweep_wall,
            "curve": curve,
            "grid_optimum": grid_summary,
        },
        "inversion": {k: v for k, v in res.items() if k != "obs_final"},
        "inversion_wall_time_s": inv_wall,
        "obs_final": [float(x) for x in res["obs_final"]],
        "summary": {
            "tau_recovered": tau_rec,
            "re_recovered": re_rec,
            "rel_err_re_recovered": abs(re_rec - re_true) / re_true,
            "tau_best": tau_best,
            "re_best": re_best,
            "rel_err_re_best": abs(re_best - re_true) / re_true,
            "residual_rms": res["residual_rms"],
            "n_evals": res["n_evals"],
            "status": res["status"],
            "criterion_re_lt_15pct": (abs(re_rec - re_true) / re_true) < 0.15,
        },
        "torch": torch.__version__,
        "n_threads": torch.get_num_threads(),
    }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=129)
    p.add_argument("--steps", type=int, default=16000)
    p.add_argument("--u-lid", type=float, default=0.08)
    p.add_argument("--re", type=float, default=RE_TARGET)
    p.add_argument("--tau-init", type=float, default=None)
    p.add_argument("--max-iter", type=int, default=12)
    p.add_argument("--re-grid", type=float, nargs="+",
                   default=[50, 62.5, 75, 87.5, 100, 112.5, 125, 150, 175, 200])
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    if args.tau_init is None:
        args.tau_init = _tau_of_re(args.n, args.re, args.u_lid)

    t0 = time.time()
    payload = run(args)
    payload["total_wall_time_s"] = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    s = payload["summary"]
    print(json.dumps({
        "n": args.n, "re_true": payload["re_true"],
            "re_recovered": s["re_recovered"],
            "rel_err_re_recovered": s["rel_err_re_recovered"],
            "re_grid_opt": (payload["sweep"]["grid_optimum"] or {}).get("re_at_min"),
            "rel_err_re_grid_opt": (payload["sweep"]["grid_optimum"] or {}).get("rel_err_re_vs_true"),
            "criterion_re_lt_15pct": s["criterion_re_lt_15pct"],
            "total_wall_time_s": payload["total_wall_time_s"],
    }, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
