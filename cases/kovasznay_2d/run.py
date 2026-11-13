#!/usr/bin/env python3
"""CASE-LBM-KOVASZNAY-EXACT -- case-pool entry & acceptance gate.

Anchor
------
``feature_ledger`` slug ``CASE-LBM-KOVASZNAY-EXACT`` (TensorLBM case pool):

    item       : 2D steady Kovasznay exact-solution flow field
    reference  : Kovasznay 1948, Proc. Camb. Phil. Soc. 44:58,
                 DOI 10.1017/S0305004100023999
    tolerance  : < 3 %
    metric     : velocity-field L2 relative error vs. the exact solution,
                 judged at Re = 40 with N >= 128

What this entry adds over ``benchmarks/verified/kovasznay_2d``
-------------------------------------------------------------
The benchmark (and the three follow-up missions m16861 / m16974 / m17161) settled
that the *full-field* L2 error is 2.55 % at ny = 128 -- i.e. it passes the gate
by 0.45 percentage points, with ~85 % of the residual attributed to the
zero-gradient outlet boundary layer, and with every alternative outlet closure
(nscbc, NEE, analytic Dirichlet) measurably worse or non-physical.  What was
never done is the thing that turns "passes the tolerance" into "verified
solution":

1. **exact reference certification** -- the continuous N-S residual of the
   analytic field is evaluated with *analytic* derivatives (``analytic.py``),
   giving ~1e-16 instead of the ~1e-3 finite-difference number quoted in the
   older benchmark docs.  The reference is exact; the residual error is
   therefore purely numerical.
2. **outlet-pollution scaling law** -- the full-field error is measured as a
   function of the downstream domain length ``xmax`` at fixed ``ny``.  This is
   the single variable that isolates the boundary term (earlier missions varied
   the *closure* instead, which is a different axis).
3. **boundary-free order of accuracy** -- the L2 error is re-measured on an
   interior window that excludes the outlet layer, then refined 32 -> 64 -> 128
   to report the observed order of accuracy ``p`` (the quantity a verification
   case is supposed to deliver) and the Richardson extrapolated ``h -> 0``
   error.

The acceptance gate itself is *unchanged*: full-field velocity L2 relative
error < 3 % at Re = 40, ny = 128.

Usage
-----
::

    python cases/kovasznay_2d/run.py single 128 out.json --xmax 3.0
    python cases/kovasznay_2d/run.py gate   out_gate.json
    python cases/kovasznay_2d/run.py study  out_study.json --threads 16
    python cases/kovasznay_2d/run.py check         # reference certification only

All lattice operators are public library functions
(``tensorlbm.d2q9.equilibrium``, ``tensorlbm.solver.collide_bgk`` /
``stream`` / ``macroscopic``, ``tensorlbm.boundaries.zou_he_inlet_velocity`` /
``zero_gradient_outlet_2d``); this module only arranges them and measures.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tensorlbm.boundaries import zero_gradient_outlet_2d, zou_he_inlet_velocity  # noqa: E402
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402

from analytic import kovasznay_lambda, make_field  # noqa: E402

RE_DEFAULT = 40.0
U0_DEFAULT = 0.03
STEPS_DEFAULT = 12_000
STEADY_TOL = 5e-5  # ||u(t)-u(t-2000)||2 / ||u(t)||2, see benchmark note on slow mass leak


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------


def simulate(
    ny: int,
    *,
    re: float = RE_DEFAULT,
    u0: float = U0_DEFAULT,
    xmax: float = 3.0,
    steps: int = STEPS_DEFAULT,
    max_steps: int | None = None,
    dtype: torch.dtype = torch.float64,
    threads: int = 16,
    device: str = "cpu",
    field_out: Path | None = None,
) -> dict:
    """Run the Kovasznay case to steady state and measure every error metric."""
    torch.set_num_threads(threads)
    dev = torch.device(device)
    field = make_field(ny, re=re, u0=u0, xmax=xmax)
    nx = field.nx
    tau = field.tau
    if tau <= 0.5:
        raise ValueError(f"tau must exceed 0.5 for BGK stability, got {tau}")

    u_ana = torch.as_tensor(field.u(), dtype=dtype, device=dev)
    v_ana = torch.as_tensor(field.v(), dtype=dtype, device=dev)
    rho0 = torch.ones((ny, nx), dtype=dtype, device=dev)
    f = equilibrium(rho0, u_ana, v_ana)
    mass0 = float(f.sum().item())

    u_in = u_ana[:, 0].contiguous()
    v_in = v_ana[:, 0].contiguous()

    max_steps = int(max_steps if max_steps is not None else max(steps, 4 * steps))

    def step(f: torch.Tensor) -> torch.Tensor:
        f = collide_bgk(f, tau)
        f = stream(f)
        f = zero_gradient_outlet_2d(f)
        return zou_he_inlet_velocity(f, u_in, v_in)

    t0 = time.time()
    hist: list[tuple[int, torch.Tensor]] = []
    n_done = 0
    steady = False
    for n_done in range(1, max_steps + 1):
        f = step(f)
        if n_done % 1000 == 0:
            _, ux, _ = macroscopic(f)
            hist.append((n_done, ux.detach().to(torch.float64)))
            if n_done >= steps and len(hist) >= 3:
                prev = hist[-3][1]
                num = float(torch.linalg.vector_norm(ux.detach().to(torch.float64) - prev))
                den = float(torch.linalg.vector_norm(hist[-1][1]))
                if num / max(den, 1e-30) < STEADY_TOL:
                    steady = True
                    break

    # time-average the last 100 steps to remove round-off noise
    acc_u = torch.zeros((ny, nx), dtype=torch.float64, device=dev)
    acc_v = torch.zeros((ny, nx), dtype=torch.float64, device=dev)
    navg = 100
    for _ in range(navg):
        f = step(f)
        _, ux, uy = macroscopic(f)
        acc_u += ux.to(torch.float64)
        acc_v += uy.to(torch.float64)
    u_num = (acc_u / navg).cpu().numpy()
    v_num = (acc_v / navg).cpu().numpy()
    elapsed = time.time() - t0

    u_a = field.u()
    v_a = field.v()
    mass_final = float(f.sum().item())

    # interior window: drop the inlet column and one full y-period (ny cells)
    # at the outlet, i.e. the band the zero-gradient closure is known to pollute.
    n_bl = min(ny, max(1, nx // 2))
    lo, hi = 1, max(1, nx - n_bl)
    interior = slice(lo, hi)

    metrics = dict(
        ny=ny,
        nx=nx,
        xmax=xmax,
        re=re,
        u0=u0,
        tau=tau,
        nu=field.nu,
        lam=field.lam,
        ma_max=field.ma_max,
        dtype=str(dtype).replace("torch.", ""),
        steps=n_done,
        steady=steady,
        elapsed_s=round(elapsed, 2),
        mass_drift_pct=(mass_final - mass0) / mass0 * 100.0,
        finite=bool(np.isfinite(u_num).all() and np.isfinite(v_num).all()),
        full=error_block(u_num, v_num, u_a, v_a),
        interior={
            "cols": [lo, hi],
            "n_cols": hi - lo,
            **error_block(u_num[:, interior], v_num[:, interior], u_a[:, interior], v_a[:, interior]),
        },
    )
    if field_out is not None:
        field_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            field_out, u_num=u_num, v_num=v_num, u_ana=u_a, v_ana=v_a,
            ny=ny, nx=nx, xmax=xmax, re=re, u0=u0, tau=tau,
        )
    metrics["velocity_l2_rel_pct"] = max(
        metrics["full"]["u_l2_rel_pct"], metrics["full"]["v_l2_rel_pct"]
    )
    return metrics


def error_block(u_num: np.ndarray, v_num: np.ndarray, u_a: np.ndarray, v_a: np.ndarray) -> dict:
    """L2-relative, masked-max-relative and absolute error metrics."""

    def l2_rel(a: np.ndarray, b: np.ndarray) -> float:
        den = float(np.linalg.norm(b))
        return float(np.linalg.norm(a - b) / den) if den > 0 else float("nan")

    def max_rel(a: np.ndarray, b: np.ndarray, thresh: float) -> float:
        mask = np.abs(b) > thresh
        if not np.any(mask):
            return float("nan")
        return float(np.max(np.abs(a[mask] - b[mask]) / np.abs(b[mask])))

    u0 = float(np.max(np.abs(u_a))) if np.any(u_a) else 1.0
    vmax = float(np.max(np.abs(v_a))) if np.any(v_a) else 1.0
    return {
        "u_l2_rel_pct": 100.0 * l2_rel(u_num, u_a),
        "v_l2_rel_pct": 100.0 * l2_rel(v_num, v_a),
        "u_max_rel_pct": 100.0 * max_rel(u_num, u_a, 0.1 * u0),
        "v_max_rel_pct": 100.0 * max_rel(v_num, v_a, 0.1 * vmax),
        "u_max_abs_over_uref": float(np.max(np.abs(u_num - u_a)) / u0),
        "v_max_abs_over_uref": float(np.max(np.abs(v_num - v_a)) / vmax),
    }


# ---------------------------------------------------------------------------
# reference certification
# ---------------------------------------------------------------------------


def reference_check(ny: int = 64, re: float = RE_DEFAULT, u0: float = U0_DEFAULT) -> dict:
    """Certify the analytic reference: exact N-S residual at machine precision."""
    field = make_field(ny, re=re, u0=u0, xmax=3.0)
    res = field.ns_residual()
    res.update(re=re, u0=u0, lam=field.lam, ny=ny)
    res["lam_check_vs_formula"] = abs(field.lam - kovasznay_lambda(re))
    res["lam_is_negative"] = bool(field.lam < 0)
    res["verdict"] = bool(
        res["res_mom_x_max"] < 1e-12
        and res["res_mom_y_max"] < 1e-12
        and res["res_div_max"] < 1e-12
        and res["compat_lhs"] < 1e-8
    )
    return res


# ---------------------------------------------------------------------------
# acceptance gate (unchanged anchor criterion)
# ---------------------------------------------------------------------------


def gate(ny: int = 128, **kw) -> dict:
    m = simulate(ny, xmax=3.0, **kw)
    ok = (
        m["finite"]
        and m["steps"] >= STEPS_DEFAULT
        and m["full"]["u_l2_rel_pct"] < 3.0
        and m["full"]["v_l2_rel_pct"] < 3.0
    )
    return {
        "case": "CASE-LBM-KOVASZNAY-EXACT",
        "lattice": "D2Q9",
        "collision": "BGK",
        "reference": "Kovasznay 1948 Proc. Camb. Phil. Soc. 44:58 (DOI 10.1017/S0305004100023999)",
        "criterion": "velocity-field L2 relative error vs exact solution < 3% at Re=40, N>=128",
        "criterion_metric_definition": (
            "max(u, v) full-field L2 relative error [%]; both components must be < 3%"
        ),
        "measured": m["full"],
        "velocity_l2_rel_pct": m["velocity_l2_rel_pct"],
        "verdict": "PASS" if ok else "FAIL",
        "extrap": "none",
        "reference_certification": reference_check(re=kw.get("re", RE_DEFAULT)),
        "run": m,
    }


# ---------------------------------------------------------------------------
# study: outlet-pollution scaling + boundary-free order of accuracy
# ---------------------------------------------------------------------------


def observed_order(err_coarse: float, err_fine: float, ratio: float = 2.0) -> float:
    if err_coarse <= 0 or err_fine <= 0:
        return float("nan")
    return math.log(err_coarse / err_fine) / math.log(ratio)


def study(
    threads: int = 16,
    steps: int = STEPS_DEFAULT,
    dtype=torch.float64,
    cache_dir: Path | None = None,
) -> dict:
    """Two orthogonal sweeps, each a single variable away from the shipped case.

    * **xmax sweep** (fixed ``ny``): isolates the outlet boundary term.
    * **ny sweep** (fixed ``xmax`` = 3): grid refinement -> observed order ``p``.

    Runs are cached one JSON per configuration so the study is resumable.
    """
    plan = [(32, x) for x in (1.0, 2.0, 3.0, 4.0)] + [(64, x) for x in (1.0, 2.0, 3.0, 4.0)]
    plan += [(128, 3.0)]
    runs: list[dict] = []
    for ny, xmax in plan:
        cpath = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cpath = cache_dir / f"ny{ny}_xmax{xmax:g}.json"
        if cpath is not None and cpath.exists():
            runs.append(json.loads(cpath.read_text(encoding="utf-8")))
            print(f"[cache] ny={ny} xmax={xmax}", flush=True)
            continue
        rec = simulate(ny, xmax=xmax, steps=steps, threads=threads, dtype=dtype)
        print(
            f"[run] ny={ny} xmax={xmax} steps={rec['steps']} "
            f"u_full={rec['full']['u_l2_rel_pct']:.4f}% "
            f"v_full={rec['full']['v_l2_rel_pct']:.4f}% "
            f"u_int={rec['interior']['u_l2_rel_pct']:.4f}% "
            f"v_int={rec['interior']['v_l2_rel_pct']:.4f}% ({rec['elapsed_s']} s)",
            flush=True,
        )
        if cpath is not None:
            cpath.write_text(json.dumps(rec, indent=2, default=float), encoding="utf-8")
        runs.append(rec)

    refine = sorted([r for r in runs if abs(r["xmax"] - 3.0) < 1e-9], key=lambda r: r["ny"])

    order = {}
    for comp in ("u", "v"):
        full = [r["full"][f"{comp}_l2_rel_pct"] for r in refine]
        interior = [r["interior"][f"{comp}_l2_rel_pct"] for r in refine]
        p_f1 = observed_order(full[0], full[1])
        p_f2 = observed_order(full[1], full[2])
        p_i1 = observed_order(interior[0], interior[1])
        p_i2 = observed_order(interior[1], interior[2])
        rich = float("nan")
        if p_i2 > 0.5:
            factor = 2.0**p_i2
            rich = interior[2] / (factor - 1.0)
        order[comp] = {
            "ny": [r["ny"] for r in refine],
            "full_l2_pct": full,
            "interior_l2_pct": interior,
            "p_full_32_64": p_f1,
            "p_full_64_128": p_f2,
            "p_interior_32_64": p_i1,
            "p_interior_64_128": p_i2,
            "richardson_interior_pct": rich,
            "richardson_note": "interior error extrapolated to h->0 with the 64->128 slope",
        }

    scaling = {}
    for ny in (32, 64):
        sub = sorted([r for r in runs if r["ny"] == ny], key=lambda r: r["xmax"])
        xs = [r["xmax"] for r in sub]
        scaling[str(ny)] = {
            "xmax": xs,
            "u_full_pct": [r["full"]["u_l2_rel_pct"] for r in sub],
            "v_full_pct": [r["full"]["v_l2_rel_pct"] for r in sub],
            "u_interior_pct": [r["interior"]["u_l2_rel_pct"] for r in sub],
            "v_interior_pct": [r["interior"]["v_l2_rel_pct"] for r in sub],
            "v_full_loglog_slope": _loglog_slope(xs, [r["full"]["v_l2_rel_pct"] for r in sub]),
            "u_full_loglog_slope": _loglog_slope(xs, [r["full"]["u_l2_rel_pct"] for r in sub]),
            "v_interior_loglog_slope": _loglog_slope(
                xs, [r["interior"]["v_l2_rel_pct"] for r in sub]
            ),
            "outlet_share_of_v_err_pct": [
                100.0
                * (1.0 - (r["interior"]["v_l2_rel_pct"] / r["full"]["v_l2_rel_pct"]) ** 2)
                for r in sub
            ],
        }

    return {
        "case": "CASE-LBM-KOVASZNAY-EXACT",
        "threads": threads,
        "steps_target": steps,
        "runs": runs,
        "refinement_at_xmax3": refine,
        "order_of_accuracy": order,
        "xmax_scaling": scaling,
        "reference_certification": reference_check(),
    }


def ma_study(
    ny: int = 64,
    xmax: float = 3.0,
    u0_list=(0.03, 0.015, 0.0075),
    threads: int = 16,
    steps: int = STEPS_DEFAULT,
    dtype=torch.float64,
    cache_dir: Path | None = None,
) -> dict:
    """Mach refinement at fixed ``ny``: split the interior error into Ma^2 + h^2.

    The interior (outlet-free) L2 error is nearly independent of the grid
    (0.110 % at ny=32 vs 0.102 % at ny=64 for ``u``), which is the signature of
    a grid-independent error term rather than of first-order convergence.  The
    candidate is the ``O(Ma^2)`` compressibility of the weakly compressible LBM:
    at fixed grid and fixed Re, halving ``U0`` (hence ``nu``) must shrink a pure
    Ma^2 term by 4x while leaving the ``O(h^2)`` discretisation term unchanged.
    Fitting ``err^2 = (a Ma^2)^2 + (b h^2)^2`` on the observed ladder separates
    the two without any assumption about which dominates.
    """
    rows = []
    for u0 in u0_list:
        cpath = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cpath = cache_dir / f"ma_ny{ny}_u0{u0:g}.json"
        if cpath is not None and cpath.exists():
            rows.append(json.loads(cpath.read_text(encoding="utf-8")))
            continue
        rec = simulate(ny, xmax=xmax, u0=u0, steps=steps, threads=threads, dtype=dtype)
        rec["ma"] = rec["u0"] * math.sqrt(3.0)
        print(
            f"[ma] ny={ny} u0={u0} ma={rec['ma']:.4f} "
            f"u_int={rec['interior']['u_l2_rel_pct']:.4f}% "
            f"v_int={rec['interior']['v_l2_rel_pct']:.4f}% "
            f"u_full={rec['full']['u_l2_rel_pct']:.4f}% "
            f"v_full={rec['full']['v_l2_rel_pct']:.4f}%",
            flush=True,
        )
        if cpath is not None:
            cpath.write_text(json.dumps(rec, indent=2, default=float), encoding="utf-8")
        rows.append(rec)

    out = {
        "case": "CASE-LBM-KOVASZNAY-EXACT",
        "arm": "mach refinement (fixed ny, fixed Re, decreasing U0)",
        "ny": ny,
        "xmax": xmax,
        "rows": [
            {
                "u0": r["u0"],
                "ma": r["ma"],
                "tau": r["tau"],
                "nu": r["nu"],
                "u_interior_pct": r["interior"]["u_l2_rel_pct"],
                "v_interior_pct": r["interior"]["v_l2_rel_pct"],
                "u_full_pct": r["full"]["u_l2_rel_pct"],
                "v_full_pct": r["full"]["v_l2_rel_pct"],
                "u_interior_max_abs": r["interior"]["u_max_abs_over_uref"],
            }
            for r in rows
        ],
    }
    # consecutive-halving ratios: 4.0 => pure Ma^2 term, 1.0 => grid-limited floor
    for comp in ("u", "v"):
        key = f"{comp}_interior_pct"
        out[f"{comp}_halving_ratios"] = [
            (rows[i][ "interior" if False else "interior"][key] / rows[i + 1]["interior"][key])
            if rows[i + 1]["interior"][key] > 0
            else float("nan")
            for i in range(len(rows) - 1)
        ]
    return out


def _loglog_slope(xs: list[float], ys: list[float]) -> float:
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if y > 0 and x > 0]
    if len(pts) < 2:
        return float("nan")
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    return sxy / sxx if sxx > 0 else float("nan")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["single", "gate", "study", "check", "ma"])
    ap.add_argument("ny", nargs="?", type=int, default=128)
    ap.add_argument("out", nargs="?", type=Path, default=None)
    ap.add_argument("--xmax", type=float, default=3.0)
    ap.add_argument("--re", type=float, default=RE_DEFAULT)
    ap.add_argument("--u0", type=float, default=U0_DEFAULT)
    ap.add_argument("--steps", type=int, default=STEPS_DEFAULT)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    ap.add_argument("--cache", type=Path, default=None, help="per-run cache dir (resumable study)")
    ap.add_argument("--field-out", type=Path, default=None, help="save the measured/analytic fields (npz)")
    a = ap.parse_args(argv)
    dtype = torch.float64 if a.dtype == "float64" else torch.float32

    if a.cmd == "check":
        out = reference_check(ny=a.ny if a.ny > 4 else 64, re=a.re, u0=a.u0)
    elif a.cmd == "single":
        out = simulate(
            a.ny, re=a.re, u0=a.u0, xmax=a.xmax, steps=a.steps, threads=a.threads,
            dtype=dtype, field_out=a.field_out,
        )
    elif a.cmd == "gate":
        out = gate(ny=a.ny if a.ny >= 128 else 128, re=a.re, u0=a.u0, threads=a.threads, dtype=dtype)
    elif a.cmd == "ma":
        out = ma_study(
            ny=a.ny if a.ny >= 16 else 64,
            xmax=a.xmax,
            threads=a.threads,
            steps=a.steps,
            dtype=dtype,
            cache_dir=a.cache,
        )
    else:
        out = study(threads=a.threads, steps=a.steps, dtype=dtype, cache_dir=a.cache)

    text = json.dumps(out, indent=2, ensure_ascii=False, default=float)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
    print(text if a.cmd in ("check", "gate") else json.dumps(
        {k: v for k, v in out.items() if k != "runs"}, indent=2, ensure_ascii=False, default=float
    )[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())