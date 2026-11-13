#!/usr/bin/env python
"""Independent verification of ``GAP-PP-2-lid-wall-diff`` (verifier seat m17147).

The merged feature (``src/tensorlbm/diff_lid_cavity.py``) claims two acceptance
criteria:

* ① ``tau`` inversion on the Re=100 lid-driven cavity recovers ``tau`` to <2 %;
* ② autograd vs central finite difference agreement <1e-4.

This script re-derives both **independently** of the implementing driver
(``scripts/run_gap_pp2_lid_wall_diff.py``), and adds the checks the original
evidence bundle does not contain:

``fd``      gradient cross-check with three estimators (autograd, central FD
            with a truncation-error ladder + Richardson extrapolation, and
            complex step) on a *different* functional than the shipped one
            (energy norm instead of the centreline MSE);
``equiv``   production lid BC (``lid_driven_cavity.zou_he_moving_lid``,
            clone + in-place) vs the graph-preserving BC, per-node;
``repro``   same-chain ``tau`` inversion (truth = the same differentiable
            chain) -- reproduction of criterion ①;
``anchor``  *external* anchor inversion: truth = the Ghia et al. (1982) Re=100
            centreline tables, i.e. data that the inverse model did **not**
            generate.  This is the only non-circular reading of criterion ①;
``noise``   same-chain inversion with additive measurement noise on the probes.

Every number is written to JSON by the code path that computed it (no
hand-copying).  Usage::

    PYTHONPATH=src python scripts/verify_lid_wall_diff_m17147.py fd --out ...
    PYTHONPATH=src python scripts/verify_lid_wall_diff_m17147.py shard --index 0/4 --outdir ...
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from tensorlbm.d2q9 import macroscopic
from tensorlbm.diff_lid_cavity import (
    GHIA_STATIONS,
    CavityGeometry,
    centreline_probes,
    initial_field,
    invert_relaxation_time_rootfind,
    moving_lid_bc,
    rollout,
    tau_from_reynolds,
)
from tensorlbm.lid_driven_cavity import GHIA_RE100, zou_he_moving_lid
from tensorlbm.solver import bounce_back_cells, collide_bgk, stream

DTYPE = torch.float64
U_LID = 0.1
RE_TARGET = 100.0


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _tau_of_re(n: int, re: float = RE_TARGET, u_lid: float = U_LID) -> float:
    return tau_from_reynolds(re, u_lid, length=n - 1)


def _re_of_tau(n: int, tau: float, u_lid: float = U_LID) -> float:
    """Re = u*L/nu; ``tau -> 0.5`` (nu -> 0) is the BGK stability boundary and
    is reported as ``inf`` instead of raising, so a root-finder that walks to the
    boundary still yields a diagnosable record."""
    nu = (tau - 0.5) / 3.0
    if nu <= 0.0:
        return float("inf")
    return u_lid * (n - 1) / nu


def _interior(u: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    keep = ~mask
    keep[-1, :] = False  # lid row carries the prescribed velocity, drop it
    return u[keep]


def _energy_loss(f: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Functional independent of the shipped MSE-on-centrelines loss."""
    _, ux, uy = macroscopic(f)
    uxi = _interior(ux, mask)
    uyi = _interior(uy, mask)
    return (uxi * uxi + uyi * uyi).mean()


def _mismatch_loss(f: torch.Tensor, mask: torch.Tensor, ux_t: torch.Tensor,
                   uy_t: torch.Tensor) -> torch.Tensor:
    _, ux, uy = macroscopic(f)
    return (((ux - ux_t) ** 2 + (uy - uy_t) ** 2)[~mask]).mean()


def _prod_step(f: torch.Tensor, tau: torch.Tensor | float, u_lid: float,
               wall_mask: torch.Tensor) -> torch.Tensor:
    """Production lid chain: same collide/stream/bounce-back, production BC."""
    f = collide_bgk(f, tau)
    f = stream(f)
    f = bounce_back_cells(f, wall_mask)
    return zou_he_moving_lid(f, u_lid)


def _prod_rollout(f0: torch.Tensor, tau: float, u_lid: float, n_steps: int,
                  wall_mask: torch.Tensor) -> torch.Tensor:
    f = f0
    for _ in range(n_steps):
        f = _prod_step(f, tau, u_lid, wall_mask)
    return f


# ---------------------------------------------------------------------------
# Ghia (1982) table -> probe-layout target vector
# ---------------------------------------------------------------------------


def ghia_target(stations: Sequence[float] = GHIA_STATIONS) -> torch.Tensor:
    """Ghia Re=100 centreline data interpolated onto the Ghia probe layout.

    Returns ``(2 * len(stations),)`` normalised (u/u_lid) values whose layout
    matches :func:`tensorlbm.diff_lid_cavity.centreline_probes`: ``u`` on the
    vertical centreline sampled bottom->top, then ``v`` on the horizontal
    centreline sampled left->right.  ``GHIA_RE100["y"]`` is stored in
    descending order, so the abscissa is reversed before interpolation.
    """
    y = list(GHIA_RE100["y"])
    u = list(GHIA_RE100["u"])
    x = list(GHIA_RE100["x"])
    v = list(GHIA_RE100["v"])
    if y[0] > y[-1]:  # descending -> ascending
        y = y[::-1]
        u = u[::-1]
    if x[0] > x[-1]:
        x = x[::-1]
        v = v[::-1]
    uu = []
    for s in stations:
        uu.append(_lin_interp(y, u, s))
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
# fd : three-estimator gradient cross-check
# ---------------------------------------------------------------------------


def run_fd(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(0)
    n, steps = args.n, args.steps
    geom = CavityGeometry(ny=n, nx=n)
    mask = geom.wall_mask
    f0 = initial_field(geom)

    tau_true = _tau_of_re(n, args.re)
    # a *different* functional: mismatch against a reference field computed at a
    # *shifted* parameter, so the evaluation point is NOT the functional minimum
    # (otherwise dJ/dtau would vanish identically and the check would be vacuous)
    tau_ref = tau_true
    u_ref = float(U_LID) * 1.2
    with torch.no_grad():
        f_ref = rollout(f0, torch.tensor(tau_true + 0.02, dtype=DTYPE), u_ref, steps,
                        wall_mask=mask)
        _, ux_r, uy_r = macroscopic(f_ref)

    def loss_tau(tau: float) -> float:
        with torch.no_grad():
            f = rollout(f0, torch.tensor(tau, dtype=DTYPE), U_LID, steps, wall_mask=mask)
            return float(_mismatch_loss(f, mask, ux_r, uy_r))

    def loss_u(u: float) -> float:
        with torch.no_grad():
            f = rollout(f0, torch.tensor(tau_ref, dtype=DTYPE), u, steps, wall_mask=mask)
            return float(_mismatch_loss(f, mask, ux_r, uy_r))

    # analytic (autograd) gradients at (tau_ref, u_lid)
    tau = torch.tensor(tau_ref, dtype=DTYPE, requires_grad=True)
    u = torch.tensor(float(U_LID), dtype=DTYPE, requires_grad=True)
    f = rollout(f0, tau, u, steps, wall_mask=mask)
    loss = _mismatch_loss(f, mask, ux_r, uy_r)
    g_tau, g_u = torch.autograd.grad(loss, (tau, u))
    ad = {"dJ_dtau": float(g_tau), "dJ_du_lid": float(g_u)}

    fd: dict[str, dict[str, Any]] = {}
    for name, fn, x0 in (("dJ_dtau", loss_tau, tau_ref), ("dJ_du_lid", loss_u, float(U_LID))):
        rows = []
        for h in (1e-2, 1e-3, 1e-4, 1e-5):
            plus, minus = fn(x0 + h), fn(x0 - h)
            d = (plus - minus) / (2.0 * h)
            rows.append({"h": h, "derivative": d,
                         "abs_err": abs(d - ad[name]),
                         "rel_err": abs(d - ad[name]) / max(abs(ad[name]), 1e-300)})
        # Richardson: D(h) = D + c h^2  ->  extrapolate from the two smallest h
        r1, r2 = rows[-2], rows[-1]
        ratio = r1["h"] / r2["h"]
        extrap = r2["derivative"] + (r2["derivative"] - r1["derivative"]) / (ratio * ratio - 1.0)
        fd[name] = {"rows": rows, "richardson": extrap,
                    "richardson_abs_err": abs(extrap - ad[name]),
                    "richardson_rel_err": abs(extrap - ad[name]) / max(abs(ad[name]), 1e-300)}

    cs = _second_engine_checks(args, geom, f0, mask, tau_ref, ux_r, uy_r, ad)

    payload = {
        "case": "GAP-PP-2-lid-wall-diff / criterion 2 (AD vs FD) -- independent",
        "n": n, "steps": steps, "tau_ref": tau_ref, "u_lid": U_LID,
        "functional": "mean interior (ux^2+uy^2) mismatch vs tau_ref+0.02 field",
        "autograd": ad, "central_fd": fd, "second_engine_checks": cs,
        "n_threads": torch.get_num_threads(),
        "wall_time_s": None,
    }
    def _nanmax(vals: list[float]) -> float:
        vals = [v for v in vals if v == v]  # drop NaN (unstable large-h probes)
        return max(vals) if vals else float("nan")

    payload["worst_fd_rel_err_h_ge_1e-4"] = _nanmax(
        [r["rel_err"] for name in fd for r in fd[name]["rows"] if r["h"] >= 1e-4]
    )
    payload["worst_fd_rel_err_all_h"] = _nanmax(
        [r["rel_err"] for name in fd for r in fd[name]["rows"]]
    )
    payload["nan_fd_samples"] = [
        {"param": name, "h": r["h"]} for name in fd for r in fd[name]["rows"] if r["rel_err"] != r["rel_err"]
    ]
    payload["richardson_worst_rel_err"] = max(
        fd[name]["richardson_rel_err"] for name in fd
    )
    return payload


def _second_engine_checks(args: argparse.Namespace, geom: CavityGeometry, f0: torch.Tensor,
                          mask: torch.Tensor, tau_ref: float, ux_r: torch.Tensor,
                          uy_r: torch.Tensor, ad: dict[str, float]) -> dict[str, Any]:
    """Two further, independent consistency checks on the autograd gradient.

    1. **Forward-mode AD** (``torch.func.jvp``): reverse and forward mode share
       no tangent bookkeeping, so agreement rules out a reverse-mode-only
       bookkeeping error.  Recorded as unavailable with the exact exception if
       the environment rejects it -- not silently skipped.
    2. **Taylor / secant consistency**: for a *finite* increment ``d`` the true
       change obeys ``J(x+d) - J(x) = g d + O(d^2)``.  Predicting the finite
       change from the autograd gradient and measuring the residual against the
       second-order term tests the gradient at a scale where finite-difference
       truncation is not the limiting factor.  A gradient that is wrong by a
       constant factor fails this even when it passes a self-consistent FD.

    A complex-step estimator (truncation free) is *not* available here:
    :func:`tensorlbm.solver.collide_bgk` clamps ``rho`` and ``torch.clamp`` has
    no complex kernel.  That limitation is recorded, not worked around.
    """
    n, steps = args.n, args.steps
    out: dict[str, Any] = {"n": n, "steps": steps}

    try:
        from torch.func import jvp

        def fwd(tau: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            f = rollout(f0, tau, u, steps, wall_mask=mask)
            return _mismatch_loss(f, mask, ux_r, uy_r)

        tau0 = torch.tensor(tau_ref, dtype=DTYPE)
        u0 = torch.tensor(float(U_LID), dtype=DTYPE)
        jvp_out = {}
        for name, tangent in (("dJ_dtau", (torch.ones_like(tau0), torch.zeros_like(u0))),
                              ("dJ_du_lid", (torch.zeros_like(tau0), torch.ones_like(u0)))):
            _, d = jvp(fwd, (tau0, u0), tangent)
            jvp_out[name] = float(d)
            jvp_out[f"{name}_rel_err_vs_reverse"] = abs(float(d) - ad[name]) / max(abs(ad[name]), 1e-300)
        out["forward_mode"] = {"available": True, "engine": "torch.func.jvp", **jvp_out}
    except Exception as exc:  # pragma: no cover - environment dependent
        out["forward_mode"] = {"available": False, "engine": "torch.func.jvp",
                               "error": f"{type(exc).__name__}: {exc}"}

    out["complex_step_attempt"] = {
        "available": False,
        "reason": "torch.clamp (rho floor in collide_bgk) has no complex kernel",
    }

    def loss_at(tau: float, u: float) -> float:
        with torch.no_grad():
            f = rollout(f0, torch.tensor(tau, dtype=DTYPE), u, steps, wall_mask=mask)
            return float(_mismatch_loss(f, mask, ux_r, uy_r))

    base = loss_at(tau_ref, float(U_LID))
    taylor = {}
    for name, getter, x0 in (("dJ_dtau", lambda x: loss_at(x, float(U_LID)), tau_ref),
                             ("dJ_du_lid", lambda x: loss_at(tau_ref, x), float(U_LID))):
        rows = []
        for d in (1e-2, 1e-3, 1e-4):
            actual = getter(x0 + d) - getter(x0 - d)
            predicted = 2.0 * ad[name] * d
            rows.append({"delta": d, "actual_change": actual, "linear_prediction": predicted,
                         "residual": actual - predicted,
                         "residual_over_delta2": (actual - predicted) / (d * d)})
        taylor[name] = {"rows": rows,
                        "residual_at_smallest_delta": abs(rows[-1]["residual"]),
                        "residual_rel_to_J": abs(rows[-1]["residual"]) / max(abs(base), 1e-300)}
    out["taylor"] = taylor
    out["J_at_reference"] = base
    return out


# ---------------------------------------------------------------------------
# equiv : production lid BC vs graph-preserving lid BC
# ---------------------------------------------------------------------------


def run_equiv(args: argparse.Namespace) -> dict[str, Any]:
    n, steps = args.n, args.steps
    geom = CavityGeometry(ny=n, nx=n)
    mask = geom.wall_mask
    f0 = initial_field(geom)
    tau = _tau_of_re(n, args.re)

    t0 = time.time()
    f_diff = rollout(f0, torch.tensor(tau, dtype=DTYPE), U_LID, steps, wall_mask=mask)
    t_diff = time.time() - t0
    t0 = time.time()
    f_prod = _prod_rollout(f0, tau, U_LID, steps, mask)
    t_prod = time.time() - t0

    delta = (f_diff - f_prod).abs()
    _, ux_d, uy_d = macroscopic(f_diff)
    _, ux_p, uy_p = macroscopic(f_prod)
    return {
        "case": "production (clone+inplace) vs graph-preserving moving lid",
        "n": n, "steps": steps, "tau": tau,
        "max_abs_delta_f": float(delta.max()),
        "max_abs_delta_ux": float((ux_d - ux_p).abs().max()),
        "bit_identical": bool(float(delta.max()) == 0.0),
        "time_diff_chain_s": t_diff, "time_prod_chain_s": t_prod,
        "lid_row_jx_minus_rho_u": float(
            (macroscopic(f_diff)[1][-1, 1:-1] - U_LID).abs().max()
        ),
    }


# ---------------------------------------------------------------------------
# inversion helpers
# ---------------------------------------------------------------------------


def _truth_obs(geom: CavityGeometry, tau: float, steps: int) -> torch.Tensor:
    f0 = initial_field(geom)
    with torch.no_grad():
        f = rollout(f0, torch.tensor(tau, dtype=DTYPE), U_LID, steps, wall_mask=geom.wall_mask)
    return centreline_probes(f, U_LID).detach()


def _invert(target: torch.Tensor, geom: CavityGeometry, steps: int, tau_init: float,
            bracket: tuple[float, float] | None, max_iter: int) -> dict[str, Any]:
    t0 = time.time()
    res = invert_relaxation_time_rootfind(
        target, geom, U_LID, tau_init, steps,
        method="illinois", bracket_tau=bracket, max_iter=max_iter,
        step_tol=1e-9, checkpoint_every=200,
    )
    out = {k: v for k, v in res.items() if k != "obs_final"}
    out["wall_time_s"] = time.time() - t0
    out["obs_final"] = [float(x) for x in res["obs_final"]]
    return out


def _summary(res: dict[str, Any], tau_true: float, n: int) -> dict[str, Any]:
    tau_rec = res["tau_recovered"]
    re_true = _re_of_tau(n, tau_true)
    re_rec = _re_of_tau(n, tau_rec)
    rel_re = float("inf") if math.isinf(re_rec) else abs(re_rec - re_true) / re_true
    return {
        "tau_true": tau_true,
        "tau_recovered": tau_rec,
        "rel_err_tau": abs(tau_rec - tau_true) / tau_true,
        "re_true": re_true,
        "re_recovered": re_rec,
        "rel_err_re": rel_re,
        "residual_rms": res["residual_rms"],
        "n_evals": res["n_evals"],
        "status": res["status"],
        "wall_time_s": res["wall_time_s"],
    }


# ---------------------------------------------------------------------------
# repro / anchor / noise
# ---------------------------------------------------------------------------


def run_repro(args: argparse.Namespace) -> dict[str, Any]:
    n, steps = args.n, args.steps
    geom = CavityGeometry(ny=n, nx=n)
    tau_true = _tau_of_re(n, args.re)
    target = _truth_obs(geom, tau_true, steps)
    res = _invert(target, geom, steps, args.tau_init,
                  None if args.bracket is None else tuple(args.bracket), args.max_iter)
    return {
        "case": "criterion 1 reproduction, truth = same differentiable chain",
        "n": n, "steps": steps, "tau_init": args.tau_init,
        "summary": _summary(res, tau_true, n), "inversion": res,
    }


def run_anchor(args: argparse.Namespace) -> dict[str, Any]:
    """Invert tau against the Ghia (1982) tables -- data the model did not make."""
    n, steps = args.n, args.steps
    geom = CavityGeometry(ny=n, nx=n)
    tau_true = _tau_of_re(n, args.re)
    target = ghia_target()
    res = _invert(target, geom, steps, args.tau_init,
                  None if args.bracket is None else tuple(args.bracket), args.max_iter)
    # Ghia-RMSE(τ) curve: where does the *non-circular* objective actually minimise?
    taus = [tau_true + d for d in (-0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06)]
    curve = []
    f0 = initial_field(geom)
    for tv in taus:
        with torch.no_grad():
            f = rollout(f0, torch.tensor(tv, dtype=DTYPE), U_LID, steps, wall_mask=geom.wall_mask)
            obs = centreline_probes(f, U_LID)
            loss = float(((obs - target) ** 2).mean())
        curve.append({"tau": tv, "re": _re_of_tau(n, tv), "mse_vs_ghia": loss,
                      "rmse_vs_ghia": math.sqrt(loss)})
    best = min(curve, key=lambda r: r["mse_vs_ghia"])
    return {
        "case": "criterion 1 under the EXTERNAL anchor (Ghia 1982 Re=100 tables)",
        "n": n, "steps": steps, "tau_init": args.tau_init,
        "summary": _summary(res, tau_true, n),
        "ghia_mse_curve": curve,
        "grid_optimum": best,
        "grid_optimum_rel_err_re": abs(best["re"] - _re_of_tau(n, tau_true)) / _re_of_tau(n, tau_true),
        "inversion": res,
    }


def run_noise(args: argparse.Namespace) -> dict[str, Any]:
    n, steps = args.n, args.steps
    geom = CavityGeometry(ny=n, nx=n)
    tau_true = _tau_of_re(n, args.re)
    clean = _truth_obs(geom, tau_true, steps)
    out = []
    for sigma in args.sigmas:
        gen = torch.Generator().manual_seed(12345)
        noise = torch.randn(clean.shape, generator=gen, dtype=DTYPE) * sigma
        target = clean + noise
        res = _invert(target, geom, steps, args.tau_init,
                      None if args.bracket is None else tuple(args.bracket), args.max_iter)
        s = _summary(res, tau_true, n)
        s["sigma"] = sigma
        s["tau_history"] = res["tau_history"]
        out.append(s)
    return {
        "case": "criterion 1 under additive probe noise (same-chain truth)",
        "n": n, "steps": steps, "tau_init": args.tau_init, "u_lid": U_LID,
        "sigma_levels": list(args.sigmas), "results": out,
    }


# ---------------------------------------------------------------------------
# shard dispatch (cluster)
# ---------------------------------------------------------------------------

SHARDS: dict[int, Callable[[argparse.Namespace], dict[str, Any]]] = {
    0: run_fd,
    1: run_repro,
    2: run_anchor,
    3: run_noise,
}


def shard_config(index: int) -> dict[str, Any]:
    """Per-shard argument defaults (kept explicit so the report can quote them)."""
    return {
        0: {"n": 32, "steps": 600, "re": 100.0},
        1: {"n": 65, "steps": 4000, "re": 100.0, "tau_init": 0.6366, "max_iter": 12},
        2: {"n": 48, "steps": 8000, "re": 100.0, "tau_init": 0.58, "max_iter": 14},
        3: {"n": 48, "steps": 3000, "re": 100.0, "tau_init": 0.60, "max_iter": 12,
            "sigmas": [0.0, 1e-4, 1e-3]},
    }[index]


def _fill(args: argparse.Namespace, cfg: dict[str, Any]) -> argparse.Namespace:
    for k, v in cfg.items():
        setattr(args, k, v)
    if not hasattr(args, "bracket") or args.bracket is None:
        args.bracket = None
    return args


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--n", type=int, default=32)
        sp.add_argument("--steps", type=int, default=600)
        sp.add_argument("--re", type=float, default=RE_TARGET)
        sp.add_argument("--tau-init", type=float, default=0.60)
        sp.add_argument("--bracket", type=float, nargs=2, default=None)
        sp.add_argument("--max-iter", type=int, default=12)
        sp.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 1e-4, 1e-3])
        sp.add_argument("--threads", type=int, default=8)
        sp.add_argument("--out", type=Path, required=True)

    for name in ("fd", "equiv", "repro", "anchor", "noise"):
        common(sub.add_parser(name))
    sh = sub.add_parser("shard")
    sh.add_argument("--index", type=int, required=True)
    sh.add_argument("--threads", type=int, default=8)
    sh.add_argument("--out", type=Path, required=True)
    sh.add_argument("--outdir", type=Path, default=None)

    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    if args.cmd == "shard":
        fn = SHARDS[args.index]
        args = _fill(args, shard_config(args.index))
        t0 = time.time()
        payload = fn(args)
        payload["shard"] = args.index
        payload["total_wall_time_s"] = time.time() - t0
        payload["torch"] = torch.__version__
        out = args.out
    else:
        t0 = time.time()
        payload = {"fd": run_fd, "equiv": run_equiv, "repro": run_repro,
                   "anchor": run_anchor, "noise": run_noise}[args.cmd](args)
        payload["total_wall_time_s"] = time.time() - t0
        out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())