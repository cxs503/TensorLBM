#!/usr/bin/env python
"""GAP-PP-2 (lid-wall-diff) driver: differentiable moving-wall lid BC evidence.

Produces the acceptance artifacts of the ``GAP-PP-2-lid-wall-diff`` anchor
(graph-preserving moving-wall lid BC for the 2D D2Q9 lid-driven cavity, used for
solver-in-the-loop ``tau`` inversion):

  steady    -- Re=100 cavity, N nodes, Ghia (1982) centreline comparison
  sweep     -- tau sensitivity of the Ghia residual (identifiability scan)
  fdcheck   -- autograd gradient vs central finite difference (d/dtau, d/du_lid)
  invert    -- tau inversion through the full time stepping (truth = same chain
               or the Ghia Re=100 tables) and its recovery error, Adam/Newton
  invert-rf -- same inverse problem solved as the scalar root-finding problem
               dL/dtau = 0 (secant / Illinois regula falsi, round-2 method)

All stages write JSON (numbers, zero hand-copying) and checkpoints into
``artifacts/gap_pp2_lid_wall_diff/``; ``figures`` turns them into the report
figures.  The module under test is :mod:`tensorlbm.diff_lid_cavity`.

Example (repro_cmd)::

    PYTHONPATH=src python scripts/run_gap_pp2_lid_wall_diff.py all \
        --artifacts artifacts/gap_pp2_lid_wall_diff
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tensorlbm.d2q9 import macroscopic
from tensorlbm.lid_driven_cavity import zou_he_moving_lid
from tensorlbm.diff_lid_cavity import (
    GHIA_STATIONS,
    CavityGeometry,
    centreline_probes,
    initial_field,
    invert_relaxation_time,
    invert_relaxation_time_rootfind,
    moving_lid_bc,
    rollout,
    step,
    tau_from_reynolds,
)
from tensorlbm.lid_driven_cavity import GHIA_RE100, compare_ghia

RE_TARGET = 100.0
U_LID = 0.1
DTYPE = torch.float64


def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def tau_target(n: int, re: float = RE_TARGET, u_lid: float = U_LID) -> float:
    """``tau`` for ``Re`` on an ``n``-node cavity (side ``H = n - 1``)."""
    return tau_from_reynolds(re, u_lid, length=n - 1)


def _fields(f: torch.Tensor, wall_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, ux, uy = macroscopic(f)
    return ux.masked_fill(wall_mask, 0.0), uy.masked_fill(wall_mask, 0.0)


def _ghia_rmse(f: torch.Tensor, wall_mask: torch.Tensor, u_lid: float = U_LID) -> dict[str, float]:
    ux, uy = _fields(f, wall_mask)
    return compare_ghia(ux, uy, u_lid, GHIA_RE100)


def _ghia_profile_at_stations() -> torch.Tensor:
    """Ghia Re=100 centreline data interpolated onto :data:`GHIA_STATIONS`."""
    import numpy as np

    y = np.asarray(GHIA_RE100["y"])
    u = np.asarray(GHIA_RE100["u"])
    x = np.asarray(GHIA_RE100["x"])
    v = np.asarray(GHIA_RE100["v"])
    sta = np.asarray(GHIA_STATIONS)
    # Ghia's stations are ordered from the far wall; sort for np.interp.
    u_interp = np.interp(sta, y[::-1], u[::-1])
    v_interp = np.interp(sta, x, v)
    return torch.as_tensor(np.concatenate([u_interp, v_interp]), dtype=DTYPE)


def _steady_run(
    n: int,
    steps: int,
    tau: float,
    dtype: torch.dtype = DTYPE,
    log_every: int = 5000,
    compile_step: bool = True,
    u_lid: float = U_LID,
) -> dict:
    geom = CavityGeometry(n, n, dtype=dtype)
    mask = geom.wall_mask
    f = initial_field(geom)
    t0 = time.time()
    history = []
    u = torch.as_tensor(u_lid, dtype=dtype)
    with torch.no_grad():
        for k in range(1, steps + 1):
            f = rollout(f, tau, u, 1, wall_mask=mask, compile_step=compile_step)
            if k % log_every == 0 or k == steps:
                err = _ghia_rmse(f.double(), mask, u_lid)
                history.append({"step": k, "rmse_u": err["rmse_u"], "rmse_v": err["rmse_v"]})
                _log(f"  step={k:6d} rmse_u={err['rmse_u']:.5f} rmse_v={err['rmse_v']:.5f}")
    ux, uy = _fields(f, mask)
    err = _ghia_rmse(f.double(), mask, u_lid)
    rho, _, _ = macroscopic(f)
    return {
        "n": n,
        "steps": steps,
        "tau": tau,
        "nu": (tau - 0.5) / 3.0,
        "re": u_lid * (n - 1) / ((tau - 0.5) / 3.0),
        "u_lid": u_lid,
        "max_abs_err_u": _max_abs_ghia_err(f.double(), mask, u_lid),
        "dtype": str(dtype).replace("torch.", ""),
        "rmse_u": err["rmse_u"],
        "rmse_v": err["rmse_v"],
        "mass": float(rho.double().sum()),
        "mass_drift": float(rho.double().sum()) - float(n * n),
        "wall_time_s": time.time() - t0,
        "history": history,
        "ux": ux.double().cpu(),
        "uy": uy.double().cpu(),
        "probes": centreline_probes(f.double(), None).cpu(),
    }


def _max_abs_ghia_err(f: torch.Tensor, wall_mask: torch.Tensor, u_lid: float) -> float:
    """Max |u_LBM - u_Ghia| / u_lid over the Ghia vertical-centreline stations."""
    import numpy as np

    ux, _ = _fields(f, wall_mask)
    n = ux.shape[0]
    xc = n // 2
    ys = np.asarray(GHIA_RE100["y"])
    ref = np.asarray(GHIA_RE100["u"])
    prof = ux[:, xc].double().cpu().numpy() / u_lid
    got = np.interp(ys, np.linspace(0.0, 1.0, n), prof)
    return float(np.max(np.abs(got - ref)))


def _save_steady(result: dict, out: Path) -> None:
    payload = {k: v for k, v in result.items() if k not in ("ux", "uy", "probes")}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"ux": result["ux"], "uy": result["uy"], "probes": result["probes"]},
        out.with_suffix(".pt"),
    )
    out.write_text(json.dumps(payload, indent=2) + "\n")
    _log(f"wrote {out} and {out.with_suffix('.pt')}")


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def stage_steady(args: argparse.Namespace) -> None:
    n = args.n
    tau = args.tau if args.tau is not None else tau_target(n)
    _log(f"[steady] n={n} tau={tau:.6f} steps={args.steps}")
    result = _steady_run(
        n, args.steps, tau, log_every=max(1000, args.steps // 8), u_lid=args.u_lid
    )
    _save_steady(result, Path(args.out))


def stage_sweep(args: argparse.Namespace) -> None:
    n = args.n
    tau0 = tau_target(n)
    rows = []
    for frac in args.fracs:
        tau = tau0 * (1.0 + frac)
        _log(f"[sweep] frac={frac:+.3f} tau={tau:.6f}")
        res = _steady_run(n, args.steps, tau, log_every=max(1000, args.steps), u_lid=args.u_lid)
        rows.append(
            {
                "frac": frac,
                "tau": tau,
                "nu": (tau - 0.5) / 3.0,
                "re": U_LID * (n - 1) / ((tau - 0.5) / 3.0),
                "rmse_u": res["rmse_u"],
                "rmse_v": res["rmse_v"],
                "probes": res["probes"].tolist(),
            }
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"n": n, "steps": args.steps, "tau_target": tau0, "rows": rows}, indent=2) + "\n"
    )
    _log(f"wrote {out}")


def _observable(
    f: torch.Tensor,
    u_lid: torch.Tensor,
    stations=GHIA_STATIONS,
) -> torch.Tensor:
    """Scalar loss-like observable: squared norm of the normalised centreline data."""
    obs = centreline_probes(f, u_lid, stations=stations)
    return (obs**2).sum()


def stage_fdcheck(args: argparse.Namespace) -> None:
    """Autograd vs central finite difference on the *same* fixed-step map."""
    n, steps = args.n, args.steps
    geom = CavityGeometry(n, n, dtype=DTYPE)
    mask = geom.wall_mask
    tau0 = tau_target(n)
    u0 = U_LID
    f0 = initial_field(geom)

    def observable_map(tau, u_lid):
        f = rollout(f0, tau, u_lid, steps, wall_mask=mask)
        return _observable(f, u_lid)

    rows = {}
    for name, param in (("tau", tau0), ("u_lid", u0)):
        if name == "tau":
            tau_t = torch.tensor(param, dtype=DTYPE, requires_grad=True)
            u_t = torch.tensor(u0, dtype=DTYPE)
        else:
            tau_t = torch.tensor(tau0, dtype=DTYPE)
            u_t = torch.tensor(param, dtype=DTYPE, requires_grad=True)
        val = observable_map(tau_t, u_t)
        (grad,) = torch.autograd.grad(val, tau_t if name == "tau" else u_t)
        ad = float(grad)
        fds = []
        for h in args.hs:
            with torch.no_grad():
                if name == "tau":
                    hpt = torch.tensor(param + h, dtype=DTYPE)
                    hmt = torch.tensor(param - h, dtype=DTYPE)
                    jp = float(observable_map(hpt, u_t))
                    jm = float(observable_map(hmt, u_t))
                else:
                    hpt = torch.tensor(param + h, dtype=DTYPE)
                    hmt = torch.tensor(param - h, dtype=DTYPE)
                    jp = float(observable_map(tau_t, hpt))
                    jm = float(observable_map(tau_t, hmt))
            fd = (jp - jm) / (2.0 * h)
            rel = abs(ad - fd) / max(abs(fd), 1e-300)
            fds.append(
                {"h": h, "fd": fd, "ad": ad, "rel_err": rel, "j_plus": jp, "j_minus": jm}
            )
            _log(f"[fdcheck] {name}: h={h:.0e} ad={ad:.10e} fd={fd:.10e} rel={rel:.3e}")
        rows[name] = {"param": param, "ad": ad, "checks": fds}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "n": n,
                "steps": steps,
                "u_lid": U_LID,
                "tau": tau0,
                "gradients": rows,
                "tolerance_rel": 1e-4,
                "all_pass": all(
                    c["rel_err"] < 1e-4 for r in rows.values() for c in r["checks"]
                ),
            },
            indent=2,
        )
        + "\n"
    )
    _log(f"wrote {out}")


def stage_invert(args: argparse.Namespace) -> None:
    n, steps = args.n, args.steps
    tau_true = args.tau_true if args.tau_true is not None else tau_target(n)
    geom = CavityGeometry(n, n, dtype=DTYPE)
    mask = geom.wall_mask
    f0 = initial_field(geom)
    u_t = torch.as_tensor(U_LID, dtype=DTYPE)

    if args.truth == "ghia":
        target = _ghia_profile_at_stations()
        truth_source = "Ghia et al. (1982) Re=100 centreline tables (JCP 48:387)"
    else:
        with torch.no_grad():
            f_truth = rollout(
                f0, tau_true, u_t, steps, wall_mask=mask, compile_step=True
            )
            target = centreline_probes(f_truth, u_t).detach()
        truth_source = f"same differentiable chain, n={n}, steps={steps}, tau={tau_true}"

    _log(
        f"[invert] truth={args.truth} n={n} steps={steps} tau_true={tau_true:.6f} "
        f"tau_init={args.tau_init:.6f}"
    )
    t0 = time.time()
    res = invert_relaxation_time(
        target,
        geom,
        U_LID,
        args.tau_init,
        steps,
        iters=args.iters,
        lr=args.lr,
        checkpoint_every=args.checkpoint_every,
        log=_log,
        compile_step=True,
        loss_tol=args.loss_tol,
    )
    tau_rec = res["tau_recovered"]
    payload = {
        "truth": args.truth,
        "truth_source": truth_source,
        "n": n,
        "steps": steps,
        "u_lid": U_LID,
        "tau_true": tau_true,
        "tau_init": args.tau_init,
        "tau_recovered": tau_rec,
        "tau_best": res["tau_best"],
        "loss_best": res["loss_best"],
        "rel_err_best": abs(res["tau_best"] - tau_true) / tau_true,
        "rel_err_best_percent": 100.0 * abs(res["tau_best"] - tau_true) / tau_true,
        "abs_err": abs(tau_rec - tau_true),
        "rel_err": abs(tau_rec - tau_true) / tau_true,
        "rel_err_percent": 100.0 * abs(tau_rec - tau_true) / tau_true,
        "nu_true": (tau_true - 0.5) / 3.0,
        "nu_recovered": (tau_rec - 0.5) / 3.0,
        "re_true": U_LID * (n - 1) / ((tau_true - 0.5) / 3.0),
        "re_recovered": U_LID * (n - 1) / ((tau_rec - 0.5) / 3.0),
        "method": res.get("method", "adam"),
        "final_step": res.get("final_step", float("nan")),
        "tau_history": res["tau_history"],
        "loss_history": res["loss_history"],
        "residual_rms": res["residual_rms"],
        "iters": args.iters,
        "lr": args.lr,
        "tolerance_rel": 0.02,
        "pass": bool(abs(tau_rec - tau_true) / tau_true < 0.02),
        "wall_time_s": time.time() - t0,
        "target": target.tolist(),
        "obs_final": res["obs_final"].tolist(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    _log(
        f"[invert] tau_rec={tau_rec:.6f} rel_err={payload['rel_err_percent']:.4f}% "
        f"pass={payload['pass']} ({payload['wall_time_s']:.1f}s)"
    )
    _log(f"wrote {out}")

def stage_invert_rf(args: argparse.Namespace) -> None:
    """Round-2 inversion: root-find dL/ds = 0 instead of descending L."""
    n, steps = args.n, args.steps
    tau_true = args.tau_true if args.tau_true is not None else tau_target(n)
    geom = CavityGeometry(n, n, dtype=DTYPE)
    mask = geom.wall_mask
    f0 = initial_field(geom)
    u_t = torch.as_tensor(U_LID, dtype=DTYPE)

    if args.truth == "ghia":
        target = _ghia_profile_at_stations()
        truth_source = "Ghia et al. (1982) Re=100 centreline tables (JCP 48:387)"
    else:
        with torch.no_grad():
            f_truth = rollout(
                f0, tau_true, u_t, steps, wall_mask=mask, compile_step=True
            )
            target = centreline_probes(f_truth, u_t).detach()
        truth_source = f"same differentiable chain, n={n}, steps={steps}, tau={tau_true}"

    bracket_tau = tuple(args.bracket) if args.bracket else None
    _log(
        f"[invert-rf] truth={args.truth} n={n} steps={steps} tau_true={tau_true:.6f} "
        f"tau_init={args.tau_init:.6f} method={args.method} bracket={bracket_tau}"
    )
    t0 = time.time()
    res = invert_relaxation_time_rootfind(
        target,
        geom,
        U_LID,
        args.tau_init,
        steps,
        method=args.method,
        bracket_tau=bracket_tau,
        bracket_frac=args.bracket_frac,
        max_iter=args.max_iter,
        step_tol=args.step_tol,
        checkpoint_every=args.checkpoint_every,
        compile_step=args.compile_step,
        log=_log,
    )
    tau_rec = res["tau_recovered"]
    payload = {
        "truth": args.truth,
        "truth_source": truth_source,
        "n": n,
        "steps": steps,
        "u_lid": U_LID,
        "tau_true": tau_true,
        "tau_init": args.tau_init,
        "tau_recovered": tau_rec,
        "tau_best": res["tau_best"],
        "loss_best": res["loss_best"],
        "loss_root": res["loss_root"],
        "grad_root": res["grad_root"],
        "rel_err_best": abs(res["tau_best"] - tau_true) / tau_true,
        "rel_err": abs(tau_rec - tau_true) / tau_true,
        "rel_err_percent": 100.0 * abs(tau_rec - tau_true) / tau_true,
        "abs_err": abs(tau_rec - tau_true),
        "nu_true": (tau_true - 0.5) / 3.0,
        "nu_recovered": (tau_rec - 0.5) / 3.0,
        "re_true": U_LID * (n - 1) / ((tau_true - 0.5) / 3.0),
        "re_recovered": U_LID * (n - 1) / ((tau_rec - 0.5) / 3.0),
        "method": res["method"],
        "root_status": res["status"],
        "n_evals": res["n_evals"],
        "iterations": res["iters"],
        "bracket_s": res["bracket"],
        "tau_history": res["tau_history"],
        "loss_history": res["loss_history"],
        "grad_history": res["grad_history"],
        "residual_rms": res["residual_rms"],
        "tolerance_rel": 0.02,
        "pass": bool(abs(tau_rec - tau_true) / tau_true < 0.02),
        "wall_time_s": time.time() - t0,
        "target": target.tolist(),
        "obs_final": res["obs_final"].tolist(),
        "step_tol": args.step_tol,
        "max_iter": args.max_iter,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    _log(
        f"[invert-rf] tau_rec={tau_rec:.6f} rel_err={payload['rel_err_percent']:.4f}% "
        f"status={res['status']} evals={res['n_evals']} "
        f"pass={payload['pass']} ({payload['wall_time_s']:.1f}s)"
    )
    _log(f"wrote {out}")


def stage_equiv(args: argparse.Namespace) -> None:
    """Production (in-place, non-differentiable) chain vs the new chain.

    The production step order of ``lid_driven_cavity.run`` is reproduced
    literally: ``collide_bgk -> stream -> bounce_back_cells ->
    zou_he_moving_lid``.  Any value difference would mean the graph-preserving
    rewrite changed the physics; the acceptance claim is "bit-identical".
    """
    from tensorlbm.boundaries import bounce_back_cells
    from tensorlbm.lid_driven_cavity import make_cavity_wall_mask
    from tensorlbm.solver import collide_bgk, stream

    n, steps = args.n, args.steps
    tau = float(tau_target(n))
    geom = CavityGeometry(n, n, dtype=DTYPE)
    mask = geom.wall_mask
    mask_prod = make_cavity_wall_mask(n, n, torch.device("cpu"), include_top=False)
    assert torch.equal(mask, mask_prod), "wall masks differ"

    f_new = initial_field(geom)
    f_prod = f_new.clone()
    t0 = time.time()
    with torch.no_grad():
        f_new = rollout(f_new, tau, U_LID, steps, wall_mask=mask)
        for _ in range(steps):
            f_prod = collide_bgk(f_prod, tau)
            f_prod = stream(f_prod)
            f_prod = bounce_back_cells(f_prod, mask_prod)
            f_prod = zou_he_moving_lid(f_prod, U_LID)
    diff = (f_new - f_prod).abs().max().item()
    payload = {
        "n": n,
        "steps": steps,
        "tau": tau,
        "u_lid": U_LID,
        "max_abs_diff": diff,
        "bit_identical": bool(diff == 0.0),
        "rel_diff": diff / float(f_prod.abs().max()),
        "wall_time_s": time.time() - t0,
        "reference": "tensorlbm.lid_driven_cavity step order "
        "(collide_bgk -> stream -> bounce_back_cells -> zou_he_moving_lid)",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    _log(f"[equiv] n={n} steps={steps} max|df|={diff:.3e} bit_identical={payload['bit_identical']}")
    _log(f"wrote {out}")

def stage_field(args: argparse.Namespace) -> None:
    """Save the Re=100 cavity field (u, |u|) as npz + metadata for figures."""
    import numpy as np

    n, steps = args.n, args.steps
    tau = args.tau if args.tau is not None else tau_target(n)
    geom = CavityGeometry(n, n, dtype=DTYPE)
    f0 = initial_field(geom)
    u_t = torch.as_tensor(U_LID, dtype=DTYPE)
    t0 = time.time()
    with torch.no_grad():
        f = rollout(f0, tau, u_t, steps, wall_mask=geom.wall_mask, compile_step=True)
        rho, ux, uy = macroscopic(f)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out.with_suffix(".npz"),
        ux=ux.numpy(),
        uy=uy.numpy(),
        rho=rho.numpy(),
        n=n,
        tau=tau,
        u_lid=U_LID,
    )
    payload = {
        "n": n,
        "steps": steps,
        "tau": tau,
        "u_lid": U_LID,
        "re": U_LID * (n - 1) / ((tau - 0.5) / 3.0),
        "max_abs_u": float(torch.sqrt(ux**2 + uy**2).max()),
        "wall_time_s": time.time() - t0,
        "npz": str(out.with_suffix(".npz")),
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    _log(f"[field] n={n} tau={tau:.5f} max|u|={payload['max_abs_u']:.4f} -> {out}")


def stage_figures_rf(args: argparse.Namespace) -> None:
    """Round-2 figures: root-finder convergence and the recovered field."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    base = Path(args.artifacts)
    figs = base / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    made: list[str] = []

    runs = sorted(base.glob("invert_rf_*.json"))
    if runs:
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
        for rp in runs:
            meta = json.loads(rp.read_text())
            tau_true = meta["tau_true"]
            losses = np.asarray(meta["loss_history"], dtype=float)
            taus = np.asarray(meta["tau_history"], dtype=float)
            grads = np.asarray(meta["grad_history"], dtype=float)
            ok = np.isfinite(losses)
            label = (
                f"n={meta['n']} T={meta['steps']} init={meta['tau_init']:.4f} "
                f"({meta['rel_err_percent']:.3f}%)"
            )
            axes[0].semilogy(np.arange(len(losses))[ok], losses[ok], "o-", ms=3, label=label)
            axes[1].semilogy(
                np.arange(len(grads))[np.isfinite(grads)],
                np.abs(grads)[np.isfinite(grads)],
                "o-",
                ms=3,
                label=label,
            )
            axes[2].plot(
                np.arange(len(taus))[np.isfinite(taus)],
                100.0 * np.abs(taus - tau_true)[np.isfinite(taus)] / tau_true,
                "o-",
                ms=3,
                label=label,
            )
            axes[2].axhline(2.0, color="r", ls="--", lw=1, label="acceptance 2%")
        axes[0].set(xlabel="oracle evaluation", ylabel="loss $L$", title="stationary loss")
        axes[1].set(
            xlabel="oracle evaluation", ylabel="$|dL/ds|$", title="stationary residual"
        )
        axes[2].set(
            xlabel="oracle evaluation",
            ylabel=r"$|\tau-\tau_{true}|/\tau_{true}$ [%]",
            title="tau recovery error",
        )
        for ax in axes:
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=7)
        fig.suptitle("GAP-PP-2 round 2: root-finding inversion of tau (dL/ds = 0)")
        fig.tight_layout()
        p = figs / "rootfind_convergence.png"
        fig.savefig(p, dpi=150)
        made.append(str(p))
        plt.close(fig)

    fields = sorted(base.glob("field_n*.npz"))
    if fields:
        npz = np.load(fields[-1])
        meta = json.loads(fields[-1].with_suffix(".json").read_text())
        ux, uy = npz["ux"], npz["uy"]
        n = int(npz["n"])
        x = np.linspace(0.0, 1.0, n)
        speed = np.sqrt(ux**2 + uy**2) / U_LID
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
        cf = axes[0].contourf(x, x, speed, levels=24, cmap="viridis")
        fig.colorbar(cf, ax=axes[0], label="$|u|/u_{lid}$")
        axes[0].streamplot(x, x, ux, uy, color="w", density=1.1, linewidth=0.6)
        axes[0].set(
            xlabel="$x/H$",
            ylabel="$y/H$",
            title=f"Re=100 lid cavity, n={n}, $\\tau$={meta['tau']:.4f}",
            aspect="equal",
        )
        axes[1].plot(ux[:, n // 2] / U_LID, x, "-", lw=1.3, label="LBM (root-find chain)")
        axes[1].plot(GHIA_RE100["u"], GHIA_RE100["y"], "ko", ms=4, label="Ghia 1982")
        axes[1].set(
            xlabel="$u_x/u_{lid}$", ylabel="$y/H$", title="vertical centreline"
        )
        axes[1].grid(alpha=0.3)
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        p = figs / f"cavity_field_n{n}_re100.png"
        fig.savefig(p, dpi=150)
        made.append(str(p))
        plt.close(fig)

    _log(f"[figures-rf] wrote {len(made)} figures: {made}")


def stage_figures(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    base = Path(args.artifacts)
    figs = base / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    made = []

    # (1) Ghia comparison, one panel per resolution
    steadies = sorted(base.glob("steady_n*.json"))
    if steadies:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        for sp in steadies:
            meta = json.loads(sp.read_text())
            data = torch.load(sp.with_suffix(".pt"))
            ux, uy = data["ux"], data["uy"]
            n = meta["n"]
            xc = n // 2
            yc = n // 2
            axes[0].plot(
                ux[:, xc].numpy() / U_LID,
                np.arange(n) / (n - 1),
                label=f"LBM n={n} (RMSE={meta['rmse_u']*100:.2f}%)",
                lw=1.4,
            )
            axes[1].plot(
                np.arange(n) / (n - 1),
                -uy[yc, :].numpy() / U_LID,
                label=f"LBM n={n} (RMSE={meta['rmse_v']*100:.2f}%)",
                lw=1.4,
            )
        axes[0].plot(
            GHIA_RE100["u"], GHIA_RE100["y"], "ko", ms=4, label="Ghia 1982 (Re=100)"
        )
        axes[1].plot(
            GHIA_RE100["x"], [-v for v in GHIA_RE100["v"]], "ko", ms=4, label="Ghia 1982"
        )
        axes[0].set(xlabel="$u_x/u_{lid}$", ylabel="$y/H$", title="Vertical centreline")
        axes[1].set(xlabel="$x/H$", ylabel="$-u_y/u_{lid}$", title="Horizontal centreline")
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("GAP-PP-2: differentiable moving-lid cavity vs Ghia (Re=100)")
        fig.tight_layout()
        p = figs / "ghia_comparison.png"
        fig.savefig(p, dpi=150)
        made.append(str(p))
        plt.close(fig)

    # (2) tau sensitivity sweep
    sweeps = sorted(base.glob("sweep_n*.json"))
    if sweeps:
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        for sp in sweeps:
            meta = json.loads(sp.read_text())
            taus = [r["tau"] for r in meta["rows"]]
            errs = [100.0 * r["rmse_u"] for r in meta["rows"]]
            ax.plot(taus, errs, "o-", label=f"n={meta['n']}")
        ax.set(xlabel=r"$\tau$", ylabel="Ghia $u$ RMSE [%]", title="tau sensitivity, Re=100")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        p = figs / "tau_sensitivity.png"
        fig.savefig(p, dpi=150)
        made.append(str(p))
        plt.close(fig)

    # (3) inversion convergence
    invs = sorted(base.glob("invert_*.json"))
    if invs:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
        for sp in invs:
            meta = json.loads(sp.read_text())
            it = np.arange(1, len(meta["tau_history"]) + 1)
            lab = f"{sp.stem}"
            axes[0].semilogy(it, meta["loss_history"], "o-", label=lab)
            axes[1].plot(it, meta["tau_history"], "o-", label=lab)
            axes[1].axhline(meta["tau_true"], ls="--", lw=1, color="k")
            axes[1].text(1, meta["tau_true"], f" tau_true={meta['tau_true']:.4f}", fontsize=8)
        axes[0].set(xlabel="iteration", ylabel="MSE loss", title="Inversion loss")
        axes[1].set(xlabel="iteration", ylabel=r"$\tau$", title="Recovered tau")
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        p = figs / "tau_inversion.png"
        fig.savefig(p, dpi=150)
        made.append(str(p))
        plt.close(fig)

    # (4) field snapshot (streamlines + |u|) of the finest steady run
    if steadies:
        sp = steadies[-1]
        data = torch.load(sp.with_suffix(".pt"))
        ux = data["ux"].numpy()
        uy = data["uy"].numpy()
        n = ux.shape[0]
        fig, ax = plt.subplots(figsize=(5.4, 4.8))
        speed = np.sqrt(ux**2 + uy**2) / U_LID
        im = ax.imshow(speed, origin="lower", extent=(0, 1, 0, 1), cmap="viridis")
        ax.streamplot(
            np.linspace(0, 1, n),
            np.linspace(0, 1, n),
            ux,
            uy,
            color="w",
            density=1.1,
            linewidth=0.6,
            arrowsize=0.7,
        )
        fig.colorbar(im, ax=ax, label="$|u|/u_{lid}$")
        ax.set(xlabel="$x/H$", ylabel="$y/H$", title=f"Cavity Re=100, n={n} (differentiable lid)")
        fig.tight_layout()
        p = figs / "cavity_field.png"
        fig.savefig(p, dpi=150)
        made.append(str(p))
        plt.close(fig)

    _log("figures: " + ", ".join(made))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("steady")
    p.add_argument("--n", type=int, default=65)
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--u-lid", type=float, default=U_LID)
    p.add_argument("--out", required=True)
    p.set_defaults(func=stage_steady)

    p = sub.add_parser("sweep")
    p.add_argument("--n", type=int, default=65)
    p.add_argument("--steps", type=int, default=25000)
    p.add_argument("--fracs", type=float, nargs="+", default=[-0.10, -0.05, 0.0, 0.05, 0.10])
    p.add_argument("--u-lid", type=float, default=U_LID)
    p.add_argument("--out", required=True)
    p.set_defaults(func=stage_sweep)

    p = sub.add_parser("fdcheck")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--hs", type=float, nargs="+", default=[1e-3, 1e-4])
    p.add_argument("--out", required=True)
    p.set_defaults(func=stage_fdcheck)

    p = sub.add_parser("invert")
    p.add_argument("--n", type=int, default=65)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--tau-true", type=float, default=None)
    p.add_argument("--tau-init", type=float, required=True)
    p.add_argument("--truth", choices=["same", "ghia"], default="same")
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--loss-tol", type=float, default=0.0)
    p.add_argument("--out", required=True)
    p.set_defaults(func=stage_invert)

    p = sub.add_parser("invert-rf")
    p.add_argument("--n", type=int, default=65)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--tau-true", type=float, default=None)
    p.add_argument("--tau-init", type=float, required=True)
    p.add_argument("--truth", choices=["same", "ghia"], default="same")
    p.add_argument("--method", choices=["illinois", "secant"], default="illinois")
    p.add_argument("--bracket", type=float, nargs=2, default=None)
    p.add_argument("--bracket-frac", type=float, default=0.05)
    p.add_argument("--max-iter", type=int, default=14)
    p.add_argument("--step-tol", type=float, default=1e-5)
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--compile-step", action="store_true")
    p.add_argument("--out", required=True)
    p.set_defaults(func=stage_invert_rf)

    p = sub.add_parser("equiv")
    p.add_argument("--n", type=int, default=65)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--out", required=True)
    p.set_defaults(func=stage_equiv)

    p = sub.add_parser("field")
    p.add_argument("--n", type=int, default=65)
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--out", required=True)
    p.set_defaults(func=stage_field)

    p = sub.add_parser("figures-rf")
    p.add_argument("--artifacts", required=True)
    p.set_defaults(func=stage_figures_rf)

    p = sub.add_parser("figures")
    p.add_argument("--artifacts", required=True)
    p.set_defaults(func=stage_figures)

    args = ap.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    # torch.compile's default backward donates buffers, which forbids the
    # create_graph backward the Newton inversion needs (autograd Hessian).
    try:
        import torch._functorch.config as _fc

        _fc.donated_buffer = False
    except Exception:  # pragma: no cover - older torch
        pass
    args.func(args)


if __name__ == "__main__":
    main()