"""Validation artefacts for the ELBM node: H-theorem, entropy condition, laminar.

Produces the figures and the JSON block that do *not* depend on the long
turbulent-channel runs, so the physics claims of the node are backed by
artefacts even if the campaign runs are still in flight:

* ``elbm_h_theorem.png``  -- per-cell entropy change of the entropic collision
  over many distorted states and relaxation times, plus the ``H(α)`` curve that
  the local Newton solve inverts;
* ``elbm_equilibrium.png`` -- the closed-form D3Q27 entropic equilibrium against
  the independent Lagrange-dual Newton solve, and ``H`` along the collision line;
* ``elbm_laminar.png``    -- the native channel kernel against the exact
  Poiseuille parabola (end-to-end check of forcing + bounce-back + geometry).

Usage: ``python tools/elbm_validation_report.py --out artifacts/elbm/validation``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensorlbm.d3q27 import W_EXACT64 as W27
from tensorlbm.d3q27 import macroscopic27
from tensorlbm.elbm import (
    ALPHA_CAP,
    collide_elbm_d3q27,
    entropic_equilibrium_dual,
    entropic_equilibrium_gauss_d3q27,
    gauss_entropy,
    solve_entropy_condition,
)

F64 = torch.float64
_W = W27.view(27, 1, 1, 1)


def _state(seed: int, amp: float = 0.3, shape=(4, 5, 6), noise: float = 0.5) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    rho = 1.0 + 0.05 * torch.randn(shape, generator=g, dtype=F64)
    u = amp * torch.randn((3,) + shape, generator=g, dtype=F64)
    feq = entropic_equilibrium_gauss_d3q27(rho, u[0], u[1], u[2])
    return torch.clamp(feq * (1.0 + noise * torch.randn(feq.shape, generator=g, dtype=F64)), min=1e-9)


def h_theorem_figure(out: Path) -> dict:
    taus = [0.5002, 0.5005, 0.501, 0.505, 0.51, 0.55, 0.7, 1.0, 1.5, 2.0]
    records = []
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for tau in taus:
        dh_all = []
        alpha_all = []
        for seed in range(4):
            f = _state(seed)
            f_post, info = collide_elbm_d3q27(f, tau, n_iter=30, return_info=True)
            dh = (gauss_entropy(f_post, _W) - gauss_entropy(f, _W)).numpy().ravel()
            dh_all.append(dh)
            alpha_all.append(info.alpha.numpy().ravel())
            assert float(f_post.min()) > 0.0
        dh = np.concatenate(dh_all)
        al = np.concatenate(alpha_all)
        records.append(
            {
                "tau": tau,
                "dh_max": float(dh.max()),
                "dh_min": float(dh.min()),
                "alpha_min": float(al.min()),
                "alpha_mean": float(al.mean()),
                "alpha_max": float(al.max()),
                "limiter_active_fraction": float((al < 0.999).mean()),
                "cells": int(dh.size),
            }
        )
        axes[0].scatter(np.full(dh.size, tau) * (1 + 0.02 * np.random.RandomState(0).randn(dh.size)),
                        dh, s=3, alpha=0.4)
    axes[0].axhline(0.0, color="k", lw=0.8)
    axes[0].set_xlabel("relaxation time tau (log)")
    axes[0].set_ylabel("H(f*) - H(f)  per cell")
    axes[0].set_xscale("log")
    axes[0].set_title("H-theorem: collision never increases H")
    axes[0].grid(alpha=0.3)

    # entropy-condition curve for one cell
    f = _state(0, amp=0.25)
    feq = entropic_equilibrium_gauss_d3q27(*macroscopic27(f))
    tau = 0.51
    alphas = np.linspace(0.0, 1.5, 121)
    cell = (0, 2, 3)
    fc = f[:, cell[0], cell[1], cell[2]].view(27, 1, 1, 1)
    fe = feq[:, cell[0], cell[1], cell[2]].view(27, 1, 1, 1)
    h0 = float(gauss_entropy(fc, _W))
    dh = [float(gauss_entropy(fc + (a / tau) * (fe - fc), _W)) - h0 for a in alphas]
    axes[1].plot(alphas, dh, "-", color="C3", label="H(alpha) - H(0)")
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].axvline(2 * tau, color="C0", ls="--", lw=1.0, label="mirror point alpha = 2 tau")
    axes[1].axvline(ALPHA_CAP, color="C2", ls=":", lw=1.2, label="admissibility cap alpha = 1 (BGK)")
    axes[1].set_xlabel("alpha (collision rate = alpha / tau)")
    axes[1].set_ylabel("H(alpha) - H(0)")
    axes[1].set_title("Entropy condition: two roots, 0 and the mirror point")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p = out / "elbm_h_theorem.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return {"table": records, "figure": str(p)}


def equilibrium_figure(out: Path) -> dict:
    shape = (3, 4, 5)
    g = torch.Generator().manual_seed(5)
    rho = 1.0 + 0.1 * torch.randn(shape, generator=g, dtype=F64)
    u = 0.25 * torch.randn((3,) + shape, generator=g, dtype=F64)
    closed = entropic_equilibrium_gauss_d3q27(rho, u[0], u[1], u[2])
    dual = entropic_equilibrium_dual(rho, u[0], u[1], u[2], lattice="d3q27", n_newton=40, tol=1e-15)
    err = float((closed - dual).abs().max())
    k = 5
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    axes[0].bar(np.arange(27), closed[:, 0, 2, 3].numpy(), color="C0", width=0.7, label="closed form")
    axes[0].plot(np.arange(27), dual[:, 0, 2, 3].numpy(), "k.", ms=5, label="dual Newton")
    axes[0].set_xlabel("population index q")
    axes[0].set_ylabel("f_eq (cell 0,2,3)")
    axes[0].set_title(f"Entropic equilibrium D3Q27 (max diff {err:.1e})")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    # entropy along a moment-preserving line through f^eq
    fc = closed[:, 0, 2, 3].view(27, 1, 1, 1)
    d = torch.randn(fc.shape, generator=g, dtype=F64)
    d = d - d.sum() * _W
    lam = np.linspace(-3.0, 3.0, 61)
    h0 = float(gauss_entropy(fc, _W))
    vals = [float(gauss_entropy(fc + lam_i * 1e-3 * d, _W)) - h0 for lam_i in lam]
    axes[1].plot(lam, vals, "-")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xlabel("lambda (moment-preserving perturbation)")
    axes[1].set_ylabel("H - H(f_eq)")
    axes[1].set_title("f_eq is the entropy minimum")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p = out / "elbm_equilibrium.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return {"closed_form_vs_dual_max_abs_diff": err, "figure": str(p), "populations": k}


def laminar_figure(out: Path) -> dict:
    from tensorlbm.elbm_channel_native import native_channel_run

    nx, ny, nz = 4, 21, 4
    re_tau, u_tau = 4.0, 0.01
    res = native_channel_run(
        nx=nx, ny=ny, nz=nz, steps=60000, warmup=30000, sample_every=200,
        re_tau=re_tau, u_tau=u_tau, n_iter=4, noise_amplitude=1e-8, log_every=20000,
    )
    height = ny - 2
    half = 0.5 * height
    g = u_tau**2 / half
    nu = u_tau * half / re_tau
    y = np.arange(ny) - 0.5
    u = res.profiles["u"]
    analytic = g / (2.0 * nu) * y * (height - y)
    interior = (y > 0.5) & (y < height - 0.5)
    rel = float(np.abs(u[interior] - analytic[interior]).max() / analytic.max())
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.plot(analytic, y, "k-", lw=1.4, label="exact Poiseuille")
    ax.plot(u, y, "o", ms=3.5, color="C3", label="ELBM (native kernel)")
    ax.set_ylabel("y (lattice units from the bottom wall plane)")
    ax.set_xlabel("u_x")
    ax.set_title("Laminar channel: ELBM + Guo force + half-way bounce-back")
    ax.text(0.03, 0.06, f"max rel. error {rel:.2%}", transform=ax.transAxes, fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out / "elbm_laminar.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return {
        "max_rel_error": rel,
        "tau": 0.5 + 3 * nu,
        "nu": nu,
        "alpha_min": res.scalars["alpha_min"],
        "alpha_mean": res.scalars["alpha_mean"],
        "min_population": res.scalars["min_population"],
        "figure": str(p),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/elbm/validation")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "h_theorem": h_theorem_figure(out),
        "equilibrium": equilibrium_figure(out),
        "laminar": laminar_figure(out),
    }
    (out / "validation.json").write_text(json.dumps(payload, indent=1, default=float))
    print(json.dumps(payload, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())