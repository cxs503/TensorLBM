#!/usr/bin/env python3
"""Figures for XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER (values read from
artifacts/results.json plus two short rollouts; no number is typed by hand)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
import oss_arm  # noqa: E402
from tensorlbm import les_bptt as L  # noqa: E402

torch.set_num_threads(1)
RES = json.loads((HERE / "artifacts" / "results.json").read_text())
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
CASE = RES["source_of_record"] and {"n": 48, "u0": 0.2, "k": 2, "tau0": 0.56, "sc": 0.17}


def fig_convergence():
    a2 = RES["A2"]
    a2b = RES["A2b_adam"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(a2b["history"]["sc"], label="Adam (BPTT, log Sc)", lw=2)
    ax[0].axhline(a2["sc_true"], color="k", ls="--", label=f"sc_true = {a2['sc_true']}")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("Sc")
    ax[0].set_title(f"A2b Adam trace (final Sc = {a2b['sc_got']:.5f})")
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].semilogy(a2b["history"]["loss"], label="Adam loss", lw=2)
    ax[1].semilogy(np.maximum(np.abs(a2["history"]["grad"]), 1e-300), "o-",
                   label="|dL/dSc| (root-solve evals)", ms=4)
    ax[1].set_xlabel("evaluation"); ax[1].set_ylabel("loss / |grad|")
    ax[1].set_title(f"A2 recovered Sc = {a2['sc_got']:.6f} (rel err {a2['rel_error']:.2e})")
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "fig1_bptt_convergence.png", dpi=150)
    plt.close(fig)


def fig_decay():
    n, u0, k = CASE["n"], CASE["u0"], CASE["k"]
    tau0, sc = CASE["tau0"], CASE["sc"]
    steps = 400
    nu0 = (tau0 - 0.5) / 3.0
    kp = 2 * math.pi * k / n
    f0, b = L.taylor_green_state(n=n, u0=u0, k=k)
    series = {}
    for les in (False, True):
        hist = L.unroll(f0, steps, tau0=tau0, sc=sc, boundary=b, les=les)
        series[les] = [float(L.tg_amplitude(h.f, k=k)[0]) for h in hist]
    t = np.arange(1, steps + 1)
    a_ref = u0 * np.exp(-2 * nu0 * kp ** 2 * t)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(t, series[False], lw=2, label="TensorLBM, LES off")
    ax.plot(t, a_ref, "k--", lw=1.5, label=r"analytic $u_0 e^{-2\nu_0 k^2 t}$")
    ax.plot(t, series[True], lw=2, label=f"TensorLBM, LES on (Sc={sc})")
    ax.set_xlabel("step"); ax.set_ylabel("Taylor-Green amplitude")
    ax.set_title("A5 linear-regime decay vs analytic reference")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "fig2_tg_decay_analytic.png", dpi=150)
    plt.close(fig)
    return series, a_ref


def fig_parity():
    n, u0, k = CASE["n"], CASE["u0"], CASE["k"]
    tau0, sc, steps = CASE["tau0"], CASE["sc"], 40
    f0, b = L.taylor_green_state(n=n, u0=u0, k=k)
    ht = L.unroll(f0, steps, tau0=tau0, sc=sc, boundary=b)
    fo = f0.numpy().transpose(0, 2, 3, 1).astype("float32")
    bo = b.numpy().transpose(0, 2, 3, 1).astype("float32")
    ho = oss_arm.unroll_sc(fo, steps, tau0, sc, bo)
    vt = ht[-1].vel[0].numpy()
    vo = ho[-1]["vel"][0][..., :2].transpose(2, 0, 1)
    du = np.hypot(vt[0] - vo[0], vt[1] - vo[1])
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    im = ax[0].imshow(du, cmap="magma")
    ax[0].set_title(f"|u_TensorLBM - u_OSS| after {steps} steps\nmax = {du.max():.2e}")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[1].plot(vt[0][n // 2], label="TensorLBM", lw=2)
    ax[1].plot(vo[0][n // 2], "--", label="OSS arm", lw=1.5)
    ax[1].set_title("$u_x$ profile (mid row)")
    ax[1].set_xlabel("x"); ax[1].legend(); ax[1].grid(alpha=.3)
    steps_ax = [p["step"] for p in RES["A1_tg"]["per_step"]]
    ax[2].semilogy(steps_ax, [p["vel_rel_l2"] for p in RES["A1_tg"]["per_step"]],
                   "o-", label="TG box")
    ax[2].semilogy(steps_ax, [p["f_rel_l2"] for p in RES["A1_tg"]["per_step"]], "s--",
                   label="TG box (f)")
    steps_d = [p["step"] for p in RES["A1_disc"]["per_step"]]
    ax[2].semilogy(steps_d, [p["vel_rel_l2"] for p in RES["A1_disc"]["per_step"]],
                   "^-", label="disc (bounce-back)")
    ax[2].axhline(RES["A1_tg"]["threshold"], color="r", ls=":", label="gate 1e-4")
    ax[2].set_xlabel("step"); ax[2].set_ylabel("relative L2"); ax[2].legend(); ax[2].grid(alpha=.3)
    ax[2].set_title("A1 forward parity vs OSS arm")
    fig.tight_layout(); fig.savefig(FIG / "fig3_forward_parity.png", dpi=150)
    plt.close(fig)


def fig_sensitivity():
    g = RES["A6"]["groups"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].plot([r["steps"] for r in g["unroll_length"]],
               [r["sc_got"] for r in g["unroll_length"]], "o-")
    ax[0].axhline(RES["A2"]["sc_true"], color="k", ls="--")
    ax[0].set_xlabel("unroll length N"); ax[0].set_ylabel("recovered Sc")
    ax[0].set_title("sensitivity to N")
    ax[1].plot([r["sc_init"] for r in g["sc_init"]], [r["sc_got"] for r in g["sc_init"]], "o-")
    ax[1].plot([r["sc_init"] for r in g["sc_init"]], [r["sc_init"] for r in g["sc_init"]],
               "k:", lw=1, label="initial value")
    ax[1].axhline(RES["A2"]["sc_true"], color="k", ls="--")
    ax[1].set_xlabel("sc_init"); ax[1].set_ylabel("recovered Sc"); ax[1].set_title("sensitivity to init")
    ax[1].legend()
    ax[2].plot([r["tau0"] for r in g["tau0"]], [r["sc_got"] for r in g["tau0"]], "o-")
    ax[2].axhline(RES["A2"]["sc_true"], color="k", ls="--")
    ax[2].set_xlabel(r"$\tau_0$"); ax[2].set_ylabel("recovered Sc"); ax[2].set_title("sensitivity to tau0")
    for a in ax:
        a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "fig4_sensitivity.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_convergence()
    fig_decay()
    fig_parity()
    fig_sensitivity()
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))
