#!/usr/bin/env python3
"""Figures + machine-extracted number tables for GAP-DIFFLBM-3D-COST.

Reads the run artifacts written by ``run.py`` (never re-types a number) and
emits ``figures/fig1_gradient.png``, ``figures/fig2_envelope.png``,
``figures/fig3_inversion.png`` and ``numbers.md``.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.environ.get("DIFF3D_ART", os.path.join(REPO, "artifacts", "gap_difflbm_3d_cost"))
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)


def load(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def fig_gradient(grad, two_tau):
    rows = grad["gradient"]
    n = [r["grid"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].semilogy(n, [abs(r["grad_ad"]) for r in rows], "o-", label="AD (checkpointed)")
    ax[0].semilogy(n, [abs(r["grad_fd_richardson"]) for r in rows], "s--",
                   label="Richardson FD (ref)")
    ax[0].semilogy(n, [abs(r["grad_analytic_anchor"]) for r in rows], "^:",
                   label="ABC continuum anchor")
    ax[0].set_xlabel("grid N (N^3 box)")
    ax[0].set_ylabel("|dJ/dtau|")
    ax[0].set_title("Gradient vs references (T=%d, tau=0.8)" % rows[0]["steps"])
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")

    ax[1].semilogy(n, [r["rel_err_ad_vs_fd"] for r in rows], "o-", label="AD vs FD")
    ax[1].semilogy(n, [r["rel_err_ad_vs_bptt"] + 1e-18 for r in rows], "s-",
                   label="AD vs full BPTT")
    ax[1].semilogy(n, [r["rel_err_ad_vs_analytic"] for r in rows], "^--",
                   label="AD vs continuum anchor")
    ax[1].axhline(1e-6, color="k", ls=":", label="gate 1e-6")
    ax[1].axhline(3e-2, color="r", ls=":", label="literature 3% band")
    ax[1].set_xlabel("grid N")
    ax[1].set_ylabel("relative error")
    ax[1].set_title("Gradient correctness (gate at 1e-6)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_gradient.png"), dpi=160)
    plt.close(fig)


def fig_envelope(env):
    timing = env["step_timing"]
    cells = np.array([r["cells"] for r in timing], float)
    step = np.array([r["step_seconds"] for r in timing], float)
    ext = env["extrapolation_256"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].loglog(cells, step, "o-", label="measured (4 threads, float64)")
    for e in ext:
        if e["threads"] == 4 and e["dtype"] == "float64":
            ax[0].loglog([e["grid"] ** 3], [e["projected_forward_step_seconds"]],
                         "r*", ms=14, label="extrapolated N=256")
    ax[0].set_xlabel("cells N^3")
    ax[0].set_ylabel("forward step [s]")
    ax[0].set_title("Forward-step cost (measured ns/cell = %.0f)"
                    % env["ns_per_cell_mean"])
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")

    labels, mem = [], []
    for e in ext:
        labels.append("%s\n%d cores" % (e["dtype"], e["threads"]))
        mem.append(e["projected_peak_bytes_model"] / 2 ** 30)
        ax[1].bar(labels[-1], mem[-1], color="C0" if e["gate_memory_8GB"] else "C3")
    ax[1].axhline(8.0, color="k", ls="--", label="gate 8 GiB")
    ax[1].set_ylabel("projected peak memory at 256^3 [GiB]")
    ax[1].set_title("Memory envelope (state-count model)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_envelope.png"), dpi=160)
    plt.close(fig)


def fig_inversion(inv):
    h = inv["history"]
    it = [r["iter"] for r in h]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].semilogy(it, [r["loss"] for r in h], "o-", ms=3, label="loss")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("1/2 ||f - f_obs||^2")
    ax[0].set_title("Inversion loss (N=%d, T=%d)" % (inv["grid"], inv["steps"]))
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")
    ax[1].plot(it, [r["tau"] for r in h], "o-", ms=3, label="tau iterate")
    ax[1].plot(it, [r["best_tau"] for r in h], "-", label="best-loss iterate")
    ax[1].axhline(inv["tau_true"], color="k", ls="--", label="tau_true")
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("tau")
    ax[1].set_title("Recovered tau (rel.err %.2e)" % inv["rel_err"])
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_inversion.png"), dpi=160)
    plt.close(fig)


def numbers(grad, two_tau, env, inv):
    L = ["# GAP-DIFFLBM-3D-COST — machine-extracted numbers", ""]
    L += ["## Case A/C: gradient vs references", "",
          "| N | T | g_AD | g_FD(Richardson) | g_full-BPTT | g_ABC-anchor | rel(FD) | rel(BPTT) | rel(anchor) | gate 1e-6 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in grad["gradient"]:
        L.append("| %d | %d | %.8e | %.8e | %.8e | %.8e | %.2e | %.2e | %.2e | %s |" % (
            r["grid"], r["steps"], r["grad_ad"], r["grad_fd_richardson"],
            r["grad_full_bptt"], r["grad_analytic_anchor"],
            r["rel_err_ad_vs_fd"], r["rel_err_ad_vs_bptt"],
            r["rel_err_ad_vs_analytic"], "PASS" if r["gate_1e-6"] else "FAIL"))
    if two_tau:
        L += ["", "## Case C: second parameter (tau) points", "",
              "| N | T | tau | g_AD | g_FD | rel | gate 1e-6 |", "|---|---|---|---|---|---|---|"]
        for r in two_tau["two_tau"]:
            L.append("| %d | %d | %.2f | %.8e | %.8e | %.2e | %s |" % (
                r["grid"], r["steps"], r["tau"], r["grad_ad"], r["grad_fd"],
                r["rel_err_ad_vs_fd"], "PASS" if r["gate_1e-6"] else "FAIL"))
    if env:
        L += ["", "## Case D: memory / time envelope", "",
              "| N | cells | fwd step [ms] | ns/cell | eff. BW [GB/s] |", "|---|---|---|---|---|"]
        for r in env["step_timing"]:
            L.append("| %d | %d | %.1f | %.0f | %.2f |" % (
                r["grid"], r["cells"], r["step_seconds"] * 1e3, r["ns_per_cell"],
                r["effective_bandwidth_gbs"]))
        L += ["", "| N | T | peak RSS over baseline [MiB] | states-equiv (RSS) | model states |",
              "|---|---|---|---|---|"]
        for r in env["memory"]:
            L.append("| %d | %d | %.1f | %.1f | %d |" % (
                r["grid"], r["steps"],
                (r["peak_bytes"] - r["baseline_bytes"]) / 2 ** 20,
                r["states_equivalent_rss"], r["model_states"]))
        L += ["", "| N | T | peak RSS over baseline [MiB] | states-equiv (RSS) |",
              "|---|---|---|---|"]
        for r in env["boundedness"]:
            L.append("| %d | %d | %.1f | %.1f |" % (
                r["grid"], r["steps"], r["peak_bytes_over_baseline"] / 2 ** 20,
                r["states_equivalent_rss"]))
        L += ["", "reverse-mode overhead measured: %.2f x forward step (N=%d, segment=%d)" % (
            env["reverse_overhead"]["overhead_factor_vs_forward"],
            env["reverse_overhead"]["grid"], env["reverse_overhead"]["segment"]), "",
            "| dtype | cores | proj. fwd step [s] | proj. rev step [s] | T | inversion [h] | T_max(2h) | peak [GiB] | mem 8GB | time 2h |",
            "|---|---|---|---|---|---|---|---|---|---|"]
        for e in env["extrapolation_256"]:
            L.append("| %s | %d | %.2f | %.2f | %d | %.2f | %.0f | %.2f | %s | %s |" % (
                e["dtype"], e["threads"], e["projected_forward_step_seconds"],
                e["projected_reverse_step_seconds"], e["assumed_T"],
                e["projected_inversion_seconds"] / 3600, e["max_T_for_2h"],
                e["projected_peak_bytes_model"] / 2 ** 30,
                "PASS" if e["gate_memory_8GB"] else "OVER",
                "PASS" if e["gate_time_2h"] else "OVER"))
    if inv:
        L += ["", "## Real inversion (tau recovery)", "",
              "| N | T | tau_true | tau0 | tau_hat | rel.err | iters | wall [s] | <=3% |",
              "|---|---|---|---|---|---|---|---|---|",
              "| %d | %d | %.4f | %.4f | %.6f | %.3e | %d | %.1f | %s |" % (
                  inv["grid"], inv["steps"], inv["tau_true"], inv["tau0"],
                  inv["tau_hat"], inv["rel_err"], inv["iters"], inv["seconds"],
                  "PASS" if inv["gate_3pct"] else "FAIL")]
    path = os.path.join(OUT, "numbers.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    g = load("run_gradient.json")
    t = load("run_two_tau.json")
    e = load("run_envelope.json")
    if e and "envelope" in e:
        e = e["envelope"]
    i = load("run_inversion.json")
    if i and "inversion" in i:
        i = i["inversion"]
    if g:
        fig_gradient(g, t)
    if e:
        fig_envelope(e)
    if i:
        fig_inversion(i)
    numbers(g, t, e, i)
    print("\nfigures ->", FIG)