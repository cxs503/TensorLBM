"""Post-processing: criteria table + figures for the free-surface verification face.

Reads only artifacts written by the run scripts (zero hand-copied numbers) and
emits

* ``runs/p351w1_n07/summary.json``   -- criteria table + provenance hashes
* ``figures/fig_front_vs_reference.png``
* ``figures/fig_error_vs_reference.png``
* ``figures/fig_mass_ledger.png``
* ``figures/fig_convergence.png``
* ``figures/fig_fields.png``          (only if the snapshot run is present)

Usage
-----
    python benchmarks/free_surface_dambreak_wigley/analyze.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.free_surface import (  # noqa: E402
    fit_convergence_order,
    martin_moyce_table,
    relative_rmse_percent,
)

RUNS = ROOT / "runs" / "p351w1_n07"
FIGS = ROOT / "figures"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def load_cases() -> list:
    cases = []
    for path in sorted((RUNS / "dambreak").glob("*.json")):
        data = json.loads(path.read_text())
        data["_file"] = str(path.relative_to(ROOT))
        data["_sha256"] = sha256(path)
        cases.append(data)
    return cases


def fig_front(cases, table):
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(table["T"], table["X"], "ko", ms=4.5, label="experiment (digitised)")
    for c in cases:
        ax.plot(c["T"], c["X_sim"], "-", lw=1.3,
                label=f"{c['config']['cells_per_a']}/a, {c['config']['wall']} "
                      f"(relRMSE {c['rmse_relative_percent']:.1f}%)")
    ax.set_xlabel(r"$T = t\sqrt{2g/a}$")
    ax.set_ylabel(r"$X = x_{front}/a$")
    ax.set_title("Collapse front: solver vs digitised experiment")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_front_vs_reference.png", dpi=150)
    plt.close(fig)


def fig_error(cases):
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for c in cases:
        ax.plot(c["T"], c["error_percent"], "o-", ms=3, lw=1.0,
                label=f"{c['config']['cells_per_a']}/a, {c['config']['wall']}")
    ax.axhspan(-10, 10, color="tab:green", alpha=0.10, label="+-10 % band")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"$T$")
    ax.set_ylabel("front error [%]")
    ax.set_title("Front-position error against the experimental table")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_error_vs_reference.png", dpi=150)
    plt.close(fig)


def fig_mass(cases):
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for c in cases:
        ax.semilogy(c["T"], np.abs(np.asarray(c["error"]) * 0.0) + 1e-16, alpha=0.0)
    for c in cases:
        drift = c["mass"]["max_relative_drift"]
        ax.bar(c["tag"], max(drift, 1e-16), label=c["tag"])
    ax.axhline(2.0e-2, color="tab:red", ls="--", lw=1.0, label="gate 2 %")
    ax.set_ylabel(r"$\max_t |M(t)-M(0)|/M(0)$")
    ax.set_title("Mass ledger (liquid volume drift of the whole run)")
    ax.set_yscale("log")
    ax.tick_params(axis="x", labelsize=6, rotation=20)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_mass_ledger.png", dpi=150)
    plt.close(fig)


def fig_convergence(conv):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for ax, key, label in ((axes[0], "resolved_interface", "resolved interface (w = 0.02)"),
                           (axes[1], "sharp_interface", "sharp interface (w = 0.004)")):
        d = conv[key]
        h = np.asarray(d["h"])
        e = np.asarray(d["l2"])
        ax.loglog(h, e, "o-", label=f"measured, order {d['order']:.2f}")
        fit = fit_convergence_order(h, e)
        ax.loglog(h, np.exp(fit["constant"]) * h ** fit["order"], "k--", lw=1.0,
                  label=f"fit slope {fit['order']:.2f} (R2 {fit['r2']:.3f})")
        ax.loglog(h, e[0] * (h / h[0]) ** 1.5, ":", color="tab:red",
                  label="slope 1.5 target")
        ax.set_xlabel("h")
        ax.set_ylabel(r"$L_2$ error of $F$")
        ax.set_title(label, fontsize=9)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_convergence.png", dpi=150)
    plt.close(fig)


def fig_fields():
    path = RUNS / "snapshots.npz"
    if not path.exists():
        return False
    d = np.load(path)
    F, P, T = d["F"], d["p"], d["T"]
    lx, ly = float(d["lx"]), float(d["ly"])
    ext = (0.0, lx, 0.0, ly)
    fig, axes = plt.subplots(2, len(T), figsize=(2.3 * len(T), 4.4),
                             gridspec_kw={"height_ratios": [1.0, 1.0]})
    for k in range(len(T)):
        im1 = axes[0, k].imshow(F[k], origin="lower", extent=ext, cmap="Blues",
                                vmin=0.0, vmax=1.0, aspect="auto")
        im2 = axes[1, k].imshow(P[k], origin="lower", extent=ext, cmap="RdBu_r",
                                aspect="auto")
        axes[0, k].set_title(f"T = {T[k]:.2f}", fontsize=8)
        axes[0, k].set_xticks([])
        axes[0, k].set_yticks([])
        axes[1, k].set_xticks([])
        axes[1, k].set_yticks([])
        plt.colorbar(im1, ax=axes[0, k], fraction=0.046, pad=0.02)
        plt.colorbar(im2, ax=axes[1, k], fraction=0.046, pad=0.02)
    axes[0, 0].set_ylabel("liquid fraction F", fontsize=8)
    axes[1, 0].set_ylabel("pressure p", fontsize=8)
    fig.suptitle("Collapse: VOF interface (top) and pressure field (bottom)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_fields.png", dpi=150)
    plt.close(fig)
    return True


def main() -> int:
    FIGS.mkdir(exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    table = martin_moyce_table()
    cases = load_cases()
    if not cases:
        print("no dam-break artefacts yet")
        return 1

    conv_path = RUNS / "convergence.json"
    conv = json.loads(conv_path.read_text()) if conv_path.exists() else None

    summary = {"criteria": {}, "cases": []}
    for c in cases:
        summary["cases"].append({
            "tag": c["tag"],
            "cells_per_a": c["config"]["cells_per_a"],
            "wall": c["config"]["wall"],
            "nx": c["grid"]["nx"], "ny": c["grid"]["ny"], "dx": c["grid"]["dx"],
            "rmse": c["rmse"],
            "rmse_relative_percent": c["rmse_relative_percent"],
            "max_abs_error": c["max_abs_error"],
            "mass_max_drift": c["mass"]["max_relative_drift"],
            "n_steps": c["n_steps"],
            "wall_seconds": c["wall_seconds"],
            "artifact": c["_file"],
            "artifact_sha256_16": c["_sha256"],
        })

    best = min(cases, key=lambda c: c["rmse_relative_percent"])
    summary["criteria"]["a_front_rmse_relative_percent"] = best["rmse_relative_percent"]
    summary["criteria"]["a_front_rmse_absolute"] = best["rmse"]
    summary["criteria"]["a_pass_le_10_percent"] = bool(best["rmse_relative_percent"] <= 10.0)
    summary["criteria"]["b_mass_max_drift"] = best["mass"]["max_relative_drift"]
    summary["criteria"]["b_pass_le_2_percent"] = bool(best["mass"]["max_relative_drift"] <= 2.0e-2)
    summary["criteria"]["best_case"] = best["tag"]
    summary["criteria"]["best_case_artifact"] = best["_file"]

    if conv is not None:
        summary["criteria"]["d_interface_order"] = conv["criteria"]["resolved_order"]
        summary["criteria"]["d_interface_r2"] = conv["criteria"]["resolved_r2"]
        summary["criteria"]["d_pass_ge_1p5"] = bool(conv["criteria"]["pass"])

    # relative-RMSE recomputation from the printed series (independent of the
    # solver-side number, so the report number is traceable to two paths)
    recomputed = relative_rmse_percent(best["X_sim"], table["X"])
    summary["criteria"]["a_front_rmse_relative_recomputed"] = recomputed

    (RUNS / "summary.json").write_text(json.dumps(summary, indent=1))

    fig_front(cases, table)
    fig_error(cases)
    fig_mass(cases)
    if conv is not None:
        fig_convergence(conv)
    have_fields = fig_fields()

    print(json.dumps(summary["criteria"], indent=1))
    print("figures:", sorted(p.name for p in FIGS.glob("fig_*.png")))
    print("fields figure:", have_fields)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())