"""Figures for the IN67-N01 micro-bubble drag-reduction report (matplotlib only).

Every figure is generated from ``results.json`` (programmatic injection, no
hand-copied numbers).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG_DPI = 150


def _save(fig, out: Path, name: str) -> str:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return str(path)


def fig_void_field(payload, out: Path) -> str:
    raw = payload["raw"]
    key = "single_d10" if "single_d10" in raw else next(iter(raw))
    src = raw[key]
    if "alpha_field" not in src:
        # the production summaries carry no field snapshots (they are transported
        # compactly); the cloud map comes from the dedicated visualisation run
        vis = out / "visual_case.json"
        if not vis.exists():
            return ""
        src = json.loads(vis.read_text())
        key = f"visualisation case ({key} parameters, 64x65, 4000 steps)"
    alpha = np.asarray(src["alpha_field"], dtype=float)
    u = np.asarray(src["u_field"], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    im0 = axes[0].pcolormesh(alpha, cmap="Blues", vmin=0, vmax=1, shading="auto")
    axes[0].set_title(f"void fraction alpha, {key}, every 2nd cell")
    fig.colorbar(im0, ax=axes[0], label="alpha")
    im1 = axes[1].pcolormesh(u, cmap="RdBu_r", shading="auto")
    axes[1].set_title("streamwise velocity u_x")
    axes[1].set_xlabel("x [lattice]")
    fig.colorbar(im1, ax=axes[1], label="u_x")
    for ax in axes:
        ax.set_ylabel("y [lattice]")
    return _save(fig, out, "fig1_void_field.png")


def fig_mean_profiles(payload, out: Path) -> str:
    raw = payload["raw"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, res in raw.items():
        prof = res["profile"]
        y = np.asarray(prof["y"], dtype=float)
        axes[0].plot(np.asarray(prof["u"], dtype=float), y, label=name)
        if float(np.max(prof["alpha"])) > 1e-6:
            axes[1].plot(np.asarray(prof["alpha"], dtype=float), y, label=name)
    axes[0].set_xlabel(r"$\bar u_x$ [lattice]")
    axes[0].set_ylabel("y from wall [lattice]")
    axes[0].set_title("mean velocity profiles")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel(r"$\bar\alpha$")
    axes[1].set_title("mean void fraction")
    axes[1].legend(fontsize=7)
    return _save(fig, out, "fig2_mean_profiles.png")


def fig_drag_vs_size(payload, out: Path) -> str:
    cases = payload["summary"]["cases"]
    order = [("single_d5", 5), ("single_d10", 10), ("single_d20", 20), ("single_d40", 40)]
    pts = [(d, cases[k]["DR"], cases[k]["alpha_mean"], cases[k]["d_plus_achieved"])
           for k, d in order if k in cases]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    d = [p[0] for p in pts]
    axes[0].plot(d, [100 * p[1] for p in pts], "o-", label=r"$DR$ (measured)")
    axes[0].axhline(0.0, color="k", lw=0.8, ls=":")
    axes[0].set_xlabel(r"$d^+$ (target)")
    axes[0].set_ylabel(r"$DR$ [%]")
    axes[0].set_title("drag reduction vs bubble size (1 bubble)")
    axes[0].legend(fontsize=8)
    axes[1].plot([p[2] for p in pts], [100 * p[1] for p in pts], "s-")
    axes[1].set_xlabel(r"achieved void fraction $\alpha$")
    axes[1].set_ylabel(r"$DR$ [%]")
    axes[1].set_title("same runs vs achieved void fraction (confound shown)")
    return _save(fig, out, "fig3_drag_vs_size.png")


def fig_void_path(payload, out: Path) -> str:
    cases = payload["summary"]["cases"]
    names = [("zero_gas", 0.0), ("single_d10", None), ("void_n2", None),
             ("void_n4", None), ("void_n8", None)]
    pts = []
    for key, _ in names:
        c = cases.get(key)
        if c is None:
            continue
        pts.append((c["alpha_mean"], c["DR"], key))
    pts.sort(key=lambda p: p[0])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    axes[0].plot([p[0] for p in pts], [100 * p[1] for p in pts], "o-")
    for a, dr, k in pts:
        axes[0].annotate(k, (a, 100 * dr), fontsize=7, rotation=0,
                         xytext=(3, 3), textcoords="offset points")
    axes[0].set_xlabel(r"void fraction $\alpha$")
    axes[0].set_ylabel(r"$DR$ [%]")
    axes[0].set_title(r"negative control: $DR\to0$ as $\alpha\to0$")
    axes[0].axhline(0.0, color="k", lw=0.8, ls=":")
    sub = [(k, cases[k]) for k in ("zero_gas",) if k in cases]
    axes[1].bar([k for k, _ in sub], [100 * c["DR"] for _, c in sub], width=0.4)
    axes[1].set_ylabel(r"$DR$ [%]")
    axes[1].set_title("zero-gas run (must vanish)")
    return _save(fig, out, "fig4_void_path.png")


def fig_mechanism_split(payload, out: Path) -> str:
    cases = payload["summary"]["cases"]
    keys = [k for k in ("single_d5", "single_d10", "single_d20", "single_d40",
                        "void_n2", "void_n4", "void_n8") if k in cases]
    if not keys:
        return ""
    slip = [100 * cases[k]["DR_slip"] for k in keys]
    visc = [100 * cases[k]["DR_viscosity"] for k in keys]
    turb = [100 * cases[k]["DR_turb_incl_profile"] for k in keys]
    total = [100 * cases[k]["DR_total"] for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.bar(x - 0.28, slip, width=0.26, label="slip")
    ax.bar(x, visc, width=0.26, label="viscosity contrast")
    ax.bar(x + 0.28, turb, width=0.26, label="turbulence (incl. profile footprint)")
    ax.plot(x, total, "k_", ms=22, mew=2, label="total DR (measured)")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=20, fontsize=8)
    ax.set_ylabel("contribution to DR [%]")
    ax.set_title("three-way mechanism split, fractions of the single-phase wall shear")
    ax.legend(fontsize=7)
    ax.axhline(0.0, color="k", lw=0.8)
    return _save(fig, out, "fig5_mechanism_split.png")


def fig_reynolds_and_utau(payload, out: Path) -> str:
    raw = payload["raw"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, res in raw.items():
        prof = res["profile"]
        y = np.asarray(prof["y"], dtype=float)
        rho_uv = np.asarray(prof["rho_uv"], dtype=float)
        if np.max(np.abs(rho_uv)) > 1e-12:
            axes[0].plot(-rho_uv, y, label=name)
    axes[0].set_xlabel(r"$-\langle\rho u'v'\rangle$")
    axes[0].set_ylabel("y [lattice]")
    axes[0].set_title("Reynolds shear stress")
    axes[0].legend(fontsize=7)
    cases = payload["summary"]["cases"]
    keys = list(cases)
    x = np.arange(len(keys))
    for k, label in (("u_tau_gradient", "wall gradient"), ("u_tau_identity", "Reynolds identity"),
                     ("u_tau", "momentum exchange"), ("u_tau_balance", "force balance")):
        axes[1].plot(x, [cases[key][k] for key in keys], "o", label=label, ms=4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(keys, rotation=25, fontsize=7)
    axes[1].set_ylabel(r"$u_\tau$ [lattice]")
    axes[1].set_title("four friction-velocity estimates (C2)")
    axes[1].legend(fontsize=7)
    return _save(fig, out, "fig6_reynolds_utau.png")


def make_all_figures(payload, out: Path) -> list[str]:
    paths = []
    for fn in (fig_void_field, fig_mean_profiles, fig_drag_vs_size, fig_void_path,
               fig_mechanism_split, fig_reynolds_and_utau):
        try:
            p = fn(payload, out)
            if p:
                paths.append(p)
        except Exception as exc:  # pragma: no cover - figures must not kill the run
            print(f"figure {fn.__name__} failed: {exc}")
    return paths


if __name__ == "__main__":  # pragma: no cover
    import sys
    payload = json.loads(Path(sys.argv[1]).read_text())
    print(make_all_figures(payload, Path(sys.argv[2] if len(sys.argv) > 2 else "artifacts/microbubble_dr_in67_n01")))