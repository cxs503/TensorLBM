"""P399W2-N04 figures: surface quadrature, force convergence, conservation, fields.

Reads ``artifacts/p399w2_n04/*.json`` (produced by ``run_validation.py`` /
``run_case.py``) and writes publication figures to ``figures/p399w2_n04/``.
Every number plotted comes from the artifacts; nothing is hand-typed.

    PYTHONPATH=src python scripts/p399w2_n04_curved_force/make_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/p399w2_n04"
FIG = ROOT / "figures/p399w2_n04"

CD_LIT = {"low": 1.26, "ref": 1.34, "high": 1.36}


def _load(name: str):
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else None


def _cases():
    out = []
    for p in sorted(ART.glob("case_*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            pass
    return out


def fig_surface_measure(quad: dict) -> Path | None:
    if not quad:
        return None
    rows = quad["cases"]["measure"]
    M = [r["n_markers"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))

    ax = axes[0]
    for key, lab, mk in (
        ("circle_rel_error", "circle $2\\pi R$", "o"),
        ("ellipse_rel_error", "ellipse $4aE(e)$", "s"),
        ("sphere_rel_error", "sphere $4\\pi R^2$", "^"),
    ):
        y = [max(r[key] or 0.0, 1e-17) for r in rows]
        ax.loglog(M, y, mk + "-", label=lab, lw=1.4, ms=5)
    ax.axhline(1e-6, color="k", ls=":", lw=1.0)
    ax.text(M[0], 1.6e-6, "1e-6 audit threshold", fontsize=8)
    ax.set_xlabel("markers $M$")
    ax.set_ylabel(r"$|\Sigma \Delta s - L_{\rm exact}|/L_{\rm exact}$")
    ax.set_title("(a) measure exactness of the curved-surface discretisation")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ell = quad["cases"]["ellipse_nonuniform"]
    M2 = [r["n_markers"] for r in ell]
    ax.loglog(M2, [max(r["quadrature_rel_error"], 1e-12) for r in ell], "o-",
              label=r"quadrature $\oint \tau\,\hat n\,ds$", lw=1.4)
    ax.loglog(M2, [max(r["unweighted_rel_error"], 1e-12) for r in ell], "s--",
              label="legacy unweighted marker sum", lw=1.4)
    ax.axhline(1e-2, color="crimson", ls=":", lw=1.0)
    ax.set_xlabel("markers $M$ (equal-angle ellipse, spacing ratio 2.5)")
    ax.set_ylabel("rel. error of $\\oint \\tau\\,\\hat n\\,ds$")
    ax.set_title("(b) non-uniform markers: measure-weighted vs unweighted")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG / "fig1_surface_quadrature.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_cd_markers(cases) -> Path | None:
    # marker-convergence plot: base box only (other boxes are separate studies)
    cyl = [
        c for c in cases
        if c["label"].startswith("re100")
        and c["params"]["box_in_diameters"] == [15.0, 10.0]
    ]
    if not cyl:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    ax = axes[0]
    ax.axhspan(CD_LIT["low"], CD_LIT["high"], color="0.85",
               label="literature band (Tritton 1.26 – Braza 1.36)")
    ax.axhline(CD_LIT["ref"], color="0.35", ls="--", lw=1.0,
               label="Rajani et al. (2009), $C_d=1.34$")
    for scheme, style, col in (("quadrature", "o-", "C0"), ("raw", "s--", "C3")):
        rows = sorted(
            [c for c in cyl if c["params"]["scheme"] == scheme],
            key=lambda c: c["params"]["n_markers"],
        )
        if not rows:
            continue
        M = [c["params"]["n_markers"] for c in rows]
        cd = [c["metrics"]["cd_body_mean"] for c in rows]
        err = [c["metrics"]["cd_body_std"] or 0.0 for c in rows]
        ax.errorbar(M, cd, yerr=err, fmt=style, color=col, lw=1.4, capsize=3,
                    label=f"{scheme} forcing")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("surface markers $M$")
    ax.set_ylabel("$C_d$ (Re = 100)")
    ax.set_title("(a) marker-count convergence of the drag integral")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for scheme, style, col in (("quadrature", "o-", "C0"), ("raw", "s--", "C3")):
        rows = sorted(
            [c for c in cyl if c["params"]["scheme"] == scheme],
            key=lambda c: c["params"]["n_markers"],
        )
        if not rows:
            continue
        M = [c["params"]["n_markers"] for c in rows]
        err = [
            100.0 * (c["metrics"]["cd_body_mean"] - CD_LIT["ref"]) / CD_LIT["ref"]
            for c in rows
        ]
        ax.semilogx(M, err, style, color=col, lw=1.4, base=2,
                    label=f"{scheme} forcing")
    ax.axhspan(-15, 15, color="honeydew", label="±15 % acceptance band")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xlabel("surface markers $M$")
    ax.set_ylabel("error vs $C_d=1.34$  [%]")
    ax.set_title("(b) deviation from the literature anchor")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG / "fig2_cd_marker_convergence.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_conservation(cases) -> Path | None:
    cyl = [c for c in cases if c["label"].startswith("re100")]
    if not cyl:
        return None
    cyl.sort(key=lambda c: c["label"])
    labels = [c["label"].replace("re100_", "") for c in cyl]
    mom = [abs(c["metrics"]["momentum_balance_rel_error"] or 0.0) for c in cyl]
    leak = [abs(c["metrics"]["leak_rate_net_mean"] or 0.0) for c in cyl]
    loc = [c["metrics"]["leak_rate_local_mean"] or 0.0 for c in cyl]
    clo = [c["metrics"]["forcing_closure_rel_mean"] or 0.0 for c in cyl]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    x = np.arange(len(cyl))
    ax.bar(x - 0.2, np.maximum(mom, 1e-16), 0.4, label="momentum balance (global)")
    ax.bar(x + 0.2, np.maximum(leak, 1e-16), 0.4, label="net mass leak")
    ax.set_yscale("log")
    ax.axhline(1e-2, color="crimson", ls="--", lw=1.0, label="1 % (momentum)")
    ax.axhline(1e-3, color="navy", ls=":", lw=1.0, label="0.1 % (mass)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("relative error [-]")
    ax.set_title("(a) conservation criteria per case")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.bar(x - 0.2, np.maximum(clo, 1e-18), 0.4, label="Eulerian vs Lagrangian budget")
    ax.bar(x + 0.2, np.maximum(loc, 1e-18), 0.4, label="local impermeability $\\oint|\\rho u_n|ds$")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("relative error [-]")
    ax.set_title("(b) forcing-budget closure and local slip")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = FIG / "fig3_conservation_audit.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_fields(cases) -> Path | None:
    quad = next((c for c in cases if c["label"] == "re100_quad_M64"), None)
    raw = next((c for c in cases if c["label"] == "re100_raw_M64"), None)
    if quad is None or "fields" not in quad or not quad["fields"]:
        return None
    f = quad["fields"]
    ny, nx = f["shape"]
    s = f["stride"]
    vort = np.array(f["vorticity"])
    fig = plt.figure(figsize=(11.6, 4.6))
    ax = fig.add_subplot(1, 2, 1)
    extent = (0, nx, 0, ny)
    vmax = max(1e-9, float(np.abs(vort).max()))
    im = ax.imshow(vort, origin="lower", extent=extent, cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    ax.set_title(f"(a) vorticity, Re=100, M=64 quadrature forcing\n"
                 f"{nx}×{ny}, every {s}th node")
    ax.set_xlabel("x [lattice units]")
    ax.set_ylabel("y [lattice units]")
    fig.colorbar(im, ax=ax, fraction=0.035, label=r"$\omega\,\Delta t$")

    ax = fig.add_subplot(1, 2, 2, projection="polar")
    th = np.array(f["marker_theta"])
    order = np.argsort(th)
    th = th[order]
    tr = np.array(f["traction"])[order]
    pres = np.array(f["pressure_proxy"])[order]
    ax.plot(th, pres, "-", lw=1.6, label=r"normal traction $\tau_n/U^2$")
    if raw is not None and raw.get("fields"):
        th2 = np.array(raw["fields"]["marker_theta"])
        o2 = np.argsort(th2)
        pres2 = np.array(raw["fields"]["pressure_proxy"])[o2]
        ax.plot(th2[o2], pres2, "--", lw=1.2,
                label=r"legacy unweighted $\tau_n/U^2$")
    ax.set_title("(b) wall-normal traction around the cylinder")
    ax.set_rlabel_position(135)
    ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(-0.15, -0.1))
    del tr
    fig.tight_layout()
    out = FIG / "fig4_vorticity_traction.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_series(cases) -> Path | None:
    quad = next((c for c in cases if c["label"] == "re100_quad_M64"), None)
    if quad is None:
        return None
    ser = quad["series"]
    step = np.array(ser["step"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    ax = axes[0]
    ax.plot(step, ser["cd_body"], lw=1.0, label=r"$C_d$ from marker budget")
    ax.plot(step, ser["cd_drive"], lw=1.0, ls="--", label=r"$C_d$ from drive force")
    ax.axhline(CD_LIT["ref"], color="0.4", ls=":", lw=1.0, label="$C_d=1.34$")
    ax.axvline(quad["params"]["n_transient"], color="k", ls="-.", lw=0.8,
               label="end of transient")
    ax.set_xlabel("time step")
    ax.set_ylabel("$C_d$")
    ax.set_title("(a) drag history and global momentum balance")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.semilogy(step, np.maximum(ser["leak_net"], 1e-18), lw=1.0, label="net mass leak")
    ax.semilogy(step, np.maximum(ser["leak_local"], 1e-18), lw=1.0,
                label="local impermeability")
    ax.semilogy(step, np.maximum(ser["closure_rel"], 1e-18), lw=1.0,
                label="Eulerian−Lagrangian budget")
    ax.axhline(1e-3, color="navy", ls=":", lw=1.0, label="0.1 % mass-leak bound")
    ax.set_xlabel("time step")
    ax.set_ylabel("relative error [-]")
    ax.set_title("(b) mass-conservation audit")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG / "fig5_conservation_history.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_hasimoto(cases) -> Path | None:
    stk = [c for c in cases if c["label"].startswith("hasimoto")]
    if not stk:
        return None
    stk.sort(key=lambda c: c["params"]["box_in_diameters"][0])
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    Ls = [c["params"]["box_in_diameters"][0] for c in stk]
    meas = [c["metrics"]["cd_body_mean"] for c in stk]
    has = [c["analytic"]["hasimoto"]["cd"] for c in stk]
    unb = [c["analytic"]["unbounded_oseen"]["cd"] for c in stk]
    ax.semilogx(Ls, has, "o-", label="Hasimoto (1959) square array (analytic)", lw=1.5)
    ax.semilogx(Ls, meas, "s-", label="LBM-IBM, measure-consistent forcing", lw=1.5)
    ax.axhline(unb[0], color="0.5", ls="--", lw=1.0,
               label="unbounded Oseen/K–L asymptote (not representable)")
    for x, y, h in zip(Ls, meas, has):
        ax.annotate(f"{100 * (y - h) / h:+.1f}%", (x, y), textcoords="offset points",
                    xytext=(6, -12), fontsize=8)
    ax.set_xlabel("periodic cell size $L/D$ (Re = 0.1)")
    ax.set_ylabel("$C_d$")
    ax.set_title("Stokes-regime drag vs periodic-cell analytic anchor")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = FIG / "fig6_hasimoto_stokes.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_couette(cou: dict | None) -> Path | None:
    if not cou:
        return None
    samples = cou["series"]["traction_samples"]
    if not samples:
        return None
    arr = np.array(samples)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    ax = axes[0]
    ax.plot(arr.T, color="C0", alpha=0.25, lw=0.6)
    ax.axhline(abs(cou["metrics"]["traction_exact_inner"]), color="crimson", ls="--",
               lw=1.4, label=r"exact $|\tau_\theta| = 2\mu B/R_1^2$")
    ax.set_xlabel("marker index on the inner cylinder")
    ax.set_ylabel(r"$\tau_\theta$ (load on fluid)")
    ax.set_title("(a) reconstructed wall traction (all sampled steps)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    tor = np.array(cou["series"]["torque_samples"])
    ax.plot(tor / cou["metrics"]["torque_exact"], lw=1.2)
    ax.axhline(1.0, color="crimson", ls="--", lw=1.2, label="exact torque")
    ax.set_xlabel("sample")
    ax.set_ylabel(r"$T/T_{\rm exact}$")
    ax.set_title("(b) torque history, Taylor–Couette (Stokes regime)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG / "fig7_couette_traction_torque.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    quad = _load("quadrature_audit.json")
    cou = _load("case_couette_torque.json")
    cases = _cases()
    made = [
        fig_surface_measure(quad),
        fig_cd_markers(cases),
        fig_conservation(cases),
        fig_fields(cases),
        fig_series(cases),
        fig_hasimoto(cases),
        fig_couette(cou),
    ]
    for p in made:
        if p:
            print(f"[figure] {p.relative_to(ROOT)}")
    if not any(made):
        print("[figure] nothing produced — artifacts missing")


if __name__ == "__main__":
    main()