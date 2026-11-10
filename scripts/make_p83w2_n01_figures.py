#!/usr/bin/env python3
"""Figures for the p83w2-n01 compressibility-error-law report.

Every number plotted is read from ``artifacts/p83w2_n01/verification.json`` (and
the field snapshots written by the campaign driver); nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TOL = 0.03


def _fig_to_b64(fig) -> str:
    from io import BytesIO

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fig_error_law(ver: dict, out: Path) -> dict:
    per_re = ver["per_re"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(per_re)))
    ma_guide = np.array([0.03, 0.5])
    c0 = None
    colors = list(colors) * (1 + len(per_re))
    for (re_key, rec), col in zip(sorted(per_re.items(), key=lambda kv: float(kv[0])), colors):
        law = rec["error_law"]
        ma = np.array(law.get("mach", []))
        eps = np.array(law.get("eps", []))
        keep = np.isfinite(ma) & np.isfinite(eps) & (ma > 0) & (eps > 0)
        ma, eps = ma[keep], eps[keep]
        if ma.size < 2:
            continue
        ax.loglog(ma, eps, "o", color=col, ms=6, label=f"Re={float(re_key):g} measured")
        mm = np.linspace(ma.min(), ma.max(), 50)
        ax.loglog(mm, law["c"] * mm ** law["n"], "-", color=col, lw=1.4,
                  label=(f"fit n={law['n']:.3f} "
                         f"[{law['n_ci95'][0]:.2f},{law['n_ci95'][1]:.2f}] "
                         f"R$^2$={law['r2_loglog']:.4f}"))
        if c0 is None:
            c0 = (ma[0], eps[0])
    if c0 is not None:
        ax.loglog(ma_guide, c0[1] * (ma_guide / c0[0]) ** 2, "k:", lw=1.0,
                  label="slope $n=2$ guide")
    ax.set_xlabel("Mach number Ma")
    ax.set_ylabel(r"$|Nu(Ma)/Nu_0 - 1|$")
    ax.set_title("Compressibility error law of the cylinder Nusselt number")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, loc="lower right")
    name = "fig1_error_law.png"
    fig.savefig(out / name, dpi=130, bbox_inches="tight")
    b64 = _fig_to_b64(fig)
    return {
        "name": name,
        "caption": ("Measured relative error of the mean Nusselt number versus Mach "
                    "number for Re = 40, 100, 200 (same grid, same integrated "
                    "physical time).  Lines are the weighted log-log fits of "
                    "eps = C Ma^n; the dotted line is the theoretical slope n = 2 "
                    "from the linearised acoustic-mode analysis."),
        "png_base64": b64,
    }


def fig_nu_mach(ver: dict, out: Path) -> dict:
    per_re = ver["per_re"]
    fig, axes = plt.subplots(1, len(per_re), figsize=(4.2 * len(per_re), 3.8), sharey=False)
    if len(per_re) == 1:
        axes = [axes]
    for ax, (re_key, rec) in zip(axes, sorted(per_re.items(), key=lambda kv: float(kv[0]))):
        pts = [p for p in rec.get("points", []) if np.isfinite(p["nu"])]
        if not pts:
            ax.set_title(f"Re = {float(re_key):g} (no complete case)")
            ax.set_xlabel("Ma")
            continue
        ma = np.array([p["mach"] for p in pts])
        nu = np.array([p["nu"] for p in pts])
        err = np.array([p["nu_stderr"] if np.isfinite(p["nu_stderr"]) else 0.0 for p in pts])
        ax.errorbar(ma, nu, yerr=err, fmt="o-", ms=5, lw=1.2, label="LBM")
        nu0 = rec["richardson"]["nu0"]
        ax.axhline(nu0, color="C3", ls="--", lw=1.2,
                   label=f"Richardson $Nu_0$={nu0:.3f}")
        ax.axhline(rec["cb_reference"], color="C2", ls=":", lw=1.2,
                   label=f"Churchill-Bernstein {rec['cb_reference']:.3f}")
        ax.set_title(f"Re = {float(re_key):g}")
        ax.set_xlabel("Ma")
        ax.set_ylabel("mean Nu")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Mean Nusselt number versus Mach number", y=1.02)
    name = "fig2_nu_vs_mach.png"
    fig.savefig(out / name, dpi=130, bbox_inches="tight")
    b64 = _fig_to_b64(fig)
    return {
        "name": name,
        "caption": ("Mean Nusselt number versus Mach number at fixed grid and Re.  "
                    "Dashed: Ma->0 Richardson extrapolation Nu0 of the same "
                    "discretisation; dotted: the external Churchill-Bernstein "
                    "correlation (no fitting)."),
        "png_base64": b64,
    }


def fig_envelope(ver: dict, out: Path) -> dict:
    rows = ver["envelope"]["rows"]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    rows = [r for r in rows if np.isfinite(r["ma_band"][1])]
    if not rows:
        fig.text(0.5, 0.5, "no complete envelope row", ha="center")
        name = "fig3_mach_envelope.png"
        fig.savefig(out / name, dpi=130, bbox_inches="tight")
        return {"name": name, "caption": "Mach band (incomplete data)",
                "png_base64": _fig_to_b64(fig)}
    for i, row in enumerate(rows):
        lo, hi = row["ma_band"]
        re = row["re"]
        ax.plot([lo, hi], [i, i], "-", lw=8, alpha=0.35, color="C0",
                label="usable Ma band" if i == 0 else None)
        ax.plot([row["ma_max_error"]], [i], "v", color="C1", ms=8,
                label="accuracy bound (3 %)" if i == 0 else None)
        ax.plot([row["ma_min_stiffness"]], [i], "^", color="C4", ms=8,
                label=r"stiffness floor ($\tau\geq\tau_{min}$)" if i == 0 else None)
        law = row["error_law"]
        mm = np.linspace(max(row["ma_min_stiffness"] * 0.5, 1e-3), 0.5, 200)
        ax.plot(mm, np.full_like(mm, i) + 0.0, alpha=0.0)
        _ = law
    # measured stability markers
    for i, row in enumerate(rows):
        rec = ver["per_re"].get(str(int(row["re"]))) or ver["per_re"].get(str(float(row["re"])))
        if rec is None:
            continue
        for p in rec.get("points", []):
            if not np.isfinite(p["nu"]):
                continue
            ax.plot([p["mach"]], [i + 0.22], "o", ms=4,
                    color="C0" if not p["diverged"] else "C3")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"Re={r['re']:g}" for r in rows])
    ax.set_xlabel("Mach number Ma")
    ax.set_title("Mach operating envelope (scheduler-facing): band, accuracy bound, stiffness floor")
    ax.grid(alpha=0.25, axis="x")
    ax.legend(fontsize=7, loc="lower right")
    name = "fig3_mach_envelope.png"
    fig.savefig(out / name, dpi=130, bbox_inches="tight")
    b64 = _fig_to_b64(fig)
    return {
        "name": name,
        "caption": ("Usable Mach band per Reynolds number: thick bar = band, "
                    "down triangle = Ma at which the O(Ma^n) error reaches the 3 % "
                    "tolerance, up triangle = stiffness floor tau >= tau_min.  "
                    "Circles: measured cases (red = unusable/diverged)."),
        "png_base64": b64,
    }


def fig_grid(grid_dir: Path, out: Path) -> dict | None:
    files = sorted(grid_dir.glob("*_grid.json"))
    if not files:
        return None
    ds, nus, errs, cbs = [], [], [], []
    for p in files:
        rec = json.loads(p.read_text())
        ds.append(rec["case"]["diameter_eff"])
        nus.append(rec["nu_mean"])
        errs.append(rec.get("nu_stderr") or 0.0)
        cbs.append(rec["cb_reference"])
    order = np.argsort(ds)
    ds = np.array(ds)[order]
    nus = np.array(nus)[order]
    errs = np.array(errs)[order]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.errorbar(ds, nus, yerr=errs, fmt="o-", label="LBM (this work)")
    ax.axhline(cbs[0], color="C2", ls=":", label=f"Churchill-Bernstein {cbs[0]:.3f}")
    ax.fill_between([ds.min() * 0.9, ds.max() * 1.1],
                    cbs[0] * (1 - 0.08), cbs[0] * (1 + 0.08), color="C2", alpha=0.12,
                    label="8 % band")
    ax.set_xlabel("effective diameter $D_{eff}$ [lattice cells]")
    ax.set_ylabel("mean Nu")
    ax.set_title("Grid study at Re = 100, Pr = 0.7, Ma = 0.1")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    name = "fig4_grid_study.png"
    fig.savefig(out / name, dpi=130, bbox_inches="tight")
    b64 = _fig_to_b64(fig)
    return {
        "name": name,
        "caption": ("Grid study of the absolute Nusselt number at Re = 100, Ma = 0.1.  "
                    "The Ma-independent discretisation offset (staircase geometry, "
                    "half-cell flux sampling radius) is what separates Nu from the "
                    "correlation; it cancels in the error law."),
        "png_base64": b64,
    }


def fig_fields(field_dir: Path, out: Path) -> dict | None:
    files = sorted(field_dir.glob("field_*.npz"))
    if not files:
        return None
    fig, axes = plt.subplots(len(files), 2, figsize=(9.0, 3.2 * len(files)), squeeze=False)
    for i, p in enumerate(files):
        z = np.load(p)
        T, ux, uy = z["T"], z["ux"], z["uy"]
        tag = p.stem.replace("field_", "")
        vort = np.gradient(uy, axis=1) - np.gradient(ux, axis=0)
        im0 = axes[i, 0].imshow(T, origin="lower", cmap="inferno")
        axes[i, 0].set_title(f"{tag}: temperature")
        fig.colorbar(im0, ax=axes[i, 0], fraction=0.035)
        vmax = np.percentile(np.abs(vort), 99.5)
        im1 = axes[i, 1].imshow(vort, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[i, 1].set_title(f"{tag}: vorticity")
        fig.colorbar(im1, ax=axes[i, 1], fraction=0.035)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    name = "fig5_fields.png"
    fig.savefig(out / name, dpi=130, bbox_inches="tight")
    b64 = _fig_to_b64(fig)
    return {
        "name": name,
        "caption": ("Temperature and vorticity fields of the heated cylinder at "
                    "Re = 100 for two Mach numbers on the same grid; the temperature "
                    "field is Mach independent to leading order, the density and the "
                    "wake unsteadiness are not."),
        "png_base64": b64,
    }


def fig_holdout(ver: dict, out: Path) -> dict | None:
    ho = ver.get("holdout")
    if not ho:
        return None
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    xs = [0, 1]
    ax.bar(xs, [ho["ma_pred"], ho["measured_critical_ma"]], color=["C0", "C1"], width=0.55)
    ax.axhspan(ho["ma_pred"] / 1.5, ho["ma_pred"] / 0.67, color="C0", alpha=0.10,
               label="acceptance band (ratio 0.67-1.5)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"predicted\nfrom Re={','.join(f'{r:g}' for r in ho['train_re'])}",
                        f"measured\nat Re={ho['re_target']:g}"])
    ax.set_ylabel("Ma at which the error reaches 3 %")
    ax.set_title(f"Hold-out: ratio = {ho['ratio_pred_over_measured']:.2f}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, axis="y")
    name = "fig6_holdout.png"
    fig.savefig(out / name, dpi=130, bbox_inches="tight")
    b64 = _fig_to_b64(fig)
    return {
        "name": name,
        "caption": ("Pre-registered hold-out check at the held-out Reynolds number: the "
                    "critical Mach number predicted from the training Reynolds numbers "
                    "versus the one measured afterwards."),
        "png_base64": b64,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/p83w2_n01")
    args = ap.parse_args()
    root = Path(args.root)
    ver = json.loads((root / "verification.json").read_text())
    figs = [fig_error_law(ver, root), fig_nu_mach(ver, root), fig_envelope(ver, root)]
    grid_fig = fig_grid(root / "grid", root)
    if grid_fig:
        figs.append(grid_fig)
    field_fig = fig_fields(root / "fields", root)
    if field_fig:
        figs.append(field_fig)
    ho_fig = fig_holdout(ver, root)
    if ho_fig:
        figs.append(ho_fig)
    (root / "figures.json").write_text(json.dumps(
        [{"name": f["name"], "caption": f["caption"]} for f in figs], indent=2) + "\n")
    for f in figs:
        (root / f["name"]).write_bytes(base64.b64decode(f["png_base64"]))
    print(f"wrote {len(figs)} figures to {root}")


if __name__ == "__main__":
    main()
