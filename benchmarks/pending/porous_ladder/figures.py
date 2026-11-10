#!/usr/bin/env python3
"""Figures for the p339w2-n06 porous-media verification ladder.

Reads ``result.json`` (machine-written by ``report.py``) plus the raw case
files and renders:

  fig1_kozeny_carman.png   K/d_p^2 vs porosity: lattice vs KC-180 / KC-150
  fig2_ergun.png           measured pressure gradient vs Ergun's two terms
  fig3_interface.png       port-A/profiles + port-B closure + Brinkman decay
  fig4_geometry.png        digitised pack and BCC bed slices (geometry audit)

Every plotted number comes from the JSON artifacts (zero hand-copying).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def load(cases: Path) -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted(cases.glob("*.json"))}


def fig_kozeny(res: dict, out: Path) -> Path:
    tiers = res["gates"][0]["detail"]["tiers"]
    fine = res["gates"][0]["detail"]["grid_refinement_phi045"]
    phi = np.linspace(0.30, 0.60, 200)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(phi, phi**3 / (180 * (1 - phi) ** 2), "-", lw=2, label="Kozeny-Carman, C=180")
    ax.plot(phi, phi**3 / (150 * (1 - phi) ** 2), "--", lw=1.2, label="Ergun-viscous, C=150")
    xs = [v["phi_nominal"] for v in tiers.values()]
    ys = [v["K_over_dp2"] for v in tiers.values()]
    ax.plot(xs, ys, "o", ms=7, mfc="white", mew=1.6, label="LBM, resolved spheres (d$_p$=16)")
    if isinstance(fine, dict):
        ax.plot([fine["fine"]["phi_nominal"] if "phi_nominal" in fine["fine"] else 0.45],
                [fine["fine"]["K_over_dp2"]], "s", ms=7, mfc="none",
                label="LBM refined grid")
    for k, v in tiers.items():
        ax.annotate(f"{100*v['err_vs_kc180']:+.0f}%", (v["phi_nominal"], v["K_over_dp2"]),
                    textcoords="offset points", xytext=(6, -12), fontsize=8)
    ax.set_xlabel("porosity $\\phi$")
    ax.set_ylabel("$K / d_p^2$")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title("G1 Kozeny-Carman ladder (annotated = error vs C=180)")
    fig.tight_layout()
    p = out / "p339w2_n06_fig1_kozeny_carman.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def fig_ergun(res: dict, out: Path) -> Path:
    tiers = res["gates"][1]["detail"]["tiers"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    labels, meas, pred, visc, inert = [], [], [], [], []
    for k, v in tiers.items():
        labels.append(f"Re$_p$={v['re_pore_measured']:.0f}")
        meas.append(v["grad_measured"])
        pred.append(v["grad_ergun"])
        visc.append(v["grad_ergun_viscous"])
        inert.append(v["grad_ergun_inertial"])
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, meas, w, label="LBM (momentum balance)")
    ax.bar(x + w / 2, pred, w, label="Ergun total, C$_v$=150, C$_i$=1.75")
    ax.bar(x + w / 2, visc, w * 0.0, color="none")
    ax.plot(x + w / 2, visc, "_", ms=22, mew=2, color="k", label="Ergun viscous part")
    ax.plot(x + w / 2, [vi + ii for vi, ii in zip(visc, inert)], "_", ms=22, mew=2,
            color="tab:red", label="viscous + inertial")
    for xi, (m, p) in enumerate(zip(meas, pred)):
        ax.annotate(f"{100*(m/p-1):+.1f}%", (xi, max(m, p) * 1.02), ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("$-dp/dx$  [lattice]")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7.5)
    title = "G2 Ergun inertial deviation ("
    title += ", ".join(f"{100*v['inertial_fraction']:.0f}% inertial" for v in tiers.values())
    ax.set_title(title + ")", fontsize=9)
    fig.tight_layout()
    p = out / "p339w2_n06_fig2_ergun.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def fig_interface(res: dict, cases: dict, out: Path) -> Path:
    bp, bs = cases["bj_pressure"], cases["bj_shear"]
    g3, g4, g5 = res["gates"][2], res["gates"][3], res["gates"][4]
    y_int = bs["y_int"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axes[0]
    for rec, lab, style in ((bp, "port A: pressure-driven", "-"),
                            (bs, "port A: shear-driven", "-")):
        y = np.asarray(rec["profile_y"])
        u = np.asarray(rec["profile_u"])
        ax.plot(u, y, style, lw=1.6, label=lab)
    # port B blind prediction of the pressure port
    K = g3["detail"]["K_bed"]
    phi = g3["detail"]["phi_bed"]
    alpha = g3["detail"]["alpha_shear"]
    import sys
    sys.path.insert(0, str(HERE.parents[2] / "src"))
    import tensorlbm.porous_ladder as pl
    pred = pl.composite_brinkman_profile(
        K=K, phi=phi, alpha=alpha, y_int=y_int,
        free_top=float(bp["profile_y"][-1] + 1.0), nu=bs["nu"],
        accel_free=bp["accel"], accel_bed=bp["accel"],
    )
    ax.plot(pred["u_super"], pred["y"], "--", lw=1.4,
            label=f"port B (K={K:.1f}, $\\alpha$={alpha:.2f} from shear)")
    ax.axhline(y_int, color="k", lw=0.7, ls=":")
    ax.text(0.02, y_int + 1.5, "interface", fontsize=7)
    ax.set_xlabel("$u_x$ (superficial average)")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)
    ax.set_title(f"G3/G5 interface closure (dev={100*g5['error']:.1f}%)", fontsize=9)

    ax = axes[1]
    y = np.asarray(bs["profile_y"])
    u = np.asarray(bs["profile_u"])
    sel = (u > 1e-4 * u.max())
    ax.semilogy(y[sel], u[sel], "o", ms=2.5, label="port A, slab-averaged")
    lam = g4["detail"]["lambda_measured"]
    band = g4["detail"]["decay_fit"].get("band")
    if band and np.isfinite(lam):
        yy = np.linspace(band[0], band[1], 50)
        u0 = u[sel][np.argmin(np.abs(y[sel] - band[0]))]
        ax.semilogy(yy, u0 * np.exp((yy - band[0]) / lam), "-",
                    label=f"exp fit $\\lambda$={lam:.2f}")
    else:
        ax.semilogy([], [], "-", label="decay fit not identifiable")
    dspec = g4["detail"]["delta_sqrt_K_over_phi"]
    ax.axvline(y_int, color="k", lw=0.7, ls=":")
    ax.set_xlabel("y")
    ax.set_ylabel("$u_x$ (superficial average)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)
    ax.set_title(f"G4 Brinkman decay: $\\lambda$/{dspec:.2f}"
                 .replace("{dspec:.2f}", f"{dspec:.2f}"), fontsize=9)
    fig.tight_layout()
    p = out / "p339w2_n06_fig3_interface.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def fig_geometry(cases: dict, out: Path) -> Path:
    pack = cases["kc_045"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    # rebuild the same pack deterministically from the recorded parameters
    import sys
    sys.path.insert(0, str(HERE.parents[2] / "src"))
    import tensorlbm.porous_ladder as pl
    pk = pl.generate_sphere_pack((pack["L"],) * 3, pack["d_p"], pack["phi_target"], seed=1106)
    mask = pk.as_mask()
    nz, ny, nx = mask.shape
    for ax, (title, sl) in zip(axes, [
        (f"z = {nz//2}", mask[nz // 2, :, :]),
        (f"y = {ny//2}", mask[:, ny // 2, :]),
        (f"x = {nx//2}", mask[:, :, nx // 2]),
    ]):
        ax.imshow(sl, cmap="gray_r", origin="lower", interpolation="nearest")
        ax.set_title(f"{title}  ($\\phi$={pack['phi_mask']:.3f} mask, {pack['phi_nominal']:.3f} nominal)",
                     fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("G1 geometry audit: digitised random sphere pack ($d_p$=16, "
                 f"{pack['n_spheres']} spheres)", fontsize=9)
    fig.tight_layout()
    p = out / "p339w2_n06_fig4_geometry.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default=str(HERE / "result.json"))
    ap.add_argument("--cases", default=str(HERE / "cases"))
    ap.add_argument("--out", default=str(HERE.parents[2] / "docs" / "figures"))
    args = ap.parse_args()
    res = json.loads(Path(args.result).read_text())
    cases = load(Path(args.cases))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    made = [fig_kozeny(res, out), fig_ergun(res, out),
            fig_interface(res, cases, out), fig_geometry(cases, out)]
    payload = {
        "node": res["node"],
        "figures": [{"name": p.name, "caption": c} for p, c in zip(made, [
            "K/d_p^2 vs porosity; lattice points vs Kozeny-Carman C=180/150",
            "Ergun pressure gradient at two pore-Reynolds tiers",
            "Beavers-Joseph interface profiles (ports A/B) and Brinkman decay fit",
            "Digitised random sphere pack slices (porosity/surface audit)",
        ])],
    }
    (out / "p339w2_n06_figures.json").write_text(json.dumps(payload, indent=1))
    for p in made:
        print(p, p.stat().st_size)


if __name__ == "__main__":
    main()