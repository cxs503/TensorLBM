#!/usr/bin/env python3
"""Generate the figures for XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION.

All numbers are read from ``artifacts/result.json``; the 2-D field snapshot is
re-computed deterministically (same protocol, R = 20, L = 4R) so the maps in
the report cannot drift from the tables.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
FIG = ART / "figures"
FIG.mkdir(parents=True, exist_ok=True)

res = json.loads((ART / "result.json").read_text())
CFG = res["config"]
LOCKED = CFG["sigma_locked_doc"]


def fig_field():
    import run_xval as X

    sol = X.TorchSolver()
    rho0, rr = X.init_droplet(80, 20.0)
    f = sol.init(rho0)
    for _ in range(5000):
        f = sol.step(f)
    rho, ux, uy = sol.fields(f)
    p = X.eos_pressure(rho)
    m = X.measure((rho, ux, uy), rr, 20.0)
    # radial profiles (same binning as run_case)
    r2 = np.sqrt(rr)
    nb = 60
    edges = np.linspace(0.0, 80 / 2.0, nb + 1)
    idx = np.clip(np.digitize(r2.ravel(), edges) - 1, 0, nb - 1)
    cnt = np.maximum(np.bincount(idx, minlength=nb), 1)
    m["profile"] = {
        "r": (0.5 * (edges[:-1] + edges[1:])).tolist(),
        "rho": (np.bincount(idx, weights=rho.ravel(), minlength=nb) / cnt).tolist(),
        "p": (np.bincount(idx, weights=p.ravel(), minlength=nb) / cnt).tolist(),
    }

    fig, ax = plt.subplots(2, 2, figsize=(10.5, 8.6))
    im0 = ax[0, 0].imshow(rho, origin="lower", cmap="viridis")
    ax[0, 0].set_title(r"density field $\rho(x,y)$  (R=20, L=80, 5000 steps)")
    fig.colorbar(im0, ax=ax[0, 0], fraction=0.046)
    im1 = ax[0, 1].imshow(p, origin="lower", cmap="inferno")
    ax[0, 1].set_title(r"SC EOS pressure $p=c_s^2\rho+\frac{G_{eff}c_s^2}{2}\psi^2$")
    fig.colorbar(im1, ax=ax[0, 1], fraction=0.046)
    ax[0, 0].set_xlabel("x"); ax[0, 0].set_ylabel("y")
    ax[0, 1].set_xlabel("x"); ax[0, 1].set_ylabel("y")

    prof = m["profile"]
    r = np.array(prof["r"])
    ax[1, 0].plot(r, prof["rho"], "o-", ms=3, label=r"$\rho(r)$ (angular average)")
    ax[1, 0].axvline(0.5 * m["R_eq"], color="k", ls="--", lw=1, label=r"$0.5R_{eq}$ band")
    ax[1, 0].axvline(1.5 * m["R_eq"], color="k", ls=":", lw=1, label=r"$1.5R_{eq}$ band")
    ax[1, 0].set_xlabel("r / lattice units"); ax[1, 0].set_ylabel(r"$\rho$")
    ax[1, 0].legend(fontsize=8); ax[1, 0].set_title("radial density profile")

    ax[1, 1].plot(r, prof["p"], "s-", ms=3, color="C3", label=r"$p(r)$")
    ax[1, 1].axhline(m["p_in"], color="C0", ls="--", lw=1, label=rf"$p_{{in}}$={m['p_in']:.5f}")
    ax[1, 1].axhline(m["p_out"], color="C2", ls="--", lw=1, label=rf"$p_{{out}}$={m['p_out']:.5f}")
    ax[1, 1].axhline(res["maxwell"]["p_c"], color="k", ls=":", lw=1,
                     label=rf"$p_c$ (Maxwell)={res['maxwell']['p_c']:.5f}")
    ax[1, 1].set_xlabel("r / lattice units"); ax[1, 1].set_ylabel("p")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title(rf"$\Delta p={m['dp']:.5f}$,  $R_{{eq}}={m['R_eq']:.2f}$,  "
                       rf"$u_{{max}}={m['u_max']:.3f}$")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_droplet_field.png", dpi=130)
    plt.close(fig)
    return m


def fig_laplace():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    marks = {"tensorlbm": ("o", "C0"), "ref_numpy": ("x", "C1")}
    for name, fit in res["laplace"].items():
        R = np.array([q["R_eq"] for q in fit["per_radius"]])
        dp = np.array([q["dp"] for q in fit["per_radius"]])
        mk, col = marks.get(name, ("s", "C2"))
        ax[0].plot(1.0 / R, dp, mk, color=col, ms=7, label=f"{name} (measured)")
        xx = np.linspace(0, 1.02 / R.min(), 50)
        ax[0].plot(xx, fit["sigma_through_origin"] * xx, "-", color=col, lw=1,
                   label=rf"{name}: through-origin $\sigma$={fit['sigma_through_origin']:.5f}")
        ax[0].plot(xx, fit["sigma_free_intercept"] * xx + fit["intercept"], "--", color=col, lw=1,
                   label=rf"{name}: free fit $\sigma$={fit['sigma_free_intercept']:.5f}")
    ax[0].set_xlabel(r"$1/R_{eq}$"); ax[0].set_ylabel(r"$\Delta p$")
    ax[0].set_title("Young-Laplace calibration  " + r"$\Delta p=\sigma/R$")
    ax[0].legend(fontsize=7)
    # residual panel (relative to the through-origin fit), tensorlbm
    fit = res["laplace"]["tensorlbm"]
    R = np.array([q["R_eq"] for q in fit["per_radius"]])
    dev = np.array([q["dev_pct_through_origin"] for q in fit["per_radius"]])
    ax[1].plot(R, dev, "o-", color="C0", label="through-origin fit")
    devf = np.array([q["dev_pct_free_intercept"] for q in fit["per_radius"]])
    ax[1].plot(R, devf, "s--", color="C1", label="free-intercept fit")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].axhline(3, color="r", ls=":", lw=1); ax[1].axhline(-3, color="r", ls=":", lw=1)
    ax[1].axhline(15, color="grey", ls=":", lw=1); ax[1].axhline(-15, color="grey", ls=":", lw=1)
    ax[1].set_xlabel(r"$R_{eq}/(\mathrm{lattice\ units})$")
    ax[1].set_ylabel(r"$\sigma_i$ deviation / %")
    ax[1].set_title(rf"per-radius $\sigma_i=\Delta p\,R$  (red dots: $\pm3\%$, grey: $\pm15\%$)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_laplace_fit.png", dpi=130)
    plt.close(fig)


def fig_sigma_vs_R():
    fit = res["laplace"]["tensorlbm"]
    R = np.array([q["R_eq"] for q in fit["per_radius"]])
    sig = np.array([q["sigma_i"] for q in fit["per_radius"]])
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.plot(R, sig, "o-", color="C0", label=r"$\sigma_i=\Delta p\,R_{eq}$ (velocity-shift)")
    ax.axhline(LOCKED, color="k", ls="--", lw=1,
               label=rf"locked library value $\sigma$={LOCKED:.4f}")
    ax.axhline(fit["sigma_through_origin"], color="C0", ls=":", lw=1,
               label=rf"through-origin fit {fit['sigma_through_origin']:.5f}")
    ax.axhline(fit["sigma_free_intercept"], color="C1", ls="-.", lw=1,
               label=rf"free-intercept fit {fit['sigma_free_intercept']:.5f}")
    ax.fill_between([R.min() - 1, R.max() + 1], LOCKED * 0.97, LOCKED * 1.03, color="r",
                    alpha=0.10, label=r"$\pm3\%$ band")
    ax.set_xlim(R.min() - 1, R.max() + 1)
    ax.set_xlabel(r"$R_{eq}$ / lattice units"); ax.set_ylabel(r"$\sigma$ (lattice units)")
    ax.set_title("curvature (finite-R) trend of the calibrated surface tension")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_sigma_vs_R.png", dpi=130)
    plt.close(fig)


def fig_sensitivity():
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.9))
    # (a) band sensitivity, R = 24 case
    rows = res["laplace"]["tensorlbm"]["rows"]
    r24 = max(rows, key=lambda q: q["R_init"])
    keys = list(r24["band_sensitivity"].keys())
    vals = [r24["band_sensitivity"][k] for k in keys]
    ax[0].bar(range(len(keys)), vals, color="C0")
    ax[0].set_xticks(range(len(keys))); ax[0].set_xticklabels(keys, fontsize=8)
    ax[0].set_ylabel(r"$\Delta p$"); ax[0].set_xlabel("pressure band  (inner-outer) x $R_{eq}$")
    ax[0].set_title(rf"band sensitivity (R={r24['R_init']:.0f})")
    for i, v in enumerate(vals):
        ax[0].text(i, v, f"{v:.5f}", ha="center", va="bottom", fontsize=7)
    ax[0].set_ylim(0, max(vals) * 1.18)
    # (b) domain extent
    ds = res.get("domain_study", {})
    if ds.get("rows"):
        Ls = [q["L"] for q in ds["rows"]]
        dps = [q["dp"] for q in ds["rows"]]
        ax[1].plot([l / 16.0 for l in Ls], dps, "o-", color="C2")
        for x, y, l in zip([l / 16.0 for l in Ls], dps, Ls):
            ax[1].annotate(f"L={l}", (x, y), textcoords="offset points", xytext=(0, 7), fontsize=7)
        ax[1].set_xlabel("domain extent  L/R"); ax[1].set_ylabel(r"$\Delta p$")
        ax[1].set_title(rf"domain extent (R=16): spread {ds['dp_spread_pct']:.2f}%")
    # (c) forcing scheme
    fs = res.get("forcing_study", {})
    if fs.get("rows"):
        labs = [q["forcing"] for q in fs["rows"]]
        sig = [q["sigma_i"] for q in fs["rows"]]
        ax[2].bar(labs, sig, color=["C0", "C4"])
        for i, v in enumerate(sig):
            ax[2].text(i, v, f"{v:.5f}", ha="center", va="bottom", fontsize=8)
        ax[2].axhline(LOCKED, color="k", ls="--", lw=1, label=f"locked {LOCKED:.4f}")
        ax[2].set_ylabel(r"$\sigma_i=\Delta p\,R_{eq}$")
        ax[2].set_title(rf"forcing scheme (R=16, L=96): spread {fs['dp_spread_pct']:.1f}%")
        ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_sensitivity.png", dpi=130)
    plt.close(fig)


def write_provenance() -> dict:
    """Stamp the committed figures against the result files they were built from.

    ``tests/test_xval_sc_laplace_artifacts.py`` compares these digests with the
    bytes on disk, so a figure that has gone stale with respect to
    ``artifacts/result.json`` (or a result file edited without regenerating the
    figures) is caught instead of silently shipping.
    """

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    prov = {
        "anchor": res.get("anchor", HERE.name),
        "generator": "make_figures.py",
        "result_json": {"sha256": sha256(ART / "result.json")},
        "field_snapshot": {"sha256": sha256(ART / "field_snapshot_R20.json")},
        "figures": {
            p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(FIG.glob("*.png"))
        },
    }
    (FIG / "PROVENANCE.json").write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
    return prov


if __name__ == "__main__":
    m = fig_field()
    fig_laplace()
    fig_sigma_vs_R()
    fig_sensitivity()
    (ART / "field_snapshot_R20.json").write_text(
        json.dumps({k: m[k] for k in ("R_eq", "dp", "p_in", "p_out", "rho_in", "rho_out", "u_max")},
                   indent=2)
    )
    prov = write_provenance()
    print("figures ->", sorted(prov["figures"]))
    print("provenance ->", FIG / "PROVENANCE.json")
    print(f"field snapshot: R_eq={m['R_eq']:.3f} dp={m['dp']:.6f} u_max={m['u_max']:.4f}")
