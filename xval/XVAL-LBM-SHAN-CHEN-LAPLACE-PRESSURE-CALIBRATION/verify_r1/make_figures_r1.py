"""Figures for the R1 adversarial verification of the Shan-Chen Laplace calibration.

Panels
  A  droplet density field + radial rho(r), p(r) for velocity_shift vs guo
     (two short local runs; the curvatures differ because the two forcing
     schemes select different coexistence states)
  B  flat-interface coexistence densities (scheme-specific) vs the analytic
     Maxwell coexistence of the SC equation of state, plus the droplet bulk
     densities that must sit on the same curve
  C  sigma per forcing scheme and sigma(tau) with the +-3% band around the
     library-locked sigma = 0.0561
  D  Laplace law dp vs 1/R for the three schemes together with the first-pass
     (m20921) four-radius calibration

All numbers come from artifacts/verify_r1.json (written by verify_r1.py) or
from artifacts/field_profiles_r1.npz (written here with --fields).
Run:  python make_figures_r1.py [--fields]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
XV_DIR = HERE.parent
REPO = XV_DIR.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(XV_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import ref_sc as REF  # noqa: E402

CFG_LOCKED_SIGMA = 0.0561
J = HERE / "artifacts" / "verify_r1.json"
NPZ = HERE / "artifacts" / "field_profiles_r1.npz"


def compute_fields(steps: int = 5000, R: float = 16.0, L: int = 64):
    """Two short droplet runs (velocity_shift / guo) -> rho, p(r) profiles."""
    import run_xval as XV

    out = {}
    for fs in ("velocity_shift", "guo"):
        solver = XV.TorchSolver(forcing=fs, threads=4)
        rho0, rr = XV.init_droplet(L, R)
        f = solver.init(rho0)
        for _ in range(steps):
            f = solver.step(f)
        rho, ux, uy = solver.fields(f)
        p = REF.eos_pressure(rho, XV.CFG["G_eff"])
        edges = np.arange(0.0, L / 2 - 1, 0.5)
        rc = 0.5 * (edges[:-1] + edges[1:])
        prof_r, prof_rho, prof_p = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (rr >= lo) & (rr < hi)
            if m.sum() < 4:
                continue
            prof_r.append(0.5 * (lo + hi))
            prof_rho.append(float(rho[m].mean()))
            prof_p.append(float(p[m].mean()))
        out[f"rho_{fs}"] = rho
        out[f"p_{fs}"] = p
        out[f"r_{fs}"] = np.array(prof_r)
        out[f"rhor_{fs}"] = np.array(prof_rho)
        out[f"pr_{fs}"] = np.array(prof_p)
        # centre-line rho(y) through the droplet centre for panel A overlay
        out[f"centre_{fs}"] = rho[:, L // 2]
    NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(NPZ, **out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", action="store_true", help="run the two short field cases")
    args = ap.parse_args()

    res = json.loads(J.read_text())
    if args.fields or not NPZ.exists():
        compute_fields()
    fld = np.load(NPZ)

    MX = res["maxwell"]
    V1 = res["legs"]["V1_flat"]
    V2 = res["legs"]["V2_droplet"]
    V3 = res.get("legs", {}).get("V3_tau", [])
    schemes = list(V1.keys())
    COL = {"velocity_shift": "#1f77b4", "edm": "#2ca02c", "guo": "#d62728"}

    fig, ax = plt.subplots(2, 3, figsize=(16.5, 9))

    # --- A: fields + radial profiles -------------------------------------
    for k, fs in enumerate(("velocity_shift", "guo")):
        a = ax[0, k]
        im = a.imshow(fld[f"rho_{fs}"], origin="lower", cmap="viridis")
        a.set_title(f"A{k + 1}  rho(x,y), {fs}  (R_init=16, L=64)")
        a.set_xlabel("x")
        a.set_ylabel("y")
        fig.colorbar(im, ax=a, fraction=0.046, label="rho")
    a = ax[0, 2]
    for fs in ("velocity_shift", "guo"):
        a.plot(fld[f"r_{fs}"], fld[f"rhor_{fs}"], "o-", ms=3, color=COL[fs], label=fs)
    a.axhline(MX["rho_l"], ls="--", c="k", lw=1, label="Maxwell rho_l")
    a.axhline(MX["rho_v"], ls=":", c="k", lw=1, label="Maxwell rho_v")
    a.set_xlabel("r")
    a.set_ylabel("radial mean rho")
    a.set_yscale("log")
    a.set_title("A3  radial density profile, two forcings")
    a.legend(fontsize=8)

    # --- B: coexistence state per scheme --------------------------------
    a = ax[1, 0]
    xs = np.arange(len(schemes))
    rl = [V1[s]["rho_l"] for s in schemes]
    rv = [V1[s]["rho_v"] for s in schemes]
    a.bar(xs - 0.18, rl, 0.34, label="rho_l (flat)", color="#4c72b0")
    a.bar(xs + 0.18, rv, 0.34, label="rho_v (flat)", color="#dd8452")
    for i, s in enumerate(schemes):
        for j, (m, col) in enumerate(((V2[s][0]["rho_in"], "#4c72b0"),
                                      (V2[s][0]["rho_out"], "#dd8452"))):
            a.plot(i + (0.06 if j == 0 else -0.06), m, "k*", ms=11)
    a.axhline(MX["rho_l"], ls="--", c="#4c72b0", lw=1, label="Maxwell rho_l")
    a.axhline(MX["rho_v"], ls="--", c="#dd8452", lw=1, label="Maxwell rho_v")
    a.set_xticks(xs)
    a.set_xticklabels(schemes)
    a.set_ylabel("density")
    a.set_title("B  coexistence: flat plateaus (bars),\n droplet bulk (stars), Maxwell (dashed)")
    a.legend(fontsize=7)

    # --- C: sigma per scheme and sigma(tau) -----------------------------
    a = ax[1, 1]
    sig = [np.mean([r["sigma_from_dp"] for r in V2[s]]) for s in schemes]
    a.bar(xs, sig, 0.5, color=[COL[s] for s in schemes])
    a.axhline(CFG_LOCKED_SIGMA, ls="--", c="k", label="locked 0.0561")
    a.fill_between([-0.5, len(schemes) - 0.5], CFG_LOCKED_SIGMA * 0.97,
                   CFG_LOCKED_SIGMA * 1.03, color="k", alpha=0.08, label="+-3%")
    a.set_xticks(xs)
    a.set_xticklabels(schemes)
    a.set_ylabel("sigma = dp * R_eq")
    a.set_title("C1  surface tension per forcing scheme")
    a.set_xlim(-0.5, len(schemes) - 0.5)
    a.legend(fontsize=7)
    if V3:
        a = ax[1, 2]
        taus = [m["tau"] for m in V3]
        s3 = [m["sigma_from_dp"] for m in V3]
        a.plot(taus, s3, "o-", color="#1f77b4")
        a.axhline(CFG_LOCKED_SIGMA, ls="--", c="k", label="locked 0.0561")
        a.fill_between([min(taus), max(taus)], CFG_LOCKED_SIGMA * 0.97,
                       CFG_LOCKED_SIGMA * 1.03, color="k", alpha=0.08, label="+-3%")
        a.set_xlabel("tau (nu = (tau - 1/2)/3)")
        a.set_ylabel("sigma = dp * R_eq")
        a.set_title("C2  tau dependence (velocity_shift, R=20)")
        a.legend(fontsize=8)
    else:
        ax[1, 2].axis("off")

    fig.suptitle("SCMP Shan-Chen Laplace calibration - R1 adversarial verification", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out1 = HERE / "artifacts" / "fig_r1_overview.png"
    fig.savefig(out1, dpi=140)
    print(f"wrote {out1}")

    # --- D: Laplace law dp vs 1/R --------------------------------------
    fig2, a = plt.subplots(figsize=(7.2, 5.6))
    for s in schemes:
        invR = [1.0 / r["R_eq"] for r in V2[s]]
        dp = [r["dp"] for r in V2[s]]
        a.plot(invR, dp, "o", color=COL[s], ms=8, label=f"{s} (this round)")
    prior = XV_DIR / "artifacts" / "result.json"
    if prior.exists():
        pr = json.loads(prior.read_text())["laplace"]["tensorlbm"]
        a.plot([1.0 / q["R_eq"] for q in pr["rows"]],
               [q["dp"] for q in pr["rows"]], "s--", mfc="none", color="#555555",
               label="m20921 4-radius run")
        s_fit = pr["sigma_through_origin"]
        xx = np.linspace(0, max(1.0 / r["R_eq"] for r in V2[schemes[0]]) * 1.15, 20)
        a.plot(xx, s_fit * xx, ":", color="#555555", lw=1,
               label=f"m20921 through-origin fit sigma={s_fit:.5f}")
    a.set_xlabel("1 / R_eq")
    a.set_ylabel("Laplace pressure  dp")
    a.set_xlim(0, None)
    a.set_ylim(0, None)
    a.set_title("D  Young-Laplace law dp = sigma / R")
    a.legend(fontsize=8)
    fig2.tight_layout()
    out2 = HERE / "artifacts" / "fig_r1_laplace_fit.png"
    fig2.savefig(out2, dpi=140)
    print(f"wrote {out2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())