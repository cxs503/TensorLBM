#!/usr/bin/env python
"""Assemble the Stokes-first-problem (t* = 1, pointwise erfc) evidence bundle.

Reads the case JSONs produced by ``run_tstar1.py matrix`` (or re-uses them), writes

  * ``summary.json``  — machine-readable verdict table (no hand-copied numbers),
  * ``table.csv``     — the same table for the report,
  * ``fig_profile.png``      — u/U vs eta: analytic vs LBM, coarse & refined, log tail,
  * ``fig_error.png``        — pointwise relative error vs eta for every variant,
  * ``fig_spacetime.png``    — u/U(y,t) space-time map, LBM vs analytic difference,
  * ``fig_convergence.png``  — error vs delta (resolution) in log-log.

Run:  python make_assets.py [--scan-dir scan_tstar1] [--no-spacetime]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent

import sys  # noqa: E402

sys.path.insert(0, str(HERE))
from run_tstar1 import erfc, run_case  # noqa: E402

CASES = [
    "U0.05_H200_bgk", "U0.01_H200_bgk", "U1e-3_H200_bgk", "U1e-3_H200_trt",
    "U1e-3_H200_bgk_topnoslip", "U1e-3_H200_bgk_tstar0.64", "U1e-3_H400_bgk_fine",
]


def load(scan: Path) -> dict[str, dict]:
    out = {}
    for tag in CASES:
        p = scan / f"{tag}.json"
        if p.exists():
            out[tag] = json.loads(p.read_text())
    return out


def fig_profile(cases: dict[str, dict], out: Path) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    eta = np.linspace(0, 2.05, 300)
    ax.plot(eta, erfc(eta), "k-", lw=1.8, zorder=0, label="analytic erfc(eta)")
    colors = {"U1e-3_H200_bgk": "C0", "U1e-3_H400_bgk_fine": "C3", "U1e-3_H200_trt": "C2"}
    labels = {"U1e-3_H200_bgk": "LBM H=200, delta=25 (coarse)",
              "U1e-3_H400_bgk_fine": "LBM H=400, delta=50 (refined)",
              "U1e-3_H200_trt": "LBM TRT(Lambda=3/16), delta=25"}
    for tag, c in colors.items():
        if tag not in cases:
            continue
        p = cases[tag]["profiles"][0]
        ax.plot(p["eta_native"], p["u_over_U_native"], "o", ms=3.0, color=c, label=labels[tag])
    ax.set_xlabel("eta = y / (2 sqrt(nu t))")
    ax.set_ylabel("u / U")
    ax.set_title("Stokes first problem, t* = 1.0: profile vs erfc")
    ax.set_xlim(0, 2.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    eta_bl = np.linspace(0, 4.2, 400)
    ax2.semilogy(eta_bl, erfc(eta_bl), "k-", lw=1.6, label="analytic")
    for tag, c in colors.items():
        if tag not in cases:
            continue
        p = cases[tag]["profiles"][0]
        ax2.semilogy(p["eta_native"], p["u_over_U_native"], "o", ms=3.0, color=c)
    ax2.set_xlabel("eta")
    ax2.set_ylabel("u / U  (log)")
    ax2.set_title("same data, log scale: erfc tail to 1e-4")
    ax2.set_xlim(0, 4.2)
    ax2.set_ylim(1e-4, 1.2)
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "fig_profile.png", dpi=140)
    plt.close(fig)


def fig_error(cases: dict[str, dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for tag, d in cases.items():
        p = d["profiles"][0]
        ax.semilogy(p["eta_grid"], np.maximum(p["rel_err_pct_at_eta"], 1e-5), "-o", ms=3, label=tag)
    ax.axhline(2.0, color="k", ls="--", lw=1.2, label="acceptance 2%")
    ax.set_xlabel("eta")
    ax.set_ylabel("pointwise relative error of u/U  [%]")
    ax.set_title("Stokes first problem: pointwise error vs eta (t* = 1.0 window)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "fig_error.png", dpi=140)
    plt.close(fig)


def fig_convergence(cases: dict[str, dict], out: Path) -> None:
    rows = []
    for tag in ("U1e-3_H200_bgk", "U1e-3_H400_bgk_fine"):
        if tag not in cases:
            continue
        p = cases[tag]["profiles"][0]
        rows.append((p["delta_lb"], p["max_rel_err_pct_eta0.1_2"], p["max_rel_err_pct_native_nodes"], tag))
    rows2 = [("U1e-3_H200_bgk_tstar0.64", cases["U1e-3_H200_bgk_tstar0.64"]["profiles"][0])] \
        if "U1e-3_H200_bgk_tstar0.64" in cases else []
    for tag, p in rows2:
        rows.append((p["delta_lb"], p["max_rel_err_pct_eta0.1_2"], p["max_rel_err_pct_native_nodes"], tag))
    if len(rows) < 2:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    d = np.array([r[0] for r in rows])
    for i, lab in ((1, "max rel err, eta in [0.1, 2]"), (2, "max rel err, resolved nodes")):
        e = np.array([r[i] for r in rows])
        ax.loglog(d, e, "o-", label=lab)
    # reference slopes
    dd = np.array([d.min(), d.max()])
    ax.loglog(dd, rows[0][1] * (dd / d[0]) ** -2.0, "k:", lw=1, label="slope -2 (2nd order)")
    ax.set_xlabel("delta = sqrt(nu t)  [lattice units]  (= resolution at fixed t*)")
    ax.set_ylabel("max relative error [%]")
    ax.set_title("resolution study at fixed t* = 1.0 (delta = 20, 25, 50)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_convergence.png", dpi=140)
    plt.close(fig)


def fig_spacetime(out: Path) -> dict:
    """u/U(y,t) space-time map (the only 'field' this 1-D problem has)."""
    tstars = [0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1.0]
    d = run_case(200, 0.65, 1e-3, 1.0, record_tstars=tuple(tstars))
    nu = d["nu_lb"]
    field, ana, err = [], [], []
    eta = np.linspace(0, 4.0, 161)
    for p in d["profiles"]:
        y = np.array(p["eta_native"]) * 2 * np.sqrt(nu * p["step"])
        u = np.array(p["u_over_U_native"])
        un = np.interp(eta * 2 * np.sqrt(nu * p["step"]), y, u, left=np.nan, right=np.nan)
        a = erfc(eta)
        field.append(un)
        ana.append(a)
        err.append(np.abs(un - a) / np.maximum(a, 1e-12) * 100)
    field, ana, err = np.array(field), np.array(ana), np.array(err)
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for ax, dat, title, kw in (
        (axs[0], field, "LBM  u/U", dict(cmap="viridis", vmin=0, vmax=1)),
        (axs[1], ana, "analytic erfc  u/U", dict(cmap="viridis", vmin=0, vmax=1)),
        (axs[2], np.log10(np.maximum(err, 1e-4)), "log10(rel err [%])", dict(cmap="magma")),
    ):
        im = ax.pcolormesh(eta, tstars, dat, shading="auto", **kw)
        ax.set_xlabel("eta = y/(2 sqrt(nu t))")
        ax.set_ylabel("t*")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.suptitle("Stokes first problem: self-similar space-time map (H=200, delta=25 at t*=1)")
    fig.tight_layout()
    fig.savefig(out / "fig_spacetime.png", dpi=140)
    plt.close(fig)
    return {"tstars": tstars,
            "max_rel_err_pct_per_tstar": [float(np.nanmax(e)) for e in err],
            "max_abs_u_over_U_err": float(np.nanmax(np.abs(field - ana)))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", type=Path, default=HERE / "scan_tstar1")
    ap.add_argument("--no-spacetime", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=HERE / "scan_tstar1")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    cases = load(a.scan_dir)
    tol = 2.0

    # machine-checked criterion: pointwise rel err on eta in [0.1, 2] at t*=1.0
    # (eta < 1/(4 delta) has no discrete node: the half-way wall sits at y = 0.5)
    table = []
    for tag, d in cases.items():
        p = d["profiles"][0]
        table.append({
            "tag": tag, "H": d["H"], "tau": d["tau"], "nu": d["nu_lb"], "U": d["U"], "Ma": d["Ma"],
            "collision": d["collision"], "top_bc": d["top_bc"], "tstar": p["tstar"],
            "delta": p["delta_lb"], "nsteps": d["nsteps"], "rho_max_dev": d.get("rho_max_dev_from_1"),
            "max_rel_pct_eta0_2": p["max_rel_err_pct_eta0_2"],
            "max_rel_pct_eta0.1_2": p["max_rel_err_pct_eta0.1_2"],
            "rel_pct_eta2": p["rel_err_pct_at_eta2"],
            "max_rel_pct_resolved_nodes": p["max_rel_err_pct_native_nodes"],
            "pass_2pct": bool(p["max_rel_err_pct_eta0_2"] <= tol),
        })
    table.sort(key=lambda r: (r["tstar"], r["delta"], r["tag"]))
    (a.out_dir / "table.csv").write_text(
        "tag,H,tau,nu,U,Ma,collision,top_bc,tstar,delta,nsteps,rho_max_dev,"
        "max_rel_pct_eta0_2,max_rel_pct_eta0.1_2,rel_pct_eta2,max_rel_pct_nodes,pass_2pct\n"
        + "\n".join(
            ",".join(str(r[k]) for k in ("tag", "H", "tau", "nu", "U", "Ma", "collision", "top_bc",
                                         "tstar", "delta", "nsteps", "rho_max_dev",
                                         "max_rel_pct_eta0_2", "max_rel_pct_eta0.1_2",
                                         "rel_pct_eta2", "max_rel_pct_resolved_nodes", "pass_2pct"))
            for r in table)
    )
    summary = {
        "case": "LBM-STOKES-FIRST-PROBLEM",
        "criterion": "<2% pointwise relative error of u/U=erfc(eta), eta in [0,2], t*=1.0",
        "tstar_definition": "t* = nu t / delta_ref^2, delta_ref = 25 coarse-lattice units "
                            "(<=> delta = sqrt(nu t) = 25 at t* = 1; eta window [0,2] = y in [0,4 delta])",
        "wall_plane_lattice": 0.5,
        "runs": table,
        "all_pass_2pct": all(r["pass_2pct"] for r in table),
        "verdict": "VERIFIED" if all(r["pass_2pct"] for r in table) else "NOT_PASSED",
    }
    if not a.no_spacetime:
        summary["spacetime"] = fig_spacetime(a.out_dir)
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    fig_profile(cases, a.out_dir)
    fig_error(cases, a.out_dir)
    fig_convergence(cases, a.out_dir)
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))
    hdr = f"{'tag':<26s}{'delta':>6s}{'t*':>5s}{'eta0-2':>9s}{'eta.1-2':>9s}{'eta2':>9s}{'nodes':>8s}"
    print(hdr)
    for r in table:
        print(f"{r['tag']:<26s}{r['delta']:6.1f}{r['tstar']:5.2f}"
              f"{r['max_rel_pct_eta0_2']:9.3f}{r['max_rel_pct_eta0.1_2']:9.4f}"
              f"{r['rel_pct_eta2']:9.4f}{r['max_rel_pct_resolved_nodes']:8.3f}")


if __name__ == "__main__":
    main()