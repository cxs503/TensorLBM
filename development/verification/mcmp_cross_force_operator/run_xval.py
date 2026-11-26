#!/usr/bin/env python3
"""XVAL-LBM-MCMP-CROSS-FORCE-OPERATOR-XVAL-SYMBOL-SP -- entry point.

Runs the four cross-validation arms of :mod:`arms` (symbolic, spectral, OSS,
physics), writes machine-readable artifacts and the report figures.

    PYTHONPATH=<repo>/src python run_xval.py                    # full
    PYTHONPATH=<repo>/src python run_xval.py --quick            # smoke (~1 min)
    PYTHONPATH=<repo>/src python run_xval.py --skip-physics     # arms S/P/O only

Artifacts (relative to this directory):
    artifacts/xval_results.json     every number in the report
    artifacts/xval_profiles.npz     relaxed fields + spectral tables
    figures/fig1_gain.png           stencil gain vs wavenumber (+ closed form)
    figures/fig2_equilibrium.png    field cloud map + interface profiles
    figures/fig3_invariants.png     invariant residuals against the 3%/15% gates
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "src"))

import torch  # noqa: E402

import arms  # noqa: E402

ART = os.path.join(HERE, "artifacts")
FIG = os.path.join(HERE, "figures")


def _dump(obj):
    """JSON-safe deep conversion (tensors / numpy scalars)."""
    if isinstance(obj, dict):
        return {k: _dump(v) for k, v in obj.items() if k != "fields"}
    if isinstance(obj, (list, tuple)):
        return [_dump(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return float(obj)
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-physics", action="store_true")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("NTHREADS", "8")))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(ART, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    res = {}
    print("[arm S] symbolic ...", flush=True)
    res["arm_symbol"] = arms.arm_symbol()
    print(json.dumps(_dump(res["arm_symbol"]), indent=1), flush=True)

    print("[arm P] spectral ...", flush=True)
    res["arm_spectral"] = arms.arm_spectral(n=64 if args.quick else 128,
                                            m_max=6 if args.quick else 16)
    sp = res["arm_spectral"]
    print("  lattice vs closed form: %.3e   fft-gain vs 1: %.3e   order %s"
          % (sp["max_rel_diff_lattice_vs_closed_form"],
             sp["max_rel_diff_fft_gain_vs_unity"], sp["refinement_order"]), flush=True)

    print("[arm O] OSS (lbmpy transcription) ...", flush=True)
    frozen = os.path.join(os.path.dirname(os.path.dirname(HERE)), "audit",
                          "xval_mcmp_cross_force_m18289", "refs", "operators.py")
    res["arm_oss"] = arms.arm_oss(frozen_ref=frozen)
    print("  max rel rms diff vs OSS operator: %.3e"
          % res["arm_oss"]["max_rel_rms_diff"], flush=True)

    fields = {}
    if not args.skip_physics:
        print("[arm F] physics (flat interface equilibrium) ...", flush=True)
        Gs = (-2.5,) if args.quick else (-2.5, -2.6)
        res["arm_physics"] = arms.arm_physics(
            n=64 if args.quick else args.n,
            G_values=Gs,
            steps=4000 if args.quick else args.steps,
            tau=1.0, u_eq="self")
        for tag, case in res["arm_physics"].items():
            if tag == "fields":
                continue
        fields = res["arm_physics"].pop("fields", {})
        for c in res["arm_physics"]["cases"]:
            print("  G=%s converged=%s u_max=%.2e p-uniformity=%.3e boltzmann=%.3e op-err=%.3e"
                  % (c.get("G"), c.get("converged"), c.get("u_max", float("nan")),
                     c.get("p_uniformity_rel", float("nan")),
                     c.get("boltzmann_relation_residual_rel", float("nan")),
                     c.get("operator_error_on_interface_rel", float("nan"))), flush=True)

    res["verdict"] = _verdict(res)

    with open(os.path.join(ART, "xval_results.json"), "w") as fh:
        json.dump(_dump(res), fh, indent=1)
    npz = {}
    for tag, d in fields.items():
        for k, v in d.items():
            npz[f"{tag}_{k}"] = np.asarray(v)
    for key in ("rows", "refinement"):
        for i, row in enumerate(res["arm_spectral"].get(key, [])):
            npz[f"spectral_{key}_{i}"] = np.asarray(
                [row.get("k", np.nan), row.get("gain_lattice", np.nan),
                 row.get("gain_closed_form", np.nan), row.get("rel_err_vs_continuum", np.nan)])
    if res.get("arm_physics", {}).get("profiles"):
        for tag, prof in res["arm_physics"]["profiles"].items():
            npz[f"profile_{tag}_rho1"] = np.asarray(prof["rho1"])
            npz[f"profile_{tag}_rho2"] = np.asarray(prof["rho2"])
    np.savez(os.path.join(ART, "xval_profiles.npz"), **npz)

    _figures(res, fields)
    print(json.dumps(_dump(res["verdict"]), indent=1), flush=True)
    print("artifacts ->", ART, "figures ->", FIG, flush=True)
    return 0


def _verdict(res) -> dict:
    """Explicit pass/fail per key quantity against the 3% / 15% gates."""
    out = {"items": []}

    def add(name, value, ref, gate, note=""):
        arr = {"quantity": name, "measured": value, "reference": ref,
               "rel_err": abs(value) if value is not None else None,
               "gate": gate,
               "pass": (value is not None and abs(value) <= gate),
               "note": note}
        out["items"].append(arr)

    s = res["arm_symbol"]
    add("symbolic stencil closed form (vs sympy sum)", 0.0 if s["stencil_closed_form_x_verified"]
        else 1.0, "exact algebra", 0.03, "identity verified symbolically")
    add("stencil gain vs cs^2 for k->0", abs(s["gain_leading_cs2_float"] / 0.3333333333333 - 1.0),
        "cs^2 = 1/3", 0.03, "leading coefficient of the small-k expansion")

    p = res["arm_spectral"]
    add("lattice gain vs closed form (all reciprocal modes)",
        p["max_rel_diff_lattice_vs_closed_form"], "closed form", 0.03,
        "float32 weight table in tensorlbm.d2q9.W bounds this at ~1e-8")
    add("FFT (spectral) gain vs unity", p["max_rel_diff_fft_gain_vs_unity"],
        "du/dx", 0.03, "independent FFT reference machinery")
    o = res["arm_oss"]
    add("library kernel vs OSS (lbmpy) operator", o["max_rel_rms_diff"], "independent OSS",
        0.03, "RMS over random/smooth/exp-psi fields")

    for c in res.get("arm_physics", {}).get("cases", []):
        if c.get("status") != "ok":
            out["items"].append({"quantity": f"physics G={c.get('G')}", "measured": None,
                                 "reference": "flat-interface equilibrium",
                                 "rel_err": None, "gate": 0.03, "pass": False,
                                 "note": f"run status={c.get('status')}"})
            continue
        gate = 0.03
        add(f"EOS pressure uniformity across flat interface (G={c['G']})",
            c["p_uniformity_rel"], "p = const (derived convention)", gate,
            "mechanical equilibrium of a flat interface")
        add(f"discrete Boltzmann relation ln rho1 - G psi2 = const (G={c['G']})",
            c["boltzmann_relation_residual_rel"], "analytic ODE integral", gate,
            "continuum-limit force balance")
        add(f"Newton 3rd law, sum(F1+F2) (G={c['G']})",
            c["newton_third_law_residual_rel"], "0", gate, "telescoping on a periodic box")
        add(f"operator error on the measured interface (G={c['G']})",
            c["operator_error_on_interface_rel"],
            "spectral force on the same profile", 0.15,
            "intrinsic high-k truncation of the D2Q9 stencil, not an implementation defect")
    out["all_pass_3pct"] = all(i["pass"] for i in out["items"] if i["gate"] <= 0.03)
    return out


def _figures(res, fields) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print("matplotlib unavailable, skipping figures:", exc, flush=True)
        return

    sp = res["arm_spectral"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for tag, marker in (("axis", "o"), ("diagonal", "s")):
        rows = [r for r in sp["rows"] if r.get("dir", "axis") == tag]
        if not rows:
            continue
        k = np.array([r["k"] for r in rows])
        g = np.array([r["gain_over_cs2"] for r in rows])
        ax[0].plot(k, g, marker, ms=4, label=f"lattice ({tag})")
    kk = np.linspace(1e-3, 2.6, 400)
    ax[0].plot(kk, 1 - kk ** 2 / 6, "k-", lw=1, label=r"$1-k^2/6$ (symbolic)")
    ax[0].plot(kk, np.ones_like(kk), "k--", lw=1, label="continuum $c_s^2$")
    ax[0].set_xlabel("wavenumber |k|  (lattice units)")
    ax[0].set_ylabel(r"stencil gain / $c_s^2$")
    ax[0].set_title("cross-force gradient operator: gain")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)
    kk = np.linspace(1e-3, 2.6, 400)
    ax[1].semilogy(kk, kk ** 2 / 6, "k-", label=r"$k^2/6$ truncation")
    for gate, ls in ((0.03, ":"), (0.15, "--")):
        ax[1].axhline(gate, ls=ls, lw=1, color="r" if gate == 0.15 else "g")
        ax[1].text(0.05, gate * 1.15, f"{int(gate*100)}% gate", fontsize=8)
    ax[1].set_xlabel("wavenumber |k|")
    ax[1].set_ylabel("relative gain error")
    ax[1].set_title("where the D2Q9 stencil stops resolving the gradient")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_gain.png"), dpi=130)
    plt.close(fig)

    # fig2: field cloud map + interface profiles
    if fields:
        tag = sorted(fields)[0]
        r1 = np.asarray(fields[tag]["rho1"])
        r2 = np.asarray(fields[tag]["rho2"])
        case = [c for c in res.get("arm_physics", {}).get("cases", [])
                if f"G{c.get('G')}" == tag]
        case = case[0] if case else {}
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.0))
        im = ax[0].imshow(r1 - r2, origin="lower", cmap="RdBu_r")
        ax[0].set_title(f"$\\rho_1-\\rho_2$ cloud map  ({tag})")
        fig.colorbar(im, ax=ax[0], fraction=0.046)
        x = np.arange(r1.shape[1])
        ax[1].plot(x, r1.mean(axis=0), label=r"$\rho_1$")
        ax[1].plot(x, r2.mean(axis=0), label=r"$\rho_2$")
        ax[1].set_xlabel("x"); ax[1].set_ylabel("density")
        ax[1].set_title("interface profiles (y-averaged)")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
        if case:
            G = case["G"]
            p_der = arms._mcmp_pressure(
                torch.as_tensor(r1.mean(axis=0)), torch.as_tensor(r2.mean(axis=0)), G)
            p_lib = arms._mcmp_pressure(
                torch.as_tensor(r1.mean(axis=0)), torch.as_tensor(r2.mean(axis=0)), G,
                convention="library")
            ax[2].plot(x, p_der.numpy() / float(p_der.mean()), label="derived (uniform)")
            ax[2].plot(x, p_lib.numpy() / float(p_lib.mean()), label="library sign")
            ax[2].set_xlabel("x"); ax[2].set_ylabel(r"$p/\bar p$")
            ax[2].set_title("flat interface: pressure must be constant")
            ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig2_equilibrium.png"), dpi=130)
        plt.close(fig)

    # fig3: invariant residuals against the gates
    items = [i for i in res["verdict"]["items"] if i["rel_err"] is not None and i["rel_err"] > 0]
    if items:
        labels = [i["quantity"] for i in items]
        vals = [i["rel_err"] for i in items]
        cols = ["tab:green" if i["pass"] else "tab:red" for i in items]
        fig, ax = plt.subplots(figsize=(9, 0.42 * len(items) + 1.6))
        y = np.arange(len(items))
        ax.barh(y, vals, color=cols)
        ax.set_yticks(y)
        ax.set_yticklabels([l if len(l) < 62 else l[:59] + "..." for l in labels], fontsize=7)
        ax.set_xscale("log")
        ax.axvline(0.03, ls=":", color="k", lw=1)
        ax.axvline(0.15, ls="--", color="k", lw=1)
        ax.set_xlabel("relative residual / error  (3% dotted, 15% dashed)")
        ax.set_title("XVAL-MCMP-CROSS-FORCE-OPERATOR: invariants vs gates")
        ax.grid(alpha=.3, axis="x", which="both")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig3_invariants.png"), dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())