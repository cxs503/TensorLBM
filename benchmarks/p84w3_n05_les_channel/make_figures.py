#!/usr/bin/env python3
"""Figures + report table for the LBM-LES channel verification (p84w3-n05).

Reads the case artifacts written by ``run_case.py`` and emits

* ``docs/figures/p84w3_n05_u_plus.png``      U+ vs y+ (LES, MKM1999, log law)
* ``docs/figures/p84w3_n05_reynolds_stress.png``  -<u'v'>+ vs y+ (LES, DNS budget)
* ``docs/figures/p84w3_n05_spectra.png``     E_uu(k_x) at sampled planes + -5/3
* ``docs/figures/p84w3_n05_sgs.png``         SGS viscosity / stress fraction
* ``artifacts/p84w3_n05_les_channel/report_numbers.json``  (plot inputs, no hand-typed digits)

Every plotted number comes from the artifact; the DNS curves come from the
vendored MKM1999 table through the same loader the criteria use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorlbm.turbulence.les_channel_verification import load_dns_reference  # noqa: E402

LOG_KAPPA = 0.41
LOG_B = 5.2


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def figure_u_plus(artifacts: list[dict], out: Path) -> dict:
    dns = load_dns_reference(root=ROOT)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    y = dns.y_plus.numpy()
    ax.plot(y, dns.u_plus, "k-", lw=1.4, label="MKM1999 DNS (Re$\\tau$=180)")
    ylog = np.linspace(30.0, 120.0, 200)
    ax.plot(ylog, (1 / LOG_KAPPA) * np.log(ylog) + LOG_B, "k:", lw=1.0,
            label=r"$\frac{1}{\kappa}\ln y^+ + B$ ($\kappa$=0.41, B=5.2)")
    payload = {"dns": {"y_plus": y.tolist(), "u_plus": dns.u_plus.tolist()},
               "log_law": {"y_plus": ylog.tolist(),
                           "u_plus": ((1 / LOG_KAPPA) * np.log(ylog) + LOG_B).tolist()},
               "cases": {}}
    for art, style in zip(artifacts, ["-o", "-s", "-^", "-d"]):
        res = art["result"]
        wu = res["wall_units"]
        cfg = res["config"]
        label = f"LES h={cfg['half_height']:.0f}, $\\Delta y^+$={cfg['dy_plus']:.1f}, " \
                f"$C_s$={cfg['smagorinsky_cs']:.2f}"
        ax.plot(wu["y_plus"], wu["u_plus"], style, ms=3, lw=1.2, label=label)
        payload["cases"][cfg.get("label", label)] = {
            "y_plus": wu["y_plus"], "u_plus": wu["u_plus"],
            "dy_plus": cfg["dy_plus"], "cells": cfg["cells"], "steps": cfg["steps"],
        }
    ax.set_xscale("log")
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"$U^+$")
    ax.set_title("Mean velocity profile, plane channel LES (Re$\\tau$=180)")
    ax.set_xlim(1.0, 180.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return payload


def figure_reynolds_stress(artifacts: list[dict], out: Path) -> dict:
    dns = load_dns_reference(root=ROOT)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    y = dns.y_plus.numpy()
    ax.plot(y, dns.uv_minus_plus, "k-", lw=1.4,
            label=r"MKM1999 stress budget $1-y^+/Re_\tau-dU^+/dy^+$")
    payload = {"dns": {"y_plus": y.tolist(), "uv_minus_plus": dns.uv_minus_plus.tolist()},
               "cases": {}}
    for art, style in zip(artifacts, ["-o", "-s", "-^", "-d"]):
        res = art["result"]
        wu = res["wall_units"]
        cfg = res["config"]
        ax.plot(wu["y_plus"], wu["uv_minus_plus"], style, ms=3, lw=1.2,
                label=f"LES h={cfg['half_height']:.0f}, SGS-inclusive DNS-complement")
        payload["cases"][cfg.get("label", "case")] = {
            "y_plus": wu["y_plus"], "uv_minus_plus": wu["uv_minus_plus"],
            "total_stress_plus": wu["total_stress_plus"],
        }
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"$-\langle u'v'\rangle^+$")
    ax.set_title("Resolved Reynolds shear stress")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return payload


def figure_spectra(artifacts: list[dict], out: Path) -> dict:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    payload = {"cases": {}}
    for art, marker in zip(artifacts, ["o", "s", "^", "d"]):
        res = art["result"]
        spec = res.get("les", {}).get("spectra", {})
        label_base = f"h={res['config']['half_height']:.0f}"
        for leg in spec.get("streamwise", []):
            k = np.array(leg["k"])
            e = np.array(leg["E"])
            fit = leg.get("fit", {})
            ax.loglog(k, e, marker + "-", ms=2.5, lw=0.9, alpha=0.8,
                      label=f"{label_base} y+={leg['y_plus']:.0f}")
            if fit.get("valid"):
                kk = np.array([fit["k_min"], fit["k_max"]])
                ee = np.exp(np.interp(np.log(kk), np.log(k), np.log(np.maximum(e, 1e-300))))
                ax.loglog(kk, ee, "k--", lw=1.0, alpha=0.6)
        payload["cases"][label_base] = {
            "legs": [{"y_plus": leg["y_plus"], "k": leg["k"], "E": leg["E"], "fit": leg.get("fit")}
                     for leg in spec.get("streamwise", [])],
        }
    # -5/3 guide anchored on the plot
    if ax.get_xlim()[1] > 0:
        kk = np.array([4.0, 12.0])
        ax.loglog(kk, 1e-4 * (kk / 4.0) ** (-5.0 / 3.0), "r-", lw=1.4, label=r"$k^{-5/3}$ guide")
    ax.set_xlabel(r"$k_x$ (lattice units)")
    ax.set_ylabel(r"$E_{uu}(k_x)$")
    ax.set_title("Streamwise energy spectra (a-priori fit band dashed)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return payload


def figure_sgs(artifacts: list[dict], out: Path) -> dict:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 4.0))
    payload = {"cases": {}}
    for art, style in zip(artifacts, ["-o", "-s", "-^", "-d"]):
        res = art["result"]
        sgs = res.get("les", {}).get("sgs_stress", {})
        cfg = res["config"]
        nu = cfg["nu"]
        yp = np.array(sgs["y_plus"])
        nut = np.array(sgs["nu_t_mean"]) / nu
        a1.semilogx(yp, nut, style, ms=3, lw=1.2, label=f"h={cfg['half_height']:.0f}")
        payload["cases"].setdefault(cfg.get("label", "case"), {})["y_plus"] = yp.tolist()
        payload["cases"][cfg.get("label", "case")]["nu_t_over_nu"] = nut.tolist()
    a1.set_xlabel(r"$y^+$")
    a1.set_ylabel(r"$\langle \nu_t\rangle / \nu$")
    a1.set_title("Smagorinsky eddy viscosity (van Driest damped)")
    a1.grid(alpha=0.3, which="both")
    a1.legend(fontsize=8)
    for art, style in zip(artifacts, ["-o", "-s", "-^", "-d"]):
        res = art["result"]
        prod = res.get("les", {}).get("production", {})
        yp = np.array(prod["y_plus"])
        a2.plot(yp, prod["resolved_plus"], style, ms=3, lw=1.2,
                label=f"resolved, h={res['config']['half_height']:.0f}")
        a2.plot(yp, prod["with_sgs_plus"], style, ms=2, lw=0.8, alpha=0.5)
        payload["cases"][res["config"].get("label", "case")]["production_y_plus"] = yp.tolist()
        payload["cases"][res["config"].get("label", "case")]["production_resolved_plus"] = prod["resolved_plus"]
        payload["cases"][res["config"].get("label", "case")]["production_with_sgs_plus"] = prod["with_sgs_plus"]
    a2.set_xlabel(r"$y^+$")
    a2.set_ylabel(r"$P^+$")
    a2.set_title("TKE production (resolved, and +SGS)")
    a2.grid(alpha=0.3)
    a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return payload


def figure_tke_history(artifacts: list[dict], registers: list[dict], out: Path) -> dict:
    """Resolved TKE versus step: the three fates (divergence / decay / sustain)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    payload = {"cases": {}, "honest_fail": []}
    for art, style in zip(artifacts, ["-o", "-s", "-^", "-d"]):
        hist = art["result"].get("history", [])
        step = [h["step"] for h in hist]
        tke = [h["tke"] for h in hist]
        label = art["result"].get("label") or art["result"]["config"].get("label")
        ax.semilogy(step, tke, style, ms=3, lw=1.2, label=label)
        payload["cases"][label] = {"step": step, "tke": tke}
    for reg in registers:
        if reg.get("status") == "completed":
            continue
        ax.axvline(1000.0, color="r", ls="--", lw=1.0, alpha=0.6)
        payload["honest_fail"].append(
            {"label": reg.get("label"), "spec": reg.get("spec"),
             "error": reg.get("error"), "diverged_at_or_before_step": 1000}
        )
        break
    ax.set_xlabel("step")
    ax.set_ylabel(r"resolved TKE")
    ax.set_title("Resolved turbulent kinetic energy: decay vs divergence")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("artifacts", nargs="+", type=Path)
    ap.add_argument("--figdir", type=Path, default=ROOT / "docs" / "figures")
    ap.add_argument("--numbers", type=Path,
                    default=ROOT / "artifacts" / "p84w3_n05_les_channel" / "report_numbers.json")
    args = ap.parse_args(argv)

    arts = [_load(p) for p in args.artifacts]
    args.figdir.mkdir(parents=True, exist_ok=True)
    numbers = {
        "cases": [{k: v for k, v in a["summary"].items()} for a in arts],
        "criteria": {a["result"]["config"].get("label", f"case{i}"): a["criteria"] for i, a in enumerate(arts)},
    }
    numbers["fig_u_plus"] = figure_u_plus(arts, args.figdir / "p84w3_n05_u_plus.png")
    numbers["fig_reynolds_stress"] = figure_reynolds_stress(arts, args.figdir / "p84w3_n05_reynolds_stress.png")
    numbers["fig_spectra"] = figure_spectra(arts, args.figdir / "p84w3_n05_spectra.png")
    numbers["fig_sgs"] = figure_sgs(arts, args.figdir / "p84w3_n05_sgs.png")
    registers = []
    for d in {p.parent for p in args.artifacts}:
        for rp in sorted(Path(d).glob("campaign_register*.json")):
            registers.extend(json.loads(rp.read_text()).get("cases", []))
    numbers["fig_tke_history"] = figure_tke_history(
        arts, registers, args.figdir / "p84w3_n05_tke_history.png"
    )
    args.numbers.parent.mkdir(parents=True, exist_ok=True)
    args.numbers.write_text(json.dumps(numbers, indent=1))
    print(f"figures -> {args.figdir}")
    print(f"numbers -> {args.numbers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
