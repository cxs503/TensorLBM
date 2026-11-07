#!/usr/bin/env python
"""Figures for the KCS self-propulsion study.

Reads the machine-readable evidence produced by
``scripts/selfpropulsion_kcs.py`` (``selfpropulsion_kcs.json`` and
``tunnel.json``) and writes publication-style PNG figures.  No number in the
figures is typed by hand -- everything is read from the evidence files or
recomputed with the same library calls.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorlbm.selfpropulsion import (  # noqa: E402
    KCS_MODEL,
    KP505Geometry,
    SectionPolar,
    blade_element_momentum,
    kp505_reference_table,
)


def fig_open_water(out: Path, geom: KP505Geometry, polar: SectionPolar) -> Path:
    rows = json.loads((out / "selfpropulsion_kcs.json").read_text())["open_water"]["rows"]
    js = np.array([r["J"] for r in rows])
    ref = kp505_reference_table()
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    ax[0].plot(js, [r["KT_bem"] for r in rows], "o-", label="BEM (this work)")
    ax[0].plot(ref["J"], ref["KT"], "s--", label="KP505 measured")
    ax[0].set_xlabel("J = V_A/(nD)")
    ax[0].set_ylabel("$K_T$")
    ax[0].grid(alpha=0.3)
    ax[0].legend()
    ax[0].set_title("(a) thrust coefficient")

    ax[1].plot(js, [10 * r["KQ_bem"] for r in rows], "o-", label="BEM")
    ax[1].plot(ref["J"], ref["10KQ"], "s--", label="KP505 measured")
    ax[1].set_xlabel("J")
    ax[1].set_ylabel("$10K_Q$")
    ax[1].grid(alpha=0.3)
    ax[1].legend()
    ax[1].set_title("(b) torque coefficient")

    ax[2].plot(js, [r["eta0_bem"] for r in rows], "o-", label="BEM")
    ax[2].plot(ref["J"], ref["eta0"], "s--", label="KP505 measured")
    ax[2].axvline(0.7, color="grey", ls=":", lw=1)
    ax[2].annotate("design J = 0.7", xy=(0.7, 0.1), rotation=90, fontsize=8, color="grey")
    ax[2].set_xlabel("J")
    ax[2].set_ylabel(r"$\eta_0$")
    ax[2].grid(alpha=0.3)
    ax[2].legend()
    ax[2].set_title("(c) open-water efficiency")
    fig.tight_layout()
    path = out / "fig_openwater.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_loading(out: Path, geom: KP505Geometry, polar: SectionPolar) -> Path:
    sol = blade_element_momentum(geom, polar, 0.7)
    rr = np.array(sol["r_over_R"])
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    ax[0].plot(rr, np.array(sol["dT_dr"]) / 1e3, "-", color="C0")
    ax[0].plot(rr, np.array(sol["dQ_dr"]) * 1e2, "-", color="C1")
    ax[0].set_xlabel("r/R")
    ax[0].set_ylabel(r"$dT/dr$ [kN/m],  $100\,dQ/dr$ [kN]")
    ax[0].set_title("(a) BEM radial loading (J = 0.7)")
    ax[0].grid(alpha=0.3)

    ax[1].plot(rr, sol["alpha_deg"], label=r"incidence $\alpha$ [deg]")
    ax[1].plot(rr, sol["phi_deg"], label=r"flow angle $\phi$ [deg]")
    ax[1].plot(rr, np.rad2deg(geom.pitch_angle(rr)), label=r"pitch angle $\beta$ [deg]")
    ax[1].set_xlabel("r/R")
    ax[1].set_ylabel("angle [deg]")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    ax[1].set_title("(b) blade-element kinematics")

    ax[2].plot(rr, sol["a"], label="axial induction a")
    ax[2].plot(rr, sol["a_prime"], label="tangential a'")
    ax[2].plot(rr, sol["F"], label="Prandtl loss F")
    ax[2].set_xlabel("r/R")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)
    ax[2].set_title("(c) induction factors")
    fig.tight_layout()
    path = out / "fig_loading.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_tunnel(out: Path) -> Path:
    data = json.loads((out / "tunnel.json").read_text())
    prof = np.array(data["axial_profile"])
    x = np.array(data["x_profile"])
    u_in = data["config"]["u_in"]
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.9))
    ax[0].plot(x, prof / u_in, "-", color="C0")
    ax[0].axhline(1.0, color="grey", ls=":", lw=1)
    ax[0].set_xlabel("x [lattice units]")
    ax[0].set_ylabel(r"$\langle u_x\rangle_{disk}/U_\infty$")
    ax[0].set_title("(a) axial velocity along the duct (disk-averaged)")
    ax[0].grid(alpha=0.3)
    ax[0].annotate("actuator disk", xy=(0.35 * data["config"]["nx"], 1.0),
                   xytext=(0.5 * data["config"]["nx"], 0.4 * (prof / u_in).max() + 0.6),
                   arrowprops={"arrowstyle": "->", "color": "C3"}, color="C3", fontsize=9)

    checks = data["checks"]
    measured = data["measured"]
    labels = ["thrust\naccounting", "$\\Delta p$ vs\n$T/A$", "$\\Delta p$ vs\n$u_a(U{+}u_a)$",
              "far-wake\nratio vs 2"]
    values = [checks["thrust_accounting_rel_error"],
              checks["delta_p_rel_error_vs_T_over_A"],
              checks.get("delta_p_rel_error_vs_momentum_theory", float("nan")),
              checks["far_wake_ratio_error_vs_2"]]
    bars = ax[1].bar(labels, [100 * v for v in values], color=["C2", "C0", "C0", "C1"])
    ax[1].set_ylabel("deviation [%]")
    ax[1].set_title("(b) momentum-theory verification residuals")
    ax[1].grid(alpha=0.3, axis="y")
    for b, v in zip(bars, values):
        ax[1].text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02,
                   f"{100 * v:.2f}%" if math.isfinite(v) else "n/a",
                   ha="center", fontsize=8)
    fig.suptitle(
        f"actuator-disk verification: $T^*$={data['disk']['thrust_lattice']:.3g}, "
        f"$u_a/U$={measured['u_induced'] / u_in:.3f}, "
        f"$\\Delta p$={measured['delta_p']:.2e}",
        fontsize=9)
    fig.tight_layout()
    path = out / "fig_tunnel.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_power_chain(out: Path) -> Path:
    res = json.loads((out / "selfpropulsion_kcs.json").read_text())
    chain = res["power_chain"]
    sp = res["self_propulsion"]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9))
    names = ["$P_E$\neffective", "$P_T$\nthrust", "$P_D$\ndelivered"]
    vals = [chain["P_E"], chain["P_T"], chain["P_D"]]
    bars = ax[0].bar(names, vals, color=["C2", "C0", "C1"])
    for b, v in zip(bars, vals):
        ax[0].text(b.get_x() + b.get_width() / 2, v * 1.01, f"{v:.1f} W",
                   ha="center", fontsize=9)
    ax[0].set_ylabel("power [W]")
    ax[0].set_title("(a) power chain at the self-propulsion point")
    ax[0].grid(alpha=0.3, axis="y")
    ax[0].text(0.02, 0.95,
               f"n* = {sp['n_rps']:.3f} rps, J = {sp['j']:.3f}\n"
               f"T = {sp['thrust']:.2f} N, Q = {sp['torque']:.3f} Nm\n"
               f"$\\eta_H$ = {chain['eta_H']:.4f}, $\\eta_R$ = {chain['eta_R']:.4f}, "
               f"$\\eta_0$ = {chain['eta_0']:.4f}\n"
               f"$\\eta_D$ = {chain['eta_D']:.4f}",
               transform=ax[0].transAxes, va="top", fontsize=8,
               bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "grey"})

    eff_names = [r"$\eta_H$", r"$\eta_R$", r"$\eta_0$", r"$\eta_D$"]
    eff_vals = [chain["eta_H"], chain["eta_R"], chain["eta_0"], chain["eta_D"]]
    bars = ax[1].bar(eff_names, eff_vals, color=["C2", "C4", "C0", "C1"])
    for b, v in zip(bars, eff_vals):
        ax[1].text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                   ha="center", fontsize=9)
    ax[1].set_ylim(0, max(eff_vals) * 1.25)
    ax[1].set_ylabel("efficiency [-]")
    ax[1].set_title(r"(b) efficiency decomposition  $\eta_D=\eta_H\eta_R\eta_0$")
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = out / "fig_power_chain.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_selfprop(out: Path, geom: KP505Geometry, polar: SectionPolar) -> Path:
    res = json.loads((out / "selfpropulsion_kcs.json").read_text())
    sp = res["self_propulsion"]
    rt = res["resistance"]["R_T"]
    w = res["wake"]["w_design"]
    t = res["wake"]["t_design"]
    target = rt / (1.0 - t)
    ns = np.linspace(max(5.0, sp["n_rps"] - 2.5), sp["n_rps"] + 2.5, 40)
    th = []
    for n in ns:
        sol = blade_element_momentum(geom, polar, KCS_MODEL.speed * (1 - w) / (n * geom.diameter),
                                     n_rps=float(n), rho=KCS_MODEL.rho)
        th.append(sol["thrust"])
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.9))
    ax[0].plot(ns, th, "-", color="C0", label=r"$T(n)$ from BEM")
    ax[0].axhline(target, color="C3", ls="--", label=r"$R_T/(1-t)$")
    ax[0].plot([sp["n_rps"]], [sp["thrust"]], "o", color="C3", ms=7,
               label=f"self-propulsion point n*={sp['n_rps']:.3f} rps")
    ax[0].set_xlabel("n [rev/s]")
    ax[0].set_ylabel("thrust [N]")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    ax[0].set_title("(a) self-propulsion balance  $T(n)(1-t)=R_T$")

    hist = res["self_propulsion"]["history"]
    res_arr = np.abs(np.array([h["residual"] for h in hist])) / target
    ax[1].semilogy(range(1, len(res_arr) + 1), np.maximum(res_arr, 1e-16), "o-")
    ax[1].axhline(1e-9, color="grey", ls=":", lw=1)
    ax[1].set_xlabel("bisection iteration")
    ax[1].set_ylabel("$|T(n)(1-t)-R_T| / R_T$")
    ax[1].grid(alpha=0.3, which="both")
    ax[1].set_title("(b) convergence of the revolution-rate search")
    fig.tight_layout()
    path = out / "fig_selfprop.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_wake(out: Path) -> Path | None:
    res = json.loads((out / "selfpropulsion_kcs.json").read_text())
    hull = res.get("hull_run")
    if not hull:
        return None
    rad = hull["radial_profile"]
    sec = hull["sector_profile"]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8), subplot_kw={"projection": "polar"})
    th = np.array([s["theta_mid"] for s in sec])
    wv = np.array([s["w"] for s in sec])
    ax[0].plot(np.append(th, th[0]), np.append(wv, wv[0]), "o-")
    ax[0].set_title("(a) sector wake fraction w(theta)")
    rr = np.array([r["r_over_R"] for r in rad])
    rw = np.array([r["w"] for r in rad])
    ax[1].bar(rr, rw, width=0.09, color="C0")
    ax[1].set_title("(b) radial wake fraction w(r/R)")
    fig.tight_layout()
    path = out / "fig_wake.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="artifacts/selfpropulsion_kcs")
    args = ap.parse_args()
    out = Path(args.dir)
    geom = KP505Geometry()
    polar = SectionPolar()
    made = [fig_open_water(out, geom, polar), fig_loading(out, geom, polar),
            fig_power_chain(out), fig_selfprop(out, geom, polar)]
    if (out / "tunnel.json").exists():
        made.append(fig_tunnel(out))
    wake = fig_wake(out)
    if wake is not None:
        made.append(wake)
    for m in made:
        print(m)


if __name__ == "__main__":
    main()