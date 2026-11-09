#!/usr/bin/env python
"""Rotating-reference-frame *interior* flow verification (p158w1-n02-lbm-rot).

Three verification surfaces, four machine-checkable criteria:

S1  rotating-frame Taylor-Couette (annular, curved, closed, moving walls)
    C1  azimuthal profile vs ``A r + B/r`` on effective (staircase-corrected)
        radii: relative error ``<= 3 %``
    C4  angular-momentum residual of the Coriolis + centrifugal operator
        ``|T_c + Omega dI/dt| / |T_c| <= 1e-8`` (float64, divergence-free field)
    (plus a *discriminating* anchor that the velocity profile cannot see: the
     reduced-pressure profile ``p(r) - p(r_i)`` including ``2 Omega u`` and
     ``Omega^2 r``; dropping the frame terms moves it by ~80 % of its range)

S2/S3 rotating plane channel with rotation about the wall-normal axis
    C2  geostrophic secondary-flow strength ``max|u_z|`` vs the direct numerical
        (FD) reference: relative error ``<= 5 %``
    C3  Ekman thickness ``delta_E = sqrt(2 nu/f)`` from a blind one-parameter
        fit of the secondary-flow profile: relative error ``<= 5 %``

Usage::

    python run.py [--outdir records] [--figdir ../../../docs/assets/rotating_interior_p158w1_n02]
                  [--quick]

Every number written to the JSON records is produced by the code path above it;
nothing is transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from tensorlbm.rotating_interior import (  # noqa: E402
    RotatingAnnulusD2Q9,
    angular_momentum_residual,
    circular_couette,
    couette_pressure_rotating_frame,
    fd_rotating_channel,
    fit_channel_ekman_delta,
    fit_couette_AB,
    invert_effective_radii,
    rotating_channel_analytic,
    run_rotating_channel_lbm,
    stream_function_field,
)

CRIT = {"C1_profile": 0.03, "C2_secondary": 0.05, "C3_ekman": 0.05, "C4_angmom": 1e-8}


# --------------------------------------------------------------------------- #
# S1: rotating-frame Taylor-Couette
# --------------------------------------------------------------------------- #
def couette_case(ri: float, ro: float, omega_i: float, omega_frame: float, nu: float,
                 box: int, tol: float = 1e-7, max_steps: int = 60000) -> dict:
    t0 = time.time()
    sim = RotatingAnnulusD2Q9(
        (box, box), 3.0 * nu + 0.5, ri, ro, omega_i,
        omega_o=0.0, omega_frame=omega_frame, centrifugal=bool(omega_frame),
    )
    # start from the analytic viscous solution (fixed-point test): the discrete
    # steady state must not drift away from it
    th = np.arctan2(sim.dy, sim.dx)
    r = np.maximum(sim.rad, 1e-9)
    u = circular_couette(r, ri, ro, omega_i)["u_theta"]
    sim.initialize_from_velocity(-np.sin(th) * u, np.cos(th) * u)
    period = int(round(2.0 * math.pi / (2.0 * omega_frame))) if omega_frame else None
    conv = sim.run_until_steady(tol=tol, max_steps=max_steps, check_every=100, period=period)
    prof = sim.azimuthal_profile()

    rr = np.array(prof["r_bin"])
    uu = np.array(prof["u_theta"])
    pp = np.array(prof["p"])
    cnt = np.array(prof["count"])
    sel = (cnt > 0) & np.isfinite(uu) & (rr >= ri + 1) & (rr <= ro - 1)
    fit = fit_couette_AB(rr[sel], uu[sel], weight=cnt[sel])
    eff = invert_effective_radii(fit["A"], fit["B"], omega_i)
    ana_eff = circular_couette(rr[sel], eff["ri_eff"], eff["ro_eff"], omega_i)
    ana_nom = circular_couette(rr[sel], ri, ro, omega_i)
    u_max = omega_i * eff["ri_eff"]
    err_eff = float(np.max(np.abs(uu[sel] - ana_eff["u_theta"])) / u_max)
    err_nom = float(np.max(np.abs(uu[sel] - ana_nom["u_theta"])) / (omega_i * ri))

    r_f = rr[sel]
    dp = pp[sel] - pp[sel][0]
    pW = couette_pressure_rotating_frame(r_f, eff["ri_eff"], eff["ro_eff"], omega_i, omega_frame=omega_frame)
    p0 = couette_pressure_rotating_frame(r_f, eff["ri_eff"], eff["ro_eff"], omega_i, omega_frame=0.0)
    dW, d0 = pW - pW[0], p0 - p0[0]
    p_scale = float(np.max(np.abs(dW)))
    p_err = float(np.max(np.abs(dp - dW)) / p_scale)
    p_err_noframe = float(np.max(np.abs(dp - d0)) / p_scale)

    # C4: angular-momentum residual of the pseudo-force operator.  Field 1: a
    # discretely divergence-free random stream function + shear (machine-precision
    # identity); field 2: the physical Taylor-Couette field of this run (finite
    # divergence, so the identity holds only to truncation -- reported honestly).
    ff = stream_function_field(box, box, seed=15802, shear=0.01)
    yy, xx = np.meshgrid(np.arange(box, dtype=float), np.arange(box, dtype=float), indexing="ij")
    # rotation axis at the grid corner: arbitrary but non-degenerate (a centred
    # axis makes every moment of a periodic field cancel exactly, which would
    # turn the identity into 0/0 -- see tests/test_rotating_interior.py)
    x, y = xx, yy
    am_divfree = angular_momentum_residual(np.ones((box, box)), ff[0], ff[1], x, y, omega_frame)
    rho_f, ux_f, uy_f = sim.fields()
    div_f = float(np.max(np.abs(
        (np.roll(ux_f, -1, 1) - np.roll(ux_f, 1, 1)) / 2
        + (np.roll(uy_f, -1, 0) - np.roll(uy_f, 1, 0)) / 2)))
    am_flow = angular_momentum_residual(rho_f, ux_f, uy_f, x, y, omega_frame)

    return {
        "case": {"kind": "rotating_couette", "ri": ri, "ro": ro, "eta": ri / ro,
                 "box": box, "tau": 3.0 * nu + 0.5, "nu": nu, "omega_i": omega_i,
                 "omega_frame": omega_frame, "centrifugal": bool(omega_frame),
                 "initialisation": "analytic viscous profile (fixed-point test)"},
        "convergence": {k: conv[k] for k in ("steps", "residual", "tol", "period", "converged")},
        "Ma": prof["Ma"],
        "radial_velocity_max": float(np.nanmax(np.abs(np.array(prof["u_rad"])))),
        "profile": {"r": [float(v) for v in rr[sel]], "u_theta": [float(v) for v in uu[sel]],
                    "u_theta_analytic_eff": [float(v) for v in ana_eff["u_theta"]],
                    "u_theta_analytic_nominal": [float(v) for v in ana_nom["u_theta"]],
                    "count": [int(v) for v in cnt[sel]]},
        "fit": fit,
        "effective_radii": eff,
        "delta_r_inner": eff["ri_eff"] - ri,
        "delta_r_outer": eff["ro_eff"] - ro,
        "profile_max_rel_err_eff": err_eff,
        "profile_max_rel_err_nominal": err_nom,
        "pressure": {"r": [float(v) for v in r_f], "measured": [float(v) for v in dp],
                     "analytic_rotating": [float(v) for v in dW],
                     "analytic_inertial_frame": [float(v) for v in d0],
                     "max_rel_err_vs_rotating": p_err,
                     "max_rel_err_vs_inertial": p_err_noframe,
                     "scale": p_scale},
        "angmom_divfree": am_divfree,
        "angmom_flow": am_flow,
        "flow_max_divergence": div_f,
        "wall_seconds": time.time() - t0,
    }


# --------------------------------------------------------------------------- #
# S2/S3: rotating plane channel
# --------------------------------------------------------------------------- #
def channel_case(h: int, delta_nodes: float, nu: float, init: str, uz_core: float = 0.02,
                 max_steps: int = 20000, tol: float = 1e-7) -> dict:
    res = run_rotating_channel_lbm(
        h=h, nu=nu, delta_nodes=delta_nodes, uz_core_target=uz_core,
        init=init, tol=tol, max_steps=max_steps,
    )
    y = np.array(res["y"])
    ux = np.array(res["ux"])
    uz = np.array(res["uz"])
    G, Om, hh = res["G"], res["Omega"], res["h"]
    ax, az = rotating_channel_analytic(y, hh, G, Om, nu)
    fd = fd_rotating_channel(G, Om, nu, float(hh), n=2001)
    ax_fd = np.interp(y, fd["y"], fd["ux"])
    az_fd = np.interp(y, fd["y"], fd["uz"])
    fd_vs_analytic = float(np.max(np.abs(fd["uz"] - rotating_channel_analytic(fd["y"], hh, G, Om, nu)[1]))
                           / np.max(np.abs(fd["uz"])))
    strength_meas = float(np.max(np.abs(uz)))
    strength_dns = float(np.max(np.abs(az_fd)))
    err_strength = abs(strength_meas - strength_dns) / strength_dns
    err_core = abs(uz[np.argmin(np.abs(y))] - az_fd[np.argmin(np.abs(y))]) / abs(az_fd[np.argmin(np.abs(y))])
    # whole-profile error, and the same with the wall-adjacent node removed
    err_prof = float(np.max(np.hypot(ux - ax_fd, uz - az_fd)) / strength_dns)
    inner = np.abs(y) <= hh - 2
    err_prof_inner = float(np.max(np.hypot(ux - ax_fd, uz - az_fd)[inner]) / strength_dns)
    err_l2 = float(np.sqrt(np.mean((ux - ax_fd) ** 2 + (uz - az_fd) ** 2)) / strength_dns)
    fit = fit_channel_ekman_delta(y, uz, hh, G, Om, nu)
    err_delta = abs(fit["delta_fit"] - fit["delta_theory"]) / fit["delta_theory"]
    return {
        "case": {"kind": "rotating_channel", "h": hh, "delta_nodes": delta_nodes,
                 "nu": nu, "tau": 3.0 * nu + 0.5, "Omega": Om, "f_coriolis": 2 * Om,
                 "G": G, "uz_core_target": uz_core, "init": init,
                 "ny": res["ny"], "period_steps": res["period"]},
        "convergence": {"steps": res["steps"], "residual": res["residual"],
                        "tol": res["converge_tol"], "converged": res["converged"]},
        "dns_reference": {"method": "2nd-order FD direct solve of the rotating NS "
                                    "(nu ux''-2Omega uz=-G, nu uz''+2Omega ux=0)",
                          "n_nodes": fd["n"], "max_rel_err_vs_closed_form_uz": fd_vs_analytic},
        "profile": {"y": [float(v) for v in y], "ux": [float(v) for v in ux],
                    "uz": [float(v) for v in uz], "ux_dns": [float(v) for v in ax_fd],
                    "uz_dns": [float(v) for v in az_fd], "uy_max": float(np.max(np.abs(res["uy"])))},
        "strength": {"max_uz_meas": strength_meas, "max_uz_dns": strength_dns,
                     "rel_err": err_strength, "core_uz_rel_err": err_core},
        "profile_rel_err_max": err_prof,
        "profile_rel_err_max_inner": err_prof_inner,
        "profile_rel_err_l2": err_l2,
        "ekman": {**fit, "rel_err": err_delta},
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(summary: dict, figdir: str) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(figdir, exist_ok=True)
    figs = []
    c1 = [c for c in summary["cases"] if c["case"]["kind"] == "rotating_couette"]
    c2 = [c for c in summary["cases"] if c["case"]["kind"] == "rotating_channel"]

    # Fig 1: Taylor-Couette profile + error
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for rec in c1:
        r = np.array(rec["profile"]["r"]); u = np.array(rec["profile"]["u_theta"])
        ua = np.array(rec["profile"]["u_theta_analytic_eff"])
        lab = f"$\\Omega_f$={rec['case']['omega_frame']:g}, {rec['case']['box']}$^2$"
        ax[0].plot(r, u, "o", ms=4, label=f"LBM {lab}")
        ax[0].plot(r, ua, "-", lw=1.0)
        ax[1].semilogy(r, np.maximum(np.abs(u - ua) / (rec["case"]["omega_i"] * rec["effective_radii"]["ri_eff"]), 1e-6),
                       "o-", ms=3, lw=1.0, label=lab)
    ax[0].set_xlabel("$r$ [lattice]"); ax[0].set_ylabel(r"$u_\theta$")
    ax[0].set_title("rotating-frame Taylor-Couette: profile (markers) vs analytic (lines)")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    ax[1].axhline(CRIT["C1_profile"], color="r", ls="--", lw=1, label="criterion 3%")
    ax[1].set_xlabel("$r$"); ax[1].set_ylabel(r"$|u_\theta^{LBM}-u_\theta^{ana}|/u_{max}$")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    fig.tight_layout(); p = os.path.join(figdir, "fig1_couette_profile.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    figs.append({"name": "fig1_couette_profile.png", "path": p,
                 "caption": "旋转参考系 Taylor-Couette：方位角速度剖面（LBM 符号 vs 有效几何解析线）与逐点相对误差（判据 3%）"})

    # Fig 2: rotating-frame pressure anchor
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.4))
    rec = [c for c in c1 if c["case"]["omega_frame"] != 0][0]
    r = np.array(rec["pressure"]["r"])
    ax.plot(r, np.array(rec["pressure"]["measured"]) * 1e3, "o", ms=4, label="LBM (measured)")
    ax.plot(r, np.array(rec["pressure"]["analytic_rotating"]) * 1e3, "-", lw=1.2,
            label=r"analytic rotating frame ($2\Omega u+\Omega^2 r$)")
    ax.plot(r, np.array(rec["pressure"]["analytic_inertial_frame"]) * 1e3, "--", lw=1.2,
            label="analytic, frame terms dropped")
    ax.set_xlabel("$r$ [lattice]"); ax.set_ylabel(r"$p(r)-p(r_i)$  [$10^{-3}$]")
    ax.set_title("rotating-frame pressure anchor (discriminates Coriolis/centrifugal)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); p = os.path.join(figdir, "fig2_couette_pressure.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    figs.append({"name": "fig2_couette_pressure.png", "path": p,
                 "caption": "旋转系折合压力剖面：实测 vs 含 Coriolis/离心项的解析解 vs 丢掉框架项的解析解（后者偏差 ~80%）"})

    # Fig 3: rotating channel secondary flow + Ekman decay
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for rec in c2:
        y = np.array(rec["profile"]["y"]); uz = np.array(rec["profile"]["uz"])
        uzd = np.array(rec["profile"]["uz_dns"]); uxd = np.array(rec["profile"]["ux_dns"])
        ux = np.array(rec["profile"]["ux"])
        d = rec["case"]["delta_nodes"]
        ax[0].plot(uz, y / d, "-", lw=1.2, label=f"LBM $u_z$, $\\delta$={d:g}")
        ax[0].plot(uzd, y / d, "--", lw=1.0, label="DNS $u_z$")
        ax[0].plot(ux, y / d, "-", lw=0.8, alpha=.6, label=f"LBM $u_x$, $\\delta$={d:g}")
    ax[0].set_xlabel("velocity"); ax[0].set_ylabel(r"$y/\delta_E$")
    ax[0].set_title("rotating channel: geostrophic $u_z$ + Ekman layers")
    ax[0].legend(fontsize=6); ax[0].grid(alpha=.3)
    rec = c2[0]
    y = np.array(rec["profile"]["y"]); uz = np.array(rec["profile"]["uz"])
    core = np.max(np.abs(uz))
    ax[1].plot(y / rec["case"]["delta_nodes"], np.abs(np.abs(uz) - core) / core + 1e-12, "o-", ms=3, lw=1,
               label="measured deficit")
    ax[1].set_xlabel(r"$y/\delta_E$"); ax[1].set_ylabel("deficit")
    ax[1].set_yscale("log"); ax[1].set_title("Ekman decay from the walls")
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    fig.tight_layout(); p = os.path.join(figdir, "fig3_rotating_channel.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    figs.append({"name": "fig3_rotating_channel.png", "path": p,
                 "caption": "旋转通道（绕壁法向轴）：展向地转流 u_z 与双壁 Ekman 层（符号=LBM，虚线=直接数值解），右：壁面 Ekman 衰减"})

    # Fig 4: criteria dashboard
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 3.8))
    names = ["C1 profile", "C2 secondary", "C3 Ekman", "C4 ang.mom."]
    vals = [summary["criteria"]["C1_profile"]["value"], summary["criteria"]["C2_secondary"]["value"],
            summary["criteria"]["C3_ekman"]["value"], summary["criteria"]["C4_angmom"]["value"]]
    tol = [CRIT["C1_profile"], CRIT["C2_secondary"], CRIT["C3_ekman"], CRIT["C4_angmom"]]
    ratio = [v / t for v, t in zip(vals, tol)]
    bars = ax.bar(names, ratio, color=["tab:green" if r <= 1 else "tab:red" for r in ratio])
    ax.axhline(1.0, color="r", ls="--", lw=1)
    ax.set_yscale("log"); ax.set_ylabel("measured error / tolerance")
    ax.set_title("criteria dashboard (bar <= 1 passes)")
    for b, v, t in zip(bars, vals, tol):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.3, f"{v:.2e}\n(≤{t:g})",
                ha="center", fontsize=7)
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); p = os.path.join(figdir, "fig4_criteria.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    figs.append({"name": "fig4_criteria.png", "path": p,
                 "caption": "四条机器判据：实测误差/容差（对数轴，柱高≤1 即达标）"})
    return figs


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "records"))
    ap.add_argument("--figdir", default=os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "docs", "assets", "rotating_interior_p158w1_n02"))
    ap.add_argument("--quick", action="store_true", help="coarse grids / few steps (smoke test)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    cases = []
    # S1: two frame rates (Omega_f = 0 inertial check, Omega_f = 0.002 rotating)
    box = 60 if not args.quick else 44
    ri, ro = (12.0, 24.0) if not args.quick else (9.0, 18.0)
    for wf in (0.0, 0.002):
        cases.append(couette_case(ri, ro, 0.004, wf, 0.05, box,
                                  max_steps=60000 if not args.quick else 3000))
    # grid refinement on the rotating-frame case
    if not args.quick:
        cases.append(couette_case(1.4 * ri, 1.4 * ro, 0.004 / 1.4, 0.002, 0.05, 84))

    # S2/S3: rotating plane channel (Ekman secondary flow)
    ch_specs = [(24, 8.0, "analytic"), (24, 8.0, "rest"), (24, 6.0, "analytic"), (36, 12.0, "analytic")]
    if args.quick:
        ch_specs = [(12, 4.0, "analytic")]
    for h, dn, init in ch_specs:
        cases.append(channel_case(h, dn, 0.1, init,
                                  max_steps=20000 if not args.quick else 1200,
                                  tol=1e-7))

    c1 = [c for c in cases if c["case"]["kind"] == "rotating_couette"]
    c2 = [c for c in cases if c["case"]["kind"] == "rotating_channel"]
    crit = {
        "C1_profile": {
            "quantity": "max |u_theta^LBM - (A r + B/r)| / (omega_i r_i,eff) over the annulus",
            "value": max(c["profile_max_rel_err_eff"] for c in c1),
            "tolerance": CRIT["C1_profile"], "direction": "<=",
            "cases": [c["case"]["omega_frame"] for c in c1]},
        "C2_secondary": {
            "quantity": "|max|u_z|_LBM - max|u_z|_DNS| / max|u_z|_DNS (geostrophic secondary flow)",
            "value": max(c["strength"]["rel_err"] for c in c2),
            "tolerance": CRIT["C2_secondary"], "direction": "<=",
            "cases": [f"h={c['case']['h']},delta={c['case']['delta_nodes']}" for c in c2]},
        "C3_ekman": {
            "quantity": "|delta_fit - sqrt(2 nu/f)| / sqrt(2 nu/f) (blind 1-parameter fit)",
            "value": max(c["ekman"]["rel_err"] for c in c2),
            "tolerance": CRIT["C3_ekman"], "direction": "<=",
            "cases": [f"h={c['case']['h']},delta={c['case']['delta_nodes']}" for c in c2]},
        "C4_angmom": {
            "quantity": "|T_coriolis+cf + Omega dI/dt| / |T_c| on a discretely divergence-free field (float64)",
            "value": max(c["angmom_divfree"]["residual_rel"] for c in c1),
            "tolerance": CRIT["C4_angmom"], "direction": "<=",
            "cases": ["stream-function field, all frame rates"]},
    }
    for k, v in crit.items():
        v["pass"] = bool(v["value"] <= v["tolerance"])
    summary = {
        "mission": "p158w1-n02-lbm-rot",
        "repo": "TensorLBM",
        "module": "src/tensorlbm/rotating_interior.py",
        "criteria": crit,
        "all_pass": all(v["pass"] for v in crit.values()),
        "cases": cases,
        "wall_seconds": time.time() - t0,
    }
    figs = make_figures(summary, args.figdir)
    summary["figures"] = [{"name": f["name"], "caption": f["caption"]} for f in figs]

    tag = "quick" if args.quick else "full"
    out = os.path.join(args.outdir, f"rotating_interior_{tag}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)

    print(f"record -> {out}")
    for k, v in crit.items():
        print(f"  {k}: {v['value']:.4e}  (<= {v['tolerance']:g})  {'PASS' if v['pass'] else 'FAIL'}")
    print("  all_pass:", summary["all_pass"], f" wall={summary['wall_seconds']:.1f}s")
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())