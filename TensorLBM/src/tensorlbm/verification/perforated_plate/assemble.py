"""Assemble per-case results into the verification report artefacts.

Produces, inside the run directory:

* ``metrics.json``  - all scalars + gate verdicts (machine readable)
* ``*.csv``         - tidy tables (C_d(KC), C_d(beta), GCI, negative controls)
* ``figures/*.png`` - publication figures
* ``REPORT.md``     - the six-element paper-style report generated from the
  numbers above (no hand-copied values)
"""

from __future__ import annotations

import csv
import json
import math
import os

import numpy as np

from .driver import build_cases
from .gates import evaluate_all

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_cases(run_dir: str) -> list[dict]:
    cases = []
    for spec in build_cases():
        path = os.path.join(run_dir, f"case_{spec['id']}.json")
        if os.path.exists(path):
            with open(path) as fh:
                data = json.load(fh)
            data.setdefault("level", spec.get("level"))
            cases.append(data)
    return cases


def _num(v, fmt="{:.4g}"):
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return "nan"
    return fmt.format(f)


def _write_csv(path: str, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _fig_kc(cases, gates, outdir):
    kc = sorted([c for c in cases if c["kind"] == "kc"], key=lambda c: c["KC"])
    if not kc:
        return None
    x = [c["KC"] for c in kc]
    cd = [c["C_d"] for c in kc]
    cm = [c["C_m"] for c in kc]
    lag = [c["phase_lag_deg"] for c in kc]
    ref = kc[0]["C_d_reference"]

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ax[0].semilogx(x, cd, "o-", color="#1f4e9c", label=r"$C_d$ (LBM fit)")
    ax[0].axhline(ref, color="k", ls="--", lw=1.0,
                  label=rf"Borda-Carnot $C_c/(1-\beta C_c)$={ref:.3f}")
    ax[0].fill_between([min(x) * 0.8, max(x) * 1.25], ref * 0.9, ref * 1.1,
                       color="k", alpha=0.08, label=r"$\pm10\%$ band")
    ax[0].set_xlabel("KC = $u_h T/d$")
    ax[0].set_ylabel(r"$C_d$")
    ax[0].set_title(r"discharge coefficient vs KC ($\beta$=%.3f)" % kc[0]["beta"])
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    ax[1].semilogx(x, cm, "s-", color="#b5651d", label=r"$C_m = K_m/t$")
    ax[1].set_xlabel("KC")
    ax[1].set_ylabel(r"$C_m$")
    ax[1].set_title("added-mass coefficient")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)

    ax[2].semilogx(x, lag, "^-", color="#2a7f3f", label=r"phase($\Delta p$ ahead of $Q$)")
    ax[2].axhline(90, color="grey", ls=":", lw=1, label="pure inertia (90$^\\circ$)")
    ax[2].axhline(0, color="grey", ls=":", lw=1, label="pure loss (0$^\\circ$)")
    ax[2].set_xlabel("KC")
    ax[2].set_ylabel("phase lag [deg]")
    ax[2].set_title(r"$\rho_{Spearman}$=%.3f" % gates["B_phase_monotonic"].get("spearman_rho", float("nan")))
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=0.3)
    for a in ax:
        a.axvspan(15, max(x) * 1.3, color="tab:blue", alpha=0.05)
    fig.tight_layout()
    path = os.path.join(outdir, "fig1_cd_cm_phase_vs_KC.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _fig_beta(cases, gates, outdir):
    rows = sorted([c for c in cases if c["kind"] == "beta"], key=lambda c: c["beta"])
    if len(rows) < 2:
        return None
    b = [c["beta"] for c in rows]
    cd = [c["C_d"] for c in rows]
    ref = [c["C_d_reference"] for c in rows]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(b, cd, "o-", color="#1f4e9c", label=r"$C_d$ (LBM)")
    ax.plot(b, ref, "k--", lw=1.2, label=r"$C_d=C_c/(1-\beta C_c)$")
    ax.set_xlabel(r"open-area ratio $\beta$")
    ax.set_ylabel(r"$C_d$")
    ax.set_title(r"$C_d(\beta)$ at KC$\approx$%.1f, $\rho_{Spearman}$=%.2f"
                 % (np.mean([c["KC"] for c in rows]),
                    gates["C_cd_vs_beta"].get("spearman_rho", float("nan"))))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, "fig2_cd_vs_beta.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _fig_gci(cases, gates, outdir):
    rows = sorted([c for c in cases if c["kind"] == "gci"], key=lambda c: c["ny"])
    if len(rows) < 3:
        return None
    d = [c["hole_size"] for c in rows]
    cd = [c["C_d"] for c in rows]
    g = gates["D_gci"]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(d, cd, "o-", color="#1f4e9c", label=r"$C_d$ (LBM)")
    ax.axhline(g["C_d_richardson"], color="k", ls="--", lw=1.0,
               label=r"Richardson $C_d$=%.4f" % g["C_d_richardson"])
    for c in rows:
        ax.annotate(f"KC={c['KC']:.2f}", (c["hole_size"], c["C_d"]),
                    textcoords="offset points", xytext=(4, 6), fontsize=7)
    ax.set_xlabel("hole width d [lattice units] (r = 2 between levels)")
    ax.set_ylabel(r"$C_d$")
    ax.set_title("grid convergence: GCI$_{fine}$=%.2f%% (limit 5%%)" % (100 * g["GCI_fine"]))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, "fig3_gci.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _fig_phase(cases, gates, outdir):
    rows = [c for c in cases if c["kind"] == "kc" and c.get("phase")]
    if not rows:
        return None
    rows = sorted(rows, key=lambda c: c["KC"])
    pick = [rows[0], rows[len(rows) // 2], rows[-1]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    colors = ["#1f4e9c", "#b5651d", "#2a7f3f"]
    for c, col in zip(pick, colors):
        ph = c["phase"]
        q = np.array(ph["q"]) * 1e3
        dp = np.array(ph["dp"]) * 1e3
        ax[0].plot(q, dp, "-", color=col, lw=1.4,
                   label=f"KC={c['KC']:.2f}  $C_d$={c['C_d']:.3f}  lag={c['phase_lag_deg']:.0f}$^\\circ$")
        ax[1].plot(ph["theta"], dp, "-", color=col, lw=1.2,
                   label=f"KC={c['KC']:.2f}")
    ax[0].set_xlabel(r"$Q \times 10^3$")
    ax[0].set_ylabel(r"$\Delta p \times 10^3$")
    ax[0].set_title(r"$\Delta p$-$Q$ orbit (area = $\Phi$)")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)
    ax[1].set_xlabel("phase [rad]")
    ax[1].set_ylabel(r"$\Delta p \times 10^3$")
    ax[1].set_title("phase-averaged pressure drop")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "fig4_phase_orbit.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _fig_nc(cases, gates, outdir):
    row = next((c for c in cases if c["kind"] == "nc"), None)
    if row is None:
        return None
    nc = row["nc"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    # (1) dp amplitude comparison
    labels = ["$\\Delta p$", "$\\tau_w$", "profile"]
    meas = [nc["dp_amp_measured"], nc["tau_amp_measured"], 1.0 - nc["profile_rel_error_l2"]]
    ref = [nc["dp_amp_analytic"], nc["tau_amp_analytic"], 1.0]
    idx = np.arange(3)
    ax[0].bar(idx - 0.2, meas, 0.4, label="LBM", color="#1f4e9c")
    ax[0].bar(idx + 0.2, ref, 0.4, label="analytic", color="#b5651d")
    ax[0].set_xticks(idx)
    ax[0].set_xticklabels(labels)
    ax[0].set_title("negative controls: amplitude match")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, axis="y")

    errs = [nc["dp_rel_error"], nc["tau_rel_error"], nc["profile_rel_error_l2"]]
    thr = [0.03, 0.05, 0.03]
    ax[1].bar(idx, [100 * e for e in errs], 0.5, color="#2a7f3f")
    for i, t in enumerate(thr):
        ax[1].hlines(100 * t, i - 0.4, i + 0.4, color="r", ls="--", lw=1)
    ax[1].set_xticks(idx)
    ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("relative error [%]")
    ax[1].set_title("relative error vs analytic (red = limit)")
    ax[1].grid(alpha=0.3, axis="y")

    y = np.array(nc["y"])
    prof = nc.get("profile_amp")
    prof_ref = nc.get("profile_amp_ref")
    if prof and prof_ref:
        ax[2].plot(prof, y, "o-", ms=3, color="#1f4e9c", label="LBM")
        ax[2].plot(prof_ref, y, "k--", lw=1.1,
                   label=r"Womersley ($\delta$=%.1f)" % nc["stokes_delta"])
        ax[2].set_xlabel(r"$|\hat u(y)|$")
        ax[2].set_ylabel("y [lattice]")
        ax[2].set_title("NC1: oscillating channel profile")
        ax[2].legend(fontsize=8)
        ax[2].grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "fig5_negative_controls.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _fig_energy(cases, gates, outdir):
    rows = sorted([c for c in cases if c["kind"] in ("kc", "beta", "gci")
                   and c.get("KC", 0) > 0.2], key=lambda c: c["KC"])
    if not rows:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    x = [c["KC"] for c in rows]
    ax[0].loglog(x, [c["phi"] for c in rows], "o", color="#1f4e9c",
                 label=r"$\Phi$ momentum exchange")
    ax[0].loglog(x, [c["phi_stations"] for c in rows], "x", color="#b5651d",
                 label=r"$\Phi$ pressure stations")
    ax[0].loglog(x, [c.get("phi_model", float("nan")) for c in rows], ".",
                 color="k", ms=3, label=r"$\Phi$ quasi-steady model")
    ax[0].set_xlabel("KC")
    ax[0].set_ylabel(r"$\Phi$ per cycle")
    ax[0].set_title("energy budget: two independent paths")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, which="both")

    ax[1].loglog(x, [abs(c["phi"] - c["phi_stations"]) / abs(c["phi"]) for c in rows],
                 "o", color="#2a7f3f")
    ax[1].axhline(0.03, color="r", ls="--", lw=1, label="3% limit")
    ax[1].set_xlabel("KC")
    ax[1].set_ylabel(r"$|\Phi_{force}-\Phi_{stations}|/\Phi$")
    ax[1].set_title("closure error (max %.2f%%)"
                    % (100 * gates["F_energy_closure"].get("max_rel_error", float("nan"))))
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(outdir, "fig6_energy_budget.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _fig_field(run_dir, cases, outdir):
    kc = sorted([c for c in cases if c["kind"] == "kc"], key=lambda c: c["KC"])
    if not kc:
        return None
    cand = [kc[len(kc) // 2], kc[-1]]
    figs = []
    for c in cand:
        path = os.path.join(run_dir, f"field_{c['id']}.npz")
        if not os.path.exists(path):
            continue
        z = np.load(path)
        ux, vort, solid = z["ux"], z["vorticity"], z["solid"]
        fig, ax = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
        umax = np.abs(ux).max()
        ax[0].imshow(ux.T, origin="lower", cmap="RdBu_r", vmin=-umax, vmax=umax,
                     aspect="auto")
        ax[0].contour(solid.T, levels=[0.5], colors="k", linewidths=0.6)
        ax[0].set_title(f"u_x at peak flow, KC={c['KC']:.1f}, "
                        r"$C_d$=%.3f" % c["C_d"])
        ax[0].set_ylabel("y")
        vm = np.percentile(np.abs(vort), 99.5)
        ax[1].imshow(vort.T, origin="lower", cmap="coolwarm", vmin=-vm, vmax=vm,
                     aspect="auto")
        ax[1].contour(solid.T, levels=[0.5], colors="k", linewidths=0.6)
        ax[1].set_title("vorticity  $\\omega_z$")
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("y")
        fig.tight_layout()
        p = os.path.join(outdir, f"fig7_field_{c['id']}.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        figs.append(p)
    return figs[0] if figs else None


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _report_markdown(cases, gates, figures, run_dir) -> str:
    kc = sorted([c for c in cases if c["kind"] == "kc"], key=lambda c: c["KC"])
    beta = sorted([c for c in cases if c["kind"] == "beta"], key=lambda c: c["beta"])
    gci = sorted([c for c in cases if c["kind"] == "gci"], key=lambda c: c["ny"])
    nc = next((c for c in cases if c["kind"] == "nc"), None)

    L = []
    L.append("# Perforated-plate oscillatory-flow verification (P333W2-N06)\n")
    L.append("## 1. Case introduction\n")
    L.append("Forced oscillation of an incompressible viscous fluid through a "
             "sharp-edged perforated plate in a 2-D channel, driven by a "
             "sinusoidal body force `F_x = F0 sin(omega t)`. The plate has "
             "`n_holes` rectangular holes of width `d` in a plate of thickness "
             "`t`; the open-area ratio is `beta = n_holes*d/(ny-2)`.\n")
    L.append("Deliverables: `C_d(KC,beta)`, `C_m(KC,beta)`, cycle dissipation "
             "`Phi` and the equivalent sloshing damping "
             "`zeta_eff = Phi/(2 pi omega 2 E_kin)` plus its quasi-steady "
             "analytic counterpart.\n")
    if kc:
        c0 = kc[0]
        L.append(f"KC sweep geometry: nx={c0['nx']}, ny={c0['ny']}, d={c0['hole_size']}, "
                 f"t={c0['plate_thickness']}, beta={c0['beta']:.4f}, "
                 f"nu={c0['nu']:.4g} (tau={c0['tau']}), T={c0['period']:.0f} steps, "
                 f"{len(kc)} force amplitudes from {kc[0]['force_amplitude']:.2e} to "
                 f"{kc[-1]['force_amplitude']:.2e}.\n")
    L.append("\n## 2. Method\n")
    L.append("* D2Q9 BGK, Guo forcing with the `u = (sum c f + F/2)/rho` correction.\n"
             "* Half-way bounce-back on every solid link (walls + plate), so the "
             "hydrodynamic plate thickness equals the geometric one.\n"
             "* Plate force by momentum exchange (Ladd): `F = sum_links 2 c_i f_i`.\n"
             "* Pressure probes on two stations upstream/downstream plus the "
             "explicit slab momentum budget (inertia, body force, wall shear, "
             "advective flux) to convert them into a plate pressure drop.\n"
             "* Initial condition: zero-mean Womersley AC profile, which removes the "
             "spurious start-up mean flow (`F0/omega`, decaying on `h^2/(pi^2 nu)`) "
             "that otherwise suppresses the oscillation by ~2 orders of magnitude "
             "through the quadratic plate drag.\n"
             "* Fit: `dp = rho K_m du_h/dt + 0.5 rho K_q u_h|u_h|` with `K_q=1/C_d^2`, "
             "`C_m = K_m/t`, by least squares on the phase-averaged cycle "
             "(both basis functions are mean free and orthogonal).\n"
             "* `Phi = oint dp Q dt`; `zeta_eff = Phi/(2 pi omega 2 E_kin)`; "
             "`E_kin` = cycle-mean kinetic energy of the simulated field.\n"
             "* Grid convergence: three self-similar levels (r = 2), Roache GCI with "
             "`Fs = 1.25`.\n")
    L.append("\n## 3. Results\n")
    if kc:
        L.append("\n### 3.1 KC sweep\n\n")
        L.append("| case | KC | Re_h | C_d | C_d ref | err | C_m | r^2 | lag [deg] | Phi | Phi stations |\n")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for c in kc:
            err = abs(c["C_d"] - c["C_d_reference"]) / c["C_d_reference"]
            L.append(f"| {c['id']} | {c['KC']:.2f} | {c['Re_h']:.1f} | {_num(c['C_d'],'{:.4f}')} | "
                     f"{c['C_d_reference']:.4f} | {100*err:.1f}% | {_num(c['C_m'],'{:.3f}')} | "
                     f"{_num(c['r_squared'],'{:.3f}')} | {_num(c['phase_lag_deg'],'{:.1f}')} | "
                     f"{_num(c['phi'],'{:.3e}')} | {_num(c['phi_stations'],'{:.3e}')} |\n")
    if beta:
        L.append("\n### 3.2 beta sweep\n\n| case | beta | KC | Re_h | C_d | C_d ref | err | C_m | r^2 |\n")
        L.append("|---|---|---|---|---|---|---|---|---|\n")
        for c in beta:
            err = abs(c["C_d"] - c["C_d_reference"]) / c["C_d_reference"]
            L.append(f"| {c['id']} | {c['beta']:.4f} | {c['KC']:.2f} | {c['Re_h']:.1f} | "
                     f"{_num(c['C_d'],'{:.4f}')} | {c['C_d_reference']:.4f} | {100*err:.1f}% | "
                     f"{_num(c['C_m'],'{:.3f}')} | {_num(c['r_squared'],'{:.3f}')} |\n")
    if gci:
        g = gates["D_gci"]
        L.append("\n### 3.3 Grid convergence\n\n| level | ny | d | KC | C_d | steps |\n|---|---|---|---|---|---|\n")
        for c in gci:
            L.append(f"| {c['id']} | {c['ny']} | {c['hole_size']} | {c['KC']:.2f} | "
                     f"{_num(c['C_d'],'{:.4f}')} | {c.get('steps')} |\n")
        L.append(f"\nApparent order p = {g['apparent_order']:.2f}; "
                 f"GCI_fine = {100*g['GCI_fine']:.2f}% (limit 5%); "
                 f"Richardson C_d = {g['C_d_richardson']:.4f}.\n")
    if nc:
        d = nc["nc"]
        L.append("\n### 3.4 Negative controls\n\n"
                 f"* NC1 (no plate) pressure budget: LBM {d['dp_amp_measured']:.4e} vs "
                 f"analytic {d['dp_amp_analytic']:.4e} -> {100*d['dp_rel_error']:.2f}% "
                 f"(limit 3%); velocity profile L2 error {100*d['profile_rel_error_l2']:.2f}%.\n"
                 f"* NC2 Stokes layer wall shear: LBM {d['tau_amp_measured']:.4e} vs "
                 f"analytic {d['tau_amp_analytic']:.4e} -> {100*d['tau_rel_error']:.2f}% "
                 f"(limit 5%), phase error {d['tau_phase_error_deg']:.1f} deg; "
                 f"delta = {d['stokes_delta']:.2f} lattice (delta/h = {d['stokes_delta_over_h']:.3f}).\n")
    L.append("\n### 3.5 Gate verdicts\n\n| gate | verdict | key number |\n|---|---|---|\n")
    keymap = {
        "A_quasi_steady": lambda g: ("%.2f%%" % (100 * g.get("worst", {}).get("rel_error", float("nan")))) if g.get("rows") else g.get("reason", "-"),
        "B_phase_monotonic": lambda g: "rho=%.3f, lag %.1f -> %.1f deg" % (
            g.get("spearman_rho", float("nan")), g.get("lag_low_kc_deg", float("nan")),
            g.get("lag_high_kc_deg", float("nan"))),
        "C_cd_vs_beta": lambda g: "rho=%.3f, C_d %.3f..%.3f" % (
            g.get("spearman_rho", float("nan")),
            min([r["C_d"] for r in g.get("rows", [{"C_d": float("nan")}])]),
            max([r["C_d"] for r in g.get("rows", [{"C_d": float("nan")}])])),
        "D_gci": lambda g: "GCI=%.2f%%" % (100 * g.get("GCI_fine", float("nan"))),
        "E_negative_controls": lambda g: "dp %.2f%%, tau %.2f%%, profile %.2f%%" % (
            100 * g.get("detail", {}).get("dp_rel_error", float("nan")),
            100 * g.get("detail", {}).get("tau_rel_error", float("nan")),
            100 * g.get("detail", {}).get("profile_rel_error_l2", float("nan"))),
        "F_energy_closure": lambda g: "max %.2f%%, median %.2f%%" % (
            100 * g.get("max_rel_error", float("nan")),
            100 * g.get("median_rel_error", float("nan"))),
    }
    for name, g in gates.items():
        try:
            kk = keymap[name](g)
        except Exception:
            kk = "-"
        L.append(f"| {name} | {'PASS' if g.get('passed') else 'FAIL'} | {kk} |\n")
    overall = all(g.get("passed") for g in gates.values())
    L.append(f"\n**Overall: {'ALL GATES PASS' if overall else 'SOME GATES FAIL'}**\n")
    if figures:
        L.append("\n## 4. Figures\n")
        for p, cap in figures:
            L.append(f"![{os.path.basename(p)}]({os.path.basename(p)})\n\n*{cap}*\n")
    return "".join(L)


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------

def assemble(run_dir: str) -> str:
    cases = load_cases(run_dir)
    if not cases:
        raise SystemExit(f"no case_*.json in {run_dir}")
    gates = evaluate_all(cases)
    gd = gates["gates"]

    figdir = os.path.join(run_dir, "figures")
    os.makedirs(figdir, exist_ok=True)

    figures = []
    for fn, cap in [
        (_fig_kc, "C_d, C_m and Delta p-Q phase lag versus KC; shaded band = quasi-steady limit +-10%."),
        (_fig_beta, "C_d versus open-area ratio beta with the Borda-Carnot sharp-edge reference."),
        (_fig_gci, "Grid convergence of C_d over three self-similar resolutions (r=2)."),
        (_fig_phase, "Phase-averaged pressure drop: Lissajous orbits (area = Phi) and phase waveforms."),
        (_fig_nc, "Negative controls: unobstructed channel pressure budget, wall shear and velocity profile."),
        (_fig_energy, "Energy budget: Phi from the momentum-exchange and pressure-station paths and its closure."),
        (_fig_field, "Velocity and vorticity field at peak flow through the perforated plate."),
    ]:
        try:
            p = fn(run_dir, cases, figdir) if fn is _fig_field else fn(cases, gd, figdir)
        except Exception as exc:  # pragma: no cover
            print(f"[warn] figure {fn.__name__} failed: {exc}")
            p = None
        if p:
            figures.append((p, cap))

    metrics = {
        "run_dir": run_dir,
        "n_cases": len(cases),
        "gates": gates["gates"],
        "all_passed": gates["all_passed"],
        "thresholds": gates["thresholds"],
        "cases": [{k: v for k, v in c.items() if k not in ("phase", "nc")} for c in cases],
        "negative_control": next((c["nc"] for c in cases if c["kind"] == "nc"), None),
        "figures": [os.path.basename(p) for p, _ in figures],
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=1)

    kc = sorted([c for c in cases if c["kind"] == "kc"], key=lambda c: c["KC"])
    beta = sorted([c for c in cases if c["kind"] == "beta"], key=lambda c: c["beta"])
    if kc:
        _write_csv(os.path.join(run_dir, "cd_vs_kc.csv"), kc,
                   ["id", "KC", "Re_h", "beta", "C_d", "C_d_reference", "C_m",
                    "r_squared", "phase_lag_deg", "phi", "phi_stations", "zeta_eff"])
    if beta:
        _write_csv(os.path.join(run_dir, "cd_vs_beta.csv"), beta,
                   ["id", "beta", "KC", "Re_h", "C_d", "C_d_reference", "C_m",
                    "r_squared", "phase_lag_deg", "phi", "phi_stations"])
    gci = [c for c in cases if c["kind"] == "gci"]
    if gci:
        _write_csv(os.path.join(run_dir, "gci.csv"), gci,
                   ["id", "ny", "hole_size", "beta", "KC", "Re_h", "C_d", "steps"])
    if any(c["kind"] == "nc" for c in cases):
        _write_csv(os.path.join(run_dir, "negative_controls.csv"),
                   [{"quantity": "dp_amp", "lbm": nc["dp_amp_measured"],
                     "analytic": nc["dp_amp_analytic"], "rel_error": nc["dp_rel_error"]}
                    for nc in [next(c["nc"] for c in cases if c["kind"] == "nc")]]
                   + [{"quantity": "tau_w_amp", "lbm": nc["tau_amp_measured"],
                       "analytic": nc["tau_amp_analytic"], "rel_error": nc["tau_rel_error"]}
                      for nc in [next(c["nc"] for c in cases if c["kind"] == "nc")]],
                   ["quantity", "lbm", "analytic", "rel_error"])

    md = _report_markdown(cases, gd, figures, run_dir)
    path = os.path.join(run_dir, "REPORT.md")
    with open(path, "w") as fh:
        fh.write(md)
    return os.path.join(run_dir, "metrics.json")