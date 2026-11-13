#!/usr/bin/env python
"""Figures + machine-written tables for XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL.

Reads only artefacts (``artifacts/results.json``, ``artifacts/oss_facts.json``,
``artifacts/snapshot.npz``) and writes
  * ``artifacts/figures/*.png``
  * ``artifacts/NUMBERS.md``  — every table used by the report, generated here
so that no number in the report is transcribed by hand.
"""

from __future__ import annotations

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

plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight"})


def load(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "✔" if x else "✘"
    if isinstance(x, (int, np.integer)):
        return f"{x:d}"
    if isinstance(x, float):
        if x != x:
            return "NaN"
        if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e5):
            return f"{x:.2e}"
        return f"{x:.{nd}f}"
    return str(x)


# ---------------------------------------------------------------------------
def fig_psi(res: dict, facts: dict) -> str:
    smp = res["e1"]["sample"]
    rho = np.array(smp["rho"])
    lbm = np.array(smp["psi_lbm"])
    oss = np.array(smp["psi_oss"])
    ok = np.isfinite(oss) & np.isfinite(lbm) & (oss > 0)

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax[0].plot(rho[ok], lbm[ok], "C0-", lw=2, label="TensorLBM $\\psi$ (PR, G=+1)")
    ax[0].plot(rho[ok], oss[ok], "C3--", lw=1.4, label="OSS kernel (c0=cs$^2$, G=−1)")
    ax[0].set(xlabel="density $\\rho$", ylabel="pseudopotential $\\psi$",
              title=f"P-R effective mass, $T_r$={res['e1']['rows'][2]['T_r']}")
    ax[0].legend()

    rel = np.abs(lbm[ok] - oss[ok]) / oss[ok]
    ax[1].semilogy(rho[ok], np.maximum(rel, 1e-18), "C2-", lw=1.6,
                   label="stencil mapping $c_0=c_s^2$")
    c0three = res["e1"]["rows"][2]["c0_three_ratio_mean"]
    ax[1].axhline(abs(c0three - 1) / c0three, color="C1", ls="--", lw=1.2,
                  label=f"macro-pressure map $c_0$=3 (factor {c0three:.3g})")
    ax[1].axhline(0.15, color="k", ls=":", lw=1, label="owner 15 % band")
    ax[1].axhline(1e-6, color="grey", ls=":", lw=1, label="prereg 1e-6 gate")
    ax[1].set(xlabel="density $\\rho$", ylabel="relative deviation", ylim=(1e-18, 10),
              title="conformance residual (same EOS parameters)")
    ax[1].legend(fontsize=7)
    p = FIG / "fig1_psi_conformance.png"
    fig.savefig(p)
    plt.close(fig)
    return p.name


def fig_planar(res: dict) -> str | None:
    cells = [c for c in res.get("cells", []) if c["geometry"] == "planar"
             and c["reference_tier"] == "ratio10" and c.get("y_profile")
             and "threads2" not in c["tag"]]
    if not cells:
        return None
    cells = sorted(cells, key=lambda c: c["spec"]["L"])
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for c in cells:
        y = np.array([r["y"] for r in c["y_profile"]])
        rho = np.array([r["rho_mean"] for r in c["y_profile"]])
        ax[0].plot(y, rho, lw=1.4, label=f"L={c['spec']['L']} (ratio={c['ratio']:.3g})")
    mw = res["maxwell"]["ratio10"]
    ax[0].axhline(mw["rho_liquid"], color="k", ls="--", lw=1,
                  label=f"Maxwell $\\rho_l$={mw['rho_liquid']:.4g}")
    ax[0].axhline(mw["rho_vapour"], color="k", ls=":", lw=1,
                  label=f"Maxwell $\\rho_v$={mw['rho_vapour']:.4g}")
    ax[0].set(xlabel="y (lattice)", ylabel="row-mean $\\rho$",
              title="planar slab: measured plateaus vs Maxwell")
    ax[0].legend(fontsize=7)

    c = cells[-1]
    y = np.array([r["y"] for r in c["y_profile"]])
    rho = np.array([r["rho_mean"] for r in c["y_profile"]])
    ax[1].semilogy(y, np.maximum(rho, 1e-6), lw=1.6, label=f"L={c['spec']['L']} profile")
    ax[1].axhline(mw["rho_liquid"], color="k", ls="--", lw=1, label="Maxwell $\\rho_l$")
    ax[1].axhline(mw["rho_vapour"], color="k", ls=":", lw=1, label="Maxwell $\\rho_v$")
    ax[1].set(xlabel="y (lattice)", ylabel="row-mean $\\rho$ (log)",
              title="vapour plateau on a log axis (error amplification)")
    ax[1].legend(fontsize=7)
    p = FIG / "fig2_planar_profiles.png"
    fig.savefig(p)
    plt.close(fig)
    return p.name


def fig_convergence(res: dict) -> str | None:
    grid = res.get("verdict", {}).get("grid_ratio10") or []
    if not grid:
        return None
    grid = [g for g in grid if g["ratio"] is not None]
    grid = sorted(grid, key=lambda g: g["L"])
    L = np.array([g["L"] for g in grid], float)
    ratio = np.array([g["ratio"] for g in grid], float)
    ref = np.array([g["ratio_maxwell"] for g in grid], float)

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
    ax[0].fill_between([L.min() * 0.9, L.max() * 1.1], ref[0] * 0.85, ref[0] * 1.15,
                       color="0.85", label="Maxwell $\\pm$15 %")
    ax[0].plot(L, ratio, "o-", color="C0", label="LBM planar coexistence ratio")
    ax[0].axhline(ref[0], color="k", ls="--", lw=1, label=f"Maxwell {ref[0]:.4g}")
    ax[0].set_xscale("log")
    ax[0].set(xlabel="grid L", ylabel="$\\rho_l/\\rho_v$",
              title="grid convergence at $T_r$=" f"{res['maxwell']['ratio10']['T_r']:.4f}")
    ax[0].legend(fontsize=7)

    sens = [c for c in res.get("cells", []) if c.get("errors")
            and c["spec"]["L"] == 64 and "threads2" not in c["tag"]]
    if sens:
        tr = np.array([c["T_r"] for c in sens], float)
        mr = np.array([c["ratio_maxwell"] for c in sens], float)
        lr = np.array([c["ratio"] for c in sens], float)
        o = np.argsort(tr)
        ax[1].semilogy(tr[o], mr[o], "k--s", ms=5, lw=1, label="Maxwell")
        ax[1].semilogy(tr[o], lr[o], "C0-o", ms=5, lw=1.2, label="LBM (L=64)")
        ax[1].set(xlabel="$T_r$", ylabel="$\\rho_l/\\rho_v$",
                  title="temperature sensitivity (L=64 planar)")
        ax[1].legend(fontsize=7)
    p = FIG / "fig3_convergence.png"
    fig.savefig(p)
    plt.close(fig)
    return p.name


def fig_droplet(res: dict) -> str | None:
    snap_path = ART / "snapshot.npz"
    if not snap_path.exists():
        return None
    d = np.load(snap_path, allow_pickle=True)
    rho, umag, r = d["rho"], d["umag"], d["r"]
    meta = json.loads(str(d["meta"]))
    cs = np.sqrt(1 / 3)

    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5))
    im0 = ax[0].imshow(rho, origin="lower", cmap="viridis")
    ax[0].set(title=f"$\\rho$ field (L={meta['L']}, $T_r$={meta['T_r']:.4f}, "
                    f"{meta['steps']} steps)", xticks=[], yticks=[])
    fig.colorbar(im0, ax=ax[0], fraction=0.046)

    im1 = ax[1].imshow(np.log10(np.maximum(umag / cs, 1e-12)), origin="lower",
                       cmap="magma")
    ax[1].set(title=f"spurious current $\\log_{{10}} Ma$ (max={meta['ma_max']:.3g})",
              xticks=[], yticks=[])
    fig.colorbar(im1, ax=ax[1], fraction=0.046)

    drops = [c for c in res.get("cells", []) if c["geometry"] == "droplet"
             and c.get("radial_profile")]
    for c in drops:
        prof = c["radial_profile"]
        ax[2].plot([b["r_mid"] for b in prof], [b["rho_mean"] for b in prof], "o-",
                   ms=2.5, lw=1.2, label=f"R={c['spec']['R']:.0f} (T={c['T_r']:.4f})")
    if not drops:
        L, R0 = meta["L"], meta["R"]
        rr = r.reshape(-1); rr = rr[rr <= 0.5 * L]
        nb = 40
        edges = np.linspace(0, 0.5 * L, nb + 1)
        idx = np.clip(np.digitize(r.ravel(), edges) - 1, 0, nb - 1)
        flat = rho.ravel()
        x, y = [], []
        for k in range(nb):
            m = idx == k
            if m.any():
                x.append(0.5 * (edges[k] + edges[k + 1])); y.append(flat[m].mean())
        ax[2].plot(x, y, "o-", ms=2.5, lw=1.2,
                   label=f"R={R0:.0f} ({meta['steps']} steps)")
    ax[2].axhline(meta["rho_l_maxwell"], color="k", ls="--", lw=1, label="Maxwell $\\rho_l$")
    ax[2].axhline(meta["rho_v_maxwell"], color="k", ls=":", lw=1, label="Maxwell $\\rho_v$")
    ax[2].set(xlabel="r (lattice)", ylabel="annulus mean $\\rho$",
              title="droplet radial profile")
    ax[2].legend(fontsize=7)
    p = FIG / "fig4_droplet_fields.png"
    fig.savefig(p)
    plt.close(fig)
    return p.name


# ---------------------------------------------------------------------------
def numbers_md(res: dict, facts: dict, figs: dict) -> str:
    L = []
    A = L.append
    e1 = res["e1"]
    A("### N1 — E1 formulation conformance (ψ from OSS source text vs TensorLBM)\n")
    A("| $T_r$ | T | compared ρ-range | n | max rel. dev (c₀=cs²) | c₀=3 map: ψ ratio | "
      "pass 1e-6 |")
    A("|---|---|---|---|---|---|---|")
    for r in e1["rows"]:
        A(f"| {fmt(r['T_r'])} | {fmt(r['T'])} | {r['rho_min']:.3g}–{r['rho_max']:.3g} | "
          f"{r['n_compared']} | {fmt(r['max_rel_dev_stencil_c0_third'])} | "
          f"{fmt(r['c0_three_ratio_mean'])} ± {fmt(r['c0_three_ratio_std'])} | "
          f"{fmt(r['max_rel_dev_stencil_c0_third'] <= 1e-6)} |")
    A(f"\nWorst deviation over all temperatures: **{fmt(e1['worst_rel_dev'])}** "
      f"(float64 eps = {fmt(e1['eps_machine'])}); prereg gate 1e-6 → "
      f"**{'PASS' if e1['verdict_pass'] else 'FAIL'}**.\n")

    A("### N2 — OSS snapshot facts (parsed, zero hand-copy)\n")
    pr = facts["pr_kernel"]
    A("| item | value |")
    A("|---|---|")
    A(f"| OSS HEAD | `{facts['git']['head'][:12]}` ({facts['git']['log']}) |")
    A(f"| P-R kernel | `{pr['kernel']}` in `{pr['file']}` |")
    A(f"| p_EOS (verbatim, de-continued) | `{pr['p_EOS_src_verbatim'][:120]}…` |")
    A(f"| ψ (verbatim) | `{pr['psi_src_verbatim']}` |")
    A(f"| call sites | `{pr['callsite_count']}` |")
    A(f"| wiring verdict | {pr['wire_up']['verdict']} |")
    A(f"| active potential | `{facts['active_potential']['rhs']}` "
      f"(potentialType = {facts['active_potential']['ini_potentialType']}) |")
    A(f"| OSS macro-pressure term | `{facts['active_potential']['macro_pressure_rhs'][-1][-80:]}` |")
    A("")
    A("### N3 — interaction-stencil isotropy moments (weights eval-ed from OSS literals)\n")
    A("| stencil | points | Σw | Σw·c_x² | A4 | A6 | A8 |")
    A("|---|---|---|---|---|---|---|")
    for k, v in facts["stencil"]["moments"].items():
        A(f"| {k} | {v['n_points']} | {fmt(v['sum_w'])} | {fmt(v['sum_w_cx2'])} | "
          f"{fmt(v['A4'])} | {fmt(v['A6'])} | {fmt(v['A8'])} |")
    A("\n(isotropic value of A4/A6/A8 is 1; both stencils are 4th-order isotropic and "
      "identical beyond it, i.e. the OSS `iso4` shell buys nothing over D2Q9.)\n")

    A("### N4 — E2 thermodynamic accuracy vs analytic PR-Maxwell (production cells)\n")
    A("| run | geometry | L | steps | stable | ρ_l | ρ_v | ratio | ratio err | ρ_l err | "
      "ρ_v err | worst | <15 % |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in res.get("cells", []):
        e = c.get("errors") or {}
        A(f"| {c['tag']} | {c['geometry']} | {c['spec']['L']} | {c['steps_run']} | "
          f"{fmt(c['stable'])} | {fmt(c.get('rho_l'))} | {fmt(c.get('rho_v'))} | "
          f"{fmt(c.get('ratio'))} | {fmt(e.get('ratio_err'))} | {fmt(e.get('rho_l_err'))} | "
          f"{fmt(e.get('rho_v_err'))} | {fmt(e.get('worst'))} | {fmt(e.get('pass_15pct'))} |")
    A("")
    A("### N5 — E2 reference values (PR equal-area Maxwell construction)\n")
    A("| tier | T_r | T | ρ_l | ρ_v | ratio | p_sat | ρ_cross | |area residual| |")
    A("|---|---|---|---|---|---|---|---|---|")
    for k, m in res["maxwell"].items():
        A(f"| {k} | {fmt(m['T_r'])} | {fmt(m['T'])} | {fmt(m['rho_liquid'])} | "
          f"{fmt(m['rho_vapour'])} | {fmt(m['ratio'])} | {fmt(m['p_sat'])} | "
          f"{fmt(m.get('rho_cross'))} | {fmt(m.get('area_residual'))} |")
    A("")
    A("### N6 — E2b pressure read-out from the measured plateaus\n")
    A("| run | p(ρ_l) | p(ρ_v) | p_sat (Maxwell) | p_l err | p_v err | plate mismatch |")
    A("|---|---|---|---|---|---|---|")
    for c in res.get("cells", []):
        ro = c.get("eos_readout")
        if ro:
            A(f"| {c['tag']} | {fmt(ro['p_l'])} | {fmt(ro['p_v'])} | {fmt(ro['p_sat_ref'])} | "
              f"{fmt(ro['p_l_err'])} | {fmt(ro['p_v_err'])} | {fmt(ro['p_plates_mismatch'])} |")
    A("")
    lap = res.get("laplace") or {}
    if lap.get("points"):
        A("### N7 — E3 Laplace / spurious-current diagnostics (droplets)\n")
        A("| run | R | Δp | Δp·R | r_eff | Ma_max |")
        A("|---|---|---|---|---|---|")
        for p in lap["points"]:
            A(f"| {p['tag']} | {fmt(p['R'])} | {fmt(p['dp'])} | {fmt(p['dp_times_R'])} | "
              f"{fmt(p.get('r_eff'))} | {fmt(p.get('ma_max'))} |")
        if lap.get("sigma_through_origin") is not None:
            A(f"\nσ (through-origin fit) = {fmt(lap['sigma_through_origin'])}; "
              f"max |Δp·R − σ|/σ = {fmt(lap['laplace_residual'])}.\n")
    A("### N8 — verdict under the preregistered rules\n")
    v = res.get("verdict", {})
    for k, val in v.items():
        if k != "grid_ratio10":
            A(f"- **{k}** = {val}")
    A("")
    A("### N9 — figures\n")
    for k, val in figs.items():
        A(f"- `{val}` — {k}")
    return "\n".join(L) + "\n"


def main() -> None:
    res = load(ART / "results.json", None)
    facts = load(ART / "oss_facts.json", None)
    if res is None or facts is None:
        raise SystemExit("missing artifacts/results.json or oss_facts.json")
    figs = {}
    for name, fn in [("fig1 ψ conformance", fig_psi), ("fig2 planar profiles", fig_planar),
                     ("fig3 convergence & sensitivity", fig_convergence),
                     ("fig4 droplet fields", fig_droplet)]:
        try:
            out = fn(res, facts) if name.startswith("fig1") else fn(res)
        except TypeError:
            out = fn(res)
        if out:
            figs[name] = out
            print("wrote", FIG / out)
    (ART / "NUMBERS.md").write_text(numbers_md(res, facts, figs))
    print("wrote", ART / "NUMBERS.md")


if __name__ == "__main__":
    main()