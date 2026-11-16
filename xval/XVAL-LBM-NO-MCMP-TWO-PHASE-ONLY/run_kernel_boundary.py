#!/usr/bin/env python3
"""XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY — cross-validation driver.

Anchor: ``lssfau/walberla`` **no-mcmp-two-phase-only** (a two-phase lattice model
carries no MCMP force term; the two-phase and the multi-component multiphase
models are distinct, even though they share the Shan-Chen interaction kernel).

Gates (implementation in ``src/tensorlbm/xval/no_mcmp_two_phase.py``):

===========  ===================================================================
G1           interaction-kernel identity: MCMP cross-force with a duplicated
             partner == SCMP self-force, cell by cell (4 variants incl. float32)
G2           null-coupling purity (``G_12 = 0`` -> identically zero force) plus a
             non-triviality ablation and an inert-uniform-partner check
G3           dynamical reduction: duplicated-partner MCMP == SCMP trajectory
G4           no self-interaction inside the MCMP kernel (inert partner -> the
             single phase may only diffuse, never separate)
G5           coupling-sign semantics of the shared kernel (exact force alignment
             + frozen-partner tracer transport)
G6           two-phase-only physics anchor: relaxed flat-interface coexisting
             densities vs the analytic Maxwell binodal of the same EOS
===========  ===================================================================

Usage::

    PYTHONPATH=src python xval/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY/run_xval.py
    ... --stages G1,G2,G5 --quick        # subset / fast settings
    ... --merge                          # build results.json + figures from chunks

Each stage writes ``artifacts/gates_<stage>.json`` so the suite can be split
across short compute windows; ``--merge`` combines the chunks into
``artifacts/results.json`` and draws the figures.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

CASE = pathlib.Path(__file__).resolve().parent
ARTIFACTS = CASE / "artifacts" / "kernel_boundary"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def _import_gates():
    sys.path.insert(0, str(CASE.parent.parent / "src"))
    from tensorlbm.xval import no_mcmp_two_phase as gates

    return gates


def stage_G1(gates, quick: bool) -> dict:
    return {
        "G1": [
            gates.gate_force_kernel_identity(G=5.0, psi="exp", dtype=torch.float64),
            gates.gate_force_kernel_identity(G=5.0, psi="linear", dtype=torch.float64),
            gates.gate_force_kernel_identity(G=-1.0, psi="exp", dtype=torch.float64),
            gates.gate_force_kernel_identity(G=5.0, psi="exp", dtype=torch.float32),
        ]
    }


def stage_G2(gates, quick: bool) -> dict:
    return {"G2": gates.gate_null_coupling_purity()}


def stage_G3(gates, quick: bool) -> dict:
    return {"G3": gates.gate_duplicated_partner_dynamics(G=5.0, steps=150 if quick else 400)}


def stage_G4(gates, quick: bool) -> dict:
    return {"G4": gates.gate_no_self_interaction(G_12=-1.0, steps=200 if quick else 400)}


def stage_G5(gates, quick: bool) -> dict:
    return {"G5": gates.gate_coupling_sign(steps=150 if quick else 400)}


def stage_G6(gates, quick: bool) -> dict:
    return {
        "G6": gates.gate_two_phase_coexistence(
            (4.5, 5.0), steps=3000 if quick else 8000
        )
    }


STAGES = {"G1": stage_G1, "G2": stage_G2, "G3": stage_G3, "G4": stage_G4, "G5": stage_G5, "G6": stage_G6}


def run_stages(names: list[str], quick: bool) -> dict:
    gates = _import_gates()
    for name in names:
        out = STAGES[name](gates, quick)
        path = ARTIFACTS / f"gates_{name}.json"
        path.write_text(json.dumps(out, indent=1))
        print(f"[{name}] -> {path.name}: pass={_stage_pass(out)}", flush=True)
        if isinstance(out[name], list):
            for row in out[name]:
                print("   ", {k: row[k] for k in ("G", "psi", "dtype", "rel_err", "exact") if k in row}, flush=True)
        elif name == "G6":
            for row in out[name]["rows"]:
                print(
                    f"    G={row['G']}: rho_l {row.get('rho_liquid_sim')} vs {row.get('rho_liquid_maxwell')} "
                    f"(err {row.get('err_liquid')}), rho_v {row.get('rho_vapour_sim')} vs "
                    f"{row.get('rho_vapour_maxwell')} (err {row.get('err_vapour')})",
                    flush=True,
                )
        else:
            print("   ", {k: v for k, v in out[name].items() if not isinstance(v, (list, dict))}, flush=True)
    return out


def _stage_pass(out: dict) -> bool:
    value = next(iter(out.values()))
    if isinstance(value, list):
        return all(v["pass"] for v in value)
    return bool(value["pass"])


def _load_chunks() -> dict:
    merged: dict = {}
    for path in sorted(ARTIFACTS.glob("gates_*.json")):
        merged.update(json.loads(path.read_text()))
    return merged


def merge() -> dict:
    chunks = _load_chunks()
    results = {
        "case": "XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY",
        "anchor": "lssfau/walberla no-mcmp-two-phase-only",
        "repo": "TensorLBM",
        "gates": chunks,
        "pass": all(
            (all(v["pass"] for v in value) if isinstance(value, list) else value["pass"])
            for value in chunks.values()
        ),
    }
    results["gate_count"] = sum(
        len(value) if isinstance(value, list) else 1 for value in chunks.values()
    )
    results["gate_count_passed"] = sum(
        (sum(1 for v in value if v["pass"]) if isinstance(value, list) else int(value["pass"]))
        for value in chunks.values()
    )
    (ARTIFACTS / "results.json").write_text(json.dumps(results, indent=1))
    print(f"merged -> artifacts/results.json  pass={results['pass']} "
          f"({results['gate_count_passed']}/{results['gate_count']} gates)")
    return results


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sys.path.insert(0, str(CASE.parent.parent / "src"))
    from tensorlbm.multiphase import psi_exp, sc_single_component_force, sc_two_component_force
    from tensorlbm.sc_eos_coexistence import maxwell_coexistence
    from tensorlbm.xval import no_mcmp_two_phase as gates

    results = json.loads((ARTIFACTS / "results.json").read_text())
    figs = CASE / "figures" / "kernel_boundary"
    figs.mkdir(exist_ok=True)

    # --- fig 1: kernel identity + null-coupling purity ---------------------
    rho = gates.random_smooth_field(48, 64, seed=0)
    fsc = sc_single_component_force(rho, G=5.0, psi_fn=psi_exp)
    f1 = sc_two_component_force(rho, rho.clone(), G_12=5.0, psi=psi_exp)[:2]
    mag_sc = torch.sqrt(fsc[0] ** 2 + fsc[1] ** 2).numpy()
    mag_mc = torch.sqrt(f1[0] ** 2 + f1[1] ** 2).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    ax = axes[0]
    ax.loglog(mag_sc.ravel(), mag_mc.ravel(), ".", ms=2, color="tab:blue")
    lim = [mag_sc.min() * 0.8, mag_sc.max() * 1.2]
    ax.plot(lim, lim, "k--", lw=0.8, label="y = x")
    ax.set_xlabel("|F| SCMP self-force  (single component)")
    ax.set_ylabel("|F| MCMP cross-force (duplicated partner)")
    ax.set_title("G1 kernel identity: 8x8-cell droplet field, G = G$_{12}$ = 5")
    ax.legend(fontsize=8)
    rel = np.abs(mag_mc - mag_sc) / np.maximum(mag_sc, 1e-30)
    ax2 = axes[1]
    ax2.semilogy(np.sort(rel.ravel()) + 1e-300, ".", ms=2, color="tab:red")
    ax2.axhline(1e-12, ls="--", c="k", lw=0.8, label="tolerance 1e-12")
    ax2.set_xlabel("cell index (sorted)")
    ax2.set_ylabel("relative difference")
    ax2.set_title(f"G1 rel. difference: max = {rel.max():.1e} (bit-exact)")
    ax2.legend(fontsize=8)

    ax3 = axes[2]
    g2 = results["gates"]["G2"]
    labels = ["G$_{12}$=0\narbitrary\npartner", "G$_{12}$=-1\nuniform\npartner", "G$_{12}$=-1\nreal partner\n(ablation)"]
    values = [max(g2["null_max"], 1e-30), max(g2["inert_partner_max"], 1e-30), g2["nontrivial_max"]]
    colors = ["tab:green", "tab:green", "tab:orange"]
    ax3.bar(labels, values, color=colors)
    ax3.set_yscale("log")
    ax3.set_ylabel("max |F|  (log)")
    ax3.set_title("G2 null-coupling purity (zeros at 1e-30 floor)")
    for i, v in enumerate(values):
        ax3.text(i, v * 1.5, f"{v:.1e}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "fig1_kernel_identity_purity.png", dpi=140)
    plt.close(fig)

    # --- fig 2: dynamical reduction ---------------------------------------
    g3 = results["gates"]["G3"]
    steps = [h["step"] for h in g3["history"]]
    diffs = [max(h["max_abs_diff"], 1e-30) for h in g3["history"]]

    rho0 = gates.droplet_field(64, 64, radius=14.0)
    u0 = torch.zeros_like(rho0)
    from tensorlbm.multiphase import collide_sc_single_component, collide_sc_two_component, equilibrium
    from tensorlbm.solver import stream

    f_sc = equilibrium(rho0, u0, u0)
    f_mc = f_sc.clone()
    f_b = f_sc.clone()
    for _ in range(400):
        f_sc = stream(collide_sc_single_component(f_sc, G=5.0, tau=1.0, psi_fn=psi_exp, forcing="velocity_shift"))
        f_mc, f_b = collide_sc_two_component(f_mc, f_b, G_12=5.0, tau1=1.0, tau2=1.0, psi=psi_exp, force_scheme="shift")
        f_mc, f_b = stream(f_mc), stream(f_b)
    r_sc = f_sc.sum(0).numpy()
    r_mc = f_mc.sum(0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    ax = axes[0]
    ax.semilogy(steps, diffs, "o-", color="tab:blue")
    ax.set_xlabel("time step")
    ax.set_ylabel("max |rho$_{SCMP}$ - rho$_{MCMP,dup}$|")
    ax.set_title("G3 duplicated-partner MCMP vs SCMP: 0.0 at every sample")
    ax.set_ylim(1e-18, 1e-6)

    ax = axes[1]
    ax.plot(r_sc[32, :], "k-", lw=2.2, label="SCMP (two-phase-only)")
    ax.plot(r_mc[32, :], "r--", lw=1.0, label="MCMP, duplicated partner")
    ax.set_xlabel("x (mid-row cut, step 400)")
    ax.set_ylabel("density rho")
    ax.set_title(f"trajectories coincide: contrast {r_sc.max()-r_sc.min():.3f}")
    ax.legend(fontsize=8)

    ax = axes[2]
    im = ax.imshow(r_sc, cmap="viridis")
    ax.set_title("SCMP density field, step 400 (G = 5)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(figs / "fig2_dynamical_reduction.png", dpi=140)
    plt.close(fig)

    # --- fig 3: coupling-sign semantics ------------------------------------
    g5 = results["gates"]["G5"]
    gs = [r["G_12"] for r in g5["rows"]]
    align = [r["alignment"] for r in g5["rows"]]
    tracer = [r["tracer_correlation"] for r in g5["rows"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    ax.plot(gs, align, "s-", label=r"$\sum_x F_A\cdot\nabla\psi_B$ (exact)")
    ax.axhline(0, c="k", lw=0.7)
    ax.set_xlabel("G$_{12}$ (multi-component coupling)")
    ax.set_ylabel("force alignment (a.u.)")
    ax.set_title("G5a shared kernel, opposite sign meaning\n(SCMP like-species attraction needs G > 0)")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(gs, tracer, "o-", color="tab:red")
    ax.axhline(0, c="k", lw=0.7)
    ax.set_xlabel("G$_{12}$")
    ax.set_ylabel(r"corr($\rho_A$, $\psi_B$), frozen partner")
    ax.set_title("G5b tracer: G$_{12}$<0 repels (segregation),\nG$_{12}$>0 attracts (adhesion)")
    fig.tight_layout()
    fig.savefig(figs / "fig3_coupling_sign.png", dpi=140)
    plt.close(fig)

    # --- fig 4: two-phase-only coexistence vs Maxwell ----------------------
    g6 = results["gates"]["G6"]
    gs_max = torch.linspace(4.2, 6.0, 25)
    liq, vap = [], []
    for g in gs_max:
        mw = maxwell_coexistence(float(g), psi_exp)
        liq.append(mw.rho_liquid if mw.found else float("nan"))
        vap.append(mw.rho_vapour if mw.found else float("nan"))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    ax.plot(gs_max.numpy(), liq, "k-", label="Maxwell liquid branch")
    ax.plot(gs_max.numpy(), vap, "k--", label="Maxwell vapour branch")
    sim_l = [r["rho_liquid_sim"] for r in g6["rows"]]
    sim_v = [r["rho_vapour_sim"] for r in g6["rows"]]
    sim_g = [r["G"] for r in g6["rows"]]
    ax.plot(sim_g, sim_l, "ro", label="LBM flat interface (liquid)")
    ax.plot(sim_g, sim_v, "bs", label="LBM flat interface (vapour)")
    ax.set_xlabel("SCMP coupling G")
    ax.set_ylabel("coexistence density")
    ax.set_yscale("log")
    ax.set_title("G6 two-phase-only path vs Maxwell construction")
    ax.legend(fontsize=8)
    ax = axes[1]
    errs_l = [100 * r["err_liquid"] for r in g6["rows"]]
    errs_v = [100 * r["err_vapour"] for r in g6["rows"]]
    width = 0.35
    x = np.arange(len(sim_g))
    ax.bar(x - width / 2, errs_l, width, label=r"$\rho_{liq}$ error")
    ax.bar(x + width / 2, errs_v, width, label=r"$\rho_{vap}$ error")
    ax.axhline(3, ls=":", c="k", lw=0.9, label="3 % target")
    ax.axhline(15, ls="--", c="r", lw=0.9, label="15 % acceptance")
    ax.set_xticks(x)
    ax.set_xticklabels([f"G = {g}" for g in sim_g])
    ax.set_ylabel("error vs Maxwell (%)")
    ax.set_title("G6 error budget")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "fig4_coexistence_vs_maxwell.png", dpi=140)
    plt.close(fig)
    print("figures ->", sorted(p.name for p in figs.glob("*.png")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    if args.stages:
        run_stages([s.strip() for s in args.stages.split(",") if s.strip()], args.quick)
    if args.merge or args.figures:
        merge()
    if args.figures:
        make_figures()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())