"""DE-X-SURV1-MMREVIEW: figure generation for the survey card / platform report.

Produces (into artifacts/de_x_surv1_mmreview/):

* fig1_capability_matrix.png   — collision family x lattice capability contract
                                 (evidence matrix, from the shipped contract)
* fig2_density_ratio.png       — reachable density ratio per multiphase family
                                 vs the case requirement (852:1, protocol.json)
* fig3_parasitic_current.png   — measured parasitic-current level per family vs
                                 the phase-field gate (1e-3) and the CAC notes
                                 baseline (2.79e-4 / 1.0-1.1e-3)
* fig4_turbulence_contract.png — turbulence family x lattice x collision
                                 contract status (all fail-closed)

All inputs are produced by this task's own probe/contract dumps; nothing is
hand-typed except the case constants that are quoted from
benchmarks/water_entry/single_drop/protocol.json and
benchmarks/pending/phasefield_water_entry/NOTES.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "de_x_surv1_mmreview"

CASE_DENSITY_RATIO = 997.0 / 1.17  # protocol.json: water/air density -> 852:1
PHASEFIELD_GATE = 1.0e-3  # ch_benchmarks.py:249 spurious_velocity gate
CAC_BASELINE = 2.79e-4  # pending/phasefield_water_entry/NOTES.md P0
CAC_CONTACTLINE = 1.06e-3  # NOTES.md P0b (1.00e-3 .. 1.06e-3)
VOF_MASS_DRIFT = 4.8e-6  # campaign anchor (mass drift, not a spurious current)


def _load(name: str) -> dict:
    p = ART / name
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def fig1(caps: dict) -> str:
    fam_order = ["BGK", "TRT", "RLBM", "MRT", "CM", "CUMULANT", "KBC"]
    lattices = ["D3Q19", "D3Q27"]
    grid = np.zeros((len(fam_order), len(lattices)))
    for i, fam in enumerate(fam_order):
        for j, lat in enumerate(lattices):
            cap = caps.get("collision", {}).get(lat, {}).get(fam)
            grid[i, j] = 1.0 if (cap and cap.get("available")) else 0.0
    fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=140)
    ax.imshow(grid, cmap="Greens", vmin=0, vmax=1.6, aspect="auto")
    ax.set_xticks(range(len(lattices)), lattices)
    ax.set_yticks(range(len(fam_order)), fam_order)
    for i in range(len(fam_order)):
        for j in range(len(lattices)):
            ax.text(j, i, "AVAILABLE" if grid[i, j] else "WITHHELD",
                    ha="center", va="center", fontsize=8,
                    color="black" if grid[i, j] else "#8b0000")
    ax.set_title("Collision capability contract\n"
                 "(tensorlbm.advanced_collision_contract:53, executed)")
    ax.set_xlabel("lattice")
    fig.tight_layout()
    out = ART / "fig1_capability_matrix.png"
    fig.savefig(out)
    plt.close(fig)
    return out.name


def fig2(probe: dict) -> str:
    # (label, reachable ratio or nan when the family cannot carry the contrast
    #  in its momentum equation, colour code, annotation)
    rows = []
    for key, label in (
        ("mcmp_coex_G-0.9", "M6 MCMP (SC) G12=-0.9"),
        ("mcmp_coex_G-1.5", "M6 MCMP (SC) G12=-1.5"),
        ("mcmp_coex_G-2.5", "M6 MCMP (SC) G12=-2.5"),
    ):
        r = probe.get(key, {})
        ratio = r.get("density_ratio")
        if isinstance(ratio, (int, float)) and ratio == ratio:
            rows.append((label, float(ratio), f"measured (probe)"))
    pseudo = probe.get("pseudo_coexistence_ratio", {})
    for k, v in pseudo.items():
        if isinstance(v, dict) and isinstance(v.get("CS_ratio"), (int, float)):
            rows.append((f"M6 PR/CS EOS {k}", float(v["CS_ratio"]), "analytic Maxwell"))

    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=140)
    labels = [r[0] for r in rows] + ["CASE REQUIREMENT\n(protocol.json 997/1.17)"]
    vals = [r[1] for r in rows] + [CASE_DENSITY_RATIO]
    colors = ["#2b6cb0"] * len(rows) + ["#c53030"]
    bars = ax.bar(range(len(vals)), vals, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("density ratio rho_liquid/rho_gas  (log)")
    ax.set_title("Reachable density ratio vs the water-entry case requirement")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.1f}",
                ha="center", fontsize=8)
    ax.grid(axis="y", ls=":", alpha=0.6)
    fig.tight_layout()
    out = ART / "fig2_density_ratio.png"
    fig.savefig(out)
    plt.close(fig)
    return out.name


def fig3(probe: dict) -> str:
    labels, vals, kinds, colors = [], [], [], []

    def add(label, value, kind, color):
        if isinstance(value, (int, float)) and value == value:
            labels.append(label)
            vals.append(float(value))
            kinds.append(kind)
            colors.append(color)

    for key, tag in (("ch_planar_lee_lin", "lee_lin"), ("ch_planar_swift", "swift")):
        r = probe.get(key, {})
        add(f"M1/M4 CH planar ({tag})", r.get("max_physical_speed"), "measured", "#2f855a")
    for key, tag in (("mcmp_coex_G-0.9", "G=-0.9"), ("mcmp_coex_G-1.5", "G=-1.5"),
                     ("mcmp_coex_G-2.5", "G=-2.5")):
        r = probe.get(key, {})
        add(f"M6 MCMP coexistence ({tag})", r.get("u_max"), "measured", "#2b6cb0")
    spur = probe.get("mcmp_spurious_G-1.5", {})
    rows = spur.get("rows", []) if isinstance(spur, dict) else []
    if rows:
        add("M6 MCMP droplet u_max", rows[0].get("u_max"), "measured", "#2b6cb0")
    add("M1 CH/CAC baseline (NOTES P0)", CAC_BASELINE, "repo note", "#805ad5")
    add("M1 CH/CAC contact line (NOTES P0b)", CAC_CONTACTLINE, "repo note", "#805ad5")

    if not vals:
        return ""
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=140)
    x = np.arange(len(vals))
    ax.bar(x, vals, color=colors)
    ax.axhline(PHASEFIELD_GATE, ls="--", color="#c53030",
               label=f"phase-field gate {PHASEFIELD_GATE:g} (ch_benchmarks.py:249)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("max |u|  (lattice units, log)")
    ax.set_title("Parasitic-current level per multiphase family\n"
                 "(green/blue = measured by this task's probe; purple = repo note)")
    for xi, v in zip(x, vals):
        ax.text(xi, v * 1.2, f"{v:.2e}", ha="center", fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(axis="y", ls=":", alpha=0.6)
    fig.tight_layout()
    out = ART / "fig3_parasitic_current.png"
    fig.savefig(out)
    plt.close(fig)
    return out.name


def fig4(caps: dict) -> str:
    turb = caps.get("turbulence", {})
    fams = ["smagorinsky", "dynamic_smagorinsky", "wale", "vreman", "rans_ke", "ddes"]
    fams = [f for f in fams if f in turb]
    lats = ["D2Q9", "D3Q19", "D3Q27"]
    cols = ["BGK", "MRT", "CG"]
    grid = np.zeros((len(fams), len(lats) * len(cols)))
    for i, fam in enumerate(fams):
        for j, lat in enumerate(lats):
            for k, col in enumerate(cols):
                c = turb.get(fam, {}).get(lat, {}).get(col, {})
                impl = c.get("impl") == "IMPLEMENTED"
                verified = c.get("ver") in ("CONTRACT_TESTED", "BENCHMARK_ONLY")
                grid[i, j * len(cols) + k] = 1.0 if impl else 0.0
                if impl and not verified:
                    grid[i, j * len(cols) + k] = 0.5
    fig, ax = plt.subplots(figsize=(8.4, 3.8), dpi=140)
    ax.imshow(grid, cmap="YlGn", vmin=0, vmax=1.6, aspect="auto")
    ticks, labels = [], []
    for j, lat in enumerate(lats):
        for k, col in enumerate(cols):
            ticks.append(j * len(cols) + k)
            labels.append(f"{lat}\n{col}")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticks(range(len(fams)), fams, fontsize=8)
    for i in range(len(fams)):
        for j in range(len(ticks)):
            v = grid[i, j]
            txt = {1.0: "contract", 0.5: "impl only", 0.0: "none"}[v]
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5)
    ax.set_title("Turbulence contract: every combination is fail-closed\n"
                 "(status = WITHHELD_NO_PHYSICS_VALIDATION / _NO_CONTRACT_TESTS)")
    fig.tight_layout()
    out = ART / "fig4_turbulence_contract.png"
    fig.savefig(out)
    plt.close(fig)
    return out.name


def main() -> None:
    caps = _load("capability_matrix_dump.json")
    probe = _load("probe_results.json")
    made = {
        "fig1": fig1(caps),
        "fig2": fig2(probe),
        "fig3": fig3(probe),
        "fig4": fig4(caps),
    }
    (ART / "figures.json").write_text(json.dumps(made, indent=1))
    print(json.dumps(made, indent=1))


if __name__ == "__main__":
    main()
