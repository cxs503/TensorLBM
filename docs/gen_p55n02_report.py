#!/usr/bin/env python3
"""Generate docs/P55-N02_lubrication_closure.md from the benchmark result.json.

Every number in the report is injected programmatically from
``benchmarks/verified/permeability_fibrous/result.json`` (zero hand
transcription).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "benchmarks" / "verified" / "permeability_fibrous" / "result.json"
OUT = REPO / "docs" / "P55-N02_lubrication_closure.md"


def fmt(x, nd=3):
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.{nd}f}"
    return str(x)


def main() -> None:
    r = json.loads(RES.read_text())
    L: list[str] = []
    a = L.append
    v = r["verdict"]
    g = r["gates"]
    pl = r["diagnostics"]["plateau_analysis"]

    a("# P55-N02 — Sub-lattice lubrication closure for near-jamming fibrous permeability")
    a("")
    a(f"*generated {r['created_utc']} from `benchmarks/verified/permeability_fibrous/result.json`*")
    a("")
    a("## 1. Case introduction")
    a("")
    a("Periodic two-dimensional D2Q9 Stokes flow (BGK, `tau` basis 1.0, base "
      f"`nu = {r['settings']['nu']:.6f}`) is driven through cylinder arrays by a "
      "fluid-masked body force; the Darcy (superficial) permeability is measured as")
    a("")
    a("    K_sim = (1 - phi_actual) * nu * <u_x>_fluid / a_body .")
    a("")
    a("Two reference channels are reported for every grid:")
    a("")
    a(f"- **channel A (nominal)**: {r['reference']['hex']} for hex, and "
      f"{r['reference']['sq']} for sq, both evaluated at the *nominal* phi;")
    a("- **channel B (realized geometry)**: the same closed forms evaluated at the "
      "solid fraction and radius of the *voxelized* geometry actually solved "
      "(it removes the O(dx^2) mask-area bias, which near jamming is amplified by "
      "d ln k / d phi = 15-30).")
    a("")
    a("| family | arrangement | phi | grids | judged levels |")
    a("|--------|-------------|-----|-------|---------------|")
    for case, d in r["cases"].items():
        a(f"| {case} | {d['arrangement']} | {d['phi']} | {d['grids']} | {d['judged_grids']} |")
    a("")
    a("## 2. Method")
    a("")
    a(f"- closure rule: `{r['closure']['rule']}`")
    a(f"- conductance: `{r['closure']['conductance']}`")
    a(f"- strict no-op: {r['closure']['no_op']}")
    a("- gap measurement: four axis-aligned rays per fluid node (exact analytic "
      "ray/cylinder intersections and, independently, the voxel mask); the local "
      "throat width is the narrowest sum of two opposed first-hit distances.")
    a("- solver: collide -> fluid-masked Guo body force -> bounce-back (post-collision) "
      "-> periodic stream (library kernel `tensorlbm.solver.stream`), float64, "
      "single-thread CPU torch.")
    a(f"- steady state: relative drift < {r['settings']['steady_tol']} over 1000 steps, "
      "then a 2000-step measurement window.")
    a("- criterion (item 4 of the node): the strict per-level error monotonicity is "
      "replaced by **a monotonically narrowing Roache GCI band with the finest band "
      "<= 3%**; an error *increase* between two levels is tolerated only if it is "
      "inside the GCI band of the finer level.")
    a("")
    a("## 3. Results")
    a("")
    a("### 3.1 Verdict")
    a("")
    a(f"**{v['overall']}** — {v['statement']}")
    a("")
    a("| case | grid | errB (%) |")
    a("|------|------|----------|")
    for lv in v["levels"]:
        a(f"| {lv['case']} | {lv['grid']} | {fmt(lv['err_pct'])} |")
    a("")
    a("### 3.2 Ladders (both channels)")
    a("")
    for case, d in r["cases"].items():
        a(f"**{case}** (phi = {d['phi']})")
        a("")
        a("| grid | delta_min/dx | closure | K_sim | K_ref(A) | errA (%) | K_ref(B) | errB (%) | n_corr | nu factor |")
        a("|------|--------------|---------|-------|----------|----------|----------|----------|--------|-----------|")
        for tag in ("ladder_on", "ladder_off"):
            for e in d[tag]:
                a(
                    f"| {e['grid']} | {fmt(e['delta_min_over_dx'], 2)} | {tag.split('_')[1]} | "
                    f"{e['k_sim']:.6g} | {e['k_ref']:.6g} | {fmt(e['err_pct'])} | "
                    f"{e['k_ref_B']:.6g} | {fmt(e['err_B_pct'])} | {e['n_corrected']} | "
                    f"{fmt(e['nu_factor_min'], 2)}..{fmt(e['nu_factor_max'], 2)} |"
                )
        a("")
    a("### 3.3 Gates")
    a("")
    a("| gate | pass | numbers |")
    a("|------|------|---------|")
    for name, gg in g.items():
        keys = [k for k in gg if k not in ("gate", "pass", "cases")]
        nums = "; ".join(f"{k}={fmt(gg[k])}" if not isinstance(gg[k], list) else f"{k}={gg[k]}" for k in keys)
        a(f"| {name} | {gg['pass']} | {nums} |")
    a("")
    a("### 3.4 Diagnostics: the error-vs-gap master curve")
    a("")
    a(f"Threshold delta_min/dx = {pl['threshold_dx']}; plateau proved: **{pl['proved']}**.")
    a("")
    a(pl["statement"])
    a("")
    a("| case | delta_min/dx | errB (%) | successive diffs (pp) | monotone | band (pp) |")
    a("|------|--------------|----------|-----------------------|----------|-----------|")
    for case, d in pl["per_case"].items():
        xs = ", ".join(fmt(x, 2) for x in d["delta_min_over_dx"])
        es = ", ".join(fmt(x) for x in d["err_pct"])
        ds = ", ".join(f"{x:+.3f}" for x in d["successive_diffs_pct"])
        a(f"| {case} | {xs} | {es} | {ds} | {d['monotone_decreasing']} | {fmt(d['band_pct'])} |")
    a("")
    a("### 3.5 GCI bands")
    a("")
    a("| case | closure | grids | p | GCI_fine (%) |")
    a("|------|---------|-------|---|--------------|")
    for case, d in r["diagnostics"]["gci"].items():
        for tag in ("triplets_on", "triplets_off"):
            for t in d.get(tag, []):
                a(f"| {case} | {tag.split('_')[1]} | {t['grids']} | {fmt(t['p'], 2)} | {fmt(t['gci_fine_pct'])} |")
    a("")
    a("### 3.6 Dilute negative control")
    a("")
    a("| case | phi | grid | delta_min/dx | K(on) | K(off) | correction (%) | pass |")
    a("|------|-----|------|--------------|-------|--------|----------------|------|")
    for name, d in r["diagnostics"]["dilute_negative_control"].items():
        a(
            f"| {name} | {d['phi']} | {d['grid']} | {fmt(d['delta_min_over_dx'], 2)} | "
            f"{d['k_on']:.6g} | {d['k_off']:.6g} | {fmt(d['correction_pct'], 4)} | {d['pass']} |"
        )
    a("")
    a("### 3.7 Figures")
    a("")
    a("- `benchmarks/verified/permeability_fibrous/figures/fig_master_curve.png` — error "
      "versus the grid-independent gap ratio, closure on/off.")
    a("- `benchmarks/verified/permeability_fibrous/figures/fig_ladders.png` — ladders with "
      "GCI bands.")
    a("- `benchmarks/verified/permeability_fibrous/figures/fig_dilute_control.png` — dilute "
      "negative control.")
    a("")
    a("## 4. Error judgement (per quantity, versus the reference)")
    a("")
    a("| quantity | reference | error | < 3% ? |")
    a("|----------|-----------|-------|--------|")
    for case, d in r["cases"].items():
        e = d["ladder_on"][-1] if d["ladder_on"] else None
        if e is None:
            continue
        ok = "yes" if e["err_B_pct"] < 3.0 else "no"
        a(
            f"| K ({case}, grid {e['grid']}, channel B) | closed form at realized geometry | "
            f"{fmt(e['err_B_pct'])}% | {ok} |"
        )
        a(
            f"| K ({case}, grid {e['grid']}, channel A) | closed form at nominal phi | "
            f"{fmt(e['err_pct'])}% | {'yes' if e['err_pct'] < 3.0 else 'no'} |"
        )
    a("")
    a("## 5. Caveats, limitations, what is NOT delivered")
    a("")
    a("See the mission report (platform) for the full honest-FAIL discussion; the "
      "machine-readable record is `result.json` -> `verdict`, `gates`, "
      "`diagnostics.plateau_analysis`.")
    a("")
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()