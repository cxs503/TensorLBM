"""CASE-RT-INSTABILITY-ATWOOD -- similarity-preserving ladder of the *terminal*
(nonlinear) front velocity.

Why this driver exists
----------------------
``run_late.py`` closed failure mode F4 of ``REPORT.md`` and gated the terminal
front velocity on the reference point (``Fr_meas = 0.5956`` vs
``Fr_LZ = 1/sqrt(pi) = 0.5642``, ``+5.56 %``, gate ``+-15 %``).  Two claims of
that increment were **not** verified:

1. **resolution independence.**  The only other point of that suite
   (``late_l096_g3e-4``) raised ``g`` together with ``lam`` (constant
   ``A g / lam``, the *linear*-phase ladder), so the lattice Mach number grew by
   ``sqrt(1.5)`` and the run tripped the Mach guard at ``Fr = 0.7833``
   (``+38.8 %``, marked INCOMPLETE).  The surviving PASS therefore rests on a
   single resolution whose nearest neighbour differs by 31 %.
2. **compressibility.**  The reference point runs at ``Ma_peak = 0.548``.  A
   lattice-BGK run at that Mach number carries an ``O(Ma^2)`` compressibility
   error of the same order as the 5.6 % agreement being claimed, so the agreement
   could be a Ma artefact rather than the potential-flow limit.

This driver re-runs the terminal-velocity measurement on a **similarity-
preserving** ladder, i.e. with the dimensionless groups of the problem held fixed
while the grid is refined, and adds a Mach arm at fixed resolution.

Groups held fixed (all arms)
----------------------------
============================  ==========================================
Atwood number                 ``A = (rho1-rho2)/(rho1+rho2) = 0.5``
initial amplitude             ``k a0 = 2 pi eta0_frac = 0.0628``
lattice Mach number           ``Ma ~ sqrt(A g lam)``      -> ``g ~ 1/lam``
viscous damping group         ``zeta = nu k^2 / sqrt(A g k)`` -> ``nu ~ lam``
box depth                     ``layers = 8`` wavelengths (4 per side)
interface model               Allen-Cahn, ``ac_sigma``/``ac_mobility`` as in
                              the case definition
============================  ==========================================

The one group that is *not* held fixed is the diffuse-interface thickness in
wavelengths, ``k xi = 2 pi ac_width / lam``: it halves per refinement because
``ac_width`` is fixed in cells.  This is the same choice as the ``kdelta`` arm of
``verify/``.  ``k xi`` therefore drifts by ``0.196 - 0.098 = 0.098`` along the
ladder, which is exactly the size of the effect this study has to separate from
the grid effect.  The arm ``S_w4`` (``ac_width = 4`` cells at ``lam = 64``, i.e.
``k xi = 0.393 = 2 x`` the reference value at the *same* grid) provides one
finite-difference point of ``dFr / d(k xi)`` in the opposite direction, so the
``k xi`` drift of the ladder can be extrapolated away instead of being lumped
into the grid error.  (An ``ac_width = 1`` cell arm would have reproduced the
finest arm's ``k xi``, but a one-cell diffuse interface is below the resolution
of the phase-field kernel and is not a valid control.)

Pre-declared analysis (fixed before any of these runs was executed)
-------------------------------------------------------------------
* the gate of the anchor is unchanged: ``|Fr / Fr_LZ(C=1) - 1| <= 15 %`` on every
  Ma-matched arm (``Fr = u / sqrt(A g lam)``, ``Fr_LZ = 1/sqrt(pi C)``);
* **resolution independence** of the ladder is accepted when the spread of ``Fr``
  over the three Ma-matched arms (``lam = 64, 96, 128``) is
  ``<= LADDER_SPREAD_PCT = 5`` percentage points;
* if the spread exceeds that, the ``k xi -> 0`` limit is extrapolated from the
  ``Fr(k xi)`` line and reported together with its ``R^2`` -- an extrapolation is
  a *diagnosis*, not a gate;
* the Mach arm (``lam = 64``, ``g`` halved) is a **falsification test** of the
  reference PASS: if ``Fr`` is a Mach artefact, ``Fr`` must move towards
  ``Fr_LZ`` (or away from it) by roughly the change in ``Ma^2``; a shift
  ``<= 2 %`` is reported as "no resolvable Ma dependence".

Usage
-----
    python3 run_simladder.py                          # full ladder (queue)
    python3 run_simladder.py --cases S_ref --quick     # <= 400 steps smoke
    python3 run_simladder.py --shard 2/5               # one arm per shard
    python3 run_simladder.py --aggregate               # tables from case.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

torch.set_num_threads(int(os.environ.get("RT_THREADS", "4")))

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))
if str(BENCH_DIR / "ref") not in sys.path:
    sys.path.insert(0, str(BENCH_DIR / "ref"))

import ref_lock  # noqa: E402  (reference values: single source of truth)
import run_late as la  # noqa: E402  (long-run driver; reused, not duplicated)
import late_analysis as lan  # noqa: E402  (pure analysis: C_2D / gate / Layzer)
from run_case import CaseSpec  # noqa: E402  (case definition; single source)

OUT_ROOT = BENCH_DIR / "artifacts" / "simladder"

#: similarity anchor: the reference point of the case (lam = 64, g = 2e-4).
LAM0, G0, TAU0 = 64, 2.0e-4, 0.515
#: pre-declared ladder acceptance (see module docstring)
LADDER_SPREAD_PCT = 5.0
#: a Mach-induced shift of Fr below this is "not resolvable"
MACH_RESOLUTION_PCT = 2.0


def _nu0() -> float:
    return (TAU0 - 0.5) / 3.0


def sim_specs() -> tuple[CaseSpec, ...]:
    """The similarity-preserving ladder (see module docstring)."""
    specs = [
        # reference point -- identical to ``late_l064_g2e-4`` (reproduction check)
        CaseSpec("S_ref_l064_g2e-4", lam=LAM0, g=G0, tau=TAU0,
                 note="reference point, Ma-matched anchor"),
        # Mach arm: same grid, half gravity -> Ma/sqrt2, zeta*sqrt2
        CaseSpec("S_mach_l064_g1e-4", lam=LAM0, g=0.5 * G0, tau=TAU0,
                 note="falsification: half gravity at fixed grid (Ma/sqrt2)"),
    ]
    # similarity ladder: lam -> s*lam0, g -> g0/s, nu -> s*nu0, layers fixed
    for s in (1.5, 2.0):
        lam = int(round(s * LAM0))
        nu = s * _nu0()
        specs.append(CaseSpec(f"S_lad{lam}_g{G0 / s:.2e}", lam=lam, g=G0 / s,
                              tau=0.5 + 3.0 * nu,
                              note=f"similarity ladder x{s:g} (Ma, zeta fixed)"))
    # interface-thickness control: k*xi of the lam = 128 arm at lam = 64
    specs.append(CaseSpec("S_w4_l064_g2e-4", lam=LAM0, g=G0, tau=TAU0, ac_width=4.0,
                          note="k*xi control (ac_width = 4 cells at lam = 64)"))
    return tuple(specs)


#: the three Ma-matched arms of the resolution ladder (pre-declared)
LADDER_NAMES = ("S_ref_l064_g2e-4", "S_lad96_g1.33e-04", "S_lad128_g1.00e-04")


def _select(names: tuple[str, ...] | None, shard: str | None) -> list[CaseSpec]:
    specs = list(sim_specs())
    if names:
        specs = [s for s in specs if s.name in names]
    if shard:
        i_s, n_s = shard.split("/")
        i, n = int(i_s), int(n_s)
        specs = [s for k, s in enumerate(specs) if k % n == i % n]
    return specs


def run(names: tuple[str, ...] | None = None, shard: str | None = None,
        quick: bool = False) -> list[dict]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cases = []
    for spec in _select(names, shard):
        print(f"[simladder] running {spec.name} "
              f"(lam={spec.lam}, g={spec.g:.3e}, tau={spec.tau}, "
              f"n_steps={spec.n_steps()}, grid={spec.lam}x{int(spec.layers * spec.lam)})",
              flush=True)
        case = la.run_case(spec, OUT_ROOT, quick=quick)
        cases.append(case)
        print(f"[simladder] {spec.name}: Fr={case['fr_meas']:.4f} "
              f"err={case['err_fr_vs_layzer_c1_pct']:+.2f}% "
              f"Ma={case.get('ma_peak', float('nan')):.3f} "
              f"stop={case.get('stop_reason')} "
              f"wall={case.get('wall_seconds', 0.0):.1f}s", flush=True)
    return cases


# ---------------------------------------------------------------------------
# aggregation (no solver: reads the case.json files only)
# ---------------------------------------------------------------------------
def _load_cases() -> list[dict]:
    out = []
    for p in sorted(OUT_ROOT.glob("*/case.json")):
        out.append(json.loads(p.read_text()))
    return out


def _extrapolate(xs: list[float], ys: list[float]) -> dict:
    """Least-squares ``y = a + b x`` with ``R^2`` (used for ``k xi -> 0``)."""
    n = len(xs)
    if n < 2:
        return {"n": n, "intercept": float("nan"), "slope": float("nan"),
                "r2": float("nan")}
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0.0:
        return {"n": n, "intercept": float("nan"), "slope": float("nan"),
                "r2": float("nan")}
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys))
    return {"n": n, "intercept": a, "slope": b,
            "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")}


def aggregate(out_root: Path | None = None) -> dict:
    out_root = out_root or OUT_ROOT
    cases = []
    for p in sorted(Path(out_root).glob("*/case.json")):
        cases.append(json.loads(p.read_text()))
    if not cases:
        raise SystemExit(f"no case.json under {out_root}")

    C = lan.C_2D
    rows = []
    for c in cases:
        lam = float(c["lam"])
        kxi = 2.0 * math.pi * float(c["ac_width"]) / lam
        zeta = float(c["nu"]) * (2.0 * math.pi / lam) ** 2 / math.sqrt(
            float(c["atwood"]) * float(c["g"]) * 2.0 * math.pi / lam)
        rows.append({
            "name": c["name"],
            "lam": lam,
            "g": float(c["g"]),
            "nu": float(c["nu"]),
            "k_xi": kxi,
            "zeta": zeta,
            "ma_peak": c.get("ma_peak", float("nan")),
            "n_steps": c.get("n_steps_run"),
            "tau_reached": c.get("tau_reached"),
            "window_complete": c.get("window_complete"),
            "stop_reason": c.get("stop_reason"),
            "u_mean": 0.5 * (float(c["u_bubble_sim"]) + float(c["u_spike_sim"])),
            "fr": float(c["fr_meas"]),
            "fr_lz": float(c["fr_lz"]),
            "err_pct": float(c["err_fr_vs_layzer_c1_pct"]),
            "c_eff": float(c["c_eff"]),
            "gate_15pct": bool(c.get("gate_terminal_15pct")),
            "subwindow_drift_pct": c.get("slope_subwindow_drift_pct"),
            "wall_seconds": c.get("wall_seconds"),
        })

    by_name = {r["name"]: r for r in rows}
    lad = [by_name[n] for n in LADDER_NAMES if n in by_name]
    spread = (max(r["fr"] for r in lad) - min(r["fr"] for r in lad)) \
        if len(lad) >= 2 else float("nan")
    spread_pct = 100.0 * spread / (sum(r["fr"] for r in lad) / len(lad)) if lad else float("nan")

    extrap = {"n": 0, "intercept": float("nan"), "slope": float("nan"), "r2": float("nan")}
    extrap_err_pct = float("nan")
    if len(lad) >= 3:
        extrap = _extrapolate([r["k_xi"] for r in lad], [r["fr"] for r in lad])
        extrap_err_pct = 100.0 * (extrap["intercept"] / lad[0]["fr_lz"] - 1.0)

    mach_row = by_name.get("S_mach_l064_g1e-4")
    ref_row = by_name.get("S_ref_l064_g2e-4")
    mach_shift_pct = (100.0 * (mach_row["fr"] / ref_row["fr"] - 1.0)
                      if (mach_row and ref_row) else float("nan"))
    w4_row = by_name.get("S_w4_l064_g2e-4")

    summary = {
        "criterion": {
            "gate_pct": lan.GATE_PCT,
            "fr_lz_c1": lan.layzer_froude(0.5, G0, LAM0, C),
            "ladder_spread_pct_max": LADDER_SPREAD_PCT,
            "mach_resolution_pct": MACH_RESOLUTION_PCT,
            "ladder_arms": list(LADDER_NAMES),
            "c_2d": C,
        },
        "rows": rows,
        "ladder": {
            "fr_spread_pct": spread_pct,
            "gate_pass_all": all(r["gate_15pct"] for r in lad) if lad else False,
            "resolution_independent": (spread_pct <= LADDER_SPREAD_PCT) if lad else False,
            "extrapolation_kxi0": extrap,
            "extrapolated_err_vs_fr_lz_pct": extrap_err_pct,
        },
        "mach_test": {
            "fr_ref": ref_row["fr"] if ref_row else float("nan"),
            "fr_mach": mach_row["fr"] if mach_row else float("nan"),
            "ma_ref": ref_row["ma_peak"] if ref_row else float("nan"),
            "ma_half_g": mach_row["ma_peak"] if mach_row else float("nan"),
            "shift_pct": mach_shift_pct,
            "resolvable": abs(mach_shift_pct) > MACH_RESOLUTION_PCT
                          if math.isfinite(mach_shift_pct) else False,
        },
        "kxi_control": {
            "fr_w4": w4_row["fr"] if w4_row else float("nan"),
            "k_xi_w4": w4_row["k_xi"] if w4_row else float("nan"),
            "shift_vs_ref_pct": (100.0 * (w4_row["fr"] / ref_row["fr"] - 1.0)
                                 if (w4_row and ref_row) else float("nan")),
        },
    }
    Path(out_root, "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    Path(out_root, "acceptance_table.md").write_text(render_table(summary))
    return summary


def render_table(summary: dict) -> str:
    crit = summary["criterion"]
    lines = [
        "# CASE-RT-INSTABILITY-ATWOOD -- similarity ladder of the terminal front velocity",
        "",
        f"Gate (anchor): |Fr / Fr_LZ(C={crit['c_2d']:g}) - 1| <= {crit['gate_pct']:g} %, "
        f"Fr_LZ = {crit['fr_lz_c1']:.4f}; ladder acceptance: spread <= "
        f"{crit['ladder_spread_pct_max']:g} pp; Ma resolution: "
        f"{crit['mach_resolution_pct']:g} pp.",
        "",
        "| case | lam | g | nu | k*xi | zeta | Ma_peak | tau_end | stop "
        "| u_mean | Fr | err vs Fr_LZ | gate | C_eff | sub-window drift |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary["rows"]:
        lines.append(
            f"| {r['name']} | {r['lam']:.0f} | {r['g']:.3e} | {r['nu']:.4g} "
            f"| {r['k_xi']:.3f} | {r['zeta']:.4f} | {r['ma_peak']:.3f} "
            f"| {r['tau_reached']:.2f} | {r['stop_reason']} | {r['u_mean']:.4e} "
            f"| {r['fr']:.4f} | {r['err_pct']:+.2f} % "
            f"| {'PASS' if r['gate_15pct'] else 'FAIL'} | {r['c_eff']:.3f} "
            f"| {r['subwindow_drift_pct']:+.2f} % |")
    lad = summary["ladder"]
    lines += [
        "",
        "## Ladder verdict",
        "",
        f"* resolution ladder (Ma-matched, zeta-matched): spread = "
        f"**{lad['fr_spread_pct']:+.2f} pp** "
        f"(accepted <= {crit['ladder_spread_pct_max']:g} pp) -> "
        f"**{'resolution independent' if lad['resolution_independent'] else 'NOT resolution independent'}**",
        f"* every ladder arm inside the anchor gate: "
        f"**{'yes' if lad['gate_pass_all'] else 'no'}**",
        f"* k*xi -> 0 extrapolation: Fr = {lad['extrapolation_kxi0']['intercept']:.4f} "
        f"(slope {lad['extrapolation_kxi0']['slope']:+.4f}, "
        f"R^2 = {lad['extrapolation_kxi0']['r2']:.4f}) -> error vs Fr_LZ = "
        f"{lad['extrapolated_err_vs_fr_lz_pct']:+.2f} %",
        "",
        "## Mach falsification and k*xi control",
        "",
        f"* Ma arm (half g at lam = 64): Ma {summary['mach_test']['ma_ref']:.3f} -> "
        f"{summary['mach_test']['ma_half_g']:.3f}, Fr {summary['mach_test']['fr_ref']:.4f} -> "
        f"{summary['mach_test']['fr_mach']:.4f} (shift "
        f"{summary['mach_test']['shift_pct']:+.2f} %) -> "
        f"**{'resolvable Ma dependence' if summary['mach_test']['resolvable'] else 'no resolvable Ma dependence'}**",
        "* k*xi control (ac_width 2 -> 4 cells at lam = 64, k*xi = "
        f"{summary['kxi_control']['k_xi_w4']:.3f}): Fr = {summary['kxi_control']['fr_w4']:.4f} "
        f"({summary['kxi_control']['shift_vs_ref_pct']:+.2f} % vs reference)",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    global OUT_ROOT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", nargs="*", default=None,
                    help="subset of case names (default: the whole ladder)")
    ap.add_argument("--shard", default=None, help="i/n -- run shard i of n")
    ap.add_argument("--quick", action="store_true", help="<= 400 steps smoke run")
    ap.add_argument("--aggregate", action="store_true", help="tables from case.json")
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args(argv)

    out_root = Path(args.out)
    OUT_ROOT = out_root
    if args.aggregate:
        summary = aggregate(out_root)
        print(render_table(summary))
        return 0
    run(names=tuple(args.cases) if args.cases else None, shard=args.shard,
        quick=args.quick)
    if not args.quick:
        aggregate(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())