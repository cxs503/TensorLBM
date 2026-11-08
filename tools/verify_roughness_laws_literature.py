#!/usr/bin/env python3
"""Independent literature re-verification of :mod:`tensorlbm.roughness_laws`.

Why this file exists
--------------------
``src/tensorlbm/roughness_laws.py`` is the **adjudication basis** of the
roughness campaign (``tools/run_roughness_sweep.py``): every measured roughness
function ``dU+`` is converted to a friction increment through it, and the
ship-scale allowance is compared against it.  Until this audit it had no test
and no independent check of its published constants.

This script does **not** re-export the module's own formulas.  Every reference
value is rebuilt from an independent statement of the classic law (as printed
in the source paper / ITTC procedure), and the module output is then compared
against it:

===============  ============================================================
anchor           independent statement
===============  ============================================================
A  ITTC-57       ``Cf = 0.075 / (log10 Re - 2)^2``, compared against its
                 published companion-table values (4 significant figures)
B  Nikuradse     fully rough velocity law ``U+ = (1/kappa) ln(y/ks) + 8.5``
                 combined with the smooth law ``U+ = (1/kappa) ln y+ + 5.0``;
                 the roughness function is their difference at ``y = ks``
C  Cebeci        transition closure ``dU+ = (1/kappa) ln(1 + 0.3 ks+)`` whose
                 fully rough intercept must be ``(1/kappa) ln(0.3)``
D  ITTC-78       allowance ``dCf = [105 (ks/L)^(1/3) - 0.64] x 1e-3``: exact
                 zero crossing, documented worked value, monotonicity
E  momentum Cf   ``Cf = 2/U+^2`` derived from ``tau_w = rho u_tau^2`` and
                 ``Cf = tau_w / (1/2 rho U^2)`` (symbol-level identity)
F  route gap     the two adjudication routes used by the campaign
                 (measured ``dU+`` -> ``dCf`` vs the ITTC-78 allowance)
                 mapped onto each other through ``ks/L = ks+ U+ / Re_L``
===============  ============================================================

Outputs (all numbers in the report are read back from these files):
  * ``artifacts/roughness_laws_audit/audit.json``     machine numbers
  * ``artifacts/roughness_laws_audit/table_*.md``     markdown tables
  * ``artifacts/roughness_laws_audit/fig_*.png``      figures

Run: ``python tools/verify_roughness_laws_literature.py`` (pure numpy, < 5 s).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm import roughness_laws as rl  # noqa: E402

OUT = ROOT / "artifacts" / "roughness_laws_audit"

#: Independent (literature) constants -- deliberately re-typed here so a typo
#: in the module cannot propagate into the reference side of the comparison.
KAPPA_LIT = 0.41
B_SMOOTH_LIT = 5.0          # smooth-wall log-law intercept (Schlichting)
B_ROUGH_LIT = 8.5           # Nikuradse (1933) fully rough intercept
CEBECI_A_LIT = 0.3          # Cebeci & Bradshaw / Knopp et al. closure

#: ITTC-57 companion-table values, 4 significant figures (ITTC 7.5-02-03-01.4;
#: reproduced in Molland, Turnock & Hudson 2017, Table 7.1).
ITTC57_TABLE = {
    1.0e5: 8.333e-3,
    1.0e6: 4.688e-3,
    1.0e7: 3.000e-3,
    1.0e8: 2.083e-3,
    1.0e9: 1.531e-3,
}


# --------------------------------------------------------------------------
# A. ITTC-57 correlation line
# --------------------------------------------------------------------------
def audit_ittc57() -> dict[str, Any]:
    rows = []
    for re, published in ITTC57_TABLE.items():
        got = float(rl.ittc57_cf(re))
        # independent arithmetic straight from the published expression
        indep = 0.075 / (math.log10(re) - 2.0) ** 2
        rows.append(
            {
                "re": re,
                "cf_module": got,
                "cf_published": published,
                "cf_independent": indep,
                "rel_vs_published": got / published - 1.0,
                "rel_vs_independent": got / indep - 1.0,
            }
        )
    max_rel_pub = max(abs(r["rel_vs_published"]) for r in rows)
    max_rel_ind = max(abs(r["rel_vs_independent"]) for r in rows)
    # monotone decreasing and positive over the published range
    cfs = [r["cf_module"] for r in rows]
    monotone = all(a > b for a, b in zip(cfs, cfs[1:]))
    return {
        "rows": rows,
        "max_abs_rel_vs_published": max_rel_pub,
        "max_abs_rel_vs_independent": max_rel_ind,
        "monotone_decreasing": monotone,
        "pass_1e-3_vs_published": max_rel_pub <= 1.0e-3,
    }


# --------------------------------------------------------------------------
# B. Nikuradse fully rough asymptote, rebuilt from the two velocity laws
# --------------------------------------------------------------------------
def delta_u_plus_from_laws(ks_plus: float) -> float:
    """``dU+ = U+_smooth(y=ks) - U+_rough(y=ks)`` from the classic laws only.

    ``U+_smooth = (1/kappa) ln y+ + 5.0`` and ``U+_rough = (1/kappa) ln(y/ks)
    + 8.5`` are evaluated at the sand-grain crest ``y = ks`` (``y+ = ks+``),
    so the difference is ``(1/kappa) ln(ks+) + 5.0 - 8.5``.
    """
    y_plus = float(ks_plus)
    u_smooth = (1.0 / KAPPA_LIT) * math.log(y_plus) + B_SMOOTH_LIT
    # at the crest y = ks the rough-wall argument y/ks is one, so the log term
    # of the rough law vanishes and only the intercept survives
    u_rough = (1.0 / KAPPA_LIT) * math.log(1.0) + B_ROUGH_LIT
    return u_smooth - u_rough


def audit_nikuradse(ks_samples: tuple[float, ...] = (5.0, 10.0, 20.0, 50.0, 90.0, 100.0, 300.0, 1000.0)) -> dict[str, Any]:
    rows = []
    for ks in ks_samples:
        ref = delta_u_plus_from_laws(ks)
        got = float(rl.nikuradse_delta_u_plus(ks))
        ceb = float(rl.cebeci_delta_u_plus(ks))
        rows.append(
            {
                "ks_plus": ks,
                "delta_u_plus_module": got,
                "delta_u_plus_from_laws": ref,
                "abs_dev": abs(got - ref),
                "cebeci": ceb,
                "nikuradse_minus_cebeci": got - ceb,
                "regime": rl.regime(ks),
            }
        )
    # clip threshold: the largest ks+ at which the asymptote is still <= 0
    clip = rl.NIKURADSE_CLIP_KS_PLUS  # 1/kappa ln(ks+) - 3.5 = 0, exp(3.5 kappa)
    # asymptotic offset of the two closures (should equal -ln(0.3)/kappa - 3.5)
    offset_pred = -math.log(CEBECI_A_LIT) / KAPPA_LIT - 3.5
    big = 1.0e8
    offset_num = float(rl.nikuradse_delta_u_plus(big)) - float(rl.cebeci_delta_u_plus(big))
    return {
        "rows": rows,
        "max_abs_dev": max(r["abs_dev"] for r in rows),
        "clip_threshold_ks_plus": clip,
        "clip_threshold_claim_doc": 4.2,
        "cebeci_asymptote_offset_pred": offset_pred,
        "cebeci_asymptote_offset_numeric": offset_num,
        "cebeci_offset_abs_dev": abs(offset_pred - offset_num),
        "pass_machine_precision": max(r["abs_dev"] for r in rows) < 1.0e-12,
    }


# --------------------------------------------------------------------------
# C/D. ITTC-78 allowance
# --------------------------------------------------------------------------
def audit_ittc78() -> dict[str, Any]:
    zero_exact = (0.64 / 105.0) ** 3          # dCf = 0 in exact arithmetic
    zero_doc = 5.3e-7                          # claim in the module docstring
    raw_at_doc = (105.0 * (zero_doc ** (1.0 / 3.0)) - 0.64) * 1.0e-3
    # numerical zero crossing of the module's clipped function
    lo, hi = 1.0e-9, 1.0e-5
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(rl.ittc78_allowance(mid)) > 0.0:
            hi = mid
        else:
            lo = mid
    zero_num = 0.5 * (lo + hi)
    worked = []
    for L in (100.0, 200.0, 300.0):
        for ks_mm in (0.05, 0.15, 0.40, 1.00):
            ks = ks_mm * 1.0e-3
            x = ks / L
            worked.append(
                {
                    "L_m": L,
                    "ks_mm": ks_mm,
                    "ks_over_L": x,
                    "dcf_module": float(rl.ittc78_allowance(x)),
                    "dcf_independent": max((105.0 * (x ** (1.0 / 3.0)) - 0.64) * 1.0e-3, 0.0),
                }
            )
    # monotone non-decreasing in ks/L
    xs = np.logspace(-8, -3, 400)
    vals = rl.ittc78_allowance(xs)
    monotone = bool(np.all(np.diff(vals) >= 0.0))
    max_rel = max(
        abs(r["dcf_module"] / r["dcf_independent"] - 1.0)
        for r in worked
        if r["dcf_independent"] > 0.0
    )
    return {
        "zero_crossing_exact": zero_exact,
        "zero_crossing_numeric": zero_num,
        "zero_crossing_doc_claim": zero_doc,
        "raw_polynomial_at_doc_claim": raw_at_doc,
        "doc_claim_falsified": bool(raw_at_doc > 0.0),
        "worked": worked,
        "max_rel_vs_independent": max_rel,
        "monotone": monotone,
        "pass_zero_crossing": abs(zero_num / zero_exact - 1.0) < 1.0e-6,
    }


# --------------------------------------------------------------------------
# E. momentum convention Cf = 2/U+^2 derived from symbols
# --------------------------------------------------------------------------
def audit_momentum_cf() -> dict[str, Any]:
    # Cf = tau_w / (1/2 rho U^2) with tau_w = rho u_tau^2  =>  2 u_tau^2 / U^2
    rows = []
    for u_plus in (12.0, 18.0, 25.0, 36.0):
        got = float(rl.cf_from_u_plus(u_plus))
        from_symbols = 2.0 * (1.0 / u_plus) ** 2
        rows.append({"u_plus": u_plus, "cf_module": got, "cf_from_symbols": from_symbols,
                     "abs_dev": abs(got - from_symbols)})
    # round trip: Cf -> U+ -> Cf
    rt = []
    for cf in (2.0e-3, 1.5e-3, 4.7e-3):
        u_plus = math.sqrt(2.0 / cf)
        rt.append({"cf_in": cf, "cf_round_trip": float(rl.cf_from_u_plus(u_plus))})
    return {
        "rows": rows,
        "max_abs_dev": max(r["abs_dev"] for r in rows),
        "pass_machine_precision": max(r["abs_dev"] for r in rows) < 1.0e-15,
        "round_trip_max_rel": max(abs(r["cf_round_trip"] / r["cf_in"] - 1.0) for r in rt),
    }


# --------------------------------------------------------------------------
# F. route gap: measured dU+ -> dCf  vs  ITTC-78 allowance
# --------------------------------------------------------------------------
def u_plus_from_ittc57(re_l: float) -> float:
    """``U+ = sqrt(2/Cf)`` with ``Cf`` from the ITTC-57 line (momentum inversion)."""
    return math.sqrt(2.0 / float(rl.ittc57_cf(re_l)))


def audit_route_gap(
    ks_plus_samples: tuple[float, ...] = (18.75, 37.5, 75.0, 150.0, 225.0),
    re_l_samples: tuple[float, ...] = (5.0e8, 1.0e9, 2.0e9),
) -> dict[str, Any]:
    """Cross-check the two dCf routes the campaign actually reports.

    ``ks/L = ks+ U+ / Re_L`` follows from ``ks+ = ks u_tau / nu``,
    ``u_tau = U/U+`` and ``Re_L = U L / nu`` (algebra, no free constant).
    """
    rows = []
    for re_l in re_l_samples:
        u_plus = u_plus_from_ittc57(re_l)
        for ks in ks_plus_samples:
            d_u_plus = float(rl.nikuradse_delta_u_plus(ks))
            dcf_mom = float(rl.delta_cf_from_delta_u_plus(u_plus, d_u_plus))
            ks_over_l = ks * u_plus / re_l
            dcf_78 = float(rl.ittc78_allowance(ks_over_l))
            rows.append(
                {
                    "re_l": re_l,
                    "u_plus": u_plus,
                    "ks_plus": ks,
                    "ks_over_l": ks_over_l,
                    "dcf_momentum_route": dcf_mom,
                    "dcf_ittc78_route": dcf_78,
                    "rel_gap": dcf_mom / dcf_78 - 1.0 if dcf_78 > 0.0 else float("nan"),
                }
            )
    # express the convention gap as an equivalent sand-grain ratio: which
    # ks_eff/ks_geom makes the momentum route reproduce the ITTC-78 number?
    for row in rows:
        target = row["dcf_ittc78_route"]

        def momentum_at(ksp: float, u_plus: float = row["u_plus"]) -> float:
            return float(rl.delta_cf_from_delta_u_plus(u_plus, float(rl.nikuradse_delta_u_plus(ksp))))

        lo, hi = 0.05, 1.0e5
        if momentum_at(lo) > target:
            row["implied_ks_ratio"] = float("nan")
            continue
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            if momentum_at(mid) > target:
                hi = mid
            else:
                lo = mid
        row["implied_ks_ratio"] = math.sqrt(lo * hi) / row["ks_plus"]

    gaps = [abs(r["rel_gap"]) for r in rows if math.isfinite(r["rel_gap"])]
    ratios = [r["implied_ks_ratio"] for r in rows if math.isfinite(r["implied_ks_ratio"])]
    return {
        "rows": rows,
        "max_abs_rel_gap": max(gaps),
        "median_abs_rel_gap": float(np.median(gaps)),
        "n_beyond_3pct": int(sum(1 for g in gaps if g > 0.03)),
        "n_compared": len(gaps),
        "implied_ks_ratio_min": min(ratios) if ratios else float("nan"),
        "implied_ks_ratio_max": max(ratios) if ratios else float("nan"),
        "implied_ks_ratio_median": float(np.median(ratios)) if ratios else float("nan"),
    }


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def make_figures(nik: dict[str, Any], it78: dict[str, Any], gap: dict[str, Any]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    OUT.mkdir(parents=True, exist_ok=True)

    # fig 1: roughness functions and their literature band
    ks = np.logspace(-0.5, 4.0, 400)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.semilogx(ks, rl.nikuradse_delta_u_plus(ks), label=r"Nikuradse $(1/\kappa)\ln k_s^+-3.5$")
    ax.semilogx(ks, rl.cebeci_delta_u_plus(ks), label=r"Cebeci $(1/\kappa)\ln(1+0.3k_s^+)$")
    band = rl.literature_band(ks)
    ax.fill_between(ks, band["lo"], band["hi"], color="grey", alpha=0.25,
                    label=r"literature band ($\pm 0.5u_\tau$)")
    ax.axvline(rl.SMOOTH_REGIME_KS_PLUS, ls=":", color="k", lw=1)
    ax.axvline(rl.FULLY_ROUGH_KS_PLUS, ls="--", color="k", lw=1)
    ax.text(rl.SMOOTH_REGIME_KS_PLUS, 0.3, " smooth|trans", fontsize=8, rotation=90, va="bottom")
    ax.text(rl.FULLY_ROUGH_KS_PLUS, 0.3, " trans|fully rough", fontsize=8, rotation=90, va="bottom")
    ax.plot([r["ks_plus"] for r in nik["rows"]], [r["delta_u_plus_from_laws"] for r in nik["rows"]],
            "o", ms=3, mfc="none", label="independent law difference")
    ax.set_xlabel(r"$k_s^+$")
    ax.set_ylabel(r"$\Delta U^+$")
    ax.set_title("Roughness functions vs independent law reconstruction")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = OUT / "fig_roughness_functions.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p.relative_to(ROOT)))

    # fig 2: two dCf routes
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for re_l in sorted({r["re_l"] for r in gap["rows"]}):
        sel = [r for r in gap["rows"] if r["re_l"] == re_l]
        ax.loglog([r["ks_over_l"] for r in sel], [r["dcf_momentum_route"] for r in sel],
                  "o-", ms=4, label=f"momentum route, $Re_L$={re_l:.1e}")
        ax.loglog([r["ks_over_l"] for r in sel], [r["dcf_ittc78_route"] for r in sel],
                  "s--", ms=4, label=f"ITTC-78 route, $Re_L$={re_l:.1e}")
    xs = np.logspace(np.log10(min(r["ks_over_l"] for r in gap["rows"]) * 0.5),
                     np.log10(max(r["ks_over_l"] for r in gap["rows"]) * 2.0), 200)
    ax.loglog(xs, rl.ittc78_allowance(xs), "-", color="k", lw=1, alpha=0.6, label="ITTC-78 allowance")
    ax.set_xlabel(r"$k_s/L$")
    ax.set_ylabel(r"$\Delta C_f$")
    ax.set_title(r"Route cross-check: measured $\Delta U^+$ vs ITTC-78 allowance")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = OUT / "fig_dcf_routes.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p.relative_to(ROOT)))

    # fig 3: ITTC-78 allowance vs ks/L for three hull lengths
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = np.logspace(-7, -3.5, 300)
    for L in (100.0, 200.0, 300.0):
        ax.semilogx(xs, 1.0e3 * rl.ittc78_allowance(xs) * 1.0e0, label=f"L = {L:.0f} m (x-axis is $k_s/L$)")
    ax.axvline(it78["zero_crossing_exact"], ls=":", color="r")
    ax.text(it78["zero_crossing_exact"] * 1.15, 0.05,
            f"exact zero\n{it78['zero_crossing_exact']:.3e}", fontsize=7, color="r")
    ax.set_xlabel(r"$k_s/L$")
    ax.set_ylabel(r"$\Delta C_f \times 10^{3}$")
    ax.set_title("ITTC-1978 roughness allowance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = OUT / "fig_ittc78_allowance.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p.relative_to(ROOT)))
    return paths


# --------------------------------------------------------------------------
# markdown tables
# --------------------------------------------------------------------------
def write_tables(audit: dict[str, Any]) -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    written = []

    a = audit["ittc57"]
    lines = ["| Re | Cf (module) | Cf (published 4 s.f.) | rel dev |", "|---|---|---|---|"]
    for r in a["rows"]:
        lines.append(
            f"| {r['re']:.0e} | {r['cf_module']:.6e} | {r['cf_published']:.4e} | "
            f"{100.0 * r['rel_vs_published']:+.4f}% |"
        )
    p = OUT / "table_ittc57.md"
    p.write_text("\n".join(lines) + "\n")
    written.append(str(p.relative_to(ROOT)))

    b = audit["nikuradse"]
    lines = ["| ks+ | regime | dU+ module | dU+ from laws | abs dev | Cebeci | Nik-Ceb |",
             "|---|---|---|---|---|---|---|"]
    for r in b["rows"]:
        lines.append(
            f"| {r['ks_plus']:g} | {r['regime']} | {r['delta_u_plus_module']:.10f} | "
            f"{r['delta_u_plus_from_laws']:.10f} | {r['abs_dev']:.3e} | {r['cebeci']:.6f} | "
            f"{r['nikuradse_minus_cebeci']:+.4f} |"
        )
    p = OUT / "table_nikuradse.md"
    p.write_text("\n".join(lines) + "\n")
    written.append(str(p.relative_to(ROOT)))

    c = audit["ittc78"]
    lines = ["| L [m] | ks [mm] | ks/L | dCf module | dCf independent |", "|---|---|---|---|---|"]
    for r in c["worked"]:
        lines.append(
            f"| {r['L_m']:.0f} | {r['ks_mm']:.2f} | {r['ks_over_L']:.3e} | "
            f"{r['dcf_module']:.6e} | {r['dcf_independent']:.6e} |"
        )
    p = OUT / "table_ittc78.md"
    p.write_text("\n".join(lines) + "\n")
    written.append(str(p.relative_to(ROOT)))

    f = audit["route_gap"]
    lines = ["| Re_L | U+ (ITTC-57) | ks+ | ks/L | dCf momentum | dCf ITTC-78 | rel gap | implied ks_eff/ks |",
             "|---|---|---|---|---|---|---|---|"]
    for r in f["rows"]:
        lines.append(
            f"| {r['re_l']:.1e} | {r['u_plus']:.3f} | {r['ks_plus']:g} | {r['ks_over_l']:.3e} | "
            f"{r['dcf_momentum_route']:.6e} | {r['dcf_ittc78_route']:.6e} | "
            f"{100.0 * r['rel_gap']:+.2f}% | {r['implied_ks_ratio']:.3f} |"
        )
    p = OUT / "table_route_gap.md"
    p.write_text("\n".join(lines) + "\n")
    written.append(str(p.relative_to(ROOT)))
    return written


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {
        "module": "src/tensorlbm/roughness_laws.py",
        "constants": {
            "kappa": rl.KAPPA,
            "b_smooth": rl.B_SMOOTH,
            "nikuradse_c": rl.NIKURADSE_C,
            "cebeci_a": rl.CEBECI_A if hasattr(rl, "CEBECI_A") else CEBECI_A_LIT,
            "smooth_regime_ks_plus": rl.SMOOTH_REGIME_KS_PLUS,
            "fully_rough_ks_plus": rl.FULLY_ROUGH_KS_PLUS,
            "nikuradse_clip_ks_plus": rl.NIKURADSE_CLIP_KS_PLUS,
        },
        "ittc57": audit_ittc57(),
        "nikuradse": audit_nikuradse(),
        "ittc78": audit_ittc78(),
        "momentum_cf": audit_momentum_cf(),
        "route_gap": audit_route_gap(),
    }
    audit["figures"] = make_figures(audit["nikuradse"], audit["ittc78"], audit["route_gap"])
    audit["tables"] = write_tables(audit)
    audit["verdict"] = {
        "ittc57_within_1e-3_of_published": audit["ittc57"]["pass_1e-3_vs_published"],
        "nikuradse_matches_law_difference": audit["nikuradse"]["pass_machine_precision"],
        "ittc78_zero_crossing_exact": audit["ittc78"]["pass_zero_crossing"],
        "ittc78_doc_claim_falsified": audit["ittc78"]["doc_claim_falsified"],
        "momentum_cf_symbol_exact": audit["momentum_cf"]["pass_machine_precision"],
        "route_gap_max_abs_rel": audit["route_gap"]["max_abs_rel_gap"],
    }
    path = OUT / "audit.json"
    path.write_text(json.dumps(audit, indent=1, sort_keys=True, default=float))
    print(json.dumps(audit["verdict"], indent=1, sort_keys=True))
    print(f"wrote {path.relative_to(ROOT)}")
    ok = all(
        [
            audit["verdict"]["ittc57_within_1e-3_of_published"],
            audit["verdict"]["nikuradse_matches_law_difference"],
            audit["verdict"]["ittc78_zero_crossing_exact"],
            audit["verdict"]["momentum_cf_symbol_exact"],
        ]
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())