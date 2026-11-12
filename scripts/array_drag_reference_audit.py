"""Audit of the periodic-array (Hasimoto) drag reference across TensorLBM.

Anchor: GAP-LBM-SPHERE (the analytic instrument that judges the case) /
CASE-LBM-CUBIC-ARRAY-DRAG (the array-drag benchmark of the same series).

Everything here is generated in code -- the historical (defective) series are
re-typed below *because* they are the objects under audit; the reference side
always goes through ``tensorlbm.array_drag_reference``.

Outputs (all under ``artifacts/array_drag_reference/``):
  audit.json  -- machine-readable numbers
  tables.md   -- the markdown tables quoted by docs/ARRAY_DRAG_REFERENCE_AUDIT.md
  fig.png     -- left: K(lambda) of every variant; right: deviation vs canonical

Run:  python scripts/array_drag_reference_audit.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorlbm.array_drag_reference import (  # noqa: E402
    PHI_PER_LAMBDA_CUBED,
    SERIES_MAX_LAMBDA,
    array_drag_factor_from_lambda,
    array_drag_factor_from_lambda_truncated,
    array_drag_factor_from_phi,
    phi_from_lambda,
)
from tensorlbm.d3q27_recovery import hasimoto_correction  # noqa: E402
from tensorlbm.particle_suspension import hasimoto_drag_correction  # noqa: E402
from tensorlbm.sc_sphere_array_drag import SC_TABLE, hasimoto_K  # noqa: E402

OUT = REPO / "artifacts" / "array_drag_reference"
OUT.mkdir(parents=True, exist_ok=True)

#: lambda of the GAP-LBM-SPHERE case (R = 6, L_cell = 36, driver convention
#: lambda = R/L; the prereg's lambda = 2R/L is audited separately below).
CASE_LAMBDA = 6.0 / 36.0
#: values of lambda at which every variant is compared
LAMBDAS = (1.0 / 20.0, 1.0 / 10.0, 1.0 / 8.0, 1.0 / 6.0, 0.25)


# ---------------------------------------------------------------------------
# Historical (audited) series
# ---------------------------------------------------------------------------
def old_sphere_driver(lam: float) -> float:
    """benchmarks/sphere_curved_drag/driver.py before the 2026-11-12 audit."""
    return 1.0 / (1.0 - 2.837 * lam + 4.19 * lam**3 - 27.4 * lam**5)


def old_force_control_surface(lam: float) -> float:
    """tensorlbm.force_control_surface.hasimoto_factor before the audit."""
    return 1.0 / (
        1.0
        - 2.10444 * lam
        + 2.08877 * lam**3
        - 0.94813 * lam**5
        - 1.372 * lam**6
        + 3.87 * lam**8
        - 4.19 * lam**10
    )


VARIANTS = (
    ("canonical (phi series)", array_drag_factor_from_lambda),
    ("lambda trunc. thru lambda^6", array_drag_factor_from_lambda_truncated),
    ("d3q27_recovery (thru lambda^3)", hasimoto_correction),
    ("sphere driver, pre-audit (lambda^5)", old_sphere_driver),
    ("force_control_surface, pre-audit", old_force_control_surface),
    ("sc_sphere_array_drag (phi entry, canonical)", lambda lam, f=hasimoto_K: f(phi_from_lambda(lam))),
)


def pct(a: float, b: float) -> float:
    return 100.0 * (a / b - 1.0)


def main() -> int:
    audit: dict = {
        "case_lambda": CASE_LAMBDA,
        "series_max_lambda": SERIES_MAX_LAMBDA,
        "variants": {},
    }

    # --- table A: every variant at every lambda, deviation vs canonical ------
    rows_a = []
    for lam in LAMBDAS:
        ref = array_drag_factor_from_lambda(lam)
        row = {"lambda": lam, "phi": phi_from_lambda(lam), "canonical": ref}
        for name, fn in VARIANTS:
            val = float(fn(lam))
            row[name] = {"K": val, "dev_pct": pct(val, ref)}
        rows_a.append(row)
    audit["table_a"] = rows_a

    # --- table B: canonical series vs the locked SC table -------------------
    rows_b = []
    for phi, k_table in sorted(SC_TABLE.items()):
        k = array_drag_factor_from_phi(phi)
        rows_b.append(
            {"phi": phi, "series": k, "locked_table": k_table, "dev_pct": pct(k, k_table)}
        )
    audit["table_b"] = rows_b
    audit["validity_max_dev_pct_within_bound"] = max(
        abs(r["dev_pct"]) for r in rows_b if r["phi"] <= 0.216
    )

    # --- table C: consequences at the GAP-LBM-SPHERE geometry ---------------
    lam = CASE_LAMBDA
    k_ref = array_drag_factor_from_lambda(lam)
    lam_prereg = 2.0 * lam  # the prereg's (wrong) definition, v3 §3.1 / amendment A1
    rows_c = {
        "case_lambda": lam,
        "case_phi": phi_from_lambda(lam),
        "K_canonical": k_ref,
        "K_pre_audit_driver": float(old_sphere_driver(lam)),
        "driver_dev_pct": pct(float(old_sphere_driver(lam)), k_ref),
        "K_pre_audit_force_control_surface": float(old_force_control_surface(lam)),
        "force_control_surface_dev_pct": pct(float(old_force_control_surface(lam)), k_ref),
        "prereg_lambda": lam_prereg,
        "K_at_prereg_lambda": array_drag_factor_from_lambda(lam_prereg),
        "prereg_lambda_dev_pct": pct(array_drag_factor_from_lambda(lam_prereg), k_ref),
        "criterion_tolerance_pct": 8.0,
        "budget_share_of_tolerance_pct": 100.0 * abs(pct(float(old_sphere_driver(lam)), k_ref)) / 8.0,
    }
    audit["table_c"] = rows_c

    # --- table D: two-term phi truncation used by particle_suspension -------
    rows_d = []
    for phi in (1e-5, 1e-4, 1e-3, 1e-2, 0.05):
        ref = array_drag_factor_from_phi(phi)
        rows_d.append(
            {
                "phi": phi,
                "two_term": float(hasimoto_drag_correction(phi)),
                "canonical": ref,
                "dev_pct": pct(float(hasimoto_drag_correction(phi)), ref),
            }
        )
    audit["table_d"] = rows_d

    # --- figure ------------------------------------------------------------
    lam_grid = np.linspace(1e-3, SERIES_MAX_LAMBDA, 400)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ref_grid = np.array([array_drag_factor_from_lambda(x) for x in lam_grid])
    for name, fn in VARIANTS:
        vals = np.array([float(fn(x)) for x in lam_grid])
        ax1.plot(lam_grid, vals, lw=1.6 if "canonical" in name else 1.1, label=name)
        ax2.plot(lam_grid, 100.0 * (vals / ref_grid - 1.0), lw=1.4, label=name)
    for ax in (ax1, ax2):
        ax.axvline(CASE_LAMBDA, color="k", ls=":", lw=0.9)
        ax.grid(alpha=0.3)
        ax.set_xlabel(r"$\lambda = a/L_{\rm cell}$")
    ax1.set_ylabel(r"$K = F/(6\pi\mu a U)$")
    ax1.set_title("Hasimoto SC array correction: six implementations")
    ax2.set_ylabel("deviation vs canonical series  [%]")
    ax2.set_title("the judge itself was the outlier")
    ax2.axhline(0.0, color="k", lw=0.8)
    ax1.text(CASE_LAMBDA, ax1.get_ylim()[1] * 0.6, " GAP-LBM-SPHERE\n $\\lambda=1/6$", fontsize=7)
    ax1.legend(fontsize=6.5, loc="upper left")
    ax2.legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig.png", dpi=160)
    plt.close(fig)

    (OUT / "audit.json").write_text(json.dumps(audit, indent=1) + "\n")

    # --- generated markdown tables ----------------------------------------
    lines: list[str] = []
    lines.append("### Table A - every variant at every lambda (deviation vs canonical)\n")
    head = "| lambda | phi | canonical K | " + " | ".join(
        f"{n} K (dev)" for n, _ in VARIANTS[1:]
    ) + " |"
    lines.append(head)
    lines.append("|" + "---|" * (3 + len(VARIANTS) - 1))
    for r in rows_a:
        cells = [f"{r['lambda']:.4f}", f"{r['phi']:.5f}", f"{r['canonical']:.5f}"]
        for n, _ in VARIANTS[1:]:
            cells.append(f"{r[n]['K']:.5f} ({r[n]['dev_pct']:+.2f}%)")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("### Table B - canonical series vs the locked SC table (NOTES §1.2)\n")
    lines.append("| phi | series K | locked table | dev |")
    lines.append("|---|---|---|---|")
    for r in rows_b:
        lines.append(
            f"| {r['phi']:.4f} | {r['series']:.4f} | {r['locked_table']:.4f} | {r['dev_pct']:+.2f}% |"
        )
    lines.append("")
    lines.append("### Table C - consequences at the GAP-LBM-SPHERE geometry\n")
    lines.append("| quantity | value |")
    lines.append("|---|---|")
    for key, val in rows_c.items():
        lines.append(f"| {key} | {val if isinstance(val, str) else f'{val:.6g}'} |")
    lines.append("")
    lines.append("### Table D - particle_suspension two-term truncation\n")
    lines.append("| phi | two-term K | canonical K | dev |")
    lines.append("|---|---|---|---|")
    for r in rows_d:
        lines.append(
            f"| {r['phi']:.0e} | {r['two_term']:.6f} | {r['canonical']:.6f} | {r['dev_pct']:+.3f}% |"
        )
    (OUT / "tables.md").write_text("\n".join(lines) + "\n")

    print(json.dumps(rows_c, indent=1))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())