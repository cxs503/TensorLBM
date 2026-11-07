#!/usr/bin/env python3
"""IN25-T6 -- emit every survey artifact from the package (no hand-typed numbers).

Products
--------
``docs/survey/in25_t6_lit_anchor_table.json``   anchor table for the T5 red team
``docs/survey/in25_t6_playbook.md``             five-measure PLAYBOOK entries
``artifacts/in25_t6/acceptance_report.json``    criteria numbers + product paths
``docs/survey/figs/*.png``                      three figures for the report

Usage
-----
    python scripts/in25_t6/emit_survey_artifacts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.external_survey import consistency as cons  # noqa: E402
from tensorlbm.external_survey import suboff_anchor as sub  # noqa: E402
from tensorlbm.external_survey.report import (  # noqa: E402
    write_acceptance_report,
    write_anchor_table,
    write_playbook,
)
from tensorlbm.external_survey.survey_doc import survey_markdown  # noqa: E402

FIGS = ROOT / "docs" / "survey" / "figs"


def figure_ct_ladder() -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    a1 = sub.aff1_ladder()
    a8 = sub.aff8_ladder()
    re1 = np.array([p.re for p in a1])
    ct1 = np.array([p.ct for p in a1])
    re8 = np.array([p.re for p in a8])
    ct8 = np.array([p.ct for p in a8])
    re_grid = np.logspace(np.log10(1.0e7), np.log10(4.0e7), 120)

    k1 = sub.form_factor_mean(1)
    k8 = sub.form_factor_mean(8)
    ittc = np.array([sub.ittc57_cf(r) for r in re_grid])
    lo, hi = sub.aff8_ct_bracket()["interval"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    ax.fill_between(re8, lo, hi, color="tab:orange", alpha=0.18,
                    label=f"AFF-8 CT interval (S_wet {sub.S_WET_AFF8_BRACKET[0]:.2f}-"
                          f"{sub.S_WET_AFF8_BRACKET[1]:.2f} m$^2$)")
    ax.plot(re1, ct1, "o-", color="tab:blue", lw=1.4, ms=5, label="AFF-1 EFD ladder (Table 14)")
    ax.plot(re8, ct8, "s-", color="tab:orange", lw=1.4, ms=5,
            label="AFF-8 EFD ladder (Table 14, mid-bracket S)")
    ax.plot(re_grid, ittc, "--", color="0.35", lw=1.2,
            label="ITTC-1957 $C_F$ (friction only)")
    ax.plot(re_grid, ittc * (1 + k1), ":", color="tab:blue", lw=1.6,
            label=f"ITTC-1957 $\\times(1+k_1)$, $k_1$={k1:.3f} (AFF-1 fit)")
    ax.plot(re_grid, ittc * (1 + k8), ":", color="tab:orange", lw=1.6,
            label=f"ITTC-1957 $\\times(1+k_8)$, $k_8$={k8:.3f} (AFF-8 fit)")
    a1p = min(a1, key=lambda p: abs(p.re - sub.PUBLISHED_AFF1_RE))
    ax.plot([a1p.re], [sub.PUBLISHED_AFF1_CT], "D", color="tab:green", ms=7,
            label=f"published AFF-1 CT={sub.PUBLISHED_AFF1_CT} (reproduced "
                  f"{100*(a1p.ct/sub.PUBLISHED_AFF1_CT-1):+.1f} %)")
    ax.plot([a8[0].re], [sub.PUBLISHED_AFF8_CT], "X", color="crimson", ms=9,
            label="widely quoted AFF-8 CT=0.004 (NOT reproducible)")
    ax.annotate("0.004 would need $S_{wet}$=5.50 m$^2$\n< AFF-1 area -> unphysical",
                xy=(a8[0].re, sub.PUBLISHED_AFF8_CT), xytext=(1.5e7, 0.00392),
                fontsize=7.5, color="crimson",
                arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8))
    ax.set_xscale("log")
    ax.set_xlabel("Re (based on hull length L=4.356 m, $\\nu$=1.139e-6 m$^2$/s)")
    ax.set_ylabel("$C_T$ (wetted-area normalised)")
    ax.set_title("IN25-T6 seat B: SUBOFF resistance anchor lane vs ITTC-1957")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6.8, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    out = FIGS / "in25_t6_ct_ladder.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_friction_lines() -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ag = cons.friction_line_agreement()
    re = np.logspace(6, 9, 160)
    diff = np.array([100 * (sub.schoenherr_cf(r) - sub.ittc57_cf(r)) / sub.ittc57_cf(r) for r in re])

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=150)
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.axvspan(1e6, 1e7, color="0.85", alpha=0.6, label="Re < 1e7 (below SUBOFF band)")
    ax.axhspan(-3, 3, color="tab:green", alpha=0.12, label="+/-3 % gate (Re $\\geq$ 1e7)")
    ax.plot(re, diff, color="tab:purple", lw=1.6, label="(Schoenherr - ITTC-1957)/ITTC-1957")
    ax.axvline(sub.aff1_ladder()[0].re, color="tab:blue", ls="--", lw=1.0,
               label=f"lowest SUBOFF ladder Re={sub.aff1_ladder()[0].re:.2e}")
    ax.axvline(sub.aff1_ladder()[-1].re, color="tab:blue", ls=":", lw=1.0,
               label=f"highest SUBOFF ladder Re={sub.aff1_ladder()[-1].re:.2e}")
    ax.set_xscale("log")
    ax.set_xlabel("Re")
    ax.set_ylabel("relative difference [%]")
    ax.set_title(
        "Friction-line cross-consistency: max "
        f"{ag.max_rel_diff_pct_model_band:.2f} % for Re$\\geq$1e7, "
        f"{ag.max_rel_diff_pct:.2f} % over Re 1e6-1e9"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = FIGS / "in25_t6_friction_lines.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_sensitivity() -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sens = sub.convention_sensitivity()
    pub = sub.PUBLISHED_AFF8_CT
    base = sens["ct_base"]
    rows = [
        (f"base (S={sens['s_wet_used']:.3f} m2, rho=999)", base),
        (f"S_wet={sub.S_WET_AFF8_BRACKET[0]:.2f} m2", base * (1 + sens["sensitivity_s_lo_pct"] / 100)),
        (f"S_wet={sub.S_WET_AFF8_BRACKET[1]:.2f} m2", base * (1 + sens["sensitivity_s_hi_pct"] / 100)),
        ("rho=1025.9 kg/m3 (sea water)", base * (1 + sens["sensitivity_rho_sea_water_pct"] / 100)),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=150)
    bars = ax.barh(labels, vals, color="tab:orange", alpha=0.75, height=0.55)
    ax.axvline(pub, color="crimson", ls="--", lw=1.4,
               label=f"widely quoted AFF-8 CT={pub}")
    for b, v in zip(bars, vals):
        ax.text(v, b.get_y() + b.get_height() / 2,
                f" {v:.5f} ({100*(v/pub-1):+.1f} % vs 0.004)", va="center", fontsize=7.5)
    ax.set_xlim(0, 0.0046)
    ax.set_xlabel("$C_T$ at Re=$%.2e$ (AFF-8, Table 14 forces)" % sens["re"])
    ax.set_title("Convention sensitivity: wetted area +-5.8 %, density +-2.6 %; "
                 "0.004 unreachable")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = FIGS / "in25_t6_convention_sensitivity.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    figs = [figure_ct_ladder(), figure_friction_lines(), figure_sensitivity()]
    table = write_anchor_table(ROOT / "docs" / "survey" / "in25_t6_lit_anchor_table.json")
    book = write_playbook(ROOT / "docs" / "survey" / "in25_t6_playbook.md")
    pytest_summary = None
    summary_path = ROOT / "artifacts" / "in25_t6" / "pytest_summary.json"
    if summary_path.exists():
        pytest_summary = json.loads(summary_path.read_text())
    doc = ROOT / "docs" / "survey" / "IN25_T6_external_mature_tech_survey.md"
    doc.write_text(survey_markdown() + "\n")
    rep = write_acceptance_report(
        ROOT / "artifacts" / "in25_t6" / "acceptance_report.json",
        pytest_summary=pytest_summary,
    )
    for p in (*figs, table, book, doc, rep):
        print(f"wrote {p.relative_to(ROOT)} ({p.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())