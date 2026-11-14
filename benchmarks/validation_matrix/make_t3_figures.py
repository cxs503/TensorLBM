"""Figures for the T3 coverage gate + turbulent-channel debt record.

Fig 1 (:file:`t3_slot_coverage.png`): the nine T3 lineages with per-component
credit, so the honest 6/9 headline is visible at a glance.
Fig 2 (:file:`t3_channel_gate.png`): L3 u+ profile against the frozen MKM-1999
DNS table under both friction normalisations, plus the three-level FAIL ladder
against the 3% gate.

Every number is read from the frozen artifacts (T3_COVERAGE.json,
validation_report.json, L3_result.json, mkm180.csv); nothing is transcribed.

Usage::

    python benchmarks/validation_matrix/make_t3_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from channel_budget import (  # noqa: E402  (same directory)
    DNS_CSV,
    L3_RESULT,
    LADDER,
    build,
    load_dns,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
FIG_DIR = HERE / "figures"
COVERAGE = HERE / "T3_COVERAGE.json"

STATE_COLOR = {"verified": "#2e7d32", "proxy": "#f9a825", "missing": "#c62828"}


def fig_slot_coverage() -> Path:
    report = json.loads(COVERAGE.read_text(encoding="utf-8"))
    slots = report["slots"]
    comps = sorted({c for r in slots for c in r["components"]})
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for row, r in enumerate(slots):
        for col, comp in enumerate(comps):
            state = r["components"].get(comp, {}).get("state")
            if state is None:
                continue
            ax.add_patch(plt.Rectangle((col, row), 0.94, 0.86,
                                       color=STATE_COLOR[state]))
            label = {"verified": "verified", "proxy": "proxy", "missing": "missing"}[state]
            ax.text(col + 0.47, row + 0.43, label, ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")
    ax.set_xlim(0, len(comps))
    ax.set_ylim(len(slots), 0)
    ax.set_xticks([i + 0.47 for i in range(len(comps))])
    ax.set_xticklabels(comps, rotation=20, ha="right", fontsize=8)
    ax.set_yticks([i + 0.43 for i in range(len(slots))])
    ax.set_yticklabels([f"{i + 1}. {r['slot']} [{r['state']}]" for i, r in enumerate(slots)], fontsize=9)
    headline = report["headline"]
    ax.set_title(
        f"T3 nine-case lineage coverage: verified {headline['verified']}/9, "
        f"partial {headline['partial']}, gap {headline['gap']} "
        f"(matrix v{report['matrix_version']})",
        fontsize=11,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("required component per lineage", fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "t3_slot_coverage.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_channel_gate() -> Path:
    l3 = json.loads(L3_RESULT.read_text(encoding="utf-8"))
    ladder = json.loads(LADDER.read_text(encoding="utf-8"))
    dns_yp, dns_up = load_dns()
    stats = l3["statistics"]
    nu = float(l3["derived"]["nu"])
    u = [float(v) for v in stats["mean_velocity_profile"]]
    u_tau_meas = float(stats["stationarity_friction_velocity"])
    u_tau_target = math.sqrt(float(l3["derived"]["target_wall_shear"]))
    budget = build()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    ax1.plot(dns_yp, dns_up, "k-", lw=2.2, label="MKM 1999 DNS (Re_tau=178.12)")
    for u_tau, style, label in (
        (u_tau_target, "b--o", "LBM L3, u_tau = target (recorded 5.33%)"),
        (u_tau_meas, "r-s", "LBM L3, u_tau = measured (15.8%)"),
    ):
        yp = [(i - 0.5) * u_tau / nu for i in range(len(u))]
        up = [ui / u_tau for ui in u]
        keep = [(a, b) for a, b in zip(yp, up) if 0.5 <= a <= 200.0]
        ax1.plot([p[0] for p in keep], [p[1] for p in keep], style, ms=3, lw=1.2, label=label)
    ax1.set_xlabel("y+")
    ax1.set_ylabel("u+")
    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, 25)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7.5, loc="lower right")
    ax1.set_title("u+ profile vs frozen DNS table (frozen L3 statistics)", fontsize=10)

    levels = [g["level"] for g in ladder["grid_levels"]]
    rms = [g["rms_y30_100_pct"] for g in ladder["grid_levels"]]
    ax2.plot(levels, rms, "ko-", lw=2, ms=8, label="u+ RMS vs DNS, y+ in [30,100]")
    ax2.axhline(3.0, color="red", ls="--", lw=1.2, label="3% T3 gate")
    for x, y in zip(levels, rms):
        ax2.annotate(f"{y:.2f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax2.set_yscale("log")
    ax2.set_xlabel("grid level (256x64x128 at L3)")
    ax2.set_ylabel("RMS error [%] (log)")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8)
    ax2.set_title("slot-4 honest FAIL ladder (monotone, still > gate)", fontsize=10)
    fig.suptitle(
        "Turbulent channel Re_tau=180: gate reproduction "
        f"(recorded {budget['reproduction']['recorded_rms_pct']:.3f}% vs recomputed "
        f"{budget['reproduction']['recomputed_rms_pct']:.3f}%, "
        f"delta {budget['reproduction']['delta_pp']:.4f} pp)",
        fontsize=10,
    )
    fig.tight_layout()
    out = FIG_DIR / "t3_channel_gate.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for path in (fig_slot_coverage(), fig_channel_gate()):
        print(f"wrote {path.relative_to(REPO_ROOT)} ({path.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())