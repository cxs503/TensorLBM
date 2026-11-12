#!/usr/bin/env python3
"""Figure for the CASE-LBM-SQUARE-CYL-NORBERG remediation (audit m16183).

Two panels, both pure re-plots of the committed archive (no simulation):

  left  — measured Cd/St per grid against the locked reference bands, plus the
          diagnostic ``wet`` force column at D=64 (the column that would FAIL
          the 3 % bar and that v1 of the case left unnamed).
  right — error vs the locked reference per grid with the 3 % tolerance line,
          showing monotone convergence and the force-column spread.

Usage: ``python make_remediation_figure.py [OUT_PNG]``
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refspec as R  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fig_square_cylinder_remediation.png")
    cases = R.load_archive(Path(__file__).resolve().parent)
    ds = sorted(cases)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6), dpi=150)

    # ---- left: values vs locked bands -----------------------------------
    cd = [cases[d]["primary_cd"] for d in ds]
    st = [cases[d]["primary_st"] for d in ds]
    ax1.plot(ds, cd, "o-", color="#1f4e79", lw=1.8, label="Cd (primary column 'pair'/summary)")
    ax1.axhline(R.REF_CD, color="#1f4e79", ls="--", lw=1.2, label=f"Cd_ref = {R.REF_CD}")
    ax1.fill_between([ds[0] - 4, ds[-1] + 4], *R.CD_BAND, color="#1f4e79", alpha=0.12, label="Cd band")
    ax1.set_xlabel("grid resolution D (cylinder diameter in lattice cells)")
    ax1.set_ylabel("Cd", color="#1f4e79")
    ax1.tick_params(axis="y", labelcolor="#1f4e79")
    ax1.set_xlim(ds[0] - 4, ds[-1] + 4)

    ax1b = ax1.twinx()
    ax1b.plot(ds, st, "s-", color="#a63a00", lw=1.8, label="St (primary column)")
    ax1b.axhline(R.REF_ST, color="#a63a00", ls="--", lw=1.2, label=f"St_ref = {R.REF_ST}")
    ax1b.fill_between([ds[0] - 4, ds[-1] + 4], *R.ST_BAND, color="#a63a00", alpha=0.12, label="St band")
    ax1b.set_ylabel("St", color="#a63a00")
    ax1b.tick_params(axis="y", labelcolor="#a63a00")

    if 64 in cases and cases[64]["schema"] == "triple":
        wet = cases[64]["columns"]["wet"]
        ax1.plot([64], [wet["cd"]], "x", ms=10, mew=2.4, color="crimson",
                 label=f"wet column D64 (diagnostic): Cd={wet['cd']:.4f}")
        ax1.annotate(f"wet Cd {R.rel_err_pct(wet['cd'], R.REF_CD):+.1f}% -> would FAIL 3%",
                     xy=(64, wet["cd"]), xytext=(44, 1.575), color="crimson", fontsize=8.5,
                     arrowprops={"arrowstyle": "->", "color": "crimson", "lw": 1.0})

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7.4, loc="lower right", framealpha=0.9)
    ax1.set_title("CASE-LBM-SQUARE-CYL-NORBERG — values vs locked 2D free-stream cluster")

    # ---- right: errors vs tolerance -------------------------------------
    ecd = [R.rel_err_pct(cases[d]["primary_cd"], R.REF_CD) for d in ds]
    est = [R.rel_err_pct(cases[d]["primary_st"], R.REF_ST) for d in ds]
    w = 3.0
    ax2.bar([d - w for d in ds], ecd, width=2 * w, color="#1f4e79", label="Cd err %")
    ax2.bar([d + w for d in ds], est, width=2 * w, color="#a63a00", label="St err %")
    ax2.axhline(R.TOL_PCT, color="k", ls="--", lw=1.2, label=f"tolerance {R.TOL_PCT}%")
    ax2.axhline(-R.TOL_PCT, color="k", ls="--", lw=1.2)
    for d, e1, e2 in zip(ds, ecd, est):
        ax2.text(d - w, e1 + 0.05, f"{e1:+.2f}", ha="center", fontsize=7.6, color="#1f4e79")
        ax2.text(d + w, e2 + 0.05, f"{e2:+.2f}", ha="center", fontsize=7.6, color="#a63a00")
    ax2.set_ylim(0, 3.4)
    ax2.set_xticks(ds)
    ax2.set_xlabel("grid resolution D")
    ax2.set_ylabel("relative error vs locked reference (%)")
    ax2.set_title("Errors vs the 3% acceptance bar (locked convention, single source)")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()