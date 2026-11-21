"""RELX1-PHASETAU-R2 figures: cavity depth vs time for the dose matrix.

Programmatic: reads runs/<arm>/result.json, overlays the Santini case-3
experimental markers (fig5_experiment.csv), saves PNGs.  No hand-copied data.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_DEPTH = 1.857434
ARMS = ["pt_ctrl", "pt_p05", "pt_p10", "pt_p20", "pt_m05"]
DOSES = {"pt_ctrl": 0.0, "pt_p05": 0.5, "pt_p10": 1.0, "pt_p20": 2.0, "pt_m05": -0.5}
COLORS = {"pt_ctrl": "black", "pt_p05": "tab:blue", "pt_p10": "tab:red",
          "pt_p20": "tab:green", "pt_m05": "tab:orange"}


def main():
    runs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else runs_dir / "phasetau_depth_time.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    # experimental markers
    csv = Path(__file__).parent / "fig5_experiment.csv"
    if csv.exists():
        data = np.genfromtxt(csv, delimiter=",", names=True)
        t_exp = np.array([float(v) for v in data["time_ms"]])
        d_exp = np.array([float(v) for v in data["depth_over_D"]])
    else:
        t_exp = d_exp = None

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    if t_exp is not None:
        ax.plot(t_exp, d_exp, "k*", ms=7, label="Santini case 3 (exp.)")
    for arm in ARMS:
        p = runs_dir / arm / "result.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        h = r.get("history") or []
        t = [row["time_ms"] for row in h]
        d = [row.get("depth_over_D", float("nan")) for row in h]
        ax.plot(t, d, color=COLORS[arm], lw=1.2,
                label=f"{arm} (dose {DOSES[arm]:+.1f})")
        pk = max(h, key=lambda x: x.get("depth_over_D", -1))
        ax.plot([pk["time_ms"]], [pk["depth_over_D"]], "o", color=COLORS[arm], ms=5)
        ax.annotate(f"{pk['depth_over_D']:.4f}D", (pk["time_ms"], pk["depth_over_D"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8, color=COLORS[arm])
    ax.axhline(EXP_DEPTH, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("cavity depth (D)")
    ax.set_title("RELX1-PHASETAU-R2 dose matrix, cells=8, SDAA lane")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
