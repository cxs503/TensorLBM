"""DE-X-MOM1-GUO_INT figures: cavity depth vs time, ctrl vs arms vs experiment."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
RUNS = HERE.parent.parent.parent / "runs" / "guo_int"


def load_history(path):
    run = json.loads(Path(path).read_text())
    h = run["history"]
    t0 = next(r["time_ms"] for r in h if r["step"] == run["contact_step"])
    sel = [r for r in h if r["step"] >= run["contact_step"]]
    t = np.array([r["time_ms"] - t0 for r in sel])
    d = np.array([r["depth_over_D"] for r in sel])
    return t, d, run


def main():
    arms = [
        ("ctrl.json", "ctrl (legacy fill-weighted Guo force)", "b", "-"),
        ("guo_int.json", "guo_int mass (mass-consistent, no-op)", "g", ":"),
        ("guo_int_full.json", "guo_int full (full force on 0<fill<1)", "r", "--"),
    ]
    exp = np.genfromtxt(HERE / "fig5_experiment.csv", delimiter=",", names=True)
    t_exp = np.array(exp["time_ms"])
    d_exp = np.array(exp["depth_over_D"])

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    ax.plot(t_exp, d_exp, "k^", ms=5, label="experiment (Santini case-3)")
    for fname, label, color, ls in arms:
        p = RUNS / fname
        if not p.exists():
            continue
        t, d, _ = load_history(p)
        ax.plot(t, d, color=color, ls=ls, lw=1.4, label=label)
        i = int(np.argmax(d))
        ax.plot(t[i], d[i], "o", color=color, ms=6)
        ax.annotate(
            f"{d[i]:.4f}D @ {t[i]:.3f}ms",
            (t[i], d[i]),
            textcoords="offset points",
            xytext=(8, -12),
            color=color,
        )
    i_exp = int(np.argmax(d_exp))
    ax.annotate(
        f"exp {d_exp[i_exp]:.4f}D @ {t_exp[i_exp]:.3f}ms",
        (t_exp[i_exp], d_exp[i_exp]),
        textcoords="offset points",
        xytext=(8, 8),
    )
    ax.set_xlabel("time after contact [ms]")
    ax.set_ylabel("cavity depth / D")
    ax.set_title("Single drop into deep pool, cells=8, full window (GUO_INT)")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(alpha=0.3)
    fig = ax.get_figure()
    out = RUNS / "fig_depth_time.png"
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
