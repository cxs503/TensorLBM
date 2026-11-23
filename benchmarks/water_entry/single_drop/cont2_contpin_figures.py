#!/usr/bin/env python3
"""DE-X-CONT2-CONT-PIN: depth-time figure for the card and the progress report.

Reads runs/cont2/{contpin,ctrl}.json and renders the cavity-depth trajectory
of both arms against the experimental peak (1.857434 D @ 10.394 ms).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_PEAK_D = 1.857434
EXP_PEAK_MS = 10.394


def series(p: Path):
    r = json.loads(p.read_text())
    h = r.get("history") or []
    pts = [
        (row["time_ms"], row["depth_over_D"])
        for row in h
        if row.get("time_ms") is not None and row.get("depth_over_D") is not None
    ]
    t = [x[0] for x in pts]
    d = [x[1] for x in pts]
    return t, d, r


def main() -> int:
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/cont2")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else base / "fig_depth_time.png"
    tc, dc, rc = series(base / "ctrl.json")
    ta, da, ra = series(base / "contpin.json")
    t1, d1, r1 = series(base / "contpin01.json")
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    ax.plot(tc, dc, color="tab:blue", lw=1.6, label="ctrl (K_pin=0)")
    ax.plot(ta, da, color="tab:red", lw=1.6, ls="--", label="CONT-PIN (K_pin=0.3, died 3.6ms)")
    ax.plot(t1, d1, color="tab:green", lw=1.6, label="CONT-PIN (K_pin=0.1)")
    ax.axhline(EXP_PEAK_D, color="k", ls="--", lw=1.0, label=f"exp peak {EXP_PEAK_D} D")
    ax.axvline(EXP_PEAK_MS, color="k", ls=":", lw=1.0, label=f"exp peak time {EXP_PEAK_MS} ms")
    pk_c = max(dc) if dc else float("nan")
    pk_a = max(da) if da else float("nan")
    if dc:
        ax.plot([t for t, d in zip(tc, dc) if d == pk_c], [pk_c], "o", color="tab:blue")
    if da:
        ax.plot([t for t, d in zip(ta, da) if d == pk_a], [pk_a], "o", color="tab:red")
    pk_1 = max(d1) if d1 else float("nan")
    if d1:
        ax.plot([t for t, d in zip(t1, d1) if d == pk_1], [pk_1], "o", color="tab:green")
    ax.set_xlabel("time after release (ms)")
    ax.set_ylabel("cavity depth (D)")
    ax.set_title("DE-X-CONT2-CONT-PIN: cavity-mouth pinning, cells=8 full window")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
