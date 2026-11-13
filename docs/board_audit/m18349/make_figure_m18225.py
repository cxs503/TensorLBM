#!/usr/bin/env python3
"""m18349 审计图：把 benchmarks/eos_maxwell/RESULTS.md §2 的数值表程序化解析后作图。

零手抄：所有数字来自 RESULTS.md 的 markdown 表（该表本身由 m8675 的
plot_figures.py 从 out/results.json 机生成）。图只陈述"本锚点既有能力现状"，
不构成本单 m18225 的数值证据（m18225 零运行）。
"""
import json
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WT = pathlib.Path("/root/tp-agent/work/m18225")
RT = WT / "benchmarks/eos_maxwell/RESULTS.md"


def num(s):
    s = s.strip().replace("*", "")
    if s in ("—", "", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_tier_table(text):
    rows = []
    for line in text.splitlines():
        if not line.startswith("| tier_ratio"):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        m = re.match(r"tier_ratio(\d+)_(\w+)_L(\d+)R(\d+)", c[0])
        if not m:
            continue
        rows.append({
            "target": float(m.group(1)), "forcing": m.group(2),
            "stable": c[2] == "True", "measured": num(c[5]), "maxwell": num(c[6]),
            "ratio_err": num(c[7]), "ma": num(c[8]), "verdict": c[11].strip("*"),
        })
    return rows


def main():
    text = RT.read_text(encoding="utf-8")
    rows = parse_tier_table(text)
    out = pathlib.Path(__file__).with_name("fig_m18225_anchor_status.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"edm": "tab:blue", "guo": "tab:orange"}
    for ax, key, label, gate in (
        (axes[0], "ratio_err", "density-ratio error vs Maxwell lock  (gate 15%)", 0.15),
        (axes[1], "ma", "spurious max|u|/cs  (gate 0.10)", 0.10),
    ):
        for forcing in ("edm", "guo"):
            xs, ys, ok = [], [], []
            for r in rows:
                if r["forcing"] != forcing:
                    continue
                xs.append(r["target"])
                ys.append(r[key] if r[key] is not None else float("nan"))
                ok.append(r["stable"])
            ax.plot(xs, ys, "o-", color=colors[forcing], label=f"{forcing} (stable run)")
            for x, y, s in zip(xs, ys, ok):
                if not s:  # 塌缩档：无测量值，画空心标记置底
                    ax.plot([x], [0.0], "x", color=colors[forcing], ms=9)
        ax.axhline(gate, color="crimson", ls="--", lw=1.2, label=f"predeclared gate {gate}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("target density ratio")
        ax.set_ylabel(label)
        ax.set_xticks([10, 100, 1000])
        ax.set_xticklabels(["10", "100", "1000"])
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    axes[0].set_title("m8675 pre-existing campaign: measured/Maxwell ratio error")
    axes[1].set_title("m8675 pre-existing campaign: spurious currents (x = collapsed)")
    fig.suptitle("EOS-Maxwell high-density-ratio anchor: status quo (NOT m18225 output; "
                 "m18225 ran nothing)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(json.dumps({"figure": str(out), "rows": rows,
                      "provenance": "RESULTS.md §2 parsed programmatically"}, indent=1))


if __name__ == "__main__":
    main()