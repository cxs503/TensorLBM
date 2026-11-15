"""Audit figure for m21802 (board audit of m21672 / CASE-BACKSTEP-FVM-RE100, LBM arm).

Reads only committed artefacts of the audited branch (agent/m21672 @ 0fb4c2ebe)
plus the converged point that exists only on an unmerged branch (agent/m17034 @
f3a211d78), and plots them against the two in-repo reference tiers.
No number is typed in by hand: everything is loaded from the artefact JSONs.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AUDITED = Path("/srv/tensor/agent/work/m21672")
OUT = Path(__file__).resolve().parent / "fig_m21802_backstep_audit.png"


def _git_show(rev: str, path: str) -> dict:
    txt = subprocess.run(["git", "-C", str(AUDITED), "show", f"{rev}:{path}"],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(txt)


def main() -> int:
    partial = json.loads(
        (AUDITED / "artifacts/case_backward_step_armaly/h1_re100/result.partial.json").read_text()
    )
    hist = partial["history"]
    refs = json.loads((AUDITED / "benchmarks/case_pool/backward_step_armaly/refs.json").read_text())
    ref_exp = refs["adopted_numeric_reference"]["points"]["100"]["x1_over_s"]
    e = refs["cross_checks"][0]
    conv = _git_show("f3a211d78", "artifacts/case_backward_step_armaly/h1_re100/result.json")

    steps = [h["step"] for h in hist]
    x1 = [h["x1_over_s_instant"] for h in hist]
    umax = [h["max_speed"] for h in hist]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    ax.axhspan(e["band"][0], e["band"][1], color="tab:green", alpha=0.18,
               label=f"Erturk 2008 2D band {e['band'][0]}-{e['band'][1]}")
    ax.axhline(ref_exp, color="tab:blue", lw=1.6,
               label=f"Armaly 1983 anchor (in-repo tier SECONDARY) = {ref_exp}")
    ax.plot(steps, x1, "o-", color="tab:red", lw=1.8,
            label="m21672 cj3613 h1_re100 (20000/200000 steps, transient)")
    ax.errorbar([partial["steps_run"] * 1.9], [conv["result"]["x1_over_s"]],
                yerr=None, fmt="D", ms=9, color="black",
                label=f"converged point NOT in delivery (agent/m17034 @f3a211d78, "
                      f"{conv['result']['steps_run']} steps) = {conv['result']['x1_over_s']:.4f}")
    ax.annotate("delivery ends here\n(steady_at_step = null)",
                xy=(partial["steps_run"], x1[-1]), xytext=(0.45, 0.25),
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.set_xlabel("step")
    ax.set_ylabel("X1/s (reattachment / step height, instantaneous)")
    ax.set_title("Reattachment length vs step budget (Re=100, Armaly caliber)")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3)

    ax2.plot(steps, umax, "s-", color="tab:purple", lw=1.6)
    ax2.set_xlabel("step")
    ax2.set_ylabel("max fluid speed (lattice units)")
    ax2.set_title("max_speed diagnostic after the m21672 fix\n(fluid speed, not the 10-30 global L2 norm)")
    ax2.grid(alpha=0.3)

    fig.suptitle("m21802 board audit — m21672 CASE-BACKSTEP-FVM-RE100 (LBM): L3, benchmark referee absent", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print("wrote", OUT)
    print("partial x1 tail", x1[-1], "conv", conv["result"]["x1_over_s"],
          f"err_vs_exp {(conv['result']['x1_over_s']-ref_exp)/ref_exp*100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())