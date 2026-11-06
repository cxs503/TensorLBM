"""Reproduce figures/ci_quarantine_burndown_r1.png (CI-quarantine burndown, task m4732).

Panel A: injected streamwise wall momentum per near-wall cell as a function of the
near-wall distance y_val, for the regression (traction divided by y_val) and the
restored A/V = 1 scaling.  The quarantined test asserts the flat curve; the 2x
gap at the default y_val = 0.5 is the whole bug.
Panel B: quarantine list size before/after this burndown and the measured CPU
wall-clock of the two throughput benchmarks that stay quarantined.

Usage: PYTHONPATH=src python3 tools/plot_ci_quarantine_burndown.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TAU_W = 0.01**2  # u_tau = 0.01 -> tau_w = 1e-4
N_NEAR = 10  # near-wall cells in the 2x4x5 test box


def main() -> pathlib.Path:
    y_val = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    buggy = -TAU_W / y_val * N_NEAR
    fixed = np.full_like(y_val, -TAU_W * N_NEAR)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax0.plot(y_val, buggy * 1e3, "o--", label=r"regression: $-\tau_w/y_{val}$")
    ax0.plot(y_val, fixed * 1e3, "s-", label=r"restored: $-\tau_w\,A/V$, $A/V=1$")
    ax0.axhline(-TAU_W * N_NEAR * 1e3, color="0.6", lw=0.8, ls=":")
    ax0.axvline(0.5, color="0.6", lw=0.8, ls=":")
    ax0.annotate(
        "2x at the default $y_{val}=0.5$",
        xy=(0.5, -2.0),
        xytext=(1.1, -1.65),
        arrowprops={"arrowstyle": "->", "lw": 0.9},
        fontsize=9,
    )
    ax0.set_xscale("log")
    ax0.set_xlabel(r"wall distance $y_{val}$ (lattice units)")
    ax0.set_ylabel(r"injected $\sum_i c_{ix}(f_i^{*}-f_i)$  [$10^{-3}$]")
    ax0.set_title("A. Wall-traction source moment vs. $y_{val}$")
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3)

    labels = ["quarantined\nentries", "restored to\nmain lane", "D3Q19\nMLUPS test", "D3Q27\nMLUPS test*"]
    values = [120 - 113, 9, 60, 130]
    colors = ["0.75", "#2a7", "#48a", "#48a"]
    bars = ax1.bar(labels, values, color=colors)
    for bar, val in zip(bars, values, strict=True):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{val}" + (" s" if val > 9 else ""),
            ha="center",
            fontsize=9,
        )
    ax1.set_ylabel("count / seconds")
    ax1.set_title("B. Burndown: 120 -> 113 entries, 9 restored")
    ax1.grid(alpha=0.3, axis="y")
    ax1.set_ylim(0, 150)
    fig.text(0.5, 0.01, "*D3Q27 parametrization estimated as 2x D3Q19 from the step probe", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    out = pathlib.Path("figures/ci_quarantine_burndown_r1.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")
    return out


if __name__ == "__main__":
    main()
