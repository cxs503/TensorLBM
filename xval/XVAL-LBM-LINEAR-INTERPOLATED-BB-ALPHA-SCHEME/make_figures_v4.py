"""Figures for verify_v4.py (numbers injected from artifacts_v4/results_*.json)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts_v4"
OUT = ART / "figures"


def load(part: str) -> list[dict]:
    return json.loads((ART / f"results_{part}.json").read_text())["rows"]


def fig_wall_position(sweep: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    deltas = sorted({r["delta_b"] for r in sweep})
    ax.plot(deltas, deltas, "k--", lw=1.0, label="ideal ($w_{bot}=\\delta$)")
    for clo, mk in (("halfway", "o"), ("bfl", "s"), ("cli", "^")):
        pts = sorted((r["delta_b"], r["w_bot"]) for r in sweep
                     if r["closure"] == clo and r["carrier"] == "bgk")
        if pts:
            ax.plot(*zip(*pts), mk + "-", ms=5, lw=1.2, label=f"{clo} (BGK)")
    ax.set_xlabel("sub-grid wall distance $\\delta$ [cell]")
    ax.set_ylabel("measured $w_{bot}$ [cell]")
    ax.set_title("production operator: wall position from parabola fit")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = OUT / "fig_wall_position.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_error_split(sweep: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for clo, mk in (("halfway", "o"), ("bfl", "s"), ("cli", "^")):
        pts = sorted((r["delta_b"], r["l2_rel"], r["l2_shape"]) for r in sweep
                     if r["closure"] == clo and r["carrier"] == "bgk")
        if not pts:
            continue
        d, full, shape = zip(*pts)
        ax.semilogy(d, full, mk + "-", ms=4, lw=1.0, alpha=0.55,
                    label=f"{clo}: raw $l_2$")
        ax.semilogy(d, shape, mk + ":", ms=4, lw=1.4,
                    label=f"{clo}: shape-only $l_2$")
    ax.axhline(0.15, color="r", ls="--", lw=1.0, label="owner 15 % line")
    ax.set_xlabel("sub-grid wall distance $\\delta$ [cell]")
    ax.set_ylabel("relative $l_2$ error")
    ax.set_title("residual split: amplitude (density) vs profile shape")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = OUT / "fig_error_split.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_init(init: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    labels, amps, rhos = [], [], []
    for r in sorted(init, key=lambda r: (r["delta_b"], r["closure"], r["init"])):
        labels.append(f"δ={r['delta_b']:.2f}\n{r['closure']}\n{r['init']}")
        amps.append(r["A_over_ref"])
        rhos.append(r["rho_fluid_mean"])
    xs = range(len(labels))
    ax.bar(xs, amps, color=["#4477aa" if "reflect" in l else
                            "#ee6677" if "rest" in l else "#ccbb44" for l in labels])
    ax.axhline(1.0, color="k", ls="--", lw=1.0)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("$A/A_{ref}$ (momentum amplitude)")
    ax.set_title("solid-slot initialization sets the mean density\n"
                 "(bar label colour: reflect / rest / zero)")
    for x, (a, rho) in enumerate(zip(amps, rhos)):
        ax.text(x, a + 0.001, f"{a:.4f}\nρ̄={rho:.4f}", ha="center", fontsize=5)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = OUT / "fig_init_sensitivity.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    for part, fn in (("sweep", fig_wall_position), ("sweep", fig_error_split), ("init", fig_init)):
        try:
            made.append(fn(load(part)))
        except FileNotFoundError:
            print(f"missing results_{part}.json")
    for p in made:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())