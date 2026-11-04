"""Figure for docs/cylinder_drag_benchmark_r1.md — built from the registry + samples."""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REG = pathlib.Path("docs/cylinder_drag_registry.json")
CSV = pathlib.Path("artifacts/cylinder_drag/samples.csv")
OUT = pathlib.Path("docs/assets/cylinder_drag_benchmark.png")


def main() -> None:
    reg = json.loads(REG.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    if CSV.exists():
        import csv as csvmod
        rows: dict[str, list[tuple[float, float]]] = {}
        with CSV.open() as fh:
            for r in csvmod.DictReader(fh):
                rows.setdefault(r["run"], []).append((float(r["cd"]), float(r["cl"])))
        for name, seq in rows.items():
            steps = range(len(seq))
            ax.plot(steps, [c for c, _ in seq], lw=0.8, label=f"{name} Cd")
    ax.set_xlabel("step in statistics window")
    ax.set_ylabel("Cd")
    ax.set_title("Drag history (real LBM runs)")
    ax.legend(fontsize=7)

    ax = axes[1]
    runs = [r for r in reg["runs"]]
    xs = [r["derived"]["blockage"] for r in runs]
    ys = [r["derived"]["cd_mean"] for r in runs]
    ax.plot(xs, ys, "o-", label="measured Cd")
    ext = reg.get("extrapolation") or {}
    if "cd_b0" in ext:
        ax.axhline(ext["cd_b0"], ls="--", color="C1",
                   label=f"B->0 fit: {ext['cd_b0']:.3f}")
    rng = reg["anchors"]["unconfined_re100_cd"]["range"]
    ax.axhspan(rng[0], rng[1], color="C2", alpha=0.25,
               label=f"unconfined anchor {rng[0]:.2f}-{rng[1]:.2f}")
    ax.set_xlabel("blockage B = D/H")
    ax.set_ylabel("Cd")
    ax.set_title("Blockage trend vs literature anchor")
    ax.legend(fontsize=7)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
