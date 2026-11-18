"""Compare numerical gauges at common times in the free-slip rectangular case."""

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.pending.dam_break_mm_channel.score import lookup


def read_wa(path: Path, step_dt: float) -> np.ndarray:
    rows = np.loadtxt(path, ndmin=2)
    # Upstream advances a step before emitting an observation labelled t.
    return np.column_stack(((rows[:, 0] + 1) * step_dt, rows[:, 2:4]))


def read_tensor(path: Path) -> np.ndarray:
    rows = np.genfromtxt(path, delimiter=",", skip_header=1, ndmin=2)
    return rows[np.isfinite(rows).all(axis=1), 1:4]


def comparison(curves: dict[str, np.ndarray]) -> dict:
    checkpoints = (0.5, 1.0, 2.0, 3.0, 4.0, 4.5, 6.0, 8.0, 9.0)
    return {
        name: {
            "last_T": float(points[-1, 0]),
            "checkpoints": [
                {
                    "T": t,
                    "Z": lookup(t, points[:, 0], points[:, 1]),
                    "H": lookup(t, points[:, 0], points[:, 2]),
                }
                for t in checkpoints
            ],
        }
        for name, points in curves.items()
    }


def plot(curves, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "tensor_bgk_n12": ("TensorLBM BGK N=12", "#1d6493", "--"),
        "tensor_mrt_n12": ("TensorLBM MRT N=12", "#1d6493", "-"),
        "walberla_n12": ("waLBerla matched N=12", "#c56724", "-"),
        "walberla_paper_n50": ("waLBerla paper N=50 (LES)", "#c56724", ":"),
        "basilisk_n16": ("Basilisk N=16", "#367750", "--"),
        "basilisk_n32": ("Basilisk N=32", "#367750", "-"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4), layout="constrained")
    for j, ax in enumerate(axes, start=1):
        for name, (label, color, style) in labels.items():
            rows = curves[name]
            rows = rows[rows[:, 0] <= 7.0]
            ax.plot(rows[:, 0], rows[:, j], label=label, color=color, ls=style)
        ax.set(xlabel="Release time T", ylabel=("Front Z" if j == 1 else "Crest H"), xlim=(0, 7))
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.savefig(path, dpi=145)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--tensor", required=True, type=Path)
    parser.add_argument("--tensor-mrt", required=True, type=Path)
    parser.add_argument("--wa12", required=True, type=Path)
    parser.add_argument("--wa50", required=True, type=Path)
    parser.add_argument("--wa-paper", required=True, type=Path)
    parser.add_argument("--basilisk16", required=True, type=Path)
    parser.add_argument("--basilisk32", required=True, type=Path)
    args = parser.parse_args()
    curves = {
        "tensor_bgk_n12": read_tensor(args.tensor),
        "tensor_mrt_n12": read_tensor(args.tensor_mrt),
        "walberla_n12": read_wa(args.wa12, 0.06 / 12),
        "walberla_n50": read_wa(args.wa50, 0.06 / 50),
    }
    # Native paper input uses tau=1/1.9995 and its supplied Ga at N=50.
    native_g = 1831123817 * ((1 / 1.9995 - 0.5) / 3) ** 2 / 50**3
    curves["walberla_paper_n50"] = read_wa(args.wa_paper, (2 * native_g / 50) ** 0.5)
    bas = {
        n: np.loadtxt(file, delimiter=",", skiprows=1, ndmin=2)
        for n, file in ((16, args.basilisk16), (32, args.basilisk32))
    }
    for n, row in bas.items():
        if not np.isfinite(row).all():
            raise ValueError(f"nonfinite Basilisk N={n} observations")
        curves[f"basilisk_n{n}"] = row[:, :3]
    result = comparison(curves)
    for n, row in bas.items():
        result[f"basilisk_n{n}"]["max_volume_drift_fraction"] = float(
            np.max(np.abs(row[:, 3] / row[0, 3] - 1.0))
        )
    print(json.dumps(result, indent=2))
    if args.plot:
        plot(curves, args.plot)


if __name__ == "__main__":
    main()
