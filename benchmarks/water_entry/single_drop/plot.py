"""Plot the experimental points and the healthy, post-contact simulation window."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.water_entry.single_drop.score import HERE, read_run, reference


def plot(paths, target):
    case = json.loads((HERE / "protocol.json").read_text())
    t, h = reference(case)
    meta = case["reference_data"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.5), layout="constrained")
    series = []
    for path in paths:
        run = read_run(path)
        contact = run["contact_step"]
        if contact is None:
            continue
        rows = [r for r in run["history"] if r["step"] >= contact and "depth_over_D" in r]
        if not rows:
            continue
        time = np.array([r["time_ms"] for r in rows]) - rows[0]["time_ms"]
        depth = [r["depth_over_D"] for r in rows]
        model = "CAC" if run["solver"].startswith("CAC") else "Free surface"
        label = f"{model}, {run['cells_per_D']} cells/D"
        if model == "Free surface":
            label = (
                f"N={run['cells_per_D']}, width {run['span_over_D']:g}D, "
                f"Δt={1e6 * run['scales']['dt_s']:.1f} µs, "
                f"U₀={run['release_speed_m_s']:.2f} m/s"
            )
        series.append((time, depth, label))
    for ax in axes:
        ax.errorbar(
            t,
            h,
            yerr=meta["depth_readout_over_D"],
            xerr=meta["time_readout_ms"],
            fmt="o",
            ms=4,
            capsize=2,
            color="black",
            label="Experiment, Fig. 5",
            zorder=8,
        )
        for time, depth, label in series:
            ax.plot(time, depth, lw=1.4, label=label)
            ax.plot(time[-1], depth[-1], marker="x", color=ax.lines[-1].get_color(), ms=6)
        ax.set_xlabel("Time from first contact (ms)")
        ax.set_ylim(bottom=-0.04)
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(alpha=0.18)
    axes[0].set_xlim(-0.2, 18)
    axes[0].set_ylabel("Atmosphere-connected crater depth / D")
    axes[0].set_title("Published observation window", loc="left")
    axes[1].set_xlim(-0.04, 2.5)
    axes[1].set_ylim(-0.02, 0.9)
    axes[1].set_title("Early evolution", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: not labels[i].startswith("Experiment"))
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="outside lower center",
        ncols=3,
        frameon=False,
        fontsize=8.5,
    )
    fig.savefig(target, dpi=150)
    plt.close(fig)


def plot_planes(path, target):
    with np.load(path) as data:
        x, z = data["x_over_D"], data["z_over_D"]
        fill, velocity, times = data["fill"], data["uz_m_s"], data["time_ms"]
    fig, axes = plt.subplots(
        1,
        len(times),
        figsize=(3 * len(times), 3.8),
        layout="constrained",
        squeeze=False,
        sharey=True,
    )
    limit = max(float(np.abs(velocity[fill >= 0.5]).max()), 0.1)
    for ax, phase, w, t in zip(axes[0], fill, velocity, times):
        water = np.ma.masked_where(phase < 0.5, w)
        im = ax.pcolormesh(x, z, water, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="nearest")
        ax.axhline(0, color="0.6", lw=0.6, ls="--")
        ax.contour(x, z, phase, levels=[0.5], colors="black", linewidths=1.0)
        ax.set_aspect("equal")
        ax.set_ylim(-2.2, 1.3)
        ax.set_xlabel("x / D")
        ax.set_title(f"{t:.2f} ms")
    axes[0, 0].set_ylabel("z / D from initial surface")
    fig.colorbar(im, ax=axes[0].tolist(), label="Vertical liquid velocity (m/s)")
    fig.savefig(target, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--planes", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.planes is not None:
        plot_planes(args.planes, args.output)
    elif args.runs:
        plot(args.runs, args.output)
    else:
        parser.error("supply run files or --planes")
