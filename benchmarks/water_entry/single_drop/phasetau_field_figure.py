"""RELX1-PHASETAU-R2 field figure: mid-plane fill at registered post-contact
times (from --planes npz), for the report.  Programmatic, no hand data.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    npz_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/pt_local/ctrl_planes.npz")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else npz_path.parent / "fill_midplane.png"
    d = np.load(npz_path)
    t_ms = d["time_ms"]
    fill = d["fill"]  # (n_times, nz, nx)
    x_over_D = d["x_over_D"]
    z_over_D = d["z_over_D"]

    # pick the sample closest to the experimental peak time 10.39446 ms
    idx = int(np.argmin(np.abs(t_ms - 10.39446)))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), dpi=140)
    for ax, i in zip(axes, (0, idx)):
        im = ax.pcolormesh(
            x_over_D, z_over_D, fill[i], shading="auto", cmap="viridis", vmin=0, vmax=1
        )
        ax.set_xlabel("x/D")
        ax.set_ylabel("z/D (up)")
        ax.set_title(f"mid-plane fill, t={t_ms[i]:.3f} ms post-contact")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, label="fill")
    fig.suptitle(
        "ctrl (dose=0) mid-plane liquid fill, cells=8 (x86 run, SDAA-bit-identical per TENS1)"
    )
    fig.tight_layout()
    fig.savefig(out)
    print(f"figure -> {out}  (times: {list(t_ms)})")


if __name__ == "__main__":
    main()
