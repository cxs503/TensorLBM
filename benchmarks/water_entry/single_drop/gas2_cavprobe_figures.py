"""DE-X-GAS2-CAVPROBE-R1 figures: p_cav(t) vs constant-pressure closure & Rayleigh-Plesset line."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ctrl = json.loads(Path(sys.argv[1]).read_text())
    arm = json.loads(Path(sys.argv[2]).read_text())
    outdir = Path(sys.argv[3])
    outdir.mkdir(parents=True, exist_ok=True)

    cp = arm["cav_probe"]
    data = cp["data"]
    p_ref = cp["vapor_pressure_lattice"]
    t = [d["time_ms"] for d in data]
    pcav = [d["p_cav_lattice"] for d in data]
    prp = [d["p_rp_lattice"] for d in data]
    pk_i = max(range(len(data)), key=lambda i: data[i]["depth_over_D"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, pcav, "b-", lw=1.6, label="measured p_cav (cavity bottom, lattice units)")
    ax.axhline(p_ref, color="g", ls="--", lw=1.4, label="constant-pressure closure p_ref = 1.0")
    ax.plot(t, prp, "r-", lw=1.4, label="Rayleigh-Plesset reference (closed bubble)")
    ax.axvline(data[pk_i]["time_ms"], color="k", ls=":", lw=1)
    ax.text(data[pk_i]["time_ms"] + 0.1, ax.get_ylim()[1] * 0.98, "peak depth", rotation=90, fontsize=8)
    ax.set_xlabel("absolute time (ms)")
    ax.set_ylabel("pressure (lattice units)")
    ax.set_title("Cavity-bottom pressure vs two reference lines (cells=8, ctrl flags)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "gas2_cavprobe_p_cav.png", dpi=140)

    # deviation panel
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    dev = [(d["p_cav_lattice"] - p_ref) / p_ref * 100 for d in data]
    ax2.plot(t, dev, "b-", lw=1.2)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.axhline(5, color="g", ls="--", lw=0.8)
    ax2.axhline(-5, color="g", ls="--", lw=0.8)
    ax2.axhline(15, color="r", ls="--", lw=0.8)
    ax2.axhline(-15, color="r", ls="--", lw=0.8)
    ax2.axvline(data[pk_i]["time_ms"], color="k", ls=":", lw=1)
    ax2.set_xlabel("absolute time (ms)")
    ax2.set_ylabel("(p_cav - p_ref)/p_ref (%)")
    ax2.set_title("Growth-phase check: green = ±5% predeclared band, red = ±15% falsify band")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(outdir / "gas2_cavprobe_dev.png", dpi=140)

    # depth context
    h = ctrl["history"]
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot([r["time_ms"] for r in h], [r["depth_over_D"] for r in h], "b-", lw=1.2)
    ax3.axhline(1.857434, color="r", ls="--", lw=1.2, label="experiment 1.857434 D @ 10.394 ms")
    ax3.set_xlabel("absolute time (ms)")
    ax3.set_ylabel("open cavity depth (D)")
    ax3.set_title("ctrl depth trajectory (anchor 1.4352 D @ 11.091 ms)")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(outdir / "gas2_cavprobe_depth.png", dpi=140)
    print("figures ->", outdir)


if __name__ == "__main__":
    main()
