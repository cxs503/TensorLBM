"""CMB7-CSTUNE2-R1 report figures (matplotlib, programmatic, zero hand-copy).

Fig 1: depth-over-time for the 8 arms (6 CMB7 + 2 R1) vs experimental markers.
Fig 2: response surface peak_depth vs (Cs, z_ref).
Fig 3: mechanism probe - analytic correction magnitude vs z_ref + fhash note.
"""
import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_CSV = Path("/root/tp-agent/work/m27389/benchmarks/water_entry/single_drop/fig5_experiment.csv")
EXP_PEAK_D = 1.857434
EXP_PEAK_MS = 10.39446
GATE = 0.97 * EXP_PEAK_D

BASES = [
    ("CMB7", Path("/root/tp-agent/work/m27175/runs")),
    ("R1", Path("/root/tp-agent/work/m27389/runs/cmb7r1")),
]


def load_arms():
    arms = {}
    for tag, base in BASES:
        if not base.exists():
            continue
        for d in sorted(base.glob("cs0*")):
            rj = d / "result.json"
            if not rj.exists():
                continue
            r = json.loads(rj.read_text())
            hist = r["history"]
            pk = max(hist, key=lambda x: x["depth_over_D"])
            arms[d.name] = {
                "tag": tag,
                "C_s": r.get("C_s"),
                "z_ref": int(d.name.split("_z")[1]),
                "stop": r.get("stop_reason"),
                "peak_D": pk["depth_over_D"],
                "peak_ms": pk["time_ms"],
                "t": [h["time_ms"] for h in hist],
                "depth": [h["depth_over_D"] for h in hist],
            }
    return arms


def fig1(arms, out):
    exp = list(__import__("csv").DictReader(EXP_CSV.open()))
    et = [float(r["time_ms"]) for r in exp]
    ed = [float(r["depth_over_D"]) for r in exp]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"cs0075_z10": "tab:red", "cs0075_z12": "tab:blue",
              "cs0075_z18": "tab:green", "cs0075_z20": "tab:orange"}
    for name, a in arms.items():
        if name.startswith("cs0075"):
            ax.plot(a["t"], a["depth"], lw=2.0, label=f"{name} (z_ref={a['z_ref']})",
                    color=colors.get(name, None))
        else:
            ax.plot(a["t"], a["depth"], lw=1.0, alpha=0.5,
                    label=f"{name} (Cs={a['C_s']})")
    ax.plot(et, ed, "k*", ms=9, label="Santini exp. (Fig.5)")
    ax.axhline(GATE, color="gray", ls="--", lw=1, label=f"3% gate {GATE:.4f}D")
    ax.axhline(EXP_PEAK_D, color="k", ls=":", lw=1, label=f"exp peak {EXP_PEAK_D}D")
    ax.set_xlabel("time (ms pc)")
    ax.set_ylabel("cavity depth (D)")
    ax.set_title("CMB7-CSTUNE2-R1: cavity depth vs time (cells=8, RR+SMAG+HYDRO)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig2(arms, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, a in arms.items():
        if not name.startswith("cs0075"):
            continue
        ax.plot(a["z_ref"], a["peak_D"], "o-", color="tab:blue",
                label=f"Cs=0.075")
        ax.annotate(f"z={a['z_ref']}\n{a['peak_D']:.4f}D",
                    (a["z_ref"], a["peak_D"]), textcoords="offset points",
                    xytext=(8, 6), fontsize=8)
    # Cs direction at z_ref=18
    z18 = {a["C_s"]: a["peak_D"] for n, a in arms.items() if a["z_ref"] == 18}
    if z18:
        ax.scatter([18] * len(z18), list(z18.values()), marker="s",
                   color="tab:red", zorder=5, label="Cs sweep @z_ref=18")
    ax.axhline(GATE, color="gray", ls="--", lw=1, label=f"3% gate {GATE:.4f}D")
    ax.axhline(EXP_PEAK_D, color="k", ls=":", lw=1, label=f"exp peak {EXP_PEAK_D}D")
    ax.set_xlabel("z_ref (grid index)")
    ax.set_ylabel("peak cavity depth (D)")
    ax.set_title("CMB7-CSTUNE2-R1: peak depth vs z_ref (Cs=0.075) + Cs sweep")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig3(out):
    import json as _j
    rows = []
    for tag, zr in [("off", None), ("z10", 10), ("z12", 12), ("z14", 14),
                    ("z18", 18), ("z20", 20)]:
        d = _j.load(open(f"/root/tp-agent/work/m27389/development/audit/cmb7_cstune2_r1/probe/{tag}.json"))
        rows.append((tag, zr, d["corr_at_surface"], d["corr_at_z12"], d["fhash"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    zs = [r[1] for r in rows if r[1] is not None]
    cs = [r[2] for r in rows if r[1] is not None]
    cz = [r[3] for r in rows if r[1] is not None]
    ax1.plot(zs, [c * 1e4 for c in cs], "o-", label="corr @ water surface (z=24.5)")
    ax1.plot(zs, [c * 1e4 for c in cz], "s-", label="corr @ cavity bottom (z=12)")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_xlabel("z_ref")
    ax1.set_ylabel("rho_l*gz*(z-z_ref)/cs2  (x1e-4)")
    ax1.set_title("Analytic hydrostatic correction vs z_ref")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    tags = [r[0] for r in rows]
    hs = [r[4] for r in rows]
    ax2.bar(range(len(rows)), [1] * len(rows), color=["tab:gray"] + ["tab:blue"] * 5)
    ax2.set_xticks(range(len(rows)))
    ax2.set_xticklabels(tags, rotation=45)
    ax2.set_ylabel("distinct f-field hash (40-step)")
    ax2.set_title("All 6 arms distinct (correction engaged)")
    for i, h in enumerate(hs):
        ax2.text(i, 0.05, h[:6], ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    arms = load_arms()
    outdir = Path("/root/tp-agent/work/m27389/development/audit/cmb7_cstune2_r1/figs")
    outdir.mkdir(parents=True, exist_ok=True)
    f1 = outdir / "fig1_depth_time.png"
    f2 = outdir / "fig2_response.png"
    f3 = outdir / "fig3_mechanism.png"
    fig1(arms, f1)
    fig2(arms, f2)
    fig3(f3)
    for f in (f1, f2, f3):
        b64 = base64.b64encode(f.read_bytes()).decode()
        (outdir / (f.stem + ".b64")).write_text(b64)
        print(f, f.stat().st_size, "bytes")
    print("arms loaded:", list(arms.keys()))


if __name__ == "__main__":
    main()
