"""Generate the P142W3-N01 report assets (figures + markdown + JSON).

Every number in the markdown is read back from ``runs/p142w3_n01/*.json`` and
``*.npz`` (programmatic injection, no hand copying).

    python scripts/gen_p142w3_report.py --outdir runs/p142w3_n01
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorlbm.lng_membrane_sloshing import fatigue
from tensorlbm.lng_membrane_sloshing.campaign import summarize
from tensorlbm.lng_membrane_sloshing.tank import MembraneTank174k


def _load(outdir: Path) -> tuple[dict, dict[str, dict]]:
    records = {}
    for path in sorted(outdir.glob("*.json")):
        rec = json.loads(path.read_text())
        records[path.stem] = rec
    return summarize(outdir), records


def _arrays(outdir: Path, stem: str) -> dict:
    with np.load(outdir / f"{stem}.npz") as data:
        return {k: data[k] for k in data.files}


def fig_free_surface(outdir: Path, records: dict, path: Path) -> None:
    stems = [s for s in records if s.startswith("impact_f")]
    stems.sort()
    fig, axes = plt.subplots(1, len(stems), figsize=(4.2 * len(stems), 3.2), squeeze=False)
    tank = MembraneTank174k()
    for ax, stem in zip(axes[0], stems):
        rec = records[stem]
        data = _arrays(outdir, stem)
        eta, t = data["eta"], data["t"]
        h = rec["depth_m"]
        n = eta.shape[1]
        x = (np.arange(n) + 0.5) * rec["similarity"]["dx_m"]
        idx = np.linspace(int(0.55 * len(t)), len(t) - 1, 4).astype(int)
        for k, i in enumerate(idx):
            ax.plot(x, eta[i], lw=1.2, label=f"t={t[i]:.1f} s")
        ax.axhline(0.0, color="k", lw=0.6, ls=":")
        ax.set_title(f"fill {rec['fill']*100:.0f}%  (h={h:.1f} m, {rec['cells_per_depth']:.0f} cells)")
        ax.set_xlabel("x [m]")
        ax.set_ylabel(r"$\eta$ [m]")
        ax.legend(fontsize=6)
    fig.suptitle("P142W3-N01 free-surface evolution in the 174k m3 membrane tank (forced sway at f1)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_pressure(outdir: Path, records: dict, path: Path) -> None:
    stems = sorted(s for s in records if s.startswith("impact_f"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    cmap = plt.get_cmap("viridis")
    for k, stem in enumerate(stems):
        rec = records[stem]
        data = _arrays(outdir, stem)
        t = data["t"]
        ramp = 2.0 / rec["f1_anchor_hz"]
        keep = t >= ramp
        press = np.nan_to_num(data["pressure"])
        iy = int(round(rec["p_max_height_m"] / rec["similarity"]["dx_m"] + 0.5))
        iy = min(max(iy, 0), press.shape[0] - 1)
        axes[0].plot(t[keep], press[iy, keep] / 1e3, lw=0.9, color=cmap(k / max(1, len(stems) - 1)),
                     label=f"fill {rec['fill']*100:.0f}% (y={rec['p_max_height_m']:.1f} m)")
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel("$p_{dyn}$ [kPa]")
    axes[0].set_title("wall gauge pressure history (post-ramp)")
    axes[0].legend(fontsize=7)
    # anchors vs LBM peak (dimensionless)
    labels, lbm, mlm, wh, stag = [], [], [], [], []
    for stem in stems:
        rec = records[stem]
        v = max(rec["v_impact_ms"], 1e-9)
        beta = max(rec["beta_eff_deg"], 1.0)
        p_mlm = 0.5 * math.pi / math.tan(math.radians(beta)) - 0.5
        labels.append(f"fill {rec['fill']*100:.0f}%\nV={v:.2f} m/s, b={beta:.0f}deg")
        lbm.append(rec["p_dyn_excess_pa"] / (470.0 * v * v))
        mlm.append(max(p_mlm, 0.5))
        wh.append(1250.0 / v)
        stag.append(0.5)
    xpos = np.arange(len(labels))
    axes[1].bar(xpos - 0.2, lbm, width=0.4, label="$p^*_{LBM}$")
    axes[1].bar(xpos + 0.2, mlm, width=0.4, label="$p^*_{MLM}$")
    axes[1].plot(xpos, wh, "r*", label="$p^*_{WH}=c/V$")
    axes[1].plot(xpos, stag, "k--", lw=0.8, label="$p^*_{stag}=0.5$")
    axes[1].set_yscale("log")
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(labels, fontsize=6)
    axes[1].set_ylabel("$p_{max}/(\\rho V^2)$")
    axes[1].set_title("impact peak vs Wagner/MLM anchors")
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_fatigue(outdir: Path, records: dict, path: Path) -> None:
    stems = sorted(s for s in records if s.startswith("impact_f"))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4))
    cmap = plt.get_cmap("plasma")
    for k, stem in enumerate(stems):
        rec = records[stem]
        data = _arrays(outdir, stem)
        t = data["t"]
        ramp = 2.0 / rec["f1_anchor_hz"]
        keep = t >= ramp
        press = np.nan_to_num(data["pressure"])
        iy = int(round(rec["p_max_height_m"] / rec["similarity"]["dx_m"] + 0.5))
        iy = min(max(iy, 0), press.shape[0] - 1)
        series = press[iy, keep]
        dt = float(t[1] - t[0])
        stress = fatigue.stress_from_pressure(series, span=0.8, thickness=0.022)
        f_cut = 6.0 * rec["f1_anchor_hz"]
        stress = fatigue.lowpass(stress, dt, f_cut=f_cut)
        moments = fatigue.spectral_moments(stress, dt, f_max=f_cut)
        s = np.linspace(1e-6, 10 * math.sqrt(moments[0]), 4000)
        axes[0].plot(s, fatigue.dirlik_range_pdf(s, moments) / fatigue.dirlik_range_pdf(s, moments).max(),
                     color=cmap(k / max(1, len(stems) - 1)), label=f"fill {rec['fill']*100:.0f}%")
        cycles = fatigue.rainflow_cycles(stress, threshold=1e-3 * float(np.max(np.abs(stress))))
        ranges = np.array([c[0] for c in cycles])
        counts = np.array([c[1] for c in cycles])
        if ranges.size:
            hist, edges = np.histogram(ranges, bins=18, range=(0, s[-1]), weights=counts)
            centres = 0.5 * (edges[:-1] + edges[1:])
            axes[1].bar(centres, hist / max(hist.max(), 1), width=0.8 * (edges[1] - edges[0]),
                        alpha=0.35, color=cmap(k / max(1, len(stems) - 1)))
    axes[0].set_xlabel("stress range S [MPa]")
    axes[0].set_ylabel("Dirlik $p(S)$ (normalised)")
    axes[0].set_title("Dirlik rainflow-range distribution")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("stress range S [MPa]")
    axes[1].set_ylabel("rainflow histogram (normalised)")
    axes[1].set_title("ASTM E1049-85 rainflow count")
    d_d, d_r = [], []
    for stem in stems:
        rec = records[stem]
        d_d.append(rec["fatigue"]["damage_dirlik"])
        d_r.append(rec["fatigue"]["damage_rainflow"])
    xpos = np.arange(len(stems))
    axes[2].bar(xpos - 0.2, d_d, width=0.4, label="Dirlik")
    axes[2].bar(xpos + 0.2, d_r, width=0.4, label="rainflow")
    axes[2].set_xticks(xpos)
    axes[2].set_xticklabels([f"{records[s]['fill']*100:.0f}%" for s in stems])
    axes[2].set_ylabel("Miner damage D")
    axes[2].set_title("fatigue damage (IIW FAT80)")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="runs/p142w3_n01")
    args = parser.parse_args(argv)
    outdir = Path(args.outdir)
    report, records = _load(outdir)
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig_free_surface(outdir, records, figdir / "fig1_free_surface.png")
    fig_pressure(outdir, records, figdir / "fig2_impact_pressure.png")
    fig_fatigue(outdir, records, figdir / "fig3_fatigue_spectrum.png")
    (outdir / "report_data.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v["pass"] for k, v in report["judges"].items()}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())