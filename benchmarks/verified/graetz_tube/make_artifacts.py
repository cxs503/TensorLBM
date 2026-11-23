"""Merge the shard JSONs into result.json (gate evaluation) and draw the figures."""

from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import reference_series as RS

ART = ROOT / "artifacts" / "graetz_tube"
GATED = [i for i, x in enumerate([0.002, 0.005, 0.01, 0.02, 0.04]) if x >= 0.005]


def load():
    out = {}
    for p in sorted(ART.glob("shard*.json")):
        d = json.loads(p.read_text())
        out[d["variant"]] = d
    return out


def evaluate(data):
    res = {
        "case": "graetz_tube",
        "reference": data["orig"]["reference"],
        "x_star": data["orig"]["x_star"],
        "nu_inf": data["orig"]["nu_inf"],
        "gated_stations": [data["orig"]["x_star"][i] for i in GATED],
        "variants": {},
    }
    for v, d in data.items():
        lv = {
            str(l["n_r"]): {
                "Nu": l["Nu"],
                "rel_err": l["rel_err"],
                "n_x": l["n_x"],
                "n_steps": l["n_steps"],
                "R": l["R"],
                "runtime_s": l["runtime_s"],
            }
            for l in d["levels"]
        }
        e40 = np.abs(np.array(lv["40"]["rel_err"]))[GATED]
        e56 = np.abs(np.array(lv["56"]["rel_err"]))[GATED]
        res["variants"][v] = {
            "levels": lv,
            "max_abs_err_gated_40": float(e40.max()),
            "max_abs_err_gated_56": float(e56.max()),
            "G1_accuracy": bool(max(e40.max(), e56.max()) <= 0.03),
            "G2_monotone": bool(np.all(e56 < e40)),
            "frontier_x0.002_err_40": float(lv["40"]["rel_err"][0]),
            "frontier_x0.002_err_56": float(lv["56"]["rel_err"][0]),
        }
        res["variants"][v]["VERIFIED"] = bool(
            res["variants"][v]["G1_accuracy"] and res["variants"][v]["G2_monotone"]
        )
    return res


def figures(data, res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    REF = RS.series(np.logspace(-3.1, -0.5, 120), n_modes=10, n_r=800)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(REF["x_star"], REF["Nu"], "k-", lw=2, label="Graetz series (independent ref)")
    styles = {"orig": ("o", "--"), "reff": ("s", "-")}
    for v, d in data.items():
        for l in d["levels"]:
            m, ls = styles[v]
            ax[0].plot(d["x_star"], l["Nu"], ls, marker=m, ms=5, label=f"{v}, n_r={l['n_r']}")
    ax[0].axhline(3.6568, color="0.6", ls=":", lw=1)
    ax[0].text(0.05, 3.70, r"$Nu_\infty=3.6568$", color="0.4", fontsize=8)
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"$x^*=\alpha x/(u_m D^2)$")
    ax[0].set_ylabel(r"$Nu(x^*)$")
    ax[0].set_title("tube Graetz entrance Nu")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)
    for v, d in data.items():
        for l in d["levels"]:
            m, ls = styles[v]
            ax[1].plot(
                d["x_star"],
                100 * np.array(l["rel_err"]),
                ls,
                marker=m,
                ms=5,
                label=f"{v}, n_r={l['n_r']}",
            )
    ax[1].axhspan(-3, 3, color="0.85", label="±3% gate")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$x^*$")
    ax[1].set_ylabel("Nu relative error [%]")
    ax[1].set_title("error vs independent series")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "fig1_nu_and_error.png", dpi=150)
    # field of the accepted variant at the coarse level
    sys.path.insert(0, str(HERE))
    import study

    best = [v for v in ("reff", "orig") if res["variants"][v]["VERIFIED"]][:1]
    best = best[0] if best else "reff"
    f = study._lbm_reff(40, study.U_MEAN, study.ALPHA, study.X_STAR) if best == "reff" else None
    if f is not None:
        fig2, ax2 = plt.subplots(figsize=(6.2, 3.4))
        X = np.arange(f["field"].shape[1]) / f["D"]
        Y = f["r_over_R"]
        im = ax2.contourf(X, Y, f["field"], levels=20, cmap="inferno")
        ax2.set_xlabel(r"$x/D$")
        ax2.set_ylabel(r"$r/R$")
        ax2.set_title(f"temperature field, {best} convention (n_r=40)")
        fig2.colorbar(im, ax=ax2, label=r"$\Theta$")
        fig2.tight_layout()
        fig2.savefig(HERE / "figures" / "fig2_field.png", dpi=150)
        res["field_figure_variant"] = best
    return res


if __name__ == "__main__":
    data = load()
    if len(data) < 2:
        raise SystemExit(f"need both shards, found {sorted(data)}")
    res = evaluate(data)
    res = figures(data, res)
    (HERE / "result.json").write_text(json.dumps(res, indent=1))
    print(
        json.dumps(
            {
                v: {
                    k: res["variants"][v][k]
                    for k in (
                        "max_abs_err_gated_40",
                        "max_abs_err_gated_56",
                        "G1_accuracy",
                        "G2_monotone",
                        "VERIFIED",
                        "frontier_x0.002_err_40",
                        "frontier_x0.002_err_56",
                    )
                }
                for v in res["variants"]
            },
            indent=1,
        )
    )
