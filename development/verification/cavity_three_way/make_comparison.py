"""Assemble the three-party comparison for XVAL-LBM-CAVITY.

Consumes every ``runs/<tag>.json`` produced by ``party_a_tensorlbm.py`` /
``party_b_pylbm.py`` and emits, with no hand-typed numbers anywhere:

    artifacts/comparison.json   machine-readable metric table
    artifacts/comparison.md     the report table (ref vs got, per item)
    artifacts/fig_*.png         centreline profiles + error bars + convergence

Missing runs are reported as ``not_run`` -- never silently dropped, never
counted as a pass.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from cavity_spec import inter_tool_delta, load_ghia

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
ART = HERE / "artifacts"

#: acceptance thresholds declared *before* looking at the numbers
THRESH = {
    "max_abs_dev_pct_ref": 3.0,     # per-point |ref-got| <= 3 % of U_lid
    "inter_tool_pct": 3.0,          # party A vs party B <= 3 % of U_lid
    "convergence_decrease": True,   # error must decrease W=64 -> W=128
}


def load_runs() -> dict:
    out = {}
    for p in sorted(RUNS.glob("*.json")):
        d = json.loads(p.read_text())
        if "metrics" in d:
            out[p.stem] = d
    return out


def pick(runs: dict, party: str, re: float, w: int):
    for tag, d in runs.items():
        if d["party"] != party:
            continue
        spec = d["spec"]
        width = d["metrics"]["geometry"]["W_lattice"]
        if abs(spec["re"] - re) < 1e-9 and abs(width - w) < 0.6:
            return tag, d
    return None, None


def pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / abs(b) if b else float("nan")


def build_table(runs: dict) -> dict:
    specs = [(100.0, 128), (100.0, 64), (400.0, 128), (400.0, 64)]
    table = {"thresholds": THRESH, "specs": []}

    for re, w in specs:
        tags = {}
        a_tag, A = pick(runs, "A", re, w)
        b_tag, B = pick(runs, "B", re, w)
        ghia = load_ghia(int(re))
        row = {
            "re": re,
            "W": w,
            "tau": (0.5 + 3 * 0.06 * w / re),
            "party_A": None,
            "party_B": None,
            "ref": {
                "u_min": float(ghia["u"].min()),
                "u_min_y": float(ghia["y"][int(np.argmin(ghia["u"]))]),
                "v_max": float(ghia["v"].max()),
                "v_max_x": float(ghia["x"][int(np.argmax(ghia["v"]))]),
                # Ghia's own y column is descending: interp needs an increasing
                # abscissa, so sort the pair (never sort one side alone).
                "u_center": float(np.interp(0.5, np.sort(ghia["y"]),
                                            ghia["u"][np.argsort(ghia["y"])])),
                "v_center": float(np.interp(0.5, np.sort(ghia["x"]),
                                            ghia["v"][np.argsort(ghia["x"])])),
                "source": "Ghia 1982 Table I/II (machine transcription, sha256 in JSON)",
            },
        }
        for party, tag, D in (("A", a_tag, A), ("B", b_tag, B)):
            if D is None:
                row[f"party_{party}"] = {"tag": f"A/B re={re} W={w}", "status": "not_run"}
                tags[party] = None
                continue
            m = D["metrics"]
            tags[party] = tag
            row[f"party_{party}"] = {
                "tag": tag,
                "status": "ok",
                "tool": D["tool"],
                "steps": D["steps"],
                "resid": D["final_resid"],
                "elapsed_s": D["elapsed_s"],
                "u_min": m["u_min"],
                "u_min_y": m["u_min_y"],
                "v_max": m["v_max"],
                "v_max_x": m["v_max_x"],
                "u_center": m["u_center"],
                "v_center": m["v_center"],
                "vortex_center": m["vortex_center"],
                "rmse_u": m["rmse_u"],
                "rmse_v": m["rmse_v"],
                "max_abs_dev_pct": m["max_abs_dev_pct"],
                "wall_convention": m["wall_convention"],
                "aspect": m["geometry"]["aspect"],
                "err_pct": {
                    "u_min": pct(m["u_min"], row["ref"]["u_min"]),
                    "v_max": pct(m["v_max"], row["ref"]["v_max"]),
                    "u_center": pct(m["u_center"], row["ref"]["u_center"]),
                    "v_center": pct(m["v_center"], row["ref"]["v_center"]),
                    "y_of_u_min": pct(m["u_min_y"], row["ref"]["u_min_y"]),
                    "x_of_v_max": pct(m["v_max_x"], row["ref"]["v_max_x"]),
                },
            }

        if A is not None and B is not None:
            row["inter_tool"] = inter_tool_delta(
                {k: np.asarray(v) for k, v in A["profile"].items()},
                {k: np.asarray(v) for k, v in B["profile"].items()},
            )
            row["inter_tool"]["du_min_pct"] = pct(A["metrics"]["u_min"], B["metrics"]["u_min"])
            row["inter_tool"]["dv_max_pct"] = pct(A["metrics"]["v_max"], B["metrics"]["v_max"])
            row["inter_tool"]["vortex_dist"] = float(
                math.dist(A["metrics"]["vortex_center"], B["metrics"]["vortex_center"])
            )
            row["inter_tool"]["vortex_dist_ref"] = float(
                math.dist(A["metrics"]["vortex_center"], [0.0, 0.0])
            )
        else:
            row["inter_tool"] = {"status": "not_run"}

        # verdict per spec
        ok_ref = {
            p: (row[f"party_{p}"].get("max_abs_dev_pct", 1e9) <= THRESH["max_abs_dev_pct_ref"]
                if row[f"party_{p}"]["status"] == "ok" else None)
            for p in ("A", "B")
        }
        ok_int = (row["inter_tool"].get("max_abs_pct", 1e9) <= THRESH["inter_tool_pct"]
                  if row["inter_tool"].get("status") != "not_run" else None)
        row["verdict"] = {"vs_ref_pass": ok_ref, "inter_tool_pass": ok_int}
        table["specs"].append(row)
    return table


def md_table(table: dict) -> str:
    L = []
    L.append("| Re | W | τ | 方 | u_min | ref u_min | 误差 | v_max | ref v_max | 误差 | "
             "u(0.5,0.5) | ref | 误差 | RMSE(u) | max_dev | ≤3%? |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in table["specs"]:
        ref = s["ref"]
        for p, name in (("party_A", "A TensorLBM"), ("party_B", "B pylbm")):
            d = s[p]
            if d["status"] != "ok":
                L.append(f"| {s['re']:.0f} | {s['W']} | {s['tau']:.4f} | {name} | not_run | "
                         f"{ref['u_min']:.5f} | — | {ref['v_max']:.5f} | — | — | — | "
                         f"{ref['u_center']:.5f} | — | — | — | not_run |")
                continue
            e = d["err_pct"]
            L.append(
                f"| {s['re']:.0f} | {s['W']} | {s['tau']:.4f} | {name} | {d['u_min']:+.5f} | "
                f"{ref['u_min']:+.5f} | {e['u_min']:+.2f}% | {d['v_max']:+.5f} | "
                f"{ref['v_max']:+.5f} | {e['v_max']:+.2f}% | {d['u_center']:+.5f} | "
                f"{ref['u_center']:+.5f} | {e['u_center']:+.2f}% | {d['rmse_u']:.4f} | "
                f"{d['max_abs_dev_pct']:.2f}% | "
                f"{'✅' if d['max_abs_dev_pct'] <= 3.0 else '❌'} |"
            )
    L.append("")
    L.append("| Re | W | 工具间差 max|Δu| | max|Δv| | Δu_min | Δv_max | 涡心距 | ≤3%? |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in table["specs"]:
        it = s["inter_tool"]
        if it.get("status") == "not_run":
            L.append(f"| {s['re']:.0f} | {s['W']} | not_run | — | — | — | — | not_run |")
            continue
        L.append(
            f"| {s['re']:.0f} | {s['W']} | {it['max_abs_du']*100:.2f}% | "
            f"{it['max_abs_dv']*100:.2f}% | {it['du_min_pct']:+.2f}% | "
            f"{it['dv_max_pct']:+.2f}% | {it['vortex_dist']:.4f} | "
            f"{'✅' if it['max_abs_pct'] <= 3.0 else '❌'} |"
        )
    return "\n".join(L)


def figures(runs: dict, table: dict) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    ART.mkdir(exist_ok=True)

    # ---- Fig 1: centreline profiles, Re=100 W=128, three parties ----------
    for re in (100.0, 400.0):
        w = 128
        _, A = pick(runs, "A", re, w)
        _, B = pick(runs, "B", re, w)
        if A is None and B is None:
            w = 64
            _, A = pick(runs, "A", re, w)
            _, B = pick(runs, "B", re, w)
        if A is None and B is None:
            continue
        g = load_ghia(int(re))
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
        for D, lab, st in ((A, "TensorLBM (this repo)", "-"), (B, "pylbm (OSS)", "--")):
            if D is None:
                continue
            pr = D["profile"]
            ax[0].plot(np.asarray(pr["u"]), np.asarray(pr["y"]), st, lw=1.6,
                       label=f"{lab} W={D['metrics']['geometry']['W_lattice']:.0f}")
            ax[1].plot(np.asarray(pr["x"]), np.asarray(pr["v"]), st, lw=1.6, label=lab)
        ax[0].plot(g["u"], g["y"], "o", ms=4, mfc="none", label="Ghia 1982 (Table I)")
        ax[1].plot(g["x"], g["v"], "o", ms=4, mfc="none", label="Ghia 1982 (Table II)")
        ax[0].set_xlabel("u/U_lid"); ax[0].set_ylabel("y/L")
        ax[1].set_xlabel("x/L"); ax[1].set_ylabel("v/U_lid")
        ax[0].set_title(f"vertical centreline, Re={re:g}")
        ax[1].set_title(f"horizontal centreline, Re={re:g}")
        for a in ax:
            a.grid(alpha=0.3); a.legend(fontsize=7)
        fig.tight_layout()
        p = ART / f"fig_centerline_re{int(re)}.png"
        fig.savefig(p, dpi=140); plt.close(fig)
        made.append(p.name)

    # ---- Fig 2: item-wise error bars + inter-tool spread -----------------
    labels, a_err, b_err = [], [], []
    for s in table["specs"]:
        for key, lab in (("u_min", "u_min"), ("v_max", "v_max"),
                         ("u_center", "u(0.5,0.5)")):
            labels.append(f"Re{s['re']:.0f}\nW{s['W']}\n{lab}")
            da, db = s["party_A"], s["party_B"]
            a_err.append(abs(da["err_pct"][key]) if da["status"] == "ok" else np.nan)
            b_err.append(abs(db["err_pct"][key]) if db["status"] == "ok" else np.nan)
    if labels:
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.45), 3.6))
        ax.bar(x - 0.2, a_err, 0.4, label="TensorLBM vs ref")
        ax.bar(x + 0.2, b_err, 0.4, label="pylbm vs ref")
        ax.axhline(3.0, color="r", ls="--", lw=1, label="3 % threshold")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6)
        ax.set_ylabel("|error| vs Ghia [% of U_lid]")
        ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        p = ART / "fig_error_items.png"
        fig.savefig(p, dpi=140); plt.close(fig)
        made.append(p.name)

    # ---- Fig 3: resolution convergence of max_abs_dev --------------------
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for party, lab, mk in (("A", "TensorLBM", "o"), ("B", "pylbm", "s")):
        for re in (100.0, 400.0):
            ws, es = [], []
            for w in (64, 128):
                _, D = pick(runs, party, re, w)
                if D is not None:
                    ws.append(w); es.append(D["metrics"]["max_abs_dev_pct"])
            if ws:
                ax.plot(ws, es, mk + "-", label=f"{lab} Re={re:g}")
    ax.set_xlabel("W [lattice spacings]"); ax.set_ylabel("max |ref-got| [% U_lid]")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3)
    ax.axhline(3.0, color="r", ls="--", lw=1)
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = ART / "fig_convergence.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    made.append(p.name)
    return made


def main():
    runs = load_runs()
    table = build_table(runs)
    ART.mkdir(exist_ok=True)
    figs = figures(runs, table)
    table["figures"] = figs
    table["runs_found"] = sorted(runs)
    (ART / "comparison.json").write_text(json.dumps(table, indent=1))
    (ART / "comparison.md").write_text(
        "# XVAL-LBM-CAVITY · three-party metric table\n\n"
        + md_table(table)
        + "\n\nGeometry conventions per party:\n"
        + "\n".join(
            f"* Re{s['re']:.0f} W{s['W']} A: {s['party_A'].get('wall_convention','not_run')} "
            f"(aspect {s['party_A'].get('aspect')}) | B: {s['party_B'].get('wall_convention','not_run')} "
            f"(aspect {s['party_B'].get('aspect')})"
            for s in table["specs"]
        )
    )
    print(md_table(table))
    print("\nfigures:", figs)


if __name__ == "__main__":
    main()