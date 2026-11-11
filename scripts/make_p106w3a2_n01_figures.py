#!/usr/bin/env python3
"""Figures for P106W3A2-N01 (heterogeneous-backend equivalence ladder).

Reads the artifacts written by ``scripts/run_p106w3a2_n01_equivalence.py`` and
produces:

* ``fig1_l0_ladder.png``    L0 deviation growth vs rounding budget + drift (L2)
* ``fig2_l1_profiles.png``  U+ / rms profiles: golden vs accelerator vs MKM-1999
* ``fig3_pareto.png``       error vs cost Pareto (L0 and L1 points)

Every number is read from the JSON artifacts (no hand-typed values).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "p106w3a2_n01"
FIG = ROOT / "figures" / "p106w3a2_n01"

L0_COLORS = {
    "cpu_fp64_serial": "k",
    "cpu_fp64_numpy_tree": "tab:blue",
    "cpu_fp64_torch_gemm": "tab:green",
    "sdaa_fp64_torch": "tab:purple",
    "cpu_fp32_torch": "tab:orange",
    "sdaa_fp32_torch": "tab:red",
    "cpu_fp64_shipped_w": "tab:brown",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fig_l0() -> Path:
    l0 = _load(ART / "l0.json")
    golden = L0_COLORS  # alias for readability
    tiers = [t for t in l0["checks"] if (ART / "l0" / f"{t}.json").exists()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for tier in tiers:
        run = _load(ART / "l0" / f"{tier}.json")
        steps = np.asarray([t["step"] for t in run["trace"]], dtype=float)
        u = np.asarray([t["u_max"] for t in run["trace"]], dtype=float)
        ref = np.asarray([t["u_max"] for t in _load(ART / "l0" / f"{l0['golden_profile']}.json")["trace"]])
        rel = np.abs(u - ref) / np.max(np.abs(ref))
        rel = np.maximum(rel, 1e-20)
        ax.semilogy(steps, rel, label=tier, color=golden.get(tier, None), lw=1.4)
    checks = l0["checks"]
    fp64_budget = np.median([c["ulp_budget_relative"] for t, c in checks.items()
                             if c["dtype"] == "float64"])
    fp32_budget = np.median([c["ulp_budget_relative"] for t, c in checks.items()
                             if c["dtype"] == "float32"])
    ax.axhline(fp64_budget, color="tab:blue", ls="--", lw=1.2,
               label=f"fp64 rounding budget {fp64_budget:.1e}")
    ax.axhline(fp32_budget, color="tab:orange", ls="--", lw=1.2,
               label=f"fp32 rounding budget {fp32_budget:.1e}")
    ax.set_xlabel("step (D3Q19 periodic box, 64$^3$)")
    ax.set_ylabel(r"$|u_{\max}^{\rm tier} - u_{\max}^{\rm golden}| \,/\, \max|u_{\max}^{\rm golden}|$")
    ax.set_title("(a) L0: deviation from the CPU-fp64 golden vs rounding budget")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    names = [t for t in tiers]
    mass = [l0["l2"][t]["mass"]["drift_per_step_per_cell"] for t in names]
    mom = [l0["l2"][t]["momentum"]["drift_per_step_per_cell"] for t in names]
    x = np.arange(len(names))
    ax.bar(x - 0.2, mass, 0.4, label="mass drift", color="tab:blue")
    ax.bar(x + 0.2, mom, 0.4, label="momentum drift", color="tab:green")
    ax.axhline(1e-12, color="k", ls="--", lw=1.2, label="L2 gate 1e-12")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("cpu_", "").replace("_torch", "").replace("sdaa_", "sdaa_")
                        for n in names], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("drift per step per cell")
    ax.set_title("(b) L2: conservation drift (fp64 passes, fp32 fails)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "fig1_l0_ladder.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_l1() -> Path:
    from tensorlbm.validation import mkm1999

    report = _load(ART / "report.json")
    golden_name = None
    for name, payload in report["l1_checks"].items():
        golden_name = payload["golden"]
    golden = _load(ART / "l1" / f"{golden_name}_full.json")
    mkm = mkm1999.load_profile(180)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    wu = golden["wall_units"]
    ax.semilogx(mkm.y_plus, mkm.u_plus, "k--", lw=1.2, label="MKM-1999 DNS (Re$_\\tau$=178.12)")
    ax.semilogx(wu["y_plus"], wu["u_plus"], "-", color="tab:blue", lw=2,
                label=f"golden {golden['profile']['name']} ({golden['profile']['dtype']})")
    for name, payload in report["l1_checks"].items():
        run = _load(ART / "l1" / f"{name}_full.json")
        w = run["wall_units"]
        ax.semilogx(w["y_plus"], w["u_plus"], "-", color="tab:red", lw=1.5,
                    label=f"tier {name} ({run['profile']['dtype']})")
    for path in sorted((ART / "l1").glob("*_negative.json")):
        run = _load(path)
        w = run["wall_units"]
        ax.semilogx(w["y_plus"], w["u_plus"], ":", color="tab:brown", lw=1.5,
                    label=f"negative control: {run.get('wall_scheme')} wall")
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"$U^+$")
    ax.set_title("(a) mean velocity profile, $Re_\\tau=180$, $10^4$ steps")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    keys = [("uu_plus", r"$u_{\rm rms}^{+}$", "-"), ("vv_plus", r"$v_{\rm rms}^{+}$", "--"),
            ("ww_plus", r"$w_{\rm rms}^{+}$", ":")]
    for key, label, ls in keys:
        ax.semilogx(wu["y_plus"], wu[key], ls, color="tab:blue", lw=2,
                    label=f"golden {label}")
        for name in report["l1_checks"]:
            run = _load(ART / "l1" / f"{name}_full.json")
            ax.semilogx(run["wall_units"]["y_plus"], run["wall_units"][key], ls,
                        color="tab:red", lw=1.2, label=f"{name} {label}")
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel("rms velocity (wall units)")
    ax.set_title("(b) Reynolds-stress rms profiles")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "fig2_l1_profiles.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_pareto() -> Path:
    report = _load(ART / "report.json")
    pts = report["cost_accuracy_ledger"]["points"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, level in zip(axes, ("L0", "L1")):
        sel = [p for p in pts if p["level"] == level and p["error"] is not None and p["cost"]]
        sel = [dict(p, error=max(p["error"], 1e-20)) for p in sel]
        front = set(report["cost_accuracy_ledger"][f"pareto_{level}"])
        for p in sel:
            ax.scatter(p["cost"], p["error"], s=70,
                       color="tab:red" if p["name"] in front else "tab:gray",
                       zorder=3)
            ax.annotate(p["name"], (p["cost"], p["error"]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        ordered = sorted([p for p in sel if p["name"] in front], key=lambda p: p["cost"])
        if len(ordered) > 1:
            ax.plot([p["cost"] for p in ordered], [p["error"] for p in ordered],
                    "k--", lw=1.0, zorder=2)
        ax.set_yscale("log")
        ax.set_xlabel("cost [s per " + ("step (64$^3$)" if level == "L0" else "convective time h/u$_\\tau$") + "]")
        ax.set_ylabel("error vs golden (relative)")
        ax.set_title(f"{level}: error-cost (red = Pareto front)")
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "fig3_pareto.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    paths = []
    for fn in (fig_l0, fig_l1, fig_pareto):
        try:
            paths.append(fn())
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"figure {fn.__name__} failed: {exc}", file=sys.stderr)
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()