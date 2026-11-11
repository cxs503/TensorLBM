#!/usr/bin/env python3
"""AUDIT-RUNTEST-6 — figures for the audit report (matplotlib, Agg backend).

fig1  convergence ladder: L2 relative error vs channel height for all four wall
      arms (before/after the D1 fix), with reference slopes p=1 and p=2.
fig2  acceptance-gate view: |error| in % for every measured benchmark, against
      the 3% gate that benchmarks/STATUS.md declares.

All series are read from runs/audit_runtest_r6/*.json — no hard-coded numbers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "artifacts/audit_runtest_6" if (REPO / "artifacts/audit_runtest_6").exists() else REPO / "runs/audit_runtest_r6"
VER = REPO / "benchmarks/verified"
OUT = REPO / "figures/audit_runtest_r6"


def jload(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def fig_ladder() -> Path | None:
    series = {}
    for name, path in [
        ("fix H=10/20", RUNS / "conv_fix_H10_20.json"),
        ("fix H=20/40", RUNS / "conv_fix_H20_40.json"),
    ]:
        d = jload(path)
        if not d:
            continue
        for arm, v in d["wall_models"].items():
            hs = v["convergence"]["h_true_list"]
            series.setdefault(arm, []).append((hs, v["convergence"]["errors_l2_rel"], name))
    old = jload(RUNS / "conv_repro.json")
    if not series:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    styles = {
        "halfway": ("o", "-", "halfway, wall truly at 0.5Δ (p≈2)"),
        "subcell_halfway": ("s", "-", "mid-point BC on a 0.25Δ wall (p≈1)"),
        "effective": ("^", "--", "same run, effective-wall reference (p≈2)"),
        "bouzidi": ("d", "-", "Bouzidi interpolated at 0.25Δ (p≈2)"),
    }
    for arm, chunks in series.items():
        for hs, errs, label in chunks:
            marker, ls, leg = styles.get(arm, ("x", "-", arm))
            ax.loglog(
                hs,
                errs,
                marker=marker,
                ls=ls,
                label=f"{leg} [{label}]",
                markersize=6,
            )
    # reference slopes through the first halfway point
    if "halfway" in series:
        hs0, e0, _ = series["halfway"][0]
        xs = [hs0[0], hs0[-1] * 2]
        for p, style in [(1, ":"), (2, "-.")]:
            ys = [e0[0], e0[0] * (xs[1] / xs[0]) ** (-p)]
            ax.loglog(xs, ys, style, color="0.55", lw=1.0)
            ax.annotate(f"slope {p}", (xs[1], ys[1]), textcoords="offset points",
                        xytext=(4, 4), fontsize=8, color="0.4")
    if old:
        e_arr = old["wall_models"]["halfway"]["convergence"]["errors_l2_rel"]
        hs = old["wall_models"]["halfway"]["convergence"]["h_true_list"] if "h_true_list" in old["wall_models"]["halfway"]["convergence"] else old["H_list"]
        ax.loglog(hs, e_arr, "k*", markersize=13,
                  label="HEAD (all 3 declared arms identical)")
    ax.set_xlabel("channel height H  (lattice units)")
    ax.set_ylabel(r"$L_2$ relative error vs analytic Poiseuille profile")
    ax.set_title("Poiseuille 2D wall-treatment ladder — AUDIT-RUNTEST-6")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "conv_ladder.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_gate() -> Path | None:
    items = []
    tg = jload(RUNS / "tg2d.json/result.json")
    if tg:
        items.append(("Taylor-Green 2D\nenergy decay", abs(tg["err_e_pct"])))
    sw = jload(RUNS / "shear_wave.json/result.json")
    if sw:
        for c in sw["cases"]:
            items.append((f"shear wave\nH={c['H']}", abs(c["err_vel_pct"])))
    p2 = jload(RUNS / "poiseuille_H60.json")
    if p2:
        items.append(("Poiseuille 2D\nH=60 u_max", abs(p2["u_max_err_pct"])))
    sp = jload(RUNS / "sphere_D40_local.json") or jload(RUNS / "sphere_D40.json")
    if sp:
        flat = {k: v for k, v in sp.items() if not isinstance(v, (list, dict))}
        if "err_pct" in flat:
            items.append(("sphere Re=100 D40\nCd", abs(flat["err_pct"])))
    claimed = jload(VER / "sphere_re100/result.json")
    if not items:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    names = [n for n, _ in items]
    vals = [v for _, v in items]
    bars = ax.bar(names, vals, color="#3b7dd8", width=0.55)
    ax.axhline(3.0, color="#c0392b", ls="--", lw=1.2, label="3% acceptance gate (benchmarks/STATUS.md)")
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3g}%", (b.get_x() + b.get_width() / 2, v), ha="center",
                    va="bottom", fontsize=8)
    if claimed:
        cd = claimed["grids"]["D40"]["err_pct"]
        ax.axhline(abs(cd), color="#27ae60", ls=":", lw=1.4,
                   label=f"sphere D40 claim {cd:.2f}% (result.json)")
    ax.set_yscale("log")
    ax.set_ylabel("|error| vs analytic / reference  [%]")
    ax.set_title("Measured vs acceptance gate — AUDIT-RUNTEST-6 rerun")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "bench_gate.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main() -> int:
    out = {}
    for fn in (fig_ladder, fig_gate):
        try:
            p = fn()
        except Exception as exc:  # noqa: BLE001
            print(f"{fn.__name__} failed: {exc}")
            p = None
        out[fn.__name__] = str(p) if p else None
        print(fn.__name__, "->", p)
    (OUT / "figures.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())