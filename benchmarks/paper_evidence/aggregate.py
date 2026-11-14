#!/usr/bin/env python3
"""Aggregate the paper-evidence item JSONs into one summary, table set and figures.

Reads   ``runs/paper_evidence/<ITEM>.json``   (written by run_evidence.py)
Writes  ``artifacts/paper_evidence/summary.json``
        ``artifacts/paper_evidence/tables.md``         (all numbers, generated)
        ``artifacts/paper_evidence/figures/*.png``
        ``docs/PAPER_EVIDENCE_BUNDLE.md``              (prose template + tables)

No number in the generated documents is typed by hand: every cell comes from a
solver return value stored in an item JSON.  The pytest gate
``tests/test_paper_evidence_bundle.py`` re-derives the verdicts from the same
JSONs, so a hand edit of the summary would be caught by re-running the gate.

Usage:
    python benchmarks/paper_evidence/aggregate.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IN_DEFAULT = REPO / "runs" / "paper_evidence"
OUT_DEFAULT = REPO / "artifacts" / "paper_evidence"
TEMPLATE = REPO / "benchmarks" / "paper_evidence" / "report_template.md"
DOC = REPO / "docs" / "PAPER_EVIDENCE_BUNDLE.md"

CATEGORY_LABEL = {
    "analytic": "解析解 analytic",
    "literature": "文献 literature",
    "cross_solver": "交叉求解器 cross-solver",
}


# --------------------------------------------------------------------------- #
# loading / summarising
# --------------------------------------------------------------------------- #


def load_items(in_dir: Path) -> dict[str, dict]:
    items = {}
    for f in sorted(in_dir.glob("*.json")):
        if f.name in ("FAILURES.json",):
            continue
        try:
            payload = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if "item" not in payload or "rows" not in payload:
            continue  # raw sub-artifacts (e.g. cavity runner output) are not items
        items[payload["item"]] = payload
    return items


def summarise(items: dict[str, dict]) -> dict:
    per_item = []
    for key, it in items.items():
        rows = it["rows"]
        fails = [r for r in rows if not r["pass"]]
        three_pct_rows = [r for r in rows if abs(r["threshold_pct"] - 3.0) < 1e-9]
        per_item.append(
            {
                "item": key,
                "category": it["category"],
                "category_label": CATEGORY_LABEL.get(it["category"], it["category"]),
                "title": it["title"],
                "anchor_kind": it.get("anchor", {}).get("kind"),
                "anchor_citation": it.get("anchor", {}).get("citation"),
                "anchor_source": it.get("anchor", {}).get("source"),
                "n_rows": len(rows),
                "n_rows_fail": len(fails),
                "max_err_pct": max(r["err_pct"] for r in rows) if rows else None,
                "max_err_pct_3pct_rows": (max(r["err_pct"] for r in three_pct_rows)
                                          if three_pct_rows else None),
                "all_under_3pct": all(r["err_pct"] < 3.0 for r in three_pct_rows)
                if three_pct_rows else None,
                "failing_rows": [r["quantity"] for r in fails],
                "observed_order": it.get("observed_order"),
                "noise_floor_pct": it.get("noise_floor_pct"),
                "settings": it.get("settings"),
                "repro_cmd": it.get("repro_cmd"),
                "wall_seconds": it.get("wall_seconds"),
                "rows": rows,
            }
        )
    order = {k: i for i, k in enumerate(
        ["A1_poiseuille_2d", "A2_square_duct_3d", "A3_shear_wave_viscosity",
         "L1_cavity_ghia_re100", "L2_cavity_ghia_re400", "L3_cylinder_array_sangani",
         "X1_cross_solver_cavity"])}
    per_item.sort(key=lambda r: order.get(r["item"], 99))
    cats = {}
    for r in per_item:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    return {
        "schema": "paper-evidence-bundle/1.0",
        "n_items": len(per_item),
        "items_per_category": cats,
        "all_rows_pass": all(r["n_rows_fail"] == 0 for r in per_item),
        "items": per_item,
    }


# --------------------------------------------------------------------------- #
# markdown tables
# --------------------------------------------------------------------------- #


def _f(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float):
        if x == 0:
            return "0"
        if abs(x) < 1e-3 or abs(x) >= 1e4:
            return f"{x:.3e}"
        return f"{x:.{nd}f}"
    return str(x)


def md_overview(summary: dict) -> str:
    lines = [
        "| # | 基准 item | 类别 | 判据条目 | 最大误差 % | 全部 <3%? | 观测阶 | 锚点出处 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, it in enumerate(summary["items"], 1):
        ok = it["all_under_3pct"]
        flag = "✅ 是" if ok else ("❌ 否" if ok is False else "—")
        lines.append(
            f"| {i} | `{it['item']}` | {it['category_label']} | {it['n_rows']} | "
            f"{_f(it['max_err_pct_3pct_rows'] if it['max_err_pct_3pct_rows'] is not None else it['max_err_pct'], 3)} | "
            f"{flag} | {_f(it['observed_order'])} | {it['anchor_citation']} |"
        )
    return "\n".join(lines)


def md_item_tables(summary: dict) -> str:
    out = []
    for i, it in enumerate(summary["items"], 1):
        out.append(f"### {i}. `{it['item']}` — {it['title']}\n")
        out.append(f"- 类别：**{it['category_label']}**  ")
        out.append(f"- 锚点（external anchor）：{it['anchor_source']}  ")
        out.append(f"- 出处（citation）：{it['anchor_citation']}  ")
        out.append(f"- 复现命令：`{it['repro_cmd']}`  ")
        if it.get("observed_order") is not None:
            oo = it["observed_order"]
            if isinstance(oo, dict):
                oo = ", ".join(f"{k}: {_f(v, 2)}" for k, v in oo.items())
            else:
                oo = _f(oo, 2)
            out.append(f"- 观测收敛阶（两档）：{oo}  ")
        if it.get("noise_floor_pct"):
            out.append(f"- 噪声地板探针：`{json.dumps(it['noise_floor_pct'], ensure_ascii=False)}`  ")
        out.append(f"- 墙钟：{it['wall_seconds']} s（本次运行）\n")
        out.append("| 量 | ref（基准） | got（本代码） | 误差 % | 门限 % | 判定 |")
        out.append("|---|---|---|---|---|---|")
        for r in it["rows"]:
            out.append(
                f"| {r['quantity']} | {_f(r['ref'])} | {_f(r['got'])} | "
                f"{_f(r['err_pct'], 3)} | {_f(r['threshold_pct'], 1)} | "
                f"{'✅ pass' if r['pass'] else '❌ fail'} |"
            )
        out.append("")
    return "\n".join(out)


def md_verdict(summary: dict) -> str:
    n = summary["n_items"]
    cats = summary["items_per_category"]
    ok = summary["all_rows_pass"]
    worst = max(summary["items"], key=lambda it: it["max_err_pct"])
    lines = [
        f"- 基准总数：**{n}**（解析 {cats.get('analytic', 0)} + 文献 {cats.get('literature', 0)}"
        f" + 交叉求解器 {cats.get('cross_solver', 0)}），判据要求 ≥6 → "
        f"{'✅ 达标' if n >= 6 else '❌ 未达标'}",
        f"- 每个基准都至少两档分辨率/参数点 → "
        f"{'✅' if all(_has_two_resolutions(it) for it in summary['items']) else '❌'}",
        f"- 全部 {sum(it['n_rows'] for it in summary['items'])} 条判据条目："
        f"{'✅ 全部在各自门限内' if ok else '❌ 存在超门限条目'}",
        f"- 单条最大误差：{_f(worst['max_err_pct'], 3)} %（`{worst['item']}`）→ "
        f"是否 <3%：{'✅ 是' if worst['max_err_pct'] < 3.0 else '❌ 否'}",
    ]
    return "\n".join(lines)


def _has_two_resolutions(item: dict) -> bool:
    st = item.get("settings") or {}
    for key in ("resolutions", "cavity_sizes"):
        v = st.get(key)
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            return True
        if isinstance(v, dict) and all(len(x) >= 2 for x in v.values()):
            return True
    return False


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def make_figures(items: dict[str, dict], summary: dict, figdir: Path) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)
    figs: list[dict] = []

    # --- 1. per-row error overview ---------------------------------------
    labels, errs, colors = [], [], []
    palette = {"analytic": "tab:blue", "literature": "tab:green", "cross_solver": "tab:red"}
    for it in summary["items"]:
        for r in it["rows"]:
            labels.append(f"{it['item'].split('_')[0]}·{r['quantity']}")
            errs.append(max(r["err_pct"], 1e-6))
            colors.append(palette.get(it["category"], "gray"))
    fig, ax = plt.subplots(figsize=(9, 0.30 * len(labels) + 1.6))
    y = range(len(labels))
    ax.barh(list(y), errs, color=colors)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xscale("log")
    ax.axvline(3.0, color="k", ls="--", lw=1.2, label="3 % acceptance")
    ax.set_xlabel("relative error vs reference [%]")
    ax.set_title("per-quantity error against the reference value (log scale)")
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    p = figdir / "fig1_error_overview.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    figs.append({"name": "fig1_error_overview.png", "path": p,
                 "caption": "逐项误差总览：每个基准的每条判据 vs 其基准值的相对误差（对数轴），"
                            "虚线为 3% 验收线。"})

    # --- 2. Poiseuille profile -------------------------------------------
    a1 = items.get("A1_poiseuille_2d")
    if a1:
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        first = a1["per_resolution"][0]
        ax.plot(first["u_exact"], first["y"], "k--", lw=1.4, label="analytic parabola")
        for pr in a1["per_resolution"]:
            ax.plot(pr["u_num"], pr["y"], "o-", ms=3, lw=1.0,
                    label=f"LBM H={pr['height_cells']} (max err {pr['err_profile_pct']:.3f}%)")
        ax.set_xlabel("u_x [lattice]")
        ax.set_ylabel("y [lattice node]")
        ax.set_title("A1 plane Poiseuille: profile vs closed form")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = figdir / "fig2_poiseuille_profile.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        figs.append({"name": "fig2_poiseuille_profile.png", "path": p,
                     "caption": "A1 平面 Poiseuille：两档分辨率速度剖面与解析抛物线对照。"})

    # --- 3. duct centre diagonal ----------------------------------------
    a2 = items.get("A2_square_duct_3d")
    if a2:
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        for pr in a2["per_resolution"]:
            ax.plot(pr["diag_coord"], pr["diag_exact"], "k--", lw=1.2,
                    label="Boussinesq series (analytic)" if pr is a2["per_resolution"][0] else None)
            ax.plot(pr["diag_coord"], pr["diag_num"], "o-", ms=3,
                    label=f"LBM L={pr['resolution']} (err {pr['err_profile_pct']:.3f}%)")
        ax.set_xlabel("cross-section coordinate along the diagonal (wall→wall)")
        ax.set_ylabel("u_x [lattice]")
        ax.set_title("A2 square duct: diagonal profile vs series solution")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = figdir / "fig3_duct_diagonal.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        figs.append({"name": "fig3_duct_diagonal.png", "path": p,
                     "caption": "A2 方管 3D Poiseuille：截面对角线速度剖面与 Boussinesq 级数解对照（含角区）。"})

    # --- 4. convergence plot ---------------------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    if a1:
        xs = [p["resolution"] for p in a1["per_resolution"]]
        ys = [p["err_profile_pct"] for p in a1["per_resolution"]]
        ax.loglog(xs, ys, "o-", label=f"A1 Poiseuille (order {_f(a1['observed_order'], 2)})")
    if a2:
        xs = [p["resolution"] for p in a2["per_resolution"]]
        ys = [p["err_profile_pct"] for p in a2["per_resolution"]]
        ax.loglog(xs, ys, "s-", label=f"A2 duct (order {_f(a2['observed_order'], 2)})")
    for key, mk in (("L1_cavity_ghia_re100", "^"), ("L2_cavity_ghia_re400", "v")):
        it = items.get(key)
        if it:
            xs = [p["resolution"] for p in it["per_resolution"]]
            ys = [p["max_abs_dev_pct"] for p in it["per_resolution"]]
            ax.loglog(xs, ys, mk + "-", label=f"{key.split('_')[0]} Ghia (order "
                                              f"{_f(it['observed_order'], 2)})")
    ax.axhline(3.0, color="k", ls="--", lw=1.0, label="3 % line")
    ax.set_xlabel("resolution (cells across the gap / side)")
    ax.set_ylabel("error [%]")
    ax.set_title("resolution study: error vs grid")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = figdir / "fig4_convergence.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    figs.append({"name": "fig4_convergence.png", "path": p,
                 "caption": "两档分辨率误差对照（解析 A1/A2 与文献 L1/L2），附实测收敛阶与 3% 线。"})

    # --- 5. permeability parity ------------------------------------------
    l3 = items.get("L3_cylinder_array_sangani")
    if l3:
        fig, ax = plt.subplots(figsize=(4.8, 4.4))
        for phi in sorted({p["phi"] for p in l3["per_resolution"]}):
            sel = [p for p in l3["per_resolution"] if p["phi"] == phi]
            ax.loglog([p["k_ref"] for p in sel], [p["k_sim"] for p in sel], "o-",
                      label=f"phi={phi} (max err {max(p['err_pct'] for p in sel):.2f}%)")
        lo = min(p["k_ref"] for p in l3["per_resolution"]) * 0.8
        hi = max(p["k_ref"] for p in l3["per_resolution"]) * 1.25
        ax.loglog([lo, hi], [lo, hi], "k--", lw=1.0, label="perfect agreement")
        ax.set_xlabel("k_ref from Sangani & Acrivos (1982) table")
        ax.set_ylabel("k_sim (Darcy, lattice)")
        ax.set_title("L3 cylinder array permeability: simulation vs literature")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = figdir / "fig5_permeability_parity.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        figs.append({"name": "fig5_permeability_parity.png", "path": p,
                     "caption": "L3 周期圆柱方阵渗透率：k_sim 与 Sangani–Acrivos 文献表 k_ref 的对角线对照。"})

    # --- 6. cavity centreline vs Ghia (cross-solver raw) ------------------
    x1 = items.get("X1_cross_solver_cavity")
    if x1 and x1.get("cavity_profiles"):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
        done_u, done_v = False, False
        for name, prof in x1["cavity_profiles"].items():
            if not prof.get("y_ref"):
                continue
            solver = name.split("_")[0].upper()
            axes[0].plot(prof["u_ref"], prof["y_ref"], "k--", lw=1.4,
                         label="Ghia 1982" if not done_u else None)
            axes[0].plot(prof["u_lbm"], prof["y_ref"], "-", lw=1.0, label=solver)
            done_u = True
            if prof.get("x_ref"):
                axes[1].plot(prof["x_ref"], prof["v_ref"], "k--", lw=1.4,
                             label="Ghia 1982" if not done_v else None)
                axes[1].plot(prof["x_ref"], prof["v_lbm"], "-", lw=1.0, label=solver)
                done_v = True
        axes[0].set_xlabel("u/U_lid"); axes[0].set_ylabel("y"); axes[0].set_title("vertical centreline")
        axes[1].set_xlabel("v/U_lid"); axes[1].set_ylabel("x"); axes[1].set_title("horizontal centreline")
        for a in axes:
            a.legend(fontsize=7)
        fig.suptitle("X1 cross-solver: LBM / FVM / FEM vs Ghia (1982), Re=100")
        fig.tight_layout()
        p = figdir / "fig6_cavity_vs_ghia.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        figs.append({"name": "fig6_cavity_vs_ghia.png", "path": p,
                     "caption": "X1 交叉求解器：LBM / FVM / FEM 三个独立离散在 Re=100 的中心线速度与 Ghia(1982) 表值对照。"})

    # --- 7. cavity fields, three solvers ---------------------------------
    fields = (x1 or {}).get("fields") or {}
    if fields.get("npz"):
        import numpy as np

        npz = np.load(REPO / fields["npz"])
        fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))
        vmax = 0.0
        panels = [("lbm", "LBM (D2Q9)"), ("fvm", "FVM (psi-omega)"), ("fem", "FEM (Q2-Q1)")]
        mags = {k: np.sqrt(npz[f"{k}_u"] ** 2 + npz[f"{k}_v"] ** 2) for k, _ in panels}
        vmax = max(float(m.max()) for m in mags.values())
        for ax, (k, title) in zip(axes, panels):
            im = ax.imshow(mags[k], origin="lower", cmap="viridis", vmin=0, vmax=vmax)
            ax.set_title(f"{title}, n={int(npz['n'])}, Re={float(npz['re']):.0f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axes, shrink=0.85, label="|u| / u_lid")
        fig.suptitle("same physical cavity, three independent discretisations (field magnitude)")
        fig.tight_layout()
        p = figdir / "fig7_cavity_fields.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        figs.append({"name": "fig7_cavity_fields.png", "path": p,
                     "caption": "同一方腔（n=64, Re=100）的速度幅值云图，由 LBM/FVM/FEM 三套独立离散分别算出。"})

    return figs


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, default=IN_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    items = load_items(args.in_dir)
    if not items:
        print(f"no item JSONs in {args.in_dir}; run run_evidence.py first")
        return 1
    summary = summarise(items)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")

    overview = md_overview(summary)
    verdict = md_verdict(summary)
    tables = "\n\n".join([
        "## 4. 逐项误差对照总表\n", overview,
        "\n## 5. 判据汇总（程序化生成）\n", verdict,
        "\n## 6. 逐基准明细（ref vs got + 误差）\n", md_item_tables(summary),
    ])
    (args.out_dir / "tables.md").write_text(tables, encoding="utf-8")

    figs = [] if args.no_figures else make_figures(items, summary, args.out_dir / "figures")
    (args.out_dir / "figures.json").write_text(
        json.dumps([{k: str(v) for k, v in f.items()} for f in figs], indent=1,
                   ensure_ascii=False), encoding="utf-8")

    # docs page = hand-written prose template with the generated tables injected
    if TEMPLATE.exists():
        prose = TEMPLATE.read_text(encoding="utf-8")
        DOC.write_text(prose.replace("<!--TABLES-->", tables), encoding="utf-8")
        print(f"docs -> {DOC.relative_to(REPO)}")

    print(json.dumps({"n_items": summary["n_items"],
                      "items_per_category": summary["items_per_category"],
                      "all_rows_pass": summary["all_rows_pass"],
                      "figures": [f["name"] for f in figs]}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())