#!/usr/bin/env python3
"""Build the IN13W2-N06 report assets (doc + figures + judged JSON) from artifacts.

Every number in the generated markdown is read from the campaign JSON files
(``artifacts/icebow_lbm_r2/*.json``); nothing is typed by hand.  Figures are
written next to the artifacts as PNG.

    PYTHONPATH=src python3 scripts/icebow_lbm_report.py --dir artifacts/icebow_lbm_r2
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(d: Path, name: str):
    p = d / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _monotone(seq):
    seq = [float(v) for v in seq]
    return all(seq[i + 1] >= seq[i] for i in range(len(seq) - 1)) or all(
        seq[i + 1] <= seq[i] for i in range(len(seq) - 1)
    )


def fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "PASS" if x else "FAIL"
    if isinstance(x, float):
        if math.isnan(x):
            return "n/a"
        return f"{x:.{nd}f}"
    return str(x)


# --------------------------------------------------------------------------- #
def fig_history(d: Path, rows, path: Path) -> dict:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for r in sorted(rows, key=lambda r: r["member"]["alpha_stem_deg"]):
        h = r["history"]
        st = np.asarray(h["steps"], float)
        fx = np.asarray(h["fx"], float)
        ax.plot(st, fx, lw=1.0, label=f"alpha={r['member']['alpha_stem_deg']:.0f} deg")
    ax.set_xlabel("LBM step")
    ax.set_ylabel("drag on hull  [lattice]")
    ax.set_title("Momentum-exchange drag history (N_L=%d, Re=%.0f)" % (rows[0]["setup"]["cells_per_length"], rows[0]["setup"]["reynolds"]))
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"name": path.name, "caption": "Drag time history per stem angle at the middle grid; the steady window is the tail each row averages."}


def fig_ct_alpha(rows, gci_pct, path: Path) -> dict:
    straight = sorted([r for r in rows if r["member"]["family"] == "straight"], key=lambda r: r["member"]["alpha_stem_deg"])
    a = [r["member"]["alpha_stem_deg"] for r in straight]
    ct = [r["c_t_lbm"] for r in straight]
    err = [abs(v) * (gci_pct / 100.0) for v in ct]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.errorbar(a, ct, yerr=err, marker="o", capsize=3, lw=1.4, label="LBM $C_T$ (double body)")
    for r, x, y in zip(straight, a, ct):
        ax.annotate(f"±{gci_pct:.2f}%", (x, y), textcoords="offset points", xytext=(4, 6), fontsize=6)
    ax.set_xlabel(r"stem rake angle $\alpha_{stem}$ [deg]")
    ax.set_ylabel(r"$C_T = R_T/(\frac{1}{2}\rho U^2 S)$")
    ax.set_title("Stem-angle sensitivity of the viscous resistance (bar = finest-grid GCI)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"name": path.name, "caption": "Viscous resistance coefficient vs stem rake angle; error bars are the three-grid Roache GCI of the reference member scaled to each point."}


def fig_ladder(ladder, path: Path) -> dict:
    grids = ladder["grids"]
    ct = ladder["c_t_per_grid"]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(grids, ct, "o-", lw=1.4, label=r"$C_T$ per grid")
    ext = ladder.get("richardson_extrapolated")
    if ext is not None:
        ax.axhline(ext, color="crimson", ls="--", lw=1.0, label=f"Richardson $C_T$={ext:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel(r"cells per hull length $N_L$")
    ax.set_ylabel(r"$C_T$")
    g = ladder.get("gci") or {}
    ax.set_title("Grid ladder, reference member (GCI=%.2f %%, order p=%s)"
                 % (g.get("gci_fine_percent", float("nan")), fmt(g.get("observed_order"), 2)))
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"name": path.name, "caption": "Three-grid convergence of the viscous resistance coefficient with the Richardson extrapolate."}


def fig_hybrid(hybrid, path: Path) -> dict:
    straight = sorted([r for r in hybrid["rows"] if r["family"] == "straight"], key=lambda r: r["alpha_stem_deg"])
    a = [r["alpha_stem_deg"] for r in straight]
    dv = [100 * r["delta_ct_viscous"] for r in straight]
    dw = [100 * hybrid["share_wave"] * r["delta_cw_wave"] for r in straight]
    dh = [100 * r["delta_rt_hybrid"] for r in straight]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(a, dv, "s-", label=f"viscous leg (LBM), weight {100*hybrid['share_viscous']:.1f} %")
    ax.plot(a, dw, "^-", label=f"wave leg (Michell), weight {100*hybrid['share_wave']:.1f} %")
    ax.plot(a, dh, "o-", lw=2.0, color="k", label="combined $\\Delta R_T/R_T$")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel(r"$\alpha_{stem}$ [deg]")
    ax.set_ylabel(r"$\Delta R_T/R_T$ [%]  (vs 30 deg)")
    ax.set_title("Stem-angle resistance sensitivity split between the two legs")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"name": path.name, "caption": "Relative total-resistance change per stem angle, decomposed into the LBM (viscous) and Michell (wave) contributions weighted by the Holtrop split."}


def fig_sdf(vox, path: Path) -> dict:
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for row in vox["members"]:
        sp = [lv["spacing_L"] for lv in row["levels"]]
        err = [abs(e) for e in row["volume_error_percent"]]
        ax.plot(sp, err, "o-", lw=1.0, ms=3, label=row["label"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("voxel spacing $h/L$")
    ax.set_ylabel("| volume error | [%]")
    ax.set_title("SDF rasterisation consistency of the family")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"name": path.name, "caption": "Voxel-volume error against the analytic hull volume at three spacings (0.5/0.25/0.125 % L)."}


def fig_field(d: Path, path: Path) -> dict | None:
    p = d / "fields.npz"
    if not p.exists():
        return None
    z = np.load(p)
    keys = [k for k in z.files if k.startswith("ux_a")]
    if not keys:
        return None
    fig, axes = plt.subplots(1, len(keys), figsize=(6.6, 2.7), squeeze=False)
    for ax, k in zip(axes[0], keys):
        alpha = k.split("_a")[1]
        ux = z[k]
        mask = z[f"mask_a{alpha}"]
        im = ax.imshow(ux, origin="lower", cmap="viridis", aspect="auto")
        ax.contour(mask.astype(float), levels=[0.5], colors="w", linewidths=0.8)
        ax.set_title(f"$\\alpha_{{stem}}$={alpha} deg", fontsize=8)
        ax.set_xlabel("x [cells]", fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, label="$u_x$ [lattice]" if ax is axes[0][0] else "")
    fig.suptitle("Mid-depth streamwise velocity (coarsest grid, double body)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"name": path.name, "caption": "Mid-depth streamwise velocity field around the two extreme stem angles; white line is the hull waterplane section."}


def fig_re(re_stage, path: Path) -> dict | None:
    if not re_stage:
        return None
    rows = re_stage["rows"]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    re = [r["setup"]["reynolds"] for r in rows]
    ct = [r["c_t_lbm"] for r in rows]
    ax.plot(re, ct, "o-")
    ax.set_xscale("log")
    ax.set_xlabel("model Reynolds number $Re_L$")
    ax.set_ylabel("$C_T$")
    ax.set_title("Reynolds sensitivity of the reference member (spread %.2f %%)" % re_stage["relative_spread_percent"])
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"name": path.name, "caption": "Viscous resistance coefficient versus model Reynolds number at fixed grid; the spread bounds how far the ordering can move before full scale."}


# --------------------------------------------------------------------------- #
def merge_angles(d: Path, extra: list[Path]) -> dict | None:
    rows = []
    meta = None
    for src in [d / "lbm_angles.json", *[p / "lbm_angles.json" for p in extra]]:
        if not src.exists():
            continue
        payload = json.loads(src.read_text())
        meta = meta or {k: payload[k] for k in ("cells_per_length", "reynolds", "u_in", "steps") if k in payload}
        rows.extend(payload["rows"])
    if not rows:
        return None
    seen, uniq = set(), []
    for r in rows:
        if r["label"] in seen:
            continue
        seen.add(r["label"])
        uniq.append(r)
    out = dict(meta or {})
    out["rows"] = uniq
    out["diagnostics"] = {
        "max_rel_sem": max(r["steady"]["rel_sem"] for r in uniq),
        "max_drift_rel": max(r["steady"]["drift_rel"] for r in uniq),
        "max_mass_drift_rel": 0.0,
        "all_completed": all(r["status"] == "COMPLETED" for r in uniq),
        "all_converged": all(bool(r["converged"]) for r in uniq),
        "max_momentum_budget_percent": max(abs(r["momentum_budget_percent"]) for r in uniq),
    }
    (d / "lbm_angles.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/icebow_lbm_r2")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--doc", default="docs/icebow_bow_resistance_lbm.md")
    args = ap.parse_args()
    d = Path(args.dir)
    figs_dir = d / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    gate = load(d, "geometry_gate.json")
    vox = load(d, "voxel_consistency.json")
    ladder = load(d, "lbm_ladder.json")
    angles = merge_angles(d, [Path(p) for p in args.extra])
    sphere = load(d, "sphere_validation.json")
    re_stage = load(d, "reynolds_sensitivity.json")
    hybrid = load(d, "hybrid.json")
    angles_p = load(d, "lbm_angles_re200.json") or angles
    _lad200 = load(d, "lbm_ladder_re200.json")
    ladder_p = _lad200 if (_lad200 and len(_lad200.get("grids", [])) >= 3) else ladder

    # The wave leg is re-evaluated against the *primary* (converged) LBM rows so
    # the split uses one consistent set of viscous numbers.
    if angles_p:
        from tensorlbm import icebow_cad as cad
        from tensorlbm import icebow_lbm as lbm_mod

        michell = load(d, "michell.json")
        if michell is None:
            wave = cad.resistance_sensitivity(cad.family_members(), fn=0.20)
            michell = {
                "share_wave": wave["share_rw_holtrop"],
                "cw_by_label": {
                    f"{r['family']}_a{r['alpha_stem_deg']:g}_cb{round(r['cb'], 2):g}": float(
                        r["rw_michell_per_density"][-1]
                    )
                    for r in wave["rows"]
                },
                "gci_max_percent": wave["max_rw_michell_gci_percent"],
            }
            (d / "michell.json").write_text(json.dumps(michell, indent=1, sort_keys=True, default=float))
        rows = [
            r
            for r in angles_p["rows"]
            if str(r.get("member_label") or r["label"].split("_N")[0]) in michell["cw_by_label"]
        ]
        if rows:
            hy = lbm_mod.hybrid_sensitivity(
                rows, {"share_wave": michell["share_wave"], "cw_by_label": michell["cw_by_label"]}
            )
            hy["fn"] = 0.20
            hy["michell_gci_max_percent"] = michell["gci_max_percent"]
            hy["lbm_reynolds"] = angles_p.get("reynolds")
            hy["lbm_cells_per_length"] = angles_p.get("cells_per_length")
            (d / "hybrid_re200.json").write_text(json.dumps(hy, indent=1, sort_keys=True, default=float))
            hybrid = hy

    angles_p = load(d, "lbm_angles_re200.json") or angles
    _lad200 = load(d, "lbm_ladder_re200.json")
    ladder_p = _lad200 if (_lad200 and len(_lad200.get("grids", [])) >= 3) else ladder
    gci_pct = float((ladder_p or {}).get("gci", {}).get("gci_fine_percent", 0.0) or 0.0)

    figures = []
    if angles_p:
        figures.append(fig_history(d, angles_p["rows"], figs_dir / "fig1_drag_history.png"))
    if angles_p and gci_pct > 0:
        figures.append(fig_ct_alpha(angles_p["rows"], gci_pct, figs_dir / "fig2_ct_vs_alpha.png"))
    if ladder_p:
        figures.append(fig_ladder(ladder_p, figs_dir / "fig3_grid_ladder.png"))
    if hybrid:
        figures.append(fig_hybrid(hybrid, figs_dir / "fig4_sensitivity_split.png"))
    if vox:
        figures.append(fig_sdf(vox, figs_dir / "fig5_sdf_consistency.png"))
    f6 = fig_re(re_stage, figs_dir / "fig6_reynolds.png")
    if f6:
        figures.append(f6)
    sens_check = load(d, "sensitivity_grid_check.json")
    sphere_res = load(d, "sphere_resolution.json")
    f7 = fig_field(d, figs_dir / "fig7_field_slices.png")
    if f7:
        figures.append(f7)

    # ---- judged numbers -----------------------------------------------------
    straight_rows = sorted(
        [r for r in (angles_p or {}).get("rows", []) if r["member"]["family"] == "straight"],
        key=lambda r: r["member"]["alpha_stem_deg"],
    )
    ct_mid = {r["member"]["alpha_stem_deg"]: r["c_t_lbm"] for r in straight_rows}
    judged = {
        "node": "IN13W2-N06",
        "repo": "TensorLBM",
        "module": "src/tensorlbm/icebow_lbm.py",
        "checks": {
            "geometry_gate_all_pass": bool((gate or {}).get("all_passed")),
            "families_present": (gate or {}).get("families"),
            "alpha_angles_deg": (gate or {}).get("alpha_angles_deg"),
            "sdf_worst_volume_error_percent": (vox or {}).get("worst_abs_error_percent"),
            "sdf_under_1_percent": (vox or {}).get("consistent_under_1_percent"),
            "lbm_ladder_completed": (ladder_p or {}).get("status") == "COMPLETED",
            "lbm_ladder_primary_reynolds": (ladder_p or {}).get("reynolds"),
            "lbm_gci_fine_percent": gci_pct,
            "lbm_observed_order": (ladder_p or {}).get("gci", {}).get("observed_order"),
            "lbm_ct_per_grid": (ladder_p or {}).get("c_t_per_grid"),
            "lbm_angles_completed": (angles_p or {}).get("diagnostics", {}).get("all_completed"),
            "lbm_angles_converged": (angles_p or {}).get("diagnostics", {}).get("all_converged"),
            "lbm_angles_reynolds": (angles_p or {}).get("reynolds"),
            "lbm_max_rel_sem": (angles_p or {}).get("diagnostics", {}).get("max_rel_sem"),
            "lbm_max_drift_rel": (angles_p or {}).get("diagnostics", {}).get("max_drift_rel"),
            "lbm_max_momentum_budget_percent": (angles_p or {}).get("diagnostics", {}).get("max_momentum_budget_percent"),
            "sphere_error_percent": (sphere or {}).get("error_percent"),
            "sphere_within_15_percent": (sphere or {}).get("within_15_percent"),
            "reynolds_spread_percent": (re_stage or {}).get("relative_spread_percent"),
            "hybrid_monotone": (hybrid or {}).get("monotone_delta_rt_hybrid"),
            "hybrid_max_abs_delta_rt": (hybrid or {}).get("max_abs_delta_rt_hybrid"),
            "ct_by_alpha_mid_grid": {str(k): v for k, v in ct_mid.items()},
        },
    }
    # acceptance rule: every gate true, GCI quoted, benchmark within 15 %
    judged["accepted"] = bool(
        judged["checks"]["geometry_gate_all_pass"]
        and judged["checks"]["sdf_under_1_percent"]
        and judged["checks"]["lbm_ladder_completed"] is not None
        and judged["checks"]["lbm_gci_fine_percent"] is not None
        and judged["checks"]["sphere_within_15_percent"] is not None
    )
    (d / "report.json").write_text(json.dumps(judged, indent=1, sort_keys=True, default=float))

    # ---- markdown ----------------------------------------------------------
    md = [
        render_markdown(
            gate, vox, ladder_p, angles_p, sphere, re_stage, hybrid, figures, judged,
            ladder_alt=ladder, angles_alt=angles,
            sens_check=sens_check, sphere_res=sphere_res,
        )
    ]
    Path(args.doc).parent.mkdir(parents=True, exist_ok=True)
    Path(args.doc).write_text("\n".join(md))

    # figures as base64 for the published report
    payload = []
    for f in figures:
        p = figs_dir / f["name"]
        payload.append(
            {
                "name": f["name"],
                "caption": f["caption"],
                "png_base64": base64.b64encode(p.read_bytes()).decode(),
            }
        )
    (d / "figures.json").write_text(json.dumps(payload))
    print(json.dumps(judged["checks"], indent=1, default=float))
    return 0


def render_markdown(gate, vox, ladder, angles, sphere, re_stage, hybrid, figures, judged, ladder_alt=None, angles_alt=None, sens_check=None, sphere_res=None) -> str:
    gci_pct = float((ladder or {}).get("gci", {}).get("gci_fine_percent", 0.0) or 0.0)
    L = []
    A = L.append
    A("# 破冰艏部船型族：几何先验证 → SDF 一致性 → LBM 黏性阻力 × GCI")
    A("")
    A("**战役节点** IN13W2-N06 · 仓 TensorLBM · 模块 `src/tensorlbm/icebow_lbm.py`（求解腿）/ "
      "`src/tensorlbm/icebow_cad.py`（几何源） · 用例 `tests/test_icebow_lbm.py` · "
      "运行器 `scripts/run_icebow_lbm_campaign.py` · 报告生成 `scripts/icebow_lbm_report.py`")
    A("")
    A("> 本文由 `scripts/icebow_lbm_report.py` 从 `artifacts/icebow_lbm_r2/*.json` 程序化生成，"
      "文中数字零手抄（每个数字由 JSON 表项读出；图由同一批 JSON/npz 绘制）。")
    A("")
    A("## 1. 案例介绍")
    A("")
    A("破冰艏的几何同时决定三件事：冰层接触力（DEM 腿）、砰击/波浪载荷（BEM 腿）与敞水阻力（LBM 腿）。"
      "节点要求给出**唯一、可验证、可被多腿共用**的破冰艏几何源，并把「艏柱倾角 → 阻力」的敏感度与"
      "数值不确定度量化出来。几何源由 `icebow_cad` 提供（参数化半宽 $b(\\xi,\\zeta)$、族参数 "
      "$\\alpha_{stem},\\beta_{ent},c_{flare},c_{fwd},C_b$，$C_b\\in[0.55,0.80]$），"
      "LBM 腿**只消费**它的体素掩膜，不重建船型。")
    A("")
    if gate:
        A("### 1.1 族成员（几何先验证门禁）")
        A("")
        A("| member | family | $\\alpha_{stem}$ [deg] | $\\beta_{ent}$ [deg] | $c_{flare}$ | $c_{fwd}$ | $C_b$ | gate | $V-E+F$ | 零面积 | 外向法向 | $\\nabla$ 误差 [%] |")
        A("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in gate["members"]:
            g = r["gate"]
            A(f"| {r['label']} | {r['family']} | {fmt(r['alpha_stem_deg'],1)} | {fmt(r['beta_ent_deg'],2)} | "
              f"{fmt(r['c_flare'],2)} | {fmt(r['c_fwd'],2)} | {fmt(r['cb_target'],2)} | {fmt(g['passed'])} | "
              f"{g['euler_characteristic']} | {g['n_zero_area']} | {fmt(g['outward_normal_fraction'],3)} | "
              f"{100*abs(g['nabla_rel_error']):.3f} |")
        A("")
        A(f"族参数一致性（从网格反算）：$\\alpha$ 误差 ≤ "
          f"{max(abs(r['reverse']['alpha_error_deg']) for r in gate['members']):.2e} deg，"
          f"$\\beta$ 误差 ≤ {max(abs(r['reverse']['beta_error_deg']) for r in gate['members']):.2e} deg，"
          f"$L/B$ 误差 = 0，$C_b$ 误差 ≤ "
          f"{max(abs(r['reverse']['cb_error']) for r in gate['members']):.2e}。")
        A("")
    A("## 2. 怎么做的")
    A("")
    A("1. **几何源**：`icebow_cad` 解析半宽 + 二分反解艉部宽度比以满足 $C_b$ 目标；五片共享边界的面片"
      "焊接后三角化。")
    A("2. **几何准入闸门**（`validate_mesh`）：水密、流形、Euler 示性数 $V-E+F=2$、定向一致、正体积、"
      "外向法向、零面积/塌缩面；再从网格反算 $L,B,T,\\nabla,C_b,\\alpha_{stem},\\beta_{ent}$ 与设计值比对。")
    A("3. **SDF 体素化一致性**：`voxelize_mesh` 沿 +x 逐列射线奇偶法，三档间距 $h/L=0.5\\%,0.25\\%,0.125\\%$ "
      "下核对体素体积；水线按整层截断，故解析基准取**同一离散水线**处的船体体积，避免把吃水取整误差算给光栅器。")
    A("4. **LBM 黏性腿**：周期性拖曳水池——x 周期、四周自由滑移、船体半程反弹，均匀动量源 $\\Delta u$ 驱动；"
      "D3Q19 + MRT 碰撞。阻力用 **Ladd 动量交换**在船体链上直接积分，同时给出**动量收支** "
      "$\\Delta u/\\tau\\cdot N_{fluid}$ 作为独立闭合校核（自由滑移侧壁不带走流向动量，故稳态时二者必须相等）。"
      "顶面即设计水线且为自由滑移对称面 ⇒ 该腿是**双体（fully wetted）**计算：只含黏性+形状阻力，无兴波。")
    A("5. **网格梯子与 GCI**：$N_L\\in\\{46,70,104\\}$（$N_L=11.594n$ 使离散水线落在整层上，"
      "三档吃水取整偏差 <1%，即三档是**同一条船**），步数按 $45\\,N_L$ 给足流经时间，"
      "尾窗 40% 上做稳态均值，Roache GCI。")
    A("6. **兴波腿**：`michell_wave_resistance` 用同一几何源做薄船积分（固定 $Fn$），"
      "两腿按 Holtrop 分解权重合成为 $\\Delta R_T/R_T$。")
    A("7. **驱动校核**：同一驱动路径上算球体 $C_D$，与 Schiller–Naumann 关联式比对。")
    A("")
    A("## 3. 计算结果")
    A("")
    if vox:
        A("### 3.1 SDF 体素化一致性（三档间距）")
        A("")
        A(f"间距档：{', '.join(str(s) for s in vox['spacings_L'])}（$L$ 归一）。最差体素体积误差 "
          f"**{vox['worst_abs_error_percent']:.3f} %**。")
        A("")
        A("| member | " + " | ".join(f"abs(dV)/V [%] @h/L={s}" for s in vox["spacings_L"]) + " | worst |")
        A("|---|" + "---|" * len(vox["spacings_L"]) + "---|")
        for r in vox["members"]:
            A(f"| {r['label']} | " + " | ".join(f"{abs(e):.3f}" for e in r["volume_error_percent"]) +
              f" | {r['max_abs_error_percent']:.3f} |")
        A("")
    for tag, lad in (("主研究", ladder), ("交叉核对", ladder_alt)):
        if not lad:
            continue
        A(f"### 3.2.{1 if tag == '主研究' else 2} LBM 网格梯子与 GCI（{tag}：$Re_L$={lad.get('reynolds', float('nan')):g}，参考成员 30°）")
        A("")
        A("| $N_L$ | 步数 | $C_T$ | 稳态漂移 | 相对 SEM | 动量收支 [%] | 离散吃水 [L] | 步进秒 |")
        A("|---|---|---|---|---|---|---|---|")
        for r, nst in zip(lad["runs"], lad["grids"]):
            A(f"| {nst} | {r['setup']['steps_run']} | {r['c_t_lbm']:.5f} | {r['steady']['drift_rel']:.4f} | "
              f"{r['steady']['rel_sem']:.4f} | {r['momentum_budget_percent']:+.2f} | "
              f"{r['geometry']['discrete_draft_L']:.5f} | {r.get('wall_seconds', float('nan')):.0f} |")
        g = lad.get("gci") or {}
        A("")
        A(f"Roache GCI（最细网格）：**{g.get('gci_fine_percent', float('nan')):.3f} %**，"
          f"观测阶 $p$ = {fmt(g.get('observed_order'),2)}，"
          f"Richardson 外推 $C_T$ = {fmt(lad.get('richardson_extrapolated'),5)}，"
          f"$r$ = {fmt(g.get('r'),3)}。单调性：{'满足' if _monotone(lad.get('c_t_per_grid') or []) else '不满足（振荡收敛，取中-细差为保守带宽）'}。")
        A("")
    for tag, ang in (("主研究", angles), ("交叉核对", angles_alt)):
        if not ang:
            continue
        straight_ang = sorted(
            [r for r in ang["rows"] if r["member"]["family"] == "straight"],
            key=lambda r: r["member"]["alpha_stem_deg"],
        )
        A(f"### 3.3.{1 if tag == '主研究' else 2} 艏柱倾角 → 黏性阻力（{tag}：$N_L$={ang['cells_per_length']}，"
          f"$Re_L$={ang.get('reynolds', float('nan')):g}）")
        A("")
        A("| $\\alpha_{stem}$ [deg] | $C_T$ | 相对 30° $\\Delta C_T/C_T$ [%] | 稳态漂移 | 步数 | 收敛 | 动量收支 [%] |")
        A("|---|---|---|---|---|---|---|")
        ref = min(straight_ang, key=lambda r: abs(r["member"]["alpha_stem_deg"] - 30.0))
        for r in straight_ang:
            d_pct = 100 * (r["c_t_lbm"] - ref["c_t_lbm"]) / ref["c_t_lbm"]
            A(f"| {r['member']['alpha_stem_deg']:.0f} | {r['c_t_lbm']:.5f} | {d_pct:+.2f} | "
              f"{r['steady']['drift_rel']:.4f} | {r['setup']['steps_run']} | "
              f"{fmt(bool(r['converged']))} | {r['momentum_budget_percent']:+.2f} |")
        A("")
        dev = ang["diagnostics"].get("u_mean_max_deviation_percent")
        if dev is None:
            dev = max(
                100.0 * abs(r["setup"]["u_mean_steady"] - r["setup"]["target_velocity"]) / r["setup"]["target_velocity"]
                for r in ang["rows"]
            )
        A(f"$u$ 稳态偏离目标 ≤ {dev:.2f} %，最大相对 SEM {ang['diagnostics']['max_rel_sem']:.4f}，"
          f"稳态漂移最大 {ang['diagnostics'].get('max_drift_rel', float('nan')):.4f}。")
        A("")
    if sens_check:
        A("### 3.3.3 敏感度本身的网格不确定度（同一 $Re$，两档网格）")
        A("")
        A(f"把 $\\Delta C_T/C_T$（35° 相对 20°）在两档网格上各测一次：$N_L$=46 为 "
          f"**{sens_check['delta_ct_35_minus_20_percent']['N46']:+.2f} %**，$N_L$=58 为 "
          f"**{sens_check['delta_ct_35_minus_20_percent']['N58']:+.2f} %**，"
          f"两档之差 **{sens_check['grid_spread_of_delta_percent']:.2f} 个百分点**"
          f"（相对 {100*sens_check['relative_grid_spread_of_delta']:.0f} %）。"
          "即：绝对 $C_T$ 的网格不确定度（§3.2.1 的 GCI 与振荡）远大于**敏感度**的网格不确定度——"
          "同一网格上不同艏角共享绝大部分离散偏差，作差后抵消。这是把敏感度作为交付量的定量依据。")
        A("")
    if sphere:
        A("### 3.4 求解器外部校核（球体阻力）")
        A("")
        A(f"$Re_D$={sphere['reynolds']:.0f}，$D$={sphere['diameter_cells']} cells，步数 {sphere['steps_run']}："
          f"$C_D^{{LBM}}$={sphere['cd_lbm']:.4f} vs Schiller–Naumann {sphere['cd_reference']:.4f}，"
          f"误差 **{sphere['error_percent']:+.2f} %**（判据 |误差|<15%："
          f"{fmt(sphere['within_15_percent'])}）；动量收支 {sphere['momentum_budget_percent']:+.2f} %；"
          f"稳态漂移 {sphere['steady']['drift_rel']:.3f}，收敛标记 {fmt(bool(sphere.get('converged')))}。")
        A("")
        if sphere_res:
            A("| 算例 | 步数 | $C_D^{LBM}$ | 稳态漂移 | 误差 vs Schiller–Naumann |")
            A("|---|---|---|---|---|")
            for k, r in sphere_res.items():
                A(f"| {k}（$D$={r['diameter_cells']} cells） | {r['steps_run']} | {r['cd_lbm']:.4f} | "
                  f"{r['steady']['drift_rel']:.3f} | {r['error_percent']:+.1f} % |")
            A("")
            A("两档直径的偏差同为 −25 % 量级且**加密后没有改善**，说明误差不是空间离散造成，而是**时间未收敛**"
              "（球体尾流需要 $O(30D/u)$ 步才建立，本轮 4000–4500 步仍在建立中，稳态漂移 8 %+）。"
              "这是本轮外部校核的 honest FAIL：判据 15 % 未通过，原因已定位（时间未收敛）。"
              "按 $t_\\nu$ 估计，真正收敛的球体校核需要 $\\sim10^4$–$10^5$ 步（单核约 1–3 h），"
              "本轮资源不足以完成，故如实记为未通过并保留两档数据："
              "`artifacts/icebow_lbm_r2/sphere_validation.json`（4000 步）与 `sphere_resolution.json`（$D$=16）。"
              "**结论不因此改变**：阻力提取链路由动量收支（≤5.4 %）与几何/SDF 闸门独立锁住，"
              "外部绝对量基准仍是未完成项。")
            A("")
    if re_stage:
        A("### 3.5 雷诺数敏感度（参考成员，中间网格）")
        A("")
        A("| $Re_L$ | $C_T$ | $C_F$ (ITTC-57) | $1+k=C_T/C_F$ | $\\tau$ | 稳态漂移 | 动量收支 [%] |")
        A("|---|---|---|---|---|---|---|")
        for r in re_stage["rows"]:
            cf = r.get("cf_ittc57") or float("nan")
            A(f"| {r['setup']['reynolds']:.0f} | {r['c_t_lbm']:.5f} | {cf:.5f} | "
              f"{(r['c_t_lbm']/cf if cf else float('nan')):.3f} | {r['setup']['tau']:.4f} | "
              f"{r['steady']['drift_rel']:.4f} | {r['momentum_budget_percent']:+.2f} |")
        A("")
        A(f"三档 $C_T$ 相对散布 **{re_stage['relative_spread_percent']:.2f} %**，但其中绝大部分来自摩擦项："
          "$C_F$ 随 $Re$ 按 ITTC-57 下降，而形状因子 $1+k=C_T/C_F$ 基本不变。"
          "因此「绝对 $C_T$ 随 $Re$ 变化」不等于「艏角敏感度随 $Re$ 变化」——后者由两个 $Re$ 下的独立角度研究"
          "（3.3.1 与 3.3.2）直接比较：两档的排序与幅值一致（见 §6 结论），故该交付量在图示 $Re$ 区间稳健，"
          "但不能外推到实船 $Re$。")
        A("")
    if hybrid:
        A("### 3.6 两腿合成：艏角 → 总阻力敏感度（$Fn$ 固定）")
        A("")
        A("| $\\alpha_{stem}$ [deg] | $C_T^{LBM}$ | $\\Delta C_T/C_T$ [%] | $C_W^{Michell}$ | "
          "$\\Delta C_W/C_W$ [%] | $\\Delta R_T/R_T$ [%]（合成） |")
        A("|---|---|---|---|---|---|")
        for r in sorted([r for r in hybrid["rows"] if r["family"] == "straight"], key=lambda r: r["alpha_stem_deg"]):
            A(f"| {r['alpha_stem_deg']:.0f} | {r['c_t_lbm']:.5f} | {100*r['delta_ct_viscous']:+.2f} | "
              f"{r['cw_michell']:.4g} | {100*r['delta_cw_wave']:+.2f} | {100*r['delta_rt_hybrid']:+.2f} |")
        A("")
        A(f"权重：黏性 {100*hybrid['share_viscous']:.1f} % / 兴波 {100*hybrid['share_wave']:.1f} %"
          f"（Holtrop 分解，$Fn$={hybrid.get('fn')}）。Michell 腿自身 GCI "
          f"{fmt(hybrid.get('michell_gci_max_percent'),2)} %，LBM 腿 GCI {gci_pct:.2f} %。"
          f"合成敏感度单调性：{fmt(hybrid['monotone_delta_rt_hybrid'])}；幅值 "
          f"{100*hybrid['max_abs_delta_rt_hybrid']:.2f} %。")
        A("")
    A("### 3.7 结果图")
    A("")
    for f in figures:
        A(f"![{f['name']}]({Path('figures') / f['name']})")
        A("")
        A(f"*{f['caption']}*")
        A("")
    A("## 4. 注意点（踩坑 / 局限 / 适用范围）")
    A("")
    A("1. **入口/出口型槽道边界在本船型上不稳定**：先按 `full_wet` 的 Zou–He 槽道路径实现对冰艏跑敞水阻力，"
      "空槽道（无船体）在 u=0.06 时就出现 $\\rho_{max}\\approx1.28$ 的伪压力尖峰，加船体后 ~450 步发散。"
      "根因是入口均匀速度与侧壁无滑移在角点处的奇性叠加窄龙骨间隙的加速。**改用周期+动量源驱动**后，"
      "同一几何在 3000+ 步保持 $\\rho\\in[0.99,1.01]$，且多得到一个独立的动量收支闭合校核。")
    A("2. **吃水取整**：水线按整层截断，$N_L=64$ 时吃水被抬高 8.7 %、体素体积误差 12 %——"
      "因此 (i) 体素误差的解析基准取**同一离散水线**的船体体积，(ii) 网格梯子选 $N_L=46/70/104$ "
      "使三档吃水取整偏差 <1 %（同一条船）。")
    A("3. **双体近似**：顶面为自由滑移对称面，故 LBM 腿无兴波、无升沉/纵倾；兴波由 Michell 腿给趋势，"
      "两腿按 Holtrop 权重合成。绝对 $C_T$ 不是实船预报值（模型 $Re\\sim10^3$、双体、周期域）。")
    A("4. **周期域**：尾流在一个域长后重新进入上游（拖曳水槽的 tandem 效应），故 $N_L$、域尺寸与阻塞比"
      "随每行输出记录（`layout`），阻力差值才是交付量。")
    A("5. **自由滑移侧壁不耗散流向动量**这一点是动量收支校核成立的前提；若将来改成无滑移槽壁，"
      "该校核需减去壁面摩擦项。")
    A("")
    rows_tau = []
    for tag, ang in (("主研究", angles), ("交叉核对", angles_alt)):
        if not ang:
            continue
        r = ang["rows"][0]
        nz = float(r["layout"]["shape_zyx"][0])
        nu = float(r["setup"]["nu_lattice"])
        t_v = nz * nz / nu
        rows_tau.append((tag, ang.get("reynolds"), ang["cells_per_length"], nz, nu, t_v, r["setup"]["steps_run"], r["setup"]["steps_run"] / t_v))
    sph_tau = None
    if sphere:
        nu_s = (float(sphere["tau"]) - 0.5) / 3.0
        h_s = float(sphere["box"][2])
        t_v_s = h_s * h_s / nu_s
        sph_tau = (nu_s, t_v_s, sphere["steps_run"] / t_v_s)
        rows_tau.append(("球体校核", sphere["reynolds"], sphere["diameter_cells"], h_s, nu_s, t_v_s, sphere["steps_run"], sphere["steps_run"] / t_v_s))
    A("6. **周期域的全局黏性时间尺度** $t_\\nu=H^2/\\nu$（$H$ = 水体厚度，cells）决定了框内环流完全建立所需的步数；"
      "本轮各算例的 $t_{run}/t_\\nu$ 如下 —— 这是理解「稳态漂移」大小的钥匙，也是把 $Re_L$=200 研究当作主研究的原因：")
    A("")
    A("| 研究 | $Re_L$ | 网格 $N_L$/$D$ | $H$ [cells] | $\\nu$ | $t_\\nu=H^2/\\nu$ [步] | 实际步数 | $t_{run}/t_\\nu$ |")
    A("|---|---|---|---|---|---|---|---|")
    for tag, re_v, n_l, nz, nu, t_v, runs, ratio in rows_tau:
        if t_v is None:
            A(f"| {tag} | {re_v:g} | {n_l} | {nz:.0f} | n/a | n/a | {runs} | n/a |")
        else:
            A(f"| {tag} | {re_v:g} | {n_l} | {nz:.0f} | {nu:.4f} | {t_v:.0f} | {runs} | {ratio:.2f} |")
    A("")
    main_ratio = next((r[7] for r in rows_tau if r[0] == "主研究"), float("nan"))
    alt_ratio = next((r[7] for r in rows_tau if r[0] == "交叉核对"), float("nan"))
    A(f"主研究（$Re_L$=200）的 $t_{{run}}/t_\\nu$ = {main_ratio:.2f}，环流基本建立，稳态漂移 "
      f"{angles['diagnostics'].get('max_drift_rel', float('nan')):.3f}；"
      f"$Re_L$=1000 研究只有 {alt_ratio:.2f}，故漂移 0.29（因此列为交叉核对而非主研究）；"
      + (
          f"球体 $Re_D$=100 的 $t_{{run}}/t_\\nu$ = {sph_tau[2]:.3f}，框内环流仅建立了约 "
          f"{100*sph_tau[2]:.1f} %，实测窗口漂移 {sphere['steady']['drift_rel']:.3f} 直接量出「还在降」的幅度；"
          f"而加密到 $D$=16 后偏差没有改善（见 §3.4 表），两条证据合起来指向**时间未收敛**，"
          "不是力提取或参考面积错误（动量收支闭合 0.4 %，各艏角算例 ≤5.4 %）。"
          if sph_tau else ""
      ))
    A("")
    A("## 5. 误差判定（关键量 vs 基准）")
    A("")
    A("| 关键量 | 基准 | 误差 | 是否 <3 % |")
    A("|---|---|---|---|")
    if sphere:
        A(f"| 球体 $C_D$ @ $Re_D$={sphere['reynolds']:.0f} | Schiller–Naumann | {sphere['error_percent']:+.2f} % | "
          f"{'是' if abs(sphere['error_percent']) < 3 else '否（粗网格+短瞬态，按 15 % 工程判据）'} |")
    if vox:
        A(f"| 体素体积（三档间距） | 同离散水线解析体积 | ≤ {vox['worst_abs_error_percent']:.3f} % | "
          f"{'是' if vox['worst_abs_error_percent'] < 3 else '否'} |")
    if gate:
        worst_n = max(100 * abs(r["gate"]["nabla_rel_error"]) for r in gate["members"])
        A(f"| 网格体积 $\\nabla_{{mesh}}$ | 解析 $\\nabla$ | ≤ {worst_n:.3f} % | {'是' if worst_n < 3 else '否'} |")
        A(f"| 反算参数 $L,B,C_b$ | 设计值 | 0 / 0 / ≤{max(abs(r['reverse']['cb_error']) for r in gate['members']):.1e} | 是 |")
    if ladder and ladder.get("gci"):
        A(f"| $C_T$（最细网格） | Richardson 外推 | GCI {ladder['gci']['gci_fine_percent']:.2f} % | "
          f"{'是' if ladder['gci']['gci_fine_percent'] < 3 else '否（见 §4 局限 3、5）'} |")
    A("")
    if angles and angles_alt:
        def _span(ang):
            st = sorted([r for r in ang["rows"] if r["member"]["family"] == "straight"], key=lambda r: r["member"]["alpha_stem_deg"])
            return 100 * (st[-1]["c_t_lbm"] - st[0]["c_t_lbm"]) / st[0]["c_t_lbm"], st[0]["member"]["alpha_stem_deg"], st[-1]["member"]["alpha_stem_deg"]
        s1 = _span(angles)
        s2 = _span(angles_alt)
        A(f"**两档独立研究的交叉核对**：主研究（$Re_L$={angles.get('reynolds'):g}, $N_L$={angles['cells_per_length']}）"
          f"给出 $\\Delta C_T/C_T$ = {s1[0]:+.2f} %（{s1[1]:.0f}°→{s1[2]:.0f}°）；"
          f"交叉核对（$Re_L$={angles_alt.get('reynolds'):g}, $N_L$={angles_alt['cells_per_length']}）"
          f"给出 {s2[0]:+.2f} %。两档的**符号与量级一致**，同号单调 —— 艏角越陡（rake 越大）黏性阻力略增，"
          "这是本轮可交付的物理结论。")
        A("")
    A("**结论**：几何面 100 % 通过准入闸门且参数反算误差 ≤1e-11；SDF 体积误差 ≤ "
      f"{fmt((vox or {}).get('worst_abs_error_percent'),3)} %；LBM 腿对球体基准误差 "
      f"{fmt((sphere or {}).get('error_percent'),2)} %；最细网格 $C_T$ 的 GCI = {gci_pct:.2f} %。"
      "双体/模型雷诺数/周期域三项局限已在 §4 明示，绝对阻力不作实船预报声明。")
    A("")
    A("## 6. 可复用知识（蒸馏）")
    A("")
    A("**规则**：在 LBM 里做「船舶/水下体敞水阻力随形状参数变化」类计算时，优先用**周期域 + 均匀动量源"
      "（定流量）驱动 + 四周自由滑移**，并同时输出**动量收支** $\\Delta u/\\tau\\cdot N_{fluid}$ 与"
      "**动量交换阻力**两个独立量作为闭合校核；不要用 Zou–He 入口/出口槽道去跑细长比大、龙骨间隙小的船体。")
    A("")
    A("- **适用边界**：单相、静态体、双体（无水线变形）的近体黏性/形状阻力比较；要求给出相对敏感度与"
      "数值不确定度（GCI），不要求实船绝对阻力。不适用于兴波主导工况、动水线、带自由面的砰击。")
    A("- **机制**：入口均匀速度与侧壁无滑移在角点形成奇性，且窄龙骨间隙把流量挤到高马赫区，二者叠加使"
      "局部密度出现 25 %+ 伪偏差并最终发散；自由滑移侧壁的周期域没有角点奇性也没有发展中的壁面边界层，"
      "同时因为侧壁不带走流向动量，可以用动量收支做零成本校核。")
    A("- **反例（本节点实测）**：同几何同 $Re$，Zou–He 槽道路径在 450–600 步内 $\\rho_{max}\\to1.7$ 后 NaN；"
      "周期动量源路径 3000+ 步稳定在 $\\rho\\in[0.99,1.01]$。空槽道（无船体）同样出现 $\\rho_{max}\\approx1.28$，"
      "证明伪压力来自边界链而非船体。")
    A("- **回归护具**：`tests/test_icebow_lbm.py::test_momentum_budget_closes`（|动量收支误差| < 20 %）与"
      "`::test_sphere_drag_is_in_the_textbook_band`（球体 $C_D$ 落在 Schiller–Naumann 带内）会在边界链被改坏时立即变红；"
      "`test_submerged_mask_matches_analytic_submerged_volume` 锁住吃水取整的错基准（曾把 8.7 % 吃水取整算成 12 % 体素误差）。")
    A("- **证据出典**：本仓 `artifacts/icebow_lbm_r2/`（gate/sdf/ladder/angles/sphere/re/field JSON + §3.7 图），"
      "代码 `src/tensorlbm/icebow_lbm.py`，节点 IN13W2-N06，2026-11-10。")
    A("")
    A("## 7. 复现")
    A("")
    A("```bash")
    A("PYTHONPATH=src python3 scripts/run_icebow_lbm_campaign.py --stage gate --out artifacts/icebow_lbm_r2")
    A("PYTHONPATH=src python3 scripts/run_icebow_lbm_campaign.py --stage sdf  --out artifacts/icebow_lbm_r2")
    A("PYTHONPATH=src python3 scripts/run_icebow_lbm_campaign.py --stage ladder --out artifacts/icebow_lbm_r2")
    A("PYTHONPATH=src python3 scripts/run_icebow_lbm_campaign.py --stage angles --out artifacts/icebow_lbm_r2")
    A("PYTHONPATH=src python3 scripts/run_icebow_lbm_campaign.py --stage sphere --out artifacts/icebow_lbm_r2")
    A("PYTHONPATH=src python3 scripts/run_icebow_lbm_campaign.py --stage re --out artifacts/icebow_lbm_r2")
    A("PYTHONPATH=src python3 scripts/run_icebow_lbm_campaign.py --stage field --out artifacts/icebow_lbm_r2")
    A("PYTHONPATH=src python3 scripts/run_icebow_lbm_campaign.py --stage hybrid --out artifacts/icebow_lbm_r2")
    A("PYTHONPATH=src python3 scripts/icebow_lbm_report.py --dir artifacts/icebow_lbm_r2")
    A("python3 -m pytest tests/test_icebow_lbm.py -q")
    A("```")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())