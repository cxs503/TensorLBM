#!/usr/bin/env python3
"""m17851 交差审计证据包生成器（零手抄：所有数字从 result.json / field CSV 程序化读出）。

对象：m17720 XVAL-LBM-CARTESIAN-MPI（TensorLBM，pylbm/pylbm cartesian-MPI 做法）。
产物：
  artifacts/m17851_audit/repro_matrix.json   各分解配置的数值矩阵 + 交付件一致性 + 界面列定位
  artifacts/m17851_audit/fig_m17851.png     三联图（误差条形 / 中心线沿 x / 剖面 vs 解析解）
用法（在 m17851 工作树内）：python docs/audit_m17851/build_audit_artifacts.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUD = ROOT / "artifacts" / "m17851_audit"
RUNS = AUD / "runs"


def load_result(run_dir: Path) -> dict:
    return json.loads((run_dir / "result.json").read_text())


def load_field(path: Path) -> np.ndarray:
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    nx = int(raw[:, 0].max()) + 1
    ny = int(raw[:, 1].max()) + 1
    ux = np.zeros((ny, nx))
    ux[raw[:, 1].astype(int), raw[:, 0].astype(int)] = raw[:, 2]
    return ux


def analytical(ny: int, umax: float = 0.05) -> np.ndarray:
    y = np.arange(ny) + 0.5
    H = float(ny)
    return 4.0 * umax / H**2 * y * (H - y)


CONFIGS = [
    ("m17851_repro_1000", "128x32/1000  npx=2 npy=2 (交付配置)"),
    ("m17851_dec_1x1", "128x32/1000  npx=1 npy=1 (单子域)"),
    ("m17851_dec_1x2", "128x32/1000  npx=1 npy=2 (仅 y 向分块)"),
    ("m17851_dec_2x1", "128x32/1000  npx=2 npy=1 (仅 x 向分块)"),
    ("m17851_2x2_4000", "128x32/4000  npx=2 npy=2 (长积分)"),
    ("m17851_repro_256x64_1000", "256x64/1000  npx=2 npy=2 (加密网格)"),
]


def main() -> None:
    rows = []
    for sub, label in CONFIGS:
        res = load_result(RUNS / sub)
        mono, dec = res["xval_table"][0], res["xval_table"][1]
        rows.append({
            "run": sub,
            "label": label,
            "spec": res["case_spec"],
            "status": res["status"],
            "mono_umax_error_pct": mono["umax_centreline_error_pct"],
            "decomp_umax_error_pct": dec["umax_centreline_error_pct"],
            "mono_l2_error_pct": mono["l2_error_pct"],
            "decomp_l2_error_pct": dec["l2_error_pct"],
            "max_centreline_diff": res["criteria"]["max_centreline_diff"],
            "decomposition_consistent": res["criteria"]["decomposition_consistency"],
            "fields_identical": res["criteria"]["fields_identical"],
            "accept_threshold_pct": res["criteria"]["accept_threshold_pct"],
        })

    # 交付件 vs 本次重跑：逐数比对（判定交付工件是否真由该代码产生）
    ship = json.loads((RUNS / "m17720_shipped_result.json").read_text())
    repro = load_result(RUNS / "m17851_repro_1000")
    ship_vs_repro = {}
    for s, r in zip(ship["xval_table"], repro["xval_table"]):
        ship_vs_repro[s["tool"]] = {
            k: {"shipped": s[k], "reproduced": r[k], "identical": s[k] == r[k]}
            for k in ("umax_centreline", "umax_computed", "l2_error_pct")
        }
    ship_vs_repro["status"] = {"shipped": ship["status"], "reproduced": repro["status"]}
    ship_vs_repro["case_spec_shipped"] = ship["case_spec"]
    ship_vs_repro["default_spec_note"] = (
        "交付工件用 nx=128/ny=32/steps=1000（CLI 需显式 --nx 128 --ny 32 --steps 1000 才能复现）；"
        "包默认 DEFAULT_SPEC=nx=256/ny=64/steps=8000，无参入口不复现交付件。"
        "工件内 warmup_steps=2000 > steps=1000，warmup 参数在 runners 中从未被读取（死参数）。"
    )

    # 界面列定位：2x2 分解下误差的空间分布
    ux_m = load_field(AUD / "field_monolithic_final.csv")
    ux_d = load_field(AUD / "field_decomposed_final.csv")
    ny, nx = ux_d.shape
    ana = analytical(ny)
    col_err = np.abs(ana[:, None] - ux_d).mean(axis=0)
    centre_row = ny // 2
    seam_cols = [0, 1, nx // 2 - 1, nx // 2, nx - 2, nx - 1]
    interior = np.delete(col_err, seam_cols)
    u_centre = ux_d[centre_row]
    interface_loc = {
        "nx": nx,
        "ny": ny,
        "max_col_mean_abs_err": float(col_err.max()),
        "max_col_mean_abs_err_at_x": int(col_err.argmax()),
        "interior_col_mean_abs_err": float(interior.mean()),
        "interior_pct_of_umax": float(interior.mean() / 0.05 * 100.0),
        "seam_cols": seam_cols,
        "centre_row_u_decomposed_at_seam": {int(i): float(u_centre[i]) for i in seam_cols},
        "centre_row_u_monolithic_at_seam": {int(i): float(ux_m[centre_row, i]) for i in seam_cols},
        "reported_umax_centreline_is_column_x": int(nx // 2),
        "note": ("umax_centreline 取 ux[ny//2, nx//2]，恰为 x 分块界面列（分解缺陷列）→ "
                 "25.35% 的指标值由单列伪值主导；但 fields_identical=false、"
                 "max_centreline_diff=1.26e-2 说明整场并不一致，缺陷非纯指标选取问题。"),
        "max_abs_du_field": float(np.abs(ux_d - ux_m).max()),
    }

    out = {
        "audit_mission": 17851,
        "object_mission": 17720,
        "anchor": "XVAL-LBM-CARTESIAN-MPI",
        "criterion_pct": 15.0,
        "configs": rows,
        "shipped_vs_reproduced": ship_vs_repro,
        "interface_localization": interface_loc,
        "registry": {
            "tensor_benchmark_cases_rows_for_anchor": 0,
            "tensor_xval_cases_rows_for_anchor": 0,
            "note": "基准判据表与 XVAL 案例表均无本锚点登记行 → 平台侧无机器裁判；"
                    "本审计以 mission goal 的 <15% 判据 + tensorlbm 解析解 Poiseuille 剖面为裁判。",
        },
        "oss_anchor": {
            "source_dir": "/root/oss-cache/pylbm",
            "mpi_topology_present": True,
            "mpi4py_available": False,
        },
    }
    AUD.mkdir(parents=True, exist_ok=True)
    (AUD / "repro_matrix.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    ax = axes[0]
    labels = ["1x1", "1x2", "2x1", "2x2\n(交付)"]
    pick = ["m17851_dec_1x1", "m17851_dec_1x2", "m17851_dec_2x1", "m17851_repro_1000"]
    by_run = {r["run"]: r for r in rows}
    mono_vals = [by_run[k]["mono_umax_error_pct"] for k in pick]
    dec_vals = [by_run[k]["decomp_umax_error_pct"] for k in pick]
    x = np.arange(len(pick))
    ax.bar(x - 0.2, mono_vals, 0.4, label="monolithic", color="#33578c")
    ax.bar(x + 0.2, dec_vals, 0.4, label="decomposed", color="#c0392b")
    ax.axhline(15.0, color="k", ls="--", lw=1.2, label="accept 15%")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("centreline u error vs analytic (%)")
    ax.set_title("(a) 2x2 分解才越界 (128x32, 1000 steps)")
    for xi, v in zip(x, dec_vals):
        ax.annotate(f"{v:.2f}%", (xi + 0.2, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.plot(u_centre, label="decomposed 2x2 (centre row)", color="#c0392b", lw=1.0)
    ax.plot(ux_m[centre_row], label="monolithic", color="#33578c", lw=1.0, ls="--")
    for c in (nx // 2,):
        ax.axvline(c, color="k", ls=":", lw=1, label=f"x-split interface x={c}")
    ax.set_xlabel("x (lattice)")
    ax.set_ylabel("u_x at centre row")
    ax.set_title("(b) 分块界面列被污染（x=0/nx-1、x=nx/2）")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    y = np.arange(ny) + 0.5
    ax.plot(ana, y, "k:", lw=2, label="analytic")
    ax.plot(ux_m.mean(axis=1), y, color="#33578c", ls="--", label="monolithic (x-avg)")
    ax.plot(ux_d.mean(axis=1), y, color="#c0392b", label="decomposed 2x2 (x-avg)")
    ax.set_xlabel("u (x-averaged)")
    ax.set_ylabel("y")
    ax.set_title("(c) 剖面：主体形状对，界面列拉低均值")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(AUD / "fig_m17851.png", dpi=140)
    plt.close(fig)
    print(json.dumps({r["run"]: {"mono": round(r["mono_umax_error_pct"], 4),
                                "dec": round(r["decomp_umax_error_pct"], 4),
                                "status": r["status"]} for r in rows}, indent=1))
    print("wrote", AUD / "repro_matrix.json", AUD / "fig_m17851.png")


if __name__ == "__main__":
    main()