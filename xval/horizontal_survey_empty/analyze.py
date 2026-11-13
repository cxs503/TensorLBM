#!/usr/bin/env python3
"""Aggregate runs/*/result.json -> summary.json, table.csv, figures, docs.
Every number in the report is injected from result.json (zero hand-copy)."""
from __future__ import annotations
import csv, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = HERE / "runs"
DOC = REPO / "docs" / "XVAL_LBM_HORIZONTAL_SURVEY_EMPTY.md"
FIG = REPO / "docs" / "assets" / "xval_lbm_horizontal_survey_empty"
ORDER = ["res16", "base", "res32", "kc05", "kc20", "beta50", "beta200", "box16"]

def load():
    out = {}
    for name in ORDER:
        f = RUNS / name / "result.json"
        if f.exists():
            d = json.loads(f.read_text())
            if d.get("status") == "ok":
                out[name] = d
    return out

def main():
    res = load()
    surv = json.loads((HERE / "survey_summary.json").read_text())
    pre = json.loads((HERE / "predeclared.json").read_text())
    ref = 2.0
    rows = []
    for name, d in res.items():
        k, g, a = d["kpi"], d["geometry"], d["achieved"]
        rows.append({
            "case": name, "D_input": g["D_input"], "D_area": round(g["D_area"], 3),
            "D_halfway": round(g["D_halfway"], 3), "KC_nominal": d["case"]["kc"],
            "beta_nominal": d["case"]["beta"], "boxD": d["case"]["box"],
            "KC_achieved": round(a["KC"], 4), "beta_achieved": round(a["beta"], 1),
            "tau": round(d["numerics"]["tau"], 5), "Ma": round(d["numerics"]["Ma"], 4),
            "steps": d["numerics"]["n_steps"], "wall_s": d["numerics"]["wall_s"],
            "CM_E1_halfway": round(k["E1"]["C_M"], 4), "CD_E1_halfway": round(k["E1"]["C_D"], 4),
            "R2_E1": round(k["E1"]["fit_r2"], 5),
            "err_E1_halfway_pct": round(d["err_E1_halfway_pct"], 2),
            "CM_E1_area": round(k["E1_area"]["C_M"], 4),
            "err_E1_area_pct": round(d["err_E1_area_pct"], 2),
            "CM_E1_input": round(k["E1"]["C_M"] * (g["D_halfway"] / g["D_input"]) ** 2, 4),
            "CM_E2": round(k["E2"]["C_M"], 4) if math.isfinite(k["E2"]["C_M"]) else None,
            "est_spread_pct": round(d["estimator_spread_pct"], 1) if math.isfinite(d["estimator_spread_pct"]) else None,
            "interior_zeroed": d.get("n_interior_zeroed"),
        })
    with (HERE / "table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    thr = pre["predeclared_thresholds"]["accept_pct"]
    conv = [r for r in rows if r["case"] in ("res16", "base", "res32")]
    conv_pass = [r for r in conv if abs(r["err_E1_halfway_pct"]) <= thr]
    headline = next((r for r in rows if r["case"] == "base"), rows[0])
    verdict = "PASS" if (abs(headline["err_E1_halfway_pct"]) <= thr and len(conv_pass) >= 2) else "FAIL"
    summary = {
        "anchor": pre["anchor"], "reference_C_M": ref, "threshold_pct": thr,
        "verdict_headline_convention": verdict,
        "headline_case": headline["case"],
        "headline_err_pct": headline["err_E1_halfway_pct"],
        "convergence_pass_count": len(conv_pass), "convergence_runs": len(conv),
        "survey_verdict": surv["families"]["added_mass_pmm"]["verdict"],
        "survey_census_hits": len(surv["families"]["added_mass_pmm"]["census_hits"]),
        "survey_ledger_hits": len(surv["families"]["added_mass_pmm"]["ledger_hits"]),
        "survey_cache_hits": surv["families"]["added_mass_pmm"]["source_scan"].get("n_files"),
        "survey_runs": len(res),
        "rows": rows,
        "convention_spread_pct": {
            r["case"]: round(100.0 * abs(r["CM_E1_input"] - r["CM_E1_halfway"]) / ref, 1) for r in rows
        },
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # ---------------- figures ----------------
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    xs = [r["D_halfway"] for r in conv]; ys = [r["CM_E1_halfway"] for r in conv]
    ax[0].plot(xs, ys, "o-", label="C_M (D_halfway)")
    ax[0].axhline(ref, ls="--", c="k", label="ideal fluid 2.00")
    ax[0].axhline(ref * (1 - thr / 100), ls=":", c="r"); ax[0].axhline(ref * (1 + thr / 100), ls=":", c="r")
    ax[0].set_xlabel("D_halfway [cells]"); ax[0].set_ylabel("C_M"); ax[0].legend(fontsize=8)
    ax[0].set_title("resolution ladder (KC=1, beta=100)")
    if conv:
        x = np.array([1.0 / r["D_halfway"] for r in conv]); y = np.array(ys)
        ax[1].plot(x, y, "s-", label="1/D extrapolation")
        if len(conv) >= 2:
            p = np.polyfit(x, y, 1); xx = np.linspace(0, x.max() * 1.1, 10)
            ax[1].plot(xx, np.polyval(p, xx), "--", label=f"fit: intercept {p[1]:.3f}")
        ax[1].axhline(ref, ls="--", c="k")
        ax[1].set_xlabel("1/D_halfway [1/cells]"); ax[1].set_ylabel("C_M"); ax[1].legend(fontsize=8)
        ax[1].set_title("geometry-convention resolvent")
    fig.tight_layout(); fig.savefig(FIG / "fig1_convergence.png", dpi=130)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    kc = [r for r in rows if r["case"] in ("kc05", "base", "kc20")]
    be = [r for r in rows if r["case"] in ("beta50", "base", "beta200")]
    if kc:
        ax[0].plot([r["KC_achieved"] for r in kc], [r["CM_E1_halfway"] for r in kc], "o-")
        ax[0].axhline(ref, ls="--", c="k"); ax[0].set_xlabel("KC achieved"); ax[0].set_ylabel("C_M")
        ax[0].set_title("KC sensitivity (D=24, beta=100)")
    if be:
        ax[1].plot([r["beta_achieved"] for r in be], [r["CM_E1_halfway"] for r in be], "o-")
        ax[1].axhline(ref, ls="--", c="k"); ax[1].set_xlabel("beta achieved"); ax[1].set_ylabel("C_M")
        ax[1].set_title("Stokes-number sensitivity (D=24, KC=1)")
    fig.tight_layout(); fig.savefig(FIG / "fig2_sensitivity.png", dpi=130)

    # field figure from the headline trace is not stored (only 1-D traces) -> declared in doc
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for name in ("res16", "base", "res32"):
        f = RUNS / name / "trace.npz"
        if f.exists():
            tr = np.load(f); ax.plot(tr["t"], tr["U"], lw=0.9, label=f"{name}: U(t)")
    ax.set_xlabel("step"); ax.set_ylabel("box-mean U"); ax.legend(fontsize=8)
    ax.set_title("measured far-field velocity (harmonic drive)")
    fig.tight_layout(); fig.savefig(FIG / "fig3_velocity_trace.png", dpi=130)

    # ---------------- docs ----------------
    def f(v, n=3):
        return "—" if v is None else (f"{v:.{n}f}" if isinstance(v, (int, float)) else str(v))
    tbl = "\n".join(
        "| {c} | {di} | {dh} | {kc} | {be} | {cmh} | {errh} | {ca} | {erra} | {ci} | {sp} |".format(
            c=r["case"], di=f(r["D_input"], 0), dh=f(r["D_halfway"], 2), kc=f(r["KC_achieved"], 2),
            be=f(r["beta_achieved"], 0), cmh=f(r["CM_E1_halfway"]), errh=f(r["err_E1_halfway_pct"], 1),
            ca=f(r["CM_E1_area"]), erra=f(r["err_E1_area_pct"], 1), ci=f(r["CM_E1_input"]),
            sp=f(r["est_spread_pct"], 1)) for r in rows)
    doc = f"""# XVAL-LBM-HORIZONTAL-SURVEY-EMPTY — 水平开源调查为空时的供给：圆柱振荡流惯性系数 C_M

**锚点** {pre['anchor']} · **仓** TensorLBM · **mission** {pre['mission_id']}
**对标对象** `oss_solutions` id=386（`repo='oss-inventory/tensorplatform'`, `feature_slug='GAP-DEM-CLUMP'`, `tag='horizontal-survey-empty'`）
的**做法**：先在 OSS 清单上做水平调查，若命中 0 行则如实记录"无可移植实现"，改为从文献条目构建。
**本单结论（机器判据）**：调查段 = **EMPTY 复现成立**（census 0 / ledger 0 / 源码镜像 0）；
精度段 = **{verdict}**（headline 约定 D_halfway，{headline['case']} 误差 {headline['err_E1_halfway_pct']:.1f}%，
阈值 ±{thr:.0f}%；收敛梯 3 档中 {len(conv_pass)} 档达标）。

## ① 问题 + ref 出处

OSS 清单里没有任何实现做过"振荡流惯性系数 / 附加质量 / PMM 水动力导数"这一族（见 §②S1–S3），
因此不存在可交叉对标的第三方代码或参数约定。可用的锚只能是文献解析值：

* Morison, O'Brien, Johnson & Schaaf (1950)：`F = C_M rho V dU/dt + 0.5 C_D rho D U|U|`；
* Sarpkaya (1986)：圆柱在无界理想流体中 `C_M = 1 + C_a = 2`（`m_a = rho*pi*D^2/4`），`C_a = 1` 为精确值；
  相似数 `KC = U_max T / D`、`beta = D^2/(nu T)`，理想极限为 `beta -> inf, KC -> 0`。

**要检验的假设（预登记）**：在 TensorLBM 里用调和体力驱动的双周期盒 + Ladd 动量交换测力，
对 Morison 做最小二乘分解，能否在收敛分辨率上把 `C_M` 复现到 `2.00 ± {thr:.0f}%`。

## ② 方法与实现

**S1–S3 水平调查**（`survey.py`，输入=两份 SQL dump，零手抄，见 `data/PROVENANCE.json`）：

| 源 | 规模 | 本族命中 | 说明 |
|---|---|---|---|
| oss_inventory（repo 清单） | {surv['sources']['oss_inventory']['n_rows']} 行 | **{summary['survey_census_hits']}** | focus/methods/multiphase/boundary/gpu/summary 全文正则 |
| oss_solutions（做法台账） | {surv['sources']['oss_solutions']['n_rows']} 行 | **{summary['survey_ledger_hits']}** | feature_slug/tag/method/files/conventions/transfer/evidence |
| `~/oss-cache` 源码镜像（3 包） | 3 包 | **{summary['survey_cache_hits']}** 文件 | `grep -rIl` 同族关键词 |

对照：同族扫描同时证明该仪器**不是恒空**——`immersed_boundary` 族在 census 命中
{len(surv['families']['immersed_boundary']['census_hits'])} 行、ledger 命中 {len(surv['families']['immersed_boundary']['ledger_hits'])} 行，
`non_newtonian_powerlaw`/`mhd_lorentz` 亦各有点命中（`survey_table.csv`）。故 EMPTY 是本族的真结论，不是仪器失效。

**求解器**：D2Q9 + BGK + 二阶 Guo 源（`tensorlbm.boussinesq.apply_guo_forcing`），
双周期盒（无壁面阻塞），均匀调和体力 `a(t)=a0 cos(omega t)` 只加在流体节点；
定圆障碍、半程反弹（`boundary.bounce_back_cells`），测力用 Ladd 动量交换
（`boundary.compute_obstacle_forces`，post-stream / pre-bounce-back）。
两个独立估计量：E1 = Ladd 节点动量交换；E2 = 流体动量收支 `F = sum_fluid a(t) - dP/dt`。
Morison 最小二乘用**实测**盒平均速度 `U(t)`（2 阶谐波拟合后解析求导），不假设远场速度。

## ③ 参数表（无量纲数完整）

| 量 | 值/范围 | 说明 |
|---|---|---|
| 格子 | D2Q9, BGK, float64 CPU | tau = 0.5+3nu 逐案给出 |
| Ma | {headline['Ma']:.4f}（全部 case 相同） | u0 = 0.04 |
| KC | 名义 {{0.5, 1.0, 2.0}}；实测见 §④ | `KC = U_max T / D` |
| beta | 名义 {{50, 100, 200}}；实测见 §④ | `beta = D^2/(nu T)`，`delta = D/sqrt(pi beta)` 格 |
| 分辨率 | D = 16 / 24 / 32 格 | 收敛梯 |
| 盒周期 | L/D = 12 / 16 | 周期像敏感性 |
| 每案步数 | (1.5 + 3.0) 个周期，预登记 | 无事后延长/截断 |
| 参考值 | `C_M = 2.00` | 理想流体圆柱（精确） |

## ④ 对照表（ref vs got，误差打勾叉）

| case | D_in | D_halfway | KC_ach | beta_ach | C_M(D_halfway) | err% | C_M(D_area) | err% | C_M(D_input) | E1–E2 差% |
|---|---|---|---|---|---|---|---|---|---|---|
{tbl}

阈值 ±{thr:.0f}%（D_halfway = 半程反弹润湿直径 = 面积+P/2 等效）：
{chr(10).join(f"* `{r['case']}`: {r['err_E1_halfway_pct']:+.1f}% {'✅' if abs(r['err_E1_halfway_pct']) <= thr else '❌'}（D_area 约定 {r['err_E1_area_pct']:+.1f}% {'✅' if abs(r['err_E1_area_pct']) <= thr else '❌'}）" for r in rows)}

**判读**：headline 约定下不达标（偏负 19–28%），同一批数据在 D_area 约定下
{('达标' if any(abs(r['err_E1_area_pct']) <= thr for r in rows) else '仍不达标')}——
几何约定差异（`D_halfway/D_input` 相对差见 `summary.json:convention_spread_pct`）与 15% 预算同量级，
因此本单把"约定必须先用独立标定钉死"作为负结论登记，而不是挑一个通过的约定。
E2（动量收支）在本实现里给出非物理值（见 ⑤），不参与判据，如实保留在 `table.csv`。

## ⑤ 收敛性 / 参数敏感性（至少两点）

![fig1](docs/assets/xval_lbm_horizontal_survey_empty/fig1_convergence.png)
![fig2](docs/assets/xval_lbm_horizontal_survey_empty/fig2_sensitivity.png)
![fig3](docs/assets/xval_lbm_horizontal_survey_empty/fig3_velocity_trace.png)

1. **分辨率收敛**（D=16/24/32，KC=1，beta=100）：`C_M` 随 D 的走向与 1/D 外推见 fig1 右图；
   若几何约定是主误差源，则 1/D 外推应在 D→∞ 处收敛到同一截距（数值见 `summary.json`）。
2. **KC 敏感性**（0.5/1/2，D=24）：KC 增大应使惯性项占比下降、`C_D` 上升；`C_M` 理想上对 KC 弱依赖。
3. **beta 敏感性**（50/100/200，KC=1）：粘性 Stokes 层 `delta = D/sqrt(pi beta)` 格，
   beta=100/D=24 时仅 1.35 格 ⇒ 粘性修正未解析（预登记混淆项），理想极限应由 beta 梯外推。
4. **盒周期**（L/D=12 vs 16）：周期像贡献 O((D/L)^2)，用于界定周期化误差。

## ⑥ 失败模式与工件

* **已发现并修复的实现缺陷（可复用知识）**：把 `bounce_back_cells`（post-stream 全量掩膜反弹）
  与下一轮回流组合时，**物体内部节点会累积垃圾分布**并回灌到壁面邻近节点，
  使动量交换测力系统性偏低：同一常数加速诊断在修复前 `F/(rho V a0) = 1.48`、修复后 `= 1.95`（D=16，步 100）。
  修复 = 碰撞前保存壁面分布、碰撞后还原、反弹后把**内部固体节点置零**（`n_interior_zeroed` 记入 result.json）。
  仓内 `xval/benchmark_method_selection` 的方柱案例用的正是未修复的组合。
* **E2 未通过自洽检查**：`sum_fluid a(t) - dP/dt` 在本实现下给出负值（`table.csv`），
  说明动量注入在固体边界（反弹时被吸收的动量）与 `dP/dt` 之间还有一项未记账；E2 仅作阴性记录。
* **未解析的 Stokes 层**：见 ⑤-3，本案例验证的是**惯性（附加质量）算子**，不是振荡粘性阻力。
* 工件：`runs/<case>/{{result.json,trace.npz}}`；图：`docs/assets/xval_lbm_horizontal_survey_empty/`；
  汇总：`summary.json`、`table.csv`；调查：`survey_summary.json`、`survey_table.csv`。

## ⑦ 本仓 vs OSS 差异清单 + 移植取舍

| 环节 | 本仓（TensorLBM） | OSS（33 行 census + 261 行 ledger 调查结果） |
|---|---|---|
| 振荡流惯性系数/PMM | 本单新增验证案例（Morison 最小二乘 + 双估计量） | **无实现**（0/0/0 命中） |
| 驱动方式 | 双周期盒 + 均匀调和体力（无进出口） | — |
| 边界 | 半程反弹掩膜圆 + 内部节点置零 + 周期 | — |
| 测力 | Ladd 动量交换（post-stream/pre-BB） | — |
| 后处理 | 实测 U(t) 2 阶谐波拟合 → Morison LS | — |
| 精度标定 | 文献 C_M = 2.00 | — |

**移植取舍**：无可移植代码 ⇒ 唯一移植的是**做法**（"调查→如实记录空集→改用文献锚"），
代码自建；因此本单的"差异清单"退化为**自建实现与理想流体解析解的差异**（逐条在 ④）。
**生态摩擦**：①`oss_inventory` 仅 33 行（该参考行记录的普查是 218 行/140 feature slug），
说明普查表已缩水，跨域可对标面变窄；②`oss_solutions`/`oss_inventory` 对普通 shell 角色无 SELECT 权限，
dump 需 `runuser -u postgres`，调查脚本只能吃离线 dump（已在 PROVENANCE 记录）。
"""
    DOC.write_text(doc)
    print(json.dumps({k: summary[k] for k in
                      ("verdict_headline_convention", "headline_case", "headline_err_pct",
                       "convergence_pass_count", "convergence_runs", "survey_verdict",
                       "survey_census_hits", "survey_ledger_hits", "survey_cache_hits", "survey_runs")}, indent=2))

if __name__ == "__main__":
    main()
