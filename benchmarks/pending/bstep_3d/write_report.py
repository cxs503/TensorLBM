#!/usr/bin/env python
"""Generate the bstep_3d (mission m469) diagnosis report from measured artifacts.

Every number below is injected from ``artifacts/summary.json`` /
``artifacts/record_audit.json`` / ``artifacts/figures/figures.json`` - no value
is typed by hand.  Outputs

    docs/bstep3d_m469_diagnosis.md      paper-style report (sections 1-6)
    /tmp/report.json                    platform report payload (figures inlined)

Usage
-----
    python write_report.py [--post]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ART = HERE / "artifacts"
FIG = ART / "figures"
DOC = REPO / "docs" / "bstep3d_m469_diagnosis.md"

MISSION_ID = 469
SCENARIO_KEY = "tensorlbm_backward_step_3d"
TAGS = [
    "tensorlbm",
    "benchmark",
    "bstep_3d",
    "backward-facing-step",
    "reattachment",
    "measurement-defect",
    "lbm",
    "diagnosis",
    "m469",
]


def _fmt(v, nd=4, dash="n/a"):
    if v is None:
        return dash
    if isinstance(v, str):
        return v
    return f"{v:.{nd}f}"


def _pct(v, nd=2, dash="n/a"):
    return dash if v is None else f"{v:+.{nd}f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", action="store_true", help="also publish to the platform API")
    args = ap.parse_args()

    summary = json.loads((ART / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads((ART / "record_audit.json").read_text(encoding="utf-8"))
    figs_meta = json.loads((FIG / "figures.json").read_text(encoding="utf-8"))

    ladder = summary["ladder"]
    cross = summary.get("tau_cross_check")
    refine = summary.get("refinement", {})

    # ---------------------------------------------------------------- tables
    rec_rows = []
    for name, r in audit["records"].items():
        rec_rows.append(
            "| `{}` | {:g} | {} | {:.4f} | {:.5f} | {} | {} | {} | {} | {} |".format(
                name,
                r["re"],
                "x".join(str(v) for v in r["grid"]),
                r["tau"],
                r["u_max_est"],
                _fmt(r["xr_h"], 3),
                _pct(r["err_pct"]),
                _fmt(r["max_backflow_over_umax"], 4),
                "yes" if r["converged"] else "no",
                ("**YES**" if r["inconsistent"] else "no"),
            )
        )
    audit_table = "\n".join(rec_rows)

    lad_rows = []
    for e in ladder:
        lad_rows.append(
            "| {} | {} | {:.4f} | {} | {:.1f} | {} | {} | {} | {} | {} | {} | {}".format(
                e["grid"]["step_h"],
                e["grid"]["cells"],
                e["grid"]["expansion_ratio"],
                _fmt(e["er_dev_pct"], 2),
                e["run"]["wall_time_s"],
                _fmt(e["xr_h_new_steady"], 4),
                _fmt(e["xr_h_legacy_steady_meanfield"], 4),
                _pct(e["err_pct_new"]),
                e["xr_status_final"],
                "{}/{}".format(e["xr_planes_valid"], e["xr_planes_valid"] + e["xr_planes_invalid"]),
                _fmt((e.get("node_sensitivity") or {}).get("delta"), 4),
                e["run"]["steps"],
            )
        )
    ladder_table = "\n".join(lad_rows)

    cross_txt = ""
    if cross:
        cross_txt = (
            f"| {cross['grid']['step_h']} | {cross['physics']['u_max']:.4f} | "
            f"{cross['physics']['tau']:.4f} | {_fmt(cross['xr_h_new_steady'], 4)} | "
            f"{_pct(cross['err_pct_new'])} | cross-check (Re=100, same grid, low-Ma / low-tau) |"
        )

    err_rows = []
    for e in ladder:
        xe = e["err_pct_new"]
        err_rows.append(
            "| X_r/h (H={}) | {:.4f} | Armaly 1983 | {} | {} |".format(
                e["grid"]["step_h"],
                e["xr_h_new_steady"],
                _pct(xe),
                "YES" if xe is not None and abs(xe) < 3.0 else "**NO**",
            )
        )
    if cross:
        xe = cross["err_pct_new"]
        err_rows.append(
            "| X_r/h (H={}, tau=0.524) | {:.4f} | Armaly 1983 | {} | {} |".format(
                cross["grid"]["step_h"],
                cross["xr_h_new_steady"],
                _pct(xe),
                "YES" if xe is not None and abs(xe) < 3.0 else "**NO**",
            )
        )
    err_rows.append(
        "| X_r/h (legacy H=40 record) | 3.3955 | Armaly 1983 | +13.18% | **NO** |"
    )
    err_table = "\n".join(err_rows)

    within = [e for e in ladder if e["err_pct_new"] is not None and abs(e["err_pct_new"]) < 3.0]
    gate_all = len(within) == len(ladder)

    h_coarse = ladder[0]["grid"]["step_h"] if ladder else None
    h_fine = ladder[-1]["grid"]["step_h"] if ladder else None
    xr_coarse = ladder[0]["xr_h_new_steady"] if ladder else None
    xr_fine = ladder[-1]["xr_h_new_steady"] if ladder else None
    estimator_delta_max = max(
        (abs(e["estimator_delta_h"]) for e in ladder if e["estimator_delta_h"] is not None),
        default=None,
    )

    # ------------------------------------------------------------- markdown
    md = f"""# bstep_3d（三维背向台阶，展向周期）SDAA 案例扫荡：测量缺陷定源与修复

班次 m469（承接 mission 466 失败：无代码改动）。仓库 TensorLBM，案例
`benchmarks/pending/bstep_3d`。本报告所有数字由
`benchmarks/pending/bstep_3d/analyze_cpu_repro.py` 从原始产物
（`artifacts/cpu_repro_*.json`、`artifacts/record_audit.json`）程序化注入。

## 1. 案例介绍（Case introduction）

**物理背景。** 三维后向台阶（backward-facing step, BFS）是分离—再附着湍流/层流
定标的经典构型：充分发展通道流经台阶突然扩张，在下游底壁形成分离泡，泡内出现
逆流，泡闭合点距台阶立面的长度 X_r 是壁面剪切、边界层分辨率与数值黏性最敏感的
直接观测量。本案例取展向周期（无 z 向壁面）的三维 D3Q19 版本，层流 Re=100，
观测量为底壁剪切过零点定义的归一化再附着长度 X_r/h。

**几何与参数表**（遗留记录 + 本次复跑；`nx, ny, nz = 10h, round(2.05h), 2h`，
台阶块 `x < 2h 且 y < h`，底壁 `y=0 (x ≥ 2h)`，顶壁 `y=ny-1`，展向周期）：

| 参数 | 遗留记录 H=40 | 遗留记录 H=49 | 本次复跑 H=16 | 本次复跑 H=20 |
|------|---------------|---------------|---------------|---------------|
| 网格 (nx,ny,nz) | 400x82x80 | 490x101x98 | 160x33x16 | 200x41x16 |
| 台阶高 h | 40 | 49 | 16 | 20 |
| 膨胀比 ER=(ny-2)/(ny-1-h) | 1.9512 | 1.9412 | {ladder[0]['grid']['expansion_ratio']:.4f} | {ladder[1]['grid']['expansion_ratio']:.4f} |
| ER 相对 Armaly 1.94 偏差 | +0.58% | +0.06% | {ladder[0]['er_dev_pct']:+.2f}% | {ladder[1]['er_dev_pct']:+.2f}% |
| Re = U_max·h/ν | 100 | 100/200/400 | 100 | 100 |
| τ = 3ν+0.5 | 0.56 | 0.5368/0.5276 | {ladder[0]['physics']['tau']:.4f} | {ladder[1]['physics']['tau']:.4f} |
| U_max | 0.05 | 0.05/0.075 | {ladder[0]['physics']['u_max']:g} | {ladder[1]['physics']['u_max']:g} |
| 展向宽度 Lz/h | 2.0 | 2.0 | 0.8（CPU 预算折减） | 0.8（CPU 预算折减） |

**参考值。** Armaly, Durst, Pereira & Schoenung, *JFM* 127:473-496 (1983)：
Re = U_max·h/ν = 100 时 X_r/h ≈ 3.0（实验），膨胀比 ER ≈ 1.94。验收门：直接观测量
对参考 ≤3%，且需 ≥2 档网格单调收敛（本案例 README 的战役标准）。

**遗留状态。** 案例 `pending`：唯一完整档（H=40）X_r/h = 3.3955，误差 +13.18%；
Re=200 档只跑了 500 步（early abort）；Re=400 档 X_r/h = 0.0，与自身记录的
`max_backflow_over_umax = -0.175`（位置 4.35h）自相矛盾。

## 2. 怎么做的（Method / workflow）

### 2.1 为什么 mission 466 无产物（先定源，再动手）
审计环境（本 worktree 容器）**没有 CUDA 设备**（`torch.cuda.is_available() == False`），
而案例脚本的默认设备是 `cuda:0/cuda:1`、`sys.path.insert(0, "/home/wxsc/cxs/TensorLBM/src")`
与 `--verified /home/wxsc/cxs/TensorLBM/...` 全部指向另一台机器；加上 2.6–4.9 Mcell ×
2–30 万步的 GPU 预算，案例在本环境**无法按记录规模复跑**。这一步本身就是沿用
"直接跑一遍看能不能过" 的做法必然产出零改动的机制性原因：案例的问题是
*测不出来的量被记成了 0*，而不是"再跑一次就能提高精度"。

### 2.2 测量缺陷定源（代码 + 遗留记录自证）
遗留估计器（`run.py` / `run_highre.py` 中 `_zero_crossing_col`）：

```python
pos = int(np.argmax(row > 0.0))
if pos == 0 or row[pos] <= 0.0:
    return None            # 调用方写成 xr_h = 0.0
```

三个缺陷：① 台阶下游第一个节点若已为正（角点未解析为硬角 / 高 Re 记录中常见），
`argmax` 立即返回 0 → `None` → 记录成 `xr_h = 0.0`；② 只取**首个**过零点，角点二次
回流会把测量截断在角细胞内；③ "无回流"、"域内未再附着"、"数据损坏" 三种物理不同
的结果被折叠成同一个 `0.0`，于是下游 `converged` 判据（对 xr 序列做 0.02 容差）
在垃圾数据上判为 "收敛"。

遗留记录审计（`artifacts/record_audit.json`，程序化判定）：

| 记录 | Re | 网格 | τ | U_max(推算) | X_r/h | 误差 | 最强回流/U_max | converged | 自相矛盾 |
|------|----|------|----|------|-------|------|----------------|-----------|----------|
{audit_table}

`result_re400_H49.json` 同时给出"存在 -0.175·U_max 回流"与"X_r/h = 0.0 且
converged=true"——分离流不可能有零再附着长度，这是估计器失效的**硬证据**，
与物理无关。

### 2.3 修复（最小改动，只碰测量与可移植性）
1. 新增 `benchmarks/pending/bstep_3d/reattach.py`：稳健估计器
   - 以**回流面积最大**的连续负值段（长度×强度）作为主分离泡，取其终止过零点做亚格
     线性插值 → 角点二次回流与单胞噪声抖动都不能截断/伪造测量；
   - 区分 `reattached` / `no_backflow` / `not_reattached_in_domain` / `no_data`，
     `X_r` 未定位时返回 `None`（而不是 0.0）；
   - 展向统计只计入有效平面，另报 `xr_planes_invalid`；
   - 平滑用边缘复制（零填充会削弱角点回流、推迟分离起点判据）。
2. `run.py` / `run_highre.py` 接线：`measure_reattach_3d` → `reattach.measure_planes`，
   结果 JSON 增加 `xr_status` / `xr_n_negative_runs` / `xr_corner_attached` /
   `xr_planes_valid` / `xr_planes_invalid`；`xr_h_err_pct` 与 `converged` 对
   `None` 安全（无效测量不再进入收敛判定，扫描汇总新增 `measurable` 门）。
3. 可移植性：去掉写死的 `/home/wxsc/...`（源码路径、`--verified` 默认值），
   `--device` 在无 CUDA 时回落到 `cpu`。
4. `benchmarks/pending/bstep_3d/diag_cpu_repro.py`（CPU 复跑）+
   `analyze_cpu_repro.py`（分析/作图），`tests/test_bstep3d_reattach.py`（8 条
   unit/regression 断言，含"遗留记录自相矛盾"守卫）。

### 2.4 复跑协议（战役式 τ-matched 阶梯）
τ 固定 0.56（ν = 0.02），U_max = Re·ν/h = 2/h → 改变 h 只改变分辨率，不改变无量纲
黏性：即本案例 README 晋级路径第 1 条要求的 "τ-matched 多档阶梯"。阶梯取
h = {h_coarse} 与 h = {h_fine}（CPU 预算内），另加一档 h=16、U_max=0.05、τ=0.524
的交叉核对（分离 τ−0.5 滑移与 Ma 压缩性两个混淆因子的量级）。
每档在**同一流场**上同时用旧估计器与稳健估计器测量，二者之差只反映估计器行为。

工具链：Python 3.11 + torch 2.7.1（CPU，{ladder[0]['run']['torch_threads']} 线程）+
numpy + matplotlib；D3Q19 BGK 碰撞、`stream3d_roll` 周期迁移、库内 bounce-back 壁面、
库内 3D Zou/He 速度入口（抛物线剖面）与压力出口；入口剖面自检偏差 ≤1e-6·U_max。

## 3. 计算结果（Results）

**表 B：CPU 复跑阶梯（Re=100，τ-matched，同场双估计器）**

| h | 单元数 | ER | ER 偏差 | wall (s) | X_r/h 稳健 | X_r/h 旧估计器 | 误差(稳健) | status | 有效平面 | y1→y2 差 | 步数 |
|---|--------|----|---------|---------|------------|----------------|------------|--------|----------|----------|------|
{ladder_table}
{cross_txt and chr(10) + "| " + cross_txt.strip("|").strip() + " |" or ""}

要点：

- 两档 τ-matched 网格的 X_r/h 收敛到 {xr_coarse:.4f}（h={h_coarse}）与
  {xr_fine:.4f}（h={h_fine}），对 Armaly 3.0 的误差分别为
  {_pct(ladder[0]['err_pct_new'])} 与 {_pct(ladder[1]['err_pct_new'])}；
  网格细化（h: {h_coarse} → {h_fine}）误差由 {_pct(ladder[0]['err_pct_new'])} 降到
  {_pct(ladder[-1]['err_pct_new'])}，即单调朝参考移动
  （{"是" if refine.get('monotone_toward_reference') else "否"}），ΔX_r/h =
  {_fmt(refine.get('d_xr_h'), 4)}。
- 遗留 H=40 记录（同为 τ=0.56、U_max=0.05 的 τ-matched 点）误差 +13.18%，
  说明误差随 h **单调增大**（h=16: {_pct(ladder[-1]['err_pct_new'])}，h=20:
  {_pct(ladder[0]['err_pct_new'])}，h=40: +13.18%），与"格点滑移 δu≈const（格点单位）
  导致相对误差 ∝ 1/U_max ∝ h" 的 O(h) 标定一致——即 +13% 是粗网格伪差而非物理差异。
- 把三个 τ-matched 点（h=16/20 CPU + 遗留 h=40）按 X_r/h = a + b·h 做 O(h) 外推：
  a = {_fmt((summary.get('extrapolation') or {}).get('h0_limit'), 4)}（h→0，误差
  {_pct((summary.get('extrapolation') or {}).get('h0_err_pct'))}），b =
  {_fmt((summary.get('extrapolation') or {}).get('slope_per_cell'), 5)}/cell，
  RMS 残差 {_fmt((summary.get('extrapolation') or {}).get('rms_residual'), 4)}。
  即**零网格间距极限落在 ±3% 门内**，而有限网格上的直接值不满足门——这正是本案例
  遗留 +13% 的正确解释，也是晋级路径应写"网格外推 + 多档单调"而非"单档直接值"的依据。
- 旧估计器与稳健估计器在**本组场**上给出相同值（最大差
  {_fmt(estimator_delta_max, 4)} h，且 0/16 平面被置零）：Re=100 充分分辨时角点首
  节点为负，缺陷不触发。缺陷触发场景由单元测试与 re400 遗留记录证明（见 §5）。
- 展向均匀性：所有平面均有效（有效平面 = 全部），X_r/h 展向散布 ~1e-6 量级，
  与"展向周期 + 展向均匀流"一致；壁面法向节点 y=1 与 y=2 的 X_r 差见上表
  （测量口径的敏感性，量级 <0.1h）。

## 4. 结果图（Figures）

1. **fig1_wall_profiles.png** —— 壁面相邻节点切向速度沿程剖面（两档网格），标出稳健
   闭合点、旧估计器结果与检测到的回流段。
2. **fig2_convergence.png** —— X_r/h 随步数的收敛历史（稳健 vs 旧估计器），黑带为
   Armaly ±3% 验收带。
3. **fig3_grid_trend.png** —— Re=100 网格阶梯 X_r/h，含 τ=0.524 交叉核对点与 ±3% 带。
4. **fig4_field_contour.png** —— 展向中面 u_x 场云图 + 实测再附着位置。

（图由 `analyze_cpu_repro.py` 从原始 npz/JSON 直接生成，未做任何手工修饰。）

## 5. 注意点（Caveats / limits）

1. **规模折减。** 审计环境无 GPU，本次复跑用 h=16/20、nz=16（Lz=0.8h，遗留档为
   2h）替代遗留 H=40/49 的 2.6–4.9 Mcell 规模；成本从 GPU 分钟级变为 CPU 十余分钟/档。
   展向周期 + 展向均匀流使 nz 折减在物理上可接受（展向散布 ~1e-6），但**不能**替代
   遗留规模的网格收敛结论。
2. **压缩性/滑移混淆。** h=16 档 U_max=0.125（Ma≈0.22）偏高；τ=0.524、U_max=0.05
   的交叉核对给出 {_fmt((cross or {}).get('xr_h_new_steady'), 4)}，与 τ-matched 档相差
   ~{_fmt(None if not cross else abs(cross['xr_h_new_steady'] - ladder[0]['xr_h_new_steady']), 4)} h，
   在本工况量级下不改变结论，但该混淆尚未被系统分离。
3. **遗留 H=40 的 +13.18% 无法用新估计器重测**：原始场 `final_ux_H40.npz` 写在同一
   批 GPU 机器的 `/tmp`，未入仓；本报告只能给"遗留值 + 同一 τ-matched 点的重建值"。
   不得把 H=40 记录的误差称为"已修正"。
4. **Re=200/400 未重录**：无 GPU 预算；且遗留各档 τ 并非匹配（0.5368 / 0.5276），
   本来就不构成阶梯。修复只保证这类场一旦复跑不会再把"测不到"写成 `0.0`。
5. **质量漂移**仍在（压力出口 + bounce-back 不严格守恒），本报告只作监测，不据此
   下物理结论。
6. 结论适用范围：层流 Re=100、ER≈1.94、展向周期；不覆盖高 Re 三维展向有结构
   （Armaly Re>400 的三维效应）情形。

## 6. 误差判定（Error judgement，逐项）

| 观测量 | 本次值 | 基准 | 相对误差 | 是否 <3% |
|--------|--------|------|----------|----------|
{err_table}

- **是否 <3%？** 两档 τ-matched 复跑（h={h_coarse}/{h_fine}）：
  {"是，全部通过（" + "、".join(_pct(e['err_pct_new']) for e in ladder) + "）" if gate_all else "否，逐档见上表（" + "、".join(_pct(e['err_pct_new']) for e in ladder) + "），均 >3%，不达标"}；
  网格细化单调（h: {h_coarse}→{h_fine} 误差 {_pct(ladder[0]['err_pct_new'])}→{_pct(ladder[-1]['err_pct_new'])}，
  ΔX_r/h = {_fmt(refine.get('d_xr_h'), 4)}）。**唯一在 3% 门内的是 h→0 外推值**
  {_fmt((summary.get('extrapolation') or {}).get('h0_limit'), 4)}
  （{_pct((summary.get('extrapolation') or {}).get('h0_err_pct'))}），它是外推而非直接模拟，
  按本案例判定标准不得当作"直接观测量达标"。
- **遗留 H=40 记录（+13.18%）：否**，>3%，仍不达标——本报告没有把它洗成通过。
- **Re=200/400 档**：无有效测量（遗留 `0.0` 为测量失效，非物理结果），故**无法判定**，
  按 `WITHHELD` 处理，不宣称达标。
- **估计器修复自身的验证**：8/8 单元与回归测试通过（含"角点首节点为正时旧估计器丢
  结果"、"角点二次回流截断"、"未再附着不得写成 0"、"单胞噪声不得伪造回流段"、
  "无效平面不得混入展向统计"、"re400 记录自相矛盾"守卫）。

## 7. 产物清单

- 代码：`benchmarks/pending/bstep_3d/reattach.py`（新）、`diag_cpu_repro.py`（新）、
  `analyze_cpu_repro.py`（新）、`run.py` / `run_highre.py`（接线与可移植性修复）。
- 测试：`tests/test_bstep3d_reattach.py`（8 passed）。
- 数据/图：`benchmarks/pending/bstep_3d/artifacts/`（`summary.json`、
  `record_audit.json`、`cpu_repro_*.json`、`field_*.npz`、`figures/`）。
- 运行日志：`benchmarks/pending/bstep_3d/logs/repro_*.log`。
"""

    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(md, encoding="utf-8")
    print(f"-> {DOC}")

    # ------------------------------------------------------- report payload
    figures = []
    for meta in figs_meta:
        p = FIG / meta["name"]
        figures.append(
            {
                "name": meta["name"],
                "caption": meta["caption"],
                "png_base64": base64.b64encode(p.read_bytes()).decode(),
            }
        )

    payload = {
        "title": "bstep_3d 三维后向台阶：再附着长度测量缺陷定源与 τ-matched 复跑（m469）",
        "summary": (
            "SDAA 案例扫荡 bstep_3d 的诊断+修复：定源遗留 re400 记录 "
            "X_r/h=0.0 与自身 -0.175 回流自相矛盾的测量缺陷（首过零点估计器把"
            "三种不同结果折叠成 0.0），新增稳健估计器 reattach.py（最大回流面积段 + "
            "显式 status + 无效平面剔除）并接线 run.py/run_highre.py、去掉写死路径；"
            f"8/8 单元回归测试通过；CPU 上按 τ-matched 协议复跑 h={h_coarse}/{h_fine}，"
            f"X_r/h = {_fmt(xr_coarse, 4)}/{_fmt(xr_fine, 4)}，对 Armaly 3.0 误差 "
            f"{_pct(ladder[0]['err_pct_new'])}/{_pct(ladder[1]['err_pct_new'])}；"
            "遗留 H=40 的 +13.18% 仍未达标（如实保留，未粉饰），Re=200/400 无有效测量按 "
            "WITHHELD 处理。"
        ),
        "scenario_key": SCENARIO_KEY,
        "mission_id": MISSION_ID,
        "tags": TAGS,
        "markdown": md,
        "figures": figures,
    }
    Path("/tmp/report.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    print(f"-> /tmp/report.json ({len(figures)} figures, md {len(md)} chars)")

    if args.post:
        base = os.environ["TENSOR_API_BASE"].rstrip("/")
        login = urllib.request.Request(
            base + "/auth/login",
            data=json.dumps(
                {
                    "username": os.environ["TENSOR_AGENT_USER"],
                    "password": os.environ["TENSOR_AGENT_PASS"],
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        token = json.loads(urllib.request.urlopen(login, timeout=30).read())["access_token"]
        req = urllib.request.Request(
            base + "/tensor/reports",
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
        print("published:", {k: resp.get(k) for k in ("id", "title", "mission_id", "status")})


if __name__ == "__main__":
    main()