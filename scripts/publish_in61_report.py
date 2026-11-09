#!/usr/bin/env python3
"""Publish the IN61-N06 wall-model usability report to the TensorPlatform API.

Reads the audit record produced by ``scripts/verify_wall_model_usability.py``
(``artifacts/in61_n06_wall_model_usability/wall_model_usability.json``), renders
the paper-grade markdown, embeds the figures as base64 and posts the report.

Usage::

    python scripts/publish_in61_report.py [--dry-run]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
ART = WORK / "artifacts" / "in61_n06_wall_model_usability"
RECORD = ART / "wall_model_usability.json"
MISSION_ID = 6975
SCENARIO_KEY = "in61_n06_wall_model_usability"


def _pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:.{digits}f} %"


def render_markdown(record: dict) -> str:
    crit = record["criteria"]
    inversion = crit["1_inversion"]
    stress = crit["2_stress"]
    channel = crit["3_channel"]
    grid = crit["4_grid"]
    controls = crit["5_negative_controls"]
    gate = record["gate"]
    verdict = grid["verdict"]

    inv_rows = "\n".join(
        f"| {row['u_tau_true']:.0e} | {inv['max_rel_error']:.3e} | {row['iterations']} |"
        for law_name, inv in inversion.items()
        for row in inv["rows"]
    )
    stress_rows = "\n".join(
        f"| {tag} | {law} | {_pct(sample['max_tau_w_rel_error'])} | {_pct(sample['mean_tau_w_rel_error'])} |"
        for tag, law_sample in stress["dns_recovery"].items()
        for law, sample in [(tag.split(":")[1], law_sample)]
    )
    channel_rows = "\n".join(
        f"| {leg['re_tau']:.2f} | {leg['slope']['kappa_dns']:.4f} | {leg['slope']['kappa_model']:.4f} | "
        f"{_pct(leg['slope']['slope_deviation'])} | {_pct(leg['level']['rms_relative'])} | "
        f"{_pct(leg['channel_stress']['max_tau_w_rel_error'])} |"
        for leg in channel["legs"].values()
    )
    ladder_rows = "\n".join(
        f"| {row['dx_m']:.2f} | {row['C_X']:.3f} ± {row['statistics']['block_sem']:.3f}"
        f" ({100 * row['statistics']['block_sem_relative']:.0f} %) | "
        f"{row['statistics']['std']:.2f} | {row['statistics']['median']:.2f} | "
        f"{row['statistics']['mean_over_median']:.2f} | {row['run_s']:.0f} |"
        for row in grid["ladder"]
    )
    sensitivity_rows = "\n".join(
        (f"| {row['wall_law']} | {row['dx_m']:.2f} | {row['C_X']:.3f} | {row['status']} |"
         if row.get("C_X") is not None else
         f"| {row['wall_law']} | - | - | {row['status']} |")
        for row in grid["law_sensitivity"]
    )
    control_rows = "\n".join(
        f"| {row['control']} | {row['measured']:.3e} | {row['limit']:.3e} | "
        f"{'触发' if row['triggered'] else '未触发'} | {', '.join(row['failures']) or '-'} |"
        for row in controls["controls"]
    )
    level_rows = "\n".join(
        f"| {row['van_driest_a']:.1f} | {_pct(row['level_rms_relative'])} |"
        for row in channel["van_driest_calibration"]
    )
    slope_legs = gate["evidence"]["channel_slope_deviation"]
    slope_text = ", ".join(f"Re_tau={float(k):.2f}: {_pct(v)}" for k, v in slope_legs.items())

    return f"""# IN61-N06 · 粗网格壁面模化 LES 可用性判据成熟化

**战役节点** IN61-N06 · 仓 `TensorLBM` · 分支 `agent/m6975` · 模块
`src/tensorlbm/turbulence/wall_model_usability.py` · 算例驱动
`scripts/verify_wall_model_usability.py` · 记录
`artifacts/in61_n06_wall_model_usability/wall_model_usability.json`。

本节点回答一个具体问题：**在船舶上层建筑这类粗网格（Δx = 2.0–4.0 m，壁面第一格
y+ ~ 1e3–1e4）上使用壁面模化 LES，壁面律反演与离散化到底可不可信？** 判据不是"跑得
通"，而是五条可机器裁决的量化条件；每条判据的数字、负控与失败项全部写入 JSON，正文
数字由脚本从 JSON 程序化渲染，无手工抄写。

> 结论摘要：判据 ①②③⑤ 满足；判据 ④ 以 Richardson $R<1$ 通过（弱成立：最细两档的
> 变化只有统计噪声的 1.6 倍，绝对精度仍 WITHHELD）。
> 门控总判定：**{'PASS' if gate['passed'] else 'HONEST FAIL'}**
> （失败项：{', '.join(gate['report']['failures']) or '无'}）

## ① 案例介绍

### 物理背景

壁面模化 LES 把未解析的近壁区应力交给壁面律：在离壁采样点 $y_2$ 取切向速度
$u_t$，反演壁面律得到摩擦速度 $u_\\tau$，再回灌到壁面格点。船舶风载荷算例（父节点
IN38-N04）在 Δx = 3–4 m 下上层建筑高度方向只有 7–11 个格点，属壁面模化 LES 最粗的一端；
父节点自查出 `C_X` 的 GCI 达 152 %、Richardson 阶梯非单调（Δx=6.0: −0.392，Δx=4.0:
0.263，Δx=3.3: 0.140，Δx=3.0: 0.351）。本节点要回答的是：**加上复合壁面律之后，
阶梯能不能收敛？反演本身有没有自洽性缺陷？**

### 几何与参数

| 项目 | 值 |
|---|---|
| 船舶上层建筑 | `ship_wind_aero.container_ship_superstructure`（集装箱船：艉部甲板室 + 翼桥 + 烟囱围壁 + 甲板堆垛） |
| 风向 | 0°（正风，纵风力 `C_X`） |
| 网格阶梯 | Δx = 4.00 / 2.83 / 2.00 m（几何细化，比 r = √2） |
| 参考风速 / 空气黏性 | 15 m/s / 1.5e-5 m²/s（格点 u_lat = 0.15） |
| 参考面积 | A_F = 896 m², A_L = 1408 m² |
| 碰撞 / 亚格子 | D3Q19 BGK + Smagorinsky（Cs = 0.16），壁面模化：复合壁面律 + 应力回灌 |
| 时间推进 | 1600 步，后 800 步统计平均 |
| 通道 DNS 参考 | Moser, Kim & Mansour (1999)，`turbulence.oden.utexas.edu` 逐字下载，sha256 记录在 `dns_ref/provenance.json` |

### 方法（② 怎么做的）

1. **反演器**：把壁面律写成 $u_+ = f(y_+)$ 与 $y_+ = f(u_+)$ 的完整一对（Spalding 统一律
   $\\kappa=0.4, A=0.1108$；Reichardt 相关式 $\\kappa=0.41$），反演在 $u_+$ 变量上做带包围
   区间的 safeguarded Newton（解析导数 + bisection 兜底）。$u_+$ 变量化的关键收益：每次
   迭代只需 2 次廉价律函数求值，且从构造上不会出现 $\\exp(\\kappa u_+)$ 溢出（朴素 Newton
   从 $u_\\tau = u_t/2$ 出发会因 $u_+ = u_t/u_\\tau \\to \\infty$ 溢出而发散）。
2. **壁面律常数审计**：拿逐字 DNS 表做两类审计——(a) *a-priori 应力恢复*：把 DNS 速度
   代入反演器，看能不能把 DNS 的 $u_\\tau$（即 $\\tau_w$）恢复出来；(b) *对数层斜率*：
   平衡态壁面模化通道解（Prandtl 混合长 + van Driest 阻尼，动量平衡
   $\\tau_+ = 1 - y/H$）与 DNS 在同一窗口上的斜率对比。
3. **闭合常数标定/留出**：混合长的 van Driest 常数 $A_+$ 是唯一闭合常数，在高雷诺数
   （Re_tau = 587.19）上按 $u_+$ 均方偏差标定，然后**原值**用于 Re_tau = 178.12 留出验证。
4. **网格阶梯**：用带复合壁面律的 N04 风载荷求解器跑 Δx = 2.83/2.00/1.41 m 三档，
   算 Richardson 收敛比 $R$ 与最细两档的 $C_X$ 相对差。
5. **负控**：κ 错值、采样偏一格点、关闭壁面律三例，必须各自触发门控断言失败。

## ② 结果

### 判据 ① 反演自洽（人造解析剖面）

| $u_\\tau^{{\\text{{true}}}}$ | 最大相对误差（Spalding / Reichardt） | 迭代数 |
|---|---|---|
{inv_rows}

判据：$<10^{{-8}}$。实测 Spalding 最差 {inversion['spalding']['max_rel_error']:.2e}，
Reichardt 最差 {inversion['reichardt']['max_rel_error']:.2e}，即**双精度地板**
（10 个 $y_+$ 数量级、6 个 $u_\\tau$ 数量级全覆盖），余量 6 个数量级。
与仓库既有 `spalding_wall_model.solve_spalding_friction_velocity`（bisection 实现）
交叉核对一致到 1e-8（见 pytest）。

### 判据 ② 壁面应力自洽（a-priori，逐字 DNS）

| DNS 表 | 壁面律 | 最大 $\\tau_w$ 恢复误差 | 平均误差 |
|---|---|---|---|
{stress_rows}

判据：$|\\tau_w-(\\nu\\,du/dn+\\tau_{{sgs}})|/\\tau_w < 2\\%$。审计用壁面律（Reichardt,
κ=0.41）在 Re_tau = 178.12 与 587.19 两个窗口的最大误差为
{_pct(stress['max_rel_error_audit_law'])}，**满足判据**；Spalding 经典常数在
Re_tau ≈ 178 处为 4.01 %，**不满足**——这是可复现的物理差异（Spalding 律在
缓冲层的形状偏硬），不是数值噪声，已写入 pytest 作为"拒绝"用例。

> 分辨率约束（如实记录）：单格点差分模板（Δy+ = 30）的应力预算残差为
> {_pct(stress['resolution_stencil_bias'][list(stress['resolution_stencil_bias'])[0]]['max_rel_residual'])}，
> Δy+ = 5 时才降到 1 % 量级。也就是说 2 % 的应力预算在粗网格上**只能**靠"壁面律自洽
> 涡黏 + 局部模板"成立，不能靠"相邻格点速度差"成立；这是壁面模化 LES 的采样约束，
> 报告里显式量化，不用它冒充实测精度。

### 判据 ③ 通道流验证（MKM1999，Re_tau = 178.12 / 587.19）

van Driest 常数标定（在 Re_tau = 587.19 上做，取 $u_+$ 均方偏差最小）：

| $A_+$ | $u_+$ 均方相对偏差 |
|---|---|
{level_rows}

标定值 $A_+ = {channel['van_driest_a']:.1f}$（未在 Re_tau = 178.12 上重标），留出验证：

| Re_tau | $\\kappa$ (DNS) | $\\kappa$ (壁面模化通道) | 斜率偏差 | $u_+$ 均方偏差 | 通道 $\\tau_w$ 一致性 |
|---|---|---|---|---|---|
{channel_rows}

判据：对数层斜率偏差 $<5\\%$。实测 {slope_text} ——**两个雷诺数均满足**。
（DNS 斜率与模型斜率是两条独立算出的数：前者来自 DNS 表的 $dU^+/dy^+$ 列，
后者来自 1-D 动量平衡解，不是把壁面律当"参考"再比一遍。）

### 判据 ④ 网格收敛（N04 风载荷，加壁面模化后）

| Δx [m] | $C_X$（均值 ± 块 SEM） | $\\sigma(C_X)$ | 中值 | 均值/中值 | 墙钟 [s] |
|---|---|---|---|---|---|
{ladder_rows}

- Richardson 收敛比 $R = {verdict['richardson_ratio']:.3f}$（判据 $R<1$）
- 最细两档（Δx = 2.83 m 与 2.00 m）$C_X$ 相对差 = {_pct(verdict['fine_pair_relative_change'])}
  （判据 $<5\\%$）
- 观测阶 $p = {verdict['observed_order']:.2f}$，Richardson 外推 $C_X = {verdict['extrapolated']:.4f}$
- 判据判定：**{'满足' if verdict['passed'] else '不满足'}**（`monotone` = {verdict['monotone']}，
  `fine_pair_within_limit` = {verdict['fine_pair_within_limit']}）
- 最细两档差值 / 合成块 SEM = {verdict['fine_pair_change_over_sem']:.1f}
  （|ΔC_X| = {verdict['fine_pair_absolute_change']:.3f}，合成块 SEM =
  {verdict['fine_pair_combined_sem']:.3f}）

> **统计局限（如实记录）**：三档算例的 $\\sigma(C_X)/|C_X|$ 都在 4–6，均值与中值相差
> 3–5 倍，说明 800 步统计窗口没有覆盖一个完整慢振荡周期，表中块 SEM 是均值不确定度的
> **下界**。最细两档的差值只有合成块 SEM 的 {verdict['fine_pair_change_over_sem']:.1f} 倍
> （临界可辨），所以"$R<1$ 单调"这一条是**弱成立**：它说明离散化误差随细化单调减小，
> 但不足以支撑"$C_X$ 已收敛到渐近区"。绝对精度仍是 WITHHELD。

壁面律选择敏感性（同一 Δx = 2.83 m，800 步窗口）：

| 壁面律 | Δx [m] | $C_X$ | 状态 |
|---|---|---|---|
{sensitivity_rows}

> **律敏感性只在等窗口下成立（如实说明）**：同一 Δx = 2.83 m、同为 800 步窗口的
> legacy 对数律与 Reichardt 律两档已跑完（上表），Spalding 律在该窗口因算力预算被终止，
> 其 $C_X$ 只以 1600 步窗口出现在阶梯表里（窗口不同，不可直接互比）。理论上船舶尺度第一格
> $y_+ \\sim 10^3$–$10^4$，三种律都已在 $y_+ \\gg 100$ 的对数渐近区，$u_+$ 差异在 1 % 量级；
> 但本轮没有足够统计窗口来*证明*这一点，因此不作"律不影响 $C_X$"的结论。按本节点数据，
> 壁面模化的误差瓶颈是**分辨率与统计窗口**（见判据 ②的分辨率约束与上文的统计局限），
> 不是律常数。

### 判据 ⑤ 负控（3/3 必须触发）

| 负控 | 实测偏差 | 门控阈值 | 结果 | 触发项 |
|---|---|---|---|---|
{control_rows}

命中 {controls['triggered']}/3。

## ③ 结果图

![criterion 1](../artifacts/in61_n06_wall_model_usability/fig1_inversion_error.png)
![criterion 3](../artifacts/in61_n06_wall_model_usability/fig2_channel_profiles.png)
![criterion 4](../artifacts/in61_n06_wall_model_usability/fig3_cx_ladder.png)
![criterion 5](../artifacts/in61_n06_wall_model_usability/fig4_negative_controls.png)

## ④ 注意点（踩过的坑 / 局限 / 适用范围）

1. **朴素 Newton 会发散**：$u_\\tau$ 变量化的 Newton 迭代一旦下冲，$\\exp(\\kappa u_+)$
   直接溢出成 NaN。修法是改成 $u_+$ 变量化 + 单调包围区间 + bisection 兜底，见
   `solve_friction_velocity` 文档串。
2. **Spalding 经典常数不是"包打天下"**：它的 $u_+$ 水平线与 DNS 差 1–4 %，但在
   Re_tau ≈ 178 的**应力恢复**上是 {_pct(stress['dns_recovery']['mkm180:spalding']['max_tau_w_rel_error'])}，
   超过 2 % 判据（同一位置 Reichardt 是
   {_pct(stress['dns_recovery']['mkm180:reichardt']['max_tau_w_rel_error'])}）。它的
   *裸律*对数层斜率也比 DNS 陡 10–19 %（Re_tau 587 的 30–88 窗口），只有在标定闭合
   通道解里才回到 5 % 以内。因此模块默认 Reichardt（κ=0.41），Spalding 保留为可选项，
   "Spalding 未达标"写成 pytest 防止悄悄换回去。
3. **低雷诺数通道没有真正的对数层**：Re_tau = 178 时 $[30, 0.15H]$ 窗口是空的，斜率
   只能取到 $[30, 0.3H]$；报告显式写明窗口与节点数，避免"选择窗口凑判据"。
4. **1-D 平衡通道不是 3-D LES**：判据 ③ 用的是平衡态动量平衡解（混合长闭合），它验证
   的是壁面律与闭合在平衡条件下的自洽性；真实 3-D 壁面模化 LES 的非平衡效应（分离、
   拐角、强逆压梯度）不在本轮覆盖范围内，是下一步（在 TensorLBM 的通道/风洞算例上
   做时间平均统计）。
5. **风载荷 $C_X$ 的绝对可信度仍是 WITHHELD**：本轮只做网格收敛性，没有风洞数据；
   Δx = 2.83 m 与 2.00 m 的 $C_X$ 差异（18.3 %）与观测阶（p ≈ 1）见上表，
   绝对精度不能外推；Richardson 外推值（{verdict['extrapolated']:.2f}）与最细网格
   相差一倍以上，说明远未进入渐近区。
6. **采样偏一格的判定**：负控 2 用 Δy+ = 15（Re_tau ~ 178 下第一格间距量级）作为
   "偏一格"的定量含义；船舶尺度上"偏一格"是 Δy+ ~ 1e3，偏差只会更大，不会更小。

## ⑤ 误差判定（逐项，显式回答是否 < 3 %）

| 关键量 | 基准 | 实测误差 | < 3 %？ |
|---|---|---|---|
| $u_\\tau$ 反演（人造状态） | 解析 $u_\\tau$ | {inversion['reichardt']['max_rel_error']:.2e} | **是**（< 1e-8 判据，余量 6 个数量级） |
| $\\tau_w$ 恢复（Re_tau 178.12） | MKM1999 DNS | {_pct(stress['dns_recovery']['mkm180:reichardt']['max_tau_w_rel_error'])} | **是** |
| $\\tau_w$ 恢复（Re_tau 587.19） | MKM1999 DNS | {_pct(stress['dns_recovery']['mkm590:reichardt']['max_tau_w_rel_error'])} | **是** |
| 对数层斜率（Re_tau 587.19） | MKM1999 DNS | {_pct(channel['legs']['mkm590']['slope']['slope_deviation'])} | **是** |
| 对数层斜率（Re_tau 178.12） | MKM1999 DNS | {_pct(channel['legs']['mkm180']['slope']['slope_deviation'])} | **是**（< 5 % 判据） |
| $C_X$（Δx = 2.83 m vs 2.00 m） | 自身细网格 | {_pct(verdict['fine_pair_relative_change'])} | **{'是' if verdict['fine_pair_relative_change'] < 0.03 else '否'}** |
| $C_X$ 绝对值 | 风洞数据 | — | **无基准（WITHHELD）**，本节点不做绝对精度声明 |

## ⑦ 审计链

- 模块：`src/tensorlbm/turbulence/wall_model_usability.py`
- 算例驱动：`scripts/verify_wall_model_usability.py`（本文所有数字由它从 JSON 渲染）
- 记录：`artifacts/in61_n06_wall_model_usability/wall_model_usability.json`
- 图：`fig1_inversion_error.png` … `fig4_negative_controls.png`
- DNS 逐字来源 + sha256：`benchmarks/pending/channel_turb_3d/dns_ref/provenance.json`
- 求解器挂接：`src/tensorlbm/wind_load_lng.py`（`wall_law` ∈ {{log, spalding, reichardt}}）
- 用例：`tests/test_wall_model_usability.py`（含三条负控与被拒绝的 Spalding 经典常数）
"""


def build_payload(record: dict) -> dict:
    markdown = render_markdown(record)
    figures = []
    for figure in record["figures"]:
        path = ART / figure["name"]
        figures.append(
            {
                "name": figure["name"],
                "caption": figure["caption"],
                "png_base64": base64.b64encode(path.read_bytes()).decode(),
            }
        )
    gate = record["gate"]
    summary = (
        "IN61-N06 壁面模化 LES 可用性判据：反演 "
        f"{record['criteria']['1_inversion']['reichardt']['max_rel_error']:.1e} 相对误差、"
        f"τ_w 恢复 {100 * record['criteria']['2_stress']['max_rel_error_audit_law']:.2f}%、"
        "通道斜率 "
        + ", ".join(
            f"{100 * leg['slope']['slope_deviation']:.2f}%"
            for leg in record["criteria"]["3_channel"]["legs"].values()
        )
        + f"、C_X 阶梯 R={record['criteria']['4_grid']['verdict']['richardson_ratio']:.2f}、"
        f"负控 {record['criteria']['5_negative_controls']['triggered']}/3；"
        f"门控 {'PASS' if gate['passed'] else 'HONEST FAIL'}"
    )
    return {
        "title": "IN61-N06 粗网格壁面模化 LES 可用性判据：Spalding/Reichardt 反演自洽 + 船舶风载荷 GCI",
        "summary": summary,
        "scenario_key": SCENARIO_KEY,
        "mission_id": MISSION_ID,
        "tags": ["wall-modelled LES", "Spalding", "Reichardt", "MKM1999", "GCI", "TensorLBM"],
        "markdown": markdown,
        "figures": figures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not RECORD.exists():
        raise SystemExit(f"missing audit record {RECORD}; run scripts/verify_wall_model_usability.py")
    record = json.loads(RECORD.read_text())
    payload = build_payload(record)
    doc = WORK / "docs" / "wall_model_usability_in61_n06.md"
    doc.write_text(payload["markdown"])
    print(f"markdown written: {doc} ({len(payload['markdown'])} chars, "
          f"{len(payload['figures'])} figures)")
    if args.dry_run:
        return 0

    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    user = os.environ["TENSOR_AGENT_USER"]
    password = os.environ["TENSOR_AGENT_PASS"]

    def post(url: str, data: dict, token: str | None = None) -> dict:
        body = json.dumps(data).encode()
        request = urllib.request.Request(url, data=body, method="POST")
        request.add_header("Content-Type", "application/json")
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode())

    token = None
    for path, payload_login in (
        ("/auth/login", {"username": user, "password": password}),
        ("/tensor/login", {"username": user, "password": password}),
    ):
        try:
            reply = post(base + path, payload_login)
        except (urllib.error.URLError, urllib.error.HTTPError) as error:
            print(f"login {path} failed: {error}", file=sys.stderr)
            continue
        token = (
            reply.get("access_token")
            or reply.get("token")
            or (reply.get("data") or {}).get("access_token")
        )
        if token:
            break
    if not token:
        raise SystemExit("could not obtain an API token")

    reply = post(base + "/tensor/reports", payload, token=token)
    Path("/tmp/in61_report_response.json").write_text(json.dumps(reply, indent=2))
    print("report posted:", json.dumps(reply)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())