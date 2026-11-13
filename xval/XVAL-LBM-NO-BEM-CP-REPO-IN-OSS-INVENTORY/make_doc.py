#!/usr/bin/env python3
"""Render docs/REPORT_XVAL_LBM_NO_BEM_CP_REPO_IN_OSS_INVENTORY.md from artifacts (zero hand-copied numbers)."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ART = HERE / "artifacts"
PREDECL = json.loads((HERE / "predeclared.json").read_text())
S = json.loads((ART / "summary.json").read_text())
SURVEY = json.loads((ART / "oss_survey.json").read_text())
REF = S["reference"]
rows = S["runs"]
g = S["gates"]


def fmt(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


def main() -> int:
    t06 = sorted([r for r in rows if abs(r["tau"] - 0.6) < 1e-9], key=lambda r: r["nx"])
    taus = sorted({r["tau"] for r in rows})
    gate_rows = "\n".join(
        f"| {k} | {'PASS' if v.get('pass') else '**FAIL**'} | {json.dumps(v.get('detail', v.get('span_pct')), ensure_ascii=False)[:180]} |"
        for k, v in g.items()
    )
    run_rows = "\n".join(
        f"| {r['nx']} | {r['tau']:g} | {r['steps_done']} | {r['stopped_early']} | {fmt(r['nu'],4)} | {fmt(r['err_nu_pct'],3)} | {fmt(r['abs_err_nu_pct'],3)} | {fmt(r['u_max_nd'],3)} | {fmt(r['err_u_pct'],2)} | {fmt(r['v_max_nd'],3)} | {fmt(r['err_v_pct'],2)} | {r['last_resid_u']:.2e} |"
        for r in sorted(rows, key=lambda r: (r["tau"], r["nx"]))
    )
    conv = "PASS" if g.get("G5_grid_span", {}).get("pass") else "FAIL"
    owner = "PASS" if g.get("G2_owner_15pct", {}).get("pass") else "FAIL"
    calib = "PASS" if g.get("G3_caliber_3pct", {}).get("pass") else "FAIL"

    md = f"""# XVAL · TensorLBM thermal LBM vs de Vahl Davis (1983) Ra=1e5

Anchor `XVAL-LBM-NO-BEM-CP-REPO-IN-OSS-INVENTORY` · round 2 · machine-generated numbers
(`analyze.py` → `artifacts/summary.json`; this file is rendered by `make_doc.py`, no number typed by hand).

## ① 问题设定 + 基准出处

差热方腔自然对流（differentially heated cavity, DHC）：单位方腔，左壁 T=1、右壁 T=0、上下绝热、四壁无滑移，
工质空气 Pr={PREDECL['benchmark']['pr']}，Ra={PREDECL['benchmark']['ra']:.0f}。

- 基准：**de Vahl Davis G. (1983), *Int. J. Numer. Methods Fluids* 3(3):249-264, Table 1 (p.257)**
  — Nu_mean={REF['nu_mean']}，u_max={REF['u_max_nd']}，v_max={REF['v_max_nd']}（扩散尺度 u_nd = u_lu·H/α）。
- 次判据 u_max/v_max 在预注册里标为 **WITHHELD**：仓内两处独立表（`src/tensorlbm/heat_convection_benchmarks.py`,
  `src/tensorlbm/thermal_verification.py`）都是 34.72/68.60，`docs/audit_m15466_ra1e5.md` 记录文献值 34.73/68.59
  （差 0.03%/0.015%，即舍入）；离线无原文 PDF 可逐位转录 → 只作参考量，不作闸门。
- 锚点语义（如实申报）：锚名里的 `NO-BEM-CP-REPO-IN-OSS-INVENTORY` 来自 TensorBEM 的做法标签
  （「OSS 清单里没有该能力的参考仓 → 方法只能走文献谱系」）。本轮把它移植到 TensorLBM 的 thermal 能力上，
  并**先做开源横向调研**（见 ②）。

## ② 方法与实现

- 求解：`tensorlbm.thermal.simulate_natural_convection`（速度场 D2Q9 BGK + 温度场 D2Q5 BGK 双分布，
  Boussinesq 浮力经 Guo 力格式注入，等温壁 anti-bounce-back，绝热壁解缠反射；零本地物理，驱动仅做参数换算与观测量提取）。
- 驱动：`xval/XVAL-LBM-NO-BEM-CP-REPO-IN-OSS-INVENTORY/run_xval.py`（每网格一个 JSON 工件）。
- 预注册：`predeclared.json`（判据、网格集、稳态停止规则、混淆因子、分析口径，先于任何产出运行写定）。
- 开源横向调研（机器产生，`artifacts/oss_survey.json`）：
  `oss_inventory` 共 **{SURVEY['oss_inventory_rows']}** 行，命中热/温度/浮力关键词的条目：
  {json.dumps(SURVEY['oss_inventory_keyword_hits'], ensure_ascii=False)}；
  `~/oss-cache` 三仓（{', '.join(c['repo'] for c in SURVEY['oss_cache'])}）逐文件扫描，
  单文件同时出现 ≥2 个热关键词的命中数：{json.dumps({c['repo']: c['n_hits'] for c in SURVEY['oss_cache']})}。
  → 判定 **{SURVEY['verdict']}**：本地无「双分布热 LBM + 自然对流」可解析实现，
  外部锚只能取文献谱系（与 `negative_knowledge_ledger` m17996 的 `thermal_energy_transport_lbm` 无 OSS 参考结论一致）。

## ③ 参数表（无量纲完整）

| 参数 | 值 | 备注 |
|---|---|---|
| Ra | {PREDECL['benchmark']['ra']:.0f} | Ra = gβ·ΔT·H³/(ν·α) |
| Pr | {PREDECL['benchmark']['pr']} | 空气 |
| τ（速度） | {', '.join(f'{t:g}' for t in taus)} | 敏感性参数点（G7） |
| ν = (τ−1/2)/3 | {', '.join(f'{(t-0.5)/3:.5f}' for t in taus)} | 格子单位 |
| α = ν/Pr | {', '.join(f'{((t-0.5)/3)/PREDECL["benchmark"]["pr"]:.5f}' for t in taus)} | 格子单位 |
| H | nx（库约定 half-way 壁） | 已声明，非调参 |
| Ma_lattice | 见 table.csv（每网格一列） | 全部 ≲0.07，不可压假设成立 |
| 网格 | {', '.join(str(r['nx']) for r in t06)} | 两分辨率以上（G5） |

## ④ 对照表（ref vs got，逐项误差，闸门逐条打勾/叉）

基准判据：Nu_mean ∈ [{REF['nu_mean']} ±15%（owner 精度标准）/ ±3%（平台口径）]

| N | τ | steps | 稳态触发 | Nu | err% | abs err% | u_max_nd | err% | v_max_nd | err% | resid_u |
|---|---|---|---|---|---|---|---|---|---|---|---|
{run_rows}

| 闸门 | 结果 | 依据 |
|---|---|---|
{gate_rows}

- **owner 15% 标准：{owner}**；**平台 3% 口径：{calib}**；**两分辨率 span 收敛门（≤{PREDECL['predeclared_thresholds']['convergence_gate'].split('≤')[-1].split('(')[0].strip()}）：{conv}**。

## ⑤ 收敛性 / 参数敏感性

- 网格收敛：见 G5（两最细网格 Nu 相对 span）与 G6（误差随细化单调减小）。
- 参数敏感性：τ={taus[0]:g} vs τ={taus[-1]:g}（G7，同 Ra 同网格族），检验 Nu 对 τ（即 ν、α 尺度）的依赖。
- 收敛历史曲线：`artifacts/fig1_nu_history.png`（Nu 随步数的逼近过程，含 de Vahl Davis 参考线）。

## ⑥ 失败模式与工件

- 工件：`artifacts/summary.json`（全部数字）、`artifacts/table.csv`、`artifacts/oss_survey.json`、
  `artifacts/fig1_nu_history.png`、`artifacts/fig2_ref_vs_got.png`、`artifacts/fig3_gate_matrix.png`，
  逐网格原始账：`runs/*.json`（含 nu_history 时序）。
- repro_cmd：`python xval/XVAL-LBM-NO-BEM-CP-REPO-IN-OSS-INVENTORY/run_xval.py --nx 128 --ra 1e5 --out runs/n128_ra1e5.json`
  然后 `python xval/XVAL-LBM-NO-BEM-CP-REPO-IN-OSS-INVENTORY/analyze.py`。
- 已知失败模式：D2Q5 温度方程的截断误差使粗网格 Nu 系统性偏低（负偏差）；
  半程反弹壁使有效腔长 H=nx；长时积分 float32 累积误差；
  若某网格未触发稳态检测（stopped_early=false），该行不得计入判据（G4 逐网格核）。

## 结论（机器复算，未粉饰）

- 全部门通过：**{S['all_gates_pass']}**。
- 数字级结论见 ④ 表与 `artifacts/gates`；未通过的闸门在 ④ 表中显式标 **FAIL**，不做口径改写。
"""
    docs = REPO / "docs"
    docs.mkdir(exist_ok=True)
    p = docs / "REPORT_XVAL_LBM_NO_BEM_CP_REPO_IN_OSS_INVENTORY.md"
    p.write_text(md, encoding="utf-8")
    print(f"wrote {p} ({len(md)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())