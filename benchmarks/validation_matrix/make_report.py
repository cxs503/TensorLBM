"""Publish the validation-matrix build report (six-element, paper-grade).

Reads the matrix registry, the generated status table and the two CI-tier
artifacts, assembles the markdown, embeds the PNG figures as base64 and POSTs
to the platform report API.

Usage::

    python benchmarks/validation_matrix/make_report.py [--dry-run]

Environment: ``TENSOR_API_BASE``, ``TENSOR_AGENT_USER``, ``TENSOR_AGENT_PASS``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CASES = HERE / "cases"
FIGURES = HERE / "figures"


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _figure(name: str, caption: str) -> dict | None:
    path = FIGURES / name
    if not path.exists():
        return None
    return {
        "name": name,
        "caption": caption,
        "png_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
    }


def build_markdown() -> tuple[str, list[dict], str]:
    matrix = _json(HERE / "matrix.json")
    db = _json(CASES / "dam_break_ritter_ci" / "result.json")
    drop = _json(CASES / "static_droplet_laplace_ci" / "result.json")
    frozen = [_json(p) for p in sorted(FIGURES.glob("provenance.json"))]

    db_rows = "\n".join(
        f"| {lv['level']} | {lv['metrics']['front_err_pct']:+.3f} | "
        f"{lv['metrics']['profile_l2_pct']:.3f} | {lv['metrics']['mass_drift_pct']:.2e} | "
        f"{lv['detail']['steps']} | {lv['detail']['elapsed_s']:.2f} |"
        for lv in db["levels"] + db["resolution_ladder"]
    )
    drop_rows = "\n".join(
        f"| {lv['level']} | {lv['metrics']['sigma_eff']:.6f} | "
        f"{lv['metrics']['laplace_max_dev_pct']:.3f} | {lv['metrics']['R2']:.6f} | "
        f"{lv['metrics']['intercept_frac_pct']:.3f} | "
        f"{lv['metrics'].get('sigma_pair_dev_pct', float('nan')):.3f} | "
        f"{lv['detail']['elapsed_s']:.1f} |"
        for lv in drop["levels"]
    )
    drop_ladder = "\n".join(
        f"| {lv['level']} | {lv['metrics']['sigma_eff']:.6f} | "
        f"{lv['metrics']['laplace_max_dev_pct']:.3f} | {lv['metrics']['R2']:.6f} | "
        f"{lv['detail']['elapsed_s']:.1f} |"
        for lv in drop.get("resolution_ladder", [])
    )
    per_radius = []
    for lv in drop["levels"] + drop.get("resolution_ladder", []):
        for r in lv["detail"]["per_radius"]:
            per_radius.append(
                f"| {lv['level']} | {r['R_init']:g} | {r['R_eq']:.3f} | "
                f"{r['dp']:.6f} | {r['sigma_i']:.6f} | {r['dev_sigma_pct']:.3f} | "
                f"{r['step']} | {r['converged']} |"
            )
    per_radius = "\n".join(per_radius)

    status_md = (HERE / "STATUS_MATRIX.md").read_text() if (HERE / "STATUS_MATRIX.md").exists() else ""

    md = f"""## ① 案例介绍（物理背景 / 几何 / 参数表）

本单交付 **TensorLBM 验证矩阵 v1**：把九条验证谱系（方腔流 / 圆柱绕流 / 球体阻力 /
湍流槽道 / Taylor-Green / 静态液滴 / Rayleigh-Taylor / 溃坝 / 船体阻力与波浪）
落成 **一个机器可判的注册表**（`benchmarks/validation_matrix/matrix.json`），
而不是继续堆算例。核心问题不是"再算一个数"，而是**现有证据能不能被自动复现与自动裁决**。

矩阵把每条谱系拆成五件事：物理算例 + 对照锚（解析解/实验/可信相关式）+ 判据阈值
+ 两档以上"级别序列"（网格或分辨率）+ 一键复现命令。

新增两个 CI 档算例（本单全链跑通）：

* **静态液滴 Laplace**（槽位 6）：二维 Shan–Chen 单组分液滴，ψ=1−e^(−ρ)，
  库内 G_lib=+5（物理 G_eff=−5），τ=1；域 L=4R，周期边界；判据 = 逐半径
  σ_i=Δp·R_eq 相对拟合 σ_eff 偏差 ≤3%（Young–Laplace 线性），两档半径集
  R=12/16/20 与 R=16/20/24。物理步进/拟合**直接 import 已验证的 laplace_droplet 算例**，
  不重写。
* **溃坝 Ritter 自相似**（槽位 8）：一维浅水有限体积（HLL + 静水重构），
  t_end=5√(h0/g)，cfl=0.4；判据 = 前沿律 x_f=2√(gh0)t（5%）、湿区水深剖面
  L2（5%）、质量闭合（1%），级别 nx=200/400，另附 nx=800/1600 分辨率阶梯。

参数表见两个 `result.json`：`config`（步数/半径/CFL）与 `levels[].resolution`。

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **锚点零手抄**：`references.json` 里每个参考值都带 `source`（仓内文件）+
   `token`（该文件里的字面串）+ 值或公式；`reference_audit.py` 重读文件、校验
   token 仍在、校验手写值与 token 中解析出的浮点数一致（或与冻结工件字段一致），
   并记录源文件 sha256。12/12 锚点审计通过；上游改动或手抄错位会直接让审计失败。
2. **统一判据**：冻结工件与现跑 runner 走同一路径——点分路径取数 →
   `criteria.judge_metric`（误差门）→ `criteria.convergence_trend`
   （跨度 ≤3%，≥3 档要求单调）→ 三级状态。同一算例可对"仅最细档设门、
   粗档仅作收敛信息"做显式声明（`tol_pct_by_level`），且该口径必须引用算例自身
   的预注册（如 SUBOFF：最细档 ≤3% 且误差随分辨率单调下降）。
3. **三级状态 + 证据链**：`runner.py status --write` 生成
   `benchmarks/{{verified,pending}}/<case>/validation_status.json`（判据逐项数值、
   收敛检查、repro 命令、工件/参照/矩阵的 sha256）与
   `benchmarks/validation_matrix/STATUS_MATRIX.md`。诚实负结果**不许被覆盖**：
   声明 `failed` 的算例即使机器判据通过也保持 failed。
4. **工具链**：PyTorch CPU；`tensorlbm.shallow_water`（HLL FV）、
   `tensorlbm.multiphase` + `solver.stream`（SCMP）；pytest 覆盖判据规则、
   矩阵 schema、锚点审计、状态一致性与一个端到端 CI 算例。

## ③ 计算结果（数字表）

九谱系矩阵状态（`STATUS_MATRIX.md` 生成）：**passed 5 / failed 4 / pending 1**。

溃坝（新 CI 档，含诊断阶梯）：

| nx | 前沿误差 [%] | 剖面 L2 [%] | 质量漂移 [%] | 步数 | 用时 [s] |
|---|---|---|---|---|---|
{db_rows}

静态液滴（受判两档）：

| 级别 | σ_eff | 逐半径最大偏差 [%] | R² | 截距占比 [%] | 两档 σ 偏差 [%] | 用时 [s] |
|---|---|---|---|---|---|---|
{drop_rows}

逐半径明细：

| 级别 | R_init | R_eq | Δp | σ_i | 偏差 [%] | 步数 | 收敛 |
|---|---|---|---|---|---|---|---|
{per_radius}

## ④ 结果图

附图 `dam_break_ritter_ci.png`（左：nx=200/400/800 水深剖面 vs Ritter 解析解；
右：前沿/剖面误差随 nx 的一阶收敛阶梯与 5% 判据门），由 `make_figures.py`
程序化生成并写 `figures/provenance.json`（读入工件的 sha256）。

**静态液滴面板缺图并说明理由**：该面板需要再跑一次液滴场（本机单次约 6–9 min，
额度内无法完成），本单以 ③ 的逐半径表 + Young–Laplace 拟合数值替代；`make_figures.py`
里对应面板代码已就绪（径向密度 + Δp vs 1/R_eq），下次门禁窗口可直接出图。
此处不含任何手绘/手抄数字。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

* **ci 档的分辨率会"合法地"判死算例**：溃坝在 nx=200/400 下剖面 L2 = 9.83%/6.29%，
  超出预注册 5% 门 → 矩阵如实记 **failed**；阶梯（800/1600：3.84%/2.32%）给出
  一阶收敛并定位 5% 门约需 nx≈700。**不改阈值让它变绿**。
* **静态液滴首建翻车**：R=8/12/16 档 max_dev 4.689%、R²=0.9961 → failed。
  根因是 R=8 仅 2 个界面宽度（W=4），低于 Laplace 定律可解半径 R≥3W。处置：
  受判档改为 R≥12 的两档，**失败的 R=8/12/16 结果保留为不设门的诊断阶梯**，
  并在 matrix note 与 README 中写明这是"半径集修正"而非"阈值放宽"。
* **圆柱算例口径敏感**：冻结工件用 Braza 带下界 0.164 作分母（D64 St=0.160 → −2.4%）；
  若用带中点 0.165 则同一数据为 −3.03%，刚好出 3% 门。矩阵按工件自身审计口径判
  passed，并把口径敏感性写进 notes——不是无条件背书。
* **两档不宜强求单调**：两档序列无法区分噪声与趋势，`require_monotone` 默认仅在
  ≥3 档时开启；suboff/cavity 等按其预注册单独声明。
* 局限：冻结重算档（SUBOFF 3.4h、球体数小时、方腔 159s/档）本单只重判不重跑，
  以 sha256 钉住；重跑由 `repro.cmd` 另行预算。

## ⑥ 误差判定（关键量 vs 基准，逐项显式回答是否 <3%）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| 溃坝前沿 x_f（nx=400） | Ritter x_f=2√(gh0)t | +2.562% | 是 |
| 溃坝湿区剖面 L2（nx=400） | Ritter 剖面 | 6.292% | **否（CI 档分辨率不足；nx=1600 时 2.315%）** |
| 溃坝质量闭合（nx=400） | 初始水量 | 2.2e-14% | 是 |
| 静态液滴 σ 逐半径一致性（R=12/16/20） | Young–Laplace 线性 | {drop['levels'][0]['metrics']['laplace_max_dev_pct']:.3f}% | {"是" if drop['levels'][0]['metrics']['laplace_max_dev_pct'] < 3 else "否"} |
| 静态液滴 σ 逐半径一致性（R=16/20/24） | Young–Laplace 线性 | {drop['levels'][1]['metrics']['laplace_max_dev_pct']:.3f}% | {"是" if drop['levels'][1]['metrics']['laplace_max_dev_pct'] < 3 else "否"} |
| 静态液滴两档 σ_eff 一致性 | 两档均值 | {drop['levels'][0]['metrics']['sigma_pair_dev_pct']:.3f}% | {"是" if drop['levels'][0]['metrics']['sigma_pair_dev_pct'] < 3 else "否"} |
| 方腔流中线偏差（128²/192²） | Ghia 1982 | 0.751% / 0.73% | 是 |
| 球体 Cd（D40/D60） | Schiller–Naumann 1.0917311 | +2.771% / +2.540% | 是 |
| SUBOFF Cd（L=80，受判档） | Blasius 0.041995 | −1.15% | 是 |
| Taylor-Green γ_E | 4νk² | −0.074% / −0.020% | 是 |

**诚实结论**：九个槽位中 5 个 passed、4 个 failed（湍流槽道、Rayleigh-Taylor、静态液滴 CI 档、溃坝 CI 档）、
1 个 pending（Michell 兴波 WITHHELD）。新算例的"未通过"是**分辨率与
模型参数面**的根因，已量化并写入状态文件，不是含糊带过。

### 生成的矩阵表（节选）

{status_md}
"""
    figures = []
    fig_db = _figure(
        "dam_break_ritter_ci.png",
        "溃坝：nx=200/400/800 水深剖面 vs Ritter 解析解（左）；前沿与剖面误差随 "
        "nx 的一阶收敛阶梯与 5% 判据门（右）",
    )
    if fig_db:
        figures.append(fig_db)
    fig_drop = _figure(
        "static_droplet_laplace_ci.png",
        "静态液滴：R=16 径向密度界面（左）；实测 dp vs 1/R_eq 与 Young-Laplace "
        "拟合线（右）",
    )
    if fig_drop:
        figures.append(fig_drop)
    return md, figures, json.dumps(frozen)[:2000]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    markdown, figures, _prov = build_markdown()
    payload = {
        "title": "TensorLBM 验证矩阵 v1：九谱系落矩阵 + 2 个 CI 算例全链（含诚实负结果）",
        "summary": (
            "九算例谱系注册表 + 零手抄锚点审计 + 统一判据（误差门 × 两档收敛）+ "
            "三级状态证据链；新增静态液滴/溃坝两个 CI 档算例，矩阵状态 5 passed / "
            "4 failed / 1 pending，失败均带量化根因。"
        ),
        "scenario_key": "lbm_validation_matrix",
        "mission_id": 15619,
        "tags": ["validation-matrix", "benchmark", "reproducibility",
                 "laplace-law", "ritter-dam-break", "lbm"],
        "markdown": markdown,
        "figures": figures,
    }
    out = Path("/tmp/report.json")
    out.write_text(json.dumps(payload))
    print(f"markdown {len(markdown)} chars, {len(figures)} figures -> {out}")
    if args.dry_run:
        return 0

    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    user = os.environ["TENSOR_AGENT_USER"]
    password = os.environ["TENSOR_AGENT_PASS"]
    token = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{base}/auth/login",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"username": user, "password": password})],
        capture_output=True, text=True, check=True,
    ).stdout
    try:
        tok = json.loads(token)
        token = tok.get("access_token") or tok.get("token") or token
    except json.JSONDecodeError:
        pass
    resp = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{base}/tensor/reports",
         "-H", f"Authorization: Bearer {token}",
         "-H", "Content-Type: application/json",
         "-d", f"@{out}"],
        capture_output=True, text=True,
    )
    print("POST /tensor/reports ->", resp.stdout[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())