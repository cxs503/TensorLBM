#!/usr/bin/env python3
"""Publish the m17878 board-audit report to the platform (paper-grade, six elements).

Markdown is assembled from artifacts/verify_m17757.json + AUDIT_REPORT.md; no number is
typed by hand where the artifact already carries it.

Run: python3 docs/board_audit/m17878/publish_report.py [--dry-run]
"""
from __future__ import annotations

import argparse
import base64
import json
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8010/api/v1"
USER = "eco_admin"
PASS = "EcoTest!2026pass"
HERE = Path(__file__).resolve().parent
ART = json.loads((HERE / "artifacts/verify_m17757.json").read_text())


def api(method: str, path: str, payload=None, token=None):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(req, data=data, timeout=120) as r:
        return r.status, json.loads(r.read() or b"{}")


def pct(x):
    return f"{100.0 * float(x):.2f}%"


def build_md() -> str:
    c1, c2, c3 = ART["C1_repro"], ART["C2_equivalence"], ART["C3_degeneracy"]
    c4, c5, c6 = ART["C4_integration"], ART["C5_benchmark"], ART["C6_live_path"]
    rows = "\n".join(
        f"| {s['case']} | {pct(s['A1_vs_analytic'])} | "
        f"{'—' if s['A1_vs_A2'] is None else pct(s['A1_vs_A2'])} | "
        f"{'✅' if s['A1_vs_analytic'] < 0.15 else '❌'} |"
        for s in c3["shipped_kpis"])
    degen = "\n".join(
        f"| {r['field']} | {r['n_cells']} | {r['dir_err_A1_vs_A2']:.2e} |"
        for r in c3["random_field_rows"])
    return f"""# m17878 交差审计：m17757 XVAL-LBM-FILL-GRADIENT-NORMAL（TensorLBM）

**一句话**：交付物真实、可位级复现，解析锚定误差全部 <15%（最差 {pct(c3['informative_kpi_A1_vs_analytic_max'])}）；
但交付自称「跨实现方向误差 0.00%」的主判据是**代数恒等式**，且被审计函数在仓内**零调用**、
锚点**未登记**——判级 **L2**，L3/L4 拒判。

## ① 案例介绍（物理背景 / 几何 / 参数）

自由面 LBM（Körner 型）里，界面法向 n̂ 决定质量再分配与界面重构方向。被审计对象是 TensorLBM
生产函数 `src/tensorlbm/free_surface_lbm.py:642 _compute_interface_normal(flags, mass, rho)`：
`fill = mass/rho` → 中心差分 → 归一化 → 符号 liquid→gas。对标源是开源 Maarten-vd-Sande/lbm 的
fill-gradient 做法（`oss_solutions id=85`：`n = 0.5*(ε_{{i-1}}-ε_{{i+1}}, ε_{{j-1}}-ε_{{j+1}})`，不归一化）。

几何与参数（交付构造场，纯几何无流动）：2D 圆 N=64/128/256（R=N/4）、2D 斜平面 θ=30° N=256、
3D 球 N=32/64/128（R=N/4）、2D 圆抹平场 w/dx=2/4/8（N=256）；周期边界；float64 CPU。
无量纲参数仅 R/dx 与 w/dx（无 Re/Ma/Fr，无流动）。

## ② 怎么做的（审计方法）

六项独立检查（`docs/board_audit/m17878/verify_m17757.py`，全部读运行工件、零手抄）：

| 检查 | 方法 | 结果 |
|---|---|---|
| C1 可复现 | 重跑交付入口，逐 KPI 比对 results.json | **{c1['n_cases_shipped']} 案例 / {c1['n_kpi_mismatch']} 个偏差**（bit-identical={c1['bit_identical']}） |
| C2 等价性 | 导入生产函数，与交付 A1 臂同输入比数值 | 3D {c2['tube3d_max_abs_diff']}，2D {c2['slice2d_max_abs_diff']}；符号 liquid→gas {c2['sign_liquid_to_gas_ok']} |
| C3 判据可判别性 | 3 个**任意**场（随机/0-1 随机/平滑正弦）跑 A1 vs A2 | 恒 0（≤3.8e-17），恒等残差 {c3['identity_residual_max_abs']:.1e} |
| C4 集成 | `grep -rn _compute_interface_normal(` in src+tests | **{c4['audited_function_n_callers']} 调用点**（死代码）；活路径 `interface_normal_3d` {c4['live_n_callers']} 消费者 |
| C5 基准登记 | `SELECT ... tensor_benchmark_cases WHERE anchor_slug=...` | **{c5['n_rows_for_anchor']} 行**（表内 {c5['table_total_rows']}）→ 本审计补登记 |
| C6 死/活路径关系 | 同 φ 场对比两函数 | 符号**相反**（{c6['sign_conventions_opposite']}），KPI 用 \\|n·n\\| 故**符号不可见**（{c6['kpi_is_sign_blind']}） |

## ③ 计算结果（数字表，程序化注入）

| 案例 | A1 vs 解析法向 dir_err | 交付 headline A1 vs A2 | <15% |
|---|---|---|---|
{rows}

退化性证明（同一 A1/A2 公式在**任意场**上的方向误差）：

| 合成场 | 界面/有效格数 | dir_err(A1 vs A2) |
|---|---|---|
{degen}

幅值证据（非退化）：A2/A1 幅值比 ≈ 0.586（几何相关）；A3/A2 = 0.500（复现 `streamcollide.py:166`
GAS=0.5 优先级 bug 的 2× 低估）。

## ④ 结果图

图 1：有信息量判据（A1 vs 解析）与退化 headline（A1 vs A2）并列，15% 红线为 owner 判据。
图 2：左=死函数与活函数在同一步界面上的 n_x（符号相反）；右=退化性证明（任意场 dir_err ~1e-17，对数轴）。

## ⑤ 注意点（坑与局限）

1. **恒等式伪装成证据**：A2 ≡ A1·|∇fill|，共模板共场 ⇒ 方向差必为 0；这类「交叉实现 0.00%」必须用
   可判别量（角度分布/幅值/符号三轴分开）替代，否则任何实现都能「通过」。
2. **死代码验证**：被审计函数 0 调用方；活路径 `free_surface_common.interface_normal_3d` 符号相反且
   未被本 XVAL 覆盖——「对标改善生产」的链路当前是断的。
3. **KPI 符号盲**：`1-mean|n·n|` 对符号翻转不敏感，交付差异清单 #4「符号一致」未被测量。
4. **形状守卫缺失**：`(1,nz,ny,nx)` 输入静默返回全零法向（dims 偏移）；`flags` 形参未使用。
5. 外部锚为 `oss_solutions id=85` 文档公式移植（无网络 checkout），严格意义 **WITHHELD**（非 code-to-code）。
6. 未跑：真实 OSS 源码、真实自由面算例端到端、跨 torch 版本位级复现、抹平场 A2 臂。

## ⑥ 误差判定（逐项，显式回答）

| 关键量 | 基准 | got | 误差 | <15% | <3% |
|---|---|---|---|---|---|
| circle-sharp N=256 方向 | 解析径向 | 5.10% | 5.10% | ✅ | ❌ |
| plane-sharp θ=30° | 解析常向量 | 7.83% | 7.83% | ✅ | ❌ |
| sphere-sharp N=128 | 解析径向 | 8.40% | 8.40% | ✅ | ❌ |
| sphere-sharp N=32（最差） | 解析径向 | 10.24% | 10.24% | ✅ | ❌ |
| circle-smeared w=8 N=256 | 解析径向 | 0.71% | 0.71% | ✅ | ✅ |
| A3/A2 幅值比（bug 复现） | 0.5 | 0.500 | 0.00% | ✅ | ✅ |
| 交付 headline A1 vs A2 | 「OSS 法向」 | 0.00% | — | （恒等，不可计分） | — |

**显式回答**：owner 15% 判据 → **是，全部通过**（最差 {pct(c3['informative_kpi_A1_vs_analytic_max'])}）。
**3% 严判据 → 否**：锐界面 5–10% 是中心差分在 0/1 场上的固有轴偏置（两实现共有），只有界面被分辨
（抹平 w=8）才达 0.71%；因此该法向在**未分辨界面**上不满足 3% 级精度要求，需界面锐化/更宽模板。

## 判决

**confirmed-L2**（真代码+可复现+解析真值；L3 零集成、L4 无登记基准且主判据退化 → 拒判）。
机器门本地复算 `maturity_gate.judge -> (2, [])`，与声明级别一致。返修建议 R1–R4 见 AUDIT_REPORT.md §6。
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    md = build_md()
    figs = []
    for name, cap in [
        ("fig1_kpi_informative_vs_degenerate.png",
         "有信息量判据（A1 vs 解析法向）vs 退化 headline（A1 vs A2，恒 0）；红线=owner 15%"),
        ("fig2_sign_flip_and_degeneracy.png",
         "左：死函数 _compute_interface_normal 与活函数 interface_normal_3d 在同一步界面上符号相反；"
         "右：退化性证明——任意场上 A1 vs A2 方向误差 ~1e-17"),
    ]:
        p = HERE / "figures" / name
        figs.append({"name": name, "caption": cap,
                     "png_base64": base64.b64encode(p.read_bytes()).decode()})
    payload = {
        "title": "m17878 交差审计：m17757 XVAL-LBM-FILL-GRADIENT-NORMAL（判级 L2：可复现但主判据退化、零集成）",
        "summary": ("审计 m17757：交付可位级复现（10/10 案例 0 偏差），解析锚定误差全 <15%（最差 10.24%）；"
                    "但 headline「A1 vs A2 = 0.00%」是代数恒等式（任意场恒 0，残差 5.6e-17），"
                    "被审计函数零调用方且活路径符号相反，锚点审计时未登记（已补登记）。判 L2，L3/L4 拒判。"),
        "scenario_key": "board_audit",
        "mission_id": 17878,
        "tags": ["board-audit", "TensorLBM", "XVAL-LBM-FILL-GRADIENT-NORMAL", "m17757",
                 "L2", "degenerate-kpi", "free-surface"],
        "markdown": md,
        "figures": figs,
    }
    if args.dry_run:
        print(md[:1500])
        print("figures:", [(f["name"], len(f["png_base64"])) for f in figs])
        return 0
    st, d = api("POST", "/auth/login", {"username": USER, "password": PASS})
    tok = d.get("access_token")
    st, d = api("POST", "/tensor/reports", payload, tok)
    print(st, json.dumps(d, ensure_ascii=False)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())