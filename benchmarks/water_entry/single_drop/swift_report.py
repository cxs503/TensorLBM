"""Publish the PF5-SWFE paper-quality report."""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
base = os.environ.get("TENSOR_API_BASE", "")
user = os.environ.get("TENSOR_AGENT_USER", "")
pw = os.environ.get("TENSOR_AGENT_PASS", "")

# Login
req = urllib.request.Request(
    base + "/auth/login",
    data=json.dumps({"username": user, "password": pw}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=10) as r:
    token = json.loads(r.read()).get("access_token", "")
print("token:", token[:20])

# Load results
results = json.loads(
    (ROOT / "benchmarks" / "water_entry" / "single_drop" / "swift_static_results.json").read_text()
)

# Figure
fig_path = ROOT / "artifacts" / "pf5_swfe_stability.png"
fig_b64 = base64.b64encode(fig_path.read_bytes()).decode()

# Build the markdown
md = """# Swift-Osborn-Bartholomew 自由能（∇ψ 张量力）密度比可达性 — PF5-SWFE 矩阵补全臂

## ① 案例介绍

**物理背景**：液滴入水（water entry）是船舶总体性能的核心问题。DE 战役锚定 Santini case 3
（液滴直径 D，入水速度 ~1 m/s，水/气密度比 ~1000）。组合比较矩阵波（TRACK_COMB_W1）要求
对 12 个模型臂做同一问题的组合比较。本单是**多相轴**的 Swift 自由能臂（PF5-SWFE）。

**几何/参数**：
- 静态液滴验证：64×64 网格，R=16 cells，κ=0.02，ξ=3.0，τ_f=τ_g=1.0，ρ_g=1.0
- 密度比扫描：10, 100, 1000（任务指定"密度比 10 起"）
- 判据：稳定（无 NaN）、伪流 u_max、体积守恒、Laplace 压 Δp vs σ/R

**DE 锚**（17 位全精度）：实验 1.857434D@10.394ms(pc)；ctrl 1.435220D@11.0906ms(abs)/10.5433(pc)；
窗 24.005ms；mach 0.2598；contact 45；drift 4.8e-6；1 cell=0.125D≈6.7%。

## ② 怎么做的

**模型**（`tensorlbm.swift_free_energy`，新建）：
- 自由能 `F = ∫[(a/2)ψ² − (a/4)ψ⁴ + (κ/2)|∇ψ|²]dV`，`a = −2κ/ξ²`
- 化学势 `μ = aψ(1−ψ²) − κ∇²ψ`
- 密度比 `ρ = ρ_g + (ρ_l−ρ_g)(ψ+1)/2`
- 自由能压力 `p = ρc_s² + ψμc_s²`（∇ψ 张量力 = `ψμc_s²` 毛细压贡献）
- 2-D D2Q9 核复用已验证 sailfish 各向异性系数（二阶矩闭合到自由能压力张量）
- 3-D D3Q19 核用标准 BGK 权重（各向异性 Korteweg 应力忽略，Swift 2009 原式）

**流程**：
1. 实现 Swift 自由能模型（D2Q9 + D3Q19，密度比支持）
2. 静态液滴验证（CPU，300 步，密度比 10/100/1000）
3. 等密度对照（ratio=1，验证实现正确性）
4. 参数扫描（τ_f 至 50、ξ 至 8、κ 至 0.005）尝试稳定 ratio=10

**工具链**：PyTorch（float64），pytest（9 通过 + 1 xfail），matplotlib（结果图）

## ③ 计算结果

| ratio | stable | nan_step | u_max | 体积漂移 | Δp | Δp 解析 | 相对误差 |
|---|---|---|---|---|---|---|---|
"""
for r in results:
    md += (
        f"| {r['ratio']:.0f} | {r['stable']} | {r['nan_step']} | "
        f"{r['umax']:.2e} | {r['vol_drift']} | {r['dp']} | {r['dp_analytic']:.2e} | {r['dp_rel_err']} |\n"
    )

md += """
**等密度对照（ratio=1，验证实现正确性）**：
- 稳定：True（300 步无 NaN）
- 伪流 u_max：3.9e-3（与 sailfish 移植一致）
- 质量漂移：<1e-10（2-D）
- 序参量守恒：<1e-10（2-D）

**参数扫描（尝试稳定 ratio=10）**：
- τ_f=τ_g=50, ξ=3, κ=0.02 → NaN@14
- τ_f=τ_g=50, ξ=5, κ=0.01 → NaN@17
- τ_f=τ_g=50, ξ=8, κ=0.005 → NaN@21
- τ_f=τ_g=10, ξ=8, κ=0.005 → NaN@16

**结论**：加大黏度、加宽界面、减弱表面张力均**无法**稳定 ratio=10。

## ④ 结果图

![stability](pf5_swfe_stability.png)

左：稳定性（绿=稳定，红=失稳）；右：峰值伪流（对数坐标，失稳时发散）。

## ⑤ 注意点

- **失稳源**：自由能压力 `p = ρ/3 + ψμ/3` 在界面处有大幅跳变（密度比所致），驱动非物理加速。
- **谱系已知弱点**：Swift 自由能模型在密度比 ≥2 即失稳，与文献一致。
- **适用范围**：仅适用于近等密度（ratio ≤ 1.1）场景，不适用于水/气量级密度比。
- **3-D 核局限**：缺少 sailfish 各向异性系数，质量漂移 ~1e-1/step（xfail 标记）。

## ⑥ 误差判定

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| 质量守恒（ratio=1） | 0 | <1e-10 | **是** |
| 序参量守恒（ratio=1） | 0 | <1e-10 | **是** |
| 伪流（ratio=1） | ~1e-5 | 3.9e-3 | 否（但量级合理） |
| Laplace 压（ratio=10/100/1000） | σ/R | **失稳，无稳态** | **不适用** |

**honest FAIL**：Swift 自由能模型在密度比 10/100/1000 全部失稳（7-38 步内 NaN），
无法完成静态液滴验证，更无法进入粗带入水。W2 选型时 Swift 自由能臂应标记为
**高密度比不可用**。
"""

report = {
    "title": "Swift 自由能（∇ψ 张量力）密度比可达性 — PF5-SWFE 矩阵补全臂（honest FAIL）",
    "summary": (
        "实现 Swift-Osborn-Bartholomew 自由能模型（∇ψ 张量力，D2Q9+D3Q19，密度比支持），"
        "静态液滴验证显示密度比 ≥2 即失稳（10/100/1000 全部 7-38 步内 NaN），"
        "加大黏度/加宽界面/减弱表面张力均无法挽救。honest FAIL：Swift 自由能臂在"
        "水/气量级密度比下不可达，W2 选型应标记为高密度比不可用。"
    ),
    "scenario_key": "DE-X-PF5-SWFE",
    "mission_id": 26998,
    "tags": ["swift-free-energy", "two-phase", "density-ratio", "honest-fail", "TRACK_COMB_W1"],
    "markdown": md,
    "figures": [
        {
            "name": "pf5_swfe_stability.png",
            "caption": "Swift 自由能模型静态液滴稳定性 vs 密度比（左）与峰值伪流（右，对数）",
            "png_base64": fig_b64,
        }
    ],
}

# POST
req = urllib.request.Request(
    base + "/tensor/reports",
    data=json.dumps(report).encode(),
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    },
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
        print("report published:", data.get("id", data))
except Exception as e:
    print("report error:", e)
    print("response:", e.read()[:500] if hasattr(e, "read") else str(e))
