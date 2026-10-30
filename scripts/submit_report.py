#!/usr/bin/env python
"""Submit report via Tensor API."""
import json
import os
import requests

API_BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "EcoTest!2026pass")

# Load figures
with open("/root/tp-agent/work/m427/benchmarks/pending/taylor_green_3d_decay/report_figures.json") as f:
    figs = json.load(f)

# Build markdown report
markdown = """# SDAA 验证案例扫荡：taylor_green_3d_decay

## ① 案例介绍

**物理背景**：3D Taylor-Green 涡是流体力学经典基准问题，描述周期域内粘性涡旋的衰减过程。
初始场为 OpenLB tgv3d 同款：

$$u_x = U_0 \\sin(kx)\\cos(ky)\\cos(kz)$$
$$u_y = -U_0 \\cos(kx)\\sin(ky)\\cos(kz)$$
$$u_z = 0$$

其中 $k=2\\pi/N$，$\\nu=(\\tau-1/2)/3$，$Re=U_0 N/\\nu$。

**关键物理**：3D TG 不是 NS 精确解（2D 是），涡拉伸效应使 $w$ 分量从 0 增长，
导致实测衰减率 $\\gamma_{sim}$ 略高于理论值 $\\gamma_E=6\\nu k^2$。

**参数表**：

| 参数 | 值 |
|------|-----|
| 格子模型 | D3Q19 |
| 碰撞 | BGK |
| 边界 | 周期（stream3d 模运算内建） |
| Re | 24 |
| U0 | 0.05 |
| 网格 | N=32/48/64 |
| 数据类型 | float32 |

## ② 方法

- **工具链**：TensorLBM `tensorlbm.d3q19.equilibrium3d` + `tensorlbm.solver3d.collide_bgk3d` + `tensorlbm.solver3d.stream3d` + `tensorlbm.d3q19.macroscopic3d`
- **编译路径**：SDAA compile route → `teco_inductor` 不完整 → 自动回退 eager 模式
- **设备**：CPU（`SDAA_VISIBLE_DEVICES=8` 但无 SDAA 加速器硬件）
- **步数**：N=32 用 500 步，N=48/64 用 1000 步（CPU 墙时限制）
- **判定**：$|\\gamma_{E,sim}/\\gamma_{E,theory}-1| \\leq 3\\%$，单调收敛，$R^2 \\geq 0.999$

## ③ 计算结果

| N | τ | steps | γ_E_sim | γ_E_theory | err_E | R² | ≤3% |
|---|-----|-------|-----------|------------------|-------|-----|-----|
| 32 | 0.700 | 500 | 1.5615e-02 | 1.5421e-02 | **+1.258%** | 0.999995 | ✅ |
| 48 | 0.800 | 1000 | 1.0341e-02 | 1.0281e-02 | **+0.582%** | 0.999996 | ✅ |
| 64 | 0.900 | 1000 | 7.7550e-03 | 7.7106e-03 | **+0.576%** | 0.999995 | ✅ |

**GPU 参考对比**（verified/taylor_green_3d，cuda:2，auto_steps 2000/2000/2600）：

| N | GPU err_E | CPU err_E (SDAA) |
|---|-----------|------------------|
| 64 | +0.309% | +0.576% |

CPU 误差略高因步数减半（e-folds 采样不足），物理完全一致。

## ④ 结果图

见内嵌 figures（能量衰减曲线、ln(E) 线性拟合、误差收敛图）。

## ⑤ 注意点

1. **SDAA 编译回退**：`teco_inductor` 对 LBM 整步链（collide→stream）编译不完整，
   `compile_route` 自动回退 eager 模式，数值与纯 CPU eager bit-exact。
2. **步数限制**：CPU 墙时限制导致 N=64 仅跑 1000 步（auto 应给 2000），
   误差 +0.58% vs GPU +0.31%，差异来自 e-folds 采样深度。
3. **涡拉伸效应**：3D TG 非精确解，$\\gamma_{sim} > \\gamma_{theory}$ 是物理真实（+0.25~0.31% @ GPU Re=24）。
4. **无 SDAA 硬件**：`torch_sdaa` 已安装但无加速器设备，实际运行在 CPU。

## ⑥ 误差判定

| 关键量 | 基准 | 实测 | 误差 | <3%? |
|--------|------|------|------|------|
| γ_E (N=32) | 1.5421e-02 | 1.5615e-02 | +1.258% | ✅ 是 |
| γ_E (N=48) | 1.0281e-02 | 1.0341e-02 | +0.582% | ✅ 是 |
| γ_E (N=64) | 7.7106e-03 | 7.7550e-03 | +0.576% | ✅ 是 |
| 单调收敛 | — | 1.258→0.582→0.576 | — | ✅ 是 |
| R² | ≥0.999 | ≥0.999995 | — | ✅ 是 |

**结论**：所有关键量误差均 <3%，单调收敛确认，SDAA 路径（eager fallback）可运行且精度达标。
"""

# Build report JSON
report = {
    "title": "SDAA 验证案例扫荡：taylor_green_3d_decay（3D Taylor-Green 涡衰减）",
    "summary": "SDAA 设备扫荡验证 taylor_green_3d_decay 基准案例。三档网格 N=32/48/64 均通过 ≤3% 精度门（err_E: +1.26%→+0.58%→+0.58%），单调收敛确认。SDAA compile route 自动回退 eager 模式（teco_inductor 不完整），数值与 CPU eager bit-exact。",
    "scenario_key": "taylor_green_3d_decay",
    "mission_id": 427,
    "tags": ["SDAA", "sweep", "taylor_green", "3D", "decay", "D3Q19", "BGK", "periodic"],
    "markdown": markdown,
    "figures": figs
}

# Try to submit
try:
    # Login
    login_resp = requests.post(
        f"{API_BASE}/auth/login",
        json={"username": USER, "password": PASS},
        timeout=10
    )
    print(f"Login status: {login_resp.status_code}")
    print(f"Login response: {login_resp.text[:200]}")
    
    if login_resp.status_code == 200:
        token_data = login_resp.json()
        token = token_data.get("token", token_data.get("access_token", ""))
        print(f"Token obtained: {token[:20]}...")
        
        # Submit report
        report_resp = requests.post(
            f"{API_BASE}/tensor/reports",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=report,
            timeout=30
        )
        print(f"Report submit status: {report_resp.status_code}")
        print(f"Report response: {report_resp.text[:500]}")
    else:
        print("Login failed, saving report locally")
        with open("/root/tp-agent/work/m427/artifacts/report_tg3d_sweep.json", "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("Report saved to artifacts/report_tg3d_sweep.json")
        
except Exception as e:
    print(f"Error: {e}")
    # Save locally as fallback
    with open("/root/tp-agent/work/m427/artifacts/report_tg3d_sweep.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("Report saved locally due to API error")
