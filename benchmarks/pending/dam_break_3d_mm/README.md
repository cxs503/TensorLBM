# 3D 溃坝 — Martin & Moyce (1952) 验证（free-surface / VOF 共性模块）

**状态：❌ 未达标（2026-08-19）** — 共性模块质量守恒/界面稳定性缺陷，结果不可信，未入库 `benchmarks/verified/`。

## 物理问题

水柱 a×a×2a（如 32×32×64）在角部（x=0, z=0 墙），重力 -y，六面固壁，其余为气体。
无量纲：T = t·sqrt(g/a)，X = x_front/a，H = h_residual/(2a)（初始 1.0）。
参考（Martin & Moyce 1952）：T=1 → X≈1.5、T=2 → X≈2.7；T=1 → H≈0.8。

## 共性模块

`tensorlbm.free_surface_lbm`（D3Q19，da550e5 顶层导出）：
- `init_flags_from_fill` / `init_mass_from_fill` / `free_surface_step`（原版，未修改）
- 水柱角部布置需**自行补 z 向壁面**（`init_fill_rectangular` 的 solid 只有 x/y 面）

## 结果（真实模拟，无外推；原版模块，a=16，400 步，T_max≈1.6）

| rho_gas | 质量漂移 | interface 格（初→终） | X(T=1) 模拟 | X(T=1) 参考 | X(T=1.6) 模拟 | X(T=1.6) 参考 |
|---------|---------|---------------------|------------|------------|--------------|--------------|
| 0.1（10:1） | **-0.4%** | 500 → **55548**（111×） | **4.75** | 1.5 | **4.75** | ≈2.1 |
| 1.0（1:1）  | **+47.9%** | 500 → **51837**（104×） | **4.88** | 1.5 | **4.88** | ≈2.2 |

- X 虚高 3 倍以上（interface 爆炸污染波前测量）；T=2 检查点同样失效。
- 修复实验（去掉非保守的 interface↔interface 质量交换，A/B 验证）后：interface 稳定，
  但 rho_gas=0.1 时质量漂移 +85%（ABB 缺陷未独立解决）、rho_gas=1.0 时波前慢 15-44%
  （1:1 密度比物理不正确）→ 仍不达标。
- 网格收敛：无法评估（两档网格均被同一缺陷破坏）。

## 运行方式（复现）

```
cd /home/wxsc/cxs/TensorLBM
PYTHONPATH=src python benchmarks/pending/dam_break_3d_mm/run.py --a 16 --g 1e-4 --steps 400 --rho_gas 0.1
# 其他参数：--a 32 --steps 2000 --rho_gas 1.0（1:1 稳定窗口，但波前偏慢）
```

## 共性模块缺口（详见 /tmp/dambreak_gap.md）

1. **interface↔interface 质量交换非保守**（`mass_delta_interface = 0.5*(f-f_opp)` 单边加、
   无配对 debit）+ clamp 不对称截断 → 正质量源；A/B 验证去掉后 interface 稳定、漂移 ~1%。
2. **ABB 重建 population 参与质量交换** → 对 rho_gas 病态敏感（0.1 爆炸/1.0 漂移 +48%）。
3. **interface 计数爆炸**（to_iface 阈值 0.01 过低 + recv_new 播散）→ 即使质量守恒
   （rho_gas=0.1：drift -0.4%）波前也被雾状 interface 层污染（X 虚高 ~3 倍）。
4. `init_fill_rectangular` 无 z 向壁面（角部水柱需自行补墙）。

## 判定

- 波前误差 ≤3% 且 ≥2 档网格收敛：**不满足**（误差 >200%）。
- 未写入 `benchmarks/verified/dam_break/`；本目录保留可复现材料与真实数据。
- 共性模块修复建议见 /tmp/dambreak_gap.md（缺陷 A 修复已验证有效）。
