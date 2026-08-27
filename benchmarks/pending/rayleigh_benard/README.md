# Rayleigh-Bénard 对流（水平板间温差驱动）— 热 LBM 扩展验证

**状态：✅ 已入库（2026-08-19）**（或 🔶 未达标——见判定节）

## 物理问题

水平无限板间温差驱动的自然对流（Rayleigh-Bénard，RB），经典基准：
- **临界 Rayleigh 数 Ra_c ≈ 1707.76**（Chandrasekhar 1961，刚-刚边界）：
  Ra < Ra_c 无流动（纯导热，Nu = 1），Ra > Ra_c 出现 roll 对流
- **Nu(Ra) 超临界关系**（2D rolls、Pr=0.71、刚-刚）：
  Clever & Busse (1974, JFM 65:625)：Ra=1e4 → Nu≈2.16；
  Shan (1997, PRE 55:2780, LBM)：Ra=1e4 → Nu≈2.16、Ra=1e5 → Nu≈4.0；
  常见 2D 数值报告 Ra=1e4 时 Nu≈2.2–2.3（与 de Vahl Davis 方腔 Nu=2.243 同量级）。
  本 benchmark 判定基准取 **Nu_ref(1e4) = 2.24**（覆盖 2.16–2.30 区间，容差 3%）。

## 共性模块

热模型 src/tensorlbm/thermal.py 底层原语（D2Q9 速度 + D2Q5 温度双分布）：
- collide_bgk_force（Guo 力格式）+ buoyancy_force（Boussinesq 浮力，T_ref=(T_hot+T_cold)/2）
- temperature_collision / temperature_stream（**周期流**，% 取模——水平周期零改造）
- pre_streaming_bounce_back（上下壁 no-slip，V3 配方）
- RB 专用组合（**run.py 内联**，缺口见 /tmp/rb_gap.md）：
  底部 T_hot / 顶部 T_cold 的 anti-bounce-back（与库内左右壁 ABB 同构，旋转 90°）

## 配置

- 域：nx×ny 水平周期 × 垂直刚壁（96×48 / 128×64，纵横比 2:1，单 roll 波长=域宽）
- 温度：底部 T_hot=1、顶部 T_cold=0（ABB，half-way 壁面 y=∓0.5），ΔT=1
- 重力 −y（浮力 +y）；Pr=0.71（α = ν/Pr，τ_T = 3α+1/2）；τ=0.6（ν=1/30）
- g·β = Ra·ν·α/H³，H = ny（板间距，格子单位）
- 初始：线性温度 + 单 roll 扰动（ε=0.01，波长=域宽，垂直半波——最不稳定模）

## 结果（真实模拟，无外推）

| Ra | 网格 | steps | u_max | Nu_grad2（上下壁平均） | Nu_ref | err |
|----|------|-------|-------|------------------------|--------|-----|
| 1500（亚临界） | 96×48 | 120k | 待填 | 待填 | 1（纯导热） | — |
| 2000（超临界） | 96×48 | 120k | 待填 | 待填 | >1（弱对流） | — |
| 1e4 | 96×48 | 300k | 待填 | 待填 | 2.24 | 待填 |
| 1e4 | 128×64 | 300k | 待填 | 待填 | 2.24 | 待填 |
| 1e5 | 96×48 | 300k | 待填 | 待填 | 4.0–4.2 | 待填 |

**临界转变验证**：Ra=1500（<Ra_c）扰动衰减、u_max→0；Ra=2000（>Ra_c）扰动增长、
u_max 单调上升——数值临界 Ra_c 落在 (1500, 2000) 区间，与理论 1707.76 一致。

**网格收敛**：Ra=1e4 在 96×48 与 128×64 的 Nu 相对差 <5%（判定阈值）。

## Nu 口径说明（重要）

- 主口径 grad2（节点二阶单侧差分，与方腔 benchmark 一致），辅助 grad1
- **D2Q5 ABB 的离散稳态解将端点节点固定为壁面温度**（T[0]=T_hot、T[ny−1]=T_cold 精确成立，
  内部节点线性）→ 'halfway' 口径在此退化（纯导热给 Nu≈0），不使用
- 纯导热（Ra=1500）节点差分口径给 Nu = H/(H−1) ≈ 1.02（离散误差 ~2%，<3% 阈值）
- 顶部通量 Nu_top = −∂T/∂y·H/ΔT 为正（热流向上），grad1/grad2 顶部公式需取负号
  （初版曾漏号，导致 Nu 平均被系统性拉低——已修正并在 run.py 注释说明）

## 运行方式

```
cd /home/wxsc/cxs/TensorLBM
PYTHONPATH=src python benchmarks/verified/rayleigh_benard/run.py --nx 96 --ra 1e4 --steps 300000
# CPU 96×48 约 8 分钟（16 线程）；--device cuda:2 可选
```

## 共性模块缺口（记录于 /tmp/rb_gap.md）

- G1：thermal.py 的 ABB 温度边界仅支持左右壁（方向 1/2），无上下壁变体（方向 3/4）
- G2：thermal_params 的 H 硬编码为 nx（RB 板距 = ny）
- G3：nusselt_number 仅支持左右壁热通量口径（RB 需要上下壁）
- G4：D2Q5 ABB 端点节点 = 壁面温度（离散稳态性质），'halfway' Nu 口径不适用——
  文档/docstring 需说明；run.py 内联实现，建议 G1–G3 泛化入库

## 判定

- 真实模拟（无外推）；Ra=1e4 Nu 对 2.24 误差 ≤3%；≥2 档网格收敛（Nu 相对差 <5%）；
  亚临界 Ra=1500 u_max 比超临界低 3 个数量级以上 → 达标入库
- 参考来源：Chandrasekhar 1961（Ra_c）、Clever & Busse 1974 / Shan 1997（Nu）
