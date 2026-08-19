# B1/B2: 球体 Re=100 阻力系数 — 力法对比（G12/G15 修复验证）

**状态：🔴 未达标（不入 verified/）** — G15 修复（bg_sub）实测无效，见下。

## 物理问题

均匀来流绕球（直径 D=1.0 m），Re=100。阻力系数参考（库内公式，判定基准）：
- **Schiller-Naumann: Cd = 24/Re·(1+0.15·Re^0.687) = 1.091731**
  （任务书引用 ~1.087 与此相差 0.4%，按仓库惯例用库内精确值 1.0917 判定）
- Clift-Gauvin: Cd = 24/Re·(1+0.1315·Re^(0.82-0.05·log10(Re))) = 1.109235

## 共性模块

**GeneralSimEngine**（src/tensorlbm/general_sim.py）：
- GeometryConfig: PARAMETRIC_SPHERE（radius=0.5, center=(0,0,0)）
- SolverConfig: D3Q19, collision=AUTO(→MRT, τ=0.56), resolution=D, wall=BOUNCE_BACK
- ForceMethod: BOTH（压力+摩擦积分 **与** 全部 MEM 变体同跑），extrap='none'，p0='far_field'
- SolverConfig.mem_variant: 'standard'|'galilean'|'bg_sub'|'all'（'all' 一次跑三种 MEM）
- MEM 变体实现：src/tensorlbm/momentum_exchange.py
  （`momentum_exchange_standard` / `momentum_exchange_galilean` /
  `momentum_exchange_background_subtracted`，G15 修复新增，均匀流 F=0 单元测试通过）

## 运行方式

```bash
cd /home/wxsc/cxs/TensorLBM
# D=40, 16000 步（收敛），mem_variant='all' 一次对比三种 MEM，GPU cuda:2
bash --noprofile --norc /tmp/run_g15_d40.sh
# 或直接:
PYTHONPATH=src /home/wxsc/anaconda3/envs/ftw-env/bin/python \
  /tmp/g15_sphere_re100_bgsub.py 40 16000 cuda:2 all far_field
```

## 结果（2026-08-19 G15 修复后首次验证，D=40，16000 步收敛实测）

收敛判定：5×1000 窗口均值漂移 <0.001（cd_mem_bgsub: 3.98142→3.98044），
非欠收敛伪值。时间平均取末 100 样本。

| 方法 | Cd | 误差 vs SN 1.0917 | 判定 |
|---|---|---|---|
| 压力+摩擦积分 cd_total（extrap='none'） | 0.9109 | **−16.56%** | G12：近壁压力无外推低估（已知） |
| MEM standard（Ladd 和形式） | 3.9804 | **+264.60%** | G15 曲面背景虚假力（复现） |
| MEM galilean（(1+1/2τ) 因子） | 7.5344 | **+590.13%** | 放大背景，更差 |
| **MEM bg_sub（G15 修复）** | **3.9804** | **+264.60%** | **与 standard 位相同 — 修复无效** |

压力积分分项：cd_pressure=0.5566（−49.0%）、cd_friction=0.3543（−67.6%）。

### bg_sub 为何无效（实测+数学，网格无关）

`momentum_exchange_background_subtracted` 从 Ladd 和中减去均匀自由流 f^eq 背景的
闭式解（逐方向：`2ρ₀w_q[1+4.5(c_q·U₀)²−1.5|U₀|²]`）。若每个方向的近壁 crossing
计数与其反方向相等（n_i = n_opp），背景项逐方向成对抵消 → **bg_sub ≡ standard**。

- 实测 D=40 与 D=60 球阶梯网格：**Σ|n_i−n_opp| = 0（18 个方向全部成对相等）**
  （D=40: 轴向 1245/对角 1434；D=60: 同构，obstacle_cells 33371→112931）
- 数字球掩膜几何对称 ⇒ crossing 计数对称不随分辨率改变 → bg_sub 对球是 no-op
- 均匀流 F=0 单元测试通过（单 solid cell 对称情形），但该性质在球面上退化为恒等
- **诊断规则**（任何背景减除类实现前必查）：`Σ|n_i−n_opp| = 0` ⇒ 减除是死代码

### 收敛性说明

16000 步时 MEM 各变体均已收敛（窗口漂移 <0.001）；4000 步欠收敛值
（standard/bg_sub=4.0017, galilean=7.5746）与收敛值（3.9804/7.5344）方向一致，
结论稳健：**MEM 三种变体在球面全部失败，与网格、步数无关**（B3 球 Re=200 D60
收敛 MEM standard 亦 +268.5%，同源）。

## 未达标原因与下一步

1. **bg_sub 修复被实测否决**（本记录）：方向对称网格上背景减除恒为 0。
   该模块对方向不对称的曲面（如带攻角翼、偏置体）仍有意义，但球不适用。
2. 剩余力法杠杆（均为**禁用于 verified/** 的方向）：
   - 压力外推（quadratic）：D=40 实测 cd_total=0.9293（−14.9%），仍不达标；
   - MEM 差形式（f_i−f_opp，Lorenz 2014）未实现，实现前先跑对称性检查。
3. **球绕流系列（B1/B2/B3）继续暂缓**（TODO.md 定案），优先其他问题类型。

## 判定标准（仓库规则）

- 真实模拟（extrap='none'），禁外推凑精度；误差 ≤3% 且 ≥2 档网格单调收敛才入库
- 本案例两档对称性均为 0 且 D=40 收敛误差 +264.6% ≫ 3% → **不入库，留 pending/**
