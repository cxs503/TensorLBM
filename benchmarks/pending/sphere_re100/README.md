# B1/B2: 球体 Re=100 阻力系数 — 力法对比（G12/G15 修复验证）

**状态：🔴 未达标（不入 verified/）** — G15 修复（bg_sub）实测无效；faces 摩擦公式
（edd3e80）实测有效但不足达标（摩擦 +16%，压力 −49% 主导残余），见下。

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
2. **压力积分参数扫描定案（2026-08-19，p0_method × friction_formula，extrap='none'）**：
   4×4 全参数空间无达标组合（详见 `scan_p0_friction_summary.json`）：
   - **p0 维度完全退化**：球阶梯表面闭合（Σn̂dA≈0 实测），常数 p0 从
     Σ(p−p0)n̂dA 中精确消去——4 种 p0_method（near_wall/far_field/domain_avg/inlet）
     D40 与 D60 均逐位相同（0.5450 / 0.5575）。G12 的 far_field 方向数学上无效，
     不存在 p0 杠杆。
   - **friction 维度 D=40（5000 步时间平均）**：

     | friction | cd_p | cd_f | cd_total | err vs SN 1.0917 |
     |---|---|---|---|---|
     | standard | 0.5564 | 0.3549 | 0.9112 | −16.53% |
     | 2nd_order | 0.5564 | −0.0255 | 0.5309 | −51.38% |
     | **central** | 0.5564 | 0.5578 | **1.1142** | **+2.05%** |
     | lagrange | 0.5564 | 0.3464 | 0.9027 | −17.31% |

   - **central 的 D=40 \"达标\"是巧合抵消，加密即翻车（未达标铁证）**：
     D=60（8000 步，统计收敛，末窗漂移 0.08%）cd_total=1.1283 = **+3.35% 超线**，
     误差随网格加密增大（+2.05%→+3.35%）。机制：central τ=ν·u2 采样离壁 1.5 格
     的第二格速度且无 /1.5 归一，对近壁线性剖面**系统性高估壁剪 1.5×**（实测
     cd_f central/standard = 1.57@D40 / 1.55@D60，u2/u1≈3.1-3.15）；而压力项
     extrap='none' 低估（G12）不随网格改善（cd_p 0.5564→0.5498），摩擦项随网格
     继续增大（0.5578→0.5785）→ 抵消失衡 → 总量反超 3%。2nd_order (3u1−u2)
     对球 BL 过校正致 cd_f<0（u2/u1>3，unphysical）；lagrange ≈ standard。
   - **结论**：压力积分在 extrap='none' 下的全部参数杠杆已穷尽，无达标组合。
     唯一剩余杠杆 = 压力外推（quadratic，B2 方向），被禁外推规则限制。
3. 剩余力法杠杆（均为**禁用于 verified/** 的方向）：
   - 压力外推（quadratic）：D=40 实测 cd_total=0.9293（−14.9%），仍不达标；
   - MEM 差形式（f_i−f_opp，Lorenz 2014）未实现，实现前先跑对称性检查。
4. **faces 摩擦公式重测（2026-08-20，edd3e80 阶梯面积修复后）**：见下节。
   球系列（B1/B2/B3）继续暂缓（TODO.md 定案），优先其他问题类型。

## faces 摩擦公式重测（2026-08-20，edd3e80 入库后）

`formula='faces'`（逐壁面剪切，dA=1/面，含阶梯内角双面）已由 GeneralSimEngine 接线
（`_sample_forces` 传 `formula=sol.friction_formula, solid=self.solid`）。实测（时间平均，
extrap='none'，p0=far_field，cuda:1）：

| D | 公式 | cd_p | cd_f | cd_total | err vs SN 1.0917 | 收敛（5×1000 窗漂移） |
|---|---|---|---|---|---|---|
| 40 | standard | 0.5564 | 0.3549 | 0.9112 | −16.53% | 复现历史 |
| 40 | **faces** | 0.5564 | 0.4137 | **0.9700** | **−11.15%** | 末窗漂移 <0.1% |
| 60 | **faces** | 0.5498 | 0.4396 | **0.9894** | **−9.37%** | 1.010→0.989，末窗 0.08% |

- **faces 有效但只 +16% 而非预期 +40%**（球 D40 面数诊断）：阶梯壁面 7470 个
  （x=y=z=2490，faces/near=1.736），但流向 Cd_x 只由法线⊥x 的 y/z 面贡献剪切
  （Σ(nfy+nfz)=4980，4980/4302 near = **1.158** ≈ 实测提升 1.166）。x 面（前驻点/尾流
  区阶梯）对流向摩擦无贡献 → +40% 预期（2D 圆柱 dA_scale 外推）在 3D 球上不成立，
  faces 公式实现本身正确（面数与方向均经独立脚本核对）。
- **误差随加密收敛**（−11.15% → −9.37%，改善 1.8pp），方向正确——与 central 的
  D40 巧合达标、加密翻车（+2.05%→+3.35%）本质不同。但两档均 >3% 硬线，**不入库**。
- **残余误差主因仍是压力项 cd_p=−49%**（G12：extrap='none' 取近壁格心压力，漏驻点
  压力升），faces 只修摩擦部分（cd_f 从 −67.5% 改善到 −62.1% / −59.7%）。
- 判定：faces 摩擦修正确认有效并收敛，但球 Re=100 达标需压力项修复（BFL/贴壁压力
  重构为唯一未穷尽方向）；本案例保持 pending/。

## 判定标准（仓库规则）

- 真实模拟（extrap='none'），禁外推凑精度；误差 ≤3% 且 ≥2 档网格单调收敛才入库
- 本案例两档对称性均为 0 且 D=40 收敛误差 +264.6% ≫ 3% → **不入库，留 pending/**
