# B26: 湍流通道 Re_τ=180 —— u+ 剖面验证

**状态：❌ 未达标（真实模拟，结果如实记录，无外推、无人工修正）**

## 问题定义

周期流向（x）、上下壁无滑移的平面通道，恒定体积力（压力梯度等价）驱动。
目标：稳态湍流后得到壁面律剖面 u+(y+)，与 DNS（Kim, Moin & Moser 1987 /
Moser, Kim & Mansour 1999，Re_τ=178.12）及对数律
u+ = (1/0.41)·ln(y+) + 5.0 在 **y+ ∈ [30, 100]**（线性-对数过渡区）对比，
**RMS 误差 ≤ 3%** 方为达标（benchmarks/problems.md B26）。

## 实现（真实模拟，无外推）

- **共性模块**：`src/tensorlbm/turbulent_channel.py` 的
  `TurbulentChannelConfig` + `run_turbulent_channel`（库原语，未修改）。
- **格子/碰撞**：D2Q9，BGK + Smagorinsky LES（`turbulence.collide_smagorinsky_bgk`，
  C_s=0.1）。
- **边界**：上下壁流后反弹（`boundaries.bounce_back_cells`，post-streaming BB），
  流向周期（solver `stream` 内建）；驱动为 Guo 型体积力
  `a_x = 2·u_τ²/H`（一阶形式，模块内 `_apply_body_force_2d`）。
- **参数**：Re_τ=180，u_τ=0.005（U_b ≈ 0.078，Ma ≈ 0.13，LBM 可接受上限），
  n_steps=100000，平均窗口 60000–100000，seed=0。
- **两档网格**（收敛性证据）：
  | 网格 | nx×ny | H | ν | τ | Δy+（每格） |
  |------|-------|-----|-----------|-----|------|
  | A | 256×64 | 62 | 8.61e-4 | 0.50258 | 5.81 |
  | B | 128×32 | 30 | 4.17e-4 | 0.50125 | 12.0 |
- **DNS 参考**：官方 Moser 组数据库
  （https://turbulence.ices.utexas.edu/ ，chan180/profiles/chan180.means，
  Re_τ=178.12，65 点），随附 `dns_ref/mkm180.csv`。
- **壁面位置标定**（层流标定实验，见下）：实测无滑移壁面位于
  y_w ≈ 0.10±0.03（不是模块隐含假设的 0.5），分析采用 y_w=0.1
  （同时报告 y_w=0.5 敏感性，不改变结论）。

## 结果（2026-08-18 实测）

| 网格 | 步数 | 稳态? | RMS vs DNS (y+30–100) | 平均相对误差 vs DNS | RMS vs 对数律 | 达标? |
|------|------|-------|----------------------|--------------------|---------------|-------|
| 128×32 | 100000 | 准稳态(max\|u\|≈0.1167 平台) | **4.59** | 27.7% | 3.88 | 否 |
| 256×64 | 100000 | 否（仍在爬坡，max\|u\|=0.080≈a_x·t） | **5.61** | 35.4% | 4.82 | 否 |

- 模块自身统计口径（30 < y+ < 0.8·Re_τ，B=5.2）：128×32 为 3.41、256×64 为 4.79 ——
  与我们的独立计算一致，均远超 3% 阈值。
- 网格加密**不收敛**：256×64 误差反而更大（其层流爬坡更慢，10 万步时离稳态更远）。
- 湍流统计（128×32，3 万步诊断 + 50 万步长程实验）：**vv ≈ 0、uv ≈ 0**，
  无壁法向脉动、无雷诺应力 —— 流场不是湍流，而是"层流 + Smagorinsky 涡粘"。

## 未达标原因（物理 + 模块设计，按重要性排序）

1. **2D 通道在该参数下不存在线性转捩路径（根本原因）**。
   2D 平面 Poiseuille 的线性稳定阈值 Re_c ≈ 5772（基于 U_max·h/ν）。
   本参数下终态 Re_max = u_max·Re_τ/u_τ ≈ 0.117·180/0.005 ≈ **4200 < 5772**，
   且 Re_max 与网格无关（= u_max·Re_τ/u_τ），因此任何网格都不会线性失稳；
   2D 无涡拉伸，Smagorinsky 只增耗散，流场停留在亚临界层流+涡粘终态
   （u_max≈0.117，u+ 中心 ≈23.4 vs DNS 19.2，剖面偏"满"）。
   → 2D LES 通道无法复现 3D 湍流的对数律剖面，这是模型维度的本质限制。
2. **从静止起跑的层流爬坡极慢，模块默认预算严重不足**。
   u_τ=0.005 由 Ma≈0.13 上限约束（U_b≈15.6·u_τ 不能更高）；
   层流动量扩散时间 H²/ν ≈ 2·H·Re_τ/u_τ ≈ 2.2e6 步（128×32）~ 4.5e6 步（256×64），
   而模块默认 n_steps=50000、averaging_start=20000 —— 平均窗口落在爬坡段
   （3 万步时 max|u| = a_x·t 精确等于活塞加速，壁面剪切层仅 ~3.5 格）。
3. **Smagorinsky 涡粘在层流中依然激活**，软化近壁剪切（层流标定显示
   有效壁面位置偏移到 y_w≈0.1、H_eff≈62.8 而非 62），进一步偏离解析层流，
   且无法区分"层流+涡粘"与真实湍流。
4. 模块的 Guo 力为一阶形式且施加在含壁面行的全场上，存在持续的质量漂移
   （10 万步 ~0.08–0.13%），说明壁面/力处理有系统性缺陷（见 /tmp/channel_gap.md）。

## 复现

```bash
cd /home/wxsc/cxs/TensorLBM/benchmarks/pending/channel_turb
PYTHONPATH=/home/wxsc/cxs/TensorLBM/src \
  /home/wxsc/anaconda3/envs/ftw-env/bin/python run.py scan \
    --out-root /tmp/channel_runs --grids 128x32 256x64 \
    --re-tau 180 --u-tau 0.005 --cs 0.1 --steps 100000 --avg-start 60000
python plot_profile.py        # -> uplus_profile.png
```

原始运行数据随附于 `runs/{128x32,256x64}/`（velocity_profile.csv +
run_metadata.json + result.json）。

## 参考

- Kim, Moin & Moser (1987), J. Fluid Mech. 177, 133–166（数值方法）。
- Moser, Kim & Mansour (1999), Phys. Fluids 11, 943–945（Re_τ=178.12 数据）。
- 数据来源：https://turbulence.ices.utexas.edu/ （官方数据库，chan180.means）。
- 2D 线性稳定性：Orr–Sommerfeld 分析，平面 Poiseuille Re_c ≈ 5772
  （标准教科书结论，如 Drazin & Reid 1981）。

## 判定

- 真实模拟（无外推、无人工修正）：**是**。
- y+∈[30,100] 误差 ≤3%（DNS 为主基准）：**否**（RMS 4.59–5.61，即 27–35% 相对误差）。
- ≥2 档网格收敛：**否**（两档均不达标，细网格离稳态更远）。
- 结论：**B26 未通过**，原因如上（2D 模型维度限制为根本原因；
  模块初始条件/步数预算/壁面力处理为次要因素）。改进方向见 /tmp/channel_gap.md。
