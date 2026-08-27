# Rayleigh-Taylor 不稳定性（VOF 自由表面共性模块首次验证）

**状态：未达标（pending/，2026-08-19）** — 如实记录，不保存 verified/。

## 物理问题
Rayleigh-Taylor（RT）不稳定性：重流体置于轻流体之上、重力指向重流体一侧（−y），
界面单模正弦扰动在失稳后以指数增长。线性理论（Chandrasekhar 1961；两无限深
无粘流体）：

    γ_theory = sqrt(At · g · k),   At = (ρ_h − ρ_l)/(ρ_h + ρ_l),   k = 2π/λ

本 benchmark 配置：3D 域 (nz,ny,nx)=(64,128,64)，重力 gy=−1e-4，重流体在上
（phi=1）/轻流体在下（phi=0），Atwood=0.9（ρ_h=1.0，ρ_l=1/19≈0.05263），
界面单模扰动 λ=64（=nx，一个波长）、振幅 a0=0.01λ=0.64，τ=0.8（ν=0.1），
封闭六面盒（bounce-back 壁）。

> **任务书文字勘误**：任务书写“下半重流体上半轻流体 + 重力 −y”——该分层在
> 重力 −y 下是**稳定**分层（无 RT）。`init_phi_rayleigh_taylor_3d` 的模块约定
> 是 phi=1 **重流体在上**（docstring：“Heavy fluid (phi=1) sits on top of light
> fluid (phi=0)”），即标准 RT 失稳分层。本 benchmark 采用模块约定（重上轻下 +
> 重力 −y = RT 失稳），README/result.json 均注明。

理论值：γ_theory = sqrt(0.9 × 1e-4 × 2π/64) = **2.973e-3**。
线性期速度量级：u ~ γ·a0 ≈ 1.9e-3。

## 共性模块路径（首次验证）
- **顶层导入验证通过**（commit da550e5 导出后首次使用）：
  `from tensorlbm import init_phi_rayleigh_taylor_3d, free_surface_vof_step` 可直接调用。
- 演化：`free_surface_vof_step(f, phi, tau, gy, rho_liquid, rho_gas, solid)`
  （VOF 碰撞+BGK+Guo 体力 → D3Q19 流播 → bounce-back → phi 迎风平流）。
- 初始场：`init_phi_rayleigh_taylor_3d(nz, ny, nx, interface_frac, amplitude, wavelength, device)`。
- 测量：phi=0.5 等值面高度亚格子插值 → sin(2πx/λ) 基模傅里叶投影振幅 a(t)；
  `mixing_layer_thickness_3d`；max|u|。
- 全部脚本见 `run.py`（无手写碰撞/流播/平衡态，纯共性模块调用）。

## 实测结果（真实模拟，GPU cuda:2，float32）
| 网格 | 步数 | a(t) 增长 | γ_sim | 结果 |
|---|---|---|---|---|
| 64×128×64 | 5000 | 无（~50 步界面即毁） | 不可测 | FAIL |
| 96×192×96 | 3000 | 无（同） | 不可测 | FAIL |

失败机制（量化证据）：
1. **锚定密度 → 平衡态压力跳变 → 固有伪流动**：碰撞平衡态用
   `rho_blend = ρ_l·φ + ρ_g·(1−φ)`，LBM 压力 p = ρ·cs² 随之在界面处跳变
   Δp = Δρ·cs²（At=0.9 时 Δρ=0.947 → Δp≈0.316），而模型**没有**自由压力场
   或表面张力机制与之平衡 → 界面被持续吹散。
2. **伪流动速度标度实测（gy=0 对照，无重力也发生）**：
   - Δρ=0.7（At≈0.54）：|u|_plateau ≈ **0.44**（峰值 0.88）
   - Δρ=0.1：≈ 0.09；Δρ=0.01：≈ 0.011；Δρ=0（无对比）：0.0（完美静止）
   - 标度 |u|_spurious ~ O(sqrt(Δρ·cs²/ρ̄))，与能量估算一致（Δρ=0.7 时
     sqrt(2Δp/ρ̄)=0.85，实测 0.44–0.88）。
3. **伪流动 vs RT 线性速度**：RT 线性期 u ~ γ·a0 ≈ 1.9e-3；伪流动 ~0.44
   （At=0.9 时 ~0.5+）→ **大 2 个数量级**，界面在第一个 e-folding（~230 步）之前
   即被撕碎（实测 ~50 步后 phi 场呈碎片/丝状结构，t=50 截面见图证据）。
4. **a(t) 无指数增长**：界面模式振幅始终 ≈ 0（±噪声），γ_sim 拟合不成立
   （R² 无意义），误差无法定义 → 判定未达标。
5. **表面张力不能修复**：σ·κ 只作用于弯曲界面，平界面 κ≈0，无法平衡 Δp；
   实测 σ=0.05/0.5 对伪流动无影响。初始场（平滑 2 格界面）与界面压缩
   （c_comp=0.5）均不改变结论。

## 对照：Boussinesq 变体与 Körner 路径（修复方向证据）
1. **Boussinesq 变体（rho_l=rho_g=1，密度对比只进重力）**：无伪流动（|u|≈0.0005），
   但碰撞把密度钉在 1 → **压力场同样被钉死**（无流体静压梯度 ∇p=ρg）→ RT 驱动
   机制缺失 → 界面中性稳定、a(t)≈0 不增长；c_comp=0.5 时压缩项后期自发制造
   丝状结构（数值噪声被反扩散放大）。⇒ 仅去掉密度对比不够，还需要自由压力。
2. **Körner 完整自由表面模型（free_surface_step，质量追踪+界面 ABB 气压）**：
   密度不被钉死（feq 用实际 ρ=Σf），可建立静压支持；但模型是**单液体+空洞气体**
   （气体 f 清零、无惯性），At 固定 = 1（非任务书 0.5–0.9），且 ABB 需
   rho_gas≈rho_liquid 才无压力爆炸（rho_gas=0 时初始即 |u|~0.5 爆炸）。
   初步测试界面可保持，RT 增长（At=1，γ=sqrt(g·k)）的定量拟合见 result.json 附录。

## 修复方向（按优先级）
1. **给 VOF 共性模块加质量追踪 + 自由压力场（Körner 式）**：密度从 phi 的
   平流/质量守恒演化（fill=mass/ρ），碰撞平衡态用实际密度 → 压力可建立
   静压梯度，RT 机制恢复；这是 free_surface_lbm.py 已验证的路线，把
   free_surface_step 的质量/ABB 机制移植进 free_surface_vof_step（或直接
   用 free_surface_step 做 RT，At=1 口径）。
2. **Boussinesq VOF + 自由压力**：密度统一进平衡态（ρ=1）但允许密度扰动
   承载压力（即**不要**把 rho_blend 塞进 feq，而是用实际 Σf 密度），密度对比
   只通过重力力 F=ρ(φ)·g 施加 → 无压力跳变、可静压支持，γ = sqrt(g·k·ρ_h/(ρ_h+ρ_l))
   （注意该口径与任务公式 sqrt(At·g·k) 的关系需按模型重新推导，At 高时两者
   差 ~2.7% @At=0.9）。
3. 若坚持现模块不改：需 σκ ≈ Δρ·cs² 的曲率项平衡压力跳变，但平界面 κ≈0
   无法平衡 → 不可行；降低 At（Δρ→0）则伪流动 ~Δρ 线性下降但 RT 信号
   ~sqrt(At) 也下降，且 At→0 不再是任务书目标区间。

## 判定
- γ_sim 不可测（界面被伪流动撕碎，无线性期）→ **未达标，不保存 verified/**。
- 达标判定标准（供修复后复测）：两档网格（64×128×64 / 96×192×96 或同级）
  γ_sim/γ_theory 误差 ≤3% 且 |err| 随网格加密单调下降、R²≥0.99、
  线性期窗口 ≥5 个 e-folding。
- 已建 `benchmarks/pending/rayleigh_taylor/`：run.py（复现脚本）+ result.json
  （实测数据）+ 本 README。

## 复现
    cd benchmarks/pending/rayleigh_taylor
    PYTHONPATH=/home/wxsc/cxs/TensorLBM/src \
      /home/wxsc/anaconda3/envs/ftw-env/bin/python run.py --device cuda:2 --grid 64 --steps 5000
    # 第二档：--grid 96 --steps 3000
