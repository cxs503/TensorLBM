# 共性模块功能缺口清单（benchmark 驱动发现，2026-08-18）

> 来源：各 benchmark 子 agent 真实模拟诊断。这些缺口是"功能缺失补充开发"的输入。

## P0 级（库 bug，影响多相流正确性）

### G1. SC 力符号约定（multiphase.py / multiphase3d.py）——非 bug，是库既定约定
- **现象**：`_sc_neighbor_weighted_sum` 用**后向 gather** ψ(x−c)（非标准 SC93 前向 ψ(x+c)），
  配合力注入 `u += τF/ρ`（加速度形式，非标准动量形式），构成**自洽的库约定**
- **实测验证**：库原样 + G=+5 → 液滴稳定（R_eq=15.44, Δp=0.00323, rho_in=1.96/out=0.16），
  Laplace benchmark（3 半径 σ 拟合 0.8% 2D/1.11% 3D）通过——**G=+5 是有效配置**
- **教训**：尝试改前向 gather 后 G=±5 均 NaN（破坏了库的自洽性），已回退。**不改库**
- **建议**：文档注明库约定（后向 gather + 加速度力注入 + G>0 为吸引）；benchmark 用 G=+5

### G2. FE 自由能 Gamma 硬编码（multiphase.py:662）
- **现象**：`free_energy_step` 中 Gamma 参数硬编码 0.5（应是 `gamma`），迁移率控制无效
- **后果**：序参量 φ 在 ~600 步溶解到 0，界面消失

## P1 级（模型参数/一致性缺陷）

### G3. psi_cavitation 默认 G=−5.5 无共存态（cavitation.py）
- EOS p=ρ/3+Gψ²/6 在 G=−5.5 时液枝永不恢复稳定（p(2.65)=−4.435），模块注释声称的 ρ_l=2.65 非麦克斯韦共存态
- 有效共存：G=−1.0 → ρ_v=0.0296/ρ_l=4.330/密度比 146

### G4. psi_exp G=−4.0 恰临界（multiphase 默认）
- Shan-Chen 1994 伪势在 G=−4.0 无旋结线（恰为临界），无法相分离
- 需 G=−5.0（密度比 35.5）或 G=−4.1（密度比 2）

### G5. CS/PR EOS 参数失效
- `psi_carnahan_starling`/`psi_peng_robinson`（a=0.5,b=4,RT=1/3）p_min 负到 ~−7.8e23，不可用

### G6. 离散共存密度 ≠ 连续 EOS 麦克斯韦值
- 离散实测（1.957/0.1596）≠ 连续 EOS（1.7505/0.0493）——初始化须用离散实测值

## P2 级（缺能力/硬编码）

### G7. GeneralSimEngine D3Q27 死枚举（general_sim.py）
- `LatticeModel.D3Q27` 枚举存在但 setup 无条件用 d3q19：equilibrium3d（L470）、collide_bgk3d/mrt3d（L1044）、macroscopic3d（L1173）
- 需：D3Q27 分支（equilibrium27/stream27/boundaries_d3q27/力）

### G8. 3D 主循环 D3Q19 硬编码
- `solver3d.stream3d` q=19 硬编码；`momentum_exchange_standard` range(1,19)；far_field_bc_3d D3Q19
- D3Q27 路径需走 d3q27.stream27 + boundaries_d3q27.far_field_bc_27 + compute_obstacle_forces_27

### G9. 多相模型无大密度比自由表面
- CG 色梯度对密度比>1 溃坝不适用（平界面压力失衡）；FE 双阱弱；SCMP 需 EOS 调参——三个模型均无稳定大密度比自由表面，溃坝 benchmark（B20）被阻塞

### G10. 湍流通道 2D 无转捩路径（turbulent_channel.py）
- 2D 平面 Poiseuille Re_c≈5772 > 模块 Re_max≈4200，无法失稳；需 3D D3Q19 或 2D 加扰动初始化
- 壁面位置隐含假设 y_w=0.5 但实测 0.10±0.03；一阶 Guo 力质量漂移

### G11. 声学无圆柱接线（acoustics.py）
- FWH 后处理需全场 rho_history（170GB 不可行）；无 2D Curle 共性基准；圆柱 St 偏高 6.5%（far-field+全格反弹格子离散）

## 修复优先级建议

1. **G1**（SC 符号翻转）——多相流基础，修后 Laplace 可用物理 G=−5
2. **G7/G8**（D3Q27 支持）——高精度碰撞 benchmark 需要
3. **G3/G4**（EOS 参数）——多相模型可用性
4. **G10**（通道 3D 化）——湍流 benchmark 需要
