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

### G12. GeneralSimEngine 压力积分系统性低估（extrap='none' 时）
- **现象**：球 Re=100/200 用 extrap='none' 时 Cd 低估 13-25%（Re=200: Cd=0.607 vs 参考 0.806，-24.6%；D=60/D=80 网格加密不改善——系统性偏差非网格问题）
- **根因**：`drag_pressure_integration`（drag_pressure.py:755）extrap='none' 时直接取近壁单元（距壁 0.5-1 格）压力积分，无壁面外推 → 压力阻力系统性低估（Cd_p≈0.41 vs 文献 0.55 量级）；叠加 `SurfaceMesh.from_sphere` dA=1.0 单元面积近似
- **后果**：历史"达标"（0.98% quadratic）全依赖压力外推——已被 benchmark 规则禁用（真实模拟禁外推）→ 球绕流类 benchmark（B1/B2/B3）全部受此阻塞
- **修复方向**：压力积分用近壁二阶外推（p_wall ≈ p_near + dp/dn·Δ）或改用 Ladd MEM 力（obstacles.compute_obstacle_forces_3d，已验证 2.3%）；SurfaceMesh dA 用真实曲面面积
- **顺带 bug**：`_sample_forces` 中 cd_mem（ForceMethod.BOTH）错存为 (fx_p+fx_f)/dpS 而非 MEM 力/dpS

### G13. NACA α=10° Re=1000 流态网格依赖（airfoil）
- c=60 周期涡脱落（Cl=0.346 ±1.75%）vs c=90 近定常分离（Cl=0.394 +16%）——流态定性不同，时均 Cl 不收敛
- 前缘半径仅 ~0.66 格（次网格），分离点受格子离散影响；St_c≈0.86 疑似数值伪频
- GeneralSimEngine NACA 无攻角字段（G1 补充）；run_airfoil_benchmark 硬编码通道 BC

### G14. Blasius 前缘台阶效应（平板边界层）
- 半程反弹壁面 y=0.5 vs 对称面 y=0 差半格 → 前缘"台阶/钝头"，上游排挤 v≈0.015 → 沿板累积顺压（类 Falkner-Skan），剖面饱满、C_f 系统性 +48%
- 不随网格/域高收敛；建议：中置薄板（plate_y=50）+ Zou-He 出口（已验证零质量漂移稳态）

### G15. MEM 曲面球平衡背景未扣除（momentum_exchange.py）
- **现象**：球 Re=200 用 MOMENTUM_EXCHANGE 力法，16000 步收敛后 Cd=2.97（+268%）
  （4000 步 3.33 → 16000 步 2.97，缓慢下降但远高于参考 0.806）
- **分析**：`momentum_exchange_standard` 是 Ladd **和形式**（f_i[fluid]+f_opp[solid]），
  注释明说"flat walls 平衡项抵消"——但**曲面球壁平衡项不抵消**（各 link 权重不同），
  产生 O(ρU·A) 量级虚假力。Couette/Poiseuille（平坦壁）精确（<0.01%）但球面不适用
- **修复方向**：MEM 减平衡背景（F -= ρU·Σn̂·dA）或用**差形式**（f_i - f_opp，Lorenz 2014
  Galilean-invariant 版，文件内已有）——该变体对曲面更鲁棒
- **影响**：B1/B2/B3 球绕流 benchmark 受阻（压力积分 extrap='none' 低估 -13~25%，
  MEM 曲面背景 +268%）——两个力法在球面都需修复，**球绕流类暂缓**，优先其他问题

### G16. NACA/Blasius 判定补录
- NACA α=10° Re=1000：c=60 时 Cl=0.346（1.75% ✅）但 c=90 时 0.394（16% ❌）——
  分离/涡脱落过渡区流态网格依赖，两档不收敛 → 未达标（pending/naca0012）
- Blasius：剖面 L2 8%、C_f +48%——前缘台阶效应（壁面 y=0.5 vs 对称面 y=0 差半格）
  → 未达标（pending/blasius_flat_plate，中置薄板是候选修复）

### G17. AMR BFL 移动壁接口缺失（sphere_amr_common.py）——阻塞 B7/B8/B9
- **现象**：`bfl_sphere_advance()` 以 `wall_velocity=`/`wall_density=`/`return_force=True`
  调用核心 `bfl_d3q19.bouzidi_bounce_back_d3q19()`，但核心只有静止壁 4 参数版
  （f, f_prev, fluid_boundary_mask, q_field，单返回值）
- **波及**：amr_sphere_shell_validate/l2/l3/l4、amr_sphere_nested_shell_validate、
  amr_sphere_cellwise_validate 全部崩溃；仅 amr_sphere_drag_validate 已适配（静止壁）
- **修复**：sphere_amr_common.py 内补 D3Q19 移动壁全模板 BFL 副本（镜像已有
  _bouzidi_bounce_back_d3q27），bfl_sphere_advance 按 lattice 分派
- **备注**：AMR 机械层 177 测试通过但 shell runner 端到端崩溃（pytest 不覆盖）
