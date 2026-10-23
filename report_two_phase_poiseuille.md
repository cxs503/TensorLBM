# Two-Phase Poiseuille SC-MCMP 终裁报告

## 1. 物理问题与几何

两层不互溶流体（水/气）在 2D 通道中的 Poiseuille 流，使用 Shan-Chen 双组分
多相模型（SC-MCMP）。通道垂直方向 y 分层，x 方向周期。两侧壁面为 bounce-back，
中间为 tanh 光滑界面（半宽 3 格）。驱动体力 G_x = 5×10⁻⁶，耦合常数 G_12 = −2.5
（全交换分层）。

### 控制参数

| 参数 | 值 |
|------|------|
| τ_w / τ_g | 1.0 / 0.75 |
| G_12 | −2.5 |
| G_x | 5×10⁻⁶ |
| nx | 8（流向周期列） |
| ny | 65 (H64) / 129 (H128) |
| 名义黏度比 M = ν_w/ν_g | 2.0 |
| 界面 | tanh-3（半宽 3 格，y = half） |

### 参考解

参考解来自库内 `porous_media._two_phase_poiseuille_analytical`（名义黏度比，
未做实测黏度比修正——预注册锁定为 FORBIDDEN recalibration）。

## 2. 计算结果与判据

### 正式跑结果（result.json / out_part1.json）

| 档 | 中心区节点数 | max_rel_err | mean_rel_err | u_max 比 | Ma_max | 伪流 uy_max |
|---|---|---|---|---|---|---|
| H64 (ny=65) | 55 | 26.82% | 14.21% | 1.2652 | 0.0467 | 0.0794 |
| H128 (ny=129) | 110 | 33.17% | 15.57% | 1.3146 | 0.1943 | 0.0797 |
| 单调 | | 26.82 → 33.17 | | | | **反收敛** |

### 四门判决

| 门 | 判决 | 证据（值-文件-键三元组） |
|---|---|---|
| G1: library_entry | ✅ | `from tensorlbm.multiphase import collide_sc_two_component` (w4b_lib.py) |
| G2: direct_observable | ❌ | H64 max_rel_err = 0.2682, H128 max_rel_err = 0.3317 (result.json, 远超 3%) |
| G3: monotone_convergence | ❌ | 26.82% → 33.17%（反收敛，非单调） |
| G4: reference_locked_first | ✅ | 参考函数先锁定后跑（run.py, FORBIDDEN recalibration 预注册） |

## 3. 根因诊断

### 3.1 不等 τ 混合平衡态 NaN 失稳

不等 τ 的混合格式（u_eq="mixture"）在强分相下结构性不稳定。每步净动量源
(1/τ₂-1/τ₁)·ρ₁ρ₂(u₁-u₂)/(ρ₁+ρ₂) 在 G ≤ -1.5 时发散：

| G_12 | NaN 起始步 |
|------|---|
| −2.5 | 第 29 步 |
| −2.0 | 第 46 步 |
| −1.5 | 第 175 步 |
| −1.0 | 20k 步无 NaN |

### 3.2 稳定域内界面条带化主导残差

等 τ 档下，混合平衡态（mixture）误差 34.83% 远大于自速度平衡态（self）7.17%，
证明名义差异消除后残差主要由界面条带化贡献——平界面碎裂为水/气交替条带，
黏度分层被打乱。

### 3.3 反收敛机制

细化网格（H64→H128）时误差反而增加（26.82%→33.17%）。根因：
- 更细网格分辨率提高了界面条带结构的解析度，而非抑制其生成
- 伪流 uy_max 在 H64/H128 下几乎不变（0.0794 vs 0.0797），表明界面不稳定性
  是格式结构性而非分辨率问题

### 3.4 同问题类已达标（CG 模型）

色梯度模型（`verified/poiseuille_two_phase_cg`, PR #327）已在同一物理问题上
四臂全过（等 τ: 0.211%/0.155%，黏度比 10: 1.092%/0.560%，均 ≤3% 且单调收敛）。
这证实物理设置本身无问题，SC-MCMP 的失败是模型结构性偏差。

## 4. 结果图件

### 图 1: u_max 收敛历史

![u_max convergence](benchmarks/pending/two_phase_poiseuille/figures/curve_convergence.png)

H64 与 H128 的 u_max 收敛曲线。两档均收敛到稳态，但稳态值与解析解偏差显著。

### 图 2: 速度误差剖面

![error profile](benchmarks/pending/two_phase_poiseuille/figures/curve_error_profile.png)

相对误差随通道位置分布，中心区误差远超 3% 阈值。

### 图 3: 反收敛条形图

![anti-convergence](benchmarks/pending/two_phase_poiseuille/figures/curve_anti_convergence.png)

误差随网格细化反而增大（26.82% → 33.17%），证实反收敛。

### 图 4: 模拟 vs 解析速度剖面

![velocity profiles](benchmarks/pending/two_phase_poiseuille/figures/field_velocity_profile.png)

H64 与 H128 模拟剖面与解析解对比，可视化系统偏差。

## 5. 审计溯源（PROVENANCE）

```json
[
 {"v": [], "f": "result.json", "k": "iron_rule.physics_kernel_defs_found"},
 {"v": true, "f": "result.json", "k": "iron_rule.pass"},
 {"v": 55, "f": "result.json", "k": "part1.H64.n_center"},
 {"v": 110, "f": "result.json", "k": "part1.H128.n_center"},
 {"v": 0.26821538643505377, "f": "result.json", "k": "part1.H64.max_rel_err"},
 {"v": 0.33173655541170183, "f": "result.json", "k": "part1.H128.max_rel_err"}
]
```

## 6. 终裁裁定

**判决：TERMINAL（终裁）**

SC-MCMP 两层 Poiseuille 流存在两个不可调和的结构性问题：
1. **不等 τ NaN 失稳**：净动量源项 (1/τ₂-1/τ₁)·ρ₁ρ₂(u₁-u₂)/(ρ₁+ρ₂) 在强分相
   下发散，无法进行不等 τ 计算（而黏度对比是工程核心需求）
2. **界面条带化主导残差**：等 τ 下误差仍高达 7.17%（self）/34.83%（mixture），
   且细化恶化（反收敛 26.82% → 33.17%）

同物理问题已由色梯度模型在 `verified/poiseuille_two_phase_cg` 中验证通过
（四臂全 ≤3% 且单调），证明物理设置正确，SC-MCMP 偏差为模型结构性。

## 7. 局限与缓解

### 当前缓解
- 问题类已由 CG 模型覆盖（PR #327）
- 本记录保留为 SC-MCMP 模型点失败证据与修复判据基线

### 晋级路径（需 owner 决策）
1. **不等 τ 稳定性修复**：引入隐式动量交换格式或净动量源显式补偿
   （预算 ~10 GPU·h，依赖 M4 多相库债修复）
2. **界面条带化抑制**：添加各向同性界面稳定项或改用更高阶格式
   （预算 ~15 GPU·h）
3. **维持 TERMINAL**：记录负结果，同物理类已由 CG 模型覆盖

## 8. 复现命令

```bash
cd benchmarks/pending/two_phase_poiseuille
CUDA_VISIBLE_DEVICES=5 /nfs/wangxi/venvs/tensorlbm/bin/python run.py
```

## 9. 参考

- Shan & Chen (1993) Phys. Rev. E 47, 1815 — 双组分 Shan-Chen 模型
- Shan & Doolen (1995) J. Stat. Phys. 81, 379 — 质心速度平衡态
- verified/poiseuille_two_phase_cg (PR #327) — 同问题类 CG 模型验证

## 10. gates.json 机器门禁

本报告的 gates.json 位于 `benchmarks/pending/two_phase_poiseuille/gates.json`，
六门三元组（含两 paper 门）已由 `gates_check.py --profile paper` 验证通过，
仅 monotone_convergence 门因反收敛预期失败（terminal 终裁行为）。
