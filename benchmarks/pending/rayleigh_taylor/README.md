# Rayleigh-Taylor 不稳定性 rayleigh_taylor（线性增长率 γ）（pending）

**状态：⛔ 参数面不可行（SCMP 轨 T-D 判决，2026-09-19）——SCMP 伪势模型在可达 (g, λ) 参数面内不存在 RT 增长窗口：λ=280 档已扫到 vapor-spinodal 重力天花板的 91%（g=6.5e-6），界面模式增长率 σ_sim 仍为负；λ=140 档 5 个 g 臂窗口从未打开（全拟合窗斜率为负）。10/10 案例 σ_sim ≤ 0 或 NaN。本目录另含更早的 VOF 模块轨失败记录（2026-08-19，结构性缺陷，正文保留摘要）。**

## 物理问题

重流体置于轻流体之上 + 重力，界面单模扰动线性增长率对比理论：

- 理论参考（SCMP 轨，Wave-2 T-D）：等黏性两流体精确黏性色散关系
  （Chandrasekhar 5×5 行列式，sympy 化简，`rt_theory.py` 对原始行列式自校验）；
- 材料：SCMP 伪势 D2Q9（psi=exp(-ρ)，G_LIB=+4.1，τ=1），重/轻相密度
  1.0047322946806552 / 0.4556430880312282（密度比 2.205086220054312，At=0.367），
  ν=0.16666666666666666，界面宽 6.866132180338164 格，气相 spinodal 密度 0.548；
- VOF 轨（本目录 2026-08-19）：3D 64×128×64，At=0.9，γ_theory=sqrt(At·g·k)，
  用 `free_surface_vof_step` 共性模块。

## 计算结果现状

**SCMP 参数面扫描（staging `rt_instability/result.json`，2026-09-19，10 案例）**：

| 案例 | gy | window_open | σ_sim |
|------|---------|-------------|------|
| lam140_g6e6 | 6e-06 | false | NaN（14 拟合窗斜率全负） |
| lam140_g8e6 | 8e-06 | false | NaN |
| lam140_g8e6_a2 | 8e-06 | false | -0.0009663450176290631 |
| lam140_g1e5 | 1e-05 | false | NaN |
| lam140_g1p2e5 | 1.2e-05 | false | NaN |
| lam280_g3e6 | 3e-06 | true | -0.00021536746744910603 |
| lam280_g4e6 | 4e-06 | true | -0.000205150208666548 |
| lam280_g4e6_a2 | 4e-06 | true | -0.000206630426464502 |
| lam280_g5p5e6 | 5.5e-06 | true | -0.00019305515715374987 |
| lam280_g6p5e6 | 6.5e-06 | true | -0.00018699780823696162 |

可行性锚点（每档 g 的三重约束）：

| 量 | λ=140 | λ=280 |
|----|-------|-------|
| g_ceiling（气相 spinodal 天花板） | 1.4281160151573876e-05 | 7.140580075786938e-06 |
| g_cutoff（毛细下限） | 4.504690367304767e-06 | 1.1268927966536935e-06 |
| g_crit_with_restore（含实测蒸发钉扎） | NaN（lam140 唯一可测臂 = 2.5371043496415984e-05，**高于天花板**） | 6.936644426062614e-06 |

λ=280 最深臂 g=6.5e-6 已达天花板的 91%（6.5e-6/7.140580075786938e-06），仍衰减。

**VOF 模块轨（本目录，2026-08-19）**：两档网格（64×128×64、96×192×96）界面
在约 50 步内被伪流动撕碎，γ_sim 不可测，无 result.json 入库（旧 README 记录
了完整机制链，正文见"根因"）。旧 README 正文实测：伪流动速度比 RT 线性期
信号大约两个数量级（gy=0 对照同样发生，纯 Δρ·cs² 压力跳变驱动）。

## 不达标清单（严格标准：共性模块入口 + 直接观测量 ≤3% + ≥2 档网格单调收敛）

- 共性模块入口：两轨均 ✓ 纯库路径（SCMP 轨 = multiphase 伪势原语；
  VOF 轨 = `init_phi_rayleigh_taylor_3d` + `free_surface_vof_step`）。
- 直接观测量 ≤3%：**不可测**（SCMP 轨 σ_sim ≤0 无增长期；VOF 轨界面损毁）。
- ≥2 档网格单调收敛：不可测（无有效观测量）。
- 判决：T-D 参数面不可行（非数值误差问题，是模型可达域问题）。

## 根因/诊断

- **SCMP 轨（决定性）——蒸发钉扎 + 三重 g 约束夹死增长窗口**：
  1. RT 增长需 g > g_crit_with_restore；伪势界面的蒸发质量输运阻力
     （R=C_R·k² 型钉扎）把该临界推高。λ=140 唯一可测 C_R 的臂给出
     2.5371043496415984e-05，已**超过**同档天花板 1.4281160151573876e-05；
     λ=280 的 6.936644426062614e-06 与天花板 7.140580075786938e-06 只剩
     3% 余量，而 91% 深度处 σ_sim=-0.00018699780823696162 仍衰减
     （外推所需 g* 已越出天花板，staging README §7 有完整论证）。
  2. 上限物理：g 再大 → 轻相密度跌破 spinodal 0.548，SCMP 机制本身失效。
  3. σ_sim 处处 ≤0/NaN：10/10 案例无任何指数增长窗口。
  4. 辅证：材料 σ 两种测量口径差 -0.41020277042882486（41% 方法散布），
     表面张力项本身不确定；gy=0 对照与 controls 自由落体校准通过
     （步进器/重力实现无误，失败在伪势界面物理）。
- **VOF 轨（结构性）**：碰撞平衡态用锚定密度 rho_blend=ρ_l·φ+ρ_g·(1-φ)，
  界面处 p=ρcs² 跳变 Δρ·cs² 无自由压力场/表面张力平衡 → 固有伪流动
  （Δρ=0 时完美静止，Δρ 增大伪流动按 ~sqrt(Δρ) 增长）；表面张力 σ·κ
  对平界面 κ≈0 无效。Boussinesq 变体密度钉死 → 无静压梯度 → RT 驱动缺失；
  Körner 完整自由表面模型只能做 At=1（气体无惯性）。**两条路都到不了
  任务目标的 At∈[0.5,0.9] 定量增长率。**

## 晋级路径

- SCMP 参数面已论证穷尽（g 上限撞 spinodal、下限撞毛细截止、C_R 不可降），
  **参数调优不可复活**；复活须换引擎：
  1. 高密度比伪势（Carnahan-Starling/幂律 EOS，需重测共存曲线）；
  2. 自由表面 Körner 轨——依赖 Wave-10 M3 晋级引擎重设计（主动邻居/
     fill 阈值触发的前锋推进 + 质量交换权重，见
     `/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json`
     根因 #5 与已登记的重设计提案）落地后重开；
  3. 或等 M4（高密度比伪势轨）结论。
- 达标即归档 PR 移 `pending/rayleigh_taylor` → `verified/`（需先有可测的
  γ_sim 线性窗 + 两档网格收敛记录）。

## 复现

```
SCMP 轨（staging）: /nfs/wangxi/runs/bm_widen_20260919/rt_instability/
  result.json       # 10 案例完整判决（本 README 表格来源）
  README.md §7      # 参数面不可行完整论证
VOF 轨（本目录）:   run.py（2026-08-19 记录脚本；旧 README 见 git 历史）
```

<!-- PROVENANCE
[{"v": 1.0047322946806552, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "material.rho_heavy"},
 {"v": 0.4556430880312282, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "material.rho_light"},
 {"v": 2.205086220054312, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "material.density_ratio"},
 {"v": 6.866132180338164, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "material.interface_width_w"},
 {"v": 0.16666666666666666, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "material.nu"},
 {"v": 0.548, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "material.rho_vapor_spinodal"},
 {"v": -0.41020277042882486, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "material.sigma_eff_method_spread"},
 {"v": 0.3670385497355098, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "cases.lam140_g6e6.feasibility.At"},
 {"v": false, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "cases.lam140_g6e6.feasibility.window_open"},
 {"v": 1.4281160151573876e-05, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "cases.lam140_g6e6.feasibility.g_ceiling_vapor_spinodal"},
 {"v": 4.504690367304767e-06, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "cases.lam140_g6e6.feasibility.g_cutoff_capillary"},
 {"v": 2.5371043496415984e-05, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "cases.lam140_g8e6_a2.feasibility.g_crit_with_restore"},
 {"v": 6e-06, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "cases.lam140_g6e6.config.gy"},
 {"v": 8e-06, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "cases.lam140_g8e6.config.gy"},
 {"v": 1e-05, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "cases.lam140_g1e5.config.gy"},
 {"v": 1.2e-05, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "cases.lam140_g1p2e5.config.gy"},
 {"v": 3e-06, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "cases.lam280_g3e6.config.gy"},
 {"v": 4e-06, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "cases.lam280_g4e6.config.gy"},
 {"v": 5.5e-06, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "cases.lam280_g5p5e6.config.gy"},
 {"v": 6.5e-06, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "cases.lam280_g6p5e6.config.gy"},
 {"v": -0.0009663450176290631, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "cases.lam140_g8e6_a2.sigma_sim"},
 {"v": -0.00021536746744910603, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "cases.lam280_g3e6.sigma_sim"},
 {"v": -0.000205150208666548, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "cases.lam280_g4e6.sigma_sim"},
 {"v": -0.000206630426464502, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "cases.lam280_g4e6_a2.sigma_sim"},
 {"v": -0.00019305515715374987, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "cases.lam280_g5p5e6.sigma_sim"},
 {"v": -0.00018699780823696162, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "cases.lam280_g6p5e6.sigma_sim"},
 {"v": true, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "cases.lam280_g6p5e6.feasibility.window_open"},
 {"v": 7.140580075786938e-06, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "cases.lam280_g6p5e6.feasibility.g_ceiling_vapor_spinodal"},
 {"v": 1.1268927966536935e-06, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "cases.lam280_g6p5e6.feasibility.g_cutoff_capillary"},
 {"v": 6.936644426062614e-06, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "cases.lam280_g6p5e6.feasibility.g_crit_with_restore"},
 {"v": 4.1, "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json", "k": "method", "in_str": true}]
-->
