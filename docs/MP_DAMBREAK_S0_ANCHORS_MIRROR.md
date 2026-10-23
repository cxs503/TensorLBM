# MP-DAMBREAK 文献锚点冻结（append-only）

> 线手册：`/nfs/wangxi/tensor_eco/agent_work/mp_dambreak/README.md`
> 纪律：零手抄——每个数值带出处坐标（论文表号/图号/数据文件路径+键）。
> append-only：修正只追加不覆盖，旧值标 RETRACTED 保留。
> 冻结日期：2026-10-23（m206 S0）
> 文件位置：`/nfs/wangxi/tensor_eco/agent_work/mp_dambreak/prereg/anchors.md`
> 镜像：本 worktree `docs/MP_DAMBREAK_S0_ANCHORS_MIRROR.md`（commit 内同步）

---

## 锚点 1：Martin & Moyce (1952) 2D 溃坝前缘位置 X(T)

### 1.1 文献信息

- **文献**：Martin, J.C. & Moyce, W.J. (1952). "An Experimental Study of the Collapse of Liquid Columns on a Rigid Horizontal Plane". Philosophical Transactions of the Royal Society of London. Series A, 244(882), 312–324.
- **DOI**：10.1098/rsta.1952.0006
- **实验内容**：矩形液柱（宽 a × 高 2a）在重力下坍塌，沿干涸水平底板扩展。无量纲前沿位置 X = x_front/a 对无量纲时间 T = t·√(2g/a)。

### 1.2 14 点前缘位置数字化表（Lethe 官方后处理脚本同源 = M&M 1952 物理实验数据）

**出处坐标**：
- 库内工件：`benchmarks/pending/dam_break_sc/result.json`（git commit `d6588f1e`，m163 终裁）键 `reference_digitisation_lethe`
- Lethe 官方脚本：`examples/multiphysics/dam-break/dam-break-2d.py` 硬编码 `x_exp`/`y_exp`，注释 "Experimental data from Martin & Moyce 1952"
- 出处钉死证据：`benchmarks/pending/dam_break_sc/REPORT_FS2D_route.md` §11.2（git commit `b7465583`）

> §11.2 原文（git show 直读）：
> "Lethe 官方后处理脚本 examples/multiphysics/dam-break/dam-break-2d.py 硬编码的 x_exp/y_exp 14 点与本战役 reference_digitisation_lethe 逐位相同，注释写明 'Experimental data from Martin & Moyce 1952'。即门禁要求匹配的是 1952 年物理实验的实测前沿，不是任何数值参考解。"

**14 点表**（从 `result.json` `reference_digitisation_lethe` 直读，逐值标注 JSON 键路径）：

| 序号 | T | X(=Z) | JSON 键 |
|------|------|-------|---------|
| 0 | 0.00 | 1.00 | `reference_digitisation_lethe.T[0]` / `.Z[0]` |
| 1 | 0.41 | 1.11 | `.T[1]` / `.Z[1]` |
| 2 | 0.84 | 1.23 | `.T[2]` / `.Z[2]` |
| 3 | 1.19 | 1.44 | `.T[3]` / `.Z[3]` |
| 4 | 1.43 | 1.67 | `.T[4]` / `.Z[4]` |
| 5 | 1.63 | 1.89 | `.T[5]` / `.Z[5]` |
| 6 | 1.82 | 2.11 | `.T[6]` / `.Z[6]` |
| 7 | 1.97 | 2.33 | `.T[7]` / `.Z[7]` |
| 8 | 2.20 | 2.56 | `.T[8]` / `.Z[8]` |
| 9 | 2.32 | 2.78 | `.T[9]` / `.Z[9]` |
| 10 | 2.50 | 3.00 | `.T[10]` / `.Z[10]` |
| 11 | 2.64 | 3.22 | `.T[11]` / `.Z[11]` |
| 12 | 2.82 | 3.44 | `.T[12]` / `.Z[12]` |
| 13 | 2.96 | 3.67 | `.T[13]` / `.Z[13]` |

> 注意：T_max = 2.96 是 Lethe 表最后一个点。T=3.0 需外推，被仓库"禁止外推"纪律排除。

### 1.3 检查点值（线性内插，禁止外推）

**出处坐标**：`result.json` 键 `reference_checkpoints`（直读）

| 检查点 | T | X_ref | JSON 键 | 内插来源 |
|--------|------|-------|---------|---------|
| 1 | 1.0 | **1.326** | `reference_checkpoints.T=1.0` | T[2]=0.84 → T[3]=1.19 |
| 2 | 2.0 | **2.360** | `reference_checkpoints.T=2.0` | T[7]=1.97 → T[8]=2.20 |
| 3 | 2.96 | **3.670** | `reference_checkpoints.T=2.96` | T[12]=2.82 → T[13]=2.96 |

**内插验证**（python3 计算，exit=0）：
```
T=1.0: interp between T[2]=0.84, T[3]=1.19 -> Z=1.326
T=2.0: interp between T[7]=1.97, T[8]=2.2  -> Z=2.360
T=2.96: interp between T[12]=2.82, T[13]=2.96 -> Z=3.670
```

**口径纪律**（`result.json` `time_convention`，直读）：
- 正确时间轴：`T_MM = t·√(2g/a)`，`X = x_front/a`（a = 初始柱宽）
- 旧错误时间轴：`T_code = t·√(g/a)`（缺 √2）——**已撤回**
- 关系：`T_MM = √2 · T_code`
- 副轴（仅报告）：`T_MM_geff = t·√(2g_eff/a)`，`g_eff = g·(1-ρ_v/ρ_l) = 0.9184g`

### 1.4 旧错误参考表（RETRACTED，保留存档）

**RETRACTED — 禁止使用**（`result.json` `convention_and_table_fix`，直读）：

| 旧表 | T=1 | T=2 | 错误原因 |
|------|-----|-----|---------|
| table_1 | 1.5 | 2.7 | 把 M&M 的 T=3 值 (2.7) 映到 T=2，T=1 值无出处 |
| table_2 | 1.1 | 1.8 | 粗糙数字化，仅在旧 t·√(g/a) 轴上有意义 |

> `convention_and_table_fix.double_error_retraction`（直读）：
> "the historical +50~82% errors were the product of a DOUBLE error: (i) the axis T=t*sqrt(g/a) was missing a factor sqrt(2); (ii) the reference table was wrong. Both are now fixed; the +50% claims are RETRACTED."

### 1.5 残柱高度 H 参考

**出处坐标**：`benchmarks/pending/dam_break_sc/README.md`（git commit `d658f1e`）§根因/诊断 第 1 条（直读）

- M&M 残柱高度参考范围：**H ~ 0.42–0.5**（H = h_residual/(2a)，初始 H=1.0）
- 模拟实测范围：H ~ 0.16–0.21（柱坍塌过快）
- 出处原文："模拟柱坍塌过快（残余高度 H~0.16-0.21 vs M&M ~0.42-0.5）"

> 注意：H 参考值目前只有范围而非精确检查点表。M&M 1952 原文的 H 数据未在库内工件中数字化为表格。S1 阶段若需 H 精确检查点，需从原文图/表提取。

---

## 锚点 2：Lobovský et al. (2014) 实验前缘带

### 2.1 文献信息

- **文献**：Lobovský, L., Botija-Vera, E., Souto-Iglesias, A., Elsaßer, B., Maza, F.C., Tafuni, A., Domínguez, J.M., Crespo, A.J.C., Gómez-Gesteira, M. (2014). "An experimental investigation of the dynamics of dam-break flow against wet and dry beds". Journal of Fluids and Structures, 49, 49–66.
- **DOI**：待核实
- **实验内容**：溃坝前缘位置随时间的实验测量，包含湿润/干涸底板对照。

### 2.2 数据提取状态

**ACCESS BLOCKED** — 无法从原文提取数值数据。

**尝试记录**：
1. `web_search("Lobovsky 2014 dam break experiment front position dimensionless results")` → no results
2. `web_search("Lobovsky Botia-Vera 2014 dam break SPH validation")` → no results
3. `web_search("Lobovsky 2014 'experimental investigation' dam break flow advancing front")` → no results
4. `web_search("Lobovsky 2014 'An experimental investigation of the dynamics of dam-break flow'")` → no results
5. `fetch_url("https://doi.org/...")` → domain not in allowlist
6. `fetch_url("https://github.com/search?q=Lobovsky+dam+break")` → timeout / 404

**库内/平台工件检索**：
- `grep -rn 'Lobovsky|lobovsky' src/ docs/ benchmarks/` → 无匹配
- `/nfs/wangxi/runs/bm_mp_20261001/` 下无 lobovsky 相关文件
- `/nfs/wangxi/scratch/dbs_eos/walberla/runs/` 下有 7 个 `breaking-dam.txt`（waLBerla 自身模拟输出，非 Lobovský 实验）

### 2.3 替代锚点：waLBerla 外部对照

**可用数据**（`/nfs/wangxi/scratch/dbs_eos/walberla/runs/`，直读）：

waLBerla commit `01a28162`（Schwarzmeier & Rüde 2022, AIP Advances 12, 035317）的 DamBreakRectangular showcase 输出，7 组运行：

| run | 配置 | 数据文件 |
|-----|------|---------|
| W40P_EQ | a=40, 论文默认(EQ) | `W40P_EQ/breaking-dam.txt` |
| W40P_AVG | a=40, AVG 重填 | `W40P_AVG/breaking-dam.txt` |
| W40P_NOCS | a=40, CS=0 | `W40P_NOCS/breaking-dam.txt` |
| W40C_EQ | a=40, Ga×4 | `W40C_EQ/breaking-dam.txt` |
| W80P_EQ | a=80, 论文默认 | `W80P_EQ/breaking-dam.txt` |
| W80P_AVG | a=80, AVG 重填 | `W80P_AVG/breaking-dam.txt` |
| W40_AV3REP | a=40, τ=0.70, Ga=2880 | （无独立目录） |

**数据格式**：4 列（step, T, Z_front, H_residual），T 轴 = `t·√(2g/a)`（与 M&M 同定义）。

> 注意：这些是**数值模拟输出**，不是物理实验锚点。它们的价值在于：(i) 作为 waLBerla 论文发表实现的复现对照；(ii) 证明论文原码在同一门禁下也不过（+5.9~+13.1 点）。不可作为"物理实验锚点"替代 M&M 1952。

### 2.4 Lobovský 锚点冻结结论

- **状态**：ACCESS BLOCKED — 平台网络环境无法获取原文 PDF 或数据。
- **S1 行动建议**：若 S1 阶段需要 Lobovský 实验带作为第二锚点，需（a）owner 提供原文 PDF 或数字化表入库，或（b）平台网络恢复后重新尝试 web_search/fetch_url。
- **当前可用锚点**：M&M 1952 14 点表（锚点 1）已冻结且充分——3 个检查点 T=1/2/2.96 → X=1.326/2.360/3.670，可独立支撑 ≤3% 判据 + ≥2 档单调收敛判据。

---

## 锚点 3：Körner et al. (2005) 自由面 LBM 理论框架

### 3.1 文献信息

- **文献**：Körner, C., Thies, M., Hofhauser, T., Rüde, U. (2005). "Lattice Boltzmann Method for Thermal Free Surface Flows: liquid droplet and bundle of tubes". International Journal of Modern Physics C, 16(8), 1271–1287.
- **DOI**：待核实（IJMPC 通常无 DOI，可通过 World Scientific 访问）
- **库内引用**：`src/tensorlbm/free_surface_lbm_2d.py` line 2 docstring（直读）："References: Körner et al. (2005), waLBerla DamBreakRectangular (2D)"

### 3.2 锚点用途

此文献为**路线 A（自由面 LBM）的理论框架锚**，不是数据锚。它定义了自由面 LBM 的：
- fill field 概念（fill = ρ/ρ_liquid）
- GAS/LIQUID/INTERFACE/SOLID 四态标记
- 质量交换 + 拓扑转换规则
- refilling 方案族

库内实现 `free_surface_lbm.py` / `free_surface_lbm_2d.py` 是 Körner 2005 路线的直接实现。

---

## 锚点 4：YS2006 伪势 EOS 参数（M4 路线）

### 4.1 文献信息

- **文献**：Yuan, P. & Schaefer, L. (2006). "Equations of state in a lattice Boltzmann model". Physics of Fluids, 18, 042101.
- **DOI**：10.1063/1.2195614

### 4.2 参数值（prereg_v2.md §2，直读）

**Carnahan-Starling EOS**：
- p = ρRT(1+η+η²-η³)/(1-η)³ - aρ², η = bρ/4
- YS2006 参数：a=1, b=4, R=1（α(T)=1 形式）
- 临界点：T_c=0.0943287031, ρ_c=0.1304438842, p_c=0.0044168134, Z_c=0.359
- 出处：`/nfs/wangxi/runs/bm_mp_20261001/m4_pseudopotential/phase0/out/cs_maxwell.json`

**Peng-Robinson EOS**：
- p = ρRT/(1-bρ) - aρ²/(1+2bρ-b²ρ²)
- YS2006 参数：a=2/49, b=2/21, R=1
- 临界点：T_c=0.0729190372, ρ_c=2.6573041565, p_c=0.0595645555, Z_c=0.307
- 出处：`/nfs/wangxi/runs/bm_mp_20261001/m4_pseudopotential/phase0/out/pr_maxwell.json`

**交叉验证**：CS 与 YS2006 发表值逐位一致（外部锚）。

### 4.3 锚点用途

此锚点为**路线 B（高密度比伪势）的参数锁**。M4 Phase-0 已验证：
- PR@T_r=0.55: 模拟密度比 400-557, Laplace σ 自测 PASS（残差 0.045-0.100%）
- 符号修正后 ψ_EOS 可达物理密度比（水/空气 ~800 需 T_r ≈ 0.4-0.5）

---

## 冻结声明

- 以上锚点 1（M&M 1952 前缘位置 14 点表 + 3 检查点）为 S1 阶段的**主判据锚点**。
- 锚点 2（Lobovský 2014）为 ACCESS BLOCKED，不阻塞 S1（M&M 单锚已充分支撑判据）。
- 锚点 3（Körner 2005）为路线 A 理论框架锚，非数据锚。
- 锚点 4（YS2006）为路线 B 参数锁，供高密度比伪势参考。
- 所有数值均带出处坐标（JSON 键路径 / commit / 文件路径），零手抄。
- append-only：后续修正只追加，旧值标 RETRACTED 保留。
