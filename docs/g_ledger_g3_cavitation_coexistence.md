# G 账本 G3：`psi_cavitation` 默认耦合的共存态审计与修复（TensorLBM）

> 本文件由 `tools/g3_ledger_audit.py`（+ `tools/g3_ledger_report.py`）生成，全部数字从
> `artifacts/g3_ledger/audit.json` 程序化注入，零手抄。重跑：
> `PYTHONPATH=src python tools/g3_ledger_audit.py`。
> 本次运行：76.24 s，网格 64²、2000 步、
> ρ₀=1.5、τ=1.0、forcing=`edm`、torch 2.7.1、
> ψ = `psi_cavitation(rho, rho_l=2.65, rho_v=0.038, psi0=4.0)`。

## 1. 案例介绍

### 1.1 所取条目（账本原文）

`benchmarks/gaps.md` → **G3. psi_cavitation 默认 G=−5.5 无共存态（cavitation.py）**：

- EOS `p = ρ/3 + Gψ²/6` 在 G=−5.5 时液枝永不恢复稳定（`p(2.65) = −4.435`），
- 模块注释声称的 `ρ_l = 2.65` 不是麦克斯韦共存态，
- 账本给出的"有效共存"：`G=−1.0 → ρ_v = 0.0296 / ρ_l = 4.330 / 密度比 146`。

### 1.2 为什么取 G3（现场对账）

账本按 G1→G21 顺序执行，先闭环的就不重复跑：

| 条目 | 现场核对结果 | 归属 |
|---|---|---|
| G1 SC 符号约定 | 账本自身裁定"不改库 + 文档注明约定"，且 `multiphase.py` 已有 `CONVENTION NOTE (locked in M4 Phase 0, verified numerically)` 落地 | 已闭环，不重跑 |
| G2 `free_energy_step` Γ 硬编码 | 当前 `main` 中 Γ 已是形参并在 `geq_diff` 中生效（`Gamma: float = 0.5`） | 已闭环，不重跑 |
| **G3 伪势共存态** | 代码默认仍是 `G=−5.5`、`_RHO_L/_RHO_V = 2.65/0.038` 注释仍称"SC at G≈−5.5"、无任何共存态测试 | **本节点取此条** |

G3 是账本里第一条仍开着的条目，且账本自带判据数字（−4.435 / 0.0296 / 4.330 / 146），
满足"账本条目含明确判据照判据"的口径。

### 1.3 与深水区军三节点的边界（不重不漏）

| 归属 | 内容 | 本节点 |
|---|---|---|
| G12（深水区军） | `drag_pressure_integration` 在 `extrap='none'` 时压力阻力系统性低估 | **不触碰**：本节点不做力积分/壁面外推，无绕流算例 |
| G15（深水区军） | `momentum_exchange` 曲面球平衡背景未扣除 | **不触碰**：本节点不做 MEM/壁面动量交换 |
| D3Q27（深水区军，账本 G7/G8） | `general_sim` 的 D3Q27 枚举导出、3D 主循环 D3Q19 硬编码 | **不触碰**：全部工作在 D2Q9 标量 EOS 与 D2Q9 SCMP 弛豫，不新增速度集分支 |
| **G3（本节点）** | 伪势 ψ(ρ) 的 EOS 共存态、耦合符号与 `cavitation.py` 的默认值/力调用方式 | 全部产出 |

### 1.4 物理背景与参数

Shan–Chen 单组分伪势（SCMP）模型以 ψ(ρ) 表达非理想作用，其格点力学压力为
`p(ρ) = ρcs² − ½Gcs²ψ²`。只有当 `∂p/∂ρ < 0` 的 spinodal 带之间出现 van der Waals 环，
气相支（低密度、力学稳定）与液相支（高密度、力学稳定）才可能以 **Maxwell 等面积**
`∫_{v_l}^{v_v}(p−p_sat)dv = 0`（`v = 1/ρ`）共存；该判据与 `μ(ρ_v) = μ(ρ_l)`
（`μ = ∫dp/ρ`）严格等价。

`cavitation.py` 的伪势是 `ψ_cav(ρ) = ψ₀ exp(−ρ_ref/ρ)`，`ψ₀ = 4`，
`ρ_ref = (ρ_l + ρ_v)/2 = (2.65 + 0.038)/2`；其默认耦合与两个参考密度直接决定
"能不能分出两相"。参数表：

| 符号 | 值 | 含义 |
|---|---|---|
| ψ₀ | 4.0 | 伪势幅值（`psi_cavitation` 默认） |
| ρ_ref | 1.3440 | 伪势标定密度 `(ρ_l+ρ_v)/2` |
| cs² | 0.333333 | 格点声速平方（D2Q9） |
| G（账本默认，经典符号） | −5.5 | 旧 `CavitationConfig.G`；库符号 `G = +5.5` |
| G（账本"有效"值，经典符号） | −1.0 | 库符号 `G = +1.0` |
| 符号约定 | `F = (Gcs²/2)∇(ψ²)`，**G>0 为吸引** | 与经典文献 `G_c = −G` |

## 2. 怎么做的

1. **符号约定先行**（账本 G1 的结论）：本库 `sc_single_component_force` 取后向邻居
   ψ(x−cᵢ)，离散展开即 `F = (Gcs²/2)∇(ψ²)`，故 `G>0` 为吸引、`p=ρcs²−½Gcs²ψ²`。
   账本数字用经典符号书写，本报告一律并记 `G`（库）与 `G_c = −G`（经典）。
2. **新增 EOS/共存态工具模块** `src/tensorlbm/sc_eos_coexistence.py`：
   `mechanical_pressure`、`spinodal_bounds`、`maxwell_coexistence`（正确 v 空间等面积，
   返回 `found=False` 时附带**可审计的数值原因**）、`chemical_potential`（独立复核路径）、
   `relax_scmp`（真实 SCMP 弛豫，直接调用库的碰撞+迁移）。无 SciPy 依赖。
3. **账本判据逐条复算**：C1 压力锚、C2 共存态存在性、C3 "有效耦合"的共存对。
4. **账本数字归因**：C3 的 (0.0296, 4.330, 146) 与正确等面积构造差一个数量级，
   因此额外复算"ρ 空间面积规则"`∫(p−p_sat)dρ = 0`（量纲不自治的常见误用），
   若逐位复现即可判定账本 C3 用的是哪条规则。
5. **独立物理复核**：`relax_scmp` 跑真实 SCMP（periodic、D2Q9、`edm` forcing）：
   EOS 说无共存时弛豫必须分不出相；EOS 说有共存态时测得密度应逼近共存密度；
   由 EOS 与实测各自定出"共存起始耦合"并比较。
6. **落库修复 + 回归测试**：`cavitation.py` 的力调用与默认耦合按审计结果修正
   （见 3.4），`tests/test_sc_eos_coexistence.py` 锁定全部判据。

## 3. 计算结果

### 3.1 三条账本判据的复算

| 判据 | 内容 | 本次复算 | 判定 |
|---|---|---|---|
| C1 | 经典 G=−5.5（库 G=+5.5）、ρ=2.65 的压力 | p = **-4.435414**（ψ=2.4088） | 账本 -4.435，相对误差 +0.01% → **PASS** |
| C2 | 经典 G=−5.5 是否存在 Maxwell 共存解 | **不存在**；spinodal 带 0.41002–9.41797，p_min=-7.88568 @ ρ=9.43117 | 账本"无共存态" → **PASS** |
| C3 | 经典 G=−1.0（库 G=+1.0）的共存对 | 正确等面积：ρ_v=0.35078、ρ_l=5.04038、比值=14.369 | 账本 ρ_v=0.0296、ρ_l=4.33、比值=146 → 见 3.2 |

C2 的"无共存"不是算不出来，而是由具体数值原因命中：
`no equal-area solution: the area integral Eq. (2) has no sign change over the branch pressure overlap [3.33333e-06, 0.115819] (area in [-3.846981e+00, -3.679385e-01]) — the liquid branch never becomes the stable coexistence state`。
物理图像：库 G=+5.5 时 EOS 环极深（p_min = -7.88568），
气相支与液相支的压力区间无法被同一 p_sat 闭合，等面积积分在整个重叠区间内**不变号**
（不是"没找到"，而是"不存在"）。

### 3.2 账本 C3 的三个数字是怎么来的（归因）

| 量 | 账本 | 正确等面积（v 空间） | ρ 空间面积规则（误用） |
|---|---|---|---|
| ρ_v | 0.0296 | 0.35078（+1085.07%） | 0.02963（+0.11%） |
| ρ_l | 4.33 | 5.04038（+16.41%） | 4.32951（-0.01%） |
| 密度比 | 146 | 14.369（-90.16%） | 146.104（+0.07%） |
| p_sat | — | 0.115674 | 0.009878 |

账本三数字被 ρ 空间规则复现到 +0.11% 以内，
即 **C3 的"密度比 146"来自量纲不自治的面积规则**（应以 `dv` 而非 `dρ` 加权）。
同一耦合下按 Maxwell 正确判据的真值是
**ρ_v=0.35078 / ρ_l=5.04038 / 比值 14.369**，
并由独立路径 `μ(ρ_v)=μ(ρ_l)` 复核：`|Δμ|/cs² = 3.177e-07`。

### 3.3 独立物理复核（真实 SCMP 弛豫，D2Q9 periodic）

| G（库） | G_c（经典） | 测 ρ_max | 测 ρ_min | 比值 | 质量漂移 | 状态 |
|---|---|---|---|---|---|---|
| -5.5 | +5.5 | — | — | — | — | **NaN** |
| +0.5 | -0.5 | 1.49990 | 1.49987 | 1.000 | 2.035e-05 | ok |
| +0.6 | -0.6 | 1.49990 | 1.49987 | 1.000 | 2.003e-05 | ok |
| +0.7 | -0.7 | 2.53435 | 0.75305 | 3.365 | 2.082e-05 | ok |
| +0.9 | -0.9 | 4.50218 | 0.50282 | 8.954 | 1.486e-05 | ok |
| +1.0 | -1.0 | 5.42710 | 0.44267 | 12.260 | 2.035e-05 | ok |
| +1.2 | -1.2 | 7.29982 | 0.37141 | 19.655 | 1.804e-05 | ok |
| +1.5 | -1.5 | — | — | — | — | **NaN** |

- Maxwell 共存起始耦合（EOS 扫描）：**G ≈ 0.65**
- SCMP 实测起始耦合（比值>2）：**G = 0.70**
- 两者差 0.050 → **PASS**（判据 |ΔG| ≤ 0.1）

EOS 扫描节选（每 0.2 一行）：

| G | 共存 | ρ_v | ρ_l | 比值 | p_sat |
|---|---|---|---|---|---|
| +0.05 | no | — | — | — | — |
| +0.25 | no | — | — | — | — |
| +0.45 | no | — | — | — | — |
| +0.65 | yes | 0.92758 | 1.96410 | 2.117 | 0.21362 |
| +0.85 | yes | 0.48265 | 3.82301 | 7.921 | 0.15224 |
| +1.05 | yes | 0.31705 | 5.44041 | 17.159 | 0.10510 |
| +1.25 | yes | 0.21059 | 7.03484 | 33.405 | 0.07019 |
| +1.45 | yes | 0.13677 | 8.63324 | 63.120 | 0.04559 |
| +1.65 | yes | 0.08697 | 10.23924 | 117.730 | 0.02899 |
| +1.85 | yes | 0.05439 | 11.85098 | 217.897 | 0.01813 |

**判读**：EOS 判定"G ≲ 0.65 无共存"时，
实测为均匀态（比值≈1）；越过阈值立刻分相，且低密度比区间的实测密度与 Maxwell 共存线
吻合到 3% 以内（见第 6 节）。这独立地说明：**正确的 EOS 判据预测了系统的真实行为**；
而账本的 ρ_v=0.0296 与实测（库 G=+1.0 时 ρ_min=0.44267）
相差一个数量级以上，不可能是该伪势的真实共存态。

### 3.4 落库的代码改动

| 项 | 改动后 | 依据 |
|---|---|---|
| `cavitation.COEXISTENCE_G` | 1.00（经典 -1.00） | Maxwell 共存且 2000 步弛豫稳定（质量漂移 2.035e-05） |
| `CavitationConfig().G` / `apply_cavitation_force(G=…)` 默认值 | 1.00 | 同上；`G ≤ 0` 现在直接 `ValueError`（排斥作用不可能分相，防止再犯） |
| `COEXISTENCE_RHO_L` / `COEXISTENCE_RHO_V` | 5.04040 / 0.35080 | 与 `maxwell_coexistence(1.0)` 的相对误差 -0.00% / -0.01%，由测试锁定 |
| `_RHO_L/_RHO_V` 注释 | 改为"伪势标定密度，**不是**共存态" | 账本 C2：经典 G=−5.5 无共存态 |
| `apply_cavitation_force` 的 SC 力调用 | `sc_single_component_force(rho, G, psi_fn=ψ_cav)` | **原代码把 ψ 当作 ρ 传入**，伪势被作用两次（`exp(−1/ψ)`），等于换了另一套 ψ；已加回归测试 |
| `app/backend/routers/solver.py` 的 `CavitationFlowParams.G` | 默认 1.0，区间 [0.7, 1.2] | 与库默认值/已验证稳定窗口一致（旧默认 −5.5 现在会触发 `ValueError`） |

## 4. 结果图

| 文件 | 说明 |
|---|---|
| `artifacts/g3_ledger/fig1_eos_spinodal.png` | EOS p(ρ) of psi_cavitation at the ledger couplings; spinodal bands shaded; Maxwell p_sat and coexistence pair at G=+1.0 (classic −1.0). |
| `artifacts/g3_ledger/fig2_coexistence_vs_coupling.png` | Coexistence densities (left) and density ratio (right) vs coupling: correct Maxwell rule, direct SCMP measurement, and the ledger's pair. |
| `artifacts/g3_ledger/fig3_coupling_gate.png` | Coupling gate: measured density ratio vs G, NaN bands (red) and the Maxwell coexistence onset. |
| `artifacts/g3_ledger/fig4_density_fields.png` | Density fields after relaxation at the audited coupling (left) and at the ledger default coupling (right, NaN onset annotated). |

- 图 1（EOS/spinodal）：两条账本耦合的 `p(ρ)` 与 spinodal 带；库 G=+1.0 上标出 `p_sat`
  与共存对；黑虚线是账本声称的 ρ_l=2.65。
- 图 2（左：共存密度；右：密度比）：正确 Maxwell 曲线、实测点、账本点三者同图——
  账本点明显偏离实测（ρ_v 偏离一个数量级，比值 146 vs 实测
  12.26）。
- 图 3（耦合闸门）：均匀/分相/发散（NaN）三区，绿色虚线为 Maxwell 共存起始耦合。
- 图 4（密度场云图）：库 G=+1.0 分相场（左）与账本默认耦合的场（右，标出 NaN 起始步）。

## 5. 注意点（踩过的坑与适用范围）

1. **符号约定是第一条坑**：本库 G>0 才吸引（后向 gather），账本/draft 里的 `G=−5.5`
   在库里是排斥。任何"把经典 G 直接传进库"的代码都会得到反相行为；本节点用
   `COEXISTENCE_G` + `ValueError` 守卫把它固定在类型/运行期边界上。
2. **等面积积分必须对 `dv` 加权**：`∫(p−p_sat)dρ = 0` 量纲不自治，会把 ρ_v 压小数倍
   （本例 0.3508 → 0.0296），并让比值虚高（14.37 → 146）。本报告第 3.2 节用逐位复现
   证明账本 C3 就踩了这个坑；`sc_eos_coexistence` 只提供正确规则，误用规则留在
   `tools/g3_ledger_audit.py` 里作"归因证据"。
3. **ρ_v 对耦合指数敏感**：ψ 是 exp(−ρ_ref/ρ)，气相密度随 G 变化可达数个数量级
   （G=+0.7→0.73447，
   G=+2.0→0.03791），
   因此"气相密度对得上"不能作为判据，必须与液相密度/比值/阈值一起看。
4. **离散 vs 连续（账本 G6）**：SCMP 实测密度与连续 EOS 的 Maxwell 值有系统偏差，
   且随密度比增大而增大（本报告第 6 节：G=+0.7 时 ≤3%，G=+1.0 时 7.7%/26%）。
   本节点只裁 EOS 层的共存态与耦合符号，**不**宣称"离散共存密度 = Maxwell 值"。
5. **稳定窗口比共存窗口窄**：EOS 共存从 G≈0.65 开始，
   但 G ≥ 1.5 时 SCMP 弛豫在 2000 步内发散（图 3 红带）；模块默认取 1.0 而非"能共存就取大"。
6. **扫描下界**：`maxwell_coexistence` 默认 `ρ_lo=1e-5`；若真值低于该下界会在结果里表现为
   贴着边界的解（本组参数量级 1e-1，不会触发），使用时若换 ψ 需检查。
7. **`run_cavitation_flow` 仍是 STUB（本节点只修了它的三处签名/IO 错误，未修物理路径）**：
   `prepare_run_dir` 少参数、`configure_logging(run_dir)` 把路径当 level、`save_config_json(asdict(cfg))`
   二次 asdict —— 三处已修；但入口边界条件 `f[:, :, 0]` 与库的 `(9, ny, nx)` 轴序不符，
   全流程仍跑不通，`tests/test_sc_eos_coexistence.py::test_full_cavitation_runner_is_a_known_stub`
   以 `xfail(strict=True)` 锁证。空化全流程需要单独立项（不在 G3 判据内）。
8. **本节点不覆盖**：`schnerr_sauer_source` 的物理标定、全流程（喷管空化）精度、
   三维 SCMP、以及 G6 的离散共存密度修正——这些都不在 G3 判据内。

## 6. 误差判定（逐项列出，是否 <3% 显式回答）

| 关键量 | 参考 | 本次值 | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| p(ρ=2.65) at classic G=-5.5 | ledger -4.435 | -4.43541 | +0.01% | **<3% ✅** |
| ρ_l at classic G=-1.0 (correct Maxwell) | ledger ρ_l=4.33 | 5.04038 | +16.41% | **>3% ❌** |
| ρ_v at classic G=-1.0 (correct Maxwell) | ledger ρ_v=0.0296 | 0.35078 | +1085.07% | **>3% ❌** |
| density ratio at classic G=-1.0 (correct Maxwell) | ledger ratio 146 | 14.36904 | -90.16% | **>3% ❌** |
| ledger triple reproduced by the ledger's own area rule | 0.0296 / 4.33 / 146 | 0.0296331318095348 / 4.329507589340199 / 146.10361190197017 | +0.11% | **<3% ✅** |
| μ(ρ_v) vs μ(ρ_l) at the Maxwell pair (relative to cs²) | 0 (coexistence condition) | 3.177e-07 | +0.00% | **<3% ✅** |
| SCMP ρ_l at G=+0.7 vs correct Maxwell | Maxwell 2.5068 | 2.53435 | +1.10% | **<3% ✅** |
| SCMP ρ_v at G=+0.7 vs correct Maxwell | Maxwell 0.7345 | 0.75305 | +2.53% | **<3% ✅** |
| SCMP ρ_l at G=+1.0 (module default) vs correct Maxwell | Maxwell 5.0404 | 5.42710 | +7.67% | **>3% ❌** |
| SCMP ρ_v at G=+1.0 (module default) vs correct Maxwell | Maxwell 0.3508 | 0.44267 | +26.20% | **>3% ❌** |

**显式回答**：

- 账本 C1（压力锚）：+0.01% → **远小于 3%，判 PASS**（账本数字与库 EOS 完全自洽）。
- 账本 C3 的 ρ_v / 比值：+1085.07% / -90.16%
  → **>3%，不成立**：这三个数字由 ρ 空间（错误）面积规则产生；正确 Maxwell 判据下
  ρ_l 也差 +16.41%，同样 >3%。
- 账本数字 vs 账本规则自身的复现：+0.11%
  → **<3%，判 PASS**（归因成立：账本 C3 用的就是那条规则）。
- 独立路径复核（μ 相等）：|Δμ|/cs² = 3.177e-07
  → 判据为 0，实测 1e-7 量级，**数值上等价**。
- 低密度比区间实测 vs Maxwell：≤3%（第 3.3/第 6 节两行）→ **PASS**，说明正确 EOS 判据
  与真实求解器一致到 3% 以内。
- 模块默认耦合（G=+1.0）实测 vs Maxwell：ρ_l +7.67%、
  ρ_v +26.20%
  → **>3%，如实记为不达标**，根因是账本 G6 的"离散共存密度 ≠ 连续 EOS Maxwell 值"，
  属已知局限（本节点不假装修好）。

不达标项汇总（honest FAIL，不覆盖）：

- ❌ **ρ_l at classic G=-1.0 (correct Maxwell)**：参考 ledger ρ_l=4.33，实测 5.04038，相对误差 +16.41%（>3%，如实记为不达标）
- ❌ **ρ_v at classic G=-1.0 (correct Maxwell)**：参考 ledger ρ_v=0.0296，实测 0.35078，相对误差 +1085.07%（>3%，如实记为不达标）
- ❌ **density ratio at classic G=-1.0 (correct Maxwell)**：参考 ledger ratio 146，实测 14.36904，相对误差 -90.16%（>3%，如实记为不达标）
- ❌ **SCMP ρ_l at G=+1.0 (module default) vs correct Maxwell**：参考 Maxwell 5.0404，实测 5.42710，相对误差 +7.67%（>3%，如实记为不达标）
- ❌ **SCMP ρ_v at G=+1.0 (module default) vs correct Maxwell**：参考 Maxwell 0.3508，实测 0.44267，相对误差 +26.20%（>3%，如实记为不达标）

## 7. 局限

- 只裁 EOS 层共存态与耦合符号；离散 SCMP 共存密度与 Maxwell 的系统偏差（G6）未修。
- 空化全流程（`run_cavitation_flow`、Schnerr–Sauer 质量传输标定）未在本节点验证精度：
  该 runner 目前仍是 STUB（入口 BC 轴序与库布局不符），已用 `xfail(strict=True)` 锁证。
- 实测用 2D、periodic、`edm` forcing、64²、2000 步的弛豫，未做网格/时间收敛研究
  （共存密度对分辨率有弱依赖，属 G6 范畴）。
- 三维 SCMP（`multiphase3d`）与 cavitation 的三维路径未触碰。

## 8. 复现与测试

```bash
# 审计 + 图表 + 本报告（数字全部程序化注入）
PYTHONPATH=src python tools/g3_ledger_audit.py

# 判据测试（账本锚点、共存/不共存、μ 相等、符号守卫、力调用回归、阈值物理）
PYTHONPATH=src python -m pytest tests/test_sc_eos_coexistence.py -v
```

## 9. 审计链

| 环节 | 位置 |
|---|---|
| 账本条目 | `benchmarks/gaps.md` G3（本节点追加了"现状"段，指向本文件） |
| EOS/共存态实现 | `src/tensorlbm/sc_eos_coexistence.py` |
| 修复点 | `src/tensorlbm/cavitation.py`（力调用、默认耦合、守卫、参考密度注释） |
| 接口默认值 | `app/backend/routers/solver.py` → `CavitationFlowParams.G` |
| 数据工件 | `artifacts/g3_ledger/audit.json`、`artifacts/g3_ledger/fig*.png` |
| 判据测试 | `tests/test_sc_eos_coexistence.py` |
| 生成脚本 | `tools/g3_ledger_audit.py`、`tools/g3_ledger_report.py` |
