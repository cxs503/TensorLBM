# XVAL-LBM-NEGATIVE-NO-OSS-REFERENCE（生成报告快照）

> 本文件是 `scripts/xval_negative_no_oss_reference.py` 生成的报告快照；设计/判据见 `xval/XVAL-LBM-NEGATIVE-NO-OSS-REFERENCE/predeclared.json`，机器可读汇总见 `artifacts/xval_negative_no_oss_reference/summary.json`。

# XVAL-LBM-NEGATIVE-NO-OSS-REFERENCE — 与成熟 OSS 实现的交叉对标 + 无参考能力的诚实登记

repo: TensorLBM · mission 17996 · branch `agent/m17996`

## ① 问题 + 参考出处

**问题**：TensorLBM 的 D2Q9/SRT 核心在与"成熟开源实现"对照时是否 <15%？以及：当某一能力在本环境里**根本没有可解析的 OSS 参考**时，判据应当如何处置（不能凭记忆/复制补造参考列）。

**采纳的做法（anchor = `XVAL-LBM-NEGATIVE-NO-OSS-REFERENCE`，来自 TensorKnowledge OSS 的 `negative-no-oss-reference` 实践）**：一条对标结论只允许两种诚实状态 ——（a）参考实现在**本环境可解析**（probe：import + version + 源码 sha256 / 可执行路径），给出数值对照；（b）参考不可解析或不存在 → 记 **NEGATIVE**（"no OSS reference"），带 probe 证据进 negative-knowledge 台账，能力置 WITHHELD，绝不允许"看着差不多就写通过"。

本单把它实现成 TensorLBM 里的机制（`src/tensorlbm/xval/oss_reference.py`）并**真跑**了对照：

- **OSS 参考实现**：`lbmpy 2.0`（FAU Erlangen / pycodegen 家族，Apache-2.0），由 `pystencils` 生成 collide+stream kernel，代码路径与我们的 `tensorlbm.d2q9` / `tensorlbm.solver` 完全独立。
  probe 指纹：version `2.0`，path `/usr/lib/python3.11/site-packages/lbmpy/__init__.py`，sha256 `080a6d37a69806f528c6b458590e68f6…`。
- **物理基准**：2D Taylor-Green 衰减的**闭式精确解** `u=-U0 cos(kx) sin(ky) e^{-2νk²t}`, `v=+U0 sin(kx) cos(ky) e^{-2νk²t}`，`K=0.5<|u|²>=K0 e^{-4νk²t}`（对流项是纯梯度，被压力吸收，故为 Navier-Stokes 精确解）。该公式与本仓 `case_spec.py` 的 TG2DSpec docstring 同源。
- **probe 快照 sha256**：`813fc5ce4dbc9ef93c6d7feb38c34e4c52ae7a79d9888df83bcabb72ae3c8e97`（`2026-11-13T11:09:51.313727+00:00`），所有对照表都绑定这一快照。

## ② 方法与实现

新增/扩展三处（全部在本仓，零平台域语义）：

1. `src/tensorlbm/xval/oss_reference.py`（新）
   - `probe_oss_references()`：对 7 个候选 OSS 参考（lbmpy / pylbm / lettuce / xlb / waLBerla / Palabos 二进制 / icoFoam）做可执行探查，记录 version / path / **sha256** / 错误原因，整份记录自带 `record_sha256`。
   - `audit_capability_coverage()`：把 13 个 TensorLBM 能力族映射到"哪条证据符号才算覆盖"（例如 `lbmpy.phasefield_allen_cahn` 才算覆盖相场；`lbmpy.partially_saturated_cells` 被**明确拒绝**为自由液面的替代），输出 OSS_REFERENCE_AVAILABLE / NO_OSS_REFERENCE。
   - `XvalReferenceGate`：PASS / FAIL / **WITHHELD** 判定机。规则：参考不可解析 → 一律 WITHHELD 并附 negative-knowledge payload；可解析 → 全部 KPI <15% 才 PASS。**代码里不存在"无参考也能 PASS"的分支**。
   - `negative_knowledge_payloads()`：按 `/tensor/negative-knowledge/register` 的 `RegisterRequest` 形状产出条目（title / hypothesis / failure_kind=`hypothesis_falsified`（平台台账枚举；语义=“该能力可直接对标 OSS”的假设被证伪） / falsification_evidence / keywords / domain_tags / source_artifacts）。
2. `src/tensorlbm/xval/runners.py`（扩展）：新增 `run_lbmpy()` OSS 适配器，消费**同一个** `TG2DSpec`，写出与 in-house 臂同名同格式的 `k_history_*.csv` / `field_*.csv` / `run_info_*.json`；数组索引在进出适配器时按 pystencils 的 `[ix,iy]` ↔ 我们的 `[iy,ix]` 规则转置，并用 step-0 场比对做**朝向自检**（见 ⑤）。
3. `scripts/xval_negative_no_oss_reference.py`（新）：按 predeclared 设计跑 8 个设计格 × 2 臂，算 KPI，过闸门，出图出表。

判据（动手前写死在 `xval/XVAL-LBM-NEGATIVE-NO-OSS-REFERENCE/predeclared.json`）：owner 带 15%；紧目标 5%；γ_K 拟合窗口固定 K/K0∈[0.2,0.9]；单臂 400 步、sample_every=10、每臂 wall 预算 600 s。

## ③ 参数表（无量纲数完整）

| cell | n | U0 | τ | ν_lattice | Re=U0·n/ν | Ma=√3·U0 | k=2π/n | steps |
|---|---|---|---|---|---|---|---|---|
| main | 64 | 0.050 | 0.8000 | 0.10000 | 32.0 | 0.0866 | 0.09817 | 400 |
| conv_n32 | 32 | 0.050 | 0.8000 | 0.10000 | 16.0 | 0.0866 | 0.19635 | 400 |
| conv_n128 | 128 | 0.050 | 0.8000 | 0.10000 | 64.0 | 0.0866 | 0.04909 | 400 |
| sens_u0_0.020 | 64 | 0.020 | 0.8000 | 0.10000 | 12.8 | 0.0346 | 0.09817 | 400 |
| sens_u0_0.100 | 64 | 0.100 | 0.8000 | 0.10000 | 64.0 | 0.1732 | 0.09817 | 400 |
| sens_tau_0.70 | 64 | 0.050 | 0.7000 | 0.06667 | 48.0 | 0.0866 | 0.09817 | 400 |
| sens_tau_1.00 | 64 | 0.050 | 1.0000 | 0.16667 | 19.2 | 0.0866 | 0.09817 | 400 |
| matched_n128 | 128 | 0.050 | 0.8000 | 0.10000 | 64.0 | 0.0866 | 0.04909 | 1600 |

两臂共享：D2Q9、单松弛 BGK/SRT、ρ0=1、dx=dt=1、初始条件在格心 `x_i=i+0.5` 上取同一公式、float64、采样步与场输出步完全相同；lbmpy 侧 `compressible=False`（不可压平衡）以对齐我们的平衡态。

## ④ 对照表（ref vs got + 误差打勾叉）

**工具对工具（got = in-house，ref = lbmpy 2.0）**

| cell | γ_K 相对差 % | u_max 相对差 % | 速度场 L2 相对差 % | <15%? |
|---|---|---|---|---|
| main | 0.000 | 0.000 | 0.022 | PASS ✓ |
| conv_n32 | 0.000 | 0.000 | 0.001 | PASS ✓ |
| conv_n128 | 0.005 | 0.010 | 0.156 | PASS ✓ |
| sens_u0_0.020 | 0.000 | 0.000 | 0.003 | PASS ✓ |
| sens_u0_0.100 | 0.005 | 0.009 | 0.089 | PASS ✓ |
| sens_tau_0.70 | 0.000 | 0.002 | 0.035 | PASS ✓ |
| sens_tau_1.00 | 0.001 | 0.000 | 0.008 | PASS ✓ |
| matched_n128 | 0.000 | 0.001 | 0.060 | PASS ✓ |

**两臂各自对精确解**

| cell | γ_K in-house | γ_K lbmpy | γ_K 精确 4νk² | in-house 对理论 % | lbmpy 对理论 % |
|---|---|---|---|---|---|
| main | 0.003854 | 0.003854 | 0.003855 | 0.047 | 0.047 |
| conv_n32 | 0.015427 | 0.015426 | 0.015421 | 0.034 | 0.034 |
| conv_n128 | 0.000965 | 0.000965 | 0.000964 | 0.128 | 0.133 |
| sens_u0_0.020 | 0.003856 | 0.003856 | 0.003855 | 0.017 | 0.017 |
| sens_u0_0.100 | 0.003845 | 0.003845 | 0.003855 | 0.266 | 0.271 |
| sens_tau_0.70 | 0.002569 | 0.002569 | 0.002570 | 0.056 | 0.056 |
| sens_tau_1.00 | 0.006425 | 0.006425 | 0.006426 | 0.004 | 0.005 |
| matched_n128 | 0.000963 | 0.000963 | 0.000964 | 0.055 | 0.055 |

**闸门判定（`XvalReferenceGate`，tolerance 15%）**

| claim | reference | status | max err % | reason |
|---|---|---|---|---|
| main::oss | oss:lbmpy | PASS | 0.022 | max KPI deviation 0.0217% < 15% against oss:lbmpy |
| main::analytic | analytic:taylor_green_2d_exact | PASS | 0.047 | max KPI deviation 0.0467% < 15% against analytic:taylor_green_2d_exact |
| conv_n32::oss | oss:lbmpy | PASS | 0.001 | max KPI deviation 0.000966% < 15% against oss:lbmpy |
| conv_n32::analytic | analytic:taylor_green_2d_exact | PASS | 0.034 | max KPI deviation 0.0342% < 15% against analytic:taylor_green_2d_exact |
| conv_n128::oss | oss:lbmpy | PASS | 0.156 | max KPI deviation 0.156% < 15% against oss:lbmpy |
| conv_n128::analytic | analytic:taylor_green_2d_exact | PASS | 0.128 | max KPI deviation 0.128% < 15% against analytic:taylor_green_2d_exact |
| sens_u0_0.020::oss | oss:lbmpy | PASS | 0.003 | max KPI deviation 0.0035% < 15% against oss:lbmpy |
| sens_u0_0.020::analytic | analytic:taylor_green_2d_exact | PASS | 0.017 | max KPI deviation 0.0167% < 15% against analytic:taylor_green_2d_exact |
| sens_u0_0.100::oss | oss:lbmpy | PASS | 0.089 | max KPI deviation 0.0889% < 15% against oss:lbmpy |
| sens_u0_0.100::analytic | analytic:taylor_green_2d_exact | PASS | 0.266 | max KPI deviation 0.266% < 15% against analytic:taylor_green_2d_exact |
| sens_tau_0.70::oss | oss:lbmpy | PASS | 0.035 | max KPI deviation 0.0351% < 15% against oss:lbmpy |
| sens_tau_0.70::analytic | analytic:taylor_green_2d_exact | PASS | 0.056 | max KPI deviation 0.0558% < 15% against analytic:taylor_green_2d_exact |
| sens_tau_1.00::oss | oss:lbmpy | PASS | 0.008 | max KPI deviation 0.00838% < 15% against oss:lbmpy |
| sens_tau_1.00::analytic | analytic:taylor_green_2d_exact | PASS | 0.004 | max KPI deviation 0.0044% < 15% against analytic:taylor_green_2d_exact |
| matched_n128::oss | oss:lbmpy | PASS | 0.060 | max KPI deviation 0.0598% < 15% against oss:lbmpy |
| matched_n128::analytic | analytic:taylor_green_2d_exact | PASS | 0.055 | max KPI deviation 0.0552% < 15% against analytic:taylor_green_2d_exact |

**参考不可解析时（pre-flight，证明闸门真的挡住）**

| capability | would-be reference | status | reason |
|---|---|---|---|
| shan_chen_pseudopotential_multiphase | lbmpy | WITHHELD | lbmpy resolves but does not implement the family (rejected look-alikes: [['lbmpy.phasefield', 'lbmpy.forcemodels']]); no other OSS reference resolves here => no XVAL claim is admissible, negative-knowledge payload emitted |
| free_surface_volume_tracking | lbmpy | WITHHELD | lbmpy resolves but does not implement the family (rejected look-alikes: [['lbmpy.partially_saturated_cells', 'lbmpy.geometry']]); no other OSS reference resolves here => no XVAL claim is admissible, negative-knowledge payload emitted |
| immersed_boundary_ibm | lbmpy | WITHHELD | lbmpy resolves but does not implement the family (rejected look-alikes: none); no other OSS reference resolves here => no XVAL claim is admissible, negative-knowledge payload emitted |
| adaptive_mesh_refinement | lbmpy | WITHHELD | lbmpy resolves but does not implement the family (rejected look-alikes: [['lbmpy.sparse']]); no other OSS reference resolves here => no XVAL claim is admissible, negative-knowledge payload emitted |
| thermal_energy_transport_lbm | lbmpy | WITHHELD | lbmpy resolves but does not implement the family (rejected look-alikes: none); no other OSS reference resolves here => no XVAL claim is admissible, negative-knowledge payload emitted |
| acoustic_pulse_absorbing_boundary | lbmpy | WITHHELD | lbmpy resolves but does not implement the family (rejected look-alikes: none); no other OSS reference resolves here => no XVAL claim is admissible, negative-knowledge payload emitted |

## ⑤ 收敛性 / 参数敏感性

（下面每个数字都是脚本从 `summary.json` 的记录里程序化取出，不手抄。）

**1. 分辨率（固定 τ=0.8、u0=0.05，ν 固定、Re 随 n 变）**：工具间场差 n=32: 0.0010% → n=64: 0.0217% → n=128: 0.1560%。注意这三个点虽然步数相同（400），但 νk²t 差 4 倍（n 越大衰减越慢），所以固定步数比较的其实是**不同物理时刻**；为此补了一个信息性臂（predeclared amendments 第 1 条，不改判据）：n=128 在 1600 步（与 n=64 的 400 步对齐 νk²t=0.386）时场差 0.0598%。对齐物理时刻后差异明显变小，但仍随 n 略升（0.02% @n=64 → 0.06% @n=128），说明残差是两码离散格式/平衡态差异在更细网格上更易见的系统性项，而不是会收敛到 0 的插值误差。

**2. u0 / Ma 敏感性**：u0=0.020 (Ma=0.035): 0.0035% → u0=0.050 (Ma=0.087): 0.0217% → u0=0.100 (Ma=0.173): 0.0889%。u0 从 0.05 到 0.10（Ma 翻倍、Ma² 变 4 倍）时场差从 0.0217% 涨到 0.0889%（比值 4.09），与「两臂平衡态可压缩项不同 → 差异 ∝ Ma²」的预期同阶，这正是把 lbmpy 侧 `compressible=False` 与我们的可压缩平衡并列时必须声明的系统差来源。

**3. τ / ν 敏感性**：τ=0.70 (ν=0.0667): 0.0351% → τ=0.80 (ν=0.1000): 0.0217% → τ=1.00 (ν=0.1667): 0.0084%。τ=0.7（更接近 0.5，BGK 更脆）时工具间差略大，但仍 ≪15%；τ=1.0 时差更小，符合「远离 τ=0.5 奇异点时 BGK 离散缺陷更弱」的常规认识。


- **朝向自检（必须记录的坑）**：pystencils 数组是 `[ix,iy]`、本仓求解器是 `[iy,ix]`；step-0 场比对给出 direct=7.126e-14% 与 transposed=141.421%。选小者作为约定，若两者都大则说明初始条件不一致，绝不能拿"看起来对称"糊过去。

## ⑥ 失败模式与工件

**失败模式 / 局限**

1. **本环境离线不可取 OSS 源码**：`github.com:443` 不可达（已验证），所以 Palabos/OpenFOAM/waLBerla 若未预装就**取不到**。probe 把这种"取不到"记成 `NO_OSS_REFERENCE`/WITHHELD，而不是记成"OSS 没实现这个能力"——环境事实与实现事实严格区分。
2. **两臂平衡态不完全同族**：我们的 equilibrium 保留 ρu² 项（可压缩平衡），lbmpy `compressible=False` 丢弃一部分；这部分差异由 u0/Ma 敏感性臂量化，不试图调参抹掉。
3. **残差非零是正常的**：两码的 push-pull/pull 格式与格子缺陷不同，工具间差应随 n 减小而不是趋零；把它当"通过/不通过"的连续量报告，不当 0 目标。
4. **时域只覆盖衰减段**：400 步、K/K0 到 ~0.2，未覆盖长时间统计定常/湍流转捩。
5. **negative-no-oss-reference 的边界**：`no_oss_reference` 是对**本环境**的判定，一旦有可解析参考，该能力必须从 WITHHELD 升级，条目里写明了 `differentiation_required`。

**能力覆盖审计（7/13 有可解析 OSS 参考）**

| capability | status | supporting OSS | note |
|---|---|---|---|
| core_d2q9_srt_bgk | OSS_REFERENCE_AVAILABLE | lbmpy | periodic single-relaxation-time D2Q9 step |
| trt_mrt_cumulant_collision | OSS_REFERENCE_AVAILABLE | lbmpy | multi-relaxation / cumulant collision operators |
| guo_body_force | OSS_REFERENCE_AVAILABLE | lbmpy | Guo/Shan-Chen style body-force source terms |
| bounce_back_walls_and_velocity_inlet | OSS_REFERENCE_AVAILABLE | lbmpy, palabos | half-way bounce-back, UBB/NoSlip boundary conditions |
| smagorinsky_les | OSS_REFERENCE_AVAILABLE | lbmpy | eddy-viscosity subgrid closures |
| non_newtonian_viscosity | OSS_REFERENCE_AVAILABLE | lbmpy | shear-rate dependent viscosity (Bingham / power law) |
| phasefield_allen_cahn_cahn_hilliard | OSS_REFERENCE_AVAILABLE | lbmpy | phase-field two-phase coupling |
| shan_chen_pseudopotential_multiphase | NO_OSS_REFERENCE | - | pseudo-potential Shan-Chen; the phase-field modules are not a substitute for the pseudo-potential force |
| free_surface_volume_tracking | NO_OSS_REFERENCE | - | free-surface fill level / mass tracking; partially saturated cells are a boundary treatment, not a volume-tracking free surface |
| immersed_boundary_ibm | NO_OSS_REFERENCE | - | Lagrangian IBM force spreading; no probed reference exposes it |
| adaptive_mesh_refinement | NO_OSS_REFERENCE | - | block-structured AMR; lbmpy.sparse is sparse storage, not refinement |
| thermal_energy_transport_lbm | NO_OSS_REFERENCE | - | double-distribution thermal LBM; not exposed by any probed reference |
| acoustic_pulse_absorbing_boundary | NO_OSS_REFERENCE | - | acoustic damping / absorbing boundary treatment |

**生成的 negative-knowledge 载荷（6 条，可直接 POST /tensor/negative-knowledge/register）**

| capability | failure_kind | title |
|---|---|---|
| shan_chen_pseudopotential_multiphase | hypothesis_falsified | TensorLBM shan_chen_pseudopotential_multiphase 无本地可解析 OSS 参考（negative-no-oss-reference） |
| free_surface_volume_tracking | hypothesis_falsified | TensorLBM free_surface_volume_tracking 无本地可解析 OSS 参考（negative-no-oss-reference） |
| immersed_boundary_ibm | hypothesis_falsified | TensorLBM immersed_boundary_ibm 无本地可解析 OSS 参考（negative-no-oss-reference） |
| adaptive_mesh_refinement | hypothesis_falsified | TensorLBM adaptive_mesh_refinement 无本地可解析 OSS 参考（negative-no-oss-reference） |
| thermal_energy_transport_lbm | hypothesis_falsified | TensorLBM thermal_energy_transport_lbm 无本地可解析 OSS 参考（negative-no-oss-reference） |
| acoustic_pulse_absorbing_boundary | hypothesis_falsified | TensorLBM acoustic_pulse_absorbing_boundary 无本地可解析 OSS 参考（negative-no-oss-reference） |

**工件清单**（`artifacts/xval_negative_no_oss_reference/`）

- `oss_reference_probe.json` — OSS 参考快照（含 sha256）
- `capability_coverage.json` — 能力覆盖审计
- `negative_knowledge_payloads.json` — 待登记载荷
- `manifest.json` — 工件 sha256（{"oss_reference_probe.json": "0287bd08e53202cfd0c335c34534259ce99b69f48428505f8de67a24d359ce7b", "capability_coverage.json": "9e43e54843317bc840f4d1f7be38d67f4f7beba8f89b9c50cdadd17d74bf07a8", "negative_knowledge_payloads.json": "2c7d20c286c97b8ac45acde8eeb02352c5cec69a0fa48d6009352c4b2fc09e4c"}）
- `cells/<cell>/...` — 每格两臂的 history/field/run_info + `cell_result.json` + `fields.npz`
- `figures/` — fig1_decay.png, fig2_convergence.png, fig3_sensitivity.png, fig4_fields.png
- `summary.json` — 全部 KPI + 闸门判定的机器可读汇总

**复现命令**

```
python scripts/xval_negative_no_oss_reference.py --out artifacts/xval_negative_no_oss_reference
python -m pytest tests/test_xval_oss_reference.py -q
```
