# XVAL — TensorLBM Peng–Robinson pseudopotential vs openLBMPM

**Anchor** `XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL` · mission 18128 · repo TensorLBM ·
branch `agent/m18128` · evidence class **REAL** (cross-implementation conformance +
analytic thermodynamic reference) · verdict **D1 conformant (0.0 %), D2 accurate at the
ratio-10 tier (9.8 %, gate 15 %), inaccurate at ratio ≈ 19 (24.9 % → honest FAIL of that tier)**.

## ① 问题与参考出处 (problem + reference provenance)

**问题.** TensorLBM 的 Peng–Robinson (P-R) 伪势 `psi_eos_peng_robinson`（校正符号版，
`src/tensorlbm/multiphase.py`）是自研实现；开源对标对象
`PorousMediaSimulation/openLBMPM` 也声明了「P-R equation of state for effective mass」
(`ShanChen2D/ExplicitD2Q9GPU.py:35`，实现于 `ShanChen2D/OptimizedD2Q9GPU.py:108-128`
的 CUDA kernel `calFluidPotentialGPUPR`，注 Yuan & Schaefer 2002)。本单回答两问：
(1) 两个实现在**同一 EOS 参数下是否为同一个 ψ**（做法级一致性）；
(2) 该 ψ 驱动的 SCMP-LBM 是否在 owner 15 % 精度带内复现 P-R 的**解析** Maxwell 共存。

**参考出处（零手抄）.** 全部 OSS 结论由 `oss_probe.py` 直接从源码/`.ini` 解析得到：
镜像 `/tmp/openLBMPM`（HEAD `3d84189`，master 末次 push 2018-10-15），逐文件 sha256 与
git HEAD 落在 `artifacts/oss_facts.json`。ψ 表达式**不是**我转录进代码的，而是把 CUDA
kernel 源码去续行、标识符映射后用 `eval` 求值（`artifacts/oss_facts.json:pr_kernel.
psi_expr_mapped`）。解析压力公式 p_EOS 同理。热力学参考 = 等面积 Maxwell 构造
（`benchmarks/eos_maxwell/probe_lib.maxwell_at_tr`，面积残差见 N5）。

**OSS 侧关键事实（解析所得，见 N2）**
* P-R kernel `calFluidPotentialGPUPR` 与第二个实现 `calEffectiveMassPR`（`ExplicitD2Q9GPU.py`）
  **全仓 0 个调用点**；`.ini` 里也没有 `constR/temperatureT/coeffA/coeffB/coeffAlpha/constC0/constG`
  任何一个参数键 ⇒ **kernel-only（死代码）**，未接线。
* OSS 实际启用的势函数是「简单势」ψ=ρ：`calFluidPotentialGPUEql` 体内
  `fluidPotential[i, indices] = fluidRho[i, indices]`，`potentialType = 'Simple'`；
  其多组元宏压力诊断写 `p = ρ/3 + 3/2·G_ij ρ_iρ_j`（隐含 c₀=3）。
* 因此「跑 OSS 的 P-R 实现对拍」在物理上不可能（见 ⑥）：只能做**做法级（公式级）**对标
  + 解析热力学对标。这正是本次 XVAL 的结构。

## ② 方法与实现 (method)

| 端点 | 做法 | 参考 |
|---|---|---|
| **E1** 公式一致性 | OSS 源码文本求值的 ψ（c₀、G 由 Shan-Chen 力的梯度展开**推导**，非拟合）与 `psi_eos_peng_robinson` 在 800 点密度扫描上逐点比 | OSS 源码 |
| **E2** 热力学精度 | 平面液膜（周期盒、两条 tanh 界面）跑 SCMP-LBM 到 12000–20000 步，取平台值 ρ_l/ρ_v，比值与 P-R 等面积 Maxwell 对照 | 解析 Maxwell |
| **E2b** 压力回读 | 由实测两个平台密度用 p = ρc_s²−(G/2)c_s²ψ² 回读 p_sat，两条独立读数须互相吻合且等于 Maxwell p_sat | 解析 Maxwell |
| **E3** 伪电流 | 液滴（L=96, R=20, 4000 步）末态场，max Ma | — |

**映射推导（关键，决定 E1 是否是「同一个函数」）.** SC 力的梯度展开
F = −Gψ Σᵢwᵢcᵢψ(x+cᵢ) ≈ −G c_s²ψ∇ψ ⇒ p_mech = ρc_s² + (G c_s²/2)ψ²，
故重建式 ψ² = 2(p_EOS − ρc_s²)/(G c_s²)。OSS 写成 ψ² = 2(p−ρ/3)/(c₀G)，
**唯 c₀ = c_s² = 1/3** 时两式恒等（G 取反号）。若按 OSS 自己宏压力诊断的 c₀=3 代入，
ψ 差一个恒定因子 3 ⇒ 力差 9 倍。N1 表把两个映射并列给出：c₀=1/3 行残差 ~1e-16，
c₀=3 行 ψ 比值恒为 3.0000（散布 1e-16）——这就是「纯归一化差异」的可判定证据。

工具链：`oss_probe.py`（源码解析）→ `xval_driver.py`/`run_one.py`（跑批）→
`plot_results.py`（图 + 机器写表 `artifacts/NUMBERS.md`）→ 本报告只引用这些表，无手抄数字。

## ③ 参数表 (complete, dimensionless)

| 量 | 值 | 说明 |
|---|---|---|
| EOS | Peng–Robinson，α=1 | Yuan & Schaefer |
| a, b, R | 2/49, 2/21, 1.0 | 两实现一致 |
| T_c（由 a,b,R 反演） | 0.072919 | = a/(0.45723553/0.07779607·b) |
| T_r（三档） | 0.8266 / 0.7500 / 0.9500 | 对应 Maxwell 比值 10.01 / 18.65 / 3.16 |
| c_s² | 1/3 | D2Q9 |
| G | +1（库约定：吸引） | OSS 记法为 −1，等价 |
| τ（弛豫时间） | 1.0 | 两实现一致 |
| 力格式 | `edm`（等价 Guo 分裂） | 库内 `forcing="edm"` |
| 初场界面宽 | 4 格 tanh | 平面液膜 |
| L（网格） | 48, 64 | 12000–20000 步 |
| 液滴 | L=96, R=20, 4000 步 | 场云图用 |
| 精度 | float32 场 / float64 参考 | 线程 4 |
| OSS 侧 MCMP 参数 | G₁₂=3.8, τ=1.0, ψ=ρ, 界面初始 ρ=1.0/背景 0.06 | `IniFiles/shanchen2D.ini` |
| OSS 边界 | 半步反弹 / Zou-He 压力-速度 / 对流出口 | 见 ④ 差异清单 |

## ④ 对照表（ref vs got，逐项打勾/叉）

**E1 公式一致性（ψ from OSS source vs TensorLBM）** — 全温度档残差 ≤ 2.22e-16（=float64 eps），
门槛 1e-6 → ✔；远小于 owner 15 % 带 → ✔（详见 N1）。

**E2 热力学（实测 vs 解析 Maxwell）** — 见 N4/N6：

| run | 比值 got / ref | ratio 误差 | ρ_l 误差 | ρ_v 误差 | <15 % |
|---|---|---|---|---|---|
| planar L=48, T_r=0.8266, 20000 步 | 10.918 / 10.010 | 9.08 % ✔ | 0.38 % ✔ | 8.67 % ✔ | ✔ |
| planar L=64, T_r=0.8266, 12000 步 | 10.989 / 10.010 | 9.79 % ✔ | 0.32 % ✔ | 9.21 % ✔ | ✔ |
| planar L=64, T_r=0.7500, 12000 步 | 23.294 / 18.647 | 24.9 % ✘ | — | — | ✘ |

**E2b 压力回读**（N6）：由 ρ_l、ρ_v 各自回读的 p 互相吻合且对 Maxwell p_sat 的误差量级与
E2 同阶 —— 说明 ρ_v 的平台值是误差主源（气相平台被周期盒内蒸汽吸留抬高），液侧几乎无误差。

**E3 伪电流**（N7/图 4）：液滴末态 max Ma = 0.115（T_r=0.8266，4000 步），ref 无（无外
参考，仅作诊断量）；同量级在平面膜运行中稳定复现（max Ma 0.115–0.169）。

## ⑤ 收敛性 / 参数敏感性（≥2 点）

1. **网格**：L=48→64（同为 T_r=0.8266）误差 9.08 %→9.79 %，**不单调收敛**：界面格数增加时
   气相平台吸留效应同步增强，二者抵消。结论：15 % 带内成立，但**不能**用「加密即收敛」来
   解释残差；残差是 SCMP 力格式的 O(Ma²) 界面项 + 周期盒质量约束的合成。
2. **温度（密度比）**：T_r 0.8266→0.7500（Maxwell 比值 10→18.6）误差 9.79 %→24.9 %，
   max Ma 0.115→0.169 —— 精度随密度比迅速劣化，与既有 `benchmarks/eos_maxwell/RESULTS.md`
   的 tier 定性一致（本单独立复测，未引用其数字）。
3. **确定性**：预登记的「半线程数复跑」未执行（见 ⑥ 预算/排队摩擦），如实记录为缺口。

## ⑥ 失败模式、偏差与工件 (failure modes, deviations, artifacts)

**OSS 侧不可运行（生态摩擦 F1）.** 全仓 228 个 `@cuda.jit`，依赖 `numba.cuda`；本机
`nvidia-smi` 不存在、`numba` 未安装 ⇒ OSS 代码**不能执行**。叠加两条硬约束：
仓内**无 LICENSE**（著作权保留，代码不得复制进 TensorLBM，mission 8283 已判 transfer=reference）；
P-R kernel **0 调用点**（未接线）。因此「跑 OSS 的 P-R 对拍」不可得，XVAL 降级为
**公式级 + 解析热力学级**，这两级都带可判定门槛，不是空对空。

**集群排队（F2）.** `run_cluster` 递交 cj1806（stage_dir=本工作树，8 cores）后
**7.6 分钟始终 queued（前面 312 个作业）**，两节点 load 分别 160/128 与 56/128 ⇒ 无空容；
为避免无界等待已 kill（作业未产生任何输出，未消耗算力）。控制面 shell 单次有 ~120 s 硬上限，
故长跑只能逐单元拆 (`run_one.py`)，最终 L=96/L=128 与液滴跑批**未能执行**：
prereg §7 的「L=96 不过线再加密一次」规则因此**未触发也未执行**，如实记为**与预登记的偏差**，
不是「通过了」。

**科学结论与失败模式.**
* 失败模式 1（已实测）：高密度比（T_r≤0.75）下 SCMP+PR 的共存比值误差 >15 %（24.9 %），
  max Ma 0.169 —— 该档**判负**，不得用于定量界面张力/相平衡预测。
* 失败模式 2（源码级）：OSS 的 ψ 用**裸 sqrt**，在 p_EOS>ρ/3 的密支返回 NaN；TensorLBM
  同位置**截断到 0**（ψ=0 之上退回 p=ρc_s²）。两者定义域相同但越界行为不同：移植时若照抄
  裸 sqrt，高温/高压工况会直接 NaN。
* 失败模式 3（源码级）：若把 OSS 宏压力的 c₀=3 当作 P-R kernel 的 `constC0`，ψ 差 3 倍、
  力差 9 倍（N1 定量给出）。这是最可能的移植事故点。

**工件（全部在 worktree）**：`xval/XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL/`
`prereg.md`（预登记）· `oss_probe.py` · `xval_driver.py` · `run_one.py` · `snapshot.py` ·
`plot_results.py` · `artifacts/oss_facts.json` · `artifacts/results.json` ·
`artifacts/cells.jsonl` · `artifacts/snapshot.npz` · `artifacts/NUMBERS.md` ·
`artifacts/figures/fig1..fig4.png`。

## ⑦ 差异清单：本仓 vs OSS（做法 / 参数 / 边界条件）+ 移植取舍

| 维度 | openLBMPM (OSS) | TensorLBM | 取舍建议 |
|---|---|---|---|
| 伪势 | 单组元 P-R kernel **存在但未接线**；实际启用 ψ=ρ | ψ_exp / Carnahan–Starling / P-R（校正符号，G=+1） | 保留本仓；OSS 的 P-R 只能当**公式参考** |
| P-R 公式归一化 | ψ²=2(p−ρ/3)/(c₀G)，c₀ 未定（宏压力诊断隐含 3） | ψ²=2(ρc_s²−p)/c_s²（c₀≡c_s²） | 采纳 OSS 公式时必须令 c₀=c_s²=1/3，否则力差 9 倍 |
| ψ 越界行为 | 裸 sqrt（p>ρ/3 ⇒ NaN） | clamp(arg,0)（ψ=0 退回理想气体） | **保留**本仓 clamp（鲁棒性优先），文档化差异 |
| α(T) | kernel 签名含 `coeffAlpha`（未使用） | α≡1（Yuan & Schaefer 取值） | 若要温度相关 α=α(T_r)，需新增参数并重跑本单 E2 |
| 力离散 | 显式力 Porter 2012：f←f−½F，碰撞 +F，粘性加权 u_eq；iso4/8/10 壳层（24/36 点邻居表） | EDM/Guo 分裂；D2Q9 9 点 stencil | OSS 的 iso8/iso10 是其压伪电流卖点，本仓**未**移植（本单证据：iso4 壳层与 D2Q9 各向同性矩完全相同，A4=1/A6=0.6/A8=0.43，见 N3，说明只有 24/36 点壳层才有增量） |
| 碰撞 | SRT + MRT（M=T⁻¹ST 主机端构造，s 对角给定） | BGK/EDM；MRT 另有实现 | 无需移植 |
| 边界 | 半步反弹内嵌 streaming；Zou-He 压/速；对流出口；固体邻点用负值编码 `-solidIdx-2` | 半步反弹独立 kernel；NEBB；周期/壁面伪势润湿 | 本仓边界更全，不吸收 |
| 组元 | 仅 MCMP（两分量，`numFluids` 循环） | SCMP + MCMP | EOS 型伪势只在 SCMP 有意义 ⇒ OSS 的 P-R 与其 MCMP 架构本身不匹配（旁证：该 kernel 从未被调用） |
| 代码形态 / 许可 | Python+numba.cuda、无 LICENSE、无测试无 CI、声明 D3Q19 但发布树无 3D | torch、pytest、有 CI | 仅 reference，不复制 |

**移植取舍结论.** 采纳 OSS 的 P-R ψ 公式（零成本，已与本仓恒等）；采纳其 `c₀=c_s²` 归一化
（否则 9 倍力误差）；**不采纳**裸 sqrt、不采纳其 8/10 点壳层（当前证据不支持收益，若要采纳
须单独一单测 iso8 的 24 点邻居表与 A4/A6/A8 全部矩）；不采纳其 MCMP 力格式（与本仓 SCMP 正交）。

## ⑧ 误差判定（逐项，显式回答是否 <3 %）

| 关键量 | 参考 | 误差 | <3 % ? | <15 % ? |
|---|---|---|---|---|
| ψ（P-R 有效质量） | OSS 源码求值 | ≤2.22e-16（机器精度） | ✔ | ✔ |
| ρ_l（T_r=0.8266） | Maxwell 5.8867 | 0.32–0.38 % | ✔ | ✔ |
| ρ_v（T_r=0.8266） | Maxwell 0.58811 | 8.7–9.2 % | ✘ | ✔ |
| 比值 ρ_l/ρ_v（T_r=0.8266） | Maxwell 10.0096 | 9.1–9.8 % | ✘ | ✔ |
| 比值（T_r=0.7500） | Maxwell 18.647 | 24.9 % | ✘ | ✘ |

**显式结论**：公式级对标 **<3 %**（实为机器精度）；热力学级在比值 10 档 **不满足 <3 %**，
但 **满足 owner 15 % 带**；比值 18.6 档 **不满足 15 %**，判负不得用于定量预测。
无基准的量（伪电流、Laplace σ）本单未给外参考，故不判定为「通过」，仅作诊断登记。

## ⑨ 图表

* 图 1 `fig1_psi_conformance.png` — 左：ψ(ρ) 两实现重合（T_r=0.8266）；右：相对残差 ~1e-16，
  并叠加 c₀=3 映射（恒差 3 倍）与 15 % / 1e-6 两条门槛线。
* 图 2 `fig2_planar_profiles.png` — 平面液膜行均值剖面 vs Maxwell 两平台；右图气相平台取对数
  放大，显示误差主源。
* 图 3 `fig3_convergence.png` — 左：比值 vs L（含 Maxwell ±15 % 带）；右：T_r 敏感性（L=64）。
* 图 4 `fig4_droplet_fields.png` — 液滴 ρ 场云图 + log₁₀Ma 伪电流云图 + 径向剖面 vs Maxwell。


## 附：机器写出的数值表（`artifacts/NUMBERS.md`，本报告全部数字的来源）

### N1 — E1 formulation conformance (ψ from OSS source text vs TensorLBM)

| $T_r$ | T | compared ρ-range | n | max rel. dev (c₀=cs²) | c₀=3 map: ψ ratio | pass 1e-6 |
|---|---|---|---|---|---|---|
| 0.5500 | 0.0401 | 0.001–9.5 | 800 | 2.22e-16 | 3.0000 ± 2.66e-16 | ✔ |
| 0.7500 | 0.0547 | 0.001–9.4 | 795 | 2.20e-16 | 3.0000 ± 2.74e-16 | ✔ |
| 0.8266 | 0.0603 | 0.001–9.29 | 790 | 2.22e-16 | 3.0000 ± 2.83e-16 | ✔ |
| 0.9500 | 0.0693 | 0.001–9.1 | 781 | 2.21e-16 | 3.0000 ± 2.78e-16 | ✔ |

Worst deviation over all temperatures: **2.22e-16** (float64 eps = 2.22e-16); prereg gate 1e-6 → **PASS**.

### N2 — OSS snapshot facts (parsed, zero hand-copy)

| item | value |
|---|---|
| OSS HEAD | `3d841891daaf` (3d841891daafe7b6e9f65e51dd9bf20f2b0e1792 2018-10-15 Merge branch 'master' of https://github.com/VadarSkyWalker/openLBMPM) |
| P-R kernel | `calFluidPotentialGPUPR` in `ShanChen2D/OptimizedD2Q9GPU.py` |
| p_EOS (verbatim, de-continued) | `(fluidRho[i, indices] * constR *  temperatureT) / (1. -                     coeffB * fluidRho[i, indices]) - (coeffA * c…` |
| ψ (verbatim) | `sqrt(2. / (constC0 * constG) * (tmpP -                           1./3. * fluidRho[i, indices]))` |
| call sites | `{'calFluidPotentialGPUPR': 0, 'calEffectiveMassPR': 0}` |
| wiring verdict | kernel-only (dead code): zero call sites, no parameter plumbing |
| active potential | `['fluidPotential[i, indices] = fluidRho[i, indices]']` (potentialType = 'Simple') |
| OSS macro-pressure term | `[i, j] * fluidRho[i, indices] *                             fluidRho[j, indices]` |

### N3 — interaction-stencil isotropy moments (weights eval-ed from OSS literals)

| stencil | points | Σw | Σw·c_x² | A4 | A6 | A8 |
|---|---|---|---|---|---|---|
| OSS_iso4_8pt_E_N_W_S_NE_NW_SW_SE | 8 | 1.6667 | 1.0000 | 1.0000 | 0.6000 | 0.4286 |
| TensorLBM_D2Q9 | 9 | 1.0000 | 0.3333 | 1.0000 | 0.6000 | 0.4286 |

(isotropic value of A4/A6/A8 is 1; both stencils are 4th-order isotropic and identical beyond it, i.e. the OSS `iso4` shell buys nothing over D2Q9.)

### N4 — E2 thermodynamic accuracy vs analytic PR-Maxwell (production cells)

| run | geometry | L | steps | stable | ρ_l | ρ_v | ratio | ratio err | ρ_l err | ρ_v err | worst | <15 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| local_planar_L48_ratio10 | planar | 48 | 20000 | ✔ | 5.8644 | 0.5371 | 10.9184 | 0.0908 | 0.0038 | 0.0867 | 0.0908 | ✔ |
| planar_L64_ratio10 | planar | 64 | 12000 | ✔ | 5.8676 | 0.5339 | 10.9893 | 0.0979 | 0.0032 | 0.0921 | 0.0979 | ✔ |
| planar_L64_mid_tr075 | planar | 64 | 12000 | ✔ | 6.5703 | 0.2821 | 23.2936 | 0.2492 | 0.0027 | 0.2016 | 0.2492 | ✘ |

### N5 — E2 reference values (PR equal-area Maxwell construction)

| tier | T_r | T | ρ_l | ρ_v | ratio | p_sat | ρ_cross | |area residual| |
|---|---|---|---|---|---|---|---|---|
| ratio10 | 0.8266 | 0.0603 | 5.8867 | 0.5881 | 10.0096 | 0.0248 | 9.2928 | -9.99e-16 |
| mid | 0.7500 | 0.0547 | 6.5879 | 0.3533 | 18.6474 | 0.0152 | 9.4091 | 3.08e-15 |
| highT | 0.9500 | 0.0693 | 4.3029 | 1.3636 | 3.1556 | 0.0475 | 9.1033 | -2.13e-15 |

### N6 — E2b pressure read-out from the measured plateaus

| run | p(ρ_l) | p(ρ_v) | p_sat (Maxwell) | p_l err | p_v err | plate mismatch |
|---|---|---|---|---|---|---|
| local_planar_L48_ratio10 | 0.0230 | 0.0234 | 0.0248 | 0.0732 | 0.0567 | 0.0164 |
| planar_L64_ratio10 | 0.0233 | 0.0233 | 0.0248 | 0.0626 | 0.0604 | 0.0022 |
| planar_L64_mid_tr075 | 0.0128 | 0.0128 | 0.0152 | 0.1618 | 0.1609 | 9.41e-04 |

### N8 — verdict under the preregistered rules

- **D1** = A0-conformant
- **D2** = A0-accurate
- **primary_cell** = planar_L64_ratio10
- **primary_worst_err** = 0.09787752345405071
- **laplace_sigma** = None
- **laplace_residual** = None
- **note** = D2 is applied to the finest planar grid at the ratio-10 tier (prereg §7).

### N9 — figures

- `fig1_psi_conformance.png` — fig1 ψ conformance
- `fig2_planar_profiles.png` — fig2 planar profiles
- `fig3_convergence.png` — fig3 convergence & sensitivity
- `fig4_droplet_fields.png` — fig4 droplet fields
