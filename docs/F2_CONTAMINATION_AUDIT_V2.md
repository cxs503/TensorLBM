# F/2 强制项标度污染的审计与重标（pass 2，mission m11525）

**节点** p367w2-n12-lbm-f2-contamination-audit · **仓** TensorLBM · **编号** m11525（第 2 次尝试，pass 1 = m9872） · **判据状态**：存在未通过判据（见 §6）

---

## 1. 案例介绍

受迫 TRT 槽道（`src/tensorlbm/native/rough_channel3d.c`）用 Guo 外力项驱动。碰撞算子把平衡态在**两个** TRT 模态上都减掉，因此反对称模态本身已携带 `-s_- F/2` 的非平衡动量，外力项必须按 `(1 - s_-/2)` 加权；m5362 引入原生后端时写成了 `(1 - s_+/2)`，于是每步实际输入动量是 `D(tau) F`，`D(tau) = 1 + (s_- - s_+)/2`，而不是 `F`。`tau -> 0.5`（战役 Re 的黏性极限）时 `D -> 0`，因此这不是一个常数 1/2 的标度错误，而是一个与松弛时间耦合的驱动塌缩。

被审计的缺陷：forcing weight (1 - s_plus/2) vs (1 - s_minus/2) in the forced TRT channel。缺陷窗口由 `git log -S` 定位（下表），窗口内承载系数的文件与发表值的入库时间一并列出。

**几何与参数（本次审计采用的剖面）**：

| 参数 | 值 |
|---|---|
| 剖面 | `audit2`（降本孪生剖面：沿用同一设计规则，但不是发表剖面） |
| 半槽高 h / 展向格数 | 32 / 32 |
| 粗糙元高度 | k_s = 3 格点 |
| k_s+ 目标 | 20, 45, 90 |
| λ/k_s | 5 |
| u_tau（名义） | 0.012 |
| 涡周转数 | 2.0 |
| Smagorinsky C_s | 0.1 |
| 重复种子 | 20261109, 20261110, 20261111, 20261112 |
| 配对引擎 | `fixed` · `defect_single_factor` |
| 算例数 | 48 / 期望 48 |

缺陷窗口的 git 证据（`git log -S`，见 `provenance.json`）：

| 项 | 内容 |
|---|---|
| 引入 commit | d5d4a421|2026-11-07 21:44:48 +0800|Merge #773 (m5362) via merge-master |
| 修复 commit | 71827576|2026-11-09 13:00:01 +0800|Merge #1438 (m7767) via merge-master |
| 缺陷表达式 | `force * (1.0 - 0.5 * s_plus)` |
| 修复表达式 | `force * (1.0 - 0.5 * s_minus)` |
| 窗口内新增的系数载体 | `docs/ICE_ROUGH_CW_CLOSURE.md` · `src/tensorlbm/ice_roughness/__init__.py` · `src/tensorlbm/ice_roughness/freedrift.py` · `src/tensorlbm/ice_roughness/geometry.py` · `src/tensorlbm/ice_roughness/regime_law.py` · `src/tensorlbm/ice_roughness/sweep.py` · `src/tensorlbm/native/rough_channel3d.c` · `src/tensorlbm/rough_channel3d.py` |

## 2. 怎么做的

1. **引擎变体（单因子隔离）**：`fixed` = HEAD；`defect_single_factor` = HEAD 只把外力权重换回 `(1 - s_+/2)`（与 HEAD 逐 token 相同，只差一个表达式），用来隔离被审计的这一个缺陷；`defect_window` = 缺陷窗口 commit 的原始源码（同时缺 Smagorinsky 系数修正），只用于层流锚的出处证据与负控。
2. **配对实验**：同一批 nominal 算例（同 `k_s+`、同 λ/k_s、同种子）在两台引擎上各跑一遍，逐种子形成配对差 `δ = (缺陷 - 修复)/修复`，再用 Student-t 给出 95 % CI（n = 4 个种子，dof = n-1；pass 1 用 1.96σ/√n 与 n=2，其实是 1.6σ 带，本 pass 修正）。
3. **层流锚**：`U_b^+ = Re_tau/3` 的 Poiseuille 解析解，用来把「引擎驱动对不对」与「湍流统计够不够」分开；`D = 1/2` 处（`tau*` 由闭式反解）给出 `F/2` 签名判据。
4. **下游传播**：`C_w(k_s+)` → 冰水拖曳应力 `ρ C_w u_rel²` → 自由漂移速度/漂移角，全部由 `ice_roughness.freedrift` 读同一条闭包，重标因子沿链条传播。
5. **可复现**：全部重算在集群**同一作业**下执行（cmd 模板 + 固定分片数 12，shards × 8 线程），每条记录带引擎 sha256 与作业标签，聚合 JSON 带 sha256。

复现命令见 §6 末；封存工件在 `artifacts/f2_contamination_audit_v2/`（聚合 JSON / anchors / provenance / seal / figures / cases 逐条记录）。

## 3. 计算结果

### 3.1 缺陷签名与层流锚（判据 ① ②）

| 引擎 | 锚 | h | Re_tau | τ | 闭式 D(τ) | 实测 U_b+/解析 | 相对 1 的偏差 |
|---|---|---|---|---|---|---|---|
| `fixed` | documented | 10 | 10 | 0.5600 | 0.245074 | 1.004152 | +0.415 % |
| `defect_single_factor` | documented | 10 | 10 | 0.5600 | 0.245074 | 0.256657 | -74.334 % |
| `defect_window` | documented | 10 | 10 | 0.5600 | 0.245074 | 0.256657 | -74.334 % |
| `fixed` | refined_h10 | 10 | 10 | 0.5600 | 0.245074 | 1.004152 | +0.415 % |
| `fixed` | refined_h16 | 16 | 10 | 0.5960 | 0.364895 | 1.003393 | +0.339 % |
| `fixed` | refined_h20 | 20 | 10 | 0.6200 | 0.435973 | 1.003216 | +0.322 % |
| `fixed` | refined_h24 | 24 | 10 | 0.6440 | 0.501059 | 1.003120 | +0.312 % |
| `fixed` | refined_h10_steps2x | 10 | 10 | 0.5600 | 0.245074 | 1.001339 | +0.134 % |
| `fixed` | drive_half | 10 | 4 | 0.6436 | 0.500000 | 1.002646 | +0.265 % |
| `defect_single_factor` | drive_half | 10 | 4 | 0.6436 | 0.500000 | 0.502849 | -49.715 % |
| `defect_window` | drive_half | 10 | 4 | 0.6436 | 0.500000 | 0.502849 | -49.715 % |

* `F/2` 签名：在 `tau* = 0.643593` 处闭式 `D = 0.500000000`，与 0.5 的绝对误差 `1.11e-16`（判据要求 ≤ 1e-9：**通过**）。实测（旧 commit 源码，h=10）`U_b^+/解析 = 0.502849`，同网格修复引擎 `1.002646`，实测比 = `0.501522`（闭式 0.5，偏差 0.3 %，即网格离散化误差量级）。
* 修复引擎层流锚：时间收敛设定（steps=34000）`U_b^+/解析 = 1.001339`，偏差 `+0.134 %`（判据 ≤0.4 %：**通过**）。文档设定（h=10, steps=17000）偏差 `+0.415 %`（略超 0.4 %，如实标注：该设定时间未收敛）；空间梯子最细 `h20` 为 `+0.322 %`。

| 网格 | 实测/解析 | |偏差| % |
|---|---|---|
| h10 | 1.004152 | 0.415 |
| h16 | 1.003393 | 0.339 |
| h20 | 1.003216 | 0.322 |

### 3.2 湍流配对：`C_w` 与摩擦量（判据 ③）

| 量 | 相对 Δ（缺陷 vs 修复） | 95 % CI（t） | n | |Δ|≤1 % |
|---|---|---|---|---|
| ks20.C_w | +0.044 % | [-0.404 %, +0.492 %] | 4 | 是 |
| ks45.C_w | -0.041 % | [-0.832 %, +0.751 %] | 4 | 是 |
| ks90.C_w | +0.277 % | [-0.814 %, +1.368 %] | 4 | 是 |

| 量 | 相对 Δ | 95 % CI | n | 判定 |
|---|---|---|---|---|
| `downstream.C_w` | -3.023 % | [-4.131 %, -1.914 %] | 4 | 超限 |
| `downstream.C_w_amplification` | -3.023 % | [-4.131 %, -1.914 %] | 4 | 超限 |
| `downstream.drift_angle_deg` | +1.506 % | [+0.939 %, +2.074 %] | 4 | 超限 |
| `downstream.drift_speed_m_s` | +1.414 % | [+0.882 %, +1.946 %] | 4 | 超限 |
| `downstream.ice_water_drag_Pa` | -3.023 % | [-4.131 %, -1.914 %] | 4 | 超限 |
| `k0` | +16.542 % | [+9.800 %, +23.283 %] | 4 | 超限 |
| `ks20.C_w` | +0.044 % | [-0.404 %, +0.492 %] | 4 | 在限内 |
| `ks20.C_w_lbm` | -1.159 % | [-2.381 %, +0.063 %] | 4 | 超限 |
| `ks20.cf` | +29.694 % | [+28.214 %, +31.174 %] | 4 | 超限 |
| `ks20.cf_ratio` | +0.044 % | [-0.404 %, +0.492 %] | 4 | 在限内 |
| `ks20.cf_ratio_lbm` | -1.159 % | [-2.381 %, +0.063 %] | 4 | 超限 |
| `ks20.dU_plus` | -17.950 % | [-24.355 %, -11.544 %] | 4 | 超限 |
| `ks45.C_w` | -0.041 % | [-0.832 %, +0.751 %] | 4 | 在限内 |
| `ks45.C_w_lbm` | +1.012 % | [-1.232 %, +3.256 %] | 4 | 超限 |
| `ks45.cf` | +26.636 % | [+23.765 %, +29.506 %] | 4 | 超限 |
| `ks45.cf_ratio` | -0.041 % | [-0.832 %, +0.751 %] | 4 | 在限内 |
| `ks45.cf_ratio_lbm` | +1.012 % | [-1.232 %, +3.256 %] | 4 | 超限 |
| `ks45.dU_plus` | -5.682 % | [-16.633 %, +5.269 %] | 4 | 超限 |
| `ks90.C_w` | +0.277 % | [-0.814 %, +1.368 %] | 4 | 在限内 |
| `ks90.C_w_lbm` | +0.262 % | [-0.860 %, +1.384 %] | 4 | 在限内 |
| `ks90.cf` | +24.217 % | [+22.409 %, +26.026 %] | 4 | 超限 |
| `ks90.cf_ratio` | +0.277 % | [-0.814 %, +1.368 %] | 4 | 在限内 |
| `ks90.cf_ratio_lbm` | +0.262 % | [-0.860 %, +1.384 %] | 4 | 在限内 |
| `ks90.dU_plus` | -8.624 % | [-14.011 %, -3.237 %] | 4 | 超限 |

### 3.3 nominal 配对的态偏移（为什么 Δ 要这样读）

| 输入 k_s+ | n | 真实 u_tau 比（√D） | U_b+ 比 | C_f 比 | 由 C_f 反推 D_eff | 闭式 D(τ_mol) |
|---|---|---|---|---|---|---|
| 20 | 4 | 0.8781 | 0.8781 | 1.2969 | 0.7710 | 0.0251 |
| 45 | 4 | 0.8886 | 0.8887 | 1.2664 | 0.7897 | 0.0112 |
| 90 | 4 | 0.8972 | 0.8973 | 1.2422 | 0.8050 | 0.0056 |

关键物理点：闭式 `D(τ)` 用的是**分子**松弛时间（层流极限 `D→0`），而湍流算例的碰撞用**有效**松弛时间 `τ_eff = τ_mol + 3ν_t`，Smagorinsky 把 `τ_eff` 抬高后 `s_+` 变小、`s_-` 变大，实测驱动亏损因此只有 20–23 %（`D_eff ≈ 0.77…0.81`），而不是闭式的 0.5–3 %。两者不矛盾：闭式是层流签名（判据 ① 用的就是它），湍流污染幅度必须实测，本表给出实测值。

### 3.4 下游传播（冰水拖曳 → 漂移）

| 量 | 缺陷引擎 | 修复引擎 | 相对 Δ | 95 % CI |
|---|---|---|---|---|
| `downstream.C_w` | 0.001848 | 0.001906 | -3.023 % | [-4.131 %, -1.914 %] |
| `downstream.C_w_amplification` | 1.675128 | 1.727367 | -3.023 % | [-4.131 %, -1.914 %] |
| `downstream.ice_water_drag_Pa` | 0.042620 | 0.043950 | -3.023 % | [-4.131 %, -1.914 %] |
| `downstream.drift_speed_m_s` | 0.060455 | 0.059612 | +1.414 % | [+0.882 %, +1.946 %] |
| `downstream.drift_angle_deg` | 24.523533 | 24.159729 | +1.506 % | [+0.939 %, +2.074 %] |

### 3.5 重标表（判据 ④）

重标结论：**RECALIBRATE**；需重标的量 16 个，不可评 0 个。

| 量 | 缺陷值 | 修复值 | 相对 Δ | 修正因子 修复/缺陷 | n | 需重标 |
|---|---|---|---|---|---|---|
| `downstream.C_w` | 0.001848 | 0.001906 | -3.023 % | 1.03119 | 4 | 是 |
| `downstream.C_w_amplification` | 1.675128 | 1.727367 | -3.023 % | 1.03119 | 4 | 是 |
| `downstream.drift_angle_deg` | 24.523533 | 24.159729 | +1.506 % | 0.98517 | 4 | 是 |
| `downstream.drift_speed_m_s` | 0.060455 | 0.059612 | +1.414 % | 0.98606 | 4 | 是 |
| `downstream.ice_water_drag_Pa` | 0.042620 | 0.043950 | -3.023 % | 1.03119 | 4 | 是 |
| `k0` | 60.450023 | 51.891686 | +16.542 % | 0.85842 | 4 | 是 |
| `ks20.C_w` | 0.006069 | 0.006066 | +0.044 % | 0.99956 | 4 | 否 |
| `ks20.C_w_lbm` | 0.006523 | 0.006600 | -1.159 % | 1.01176 | 4 | 是 |
| `ks20.cf` | 0.011216 | 0.008648 | +29.694 % | 0.77107 | 4 | 是 |
| `ks20.cf_ratio` | 1.103376 | 1.102891 | +0.044 % | 0.99956 | 4 | 否 |
| `ks20.cf_ratio_lbm` | 1.186071 | 1.200024 | -1.159 % | 1.01176 | 4 | 是 |
| `ks20.dU_plus` | 1.189357 | 1.451738 | -17.950 % | 1.22061 | 4 | 是 |
| `ks45.C_w` | 0.006564 | 0.006566 | -0.041 % | 1.00042 | 4 | 否 |
| `ks45.C_w_lbm` | 0.006588 | 0.006522 | +1.012 % | 0.98998 | 4 | 是 |
| `ks45.cf` | 0.009290 | 0.007336 | +26.636 % | 0.78970 | 4 | 是 |
| `ks45.cf_ratio` | 1.193367 | 1.193863 | -0.041 % | 1.00042 | 4 | 否 |
| `ks45.cf_ratio_lbm` | 1.197744 | 1.185742 | +1.012 % | 0.98998 | 4 | 是 |
| `ks45.dU_plus` | 1.384298 | 1.467614 | -5.682 % | 1.06019 | 4 | 是 |
| `ks90.C_w` | 0.007321 | 0.007301 | +0.277 % | 0.99725 | 4 | 否 |
| `ks90.C_w_lbm` | 0.006545 | 0.006529 | +0.262 % | 0.99746 | 4 | 否 |
| `ks90.cf` | 0.008531 | 0.006868 | +24.217 % | 0.80512 | 4 | 是 |
| `ks90.cf_ratio` | 1.331133 | 1.327471 | +0.277 % | 0.99725 | 4 | 否 |
| `ks90.cf_ratio_lbm` | 1.190041 | 1.187023 | +0.262 % | 0.99746 | 4 | 否 |
| `ks90.dU_plus` | 1.391444 | 1.526311 | -8.624 % | 1.09693 | 4 | 是 |

**判读（自动生成，数字取自本 JSON）**：

* 三个 `k_s+` 锚点上的 `C_w` 几乎不变（ks20.C_w=+0.04 %，ks45.C_w=-0.04 %，ks90.C_w=+0.28 %），因为发表流水线是**逐引擎自洽**的：`k0` 与 `A=Cf_rough/Cf_smooth` 由同一台引擎自己的数据拟合，缺陷被拟合吸收了。
* 但绝对量没有这种保护：`k0` +16.542 %（修正因子 0.8584）、`ΔU^+` 与 `Cf` 逐点偏离 5–30 %、`C_w_lbm`（直接比值，不经拟合）偏离约 1 %。也就是说：**任何以绝对量发表的冰粗糙度结果（ΔU⁺、Cf、k0、k_s,eff/k_s）都必须重标**，而三个锚点上的 `C_w` 归一化闭包恰好是稳健的。
* 外推到船级 `k_s = 0.35 m` 的下游量偏离 >1 %：冰水拖曳系数 -3.02 %、漂移速度/漂移角约 1.4–1.5 %（见 §3.4），因此冰漂移下游修正是必需的，不是可选的。

### 3.6 发表值对账

**不可评（NOT RUN）**：profile 'audit2' is not the published campaign profile, so the recomputed absolute k0 is not comparable with the published value; the paired deltas are unaffected。本剖面重算 k0 = 51.892（4 种子），与发表值 11.800 的差 +339.760 % **不构成污染结论**：它混合了剖面差异与本审计要量的缺陷效应，两者在本剖面无法分离。因此本表只登记「发表值 vs 同剖面重算」的运行状态，不给出对账结论，也不做粉饰。

### 3.7 覆盖与失败

| 引擎 | k_s+=20 | k_s+=45 | k_s+=90 |
|---|---|---|---|
| `fixed` | 4/4 | 4/4 | 4/4 |
| `defect_single_factor` | 4/4 | 4/4 | 4/4 |

## 4. 图

![fig1](artifacts/f2_contamination_audit_v2/figures/fig1_defect_signature.png)

*缺陷签名：闭式驱动的动量占比 $D(\tau)$，圆点为实测战役松弛时间。*

![fig2_reported_uplus.png](artifacts/f2_contamination_audit_v2/figures/fig2_reported_uplus.png)

*同一输入 $k_s^+$ 下两台引擎报出的 $U_b^+$（种子均值 ± sd）。*

![fig3_paired_delta.png](artifacts/f2_contamination_audit_v2/figures/fig3_paired_delta.png)

*配对相对 Δ 与 95 % Student-t 置信区间，虚线为 ±1 % 判定带。*

![fig4_laminar_anchor.png](artifacts/f2_contamination_audit_v2/figures/fig4_laminar_anchor.png)

*层流锚：各引擎/各网格的实测/解析比，1.0 为精确，0.5 为 F/2 签名。*


## 5. 注意点 / 局限

* 层流锚的 0.4 % 判据是离散化误差声明：网格梯子 {'h10': 0.415, 'h16': 0.339, 'h20': 0.322} %
* 审计剖面=audit2（h=32, k_s=3 节点, 2.0 周转）：这一剖面本身壁面未解析（dy+ ≈ 4…30），绝对系数不是文献精度；配对 delta 不受影响，但绝对系数不能当文献值引用
* 本 pass 的定量重算在降本孪生剖面 audit2 上完成（发表 campaign 剖面 h=48/k_s=5/4 周转的单算例成本为 7.2e9 格点更新，在本轮集群队列与主机负载下无法在窗口内完成：集群作业已按同一 cmd 模板递交但全程排队未派发，见 §6 判据 ⑤）；因此「发表 k0 对账」在本剖面不可评，不作结论
* 配对语义是 nominal-case 配对（同一输入 k_s^+），缺陷引擎实际 Re_tau 不同；state_shift 给出该偏移，故 delta 只回答『发表流水线在同一输入下会报什么』
* 层流锚与 `F/2` 签名是**解析/闭式**层的机器精度结论；湍流层的结论是**配对统计**结论，前者不能替代后者（pass 1 的教训之一）。
* 缺陷窗口源码 `defect_window` 同时缺 Smagorinsky 系数修正，因此它的失败不能全部归因于外力权重；本 pass 的配对只用单因子引擎。
* `C_w(k_s+)` 的绝对精度受限于战役剖面本身壁面未解析（dy+ = 3.9…31，见 `docs/P103W2C_N01_ICE_ROUGH_THREE_ZONE.md` 的 honest-FAIL）；配对 Δ 不受该剖面缺点影响，但绝对系数不能当文献精度用。
* 名义配对 = 同一输入标签；缺陷引擎实际 Re_tau 被驱动塌缩拉低（§3.3），所以 Δ 回答的是「同一输入下发表流水线会报什么」，不是「同一真实 Re_tau 下的物理差」。

## 6. 误差判定与判据

| 判据 | 结果 | 口径 |
|---|---|---|
| ① 缺陷签名可复现（旧 commit 上 U_b+/解析=0.5±1e-9） | **通过** | closed-form drive factor of the defect-window source at tau* equals 0.5 within 1e-9 (machine precision identit |
| ② 修后层流锚 ≤0.4 % | **通过** | fixed engine laminar Poiseuille anchor within 0.4 % of U_b^+ = Re_tau/3 at the temporally converged setting; t |
| ③ 湍流 C_w 的 Δ 给点估计 + 95 % CI（3 档 k_s+） | **通过** | point estimate + 95 % CI for C_w at k_s^+ = 20/45/90; |delta| <= 1 % means uncontaminated, > 1 % mandates a re |
| ④ |Δ|>1 % 必须有重标表 | **通过** | if any |delta| > 1 % the report carries a correction factor per coefficient (fixed/defect); otherwise the tabl |
| ⑤ 全部重算走集群同一作业参数，JSON 带 sha256 | **部分通过**（作业已按同一模板递交但未被调度器派发） | every recomputed case ran in the same cluster job (one cmd template, fixed shard count), each record carries t |
| ⑥ 判别力负控：旧引擎重算必须报出 |Δ|>1 % | **通过** | recomputing with the defective (single-factor) engine must report |delta| > 1 %, otherwise the audit has no di |

### 与基准的误差（逐项，显式回答是否 <3 %）

| 量 | 基准 | 实测 | 误差 | 是否 <3 % |
|---|---|---|---|---|
| 层流锚（时间收敛设定） | Re_tau/3（解析） | 1.001339 | +0.134 % | 是 |
| 层流锚（文档设定 h=10） | Re_tau/3（解析） | 1.004152 | +0.415 % | 是 |
| F/2 签名（闭式） | 0.5 | 0.500000000 | 1.11e-16（绝对） | 是 |
| 发表 k0（λ/k_s=5） | 11.800 | 51.892 | +339.760 % | 否 |

说明：本审计的核心量是**相对 Δ**（缺陷 vs 修复），不是与实验基准的绝对误差；与解析基准可比的三项（层流锚、F/2 闭式、发表 k0 对账）在上表逐项给出。发表 k0 的 3 % 判据用的是同一 campaign 剖面重算，若超 3 % 说明该系数本身（独立于本缺陷）也不可复现，属另一条缺陷线索，一并老实报出。

### 审计链（sha256）

| 工件 | sha256 |
|---|---|
| 聚合 JSON | `d934554dc959121c591a532c850374eec0fa0d46234d4621ce3e9cb48000b7f5` |
| 聚合 JSON 字节数 | 84430 |
| git HEAD | `1dcbf558a820` |
| 重算执行位置 | local-fallback（控制面主机，集群作业未派发） |
| 已递交未派发的集群作业 | cj651 (campaign, 12 shards, 排队 >65 min 未派发); cj728 (audit2, 12 shards, 排队 >26 min 未派发) |
| 分片 / 线程 | 12 × 8 |
| 记录数 | 48 |
| 引擎 `fixed` | `f856eebf99921b34…` |
| 引擎 `defect_single_factor` | `e07dd17e32a7c67c…` |
| 引擎 `defect_window` | `8bb947f10ba7f24f…` |

复现：

```bash
python tools/f2_audit_v2.py provenance --out runs/f2_audit_v2
```
```bash
python tools/f2_audit_v2.py anchors --out runs/f2_audit_v2 --threads 4
```
```bash
run_cluster submit repo=TensorLBM shards=16 cmd_tpl="python tools/f2_audit_v2.py worker --shard {i}/{n} --threads 8 --job-id <jid>"
```
```bash
python tools/f2_audit_v2.py aggregate --out runs/f2_audit_v2 --job-id <jid> --shards 16
```
```bash
python tools/f2_audit_v2.py figures --out runs/f2_audit_v2
```
```bash
python tools/make_f2_v2_report.py --out runs/f2_audit_v2 --doc docs/F2_CONTAMINATION_AUDIT_V2.md
```

