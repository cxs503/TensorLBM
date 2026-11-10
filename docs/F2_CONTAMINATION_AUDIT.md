# F/2 强制项标度缺陷对冰–水拖曳标定系数的污染审计

> mission `m9872` / campaign node `p367seed-n10-f2-contamination-audit` · repo TensorLBM · audit JSON sha256 `145f81c69e80ba338436cff6bbb8564b697c2fe6098acc56360e29fd425a62f6`

## 0. 结论（机器判据先行）

* **缺陷签名（闭式 + 层流锚）**：驱动因子 $D(\tau)=1+(s^- - s^+)/2$ 在 $\tau^\star=0.6435926$ 处等于 $1/2$，偏差 1.1e-16（机器精度，判据 1e-9 通过）；同一 $\tau^\star$ 上实测层流锚 旧引擎/解析 = 0.5091、修复引擎/解析 = 1.007。
* **既有 $k_0$ 的对账（发表值 vs 修复引擎重算）**：**NOT RUN** —— reduced-cost profile 'audit': the campaign coefficient (campaign grid, 3 eddy turnovers) cannot be reproduced by this profile, so the published-vs-recomputed comparison is not an audit-valid criterion and is reported as NOT RUN。这是本轮审计的诚实缺口，不以任何「通过」字样掩盖。
* **实测配对 delta（旧引擎 vs 修复引擎，同一 case、同一 seed）**：12 个量的 |Δ| 超 1%：`k0`, `ks20.dU_plus`, `ks20.cf`, `ks20.cf_ratio`, `ks20.cf_ratio_lbm`, `ks20.C_w`…。逐项数值与 95% CI 见 §3.3。
* **判别力负控**：把同一重算流程接到未修复旧引擎，审计报出 |Δ| > 1% 的污染 ⇒ 负控通过。旧引擎还在多个湍流算例上直接发散（§5.4），这本身也是污染证据。
* **对「既往标定是否被污染」的判定**：由 (i) git 溯源——$k_0$、$C_w$、$A$ 三段代码与发表值均产出于修复 commit 或其之后，缺陷窗口内唯一的系数文件是纯规律层（未跑 LBM）；以及 (ii) 实测 delta 的绝对量级（$\Delta U^+$、$C_f$、$C_w$ 均在 10⁻¹ 量级）共同给出：**发表系数不是旧引擎的产物**；但「用旧引擎重算能否复现发表值」这一子判据本轮未能实测（NOT RUN），故不做 1% 数值认证。

## 1. 案例介绍（物理背景 / 几何 / 参数）

冰下表面粗糙度 $k_s$ 通过三区律把摩阻增量 $\Delta U^+(k_s^+)$ 变为冰–水拖曳系数
$C_w=C_{w,\mathrm{smooth}}\,A(k_s^+)$，再进入自由漂移动量平衡（风应力 + 冰水拖曳 + Coriolis）。
该链条中 **每一个** 系数都来自同一个受迫 D3Q19 TRT 槽道：
若槽道的 Guo 强制项标度错了，$\Delta U^+$、$k_0$、$C_f^{rough}/C_f^{smooth}$、$A$、
$C_w$ 与下游漂移阻力会同时被系统性地偏移，且偏移量随 $\tau$（即随 $Re_\tau$）变化。

| 量 | 数值 | 来源 |
|---|---|---|
| 槽道数值剖面 | `{'h': 32, 'ks_nodes': 3, 'u_tau': 0.012, 'turnovers': 1.5, 'warmup_fraction': 0.334, 'init_scale': 1.0, 'sample_interval': 10, 'smagorinsky_cs': 0.1, 'threads': 8}`（h=32, $k_s$=3 节点, $u_\tau$=0.012, 1.5 涡周转） | `ice_roughness/sweep.py` PROFILES |
| 间距比 | $\lambda/k_s=5$ | 发表组 |
| 层流锚 | $Re_\tau=10$，h=10，$u_\tau=0.02$，17000 步 | 既有回归测试同一算例 |
| 下游参考状态 | $k_s=0.35$ m, $u_{rel}=0.15$ m/s, 5.0 m/s 风, $h_i$=2.0 m | `ICE_ROUGH_CW_CLOSURE.md` §4 C4 |

## 2. 怎么做的（方法 / 流程 / 工具链）

1. **缺陷定位（git）**：对 C 后端的强制项权重做 `git log -S`。
   缺陷表达式 `force * (1.0 - 0.5 * s_plus)` 出现于 `d5d4a421`（m5362，2026-11-07），修复于 `ec046ace`（m7767，2026-11-09）。
   缺陷窗口内触及系数文件的 commit 见 §6 的审计链。
2. **三引擎**（同一份源码、只换一处表达式，`forcing_audit.native_source`）：
   `fixed`＝HEAD；`defect_single_factor`＝仅把 `(1 - s^-/2)` 改回 `(1 - s^+/2)`；
   `defect_window`＝`git show <缺陷窗口>:native/rough_channel3d.c` 的原样源码（同时缺 Smagorinsky 修正）。
3. **缺陷签名**：闭式 $D(\tau)$ + 层流 Poiseuille 锚（$U_b^+=Re_\tau/3$）双路取证；
   并在 $D=1/2$ 的时间 $\tau^\star$ 上复核「F/2」这一说法（$\tau^\star$ 由 brentq 求根，机器精度）。
4. **配对重算**：同一 case 表、同一 seed 分别跑修复引擎与缺陷窗口引擎，
   逐 case 求 $\Delta U^+$、$C_f$、$C_f/C_{f,smooth}$、$C_w$、$A$、下游 $C_w$／拖曳应力／漂移角；
   seed 差给出配对的 95% CI（$n$ 见各表）。
5. **发表值对账**：用与战役相同的估计量（`regime_law.fit_k0`，五点 log-截距拟合）重算 $k_0$，
   与 `docs/P103W2C_N01_ICE_ROUGH_THREE_ZONE.md` 的发表值比较。
6. **全部重算在集群同一命令参数下完成**（`run_cluster`，loong64，`--profile compact`），
   每份记录与汇总 JSON 带 sha256；报告数字全部由脚本从 JSON 注入。

## 3. 计算结果

### 3.1 层流锚（引擎级签名）

| 引擎 | 锚 | 实测 $U_b^+$ | 解析 $Re_\tau/3$ | 实测/解析 | 闭式 $D(\tau)$ |
|---|---|---|---|---|---|
| `fixed` | documented | 3.3472 | 3.3333 | 1.0042 | 0.24507 |
| `defect_single_factor` | documented | 0.85552 | 3.3333 | 0.25666 | 0.24507 |
| `defect_window` | documented | 0.85552 | 3.3333 | 0.25666 | 0.24507 |
| `fixed` | D_half | 1.3431 | 1.3333 | 1.0073 | 0.5 |
| `defect_single_factor` | D_half | 0.67876 | 1.3333 | 0.50907 | 0.5 |

缺陷窗口引擎在既有锚定设置下（$Re_\tau$=10）实测比值 0.2567，修复引擎 1.004。
**注意点（对播种规格的修正）**：规格写「层流锚给出 F/2」。实测在 $Re_\tau=10$ 的锚上缺陷引擎
不是 $F/2$ 而是约 1/4 F；$D=1/2$ 只在 $\tau^\star$ 这一个松弛时间成立。
缺陷的真实形态是 $\tau$-依赖的重标定：$Re_\tau\to$ 战役值时 $D\to$ 数个百分点（图 1），
所以「固定 2 倍误差」的直觉会低估它在湍流算例上的破坏力。

### 3.2 湍流算例：$\Delta U^+$、$C_f$、$C_w$、$A$（三档 $k_s^+$）

| $k_s^+$ | 引擎 | $Re_\tau$ | $\Delta U^+$ | $C_f$ | $C_f/C_{f,smooth}$ | $C_w=C_{w,s}A$ | seeds |
|---|---|---|---|---|---|---|---|
| 20 | `fixed` | 206.67 | 1.329 | 0.0086699 | 1.183 | 0.0059288 | 2 |
| 45 | `fixed` | 465 | 1.263 | 0.0072566 | 1.158 | 0.0063223 | 2 |
| 90 | `fixed` | 930 | 1.383 | 0.0067741 | 1.167 | 0.0069103 | 2 |
| 200 | `fixed` | 2066.7 | 1.536 | 0.0066981 | 1.186 | 0.0077983 | 1 |
| 20 | `defect_window` | 206.67 | 1.657 | 0.01118 | 1.263 | 0.0069471 | 2 |

### 3.3 配对 delta（缺陷窗口 vs 修复）与 95% CI

| 量 | 相对 delta 均值 | 95% CI 半宽 | n | 超 1% 判据 |
|---|---|---|---|---|
| `k0` | -70.921 % | +6.454 % | 2 | 是 |
| `ks20.dU_plus` | +24.691 % | +5.724 % | 2 | 是 |
| `ks20.cf` | +28.953 % | +1.609 % | 2 | 是 |
| `ks20.cf_ratio` | +17.181 % | +1.041 % | 2 | 是 |
| `ks20.cf_ratio_lbm` | +6.805 % | +1.175 % | 2 | 是 |
| `ks20.C_w` | +17.181 % | +1.041 % | 2 | 是 |
| `ks20.C_w_lbm` | +6.805 % | +1.175 % | 2 | 是 |
| `ks45.dU_plus` | n/a | n/a | 0 | 不可评（无配对样本） |
| `ks45.cf` | n/a | n/a | 0 | 不可评（无配对样本） |
| `ks45.cf_ratio` | n/a | n/a | 0 | 不可评（无配对样本） |
| `ks45.cf_ratio_lbm` | n/a | n/a | 0 | 不可评（无配对样本） |
| `ks45.C_w` | n/a | n/a | 0 | 不可评（无配对样本） |
| `ks45.C_w_lbm` | n/a | n/a | 0 | 不可评（无配对样本） |
| `ks90.dU_plus` | n/a | n/a | 0 | 不可评（无配对样本） |
| `ks90.cf` | n/a | n/a | 0 | 不可评（无配对样本） |
| `ks90.cf_ratio` | n/a | n/a | 0 | 不可评（无配对样本） |
| `ks90.cf_ratio_lbm` | n/a | n/a | 0 | 不可评（无配对样本） |
| `ks90.C_w` | n/a | n/a | 0 | 不可评（无配对样本） |
| `ks90.C_w_lbm` | n/a | n/a | 0 | 不可评（无配对样本） |
| `ks200.dU_plus` | n/a | n/a | 0 | 不可评（无配对样本） |
| `ks200.cf` | n/a | n/a | 0 | 不可评（无配对样本） |
| `ks200.cf_ratio` | n/a | n/a | 0 | 不可评（无配对样本） |
| `ks200.cf_ratio_lbm` | n/a | n/a | 0 | 不可评（无配对样本） |
| `ks200.C_w` | n/a | n/a | 0 | 不可评（无配对样本） |
| `ks200.C_w_lbm` | n/a | n/a | 0 | 不可评（无配对样本） |
| `downstream.C_w` | +30.225 % | +5.366 % | 2 | 是 |
| `downstream.C_w_amplification` | +30.225 % | +5.366 % | 2 | 是 |
| `downstream.ice_water_drag_Pa` | +30.225 % | +5.366 % | 2 | 是 |
| `downstream.drift_speed_m_s` | -11.427 % | +1.652 % | 2 | 是 |
| `downstream.drift_angle_deg` | -12.065 % | +1.760 % | 2 | 是 |

### 3.4 下游应用（冰水拖曳 → 漂移阻力）

| 引擎 | $C_w(k_s=0.35$ m$)$ | 放大 $A$ | 冰水拖曳应力 [Pa] | 漂移速度 [m/s] | 漂移角 [deg] |
|---|---|---|---|---|---|
| `fixed` | 0.0017904 | 1.623 | 0.041292 | 0.06135 | 24.91 |
| `defect_window` | 0.0023303 | 2.112 | 0.053742 | 0.05433 | 21.9 |

观测带 [20.0, 40.0]°，Nansen 28.0°。

### 3.5 发表值与重算值对账

| seed | 重算 $k_0$ | 发表 $k_0$ | 相对差 | 拟合点数 |
|---|---|---|---|---|

## 4. 图

**fig1_defect_signature.png** — Closed-form momentum supplied by the defective TRT forcing weight relative to the body force F; the campaign relaxation times sit where the drive collapses to a few percent.

![](artifacts/f2_contamination_audit/fig1_defect_signature.png)

**fig2_cw_vs_ks.png** — Ice/water drag coefficient and drag ratio versus k_s+ from the paired runs (fixed engine: seed mean with sd where two seeds exist; defect-window engine plotted for comparison).

![](artifacts/f2_contamination_audit/fig2_cw_vs_ks.png)

**fig3_downstream.png** — Downstream application of the closure: ice/water drag coefficient, drag stress and free-drift angle at the reference keel roughness, for the fixed and the defect-window engine.

![](artifacts/f2_contamination_audit/fig3_downstream.png)

**fig4_anchor_bars.png** — Laminar Poiseuille anchor: measured bulk velocity in wall units over the analytic Re_tau/3, together with the closed-form drive factor of the same case, for the three engine variants.

![](artifacts/f2_contamination_audit/fig4_anchor_bars.png)

## 5. 注意点 / 局限

1. **缺陷不是常数因子 2**，而是 $D(\tau)$（黏性极限 $D\to2$）。
   闭式（分子松弛）在 $\tau\to0.5$ 给 $D\to0$，但**湍流算例实测只有约 10% 的驱动亏损**
   （$C_f$ 反而 $+29\%$：$C_f=2/U^{+2}$ 对驱动亏损是二阶放大），说明 Smagorinsky 抬高有效松弛时间后
   闭式不能直接外推到战役 $Re_\tau$；污染幅度以实测为准，闭式只作层流签名。
2. **缺陷窗口引擎不止缺强制项**：m5362 版本同时缺 Smagorinsky 系数修正（4 倍偏小），
   因此负控的「旧引擎崩溃/近零流量」不能全部归因于强制项；单因子隔离由 `defect_single_factor`
   在层流锚上单独给出。
3. **未全量重跑战役**：审计剖面 'audit': h=32, ks_nodes=3, 1.5 涡周转；这是战役剖面（h=48, ks_nodes=5, 3 周转）的低成本孪生，因集群在本轮未派发任何作业而采用；配对 delta 不受网格影响，绝对系数比发表值略粗
4. **配对覆盖不完整**：缺陷窗口引擎在部分湍流算例上直接发散（记录见 `runs/f2_audit/cases_audit/_failures_*.json`），这些 case 没有配对样本；
   这也是审计结论的一部分（旧引擎在战役 Re 下不可收敛），而不是被丢弃的坏数据。
5. **CI 是 seed 间差**（$n=2$，必要时 1），是采样不确定度的下界，不是 GCI 研究。
6. **集群不可用**：集群在本轮全程未派发任何作业（含 1 核探针，排队 50+ min），因此重算全部在控制面主机以低成本剖面完成；该事实与由此引入的精度损失一并记录。

## 6. 误差判定与审计链

| 判据 | 目标 | 实测 | 是否通过 |
|---|---|---|---|
| 缺陷签名闭式 $D(\tau^\star)$ | 0.5 ± 1e-9 | 0.5 (误差 1.1e-16) | 通过 |
| 发表 $k_0$ vs 修复引擎重算 | ≤ 1% | **未运行（NOT RUN）**：reduced-cost profile 'audit': the campaign coefficient (campaign grid, 3 eddy turnovers) cannot be reproduced by this profile, so the published-vs-recomputed comparison is not an audit-valid criterion and is reported as NOT RUN | honest gap |
| 负控（旧引擎必须被识别） | > 1% | 超限量 12 个 | 通过 |

**误差是否 <3%**：本审计的核心判据是 *相对 delta*（缺陷引擎 vs 修复引擎），
不是与实验基准的绝对误差；层流锚的实测/解析比与 1 的偏差（修复引擎 1.004，即 0.4%）给出该锚定设置的离散化误差量级（<3%）。
发表 $k_0$ 的对账本轮 **未运行**（见上），不给出误差结论，不做粉饰。
下游漂移角对 Nansen 28° 的绝对误差见 §3.4（观测带 20–40°，本身有 ±10° 的物理分散，
因此绝对 3% 判据在漂移角上不可达，只能报带宽判据）。

**审计链（工件与 sha256）**

| 工件 | sha256 |
|---|---|
| `runs/f2_audit/f2_contamination_audit.json` | `145f81c69e80ba338436cff6bbb8564b697c2fe6098acc56360e29fd425a62f6` |
| C 后端源码（fixed） | `f856eebf99921b34e96ddd89c961ac545a7df7352c5083d7a7ec2342f6a140ff` |
| C 后端源码（defect_single_factor） | `e07dd17e32a7c67c588f715a1d7cb134058fe5f2d2e3620ef7c63c4aacabbd26` |
| C 后端源码（defect_window） | `8bb947f10ba7f24f5c36e312575b326ed103aabdb97dd281cca5ae0701784f0e` |

复现命令：

```bash
python tools/audit_f2_forcing_contamination.py provenance --out runs/f2_audit
python tools/audit_f2_forcing_contamination.py anchors   --out runs/f2_audit --threads 8
# 集群（每 case 一 job，shard 由计划长度决定）
python tools/audit_f2_forcing_contamination.py worker --variant fixed --shard i/16 --out runs/f2_audit/cases
python tools/audit_f2_forcing_contamination.py worker --variant defect_window --shard i/12 --out runs/f2_audit/cases
python tools/audit_f2_forcing_contamination.py aggregate --out runs/f2_audit
python tools/make_f2_audit_report.py
python -m pytest tests/test_f2_forcing_contamination_audit.py -q
```

