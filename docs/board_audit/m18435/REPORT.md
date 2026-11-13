# 交差核验 m18327 · XVAL-LBM-SHAN-CHEN-MCMP-TAU-MIXED-VELOCITY：判级 **L2**（功能实现真、集成/基准未达）

审计单 m18435（verdict_by=board-audit）。对象 = m18327（`status=succeeded`，head_commit=null，
交付文件在工作树 `/root/tp-agent/work/m18327` **未提交**）。本单是交差核验，不重做。

> 一句话结论：**代码是真的、数是真的、可复现**（13 个单测绿；parity/momentum 重跑逐位一致；
> channel 8 例逐位复现），但**四臂预声明只交了三臂**（separation 臂无工件、5 档 G 只跑了 3 档、
> 1 档过 threshold）、**channel 最坏 12.34% 是生态 3% 线的 4.1 倍**、
> 且**没有任何集成调用方/从未合并**。故 L2，不认 L3/L4。

## ① 案例介绍（物理背景 / 基准 / 参数）

**物理问题**：两组分 Shan-Chen MCMP 碰撞里"不等 τ 时把两组分塌缩成一个共同速度"的做法。
OSS 参考 `loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow` 用
`vel_mix = Σ(Vel·Rho/τ)/Σ(Rho/τ)`（平台 `oss_solutions` id=35，evidence report 9555，
源码行 `Domain.py:151-153`，每组 τ 取自 `Domain.py:77-78`）。本单核验的是
TensorLBM `collide_sc_two_component(..., u_eq="viscous")` 是否施加同一公式、
该公式的连续极限混合物黏度 `ν_eff = Σρ_k ν_k/ρ` 是否被数值解复现。

**基准性质（关键局限）**：沙箱无 GitHub，**对标对象是被核验过的公式与源码行，而非 OSS 跑出来的数字**。
所以 parity 臂只能证"本仓施加的就是这个公式"（转写一致性），其独立数值含量来自
解析平面 Poiseuille（用同一 ν_eff 推导）——**推导在同一份交付里**，不是外部数据。

**参数（格子单位，predeclared.json 动手前写死）**：D2Q9 BGK，`dx=dt=1`、`ρ₀=1`、`c_s²=1/3`；
槽道 `ny×4`，`y=0` 与 `y=ny−1` 反弹壁、x 周期；`τ=(1.0,0.7)` 主例，
τ₂∈{1.00,0.90,0.70,0.60}、ny∈{17,33,65}、gx∈{5e-7,1e-6,2e-6}；
动量臂 τ=(1.0,0.7)/200 步/周期盒；分离臂 48×48/4000 步/G₁₂∈{−0.4…−2.0}。

## ② 怎么做的（审计方法/工具链）

1. **取证**：读 m18327 的 result JSON（`tensor_dev_missions.id=18327`）、工作树、
   `xval/…/{predeclared.json,run_xval.py,results/xval_results.json,figures/}`、`docs/…md`。
   工件快照复制进本单 `audit/m18327/` 并记 sha256（`artifacts/audit_m18327/sha256.txt`）。
2. **基准裁判**：先查 `tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-SHAN-CHEN-MCMP-TAU-MIXED-VELOCITY'` → **0 行**
   （交付时未登记，本单补登记，status=failed，见 ④）；再按 predeclared.json 的四臂判据逐条判。
3. **独立重跑**：在 m18327 工作树内 `PYTHONPATH=src python3 xval/…/run_xval.py
   --arm parity|momentum|channel --out <本单工件>`，与本单出工件逐数比对；
   `pytest tests/test_mcmp_tau_mixed_xval.py -q` → **13 passed in 7.51s**。
4. **声称 vs 实测**机器对照：`docs/board_audit/m18435/verify_claims.py` →
   `artifacts/audit_m18327/claim_vs_measured.json`（21/29 条重现）。
5. **落表**：`docs/board_audit/m18435/insert_verdict.py` 写 `tensor_maturity_verdicts`
   （level=2，键=impl_files/entry/run_artifacts/tests/integration_callers/benchmark/repro_cmd/external_anchor）。

## ③ 计算结果（程序化注入，零手抄）

**槽道臂（8 例，交付 vs 审计重跑）——判据 rel_err_max ≤15%、ν 误差 ≤15%、steady_resid ≤1e-4**

| case | ny | τ | gx | max% | L2% | ν 实测 | ν 误差% | 交付 rel_err_max | 审计重跑 | 交付自身子门 |
|---|---|---|---|---|---|---|---|---|---|---|
| primary | 33 | (1.00,0.70) | 1.0e-06 | 6.21 | 8.34 | 0.11666 | 0.01 | 0.062097 | 0.062097 | PASS |
| refine_17 | 17 | (1.00,0.70) | 1.0e-06 | 12.34 | 16.27 | 0.11659 | 0.06 | 0.123387 | 0.123387 | PASS |
| refine_65 | 65 | (1.00,0.70) | 1.0e-06 | 3.18 | 4.29 | 0.11675 | 0.07 | 0.031810 | 0.031810 | FAIL(sub-gate) |
| tau_equal_control | 33 | (1.00,1.00) | 1.0e-06 | 6.18 | 8.34 | 0.16667 | 0.00 | 0.061849 | 0.061849 | PASS |
| tau_ratio_1p11 | 33 | (1.00,0.90) | 1.0e-06 | 6.20 | 8.35 | 0.15000 | 0.00 | 0.062011 | 0.062011 | PASS |
| tau_ratio_3 | 33 | (1.00,0.60) | 1.0e-06 | 6.20 | 8.32 | 0.09999 | 0.01 | 0.061961 | 0.061961 | PASS |
| force_half | 33 | (1.00,0.70) | 5.0e-07 | 6.21 | 8.34 | 0.11666 | 0.01 | 0.062097 | 0.062097 | PASS |
| force_double | 33 | (1.00,0.70) | 2.0e-06 | 6.21 | 8.34 | 0.11666 | 0.01 | 0.062097 | 0.062097 | PASS |

- 最坏 max 误差 **12.34%**（ny=17），主例 6.21%，ny=65 3.18%
  → 15% owner 线内，但**生态 3% 线外**（主例 2.1×、最坏 4.1×）。
- ν 实测误差全部 ≤0.07% ⇒ 黏度混合权重（ρ 加权算术平均）证实。
- **子门漏报**：`refine_65` 的 steady_resid = 1.142e-04 > predeclared 1e-4
  ⇒ 该例自判 FAIL；但驱动 `arm_channel` 的臂级判定只取 `rows[0]['pass']`，
  故 results JSON 里 channel 臂 = PASS、`overall` = PASS（doc 表格里倒如实写了 FAIL）。

**parity 臂（黑盒探针，5 前碰撞态 × 2 dtype）**

| 指标 | 交付 | 审计重跑 | 判据 |
|---|---|---|---|
| worst rel dev float64 | 0.000 | 0.000 | ≤1e-10 ✔ |
| worst rel dev float32 | 0.000 | 0.000 | ≤1e-4 ✔ |

探针由 `ΔJ_k = −(J_k − ρ_k u_mix)/τ_k` 反解库内真正用的 u_mix——比"两个自写函数互相对"强一档，
但仍只证"公式同一"，不证"OSS 数字也这样"。

**momentum 臂（200 步，周期盒，判据 ≤1e-12）**

| u_eq | 交付 rel_drift | 审计重跑 | 判定 |
|---|---|---|---|
| viscous | 0.0000 | 0.0000 | PASS |
| mixture | 0.5294 | 0.5294 | FAIL |
| self | 0.0000 | 0.0000 | PASS |

负控（`mixture` 模式不守恒 0.529）⇒ 该臂有判别力，不是"太宽松才过"。

**separation 臂：无工件（本单核心发现）**。`xval_results.json` 无 `separation` 键，
全工作树无分离结果 JSON；交付 doc §4 只以正文列出 control 3.16e-07 / G=−0.4 7.666e-06 /
G=−0.8 1.911e-04 / G=−1.2 1.571e+00，**G=−1.6/−2.0 从未执行**。
按预声明（control ≤1e-3、SC 臂 contrast ≥0.5、G 扫 5 档）⇒ 该臂 **未交差**；
且 JSON `overall='PASS'` 是对"存在的那几臂"求 all()，三臂交付也印 PASS（F2）。

## ④ 结果图

![audit summary](artifacts/audit_m18327/figures/audit_m18327_summary.png)

(A) 槽道臂逐例 max/L2 误差（交付 vs 审计重跑）对 15% 判据与 3% 生态线；
`refine_65` 的 steady_resid 子门 miss 已标注。(B) 主例剖面：交付 vs 解析 ν_eff 解 vs 审计重跑
（误差集中在贴壁第一层流体格点，ν 本身只差 0.01%）。
审计另核交付三张图（channel_profiles 1650×630、convergence_viscosity 1500×600、momentum 780×540，均有效 PNG）。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **臂级判定只取 rows[0]**（`run_xval.py` `arm_channel`: `"verdict": "PASS" if rows[0]["pass"]`）：
   子门 miss 不上浮 → "channel PASS" 这句话在臂级/总级都是**被放宽的**。
2. **`overall` 是"存在臂的 all()"**：声明 4 臂、实交 3 臂仍印 PASS。报告/JSON 的 PASS 不能当"四臂全过"读。
3. **外部锚是公式级**：parity 无法排除"两边同一误读"；唯一有独立数值含量的是解析 ν_eff 槽道，
   而该解析式由同一交付推导，强度取决于推导本身（本单复核：动量守恒⇒τ 加权 ⇒ Π^neq 求和 ⇒ ρ 加权算术 ν，链条自洽）。
4. **12.34% 的来源是壁面约定**（半格 vs 格心）而不是黏度：ν 误差 ≤0.08%，壁位置拟合 0.504/31.496 = 格心约定；
   换约定会伪造 6–12% 假误差——引用该数字必须带上约定。
5. **未合并**：无 commit、无分支、`head_commit=null`，`git log --all` 无该文件 → 下游引用不了（故无 L3）。
6. **交付未发平台报告**（`tensor_reports` mission_id=18327 → 0 行），也未登记基准案例行。

## ⑥ 误差判定（逐项 vs 基准，是否 <3%）

| 关键量 | 基准 | 实测 | 误差 | <15% owner | <3% 生态 |
|---|---|---|---|---|---|
| parity 混合速度 rel dev (f64) | REF-OSS 公式 `Σ(Vel·Rho/τ)/Σ(Rho/τ)` | 0.000 | 舍入级 | ✔ | ✔ |
| parity 混合速度 rel dev (f32) | 同上 | 0.000 | 0.00% | ✔ | ✔ |
| 混合黏度 ν（8 例） | ν_eff=Σρ_kν_k/ρ | 见 ③ 表 | ≤0.07% | ✔ | ✔ |
| 剖面 u_x 主例 ny=33 | 解析 Poiseuille(ν_eff) | max 6.21% | 6.21% | ✔ | **✘** |
| 剖面 u_x 最坏 ny=17 | 同上 | max 12.34% | 12.34% | ✔ | **✘** |
| 动量守恒 viscous(200 步) | 精确守恒 | 0.000 | 机器精度 | ✔ | ✔ |
| 相分离 contrast（5 档 G） | control ≤1e-3 / SC ≥0.5 | **无工件**（正文 3 档 / 5 档） | **未判定** | **✘ 未交差** | ✘ |

**是否 <3%：关键几何量（黏度、动量、parity）<3%，关键场量（剖面 u_x）不满足（6.21% / 12.34%），
相分离臂无法判定（工件缺失）。** honest-FAIL 部分不粉饰：4 臂预声明 gate 整体未过。

## ⑦ 声称 vs 实测（机器对照 21/29 条重现）

| 项 | 重现 | 实测 |
|---|---|---|
| predeclared arms present in delivered JSON | ✘ | 3/4 present: parity,channel,momentum; missing=separation |
| parity worst rel dev f64 | ✔ | 6.890896e-15 (auditor re-run) |
| parity worst rel dev f32 | ✔ | 3.353399e-06 (auditor re-run) |
| parity vs predeclared threshold f64 | ✔ | 6.891e-15 -> PASS |
| parity vs predeclared threshold f32 | ✔ | 3.353e-06 -> PASS |
| momentum rel drift u_eq=viscous | ✔ | 2.3072e-14 (auditor re-run) |
| momentum rel drift u_eq=mixture | ✔ | 5.2941e-01 (auditor re-run) |
| momentum rel drift u_eq=self | ✔ | 5.2042e-16 (auditor re-run) |
| momentum: viscous arm vs predeclared 1e-12 | ✔ | 2.307e-14 -> PASS |
| momentum: negative control u_eq=mixture must NOT conserve | ✔ | 5.294e-01 (arm has discriminating power) |
| channel primary rel_err_max | ✔ | 6.2097% (auditor re-run) |
| channel refine_17 rel_err_max | ✔ | 12.3387% (auditor re-run) |
| channel refine_65 rel_err_max | ✔ | 3.1810% (auditor re-run) |
| channel tau_equal_control rel_err_max | ✔ | 6.1849% (auditor re-run) |
| channel tau_ratio_1p11 rel_err_max | ✔ | 6.2011% (auditor re-run) |
| channel tau_ratio_3 rel_err_max | ✔ | 6.1961% (auditor re-run) |
| channel force_half rel_err_max | ✔ | 6.2097% (auditor re-run) |
| channel force_double rel_err_max | ✔ | 6.2097% (auditor re-run) |
| channel worst rel_err_max vs predeclared 15% | ✔ | 12.34% -> PASS |
| channel worst rel_err_max vs ecosystem 3% bar | ✘ | 12.34% -> FAIL (5.7x over at primary, 4.1x worst) |
| channel per-case sub-gates (rel+nu+steady_resid) | ✘ | cases failing their own predeclared sub-gate: ['refine_65'] (refine_65 steady_resid=1.142e-04 > 1e-4) |
| doc table primary max% | ✔ | artifact gives 6.21% |
| doc table refine_17 max% | ✔ | artifact gives 12.34% |
| doc table refine_65 max% | ✔ | artifact gives 3.18% |
| doc §4 arm-4 (separation) numbers have an artifact | ✘ | no separation key in xval_results.json and no separation JSON anywhere in the worktree -> those four numbers are transcribed from stdout only |
| delivered code is in the repo | ✘ | worktree /root/tp-agent/work/m18327 git_status shows '?? src/tensorlbm/mcmp_tau_mixed.py'; git log --all in the ecosystem checkout: no commit touches it; mission result head_commit=null |
| module exported / called by the library | ✘ | grep: only tests/test_mcmp_tau_mixed_xval.py imports tensorlbm.mcmp_tau_mixed; __init__.py untouched, no solver path references it |
| platform report published for m18327 | ✘ | SELECT from tensor_reports WHERE mission_id=18327 -> 0 rows |
| benchmark case registered | ✘ | SELECT ... WHERE anchor_slug='XVAL-LBM-SHAN-CHEN-MCMP-TAU-MIXED-VELOCITY' -> 0 rows (registered by this audit) |

## ⑧ 判级与落表

- **L2**（独立功能实现：真模块+入口+运行工件+13 绿单测）。不认 L3（`integration_callers` 空：未导出、
  无调用方、从未合并）、不认 L4（无外部数值锚；剖面误差 6.21%/12.34% > 3%；
  四臂之一无工件）。百页报告不改变级别，级别=功能实质。
- 落表：`tensor_maturity_verdicts (repo='TensorLBM', scope='m18327 XVAL-LBM-SHAN-CHEN-MCMP-TAU-MIXED-VELOCITY', mission_id=18435,
  level=2, verdict_by='board-audit')`；补登记基准案例行（交付时缺，status=failed，
  逐臂判据/实测写进 metric）。maturity_gate 机器复核后 confirmed=消账 / reopened=返修。
- 返修建议（给 reopened 轮用）：①`arm_*` 判定改 `all(rows)`；②补跑 separation 臂并把 JSON 落盘；
  ③把 `mcmp_tau_mixed` 导出进 `__init__` 并让求解器/平台侧真调用（补 L3）；
  ④若要 L4，需外部数值锚或把剖面误差压到 3% 以下（壁面约定 + 一阶收敛要交代）。
