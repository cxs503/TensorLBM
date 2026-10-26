# 毛细侵入 capillary_invasion_washburn（Washburn 律 β 与 L(t)）（终裁）

历史完整报告：[归档报告](../../../docs/reports/report_capillary_invasion_washburn.md)。2026-10-26 仅整理文件与链接，未重新验证或变更本案例状态。

**状态：🛑 终裁 TERMINAL（m195 闭环，2026-10-23）——SC-MCMP 结构性缺陷：β 判据 19.08%（W32）/3.17%（W64）；L(t) 绝对判据 689%/949%（b_max_rel_diff 6.894/9.492，档间反收敛）；湿润相质量漂移 -60.7%/-67.0%；W128 确定性 NaN@164 步；σ 原理性不可测。根因：多组分伪势耦合的结构性搅动，需库级重构方可修复。**

## 物理问题

两相 SC-MCMP 流体在压差 dP 驱动下侵入宽度 W 的细管（毛细侵入/Washburn）：

- 判据 a（指数）：侵入前沿 L(t) 的 Washburn 指数 β（理论 0.5）偏差 ≤3%；
- 判据 b（绝对）：L(t) 对 ODE 参考解的最大相对偏差 ≤3%。参考 ODE 用
  nominal-μ（**无毛细项**——SC-MCMP 的 σ 无可测量路径，
  `b_reference = 'nominal-mu ODE (no capillary; sigma unmeasurable)'`）；
- 预注册冻结（snapshot/notes sha256 双锁，iron_rule pass），
  判据、窗口协议（thr1 阈值触发 t1，跨 t2 关窗）全部预注册。

模型：库 SC-MCMP 多相原语（D2Q9 多组分伪势），drho_rule_check 通过；
W ∈ {32, 64, 128} 三档，40k 步预算。

## 计算结果现状（result.json `part2`，2026-09-21）

| 档 | 状态 | steps | β | β 判据偏差% | b_max_rel_diff | 湿润相质量漂移% | 判定 |
|----|------|-------|---|-------------|----------------|------------------|------|
| W32 | ok | 40000 | 0.5954203105406733 | **19.08406210813467** | **6.894145455199683** | **-60.677107119080844** | A ✗ B ✗ |
| W64 | ok | 40000 | 0.515863171986898 | **3.1726343973796034** | **9.491756800333688** | **-66.97527211128853** | A ✗ B ✗ |
| W128 | DIVERGED | 100 | — | — | — | — | NaN@164（确定性） |

窗口健康度（W32：t∈[1500,12600]，112 样本，u_max=0.074523，
Ma=0.12907762233245546，minf=0.000621；W64：t∈[1900,6900]，51 样本，
u_max=0.072953，Ma=0.1263583025645723）——拟合窗本身健康，
失败不是窗口选择问题。β 的 ODE 自参对照：W32 beta_ode=0.19952327471976875
（偏离 sim β 3 倍），W64 beta_ode=0.153640644256196——ODE 参考自身与
0.5 差距巨大，再次说明 σ 缺失下参考解不具判别力（fit_rms_rel 0.148/0.364）。

W128 发散归因（result.json `attribution`）：确定性 NaN@164，与驱动无关
（drho=0 复现）与域长无关（nx=300 复现）；水汇处气相耗尽层 → rho_g→0 →
分布函数变负（exp39/40/41）。

**姊妹案例（同库同预注册）**：`../two_phase_poiseuille/result.json`
（W4-B Part 1）：H64 max_rel_err=0.26821538643505377 → H128
0.33173655541170183（**反收敛**），criterion_monotone_decreasing=False，
verdict FAIL。

## 不达标清单（严格标准：共性模块入口 + 直接观测量 ≤3% + ≥2 档网格单调收敛）

- 共性模块入口：✓（SC-MCMP 库原语；prereg sha256 冻结，iron_rule 通过）。
- 直接观测量 ≤3%：✗（β 19.08%/3.17%；L(t) 绝对判据 689%/949%）。
- ≥2 档网格单调收敛：✗（W64 比 W32 更差：b_max_rel_diff 6.894→9.492，
  质量漂移 -60.7→-67.0；W128 直接发散；姊妹 Part 1 同样反收敛）。
- 附：σ 不可测 → 判据 b 的参考解退化为无毛细 ODE，判别力受损（如实披露）。

## 根因/诊断

1. **SC-MCMP 结构性耦合偏差**：多组分伪势的组分间动量耦合在侵入前沿
   附近产生持续搅动——湿润相质量在 40k 步内漂移 -60~-67%（质量账严重
   不守恒），前沿动力学偏离 Washburn 自相似律（β 偏离 0.5 且档间不稳）。
2. **σ 无测量路径**：表面张力在 SC-MCMP 参数化下原理性不可测，ODE 参考
   只能去毛细项（nominal-μ），绝对判据的 +689/+949% 部分来自参考本身
   缺项——这是"参考受限"型失败与"物理错误"型失败的叠加，判决如实双 FAIL。
3. **W128 负分布 NaN**：水汇气相耗尽 → rho_g→0 → 分布负值（确定性、
   与驱动/域长无关）——密度比下限路径的库级缺陷。
4. 战役侧记录的库缺陷清单（W4-B 报告）：SC-MCMP 库 validate 逻辑把唯一
   可用域拦死、washburn 指数器口径（返回 2β）、0.05 台阶初始化等 10 项
   已登记——本案例的搅动与质量漂移与该清单同源。

## 晋级路径

- 先修库再重验：SC-MCMP 湿润相质量守恒（侵入前沿质量账）、可测 σ 路径
  （或改用可测 σ 的自由能 MultiRelaxation 类多相）、气相耗尽层防护
  （W128 类负分布）。
- 重验须新预注册（判据 b 的参考解在 σ 可测后重锁）；σ 不可测的现状下
  该案例无可判别参考，不宜复跑。
- 达标即归档 PR 移 `pending/capillary_invasion_washburn` → `verified/`
  （建议与 Part 1 two_phase_poiseuille 同批处置，同库同根因）。

## 复现

```
cd benchmarks/pending/capillary_invasion_washburn
PYTHONPATH=../../../src python run.py --W {32,64,128}
# 判决: result.json part2.criteria（a/b 判据 + W128 状态）
# 预注册哈希: prereg.snapshot_sha256 / prereg.notes_sha256（见 result.json 头部）
```

<!-- PROVENANCE
[{"v": true, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "iron_rule.pass"},
 {"v": 40000, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W32.completed_steps"},
 {"v": 1500, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W32.window.t1"},
 {"v": 12600, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W32.window.t2"},
 {"v": 112, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W32.window.n_samples"},
 {"v": 0.5954203105406733, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W32.beta"},
 {"v": 19.08406210813467, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W32.beta_criterion_dev_pct"},
 {"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W32.criterion_a_pass"},
 {"v": 0.19952327471976875, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W32.beta_ode"},
 {"v": 6.894145455199683, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W32.b_max_rel_diff"},
 {"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W32.criterion_b_pass"},
 {"v": -60.677107119080844, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W32.massw_drift_pct"},
 {"v": 0.074523, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W32.umax_in_window"},
 {"v": 0.12907762233245546, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W32.Ma_in_window"},
 {"v": 0.000621, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W32.minf_overall"},
 {"v": 0.14812899907521598, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W32.fit_rms_rel"},
 {"v": 0.03018018018018018, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W32.front_speed_mean_in_window"},
 {"v": 1900, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W64.window.t1"},
 {"v": 6900, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W64.window.t2"},
 {"v": 51, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W64.window.n_samples"},
 {"v": 0.515863171986898, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W64.beta"},
 {"v": 3.1726343973796034, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W64.beta_criterion_dev_pct"},
 {"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W64.criterion_a_pass"},
 {"v": 0.153640644256196, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W64.beta_ode"},
 {"v": 9.491756800333688, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W64.b_max_rel_diff"},
 {"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W64.criterion_b_pass"},
 {"v": -66.97527211128853, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W64.massw_drift_pct"},
 {"v": 0.072953, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W64.umax_in_window"},
 {"v": 0.1263583025645723, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W64.Ma_in_window"},
 {"v": 0.3644854864941034, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W64.fit_rms_rel"},
 {"v": 100, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.W128.completed_steps"},
 {"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.criteria.a_beta_vs_0.5_le_3pct.W32"},
 {"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.criteria.a_beta_vs_0.5_le_3pct.W64"},
 {"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.criteria.b_L_vs_ODE_le_3pct.W32"},
 {"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json", "k": "part2.criteria.b_L_vs_ODE_le_3pct.W64"},
 {"v": 0.26821538643505377, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/result.json", "k": "part1.H64.max_rel_err"},
 {"v": 0.33173655541170183, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/result.json", "k": "part1.H128.max_rel_err"},
 {"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/result.json", "k": "part1.criteria.criterion_monotone_decreasing"}]
-->

## 终裁裁定（m195，2026-10-23）

**verdict = terminal**

根因证据链：
1. **质量守恒失败**：湿润相质量漂移 -60.7%（W32）/ -67.0%（W64），远超 1% 可接受范围
2. **σ 原理性不可测**：SC-MCMP 伪势交互产生排斥力驱动相分离，但界面力为数值构造，不与物理表面张力直接对应
3. **负分布 NaN**：W128 在 164 步确定性发散，气相耗尽层导致分布函数变负
4. **档间反收敛**：b_max_rel_diff 从 W32 的 6.894 恶化到 W64 的 9.492，违反单调收敛要求

四门判决：
- G1 共性模块入口：✓（from tensorlbm.multiphase import collide_sc_two_component）
- G2 直接观测量 ≤3%：✗（β 偏差 19.08%/3.17%，b_max_rel_diff 689%/949%）
- G3 ≥2 档单调收敛：✗（W64 比 W32 更差，反收敛）
- G4 预注册冻结：✓（snapshot_sha256 + notes_sha256 双锁，iron_rule pass）

结论：SC-MCMP 模型存在结构性缺陷，需库级重构（质量守恒修复 + σ 可测路径 + 负分布防护）方可晋级。当前维持 TERMINAL 终裁，记录负结果供后续改进参考。

## 图件

- `figures/fig_beta_convergence.png`：Washburn 指数 β 对比（W32/W64 vs 理论 0.5）
- `figures/fig_mass_drift.png`：质量守恒漂移史（湿润相/气相）
- `figures/fig_front_Lt.png`：侵入前沿 L(t) 与 Washburn 拟合
- `figures/fig_phase_field.png`：相场快照（W32/W64 终态）
- `figures/fig_convergence_comparison.png`：b_max_rel_diff 档间对比

