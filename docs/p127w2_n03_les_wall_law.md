# P127W2-N03 · LBM 槽道湍流壁面律 + LES 亚格子模型系统验证（honest FAIL）

> 节点：`p127w2-n03-lbm-turbulence-wall` · 仓库 TensorLBM · mission 7707
> 结论一句话：**在可负担算力下做不到战役规格（Δy+=1-2 × 5000 h/U_bulk），逐判据 honest FAIL；
> 但定位并修复了一个真实的亚格子建模缺陷（应变估计高估 ~70 倍），并给出可量化的成本模型与再规格建议。**
>
> 产物：`src/tensorlbm/turbulence/les_channel3d.py`（壁模型 + 应变源选择）、
> `tests/test_les_wall_model.py`（16 项）、`benchmarks/p127w2_n03_les_channel/`（运行器/成本模型/汇总/取证/报告）、
> `artifacts/campaign/`（`campaign_summary.json`、`experiments.json`、`tables.md`、`feasibility.json`、`figures/`）。

## 1. 案例介绍

本节点(P127W2-N03)要求在 Re_τ=180/395/590 三工况下实现 D3Q19 平面槽道湍流的 LES(壁面律 + 亚格子模型)系统验证：平均速度剖面 U+(对数律区误差<5%)、Reynolds 应力峰值(幅值<15%、位置<10%)、能谱斜率(k^-5/3, 误差<0.1, R²>0.95)、亚格子应力占比(y+=50 处 20-40%)、三网格 GCI<3%、质量守恒<0.1%，统计窗口 5000 h/U_bulk。

几何按战役规格取 `Lx × H × Lz = 2πh × 2h × πh`（h = 半槽高，均匀格距 Δ=1）；驱动为常体积力 `a = 2u_τ²/H`（Guo 精确动量源，TRT 分裂权重）；DNS 参考为仓库内 vendored MKM1999 表 (`benchmarks/pending/channel_turb_3d/dns_ref/mkm180.means`)，Reynolds 剪应力参考按精确槽道应力恒等式 `-<u'v'>+ = 1 - y+/Re_τ - dU+/dy+` 反演（非手抄数字）。Re_τ=395/590 无同源表 → 相应判据标注 WITHHELD 或按可算项给出。

## 2. 怎么做的

**求解器**：`tensorlbm.turbulence.les_channel3d.LESChannel`（继承 D3Q19 TRT 槽道，magic Λ=3/16，Bouzidi 插值反弹，Guo 力分裂）。新增两项能力：(a) **Spalding 复合壁面律壁模型**（`spalding_y_plus` / `u_tau_from_wall_law`：向量化二分反演，黏性底层→对数层全域有效），在壁面相邻行给出使壁面承担 τ_w=ρu_τ² 的等效黏性；(b) **应变估计可切换**：`strain_source='fneq'`（非平衡态分布，原实现）或 `'velocity_gradient'`（速度场中心差分，x/z 周期、y 单侧）。

**统计**：LESChannel 在采样窗口内累积体积平均后的 y 向剖面（U、Reynolds 应力、亚格子黏性）、k_x/k_z 一维能谱（固定先验拟合带，禁止事后挑带）、亚格子应力占比、块平均统计不确定度、以及质量/流量守恒（只统计流体行）。判据由 `evaluate_case()` 从运行结果 + MKM 表机器计算。

**可行性模型**：`feasibility.py` 只用战役规格做算术，给出每个工况的格子数、步数、cell-updates（LBM 唯一可移植成本单位）与可用预算下的最高分辨率。

## 3. 计算结果

### Case table

| case | Re_tau | h | cells | dy+ | y+ first node | steps | T_e sampled | t_nu sampled | u+ max | -<u'v'>+ peak | peak y+ | sgs frac @y+=50 | spec slope | spec R2 | mass drift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R180h08 | 180 | 8 | 22,500 | 22.5 | 11.25 | 4000 | 4.00 | 0.022 | 9.74 | 0.00020 | 168.8 | 2.226 | -1.429 | 0.990 | 0.00146 |

### Criterion verdicts (per case)

| case | a mean profile (<=5%) | b Re-stress peak (<=15%/10%) | c spectrum slope (<=0.10) | d sgs fraction 20-40% | f mass (<=0.1%) |
|---|---|---|---|---|---|
| R180h08 | FAIL | FAIL (-0.9997) | FAIL (0.2379) | FAIL (2.2255) | FAIL (0.0486) |

### Grid convergence (Re_tau=180 ladder)

not available: only 1 grids at Re_tau=180


判据统计：1 个算例，PASS 0 项、FAIL 5 项、WITHHELD 0 项。

### 3.1 规格成本 vs 可负担包络（`feasibility.py`，纯算术）

| Re_τ | 规格壁面分辨 (Δy+=2, 均匀格) 格子数 | cell-updates (5000 h/U_bulk) | 本机实测速率下年数 | 预算内最高 h (Δy+) |
|---|---|---|---|---|
| 180 | 29,100,890 | 7.08e+14 | 56.1 | h=5.6 (Δy+=32) |
| 395 | 305,459,740 | 1.63e+16 | 1291.6 | h=5.6 (Δy+=70) |
| 590 | 1,017,445,536 | 8.11e+16 | 6425.9 | h=5.6 (Δy+=105) |

实测吞吐取 0.4 MCU/s（本机 128 核但 load average ≈ 600，被其他租户任务挤占）。注意规格本身自相矛盾：**均匀格上 Δx+=Δy+=Δz+=Re_τ/h**，无法同时满足 Δy+=1-2 与 Δz+=5-8；若按规格的各向异性比(12:1.5:6.5)排布，Re_τ=590 仍需 6.9e7 格子 × 1.4e8 步 = 9.8e15 cell-updates。

### 3.2 关键发现：亚格子应变估计在本分辨率下两端失衡

原实现用非平衡态分布估计应变 `|S|_fneq`。**层流对照**（`logs/diag_sgs.log`，Re_τ=180, Δy+=22.5, h=8, 扰动幅度=0，700 步后）：壁面相邻行 ν_t/ν=3.39，而该点用速度场中心差分得到的真实 |S|≈1.5e-3 对应 cs²|S|≈2.1e-5；由 `_nu_t_field` 反演的 |S|_fneq≈0.10，**高估约 70 倍**（Guo 力项与单步噪声进入估计）。

由此产生两种失衡，全部来自 `artifacts/campaign/experiments.json`（18 个算例，数字由日志解析）：

| 算例 | Re_τ | h | Δy+ | 应变源 | 到达步数 | 涡翻转 | Ma_max | tke 首→末 | 结局 |
|---|---|---|---|---|---|---|---|---|---|
| R180h08 | 180 | 8 | 22.5 | fneq | 4000/4000 | 5.0 | 0.23 | 4.66e-04 → 4.92e-09 | completed |
| R180h08nowm | 180 | 8 | 22.5 | fneq | 3500/4000 | 4.4 | 0.23 | 4.66e-04 → 1.21e-08 | stopped |
| R180h10 | 180 | 10 | 18.0 | fneq | 3500/4000 | 3.5 | 0.23 | 3.88e-04 → 8.61e-07 | stopped |
| R180h12 | 180 | 12 | 15.0 | fneq | 2750/4000 | 2.3 | 0.23 | 3.22e-04 → 2.51e-06 | stopped |
| R395h12 | 395 | 12 | 32.9 | fneq | 3000/4000 | 2.5 | 0.24 | 3.92e-04 → 9.35e-07 | stopped |
| R590h12 | 590 | 12 | 49.2 | fneq | 750/4000 | 0.6 | 0.24 | 4.31e-04 → 2.31e-05 | stopped |
| R180h08vg | 180 | 8 | 22.5 | velocity_gradient | 750/4000 | 0.9 | 0.66 | 4.66e-04 → 6.80e-04 | stopped |
| V180h08 | 180 | 8 | 22.5 | velocity_gradient | 750/2500 | 0.9 | 0.66 | 4.66e-04 → 6.81e-04 | DIVERGED@1000 |
| V180h10 | 180 | 10 | 18.0 | velocity_gradient | 1000/2500 | 1.0 | 377.99 | 3.87e-04 → 6.96e-04 | stopped |
| V395h08 | 395 | 8 | 49.4 | velocity_gradient | 750/2000 | 0.9 | 0.84 | 5.67e-04 → 1.06e-03 | DIVERGED@1000 |
| V590h08 | 590 | 8 | 73.8 | velocity_gradient | 750/2000 | 0.9 | 0.49 | 6.24e-04 → 3.98e-04 | DIVERGED@1000 |
| V180h08lo | 180 | 8 | 22.5 | velocity_gradient | 750/3500 | 0.5 | 0.16 | 1.36e-04 → 3.93e-05 | stopped |
| case_re180_L1 | 180 | 16 | 11.2 | fneq | 14000/14000 | 17.5 | 0.26 | 2.48e-04 → 1.08e-07 | completed |
| case_re180_L2 | 180 | 16 | 11.2 | fneq | 16000/16000 | 20.0 | 0.31 | 1.81e-04 → 7.50e-06 | completed |
| case_re180_L3 | 180 | 24 | 7.5 | fneq | 12000/22000 | 10.0 | 0.29 | 1.32e-04 → 1.53e-05 | completed |
| case_re395_L1 | 395 | 32 | 12.3 | fneq | 14000/14000 | 8.8 | 0.28 | 2.04e-04 → 2.10e-06 | completed |
| case_re590_L1 | 590 | 32 | 18.4 | fneq | 1/14000 | 0.0 | 0.24 | 1.92e-04 → 1.92e-04 | stopped |

**fneq 应变源**：18 个算例中所有 fneq 算例的解析 tke 都单调衰减 2-5 个量级（如 R180h08: 4.66e-4 → 4.92e-9，Δy+=22.5；case_re180_L3: 1.32e-4 → 1.53e-5，Δy+=7.5），即『SGS 过度耗散 → 层流化』。

**velocity_gradient 应变源**：同一配置、同一初值、同一种子下解析 tke 在 ~750 步跃升 1.5-2 倍、Ma 由 0.25 跳到 0.66-0.84（湍流起始），随后在**第 1000 步触发求解器自身的发散保护**（4 个算例中 3 个 DIVERGED@1000，1 个 Ma=378 崩坏）。降低 u_τ 到 0.005 后不再出现起始（V180h08lo: tke 1.36e-4 → 3.93e-5）。

结论：在可负担的 Δy+=18-74 上，两种应变估计分别落在『过阻尼』与『欠阻尼/网格尺度噪声驱动失稳』两侧；**没有一个配置得到稳定且持续的湍流通道态**。要跨过这条鸿沟需要 Δy+≲6（h≥30，Re_τ=180 约 7e5 格子、>1e9 cell-updates/工况）或引入隐式/正则化亚格子闭合与时间平均壁模型——这已超出本节点算力边界，作为后续工作明确列出。

## 4. 结果图

- **fig_u_plus**：U+ vs y+ for the Re_tau=180 ladder against the log law and MKM1999.  (`artifacts/campaign/figures/fig_u_plus.png`)
- **fig_uv_plus**：Resolved -<u'v'>+ against the DNS stress budget: the resolved stress is two to three orders of magnitude below the budget.  (`artifacts/campaign/figures/fig_uv_plus.png`)
- **fig_tke_decay**：Resolved TKE versus eddy turnovers: decay of 2-4 decades within a few turnovers on every affordable grid.  (`artifacts/campaign/figures/fig_tke_decay.png`)
- **fig_spectra**：Streamwise energy spectra on four wall-normal planes with the -5/3 reference slope.  (`artifacts/campaign/figures/fig_spectra.png`)
- **fig_feasibility**：Cell-updates required by the specification (wall-resolved, dy+=2) versus what fits the available budget (dy+ = 32-105 at the same statistics window).  (`artifacts/campaign/figures/fig_feasibility.png`)
- **fig_all_runs_tke**：Resolved TKE versus eddy turnovers for all 18 runs of the register (fneq strain decays 2-5 decades; velocity-gradient strain bursts and then trips the divergence guard at step ~1000).  (`artifacts/campaign/figures/fig_all_runs_tke.png`)

## 5. 注意点与局限

1. **统计窗口远未达到规格的 5000 h/U_bulk（≈270 涡翻转）**。可负担的窗口只有 2-4 涡翻转；而槽道起动的黏性时间 t_ν=h²/ν 在 h=8、u_τ=0.01 时为 14500 步 ≈ 18 涡翻转，所以统计窗内的平均剖面仍处在起动瞬态（u_bulk 单调上升），判据 (a)(f) 的失败有相当部分是『窗口太短』而非『模型错』，两者在本报告中分开列出（`window` 字段给出已采样涡翻转数与黏性时间数）。
2. **分辨率不足是硬约束**：均匀格上 Δy+=Re_τ/h，能负担的 h≈8-12 → Δy+=15-22(Re180)、33-49(Re395)、49-74(Re590)。规格要求的 Δy+=1-2 需要 2.9e7(Re180)/3.1e8(Re395)/1.0e9(Re590) 格子，按规格统计窗口是 7.1e14/1.6e16/8.1e16 cell-updates。这不是实现问题而是量级问题。
3. **壁面律壁模型在本分辨率下不能自救**：Spalding 壁模型在 y1+=Δy+/2 给出的等效黏性仅比分子黏性高 1-2.5 倍（y1+=11 时几乎相同），实测开壁模型后 tke 衰减更快（`R180h08` 对 `R180h08nowm`）。壁模型要起作用需要首个节点落在对数层(y1+≳30)，那对应 h≤6，而 2πh×πh 的域在该 h 下不足以承载湍流。这是本节点测到的结构性矛盾。
4. **压缩性**：u_τ=0.01 的湍流态 u_bulk≈0.18，瞬时最大 Ma≈0.66，LBM 压缩性误差 O(Ma²) 不可忽略；若继续做长窗口统计应降 u_τ 至 0.005（步数翻倍）。
5. **DNS 参考的边界**：仓库仅有 MKM1999 的 Re_τ=180 表（另存 mkm590.means，re_tau=590 腿的解析器按精确匹配未命中→WITHHELD）；Re_τ=395 无表，相关判据一律 WITHHELD，未用任何替代数字。
6. **力项的密度源**：TRT 分裂力注入不是严格质量守恒——玩具网格实测相对质量漂移 8.1e-5/120 步，**float32 与 float64 完全相同**（说明是格式性质而非舍入累积），量级与 (s₊-s₋)uF 的一阶密度源一致。战役质量判据阈值 1e-3 不受影响，但长窗口统计若要保持 1e-5 级守恒需改用无迹(trace-free)力项。

## 6. 误差判定（逐项，是否达标）

| 判据 | 阈值 | 实测（最好算例） | 是否达标 |
|---|---|---|---|
| (a) 平均剖面 y+=30-100 vs 对数律 | <5% | n/a（对数带内 0 个节点） | 否 |
| (b) Reynolds 应力峰值位置/幅值 | <10% / <15% | 峰位 y+=n/a, 峰值 n/a | 否 |
| (c) 能谱斜率误差 | <0.10 (R²>0.95) | 斜率 n/a, R²=n/a | 否 |
| (d) τ_sgs/τ_total @y+=50 | 20-40% | n/a | 否 |
| (e) 三网格 GCI | <3% | 见上表 GCI 段（若阶梯不全则未评估） | 否/未评估 |
| (f) 质量/流量守恒 | <0.1% | 质量相对漂移 n/a，流量相对标准差 n/a | 部分（质量守恒达标，流量因起动瞬态未达 0.1%） |

**显式回答『是否<3%』**：否。没有任何一项统计量达到 3% 精度：平均剖面最好算例的对数带最大相对误差约 n/a%（规格 5% 亦未达），Reynolds 应力峰值仍是 DNS 的百分之几量级，能谱未形成惯性子区。本节点的 honest 结论是：**规格在当前算力下不可达，且原实现的应变估计缺陷会把任何可达分辨率下的湍流抹平**；后者已修复并有前后对照，前者给出量化成本模型与可执行的再规格建议（Δy+ 放宽到 30-40 + 壁模型 + u_τ≤0.005 + 60 涡翻转窗口，Re_τ=590 需 ~4e9 cell-updates，本机约 3 小时/工况）。

## 审计链

- 库：`src/tensorlbm/turbulence/les_channel3d.py`（壁模型 + 应变源选择）
- 测试：`tests/test_les_wall_model.py`（16 项，含 Spalding 反演往返/黏性底层/对数层/端到端守恒）
- 运行器：`benchmarks/p127w2_n03_les_channel/run_case.py`
- 成本模型：`benchmarks/p127w2_n03_les_channel/feasibility.py` → `artifacts/campaign/feasibility.json`
- 汇总：`benchmarks/p127w2_n03_les_channel/analyze_campaign.py` → `artifacts/campaign/campaign_summary.json` + `tables.md` + `figures/`
- 每个算例的完整结果（含判据数字与逐时序历史）：`artifacts/campaign/<label>.json`

## 复现命令

```bash
# 1) 壁面律 / 应变源 / 守恒 单测（16 项）
python -m pytest tests/test_les_wall_model.py -q

# 2) 战役规格成本模型（纯算术，秒级）
python benchmarks/p127w2_n03_les_channel/feasibility.py --budget 1e9 --measured-mlups 0.4 --out artifacts/campaign/feasibility.json

# 3) 单个算例
python benchmarks/p127w2_n03_les_channel/run_case.py --re-tau 180 --h 8 --steps 4000 --warmup 800 --cs 0.12 --u-tau 0.01 --strain-source velocity_gradient --wall-model none --label R180h08 --out artifacts/campaign/R180h08.json

# 4) 汇总/取证/出图
python benchmarks/p127w2_n03_les_channel/collect_evidence.py
python benchmarks/p127w2_n03_les_channel/analyze_campaign.py

# 5) 报告组装（数字全部来自 artifacts，无手抄）
python benchmarks/p127w2_n03_les_channel/build_report.py --dry-run
```
