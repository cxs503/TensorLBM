# CASE-BACKWARD-STEP-LBM · 后台阶再附长度（Armaly 1983）· 第1轮

## ① 问题与参考出处
层流后台阶一级再附长度 X1/S（下壁面剪切变号点，以台阶高 S 归一）。
- 主参考：Armaly, Durst, Pereira & Schönung, *J. Fluid Mech.* **127**(4):473–496 (1983)；ER=1.9423；Re = ū·D_h/ν（D_h = 入口通道水力直径）。
- **参考获取状态 = PRIMARY_PDF_UNREACHABLE（诚实登记，未粉饰）**：unpaywall 判该 DOI 非 OA；cambridge core 付费墙；duckduckgo/startpage/qwant/ecosia 被沙箱拦截，bing 只回 JS 壳，scholar.archive.org 拒连，core.ac.uk 403；OpenAlex 全文检索 100 篇 OA 引文→取回 11 篇 PDF、22 页出现 "Armaly"，**无一页给出 Table 1 数值**。因此本轮**没有**声称“引原文原表原页”，数值参考降级为仓内既有档（tier = SECONDARY_IN_REPO，注明前序战役手工录入、本轮未获原文核验）。
- 机器核实（可复核，见 `benchmarks/case_pool/backward_step_armaly/refs.json`）：
  - arXiv:2507.16509 p.2：*"experimental data, such as those of Armaly et al. [6] for ER = 1.942"*；*"2D ... 70 < Re_D < 8000, based on the hydraulic diameter of the inlet channel. Up to Re_D = 400, 2D numerical results are matched with experimental data."*
  - arXiv:2507.16509 p.8 Table 2：Re=800 二维数值共识 xr/h = 11.48–12.20（Gartling 12.20 / Erturk 11.834 / Keskar&Lyn 12.19）。
- 采用的数值参考（tier SECONDARY_IN_REPO）：X1/S = 3.0 / 5.5 / 8.0 @ Re=100/200/400；容差 <15%。

## ② 方法与实现（复用优先，未另起炉灶）
复用 TensorLBM 库内核：`tensorlbm.d2q9.equilibrium/macroscopic`、`tensorlbm.solver.collide_bgk|collide_rlbm|stream`、`tensorlbm.boundaries.zou_he_inlet_velocity|zou_he_outlet_pressure|make_sponge_strength`。
标量 BFS 循环：pre-stream 半程 bounce-back（半格壁）→ Zou/He 抛物线速度入口（行均值 = ū）→ Zou/He 压力出口 → 出口非平衡态海绵层（消声学驻波）。
观测量双口径：**row1**（y=1 行 ux 的 −→+ 亚格零穿越，主口径）与 **K 行最小二乘壁面剪切斜率**（前序引擎口径，本轮判其不可靠，见 ⑤）。
入口剖面按离散行均值归一（质量守恒核对：入口流量与出口通量一致，无人工调节）。

## ③ 参数表（无量纲数完整）
| 量 | 值 |
|---|---|
| 几何 | 台阶高 S=m 格，入口高 m，出口高 H=2m ⇒ **ER = 2**（Armaly 1.9423，偏差 2.9%） |
| 雷诺数 | **Re = ū·H/ν = ū·D_h/ν**（Armaly 口径，D_h=2×入口高）；跑 100–400 |
| 格子速度/Mach | ū = 0.1，c_s=1/√3 ⇒ Ma ≈ 0.17，可压误差 O(Ma²) ≈ 3% |
| 黏性/松弛 | ν = ū·H/Re；τ = 0.5+3ν ∈ [0.512, 0.596] |
| 域长 | 下游 L/H = 12（≈24 S > Re=400 时 X1≈8 S） |
| 网格 | S = 8 / 16 格（H = 16 / 32）；上游 8 格 |
| 时间 | 12000–16000 步；时间平均 = 末 30% 窗口、步长 100 |
| 碰撞 | BGK |

## ④ 对照表（ref vs got，逐项打勾叉）
| Re | S(格) | steps | τ | got X1/S (row1) | ref X1/S | 误差 % | 判定(<15%) | 剪切斜率口径 X1/S |
|---|---|---|---|---|---|---|---|---|
| 100 | 16 | 12000 | 0.5960 | 2.6932 | 3.0000 | -10.2 | ✓ PASS | 1.884 |
| 100 | 8 | 12000 | 0.5480 | 2.4706 | 3.0000 | -17.6 | ✗ FAIL | 1.242 |
| 200 | 16 | 12000 | 0.5480 | 4.6892 | 5.5000 | -14.7 | ✓ PASS | 3.324 |
| 200 | 8 | 14000 | 0.5240 | 4.1976 | 5.5000 | -23.7 | ✗ FAIL | 1.937 |
| 400 | 16 | 12000 | 0.5240 | 0.3430 | 8.0000 | -95.7 | ✗ FAIL | 5.289 |
| 400 | 8 | 16000 | 0.5120 | nan | 8.0000 | nan | ✗ FAIL | nan |

瞬态档（进程被沙箱裁剪，**不得当结果引用**）：
| run | 步 | X1/S(inst) | 状态 |
|---|---|---|---|
| re200_m16 | 12000 | 3.336 | 瞬态，未 flush JSON |
| re400_m16 | 12000 | 5.531 | 瞬态，未 flush JSON |

## ⑤ 收敛性 / 参数敏感性（≥2 点）
1. **网格敏感性（决定性）**：Re=100，S=8 → S=16：X1/S = 2.471 → 2.693，误差 -17.6% → -10.2%，**单调向参考收敛**（网格加倍误差降 ~7 个百分点）⇒ 主误差源是台阶角/壁面离散分辨率，非物理模型。
2. **Re 敏感性**：S=8 档 Re=100→200：X1/S = 2.471 → 4.198，dX1/dRe ≈ 0.0173/单位 Re，近线性（与层流分离泡随 Re 增长定性一致），但斜率偏小 → 同一分辨率缺陷。
3. **测量口径敏感性（负面结果）**：同一次运行内，K 行最小二乘剪切斜率给出 X1/S ≈ row1 的 30–50%（Re=100 S=8：1.242 vs 2.471）⇒ 前序战役引擎“多行剪切拟合再附判据”在 BFS 上系统性偏短，**判为不可用**（台阶角一阶误差让拟合斜率提前变号）。

## ⑥ 误差判定与失败模式
- 最好档 |误差| = **10.2%**（Re=100, S=16）→ **<15% 容差：该点达标；是否 <3%：否**（3% 论文级门槛未达，最好 10.2%）。
- 逐项：Re=100 S=16 ✓ PASS；Re=100 S=8 ✗；Re=200 S=8 ✗；Re=400 S=8 全 16000 步无零穿越（未收敛/未形成可测再附）✗。
- 失败模式三条（生态摩擦，如实登记）：
  1. **参考源不可达**：Armaly 原文付费墙 + 全部常规检索通道被沙箱拦截 → 数值参考只能降级为 SECONDARY_IN_REPO。
  2. **算力被沙箱裁剪**：单次 shell 调用 120 s 上限，`nohup` 后台进程随调用结束**整组被收割**（`ps` 证实进程消失、日志仅剩首行）→ Re≥200 稳态无法在一次调用内跑完；`run_grid` 引擎首次尝试即因此静默死亡。
  3. **库内 BFS 入口口径非 Armaly**：`tensorlbm/backward_facing_step.py` 用**均匀入口** + Re=u_in·step_h/ν + 单行测量，不能满足实验口径；本轮改为复用库内核 + 自建循环（未改库文件）。
- 工件：`runs/case_backward_step_armaly/re*_m*.json|log`、`results_table.json`、`benchmarks/case_pool/backward_step_armaly/{run.py,refs.json}`、`figures/CASE_BACKWARD_STEP_LBM.png`。复现：`python3 benchmarks/case_pool/backward_step_armaly/run.py --re 100 --m 16 --steps 12000 --u 0.1`。
- **honest 结论：本单 honest-PARTIAL/FAIL** —— Re=100 细网格单点达标（10.2%），Re=200/400 未收敛，主参考原文未取得。下一轮接棒要点：优先以 stage_dir 上集群跑 Re=200/400/800 稳态（≥6 万步）+ 三档网格 Richardson；并优先解决 Armaly Table 1 原文获取。


## 附：S=16 档全部完成后追加（同一批次进程实际已 flush JSON，报告正文生成时按当时可见性列）
| Re | S | steps | got X1/S (row1) | ref | 误差% | 判定 | 说明 |
|---|---|---|---|---|---|---|---|
| 200 | 16 | 12000 | 4.6892 | 5.5 | -14.7 | ✓ PASS | 网格加倍后由 -23.7% 收敛到 -14.7%（网格敏感性第二个证据点） |
| 400 | 16 | 12000 | 0.3430 | 8.0 | -95.7 | ✗ 无效 | **伪零穿越**：τ=0.524、step 角附近 row1 出现极早符号翻转，测量判据失效，不得当作物理结果 |
| 400 | 8 | 16000 | nan | 8.0 | nan | ✗ | 无零穿越（未收敛/未形成可测再附） |

**判据缺陷结论（负面知识）**：row1 单行零穿越与 K 行剪切拟合在 BFS 上都不稳健 —— 前者在低 τ 档会被台阶角伪信号俘获（Re=400 S=16），后者系统性偏短 50–70%。正确判据应取「再附点附近速度剖面积分/流函数等值线 + 多点稳健拟合 + 稳态判据（滑窗变化 <0.2%）」，并强制 τ ≥ 0.55 或改用 MRT/RLBM。
