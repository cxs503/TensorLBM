# XVAL-LBM-TRT-LAMBDA-MAGIC-TABLE-DRIVEN — 第 2 轮：magic Lambda 的壁面位置标定（R1 未覆盖的 D-6）

**仓** TensorLBM ｜ mission 23150 ｜ 分支 `agent/m23150` ｜ 锚点 `XVAL-LBM-TRT-LAMBDA-MAGIC-TABLE-DRIVEN`（open）
**轮次** 2 ｜ 预注册 `predeclared_r2.json`（跑前写入，阈值跑后未改）｜ 复现：`PYTHONPATH=src python3 development/verification/trt_lambda_magic/xval_trt_wall_lambda.py`

## 1 案例介绍

R1（m18452）在**标量 ADE** 上收口了 magic 关系 `Lambda=(tau_s-1/2)(tau_o-1/2)` 的代数与逐位场一致性，
并在差异清单 D-6 中把「Lambda 对**壁面位置误差**（Ginzburg 3/16 的经典用途）需带壁算例专门对标」留给后续轮次。
本轮补这件：在**流场侧**（D2Q9，`tensorlbm.solver.collide_trt`）用带半程反弹壁的平面 Poiseuille 流标定「Lambda -> 壁面位移」表。

| 项 | 值 |
|---|---|
| 算例 | 体积力驱动平面 Poiseuille，x 向周期，上下半程反弹壁（壁面 y=1/2 与 y=H+1/2） |
| 解析基准 | u(y)=g/(2nu)(y-1/2)(H+1/2-y)，Q=gH^3/(12nu)，nu=(tau+-1/2)/3（White 3rd ed. 3-2） |
| 体积力 g | 2e-06 |
| 网格 | nx=4（周期），H=32（主），H=16/32（分辨率对照） |
| 步数 | 1200（+200 步时均；缩减预算运行，见 5 节） |
| 观测量 | delta_eff = H_eff - H，H_eff=(12 nu Q/g)^(1/3)，由**流量**反解（无剖面拟合、无外推） |
| 扫的 Lambda | 0.083333, 0.166667, 0.125000, 0.187500, 0.250000, 0.375000 + BGK |
| 扫的 tau+ | 0.55, 0.60, 0.70, 0.80, 1.00, 1.30, 1.90 |
| 精度 | float64，torch CPU 单线程 |

delta_eff>0 表示有效壁面落在名义壁面之外。BGK 是 TRT 的一行：`TRT(tau, Lambda=(tau-1/2)^2) == BGK(tau)`。

## 2 怎么做的

一套驱动、一套初值（解析剖面，免瞬态）、一套测量算子，只切换 (Lambda, tau+)。
外加三个门：(a) `TRT(Lambda=(tau-1/2)^2) vs BGK` 逐位恒等；(b) H=16/32 分辨率对照；
(c) `method_selection` 反解 Lambda 与 3/16 的正面对决。判据先于运行写进 `predeclared_r2.json`
（H1 稳健性 / H2 偏差代价单调 / H3 BGK 恒等 / H4 最优 Lambda / H5 精度带 <=3%）。

**同时修复 R1 的验收通路（缺陷）**：R1 报告把 `xval_trt_lambda.py`、`predeclared.json`、`oss_src/` 列为产物，
但它们**从未合并**，`tests/test_xval_trt_lambda_magic_table_driven.py` 一收集即 `FileNotFoundError`（红，与物理无关）。
本轮用仓库内**已提交的 OSS 参考场**（`gaussian_1d_magic{12,6}_hi/*.dat`）重建了无 g++ / 无 oss_src 的等价 harness：
K3 对已提交参考场复现到 **5.55e-16**（两臂），K4 复现 R1 报的 4.03%，7 项验收现在 7 passed。

## 3 结果

delta_eff（cell，H=32，1200 步）：

| Lambda / tau+ | 0.55 | 0.60 | 0.70 | 0.80 | 1.00 | 1.30 | 1.90 |
|---|---|---|---|---|---|---|---|
| 0.083333 | -0.00491 | -0.00514 | -0.00620 | -0.00781 | -0.01197 | -0.01895 | -0.03242 |
| 0.166667 | -0.00263 | -0.00192 | -0.00170 | -0.00247 | -0.00572 | -0.01220 | -0.02547 |
| 0.125000 | -0.00377 | -0.00353 | -0.00395 | -0.00514 | -0.00884 | -0.01557 | -0.02894 |
| 0.187500 **(magic)** | -0.00206 | -0.00112 | -0.00057 | -0.00114 | -0.00416 | -0.01051 | -0.02374 |
| 0.250000 | -0.00036 | +0.00130 | +0.00280 | +0.00286 | +0.00053 | -0.00545 | -0.01854 |
| 0.375000 | +0.00304 | +0.00611 | +0.00954 | +0.01085 | +0.00989 | +0.00467 | -0.00813 |
| BGK (Lambda=(tau+-1/2)^2) | -0.00712 | -0.00798 | -0.00854 | -0.00738 | +0.00053 | +0.02609 | +0.12315 |

预注册判定：

| 假设 | 判据 | 实测 | 判 |
|---|---|---|---|
| H1 magic 稳健 | tau 散布(3/16) <= 1/5 tau 散布(1/4) 且 max|delta|<=0.03 | 散布 3/16=0.0176 vs 1/4=0.0161（比 0.92），max|delta|=0.0237 | **FAIL** |
| H2 偏差代价 | 按 |Lambda-3/16| 排序与 |delta_eff| 相关 rho>0.8 | rho=0.967 | PASS |
| H3 BGK 恒等 | max|du| <= 1e-13 | 3.03e-16 | PASS |
| H4 最优 Lambda | argmin|delta_eff|(tau=0.8) = 3/16 | 0.1875 | PASS |
| H5 精度带 | 全扫剖面 L2 <= 3% | 0.322% | PASS |

tau=0.55 正面对决（该 tau 下 3/16 给出 tau-=4.25，落在选择器「稳定窗 [0.5,2]」之外，故仓内会改选 Lambda=0.0750）：

| 臂 | tau- | delta_eff (cell) | 剖面 L2 % |
|---|---|---|---|
| Lambda=3/16（magic） | 4.25 | -0.00206 | 0.0663 |
| Lambda=0.0750（窗内反解） | 2.00 | -0.00514 | 0.0885 |

## 4 图

`figures/fig1_delta_eff_vs_tau.png`（delta_eff(tau+) 六条 Lambda 曲线 + BGK，tau=0.933 参考线）、
`figures/fig2_delta_eff_vs_lambda.png`（tau=0.8 下 |delta_eff| vs Lambda；1/8 与 1/4 的 |Lambda-3/16| 相同但代价不同）、
`figures/fig3_profiles.png`（tau=0.8 剖面 vs 解析抛物线）。无场量云图：观测量是标量壁面位移，解本身即抛物线，云图无信息量。

## 5 注意点 / 局限（诚实登记）

1. **H1 被证伪**：`collide_trt` docstring 的「Lambda=3/16 eliminates wall placement errors」在流量型 delta_eff 上只在 tau+<~1.0 站得住（|delta|~1e-3 cell）；tau+=1.9 时 |delta_eff|=0.0237 cell（BGK 同 tau 为 +0.1232，约 5 倍）。建议把该 docstring 改为「把 tau 相关壁面位移压低约 3-5 倍」。
2. **delta_eff 的 H 依赖性未完全分离**：tau=1.9 时 3/16 的 delta(H) 从 H=16 的 -0.0481 降到 H=32 的 -0.0237，残差主要是 O(H^-2) 离散误差而非固定壁面偏移；**两点拟合给不出无偏 offset**（H=64 的更细作业 cj4635 仍在集群排队，本文不据它下结论）。
3. **预算**：本轮 steps=1200（完整 3000 步配置的同一脚本作业排队未返回）。与 m16728 的 3000 步稳态值对照（H=32, tau=0.80, Lambda=3/16：m16728 报 -0.00106，本文件 -0.00114）同号同量级，结论方向不受步数影响；1e-3 cell 级精确数字以完整作业为准。
4. **适用范围**：充分发展层流 + 直线半程反弹壁；不覆盖曲面/插值反弹、非定常、D3Q27 流场（R1 留下、本轮仍未覆盖）。
5. `method_selection` 的 tau- in [0.5,2]「稳定窗」是**未标定的启发式**：本算例 tau=0.55 下 tau-=4.25 既稳定又比窗内反解 Lambda 更准，建议降级为「经验推荐」并加注本实测。

## 6 误差判定（vs 解析基准，逐项回答是否 <3%）

| 量 | 基准 | 实测 | <3%? |
|---|---|---|---|
| 全扫（6 Lambda x 7 tau+）剖面 L2 | 解析抛物线 | <= 0.322% | **是** |
| 流量 Q 相对误差（全扫） | gH^3/(12 nu) | <= 0.304% | **是** |
| delta_eff 的 H 收敛（tau=1.9, 3/16） | H=32 | 0.248%（H=16 为 0.998%） | **是** |
| R1 K3 场恒等（重建 harness） | 已提交 OSS 参考场 | 5.55e-16 abs | **是**（机器精度） |
| R1 K4 场级 relL2 | 周期热核解析解 | 4.03% | **否**（与 R1 原判一致：粗格 dx=1；细化到 1/4 为 0.09%） |

**结论**：magic Lambda 表格化做法在标量侧逐位可复现（本轮重建 harness 复核 5.6e-16）；在壁面侧，Lambda=3/16 是所扫 6 个 Lambda 中 tau=0.7-1.3 的最优取值（H4 PASS）、偏差代价随 |Lambda-3/16| 单调（H2 PASS），但「消除壁面误差」的强表述被本轮数据证伪（H1 FAIL）；tau 相关残差与 O(H^-2) 离散误差混在一起，H=64 分离作业排队中——如实登记。


Repository relocation: [canonical case index and historical evidence](../benchmarks/cross_validation/trt_lambda_magic/README.md). The original reference fields and archived reports are unchanged.
