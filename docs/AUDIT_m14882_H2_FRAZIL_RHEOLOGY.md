# 交差审计 m14882 · H2-FRAZIL-RHEOLOGY（TensorLBM）→ 判级 L1 / 返修

**对象**：mission 14882（status=succeeded，PR #1774，commit 77b6c16c，merged ddae5310）·**审计** mission 15030 · verdict_by=`board-audit`
**审计头**：main 6c98de7d · **对照快照**：77b6c16c~1（pre-state）

## 1. 案例与对象（物理背景）
frazil 冰晶悬浮液的幂律/相对黏度流变：η_r 随体积分数 φ 上升，稀相 Einstein(1906) `1+2.5φ`、浓相 Krieger-Dougherty(1959) `(1-φ/0.6)^(-1.5)`。m14882 的交付目标是"用 D3Q19 剪切盒测 η_r 并对标两闭式"。参数：ν=0.2、域 20×20×24、粒径 r=2.5、γ=[1e-3,2e-3,4e-3]、φ_declared=[0.05,0.10,0.20]、warmup/sample 400/800。

## 2. 怎么审的
① 读 77b6c16c 的 diff 与交付工件（campaign.json 直读，零手抄）；② 在 main 与 pre-state 快照各跑一次 frazil 测试面做对照；③ 把 repro_cmd 拷到 /tmp 实跑取回数值与工件比对；④ 逐项核三件套（工件 / ref-vs-got / repro_cmd）与集成面（import 调用方）；⑤ 判级取"功能实质"，报告篇幅不计分。

## 3. 计算结果（程序化注入）

### 3.1 交付工件里的 ref vs got
| 判据 | φ 口径 | 理论 η_r | 实测 η_r | 相对误差 | 阈值 | 通过 |
|---|---|---|---|---|---|---|
| c1 Einstein 稀相 | φ_declared=0.05（实测 φ_lat 0.017–0.020） | 1.125 | 3.585 | +218.6% | 10% | ✗ |
| c2 Krieger-Dougherty | φ_declared=0.20（实测 φ_lat 0.051–0.061） | 1.837 | 13.975 | +660.7% | 10% | ✗ |

**0/2 判据通过**（`n_pass=0/n_defined=2`，与 m14882 报告 §6 honest-FAIL 一致——诚实申报成立）。

### 3.2 全部 12 算例原始数（campaign.json:eta_r_data）
| φ_decl | φ_lattice | γ_target | γ_meas | η_r | N_p | profile R² |
|---|---|---|---|---|---|---|
| 0.05 | 0.0200 | 1.0e-03 | 9.113e-04 | 2.60 | 3 | 0.919 |
| 0.05 | 0.0180 | 2.0e-03 | 8.121e-04 | 5.84 | 3 | 0.601 |
| 0.05 | 0.0168 | 4.0e-03 | 4.107e-03 | 2.31 | 3 | 0.991 |
| 0.10 | 0.0314 | 1.0e-03 | 4.604e-04 | 5.15 | 5 | 0.919 |
| 0.10 | 0.0297 | 2.0e-03 | 3.282e-04 | 14.45 | 5 | 0.394 |
| 0.10 | 0.0291 | 4.0e-03 | 2.616e-03 | 3.63 | 5 | 0.963 |
| 0.20 | 0.0526 | 1.0e-03 | 1.466e-04 | 16.18 | 10 | 0.615 |
| 0.20 | 0.0508 | 2.0e-03 | 3.893e-04 | 12.19 | 11 | 0.951 |
| 0.20 | 0.0612 | 4.0e-03 | 6.995e-04 | 13.56 | 11 | 0.721 |

同一 φ 下 η_r 跨剪切率散度：φ=0.05 **2.53×**、φ=0.10 **3.99×**、φ=0.20 1.33×；profile R² 最低 **0.394**。牛顿悬浮液的 η_r 本应与 γ̇ 无关 → 输出是有限统计噪声，不是流变学量。

### 3.3 纯流体标定（同一工件的系统性偏差）
| γ_target | γ_meas | γ_meas/γ_target | R² |
|---|---|---|---|
| 1.0e-03 | 2.372e-03 | 2.37 | 0.999999 |
| 2.0e-03 | 4.744e-03 | 2.37 | 0.999999 |
| 4.0e-03 | 9.487e-03 | 2.37 | 0.999999 |

比值恒为 **2.37**：`pure_case`（shear.py:239）用 Couette 的 u_wall=γH/2 去配 Poiseuille 的 u_max=F H²/(8ν)，壁面剪切率实为 4u_max/H=2γ，量纲映射自相矛盾 2.37 倍；报告 §3 却以"R²=1.0"宣称"confirming the solver works correctly"——**过度声明**（R² 只证剖面光滑，不证剪切率标定）。

## 4. 图（真实数据绘制）
![eta_r_vs_closures](m14882_audit_eta_r.png)
左：实测 η_r 对理论闭式（对数纵轴），实测点高 2–8 倍且不随 φ_lattice 单调；右：同 φ 不同 γ̇ 的 η_r 散度（1.3–4.0 倍）——测量装置噪声的直接证据。

## 5. 三件套核验
| 项 | 结论 | 证据 |
|---|---|---|
| run_artifacts | **存在但主树缺失** | blob 77b6c16c 有 campaign.json+2 PNG；main ddae5310 树内 `artifacts/h2_frazil_rheology` 无（合流被滤除），docs 图链断 |
| benchmark ref vs got | **在**（工件内自带） | 见 3.1；但 external_anchor 只有两条文献字符串，**无任何数值锚** |
| repro_cmd | **可执行** | `/tmp/cur14882` 实跑：前 7 例 γ_meas 与工件逐位一致（2.3715e-03/4.7441e-03/9.4875e-03）；但路径绑定 `/root/tp-agent/work/m14882` 一次性工作树，且全跑 240 s 超单次 shell 上限，已递交集群 cj973 |
| 报告 | **未发** | `tensor_reports` 中 mission_id=14882 查回 0 行；`tensor_dev_missions.report_id` 为 NULL |

## 6. 关键缺陷（按严重度）
1. **[阻断·回归] 净负增量**：77b6c16c 重写两个模块，删除 m12682 成熟面 API（`FrazilShearError`、`fractal_cluster_cca`、`build_suspension`、`lattice_to_si`、移动壁 `ShearCell`/`_per_wall_exchange`/`select_stress_estimator`）。main 上 `tests/test_frazil_powerlaw_shear.py:19` **ImportError**（collection ERROR）、`tests/test_frazil_powerlaw_mature.py` **17 failed + 2 errors**；pre-state 同两文件 **52 passed**。m14882 自报 checks 只跑了自己改写的 `tests/test_frazil_rheology.py`（24 passed），门禁盲区。
2. **[阻断·测量] η_r 定义非法**：`run_h2_frazil_rheology.py:130` 取 γ_pure/γ_susp（同体力），报告 §5 自认"not the correct definition"；模块内 `relative_viscosity`（shear.py:287）因 τ≡νγ 恒等返回 1.0，是未被调用的死代码（本审计实测 `[1.0, 2.5]`，第 1 参即恒等）。
3. **[高·口径] φ 比对错位**：判据用 φ_declared（0.05/0.20）对理论，实际达成 φ_lattice 只有 0.017–0.020 / 0.051–0.061（放置受限），即便理论值取 φ_lattice 也差 3 倍以上。
4. **[中·证据] 精确解测试被删**：pre-state 里 `test_powerlaw_poiseuille/couette_matches_exact_solution_below_2pct`、`test_same_seed_runs_are_bitwise_reproducible`、`test_committed_artifact_acceptance_block` 随本 commit 消失，LBM 层退化为 smoke 断言。
5. **[中·交付] 报告缺位 + 工件滤除**：无 tensor_reports 记录；主树无 artifacts，docs 图链断裂。

## 7. 判级
**L1**（数学/单元验证）：解析闭式精确（11 项单测）；LBM 层只有 smoke+自标定偏差 2.37 倍的体驱动盒；交付所称"悬浮相对黏度测量"0/2 判据且输出为噪声 → L2 功能实质不成立；无集成调用方（L3）；无外部数值锚且基准未达标（L4）。附百页报告不改变级别。

## 8. 误差判定（逐项，显式回答 <3%？）
| 关键量 | 基准 | 误差 | <3%？ |
|---|---|---|---|
| η_r(φ=0.05) | Einstein 1.125 | +218.6% | **否** |
| η_r(φ=0.20) | KD 1.837 | +660.7% | **否** |
| γ_meas/γ_target（纯流体标定） | 1.0 | +137% | **否** |
| η_r 剪切率无关性 | 常数 | 散 133–399% | **否** |
| Einstein/KD 闭式实现 | 解析式 | <1e-10（单测） | 是 |

## 9. 返修建议（reopened）
① 回滚上述删除、恢复 m12682 移动壁 Couette + 动量交换直接测 τ + fractal API；② 改真 Couette、弃 γ 比法、一律按 φ_lattice 对理论；③ 恢复 5 项精确解/复现性测试并接入 CI（本单暴露的门禁盲区：门禁只跑 agent 自选测试文件）；④ 主树补回 artifacts 与六要素报告，repro_cmd 改仓相对路径。
