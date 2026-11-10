# TensorLBM · 壁面模型大涡模拟验证面（m8872 / P92W3-N01）

**仓库** TensorLBM ｜ **分支** `agent/m8872` ｜ **模块** `tensorlbm.wall_model_les`
**判据** (a) 对数律区 y+=30-300 剖面误差 ≤3 % · (b) 雷诺应力峰值位置误差 ≤5 % ·
(c) 平板 C_f 误差 ≤5 %（vs ITTC-1957）· (d) KVLCC2 C_T 误差 ≤7 %（vs 实验）

> 本单是第二次尝试。第一次的结论里有两处口径需要纠正，本报告在 ⑤ 中逐条列出：
> (i) 判据 (a) 曾被记为 PASS，依据是**锚定到 DNS 截距之后**的剖面误差 0.0172——
> 锚定把截距误差人为消掉，那是形状诊断，不是剖面预报；(ii) 判据 (b) 的模型峰值曾取在
> 被截断剖面的边界上（y+=178.1 处 1.009），那是截断伪峰。两处都已修正并钉进测试。

## ① 案例介绍

| 算例 | 几何 / 工况 | 关键参数 | 用途 |
|---|---|---|---|
| 槽道 DNS 参照 | MKM 1999 槽道，Re_tau=180（仓内 `mkm180.csv`，65 点） | u_tau=1，h+=180，截面 y+ 到 178.12 | 判据 (a)(b) 的数据基准 |
| 壁面模型闭合层（1-D 全槽道）| 由 u_tau 归一的一维槽道，闭合=van Driest 混合长 | κ=0.41, B=5.2, A+=26；Re_tau 扫 180…1e4 | 判据 (a)(b) 的闭合层测量 |
| DNS 峰值锚点 | 同上 | -<u'v'>+ 峰值 y+=31.98，值 0.724；P+ 峰值 y+=12.23 | 峰值位置基准 |
| 壁模化槽道 LES | D3Q19 + cumulant + Smagorinsky，x/z 周期，y 双壁 | u_tau=0.02，Cs 与 ny 见稳定域表 | 判据 (a) 的解析层尝试 |
| 零压力梯度平板 | 壁律 + Coles 尾流积分边界层 | Re_theta=1e4…5e4，κ=0.41，B=5.2，Π=0.45 | 判据 (c) |
| KVLCC2 母型（全尺度 Lpp=320 m）| 模型尺度 Lpp=10.65 m，湿表面积 S=30.1 m² | Re=1.30e+07，Fr=0.142，1+k=1.20 | 判据 (d) |

参照数据全部来自仓内或程序化计算，无手抄：MKM-180 表（经完整性门放行）、ITTC-1957 解析线、
壁律解析式。KVLCC2 拖曳实验数据在本环境**不存在**（⑤ 有检索记录），因此判据 (d) 的实验腿保持 WITHHELD。

## ② 怎么做的

1. **参考数据先过门**（`tensorlbm.wall_model_les.reference_gate`）：任何 DNS 表在参与误差计算之前，
   机器检查 (i) 平均剖面单调、(ii) 中线 u+ 与对数律估计 1/κ·ln Re_tau+B 相差 ≤15 %、
   (iii) 存储的相关量 |uv+|≤1 且其峰值落在 y+=20…60。门只拒绝、不修补，拒绝理由带数字。
2. **闭合层一维求解**（`closure_channel.solve_channel_closure`）：把槽道精确应力平衡
   τ+=1-y+/h+ 与混合长闭合 ν_t+=l+²·du+/dy+ 联立，闭式反解 du+/dy+ 后沿壁法向积分，
   得到 u+(y+)、-<u'v'>+(y+)、P+(y+)。应力恒等式误差机器记录（本单 <1e-10）。
   这一步把「闭合律对不对」与「网格能不能分辨」两件事分开：前者毫秒级可测，后者要看 LES。
3. **平板边界层**（`flat_plate`）：壁律 + Coles 尾流剖面族，动量积分推进 Re_theta，
   局部 c_f=2/Ue+²，与 ITTC-1957 局部线、Prandtl-Schlichting 线双基准比较。
4. **壁模化槽道 LES**（`channel` / `stability_map`）：D3Q19 + cumulant + Smagorinsky + Guo 体积力
   + halfway bounce-back，壁面模型在交换节点给 u_tau。本单系统扫 (Re_tau, Cs) 的收敛域。
5. **KVLCC2 离线链**（`kvlcc2`）：不再依赖船舶平台 API（本次运行时它不可达），
   用壁律积分模型的**平均**摩阻 C_F=2θ(L)/L 与 ITTC-1957 线两条独立摩擦链，
   在同一 (1+k) 下折成 C_T 比较；C_A 是两条链共用的输入，在相对比较中相消，故不臆造数值。
6. **判据表**（`acceptance.py`）把工件折成 (a)-(d) 的裁决，并显式区分子口径（预测腿/形状腿/验证腿）。

## ③ 计算结果（数字表，程序化注入）

**判据表（`artifacts/wall_model_les/acceptance.json`）**

| 判据 | 阈值 | 裁决 | 依据 |
|---|---|---|---|
| a_log_layer_profile | 3.00 % | **FAIL** | predictive limb vs the vetted channel DNS (MKM-180, y+ = 30...178, the whole criterion band the table covers): best model = 0.0393 vs threshold 0.03. The 0.0172 of the m7797 record was the anchored shape-only limb and is not a profile prediction |
| b_reynolds_stress_peak | 5.00 % | **FAIL** | no resolved-stress artifact for this pass (see stability_map.json): the graded number is the closure tier, modelled -<u'v'>+ peak vs the DNS peak, rel error 0.0843 > 0.05; the abscissa drifts with Re_tau under an eddy-viscosity closure while the DNS peak does not |
| c_flat_plate_cf | 5.00 % | **PASS** | max rel error 0.0363 vs threshold 0.05 over Re_theta = 1e4...5e4 (and 0.0365 against the independent Prandtl-Schlichting line); before the m7797 Coles-wake fix the same metric was 0.1141, i.e. the criterion now passes because the wake double-counting was removed, not because the threshold moved |
| d_kvlcc2_ct | — | **WITHHELD** | the literal limb (C_T vs a measured KVLCC2 resistance) has no reference in this environment and stays WITHHELD; the friction limb is decidable offline and compares two independent friction chains at Re = 1.3e7: C_T = (1+k) C_F with C_F from the wall-law boundary-layer model vs the ITTC-1957 line, relative difference 0. |
| reference_integrity | — | **INFO** | the shipped Re_tau = 395 table is REJECTED by the integrity gate (['mkm395']); its centreline u+ is 15.96 against a log-law estimate of 19.78 and the stored uv_plus column reaches 6.0, so it is not used as a reference anywhere in this record |

计数：{"n_pass": 1, "n_fail": 2, "n_withheld": 1}

**(a) 对数律区剖面：三条腿分开记**

| 口径 | 模型 | 最大相对误差 | 说明 |
|---|---|---|---|
| 预测腿（模型自预报 u_tau，含截距误差） | equilibrium_spalding | 6.29 % | 与 DNS 的 y+=30…178 全域比较 |
| 预测腿（模型自预报 u_tau，含截距误差） | log_law_only | 3.93 % | 与 DNS 的 y+=30…178 全域比较 |
| 预测腿（模型自预报 u_tau，含截距误差） | nonequilibrium_ode | 4.97 % | 与 DNS 的 y+=30…178 全域比较 |
| 形状腿（锚定到 DNS 截距） | nonequilibrium_ode_anchored_at_y_plus_30 | 1.72 % | 仅作形状诊断（上一轮被误读为 PASS 的那一项） |
| —— | 锚定平移量 | 0.6894 u+ 单位 | 形状腿把截距误差平移掉，故不得当剖面预报 |

| 闭合 | Re_tau | 拟合 κ | 拟合 B | κ 相对误差 | 拟合带内最大相对误差 |
|---|---|---|---|---|---|
| mixing_length | 10000 | 0.3976 | 4.873 | 3.02 % | 1.83 % |
| log_layer | 10000 | 0.4180 | -0.995 | 1.96 % | 0.23 % |

   验证腿：高 Re（1e4）时 y+=30…300 才真在对数层内，闭合积分回推出的 κ 与它自己的
   0.41 相差几个百分点（van Driest 阻尼在 y+<100 仍有贡献），说明实现无误；
   预测腿与 DNS 的 4-6 % 差距来自 κ/B 标定与低 Re 尾流，不是实现缺陷。
   解析层（LES）：最高收敛 Re_tau=180，判据带可达=no（见 ⑤ 稳定域）。

**(b) 雷诺应力峰值位置**

| 闭合 | 模型 -<u'v'>+ 峰值 y+ | DNS 峰值 y+ | 相对误差（阈值 5 %） | 生成项 P+ 峰值相对误差 |
|---|---|---|---|---|
| mixing_length | 29.28 | 31.98 | 8.43 % | 16.69 % |
| log_layer | 29.28 | 31.98 | 38.36 % | 72.67 % |

闭合层峰值横坐标随 Re_tau 漂移（DNS 峰值 y+=32.0 不随 Re 变）：

| Re_tau | 180 | 395 | 590 | 1000 | 2000 | 10000 |
|---|---|---|---|---|---|---|
| mixing_length 峰值 y+ | 29.3 | 39.9 | 46.7 | 57.5 | 76.2 | 156.6 |

**参考数据完整性门**

| 表 | Re_tau | 裁决 | 中线 u+ | 对数律估计 | 存储 \|uv\| 最大值 | 理由 |
|---|---|---|---|---|---|---|
| mkm180 | 180 | **PASS** | 18.30 | 17.87 | n/a | — |
| mkm395 | 395 | **REJECTED** | 15.96 | 19.78 | 6.012 | u+ is not monotone: first decrease at y+ = 203.504 (17.212 -> 17.210), max backward step -0.100；centreline u+ = 15.96 vs log-law estimate 19.78 (19.3% > 15%: wrong Re_tau, wrong normalisation, or not a channel)；stored |u |

**壁模化槽道 LES 收敛域（`stability_map.json`）**

| Re_tau 目标 | Cs | 网格 (nx,ny,nz) | dy+ | 交换点 y+ | 步数 | 收敛 | 首个非有限步 |
|---|---|---|---|---|---|---|---|
| 180 | 0.20 | (32,49,32) | 7.5 | 11.2 | 800 | yes | n/a |
| 590 | 0.05 | (32,33,32) | 36.9 | 55.3 | 2400 | no | 1200 |
| 1000 | 0.05 | (32,49,32) | 41.7 | 62.5 | 2400 | no | 1080 |
| 2000 | 0.05 | (32,65,32) | 62.5 | 93.7 | 2400 | no | 1080 |

**KVLCC2 摩擦链（离线，`kvlcc2.json`）**

| 量 | 壁律积分边界层模型 | ITTC-1957 线 | 相对差 |
|---|---|---|---|
| C_F (Re=1.3e7) | 0.002995 | 0.002868 | 4.44 % |
| C_T=(1+k)C_F, 1+k=1.20 | 0.003594 | 0.003441 | 4.44 % |

裁决：摩擦腿 **PASS**（阈值 7 %）；实验腿 **WITHHELD**。
平台 API 状态：ship platform ITTC-78 reference unavailable: <urlopen error [Errno 111] Connection refused>

## ④ 结果图

![fig_high_re_profile.png](fig_high_re_profile.png)

*图1 壁面模型闭合律剖面：mixing_length/log_layer 两条闭合在 Re_tau=1e4 的 u+(y+) 与预声明对数律、MKM-180 DNS 对照；灰带为判据带 y+=30-300。*

![fig_high_re_stress.png](fig_high_re_stress.png)

*图2 左：Re_tau=180 的雷诺剪应力 -<u'v'>+ 与生成项 P+（闭合模型 vs DNS）；右：两个量的峰值横坐标随 Re_tau 的漂移，DNS 峰值位置（点线）不随 Re 变化。*

![fig_reference_gate.png](fig_reference_gate.png)

*图3 参考数据完整性门：两条随仓发布的槽道 DNS 表。绿色 mkm180 通过；红色 mkm395 被机器拒绝（u+ 非单调、中线 15.96 远低于对数律估计 19.78、存储的 uv+ 冲到 6.0 越过物理上界 1）。*

![fig_flat_plate_cf.png](fig_flat_plate_cf.png)

*图4 零压力梯度平板局部摩阻系数（壁律+Coles 尾流积分模型）与 ITTC-1957 局部线、Prandtl-Schlichting 线对照，判据 (c) 的基准。*

![fig_profiles_vs_dns.png](fig_profiles_vs_dns.png)

*图5 壁律（Spalding / 纯对数律）与 MKM-180 DNS 的剖面对照，右图为由槽道精确应力平衡重构的雷诺应力与生成项。*

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **仓内 `mkm395.csv` 不是槽道剖面**：u+ 在 y+≈150 之后回落（中线 15.96，对数律估计 19.78），
   存储的 uv+ 单调涨到 6.0（物理上界 1）。它被完整性门拒绝，本轮任何误差都不使用它。
   若后续有任务以它为参照，结论应当重判——这是本轮最值得扩散的一条。
2. **判据带 30…300 在 Re_tau=180 只有 30…178 可用**，且 y+>0.3h+ 已是尾流。
   所以 (a) 的 DNS 腿必然包含尾流偏离；把它当作对数层常数错误来读数会重复上一轮的口径。
3. **壁模化槽道 LES 在本仓的组合下收敛域很窄**：Re_tau≥590 的算例在 1000-2000 步内发散
   （Cs=0.05/0.1 都试过），因此判据 (a) 的解析层（LES 实测 y+=30…300）本单 WITHHELD。
   这不是阈值挑选，是求解器边界，已写成稳定域工件。
4. **涡粘闭合结构上无法复现缓冲层应力峰**：-<u'v'>+=ν_t+·du+/dy+ 两项都非负，
   峰值横坐标由混合长增长 l+~κy+ 决定，随 Re_tau 漂移（本单 29.3→39.9），而 DNS 峰值不漂。
   要 (b) 真正通过，需要解析湍流或雷诺应力输运闭合，壁函数不够。
5. **KVLCC2 无实验参照**：本仓、船舶平台仓、平台 API、外网四条路都试过（检索记录在工件里），
   没有可溯源的拖曳 C_T。因此 (d) 的实验腿 WITHHELD，只用两条独立摩擦链互校。
   同时 C_R（剩余阻力）不是壁面模型能预报的量，本单不臆造。
6. **平板摩阻的 c_f 是局部量、C_F 是平均量**：KVLCC2 比较用的是平均量 C_F=2θ(L)/L（恒等式），
   两处不要混用。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3 %）

| 量 | 本单 | 基准 | 相对误差 | <3 %？ |
|---|---|---|---|---|
| 对数律带 u+（预测腿） | 见 ③ | MKM-180 DNS | 3.93 % | **否** |
| 对数律带 u+（形状腿，锚定后） | — | MKM-180 DNS | 1.72 % | 是（仅形状） |
| -<u'v'>+ 峰值位置（最优闭合） | y+=29.28 | DNS y+=31.98 | 8.43 % | **否** |
| 平板 c_f（局部） | — | ITTC-1957 局部线 | 3.63 % | 否（>3 %，但 ≤5 % 判据） |
| 平板 c_f vs Prandtl-Schlichting | — | 独立基准 | 3.65 % | 否（同上） |
| C_F(Re=1.3e7) 壁律模型 | 0.002995 | ITTC-1957 0.002868 | 4.44 % | **否**（4.4 %，≤7 % 判据） |

**结论（不粉饰）**：本轮 (c) PASS、(a)(b) FAIL、(d) 实验腿 WITHHELD 而摩擦腿 PASS(4.4 %)。
三个 <3 % 的问题都不是阈值问题：它们分别受限于参考带物理可用性、涡粘闭合的结构限制、
以及本仓 LES 的 Re_tau 收敛上限。

## 附：工件清单

```
artifacts/wall_model_les/acceptance.json
artifacts/wall_model_les/apriori.json
artifacts/wall_model_les/fig_flat_plate_cf.png
artifacts/wall_model_les/fig_flat_plate_sensitivity.png
artifacts/wall_model_les/fig_high_re_profile.png
artifacts/wall_model_les/fig_high_re_stress.png
artifacts/wall_model_les/fig_profiles_vs_dns.png
artifacts/wall_model_les/fig_reference_gate.png
artifacts/wall_model_les/flat_plate.json
artifacts/wall_model_les/flat_plate_profiles.npz
artifacts/wall_model_les/high_re.json
artifacts/wall_model_les/kvlcc2.json
artifacts/wall_model_les/log_law_fit.json
artifacts/wall_model_les/stability_map.json
artifacts/wall_model_les/stability_re1000_cs0.05.json
artifacts/wall_model_les/stability_re180_cs0.2.json
artifacts/wall_model_les/stability_re2000_cs0.05.json
artifacts/wall_model_les/stability_re590_cs0.05.json
```

