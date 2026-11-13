# CASE-LBM-KOVASZNAY-EXACT — 二维定常 Kovasznay 精确解流场（算例池条目）

锚点：`feature_ledger` slug **`CASE-LBM-KOVASZNAY-EXACT`**｜判据：**速度场 L2 相对误差 < 3 %（Re=40，N≥128）**
｜ref: **Kovasznay, L. I. G. (1948), _Laminar flow behind a two-dimensional grid_,
Proc. Cambridge Philos. Soc. 44:58–62, DOI 10.1017/S0305004100023999**
｜仓 TensorLBM · 分支 `agent/m17704` · mission m17704

> **全部数字来自 `cases/kovasznay_2d/RESULTS.md`（由 `make_report.py` 从
> `artifacts/kovasznay_2d/*.json` 程序化生成，零手抄）**；本 README 只做解释与判读。

## 裁决（先给结论）

1. **锚点判据达成**：ny=128、Re=40、xmax=3 的全场速度 L2 相对误差
   u = **0.1601 %**、v = **2.5613 %**，裁据量 max = **2.5613 % < 3 %**（余量 0.439 个百分点），
   `extrap: none`，verdict = **PASS**。
2. **但 PASS 的物理含义被本单改写**：把出口一层（1 个 y 周期）从度量域里剔除后，
   全场 v 误差的 **99.6 %–99.9 %（按 L2 能量分解）** 来自出口零梯度边界层，
   内部误差只有 **0.10 %–0.18 %**。也就是说 **2.56 % 不是"离散精度"，是"出口边界层"**。
3. **新给出的定量规律**：全场误差随下游域长按 **v ∝ xmax^(−2.08)** 衰减（ny=32/64 两个网格
   独立拟合到 −2.08）。**xmax=2 时同一算例是 7.15 %（判据 FAIL），xmax=3 恰好 2.71 %（擦线 PASS），
   xmax=4 是 1.04 %**。→ 该算例的 3 % 通过是**域长选择**的结果，不是精度裕量的结果。
4. **内部误差与网格无关（诚实负面结论）**：内部 L2 在 ny=32/64/128 基本不动
   （u 0.1102→0.1023→0.1017 %，p ≈ 0.11→0.01），**因此本算例在其出厂工况下不能给出
   收敛阶**。原因经 Mach 阶梯定量定位：**内部误差是 O(Ma²) 可压缩性底噪**——
   固定网格固定 Re、把 U0 减半（Ma 0.052→0.026）内部误差降 **3.5×（u）/ 3.7×（v）**，
   Ma² 预测为 4×。
5. **压不下去的墙（诚实局限）**：继续降 Ma（U0=0.0075，τ=0.536 / 0.518）BGK 直接发散
   （NaN）。**BGK + Zou-He 入口 + 零梯度出口在 τ ≲ 0.57 失稳**，所以用这套组合无法把 Ma² 底噪压到
   离散误差之下。要把它变成真正的"收敛阶算例"，需要换碰撞模型（TRT / regularized）
   或改用体积力驱动的全周期 formulation —— 列为后续条目建议，本单不做。

---

## ① 案例介绍

二维定常不可压 N-S 精确解（格栅后层流尾迹；y 向周期 1，x′ 向指数衰减）：

    u = U0(1 − e^{λx′}cos 2πy′),  v = U0(λ/2π)e^{λx′}sin 2πy′,
    p = ½U0²(1 − e^{2λx′}),        λ = Re/2 − √(Re²/4 + 4π²)   （λ² − Re·λ − 4π² = 0 的负根）

| 参数 | 值 | 说明 |
|---|---|---|
| 格子 / 碰撞 | D2Q9 / BGK | ω = 1/τ，ν = (τ−½)/3 |
| Re / U0 | 40.0 / 0.03 | λ = **−0.9637405**（精确根），Ma_max = 0.1039 |
| 域 | nx = xmax·ny，xmax=3，y 向周期 | 主配置；本单额外扫 xmax ∈ [1,4] |
| 入口 x=0 | 解析 Zou–He 速度入口（`boundaries.zou_he_inlet_velocity`） | 每行给解析 u、v |
| 出口 x=nx−1 | 零梯度 Neumann（`boundaries.zero_gradient_outlet_2d`） | 本单的被诊断对象 |
| 网格 | ny = 32 / 64 / 128（nx = 96 / 192 / 384） | 判据取 ny=128 |
| 稳态 | 场漂移监测（每 1000 步），末 100 步时间平均 | 实际 6000–7000 步 |

**参考解的证书升级**：旧 benchmark 文档用**差分**核 N-S 残差、报 ≈1e-3（读起来像"参考解只是近似解"）。
本条目改用**解析导数**在无量纲形式（U·∇U = −∇P + (1/Re)∇²U）下核：
动量残差 **2.4e-15 / 3.3e-16**、连续性散度 **0.0**、λ 相容性残差 **8.5e-14**
⇒ **参考解是精确解，测量到的误差全部归于数值**。

**任务书勘误（再次确认）**：旧任务书写"Re=40 时 λ≈−0.5"是错的；λ=−0.5 对应 Re ≈ 78.46。

## ② 怎么做的

- 工具链：`cases/kovasznay_2d/analytic.py`（解析解 + 解析残差证书）·
  `run.py`（步进/度量/CLI：`check|single|gate|study|ma`）·`make_report.py`（汇总+出图+生成 RESULTS.md）。
- 求解链：`collide_bgk → stream(周期 gather) → 零梯度出口 → Zou–He 入口`，
  全部是**库内公开函数**（无内联 hack），float64 步进，末 100 步 float64 平均去浮点噪声。
- 度量：全场 u/v L2 相对误差；**内部窗口** = 列 `[1, nx−ny)`（剔除入口列 + 出口 1 个 y 周期）；
  掩模最大点相对误差（|u_ana|>0.1·max|u|）；质量漂移。
- 三个**单变量**实验臂（每次都只动一个变量）：
  A. **xmax 扫描**（ny=32/64 固定，xmax ∈ {1,2,3,4}）→ 分离出口边界项；
  B. **网格细化**（xmax=3 固定，ny ∈ {32,64,128}）→ 观测阶 p；
  C. **Mach 阶梯**（ny=64 固定、Re 固定，U0 ∈ {0.03,0.015,0.0075}）→ 分离 O(Ma²) 与 O(h²)。
- 复现（每条 ≤ 几分钟）：
  ```bash
  python cases/kovasznay_2d/run.py check                                   # 参考解证书
  python cases/kovasznay_2d/run.py gate 128 artifacts/kovasznay_2d/gate.json
  python cases/kovasznay_2d/run.py study 128 artifacts/kovasznay_2d/study_raw.json \
      --steps 6000 --threads 8 --cache artifacts/kovasznay_2d/cache        # 可断点续跑
  python cases/kovasznay_2d/run.py ma 64 /tmp/ma.json --steps 6000 --cache artifacts/kovasznay_2d/cache
  python cases/kovasznay_2d/make_report.py                                 # → results.json + RESULTS.md + 图
  python -m pytest -q tests/test_case_pool_kovasznay_2d.py                 # 5 passed / ~40 s
  ```
- **交叉校验**：本单独立实现给出的 ny=64、xmax=3 全场误差 0.1653 % / 2.7095 %，
  与 `benchmarks/verified/kovasznay_2d/result.json` 记录的 0.17 % / 2.71 % 一致（3 位有效数字）。

## ③ 计算结果

完整表格见 **[`RESULTS.md`](RESULTS.md)**（自动生成）。摘要：

| 臂 | 关键量 | 数值 |
|---|---|---|
| 锚点 gate（ny=128, xmax=3） | u / v 全场 L2 | **0.1601 % / 2.5613 %** → PASS |
| xmax 扫描（ny=64） | v 全场：xmax=1/2/3/4 | 20.09 / 7.15 / 2.71 / 1.04 % |
| xmax 扫描（ny=64） | 对数斜率 | **−2.08**（u: −1.80） |
| 出口占 v 误差 | 能量分解 | **99.63 %–99.95 %** |
| 内部窗口（ny=32→64→128） | u / v | 0.1102→0.1023→0.1017 % / 0.1764→0.1111→0.1152 % |
| 观测阶 p（内部） | u 32→64 / 64→128 | 0.107 / 0.0096（**≈0，网格无关**） |
| Mach 阶梯（ny=64） | 内部 u / v 随 U0 减半的比 | **3.51 / 3.74**（Ma² 预测 4.0） |
| 稳定性墙 | τ=0.536（ny=64,U0=0.0075）/ τ=0.518（ny=32） | 均发散 NaN；最后稳定 τ = 0.572 |

## ④ 结果图

`figures/kovasznay_case_pool.png`（`make_report.py` 生成，4 面板）：
(a) ny=64 数值速度幅值场 + 解析 v 等值线；(b) 误差场 100·(|Δu|+|Δv|)/max|u_ana|（虚线标出被剔除的出口带）；
(c) 双对数：全场 v/u 误差 ∝ xmax^(−2.08) 与内部误差（平）＋ 3 % 判据线；
(d) 网格细化：全场下降 / 内部网格无关 / U0 减半的内部误差线（Ma² 证据）。

## ⑤ 注意点（坑与适用范围）

1. **内部窗口不能取"去掉 1 列"**：旧代码里的 `interior` 是 `[1, nx−2)`（只去掉 1 列），
   仍含整个出口边界层，会让人误以为"内部也差 2.5 %"。必须剔除 ≥1 个 y 周期。
2. **xmax=1 时内部窗口退化**：nx = ny 时窗口按半域截断（列 1..nx/2），仍被出口污染
   （v 内部 1.35 %），此时"内部误差"不可用——表里已标 `n_cols`。
3. **掩模最大点相对误差在 v 零穿越处不可靠**（m16861 已指出 2.66 % vs 1.49 % 的口径差）；
   本条目的判读一律以 **L2** 为准。
4. **Ma 阶梯不能随便加档**：τ ≲ 0.57 时 BGK + 本算例 BC 组合失稳（实见 NaN），
   这是"降 Ma 压底噪"路线的硬墙，不是实现 bug。
5. 出口闭合的备选（nscbc / NEE / 解析 Dirichlet）**都更差或非物理**，见
   `docs/CASE-LBM-KOVASZNAY-OUTLET-BC-LADDER.md`（honest 负面结果，本单不重复）。
6. 本条目**没有**改碰撞模型、没有改判据、没有外推（`extrap: none`）；判据仍是原锚点。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 < 3 %）

| 关键量 | 参考值 | 本单计算 | 误差 | < 3 % ? |
|---|---|---|---|---|
| u 全场 L2（ny=128, 判据量） | 解析解（残差 2.4e-15 ⇒ 精确） | 0.1601 % | **0.1601 %** | **是** |
| v 全场 L2（ny=128, 判据量） | 同上 | 2.5613 % | **2.5613 %** | **是**（余量 0.44 pp） |
| 判据量 max(u,v) | — | 2.5613 % | — | **是**（verdict PASS） |
| u / v 内部 L2（ny=128） | 同上 | 0.1017 / 0.1152 % | — | 是（且与网格无关） |
| λ（Re=40） | −0.9637405（公式精确根） | −0.9637405441957689 | 0（机器精度） | 是 |
| 参考解连续 N-S 残差 | 0（精确解） | 2.4e-15 / 3.3e-16，散度 0.0 | — | 是 |
| ny=64 全场 v（跨实现校验） | benchmark 记录 2.71 % | 2.7095 % | 差 0.001 pp | 是 |

**是否 < 3 %？→ 是**（判据量 2.5613 %）。但必须同时说明：这 2.56 % **几乎全是出口边界层**（99.6 %），
内部离散精度实际 **< 0.12 %**，而内部误差本身被 **O(Ma²)=0.10 % 的可压缩性底噪**占据，
**因此"PASS"不等于"该算例已给出收敛阶"**——收敛阶在本工况下不可测（p≈0），
要测必须先解决 τ≲0.57 失稳（换 TRT/regularized 碰撞或体积力驱动全周期形式）。

## 参考

- Kovasznay, L. I. G. (1948). Proc. Camb. Phil. Soc. **44**, 58–62. DOI 10.1017/S0305004100023999.
- Zou, Q. & He, X. (1997). Phys. Fluids **9**, 1591–1598.
- 既有证据：`benchmarks/verified/kovasznay_2d/`（首次验证）、
  `docs/CASE-LBM-KOVASZNAY-EXACT_independent_audit.md`（m16861 独立复核 + 误差归因）、
  `docs/CASE-LBM-KOVASZNAY-OUTLET-BC-LADDER.md`（m16974 出口口径阶梯，负面结果）。
- 本条目新增：解析残差证书、xmax 标度律、网格无关性、Mach 阶梯与 τ 失稳墙。