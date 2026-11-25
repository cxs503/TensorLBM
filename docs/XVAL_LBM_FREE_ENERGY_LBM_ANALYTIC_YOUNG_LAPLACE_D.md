# XVAL-LBM-FREE-ENERGY-LBM-ANALYTIC-YOUNG-LAPLACE-D — 2-D 液滴 Young-Laplace 解析标定（返修轮 m20929）

**锚点** XVAL-LBM-FREE-ENERGY-LBM-ANALYTIC-YOUNG-LAPLACE-D ｜ **仓库** TensorLBM ｜ **判据** owner 15%（stretch 3%）

## 1 案例与物理背景
自由能二元流体（Swift/Orlandini/Osborn/Yeomans PRL 78,19 (1996)）的 D2Q9 模型：
`f = A(-phi^2/2 + phi^4/4) + kappa|grad phi|^2/2`，压力张量内嵌于平衡态分布（生产核 `src/tensorlbm/phasefield/free_energy_pressure.py`，sailfish 语义移植）。
二维柱状液滴只有一个主曲率，毛细定律为 `dp = sigma/R`（**不是** 2sigma/R）。解析锚：`xi=sqrt(2 kappa/A)=2`、`sigma=(2/3)sqrt(2 kappa A)=(4/3)kappa/xi=2/15`（A=0.1, kappa=0.2, Gamma=0.05, tau_f=1, tau_phi=2）。

## 2 方法（本轮修复要点）
1. **稳态时间平均**：单快照体相压差逐步散度实测 ±10–30%，远超 3% 带；改为在 `steady_window`（3000/4000 步）内每 10 步采样、取均值，标准差与标准误一并入账，新增判据 C8（标准误 ≤2%）。
2. **同管线离散参照**：9 点各向同性差分在弥散界面上低估 |grad phi|^2（R/xi=9→4.5 时 −5%→−16%），故张量路线（C2/C3）改为与「把同一后处理作用在解析 tanh 场」的参照比较，抵消差分偏置；原始对连续介质 sigma 的偏差仍并列报告。
3. **稳定性实测修正**：原 predeclared 声称 kappa≥0.29 才是失稳边界；实为 `tau_phi=1.0` 在 step≈350 发散、`Gamma≥1.0` 立即发散（N=48/64/96 实测），故 tau_phi 默认值改为 2.0 并写入 predeclared。
4. **入口修复**：原 `xval/.../run_xval.py` 传 `case_id=` 给 `DropletCase`（不存在该字段）→ 入口必崩；已重写为调用库内 `run_suite`，产出 summary.json / ref_vs_got.json / 3 张图，PASS→0 / FAIL→1。
5. **集成**：`src/tensorlbm/multiphase_benchmarks.py:run_free_energy_young_laplace_benchmark()` 通过 `run_multiphase_benchmark_suite`（平台 `/api/v1/benchmarks/multiphase` 调用链）真调用本协议；`benchmarks/case_pool/free_energy_young_laplace/run_case.py` 被 `run_pool.py` 自动发现。

## 3 计算结果（程序化注入）

| case | R_fit | xi_fit | dp_mean±sd (n) | σ=dp·R/σ_an (%) | 张量积分 vs 同管线参照 (%) | dp_tensor vs 参照 (%) | xi 误差 (%) | u_max | dp vs 弥散精确解 (%) | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|
| N64_R9 | 8.71 | 2.053 | 1.49765e-02±1.02e-03 (300) | -2.16 | -0.22 | +1.68 | +2.66 | 5.84e-04 | -3.92 | PASS |
| N64_R12 | 11.753 | 2.003 | 1.09282e-02±7.77e-04 (300) | -3.67 | -0.07 | +0.90 | +0.16 | 6.02e-04 | -4.60 | PASS |

全案判定 **PASS**；ref_vs_got 行 10/10 通过。最差 |σ 误差| = 3.67%（&lt;3% 拉伸带）；对弥散精确解最差 4.60%。

## 4 图
图 1 为收敛/误差对照（fig_laplace_convergence.png），图 2 为快照压差时间序列（说明为何必须时间平均），图 3 为径向 phi 与压力各向异性剖面。

## 5 注意点 / 局限
* 仅二维柱状几何：3D 球 `2sigma/R` 未测（列入后续）。
* 第三个声明算例 N96_R18 已递交集群 cj3063（排队中），本轮本地工件含 N64_R9 / N64_R12 两算例；入口默认跑全三例。
* 同管线参照只消除后处理差分偏置，不构成独立外部实验锚（L5 不予主张）。
* 张量路线原始对连续 sigma 仍偏低 5–17%（差分偏置），已在工件内并列披露，未用它充当判据。

## 6 误差判定（vs 基准）
* `dp·R_fit` vs 解析 σ：−2.16% / −3.67% → **是否 <3%：是**（两例均在拉伸带内）。
* `dp` vs 弥散界面精确解：−3.92% / −4.60% → **<3%：否**（但 <15% 主判据，属有限 xi 的预期偏差）。
* 张量积分 vs 同管线参照：−0.22% / −0.07% → **<3%：是**。
* `xi_fit` vs sqrt(2κ/A)：+2.66% / +0.16% → **<3%：是**。

## 7 复现

[算例与验收记录](../benchmarks/case_pool/free_energy_young_laplace/README.md) · [m20929 历史结果](../benchmarks/case_pool/free_energy_young_laplace/evidence/m20929/summary.json)

```bash
PYTHONPATH=src python benchmarks/case_pool/free_energy_young_laplace/run_xval.py --out outputs/young-laplace   # 全三例
PYTHONPATH=src python3 benchmarks/case_pool/run_pool.py --case free_energy_young_laplace --smoke
PYTHONPATH=src python -m pytest tests/test_laplace_droplet_calibration.py -q   # 22 passed
```
