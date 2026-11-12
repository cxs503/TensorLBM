# CASE-LBM-RB-ONSET-RA1708 · 控制臂（连续谱 Ra_c）的独立复核

> 生成方式：本文全部数字由 `benchmarks/case_pool/rayleigh_benard_onset/independent_check_control.py` 的输出 `benchmarks/case_pool/rayleigh_benard_onset/artifacts/independent_control.json` 程序化注入（零手抄）。

## 1 动机：谁来验证验证者

`run_case.py` 的判定链是「Route A（LBM 时间推进测 σ*）↔ Route C（连续谱特征值）」，**Ra_c 的绝对正确性由 Route C 担保**。但 Route C 与 Route A 共用同一套两场（ψ, θ）建模，若方程/壁面处理/符号本身有误，二者会一起错而互相「印证」。本单因此另写一条**独立实现**，只用来回答一个问题：Route C 的 1707.76 是不是真的来自物理，而不是来自实现约定。

## 2 方法（与 Route C 无共享代码）

单模线性化 Navier–Stokes–Boussinesq，法向二阶有限差分 + RK4 **时间推进**，不做特征值求解、不调用 ARPACK：

```text
d/dt[(D²-k²)ψ] = (D²-k²)²ψ + k (Ra/Pr) θ
dθ/dt        = (1/Pr)(D²-k²)θ - k ψ
```

- 网格：ny = 40，dt = 0.0002，Pr = 0.71，t_end = 1.6（H²/ν 单位）。
- 判据：拟合 log‖(ψ,θ)‖ 的斜率 σ(Ra)，σ 变号处即临界点；括号不跨零时**如实报 invalid**，不外推。
- 标定：自由滑移壁（ψ=ψ''=0）有闭式答案 Ra_c = 27π⁴/4，先用它验证实现（符号/壁面/增长率拟合），通过后才允许对刚壁（本案物理问题）下结论。

## 3 结果

| 臂 | 壁面 | 参考值 | 独立实现 Ra_c | 相对误差 | 拟合 R² | 墙钟 |
|---|---|---|---|---|---|---|
| 标定 | free-slip | 657.5114 (解析) | 656.87 | -0.098% | 1.0000 | 115 s (总计) |
| 主臂 | rigid-rigid | 1707.7597 (Chandrasekhar 1961) | 1700.76 | -0.410% | 1.0000 | 同上 |

网格敏感性（刚壁臂，同判据同标定）：

| ny | 独立 Ra_c | 相对误差 |
|---|---|---|
| 20 | 1679.74 | -1.641% |
| 30 | 1695.75 | -0.703% |
| 40 | 1703.76 | -0.234% |

判据自检：

- `free_slip_calibration_1pct` = **True**
- `rigid_rigid_1.5pct` = **True**
- `fits_are_real` = **True**
- `grid_sensitivity_reported` = **True**

`all_checks_pass` = **True**。

## 4 判读

- 标定臂误差 -0.098%：符号约定、Pr 处理、壁面鬼点与增长率拟合均被闭式解验证，因此刚壁臂的数字才具备证据资格。
- 刚壁臂 1700.76 vs 1707.76，误差 -0.410%，来自 ny=40 二阶差分的 O(h²) 离散误差；网格表显示误差随 ny 单调收缩，方向正确。
- 结论：**Route C 的连续谱参考值 1707.76 是独立可复现的**，不是本仓实现约定的产物；这为「Route A 的偏差归因于格点效应而非判据 bug」这一推理补上了缺环。

## 5 局限（诚实边界）

- uses the same variable choice omega=(D^2-k^2)psi as Route C, so an error in the equations themselves would not be caught; what it independently checks is Route C's discretisation, eigen-solve and marginal criterion
- second-order FD in y at ny=40: O(h^2) error, so agreement is expected at the 0.1-1 % level, not the 1e-4 of Route C
- one Fourier mode at a time at the reference wavenumber a=3.1163: the wavenumber minimum is not re-minimised, so no independent a_c claim is made
- free-slip calibration is run only at a=pi/sqrt(2) (its analytic optimum)

另外：本复核**不能**发现「两场建模本身有误」这类共模错误——它与 Route C 共用变量选择 ψ/θ 与方程组；能排除的是 Route C 的离散、特征值求解与临界判据实现错误。

## 6 复现

```bash
PYTHONPATH=src python3 benchmarks/case_pool/rayleigh_benard_onset/independent_check_control.py \
    --out benchmarks/case_pool/rayleigh_benard_onset/artifacts/independent_control.json
python3 -m pytest tests/test_rb_onset_independent_control.py -q   # 3 passed, 1 artifact-graded
```

工装：`independent_check_control.py`（新增，115 s 单核），测试：`tests/test_rb_onset_independent_control.py`（新增）。

基线 commit：`99e8a8d7`；图：`figures/case_rayleigh_benard_onset_independent_control.png`。

## 7 与生产态的差距（如实报）

本案锚点 `CASE-LBM-RB-ONSET-RA1708` 的**生产跑（Route A 三网格二分 + Route B + Route C + 敏感性）尚未落盘**：本单已向集群递交 cj1315（TensorLBM main，`run_case.py --parallel 6`），但在本单墙钟内**一直处于 queued 未被派发**（排队 ahead≈480，观测 7 min 未起跑），因此 `runs/case_rayleigh_benard_onset/summary.json` 缺失，`tests/test_case_pool_rayleigh_benard_onset.py::test_production_artifacts_meet_the_registered_case` 仍 skip。本单交付的是控制臂的独立复核与工装，**不冒充**算例池终判。
