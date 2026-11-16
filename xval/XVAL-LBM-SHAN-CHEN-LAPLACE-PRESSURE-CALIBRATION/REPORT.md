# XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION — Cross-validation Report

> **Anchor**: `XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION` (mission m20921)
> **Reference practice**: `lbm-principles-practice/code` — SCMP Shan-Chen Laplace
> pressure calibration (multi-radius droplet, `Δp = σ/R` fit).
> **Criterion**: <15% (owner precision standard); the report also answers <3% explicitly.
> **Repo**: TensorLBM · worktree branch `agent/m20921`

## ① 案例介绍（物理背景 / 参数）

单组分 Shan–Chen 伪势模型（SCMP）在周期盒内自发形成液-气共存。盒中心放置一个静止液滴，
界面张力使液滴内部压力高于外部，二维圆柱界面满足 **Young–Laplace 律 Δp = σ/R**。
对多个半径测量稳态 Δp、线性拟合 σ，即「Laplace 压力标定」——这是 SC 类模型标定界面
张力最常用的做法（Shan & Chen 1993/1994；Krüger et al. *The LBM: Principles and Practice*
第 9 章及配套代码）。本单不复用 `benchmarks/verified/laplace_droplet` 的结论，而是**独立复核**它。

| 参数 | 值 | 说明 |
|---|---|---|
| 模型 | SCMP single-component Shan–Chen | D2Q9 |
| 伪势 | ψ(ρ)=1−e^(−ρ) | rho0=1 |
| 耦合 | G_lib = +5.0（= 教科书 G_eff = -5.0） | 库符号约定 G>0 为吸引 |
| 松弛 | τ = 1.0（ν=1/6） | BGK |
| 力格式 | velocity-shift（默认）/ 对照 guo | 见 ⑤ |
| 初值 | tanh 液滴，ρ_l=1.957，ρ_v=0.1596，宽度 4.0 | 实测共存密度 |
| 域 | 周期方盒，L = 4R | 与既有基准一致；另做 6R/8R 敏感性 |
| 半径 | R = 12, 16, 20, 24 | 4 个 |
| 压力 | p = c_s²ρ + (G_eff c_s²/2)ψ²，c_s²=1/3 | SC 状态方程 |
| 测量带 | 内 r ≤ 0.5R_eq，外 r ≥ 1.5R_eq；R_eq 由液相面积自洽反推 | 避开弥散界面 |

## ② 怎么做的（方法 / 流程）

三条**互相独立**的腿：

**L1 实现腿（从零复现）** — 在 `ref_sc.py` 中从教科书配方重写一套 SCMP D2Q9 求解器
（BGK + ψ=1−e^(−ρ) + 后向 gather 力 + 周期 streaming + velocity-shift），
**不 import 任何 tensorlbm 代码**，只沿用同一 D2Q9 速度排序以便逐格点比对分布函数。
两条实现跑完全相同的 4 个半径扫描；`tests/test_xval_sc_laplace.py` 强制逐格点等价
（步 0 与步 N 的 max|Δf|）。

**L2 热力学腿（解析基准）** — 在 SC 状态方程上做 Maxwell 等面积构造
（等压 + 等化学势 `∫ dp/ρ = 0`，在两条 spinodal 压力之间对 p_c 二分），
得到共存密度与共存压力的**解析**参考值，与格点模拟的体相密度对照。这条腿完全不含格点求解器。

**L3 协议腿（验收没探过的自由度）** — 同样收敛的液滴，改变 (a) 压力测量带，
(b) 周期域尺寸 L/R ∈ {4,6,8}，(c) 力格式 velocity-shift / guo，看标定出的 σ 变多少。

工具链：Python + NumPy（参考实现）/ PyTorch（TensorLBM 路径），matplotlib 出图；
`run_xval.py` 一键复算，所有数字写入 `artifacts/result.json`，报告数字由
`make_report.py` 程序化注入（零手抄）。

## ③ 计算结果

### 3.1 Maxwell 等面积构造（解析基准，SC EOS）

| 量 | 值 |
|---|---|
| ρ_v (coexistence) | 0.15289 |
| ρ_l (coexistence) | 1.92887 |
| p_c | 0.034214 |

### 3.2 液滴 Laplace 扫描（TensorLBM 路径，velocity-shift，L=4R）

| R_init | L | R_eq | Δp | ρ_in | ρ_out | σ_i = Δp·R_eq | dev vs through-origin fit | u_max | steps |
|---|---|---|---|---|---|---|---|---|---|
| 12 | 48 | 12.412 | 0.004448 | 1.9718 | 0.1619 | 0.05521 | -0.52% | 0.137 | 4000 |
| 16 | 64 | 16.195 | 0.003414 | 1.9628 | 0.1606 | 0.05529 | -0.38% | 0.139 | 4000 |
| 20 | 80 | 20.248 | 0.002760 | 1.9571 | 0.1598 | 0.05589 | +0.70% | 0.139 | 5000 |
| 24 | 96 | 24.227 | 0.002332 | 1.9533 | 0.1592 | 0.05650 | +1.81% | 0.139 | 4500 |

### 3.3 独立实现（ref_sc.py，从零复现）对照

| R_init | L | R_eq | Δp | ρ_in | ρ_out | σ_i = Δp·R_eq | dev vs through-origin fit | u_max | steps |
|---|---|---|---|---|---|---|---|---|---|
| 12 | 48 | 12.412 | 0.004448 | 1.9718 | 0.1619 | 0.05521 | -0.52% | 0.137 | 4000 |
| 16 | 64 | 16.195 | 0.003414 | 1.9628 | 0.1606 | 0.05529 | -0.37% | 0.139 | 4000 |
| 20 | 80 | 20.248 | 0.002760 | 1.9571 | 0.1598 | 0.05588 | +0.69% | 0.139 | 5000 |
| 24 | 96 | 24.227 | 0.002332 | 1.9533 | 0.1592 | 0.05651 | +1.81% | 0.139 | 4500 |

实现间偏差（同半径逐项）：max |Δp| = **0.007%**，
σ 偏差 = **0.001%**，
ρ_in 偏差 0.000%，
ρ_out 偏差 0.000%
（残余差异来自 float32 与 float64 的时间积分舍入，属数值噪声级）。

### 3.4 拟合与表面张力

| solver | σ (through origin) | σ (free intercept) | intercept | R² | max per-radius dev | <3%? | <15%? |
|---|---|---|---|---|---|---|---|
| ref_numpy | 0.05550 | 0.05386 | +1.01e-04 (2.34%) | 0.99988618 | 1.81% | YES (1.07%) | YES |
| tensorlbm | 0.05550 | 0.05386 | +1.02e-04 (2.34%) | 0.99988492 | 1.81% | YES (1.07%) | YES |

### 3.5 格点体相密度 vs 解析共存（L2）

| quantity | analytic (Maxwell on the SC EOS) | discrete droplet, R=24 | deviation |
|---|---|---|---|
| ρ_l | 1.92887 | 1.95330 | +1.27% |
| ρ_v | 0.15289 | 0.15922 | +4.14% |
| p_liquid | 0.03421 | 0.03735 | +9.17% |
| p_vapour | 0.03421 | 0.03502 | +2.35% |

液滴体相密度随 R 单调趋近解析共存值（曲率越大偏差越大），说明有限半径曲率效应是主因；
残余的 ρ_v 偏差（4.1%）
是**格点 SC 模型 EOS 相对解析 EOS 的离散偏差**，与库中「locked」实测值
（ρ_l=1.957, ρ_v=0.1596）自洽。

## ④ 结果图

- `artifacts/figures/fig1_droplet_field.png` — 场量云图（ρ(x,y)、p(x,y)）＋ 径向 ρ(r)、p(r) 剖面，
  标出 p_in / p_out 与解析 p_c。
- `artifacts/figures/fig2_laplace_fit.png` — Δp–1/R 拟合（两条实现、过原点与自由截距两种拟合）
  ＋ 逐半径残差。
- `artifacts/figures/fig3_sigma_vs_R.png` — σ_i = Δp·R_eq 的有限半径（曲率）趋势与 ±3% 带。
- `artifacts/figures/fig4_sensitivity.png` — 测量带 / 域尺寸 / 力格式三种敏感性。
- **图已入库**：上述 4 张 PNG 随 `artifacts/figures/` 一起提交，`artifacts/figures/PROVENANCE.json`
  记录每张图与 `artifacts/result.json` / `artifacts/field_snapshot_R20.json` 的 sha256；
  `tests/test_xval_sc_laplace_artifacts.py` 强制「报告引用的图必须存在且哈希自洽、REPORT.md 的数字与
  result.json 逐项一致、REPORT.md 可由 make_report.py 原样重生成」，防止报告与产物漂移。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **界面伪流速度不小**：所有稳态液滴的 u_max ≈ 0.14
  （≈0.24 c_s），集中在一两个格子的界面带上，且**与力格式无关**（guo 亦 0.15）。
  它不影响本报告的结论（Δp 由体相带平均得到，且域尺寸敏感性仅 0.45%），
  但它意味着「静止液滴」并非严格静止；任何依赖界面内速度/黏性耗散的算例都要注意。
2. **力格式会改变模型本身**：velocity-shift 与 guo 两种收敛态给出不同共存密度与不同 Δp
  （σ_i 差 56.8%）。因此 σ 的标定值**必须**与力格式绑定，
  不可跨格式搬用；库的 locked 值 0.0561 只对应 velocity-shift（τ=1 时 EDM 与其代数等价）。
3. **测量带**（内 0.5R/外 1.5R）是协议性选择，见下表；带越窄越贴近界面、越易被弥散界面污染。
4. **有限半径**：σ_i 随 R 单调上升（0.05521 → 0.05650），
  外推到 R→∞ 才是真实 σ；过原点拟合在 R∈[12,24] 上系统性偏低约
   -3.0%
  （自由截距拟合 σ = 0.05386）。
5. **2D 圆柱**：Δp = σ/R（不是 3D 的 2σ/R）；库 3D 版本另见 `benchmarks/verified/laplace_droplet`。

### 协议敏感性表

| R_init | Δp (0.4–1.6) | Δp (0.5–1.5) | Δp (0.6–1.4) | Δp (0.7–1.3) | spread |
|---|---|---|---|---|---|
| 12 | 0.004456 | 0.004448 | 0.004402 | 0.004163 | 6.72% |
| 16 | 0.003414 | 0.003414 | 0.003410 | 0.003377 | 1.08% |
| 20 | 0.002760 | 0.002760 | 0.002760 | 0.002754 | 0.22% |
| 24 | 0.002333 | 0.002332 | 0.002331 | 0.002329 | 0.19% |

| L/R | L | Δp | σ_i |
|---|---|---|---|
| 4 | 64 | 0.003414 | 0.05529 |
| 6 | 96 | 0.003421 | 0.05528 |
| 8 | 128 | 0.003429 | 0.05540 |

Spread over L/R ∈ {4,6,8}: **0.45%**.

| forcing | R_eq | Δp | ρ_in | ρ_out | σ_i | u_max |
|---|---|---|---|---|---|---|
| velocity_shift | 16.156 | 0.003421 | 1.9628 | 0.1605 | 0.05528 | 0.139 |
| guo | 18.089 | 0.001908 | 1.9114 | 0.1248 | 0.03452 | 0.151 |

Spread over forcing schemes: **56.79%**. The Guo run was re-checked to 16000 steps and is bit-stationary from step 4000 (same R_eq, Δp, ρ_in, ρ_out), so the difference is a converged model difference, not a transient.

## ⑥ 误差判定（逐项，对基准显式回答 <3%）

- **ref_numpy**: σ = 0.05550 vs library-locked σ = 0.0561 → **1.07%** deviation → <3% ✓; owner 15% standard: PASS. Laplace linearity R² = 0.99988618 (≥0.999 ✓).
- **tensorlbm**: σ = 0.05550 vs library-locked σ = 0.0561 → **1.07%** deviation → <3% ✓; owner 15% standard: PASS. Laplace linearity R² = 0.99988492 (≥0.999 ✓).

- **压强标定**：液滴 p_in/p_out 与解析 p_c 的差分别为
  +9.17% /
  +2.35%（R=24），
  符号与 Laplace 压力方向一致（曲率越大 p_in 越高），说明**压力标定自洽、无系统性偏移**。
- **实现间一致性**：σ 偏差 0.001% ⇒ **<3% ✓**（这条腿是严格的）。
- **对文献/既有基准**：本单 4 半径（小 R）过原点 σ = 0.05550；
  既有 `benchmarks/verified/laplace_droplet`（R=15/25/40）报 σ_eff = 0.0560–0.0566。二者差
  1.07%（<3% ✓），
  差异可由 R 区间不同（有限半径曲率趋势）解释，**既有结论在本次独立复核下成立**。
- **诚实说明**：σ 无第一性原理解析值（SC 伪势模型的界面张力依赖离散化细节，无闭式解），
  故以「实现间一致 + Laplace 线性性 + 解析 Maxwell 共存密度」三条独立基准构成判据；
  未做 3D 与更高分辨率（ν 扫描）复核，见 ⑦。

## ⑦ 未做 / 后续

- 3D 球形液滴（Δp = 2σ/R）的独立实现复核（本次只做 2D；库内 3D 基准已存在）。
- 分辨率/黏性扫描：σ 的 ν（τ）依赖在 SC 模型中已知存在，本轮 τ 固定为 1。
- 界面张力与格点离散化的关系（离散压力张量路线）仅作讨论，未作为判据纳入。
- guo 格式下体相 EOS 偏移的机理（半力修正与 ψ 力的耦合）未追到底，仅记录事实。

## 复算方式

```bash
cd xval/XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION
python run_xval.py --stage main          # L1/L2：4 半径 × 2 实现 + Maxwell
python run_xval.py --stage sensitivity   # L3：域尺寸 + 力格式
python make_figures.py && python make_report.py
pytest ../../tests/test_xval_sc_laplace.py -q
```

产物：`artifacts/result.json`（全部数字）、`artifacts/figures/*.png`、`REPORT.md`（本文件）。
