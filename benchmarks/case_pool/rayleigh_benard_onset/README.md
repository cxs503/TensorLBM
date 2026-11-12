# CASE-LBM-RB-ONSET-RA1708 — Rayleigh-Bénard 对流起始临界 Rayleigh 数（TensorLBM · 算例池）

> 算例池记录：`case_code=CASE-LBM-RB-ONSET-RA1708`，repo=TensorLBM，
> ref=Chandrasekhar (1961) *Hydrodynamic and Hydromagnetic Stability* §II 线性理论（刚-刚/等温壁 2-D 辊
> `Ra_c = 27π⁴/4 = 1707.76`、`a_c = 3.11632`），tol_policy=<2%。
> 本目录是**复用型**执行器：不新写解法器，直接驱动仓内既有的三条能力，按**静止态线性稳定性判据**
> 对 Ra 做二分扫描求 `Ra_c`，并与经典值逐项对照。
> 全部数字由 `make_report.py` 从 `summary.json` 程序化注入（零手抄），见 `REPORT.md`。

## 算什么

两层无限大刚性平行板间的 Boussinesq 流体，下热上冷（ΔT），x 方向周期、Γ=nx/ny=2。
静止（纯导热）基态在 `Ra > Ra_c` 后线性失稳，先出现二维定常辊。
本案问的是**起始临界 Rayleigh 数**：导热基态上无穷小扰动的增长率 σ 随 Ra 变号的位置。

判据（`run_case.py::PREDECLARED`，运行前冻结）：

```text
sigma* = sigma H^2 / alpha   (单模投影 A(t) 的 log 斜率)
unstable  : sigma* > +0.02      stable  : sigma* < -0.02      marginal : |sigma*| <= 0.02
Arm 否决门 : valid=False 或 fit_r2 < 0.999 的臂不投票
扫描       : 先验证括号 Ra/Ra_ref in [0.90, 1.10] 两端异号，再 Ra 二分 + sigma 零点插值
             （中点回退），停在 (hi-lo)/mid < 0.2%
Ra_c       : 最终括号内 sigma*(Ra) 的零点插值；不确定度 = 括号半宽（或 marginal 带折算宽度）
通过线     : 主网格 ny=64 的 Ra_c 与 1/ny 外推 Ra_inf 同时 <2%；连续谱控制臂 <0.05%
```

## 三条证据面

| 路径 | 仓内实现 | 作用 |
|---|---|---|
| **Route A（主，本案指标）** | `tensorlbm.rb_convection_validation.measure_growth_rate`（`Convection2D`：D2Q9 + 热标量，BGK + Boussinesq，pre-streaming 半程反弹壁） | LBM 时间推进测 σ*，二分扫 Ra 定 Ra_c |
| **Route B（交叉）** | `tensorlbm.rb_discrete_stability.critical_rayleigh`（同一格点步进算子的 FD Jacobian 谱） | 不推进时间，直接对 `γ(Ra)=0` 二分；校验 σ* 观测是否就是算子增长率 |
| **Route C（控制）** | `tensorlbm.rb_continuum_stability`（N-S-Boussinesq 线性稳定性，二阶 FD + QZ 全谱 + Richardson） | 独立于 LBM 重算经典 `Ra_c`；控制臂失败则 LBM 结论无效 |

**口径告警**：Route A 的 Ra 用实测流动间隙 `H_eff = ny-2`，Route B 用 `H = ny`
（`rb_discrete_stability.RBLatticeParams`）。跨模块比较必须乘换算因子 `((ny-2)/ny)³`
（ny=16 时 0.67）。本单在 `routeB_lattice_eigen.json` 里同时给出两种口径。

## 跑

```bash
# 生产（三网格二分 + 交叉臂 + 控制臂 + 两项敏感性；约 25 min 单节点）
PYTHONPATH=src python3 benchmarks/case_pool/rayleigh_benard_onset/run_case.py \
    --out-dir runs/case_rayleigh_benard_onset --parallel 3

# 快速子集（测试用，ny=16 二分 + 一个控制网格；约 1-2 min）
PYTHONPATH=src python3 benchmarks/case_pool/rayleigh_benard_onset/run_case.py --quick

# 图 + 报告（数字全部从 summary.json 注入）
PYTHONPATH=src python3 benchmarks/case_pool/rayleigh_benard_onset/make_figures.py \
    --run-dir runs/case_rayleigh_benard_onset --asym
PYTHONPATH=src python3 benchmarks/case_pool/rayleigh_benard_onset/make_report.py \
    --run-dir runs/case_rayleigh_benard_onset
```

一键：`bash benchmarks/case_pool/rayleigh_benard_onset/repro.sh`

## 测试

```bash
python3 -m pytest tests/test_case_pool_rayleigh_benard_onset.py -q          # 判据/二分逻辑（秒级）
RB1708_SLOW=1 python3 -m pytest tests/test_case_pool_rayleigh_benard_onset.py -q -k slow   # 端到端 LBM（~2 min）
```

* 判据逻辑用**解析已知零点**的合成 σ(Ra) 检验（括号验证、marginal 带、否决门、停机规则、外推形式），
  不依赖 LBM，因此能在秒级抓住判据实现的 bug（首版即被该测试抓出「零点插值被括号中点稀释」的偏差）。
* 工件层测试从 `summary.json` 复检登记判据（控制臂、主网格 <2%、括号真跨零、投票臂 fit_r2）。

## 文件

| 文件 | 内容 |
|---|---|
| `run_case.py` | 驱动器：判据/二分/三路线/敏感性/判定（`PREDECLARED` 冻结阈值） |
| `make_figures.py` | σ*(Ra) 扫描图、网格收敛图、σ*(a) 抛物线、控制臂图、超临界辊场云图 |
| `make_report.py` | 从 `summary.json` 渲染 `REPORT.md`（六段式）+ 平台报告 payload |
| `repro.sh` | 一键复现（生产 + 图 + 报告） |
| `REPORT.md` | 自动生成的论文级六段报告（数字零手抄） |