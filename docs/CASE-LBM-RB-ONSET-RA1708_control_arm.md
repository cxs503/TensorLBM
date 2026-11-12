# CASE-LBM-RB-ONSET-RA1708 — 控制臂核实与「驱动波数参考偏移」定量（诚实部分交付）

**锚点** `CASE-LBM-RB-ONSET-RA1708`（feature_ledger, status=open）· mission m16479 · repo TensorLBM · 分支 `agent/m16479`
**本单**：paper 席自主工时。上游 m16209（Route A ny=32 `Ra_c=1711.74`，+0.23%）把生产扫描、图、REPORT 留给后续；
m16337 交差失败（no_progress）。本单**不重复**那部分，只做上游从未做过的一件事：
**在拿 LBM 数字去比 1707.76 之前，先把「参考值该用哪个波数」这件事定量**，并独立核实控制臂。

---

## 1. 案例介绍（物理背景 / 几何 / 参数）

两层无限大刚性平行板、下热上冷、Boussinesq 流体；x 向周期 `Γ = nx/ny = 2`。
导热基态在 `Ra > Ra_c` 后线性失稳出二维定常辊。经典线性理论（Chandrasekhar 1961 §II，
刚-刚等温壁）：`Ra_c = 27π⁴/4 = 1707.7597`，对应的最不稳定波数 `a_c = kH = 3.11632`。

**本单要回答的隐患**：算例池的 LBM 生产臂用 `mode m = 1, Γ = 2`，即 `kH = 2πm/Γ = π = 3.14159`，
**并不在** `a_c` 上。若直接拿 `Ra_c(π)` 的 LBM 判决值去比 `1707.7597`，比较就带进一个
「参考值取错波数」的系统偏移。该偏移必须**先量化**，否则 2% 门内说不清哪部分是物理误差、
哪部分是参考口径误差。

| 参数 | 值 | 来源 |
|---|---|---|
| `Ra_ref`（教科书记载） | 1707.7597 | Chandrasekhar 1961 §II（`PREDECLARED['ra_ref']`） |
| `a_ref` | 3.11632 | 同上 |
| LBM 驱动波数 `a_lbm` | π = 3.141593 | `PREDECLARED['mode']=1`, `production nx/ny = 2` |
| `Pr`（Route C） | 0.71 | `PREDECLARED['pr']` |
| 控制臂网格 `M`（内部节点） | 50 / 100 / 200 | `PREDECLARED['control_grids']` |
| 控制臂门限 | 0.05% | `PREDECLARED['control_tol_pct']` |
| 算例门限 | 2% | `TOL_PCT` |

## 2. 怎么做的（方法 / 流程 / 工具链）

新脚本 `benchmarks/case_pool/rayleigh_benard_onset/control_arm_check.py`（只**编排**、不重写解法器）：

1. **Route C 控制臂**：调 `rb_continuum_stability`（N-S-Boussinesq 线性稳定性，二阶 FD 离散 +
   QZ 全谱 + Richardson 外推 `M→∞`）——完全独立于 LBM 算子，重算经典 `Ra_c`。
2. **驱动波数参考值**：调同一 continuum 稳定性求解器的 `_ra_c_at_wavenumber(a)`，在
   `a = a_ref`、`a = π` 两点求边际曲线值，并用中心差分给出 `d²Ra/da²|_π`（说明「平底」程度）。
3. 全部数字写 `artifacts/ra1708_control/control_arm.json`（含 `PREDECLARED` 输入、Python/NumPy 版本、
   耗时），图由该 JSON 直接生成（零手抄）。

复跑：

```bash
PYTHONPATH=src python3 benchmarks/case_pool/rayleigh_benard_onset/control_arm_check.py
python3 /tmp/fig_ctrl.py    # 或见 §6 说明
```

## 3. 计算结果

| 量 | 数值 | 与 1707.7597 之差 |
|---|---|---|
| Route C Richardson `M→∞` | **1707.7685** | **+0.00051%**（门限 0.05% → **PASS**） |
| 边际曲线 `Ra_c(a = a_c = 3.11632)` | 1706.4512 | −0.0767% |
| 边际曲线 `Ra_c(a = π = LBM 驱动)` | **1706.6159** | **−0.0670%** |
| `d²Ra/da²` at `a = π` | 491.74 | —（平底：`Δa = 0.0253` ⇒ `ΔRa ≈ 0.157`） |

解读（三条，全带数字）：

- **控制臂合格**：独立离散（FD+QZ+Richardson）复算 `Ra_c` 误差 `0.00051% ≪ 0.05%` 门，
  说明「控制臂失败则 LBM 结论无效」这条前置条件成立，LBM 侧的任何偏差都归不到「参考值本身算错」。
- **波数参考偏移可忽略但非零**：LBM 驱动的 `a = π` 处参考值应为 `1706.62`，比教科书记载的
  `1707.76` **低 0.067%**。相对 2% 判据，这个口径差占判据预算的 `0.067/2 = 3.4%`，
  因此**沿用 1707.7597 作参考是可接受的**，但报告必须写明（此前没有任何一单写过这个数）。
- **平底**：`d²Ra/da² ≈ 4.9e2`，`a` 偏离 0.8% 只带来 `Ra` 0.009% 的变化 ⇒ 用 `m=1` 单模驱动
  不会因「没打在 `a_c` 上」而系统性偏离 2% 门。

## 4. 结果图

`artifacts/ra1708_control/control_arm.png`（两联图，由 `control_arm.json` 直接绘制）：
左 = 边际曲线 `Ra_c(a)`（控制臂 5 点）叠 Richardson 值、教科书值、LBM 驱动波数点；
右 = 三个「对教科书值的偏差」柱状（Richardson / `a=π`）叠 ±2% 判据带。

## 5. 注意点与局限（诚实）

- **本单未完成 Route A/B 的生产扫描**：`run_case.py` 全矩阵（ny=32/48/64 二分 + Route B 谱 +
  两项敏感性）上游估计单节点约 25 min。本机 load15 ≈ 100（容量纪律：只做轻量活），
  故本单把重算力递交集群（作业 `cj1316`，`stage_dir` = 本工作树）并在本地以单线程 3 并行
  起了同一命令的**后台**运行（`runs/rb1708_run.log`，产物目录 `runs/case_rayleigh_benard_onset/`）。
  两者在本单收尾时**均未产出 `summary.json`**，因此 **LBM 判决值 `Ra_c` 本单不给**（不给未跑完的数字）。
- 已跑过的**单臂探针**（仅作可行性/成本标定，不是验收证据）：`ny=32` @ `Ra/Ra_ref=1.02`、
  2000 步 → `σ* = +0.1407`（`valid=True`，12.5 s）；`ny=64` 同条件 → `σ* = −4.69`（16.6 s，
  2000 步远未进入单模指数段，**不可解读为稳定判决**，只说明成本量级 ~8 ms/step）。
- 口径差异（上游 README 已记，本单沿用）：Route A 用实测间隙 `H_eff = ny−2`，Route B 用 `H = ny`，
  跨模块比较必须乘 `((ny−2)/ny)³`。本单**没有**触碰这条换算。
- `_ra_c_at_wavenumber` 单网格值（1706.45）比教科书低 0.077%，是 FD 离散残差，非物理；
  Richardson 值（1707.77）才是控制臂读数。
- 结论适用范围：刚-刚、等温壁、二维定常辊、`Pr = 0.71` 的**线性起始**问题；不含有限幅值、含
  overstable 分支（`Pr < 0.68` 附近）或三维胞。

## 6. 误差判定（逐项 vs 基准，显式回答「是否 <3%」）

| 关键量 | 计算值 | 基准 | 误差 | 门限 | <3%? |
|---|---|---|---|---|---|
| Route C `Ra_c`（Richardson） | 1707.7685 | 1707.7597（Chandrasekhar） | **+0.00051%** | 0.05% | **是**（且 <0.05%） |
| 驱动波数参考偏移 | 1706.6159 | 1707.7597 | −0.0670% | —（诊断量） | **是** |
| Route A `Ra_c`（LBM，主判据） | **未产出** | 1707.7597 | — | 2% | **无法判定（未跑完，不粉饰）** |

honest-FAIL 声明：**本单不构成锚点 `CASE-LBM-RB-ONSET-RA1708` 的关闭证据**。它关闭的是该锚点
上一处**从未被写下的口径隐患**（驱动波数 ≠ `a_c` 的参考偏移 = −0.067%）并独立复核了控制臂；
主判据（LBM `Ra_c` vs 1707.76 <2%）仍缺生产扫描，接棒单须先 `fetch cj1316` 或续跑
`runs/rb1708_run.log` 指向的后台矩阵，再出 `summary.json → make_figures.py → make_report.py`。

## 7. 复现与工件

```bash
# 本单（轻量，~75 s）
PYTHONPATH=src python3 benchmarks/case_pool/rayleigh_benard_onset/control_arm_check.py
# 上游测试面（不依赖重算）
PYTHONPATH=src python3 -m pytest tests/test_case_pool_rayleigh_benard_onset.py -q
```

| 工件 | 内容 |
|---|---|
| `artifacts/ra1708_control/control_arm.json` | Route C 数值、边际曲线 5 点、`d²Ra/da²`、参考偏移、provenance |
| `artifacts/ra1708_control/control_arm.png` | 边际曲线 + 偏差柱状图（±2% 带） |
| `benchmarks/case_pool/rayleigh_benard_onset/control_arm_check.py` | 本单新增编排脚本（不重写解法器） |
| `runs/rb1708_run.log` / `runs/case_rayleigh_benard_onset/` | 本地后台 Route A/B 矩阵（本单未跑完，见 §5） |