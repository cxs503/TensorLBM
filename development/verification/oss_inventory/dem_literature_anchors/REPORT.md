# XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY — 第 2 轮（调研结合）：没有 OSS DEM 孪生时，DEM/LBM 耦合怎么验

`repo TensorLBM · branch agent/m18215 · mission 18215 · 2026-11-13 · anchor XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY`

一句话：**OSS 台账里确实没有 DEM 孪生仓（证据见 §1/`survey_oss.md`），所以本锚点改用
"闭式解 + 文献曲线"分层验收**。DEM 侧三条非自明臂全部落在 15% 内（接触时长 −0.63%、
峰值重叠 +0.15%、恢复系数 +8.90%、沈降终速 vs Clift–Gauvin ≤ 5.1e-6 相对误差），
动量交换账本残差 **0.0**；同时**如实记一条不达标**：流体侧 LBM 限域盒 Cd 在 Re=50 偏
**−28.5%**，故本轮**不**宣称"LBM–DEM 整链已对标"，只宣称"DEM 侧已对标 + 流体侧体检结果"。

---

## ① 问题设定 + 基准出处

**物理背景**：TensorLBM 的 DEM 腿（`src/tensorlbm/dem/`）给冰区/颗粒流提供刚体颗粒动力学，
流体侧力通过阻力律注入。上一轮（m18066）想让 DEM 腿与 OSS 孪生仓互校，结果为 honest-FAIL 且
未留 stage/reason；本轮第一步是把这个前提**证伪或证实**：

| 检查 | 命令 | 结果 |
|---|---|---|
| 缓存仓数量 | `ls -1 /root/oss-cache \| wc -l` | `3` |
| 仓名/版本 | `git -C … log -1` | `Maarten-vd-Sande_lbm@573a01c`、`pylbm@f9f5d16`、`hschwane/GraSPH@ae8c6ff` |
| DEM/颗粒接触代码 | `grep -ril -E "discrete element\|spring.?dashpot\|granular\|contact model\|restitution" --include=*.py --include=*.f90 --include=*.cpp /root/oss-cache` | **无输出** |

⇒ 缺口 `NO-DEM-REPO-IN-OSS-INVENTORY` **成立**，常规"读 OSS 源码逐行对标"的打法在此不可用。

**基准出处（external_anchor，逐条）**：Cundall & Strack 1979（线性弹簧-阻尼 DEM）；
Di Renzo & Di Maio 2004（`e = exp(−πζ/√(1−ζ²))`）；Tsuji et al. 1993（软球+LBM 耦合范式）；
Coetzee 2017（标定层级：闭式→单元→整体）；Schiller & Naumann 1935 与 Clift–Grace–Weber 1978
（阻力曲线）；Ladd 1994（动量交换）；Hasimoto 1959（周期限域修正）；ten Cate et al. 2002
（单球沉降标准算例，本单只取其**无界极限**，不手抄实验表）。全文见 `lineage.md`。

**验收判据（先于计算声明，`predeclared.json`）**：owner 精度标准 `|sim−ref|/|ref| < 15%`；
in-house 对闭式接触臂更严 2%；并预先声明 H3——Stokes 臂与求解器**同式**，属自明性检查，
**不得**充当物理验收；H4——若流体侧 LBM 自身 >15%，必须报出、不得选有利频段掩盖。

## ② 方法与实现

驱动 `xval/XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY/run_xval.py`（全部走**本仓真实模块**：
`tensorlbm.dem.DEMSolver`、`dem.contact.LinearSpringDashpot`、`dem.cw_coupling.MomentumExchange`；
参考值一律在本文件内**独立写出**，不从被测代码导入）：

| 臂 | 算式 | 实现指针 |
|---|---|---|
| A/B 接触 | Kelvin–Voigt `f_n = kδ − η v_n`，`ζ = η/(2√(km))`；解析 `e=exp(−πζ/√(1−ζ²))`、`t_c=π/ω_d`、`δ_max=(v0/ω_d)e^{−ζω_n t*} sin ω_d t*`，`t*=atan(ω_d/(ζω_n))/ω_d` | `contact_arm()`、`analytic_contact()` |
| C Stokes 沈降 | `|Δρ| g D²/(18μ)`（DEM `drag_model="stokes"`，浮力折入 `g_eff=g(1−ρ_f/ρ_p)`） | `settling_arm()` |
| D SN 沈降 | 求解器用 `Cd=24/Re(1+0.15Re^0.687)`，参考为 Clift–Gauvin 同 Re 终速（`drag_closure.terminal_velocity(model="clift_gauvin")`） | `settling_arm()` |
| E 流体侧 | 读本仓既有 LBM 限域盒结果 `benchmarks/pending/sphere_drag_closure/results.json` 的 Cd(Re) 与 Clift–Gauvin 比 | `lbm_leg()` |
| F 动量账本 | `MomentumExchange.record()` 记录粒子冲量，`water_reaction()` **独立重算**水侧反冲量，残差应→0 | `ledger_arm()` |

关键控制：**A 臂的 `clip=False` 对照模型**（`_NoClipKelvinVoigt`）——只去掉
`f_n = max(0, …)` 的无张力限幅，其余完全不变，用来把"限幅偏置"与"积分器误差"分开。
（这也是本轮的方法学收获：一个 `max(0,·)` 就让恢复系数偏 8.9%，见 ③。）

## ③ 参数表（无量纲量齐全）

| 量 | 取值 |
|---|---|
| 接触：`k` / `η` / `ζ` / `m` / `ω_n` | `1e5 N/m` / `0.2479 N·s/m` / `0.2` / `3.8401e-6 kg` / `1.6142e5 rad/s` |
| 接触：`Δt/t_c` | `1/40, 1/80, 1/160` |
| 接触：`δ_max/R` | `2.0e-4`（小变形，线性模型适用区） |
| 碰撞速度 `v0` / 半径 `R` | `0.05 m/s` / `1e-3 m` |
| 沈降：`ρ_p/ρ_f/μ/g` | `917 / 1000 kg/m³`、`1e-3 Pa·s`、`9.81 m/s²` |
| 沈降：`D` | `1e-4, 2e-4, 5e-4 m` |
| 沈降：`Δt/τ_p`（`τ_p=2ρ_p a²/(9μ)`） | `1/25, 1/100` |
| 沈降：`Re_measured` | `0.0444, 0.3378, 4.0598`（Clift–Gauvin 参考区间 0.04–4） |
| 账本：`C_w` / `Δt` / 步数 | `5.5e-3` / `1e-5 s` / `50` |

## ④ 对照表（ref vs got，逐项打勾）

**A/B 接触臂**（解析 e=0.526621，解析 t_c=1.9872e-5 s）

| `t_c/Δt` | `e_sim` | e 误差 | `t_c` 误差 | `δ_max` 误差 | 判定 |
|---|---|---|---|---|---|
| 40 | 0.579295 | **+10.00%** | −2.50% | +0.68% | 15% ✓ / 2% ✗ |
| 80 | 0.575345 | **+9.25%** | −1.25% | +0.32% | 15% ✓ / 2% ✗ |
| 160 | 0.573489 | **+8.90%** | **−0.63%** | **+0.15%** | 15% ✓ / 2% ✗ |

**A-控制臂（去掉无张力限幅，其余不变）**

| `t_c/Δt` | `e_sim` | e 误差 |
|---|---|---|
| 40 | 0.535948 | **+1.77%** |
| 80 | 0.531169 | **+0.86%** |
| 160 | 0.528867 | **+0.43%** |

⇒ `t_c`、`δ_max` 随 Δt **线性收敛**（−2.50/−1.25/−0.63%；+0.68/+0.32/+0.15%），说明积分正确；
`e` 的 +8.9% 残差**几乎全部来自 `max(0, kδ−η v_n)` 限幅**（去掉后 0.43%）。这是缺陷级的
模型偏置（恢复系数被系统性抬高，影响碰撞能量账），本单如实报告，**未**在本轮修（见 §6）。

**C/D 沈降臂（DEMSolver 真实积分，尾窗漂移 ≤1.2e-7）**

| `D [m]` | 臂 | `u_sim [m/s]` | 参考 [m/s] | 相对误差 | 判定 |
|---|---|---|---|---|---|
| 1e-4 | C Stokes | 4.52350e-4 | 4.52350e-4 | −0.0 | 自明 ✓（H3） |
| 2e-4 | C Stokes | 1.80940e-3 | 1.80940e-3 | −0.0 | 自明 ✓ |
| 5e-4 | C Stokes | 1.130875e-2 | 1.130875e-2 | −0.0 | 自明 ✓ |
| 1e-4 | D SN vs CG | 4.444970e-4 | 4.444970e-4 | **−0.0** | ✓ |
| 2e-4 | D SN vs CG | 1.689177e-3 | 1.689177e-3 | **+0.0** | ✓ |
| 5e-4 | D SN vs CG | 8.119628e-3 | 8.119587e-3 | **+5.1e-6 相对** | ✓ |

**F 动量账本**：`relative_residual = 0.0`（`n_events=50`，量级 ~1e-2 N·s 的冲量对消到机器零）✓

**E 流体侧（不设门，体检）**

| Re | `Cd`(LBM, 运行 λ) | `Cd`(Clift–Gauvin) | 误差 | 判定 |
|---|---|---|---|---|
| 1.009 | 31.874 | 27.366 | **+16.5%** | ✗ |
| 2.016 | 17.794 | 14.797 | **+20.3%** | ✗ |
| 5.037 | 7.795 | 6.935 | +12.4% | ✓ |
| 10.078 | 4.739 | 4.128 | +14.8% | ✓（分辨率外推值 4.311 → +4.4%） |
| 20.147 | 2.520 | 2.598 | −3.0% | ✓ |
| 50.366 | 1.095 | 1.533 | **−28.5%** | ✗ |

⇒ **4/6 在 15% 内，最差 −28.5%（Re=50）**。根因：每个 Re 只有一个域尺寸（λ=D/L=1/8…1/6），
没有做 λ-外推；Hasimoto 修正（因子 1.518）在 Re≥5 明显过修（−20…−53%），只能当诊断用。
**故本轮不宣称 LBM 侧可作 DEM 阻力律的 <15% 参考。**

## ⑤ 收敛性 / 参数敏感性

* 接触：Δt 从 `t_c/40 → t_c/160`（×4 细化），`t_c` 误差 −2.50→−1.25→−0.63（一阶，比值≈2），
  `δ_max` 0.68→0.32→0.15（同样一阶），`e` 10.00→9.25→8.90（指向 ≠0 的平台，符合限幅偏置的签名）。
* `e` 的极限估计：用 clip=False 的一阶外推 `e(Δt→0) ≈ 0.5280`（0.528867 − (0.535948−0.528867)/3）
  vs 解析 0.526621 → **+0.26%**；clip 臂的极限 ≈ `0.573489 − (0.579295−0.573489)/3 = 0.5716`
  → **+8.5%**，与外推前一致 ⇒ 偏置是模型性的，不是离散性的。
* 沈降：`Δt/τ_p` 从 1/25 → 1/100 两档，终速在 8 位有效数字内不变（一阶迎风式指数松弛的稳态
  是不动点，离散误差只影响暂态），尾窗漂移 ≤1.2e-7 ⇒ 收敛到稳态。
* 沈降对 `D` 的敏感性：`u_t ∝ D²`（Stokes）在 1e-4→2e-4 上比值 4.000（4.5235e-4 vs 1.8094e-3），
  与解析缩放一致；5e-4 处 SN 与 Stokes 分离 28%（Re=4 惯性修正），与文献趋势一致。

## ⑥ 失败模式与工件

**诚实不达标（3 条）**

1. **接触恢复系数 +8.90%**：过 owner 15%，**未过** in-house 2%。根因 = `LinearSpringDashpot`
   的 `f_n = max(0, kδ − η v_n)` 在回弹相把阻尼截掉（160 步接触里有 20 步触发），
   使碰撞能量被人为抬高。控制臂证明去掉限幅即回到 +0.43%。**修法已明（限幅改为只在 δ<0 时截断，
   或分离相使用速度相关的张力上限），但本轮只做验证不改本构**——改本构要连带重跑所有 DEM 用例，
   超出单轮预算，已登记为后续项。
2. **流体侧 LBM Cd 在 Re=1/2/50 出 15% 带**（+16.5%、+20.3%、−28.5%）：限域 λ 单一、无 λ-外推；
   Hasimoto 修正在 Re≥5 不可用。⇒ 流体侧**尚不可**作为 DEM 阻力律的参考。
3. **未做**：ten Cate 2002 原始实验值对标（无原文表可复核，拒绝手抄）；体标定（休止角/卸料）；
   Hertz–Mindlin 非线性接触；LBM 侧 λ-外推（需多域尺寸扫描，>10 min，应走 cluster_queue）。

**回归测试**：`tests/test_xval_dem_literature_anchors.py`（锁住 `e` <15% 与 `t_c` <5% 两条不变量，
防止未来改动把已达标项做坏；`clip=False` 的 2% 不变量以 xfail 记录待修缺陷）。

**复现命令**

```bash
cd xval/XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY
python3 run_xval.py          # 全量，约 40 s 单核；写 results.json
python3 make_figures.py      # 写 fig_xval_dem.png
python3 -m pytest ../../tests/test_xval_dem_literature_anchors.py -q
```

**工件**

| 路径 | 内容 |
|---|---|
| `predeclared.json` | 判据/假设/臂/阈值（运行前登记） |
| `survey_oss.md` | ① OSS 横向调研（缺口证据 + 第 1 轮死因复盘） |
| `lineage.md` | ② 方法谱系 + external_anchor + 迁移方案 M1–M5 |
| `run_xval.py` | 驱动（A/B/C/D/E/F 六臂，全走本仓真实模块） |
| `results.json` | 全部原始数字（本报告数字的唯一来源） |
| `fig_xval_dem.png` | 三联图：(a) 恢复系数收敛 (b) 沈降 vs 文献 (c) LBM 侧体检 |
| `oss_solutions.jsonl` | ③ 方法选型 + 迁移方案记录（待入库，见下） |

**③ 入库状态（如实）**：本会话对 `oss_solutions` / `oss_inventory` / `tensor_xval_cases` 三张表
**无 INSERT/查询权限**（`ERROR: permission denied for table oss_solutions`），故选型记录以
`oss_solutions.jsonl`（可直接 `COPY … FROM` 的 JSON 行）落在工件里，等有权限的席位入库；
本单**不谎称已入库**。

**能力增量登记**：DEM 侧首次拥有"文献锚定"的可机器复核验收集（此前 `dem_verification.json`
只做同式自明检查，误差 1e-11 属 H3 类，不构成物理验收）。