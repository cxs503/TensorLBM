# 热对流验证欠账结清：AUDIT-RUNTEST-6 发现 D12 的根因、修复与复验

`锚点=NEW: 审计 D12 热对流（RB/DHC）真测试失败无总账锚点` · 仓库 TensorLBM · 任务 m16100 · 2026-11-12
证据分级：本节所有数字来自 `artifacts/thermal_rb_debt/*.json`（脚本 `scripts/diag_rb_spinup.py`、
`scripts/diag_dhc_spinup.py`、`scripts/diag_thermal_invariance.py` 生成），零手抄。

---

## 0. 一句话结论

`AUDIT-RUNTEST-6` 抽样套件里 5 条物理判据失败（Rayleigh–Bénard Nu 偏差 64.7%、能量平衡 8.24%、
DHC Nu 偏差 29.4%、两条维度一致性 spread 178%/19.5%）**不是一类问题，是两类**：

| 类 | 判据 | 根因 | 处置 |
|---|---|---|---|
| A（真缺陷） | C4 维度一致性（RB+ DHC，spread 178% / 19.5%） | `thermal_params()` / `simulate_rayleigh_benard_2d()` 由 Ra 反解浮力系数时把 ΔT **硬写成 1**：`g_beta = Ra·ν·α/H³`。Ra 的定义含 ΔT（`Ra = g·β·ΔT·H³/(ν·α)`），于是同一物理工况（同 Ra）在 `t_hot=2` 时被静默算成 **Ra_eff = 2·Ra** | 代码修复：`g_beta = Ra·ν·α/(ΔT·H³)`；`ΔT ≤ 0` 抛 `ValueError`；结果对象新增 `dT` / `ra_effective` 审计字段 |
| B（验证设计缺陷，非物理缺陷） | C2 RB Nu、C2 能量平衡、C3 DHC Nu | 判据取样在**瞬态**上：5000 / 3000 步对热扩散时标 `H²/α ≈ 2.05e4` 步而言只有 0.25 / 0.15 个时标，测到的是线性增长段的导热态（Nu≈1），却拿稳态对流参考值（2.70 / 2.243）去比 | 判据改为收敛采样：RB 12000 步（Nu 稳态漂移 <2% 门）、DHC 40000 步（resid_u ~1e-6），标 `slow`；docstring 内置实测自启动表 |

修复后实测：RB Ra=1e4 → **Nu = 2.424**（参考 2.70，−10.2%，判据门 20%）；能量失衡 **1.4%**（门 5%）；
DHC Ra=1e4 (nx=48) → **Nu = 2.086**（de Vahl Davis 2.243，−7.0%，门 10%）；
维度一致性 spread：RB `ΔT=1 vs 2` → **2.7e-6**（修复前 178%），DHC → 见 §4。

---

## 1. 案例介绍（物理背景 / 几何 / 参数表）

两套自然对流验证算例，均在 D2Q9 双分布（DDF）热格子 Boltzmann 框架内，Boussinesq 近似：

**(a) Rayleigh–Bénard（RB）**：水平层，下壁 T_hot、上壁 T_cold、左右周期；上下壁为 no-slip 静止壁
（`_rb_wall_mask` 把 y=0、y=ny−1 两行设为固体，pre-streaming 半程反弹）；浮力沿 +y（热升）。

**(b) 差分加热方腔（DHC）**：左壁热、右壁冷、上下绝热；四壁 no-slip。

| 量 | RB (a) | DHC (b) |
|---|---|---|
| 网格 | nx=64, ny=32（Γ=2） | nx=ny=48 |
| Ra / Pr | 1e4 / 0.71 | 1e4（另 1e3）/ 0.71 |
| τ / τ_T | 0.6 / 0.64085 | 0.6 / 0.64085 |
| ν / α（格子） | 0.033333 / 0.046948 | 0.033333 / 0.046948 |
| H（壁到壁） | ny−1 = 31 | nx = 48（half-way 壁约定） |
| 浮力系数 g·β | `Ra·ν·α/(ΔT·H³)` = 5.2531e-4 (ΔT=1) | `Ra·ν·α/(ΔT·H³)` = 1.4151e-4 (ΔT=1) |
| 参考值 | `RB_NU_REFERENCE[1e4] = 2.70` | de Vahl Davis (1983) `2.243` |
| 热扩散时标 | `H²/α = 2.05e4` 步 | `H²/α = 4.9e4` 步 |

## 2. 怎么做的（方法 / 流程）

1. **复现**：`pytest tests/test_thermal_verification_p96w2n04.py -k "rb_nu_at_ra1e4 or rb_energy_balance"`
   → 两条 FAIL（`Nu=0.954 vs ref 2.700, err=0.6468`；`Energy imbalance 0.0824 > 5%`），与审计 D12 一致。
2. **分裂问题**：把「Nu 对参考值偏差」与「同一 Ra 下 Nu 随 ΔT 变」分开看——前者可能是瞬态取样、
   后者是真缺陷。对后者做量纲分析：`Ra = g·β·ΔT·L³/(ν·α)`，代码却用 `g_beta = Ra·ν·α/H³`，
   于是 `Ra_eff = g_beta·ΔT·H³/(ν·α) = Ra·ΔT`。**可直接用一次 1 步跑动机器判定**（新增 `ra_effective` 字段）。
3. **自启动（spin-up）标定**：`scripts/diag_rb_spinup.py` 40000 步、每 4000 步记 Nu/u_max/能量失衡；
   `scripts/diag_dhc_spinup.py` 40000 步、每 4000 步记 Nu/resid。用它们定判据的采样点（见 §3）。
4. **修复**（最小改动，两处）：
   - `src/tensorlbm/thermal.py::thermal_params` 增加 `dT` 形参（缺省 1.0，向后兼容），
     `g_beta = ra*nu*alpha/(dT*H**3)`；`dT ≤ 0 → ValueError`。`simulate_natural_convection` 把
     `dT = t_hot - t_cold` 传进去。
   - `src/tensorlbm/thermal_verification.py::simulate_rayleigh_benard_2d` 同样除 ΔT，
     `dT ≤ 0 → ValueError`；`RBResult` / `DHCResult` 新增 `dT`、`ra_effective`
     （`g_beta·ΔT·H³/(ν·α)`，即**反算出来的有效 Ra**，供机器审计）。
5. **判据改造**（`tests/test_thermal_verification_p96w2n04.py`）：收敛采样 + `slow` 标记 +
   稳态门（末两次 Nu 相差 <2%）；同时把 C4 的两条 spread 测试**换成分层结构**：
   - `test_rb_effective_rayleigh_is_dT_invariant` / `test_dhc_effective_rayleigh_is_dT_invariant`：
     纯参数级、毫秒级、判据 `|Ra_eff−Ra|/Ra < 1e-12`，ΔT ∈ {0.5,1,2,4}——这是缺陷回归守卫；
   - `test_dT_must_be_positive`：ΔT ≤ 0 必须抛（负控，防静默 NaN）；
   - 原 spread 测试保留但移到收敛采样（`slow`），作为物理层的一致性证据。
6. **复验**：见 §3/§4 的表；`scripts/diag_thermal_invariance.py` 在修复后跑 ΔT=1/2 对照。

## 3. 计算结果（程序化注入，零手抄）

### 3.1 RB Ra=1e4, Γ=2（`artifacts/thermal_rb_debt/base_dT1.json`，40000 步）

| step | Nu_bot | Nu_top | Nu | u_max | 能量失衡 |
|---|---|---|---|---|---|
| 4000 | 0.9103 | 0.9926 | 0.9514 | 6.426e-04 | 8.65% |
| 8000 | 2.6145 | 2.5729 | 2.5937 | 4.559e-02 | 1.60% |
| 12000 | 2.4281 | 2.4072 | 2.4176 | 4.168e-02 | 0.87% |
| 20000 | 2.4331 | 2.4075 | 2.4203 | 4.168e-02 | 1.06% |
| 30000 | 2.4371 | 2.4071 | 2.4221 | 4.167e-02 | 1.32% |
| 40000 | 2.4414 | 2.4069 | 2.4241 | 4.167e-02 | 1.42% |

读法：**4000 步仍在导热态（Nu=0.95、u_max=6e-4）**，8000 步已完成对流起始与饱和；
旧判据的 5000 步恰好落在这条上升沿上——这就是「64.7% 偏差」的来源，不是求解器错。

### 3.2 DHC Ra=1e4, nx=48（`artifacts/thermal_rb_debt/dhc_nx48_dT1_before.json`，40000 步）

| step | Nu_left | Nu_right | Nu | resid_u | resid_T |
|---|---|---|---|---|---|
| 12000 | — | — | 2.078 | 2.13e-04 | 9.69e-03 |
| 32000 | 2.0986 | 2.0693 | 2.0839 | 1.25e-06 | 5.39e-05 |
| 40000 | 2.1034 | 2.0690 | 2.0862 | 1.29e-06 | 5.20e-05 |

### 3.3 修复后的维度一致性（`artifacts/thermal_rb_debt/post_fix_invariance.json`）

| 算例 | ΔT | g·β | Ra_eff（反算） | Nu | u_max | 备注 |
|---|---|---|---|---|---|---|
| RB 64×32, 12000 步 | 1.0 | 5.2531e-04 | 10000.000 | 2.4176 | 4.1679e-02 | 修复前基线（Δt=1 情形未被改动） |
| RB 64×32, 12000 步 | 2.0 | 2.6265e-04 | 10000.000 | 2.4176 | 4.1679e-02 | 修复后 |
| DHC 48², 12000 步 | 1.0 | 1.4151e-04 | 10000.000 | 2.0782 | 1.819e-02 | 修复后 |
| DHC 48², 12000 步 | 2.0 | （见 json） | 10000.000 | （见 json） | — | 修复后 |

RB 两个 ΔT 的 Nu 相对差 **2.7e-6**；修复前同一对照是 base=0.9519 vs dt2=2.6508（spread 178%，
`artifacts/audit_runtest_6/pytest_sample.log`）。

## 4. 图

`figures/thermal_d12/thermal_d12_spinup.png`（`scripts/plot_thermal_d12.py` 从上述 json 生成，
无手写数字）：

* 左：RB 的 Nu 与 u_max 随步数自启动曲线，叠加参考值 2.70、2D RB 文献带 2.2–2.45、导热极限 1，
  以及**旧判据取样点 5000 步**的竖线；
* 右：DHC 的 Nu 随步数收敛曲线，叠加 de Vahl Davis 2.243 与 ±10% 带、旧取样点 3000 步。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **ΔT 缺陷是"看得见却没人看"的典型**：`thermal_params` docstring 原文就写着「（ΔT = 1）」，
   因为两条测试都用 ΔT=1 所以一直绿；只有 C4 换了 ΔT 才暴露。教训：**任何由无量纲数反解系数的地方，
   都要把「无量纲数 → 系数」的映射本身写成可机器反算的断言**（本次落成 `ra_effective` 字段 + 1e-12 断言）。
2. **Nu 是稳态量，不能在瞬态上判**。判据必须自带采样点理由（本次给的是 `H²/α` 时标与实测自启动表），
   否则「验证失败」与「物理缺陷」无法区分——本轮 D12 里 5 条里 4 条属于后者，只有维度一致性那条是真缺陷。
3. **RB 的 Nu 仍有缓慢长期漂移**：12000→40000 步 Nu 从 2.4176 涨到 2.4241（+0.27%），
   Nu_bot 与 Nu_top 分别 2.4414 / 2.4069（差 1.4%）。这是 ny=32 分辨率下上下壁面非平衡反弹修正不对称的
   离散效应，不随步数消失；因此本算例的**分辨率误差约 ±1–2%**，不能当作 <1% 的参考数据源。
4. **参考值本身有不确定度**：`RB_NU_REFERENCE[1e4]=2.70` 落在 2D RB 文献带（2.2–2.45，Γ 与 Pr 相关）
   之外，本次实测 2.42 落在文献带内。判据门（20%）宽松到能同时容纳两者，但**不应把它当作精度证据**；
   已在测试 docstring 与本文中显式标注（honest：本单未把 2.70 换成更可信的 2.3，因为换参考值属于
   独立取证工作，需重新查文献表而不是"照实测值反推"）。
5. **计算成本**：RB 12000 步（2048 格）≈ 65 s；DHC 40000 步（2304 格）≈ 250 s；故这两条判据标 `slow`，
   PR 快车道（`-m "not slow"`）只跑参数级不变性与导热极限，代价毫秒级。

## 6. 误差判定（关键量 vs 基准，逐项回答「是否 <3%」）

| 关键量 | 实测 | 基准 | 相对误差 | 判据门 | <3% ? |
|---|---|---|---|---|---|
| RB Nu @Ra=1e4 | 2.424 | `RB_NU_REFERENCE` 2.70 | **−10.2%** | 20% | **否**（且参考值本身可疑，见 §5.4） |
| RB Nu @Ra=1e4 vs 2D 文献带 2.2–2.45 | 2.424 | 2.2–2.45 | 带内 | — | 带内（非严格基准） |
| RB 能量失衡 @12000 步 | 0.87% | 0（稳态精确） | 0.87% | 5% | **是** |
| RB 能量失衡 @40000 步 | 1.42% | 0 | 1.42% | 5% | **是** |
| DHC Nu @Ra=1e4, nx=48 | 2.086 | de Vahl Davis 2.243 | **−7.0%** | 10% | **否**（有限分辨率 + nx=48 未做 GCI） |
| DHC Nu @Ra=1e3（原判据） | 1.118 级 | de Vahl Davis 1.118 | 见测试（原已 PASS） | 10% | 是（既有绿判据） |
| Ra_eff 维度不变性（RB，ΔT∈{0.5,1,2,4}） | 10000.000 | Ra=10000 | **<1e-12** | 1e-12 | **是** |
| Ra_eff 维度不变性（DHC，ΔT∈{0.5,1,2}） | 10000.000 | Ra=10000 | **<1e-12** | 1e-12 | **是** |
| RB Nu spread（ΔT=1 vs 2，12000 步） | 2.417637 vs 2.4176 | 同工况 | **2.7e-6** | 2% | **是** |

**Honest**：Nu 对文献参考的两个 10% 级偏差（−10.2% / −7.0%）**没有压到 <3%**，原因是有限分辨率
（ny=32 / nx=48 且未做网格 GCI）与参考值不确定度，本单未做网格加密研究，故不宣称 <3%。
本单结清的是**判据的可信度**（采样在稳态、缺陷可机器反算）与**一个真缺陷**（ΔT），不是精度指标。

## 7. 复验命令

```bash
# 快速层（PR 快车道）：参数级不变性 + 负控 + 导热极限
PYTHONPATH=src pytest tests/test_thermal_verification_p96w2n04.py -m "not slow" -q
# 收敛层：RB/DHC 稳态 Nu + 维度一致性
PYTHONPATH=src pytest tests/test_thermal_verification_p96w2n04.py -m slow -q
```