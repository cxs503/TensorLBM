# XVAL-LBM-HORIZONTAL-SURVEY-EMPTY — 水平开源调查为空时的供给：圆柱振荡流惯性系数 C_M

**锚点** XVAL-LBM-HORIZONTAL-SURVEY-EMPTY · **仓** TensorLBM · **mission** 17844
**对标对象** `oss_solutions` id=386（`repo='oss-inventory/tensorplatform'`, `feature_slug='GAP-DEM-CLUMP'`, `tag='horizontal-survey-empty'`）
的**做法**：先在 OSS 清单上做水平调查，若命中 0 行则如实记录"无可移植实现"，改为从文献条目构建。
**本单结论（机器判据）**：调查段 = **EMPTY 复现成立**（census 0 / ledger 0 / 源码镜像 0）；
精度段 = **FAIL**（headline 约定 D_halfway，base 误差 -21.3%，
阈值 ±15%；收敛梯 3 档中 0 档达标）。

## ① 问题 + ref 出处

OSS 清单里没有任何实现做过"振荡流惯性系数 / 附加质量 / PMM 水动力导数"这一族（见 §②S1–S3），
因此不存在可交叉对标的第三方代码或参数约定。可用的锚只能是文献解析值：

* Morison, O'Brien, Johnson & Schaaf (1950)：`F = C_M rho V dU/dt + 0.5 C_D rho D U|U|`；
* Sarpkaya (1986)：圆柱在无界理想流体中 `C_M = 1 + C_a = 2`（`m_a = rho*pi*D^2/4`），`C_a = 1` 为精确值；
  相似数 `KC = U_max T / D`、`beta = D^2/(nu T)`，理想极限为 `beta -> inf, KC -> 0`。

**要检验的假设（预登记）**：在 TensorLBM 里用调和体力驱动的双周期盒 + Ladd 动量交换测力，
对 Morison 做最小二乘分解，能否在收敛分辨率上把 `C_M` 复现到 `2.00 ± 15%`。

## ② 方法与实现

**S1–S3 水平调查**（`survey.py`，输入=两份 SQL dump，零手抄，见 `data/PROVENANCE.json`）：

| 源 | 规模 | 本族命中 | 说明 |
|---|---|---|---|
| oss_inventory（repo 清单） | 33 行 | **0** | focus/methods/multiphase/boundary/gpu/summary 全文正则 |
| oss_solutions（做法台账） | 261 行 | **0** | feature_slug/tag/method/files/conventions/transfer/evidence |
| `~/oss-cache` 源码镜像（3 包） | 3 包 | **0** 文件 | `grep -rIl` 同族关键词 |

对照：同族扫描同时证明该仪器**不是恒空**——`immersed_boundary` 族在 census 命中
20 行、ledger 命中 20 行，
`non_newtonian_powerlaw`/`mhd_lorentz` 亦各有点命中（`survey_table.csv`）。故 EMPTY 是本族的真结论，不是仪器失效。

**求解器**：D2Q9 + BGK + 二阶 Guo 源（`tensorlbm.boussinesq.apply_guo_forcing`），
双周期盒（无壁面阻塞），均匀调和体力 `a(t)=a0 cos(omega t)` 只加在流体节点；
定圆障碍、半程反弹（`boundary.bounce_back_cells`），测力用 Ladd 动量交换
（`boundary.compute_obstacle_forces`，post-stream / pre-bounce-back）。
两个独立估计量：E1 = Ladd 节点动量交换；E2 = 流体动量收支 `F = sum_fluid a(t) - dP/dt`。
Morison 最小二乘用**实测**盒平均速度 `U(t)`（2 阶谐波拟合后解析求导），不假设远场速度。

## ③ 参数表（无量纲数完整）

| 量 | 值/范围 | 说明 |
|---|---|---|
| 格子 | D2Q9, BGK, float64 CPU | tau = 0.5+3nu 逐案给出 |
| Ma | 0.0693（全部 case 相同） | u0 = 0.04 |
| KC | 名义 {0.5, 1.0, 2.0}；实测见 §④ | `KC = U_max T / D` |
| beta | 名义 {50, 100, 200}；实测见 §④ | `beta = D^2/(nu T)`，`delta = D/sqrt(pi beta)` 格 |
| 分辨率 | D = 16 / 24 / 32 格 | 收敛梯 |
| 盒周期 | L/D = 12 / 16 | 周期像敏感性 |
| 每案步数 | (1.5 + 3.0) 个周期，预登记 | 无事后延长/截断 |
| 参考值 | `C_M = 2.00` | 理想流体圆柱（精确） |

## ④ 对照表（ref vs got，误差打勾叉）

| case | D_in | D_halfway | KC_ach | beta_ach | C_M(D_halfway) | err% | C_M(D_area) | err% | C_M(D_input) | E1–E2 差% |
|---|---|---|---|---|---|---|---|---|---|---|
| res16 | 16 | 17.19 | 0.94 | 115 | 1.615 | -19.2 | 1.802 | -9.9 | 1.864 | 185.6 |
| base | 24 | 24.77 | 0.99 | 106 | 1.574 | -21.3 | 1.694 | -15.3 | 1.677 | 222.5 |
| kc05 | 24 | 24.77 | 0.49 | 106 | 1.434 | -28.3 | 1.543 | -22.9 | 1.528 | 103.3 |
| beta50 | 24 | 24.77 | 0.98 | 53 | 1.671 | -16.4 | 1.798 | -10.1 | 1.781 | 200.3 |
| beta200 | 24 | 24.77 | 0.98 | 213 | 1.513 | -24.3 | 1.628 | -18.6 | 1.613 | 210.7 |
| box16 | 24 | 24.77 | 0.98 | 106 | 1.169 | -41.6 | 1.257 | -37.1 | 1.245 | 224.5 |

阈值 ±15%（D_halfway = 半程反弹润湿直径 = 面积+P/2 等效）：
* `res16`: -19.2% ❌（D_area 约定 -9.9% ✅）
* `base`: -21.3% ❌（D_area 约定 -15.3% ❌）
* `kc05`: -28.3% ❌（D_area 约定 -22.9% ❌）
* `beta50`: -16.4% ❌（D_area 约定 -10.1% ✅）
* `beta200`: -24.3% ❌（D_area 约定 -18.6% ❌）
* `box16`: -41.6% ❌（D_area 约定 -37.1% ❌）

**判读**：headline 约定下不达标（偏负 19–28%），同一批数据在 D_area 约定下
达标——
几何约定差异（`D_halfway/D_input` 相对差见 `summary.json:convention_spread_pct`）与 15% 预算同量级，
因此本单把"约定必须先用独立标定钉死"作为负结论登记，而不是挑一个通过的约定。
E2（动量收支）在本实现里给出非物理值（见 ⑤），不参与判据，如实保留在 `table.csv`。

## ⑤ 收敛性 / 参数敏感性（至少两点）

![fig1](docs/assets/xval_lbm_horizontal_survey_empty/fig1_convergence.png)
![fig2](docs/assets/xval_lbm_horizontal_survey_empty/fig2_sensitivity.png)
![fig3](docs/assets/xval_lbm_horizontal_survey_empty/fig3_velocity_trace.png)

1. **分辨率收敛**（D=16/24/32，KC=1，beta=100）：`C_M` 随 D 的走向与 1/D 外推见 fig1 右图；
   若几何约定是主误差源，则 1/D 外推应在 D→∞ 处收敛到同一截距（数值见 `summary.json`）。
2. **KC 敏感性**（0.5/1/2，D=24）：KC 增大应使惯性项占比下降、`C_D` 上升；`C_M` 理想上对 KC 弱依赖。
3. **beta 敏感性**（50/100/200，KC=1）：粘性 Stokes 层 `delta = D/sqrt(pi beta)` 格，
   beta=100/D=24 时仅 1.35 格 ⇒ 粘性修正未解析（预登记混淆项），理想极限应由 beta 梯外推。
4. **盒周期**（L/D=12 vs 16）：周期像贡献 O((D/L)^2)，用于界定周期化误差。

## ⑥ 失败模式与工件

* **已发现并修复的实现缺陷（可复用知识）**：把 `bounce_back_cells`（post-stream 全量掩膜反弹）
  与下一轮回流组合时，**物体内部节点会累积垃圾分布**并回灌到壁面邻近节点，
  使动量交换测力系统性偏低：同一常数加速诊断在修复前 `F/(rho V a0) = 1.48`、修复后 `= 1.95`（D=16，步 100）。
  修复 = 碰撞前保存壁面分布、碰撞后还原、反弹后把**内部固体节点置零**（`n_interior_zeroed` 记入 result.json）。
  仓内 `xval/benchmark_method_selection` 的方柱案例用的正是未修复的组合。
* **E2 未通过自洽检查**：`sum_fluid a(t) - dP/dt` 在本实现下给出负值（`table.csv`），
  说明动量注入在固体边界（反弹时被吸收的动量）与 `dP/dt` 之间还有一项未记账；E2 仅作阴性记录。
* **未解析的 Stokes 层**：见 ⑤-3，本案例验证的是**惯性（附加质量）算子**，不是振荡粘性阻力。
* 工件：`runs/<case>/{result.json,trace.npz}`；图：`docs/assets/xval_lbm_horizontal_survey_empty/`；
  汇总：`summary.json`、`table.csv`；调查：`survey_summary.json`、`survey_table.csv`。

## ⑦ 本仓 vs OSS 差异清单 + 移植取舍

| 环节 | 本仓（TensorLBM） | OSS（33 行 census + 261 行 ledger 调查结果） |
|---|---|---|
| 振荡流惯性系数/PMM | 本单新增验证案例（Morison 最小二乘 + 双估计量） | **无实现**（0/0/0 命中） |
| 驱动方式 | 双周期盒 + 均匀调和体力（无进出口） | — |
| 边界 | 半程反弹掩膜圆 + 内部节点置零 + 周期 | — |
| 测力 | Ladd 动量交换（post-stream/pre-BB） | — |
| 后处理 | 实测 U(t) 2 阶谐波拟合 → Morison LS | — |
| 精度标定 | 文献 C_M = 2.00 | — |

**移植取舍**：无可移植代码 ⇒ 唯一移植的是**做法**（"调查→如实记录空集→改用文献锚"），
代码自建；因此本单的"差异清单"退化为**自建实现与理想流体解析解的差异**（逐条在 ④）。
**生态摩擦**：①`oss_inventory` 仅 33 行（该参考行记录的普查是 218 行/140 feature slug），
说明普查表已缩水，跨域可对标面变窄；②`oss_solutions`/`oss_inventory` 对普通 shell 角色无 SELECT 权限，
dump 需 `runuser -u postgres`，调查脚本只能吃离线 dump（已在 PROVENANCE 记录）。
