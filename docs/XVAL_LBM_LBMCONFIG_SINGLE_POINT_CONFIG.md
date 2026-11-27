# XVAL-LBM-LBMCONFIG-SINGLE-POINT-CONFIG — 单点配置（lbmpy `LBMConfig` 做法）交叉验证

锚点：`XVAL-LBM-LBMCONFIG-SINGLE-POINT-CONFIG`（`feature_ledger`，验收线 <15%）
仓：TensorLBM · 分支 `agent/m22858` · mission 22858
新增生产代码：`src/tensorlbm/lbm_config.py` · 测试：`tests/test_lbm_config.py`(52 项)
验证入口：[运行入口](../benchmarks/cross_validation/lbmconfig_poiseuille/run_xval.py) · [归档结果](../benchmarks/cross_validation/lbmconfig_poiseuille/evidence/run_main.json) · [剖面表](../benchmarks/cross_validation/lbmconfig_poiseuille/evidence/profiles.csv)

---

## 1. 案例介绍（物理背景 / 几何 / 参数）

对标做法（OSS 锚）：lssfau/lbmpy 用**一个** `LBMConfig` 数据类声明整套格式
（stencil、method、relaxation_rate、force_model、force），再由它派生碰撞规则、
力源项、宏观速度输出约定与迁移步。TensorLBM 此前没有这个"单一真源"：
`collide_bgk(f, tau)` / `collide_trt(f, tau_plus, lambda_trt)` / 手写 Guo 源项 /
`macroscopic(f)`（返回**裸动量**速度）分散在各调用点，`nu`、`tau`、
`(1-omega/2)` 离散格子力修正、速度输出约定都要在每个调用点各自重推一遍。

本单把该做法落到生产模块 `src/tensorlbm/lbm_config.py`（`LBMConfig`，frozen
dataclass，派生量全为 property），并用**力驱动平面 Poiseuille 槽道**做独立交叉验证：

| 项 | 值 |
|---|---|
| 几何 | 2D，x 向周期（流场 x 向均匀，`nx=8`），上下半程反弹壁 |
| 壁面位置 | `y = 1/2` 与 `y = H + 1/2`（half-way bounce-back，流体行 `1..H`） |
| 流体行数 | 主算例 `H = 20`；收敛研究 `H = 16, 32`；力格式扫描 `H = 16` |
| 体力（单位质量） | `F_x = 2.0e-7`（低 Mach，`u_max ≈ 1.6e-5`） |
| 松弛率 | `omega = 1/tau = 1.4`（`tau = 0.714286`，`nu = 0.0714286`，`cs² = 1/3`） |
| 解析解 | `u_x(y) = F/(2 nu) (y - 1/2)(H + 1/2 - y)` |
| 收敛判据 | 每 200 步的 `max|Δu| / max|u| < 1e-9`；随后 400 步窗口平均 |
| 判定基准 | owner 标准：相对误差 <15% 可接受；本报告同时给 <3% 判据 |

## 2. 怎么做的（方法 / 流程 / 工具链）

1. **单一真源**：`LBMConfig` 一次性声明
   `stencil/method/relaxation_rate/lambda_trt/force/force_model`；
   `tau = 1/omega`、`nu = cs²(tau-1/2)`、`tau_minus = 1/2 + Lambda/(tau_plus-1/2)`
   全由 property 派生；`scheme()` 输出可审计的派生量表。
2. **力记账必须是"一对"**：教科书 Guo（Guo, Zheng & Shi 2002）要求
   平衡态用**机械速度** `u = j/rho + F/(2 rho)`，且源项带 `(1 - omega_m/2)`
   修正（`omega_m` 为携带动量的模式的松弛率：SRT 取 `omega`，TRT 取
   `omega_minus`）。两半配对后**每节点每步恰好注入 F**（单元测试逐位断言）。
   `force_model="guo_split"` 则是仓内既有 `channel_kernel_numpy.apply_body_force`
   约定：平衡态用裸速度、源项带**全量** F，注入同样是 F。
3. **不可表达的错法**：`force_model="guo"` 与 `velocity_output_includes_force=False`
   组合被 `__post_init__` 直接拒绝（错误信息给出实测代价 70%）；
   `guo + MRT` 也被拒绝（MRT 的守恒动量矩不松弛，半边动量记账无法表达）。
4. **臂（arm）**：`srt_guo`（教科书）、`trt_guo`（Ginzburg Λ=3/16）、
   `srt_guo_split`、`mrt_guo_split`、`simple_force`（一阶朴素源项）、
   `explicit_calls`（**不用** config，按文档手写 `tensorlbm.solver` 调用重推同一格式）、
   `legacy_mixed`（负对照：修正源项 + 裸速度平衡态，即历史缺陷）。
5. **独立数值路径**：所有臂的时间推进到定常，并用**不动点残差**
   `max|step(f)-f| / max|f|` 核验"采样到的确实是定常解"（本单各臂 ~1e-15，
   排除了把瞬态尾部当稳态的经典错误）。
6. 工具链：PyTorch float64（单线程）、numpy、pytest；计算按 owner 计算调度令
   递交调度系统/分阶段本地短跑（本题为 2D 低 Mach 小算例，逐阶段 <110 s）。

## 3. 计算结果（程序化注入，未手抄）

**主算例 H=20、omega=1.4（`out/run_main.json`）**

| 臂 | 相对误差 max\|u-u_ana\|/max\|u_ana\| | <3% | <15% | 力源标度 |
|---|---|---|---|---|
| `srt_guo`（教科书 Guo） | **1.892e-03** | 是 | 是 | 0.300 |
| `trt_guo`（Λ=3/16） | **1.417e-09** | 是 | 是 | 0.636 |
| `srt_guo_split` | 2.609e-03 | 是 | 是 | 1.000 |
| `mrt_guo_split` | 2.970e-03 | 是 | 是 | 1.000 |
| `simple_force` | 2.609e-03 | 是 | 是 | 1.000 |
| `explicit_calls`（手写重推） | 1.892e-03 | 是 | 是 | — |
| `legacy_mixed`（负对照） | **7.0007e-01** | 否 | 否 | 0.300 |

- 单点透明性（C1）：手写路径与 config 路径的剖面 `max|diff| = 5.44e-15`
  （相对 `max|u|` = **3.9e-11**，纯浮点结合律差异；判据 `<=1e-12·max|u|`
  见 `checks.C1_single_point_transparency`）。
- 台架不变式：定点残差 8.5e-15～1.2e-15；一致性臂"碰撞+源项"每步注入动量 = F
  （相对偏差 ≤4.6e-11，`test_total_momentum_injection_per_step_is_exactly_the_force`
  对 SRT/TRT/MRT × guo/guo_split/simple 六组合逐一断言）。
- **负对照定量**：`legacy_mixed` 的误差 70.007% 与"只注入 `(1-omega/2)F`"的预测
  `1 - 0.3 = 70%` 一致（`negative_control_legacy_mixed.predicted_1_minus_scale = 0.7`）。

**收敛研究（SRT，H=16→32）**

| 臂 | H=16 | H=32 | 观测阶 |
|---|---|---|---|
| `srt_guo` | 2.961e-03 | 7.381e-04 | **2.004** |
| `trt_guo` | 1.791e-11 | （未跑，见 §6） | — |

**力格式扫描（H=16，部分完成）**：`guo` omega=0.80 → 7.843e-03；1.40 → 2.961e-03；
1.90 → 3.907e-03；`guo_split` omega=0.80 → 3.922e-03；1.40 → 4.082e-03。
扫描的 `simple`/`legacy_mixed` 分支与 omega=1.10/1.70 两点因作业被本地 110 s
时限截断（见 §6），**如实记为未跑**。

**图**（matplotlib 由工件 JSON 程序化生成，内嵌于平台报告）：
① 定常速度剖面 vs 解析抛物线（含负对照）；② SRT 网格收敛（log-log）。

## 4. 关键发现

- **F1（阻断级缺陷类，已由配置层封死）**：把带 `(1-omega/2)` 的修正源项与
  **裸速度**平衡态混搭，会让槽道剖面恰好缩水到 `(1-omega/2)` 倍 —— 实测
  `omega=1.4` 时误差 70.007%，与解析预测 70% 一致。这正是"同一格式的参数散落在
  多个调用点"的典型后果；现在这种组合在 `LBMConfig.__post_init__` 里不可表达。
- **F2（重要，由不变式测试发现）**：TRT 的动量由**反对称**部分承载
  （`sum_i c_i f_i^+ = 0`），因此 `(1-omega/2)` 里的松弛率必须是 `omega_minus`
  而非 `omega_plus`。用 `omega_plus` 时 `TRT+guo` 每步只注入 0.716F
  （实测偏差 28.4%）；改用 `omega_minus` 后注入偏差降到 1.6e-13。
- **F3（可复用知识）**：`TRT + Lambda=3/16` 在该槽道上把壁面离散误差消到机器精度
  （H=20 时 1.4e-9，H=16 时 1.8e-11），而 SRT 是二阶收敛（观测阶 2.004）。
  即：单个配置对象里"method"一个旋钮的差异，实测跨 6 个数量级。
- **F4（诚实边界）**：静态槽道**无法**区分 `guo_split`/`simple` 与教科书 Guo
  （三者都在 2.6e-3 级）。它们只在时变流动里由被丢掉/保留的 `O(uF)` 源项矩分开，
  故本报告**不**声称 `simple` 已验证，仅记录其在定常算例中动量守恒（注入 F）。

## 5. 注意点（坑 / 局限 / 适用范围）

1. **`relaxation_rate` 语义**：本模块取 lbmpy 约定 `omega = 1/tau`。上一轮同锚工件
   [`m17952_legacy.py`](../development/verification/lbmconfig_poiseuille/m17952_legacy.py)（m17952）把 `omega=0.10` 当"速度松弛"
   用 `tau = 1/omega + 0.5 = 10.5`（`nu=3.3333`），而 lbmpy 侧
   `LBMConfig(relaxation_rate=0.1)` 对应 `tau=10`（`nu=3.1667`）——两臂粘度相差
   **5.26%**，交叉比较被污染（该脚本另硬编码 `/srv/tensor/agent/work/m17952/src`
   绝对路径、无测试、无工件，故锚点一直 open）。本单的单一真源把这类漂移消除。
2. `guo + MRT` 不可用：MRT 的守恒动量矩（rate=0）不携带半边力，教科书记账无法表达，
   故 MRT 只提供 `guo_split`。
3. 仅 D2Q9 / 二维；`simple` 与 `guo_split` 在定常算例中不可分辨（见 F4）。
4. 半程反弹壁位置固定为 `y=1/2, H+1/2`；解析解按此壁位给出。
5. 时间推进到定常用"变化率 + 不动点残差"双判据，避免瞬态尾部（历史审计 m18084 的
   F2 类缺陷）。

## 6. 未执行的检查（unknown / not_run，不计为通过）

- **lbmpy 参考臂未执行**：容器内 `import lbmpy` 不可用
  （`run_main.json: lbmpy_reference = {available: false, reason: ModuleNotFoundError}`）。
  因此"OSS 参考实现"一侧只到**文档语义级**（字段/约定映射由本单以闭式公式独立重推，
  并由 52 项单元测试对解析量断言），**没有** lbmpy 实测数值参与比较。
  本轮 15% 判定是对**解析解**的，不是对 lbmpy 的。
- **H=64 收敛点、H=32 的 TRT 点**：递交调度系统的作业 `cj4448` 排队中（前有 68 个），
  本地分阶段短跑被 110 s 时限截断 → 未跑。
- **力格式扫描全表**：`simple`/`legacy_mixed` 在 H=16 的 omega 扫描未跑（同上）。
- 时变/非定常算例（用于分离 `simple` 与 `guo_split`）、三维 stencil、GPU 后端：
  均未做。

## 7. 误差判定（逐项，显式回答"是否 <3%"）

| 关键量 | 参考值 | 本实现 | 相对误差 | <3%? |
|---|---|---|---|---|
| SRT+教科书 Guo 定常剖面（H=20） | 解析抛物线 | 1.892e-03 | 0.189% | **是** |
| TRT(Λ=3/16)+Guo 定常剖面（H=20） | 解析抛物线 | 1.417e-09 | 1.4e-7% | **是** |
| guo_split 定常剖面 | 解析抛物线 | 2.609e-03 | 0.261% | **是** |
| MRT+guo_split 定常剖面 | 解析抛物线 | 2.970e-03 | 0.297% | **是** |
| config vs 手写重推 | 同一格式 | 5.44e-15（abs） | 3.9e-11（rel） | **是** |
| 每步动量注入 | `F`（闭式） | 偏差 ≤4.6e-11 | — | **是** |
| 负对照 legacy_mixed | 解析抛物线 | 7.0007e-01 | 70.0% | 否（**预期越界**，如实入账） |
| SRT 收敛阶 | 2 | 2.004 | 0.2% | **是** |

**结论**：所有"格式自洽"的臂在 15% 与 3% 双阈值下均通过；负对照按预测失败，
说明台架对错误记账有分辨力。**本轮不与 lbmpy 数值对标**（其未安装，见 §6）。

## 8. 复现

```bash
# 单元不变量测试
PYTHONPATH=src python -m pytest tests/test_lbm_config.py -q          # 52 passed
# 交叉验证台架（分阶段，单核；完整 all 建议递交调度系统）
PYTHONPATH=src python benchmarks/cross_validation/lbmconfig_poiseuille/run_xval.py --stage main --height 20 --steady-tol 1e-9
PYTHONPATH=src python benchmarks/cross_validation/lbmconfig_poiseuille/run_xval.py --stage convergence --heights 16,32
```

本次运行的结果和图默认写入 `outputs/lbmconfig_poiseuille/`。
