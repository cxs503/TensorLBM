# 方腔 Ghia 度量的「几何本体」审计与约定落盘（m16740）

**锚点**：REPRO-LBM-HOU1995（论文复现·Simulation of Cavity Flow by the Lattice Boltzmann Method [Hou et al. 1995]）——
该复现的全部误差数字都由 `compare_ghia` 这个度量产出，本单先把度量本身钉死。
**结论一句话**：`ghia_errors` 是 TensorLBM 全部二维方腔验收（Re=100/400/1000/5000）的判据本体，
但它隐含的「节点索引 → 物理坐标」映射与**本求解器自己的边界配方对不上**：真实腔体是
`H_x = nx-2`、`H_y = ny-1.5`（**不是正方形**），而 `Re = u_lid·nx/ν` 用的 `L = nx` 与这三条长度
**都不相等**（n=128 时差 1.2 %–1.6 %）。本单量化了「完美解在不同映射下被判多少误差」的地板，
把约定、壁面平面与整条中线场一并落盘，并**否证**了 m16339 遗留的建议「改用对称 w=+0.5」——
本配方下它是最差的选择（Re=1000/n=128 横向地板 2.752 % vs 历史默认 1.350 % vs 配方一致映射 0.434 %）。

## 1. 案例介绍（物理背景 / 几何 / 参数）

顶盖驱动方腔流是不可压 N-S 的经典验证算例，Ghia, Ghia & Shin（*J. Comput. Phys.* **48**(1982) 387-411）
给出 Re=100/400/1000/3200/5000/7500/10000 的中心线速度表值，是 LBM 界的公共标尺。仓库内
`GHIA_RE100/400/1000/5000` 四张表被 `run_lid_driven_cavity`、`/benchmarks/ghia` 端点、
`generic_run`、`cross_solver_verification`、`scripts/run_gap_pp2_lid_wall_diff.py` 共同消费。

`run_lid_driven_cavity` 的离散与边界（决定了度量的几何本体）：

| 项 | 实现 | 推出的壁面平面（节点索引单位） |
|---|---|---|
| 底壁 / 左壁 | 固体节点行/列 + `bounce_back_cells` | `0.5` / `0.5` |
| 右壁 | 固体节点列 `nx-1` + `bounce_back_cells` | `nx-1.5` |
| 顶盖（动壁） | `zou_he_moving_lid` 直接写在**流体节点行** `ny-1` | `ny-1` |
| 参考长度（Re） | `ν = u_lid·nx/Re` → `L = nx` | 与上述三条长度均不同 |

因此「方腔」在本实现里是 `H_x = nx-2`（=126@n=128）、`H_y = ny-1.5`（=126.5@n=128）、
长宽比 `H_y/H_x = 1.00397`；而 Re 用的 `L = nx = 128`，比 `H_x` 大 1.59 %、比 `H_y` 大 1.19 %。

## 2. 怎么做的（方法 / 流程 / 工具链）

1. **几何解析**：从 BC 实现直接推出四条壁面平面，新增 `cavity_wall_planes(nx, ny)`（`src/tensorlbm/lid_driven_cavity.py`）。
2. **度量地板（metric floor）**：把 Ghia 表经单调三次（PCHIP）变成主剖面，在**配方真实壁面平面**上采样
   得到「完美解场」，再用候选映射（历史 `w=0` / 对称 `w=+0.5` / 配方一致）逐点插值回 Ghia 参考行，
   计算 RMSE → 该误差**不含任何求解误差**，纯度量人造项（新增 `ghia_metric_floor`）。
3. **约定显式化**：`centreline_coordinates_between(n, lower, upper)` 支持任意壁面平面；`compare_ghia`
   新增 keyword-only 的 `y_planes/x_planes/exclude_walls`（默认行为逐位不变），返回值加 `n_scored_*`。
4. **落盘**：`ghia_metric_report()` 一次算全三种约定义（full + 只算内行），`run_lid_driven_cavity`
   把 `ghia_metric`（壁面平面 / 四组分数 / 解析地板 / **整条中线 u、v 场**）写进 `run_metadata.json`，
   CSV 坐标改用配方一致映射；新增配置项 `ghia_metric_convention ∈ {legacy_w0, symmetric_w+0.5, recipe_planes}`
   （默认 `legacy_w0`，历史数字不动）决定 headline `ghia_errors` 用哪种约定。
5. **工具链**：`scripts/audit_cavity_metric_geometry.py`（零仿真，输出 JSON+图）；
   `tests/test_cavity_metric_geometry.py`（26 条不变量测试）。

## 3. 计算结果（程序化输出，零手抄）

来源：`audit/m16740/cavity_metric_geometry.json`、`runs/m16740_cavity_metric/measured_transient.json`。

### 3.1 完美解在 n=128 上的度量地板（RMSE，% of `u_lid`，全参考行）

| Re | 映射 | u 向（全行） | v 向（全行） | u 向（只算内行） | v 向（只算内行） |
|---|---|---|---|---|---|
| 100 | 历史 w=0（默认） | 0.142 % | 0.392 % | 0.151 % | 0.417 % |
| 100 | 对称 w=+0.5 | 1.172 % | 0.793 % | 1.248 % | 0.844 % |
| 100 | 配方一致（本求解器 BC） | 0.038 % | 0.126 % | 0.015 % | 0.006 % |
| 400 | 历史 w=0（默认） | 0.283 % | 0.857 % | 0.301 % | 0.913 % |
| 400 | 对称 w=+0.5 | 1.629 % | 1.725 % | 1.734 % | 1.836 % |
| 400 | 配方一致（本求解器 BC） | 0.084 % | 0.255 % | 0.039 % | 0.023 % |
| 1000 | 历史 w=0（默认） | 0.467 % | 1.350 % | 0.497 % | 1.437 % |
| 1000 | 对称 w=+0.5 | 1.932 % | 2.752 % | 2.056 % | 2.930 % |
| 1000 | 配方一致（本求解器 BC） | 0.188 % | 0.434 % | 0.047 % | 0.082 % |
（`u` 向取 `u` 参考行，`v` 向取 `v` 参考行；三种映射的同轴对比见 JSON 中 `metric_floor_percent_of_u_lid`。）

要点：
* 对称 `w=+0.5` **最差**：它把参考表的壁面行（`y,x = 0, 1`）推到节点跨度之外，`np.interp` 静默钳制
  （m16339 的 F3 已被本单用数字证实并放大到 2.75 %）。
* 配方一致映射的地板比历史默认再降 2.5–3.3 倍（Re=1000 横向 1.350 %→0.434 %，只算内行 1.437 %→0.082 %）。
* 地板随网格**只按一阶下降**（n 加倍约减半，测试 `test_metric_floor_is_not_a_high_order_small_quantity`
  固定 `1.3 < 比值 < 3.0`）——即它是 O(h) 系统项，**不会**随网格收敛消失。

### 3.2 解析预测 vs 实测（合成「完美解」场，n=128）

| Re | 量 | 解析预测（`ghia_metric_floor`） | 实测（`compare_ghia` 同一合成场） | 判定 |
|---|---|---|---|---|
| 400 | rmse_u（全行，历史映射） | 0.283 % | 0.283 % | 一致（测试容差 rel 5 %，实测 ~1e-12） |
| 400 | rmse_v（全行，历史映射） | 0.857 % | 0.857 % | 一致 |
| 1000 | rmse_u（全行，历史映射） | 0.467 % | 0.467 % | 一致 |
| 1000 | rmse_v（全行，历史映射） | 1.350 % | 1.350 % | 一致 |
| 400 | 内行 rmse_u / rmse_v（配方映射） | 0.039 % / 0.023 % | < 0.1 %（测试阈值） | 通过 |
| 1000 | 内行 rmse_u / rmse_v（配方映射） | 0.047 % / 0.082 % | < 0.1 %（测试阈值） | 通过 |

### 3.3 真实流场（nx=32/48，Re=100，3000 步，**未收敛**）

| nx | 映射 | rmse_u | rmse_v | 内行 rmse_u | 内行 rmse_v |
|---|---|---|---|---|---|
| 32 | legacy_w0 | 4.852 % | 3.446 % | 5.166 % | 3.669 % |
| 32 | recipe_planes | 5.154 % | 4.906 % | 5.185 % | 5.020 % |
| 32 | symmetric_w+0.5 | 7.240 % | 2.762 % | 7.708 % | 2.940 % |
| 48 | legacy_w0 | 4.800 % | 1.794 % | 5.110 % | 1.909 % |
| 48 | recipe_planes | 5.067 % | 2.852 % | 5.033 % | 2.929 % |
| 48 | symmetric_w+0.5 | 6.059 % | 1.282 % | 6.450 % | 1.365 % |

### 3.4 图

`figures/m16740_cavity_metric_geometry.png`：三面板（a）Re=100/400/1000 的 u 向地板 vs n（log-log，三映射）；
（b）n=128 时 v 向地板三映射柱状对比；（c）几何偏差 `H_y/H_x-1` 与 Re 长度亏损 `1-H_x/nx` vs n。

## 4. 注意点（坑 / 局限 / 适用范围）

* **瞬态场证伪了「换约定能降误差」的直觉**：3.3 节 3000 步场（求解误差 ~5 %）上，配方映射反而比历史默认
  更差（32²：5.15 % vs 4.85 %）。地板量级（0.1 %–2.8 %）比未收敛求解误差小 1–2 个量级，**约定排序
  只有在收敛场（残差 ≲1e-4）上才可分辨**。因此本单不把「选某映射」当作精度旋钮，只把它当作**可比性前提**。
* **本单未跑收敛的真实场**：load15 ≥ 100（容量纪律只做轻量），nx=64/4000 步的尝试在 115 s 内未跑完即放弃；
  3.3 节数字全部标注为瞬态。收敛场复核步骤已固化为命令（见第 6 节）。
* **`Re` 与几何长度不一致（1.2 %–1.6 %）本单不静默修正**：改 `ν` 会改变 `τ` 与全部历史数字；
  只在元数据里给出 `implied_re_ratio_*` 供引用。这是一条待裁决项（若裁决改用 `H_x`，Re=1000 实际应为 984）。
* **`np.interp` 钳制仍然存在**（未改库语义，避免静默改历史数字）：只提供了 `exclude_walls=True`
  与「壁面行承担残差」的定量证据；任何报价必须同时给 `wall_planes`。
* 本单只覆盖 2D 方腔；3D 方腔（`lid_driven_cavity_3d`）与 AMR 方腔（`ghia_comparison`）另有独立度量，未审。

## 5. 发现与修复

| # | 严重度 | 发现 | 证据 | 处置 |
|---|---|---|---|---|
| F1 | 高（可比性） | 标称「方腔」实为 `H_x=nx-2` × `H_y=ny-1.5`，长宽比 1.00397@n=128；`Re` 的 `L=nx` 与三者均不等（1.19 %–1.59 %） | `cavity_wall_planes(128)` | 几何显式化 + 入元数据 `wall_planes` / `implied_re_ratio_*` |
| F2 | 高（可比性） | 历史 `ghia_errors` 未落盘约定，且 run 只存 RMSE 不存中线场 → 数字不可复核（m16339 §5 明列的后续单） | 元数据对比 | 新增 `ghia_metric`（三约定义 + 解析地板 + 整条中线 u/v + 壁面平面） |
| F3 | 中（否证遗留建议） | m16339 建议「半程反弹壁宜用 `w=+0.5`」——本配方下它把壁面行钳制，地板 2.75 % 反而最差 | 3.1 节 | 文档化 + 测试 `test_recipe_map_beats_symmetric_half_cell_map` 固定排序 |
| F4 | 低（一致性） | `ghia_comparison.csv` 坐标写死 `linspace(0,1,n)`，与库内任何约定都不对应 | 源码 | 改用配方一致映射，测试比对 CSV 坐标 |
| F5 | 阻断（无） | 未发现会静默判 PASS 的路径：非有限场三约定义全部 NaN（测试固定） | `test_diverged_field_never_passes` | 无需修 |

## 6. 误差判定（关键量 vs 基准，逐项）

| 关键量 | 基准 | 本单值 | 误差 | <3 % ? |
|---|---|---|---|---|
| 合成完美解的内行 rmse（Re=400/1000，配方映射，n=128） | 解析地板 0.039 %/0.047 % | 实测同值 | 相对差 ~1e-12 | **是**（度量实现与解析预测一致） |
| 历史默认映射引入的地板（Re=1000，n=128，v 向） | 0（理想可比性） | 1.350 % | — | 是（但为 O(h) 系统项，不可忽略） |
| 对称 w=+0.5 映射地板（Re=1000，n=128，v 向） | 0 | 2.752 % | — | 是（但为历史默认的 2.04 倍，属最差选择） |
| 配方一致映射地板（Re=1000，n=128，v 向，只算内行） | 0 | 0.082 % | — | **是** |
| 几何长宽比偏差（n=128） | 1（正方） | 1.00397 | 0.40 % | 是 |
| `Re` 长度亏损（n=128，对 H_x） | 0 | 1.56 % | — | 是，但直接进入 Re 定义 |
| 收敛场三约定义实测排序 | 解析地板排序 | **未测**（瞬态场不成立，见 §4） | — | **未判定（not_run）** |

## 7. 复现命令

```bash
# 零仿真度量几何审计（秒级）
PYTHONPATH=$PWD/src python scripts/audit_cavity_metric_geometry.py

# 不变量测试（26 条）
python -m pytest tests/test_cavity_metric_geometry.py -q

# 收敛场复核（容量允许时；本机 load15 ≥100 未执行）
PYTHONPATH=$PWD/src python -c "
from pathlib import Path
from tensorlbm.lid_driven_cavity import LidDrivenCavityConfig, run_lid_driven_cavity
run_lid_driven_cavity(LidDrivenCavityConfig(nx=128, re=1000.0, u_lid=0.1, n_steps=200000,
    output_interval=50000, output_root=Path('runs/m16740_cavity_metric'), overwrite=True))
"
# 然后读 run_metadata.json -> ghia_metric.scores 三约定义即为收敛场排序
```