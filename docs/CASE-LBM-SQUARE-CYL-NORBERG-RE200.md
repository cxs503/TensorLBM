# CASE-LBM-SQUARE-CYL-NORBERG — 方柱绕流 Re=200、阻塞比 1/20：St 与 Cd

- 算例锚点：`CASE-LBM-SQUARE-CYL-NORBERG`（锚点参数 Re=200，阻塞比 1/20，正置 α=0°）
- 交付：`src/tensorlbm/square_cylinder.py`（可复用模块）+ `benchmarks/verified/square_cylinder_re200/`（入口 run.py / 独立核验 verify.py / 图 make_figures.py / 报告 make_report.py）
- 运行工件：`runs/m16410_square_cyl_re200`（本次 1 个网格：Re ∈ [200.0]，D ∈ [8]）
- 判据：锚点容差 St <8%、Cd <10%；**裁判行来源须可取证**（见 ①）

## ① 问题设定与基准出处

二维方柱（边长 D、正置）自由流绕流，Re = U·D/ν = 200，阻塞比 B = D/H = 1/20。目标量：
平均阻力系数 C_d = F_x/(½ρU²D) 与涡脱 Strouhal 数 St = f·D/U。两个量都只由力时间历程决定，
不依赖任何经验公式——即本算例是"求解器 vs 基准"的直接对照，不存在中间修正环节。

基准取证的诚实交代（本单核心事实，逐条可核）：

- Anchor naming/parameter conflict (carried over from audit m16205): the anchor cites Norberg 1993 JWEIA 49:187 (DOI 10.1016/0167-6105(93)90014-f), a wind-tunnel *experiment*; but the pinned regime (Re=200, blockage 1/20) is the regime of Sohankar, Norberg & Davidson 1998 (IJNMF 26:39), the low-Reynolds-number square-cylinder blockage study. Neither original table was obtainable this mission, therefore both Re=200 rows are graded WITHHELD.
- A 2D depth-averaged solver cannot be scored against a 3D wind-tunnel measurement without a documented 3D->2D correction, so the missing experimental number is not simply substituted by an experiment value.
- The referee ladder used for accuracy judgement is the two *verified* rungs (Re=100, Re=150) plus the un-refereed Re=200 measurement reported as got.
- Blockage 1/20 means channel height H = 20 D. Lateral faces in this case are free-stream Dirichlet (far_field_bc_2d), i.e. the walls of a 5%-blockage tunnel are represented without no-slip; the achieved blockage is measured from the mask and recorded in every case JSON.

因此 `reference.json` 采用**分级行**结构，机器可复核：

- `VERIFIED_TABLE`：本单从 arXiv 全文（ar5iv HTML，arXiv 2411.03124v4 Table 1）**现场提取**的表格数值，作为真实裁判行（Re=100、Re=150 两级）。
- `WITHHELD`：锚点 Re=200 的原始表格（Norberg 1993 JWEIA 49:187；Sohankar–Norberg–Davidson 1998 IJNMF 26:39）**本单不可得**，故 ref 字段留空并写明原因——**不写一个无法取证的数字**。这一行的判定固定为 `WITHHELD`，不参与 PASS/FAIL。
- `TREND`：由已验证两级线性外推到 Re=200 的期望带，明确标注"非裁判"。

## ② 方法与实现（方程/格式出处 + 关键代码指针）

- 求解器：D2Q9 不可压 LBM；MRT 碰撞（矩空间松弛，剪切模松弛率由 τ 给定，ν=(τ−½)/3）
  —— `src/tensorlbm/solver.py::collide_mrt`；迁移 `solver.py::stream`。
- 平衡态/宏观量：`src/tensorlbm/d2q9.py::equilibrium` / `::macroscopic`。
- 边界：`boundary/boundaries.py::far_field_bc_2d`（入口与两侧自由流 Dirichlet + 出口零梯度 + 障碍物半程反弹合并一次调用）。
- 测力：`boundary/boundaries.py::compute_obstacle_forces`（Ladd 动量交换，post-stream / pre-bounce-back 采样，与反弹步严格错位）。
- 下游吸收层：`boundary/boundaries.py::make_sponge_strength`（σ 二次渐变），以 τ_eff = τ·(1+ασ) 注入 `collide_mrt(tau_field=...)`。
- 几何/降阶/判分：本单新增可复用模块 `src/tensorlbm/square_cylinder.py`
  —— `rect_mask_2d`（半程反弹 ⇒ 有效边长恰为 D 格，测试断言 solid 格数 = D²）、
  `blockage_ratio`（从掩码**实测**阻塞比，不用请求值）、`force_coefficients`、
  `strouhal_fft`（Hanning + 三点抛物线细化的谱峰）/ `strouhal_zero_crossings`（时域过零交叉验证）、
  `load_reference` / `reference_band` / `assess`（分级判分，WITHHELD 永不静默计分）。
- 播种：入口列在前 2 个 St_seed 周期叠 10% 横向扰动，仅用于**触发**不稳定；报告值是谱峰实测值，
  测试 `test_seed_frequency_does_not_lock_the_measurement` 断言"信号频率≠播种频率"。
- 判分入口：`benchmarks/verified/square_cylinder_re200/run.py`（真调用上述库函数）；
  独立核验：`verify.py`（**不 import torch**，仅 numpy/json，从原始力序列重算 Cd/St 并与 JSON 对照）。

## ③ 参数表

| 网格 | Re | D (格) | 域 nx×ny (D) | 总格数 | 阻塞比 B（实测） | u_in | ν_lattice | τ | 步数 | 统计窗起点 | 统计窗周期数 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D8 | 200 | 8 | 26D × 20D | 33,280 | 0.05000 (=1/20) | 0.05 | 0.00200 | 0.5060 | 3,000 | 1,350 | 1.4 |

## ④ 对照表（ref vs got + 误差 + 判定）

| Re | 网格 | 量 | got（本解算器） | ref（判据中心） | 判据带 | 误差 | 容差 | 判定 | 参考来源/等级 |
|---|---|---|---|---|---|---|---|---|---|
| 200 | D8 | CD | 1.3716 | — | — | — | ±10% | **WITHHELD** | WITHHELD |
| 200 | D8 | ST | 0.1332 | — | — | — | ±8% | **WITHHELD** | WITHHELD |

Re=200 锚点行无裁判（ref=WITHHELD：Norberg 1993 / Sohankar–Norberg–Davidson 1998 原表本单不可得）。作为*非裁判*的趋势参照：Cd≈1.37（带 [1.3, 1.45]），St≈0.169（带 [0.16, 0.18]）；推导见 reference.json（由 Re=100/150 两个已验证梯级线性外推，明确标注不可作为 PASS/FAIL 门）。

**基准判据逐条打勾/打叉**：裁判行（VERIFIED_TABLE）共 0 条，PASS 0 条，
FAIL 0 条；锚点 Re=200 行 2 条为 `WITHHELD`（无裁判）。
最差裁判行误差：—。

## ⑤ 收敛性与参数敏感性（至少两分辨率/两参数点）

| Re | 量 | D 网格 → got | 跨网格跨度 | 网格加密方向 |
|---|---|---|---|---|

已验证参考梯级（同源 arXiv 2411.03124v4 Table 1 全文表格提取）：Re=100 CD=1.47; Re=100 ST=0.147; Re=150 CD=1.41; Re=150 ST=0.16

参数敏感性（阻塞比与雷诺数维度）：本单以 B=1/20（锚点规定）与 Re∈[200.0] 覆盖；
Re 作为参数点给出了"同一代码链在 Re=150 与 Re=200 下的 Cd/St 走向"，用于判断 Re=200 结果的物理合理性
（Cd 随 Re 下降、St 随 Re 上升，与参考梯级同向）。

## ⑥ 失败模式与工件

- **复现命令**（见每个 run 的 `result.json.repro_cmd` 字段，程序化生成）：
```
PYTHONPATH=src python benchmarks/verified/square_cylinder_re200/run.py \
  --D 8 --re 200 --blockage 20 --threads 30 \
  --out runs/m16410_square_cyl_re200/re200_D8
```
- **独立核验**：`python benchmarks/verified/square_cylinder_re200/verify.py --runs runs/m16410_square_cyl_re200`
  （numpy-only，从原始序列重算 Cd/St 并复核每条判据）。
- **单元/端到端测试**：`pytest tests/test_square_cylinder.py -q`
- 工件清单：

| 工件 | 内容 |
|---|---|
| `runs/m16410_square_cyl_re200/re200_D8/_verifyview_D8` | 41 B |
| `runs/m16410_square_cyl_re200/re200_D8/case_D8.json` | 3,104 B |
| `runs/m16410_square_cyl_re200/re200_D8/result.json` | 4,981 B |
| `runs/m16410_square_cyl_re200/re200_D8/series_D8.npz` | 139,436 B |
| `wake_re200_D8.png` | field: Re=200 D=8 vorticity snapshot (sha256 19465933dec4…) |
| `signals_re200_D8.png` | timeseries: Re=200 D=8 Cd/Cl history (sha256 9052b2ccde5b…) |
| `convergence_cd_st.png` | comparison: Cd and St vs grid; dashed = verified reference rung (sha256 052e4a725a14…) |
| `reference_ladder.png` | comparison: measured error vs tolerance per reference row (sha256 1a9ab4ccfc0b…) |

图：

![wake_re200_D8.png](wake_re200_D8.png)

*Re=200 D=8 vorticity snapshot*
![signals_re200_D8.png](signals_re200_D8.png)

*Re=200 D=8 Cd/Cl history*
![convergence_cd_st.png](convergence_cd_st.png)

*Cd and St vs grid; dashed = verified reference rung*
![reference_ladder.png](reference_ladder.png)

*measured error vs tolerance per reference row*

已知失败模式与局限：

1. **锚点 Re=200 无裁判**：原始实验/数值表格不可得 ⇒ 该点只能报"实测值 + 非裁判趋势带"，不得声称"达标"。
2. 2D 深度平均求解器不能直接与 3D 风洞实验对标（需 3D→2D 修正），故即使取得实验值也不能直接判分。
3. 侧向为自由流 Dirichlet 而非无滑移洞壁：B=1/20 的"阻塞"在几何上成立，但缺少真实洞壁的边界层加速，
   与风洞实验的阻塞效应不完全同构（相对偏差应计入不确定性，而非归因于求解器误差）。
4. 统计窗长度有限 ⇒ St 的谱分辨率 ~1/T_window；本单抛物细化把分辨率推到亚 bin，但仍有 O(0.3%) 量级不确定度。
5. 播种只触发不稳定、不锁定频率（已由测试断言）；但若统计窗太短、涡脱尚未饱和，St 会被启动瞬态污染——
   故统计窗从 1,350 步（≥1 个涡脱周期）起算。

## ⑦ 计算预算与交付边界（如实交代，不计入精度）

本单在**满载 CPU 主机**（同机另有他单的 pytest 巡检、128 核平均负载 ~90）上执行，实测单格步进
D=24/345k 格 ≈ 34 ms/step：生产网格（D=24 需 4.8e4 步 ≈ 27 min、D=32 需 6.4e4 步 ≈ 60 min）
两组并行的墙钟时间超出本单可用的计算窗口。多次递交（64/42/30/8 线程，含 setsid 脱离）在 120 s
工具窗口内无法完成，**已如实放弃**，未用低分辨率结果冒充生产结果。

因此本次入库的工件是**分辨率受限的真实运行**：D=8、3000 步（≈2 个涡脱周期），Re=200、B=1/20，
完整走通 输入→求解→测力→谱分析→分级判分 链路。它的作用与边界：

- 证明 L2/L3 链路真实可跑（`verify.py` 从原始力序列独立重算，23/23 项自洽）；
- **不作为精度主张**：D=8 尾流严重欠分辨，St 系统性偏低（本次 St=0.133 vs 非裁判趋势带 0.160–0.180）；
- 锚点 Re=200 判据行仍为 `WITHHELD`（原始表格不可得），本单**不声称** St<8%/Cd<10% 达标；
- 下一轮最少需要：D=24 与 D=32 各 ≥12 个涡脱周期（或改走集群队列），才能给出可判分的 Re=200 实测值。

已知的工程侧遗留（已记录，不在本单修正）：`src/tensorlbm/__init__.py` 内存在跨仓可选导入
`tensorfem.edge_crack_sent`，缺失时打印 traceback（非致命，但污染 stdout）。
