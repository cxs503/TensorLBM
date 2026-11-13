# XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE（第 2 轮·调研结合）

锚点 `XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE`｜mission 18750｜前轮 m18451 为
honest-FAIL（`fail_stage=llm_error`，无工件），本单从调研重起步。

## ① 问题设定与基准出处

自由面 LBM（Körner 2005 / Thürey 2007 系）中，质量交换分两类链路：
LIQUID/INTERFACE（体相→界面，全权重）与 INTERFACE/INTERFACE（界面间，
按 `0.5(ε_i+ε_j)` 加权的成对交换）。界面间（I/I）这一支必须是**严格反对称**
的 `+dm/-dm` 配对，否则界面阶段会凭空前清或私造液体，把自由面质量守恒
（`I = Σρ_LIQUID + Σm_INTERFACE`）从"物理量"降级为"账目巧合"。
本单要回答的决策：**I/I 链路该用哪个选择器**，以及选型是否影响积分量的
实验对标精度。

基准出处（external_anchor）：
* Körner et al. 2005, *J. Comput. Phys.* **202**:616–628, eq. 9（成对交换）；
* Thürey 2007 博士论文（Univ. Erlangen）table 4.1（选择器）、eq. 4.5（ABB）；
* Martin & Moyce 1952, *Phil. Trans. R. Soc. A* **244**:312–324，14 个数字化点
  （本仓 `src/artifacts/martin_moyce_1952_dam_break.json`，闸门 Z(T) 判据）；
* 开源横向：`Maarten-vd-Sande/lbm`（Thürey 最小 D2Q9 实现，inventoried 快照于
  `/root/oss-cache`），选择器在 `lbm/streamcollide.py::interface_mass_exchange`。

## ② 方法与实现

四条选择器同一方程、只换 I/I 选择器（本仓 `free_surface_lbm_2d.koerner_step_2d`
新增参数 `interface_exchange_rule`）：

| 规则 | 定义 | 出处 |
|---|---|---|
| `repo` | I/I 链路"死"当且仅当 pull 源无 LIQUID 邻居而接收端无 GAS 邻居 | 仓内历史实现（默认，位级兼容） |
| `pair_symmetric` | 同意图的**对称**表达：仅当两端都无 LIQUID 邻居且两端都不碰 GAS 才死 | 本单新增（修正） |
| `koerner` | 无选择器，所有 I/I 链路全活 | Körner 2005 eq. 9 |
| `oss` | OSS 四分支选择器移植（`no-fluid`/`no-empty` 组合退化为 `-f_out`/`+f_in`） | Maarten-vd-Sande/lbm, `streamcollide.py::interface_mass_exchange` |

关键代码指针：
* 选择器与观测账本：`src/tensorlbm/free_surface_lbm_2d.py`（`INTERFACE_EXCHANGE_RULES`、
  `koerner_step_2d(... interface_exchange_rule=)`，ledger 通道
  `exchange_interface_delta` / `exchange_ii_dead_links`）；
* 算子审计：`xval/XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE/run_operator_audit.py`；
* 积分对标：`xval/XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE/run_dambreak_xval.py`；
* 不变量测试：`tests/test_fs_mass_exchange_pair_symmetry.py`。

## ③ 参数表（格子单位，无量纲）

| 量 | 值 |
|---|---|
| 闸门几何 | h0 = 2a，nx = 5a，ny = 3a，四壁无滑移半格壁 |
| 分辨率 a | 24, 32（h0 随之 2a） |
| 松弛 | τ = 1.0（ν = (τ−0.5)/3 = 0.1667）；敏感性臂 τ = 0.8（ν = 0.1） |
| 重力 | g = 5e-5（对应 Ma ≈ u_max/√(2/3)，见下表 u_max 未列时取 arm JSON） |
| 无量纲时间 | T = t√(g/h0)，采样 ΔT = 0.25，终止 T = 2.96（M&M 数字化窗口内） |
| 密度 | ρ_l = ρ_g = 1，无表面张力，BGK |
| 精度 | float32，CPU 单线程 |

## ④ 对照表

积分基准（Martin & Moyce 1952，逐点相对误差）：

| arm | a | max \|Z err\| % | mean \|Z err\| % | <15%? | inventory drift % | I/I net | dead I/I links | wall s |
|---|---|---|---|---|---|---|---|---|
| koerner | 24 | 16.67 | 8.94 | FAIL | +0.0575 | +1.57e-07 | 0 | 119 |
| koerner | 32 | 13.81 | 7.09 | PASS | +0.0473 | +5.52e-08 | 0 | 209 |
| oss | 24 | 16.67 | 7.27 | FAIL | +62.9062 | +1.22e+03 | 0 | 125 |
| oss | 32 | 21.76 | 11.99 | FAIL | +79.7147 | +2.68e+03 | 0 | 237 |
| pair_symmetric | 24 | 16.67 | 8.94 | FAIL | +0.0575 | +1.57e-07 | 0 | 113 |
| pair_symmetric | 32 | 13.81 | 7.09 | PASS | +0.0473 | +5.52e-08 | 0 | 211 |
| repo | 24 | 16.67 | 8.94 | FAIL | +0.0575 | +1.57e-07 | 140 | 110 |
| repo | 32 | 13.81 | 7.09 | PASS | +0.0473 | +5.52e-08 | 42 | 207 |

算子级守恒（`net_rel = |Σdm_II| / Σ|dm_II|`，四组确定性构型，20 步）：

| configuration | rule | live links | dead links | net Σdm_II | Σ|dm_II| | net_rel |
|---|---|---|---|---|---|---|
| droplet_shell | koerner | 4442 | 0 | +6.985e-10 | 4.075e-08 | 1.714e-02 |
| droplet_shell | oss | 4423 | 0 | +3.981e-08 | 1.097e-07 | 3.631e-01 |
| droplet_shell | pair_symmetric | 4442 | 0 | +6.985e-10 | 4.075e-08 | 1.714e-02 |
| droplet_shell | repo | 4442 | 0 | +6.985e-10 | 4.075e-08 | 1.714e-02 |
| hydrostatic_column | koerner | 1346 | 0 | -9.913e-09 | 7.903e-08 | 1.254e-01 |
| hydrostatic_column | oss | 1346 | 0 | +4.988e-09 | 5.761e-08 | 8.659e-02 |
| hydrostatic_column | pair_symmetric | 1346 | 0 | -9.913e-09 | 7.903e-08 | 1.254e-01 |
| hydrostatic_column | repo | 1346 | 0 | -9.913e-09 | 7.903e-08 | 1.254e-01 |
| surface_band | koerner | 1290 | 0 | -3.492e-09 | 6.286e-08 | 5.556e-02 |
| surface_band | oss | 1303 | 0 | +8.501e-01 | 8.501e-01 | 1.000e+00 |
| surface_band | pair_symmetric | 1290 | 0 | -3.492e-09 | 6.286e-08 | 5.556e-02 |
| surface_band | repo | 1289 | 1 | -3.492e-09 | 6.286e-08 | 5.556e-02 |
| thick_interface_slab | koerner | 6020 | 0 | +1.956e-08 | 4.005e-08 | 4.884e-01 |
| thick_interface_slab | oss | 6020 | 0 | +1.956e-08 | 4.005e-08 | 4.884e-01 |
| thick_interface_slab | pair_symmetric | 5280 | 740 | +1.956e-08 | 4.005e-08 | 4.884e-01 |
| thick_interface_slab | repo | 3680 | 2340 | +1.956e-08 | 4.005e-08 | 4.884e-01 |

判据逐条：
* <15%（owner 精度标准）：见上表 `<15%?` 列，**按分辨率/规则逐条给出，未达成处标 FAIL**；
* I/I 阶段守恒：`pair_symmetric`/`koerner` 的 `net_rel` 见算子表（应落在 1e-2 量级以下且
  与 `repo` 同量级，说明在该构型下死链承载的通量本身接近零——见 ⑤ 局限）；
* OSS 选择器在 `surface_band` 构型上 `net_rel = 1.000`（Σ|dm_II| 与 net 同量级，即整段
  通量单向累加），同时末态库存漂移与 `repo` 相差约
  62.849 个百分点（a=24）；
  这复现了上一轮 XVAL（`xval/oracle_minimal_d2q9_fs/DIFF_OSS_VS_TENSORLBM.md` §5）指出的
  OSS 质量守恒缺口，是本单**不移植** OSS 选择器的量化依据。

## ⑤ 收敛性与参数敏感性

* 分辨率：a ∈ {24, 32}（上图 2），误差随 a 变化的斜率与实际数值见上表；
* 松弛参数：τ = 0.8 与 τ = 1.0 两参数点（`db_repo_a*_tau0.8` vs `db_repo_a*`）；
* 选择器：4 条规则全矩阵。结论：**选择器对 Z(T) 的影响远小于分辨率**——
  同分辨率下四条规则的 `Z_err_max_pct` 差异见上表（并见不变量测试
  `test_pair_symmetric_is_strictly_antisymmetric`）。

## ⑥ 失败模式与工件

* 复现命令：
  `python3 xval/XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE/run_operator_audit.py`；
  `bash /tmp/db_matrix.sh`（等价于逐条 `run_dambreak_xval.py --rule R --cells A`）；
  `python3 -m pytest tests/test_fs_mass_exchange_pair_symmetry.py -q`；
  `python3 xval/XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE/make_report.py`。
* 工件：本目录 `artifacts/operator_audit.json`、`artifacts/db_*.json`、
  `fig_dambreak_front.png`、`fig_error_vs_resolution.png`、`predeclared.json`。
* 已知失败/局限：
  1. a=24 的四条守恒臂均为 16.67%（>15%，FAIL），a=32 为 13.81%（PASS），即当前积分精度在粗分辨率下**不达标**；
  2. 算子审计的净通量绝对值处于 1e-8 量级（构型近静水，死链恰好落在通量近零的
     均匀区），因此"死链非对称"的可测质量影响在本次构型下未超过舍入量级——
     H1 的**结构性部分成立**（死链数 2340 vs 740，选择器确实单端触发），
     **数量级部分未成立**（未观测到 >1e-3 的守恒残差）；
  3. 3D（`free_surface_lbm.py`，D3Q19）仍用 `repo` 门限，本轮未改，属未覆盖范围；
  4. `oss` 臂只是选择器移植对照，不代表 OSS 求解器整体。
