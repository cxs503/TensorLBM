# T-A2: de Vahl Davis 自然对流方腔 — thermal 2D 壁 BC 修复 + 基准重扫

## 概述

- **问题**：方腔自然对流标准基准（de Vahl Davis 1983）：左壁等温 T_hot、右壁
  等温 T_cold、上下绝热、四壁 no-slip；Pr=0.71（空气），Ra∈{1e3, 1e4}，
  网格 N∈{64, 128, 256} 共六案。
- **本条目双重性质**：(1) 修复库 `tensorlbm/thermal.py` 2D 温度壁 BC 的两个
  缺陷（T-A 诊断的绝热壁热泄漏；修复过程中追加发现的第二层缺陷，见下）；
  (2) 用修复后的包完成六案基准重扫。
- **参考值**（预注册于任何模拟数字之前，≥2 独立来源交叉 + 控制器独立核对）：

  | 量 | Ra=1e3 | Ra=1e4 |
  |----|--------|--------|
  | **Nū₀（主判据）** | **1.118** | **2.243** |
  | **u_max（扩散尺度 u_nd=u_lu·H/α）** | **3.649**（y/H=0.813） | **16.178**（y/H=0.823） |
  | **v_max** | **3.697**（x/H=0.178） | **19.617**（x/H=0.119） |
  | Nu_max / Nu_min（形态自检） | 1.505 / 0.692 | 3.528 / 0.586 |

  变体披露：v_max(Ra=1e4) 文献有 19.617/19.643 两值（差 0.13%），取 19.617。

## 库缺陷与修复（两层）

### 缺陷 1：绝热壁热泄漏（controller T-A 诊断复现）

- 原库 `apply_temperature_boundaries`：上下绝热用逐方向 bounce-back、左右
  等温用 anti-bounce-back，但 `temperature_stream` 是**全周期** stream——
  从下壁流出的分布绕行到顶行。节点级事后覆写销毁/凭空产生绕行能量：
  上下壁流场不对称（本构型必然）⇒ 净热泄漏。
- fp64 守恒预算直测（Ra=1e4 N=64 发展态）：**−2.152e-2 ΣT/步**（独立复现
  controller T-A 的 −2.208e-2/步，不同代码路径）；Dirichlet 壁 +2.103e-2/步。
- thermal3d 的 `stream_thermal_3d`/六面复制同病（全周期 + 整面覆写）——
  3D 从未量过壁通量，"无此缺陷"应更正为"未测量"。
- **修复（thermal_fix.diff，包内 `apply_temperature_boundaries` v3b）**：
  1) 上下绝热 = **解缠反射**：`g[3,0,:] ← g_pre[4,−1,:]`、`g[4,−1,:] ←
     g_pre[3,0,:]`（从对侧取回本壁出射分布反射回本壁行，签名不变）；
  2) 左右等温列整列冻结 `w_i·T_wall` + **指向群体非平衡修正零和转移**
     `g[1,:,0] += neq_left; g[0,:,0] −= neq_left`（右壁镜像），
     `neq_left = g_pre[1,:,1] − w1·T1`（首流体列携带非平衡）。
  自洽性：交付值 = w1·T_w + neq·(1−1/τ_T)，稳态 neq = w1·τ_T·s 恒等还原
  体稳态形式 ⇒ 有效等温面精确落壁节点（1D 探针 gap1/s = 0.99992）；
  零和转移钉住壁列 ΣT，T(0) 读数与 [0,1] 界不受污染。
- **预算判决**：绝热净泄漏 **0.0/步（逐位）**；Dirichlet 净 −1.139e-4/步
  （其 −9.52e-5 为 W5 float32 权重和 Σw=1+3e-8 的构造性簿记——碰撞注入
  +S·(s_w−1)/τ_T 与 BC 冻结移除相消、不动点净漂 <3e-7/步、intT 偏置
  +0.028%，**非物理泄漏**；n64 探针实测 fp32/fp64 双证）。

### 缺陷 2：壁节点垃圾宏观 u 污染温度碰撞（修复中追加发现）

- 现象：v3b 六案 ra1e3 全 N 稳态永不触发（nu_left 线性膨胀，n128 达 1.894）、
  nu_sym 随 N 恶化（0.169→0.79）；而 fp64 预算能量守恒无恙。
- 根因（wallu_probe 直测）：半程反弹固壁节点的宏观 u 是反弹分布的垃圾值
  （热壁列 mean u_x = −7.96e-5 系统值、上下壁行 −3.6e-5/+1.3e-4，R 破缺 4×），
  却进入温度碰撞平衡态 (1+3cu)：热壁交付被 3w1·T_w·u_x/τ_T 污染
  （T_cold=0 侧天然免疫 → 左右不对称），造成 +0.16 Nu 加性读数偏置与
  Ra 无关净能量抽取（v3b 预算拆分出真物理汇 −1.87e-5/步）⇒ 内区恒速慢冷、
  grad1 读数线性膨胀、步率 ∝N。
- **修复（driver 层，v6）**：`driver.advance` 在 `temperature_collision` 前
  把壁环 u 置零（`mask_wall_u=True`，方案选择、全文披露；库的
  `simulate_natural_convection` 未动）。修复后 grad1 读数与真实链路通量
  **逐位吻合**、sym 0.021→0.0019（11×）、真物理汇归零、300k 步读数钉死。

## 实现（真实模拟，无外推，extrap: none）

- 格式：D2Q9 BGK（`solver.collide_bgk`/`stream` 同族入口 `collide_bgk_force`）
  + D2Q5 温度（`thermal.temperature_collision/temperature_stream`），浮力
  Guo 格式（`thermal.buoyancy_force`，t_ref=T_cold），速度壁 pre-streaming
  半程反弹（`thermal.pre_streaming_bounce_back`）。**run.py 零本地物理**
  （`grep -nE "def (collide|stream|equilibrium|bounce|zou_he|far_field)" run.py`
  零命中），driver.py 只做编排/参数换算/观测提取。
- 参数：τ=0.6、Pr=0.71、ν=(τ−1/2)/3、α=ν/Pr、τ_T=3α+1/2、H=nx−1（等温壁
  在节点上，与 thermal3d length=nx−1 同构）、gβ=Ra·ν·α/H³、ΔT=1、初场
  线性导热态 T(x)=1−x/H、ρ=1、u=0。
- 稳态判据：Nu 漂移 <1e-4/1000 步（nu/nu_left/nu_right 三者 max）连续 3 窗
  + 触发许可下限 floor = 1.15·t_stop(N=64)·(H/63)²（**修订披露**：v6 首轮
  n256 两案在初始发展段假触发——弱流期 Nu 几乎不动使 drift 提前满足，
  验证段如设计暴露；门限本身不变）+ 验证段 max(10000, 10%·t_stop)。
- 观测量（稳态终场直接提取，无拟合/外推/修正）：Nu = 0.5(Nu_hot+Nu_cold)
  整壁 grad1 含角点（库 `nusselt_number` mode='grad1'）；u_max/v_max 中线
  线性插值峰值，u_nd = u_lu·H/α（DVD 扩散尺度）；自检 |Nu_hot−Nu_cold|、
  T∈[0,1]、Ma。
- **精度档（披露）**：基准 fp32，**唯 ra1e3_n256 用 fp64**——该档
  gβ=9.44e-8 使浮力步进 ~0.5 ULP(fp32)，fp32 力量化把 u_max 抬高 +10.5%/
  v_max +21.9%（fp32/fp64 同序列 A/B 实锤，probe_256.log；fp64 复现 DVD
  到 1%）。fp32 失败值原样归档 out_v6_premature/。n128 fp32 有效
  （力步进 ~2.8 ULP，fp64 A/B 差 ≤0.2%）。

## 结果

| 案 | dtype | 步数(t_stop) | Nu (err%) | u_max (err%) | v_max (err%) | 判 |
|----|-------|--------------|-----------|--------------|--------------|----|
| ra1e3_n64 | fp32 | 37k (27k) | 1.09899 (−1.70) | 3.5024 (−4.02) | 3.5364 (−4.34) | ✗ u,v |
| ra1e3_n128 | fp32 | 142k (129k) | 1.10754 (−0.94) | 3.5801 (−1.89) | 3.6243 (−1.97) | ✓ |
| ra1e3_n256 | fp64 | 563k (512k) | 1.10873 (−0.83) | 3.6160 (−0.91) | 3.6555 (−1.12) | ✓ |
| ra1e4_n64 | fp32 | 32k (22k) | 2.10977 (−5.94) | 15.8561 (−1.99) | 18.9611 (−3.34) | ✗ Nu,v |
| ra1e4_n128 | fp32 | 116k (105k) | 2.17759 (−2.92) | 16.0124 (−1.02) | 19.2775 (−1.73) | ✓ |
| ra1e4_n256 | fp32 | 459k (417k) | 2.21132 (−1.41) | 16.0594 (−0.73) | 19.4428 (−0.89) | ✓ |

- **单调收敛**：六量（2 Ra × Nu/u/v）|误差| 64→128→256 **全部严格下降**
  （ra1e3 Nu 1.70→0.94→0.83、u 4.02→1.89→0.91、v 4.34→1.97→1.12；
  ra1e4 Nu 5.94→2.92→1.41、u 1.99→1.02→0.73、v 3.34→1.73→0.89）。
- **判决 = PARTIAL**：N=128/256 四案全过门 + 单调收敛（满足 ≥2 档铁律）；
  N=64 两案粗网格离散误差超 3%（6 个子门中 5 个，最差 Nu −5.94%）——
  如实报告，无修饰。形态自检：u_max_y 0.810/0.827 vs 0.813/0.823 ✓、
  v_max_x 0.181/0.169 vs 0.178/0.119 档内、nu_wall_max/min 包络正确
  （如 ra1e3_n256 1.532/0.638 vs 1.505/0.692）。
- 已知残余（披露）：|Nu_hot−Nu_cold| ∝H²（0.0019→0.0078→0.0314，两 Ra 同值、
  fp64 不变）= v3b neq 携带近似的二阶 u 省略在读数层；n64 链路通量实测
  左右平衡 1.4e-6/行（物理通量对称），平均 Nu 不受影响。

## 运行

```bash
cd /nfs/wangxi/runs/bm_widen_20260920/thermal_cavity
export CUDA_VISIBLE_DEVICES=5
export PYTHONPATH=src_patched3:.
PY=/nfs/wangxi/venvs/tensorlbm/bin/python

# 六案（含触发下限；ra1e3_n256 需 fp64）
$PY run.py --cases all                                   # fp32 五案
$PY run.py --cases ra1e3_n256 --dtype float64            # fp64 一案
$PY summarize.py                                         # → result.json

# 回归（58/58；注意 worktree pyproject pythonpath 劫持，须 tests_staged + 关插件）
cd tests_staged && $PY -m pytest -p no:pythonpath test_thermal.py test_thermal_common.py test_thermal3d.py

# fp64 守恒预算（--umask=v6 驱动同款；换 PYTHONPATH 可跑 port/v2/old 对照）
$PY budget.py --rule v3 --umask --out budget_v6.json
```

## 交付物（本目录）

- `run.py`（基准入口）/`driver.py`（胶水驱动）/`budget.py`（fp64 预算）/
  `summarize.py`（汇总）+ 诊断脚本（dump_1e3、wallu_probe、v5_test、v5_long、
  probe_drain、probe_256、probe_nx、sym_probe*、corner_probe、variants_test 等）
- `thermal_fix.diff`（库修复：orig → src_patched3/tensorlbm/thermal.py，
  sha12 f5bfd676625b；旧 port 版存证 thermal_fix_port_v1.diff.stale）
- `src_patched3/`（patched 包）；`out_v6/`+`logs_v6/`（六案终值）；
  `out_v6_premature/`+`logs_v6_premature/`（假触发首轮 + fp32 n256 失败值，
  存证不删）；`budget_*.json`（old/port/v3/v6/ra1e3 五方预算）
- `NOTES.md`（完整时间线、预注册、诊断链、全部披露项）；`result.json`

## 参考

- G. de Vahl Davis, "Natural convection of air in a square cavity: a bench
  mark numerical solution", IJNMF 3:249-264, 1983.
- Guo, Z., Shi, B. & Zheng, C. (2002). A coupled lattice BGK model for the
  Boussinesq equations. IJNMF 39, 325-342.
- 控制器 T-A 诊断（绝热泄漏 −2.2e-2/步）与参考值独立交叉核对。

## 判定

- 真实模拟（无外推、无人工修正）：是（BC/方案选择全部披露，见 NOTES.md）。
- 误差 ≤3%：**N=128/256 四案全过**（最大 2.92%）；N=64 两案 5/6 子门超
  （最差 Nu −5.94%，粗网格离散误差、单调收敛消失中）。
- ≥2 档网格单调收敛：是（六量全部 64→128→256 严格单调）。
- 库缺陷修复验收：绝热泄漏 −2.15e-2/步 → **0.0 逐位**；真物理汇归零。
- `result.json` verdict = **PARTIAL**（N=64 粗档如实判 FAIL，未凑数）。
