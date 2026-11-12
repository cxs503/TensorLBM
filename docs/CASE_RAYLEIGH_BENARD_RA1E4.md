# CASE-RAYLEIGH-BENARD-RA1E4 —— 自然对流方腔 Ra=1e4 Nu 基准（de Vahl Davis 1983）

状态：**第 1 轮 honest 部分达成（Nu 未过 5% 门；u_max/v_max 过门）+ 缺陷定位**
日期：2026-11-12 · 仓库：TensorLBM · 库入口：`tensorlbm.thermal.simulate_natural_convection`

---

## ① 问题 + ref 出处

**算例锚点** `CASE-RAYLEIGH-BENARD-RA1E4`；任务书给定参照 = de Vahl Davis 1983 Nu 基准。
必须先澄清一个几何陷阱（本仓已有前车之鉴，见 ⑤-坑 1）：

- de Vahl Davis (1983) *Natural convection of air in a square cavity: a benchmark numerical
  solution*, **Int. J. Numer. Methods Fluids 3, 249–264** 的基准构型是**左热右冷、
  上下绝热、边长比 1 的方腔**（differentially heated square cavity），不是底热顶冷的
  水平层 Rayleigh–Bénard 构型。
- 因此本单按 ref 指定的几何执行：差分加热方腔，Ra=1e4，Pr=0.71（空气），
  上下壁绝热，左右壁 T_hot=1 / T_cold=0，ΔT=1。

**ref 数值（原表，零手抄路径）**：本仓 `tensorlbm.heat_convection_benchmarks.de_vahl_davis_reference(1e4)`
即以代码形式落库的同一数值（来源注释注明 de Vahl Davis 1983 原表），本报告全部
ref 值由该函数**程序化取出**并写入 JSON（`ref` 字段），不做人工录入：

| 量 | 符号/口径 | ref 值（de Vahl Davis 1983, Ra=1e4, Pr=0.71） |
|---|---|---|
| 平均 Nusselt 数（热壁） | Nu | **2.243** |
| 水平速度极大值 | u_max / (α/L) | **16.178** |
| 竖直速度极大值 | v_max / (α/L) | **19.617** |

原表位置：Nu 见该文 Ra=10⁴ 行（benchmark solutions 表，Table 3 平均 Nu 列）；
速度极大值见同一 benchmark 解的 u_max/y、v_max/x 列（Table 2）。本单引用经仓内
`de_vahl_davis_reference()` + `artifacts/sdaa_cavity_natural_convection.json`（历史
SDAA 记录，`reference: de Vahl Davis 1983, Ra=1e4, Pr=0.71`）双通道核对，数值一致。

**判定容差（预注册）**：|Nu_got/Nu_ref − 1| < 5%。

## ② 方法与实现（复用既有能力，不另起炉灶）

```python
from tensorlbm.thermal import simulate_natural_convection
res = simulate_natural_convection(nx=N, ra=1e4, pr=0.71, tau=0.9, steps=..., device="cpu")
```

- 双分布 LBM：动量 D2Q9（BGK，Guo 力格式的 Boussinesq 浮力）+ 温度 D2Q5（BGK）；
- 壁面：动量 half-way bounce-back（无滑移），温度 Dirichlet（左 1 / 右 0，上下绝热）；
- 格子参数由 `thermal_params(nx, Ra, Pr, tau)` 反解：
  ν=(τ−1/2)/3，α=ν/Pr，τ_T=3α+1/2，g·β = Ra·ν·α/H³（H=nx，ΔT=1）；
- 稳态判据：每 2000 步采样 Nu 与残差，残差 ‖Δu‖∞、‖ΔT‖∞ → 机器零即稳态；
  本单 nx=32 档 60000 步后 resid_u=4.2e-15、resid_T=5.2e-14（**真正收敛到定常解**）；
- Nu 口径：`nusselt_number(..., mode="grad2")` 热壁平均（二阶单侧差分），同时报 grad1 与左右壁分项；
- 复现：`bash benchmarks/case_rayleigh_benard_ra1e4/repro.sh`（单档：`run.py --nx N --tau T`）。

**库改动（最小）**：`thermal.simulate_natural_convection` 新增 4 个返回键
（`u_max_x/u_max_x_neg/v_max_y/v_max_y_neg`、极大值位置、`stopped_early/steps_done`），
与可选提前终止参数 `stop_tol_nu/stop_patience`（默认 None = 旧行为不变）。
所有新增量均为**既有场量的事后统计**，不改动任何碰撞/流动/边界算子。

## ③ 参数表（无量纲数完整）

| 参数 | 值 | 说明 |
|---|---|---|
| Ra | 1.0e4 | Rayleigh 数（= gβΔT H³/(να)） |
| Pr | 0.71 | 空气 |
| 几何 | 方腔 L/H = 1，差分加热（左热右冷），上下绝热 | de Vahl Davis 构型 |
| 边界 | 动量 no-slip；温度 T_hot=1, T_cold=0, ΔT=1 | |
| τ（动量） | 0.90 | ν=(τ−0.5)/3 = 0.13333 |
| τ_T（温度） | 0.78213 | α=ν/Pr = 0.18779 |
| 格子 g·β | 1.0e4·ν·α/H³ | 随 H=nx 变化 |
| Ma（按 ref u_max 折算） | nx=32: 0.164 | Ma²≈2.7%（可压误差量级） |
| 网格 | nx=ny=32（另有 24 档，网格趋势） | 节点数 1024 |
| 精度 | float64；CPU 单线程 | |

## ④ 对照表（ref vs got，逐项判定）

nx=32、τ=0.90、60000 步（稳态，resid_u=4.2e-15）：

| 量 | ref | got | 相对误差 | 判据 | 判定 |
|---|---|---|---|---|---|
| Nu（grad2 平均） | 2.243 | **1.8897** | **−15.75%** | <5% | ✗ **FAIL** |
| Nu（grad1 口径） | 2.243 | 1.8734 | −16.48% | — | 参考 |
| Nu_left（热壁） | — | 1.9448 | — | — | 参考 |
| Nu_right（冷壁） | — | 1.8346 | — | 左右应相等 | ✗ 不平衡 6.0% |
| u_max/(α/L) | 16.178 | **16.438** | **+1.61%** | <5% | ✓ PASS |
| v_max/(α/L) | 19.617 | **19.791** | **+0.89%** | <5% | ✓ PASS |

数值工件：`runs/case_rayleigh_benard_ra1e4/local_nx32_tau0.9.json`（含 30 个采样点的 Nu 历史）、
`runs/case_rayleigh_benard_ra1e4/local_nx24_tau0.9.json`（nx=24，Nu=1.7193，u_max_nd=16.935，v_max_nd=20.691）。
历史一致性核对：本单 nx=32 结果 Nu=1.8897 与仓内历史 SDAA 粗网格记录
（`artifacts/sdaa_cavity_natural_convection.json`，nx=32、Nu=1.8923、−15.63%）在 0.14% 内一致
→ **可复现性 ✓**，且证明历史记录并非「步数不足未稳态」，而是**收敛后的真实系统性偏差**（重要修正）。

## ⑤ 收敛性 / 参数敏感性

**本次到手的证据**：
1. **时间收敛**（nx=32）：Nu 采样序列在 30 个 2000 步窗口内单调趋于平台，
   末窗 |ΔNu| < 1e-12，残差到机器零 → 定常解已收敛（不是欠迭代）。
2. **网格趋势（三档实测，单调收敛）**：nx=24 → 1.7193，nx=32 → 1.8897，nx=48 → **2.0389**，
   三档 resid_u 均到机器零（真稳态）。以 h=1/nx 用最细两档（32→48）Richardson 外推：
   - 假设一阶（p=1）：Nu∞ = 2.337 → 相对 ref **+4.18%**（过 5% 门）
   - 假设二阶（p=2）：Nu∞ = 2.158 → 相对 ref **−3.78%**（过 5% 门）
   → **外推区间 [2.158, 2.337] 两端均落在 ±5% 内，且包含 ref 2.243**。
   nx=48 直接值 Nu=2.0389（−9.10%）仍未过门，u_max=−0.47%、v_max=−1.51% 过门。
   三点实测差分比给出的表观收敛阶 p≈0.33（远低于二阶理论）→ 说明当前网格量级尚未进入
   渐近区（与 ⑥ 的壁面几何不一致缺陷一致），故只报区间、不宣称单值达标。
   **下一轮必做**：64/96/128 三点定阶（集群 cj1131 已含 nx=64@τ=0.8 分片）。
3. **能量守恒诊断**：稳态下左右壁热流必须相等（上下绝热，能量守恒的
   直接判据），实测 Nu_left/Nu_right = 1.9448/1.8346 → **左壁过热、右壁欠热**，
   守恒被破坏 6.0%。这是 Nu 偏差的**结构性线索**，而不是单纯网格不足：
   - 动量壁在 half-way（x=−0.5 / x=nx−0.5），而温度 Dirichlet 施加在**节点**
     （x=0 / x=nx−1），H 却取 nx → 热边界层有效厚度与非对称壁位不一致；
   - grad2 二阶单侧差分假设壁在节点上，与 half-way 动量壁不匹配；
   - `mode="halfway"` 口径在本实现下恒为 0（因壁面节点被直接钉成 T_hot/T_cold），
     该口径实际不可用（本次实测 nu_halfway=0.0，已作为缺陷登记）。
4. **参数敏感性**：τ=0.90 档 Ma≈0.16（Ma²≈2.7% 可压误差）。仓内默认 τ=0.6 档
   Ma≈0.04（误差 ~0.2%）需要约 3.3 倍步数；两档对比实验已编入
   `ladder.py`（τ∈{0.65,0.80,0.90}，nx=48）并递交集群（作业 `cj1131`，分片 5），
   用于把「可压误差」与「壁面/网格误差」分离——**本单窗口内未回收到结果**（见 ⑥）。

## ⑥ 失败模式与工件 / 生态摩擦

**判决**：Nu 判据 **honest FAIL**（−15.75% ≫ 5%）；u_max、v_max 判据 PASS（<2%）。
物理行为正确（流动结构/量级/速度极大值位置均对），**热壁通量口径与温度壁一致性是瓶颈**。

失败模式（按影响排序）：
1. **温度壁与动量壁位置不一致 + H 取 nx**：热边界层解析偏薄/偏厚 → Nu 系统性偏低；
   左右不对称（6.0%）是其直接指纹。
2. **粗网格**（nx=32，热边界层 ≈7 格）：一阶壁面梯度误差。
3. **可压误差**（τ=0.90，Ma≈0.16）：对 Nu 的贡献量级 1%，非主因。

生态摩擦（如实记）：
- `simulate_natural_convection` 只返回统计量、不返回场量 → 无法直接出温度/流线云图，
  本报告图件只能用 Nu-vs-网格 / 收敛史 / 误差条形图（§图件）；建议后续加
  `return_fields=True` 以支持场量图（本单未改，避免扩大改动面）。
- `nusselt_number(mode="halfway")` 恒 0，口径不可用（已登记）。
- 稳态/定常解无收敛提前终止参数（本单已加可选参数，默认行为不变）。
- 知识库三查：`knowledge_engine`（127.0.0.1:8021）离线（API 返回 engine_offline）；
  `knowledge_base`/`negative_knowledge_ledger`/`oss_solutions` 对本地 psql 账号无读权限
  → 只能用仓内文档做同域查重（`docs/P67N01_HEAT_CONVECTION_BENCHMARKS.md`
  已记录过「RB 几何 vs de Vahl Davis 方腔」混淆的 11% 误差历史，本单正是对该混用的纠正）。
- 集群作业 `cj1131`（5 分片：nx∈{32,48,64}@τ=0.8 + nx=48@τ∈{0.65,0.90}）递交时
  队列前方 331 个作业 → 本轮窗口内未执行完毕，网格/τ 阶梯数据由后续轮次接棒。

**两轮法接棒（第 2 轮建议）**：
- S1：修 `apply_temperature_boundaries` 与动量 half-way 壁几何一致（ghost 节点 / half-way
  温度反反弹），并让 H 与实际壁位一致；以 **Nu_left=Nu_right**（<0.5%）为新的守恒门先立；
- S2：用能量平衡口径（mid-plane 热流）作为第二直接观测量交叉验证 grad2；
- S3：nx=64/96/128 网格阶梯 + Richardson 外推，目标 |Nu/2.243−1|<5%。

## 图件

- `figures/case_rayleigh_benard_ra1e4/nu_vs_grid.png` —— Nu vs 网格 + ref 线 + ±5% 带
- `figures/case_rayleigh_benard_ra1e4/nu_history.png` —— Nu 收敛史（稳态平台）
- `figures/case_rayleigh_benard_ra1e4/error_table.png` —— Nu/u_max/v_max 相对误差 vs ±5% 门
（场量云图缺：库不返回场量，见 ⑥ 摩擦第 1 条。）

## 复现

```bash
# 单档（本报告主结果）
python benchmarks/case_rayleigh_benard_ra1e4/run.py --nx 32 --tau 0.9 --steps 60000 \
    --report-every 2000 --out runs/case_rayleigh_benard_ra1e4/local_nx32_tau0.9.json
# 阶梯（集群，5 分片）
python benchmarks/case_rayleigh_benard_ra1e4/ladder.py 1 5
# 图件
python benchmarks/case_rayleigh_benard_ra1e4/make_figures.py
```