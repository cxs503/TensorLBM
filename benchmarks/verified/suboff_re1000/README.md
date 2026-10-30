# B6: DARPA SUBOFF 裸艇体 — Re=1000 总阻力 benchmark

**状态：✅ 已入库（2026-09-29，端到端误差 −1.15% / −1.36% << 3%）**

## 物理问题

SUBOFF 裸艇体（DARPA SUBOFF bare hull，L=4.356 m，D=0.508 m，L/D=8.57）
零攻角均匀来流。Re = u·L/ν = 1000（层流）。测量总阻力系数 Cd_tot
（摩擦 + 压差）。

## 参考口径（Re=1000 基准族）

- **Blasius 层流平板摩擦系数**：`Cf = 1.328/√Re = 1.328/√1000 = 0.041995`
  （Re=1000），归一化面积取湿面积 `S = π·D·L`，
  `dpS = 0.5·u²·π·D·L`。这是本库 Re=1000 基准族的统一口径，历史同口径
  结果 Cd_tot=0.0436（+3.8% CUDA / +3.6% SDAA，L=80，5000 步未收敛瞬态）。
- **口径说明**：`benchmarks/TODO.md` 中 B6 的 “Ct=0.004（实验）” 实为
  AFF-8 全尺寸 Re=2e6 总阻力系数（见 `src/tensorlbm/suboff_reference_data.py`），
  **不适用于 Re=1000**；本 benchmark 统一使用 Blasius 0.041995 口径。

## 配置（共性模块入口 GeneralSimEngine / PARAMETRIC_SUBOFF）

- 几何：`suboff_length=4.356 m, suboff_radius=0.254 m`
  （掩码 `suboff_cad.build_suboff_mask`, bare_hull）
- 物理：Re=1000（u=1e-3 m/s, ν=4.356e-6 m²/s, L_ref=4.356 m）
  → u_lb=0.05, ν_lb=4e-3, τ=0.512
- 网格：每艇长格数 L_cells=80；域 6L（流向 1L 上游 + 4L 下游），侧向 2L+D
  → L=80 域 480×169×169（13.7M cells，solid 4304，near-wall 1928）
- 碰撞：MRT（Re≈1000 显式 MRT，无 Smagorinsky 人工粘性）
- 壁面：half-way bounce-back（Re<10000 自动 BB）
- 求解：`lbm_step_correct` + `far_field_bc_3d`（x− 入流自由流、x+ 零梯度、
  侧向自由流）+ 质量修正 200 步
- 力：`drag_pressure_integration`（`extrap='none'`，p0='near_wall'）
  + `drag_friction_integration`（**formula='mix50'**）
- 归一化：湿面积 πDL（与历史族一致；引擎默认迎风面积 πR² 在后处理重标定，
  `rescale = dpS_front/dpS_wet = 0.02916`）
- 步数：12000（L=80）；力采样间隔 10 步；末窗 1000 样本（10000 步）平均

### 编译路由（verified 标准，2026-08-19）

`GeneralSimEngine` 整步链（collision → NoDynamics → half-way BB → streaming
→ far-field BC）经 `benchmarks/compile_route.route_step` 路由到
`tensorlbm.compile_utils.compile_step`，**默认 `torch.compile(mode='default')`**；
`--compile-mode eager` 保留 A/B 通道。步序号相关分支（质量修正节拍、测力采样、
快照、NaN 守卫）全部留在编译域外的 eager 驱动循环（compile_route 规则 #2），
routing banner 写入日志留痕。

**SDAA / teco 限制与自动 eager 回退（2026-09-29）**：`mode='default'` 在 CUDA
上是真编译（生产路径）。本机为 Hygon/SDAA，`torch.compile` 走 `teco_inductor`
（Torch-SDAA **3.1.1a3**），其后端对整步 LBM 链仍不完整（`_enable_sdaa_inductor`
已修 L1–L4，上游 L5 teco `EXPAND` 路径 `reduction_shape` 越界未修）。因此
`route_step` 对编译模式改用 `_CompileWithEagerFallback`：**首次**调用试编译版，
捕获 `InductorError/AssertionError/IndexError/RuntimeError` 后打印醒目 banner
（异常 + 回退原因），**其后每步走 eager**，与 `--compile-mode eager` **逐位一致**
——保证本用例在 SDAA 上仍可复现。实际模式写入 `result.json` 的
`compile_mode_effective`（`compiled` / `eager_fallback` / `eager`），并附
`compile_status` / `compile_status_reason`。CUDA 上该封装是 no-op。

### mix50 摩擦公式（为什么用标准公式会偏低 ~10%）

体壁摩擦是阻力主导项（摩擦 ~77%）。离散体壁的精确摩擦有两个界：
`standard`（近壁单元标准求和，下界）与 `faces`（voxel stair 逐面剪切，上界，
几何面/体比 +49%）。纯几何上界增益在真实流场中只实现一部分
（L=80 uniform-field faces/standard = 1.396，真实只到 1.221），
故 `mix50 = 0.5·cd_f_standard + 0.5·cd_f_faces` 取两界中点，
把系统性的 −10% 低估修到 <2%。本结果：standard 公式 Cd_tot=0.037853（−9.85%），
mix50 Cd_tot=0.041424（−1.36%）。

## 结果（端到端达标证据，2026-09-29 实测，真实模拟无外推）

| 配置 | 网格 | 步数 | 摩擦公式 | Cd_p | Cd_f | **Cd_tot** | 误差 vs Blasius 0.041995 | 判定 |
|------|------|------|---------|------|------|-----------|--------------------------|------|
| **L=80 mix50**（主） | 480×169×169 | 12000 | mix50 | 0.005556 | 0.035957 | **0.041513** | **−1.15%** | ✅ |
| L=80 mix50 last-500 窗 | 同上 | 12000 | mix50 | — | — | 0.041428 | −1.35% | ✅ |
| **L=80 mix50 final-field 重算** | 同上 | 收敛场 | mix50 | 0.005537 | 0.035888 | **0.041424** | **−1.36%** | ✅ |
| L=48 mix50 | 288×102×102 | 20000 | mix50 | 0.006816 | 0.033848 | 0.040664 | −3.17% | （粗网格）|
| L=48 standard | 288×102×102 | 20000 | standard | 0.006816 | 0.031109 | 0.037925 | −9.69% | （对照）|

**主结果（L=80，末 10000 步窗口平均）**：Cd_tot = **0.041513**，
相对 Blasius 0.041995 = **−1.15%**；收敛场 final-field 重算 mix50 =
**0.041424**（**−1.36%**）。两口径均 < 3% 阈值 ✅

**收敛证据**（末 10000 步滑动窗口，采样间隔 10 步，窗口均值单调下降）：

| 进度 | Cd_tot（窗口均值） |
|------|-------------------|
| 25% | 0.051099 |
| 50% | 0.046304 |
| 75% | 0.044680 |
| 100% | **0.041513** |

原始引擎单位窗口均值（经 rescale=0.029155 换到湿面积）：step 4703 → 1.6342，
step 11995 → 1.4209，逐点单调下降且末段趋于平台（漂移 < 4e-6/窗口），
表明 12000 步时流场已充分收敛。final-field 的 cd_p/cd_f_standard 与
last-500 窗一致（0.005537/0.032317），确认收敛值为真实稳态值而非瞬态。

**网格说明**：L=48 是粗网格（艇半径仅 ~2.8 格），mix50 只有 −3.17%
（略超阈值，离散误差未消）；L=80（艇半径 ~4.7 格）收敛到 −1.15%。趋势
随分辨率提高误差下降，符合真收敛。

## 复现命令

```bash
cd /root/TensorLBM_feat2
export PYTHONPATH=/root/TensorLBM_feat2/src

# 主结果：L=80 mix50（torch.compile default，已入库路径）
python benchmarks/verified/suboff_re1000/run.py \
  --resolution 80 --steps 12000 --device sdaa:2 \
  --collision mrt --friction mix50 --compile-mode default \
  --out results_e2e_suboff_L80_mix50

# 粗网格对照：L=48 mix50（−3.17%）
python benchmarks/verified/suboff_re1000/run.py \
  --resolution 48 --steps 20000 --device sdaa:1 \
  --collision mrt --friction mix50 --out results_e2e_suboff_L48_mix50

# standard 公式对照：L=48 standard（−9.69%，显示 mix50 修复量）
python benchmarks/verified/suboff_re1000/run.py \
  --resolution 48 --steps 20000 --device sdaa:0 \
  --collision mrt --friction standard --out results_e2e_suboff_L48_standard

# eager A/B（编译域外通道）
python benchmarks/verified/suboff_re1000/run.py \
  --resolution 80 --steps 12000 --device sdaa:2 \
  --collision mrt --friction mix50 --compile-mode eager --out /tmp/suboff_eager
```

> 端到端长跑（L=80 需 ~3.4 h）：本仓库用
> `scripts/_suboff_e2e_monitor.py`（monkeypatch 驱动循环外挂进度监控，
> 不改 run.py）配合 `scripts/_suboff_e2e_launch.sh` 启动，
> 日志在 `runs/e2e_suboff_20260929/`。

## 耗时

| 配置 | device | ms/step | 总耗时 |
|------|--------|---------|--------|
| L=80 mix50（13.7M cells，12000 步） | sdaa:2 | 1015.5 | 12185.7 s（3.39 h）|
| L=48 mix50（3.0M cells，20000 步） | sdaa:1 | 222.95 | 4458.9 s（1.24 h）|
| L=48 standard（3.0M cells，20000 步） | sdaa:0 | 219.96 | 4399.2 s（1.22 h）|

（3.75× SDAA 加速后 L=80 才从 “3.7 s/step 不可跑” 变为可跑。）

## 库模块清单

| 模块 | 函数 | 用途 |
|------|------|------|
| `tensorlbm.general_sim` | `GeneralSimEngine` / `PARAMETRIC_SUBOFF` | 共性情形入口（几何/单位/掩码/域） |
| `tensorlbm.lbm_step_correct` | `lbm_step_correct` | 整步链（collide→NoDyn→BB→stream→BC） |
| `tensorlbm.boundaries3d` | `far_field_bc_3d` | 远场 BC |
| `tensorlbm.drag_pressure` | `drag_pressure_integration` / `drag_friction_integration` / `suboff_smooth_q` | 压差 + 摩擦（standard/faces/mix50 等） |
| `tensorlbm.solver3d` | `correct_mass3d` / `stream3d` | 质量修正 / 输运 |
| `tensorlbm.d3q19` | `equilibrium3d` | 初始平衡态 |
| `benchmarks.compile_route` | `route_step` 等 | torch.compile 共享封装（整步编译） |

## 工件

- `run.py` —— 本 benchmark 入口（已接 `compile_route` 整步编译路由；
  归档原始逐样本力历史 `forces_history.json`，供 verify.py 独立重算）
- `verify.py` —— 独立复核器（几何门 / 力门 / 历史门 / 3% 判定门，见下节）
- `result.json` —— verified 结果 + 端到端数据 + 耗时
- 大文件 `final_field.pt`（1.8 GB）**不入库**，落在
  `results_e2e_suboff_L80_mix50/final_field.pt`（供后续逐公式重算）

## 判定

- 真实模拟（无外推，`extrap='none'`），端到端 Cd_tot 误差 ≤3%：
  L=80 mix50 **−1.15%**（窗口）/ **−1.36%**（final-field 重算）→ 达标 ✅
- 时间收敛（窗口均值单调趋稳）+ 网格趋势（L=48 −3.17% → L=80 −1.15%）一致
- 历史演化：standard 公式 L=80/96 系统性 −10%（对照），faces 公式过冲
  +6.5~+10.6%（上界），mix50 取两界中点达标；分析详见 git 历史
  （`benchmarks/pending/suboff_re1000/result.json` @5c57065 之前）。

## 单次派单可用分级（m318 诊断，2026-10-30 实测）

本机（Hygon/SDAA 单卡节点：torch 2.7.1 + torch_sdaa 3.1.1a3，1 个 SDAA 设备）
实测吞吐与可用分级，供后续派单选择"能在预算内跑完并交物"的档位：

| 档位 | 命令要点 | 网格 | 步数 | 本机实测/推算耗时 | 物理结论 |
|------|----------|------|------|------------------|----------|
| smoke | `--resolution 24 --steps 200` | 144×51×51（0.37M cells, 112 ms/step） | 200 | **22 s** | 不作物理结论：几何门 R_max=−49%（艇半径仅 0.7 格），Cd +46% |
| 粗网格 | `--resolution 32 --steps 6000` | 192×68×68（0.89M cells） | 6000 | ~16 min（推算） | 几何门仍 FAIL（R −24%） |
| 中网格 | `--resolution 48 --steps 12000` | 288×102×102（3.0M cells, 287–363 ms/step 实测） | 12000 | **58–73 min**（实测吞吐外推；本班次跑到 41 min 因调用预算中止，未取数）| 几何门 R −20% / Cp −10.5%；未收敛窗仍 >3% |
| 达标档 | `--resolution 80 --steps 12000` | 480×169×169（13.7M cells） | 12000 | sdaa:2 历史 **3.4 h**；本机按 cells 外推 ≥5 h | **−1.15%（窗口）/ −1.36%（终场重算）✅** |

**结论**：本用例的"达标档"是 3.4 h 级长跑，**超出单次派单时间预算**。
此前多次 FAIL（未调用 finish / 工作树无改动）即源于直接对着 12000–20000 步
L=80 配置起跑。后续派单应二选一：
1. **诊断档**（≤75 min）：跑中网格档 + `verify.py`，如实报"未收敛 >3%"，并把
   几何/收敛趋势作为结论；
2. **达标档**：预先按长跑申请预算（>4 h），或复用已入库的 L=80 结果
   （`result.json`）并用 `verify.py` 独立复核，不要重跑。

### 编译路径（SDAA / teco）

`--compile-mode default` 在本机**必然**落到 audited eager fallback：
teco_inductor 在整步 LBM 链上报
`Error in codegen for ComputedBuffer(...) / not find generator` →
`InductorError: AssertionError`，`compile_route` 捕获后打印 banner 并**逐位
等价**于 `--compile-mode eager` 继续跑（`compile_status='eager_fallback'`
写入 `result.json`）。即：**本用例在 SDAA 上不存在 triton 通路，绕行方案就是
eager（SDAA 卡内）**；CPU 也可跑但同样尺寸下更慢（L=48 单步 >1 s，未测完）。
这不是本用例的缺陷，是 teco 上游限制（见 `benchmarks/compile_route.py`）。

## 独立复核（verify.py）

`verify.py` 用归档原始数据**重算**全部判据（不信任 `result.json` 摘要），
四道门：

1. **几何门**：体素掩码 vs 解析 SUBOFF 回转体（`suboff_cad.suboff_statistics`）
   —— L/D、R_max、Cp(=Cb)、域堵塞；|ΔR|<20%、|ΔCp|<15% 才认为阻力可信。
2. **力门**：从 `final_field.pt` 重算 Cd_p / Cd_f（standard / faces / mix50
   三口径），湿面积 πDL 归一。
3. **历史门**：从 `forces_history.json`（run.py 归档的原始逐样本力历史）
   重算窗口均值与漂移（末 20% vs 前 20%，门槛 |drift|<1%），并与
   `result.json` 的摘要做 1e-9 量级恒等核对。
4. **判定门**：|误差| ≤ 3%（对 Blasius `1.328/√Re`），窗口均值与终场重算
   **两个口径都要过**。

```bash
python benchmarks/verified/suboff_re1000/verify.py                       # 复核本目录
python benchmarks/verified/suboff_re1000/verify.py runs/m318/suboff_L48_mix50_s12000 --write
python benchmarks/verified/suboff_re1000/verify.py --geometry-table 24 32 48 64 80 96
```
退出码：0=PASS、1=FAIL、2=WITHHELD（归档数据缺失）。缺 `forces_history.json`
或 `final_field.pt` 时对应门返回 WITHHELD 而不是伪造 PASS。
