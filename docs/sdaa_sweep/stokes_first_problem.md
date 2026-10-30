# SDAA 扫荡：stokes_first_problem（B24 · Stokes 第一问题 / Rayleigh 问题，D2Q9）

| 项 | 值 |
|---|---|
| 案例目录 | `benchmarks/verified/stokes_first_problem` |
| 运行入口 | `benchmarks/verified/stokes_first_problem/run.py`（**未修改**，`git diff benchmarks/` 为空） |
| 独立脚本 | `scripts/sdaa_sweep_stokes_first_problem.py`（import 案例 `run_case`/`compare_grids`，仅把模块级 `DEVICE` 指向 sdaa） |
| 机器产物 | `artifacts/sdaa_stokes_first_problem.json` |
| 总体结论 | **PASS** — 14/14 条判据通过 |
| 墙钟（整轮脚本） | 133.43 s（≪ 20 min 上限） |
| 运行状态 | `OK` |

## 1. 设备证据（禁止 CPU 冒充 sdaa）

```
torch.sdaa.get_device_name(0) = '/dev/tcaicard25'   (device_count=1)
torch                   = 2.7.1
torch_sdaa              = 3.1.1a3+gite3971b9
sdaa smoke probe        = 67.88045501708984  on sdaa:0
device_properties       = _SDAADeviceProperties(name='/dev/tcaicard25', total_memory=15296MiB)
triton installed        = True   TRITON_INTERPRET=None
```

跑前**张量落位断言**（用案例原语 `equilibrium`/`macroscopic` 现场构造并检查设备，不只信设备名）：`f.device=sdaa:0`、`ux.device=sdaa:0`、`is_target_device=True`。（`SPECULAR` 索引张量留在 `cpu`，业务代码内 `.to(f.device)` 迁移，不影响落位。）

## 2. 配置（案例 README 推荐档，未降档）

| 参数 | 值 | 依据 |
|---|---|---|
| 网格 H | [100, 200]（ny=H+2，nx=8 周期） | README「两档网格 H=100/200」收敛判据 |
| 记录时刻 t | [1000, 4000, 9000] 格子步 | README 推荐 |
| τ | 0.65（ν=0.050000） | README：4δ=85<H=100，半无限近似成立 |
| U | 0.05 | README（Ma≈0.0866） |
| dtype | float32 | 案例原设定 |
| compile-mode | `eager` | 见 §5 绕行记录 |
| 判据阈值 | 剖面 max 相对误差 ≤ 3.0% | README「判定标准」原口径，**未放宽** |

## 3. 结果表（脚本算出，零手抄）

| H | t | δ=√(νt) | max_rel_err | l2_rel_err | max_abs_err/U | u(壁)/U | u(顶)/U |
|---|---|---|---|---|---|---|---|
| 100 | 1000 | 7.07 | **0.2590%** | 9.848e-04 | 0.0695% | 0.96019 | 1.79e-06 |
| 100 | 4000 | 14.14 | **0.0743%** | 2.858e-04 | 0.0203% | 0.98003 | 1.86e-06 |
| 100 | 9000 | 21.21 | **0.0523%** | 4.584e-04 | 0.0725% | 0.98663 | 1.64e-03 |
| 200 | 1000 | 7.07 | **0.2579%** | 9.844e-04 | 0.0696% | 0.96019 | 1.86e-06 |
| 200 | 4000 | 14.14 | **0.0741%** | 2.929e-04 | 0.0204% | 0.98004 | 1.71e-06 |
| 200 | 9000 | 21.21 | **0.0493%** | 2.313e-04 | 0.0142% | 0.98665 | 1.71e-06 |

两档网格逐时刻剖面差（%·U，coarse 插值到 fine 的共同 y 区间）：

| t | max profile diff | mean profile diff |
|---|---|---|
| 1000 | 0.0004% | 0.00009% |
| 4000 | 0.0013% | 0.00033% |
| 9000 | 0.0793% | 0.00664% |

## 4. 判据逐条 PASS/FAIL

G1/G2 为案例 README **原判据**；D1/D2 为 README「结果」段记录量的**诊断门**（标注 `[诊断]`，阈值由本脚本设定，不计入原判据）。

| # | 判据 | 范围 | 实测值 | 门 | 结果 |
|---|---|---|---|---|---|
| 1 | `G1` 剖面最大相对误差 ≤ 3%（u_ana > 5%·U 区域） | H=100 t=1000 | 0.258974% | `<= 3.0` | **PASS** |
| 2 | `G1` 剖面最大相对误差 ≤ 3%（u_ana > 5%·U 区域） | H=100 t=4000 | 0.074299% | `<= 3.0` | **PASS** |
| 3 | `G1` 剖面最大相对误差 ≤ 3%（u_ana > 5%·U 区域） | H=100 t=9000 | 0.052268% | `<= 3.0` | **PASS** |
| 4 | `G1` 剖面最大相对误差 ≤ 3%（u_ana > 5%·U 区域） | H=200 t=1000 | 0.257935% | `<= 3.0` | **PASS** |
| 5 | `G1` 剖面最大相对误差 ≤ 3%（u_ana > 5%·U 区域） | H=200 t=4000 | 0.074090% | `<= 3.0` | **PASS** |
| 6 | `G1` 剖面最大相对误差 ≤ 3%（u_ana > 5%·U 区域） | H=200 t=9000 | 0.049272% | `<= 3.0` | **PASS** |
| 7 | `G2` 网格收敛：H_hi 的 max_rel ≤ max(H_lo 值, 3%) | t=1000 (H_lo=100 err=0.2590%, H_hi=200 err=0.2579%) | 0.257935% | `<= 3.0000` | **PASS** |
| 8 | `G2` 网格收敛：H_hi 的 max_rel ≤ max(H_lo 值, 3%) | t=4000 (H_lo=100 err=0.0743%, H_hi=200 err=0.0741%) | 0.074090% | `<= 3.0000` | **PASS** |
| 9 | `G2` 网格收敛：H_hi 的 max_rel ≤ max(H_lo 值, 3%) | t=9000 (H_lo=100 err=0.0523%, H_hi=200 err=0.0493%) | 0.049272% | `<= 3.0000` | **PASS** |
| 10 | `G2` 网格收敛总体（compare_grids().converged） | H=100 vs H=200 | True | `True` | **PASS** |
| 11 | `D1` [诊断] 波场有限 finite == True | H=100 | True | `True` | **PASS** |
| 12 | `D1` [诊断] 波场有限 finite == True | H=200 | True | `True` | **PASS** |
| 13 | `D2` [诊断] \|质量漂移\| ≤ 0.1% | H=100 | 0.009784% | `<= 0.1` | **PASS** |
| 14 | `D2` [诊断] \|质量漂移\| ≤ 0.1% | H=200 | 0.007697% | `<= 0.1` | **PASS** |

合计 **14/14 PASS**，`all_pass=True`。

### 与 NVIDIA 侧 README 记录值的对照（如实列出差异，不修改任何判据）

| H | t | README（NVIDIA 侧） | 本次 SDAA | 差异 |
|---|---|---|---|---|
| 100 | 1000 | 0.248% | 0.2590% | +0.0110 pp |
| 100 | 4000 | 0.054% | 0.0743% | +0.0203 pp |
| 100 | 9000 | 0.084% | 0.0523% | -0.0317 pp |
| 200 | 1000 | 0.249% | 0.2579% | +0.0089 pp |
| 200 | 4000 | 0.048% | 0.0741% | +0.0261 pp |
| 200 | 9000 | 0.074% | 0.0493% | -0.0247 pp |

说明：近壁首格 `u(壁)/U` 与 README 记录的 t=1000 值 0.9602 在四位小数上一致；t=9000 的 max_rel 与 README 记录值有 0.032 pp 量级差异，属不同设备 float32 规约顺序对 `max_y`（erfc 尾区边界上的单点）的敏感性，**全部结果仍远低于 3% 门**，不影响判据结论。未做任何调参/修正因子。

## 5. 绕行记录（如实）

- **triton**：案例入口 `run.py` **不依赖 triton**（仅 `torch` / `numpy` / `tensorlbm` / `compile_route`）。`TRITON_INTERPRET=1` 对本案例的可执行路径**无作用**，因此**未设置**（环境中 `TRITON_INTERPRET=None`，triton_installed=True）。不使用即不冒认。
- **compile-mode 绕行**：README 标准口径为 `--compile-mode default`（`torch.compile`）。脚本按该口径**实测探测**了一次，结果如下，随后主判定改用 `eager`：

```
[compile_route] stokes_first_problem[H32]: mode='default' -> torch.compile(mode='default') + auto eager fallback [sdaa-teco_inductor]
[compile_route] !!! torch.compile FAILED on this device -> EAGER FALLBACK
[compile_route]   step    : stokes_first_problem[H32]
[compile_route]   mode    : 'default' (teco/inductor tag: sdaa-teco_inductor)
[compile_route]   reason  : InductorError: AssertionError: self.reduction_shape [1, 9, 272], index 2*i1, self.ncw [9, 272, 1],                                                          graph_ncw [9, 272, 1], graph_reduction_shape [1, 9, 272]
[compile_route]   effect  : this step + every later step run EAGER
[compile_route]             (bitwise-identical to --compile-mode eager)
[compile_route]   detail  : see compile_route.py docstring, 'Compile-time device support & the automatic eager fallback'
```

- 探测结论：`fell_back_to_eager=True`；`requested_compile_mode=default`；探测档 H=32/steps=[50]，其 `max_rel=5.330%` *仅用于触发编译路径*（边界层仅 ~1.6 格，**不是判据**，已标注 `probe_only_not_a_criterion`）。
- 因此主判定采用 `compile-mode=eager`：`compile_route` 的 fallback 语义保证 eager 与回退后轨迹**逐位一致**，且 eager 是 `compile_step` 的 passthrough（业务数值路径不改）。这是**编译路径不可用的绕行**，不是放宽判据——判据与阈值未变，失败模式（teco inductor L5 `reduction_shape`）如实留档于 artifacts JSON。
- 其它加速开关：未启用 AMP/多卡；单卡 `sdaa:0` 串行跑完 2 档网格。

## 6. 复现

```bash
cd /root/tp-agent/work/m50
# 主判定（两档网格，约 2 min 墙钟）
python scripts/sdaa_sweep_stokes_first_problem.py --out artifacts/sdaa_stokes_first_problem.json
# 附带 compile-mode=default 探测（记录 teco inductor 失败原因，约 +3 s）
python scripts/sdaa_sweep_stokes_first_problem.py --out artifacts/sdaa_stokes_first_problem.json --probe-compile-default
```

每档网格墙钟：H=100 → 57.9 s（案例内部计时 57.9 s）；H=200 → 58.44 s（案例内部计时 58.4 s）。

## 7. 结论与局限

1. Stokes 第一问题在 SDAA 上 **14/14 判据全 PASS**：六条 (H,t) 组合的剖面最大相对误差 最大 0.2590%（门 3%），网格收敛成立（H=200 各时刻不劣于 max(H=100, 3%)）。
2. 边界条件实现（移动壁 pre-streaming 半程反弹 + 顶壁自由滑移镜面）在 SDAA 上与 NVIDIA 侧一致可用；近壁首格速度与解析值吻合。
3. 局限：编译路径（teco inductor）在本栈不可用，故本案例的 SDAA 证据为 **eager 数值路径**；若需要 compiled 证据，需等 Torch-SDAA 修复 L5（`reduction_shape` 索引越界）。本案例为 2D 极小网格（nx=8），墙钟由 launch 开销主导，未测吞吐——**不作性能结论**。
