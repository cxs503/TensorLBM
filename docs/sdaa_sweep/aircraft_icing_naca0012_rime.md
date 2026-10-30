# aircraft_icing_naca0012_rime — SDAA 验证扫荡（honest-FAIL：6/11 门 PASS）

- **案例**：`benchmarks/verified/aircraft_icing_naca0012_rime`（NACA 0012 IRT rime 结冰生产点 +
  网格收敛阶梯 + LWC 线性族 + 确定性对，状态 `verified`，2026-10-06）
- **独立入口**：`scripts/sdaa_sweep_aircraft_icing_naca0012_rime.py`（未改任何业务代码；
  不 import 案例 `run.py`，因为后者硬断言 `torch.cuda.is_available()`）
- **机器输出**：`artifacts/sdaa_aircraft_icing_naca0012_rime.json`
  （逐案 `artifacts/sdaa_runs/aircraft_icing_naca0012_rime/<run>/`）
- **总判定**：**FAIL** —— 6/11 判据 PASS，5 条（G7 确定性、G10a–d 收敛阶梯）在 20 min 墙钟内
  跑不到，按纪律记 FAIL + 原因，未降判据、未用 CPU 结果充数
- **墙钟**：单次扫荡 **1048 s ≈ 17.5 min**（≤20 min 上限）

## 1. 案例判据口径（原样沿用，未改）

案例判据族是**内部一致性**（无外部实验锚）：G1 Lagrangian 质量闭合、G2 Eulerian 闭合、G7 IC-A
逐位确定性（15 数组 sha256）、G9 LWC 线性（log-log 斜率 / 比例拟合 R² / 比例偏差 / 单调）、
G10 网格阶梯 + 三点 Richardson 外推（并与 IC-A 先验 `p`、`f_ext` 比对）。门值定义与阈值取自
案例 `README.md` + `result.json::gates`；**本报告中所有判据数值均由本次 SDAA 运行现算**，
脚本内无任何门值字面量（仅 G10d 的*先验* `p=4.593597963850595`、`f_ext=10.50799579713059 mm`
与各门 prior 列是用运行时读取的 `result.json` 做对照）。

反解出的门算术（与案例先验逐位对齐验证过）：

| 门 | 算式 |
|---|---|
| G1/G2 | `max over runs` of `audit_lagr.closure_error` / `audit_euler.closure_error` |
| G7 | 逐数组 `sha256(np.ascontiguousarray(npz[k]).tobytes())`，`det_run1` vs `det_run2` 全等 |
| G9a | `np.polyfit(log LWC, log deposited, 1)[0]` |
| G9b | 过原点比例拟合 `k=Σxy/Σx²`，`R²=1-SS_res/SS_tot(about mean)` |
| G9c | `max |(y-kx)/(kx)| · 100` |
| G9d | `np.all(np.diff(deposited) > 0)` |
| G10b/c | `t = f_ext - a·dx^p` 三点非均匀 Richardson（`dx ∝ 1/nx`，解析消元 + brentq 求 p），`err=(t-f_ext)/f_ext·100` |

自检：用 NVIDIA 先验冰厚代入上式复现出 `p=4.593597963850595`、`f_ext=10.50799579713059 mm`、
`err L3=-0.18837%`、`err L4=-0.85675%`，与案例 `result.json` 的 prior 相对误差 0.0（1e-16 级）。

## 2. 运行配置与设备证据

计算张量全部放 `sdaa` 设备（`IcingConfig.device="sdaa"`，其余配置与案例 `run.py` 逐字段一致）：
NACA 0012、chord 0.5334 m、V∞ 67 m/s、AoA 4°、LWC 0.5 g/m³、MVD 20 µm、T∞ −10 °C、
曝光 360 s、Re 2.5e6；D3Q19 cumulant（c_s=0.1）+ Eulerian donor2 液滴相 + macklin rime 密度；
生产格 320×160（弦长 128 lu，dx=4.1672 mm，Stokes=0.1551），steps=3000，warmup=6000。

**设备证据行**（脚本直接调用、写入 JSON `device_evidence_line`）：

```
torch.sdaa.get_device_name(0) = '/dev/tcaicard24' (sdaa available=True, count=1,
torch=2.7.1, compute device='sdaa', cuda used=False)
```

补充证据：`torch.sdaa.is_available()=True`、`device_count=1`；设备侧 64×64 matmul 冒烟通过；
每次运行后 `torch.sdaa.max_memory_allocated(0)=47904768` B（≈45.7 MiB，证明张量确实驻留 SDAA）；
全程未创建任何 CUDA 张量（机器无可用 CUDA 设备，也不允许）。

**triton 绕行记录**：案例入口走的是 `compile_mode=None`（纯 torch 算子），`compile_unit` 的两个
编译变体从未构建，因此**不进入 triton kernel 路径**。仍按要求以 `TRITON_INTERPRET=1` 运行
（belt-and-braces），并如实登记：进程结束时 `"triton" in sys.modules = True`（由 torch/sdaa 后端
导入），但本案例执行路径未解释执行任何 triton kernel —— 计时差异不来自 interpret 模式。

复现命令：

```bash
cd /root/tp-agent/work/m13
TRITON_INTERPRET=1 python scripts/sdaa_sweep_aircraft_icing_naca0012_rime.py \
    --budget-s 1150
```

## 3. 结果表（3 案实跑 + ladder_L1 由 b1_prod 复用）

| run | 格子 | steps/warmup | LWC (g/m³) | 墙钟 (s) | β_e,peak | Eulerian 沉积 (kg) | t_max (mm) | 上/下角峰 (mm) | t_stag (mm) | n_ice | 冰面积 (% c²) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `b1_prod` | 320×160 | 3000/6000 | 0.50 | 356.9 | 0.920717 | 1.753473e-03 | 8.334375 | 4.167 / 8.334 | 12.5016 | 15 | 0.09155 |
| `m20_lwc025` | 320×160 | 3000/6000 | 0.25 | 352.3 | 0.919848 | 8.843855e-04 | 4.167188 | 0.000 / 4.167 | 8.3344 | 5 | 0.03052 |
| `m20_lwc10` | 320×160 | 3000/6000 | 1.00 | 333.3 | 0.928712 | 3.474131e-03 | 20.835938 | 12.502 / 20.836 | 25.0031 | 37 | 0.22583 |
| `ladder_L1` | 320×160 | 3000/6000 | 0.50 | — | 0.920717 | 1.753473e-03 | 8.334375 | 4.167 / 8.334 | 12.5016 | 15 | 0.09155 |

`ladder_L1` 与 `b1_prod` 配置逐字段相同（案例证据中两者也逐位相同），因墙钟预算未重复跑，
按 `reused_from="b1_prod"` 复用并在 JSON 中标注。生产点流态：cd0=0.060514，cl0=0.199712；
三案 prereg §4 护栏（m_w/impact_mass 有限、deposited>0、冰胞非零、wall<3600 s）全部 True。

## 4. 判据逐条比对（6 PASS / 5 FAIL）

| 门 | 判据 | SDAA 测得 | 结果 | 原因 / 备注 |
|---|---|---|---|---|
| G1 Lagrangian 质量闭合（最差案） | ≤1e-12 | 0.00e+00 | **PASS** | 3 案 + 复用 L1 全为 0 |
| G2 Eulerian 闭合（相对） | ≤1e-6 | 1.7692e-08 | **PASS** | 最差案 b1_prod，(NVIDIA prior 1.9799e-08) |
| G7 逐位确定性（det 对，15 数组 sha256） | 全等 | — | **FAIL** | NOT_RUN：det_run1/det_run2 未跑到（墙钟），无值可判 → 记 FAIL 而非 PASS |
| G9a 沉积量 log-log 斜率 | [0.97, 1.03] | 0.986952 | **PASS** | (prior 0.985732) |
| G9b 比例拟合 R² | ≥0.999 | 0.9998826 | **PASS** | k=3.483401 kg per (kg/m³) (prior 0.9998320) |
| G9c 比例偏差（最大点） | ≤3 % | 1.5543 % | **PASS** | 逐点 1.554 / 0.676 / 0.266 %（prior 1.6638 %） |
| G9d 沉积 vs LWC 单调 | 严格递增 | true | **PASS** | 8.844e-04 < 1.753e-03 < 3.474e-03 kg |
| G10a 阶梯 L1<L2<L3 | 严格单调 | — | **FAIL** | NOT_RUN：ladder_L2/L3 未跑到 |
| G10b Richardson 误差 L3 | \|err\|≤3 % | — | **FAIL** | NOT_RUN：无 L2/L3 冰厚 |
| G10c Richardson 误差 L4 | \|err\|≤3 % | — | **FAIL** | NOT_RUN：无 L4 |
| G10d Richardson 重现 IC-A 先验 | 相对 ≤1e-9 | — | **FAIL** | NOT_RUN：无三点阶梯 |

汇总（JSON `gates_summary`）：`n_pass=6, n_total=11, all_pass=False` → `verdict.overall="FAIL"`。

## 5. 墙钟与预算分析（为何 G7/G10 在 20 min 内跑不到）

实测单步成本：`b1_prod` 9000 步 / 356.9 s → **39.65 ms/step**；同配置 NVIDIA RTX 5090 为 28.8 s
（3.20 ms/step），即 SDAA 侧慢 **≈12.4×**。

按同一步成本外推其余格子（JSON `skipped_runs` 记录的即此判断）：

| run | 格子 | steps+warmup | 外推墙钟 |
|---|---|---|---|
| `ladder_L2` | 640×320 | 18000 | ≈2855 s |
| `ladder_L3` | 890×445 | 24934 | ≈7648 s |
| `ladder_L4` | 1280×640 | 39940 | ≈25340 s |
| 阶梯小计 | — | — | **≈35844 s（≈10 h）** |
| 全 9 案矩阵 | — | — | ≈37628 s（≈10.5 h，RTX 5090 为 472 s） |

因此按 1150 s 预算的实际排程为：`b1_prod`(357) → `m20_lwc025`(352) → `m20_lwc10`(333) = 1048 s，
之后 `det_run1` 起即触发预算闸（`spent=1048 s + est 333 s > 1150 s`）全部 SKIP。
选择 LWC 族优先而非 det 对，是因为前者一次覆盖 4 条门（G9a–d），后者只覆盖 1 条（G7）；
在只能跑 3 案的前提下这是门覆盖最大化，但仍不足以关门 —— 如实记 FAIL。

## 6. 与 NVIDIA 先验对照（复现性 + 设备敏感性披露）

同配置下 SDAA vs NVIDIA/RTX 5090（`evidence/b1_prod`）：

| 量 | SDAA | NVIDIA 先验 | 相对偏差 |
|---|---|---|---|
| t_max | 8.334375 mm | 8.334375 mm | **0.000 %** |
| t_stag | 12.501563 mm | 12.5015625 mm | **0.000 %** |
| n_ice_cells / 冰面积 | 15 / 0.0915527 % c² | 15 / 0.0915527 % c² | **0.000 %** |
| Eulerian 沉积 | 1.753473e-03 kg | 1.757730e-03 kg | −0.242 % |
| β_e,peak | 0.920717 | 0.911886 | +0.968 % |
| cd0 | 0.060514 | 0.058272 | **+3.85 %** |
| cd_end | 0.097459 | 0.095279 | +2.29 % |
| **cl0** | 0.199712 | 0.154396 | **+29.35 %** |
| **cl_end** | 0.073852 | 0.132712 | **−44.35 %** |
| cd_drift_pct | 61.05 % | 63.51 % | −3.87 % |

同样 `m20_lwc025`/`m20_lwc10` 的 t_max 与先验逐位一致（4.1671875 / 20.8359375 mm）。

**披露（非门，但与"SDAA 可用于本案例"的判断相关）**：
1. 冰形积分量（t_max / t_stag / 冰胞数 / 冰面积）在 SDAA 上与 NVIDIA **逐位相同**，沉积质量差 0.24%，
   β_pk 差 0.97%，说明结冰链（Eulerian 沉积 → macklin 密度 → 冰形量化）在 SDAA 上等价。
2. 但**流场本身不等价**：cl0 偏 +29.35%、cl_end 偏 −44.35%、cd0 偏 +3.85%。案例判据族没有气动门
   （无外部锚），所以这条不判 FAIL，但它是本次扫荡暴露的真实设备敏感性，需在把本案例升级为
   "SDAA verified" 之前给定论（候选原因：warmup 6000 步流场未充分收敛时 cumulant/float32 归约顺序差异
   被放大；本脚本不修改业务代码，故未做定位）。
3. cd_drift_pct 在三个 LWC 臂上量级差异大（+61% / +64% / −33%），与 NVIDIA 先验同量级，
   属案例已知口径（阻力随结冰冰形演化），非本扫荡新问题。

## 7. 限制与未完成部分（如实记录）

- **未跑**：`ladder_L2/L3/L4`（→G10a–d）、`det_run1/det_run2`（→G7）。原因：SDAA 单案 ≈350 s、
  阶梯外推 ≈10 h，远超 20 min 墙钟上限。要关门需放宽墙钟（约 10.5 h 单机）或改用
  `compile_mode` 提速（会改变数值口径，本次不做）。
- **Δt 内未做的定位**：cl0/cl_end 设备差异的根因排查（需逐步场对比，本次只交证据）。
- G1/G2 的 "worst over runs" 里包含 `ladder_L1`（= `b1_prod` 复用），因此覆盖面等价于 3 个独立算例。
- 未使用任何 CPU 计算结果：所有数值来自 `device="sdaa"` 的运行；`artifacts/sdaa_runs/.../result.npz`
  为对应设备侧产物（sha256 见 JSON `runs[*].array_sha256`）。

## 8. 结论

- 设备链路：**SDAA（`/dev/tcaicard24`，torch 2.7.1）上本案例的结冰求解链可跑通**，生产点与
  LWC 线性族的 6 条内部一致性判据全部 PASS，冰形与 NVIDIA 先验逐位一致（沉积差 0.24%）。
- 判据层面：**整体 honest-FAIL**，因为 G7（确定性）与 G10a–d（收敛/外推）在 20 min 内无法取得机器值，
  未获值即不判 PASS。
- 需追一项：SDAA 侧气动量 cl0/cl_end 与 NVIDIA 先验偏差 29%/44%，虽不在本案例门族内，
  但在把该案例认定为 "SDAA verified" 之前应给出解释。
## 9. 产物自检（本次实际执行）

```bash
python -m py_compile scripts/sdaa_sweep_aircraft_icing_naca0012_rime.py   # OK
TRITON_INTERPRET=1 python scripts/sdaa_sweep_aircraft_icing_naca0012_rime.py --budget-s 1150
# -> artifacts/sdaa_aircraft_icing_naca0012_rime.json, gates 6/11 PASS, overall=FAIL, wall=1048 s
```

- JSON 结构自检：`device_evidence_line` 含 `/dev/tcaicard24`；`gates` 共 11 条；
  `gates_summary.n_pass == Σ gate.pass`。
- **零手抄复核**：从 JSON 中 3 案存储值重算 G9a/G9b/G9c（0.9869524395629171 / 0.9998826455789217 /
  1.554259549798597）与 G1/G2，与 JSON 记录值差 0（逐位相等）——判据数值确由脚本现算，
  非从 NVIDIA 证据拷贝。
- 产物：`artifacts/sdaa_runs/aircraft_icing_naca0012_rime/{b1_prod,m20_lwc025,m20_lwc10}/result.npz`
  三份设备侧 npz + 冰形/β/历史图 CSV/PNG；`git status` 仅显示 4 项新增（无业务代码改动）。
