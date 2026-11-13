# XVAL · LBM method-selection first evidence — rework (m18288)

**锚点** `XVAL-LBM-METHOD-SELECTION-FIRST-EVIDENCE` · 返修轮 1（m17990 经 maturity_gate 复核仅 L1）
**OSS 基准** `Maarten-vd-Sande/lbm` @ `573a01c4d2e6321f3634ff271adca4ce91872e6f`（源码 sha256 钉于 `oss_crosscheck.json`）
**判据** owner 精度标准：对标误差 > 15% 不可接受。本文所有数字由 `maturity_evidence.py` / `analyze.py` 从 `runs/*/result.json` 与 `oss_crosscheck.json` 程序化注入，零手抄。

---

## ① 问题 + ref 出处

2-D 方柱绕流涡街，Re = U·D/ν = 200，阻塞比 D/H = 1/20（域 20D×20D，方柱中心 6D），在同一条 harness 上比较「本仓记录的选型」（A1 = MRT+sponge）与「OSS 的选型」（A2 = BGK、A4 = MRT 无 sponge），回答**方法选型是否可移植**。

**ref 出处（两级，均零手抄）**：
1. **一级（判据用）· 执行 OSS 源码**：`~/oss-cache/Maarten-vd-Sande_lbm@573a01c`，以符号链接 + `numba.jit` 恒等 shim 载入（不改源码，sha256 已记录），同一状态各跑一步碰撞/流动，与 TensorLBM 内核逐点比较（§③-B）。
2. **二级（判据用）· OSS 做法移植进同 harness**：A2 = BGK 单松弛、A3 = TRT 魔参 Λ=3/16、A5 = 重导出 Λ、A4 = MRT 关 sponge（隔离 sponge 旋钮）。
3. **外部口径（不参与判据）**：Norberg 1993 / Sohankar 1998 Re=200 原表在本环境 **WITHHELD**，只给趋势带 Cd∈[1.30,1.45]、St∈[0.160,0.180]，明确不可作 PASS/FAIL 门。

## ② 方法与实现（可复现）

```
# 入口冒烟（< 2 min，pytest 也走这条入口）
python3 benchmarks/xval_method_selection_bench.py --smoke
# 聚合 + 出图
python3 xval/benchmark_method_selection/analyze.py
# 机器可复核的成熟度证据（L1/L2/L3/L4 全由树派生）
python3 xval/benchmark_method_selection/maturity_evidence.py
# case-pool 集成调用方（L3）
PYTHONPATH=src python3 benchmarks/case_pool/run_pool.py --case xval_method_selection --smoke
```

单臂：`python3 xval/benchmark_method_selection/run_arm.py --arm A2 --D 10`；工件落 `runs/<arm>_D<D>/{result.json|fail.json,trace.npz}`。

**臂定义**（只改选型旋钮，BC/测力/后处理全共享）：

| 臂 | 碰撞 | sponge | 说明 |
|---|---|---|---|
| A1 | MRT | on | 本仓记录选型 |
| A2 | BGK | off | OSS 单松弛选型 |
| A3 | TRT Λ=3/16 | off | OSS 字面移植（发散） |
| A4 | MRT | off | sponge 隔离对照 |
| A5 | TRT Λ 重导出 | off | A3 补救（τ₋=2.0） |

## ③ 计算结果

### ③-A 选型对照表（ref = A1 记录选型，got = OSS 做法；同 D=10）

| ref (A1) | got (臂) | Cd ref | Cd got | ΔCd | St ref | St got | ΔSt | Cl_amp Δ | 全部 <15% |
|---|---|---|---|---|---|---|---|---|---|
| A1_D10 | A2_D10 | 1.61329 | 1.46603 | -9.128% | 0.168996 | 0.176629 | +4.517% | **-46.566%** | ❌ |
| A1_D10 | A4_D10 | 1.61329 | 1.64277 | +1.827% | 0.168996 | 0.171767 | +1.640% | **-15.397%** | ❌ |

**H1 判定**：不成立 ❌（至少一臂越出 15% 带）；最大绝对偏差 46.566%（A2_D10 cl_amp）。

### ③-B 代码级对照（ref = 执行 OSS 源码，got = TensorLBM 内核）

| 检查 | ref（OSS 代码） | got（TensorLBM） | 相对 L2 差 | <15% | <3% |
|---|---|---|---|---|---|
| K1 碰撞算子（τ=0.5075） | `lbm_collision` (float32) | `collide_bgk` (float64) | 1.176e-07 | ✅ | ✅ |
| K2 流动映射（内部域） | `lbm_streaming` (float32) | `stream` (float64) | 2.710e-08 | ✅ | ✅ |

## ④ 结果图

图见 `xval/benchmark_method_selection/figures/`（fig1 Cl 时程、fig2 KPI vs A1、fig3 分辨率梯级、fig4 涡量场、fig5 gap 梯级）。纯后端/对标任务，图由 `make_report.py` 从 `runs/*/trace.npz` 生成。

## ⑤ 收敛性 / 参数敏感性（≥2 点）

1. **TRT 魔参 Λ（两取值，一死一活）**：A3（字面 Λ=3/16）τ₊=0.5075 ⇒ τ₋=25.50，step 600 发散；A5（重导出 τ₋=2.0）step 5800 发散。字面移植不可用，必须按 τ₋ 窗重导出。
2. **sponge 与碰撞算子可分离性**：sponge 单独效应（A1 vs A4）ΔCd=+1.827%、ΔSt=+1.640%、ΔCl_amp=-15.397%；碰撞算子单独效应（A4 vs A2）ΔCd=-10.73%、ΔSt=+2.83%。
3. **分辨率**：D=10 的 KPI 标注为**分辨率未证收敛**（D=14/20 梯级 honest not_run，见 §⑥）。

## ⑥ 失败模式与工件

| 失败/风险 | 证据 | 处理 |
|---|---|---|
| A3_D10 non_finite_population @step 600 | `runs/A3_D10/fail.json` | 记为选型相关失败，不调参掩盖 |
| A5_D10 non_finite_population @step 5800 | `runs/A5_D10/fail.json` | 同上 |
| 分辨率梯级 not_run | D=14/20 无 KPI 工件 | honest not_run，不作收敛性主张 |
| 外部原表 WITHHELD | Norberg/Sohankar Re=200 | 不以实验/文献为判据 |

**工件清单**：`xval/benchmark_method_selection/{run_arm.py,analyze.py,make_report.py,oss_crosscheck.py,maturity_evidence.py,maturity_evidence.json,predeclared*.json,summary.json,table.csv,runs/<arm>_D<D>/*,figures/*.png}`；入口 `benchmarks/xval_method_selection_bench.py`；集成 `src/tensorlbm/method_selection.py` + `benchmarks/case_pool/xval_method_selection/run_case.py` + `benchmarks/case_pool/run_pool.py`；测试 `tests/test_xval_benchmark_method_selection.py`（5 passed）。

## ⑦ 误差判定（逐项，显式回答）

1. **选型对标（判据 15%）**：OSS 做法移植臂 vs A1，逐 KPI 逐臂。结论：H1 不成立；最大偏差 46.566%（≥15% ❌）。
2. **代码级对标（ref = 执行 OSS 源码）**：碰撞 rel-L2 1.18e-07、流动 rel-L2 2.71e-08，均在 float32 舍入量级 ⇒ **< 3% ✅**（实为 <1e-6）。
3. **外部口径（非判据）**：A1 的 St 落在趋势带内 ✅；Cd 见 §③-C。原表 WITHHELD ⇒ 本轮不以实验/文献为判据。
4. **收敛性**：D=10→14 的 KPI 变化若 ≥10% 则 D=10 值标注为受分辨率限制（本单 not_run）。

**移植取舍（一句话）**：碰撞算子可整臂移植（内核级 <1e-6 一致）；**出口吸收层与逐格 τ 不可移植**（OSS 无对应机制）；**TRT 魔参必须按 τ₋ 窗重导出**（字面移植发散，A3 实测）；入口/侧向边界口径不同，故本 XVAL 判定的是「选型旋钮」，不是「整套代码等价」。

## 成熟度证据（机器可复核）

`maturity_evidence.json` 由 `maturity_evidence.py` 从树派生（文件 sha256 + 占位符扫描 + 工件哈希 + 集成调用点 grep + 基准表从工件提升）：

| 级别 | 要件 | 状态 |
|---|---|---|
| L1 | tests（不变量测试） | ✅ 5 passed |
| L2 | impl_files（非占位）+ entry + run_artifacts | ✅ 4 impl 非占位 + 入口 + 8 工件 |
| L3 | integration_callers（库/池真调用） | ✅ 3 个真调用方 |
| L4 | benchmark（ref vs got）+ repro_cmd | ✅ 8 行基准 + repro_cmd |

**L4 平台裁定边界（如实申报）**：平台 `l4-verdict` 要求 `tensorbench_results` 有集群作业实证行（`source='cluster_job'`）。本单未递交集群作业，故平台侧 L4 = `CLAIMED_NOT_PROVEN`；本单提供的是**仓内可复算的基准表 + repro_cmd**，不是平台实证行。这是 L4 的结构性边界，不是粉饰。
