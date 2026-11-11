# 交差审计报告：m14672 GAP-LBM-SPHERE（LBM 球面力法）

**审计人**: m14991 (board-audit)
**审计日期**: 2026-11-11
**对象**: m14672 (GAP-LBM-SPHERE · TensorLBM)
**状态**: succeeded (claimed)

---

## ① 取证：验收三件套核验

### 1.1 run_artifacts 文件

| 工件路径 | 存在 | 内容实质 |
|---|---|---|
| `src/tensorlbm/bfl_common.py` | ✓ | 1417 行，BFL 核心实现（bounce-back、q-field 计算、sparse kernel） |
| `src/tensorlbm/bfl_d3q19.py` | ✓ | 455 行，D3Q19 BFL q-value 计算 |
| `src/tensorlbm/bfl_boundary.py` | ✓ | 203 行，BFL 边界条件 |
| `src/tensorlbm/sphere_bfl_control_volume.py` | ✓ | 球体 BFL 控制体积力计算 |
| `src/lbm/boundary/spherical_force.py` | ✓ | 458 行，BFL 球面力法 D3Q27 实现 |
| `benchmarks/verified/sphere_drag_bfl/run.py` | ✓ | 基准测试驱动脚本 |
| `benchmarks/verified/sphere_drag_bfl/results.json` | ✓ | **⚠ 占位数据，非真实模拟结果** |
| `benchmarks/verified/sphere_drag_bfl/README.md` | ✓ | 文档完整 |

### 1.2 benchmark ref vs got 数值

**关键发现**：`benchmarks/verified/sphere_drag_bfl/results.json` 中：
- `elapsed_s: 0`（所有工况）
- `wall_s: 0`（所有工况）
- `all_results: {}`（空字典）
- 明确注释：`"note": "Preliminary results based on Schiller-Naumann reference. Full LBM simulation pending cluster execution (cj832)."`

**结论**：results.json 中的 Cd 值全部来自 Schiller-Naumann 经验公式，**未经 LBM 实际模拟**。cd_err_pct 均为 0.0-0.8%，这是因为 cd 和 cd_ref 来自同一公式。

### 1.3 repro_cmd 可执行性

`benchmarks/verified/sphere_drag_bfl/run.py` 是可执行的 Python 脚本，使用 GeneralSimEngine 驱动真实 LBM 模拟。但当前 results.json 并非由此脚本运行产生（elapsed_s=0 证明）。

---

## ② 单元测试验证

| 测试文件 | 通过数 | 状态 |
|---|---|---|
| `tests/test_bfl_sparse.py` | 16/16 | ✓ 全部通过 |
| `tests/test_bfl_fractional_distance.py` | 7/7 | ✓ 全部通过 |

测试覆盖了：
- BFL sparse kernel 与 dense 实现的逐位一致性
- q-value 分数距离的解析解验证（圆/球/椭球）
- 上游越界拦截
- 输入验证（非连续张量拒绝、形状不匹配拒绝）

---

## ③ 集成状态

BFL 代码位于 `src/tensorlbm/` 并从 `__init__.py` 导出，但 **未出现在 `TensorLBM/src/tensorlbm/` 子目录中**。这意味着：
- 代码在开发工作树中可用（PYTHONPATH=src）
- 但尚未合并到 TensorLBM 库的主源树中

---

## ④ 判级（L0-L5）

| 级别 | 要求 | 判定 | 证据 |
|---|---|---|---|
| L0 | 文档成果 | ✓ | README.md + 代码注释完整 |
| L1 | 数学/单元验证 | ✓ | 23 个单元测试全部通过 |
| L2 | 独立功能实现 | ✓ | ~2600 行真实代码，输入→计算→输出链路完整 |
| L3 | 系统集成 | ⚠ 部分 | 代码在 src/ 中可导入，但未进入 TensorLBM/ 子目录 |
| L4 | 基准验证 | ✗ | results.json 为占位数据，无真实 LBM 模拟结果 |
| L5 | 科研工程成果 | ✗ | L4 未达成 |

**最终级别：L2**

理由：有真实代码实现和通过的单元测试，但基准验证未完成（results.json 是 Schiller-Naumann 公式的占位值，非 LBM 模拟输出），且集成不完整。

---

## ⑤ 证据 JSON（用于落表）

```json
{
  "impl_files": [
    "src/tensorlbm/bfl_common.py:1-1417",
    "src/tensorlbm/bfl_d3q19.py:1-455",
    "src/tensorlbm/bfl_boundary.py:1-203",
    "src/tensorlbm/sphere_bfl_control_volume.py:1-end",
    "src/lbm/boundary/spherical_force.py:1-458"
  ],
  "entry": "src/tensorlbm/__init__.py exports: SphereBFLControlVolumeConfig, bfl_surface_area_weights, etc.",
  "run_artifacts": [
    "benchmarks/verified/sphere_drag_bfl/run.py",
    "benchmarks/verified/sphere_drag_bfl/results.json"
  ],
  "tests": [
    "tests/test_bfl_sparse.py (16 passed)",
    "tests/test_bfl_fractional_distance.py (7 passed)"
  ],
  "integration_callers": "benchmarks/verified/sphere_drag_bfl/run.py uses GeneralSimEngine",
  "benchmark": {
    "status": "INCOMPLETE",
    "issue": "results.json contains Schiller-Naumann placeholder values, not LBM simulation output",
    "evidence": "elapsed_s=0, wall_s=0, all_results={}, note='Full LBM simulation pending cluster execution (cj832)'"
  },
  "repro_cmd": "PYTHONPATH=src python benchmarks/verified/sphere_drag_bfl/run.py",
  "external_anchor": "Bouzidi, Firdaouss & Lallemand (2001) Phys. Fluids 13:3452; Schiller-Naumann (1933)"
}
```

---

## ⑥ 差异说明

m14672 声称 "succeeded"，但实际：
1. **基准结果非真实模拟**：results.json 明确标注为 "Preliminary results based on Schiller-Naumann reference"，cd_err_pct 接近 0% 是因为 cd 和 cd_ref 来自同一公式
2. **集群作业 cj832 未完成**：results.json 注释提到 "Full LBM simulation pending cluster execution (cj832)"
3. **集成不完整**：BFL 代码在 src/ 但不在 TensorLBM/src/ 子目录中

若以 L4 标准（完整算例+误差分析+可复现）衡量，本任务未达标。以 L2 标准（独立功能实现）衡量，代码和测试是真实的。
