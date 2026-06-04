# SUBOFF 平台使用说明书 / SUBOFF Platform Manual

## 1. 目标 / Goal

本文档说明如何在 TensorLBM 中运行 **SUBOFF 全附件（AFF-8）完整案例**，并检查 LBM 阻力计算是否达到 `final_error_pct <= 3.0` 的精度目标。

This document explains how to run the **SUBOFF full-appendage (AFF-8) case** in TensorLBM and verify that the LBM resistance calculation satisfies the `final_error_pct <= 3.0` acceptance target.

## 2. 环境准备 / Environment Setup

在仓库根目录执行：

```bash
pip install -e ".[dev]"
pip install -r platform/requirements.txt
```

启动平台：

```bash
cd platform
bash start.sh
```

浏览器访问：`http://localhost:8000`

## 3. 完整 SUBOFF 精度验证（推荐） / Full SUBOFF Accuracy Validation (Recommended)

### 3.1 命令行基准 / CLI benchmark

```bash
cd /tmp/workspace/cxs503/TensorLBM
PYTHONPATH=src python benchmarks/bench_marine.py --cases suboff --full
```

通过判据：

- 输出中 `target_met` 为 `true`
- `final_error_pct <= 3.0`

典型输出字段：

- `reference.cd_analytical`
- `reference.cd_richardson`
- `simulated.cd`
- `iterations[*].grid`
- `iterations[*].error_pct`

### 3.2 平台 API 方式 / Platform API route

平台后端支持只运行 SUBOFF 基准：

```bash
curl -X POST http://localhost:8000/api/benchmarks/marine \
  -H "Content-Type: application/json" \
  -d '{"cases":["suboff"],"fast":false,"device":"cpu"}'
```

返回结果包含 `job_id`。随后查询：

```bash
curl http://localhost:8000/api/jobs/<job_id>
```

当任务完成后，在 `result.suboff` 中检查：

- `name == "suboff_resistance"`
- `target_met == true`
- `final_error_pct <= 3.0`

## 4. 平台 CAD 几何工作流 / Platform CAD Geometry Workflow

虽然前端基准测试卡片会统一提交 Marine Suite，SUBOFF 的几何建模能力可直接通过 API 使用：

### 4.1 预览 SUBOFF 外形 / Preview hull

```bash
curl -X POST http://localhost:8000/api/cad/suboff/preview \
  -H "Content-Type: application/json" \
  -d '{"hull_type":"full","length":48.0,"radius":0.0,"bow_fraction":0.18,"stern_fraction":0.30,"stern_exponent":2.0}'
```

返回：

- `image`：base64 PNG 预览图
- `stats`：长径比、排水体积、湿表面积等几何统计

### 4.2 生成体素掩码 / Build voxel mask

```bash
curl -X POST http://localhost:8000/api/cad/suboff/hull-mask \
  -H "Content-Type: application/json" \
  -d '{"hull_type":"full","nx":160,"ny":64,"nz":64,"length":96.0,"radius":0.0,"device":"cpu"}'
```

返回：

- `image`：顶视图体素预览
- `stats.solid_cells`：实体格点数
- `stats.wetted_area_lu2`：湿表面积

### 4.3 导出 STL / Export STL

```bash
curl -X POST http://localhost:8000/api/cad/suboff/export-stl \
  -H "Content-Type: application/json" \
  -d '{"hull_type":"full","length":48.0,"radius":0.0,"n_axial":160,"n_circ":96}' \
  --output suboff_full.stl
```

## 5. 结果判读 / Result Interpretation

### 5.1 重点字段 / Key fields

| 字段 | 含义 |
|---|---|
| `target_met` | 是否达到精度目标 |
| `final_error_pct` | 最终网格相对 Richardson 外推值的误差 |
| `reference.cd_analytical` | ITTC-1957 理论摩擦阻力系数 |
| `reference.cd_richardson` | 网格细化后的收敛参考值 |
| `simulated.cd` | 最终 LBM 阻力系数 |
| `iterations[*].grid` | 每一级细化网格 |
| `iterations[*].wetted_area_m2` | 体素近似湿表面积 |

### 5.2 建议验收标准 / Suggested acceptance criteria

1. `target_met` 为 `true`
2. `final_error_pct <= 3.0`
3. `iterations` 至少包含 2 个细化层
4. 最后一层 `error_pct` 小于前一层或已满足阈值

## 6. 已验证命令 / Validated Commands

以下命令已用于本仓库的本地验证：

```bash
PYTHONPATH=src python benchmarks/bench_marine.py --cases suboff --full
PYTHONPATH=src python -m pytest -q
```

## 7. 故障排查 / Troubleshooting

- 若 `final_error_pct` 大于 3%，优先检查是否使用了 `--full` 或 `fast:false`
- 若运行时间过长，可先用 `fast:true` 做功能烟测，再用完整模式做精度验收
- 若平台任务无结果，检查 `/api/jobs/<job_id>` 中的 `status` 和 `error`
- 若需要更高吞吐，可将 `device` 切换为 `cuda:N`
