# Pre-trained Models (v0.3)

本文档列出当前仓库可直接使用的模型 checkpoint，并说明输入约定与加载方式。

## 1) SUBOFF v0.3（平台端到端产物）

- **文件**：[`models/suboff_v0.3.pt`](../models/suboff_v0.3.pt)
- **来源**：`tensorlbm.ai.suboff_platform_pipeline.SuboffPlatformPipeline` 端到端跑通（数据登记 → 训练作业 → 模型注册）
- **格式**：PyTorch checkpoint（包含 `encoder`、`decoder`、优化器与调度器状态）
- **适用场景**：SUBOFF surrogate 流程联调、平台演示、checkpoint 加载验证

### 快速加载

```python
import torch

ckpt = torch.load("models/suboff_v0.3.pt", map_location="cpu")
print(ckpt.keys())  # encoder / decoder / n_iter / enc_optim / enc_sched
```

## 2) 历史 SUBOFF checkpoints（基线对照）

以下模型位于 `checkpoints/suboff/`，可作为历史版本对照与回归验证输入：

- `checkpoints/suboff/suboff_mse.ckpt`
- `checkpoints/suboff/suboff_full.ckpt`
- `checkpoints/suboff/suboff_600x150.ckpt`
- `checkpoints/suboff/suboff_4k.ckpt`
- `checkpoints/suboff/suboff_2000ep.ckpt`

## 3) 数据与输入约定

SUBOFF 训练/推理数据目录采用统一结构：

```text
<data_dir>/
  p/*.npy
  ux/*.npy
  uy/*.npy
  uz/*.npy
```

`suboff_train.py` 在训练时会对快照执行 `[49:149, :, 49:149]` 裁剪并展平，因此输入快照尺寸需满足对应空间范围。

## 4) v0.3 闭环演示（checkpoint → 推理 → 流场图）

目标：固定 `models/suboff_v0.3.pt`，执行真实推理并导出标准化流场图与元数据，形成可复现外部演示。

### 4.1 一键命令

```bash
PYTHONPATH=src python examples/suboff_v03_demo.py \
  --data-dir /abs/path/to/suboff8 \
  --output-dir outputs/suboff_v03_demo
```

### 4.2 输入契约

- checkpoint：`models/suboff_v0.3.pt`
- checkpoint 必须包含 keys：`encoder`, `decoder`, `n_iter`, `enc_optim`, `enc_sched`
- 数据目录：`data_dir/{p,ux,uy,uz}/*.npy`

### 4.3 输出产物（固定命名）

- `outputs/suboff_v03_demo/run_metadata.json`
- `outputs/suboff_v03_demo/suboff_v03_<channel>_<kind>_slice-<axis><idx>.png`
  - `channel ∈ {u, p}`（默认）
  - `kind ∈ {real, pred, abs_error}`

### 4.4 发布成功判据

1. 推理完成且返回指标：`mape` / `rel_l2_avg` / `mse_avg`
2. 流场图成功生成并可在输出目录定位
3. `run_metadata.json` 包含 checkpoint、数据路径、指标与图像清单

## 5) 后续计划

- 继续补齐 AI-LES 与 Flow-Transformer 的可下载 checkpoint；
- 在平台 API 中增加模型版本元数据与可追踪下载入口。
