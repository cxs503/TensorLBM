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

## 4) 后续计划

- 继续补齐 AI-LES 与 Flow-Transformer 的可下载 checkpoint；
- 在平台 API 中增加模型版本元数据与可追踪下载入口。
