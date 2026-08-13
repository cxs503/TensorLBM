# TensorLBM 模型服务化模块 — 干净室重实现规格

> 合规声明：本文档只记录**功能设计思想、接口契约、数据模型结构**（思想层面，
> 不受版权保护），不包含任何昆仑新能平台（KIEAP）的实现代码。所有代码均根据
> 本规格在 TensorLBM 中独立编写。

## 1. 背景与定位

TensorLBM 平台定位为超算 HPC+AI 一体化应用平台。数据链路：

```
HPC 数据生成（LBM 求解）→ 数据管理 → 模型训练 → 模型服务化 → AI 应用
      ✅ 已有              ✅ 已有      ✅ 已有       ← 本模块 →   待建
```

模型服务化模块的目标：把训练好的湍流闭包模型（`EddyViscosityMLP`）与自监督流场
Transformer（`FlowFieldTransformer`）**注册进模型库、加载并运行推理、导出为可移植
的 ONNX 格式**，为下游部署（推理服务、离线预测、嵌入式推理）提供统一入口。

## 2. 功能清单（借鉴设计思想，适配 CFD 场数据）

参考业界通行的模型服务化 / MLOps 能力（模型注册、推理服务、模型导出、模型量化压缩），
结合 TensorLBM 已有基础设施（`ai/database.py` 的 `models` 表、`ai/model.py`、
`ai/transformer.py`），本模块提供：

| # | 功能 | 说明 | 借鉴的设计思想 |
|---|------|------|----------------|
| 1 | 模型注册 | 把训练产物登记进 `models` 表，含版本、输入/输出 shape、血缘 | 模型注册表 / 模型仓库 |
| 2 | 模型检索 | 按 id / 列表 / 元数据查询已注册模型 | 模型目录 list/get |
| 3 | 模型加载 | 从注册记录定位磁盘 checkpoint 并重建模型对象 | 按路径多候选加载 + 缓存 |
| 4 | 推理服务 | 对已加载模型运行前向推理（numpy / tensor 双向） | 推理端点 predict |
| 5 | 模型导出 | 把 PyTorch 模型导出为 ONNX（动态 batch、opset 控制） | torch.onnx 导出器 |
| 6 | 模型元数据 | 记录输入/输出 shape、训练数据血缘、版本 | 模型描述符 / 元数据 |
| 7 | 依赖缺失降级 | ONNX 等可选依赖缺失时给出清晰错误而非崩溃 | 可选依赖检测 |

> 说明：KIEAP 还包含 sklearn/LightGBM/XGBoost 多框架导出、BentoML/Triton 部署、
> INT8 量化 / 剪枝 / 知识蒸馏等能力。本模块第一阶段只落地 PyTorch（TensorLBM
> 模型均为 PyTorch）的注册、推理与 ONNX 导出；多框架导出与量化压缩列入后续规划
> （见第 6 节），保持接口可扩展。

## 3. 数据模型（复用 ai/database.py `models` 表）

`serving.py` **不新建表**，直接对接 `ai/database.py` 的 `models` 表：

```
models(id, name, dataset_id, path, arch_json, metrics_json, created_at)
```

`arch_json` 字段承载完整的模型描述符（服务化元数据 + 底层架构），结构如下：

```json
{
  "family": "eddy_viscosity_mlp",        // 决定加载器：eddy_viscosity_mlp | flow_transformer_ssl
  "framework": "torch",
  "version": "1",
  "input_shapes":  {"input":  [null, 3]},   // null 表示动态 batch 维
  "output_shapes": {"output": [null, 1]},
  "lineage": {                               // 训练数据血缘
    "dataset_id": null,
    "training_run_id": null,
    "notes": ""
  },
  "arch": {                                  // 底层架构（ModelArch / FlowTransformerArch 的 asdict）
    "in_features": 3, "hidden_features": 16, "n_hidden_layers": 2, "activation": "tanh"
  }
}
```

`metrics_json` 继续承载训练指标（如 `final_val_loss`），与架构描述分离。

### 3.1 ModelMetadata（服务化元数据对象）

| 字段 | 类型 | 说明 |
|------|------|------|
| `version` | str | 模型版本号（默认 `"1"`） |
| `framework` | str | 后端框架（默认 `"torch"`） |
| `family` | str | 模型家族，决定加载器 |
| `input_shapes` | dict[str, list[int\|None]] | 输入名 → shape（`None` 表示动态维） |
| `output_shapes` | dict[str, list[int\|None]] | 输出名 → shape |
| `lineage` | dict | 训练数据血缘（dataset/run/备注） |
| `arch` | dict | 底层架构参数 |

## 4. 接口契约

### 4.1 ModelRegistry（SQLite，对接 models 表）

```python
reg = ModelRegistry.open("path/to/models.db")
model_id = reg.register_model(
    name="eddy-mlp", path="checkpoints/model.pt",
    arch={"in_features": 3, ...},
    dataset_id=None, metrics={"final_val_loss": 0.001},
    version="1", family="eddy_viscosity_mlp",
    input_shapes={"input": [None, 3]}, output_shapes={"output": [None, 1]},
    lineage={"training_run_id": "run-1"},
)
rec = reg.get_model(model_id)          # dict | None
rows = reg.list_models(limit=50)       # list[dict]
```

职责：仅做持久化与检索，不做模型对象加载（加载由 InferenceService 负责）。

### 4.2 InferenceService（推理服务）

```python
svc = InferenceService(reg)
model = svc.load_model(model_id)       # 按 family 选择加载器，带内存缓存
y = svc.predict(model_id, x)           # numpy/tensor 输入 → numpy 输出
svc.unload_model(model_id)             # 释放缓存
```

- 加载器选择：`family == "flow_transformer_ssl"` → `ai.transformer.load_flow_transformer_model`；
  其余 → `ai.model.load_model`（`EddyViscosityMLP`）。
- 推理统一走 `eval()` + `no_grad()`，返回 `numpy.ndarray`（float32）。

### 4.3 ONNX 导出

```python
path = export_onnx(model, example_inputs, output_path,
                   input_names=("input",), output_names=("output",),
                   dynamic_axes=None, opset_version=14, validate=True)
```

- 使用 `torch.onnx.export`（导出参数、常量折叠、动态 batch 轴）。
- 依赖缺失（`onnx` 未安装）时抛出 `OnnxUnavailableError`（`RuntimeError` 子类），
  并给出安装提示——不静默失败、不伪造产物。
- 导出前调用 `infer_io_shapes()` 对模型做前向校验并推导输入/输出 shape（该函数
  不依赖 onnx，任何时刻可用）。

## 5. 实现要点（独立重写约束）

1. **只依赖** 标准库 + `torch` + `numpy` + `tensorlbm.ai`（`database`/`model`/`transformer`）。
2. `onnx` 为可选依赖：模块顶层用 try/except 探测，缺失时导出路径给出明确错误。
3. 不复制 KIEAP 代码：加载、导出、元数据逻辑全部按 TensorLBM 既有模型 API 重写。
4. 与 `ai/database.py` 的 `LBMDatabase` 复用同一连接与 `insert_model`/`list_models`/
   `get_model_record` 助手函数，不重复实现 SQL。

## 6. 后续规划（本阶段不实现）

- 多框架导出（sklearn/LightGBM/XGBoost → ONNX，需 skl2onnx/onnxmltools）。
- ONNX Runtime 推理会话与 sklearn/ONNX 性能基准对比。
- 部署层：REST 推理端点（对齐 BentoML 打包 / Triton 客户端思想）。
- 模型压缩：动态 INT8 量化、剪枝、知识蒸馏。
- 端点运行统计（请求数 / 延迟 / 成功率）。
