# AI4S Applications Catalog (v0.3)

## 1) SUBOFF Surrogate Pipeline

**应用名**：SUBOFF 全流场重建与阻力评估代理模型  
**解决问题**：面向潜艇外流高维三维场景，传统高保真 LBM 工况扫描成本高、等待长。该应用将 `p/ux/uy/uz` 快照接入统一数据目录，先进行数据登记和质量检查，再执行 surrogate 训练并在平台注册模型，实现“仿真数据→模型→推理服务”的闭环。它适合用于阻力趋势筛选、参数预估和工程方案前置淘汰，把高成本全量仿真留给少量候选方案。  
**输入 / 输出**：输入为 `data_dir/{p,ux,uy,uz}/*.npy`、训练配置和可选推理配置；输出为数据资产记录、训练作业状态、模型注册记录、推理误差（如 `mape` / `rel_l2_avg`）与结果张量。  
**使用方法（一段代码）**：
```python
from tensorlbm.ai.suboff_platform_pipeline import SuboffPlatformPipeline
from tensorlbm.ai.suboff_train import SuboffTrainConfig

p = SuboffPlatformPipeline("artifacts/suboff_platform.db")
out = p.run_full_pipeline(
    data_dir="artifacts/suboff8",
    train_cfg=SuboffTrainConfig(data_dir="artifacts/suboff8", iters=100),
)
p.close()
```
**v0.3 对外演示（闭环）**：
```bash
PYTHONPATH=src python examples/suboff_v03_demo.py \
  --data-dir /abs/path/to/suboff8 \
  --output-dir outputs/suboff_v03_demo
```
固定输入契约：`models/suboff_v0.3.pt` + `data_dir/{p,ux,uy,uz}/*.npy`；固定产物：推理指标、标准命名流场图、`run_metadata.json`。
**性能（真实数据）**：当前仓库已具备可运行骨架与 checkpoint 训练/推理链路；标准 500k 点级单快照评估支持误差指标回传。公开 README 侧重流程能力展示，精确耗时/误差建议以目标机型复现实测为准。  
**示例链接**：`src/tensorlbm/ai/suboff_platform_pipeline.py`，`tests/test_suboff_platform_pipeline.py`，`docs/suboff_platform_manual.md`

---

## 2) AI-LES Pipeline

**应用名**：AI-LES 湍流闭合学习与嵌入式推理流程  
**解决问题**：传统 LES 在复杂参数空间下存在“精度-成本”矛盾，且手工调参与模型替换流程割裂。AI-LES 将高保真样本生成、特征/标签管理、训练与验证统一到平台流水线中，目标是让 AI 湍流闭合模型在保持可解释监控的前提下嵌入 LBM 求解器，减少多场景试错成本。其价值在于把“离线训练实验”变为可追溯、可复现实验资产。  
**输入 / 输出**：输入为 DNS/LES 数据源路径、训练超参数、实验标识；输出为模型工件、误差统计、训练日志与平台可查询的治理元数据（版本、配置、指标）。  
**使用方法（一段代码）**：
```python
from tensorlbm.ai.ai_les_platform_pipeline import AILesPlatformPipeline

pipe = AILesPlatformPipeline("artifacts/ai_les.db")
result = pipe.run_full_pipeline(
    data_dir="artifacts/les_dataset",
    train_cfg={"epochs": 20, "batch_size": 8},
)
pipe.close()
```
**性能（真实数据）**：目前仓库提供端到端流程骨架与测试覆盖，便于快速接入真实数据集并做统一指标留档；正式对外性能（训练时长、误差门限）依赖具体网格尺度与硬件资源。  
**示例链接**：`src/tensorlbm/ai/ai_les_platform_pipeline.py`，`tests/test_ai_les_platform_pipeline.py`，`docs/ai_turbulence.md`

---

## 3) Flow-Transformer Pipeline

**应用名**：Flow-Transformer 自监督流场建模与代理推理  
**解决问题**：工业流场常存在跨工况、跨尺度分布偏移，传统局部模型迁移性弱。Flow-Transformer 流水线通过序列化/结构化流场表示进行自监督学习，支持在平台内完成数据编目、训练作业管理和推理服务注册，使模型可以在不同工况下做快速响应预测与先验筛选。该应用定位于“高吞吐预估层”，用于缩短设计空间探索周期。  
**输入 / 输出**：输入为时间序列或快照化流场数据、模型结构参数、训练轮次和推理请求；输出为训练好的 Transformer 权重、验证指标、推理结果与全流程 lineage。  
**使用方法（一段代码）**：
```python
from tensorlbm.ai.flow_transformer_platform_pipeline import FlowTransformerPlatformPipeline

pipe = FlowTransformerPlatformPipeline("artifacts/flow_transformer.db")
result = pipe.run_full_pipeline(
    data_dir="artifacts/flow_transformer_dataset",
    train_cfg={"epochs": 10, "lr": 1e-4},
)
pipe.close()
```
**性能（真实数据）**：已完成平台级流程对接与测试支撑，重点价值是作业可追踪和模型可治理；建议在目标业务数据上补齐吞吐、延迟与泛化误差基线后再作为生产级默认模型。  
**示例链接**：`src/tensorlbm/ai/flow_transformer_platform_pipeline.py`，`tests/test_flow_transformer_platform_pipeline.py`
