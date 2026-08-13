# TensorLBM 模型训练模块 — 干净室重实现规格

> 合规声明：本文档只记录**功能设计思想、接口契约、数据模型结构**（思想层面，
> 不受版权保护），不包含任何昆仑新能平台（KIEAP）的实现代码。KIEAP 代码位于
> `/data/kunlun_ai_platform/kunlun-ai-platform/backend/`（专有闭源，无 LICENSE），
> 仅用于提炼功能设计思想；TensorLBM 的训练作业管理代码将根据本规格独立编写，
> 使用原生 `sqlite3` + `dataclass`，不采用 KIEAP 的 SQLAlchemy/线程内字典实现。

## 1. 背景与定位

TensorLBM 平台定位为超算 HPC+AI 一体化应用平台。训练链路：

```
HPC 数据生成（LBM 求解）→ 数据管理（catalog）→ 模型训练 → AI 应用
   ✅ 已有                ✅ 已有            ← 本模块 →   待建
```

训练模块的目标：让训练作业**可登记、可跟踪生命周期、可记录指标、可追溯
数据血缘**，并把训练完成的模型落库到已有 `ai/database.py` 的 `models` 表。

本模块第一阶段只做**训练作业管理（TrainingJobRegistry）**——即作业的
CRUD、状态流转、指标记录与数据血缘，不做训练执行器本身（执行器复用
`ml/torch_flow_training.py` 等已有证据门控训练代码）。

## 2. 功能清单（借鉴设计思想，适配 CFD 场数据）

参考业界通行的 MLOps 训练编排能力，结合 TensorLBM 已有数据契约，本模块提供：

| # | 功能 | 说明 | 对应 KIEAP 设计思想 |
|---|------|------|---------------------|
| 1 | 训练作业登记 | 创建训练作业，记录超参配置 + 数据集/模型引用 | TrainingJob / create_job |
| 2 | 作业生命周期 | 状态机流转：created→queued→running→completed/failed/cancelled | pending→running→completed/failed |
| 3 | 作业检索 | 按状态过滤 + 分页列出训练作业 | list_tasks / list_jobs |
| 4 | 指标记录 | 训练过程/最终指标写入作业（可合并多轮指标） | update_progress / record metrics |
| 5 | 模型落库 | 训练完成的模型登记进 models 表，回写 job.model_id | register_model |
| 6 | 数据血缘 | 记录 training-job ← dataset ← field-product 派生关系 | lineage / MLflow run lineage |
| 7 | 状态错误信息 | 失败作业记录 error 字段 | job.error |
| 8 | 实验追踪（预留） | 超参/指标/产物的实验级追踪（后续阶段） | MLflow experiment/run/log_metric |
| 9 | 分布式训练（预留） | 数据并行/梯度累积/混合精度配置（后续阶段） | DistributedTrainingJob |
| 10 | 模型版本（预留） | 模型版本递增 + 当前版本指向（后续阶段） | ModelVersionManager |

> 第 1–7 项为第一阶段实现目标（`src/tensorlbm/ml/training_job.py`）；
> 第 8–10 项仅记录设计思想，供后续阶段参考。

## 3. 数据模型（借鉴字段结构，重写持久化）

底层复用 TensorLBM 已有基础设施：
- `ai/database.py`：`runs` / `datasets` / `models` 三表（`insert_model` 落库模型）；
- `data/catalog.py`：`assets` / `lineage` 表（`add_lineage` / `upstream` 追踪血缘）。

新增训练作业管理层的一张 SQLite 表（与 `ai/database.py` 同库共存）：

### 3.1 training_jobs 表
- `job_id`: TEXT 主键（形如 `job_<12 位随机十六进制>`）
- `status`: TEXT，取值 created/queued/running/completed/failed/cancelled
- `config_json`: TEXT（超参配置，JSON）
- `model_id`: INTEGER（外键 → models.id，训练完成后回写）
- `dataset_id`: INTEGER（外键 → datasets.id）
- `metrics_json`: TEXT（指标，JSON，可合并多轮）
- `error`: TEXT（失败原因）
- `created_at` / `updated_at`: TEXT（ISO-8601 UTC 时间戳）

### 3.2 TrainingJob 数据类（内存视图）
- `job_id: str`
- `status: str`
- `config: dict[str, Any]`
- `model_id: int | None`
- `dataset_id: int | None`
- `metrics: dict[str, Any]`
- `error: str | None`
- `created_at: str`
- `updated_at: str`

### 3.3 状态机
```
created   → queued | running | failed | cancelled
queued    → running | failed | cancelled
running   → completed | failed | cancelled
completed → (终态)
failed    → (终态)
cancelled → (终态)
```
终态不允许再流转（防止已归档结果被误改）；非法流转抛 `ValueError`。

## 4. API 接口（借鉴端点设计，重写实现）

`TrainingJobRegistry`（SQLite，类 `FieldDataCatalog` 风格）：

| 方法 | 签名 | 说明 |
|------|------|------|
| `open` | `(db_path) -> TrainingJobRegistry` | 打开/创建库，建立表结构 |
| `close` | `()` | 关闭连接 |
| `create_job` | `(config, *, job_id=None, dataset_id=None, model_id=None) -> TrainingJob` | 登记作业，可指定 dataset_id |
| `get_job` | `(job_id) -> TrainingJob | None` | 查单个作业 |
| `list_jobs` | `(status=None, limit=50) -> list[TrainingJob]` | 列表，按状态过滤 |
| `update_status` | `(job_id, status, error=None) -> TrainingJob` | 状态流转（校验状态机） |
| `record_metrics` | `(job_id, metrics) -> TrainingJob` | 合并指标 |
| `register_model` | `(job_id, *, name, path, arch, metrics=None) -> int` | 落库 models 表并回写 model_id |
| `record_lineage` | `(catalog, job_asset_id, *, dataset_asset_id=None, product_asset_id=None)` | 写血缘边 |

`register_model` 对接 `ai/database.py`：调用 `insert_model` 时自动带上
`job.dataset_id` 作为模型的数据集引用，并把返回的 `model_id` 写回作业记录。

`record_lineage` 复用 `data/catalog.py` 的 `add_lineage`，记录两条有向边：
- `product_asset_id → dataset_asset_id`（`relation_type="derived_from"`）
- `dataset_asset_id → job_asset_id`（`relation_type="trained_on"`）

## 5. 与 KIEAP 的差异（CFD 场数据 vs 工业时序）

| 维度 | KIEAP（工业时序） | TensorLBM（CFD 场数据） |
|------|-----------------|----------------------|
| 持久化 | SQLAlchemy + 线程内字典 + JSON 文件 | 原生 sqlite3 + dataclass（同 catalog） |
| 训练后端 | sklearn / PyTorch MLP / LightGBM / XGBoost | 证据门控 Torch flow-transformer |
| 任务类型 | classification / regression / time_series | FIELD_RECONSTRUCTION / TURBULENCE_CLOSURE / SURROGATE |
| 实验追踪 | MLflow（experiment/run/metric/artifact） | SQLite training_jobs 表（轻量） |
| 分布式 | DDP / 梯度累积 / 混合精度 | 超算调度（app/backend hpc_scheduler.py，后续阶段） |
| 数据形态 | 表格/时序特征向量 | 网格场量（shape + dtype + units） |
| 血缘 | SQL 转换/列映射 | run→product→dataset→training-job |

**明确不借鉴**：KIEAP 的合成数据训练（随机矩阵造数）、业务场景硬编码的
预测默认值（光伏/故障/能耗等）、文件级 `registry.json`/`versions.json`
持久化、线程内全局单例字典。

## 6. 后续阶段（仅记录思想，不在本阶段实现）

- 实验追踪：以 `job_id` 为 run 维度，记录超参/指标/产物清单（对应 MLflow 的
  experiment → run → metric/param/artifact 模型），轻量落地到 SQLite。
- 模型版本：同一逻辑模型多版本递增 + `is_current` 指向 + 版本对比（对应
  ModelVersionManager 的 create_version / get_current_version / compare）。
- 分布式训练：作业 config 携带 `num_workers` / `gradient_accumulation_steps` /
  `mixed_precision`，由超算调度器（hpc_scheduler.py）消费。
