# TensorLBM 数据管理模块 — 干净室重实现规格

> 合规声明：本文档只记录**功能设计思想、接口契约、数据模型结构**（思想层面，
> 不受版权保护），不包含任何昆仑新能平台（KIEAP）的实现代码。TensorLBM 的
> 实现代码将根据本规格独立编写。

## 1. 背景与定位

TensorLBM 平台定位为超算 HPC+AI 一体化应用平台。数据链路：

```
HPC 数据生成（LBM 求解）→ 数据管理 → 模型训练 → AI 应用
        ✅ 已有                 ← 本模块 →   待建      待建
```

数据管理模块的目标：让 HPC 生成的 CFD 场数据**可登记、可检索、可校验、
可追溯、可版本化**，为下游模型训练提供可靠输入。

## 2. 功能清单（借鉴设计思想，适配 CFD 场数据）

参考业界通行的数据治理能力，结合 TensorLBM 已有数据契约，本模块提供：

| # | 功能 | 说明 | 对应 KIEAP 设计思想 |
|---|------|------|---------------------|
| 1 | 数据资产目录 | 登记/检索 CFD 场数据产品 | DataAsset 资产目录 |
| 2 | 资产元数据 | 键值元数据 + 置信度 + 来源 | AssetMetadata |
| 3 | 数据血缘 | 追踪数据从 run→product→dataset 的派生关系 | LineageNode/Edge |
| 4 | 数据质量 | 场数据完整性/有限性/守恒校验 + 得分 | DataValidationResult |
| 5 | 数据集管理 | 数据集登记、版本、train/val/test 划分 | DatasetManifest |
| 6 | 检索与索引 | 按字段名/工况/时间/标签检索 | — |

## 3. 数据模型（借鉴字段结构，重写 ORM）

底层复用 TensorLBM 已有契约 `data/contracts.py`：
`FieldProduct` / `DatasetSampleRef` / `DatasetManifest` / `RunManifest`。

新增数据管理层的持久化模型（SQLite，复用 `ai/database.py` 的 LBMDatabase 模式）：

### 3.1 DataAsset（数据资产登记）
- asset_id: str（主键，如 field product 的 product_id）
- name / description
- kind: field_product | dataset | run | model
- field_name / units / shape / dtype（场数据特有，来自 FieldProduct）
- tags: list[str]
- quality_score: int (0-100)
- sensitivity_level: str (public/internal/restricted)
- source_run_id: str（来源求解运行）
- status: active/archived
- version: str
- created_at / updated_at

### 3.2 AssetMetadata（资产元数据）
- asset_id: str（外键）
- key / value / source(manual/auto/system) / confidence: float

### 3.3 LineageRecord（数据血缘）
- source_id / target_id: str
- relation_type: derived_from | split_of | trained_on
- transformation: str（派生描述）
- resource_type: run/product/dataset/model

### 3.4 DataQualityReport（质量报告）
- asset_id: str（外键）
- checks: list（每项：check_name/passed/detail）
- overall_score: int (0-100)
- status: passed/failed/warning
- created_at

## 4. API 接口（借鉴端点设计，重写实现）

### 4.1 资产目录 `/api/data/assets`
- `GET  /assets`（列表，支持 field_name/tags/kind/status 过滤 + 分页）
- `POST /assets`（登记资产）
- `GET  /assets/{asset_id}`
- `PUT  /assets/{asset_id}`
- `DELETE /assets/{asset_id}`
- `POST /assets/{asset_id}/metadata`（加元数据）
- `GET  /assets/{asset_id}/metadata`
- `GET  /assets/{asset_id}/lineage`（查血缘）
- `POST /assets/{asset_id}/lineage`（加血缘）

### 4.2 数据质量 `/api/data/quality`
- `POST /quality/check`（对场数据运行校验：有限性、守恒、形状）
- `GET  /quality/{asset_id}/reports`（查质量报告）

### 4.3 数据集 `/api/data/datasets`
- `POST /datasets`（登记数据集 + train/val/test 划分）
- `GET  /datasets`
- `GET  /datasets/{dataset_id}`

## 5. 与 KIEAP 的差异（CFD 场数据 vs 工业时序）

| 维度 | KIEAP（工业时序）| TensorLBM（CFD 场数据）|
|------|-----------------|----------------------|
| 数据源 | mysql/influxdb/opcua/tdengine | LBM 求解产物（npy 场数据）|
| 数据形态 | 传感器时序、表格 | 网格场量（shape + dtype + units）|
| 质量校验 | 缺失值/异常值/一致性 | 有限性/质量守恒/形状完整性 |
| 血缘 | SQL 转换/列映射 | run→product→dataset 派生 |

**明确不借鉴**：KIEAP 的工业传感器协议（opcua/tdengine/mqtt）、场景化
业务（钢铁烧结/光伏/除尘）——这些与 CFD 场数据无关。

## 6. 实施顺序

1. 数据资产目录 + 元数据（对接 FieldProduct 契约）
2. 数据血缘（run→product→dataset 派生链）
3. 数据质量校验（有限性/守恒）
4. 数据集管理 + 版本 + 划分
5. API 端点（FastAPI router，对接 TensorLBM 现有 app/backend）
