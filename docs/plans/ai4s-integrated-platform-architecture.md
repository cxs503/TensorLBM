# TensorLBM AI4S 集成开发平台 — 架构设计

> 定位：以 CFD 为切入点的 AI4S 综合平台，提供「集成开发平台 + 超算全栈服务」，
> 实现从 HPC 数据生产到 AI 模型开发的端到端闭环。

## 1. 平台总架构（三层）

```
┌─────────────────────────────────────────────────────────────┐
│  应用层：AI4S 应用（神经算子/PINN/GNN/生成式/逆问题/UQ...）   │
│  开发者通过「应用开发框架 SDK」快速构建新应用                 │
├─────────────────────────────────────────────────────────────┤
│  平台层：统一服务（数据管理/训练作业/模型服务/血缘/编排）      │
│  全栈编排：produce → register → train → serve 闭环自动化      │
├─────────────────────────────────────────────────────────────┤
│  基础设施层：HPC 求解器 + 超算调度(SLURM/PBS) + GPU + 存储     │
└─────────────────────────────────────────────────────────────┘
```

三层职责清晰分离：
- **基础设施层**：TensorLBM 自研求解器（数据生产）+ 超算调度 + GPU 资源
- **平台层**：数据管理（catalog）、训练作业（training_job）、模型服务（serving）、血缘——已具备
- **应用层**：AI4S 应用开发框架（核心新增）——让开发者快速开发新应用

## 2. 核心新增：AI4S 应用开发框架（Application SDK）

这是平台从「有几个案例」跃升为「集成开发平台」的关键。一个统一的应用接口，
开发者实现几个方法即可开发一个新的 AI4S 应用，自动获得全栈服务。

```python
# src/tensorlbm/apps/base.py
class AI4SApplication(ABC):
    """AI4S 应用的标准接口——继承并实现即可接入全栈平台。"""

    name: str          # 应用名，如 "neural_operator_fno"
    family: str        # 模型家族，如 "fno" / "pinn" / "gnn" / "diffusion"

    # ---- 开发者需实现的 5 个方法 ----
    @abstractmethod
    def produce_data(self, cfg) -> DataProduct:
        """数据生产：调用 HPC 求解器生成场数据（返回数据产品）。"""

    @abstractmethod
    def build_model(self, arch) -> nn.Module:
        """模型定义：构建神经算子/PINN/GNN 等模型。"""

    @abstractmethod
    def make_dataset(self, product: DataProduct) -> Dataset:
        """数据集：从数据产品构造训练数据集。"""

    @abstractmethod
    def train(self, dataset, model, cfg) -> TrainingResult:
        """训练：训练循环（返回 metrics + 模型权重路径）。"""

    @abstractmethod
    def infer(self, model, sample) -> Prediction:
        """推理：单样本/批量推理。"""

    # ---- 框架自动提供（无需开发者实现）----
    def run(self, *, hpc: HpcConfig, train: TrainConfig) -> RunReport:
        """全栈闭环：produce → register → train → serve，自动记录血缘。"""
```

**框架自动能力**（开发者继承即获得）：
1. **数据生产**：`produce_data` 通过 `hpc_scheduler` 提交到超算（SLURM/PBS）
2. **数据管理**：数据产品自动登记到 `FieldDataCatalog`（资产 + 质量 + 血缘）
3. **训练作业**：`train` 自动包装为 `TrainingJobRegistry`（状态机 + metrics）
4. **模型服务**：训练完成的模型自动注册到 `ModelRegistry`（+ 推理服务）
5. **血缘**：`data → dataset → job → model` 全链路自动记录
6. **超算/本地双模式**：`hpc` 配置为超算则走 SLURM，否则本地 GPU

## 3. 全栈服务编排（Orchestration）

统一编排器，把「数据生产 → 数据管理 → 模型开发 → 模型服务」串成自动化闭环：

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ 数据生产     │ → │ 数据管理     │ → │ 模型开发     │ → │ 模型服务     │
│ HPC 求解    │   │ catalog 登记 │   │ GPU 训练    │   │ 推理 API    │
│ (SLURM/PBS) │   │ 质量+血缘    │   │ TrainingJob │   │ serving     │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
        ↑ 自动调度          ↑ 自动登记          ↑ 自动记录          ↑ 自动注册
```

## 4. 超算全栈部署方案

数据生产（CPU 密集，多节点）+ 模型开发（GPU 密集，单/多卡）分而治之：

| 阶段 | 资源 | 调度 | 现状 |
|------|------|------|------|
| 数据生产 | 多节点 CPU | SLURM/PBS | ✅ hpc_scheduler 已有 |
| 数据管理 | 存储 + DB | 本地/共享 | ✅ catalog 已有 |
| 模型开发 | GPU（3090 多卡）| 训练作业 | ✅ training_job 已有 |
| 模型服务 | GPU/CPU 推理 | serving | ✅ serving 已有 |

**关键集成点**：`hpc_scheduler`（数据生产）→ `catalog`（数据落库）→ `training_job`（模型开发）→ `serving`（推理）的**数据流自动衔接**，这是应用框架要解决的。

## 5. 与现有能力的关系（不重造轮子）

| 现有能力 | 在新架构中的角色 |
|---------|----------------|
| 300 模块 LBM 求解器 | 数据生产的引擎 |
| hpc_scheduler（SLURM/PBS）| 超算数据生产调度 |
| data/catalog.py | 数据管理 |
| ml/training_job.py | 模型开发作业管理 |
| ml/serving.py | 模型服务 |
| 三个 AI 案例（SUBOFF/AI-LES/Flow Transformer）| 重构为应用框架的首批实例 |

## 6. 首批应用（应用框架的实例化）

把三个现有案例重构为 `AI4SApplication` 子类，同时开发首批新 AI4S 应用：

| 应用 | family | 类型 | 优先级 |
|------|--------|------|--------|
| neural_operator_fno | fno | 神经算子 | P0 |
| physics_informed_lbm | pinn | PINN | P0 |
| mesh_gnn_flow | gnn | GNN | P1 |
| generative_flow | diffusion | 生成式 | P1 |
| suboff_surrogate | suboff | 代理模型（已有）| 重构 |
| ai_les | eddy_mlp | 代理模型（已有）| 重构 |
| flow_transformer | flow_transformer_ssl | 自监督（已有）| 重构 |

## 7. 实施路线（三阶段）

### 阶段 A：应用开发框架（1-2 周）
1. `src/tensorlbm/apps/base.py` — AI4SApplication 抽象基类 + run() 全栈闭环
2. 把三个现有案例重构为应用框架实例（验证框架设计）
3. 应用注册表（应用发现 + 元数据）

### 阶段 B：超算全栈服务（2-3 周）
1. 应用框架接入 hpc_scheduler（数据生产上超算）
2. 数据生产 → 数据管理 → 模型开发的自动衔接
3. FastAPI 端点（应用 CRUD + 运行 + 状态查询）

### 阶段 C：新 AI4S 应用（持续）
1. neural_operator_fno（P0）—— 统一神经算子框架
2. physics_informed_lbm（P0）—— LBM-PINN
3. mesh_gnn_flow（P1）—— 图神经网络
4. generative_flow（P1）—— 生成式流场
5. 逆问题 + 数据同化 + UQ（P2）

## 8. 实施进度（2026-08 已固化）

已完成的 9 个 AI4S 应用（`src/tensorlbm/apps/`）：

| 应用 | family | 方法 | 状态 |
|------|--------|------|------|
| suboff_surrogate | suboff_surrogate | 代理模型（重构）| ✅ |
| ai_les | eddy_viscosity_mlp | 代理模型（重构）| ✅ |
| flow_transformer | flow_transformer_ssl | 自监督（重构）| ✅ |
| neural_operator_fno | fno2d | FNO 神经算子 | ✅ |
| physics_informed_lbm | pinn | PINN | ✅ |
| mesh_gnn_flow | gnn | MeshGraphNet GNN | ✅ |
| generative_flow | diffusion | DDPM 扩散生成 | ✅ |
| inverse_problem | inverse | 梯度反演 | ✅ |
| uncertainty_quantification | uq | MC-dropout UQ | ✅ |

配套设施：
- `apps/base.py` — AI4SApplication SDK + ApplicationRegistry
- `apps/hpc.py` — 超算全栈（HpcRunSpec + submit_app_hpc）
- `app/backend/routers/apps.py` — 应用管理 API（/api/apps）
- 95 个测试全过（12 应用/API 套件 + serving）

剩余 P2 方向：数据同化（DA/EnKF）、科学基础模型、LLM 科学 agent。
