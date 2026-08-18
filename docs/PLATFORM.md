# TensorLBM Browser Platform (v0.3)

## 1. 平台架构图

```text
┌────────────────────────────────────────────────────────────────────┐
│                          Browser / Frontend                        │
│  Job Console · AI Agent Chat · Case Templates · KPI Dashboards     │
└───────────────┬────────────────────────────────────────────────────┘
                │ HTTP / WebSocket
┌───────────────▼────────────────────────────────────────────────────┐
│                      FastAPI Backend (app/backend)                 │
│  28 Routers · Auth/Middleware · Job Manager · Agent Core           │
│  AI Suboff / AI LES / Flow Transformer / Orchestration APIs        │
└───────────────┬────────────────────────────────────────────────────┘
                │ Service / Public API calls
┌───────────────▼────────────────────────────────────────────────────┐
│                         TensorLBM Library (src/)                   │
│  LBM Solvers · Turbulence Models · CAD/Physics · AI Modules        │
│  Stable Public API: tensorlbm.*                                    │
└───────────────┬────────────────────────────────────────────────────┘
                │
   ┌────────────▼────────────┐      ┌──────────────────────────────┐
   │ Data & Artifacts Layer  │      │ Compute Layer                 │
   │ SQLite / Outputs / Logs │      │ CPU / CUDA / HPC Schedulers   │
   └─────────────────────────┘      └──────────────────────────────┘
```

## 2. 用户旅程（User Journey）

1. **选择场景**：用户在浏览器选择模板（如 SUBOFF、圆柱、船体、AI 任务）或直接用 Agent 输入自然语言目标。  
2. **提交任务**：前端通过 FastAPI 提交求解/训练请求，后端落地参数、创建作业、分配执行资源。  
3. **实时观测**：通过 WebSocket 查看进度、日志、关键指标与中间结果图；可中断、重试或复用历史配置。  
4. **结果分析**：任务完成后在平台查看输出文件、图像、误差指标和 benchmark 对照，并生成可追溯报告。  
5. **资产沉淀**：可将数据与模型纳入平台目录/注册表，形成后续复现实验与 AI 推理的可复用资产。  

## 3. 与库的边界（Library vs Platform）

- **库（`src/tensorlbm/`）负责**：数值方法、物理模型、算法实现、可复用 Python API。  
- **平台（`app/backend/`）负责**：任务编排、路由协议、作业生命周期、可视化入口、Agent 协同。  
- **边界约束**：平台只能依赖 `tensorlbm` 公共 API；库层不得反向依赖 `app`。这样可保持“研究库可独立运行，平台可独立演进”。  
- **演进策略**：新增工业能力先在库中抽象成稳定接口，再在平台封装成业务路由或 Agent 工具。  

## 4. 部署方式

### 4.1 Docker（推荐开发/演示）
- 使用单容器或前后端同节点部署，快速启动 API、Web UI 与作业管理。  
- 适合 PoC、团队演示和小规模并发。

### 4.2 Kubernetes（推荐生产/HPC 网关）
- 后端服务无状态化部署，配合持久化卷保存输出和模型。  
- 可将作业执行对接 GPU 节点池或外部 HPC 调度器，支持弹性扩缩容与多队列策略。  

### 4.3 单机（研究者个人环境）
- 直接 `uvicorn backend.main:app` 启动，适合算法调试和本地复现实验。  
- 可在 CPU 或单卡 CUDA 上运行，最小运维成本。

## 5. 谁在用（Who uses it）

- **CFD 研究工程师**：用统一入口批量跑工况、做误差对照和收敛分析。  
- **AI4Science 工程团队**：将高保真仿真数据沉淀为可训练数据集，迭代 surrogate/transformer 模型。  
- **平台与 MLOps 团队**：治理作业、模型版本与推理服务，把“库能力”交付为“可运营系统”。  
- **教学与验证用户**：以浏览器 + Agent 方式快速体验复杂流体问题，降低入门门槛。  
