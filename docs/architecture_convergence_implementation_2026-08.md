# TensorLBM 架构问题分析与优化改进（实施版，2026-08）

## 1. 评估范围与成功标准

### 范围
- 库层：`/home/runner/work/TensorLBM/TensorLBM/src/tensorlbm`
- 平台层：`/home/runner/work/TensorLBM/TensorLBM/app/backend`
- 工程层：仓库根目录 worker/launcher 脚本与 `experiments/`

### 成功标准（可量化）
- **状态一致性**：作业生命周期状态仅由 `job_manager` 维护（单一真源）。
- **重复逻辑下降**：路由层不再维护与作业管理并行的状态缓存。
- **接口稳定性**：现有 `/api/simulations/generic-run/*` 对外路径保持不变。
- **迁移风险可控**：改动分阶段、可回滚、每阶段可验证。

### 架构问题清单模板
| 字段 | 说明 |
|---|---|
| Problem | 问题描述 |
| Impact | 稳定性/性能/维护性影响 |
| Priority | P0/P1/P2 |
| Evidence | 代码位置与证据 |
| Optimization | 改造方案 |
| Risk | 迁移风险与回滚策略 |

## 2. 架构现状审计（证据）

### 三层现状
1. **领域层（Domain）**：`src/tensorlbm` 内存在大量物理与求解模块，能力丰富但公开面较大。  
2. **平台层（Platform）**：`app/backend/routers` 端点较多，存在 case-specific 与 generic-run 双轨。  
3. **实验层（Experiment）**：仓库根目录存在大量 `*_worker.py`、launcher 脚本，工程边界较松散。

### 高风险热点分级
- **P0（稳定性/一致性）**  
  - 问题：`simulations.py` generic-run 使用路由内状态与 `job_manager` 双状态源并存。  
  - 证据：改造前存在 `_generic_jobs` 与 `job_manager` 并行维护。
- **P1（可维护性/扩展性）**  
  - 问题：`solver.py` 与 `simulations.py` 在提交流程与运行编排上有职责重叠趋势。  
- **P2（工程整洁性）**  
  - 问题：根目录 worker/launcher 数量大，主产品能力与实验能力边界不够清晰。

## 3. 目标架构与依赖规则

### 四层目标架构
1. **Core Kernel Layer**：格子、碰撞、边界等基础能力。  
2. **Domain Feature Layer**：场景级求解编排与物理组合。  
3. **Platform Service Layer**：API、任务、调度、可观测。  
4. **Experiment Layer**：一次性实验、批处理与探索脚本。

### 依赖规则
- 上层可依赖下层，不允许下层反向依赖上层。  
- 平台路由层只负责请求编排，不保留并行生命周期状态。  
- 同层通过契约对象交互，避免隐式全局状态共享。  

## 4. 分阶段实施 Backlog（优先级排序）

### Phase A（接口收敛）
- 收敛公共 API 暴露面，划分 stable/experimental 出口。
- 对高风险导出建立兼容清单与弃用窗口。

### Phase B（服务收敛）
- 合并重复提交流程，减少路由内重复业务拼装。
- 抽象统一的作业提交与配置标准化入口。

### Phase C（状态收敛）**（本次已落地第一步）**
- generic-run 生命周期状态统一以 `job_manager` 为准。
- 删除路由侧并行状态缓存，保留现有 API 路径兼容。

### Phase D（配置收敛）
- 统一 schema/config 命名与默认值语义，减少散点配置。

### Phase E（目录收敛）
- 将 root 级 worker/launcher 逐步归档到实验分层目录。

### Phase F（验证收敛）
- 接口兼容回归 → 功能回归 → 性能基线对比。

## 5. 迁移兼容清单（本次变更）

### 保留不变
- `POST /api/simulations/generic-run`
- `GET /api/simulations/generic-run/{job_id}/status`
- `GET /api/simulations/generic-run/{job_id}/results`
- `GET /api/simulations/generic-run/{job_id}/fields/{field_name}`
- `WS /api/simulations/generic-run/{job_id}/ws`

### 内部变化
- generic-run 状态来源由“路由内字典 + job_manager”改为“仅 job_manager”。
- 结果中的大字段 `fields_data` 继续对 results 接口隐藏，但用于字段切片接口。
- 数值发散时由“静默中断并返回完成态”改为“显式失败态（failed）”。

### 回滚策略
- 若发现兼容问题，可回退到本次提交前版本；对外路径不变，回滚风险集中于内部状态管理。

## 6. 验收方式

- API 兼容：generic-run 端点仍可提交、查询、取结果、取字段切片、WS 推送。  
- 状态一致性：作业状态以 `job_manager.get_job(job_id)` 返回为唯一依据。  
- 失败可诊断：发散场景返回 failed 而非 completed。  
- 测试：新增 generic-run 路由回归测试覆盖关键接口行为。
