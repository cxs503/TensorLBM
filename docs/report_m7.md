# Mission 7 — 船舶总体性能平台审计报告（补报）

> **Mission ID:** 7 · **Repo:** ship-platform · **Scenario:** report-backfill  
> **日期:** 2025-05-30 · **状态:** honest-FAIL（原单审计已完成交付，但缺论文级进展报告，本次补报）

---

## 一、目的

对船舶总体性能平台（:8001）进行系统审计，覆盖 21 项能力的证据等级评定、6 个核心 API 端点探测、标模库数据质量检查、TensorLBM benchmark 状态确认，以及源码可追溯性验证。最终产出 `docs/platform_audit.md`，为后续改进提供优先级路线图。

本补报单的目标是：**为已交付的审计结果补充论文级进展报告**，确保平台知识链完整。

---

## 二、方法

### 2.1 审计流程

| 步骤 | 工具/方法 | 目标 |
|------|----------|------|
| 1 | `audit_capabilities()` | 获取 21 项能力的证据等级（REAL/MOCK/STUB/WITHHELD/NOT_SUPPORTED） |
| 2 | `call_ship_api` × 6 | 探测 resistance / stability / seakeeping / platforms / list-models / benchmarks |
| 3 | 源码目录扫描 | 验证后端实现代码是否在工作树中可追溯 |
| 4 | 交叉比对 | 将能力清单、API 响应、源码存在性三者交叉验证 |

### 2.2 证据等级定义

| 等级 | 含义 |
|------|------|
| **REAL** | 存在真实算法实现，返回非硬编码结果 |
| **MOCK** | 硬编码或随机返回，无真实计算 |
| **STUB** | 空壳实现，仅返回固定结构 |
| **WITHHELD** | 证据不足，无法判定 |
| **NOT_SUPPORTED** | 能力完全不存在 |

### 2.3 精度标准

- 对标实验/参考值误差 >15% 不可接受
- 几何参数必须先验证 Cb/L/B 合理性
- 无实验对标的能力不得标注为"已验证"

---

## 三、关键结果

### 3.1 能力总览（21 项）

| 统计项 | 数量 | 占比 |
|--------|------|------|
| REAL | 18 | 85.7% |
| NOT_SUPPORTED | 2 | 9.5% |
| STUB | 1 | 4.8% |
| MOCK | 0 | 0% |
| WITHHELD | 0 | 0% |

### 3.2 能力明细表

| # | 能力标识 | 证据等级 | 描述 | 通过 |
|---|---------|---------|------|------|
| 1 | `calc.resistance` | REAL | 阻力计算 | ✅ |
| 2 | `calc.propulsion` | REAL | 推进计算（敞水） | ✅ |
| 3 | `calc.stability` | REAL | 稳性计算 | ✅ |
| 4 | `geometry.generate` | REAL | 几何生成 | ✅ |
| 5 | `geometry.hydro_curves` | REAL | 几何水动力曲线 | ✅ |
| 6 | `geometry.list_models` | REAL | 几何模型列表 | ⚠️ 路由 Bug |
| 7 | `geometry.verify_official` | REAL | 几何验证 | ✅ |
| 8 | `shiphull.generate` | REAL | 船体生成 | ✅ |
| 9 | `stdmodels.lookup` | REAL | 标模查询 | ✅ |
| 10 | `dev.gate_5415` | REAL | 开发工具 | ✅ |
| 11 | `dev.git` | REAL | 开发工具 | ✅ |
| 12 | `dev.logs` | REAL | 开发工具 | ✅ |
| 13 | `dev.nginx_audit` | REAL | 开发工具 | ✅ |
| 14 | `dev.probe` | REAL | 开发工具 | ✅ |
| 15 | `dev.restart` | REAL | 开发工具 | ✅ |
| 16 | `dev.services` | REAL | 开发工具 | ✅ |
| 17 | `gov.stats` | REAL | 治理统计 | ✅ |
| 18 | `knowledge.docs` | REAL | 知识文档 | ✅ |
| 19 | `ai.predict` | **NOT_SUPPORTED** | AI 预测 | ❌ |
| 20 | `methods.computation` | **NOT_SUPPORTED** | 计算方法 | ❌ |
| 21 | `misc.stubs` | **STUB** | 杂项桩 | ❌ |

### 3.3 API 端点探测结果

| 端点 | 方法 | HTTP 状态 | 关键发现 |
|------|------|----------|---------|
| `calculations/resistance` | GET | 405 | 仅支持 POST |
| `calculations/stability` | GET | 405 | 仅支持 POST |
| `calculations/seakeeping` | GET | 405 | 端点存在但未注册为能力 |
| `standard-models/platforms` | GET | 200 | 返回 8 个平台完整几何参数 |
| `geometry/list-models` | GET | 404 | **路由 Bug**: "list-models" 被当作船名查询 |
| `tensorlbm/benchmarks` | GET | 200 | 11 个 benchmark 全部 `never_run` |

### 3.4 标模库数据质量（8 个平台模型）

| 模型 | 类型 | Lpp (m) | B (m) | T (m) | Cb | 完整性 |
|------|------|---------|-------|-------|-----|--------|
| MV Blue Marlin | semi_submersible | 216 | 42 | 10 | 0.70 | ✅ |
| Typical FPSO | offshore_fpso | 300 | 60 | 22 | 0.85 | ✅ |
| Cylindrical FPSO | offshore_fpso | 80 | 80 | 25 | 0.785 | ✅ |
| Semi-sub Platform | offshore_platform | 100 | 80 | 20 | 0.25 | ✅ |
| Spar Platform | offshore_platform | 40 | 40 | 200 | 0.785 | ✅ |
| TLP | offshore_platform | 80 | 80 | 35 | 0.30 | ✅ |
| Jack-up Platform | offshore_platform | 70 | 60 | 6 | 0.60 | ✅ |
| FLNG | offshore_fpso | 488 | 74 | 26 | 0.82 | ✅ |

> 所有模型均包含 Lpp/B/T/Cb/Cm/Cp/Cw/KG/DWT/displacement/design_speed 完整参数。

### 3.5 TensorLBM Benchmark 状态

11 个 benchmark 全部为 `never_run`：  
`accuracy`, `all_models`, `bluff_body_matrix`, `dam_break`, `dam_break_3d`, `dam_break_quant`, `fs_2d`, `marine`, `mlups`, `multiphase`, `propeller`

### 3.6 源码可追溯性

| 预期路径 | 状态 |
|---------|------|
| `backend/api/calculations.py` | ❌ 不存在 |
| `backend/platform_harness/capabilities.json` | ❌ 不存在 |
| `backend/platform_harness/guard.py` | ❌ 不存在 |

> 后端 API 服务运行在 :8001 但源码不在本仓库工作树中。

---

## 四、缺陷清单与改进优先级

### 4.1 缺陷清单（14 项）

| # | 能力/端点 | 等级 | 缺陷描述 |
|---|---------|------|---------|
| 1 | `ai.predict` | NOT_SUPPORTED | AI 预测能力完全缺失 |
| 2 | `methods.computation` | NOT_SUPPORTED | 计算方法能力缺失 |
| 3 | `misc.stubs` | STUB | 杂项桩为空壳 |
| 4 | `calc.resistance` GET | REAL | GET 返回 405 |
| 5 | `calc.stability` GET | REAL | GET 返回 405 |
| 6 | `geometry/list-models` | REAL | **路由 Bug**: 返回 404 |
| 7 | `calculations/seakeeping` | 未注册 | 端点存在但未注册为能力 |
| 8 | TensorLBM benchmarks | — | 11 个全部 never_run |
| 9 | 操纵性 | NOT_SUPPORTED | 9 大学科中完全缺失 |
| 10 | 结构强度 | NOT_SUPPORTED | 完全缺失 |
| 11 | 振动噪声 | NOT_SUPPORTED | 完全缺失 |
| 12 | 绿色航行 | NOT_SUPPORTED | EEDI/CII 缺失 |
| 13 | 冰区航行 | NOT_SUPPORTED | 有场景脚本但无平台能力 |
| 14 | 后端源码 | — | `backend/` 不在工作树 |

### 4.2 优先级矩阵

| 优先级 | 缺陷 # | 描述 | 理由 |
|--------|--------|------|------|
| **P0** | #6 | geometry/list-models 路由 Bug | 基础功能不可用 |
| **P0** | #14 | 后端源码不在工作树 | 阻断源码级审计 |
| **P1** | #7 | seakeeping 未注册 | 核心学科不被追踪 |
| **P1** | #8 | benchmarks 从未执行 | 精度无法验证 |
| **P1** | #4, #5 | GET 返回 405 | API 设计缺陷 |
| **P1** | #2 | methods.computation 缺失 | 阻力/推进基础设施 |
| **P1** | #3 | misc.stubs 空壳 | 影响可信度 |
| **P2** | #9-#13 | 学科覆盖缺口 | 中长期路线图 |
| **P2** | #1 | ai.predict 缺失 | 增强能力 |

---

## 五、审计结论与局限

### 5.1 总体评估

**优势：**
- 核心计算能力（阻力/推进/稳性）均为 REAL，无 MOCK 项
- 几何/标模基础设施完善，8 个标模参数齐全
- 7 个 dev.* 运维工具均为 REAL
- 研究运行时框架（TensorPhysics）具备完整的 artifact provenance、fail-closed policy、S3 存储和平台注册能力

**风险：**
- 9 大船舶性能学科仅覆盖 3 个（33%）
- TensorLBM 11 个 benchmark 全部从未执行，精度未验证
- 后端源码不可追溯，REAL 能力无法源码级验证
- geometry/list-models 路由 Bug 影响基础可用性

### 5.2 局限（honest-FAIL 声明）

1. **源码级精度验证未完成**：后端实现代码不在工作树中，18 项 REAL 能力仅通过 API 响应判定，未进行代码审查。
2. **实验对标未执行**：阻力/稳性/推进计算结果未与实验数据或经典参考值对比，无法确认误差是否在 15% 以内。
3. **TensorLBM 精度未验证**：所有 benchmark 均为 never_run，LBM 计算精度完全未知。
4. **学科覆盖率不足**：操纵性、结构强度、振动噪声、绿色航行、冰区航行 5 个学科完全缺失平台能力。
5. **耐波性（seakeeping）**端点存在但未纳入能力追踪系统，存在管理盲区。

### 5.3 建议行动项

1. **立即**：修复 geometry/list-models 路由 Bug（P0）
2. **立即**：确认后端代码仓库位置并纳入审计范围（P0）
3. **短期**：注册 seakeeping 为能力；执行 marine/propeller benchmark（P1）
4. **短期**：为 resistance/stability 增加 GET 查询接口（P1）
5. **中期**：规划操纵性、结构、绿色航行能力建设（P2）
6. **长期**：集成冰区航行场景脚本为正式平台能力（P2）

---

## 附录：关联工件

| 工件 | 路径 | 说明 |
|------|------|------|
| 平台审计报告 | `docs/platform_audit.md` | 原单主交付物 |
| 集成状态 | `docs/INTEGRATION_STATUS.md` | 跨仓库集成边界与验收门 |
| 可执行研究方法论 | `docs/EXECUTABLE_RESEARCH_METHODOLOGY.md` | v1 策略内核 |
| DEM3D 弯曲基准 | `docs/RESEARCH_BENCHMARK_DEM3D_BENDING.md` | 三点弯曲研究环 |
| DEM3D 定量响应 | `docs/DEM3D_QUANTITATIVE_RESPONSE.md` | 载荷灵敏度指标 |
| 平台注册研究环 | `docs/RESEARCH_LOOP_PLATFORM_REGISTRATION.md` | S3→平台注册路径 |
| 持久化工作器 | `docs/PERSISTENT_RESEARCH_WORKER.md` | 长运行任务轮询 |
| Agent 子进程适配器 | `docs/AGENT_SUBPROCESS_ADAPTER.md` | CLI Agent 桥接 |
| 工件存储 | `docs/ARTIFACT_STORAGE.md` | Local/S3 双后端 |
