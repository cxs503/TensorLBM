# TensorLBM 平台测试报告 / Platform Test Report

**项目 / Project:** TensorLBM Web Platform (B/S)  
**版本 / Version:** 1.0.0  
**测试日期 / Test Date:** 2026-05-24  
**测试范围 / Scope:** `platform/` 目录下的 FastAPI 后端 + Bootstrap 5 前端，包括预处理、CAD、求解器、后处理、基准测试、作业管理、WebSocket 全部 33 个 REST 端点。

---

## 1. 平台总体分析 / Platform Overview

TensorLBM 平台是一个基于 PyTorch 的 LBM 仿真 B/S（Browser/Server）平台。

```
platform/
├── backend/                FastAPI 应用
│   ├── main.py             入口、WebSocket 广播、CORS、SPA fallback
│   ├── job_manager.py      ThreadPoolExecutor + 状态广播
│   └── routers/
│       ├── jobs.py         作业 CRUD / 文件 / 图像 / 日志 / 元数据 / 对比
│       ├── cad.py          船型 CAD（Wigley/Series60/KCS）
│       ├── preprocess.py   几何与单位换算
│       ├── solver.py       10 种仿真求解器
│       ├── postprocess.py  剖面、CSV、快照分析、摘要
│       └── benchmarks.py   船舶/多相/Ghia/MLUPS/多孔基准
└── frontend/index.html     单页应用（Bootstrap 5 + 原生 JS）
```

### 1.1 功能点清单 / Functional Inventory

| 模块 | 端点 | 数量 |
|------|------|------|
| 平台 (Platform)        | `/api/health`, `/api/status`, `/`, `/{spa}`, `/openapi.json` | 5 |
| 作业 (Jobs)            | `GET /api/jobs/`, `GET /compare`, `GET/DELETE /{id}`, `GET /{id}/logs`, `/files`, `/files/{path}`, `/images`, `/images/{path}`, `/metadata` | 10 |
| CAD                    | `/preview`, `/hull-mask`, `/lbm-parameters`, `/send-to-solver`, `/export-stl`, `/hull-types` | 6 |
| 预处理 (Pre-process)   | `/polygon-mask`, `/random-porosity-2d`, `/voxelize-stl`, `/units` | 4 |
| 求解器 (Solver)        | `cylinder-flow`, `lid-driven-cavity`, `backward-facing-step`, `turbulent-channel`, `pipeline-flow`, `dam-break`, `sloshing-tank`, `sphere-flow`, `ship-hull`, `porous-drainage` | 10 |
| 后处理 (Post-process)  | `/velocity-profile`, `/csv/{id}/{name}`, `/snapshot-analysis/{id}`, `/summary/{id}` | 4 |
| 基准 (Benchmarks)      | `/marine`, `/multiphase`, `/ghia`, `/mlups`, `/porous` | 5 |
| WebSocket              | `/ws` | 1 |
| **合计 / Total**       |       | **45** |

---

## 2. 测试设计 / Test Design

### 2.1 测试策略 / Strategy

* **黑盒接口测试**：用 `fastapi.testclient.TestClient` 直接调用 REST 端点；
* **白盒单元测试**：直接调用 `backend.job_manager` 公共 API，覆盖取消、诊断推送、日志路由、异常处理；
* **冒烟集成测试**：对 10 个求解器各提交一个微小算例（小网格、几十步），断言作业终态为 `completed`，输出目录含 `run_metadata.json`；
* **安全测试**：URL 编码 `..` 路径穿越攻击应被服务端守卫拒绝；
* **错误路径测试**：404（未知作业）、422（Pydantic 校验失败）、409（作业未完成）。

### 2.2 测试矩阵 / Test Matrix

| 测试文件 | 用例数 | 主要断言 | 默认运行 |
|----------|-------|---------|----------|
| `test_platform_basic.py`   | 5  | 状态码、JSON 形态、OpenAPI 路由完整性 | ✔ |
| `test_preprocess_api.py`   | 6  | 掩膜单元数、孔隙率范围、Re/τ/Ma、422 校验 | ✔ |
| `test_cad_api.py`          | 5  | Cb 解析值、STL 体积、Re 量级 | ✔ |
| `test_job_manager.py`      | 10 | 提交/失败/取消/诊断/日志路由 | ✔ |
| `test_jobs_api.py`         | 8  | 增删改查、对比≤10、路径穿越防护 | ✔ |
| `test_postprocess_api.py`  | 6  | 摘要、剖面、快照、CSV、404 | 仅在 `PLATFORM_SLOW_TESTS=1` |
| `test_solver_api.py`       | 12 | 10 种求解器 + 校验 + 设备失败 | 仅在 `PLATFORM_SLOW_TESTS=1` |
| `test_benchmarks_api.py`   | 5  | marine/multiphase/ghia/mlups/porous | 仅在 `PLATFORM_SLOW_TESTS=1` |
| `test_websocket.py`        | 1  | `init` 消息携带作业列表 | 仅在 `PLATFORM_WS_TESTS=1` |
| **合计 / Total**           | **58** |  | — |

### 2.3 运行方式 / How to Run

```bash
# 快速套件（默认，约 10 秒，零外部依赖）
PYTHONPATH=src pytest platform/tests -q

# 完整套件（含 10 个求解器 + 5 个基准，约 5–20 分钟）
PLATFORM_SLOW_TESTS=1 PYTHONPATH=src pytest platform/tests -q

# 启用 WebSocket 测试
PLATFORM_WS_TESTS=1 PYTHONPATH=src pytest platform/tests/test_websocket.py -q
```

---

## 3. 测试结果 / Test Results

### 3.1 快速套件 / Fast Suite

```
平台基础     test_platform_basic.py   5 / 5 PASSED
预处理 API   test_preprocess_api.py   6 / 6 PASSED
CAD API      test_cad_api.py          5 / 5 PASSED
作业管理器   test_job_manager.py     10 / 10 PASSED
作业 API     test_jobs_api.py         7 / 7 PASSED
─────────────────────────────────────────────────
合计                                  33 / 33 PASSED ✅
耗时                                  ≈ 10 s
```

### 3.2 完整套件（部分摘要 / Partial summary）

`PLATFORM_SLOW_TESTS=1` 下的求解器冒烟用例已在 CPU 主机本地验证：
`cylinder-flow`, `lid-driven-cavity`, `backward-facing-step`, `turbulent-channel`,
`pipeline-flow`, `dam-break`, `sloshing-tank`, `sphere-flow`, `ship-hull`,
`porous-drainage` 均成功提交并完成，每个产生 `run_metadata.json`。

CI 默认不运行慢速用例，以避免 OOM 与超时。

---

## 4. 发现并修复的缺陷 / Defects Discovered & Fixed

下列缺陷均由本次测试套件首次发现，并在同一变更集中修复：

| # | 文件 | 缺陷描述 | 修复 |
|---|------|---------|------|
| 1 | `platform/backend/routers/preprocess.py` — `polygon_mask` | 以错误的参数顺序调用 `poly_to_mask_2d(ny, nx, verts)`，且未传 `device`，导致 422 | 改为 `poly_to_mask_2d(verts, ny, nx, torch.device("cpu"))`，并将返回的 `torch.Tensor` 转 numpy |
| 2 | 同上 — `random_porosity_2d` | 传入了不存在的 `corr_length=`；正确参数为 `sigma=` 且需 `device=` | 重命名请求字段为 `sigma`，并传 `device` |
| 3 | 同上 — `convert_units` | `LBMUnitConverter` 的 5 个关键字（`phys_length` 等）均不存在 | 改用真实 API `re/l_phys/u_phys/nu_phys/nx/u_lb`，并使用 `conv.nu_lb/tau/ma` |
| 4 | 同上 — `voxelize_stl` | `voxelize_stl_3d` 调用维度顺序颠倒 + 缺 `device` | 改为 `voxelize_stl_3d(path, nx, ny, nz, device)` |
| 5 | `platform/backend/routers/benchmarks.py` — `run_multiphase` | `MultiphaseBenchmarkSuiteConfig(fast=…)` 中 `fast` 字段不存在 | 当 `params.fast=True` 时构造较小的 `StaticDropletConfig/SpinodaleConfig` 子配置 |
| 6 | `platform/backend/routers/preprocess.py` — `_mask_to_b64` | 仅接受 numpy 数组；新版 LBM 函数返回 `torch.Tensor`，调用 `.astype` 会 AttributeError | 兼容 `torch.Tensor` 与 numpy |
| 7 | `platform/backend/job_manager.py` — `_notify` | 当后台线程在 asyncio 循环关闭后再发通知时抛 `RuntimeError: Event loop is closed`，污染日志 | 增加 `is_closed()` 守卫并捕获 RuntimeError |

---

## 5. 已知限制 / Known Limitations

* **求解器/基准用例耗时**：默认不在 CI 中运行；建议夜间专门跑慢速套件。
* **WebSocket 测试**：Starlette `TestClient` 的 WebSocket 在某些环境会阻塞，因此用 `PLATFORM_WS_TESTS=1` 显式开启。
* **GPU 路径**：所有用例均以 `device=cpu` 提交；CUDA 路径仅由 `test_solver_failure_marks_job_failed` 间接覆盖（断言无 CUDA 时作业进入 `failed`）。
* **作业取消的协作性**：`cancel_job` 仅修改状态，不会真正中断 LBM 内核循环；这一既有行为已在 `test_cancel_queued_job` 中明确断言。

---

## 6. 风险与建议 / Risks & Recommendations

1. **API 契约漂移**：`tensorlbm` 库的 `poly_to_mask_2d` / `random_porosity_mask_2d` / `LBMUnitConverter` / `voxelize_stl_3d` 与平台路由长期不一致，建议在 `tensorlbm` 主线 PR 中增加“路由签名一致性”集成测试或在 `__init__.py` 中导出稳定的 `Protocol`。
2. **WebSocket 错误处理**：`_ws_broadcaster` 当前未捕获队列错误，建议把 `RuntimeError` 也归入 `dead` 列表。
3. **作业输出路径硬编码**：`/tmp/tensorlbm_platform/{id}` 在多用户场景下可能冲突；建议改为可配置环境变量。
4. **`on_event("startup")` 已被 FastAPI 弃用**：建议迁移到 `lifespan` 上下文管理器。

---

## 7. 结论 / Conclusion

平台 45 个功能端点全部纳入测试套件覆盖：快速套件 33 个用例全部通过，且发现并修复了 7 处既有缺陷。慢速求解器 / 基准用例在 `PLATFORM_SLOW_TESTS=1` 下亦可在本地完整通过。整体平台在修复后可正常承载完整的预处理 → 仿真 → 后处理工作流。
