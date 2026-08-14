# TensorLBM 前端 Vue3 系统化重构方案

> 方向 B：从原生 JS + Bootstrap 迁移到 Vue3 + Vite，原有前端集成到「数据生产模块」，
> 补齐 AI4S 应用模块，系统化重构。

## 1. 重构目标

把 3658 行单文件 HTML + 18 个全局 JS 脚本，重构为组件化的 Vue3 SPA，分为四大模块：

```
数据生产模块（集成原有 CFD 工具）  ← 原有 16 个 panel 的 CFD 部分
AI4S 应用模块（9 个应用，新）      ← 对接 /api/apps（后端已有，前端缺）
数据管理模块（数据目录，新）       ← 对接 /api/data（后端已有，前端缺）
项目/作业/报告模块（原有）          ← 集成原有功能
```

## 2. 技术栈（对齐 KIEAP，均为 MIT 开源）

| 层 | 选型 | 说明 |
|----|------|------|
| 框架 | Vue 3.4 | Composition API + `<script setup>` |
| 构建 | Vite 5 | 快速开发 + HMR |
| 语言 | TypeScript 5 | 类型安全 |
| UI | Element Plus 2.5 | 组件库（对齐 KIEAP）|
| 路由 | vue-router 4 | SPA 路由 |
| 状态 | pinia 2 | 状态管理 |
| HTTP | axios | API 调用 |
| 图表 | echarts 5 | 收敛曲线/流场图 |
| 3D | Three.js | 保留现有（几何可视化）|
| 编排 | @vue-flow/core | 工作流/管线编排图 |

**版权边界**：以上均为 MIT 开源库，可直接使用。但 KIEAP 的 Vue3 代码（组件/业务逻辑）是专有闭源，**只能借鉴架构思想，代码独立重写**（干净室原则，同后端一致）。

## 3. 目录结构

```
frontend-vue/
├── package.json / vite.config.ts / tsconfig.json
├── index.html
└── src/
    ├── main.ts              # 入口
    ├── App.vue              # 根组件（布局 + 侧边栏 + 路由出口）
    ├── router/index.ts      # 路由
    ├── stores/              # pinia（auth/ui/app 状态）
    ├── api/                 # axios 封装 + 各模块 API 客户端
    │   ├── request.ts       # axios 实例 + 拦截器
    │   ├── production.ts    # cad/solver/postprocess/benchmarks
    │   ├── apps.ts          # /api/apps（AI4S 应用）
    │   ├── data.ts          # /api/data（数据目录）
    │   └── projects.ts      # 项目/作业/报告
    ├── views/
    │   ├── production/      # 数据生产模块（集成原有）
    │   │   ├── DashboardView.vue
    │   │   ├── CadView.vue
    │   │   ├── PreprocessView.vue
    │   │   ├── SolveView.vue
    │   │   ├── PostprocessView.vue
    │   │   └── BenchmarksView.vue
    │   ├── ai4s/            # AI4S 应用模块（新）
    │   │   ├── AppsView.vue         # 应用列表（9 应用）
    │   │   ├── AppRunView.vue       # 运行应用（配置 + HPC/本地）
    │   │   └── AppLineageView.vue   # 血缘追溯
    │   ├── data/            # 数据管理模块（新）
    │   │   └── DataCatalogView.vue  # 数据目录（资产/质量/血缘）
    │   └── misc/            # 项目/报告/通知
    ├── components/          # 复用组件
    │   ├── JobRunner.vue    # 作业运行卡片（复用）
    │   ├── LineageGraph.vue # 血缘图（@vue-flow）
    │   ├── FieldViewer3D.vue# 3D 场视图（Three.js）
    │   └── MetricChart.vue  # 图表（echarts）
    └── i18n/                # 中英双语
```

## 4. 模块划分（功能映射）

### 4.1 数据生产模块（集成原有 16 panel 的 CFD 部分）
| 原 panel | Vue view | 后端 API |
|----------|----------|----------|
| dashboard | DashboardView | /api/jobs |
| cad | CadView | /api/cad |
| preprocess | PreprocessView | /api/preprocess |
| solve | SolveView | /api/solve |
| postprocess | PostprocessView | /api/postprocess |
| benchmarks | BenchmarksView | /api/benchmarks |
| cylinder/suboff | （并入 CadView/SolveView）| /api/cylinder-* /api/suboff |

### 4.2 AI4S 应用模块（新，对接后端 AI4S 框架）
| Vue view | 功能 | 后端 API |
|----------|------|----------|
| AppsView | 列出 9 应用（name/family/version）| GET /api/apps |
| AppRunView | 运行应用（本地/HPC 派发）| POST /api/apps/{name}/run |
| AppLineageView | 血缘追溯（data→dataset→job→model）| 应用结果 + catalog |

### 4.3 数据管理模块（新）
| Vue view | 功能 | 后端 API |
|----------|------|----------|
| DataCatalogView | 数据资产/质量/血缘 | GET /api/data/assets 等 |

### 4.4 项目/作业/报告（原有集成）
| 原 panel | Vue view | 后端 API |
|----------|----------|----------|
| projects | ProjectsView | /api/projects |
| reports | ReportsView | /api/reports |
| orchestration | OrchestrationView | /api/orchestration |
| agent | AgentView | /api/agent |

## 5. 分阶段实施（4 阶段）

### 阶段 1：脚手架 + 布局 + 路由（1 周）
- Vite + Vue3 + TS 项目初始化
- 布局（侧边栏 + 顶栏 + 路由出口）+ vue-router + pinia
- axios 封装 + API 客户端骨架
- i18n 基础设施

### 阶段 2：数据生产模块（集成原有，2 周）
- Dashboard/Cad/Preprocess/Solve/Postprocess/Benchmarks 六个 view
- 复用组件（JobRunner/FieldViewer3D/MetricChart）
- 迁移原有 JS 逻辑到 Vue 组件（干净室重写，不复用 KIEAP）

### 阶段 3：AI4S 应用 + 数据管理模块（新，1-2 周）
- AppsView（9 应用列表）+ AppRunView（运行）+ AppLineageView（血缘 @vue-flow）
- DataCatalogView（数据目录）
- 这是补后端 AI4S 框架的前端缺口

### 阶段 4：项目/作业/报告 + 打磨（1 周）
- Projects/Reports/Orchestration/Agent
- 中英双语完整化、样式统一、测试

## 6. 关键决策点（需确认）

1. **TypeScript vs JS**：建议 TS（类型安全，对齐 KIEAP）
2. **UI 库**：Element Plus（对齐 KIEAP，MIT）
3. **3D 可视化**：保留 Three.js（现有 OrbitControls + 几何）
4. **图表**：echarts（对齐 KIEAP，替代 Chart.js）
5. **新旧过渡**：新 Vue3 前端独立部署，旧前端保留到迁移完成再删

## 7. 实施策略

- **并行**：阶段 1 脚手架先做，阶段 2/3 可并行（数据生产模块迁移 + AI4S 应用模块开发互不依赖）
- **干净室**：所有 Vue 代码独立编写，只借鉴 KIEAP 的技术栈选择和架构分层思想
- **验证**：每个 view 完成后对接真实后端 API 验证
