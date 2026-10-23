# TensorLBM AI4S 平台现状盘点（2026-08-20）

- 盘点对象：`origin/main @ f7765ff`（2026-08-20 07:53, PR #181 mass-gather OOM fix）
- 方式：纯只读（`git ls-tree` / `git show` / `git grep`），未动 5090 主工作区
- 规模：全仓 1448 commits；`src/tensorlbm/` 404 个 .py；tests 379 个 .py；examples 140 个；docs 72 个 md + ~150 个 evidence JSON
- 标注约定：【确】= 读到文件确认存在；【推】= 从命名/文档推断，未逐行验证

---

## 1. 模块清单与分类（src/tensorlbm/ 全量 404 个 .py）

### 1.1 子包结构（14 个子包，共 153 个 .py）【确】

| 子包 | .py 数 | 内容 |
|---|---|---|
| `ai/` | 22 | FNO2d、自监督 flow transformer、MLP 涡粘、DNS→SQLite→训练→嵌入闭环管线、SUBOFF 3D 编解码代理（train/finetune/infer/DDP）、SQLite 持久层 |
| `apps/` | 12 | **AI4S 应用框架 SDK**（base.py 五方法抽象 + run() 全栈编排）+ 9 个应用 + hpc.py |
| `ml/` | 9 | training_job（状态机）、serving（ModelRegistry/InferenceService/ONNX 导出）、torch 训练/物化/holdout 评估 |
| `data/` | 6 | catalog（SQLite 资产/血缘/质量）、field_r2 + field_dataset_r2（防泄漏数据集契约）、quality |
| `models/` | 3 | 模型契约 + torch 执行 |
| `backends/` | 7 | torch（默认）/ paddle / mindspore / triton 后端 + contracts + torch_5op_step |
| `core/` | 6 | lattice / collision / d3q19_stencil / d3q27_stencil / turbulence（新归组命名空间） |
| `phasefield/` | 10 | CH 验证、free_energy、演化适配、算子、相库存通量 |
| `octree_boundary/` | 9 | 八叉树 BFL/力/几何/LES/qfield/分片/步进/拓扑 |
| `performance/` | 3 | 性能契约 + d3q19_mrt 热路径 |
| `runtime/` | 2 | evidence |
| `cad/` `lattice_models/` `physics/` | 各 1 | 空壳 `__init__`（预留命名空间）【确】 |

### 1.2 顶层模块分类计数（按文件名归组，**类间有重叠**，总计>404 属正常）【确：基于全量文件名】

| 类别 | 数量 | 代表模块 |
|---|---|---|
| 船舶/潜艇/螺旋桨/海工 | 56 | suboff_*（40+，含 campaign/lifecycle/segmented/distributed）、ship_cad3d、propeller_ibm、hull_free_surface_v2、sloshing_tank、wave_body_rao |
| 多相/相场/自由表面/空化 | 47 | multiphase3d_d3q27、free_surface_lbm_27、cavitation、allen_cahn_lbm、full_wet、dam_break_3d |
| AI/ML/应用/数据平台包 | 53+4 | 上述四子包 + neural_wall_law、adjoint、pod、doe |
| 湍流/壁面模型 | 35 | turbulence、rans_ke（含 KOmegaSSTSolver，Menter 1994，~200 行【确】）、ddes、spalding_wall_model、synthetic_inflow |
| 边界条件/BFL | 35 | boundaries3d、bfl_d3q19_vec、interpolated_bc_suboff_d3q27、sponge_bc、wave_bc、external_open_boundary |
| 契约/证据/战役/平台治理 | 35 | *_capability_contract（10 个）、evidence_io、suboff_campaign_lifecycle、regression_gate、benchmark_observability |
| AMR/八叉树/嵌套网格 | 33 | static_block_amr、fixed_nested_*（4）、adaptive_refinement、amr_checkpoint |
| 外流基准案例 | 28 | cylinder_flow、sphere_flow、d3q27_sphere_flow、airfoil_benchmark、backward_facing_step、wind_comfort |
| 格子/核心步进 | 20 | d2q9、d3q19、d3q27、planar_d3q19、lbm_step_correct、fused_step |
| 分布式/性能/Triton | 21 | multi_gpu（MultiGPUSolver2D/3D【确】）、triton_fused_distributed、triton_suboff_step_distributed、compile_utils、cuda_memory_budget |
| DG/棱柱混合 | 11 | dg_lbm、dg_curv、spherical_dg、prism_solver、fdlbm_prism |
| 碰撞算子 | 15 | advanced_collision、cascaded_collision、entropic_kbc、cumulant_smag、chunked_collision、collision_d3q19_advanced |
| 力/阻力/积分 | 15 | drag_pressure、momentum_exchange、control_volume_force、force_methods、surface_integrals |
| IO/后处理/可视化 | 10 | io（VTK/HDF5/VTS/XDMF）、cgns_export、isosurface、streamlines、animation_export、particle_tracker |
| FSI/六自由度/滑移网格 | 10 | fsi、sixdof、rigid_body_6dof、sliding_mesh |
| 几何/CAD/STL | 9 | stl_geometry、preprocess_geo、suboff_cad、marine_geometry |
| 热/结冰 | 6 | thermal3d、conjugate_ht、aircraft_icing（2a+2b+glaze 三阶段均在【确】）、thermal_radiation |
| 声学 | 3 | acoustics、acoustic_beamforming |

wxsc 近期导出确认：`phasefield/`、`free_surface_*`、`cavitation.py`、`cascaded_collision.py`（D3Q19/27）、`entropic_kbc.py` 均已从顶层 `tensorlbm` 导出（commit da550e5、eac50cb，2026-08-19）【确】。

### 1.3 顶层非 src 目录【确】

- `app/`：FastAPI 平台后端（23+ router：simulations/solver/jobs/benchmarks/cad/preprocess/postprocess/**data_catalog/apps/ai_suboff/ai_transformer/agent/orchestration/ai_governance/xflow_streaming**…）+ 旧 HTML/JS 前端 + 50 个 API 测试；`hpc_scheduler.py`（真实 sbatch/qsub，由 `TENSORLBM_HPC_MODE` 环境变量切换）
- `frontend-vue/`：Vue3+TS 新前端（39 文件），views 分 ai4s（Apps/AppRun/AppLineage）、data（DataCatalog）、production（Dashboard/Preprocess/Solve/Postprocess/Benchmarks/Cad）——2026-08-14 落地【确】
- `benchmarks/`：verified/（18 个 result.json 案例、13 族）+ pending/（18 个）+ matrix.md 功能矩阵
- `checkpoints/suboff/`：5 个预训练 .ckpt（各 ~6.5MB）【确】——仓库内唯一的预训练权重
- 根目录 **~150 个 worker/launcher/diagnose 脚本 + ~30 个 results_*/log_* 目录**（历史战役残留，未归档进 experiments/）
- `cases/suboff_hybrid`、`configs/`（4 个 gate json）、`teaching/`（12 个教学脚本）、`notebooks/`（仅 quickstart.ipynb 1 个）

---

## 2. examples/ 与 docs/ 全景

### 2.1 examples/（140 个 .py）【确：按文件名+README 分组】

| 组 | 数 | 一句话 |
|---|---|---|
| AI/ML | 3 | `ai_dns_case.py`（DNS→SQLite→训练→AI-LES 嵌入，nx=48、60 步、30 epoch 级【确】）；`ai_turbulence_pipeline.py`（同管线另一入口）；`ai_fno2d_demo.py`（FNO2d 代理 vs LBM，宣称 ~195x 加速，产物在 docs/benchmarks/ai_fno2d/【确】） |
| SUBOFF | 21 | 阻力验证/网格收敛/AMR/nested/static_amr/engineering resistance 等，多数走 `suboff_validation_runner` 管线【推】 |
| AMR/octree/nested 验证 | 22 | 球阻力逐级 shell 验证（l2/l3/l4）、嵌套多卡、八叉树多卡 |
| DG-LBM | 21 | 圆柱/球/SUBOFF 混合、壁面函数、MRT D3Q19/27、多卡 |
| 球绕流 | 34 | 阻力参考、边界敏感性、网格/域收敛 assess |
| 多相/相场/自由表面/入水 | 16 | 气泡上浮(RP 对比)、Stefan 冻结、gallium 相变、入水砰击、hull+FS |
| 声学 | 13 | FWH 远场、偶极子/四极子、涡声、腔体共振 |
| 经典验证 | 33 | *_assess.py 收敛评估族 + kovasznay 等 |
| 其他 | ~14 | 风舒适度、多孔排水、活塞、param_scan、wigley |

配套测试：`tests/test_example_cli.py` 仅覆盖 cylinder CLI + sphere D3Q27 两个 smoke——**140 个 example 中 ~139 个无直接 CLI 测试**【确】。

### 2.2 docs/（72 个 md，27 个近 30 天有改动）【确】

- **平台规划（新，2026-08）**：`plans/ai4s-integrated-platform-architecture.md`（三层架构 + 9 应用进度表，自称"95 个测试全过"）、`plans/data-management / model-training / model-serving-cleanroom-spec.md`、`plans/frontend-vue3-refactor.md`
- **核验证据链**：`evidence/` ~150 个 JSON（suboff-nested v1~v37 迭代、sphere/cylinder/flat-plate 收敛、collision-viscosity 审计……）——这是该仓库最独特的资产：每个物理结论有机器可读证据文件
- **回归报告**：REGRESSION_REPORT*.md（AMR/RANS/interp_bc/sliding_mesh/wall_multi_gpu/WAVE_BC）
- **能力契约文档**：advanced_collision / boundary / wall_function / turbulence / amr capability matrix 系列
- **用户文档**：software_manual（软件说明书）、platform_user_manual、suboff_platform_manual、observability、development_workflow
- **综述/对标**：lbm-open-source-survey-2026-07-02、mature-lbm-gap-analysis-2026-08-01、amr-interface-literature-audit
- 顶层另有：PLATFORM_ANALYSIS.md（平台自诊断，见 §6）、MATURE_CFD_BENCHMARK_RESEARCH.md、IBM_RESEARCH_SUMMARY.md、WATER_ENTRY_SLAMMING_SUMMARY.md 等

---

## 3. 测试覆盖地图（tests/ 379 + app/tests 50 = 429 个测试文件）【确】

| 域 | 测试数 | 备注 |
|---|---|---|
| SUBOFF | 43 | 含多卡不变量、战役生命周期、阻力合同 |
| 自由表面 | 33 | 拓扑事务/账本/审计极其细 |
| ML/apps/AI | 33 | tests/ml 9 + tests/data 4 + test_app_* 12 + ai/serving/training/data_catalog |
| 湍流/碰撞 | 35 | 含 entropic/cumulant/dynamic smagorinsky |
| AMR/nested/octree | 31 | |
| 壁面/壁函数 | 30 | |
| 边界/BFL | 21 | 含 far_field 标签回归（我方 #178） |
| 圆柱/球 | 20 | |
| 多相/相场 | 19 | |
| 船舶/螺旋桨 | 17 | |
| triton/分布式 | 12 | test_triton_fused 等（我方 #171-173 相关） |
| 声学 5、热/结冰/CHT 6、波浪/RAO 6、DG 5、IBM 4、FSI 4、多孔/非牛顿 4 | | |
| **cavitation** | **0** | 【确】零测试 |
| backends 2、core 4、performance 2、runtime 1 | | 新契约层测试 |

**零/弱测试域**：cavitation.py（0 测试）；acoustic_beamforming 仅在 test_beamforming；`neural_wall_law.py`、`adjoint.py`、`pod.py`、`doe.py`、`wind_comfort.py`、`particle_tracker.py`、`cgns_export.py`、`porous_media3d` 未见专属测试【确：grep tests/ 无匹配】。app/tests 50 个覆盖 API 合同/前端一致性但为合同级冒烟。

CI（.github/workflows/ci.yml）：ruff lint+format、mypy 核心门（4 文件）+ 全量 advisory、`scripts/check_platform_docs_consistency.py` 文档-实现一致性检查、pytest 全量带 coverage【确】。

---

## 4. README 宣传 vs 实际【逐条对照】

| README 声称 | 状态 | 证据 |
|---|---|---|
| D2Q9/D3Q19/D3Q27 格子 | **完备** | d2q9/d3q19/d3q27.py + tests |
| BGK/MRT/TRT/RLBM/Cumulant 碰撞 | **完备** | + cascaded/entropic KBC（08-19 新导出） |
| AMR 5 级 Filippov–Hänel | **完备（工程量大）** | amr_* 33 模块 + 31 测试 + octree |
| DG-LBM 混合（P1-Lobatto/SSP-RK3） | **完备** | dg_* 11 模块 + 21 examples |
| LES 四模型（Smag/Dyn/WALE/Vreman，3 种格子） | **完备** | 测试覆盖 |
| RANS k-ε + k-ω SST | **完备** | KOmegaSSTSolver 真实实现（rans_ke.py:716，Menter 1994）【确】 |
| 幂律非牛顿 | **完备（新）** | powerlaw.py + 测试（08-19） |
| 多相（Shan-Chen/CG/自由能相场） | **完备** | 但 wxsc benchmark 显示 VOF/自由表面公用模块在外部验证中 RT+dam break **未通过**（commit 649e147） |
| IBM 2D/3D | **完备** | |
| 热双分布 + CHT | **完备** | |
| FWH 气动声学 | **完备** | 13 声学 examples |
| AI 湍流（MLP/FNO2d/Transformer/DNS-LES 管线/嵌入） | **部分**——存在且闭环，但规模 smoke 级（nx=48、40-60 步、30 epoch）【确】 | ai/pipeline.py |
| 多 GPU MultiGPUSolver2D/3D | **完备** | multi_gpu.py:252/342 + 7 个 multi_gpu 测试 |
| 多后端 paddle/mindspore | **部分**——后端+合同测试存在，非全部算子在异构后端验证【推】 | backends/ |
| HDF5/XDMF 后处理 | **部分**——io.save_hdf5 存在，但 src 内仅 general_sim.py 调用，**examples 0 调用**【确】 | git grep |
| Marine CAD（Wigley/S60/KCS/SUBOFF/KP-505） | **完备** | marine_geometry + CAD 测试 |
| 定量验证表（St、ω1、κ、Ghia、MMS…） | **部分**——与 benchmarks/verified 18 案例吻合，但 README 未收录 benchmark matrix 的"未通过"结论（VOF、Blasius、cavity 自然对流等 pending 18 例） | benchmarks/ |

README 未宣传但实际存在的重大能力：**AI4S 应用框架（9 应用+SDK）、数据目录/血缘、训练作业/模型服务、FastAPI 平台 + Vue3 前端、HPC 调度、Triton 融合内核（我方）**——README 的"CPU-first reproducible LBM"定位已明显落后于仓库实际形态。

---

## 5. 近期活跃度（origin/main）

### 5.1 近 30 个提交分类（2026-08-19~20）【确】

- **我方（LBM/Claude 账号 + chuxuesen merge）约 12 个**：icing Phase 3 glaze（b16e01f）、bench compile_route 全量接入（4c4d391、c893376）、mass-gather OOM 修复（da26a80、f7765ff）、远场标签修复（17600da、#178）、分布式包装 BC 修复（debe177、#177）、共性模块导出合并
- **wxsc 约 14 个**：benchmark matrix 战役——verified 新增 cylinder_re200（±0.9% Cd/St）、Sod 激波管 B31；记录 NOT verified：VOF RT/dam break、cavity 自然对流、cylinder_re40、DIT 衰减湍流、Blasius（C_f +76~135% 系统偏）、SUBOFF L=128 OOM——负结果入库是亮点
- **共性模块导出 2 个**：cascaded/entropic KBC、phasefield/FS/cavitation

### 5.2 90 天作者分布【确】

cxs503 923 / copilot-swe-agent[bot] 255 / chuxuesen（merge）194 / wxsc 66 / LBM 5 / Claude 2 / 昆仑新能AI平台 2 / TensorLBM Bench 1。总计 1448 commits。

### 5.3 近 30 天目录级 churn（文件改动次数）【确】

src 873 > tests 744 > docs 413 > benchmarks 352 > examples 337 > scripts 216 > experiments 77 > app 73 > frontend-vue 69 > teaching 28。
src/tensorlbm 最热：`__init__.py` 41、octree_boundary 34、wall_model 33、ai 28、sphere/cylinder_bfl_control_volume 19/17、suboff_nested_convergence 17、static_block_amr 17、apps 13、ml 12。

平台层（app/、frontend-vue/、ml/、apps/）集中落在 2026-08-13~15（cleanroom 三件套 + 应用框架 + Vue 前端同期），之后 30 天主战场转回物理内核与 benchmark matrix【确：first/last commit 日期】。

---

## 6. AI4S 闭环断点分析（核心交付）

四环 = **求解器 → 数据生成 → 训练 → 应用**。

### 现有基础【确】

| 环 | 现状 | 关键文件 |
|---|---|---|
| 求解器 | 极厚：404 模块、40+ SUBOFF 模块、AMR/DG/多相/Triton | src/tensorlbm/* |
| 数据生成 | **平台侧有目录无内容，AI 侧有三条小管线** | ① `ai/pipeline.py`：内存 smoke 级 DNS→快照（nx=48、data_steps≈40-60）② `ai/suboff_dataset.py`：读外部 .npy ③ `data/field_r2` + `field_dataset_r2`：防泄漏数据集契约（仅测试构造载荷） |
| 训练 | 厚：AI4SApplication SDK（produce/build/dataset/train/infer 五方法）+ run() 自动 catalog→training_job→serving→lineage；9 个应用（FNO/PINN/GNN/DDPM/逆问题/UQ/AI-LES/FlowTransformer/SUBOFF 代理）；ml/ 有 holdout 评估 | apps/base.py（261 行，全栈编排完整）、ml/* |
| 应用 | 双通道：① AI 模型嵌回 LBM（AI-LES 涡粘嵌入碰撞，真实闭环但小规模）② FastAPI 路由 ai_suboff/ai_transformer/apps + Vue3 ai4s 视图 + hpc.py（SLURM/PBS 提交 produce_data） | app/backend/routers/、apps/hpc.py |

### 断点 TOP3（附直接证据）

**断点 1（最严重）：求解器输出 → 训练数据集的物理断开。**
- `git grep save_hdf5 origin/main -- src/` 仅 `general_sim.py` 与 io.py 自身；**examples/ 下 0 个调用**【确】。标准案例（cylinder_flow/sphere_flow/SUBOFF runner）默认产出 PNG+CSV，不落 HDF5 场数据。
- `data/field_dataset_r2.py` + `ml/torch_dataset_materialize.py` 的 `FieldDataProductR2/BlobRef` 载荷**只有 tests/ 在构造**，src 内无任何求解器路径生产它【确：grep FieldDataProductR2 生产者仅 data/ml 模块自身+测试】。
- `ai/suboff_dataset.py` 硬编码 `"../../../../mnt/data3/xzx/suboff1"` 外部数据路径读 .npy【确】——数据来自他人离线生成，仓库内无生产脚本。
→ 结果：FieldDataCatalog 血缘图的第一环（field_product）在生产环境中是空的；SDK 的 `produce_data` 大多为**合成解析场**（PINN 用 Taylor-Green、inverse_problem 用解析 Couette-Poiseuille、DDPM 可选随机涡）或 `_run_les_smoke`（48² 网格 40 步）【确：apps 各文件 docstring 明言 "injectable ... testable without a real LBM run"】。

**断点 2：平台 API 与共性求解内核两张皮。**
- 仓库自诊断 `PLATFORM_ANALYSIS.md`（顶层）明言："9 个共性模块已验证（45 bug 修复、30+ benchmark）但平台 API 不调用它们！general_sim.py 有自己的 voxelize/BC/loop/force；solver.py 是 case-specific（20 个端点，每 case 独立参数/几何/力）"【确】。
- 后果：Web 平台跑的不是最高质量的内核链路（含我方 Triton/compile 优化与修复），benchmark matrix 验证结论无法直接映射到平台 API 结果。
- 该文档提出的 generic-run 融合方案是否已落地未验证——xflow_streaming/simulations 路由存在但与共性模块的接合度【推：部分落地，2026-08-15 后 app/ churn 仅 73，低于 src 873】。

**断点 3：训练产物 → 工程应用的反哺面窄 + 数据/权重约定缺失。**
- 全仓预训练权重仅 `checkpoints/suboff/*.ckpt` 5 个（~6.5MB each），无 FNO/GNN/DDPM 权重，无 ModelRegistry 指向的模型库目录约定【确】。
- 9 个应用中仅 ai_les（涡粘嵌入碰撞）真正回写求解器行为；FNO 超分、DDPM 生成、GNN、UQ 的 `infer` 输出没有任何工程模块（marine/icing/SUBOFF resistance）消费——应用层与领域应用模块（56 个 marine 类）之间无接口【确：apps/infer 均为独立推理，marine_* 不 import ai/apps】。
- `apps/hpc.py` 的 SLURM 提交是 sbatch 字符串包装 + lazy import，`TENSORLBM_HPC_MODE=none` 默认本地，无真实集群使用痕迹【确：代码 + 无相关 evidence】。

### 其他薄弱点（次级）

- cavitation.py 零测试；VOF/自由表面公用模块外部验证未通过（wxsc 08-19 记录）而 README 仍宣称完备多相。
- 根目录 ~150 个 worker 脚本 + ~30 个 results_* 目录未归档，新用户第一眼是"垃圾场"，与 AI4S 平台定位严重不符。
- examples 140 个只有 2 个有 CLI 冒烟测试。
- notebooks 仅 1 个，Colab 入口指向它，教学资产（teaching/ 12 脚本）未接入 README。

---

## 7. 平台最值得补的三件事（建议）

1. **打通"求解器 → FieldDataCatalog"数据落库管线**（对应断点 1）：给 5-6 个主力 runner（cylinder/sphere/SUBOFF/DG/wave）加 `--export-hdf5 + register-catalog` 出口，写一个 `solver_output → FieldDataProductR2` 适配器（io.save_hdf5 与 field_r2.BlobRef 已经各就各位，缺的只是 100 行胶水 + 1 个约定目录布局）。这是把 SDK 从"合成数据演示"变成真平台的第一步，也是血缘图的第一环。
2. **落实 PLATFORM_ANALYSIS.md 的 generic-run 融合**（对应断点 2）：重写 general_sim.py 走 9 个共性模块 + compile_utils/Triton 路由，平台 `POST /api/simulations/generic-run` 一个端点收敛 20 个 case-specific 端点。让"任意 STL → 体素化 → 通用 BC → 通用力 → HDF5 → catalog"成为默认路径（恰好也是断点 1 的数据来源）。
3. **建立模型资产层与一个"真数据"旗舰应用**（对应断点 3）：约定 `checkpoints/<family>/` + ModelRegistry 路径规范，把 FNO 超分或 SUBOFF 代理跑成一个消费 catalog 真实 LBM 数据、权重入库、被 ai_suboff 路由调用的端到端样板（含 holdout 评估报告），替代现在的 `_run_les_smoke` 演示档。

---

## 附：本盘点未覆盖/不确定项

- 未运行任何测试/代码（纯静态），“95 个 AI4S 测试全过”引自架构文档自述，未复核。
- paddle/mindspore 后端的算子覆盖深度、xflow_streaming 与共性模块的接合度、【推】标注项均为命名/文档级判断。
- benchmarks/pending 18 案例的 result.json 细节未逐一读取。
