# Lettuce 与 XLB → TensorLBM 可借鉴性分析

日期：2026-08-20 ｜ 作者：AO ｜ 性质：纯调研，未改任何代码、未占用 GPU（FluidX3D 报告 `<scratch>/fluidx3d_lessons_20260820.md` 的姊妹篇，体例保持一致）

调研方式：两个并行子调研分别对 lettucecfd/lettuce（master `121359d2`，2026-04-13 push；发布 tag 0.2.3；另核 distributed / precision_lattice 未合并分支）与 Autodesk/XLB（main HEAD `9470e54a8`，2026-05-29；PyPI 0.3.1）做逐文件源码阅读（本机 GitHub 直连被墙，走 `cdn.jsdelivr.net/gh/<owner>/<repo>@<branch>` 单文件直取 + GitHub API 文件树；注：XLB 仓库超 jsdelivr 50MB 树形列出上限，树走 API、文件走 CDN）；论文编号由子调研核验后，主笔又对 4 个 arXiv 编号（export.arxiv.org 标题页）与 1 个 Crossref DOI 独立复验。TensorLBM 侧对照取自 5090 `<repo>` 只读 ssh（src/tensorlbm 325 模块、tests 364 项、app/backend、docs/）。

---

## 0. 两个先说清楚的事实（一好一坏）

**好消息（法律绿灯，与 FluidX3D 相反）**：两家许可证都是宽松许可，**代码可以合法参考、改编、甚至直接移植**（注明出处即可）：
- **lettuce = MIT**（LICENSE 全文核读：Copyright (c) 2019, Andreas Kraemer）。允许使用/复制/修改/再发布/再许可/出售，唯一义务是在副本中保留版权与许可声明。仓库无额外 NOTICE。
- **XLB = Apache License 2.0**（LICENSE 全文核读：Copyright 2023 Autodesk Inc）。同样允许商用/修改/再发布，另附**专利授权**；义务为保留版权与许可证文本、分发时附 license，**修改过的文件须显著标注（state changes）**；XLB 仓库根目录无独立 NOTICE 文件（root 文件列表核验）。
- 对照：FluidX3D（非商用+禁军事+禁 AI 训练源码，一行不能搬）→ 这两家不存在该红线。本文 TOP 项凡标注"可抄实现"的，均指按上述署名义务引用。

**坏消息（任务前提三处纠错，均经核验）**：
1. **lettuce 主论文不是 Computers & Fluids，与 Forschungszentrum Jülich 无关，Mathis Bode 不是作者**。核验结果：Bedrunka, Foysi, Grave, Kraemer, *"Lettuce: PyTorch-based Lattice Boltzmann Framework"*，arXiv:2106.12929（标题页核验），正式版发表在 **ISC High Performance 2021 Digital（Springer LNCS）**，DOI 10.1007/978-3-030-90539-2_3；作者单位为锡根大学 / Bonn-Rhein-Sieg / Fraunhofer SCAI / 柏林自由大学。与 Computers & Fluids 真正相关的是其 2024 后续（神经碰撞算子人工体粘性，见 §5）。Bode 的 arXiv 记录全部为燃烧/湍流主题。
2. **XLB 论文不是 arXiv 2310.xxxx，也不是 WinterSim/ASC 会议**。核验结果：Ataei, Salehipour, *"XLB: A differentiable massively parallel lattice Boltzmann library in Python"*，**arXiv:2311.16080**（标题页核验；v1 2023-11-27 曾用副标题 "Distributed Multi-GPU … for Differentiable Scientific Machine Learning"，v2 2023-12-12 改现题）；期刊版 *Comput. Phys. Commun.* **300: 109187, 2024**，DOI 10.1016/j.cpc.2024.109187。对 WSC/SIGSIM-PADS 的定向检索无命中。
3. **XLB 里不存在"vmap 批量参数扫描"**（本报告最重要的意外发现）。全历史（2023-05-09 首提交 → 2026-05-29 main）逐文件核验：`jax.vmap` 仅用于 streaming 的**格子方向向量化**（`xlb/operator/stream/stream.py`：`vmap(_streaming_jax_i, in_axes=(0,0), out_axes=0)(f, c.T)`，轴是 Q 不是 batch）；examples、论文（2311.16080 全文）、docs 均无批量扫 Re/松弛因子/初始条件的示例（论文 §6.1 的 batching 是 Flax **数据批**，batch=20 over 时间步）。omega 与 f 虽均为 traced 参数、pull 方案下理论可 vmap，但 Grid 的 sharding mesh 不为 batch 维设计，官方从未演示。**我们的 AI4S 数据生成不能指望从 XLB 搬现成方案，须自建（见 TOP-5）。**

---

## 1. 两框架概况

| 维度 | Lettuce | XLB |
|---|---|---|
| 定位 | 可微 PyTorch LBM，教学/科研，卖点是 autograd 与 NN 耦合 | 可微大规模并行 LBM（Autodesk Research），面向工程风洞/可微 ROM |
| 后端 | PyTorch 张量原语（roll/einsum）为主路径 + **opt-in 运行时 CUDA C++ 代码生成**（`cuda_native/`，融合 collide+stream 单核，代价是弃 autograd） | **JAX + NVIDIA Warp + Neon 三后端** Operator 架构：同一算子树三种实现，`Operator.register_backend` 装饰器分发；2025-26 重心转 Warp/Neon（JAX 为 Φ-ROM 等内部依赖保留） |
| 布局 | `(Q, *dims)`，f[i] 逐方向张量 | Q-first `(Q, nx, ny[, nz])`（旧 v0.1 为 Q-last，两代无自动转置层） |
| streaming | `torch.roll` 逐方向 + `no_streaming_mask` 的 torch.where；**无 esoteric/AA** | pull 默认（jnp.roll + vmap over Q）+ push（2025-10 为 Φ-ROM 加）；Warp/Neon 后端单 kernel 全融合；无 AA/esoteric |
| 碰撞算子 | BGK / KBC（熵稳定）/ MRT（通用矩变换：D2Q9 Dellar、Lallemand、D3Q27 Hermite）/ TRT / Regularized / Smagorinsky / NoCollision | BGK / KBC / Smagorinsky-LES / Forced（包装器）；**JAX 版 Smagorinsky 2026-01 才补齐**——两家都比我们窄（无 CM/CUMULANT） |
| 多 GPU | **无**（未合并 `distributed` 分支的 `domdec.py` 骨架 communicate 为 `pass`；论文只承诺 in progress） | JAX：`shard_map` + `jax.lax.ppermute` 环形 halo，**x 一维域分解，只交换 `left/right_indices` 指向邻居的种群（各 1 层）**；多进程 = 外部 MPI 起进程 + `jax.distributed.initialize()`；另有 Neon 后端多 GPU（OCC 通信重叠） |
| 精度 | fp16/32/64 显式准入（Context assert），**测试只覆盖 64/32**；fp16 专项（f 中心化 PrecisionLattice）滞留未合并分支 | `PrecisionPolicy` **compute/store 五档**（FP64FP64 / FP64FP32 / FP64FP16 / FP32FP32 / FP32FP16），stepper 首尾 `cast_to_compute/cast_to_store`；无 bfloat16 |
| 单位换算 | `UnitConversion`：Re/Ma 双锚点，`τ = ν/cs² + ½` 派生，velocity/pressure/time/energy 成对 `*_to_pu/*_to_lu` | 仅 ~60 行 `UnitConvertor` + 各 example 手写 ω 换算——**弱项** |
| 测试/CI | 63 个测试函数；conftest 参数化矩阵 dtype×stencil×device×native，**碰撞算子 `get_subclasses` 自动发现进守恒测试**；CI = ubuntu py3.12 × {cpu, cu124-130} + macos-cpu + **每周一 cron** + CLI 集成作业 | sdist 只带 2 个测试文件（main 已扩）；CI 仅 CLA bot + ruff（**且只在 PR 到 major-refactoring 分支时触发**）+ mkdocs 部署——**无测试 CI** |
| 打包/文档 | pyproject（versioneer），`requires-python "==3.12.*"`；**不在 PyPI**（`lettuce` 名被 2016 年停更的 BDD 框架占用，`lettuce-cfd` 404）；Sphinx + readthedocs | setup.py（无 pyproject），0.3.1（2026-01-12）；warp-lang 为 import 期核心依赖；mkdocs-material **单页文档**（内容停在 0.2.0 时代） |
| 维护活跃度 | master 最后 push 2026-04-13（近半年仍动），但 **0.2.3（2021-03）后未再发版**，大规模重构滞留 master、弃用警告成堆；核心 3-4 人（Kraemer/Wilde/Bedrunka/Spelten） | 活跃：364 commits（2023-05 起），0.3.0（2025-11，IBM 大特性）、0.3.1（2026-01）、main 0.3.2 未发；4 名主力（Ataei/Salehipour/Hennigh（NVIDIA）/Meneghin）+ Naser 等 |
| 性能量级+口径 | 论文（TGV）：V100 双精度 2D 512² ≈ **140 MLUPS**、3D D3Q19 ≈ **83**；RTX 2070S 58→44；单精度快 60-75%（V100）/ ~180%（2070S）。口径 = `Simulation.__call__` 返回 MLUPS（**格点数/秒，不含 Q**） | 论文（mlus_3d.py，完整一步含 BC、丢弃首次 JIT、mean±std）：DGX A100 8 卡强扩展效率 ~90%；弱扩展 512×A100 4096³（>6.7e10 格）**220,332 MLUPS**（~430 MLUPS/GPU，64 节点效率降至 ~30% 通信瓶颈）；单卡 SOTA 走 Warp 后端（GH200，NVIDIA 博客）。README **无 MLUPS 表** |
| 许可 | **MIT**（2019 Andreas Kraemer） | **Apache-2.0**（2023 Autodesk，含专利授权） |

**它们没有而我们有的**（不必妄自菲薄的部分）：
- **碰撞算子**：lettuce 无 CM/CUMULANT，XLB 连 Smagorinsky 的 JAX 版都是 2026 年才补——我们的 BGK/CM/CUMULANT(LES)/MRT/KBC 全家桶仍是高 Re 外流（SUBOFF/KVLCC2）的实打实依赖。
- **性能档位**：lettuce 主路径（roll 循环）性能病根显著（这正是它做 cuda_native 代码生成自救的原因）；XLB JAX 路径 8×A100 1024³ = 11.4 GLUPS（CPC 论文 Table 1），对照我们 8×5090 周期场 69 GLUPS（`triton_fused_distributed.py:35`）——代际与口径不同不可严格横比（A100 vs 5090、其数含 JAX dispatch），但量级差说明 **Triton 单 pass 融合 + NCCL slab 路线不输"jit 整步"方案**，没有理由动摇现有内核架构。
- **都没有**：结冰线、自由表面、共轭传热、声学、壁面函数、AMR（XLB 的 multires 是嵌套 cuboid 细化，IPDPS 2024）、力采样生产链（SUBOFF/KVLCC2/DrivAer）、FastAPI/Vue 平台、数据 catalog+lineage（我们 `data/catalog.py` SQLite 资产注册/血缘/质量体系在两家中无对应物）、AI4S apps（FNO/Transformer/UQ/逆问题）。

---

## 2. 逐项对照表

标注：★=直接可借鉴；◇=需改造；✕=不适用（附原因）。可合法抄代码的注明许可证义务。

### 2.1 性能手法

| 手法 | 出处与做法 | TensorLBM 现状 | 可借鉴性 |
|---|---|---|---|
| torch.roll 逐方向 streaming（lettuce 主路径） | `lettuce/_simulation.py`：`for i in range(1,q): f[i]=torch.roll(f[i], shifts=e[i], dims=...)`，q=0 跳过；有 no_streaming_mask 时 torch.where | solver3d 已是"单次 gather（索引缓存）+ roll 回退"双路径，生产走 Triton 融合 | ✕ 性能倒退方向；roll 循环正是其要自救的病根。仅 `stream3d_roll` 保留同用途（省索引内存） |
| cuda_native 运行时 CUDA C++ 代码生成 | `Generator` 现场生成融合 collide+stream 扩展并编译，`Context(use_native=True)` **opt-in**；标准 pull 就地读邻居写自身；代价 = 丧失 autograd 与精度灵活性 | Triton 融合内核（碰撞+迁移单 pass）已达成同目标 | ★ 借**产品设计**而非实现：快慢双路径常青（eager 可微参考路径永远保留，fused 生产路径 opt-in）——与 compile_utils 教训同构 |
| jit 单元 = 整个时间步，算子树作编译期闭包 | XLB `nse_stepper.py`：`@partial(jit, static_argnums=(0,))`，`self`（含 stream/collision/BC 组合树）静态进闭包；签名 `(f_0, f_1, bc_mask, missing_mask, omega, timestep)` 纯函数式返回新缓冲；双缓冲交换在用户代码 | triton_fused 已单 pass，但组合方式是模块级硬拼 | ★ 纯函数式 step 签名 + 组合树闭包化，是 generic-run 落地"平台 API 与内核统一"的现成范式（并入 TOP-2） |
| **选择性方向 halo 交换** | XLB `distribute/distribute.py`：包住算子算完后 `result[right_indices,:1,...]` / `[left_indices,-1:,...]` 取出，`lax.ppermute` 环形传递，仅 **left/right_indices（指向邻居的 q 子集）**，各 1 层 | staging = `(19, ny, nx)` fp32 **全 q 面** ×4 块 | ★ TOP-3：与 FluidX3D `transfers` 常量（D3Q19=5/D3Q27=9）**双重独立交叉验证** |
| distribute() 的 BC 相位保守策略 | XLB：stepper 含 post-streaming BC 时**只分布式化 stream 算子**（整步包 shard_map 会把 BC 算错） | 我们的 distributed 已正确处理 BC 相位（异步两段式） | ★ 语义正确性对照案例：TOP-3 改造时按此核对相位 |
| compute/store 分离精度策略 | XLB `precision_policy.py` 五档枚举，stepper 首尾显式 cast | fp16 存储仅在周期内核验证（rel err ~2e-4），生产链 "fp32 only" | ★ TOP-4：配置面直接抄（Apache-2.0 须标注修改） |
| 基准脚本口径 | XLB `examples/performance/mlus_3d.py`：后端×stencil×collision×精度网格、`--measure_scalability`、`--repetitions` 出 mean±std、warmup 10 步丢弃首次 | benchmark_observability.py 记录元数据；GLUPS 多在 docstring/evidence | ★ 1 人日：加 repetitions+mean±std+bytes/cell（与 FluidX3D 报告 TOP-5 合并执行） |
| 主循环统一返回吞吐 | lettuce `Simulation.__call__` 末尾 `return num_steps*格点数/1e6/Δt`（MLUPS） | 各 worker 自行计时 | ★ 小甜点，并入 TOP-1 的 Reporter 协议 |
| Warp/Neon 后端、OOC tiles、IBM | XLB 2025-26 主推方向 | Triton+多后端（含 sdaa）已定 | ✕ 栈不同；OOC 对 n>1024 留观 |

### 2.2 功能面

| 功能 | 做法 | TensorLBM 现状 | 可借鉴性 |
|---|---|---|---|
| **案例基类 + 名字注册表** | lettuce `ExtFlow`：抽象 `initial_pu()/make_resolution()/make_units()`，可选覆写 `pre_/post_boundaries` 与 `initialize_pressure/initialize_fneq`；`_flow_by_name.py` dict 注册（'taylor3d_d3q27' 等 9 键）供 CLI | 仓库根 **82 个 `*_worker.py` + 42 个 launcher 脚本**，平台无法枚举案例 | ★ TOP-2（一半）：注册表是 generic-run 端点（`app/backend/routers/simulations.py:376` 已有）缺的"案例"抽象 |
| **BC registry（整数 id 掩码）** | XLB `boundary_condition_registry.py` 单例 dict，每个 BC **实例**拿唯一 int id（0 保留给固体）；应用时 `bc_mask == self.id` 布尔乘法（jit 安全，无数据依赖分支），Warp 路径 `wp.static` 编译期展开；`check_bc_overlaps` 查重叠 | boundaries3d/boundaries_d3q27 为显式函数集合，无注册机制 | ★ TOP-2（另一半）；registry 模式同一套可复用给模型资产（缺口③） |
| **missing 方向掩码"流一遍"生成法** | XLB `indices_boundary_masker.py`：把布尔场 pad 一圈后**直接过 stream 算子**，自动推导每个边界点缺哪些方向——零手写 q 表 | 手写 q 索引（triton_fused.py 头部记录过手抄符号错误教训） | ★ 妙招：掩码与 lattice 定义自动一致，移植成本 0.5 人日 |
| BC 相位声明 | lettuce `pre_boundaries/post_boundaries`（碰撞前/后）；XLB `ImplementationStep.COLLISION/STREAMING` | BB/力相位约定在生产链文档中，但非类型化 | ◇ 与 TOP-2 一并类型化 |
| 单位换算层 | lettuce `UnitConversion`：Re+Ma 双锚点，Ma 决定 U_lu=cs·Ma（压缩性可控）、τ 派生属性、成对 `*_to_pu/*_to_lu` + `convert_density_lu_to_pressure_pu` | `unit_converter.py::LBMUnitConverter` 单类 | ★ 对照补齐 Ma 锚点与成对接口（0.5-1 人日，小） |
| 高阶初始化 | lettuce `initialize_fneq`：六阶差分重构 f^(1) 非平衡部分（Krüger et al. 2017）+ `initialize_pressure`（Jacobi 压力泊松） | 冷启动 feq(rho=1, u=0) | ◇ 有望缩短 SUBOFF/外流达稳态时间（观察项：先在 n=128 案例测收敛步数缩短量再决定） |
| Reporter/回调系统 | lettuce：`Reporter` ABC 持 interval、`__call__(simulation)`；VTK（pyevtk .vti）/Observable（动能/涡量/能谱 FFT/质量）/Error（对解析解 L2）/Progress/Failure；`BreakableSimulation` 允许 reporter 改 `flow.i` **提前终止** | lbm_step 以 `boundary_kwargs/force_kwargs` 零散传参；各 worker 自写导出 | ★ TOP-1 |
| 网格体素化 masker 家族 | XLB `mesh_boundary_masker`：AABB / Ray / Winding / AABBClose 多算法可选 | stl_geometry.py 纯 numpy z-ray（CPU） | ◇ 与 FluidX3D TOP-4（GPU 体素化）同方向；XLB 的算法分层与"方法枚举"接口可参考（Apache-2.0） |
| 湍流槽道 DNS 对照 | XLB `turbulent_channel_3d.py` 对 chan180（Moser-Kim）DNS 数据验证 u+，参考数据 JSON 内嵌 repo | 有 validation 体系与 cross-validation matrix | ✕ 我们已有；"参考数据随 repo 内嵌 JSON"的做法可借 |
| multires（嵌套 cuboid 细化）/IBM/OOC | XLB 2025-11 起 | 结冰线拉格朗日自研；无网格细化计划；显存当前够 | ✕ 留观（OOC 在 n>1024 或单卡装不下时再评估） |

### 2.3 AI4S / 可微分

| 能力 | 做法 | TensorLBM 现状 | 可借鉴性 |
|---|---|---|---|
| **端到端 autograd（lettuce 的立身之本）** | 全库审计仅 2 处 `no_grad`（初始化泊松、数值差分）——collide（einsum）→ stream（roll/where）主链**天然可微**；无显存代价/checkpointing 讨论 | solver.py/solver3d.py 的 eager 路径（gather/roll）**同样可微但从未被声明/测试**；生产走 Triton（不可导）；adjoint.py 是**冻结场代理**（AD 只过目标函数，不过时间步进） | ★ 0.5-1 人日：把 eager 路径立为"可微参考路径"（文档+一个 `test_autograd.py`：Taylor-Green 上 loss 对 τ/初始场反传冒烟）——教学、研究、学习型碰撞的挂点全靠它 |
| 碰撞算子接受任意 nn.Module | lettuce `Collision` 是鸭子类型 ABC（任何 `__call__(flow)` 皆可）；姊妹 repo `lettuce-paper/02_neural_collision_model` 的 `LearnedMRT`（Linear(9,24)+ReLU+Linear(24,1)）端到端训练；正式成果 C&F 2024（神经体粘性）+ PRE 2025（不变网络碰撞算子） | ai/ 有训练设施、apps/ 有 FNO，但碰撞算子层无 NN 挂点 | ◇ 中期：在 TOP-2 的 collision registry 里给 nn.Module 留挂点（学习型 MRT/LES 修正的入口） |
| 可微示例范式 | XLB `examples/cfd/differentiable_lbm.py`（2026-05 合入）：`value_and_grad(loss)(f_0)` 对初始分布反传，50 步前向，SGD+物理范围 clip；OOC 版手写 **checkpointed adjoint**（checkpoint_frequency=16，前向存档/反向重放，手动播种伴随，mpi4py）——Warp 后端无 adjoint（wp.Tape 梯度为 0，有测试验证） | adjoint.py 冻结场代理 | ★ 示例可直接改编进 examples/（Apache-2.0 注明）；**手写分段 checkpoint 反传是多步可微的实用姿势**（比 remat 透明可控），做"可微 SUBOFF 小案例"时照此 |
| Solver-in-the-loop 修正器 | XLB 论文 §6.1：粗网格模拟 + NN 体积力，损失反传**穿过求解器**逼近细网格参考（引 Um et al. NeurIPS 2020）；明确用 gradient checkpointing + 时间步数据批 | apps/neural_operator_fno.py 是"求解器产数据→离线训练 FNO" | ★ 给 FNO app 加第二条路线：in-loop 修正器训练（我们 eager 路径可微是前提，恰好上一条已立） |
| **vmap 批量参数扫描** | **不存在**（§0-3，全历史 0 处；vmap 仅 for Q 方向） | doe.py 有采样计划（LHS/sobol/factorial/CCD），执行 = 进程级逐点（naca sweep 一卡一点 subprocess） | ✕ 照搬对象不存在；等价能力须自建 = TOP-5（eager 路径 batch 轴 + case 级多卡并行） |
| Φ-ROM（push-streaming 动因） | arXiv:2505.14595：可微求解器训练物理约束 ROM | ml/ 有 training_job+serving | ◇ 了解跟踪；其"为可微 ROM 加 push streaming"说明可微性是真实生产需求而非玩具 |
| JAX-LaB（第三方 XLB 扩展） | arXiv:2506.17713，多相/多物理 | 结冰线自研 | ✕ |

### 2.4 工程面

| 维度 | 做法 | TensorLBM 现状 | 可借鉴性 |
|---|---|---|---|
| 参数化测试矩阵 + 自动发现 | lettuce conftest：dtype{64,32}×stencil 全家×device{cpu,cuda}×native{on,off}；碰撞算子 `get_subclasses` 自动发现，新增算子自动进守恒测试集 | 364 测试文件 + 能力契约测试（已很强） | ★ 自动发现模式 1 人日：新碰撞/BC 模块自动进"守恒+宏观量不变"测试，防漏测 |
| CI 形态 | lettuce：ubuntu py3.12 × extras{cpu,cu124,cu126,cu128,cu130} + macos-cpu + **每周 cron 全量** + CLI 集成作业 | ci.yml：ruff/mypy/pytest+cov/包构建（push+PR） | ★ 两个增量：extras 按 CUDA 轮子版本分档（我们现依赖用户自装 torch）、周 cron 跑全量重回归 |
| PyPI 命名教训 | lettuce 因名字被 2016 年死项目占用而不在 PyPI，传播受损 | 有 publish.yml，TensorLBM 名字唯一 | ◇ 教训记档：发包前核名 |
| 反面教材：无测试 CI / 文档停滞 / lint 只在分支触发 | XLB CI 仅 CLA+lint（分支级）+mkdocs；docs 单页停在 0.2.0 | 我们领先 | ✕ 不学；保持 |
| 反面教材：未发版大重构滞留 master | lettuce 0.2.3（2021）→ 2026 仍未发 0.3，API 弃用警告成堆，用户踩坑（社区博客记录修弃用警告） | CHANGELOG+release.sh 快节奏 | ✕ 不学；重构快进快出 tag |
| 论文复现仓库 | lettucecfd/lettuce-paper（论文图表全复现脚本）+ README bibtex + Zenodo DOI | CITATION.cff 已有 | ★ 每篇自家论文配复现仓库（学术信誉资产，2-3 人日/篇） |

---

## 3. TOP 5 可落地项

> 与 FluidX3D 报告不同：本批两家均为宽松许可，以下凡"抄实现"处均可直接移植代码（MIT：保留版权与许可声明；Apache-2.0：另须在修改文件标注 changes）。TOP-3 与 FluidX3D 报告 TOP-3 同主题（两库交叉验证后合并执行）。

### TOP-1 Reporter/回调协议（源自 lettuce，MIT 可抄实现）
- **动机（缺口①求解器→数据断点）**：solver_export.py 正在补，但采样/导出/诊断缺统一挂点——现状 lbm_step 靠 `boundary_kwargs/force_kwargs` 零散传参，82 个 worker 各写各的导出。lettuce 的 `Reporter(interval) + __call__(simulation)` + `BreakableSimulation`（reporter 可改步计数提前终止 = 稳态检测的天然载体，我们已有 cp_measurement 但无统一协议）是成熟的最小设计。
- **落点**：新建 `src/tensorlbm/reporters.py`（Reporter 协议 + VTKReporter/FieldSampleReporter/ThroughputReporter/EarlyStopReporter）；`lbm_step.py` 与 triton 融合 step 主循环加一个 `_report(ctx)` 调用点；`data/solver_export.py` 从 FieldSampleReporter 派生（采样→FieldProduct→catalog 注册，一跳打通缺口①）。
- **工作量**：2-3 人日。
- **验证**：①现有 worker 产物回归（逐字节或 1e-6）；②新测试：interval 语义（steps=100, interval=25 → 恰 4 次回调）、EarlyStop 触发；③generic-run 端点输出 VTK/切片序列。
- **来源**：`lettuce/_simulation.py:311-323`（主循环与 MLUPS 返回）、`lettuce/reporters.py`、`lettuce/ext/_reporter/`（MIT，文件头保留其版权声明）。

### TOP-2 案例 + BC 双注册表（lettuce ExtFlow × XLB BC registry，MIT/Apache-2.0 可抄实现）
- **动机（缺口②两张皮 + 缺口③模型资产 registry）**：generic-run 端点已存在（`app/backend/routers/simulations.py:376`），但平台侧没有"案例"抽象可以枚举（82 worker 脚本无法被 API 表达）；BC 同样无注册机制。XLB 的实例级整数 id + 掩码乘法应用是 jit/Triton 兼容的注册表标准答案；registry 基建一次落成，第三处复用给模型资产（ml/ 的 model registry，缺口③）。
- **落点**：`src/tensorlbm/cases/`：ExtFlow 式基类（`make_resolution/make_units/initial_pu` + `pre_/post_boundaries`）+ 名字注册 dict；generic-run 的 Geometry/PhysicalConditions schema 直接映射到基类构造参数。`src/tensorlbm/boundaries/registry.py`：XLB 式注册（实例唯一 int id，0=固体/无 BC）+ `bc_mask==id` 应用；missing 方向掩码用 XLB 的"布尔场 stream 一遍"法生成（从 d3q19/d3q27 常量程序推导，杜绝手抄 q 表——历史教训）。
- **工作量**：4-6 人日（3 个标杆 case：cavity / SUBOFF-n128 / NACA 入域）。
- **验证**：①generic-run 跑 3 案例与对应 worker 结果 1e-6 对齐；②BC id 冲突/重叠检测测试（借 XLB `check_bc_overlaps`）；③run 结束自动在 FieldDataCatalog 注册资产（与 TOP-1 串联成缺口①闭环）。
- **来源**：`lettuce/ext/_flows/_ext_flow.py`、`lettuce/ext/_flows/_flow_by_name.py`（MIT）；`xlb/operator/boundary_condition/boundary_condition_registry.py`、`xlb/operator/boundary_masker/indices_boundary_masker.py`、`xlb/helper/check_boundary_overlaps.py`（Apache-2.0，标注修改）。

### TOP-3 选择性方向 halo 交换（XLB ppermute left/right_indices；与 FluidX3D 报告 TOP-3 合并为同一项，双重交叉验证）
- **动机（缺口⑥）**：现状 `_alloc_halo_staging` 为 `(19, ny, nx)` fp32 全 q 面 ×4 块；物理上 z-slab 每步只需穿越面的方向（D3Q19=5、D3Q27=9）。FluidX3D 的 `transfers` 常量与 XLB 的 `velocity_set.left/right_indices` 环形 ppermute 交换（各 1 层）在两个独立代码库给出同构答案——该改造的正确性风险已由别人趟过。
- **落点**：`triton_fused_distributed.py::_alloc_halo_staging/_start_halo_exchange/_finalize_halo`：staging 改 `(n_cross, ny, nx)`（+可选 fp16 传输），穿越方向索引表由 d3q19/d3q27 常量程序生成或用"stream 掩码"法推导；核对 XLB 的相位教训——**有 post-streaming BC 时只分布式化 stream 段**（我们现有两段式异步已兼容，按其案例复核一遍）。
- **工作量**：2-3 人日。
- **验证**：①多卡周期场 vs 单卡容差对齐；②nsys/torch profiler 8 卡 n=512 通信时间占比下降；③8 卡 GLUPS 提升 ≥5%。
- **来源**：`xlb/distribute/distribute.py`（110 行，Apache-2.0 语义参考；实现仍走我们 Triton+NCCL 栈）。

### TOP-4 compute/store 分离的精度策略枚举（源自 XLB，Apache-2.0 可抄实现）
- **动机（缺口⑤ fp16 未进生产链）**：FluidX3D 报告 TOP-1 已定"fp16 存储进生产链"的方向，但缺一个干净的配置面。XLB 的 `PrecisionPolicy`（FP64FP64/FP64FP32/FP64FP16/FP32FP32/FP32FP16 五档，"计算精度×存储精度"显式正交）+ stepper 首尾 `cast_to_compute/cast_to_store` 边界，正是该配置面的标准答案，也顺手统一 benchmark 报数口径。
- **落点**：新建 `src/tensorlbm/precision.py`（枚举+cast 边界约定）；`triton_fused_obstacle.py`/`triton_fused_distributed.py` 的 dtype 参数化接入；`benchmark_observability.py` 记录档位字段。
- **工作量**：1-2 人日（配置面）；内核 fp16 化本体沿用 FluidX3D TOP-1 估算（3-5 人日）。
- **验证**：精度回归矩阵按 lettuce conftest 的 `dtype_params` 模式（{FP32FP32, FP32FP16} × {周期, 障碍} × {D3Q19, D3Q27}）；SUBOFF n=256 的 C_t 对 fp32 基线偏差 ≤2%（ITTC ±10% 内）；GLUPS ≥1.6× 目标不变。
- **来源**：`xlb/precision_policy.py`（Apache-2.0，标注修改）；`lettuce/tests/conftest.py` 矩阵模式（MIT）。

### TOP-5 参数扫描执行器：doe.py → 批量运行 → catalog 一条链（自建，XLB 无现成方案）
- **动机（缺口①数据生成 + 修正对 XLB 的期待）**：AI4S 数据生成需要"扫描计划→执行→带 lineage 的数据集"管线。已核验 XLB 无 vmap 批扫（§0-3），故不抄方案、自建：执行层用 **case 级多卡并行**（每卡一 case 进程/流，与现有 launcher 的卡池思路一致但进程内调度、共享 JIT 内核），小网格教学案例可用 eager 路径的 **batch 轴**（`(B,Q,nz,ny,nx)` 对 eager solver 免费支持，这是我们栈里唯一接近 vmap 的原生能力）。
- **落点**：新建 `src/tensorlbm/scan_runner.py`：输入 `DoEPlan`（doe.py 已有 LHS/sobol/factorial/CCD 生成器）→ 每点实例化 TOP-2 注册表的 case → TOP-1 Reporter 采样 → FieldDatasetR2 + catalog lineage 注册；GPU 分配器沿用 82-worker 生态验证过的"卡→任务表"模式。
- **工作量**：5-8 人日。
- **验证**：①NACA α-sweep 用新链复现旧 worker 结果；②一次 32 点扫描产出带完整 lineage 的数据集且 catalog 可查；③apps/neural_operator_fno.py 直接消费该数据集完成一次训练（端到端闭环=缺口①验收）。
- **来源**：模式参考 XLB `examples/performance/mlus_3d.py` 的设备调度与 lettuce CLI（均宽松许可）；doe.py/catalog.py/FieldDatasetR2 为自有资产。

---

## 4. 不建议照搬的项与原因

1. **torch.roll 逐方向 streaming 本体**（lettuce 主路径）：我们 Triton 融合已到 8.6 GLUPS/单 5090（`triton_fused.py:16`），roll 循环是 lettuce 自己都要用 cuda_native 救的病根；`stream3d_roll` 仅作内存回退保留，不做生产。
2. **cuda_native 运行时 CUDA C++ 代码生成**：目标（融合 collide+stream）Triton 已达成，再引入"NVRTC/扩展编译"只会加重 compile_utils 已记录的冷编译问题；只借"opt-in 快慢双路径"的产品语义。
3. **迁移 JAX / shard_map 本体**：后端栈不换（PyTorch+Triton+NCCL，还要兼容 sdaa）；只借 ppermute 选择性方向交换的语义（TOP-3）。
4. **指望 vmap 批量参数扫描**：XLB 全历史不存在（§0-3）；JAX 中理论可行但 Grid sharding 不为 batch 维设计。任何按"XLB 有现成方案"排的计划都要改写成 TOP-5 的自建路线。
5. **XLB 的工程反面教材**：无测试 CI、lint 仅分支触发、单页文档停在 0.2.0、setup.py 无 pyproject、warp-lang 作 import 期硬依赖——均不学。
6. **lettuce 的发版纪律**：0.2.3（2021）后重构滞留 master 数年未发版、弃用警告成堆——我们 CHANGELOG+release.sh 的快节奏是对的。
7. **lettuce 案例命名/CLI 全套照搬**：已有 82 worker 生态，注册表做增量吸纳（TOP-2 的 3 个标杆先行），不推倒重来。
8. **XLB OOC/mpi4py 离核、IBM、multires**：当前显存与需求之外（n=1024 8×5090 已跑通、结冰线自研、无嵌套细化计划）；OOC 在单案例超显存时再评估。
9. **lettuce 未合并的 PrecisionLattice fp16 中心化分支**：与 FP32FP16 存储档（TOP-4）路线重叠，且该分支未合并未经验证；f 的动态范围问题在我们的"算术恒 FP32"方案下已规避大半。

---

## 5. 参考来源

- **lettuce 仓库**（master `121359d2`、tag 0.2.3、分支 distributed/precision_lattice，经 CDN/API 逐文件核读）：`LICENSE`（MIT）、`lettuce/_simulation.py`（streaming/主循环/MLUPS/Collision ABC）、`_context.py`（三精度 assert）、`_unit.py`（UnitConversion）、`_flow.py`（Boundary ABC/initialize_fneq/pressure_poisson）、`ext/_collision/`（7 算子）、`ext/_boundary/`（5 BC）、`ext/_flows/`（8 案例 + `_flow_by_name.py`）、`ext/_reporter/`、`cuda_native/_default_code_gen.py`（StreamingStrategy 位掩码与 pull 代码生成）、`tests/conftest.py`、`.github/workflows/CI.yml`、`pyproject.toml`、`native_cuda_synopsis.md`；姊妹仓库 `lettucecfd/lettuce-paper/02_neural_collision_model`（LearnedMRT）。
- **XLB 仓库**（main HEAD `9470e54a8` 2026-05-29、PyPI 0.3.1 2026-01-12，经 CDN/API 逐文件核读）：`LICENSE`（Apache-2.0，无 NOTICE）、`xlb/operator/stepper/nse_stepper.py`（jit 整步/纯函数式签名）、`operator/operator.py`（register_backend 分发）、`distribute/distribute.py`（ppermute 选择性 halo）、`operator/boundary_condition/boundary_condition_registry.py` + `indices_boundary_masker.py`（registry 与"流掩码"）、`operator/collision/`、`operator/stream/stream.py`（vmap over Q）、`precision_policy.py`、`utils/utils.py`（UnitConvertor）、`examples/cfd/differentiable_lbm.py`、`examples/out_of_core/autodiff_lbm.py`（手写 checkpointed adjoint）、`examples/performance/mlups_3d.py`、`setup.py`、`.github/workflows/`。
- **论文（全部经 export.arxiv.org 标题页或 Crossref 核验，未沿用任务提示中未经核实的编号）**：
  1. Bedrunka, Foysi, Grave, Kraemer, *Lettuce: PyTorch-based Lattice Boltzmann Framework*, ISC High Performance 2021 Digital (Springer LNCS), DOI 10.1007/978-3-030-90539-2_3; arXiv:2106.12929。
  2. Horstmann, Bedrunka, Foysi, *Lattice Boltzmann method with artificial bulk viscosity using a neural collision operator*, Computers & Fluids 272:106191, 2024, DOI 10.1016/j.compfluid.2024.106191。
  3. Bedrunka, Horstmann, Picard, Reith, Foysi, *Machine-learning-enhanced collision operator for the lattice Boltzmann method based on invariant networks*, Phys. Rev. E 112(5), 2025-11-12, DOI 10.1103/rbf2-p8tf（APS 新式 DOI，Crossref 核验，CC-BY 4.0）。
  4. Ataei, Salehipour, *XLB: A differentiable massively parallel lattice Boltzmann library in Python*, arXiv:2311.16080；期刊版 Comput. Phys. Commun. 300:109187, 2024, DOI 10.1016/j.cpc.2024.109187。
  5. Mahmoud, Salehipour, Meneghin, *Optimized GPU Implementation of Grid Refinement in Lattice Boltzmann Method*, IPDPS 2024, pp. 398-407。
  6. Meneghin, Mahmoud, Jayaraman, Morris, *Neon: A Multi-GPU Programming Model for Grid-based Computations*, IPDPS 2022, pp. 817-827。
  7. Dashtbayaz, Salehipour, Butscher, Morris, *Physics-informed Reduced Order Modeling of Time-dependent PDEs via Differentiable Solvers*, arXiv:2505.14595（XLB push-streaming 的动因）。
  8. Um et al., *Solver-in-the-loop: learning from differentiable physics…*, NeurIPS 2020（XLB 论文 §6.1 所引修正器范式）。
  9. Pradhan et al., *JAX-LaB: A High-Performance, Differentiable, Lattice Boltzmann Library…*, arXiv:2506.17713。
- **许可证**：lettuce LICENSE（MIT, 2019 Andreas Kraemer）与 XLB LICENSE（Apache-2.0, 2023 Autodesk）均全文核读；XLB 仓库无 NOTICE 文件。
- **TensorLBM 5090 只读对照**：`<repo>/src/tensorlbm/`（325 模块：solver3d.py 双 streaming 路径、lbm_step.py kwargs 回调、doe.py、unit_converter.py、data/{catalog,field_dataset_r2}.py、ml/、apps/、adjoint.py 冻结场代理、triton_fused.py:16 与 triton_fused_distributed.py:35 的 GLUPS 口径、compile_utils.py）、`app/backend/routers/simulations.py`（generic-run 端点）、`.github/workflows/ci.yml`、`pyproject.toml`、仓库根 82 `*_worker.py` + 42 launcher 计数、`docs/lbm-open-source-survey-2026-07-02.md`（其中无 lettuce/XLB 条目，本报告为首次覆盖）。
- **性能口径备注**：lettuce MLUPS=格点数/秒（不含 Q），XLB=完整一步（collision+streaming+BC）含 JIT 丢弃与 mean±std，我们 GLUPS 同为格点更新口径；XLB 8×A100 1024³=11.4 GLUPS vs 我们 8×5090 周期 69 GLUPS / SUBOFF n=1024 17.69 GLUPS——代际（A100 vs 5090）与负载（含 BC/障碍与否）不同，仅供量级感，不作横比结论。

（报告完。本地：/root/lettuce_xlb_lessons_20260820.md；同步：5090 <scratch>/lettuce_xlb_lessons_20260820.md）
