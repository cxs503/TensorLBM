# TensorLBM AI4S 平台路线图

日期：2026-08-20 ｜ 状态基线：main @ 814b5a4（#182 solver→data 导出、#183 模型资产层已合入）

定位（承接 README #184 改写）：基于 PyTorch + Triton 的张量化 LBM 求解器 + AI4S 综合平台——求解器产数据、数据训模型、模型回注求解器与服务的完整闭环。

## 一、现状基线（已具备）

- **闭环骨架已通**（2026-08-20）：solver→HDF5→catalog 注册（#182 `data/solver_export.py`，试点数据集 27 产品质量分全 100）→ FNO 训练（loss 3.94e-3→8.12e-7）→ 模型入库（#183 `ml/model_registry.py`，生命周期+指标+serving 关联）→ 推理（rel-L2 0.048 vs 双线性 0.063）→ lineage 一次回溯全链
- **求解器线**：Triton 融合栈三修复+wrapper 四修复全验证（多卡 vs 单卡逐位）；SUBOFF 生产验证（8×5090 n=1024，60k 步无 NaN，C_t/C_F=1.20 vs ITTC，17.69 GLUPS）；15 个 verified 基准全 <3% 且统一走 compile 路由（#180）
- **结冰线**：拉格朗日霜冰 → 欧拉两相 → 明冰 Messinger 全温域（#179）
- **平台层**：FastAPI 23+ 路由（50 API 测试）、generic-run 端点走共性模块（#185）、Vue 前端
- **横向调研完成**：FluidX3D / lettuce / XLB 三份对标分析（见 `docs/surveys/`）

## 二、缺口地图

### A. 求解器深度（CFD 可信度上限）

| # | 缺口 | 判断依据 | 量级 |
|---|---|---|---|
| A1 | 壁面/障碍内核成本 −48% + kernel 内 Smag NaN | SUBOFF 生产 GLUPS 损失近半 | 修复中 |
| A2 | fp16 存储未进生产链 | 周期内核已验证 rel err ~2e-4；带宽受限 77% roofline，理论上限近 2× | 3-5 人日 |
| A3 | halo 全 q 面传输（19/32 面 fp32） | FluidX3D transfers 常量与 XLB ppermute 两库交叉验证：只需穿越方向（D3Q19=5） | 2-3 人日 |
| A4 | 几何输入仅解析/CPU 体素化 | 接不了真实车船 STL；GPU 体素化解锁 DrivAer/KVLCC2 | 5-8 人日 |
| A5 | 均匀网格：AMR 33 模块不在生产链 | 壁面分辨率是 C_D 误差主源（SUBOFF 标定结论） | 3-6 月 |
| A6 | 可微分 LBM 未声明 | eager 路径天然可微但从未测试；逆问题/学习型碰撞/solver-in-the-loop 的底座 | 先 0.5-1 人日立参考路径 |
| A7 | 长稳性：无 checkpoint 约定 | 多日生产 campaign 刚需 | 1-2 人日 |
| A8 | 物理面广而不深（空化/共轭传热/声学零生产验证） | 建议选 wall-model LES 做深（对标品自认的短板） | 选做 |

### B. 数据引擎（AI4S 核心差异化）

| # | 缺口 | 说明 |
|---|---|---|
| B1 | 试点 → 规模化生成 campaign | doe.py（LHS/sobol/factorial/CCD 已有）→ scan_runner 批量执行 → catalog lineage；case 级多卡并行 + 小网格 eager batch 轴。注意：XLB 无现成 vmap 批扫方案（全历史核验），须自建 |
| B2 | 数据卡片/版本化 | 数据集自动 datasheet（来源、参数窗、质量分、已知偏差） |
| B3 | 合成→物理反哺 | FNO rel-L2 0.048 已证明可行；缺 ML 壁面/亚格子嵌入主循环的工程接口（solver-in-the-loop 第二路线） |
| B4 | Foundation-model 接口 | 场 tokenizer、预训练-微调、跨 case 泛化评测——下一代卡位 |

### C. 模型资产与评测

| # | 缺口 | 说明 |
|---|---|---|
| C1 | registry 已有 → 缺模型卡片 + 模型 zoo | 可发布权重集 + 标准评测数字 |
| C2 | AI4S 评测基准 | leaderboard：代理精度/超分误差/阻力预测/推理吞吐，公开可复现脚本——AI4S 的 "MLUPS 口径" |
| C3 | 训练编排 | 单机 → 多机 DDP/超参搜索/实验跟踪 |

### D. 平台工程与生态

| # | 缺口 | 现状 |
|---|---|---|
| D1 | PyPI 发布/版本化文档/版本策略 | 世界级门槛，1-2 周 |
| D2 | CI 仅冒烟 2/140 examples；零测试域（空化/neural_wall_law/adjoint/POD/DOE）；multi_gpu 5 预存在失败 | 测试矩阵扩面 + GPU CI + 周cron 全量（lettuce 模式）+ 碰撞算子自动发现测试 |
| D3 | 教程断层 | user journey：5 分钟上手 → AI4S 全环 |
| D4 | hpc_scheduler 无真实集群使用痕迹；多节点（>8 卡）未证 | 现有超算资源即验证场 |
| D5 | 学术与社区：平台 paper/CITATION/贡献指南/公开 leaderboard | 领先需要第三方可验证 |

### E. 已领先项（守住并放大）

单仓全环闭环可复现；高 Re 碰撞算子族（CM/CUMULANT/KBC）+LES+150 份 evidence 工程验证链；结冰全链条（开放生态唯一）；舰船外流场景专注；国产硬件（sdaa）适配位。

## 三、优先级

**P0（本月）**：A1 收口；A2+A3（fp16 生产链 + 选择性 halo）；D1 PyPI；D2 测试清账；#185/#186 合并。
**P1（1-3 月）**：国产硬件 P1a/P1b（见第五节：LSF+神威桥、可移植性门禁）；A4 GPU STL 体素化；B1 scan_runner 数据 campaign；C2 评测基准 v1；A6 可微参考路径 + solver-in-the-loop 示范；A7 checkpoint；Reporter/注册表（进行中）吸纳 82 worker。
**P2（3-6 月）**：A5 AMR 生产化；B3 ML 反哺主循环；D4 多节点 HPC 验证；C1 模型 zoo + D5 平台 paper。
**卡位研究**：B4 foundation-model 接口。

## 四、对标要点（摘自 docs/surveys/ 三份报告）

- **FluidX3D**（性能标杆）：许可证禁军事+禁 AI 训练源码——**只借论文思想，一行代码不碰**。可借鉴：fp16 存储、固体格早退+mid-grid 湿节点 BB、闭式 Smag τ、选择性 halo、GPU 体素化、编译开销治理。
- **lettuce**（MIT，可抄实现）：Reporter 协议、ExtFlow 案例注册、UnitConversion Ma 锚点、六阶差分初始化、conftest 自动发现测试矩阵。
- **XLB**（Apache-2.0，可抄实现）：BC 整数 id registry、"流一遍"掩码推导、PrecisionPolicy 五档、选择性 halo（与 FluidX3D 交叉验证）、手写分段 checkpoint 反传、solver-in-the-loop 范式。反面教材：无测试 CI、文档停滞。
- **性能口径**：FluidX3D 19.1 GLUPS@5090 vs 我们 8.6，差距主因是内存格式（77 vs 153 B/格/步）与工况（空箱 D3Q19 vs 带障碍 D3Q27），双方都在 roofline 上（82%/77%）——A2/A3 落地后差距预期基本消除。

## 五、国产硬件 HPC+AI 维度（2026-08-20 追加）

原则：**不是全栈适配每一款芯片，而是每层有明确的可移植性边界，用 L0 契约层钉住跨硬件闭环。**

### 分层可移植性边界

| 层 | 内容 | 可移植性策略 |
|---|---|---|
| L0 数据/契约 | NPY/HDF5/SQLite、FieldDataProductR2、lineage | 天然全平台（含神威 lustre）；SWLBM↔TensorLBM 桥梁的通用语 |
| L1 eager 路径 | solver3d gather/roll、可微参考路径（#193） | 可移植基线：torch 插件跑哪它跑哪（CUDA/NPU/MLU/SDAA/MUSA）；零 CUDA 硬编码门禁 |
| L2 加速层 | Triton 融合内核 | CUDA 专属（ROCm 理论可试）；不追求一份代码全平台，追求"同一契约多实现"（XLB Operator.register_backend 范式）；昇腾走 torchair/torch_npu 图模式另立项 |
| L3 分布式 | z-slab + collectives | collective 库可插拔：NCCL（现状）→ HCCL/RCCL/MCCL 探测点已入 hardware.py；halo 流量已降 3.8-7.6×（#190），国产互联上收益更大 |
| L4 平台/AI4S | FastAPI/Vue、ai/ 训练、ml/ 服务 | ai/ 层已有 npu/sdaa 设备处理痕迹；FNO/Transformer 纯 torch 基本免费可移植，缺测试矩阵 |
| L5 调度 | hpc_scheduler | 已有 slurm+pbs；**LSF（神威）落地中**；后续按中心扩展 |

### 硬件版图（按可达性）

- **Tier A 已有**：神威 psn002（SWLBM C 线 + LSF，BGK 已验证）、无锡超算 x86+3090、5090 NVIDIA 集群
- **Tier B 高概率可达**：昇腾 910B（国产智算中心主力）、海光 DCU（ROCm 系）
- **Tier C 探测就绪**：天数 sdaa（仓库已有痕迹）、寒武纪/摩尔线程（hardware.py import 试探覆盖）

### 国产化优先级（并入主 P0-P2）

- **P1a** LSF 后端 + 神威数据桥规范（进行中）：神威变成 AI4S 闭环的数据生成节点——SWLBM 千核产数据 → L0 契约 → GPU（任意国产/NVIDIA）训练 → serving → 回注
- **P1b** 可移植性门禁：hardware.py 能力探测 + eager CPU 门禁测试 + 静态无 `.cuda()` 扫描 + hardware profile 入 benchmark 记录
- **P2** 昇腾/海光实测（拿到硬件后）：HCCL 插拔、torchair 降级路由、PrecisionPolicy 厂商组合验证矩阵
- **P2** compile_utils 厂商路由：NVIDIA 档（现状）/ CPU inductor 档 / torchair 档，探测降级

### 国产 HPC+AI 样板叙事

数据在神威 HPC 生成（SWLBM 百核千核、成本低）→ 大盘/对象存储 → NVIDIA 或国产 GPU 训练 → 模型入库 → serving 回注求解器。闭环每一环都可替换硬件，L0 契约层保证互操作——这是对"国产 HPC 算力和 AI 算力异构协同"最直接的工程回答。
