# FluidX3D → TensorLBM 可借鉴性分析

日期：2026-08-20 ｜ 作者：AN ｜ 性质：纯调研，未改任何代码、未占用 GPU

调研方式：jsdelivr CDN 拉取 FluidX3D master 全量源码（README/DOCUMENTATION/LICENSE/lbm.hpp/lbm.cpp/kernel.cpp/setup.cpp/defines.hpp/info.cpp）逐行阅读；arXiv 标题核验；5090 仓库 `<repo>` 只读 ssh 对照（triton_fused*.py、compile_utils.py、stl_geometry.py、docs/）。本机 github 直连被墙，改走 `cdn.jsdelivr.net/gh/ProjectPhysX/FluidX3D@master`。

---

## 0. 两个先说清楚的事实（一好一坏）

**坏消息（法律红线）**：FluidX3D 是 **"source-available no-cost non-commercial"**，不是开源。LICENSE.md 明确：①禁止商业用途；②**禁止军事/国防用途**（"You may not use this software ... for military research or any military or defense industry purposes"）；③**禁止用其源码训练 AI 模型**。我们的核心场景 SUBOFF/KVLCC2/DTMB-5415 是舰船（潜艇）阻力，正落在其军事禁令的射程内。**结论：一行代码都不能搬，也不能把它的源码喂给任何代码生成/AI 训练管线；只能借鉴其公开论文（MDPI Computation、PRE，均为公开学术成果）中的算法思想并自行实现。**

**好消息（编号纠错）**：任务里给的 arXiv:2005.11560 **不是 FluidX3D 论文**——arXiv 页面标题核验为 *"Adversarial Attack on Hierarchical Graph Pooling Neural Networks"*。FluidX3D 真正的两篇方法论文是：
- Lehmann, *Esoteric Pull and Esoteric Push: Two Simple In-Place Streaming Schemes for the Lattice Boltzmann Method on GPUs*, Computation 10(6):92, 2022, DOI 10.3390/computation10060092（就地 streaming 方案）
- Lehmann et al., *Accuracy and performance of the LBM with 64-bit, 32-bit, and customized 16-bit number formats*, Phys. Rev. E 106, 015308, 2022（FP16S/FP16C 精度方案）

---

## 1. FluidX3D 概况

| 维度 | 事实 | 出处 |
|---|---|---|
| 定位 | "最快、最省内存"的 LBM CFD 软件，OpenCL 单语言跨全硬件（N/A/I 卡、CPU、手机 ARM GPU） | README 首行 |
| 代码形态 | C++17 宿主 + 运行时生成的 OpenCL C 内核字符串，全部内嵌在 ~10 个源文件里（src/lbm.cpp 83KB、kernel.cpp 201KB 含全部内核）；无构建依赖、make.sh 一把梭 | repo 文件树 |
| 维护状态 | 单人开发（Dr. Moritz Lehmann），v1.0(2022-08) → v3.7(2026-05-14)，47 个月 33 个版本，节奏约 1 版/6 周；GitHub Stars Program 成员，社区报数 issue #8 持续更新硬件榜 | README Update History |
| 性能量级 | RTX 5090 单卡 **19141 MLUPs/s**（D3Q19 SRT 空箱、FP32 算术+FP16S/FP16C 内存）；MI355X 54.5 GLUPS；B200 55.6 GLUPS；8×MI355X 362 GLUPS（83% 并行效率）、8×A6000 40 GLUPS（57%） | README benchmark 章节 |
| 报数口径 | 明确可复现：D3Q19 SRT、无扩展（隐式 mid-grid bounce-back）、空立方体（默认 256³）、取 1000×10 步的峰值平滑值；**每格成本逐项公开**：93 B/格（FP32）或 55 B/格（FP16）、153 或 77 B/格/步带宽、363/406/1275 FLOPs/格/步，算术强度 2.37/5.27/16.56 FLOPs/B，附 roofline 百分比 | README "Single-GPU Benchmarks" 引言 + setup.cpp `#ifdef BENCHMARK` |

核心架构一句话：**一个线程一个格子的单一 `stream_collide` 内核**，周期性 + 隐式 mid-grid bounce-back 全部内联，其余一切（自由表面 4 内核、力场、温度 D3Q7、体素化、渲染、多卡传输）都是围绕它的补充内核，按 `defines.hpp` 宏在编译期裁剪。

**它没有而我们有的**（不必妄自菲薄的部分）：
- 碰撞算子只有 **SRT/TRT 两种**——README FAQ 明确拒绝 MRT（"实践中零收益甚至负收益"）。我们有 BGK/CM/CUMULANT(LES)/MRT/KBC 全家桶，高 Re 湖海工况的稳定性靠 cumulant/KBC 是真实科学需求（DrivAer/KVLCC2/SUBOFF 高 Re 生产链），这是**方法论级差异化**而非工程欠账。
- 无 AI4S 层（FNO/Transformer 代理）、无数据 catalog/lineage、无 FastAPI/Vue 平台、无 Python 生态——TensorLBM 300+ 模块 vs 它 ~10 个文件。
- 无 checkpoint（作者明确拒绝："程序足够稳不需要"）、无 AMR、无结冰、无壁面函数（文档自己承认 Ahmed body 高湍流区力偏大至 2 倍、需要 wall function——而我们有 wallfn 线）、无共轭传热/声学/6DOF。

---

## 2. 逐项对照表

标注：★=直接可借鉴；◇=需改造；✕=Triton/我们栈下不适用（附原因）。

### 2.1 性能手法

| FluidX3D 手法 | 具体做法（出处） | TensorLBM 现状 | 可借鉴性 |
|---|---|---|---|
| Esoteric-Pull 就地 streaming | 单份 f 缓冲，偶/奇时间步按 `j = t%2 ? i : (i%2? i+1 : i-1)` 交换反方向对的下标读写，零拷贝完成迁移；mid-grid bounce-back 被 twist **隐式**完成（kernel.cpp load_f/store_f，L1326-1339） | 双缓冲 ping-pong（triton_fused.py 文档"writes to a second buffer"），每步全量读写两份 f | ◇ 内存省 2 倍可由 fp16 存储等价获得；就地 twist 需重排 (Q,nz,ny,nx) 布局的 q-lane 语义+BB 相位重对齐，收益与 fp16 重叠，**优先做 fp16** |
| FP16S/FP16C 内存压缩 | 算术恒 FP32，仅 f 内存压 16bit：FP16S=范围平移 IEEE half（硬件转换），FP16C=自定义 1-4-11（软件位运算转换，kernel.cpp L848-859）；带宽 153→77 B/格/步，近 2× 速度（PRE 106,015308） | triton_fused.py **已支持** fp16/bf16 存储（rel err ~2e-4），但 obstacle/分布式/SUBOFF 生产链 "D3Q19, fp32 storage only" | ★ 最高性价比：把已验证的 fp16 存储推进 obstacle+distributed 生产链 |
| 固体格早退 | `if(flags==TYPE_S\|\|TYPE_G) return;` 固体/气体格零工作量；BB 不设独立 pass | full-way BB：固体格仍写 bounced 分布；障碍引入 −48% 成本待恢复 | ◇（见 TOP-2：湿节点 mid-grid BB + 早退） |
| equilibrium 省算（DDF-shift 代数） | f_eq 手写全展开：预计算 u0..u9 组合、rhom1=rho−1 抑制尾数相消、fma 链（kernel.cpp L1004-1061）；rho 求和把 +1 留最后加 | Triton 编译器可做部分公共子表达式消去，但未见等价手工展开 | ◇ 中等收益（算术强度只占次要地位，带宽主导）；Smagorinsky 的 clamp ±c 手法对 NaN 修复直接有用 |
| 宏观量懒更新 | `UPDATE_FIELDS` 关闭时 rho/u 不落内存，`update_fields()` 按需惰性执行（lbm.hpp L39, kernel.cpp L1565） | 每步算 rho/u 用于力与诊断（部分 pass 已融合） | ◇ 已部分做到；生产链审计一遍"每步是否真的需要写 rho/u 到 DRAM" |
| 邻居下标一次性整数计算 | `neighbors()` 展开 9 个基址组合出 26 邻居，无取模（周期 wrap 折进下标） | triton_fused 已用"单次 compare-fixup 替代真取模"（模块文档） | ★ 已对齐，无行动项 |
| SoA + 64bit 索引自适应 | `index_f: i*def_N+n`（SoA >2× 快于 AoS）；N>2³² 自动编译 64bit 索引（v2.17） | (Q,nz,ny,nx) SoA fp32；Q=19 pad 到 32 | ◇ Q_PAD=32 在 halo 交换时多传 68% 无效 lane（见 TOP-3） |
| Smagorinsky 闭式 τ | 由非平衡张量 Π=Σcαcβ(f−feq) 得 Q，τ=½(τ0+√(τ0²+18√2(CΔ)²√Q/ρ))，速度先 clamp ±c，全部寄存器内完成、无除零路径（kernel.cpp L1579-1593） | kernel 内置 Smagorinsky 有 NaN bug（修复中） | ★ 直接按公式重实现（公式源头是公开论文，非其专属） |
| 温度 D3Q7 子格 | 热场独立 D3Q7+Esoteric-pull，复用同一套 twist | 有 conjugate_ht/温度模块但未融合 | ✕/◇ 次要；结冰线如需热耦合再评估 |

### 2.2 功能面

| FluidX3D 功能 | 做法 | TensorLBM 现状 | 可借鉴性 |
|---|---|---|---|
| STL 体素化 | GPU 内核：每格发射线与三角求交+插入排序，v2.1 把分钟级 CPU 算法做到毫秒级；支持运动网格重体素化（每 1-10 步）与速度初始化；要求水密+法向朝外 | stl_geometry.py 有纯 numpy z-ray CPU 体素化（对接 drag_pressure SurfaceMesh）；生产 SUBOFF 用解析几何 build_suboff_solid_slab | ★ TOP-4：GPU 体素化解锁任意外形（DrivAer/KVLCC2 已有 worker，几何输入受限） |
| 多 GPU | 单节点跨厂商：域分解，**每轴只传穿越该面的 5/19 个方向 DDF**（lbm.cpp `transfers`：D2Q9=3/D3Q15=5/D3Q19=5/D3Q27=9），GPU→pinned CPU→指针交换→插入，PCIe 无需 NVLink/MPI | NCCL z-slab halo：staging 发**全部 q 面**（pad 后 32 个），fp32；已有异步 overlap（先 post 交换再算） | ★ TOP-3：选择性打包 5 方向（D3Q27 为 9）+fp16 传输，halo 流量降 6-12× |
| 力/力矩 | update_force_field 只对 solid 格跑，wet-node 动量交换 F=2·Σc·f（kernel.cpp L1873-1884）——**与我们的 Ladd 公式同型**；求和用 GPU 树归约+原子加（v3.2，~20× 快于 CPU 多线程） | Ladd 公式一致；all-reduce 力为 NCCL 标量；triton_fused_obstacle 内有 `_obstacle_force_reduction_kernel` | ★ 已同型；核对"每步每格"vs"只对 solid 格"的归约规模 |
| 交互可视化 | 渲染即计算：OpenCL 光栅/光线追踪直接读 VRAM 原始场，无需导出体数据；SSH 下有 ASCII 模式；多卡各渲各域+zbuffer 合成 | Vue+WebSocket 流（LBM-Platform/frontend-vue），后端切片下推 | ◇ 借"**in-situ 渲染不落盘**"思想：服务器端 GPU 直接出切片/等值面 PNG/箭头图推 WebSocket，避免整场拷回（见 4 节保留意见） |
| 初始化/参数 | `resolution(aspect, MB)` 按显存反推最大格子；units.hpp SI↔LBM 单换算自动打印系数；get_Re_max 提示稳定上限 | config/case 体系更全，但无"按显存反推分辨率"小工具 | ★ 0.5 人日的小甜点 |
| 稳态检测 | 无专门稳态检测（FAQ 只谈振荡=声波/参数） | 有 cp_measurement/时间收敛 study | ✕ 我们领先 |

### 2.3 工程面

| 维度 | FluidX3D | TensorLBM | 可借鉴性 |
|---|---|---|---|
| Benchmark 口径 | 配置、格子、步数、每格字节/FLOPs、roofline % 全公开；第三方可在 issue #8 按同一脚本复现 | benchmark_observability.py 记录设备/版本元数据；GLUPS 数字多在模块 docstring 与 evidence 文档，缺统一"每格字节+roofline%"口径 | ★ 1 人日：在 benchmark_observability 加 bytes/cell/step 与 roofline% 字段，报告统一化 |
| 文档组织 | 单一 DOCUMENTATION.md（41KB）+README（173KB，含 33 版更新史与全部 benchmark 表）；FAQ 直球（为何不用 CUDA/为何无 MRT/为何无 checkpoint） | docs/ 30+ 文件（evidence/capability matrix/survey），工程化更深但入口分散 | ◇ 各有取舍；可补一份"FAQ 直球式"入口文档 |
| 硬件适配层 | 运行时按设备查询生成 OpenCL `#define`（workgroup 大小、local memory 容量探测后决定 LS 优化开关并告警降级、NVIDIA compute capability 走内联 PTX 原子加、AMD/Intel 走 builtin/扩展、fp16 硬件转换缺失时软件模拟、N>2³² 自动 64bit 索引） | Triton 编译器自动处理大部分；num_warps/BLOCK 有 sweep 推荐值 | ✕ 生态职责不同（Triton 替我们做了）；仅"特性探测→降级+告警"的模式值得记住 |
| 编译开销 | C++ 全量编译 ~5 秒（docs §2），之后零 JIT | torch.compile 冷编 10-42s、Triton 首启 JIT、微案例 4-8s 编译开销净变慢（compile_utils 教训 5） | ◇ 见 TOP-5（缓存+形状特化+微案例 eager 降级） |

---

## 3. TOP 5 可落地项

> 全部为"借鉴公开算法思想、自行实现"，规避 FluidX3D 许可证（非商用/禁军事/禁 AI 训练源码）。

### TOP-1 把 fp16 内存存储推进 SUBOFF 生产链（内存带宽是第一瓶颈）
- **动机**：triton_fused 周期内核单 5090 fp32 已达 8.6 GLUPS=77% 带宽（1530/1790 GB/s），纯带宽受限；fp16 存储使每步 f 流量 216→108 B/格（D3Q27），理论上限近 2×。FluidX3D 5090 的 19.1 GLUPS（D3Q19+FP16S，口径不同不可直比，但其 77 vs 153 B/格/步的差正来自此）证明 LBM 生产精度在 fp16 内存下普遍可用（PRE 106,015308 系统量化）。我们自己在周期内核已测得 rel err 2e-4。
- **落点**：`src/tensorlbm/triton_fused_obstacle.py`（现为 "D3Q19, fp32 storage only"）、`triton_fused_distributed.py::_alloc_halo_staging(torch.float32)`、`triton_suboff_step_distributed.py`；load 时 `.to(tl.float32)` 计算、store 时 `.to(tl.float16)`。
- **工作量**：3-5 人日（内核 dtype 参数化 1-2 天；端到端精度回归 2-3 天）。
- **验证**：① tests/test_triton_fused.py 全绿 + 新增 fp16 存储 vs fp32 参考的最大相对误差断言（周期场）；② SUBOFF n=256 小案例 C_t/C_F 对 fp32 基线偏差 ≤2%（ITTC ±10% 容差内）；③ **重点验证力采样相位**（post-stream pre-BB 的 wet-node 力在 fp16 内存下的方差）；④ GLUPS 提升目标 ≥1.6×。

### TOP-2 障碍内核重构：固体格早退 + mid-grid 湿节点 BB + Smagorinsky 闭式 τ（直击 −48% 壁面成本与 NaN bug）
- **动机**：两个已知瓶颈同源于障碍内核。FluidX3D 的答案是：solid 格 `return` 零成本；BB 用 Esoteric twist 隐式完成不设独立相位；wet-node 力只在 solid 格上单独内核计算（`update_force_field` 与主内核分离，按需调用）。其 Smagorinsky 对速度 clamp ±c 后走闭式 τ，无除零/负数开方路径——正是我们 NaN bug 的对照解。
- **落点**：`triton_fused_obstacle.py::_fused_v2_kernel_xfar_les`（full-way BB gather 改为：流体格 pull 时遇固体邻居取本格反向分布=mid-grid 湿节点；solid 格整块跳过）；力采样移出主内核为独立小内核（只遍历 solid 格）；Smagorinsky 段按 Π 张量闭式重写。
- **工作量**：4-6 人日（语义对齐 production `bounce_back_cells_3d` 与力相位最费时）。
- **验证**：① cylinder/SUBOFF n=128 力时序与现 production 相位 bit 级或 1e-6 对齐（我们的约定：post-stream、pre-BB、F=2Σc·f）；② 有障碍 GLUPS ≥ 0.9× 无障碍周期内核（恢复 −48%）；③ 高 Re LES 案例跑 10⁴ 步无 NaN（对应 kernel 内置 Smagorinsky bug）。

### TOP-3 多卡 halo 选择性打包：只传穿越面的方向 + fp16
- **动机**：现状 staging 交换整 q 面（Q_PAD=32，fp32）：一帧 halo = 32×ny×nx×4B。物理上 z 向 slab 每步只需 cz>0 的方向：D3Q19=5 个、D3Q27=9 个。FluidX3D 的 `transfers` 常量正是这么定义的（lbm.cpp L10-22）。打包 5 方向+fp16 后 halo 流量降 ~12.8×（32×4→5×2），通信隐藏更容易，8 卡 SUBOFF 的 17.69 GLUPS 还有通信侧余量。
- **落点**：`triton_fused_distributed.py::_alloc_halo_staging / _start_halo_exchange / _finalize_halo`：staging 改 `(n_cross, ny, nx)` fp16，发送前 gather 穿越方向，接收后 scatter 进 ghost 面；q 索引表从 d3q19/d3q27 常量生成（勿手抄——triton_fused.py 头部记录过手抄符号错误教训）。
- **工作量**：2-3 人日。
- **验证**：① 多卡周期场 vs 单卡结果容差对齐（fp16 传输引入 ~1e-3 级 halo 误差，需评估对力统计的影响）；② nsys/torch profiler 量化 8 卡 n=512 通信时间占比下降；③ 8 卡 GLUPS 提升 ≥5%。

### TOP-4 GPU 上 STL 体素化（解锁任意外形输入）
- **动机**：生产 SUBOFF 是解析几何（`build_suboff_solid_slab`）；stl_geometry.py 的 numpy z-ray 是 CPU 路径，n=1024 网格上会成为新瓶颈（对照：v2.1 之前 FluidX3D CPU 体素化要分钟级，GPU 后毫秒级）。我们要接 DrivAer/KVLCC2/真实艇体 STL，这是功能缺口。
- **落点**：`src/tensorlbm/stl_geometry.py` 新增 `voxelize_stl_gpu`（Triton：每格 z 向射线 vs 三角列表求交+奇偶填充，或按其思路做包围盒裁剪+插入排序处理共线交点）；沿用现有 `read_stl`/`SurfaceMesh_from_stl` 下游。
- **工作量**：5-8 人日（watertight 容错、法向一致性、与近壁/力积分管线的接口回归占大头）。
- **验证**：① 与 CPU z-ray 体素化在 sphere/cylinder/NACA 解析 STL 上 mask 一致率 ≥99.9%，差异格全部位于表面 1 格层；② 体素化耗时 benchmark（目标：n=512 网格 <1s）；③ 体素化后 SUBOFF 小案例力与解析几何版一致。

### TOP-5 微案例编译开销治理（4-8s 导致净变慢）
- **动机**：FluidX3D 全量编译 ~5s、之后零 JIT；我们冷编译 default ~10s / max-autotune-no-cudagraphs 35-42s（compile_utils 教训 5），微案例（n=64/128）摊不平。它的启示不是"用什么编译器"而是"**编译是一次性固定成本**"：把形状特化收敛到少数枚举 + 产物落盘复用 + 小案例干脆走 eager。
- **落点**：`compile_utils.py`：① Triton/torch.compile 缓存目录固化到共享盘（`TRITON_CACHE_DIR`/`TORCHINDUCTOR_CACHE_DIR`），worker 冷启时预热；② 形状分档（n≤128 eager、其余按 256/512/1024 三档预编译）；③ benchmark 报数把"冷/热"分开列。
- **工作量**：2-3 人日。
- **验证**：微案例端到端 wall-time 对比（目标：n=64 净时间 ≤ eager 的 1.1×）；同形状二次运行 <2s 进入稳态。

---

## 4. 不建议照搬的项与原因

1. **OpenCL 全栈 / OpenCL-Wrapper**：跨厂商这件事在我们的栈里由 PyTorch+Triton 后端承担（还要兼容 SW/天数智芯 sdaa——benchmark_observability 里已有 sdaa 分支）。OpenCL-Wrapper 是作者自有库且同许可证；重写无收益。其 FAQ "OpenCL 与 CUDA 等效"的论断对我们没有迁移价值。
2. **任何源码级复制**（再次强调）：非商用+禁军事+禁 AI 训练三条款，叠加"发布改版结果须公开改版源码"的传染条款，与我们的舰船生产方向冲突。只读论文、自己写。
3. **编译期宏裁剪的单体架构**（defines.hpp 决定一切）：适合单人 C++ 项目，不适合 Python 平台的动态配置/组合（我们的 capability contract 体系是对的）。
4. **"无 checkpoint"哲学**：我们 n=1024 生产 campaign 一跑数天，checkpoint 是刚需；FluidX3D 的"重跑比重启快"只在它那个性能档位成立。
5. **拒绝 MRT/高阶碰撞算子的立场**：我们 CM/CUMULANT/KBC 是高 Re 外流稳定性的实际依赖，也是与 FluidX3D 拉开差异的科学资产。
6. **自研渲染引擎**：我们已有 Vue/WebSocket/FastAPI 栈和前端工程投入，重造光栅/光追内核不值；只借"in-situ 渲染、不导出体数据"的思路（服务器端出切片/等值面图像流）。
7. **就地 Esoteric-pull 本体**（在 fp16 已做的前提下）：收益与 TOP-1 重叠，且要重排 q-lane 偶奇语义+BB 相位，风险/收益比差于 fp16；列为 TOP-1 完成后的可选二期。

---

## 5. 参考来源

- FluidX3D repo（master，经 jsdelivr CDN 镜像逐文件核对）：README.md（benchmark 口径与全部性能数字、FAQ、更新史）、DOCUMENTATION.md（BC/STL/力/视频/导出章节）、LICENSE.md（非商用/军事/AI 训练条款）、src/lbm.hpp、src/lbm.cpp（stream_collide 调度、communicate_field 选择性传输、transfers 常量）、src/kernel.cpp（stream_collide L1454-1636、load_f/store_f L1326-1339、calculate_f_eq L1004-1061、FP16C 位运算 L848-859、update_force_field L1873-1884、SUBGRID L1579-1593）、src/setup.cpp（`#ifdef BENCHMARK` 场景）、src/defines.hpp（精度/扩展开关）、src/info.cpp（MLUPs/GB/s 报数）。
- 论文：Lehmann, Computation 10(6):92, 2022, DOI 10.3390/computation10060092（Esoteric pull/push）；Lehmann et al., Phys. Rev. E 106, 015308, 2022（16-bit 格式精度）；Lehmann, IWOCL'22（OpenCL 交互光追）。
- arXiv:2005.11560 标题核验（export.arxiv.org/abs）→ 确认**非** FluidX3D 论文。
- TensorLBM 5090 仓库只读对照：`<repo>/src/tensorlbm/{triton_fused,triton_fused_obstacle,triton_fused_distributed,triton_suboff_step_distributed,compile_utils,stl_geometry,benchmark_observability}.py`、`docs/lbm-open-source-survey-2026-07-02.md`（其中无 FluidX3D 条目，本报告为首次覆盖）。
- 性能数字口径备注：FluidX3D 单卡表 = D3Q19 SRT 空箱、FP32 算术+(FP32/FP16S/FP16C 最快)内存、256³ 峰值平滑；多卡表 = 最大立方域 2×1×1/2×2×1/2×2×2。TensorLBM 数字 = triton_fused 模块 docstring（8.6 GLUPS n=256 单 5090 fp32、69 GLUPS 8 卡周期）与 SUBOFF 生产记录（17.69 GLUPS 8×5090 n=1024 D3Q27 含障碍/BC/力，口径不同不可直接横比）。RTX 5090 roofline：19141 MLUPs×77 B≈1474 GB/s≈82% of 1792 GB/s spec；我们 fp32 路径 1530/1790=77%——**双方都在 roofline 上，差距的主要构成就是内存格式（153 vs 77 B/格/步）与工况（空箱 vs 带障碍 D3Q27）**。

（报告完。本地：/root/fluidx3d_lessons_20260820.md；已同步：5090 <scratch>/fluidx3d_lessons_20260820.md）
