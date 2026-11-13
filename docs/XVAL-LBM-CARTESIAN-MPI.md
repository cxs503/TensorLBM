# XVAL-LBM-CARTESIAN-MPI · TensorLBM 笛卡尔域分解 vs pylbm 0.12.0 cartesian-mpi

任务锚：`XVAL-LBM-CARTESIAN-MPI`（第 1 轮，开源对标）。判据：误差 < 15%（owner 精度标准）。

- **本仓（subject）**：`src/tensorlbm/domain_decomposition.py`（`DomainDecomposer` / `HaloExchange` / `SimulatedMultiDevice`）
- **对标 OSS（ref）**：`pylbm` 0.12.0，PyPI wheel `pylbm-0.12.0-py3-none-any.whl`，
  关键文件 `pylbm/mpi_topology.py`、`pylbm/storage.py`、`pylbm/domain.py`（行号见 §1 表）
- **工件**：`tools/xval_cartesian_mpi/run_xval_cartesian_mpi.py`（驱动）、
  `tools/xval_cartesian_mpi/shims/mpi4py/MPI.py`（仅执行用 mpi4py 替身，见 §6）、
  `artifacts/xval_cartesian_mpi/{summary.json,table_decomposition.csv,table_equivalence.csv}`

> 零手抄：本报告中所有 pylbm 侧的分解尺寸/邻居表/消息数都由 **pylbm 自己的函数**
> （`MpiTopology.get_region_indices_`、`get_region`、`CartComm.Get_cart_rank`、
> `Storage._set_subarray` 的 subsizes 语义）在运行时算出；本仓侧所有字节数由
> `HaloExchange` 真实分配的 staging buffer（`numel()*element_size()`）读出。

---

## 1 问题 + ref 出处

LBM 的多进程/多卡并行靠“笛卡尔域分解 + halo 交换”。两个实现路线差异很大：

| 维度 | pylbm cartesian-mpi（ref） | TensorLBM（本仓） |
|---|---|---|
| 分解轴 | **ND 笛卡尔**：`comm.Create_cart(split, period)`，`split=Compute_dims(size,dim)` 或用户 `-npx/-npy/-npz`（`mpi_topology.py:66-79`） | **单轴 slab（z）**：`DomainDecomposer`，`decomposition_axis` 字段存在但只按 z 切 |
| 整除要求 | **不要求**：`get_region_indices_` 把余数分给低 rank：`n//nproc + ((n%nproc) > i)`（`mpi_topology.py:88-116`） | **要求整除**：`nz_global % n_devices != 0` 直接 `ValueError`（模块 `__post_init__`） |
| ghost 宽度 | `vmax`（D3Q19 为 1），数组本身带 halo（`domain.py:423-440`） | 固定 1 层（`(Q, nz_local+2, ny, nx)`） |
| 交换粒度 | **整个 ghost 面、全部 nv 个分布**，用 MPI 派生数据类型 `Create_subarray`（零拷贝，strided，`storage.py:263-303`） | **选择性**：只交换穿越该面的 5 个（D3Q19, cz=±1）方向，显式打包进连续 buffer（`_cross_up/_cross_dn`） |
| 通信语义 | 每方向 `Irecv/Irecv/Isend/Isend` + `Waitall`（按方向串行等待），邻居=面邻居 2·dim 个（`storage.py:307-368`） | 1 次 `dist.batch_isend_irecv` 批处理 4 个 P2P op（NCCL），或 simulated 直拷贝 |
| 周期性 | `period=[True]*dim` 在 `Domain.construct_mpi_topology` 内**硬编码全周期**（`domain.py:373-383`） | 隐式周期（`get_neighbor_ranks` 取模），无物理边界在交换层 |
| 浮点 | 派生数据类型硬编码 `mpi.DOUBLE`（float64） | `dtype` 可配（**默认 float32**） |
| 后端 | CPU numpy / cython / loopy(OpenCL) | PyTorch（CUDA 多卡 / CPU） |

## 2 方法与实现

三条独立证据臂，先声明判据再跑：

- **A 分解几何**：网格 N ∈ {16,24,32,48} × rank 数 2/3/4/4(2×2)/8(2×2×2)/6(3×2)，
  两侧各自用**自己的代码**算 local 尺寸、非均衡度 `max/min−1`、可分解轴数、整除要求。
- **B 描述符/通信量**：**每步消息数**与**每条消息字节数**——本仓从真实 staging buffer 读，
  pylbm 从其 `_set_subarray` 的 `subsizes=[nv]+vshape, subsize[d+1]=vmax` 与 `MPI_DOUBLE` 推出（同一公式由其实参驱动）。
- **C 数值等价性**：三维 Taylor–Green 算例，**沿分解轴变化**（在 (z,x) 平面内）：
  `u_x=-U0 cos(k(z-½)) sin(k(x-½))`, `u_z=U0 sin(k(z-½)) cos(k(x-½))`, `u_y=0`，
  全周期，D3Q19 BGK，`τ=3ν+0.5`。三条对照臂：
  ① 单域串行（`stream_full`，全周期 roll）；② 本仓 slab 分解（P=1/2/4，simulated 模式，**模块原样使用**）；
  ③ 解析解 `K(t)=U0²/4·e^{-4νk²t}`。
  指标：终场速度场相对 L2（分解 vs 单域，分母=单域 RMS）、`gamma_K=ln(K0/K_end)/(2·steps)` 相对解析的误差。

**参数表**（无量纲/格点单位，`summary.json["case"]`）：

| 参数 | 值 |
|---|---|
| 网格 | 24³（周期） |
| dx=dt=1, la=1 | 格点单位 |
| U0 | 0.05 |
| Ma | 0.0866 |
| ν | 0.02 |
| τ_BGK | 0.56 |
| k | 2π/24 = 0.261799 |
| Re（U0/(νk)） | 9.55 |
| 步数 | 100 |
| 域中心口径 | x_i=i+0.5（初始场与解析解同口径） |
| 格点/碰撞 | D3Q19 / BGK(SRT) / 全周期 |

## 3 计算结果

### 3.1 C 臂：分解等价性（`table_equivalence.csv`，程序化注入）

| dtype | ranks | K_end | gamma_K 实测 | gamma_K 解析 | vs 解析误差 | **场相对 L2（vs 单域串行）** | 判定 |
|---|---|---|---|---|---|---|---|
| float64 | 1 | 3.529609e-4 | 2.856972e-3 | 2.741557e-3 | 4.21% | **0.0**（比特一致） | ✅ <15% |
| float64 | 2 | 3.529609e-4 | 2.856972e-3 | 2.741557e-3 | 4.21% | **0.0** | ✅ |
| float64 | 4 | 3.529609e-4 | 2.856972e-3 | 2.741557e-3 | 4.21% | **0.0** | ✅ |
| float32 | 1 | 3.529559e-4 | 2.857042e-3 | 2.741557e-3 | 4.21% | 8.74e-4 % | ✅ |
| float32 | 2 | 3.529559e-4 | 2.857042e-3 | 2.741557e-3 | 4.21% | 8.74e-4 % | ✅ |
| float32 | 4 | 3.529559e-4 | 2.857042e-3 | 2.741557e-3 | 4.21% | 8.74e-4 % | ✅ |

- 分解**不是**误差源：float64 下 P=1/2/4 与单域逐比特一致（相对 L2 = 0）；float32 下 P≥1 的差异来自
  逐位求和顺序，2.4e-3 相对量级远低于 15% 判据。
- `gamma_K` 相对解析 4.21%（24³、Ma=0.087 的分辨率/压缩性误差，另一条独立的解析-臂参照），
  `K_end` 相对解析 2.28%（`summary.json["analytic_check_float64_serial"]`）。

### 3.2 A/B 臂：分解几何与通信量（`table_decomposition.csv`）

结论（数值见 CSV，全部由两侧代码算出）：

- **可分解自由度**：pylbm 3 轴（2×2 / 2×2×2 / 3×2 全部接受）；TensorLBM **仅 1 轴**，
  且 `decomposition_axis ∈ {x,y,z}` 三者在同一 `(nz,ny,nx,P)` 下给出**相同** `nz_local` 与 slab 形状
  （`summary.json["axis_param_probe"]`）——即该参数是**装饰性字段**，实际恒按 z 切。
- **非均衡度**：pylbm 在 N 不能被 P 整除时自行分配余数（`imbalance>0`，例如 24=8+8+8、3 rank 时
  8/8/8 → 0；N=16,P=3 → 6/5/5 → imbalance=20%）；TensorLBM 对同样的组合**直接拒绝**（`accepted=False`）。
- **通信量（同一 slab 切分，D3Q19，每步每 rank）**：
  - TensorLBM：2 条消息 × 5 个分布 × ny·nx × **4 B**（float32 默认）
  - pylbm：2·dim 条消息 × 19 个分布 × 面 × **8 B**（`MPI_DOUBLE` 硬编码）
  - 3D 比值 = (6·19·8)/(2·5·4) = **22.8×**（TensorLBM 少 22.8 倍线上字节）；即使 pylbm 换 float32 仍是 11.4×。
  - 消息数：TensorLBM 每步 2 条（+2 条 recv，NCCL 中一次 `batch_isend_irecv` 提交）；pylbm 每步 2·dim=6 条且**按方向串行 Waitall**。
  - 代价：TensorLBM 必须打包/解包（额外 `5·ny·nx` 的内存拷贝），pylbm 用派生数据类型零拷贝。

## 4 差异清单与移植取舍

**差异清单**（做法/参数/边界条件）：

1. 分解维度：pylbm ND 笛卡尔 vs 本仓 1D slab → 本仓在“细长域 + 大量 rank”场景面临板厚趋零/`nz_local<P` 的扩展性天花板。
2. 整除策略：pylbm 余数分配（总能跑）vs 本仓硬性整除（拒绝运行）→ 本仓在任意 `(N,P)` 组合下**不可用**，这是可移植性缺陷而非精度缺陷。
3. 交换粒度：本仓选择性 5/19 分布（省 3.8× 字节）+ 连续 buffer（NCCL 友好）vs pylbm 全平面 + 派生类型（零拷贝、CPU 友好）——**本仓做法对 GPU/NCCL 更优，不建议反向移植**。
4. 周期性：pylbm 硬编码全周期（物理边界只能靠 boundary elements 在域内做）；本仓隐式周期、无边界层语义——边界条件归属更清晰。
5. dtype：pylbm 硬编码 float64 vs 本仓可配但**默认 float32**（本项目 §3.1 显示 float32 在此算例仍达 8.7e-4%，故默认值可接受；但低 Ma/长时程验证应显式 float64）。
6. 通信等待策略：pylbm 按方向串行 Waitall（可重叠度低）vs 本仓单次批处理——本仓更优。

**移植取舍（建议）**：

- **建议移植**：① 余数分配分解（`n//P + ((n%P)>i)`，让 `(N,P)` 全组合可用）——这是本单唯一测到的“pylbm 有、本仓无”的能力缺口，且改动局部（`get_slab_bounds` 改前缀和 + 每 rank `nz_local` 由形状推出）。
  ② 可选 ND 分解（`splits=(nx,ny,nz)`）——收益随 rank 数增长，但需重写 `HaloExchange` 的打包面逻辑（x/y 面同样只需交换穿越该面的分布），建议按需第二轮做。
- **不移植**：全平面 + 派生数据类型（字节数 22.8×，且 PyTorch/NCCL 需连续 buffer）；硬编码全周期；硬编码 float64。
- **保留本仓特有**：选择性分布交换、批处理 P2P、可配 dtype。

## 5 收敛性/参数敏感性（至少两点）

1. **ranks 敏感性**：P=1,2,4 在 float64 下终场与单域**逐比特一致**（相对 L2=0.0）→ 分解/交换算子对 rank 数**无误差敏感性**（这与“分解等价性”强判据等价）。float32 下 P 从 1→4 的差异恒为 8.74e-4%（由逐位求和顺序决定，与 P 无关量级）。
2. **精度格式敏感性**：float64 → float32 使场差从 0 升到 8.74e-4%（约 9e-4 个百分点），仍远低于 15% 判据；说明此前序贯实现若默认 float32，其分解误差不是瓶颈，**打包/解包的正确性（顺序）才是**（见 §6 第 1 条）。
3. **网格/解析对照敏感性**：24³ 下 `gamma_K` 相对解析 +4.21%、`K_end` 相对解析 +2.28%（离散/压缩性误差，非分解误差）。

## 6 失败模式、坑与工件

**坑 1（本单最重要的实证发现，必须写进库）**：交换必须发生在 **collide 之后、stream 之前**。
第一版我把顺序写成 collide→stream→exchange，结果 P=1 时就已偏离单域 8.4%（P=2/4 为 10.6%/9.8%，且误差随界面数增长），
一开始会误判成“TensorLBM halo 协议有 15% 量级缺陷”。判据设计救了这一单：**P=1 必须与单域逐比特一致**（无界面，任何偏差都是装配错误），
该自检 5 分钟定位到顺序问题。修正后 P=1/2/4 全部 0.0。→ 建议把 `P=1 等价性` 固化为该模块的回归断言。
**坑 2**：`HaloExchange` 的 staging buffer 以张量（`buf["up"]`）存放而非 dict，`.values()` 会命中 torch 的 `values` 属性并抛 `RuntimeError`；测字节数应 `numel()*element_size()`。
**坑 3**：`HaloExchange` 默认 `dtype=float32`，与 `DomainDecomposer` 的 fp64 工作流混用会静默降精度；做验证必须显式传 `dtype=torch.float64`。

**失败/未完成（honest）**：
- **pylbm 求解器臂 = WITHHELD**：① 本机 loongarch64 / Loongnix 无 MPI 实现，`pip install mpi4py` 在 C 扩展编译阶段失败
  （`Requires-Dist: mpi4py`），故**无法**跑多 rank pylbm；② 单 rank pylbm `Simulation()` 另被其 cerberus `symbol` 校验阻断
  （schema 要求 sympy.Symbol 类型，而 cerberus 在错误路径排序时对 sympy 关系式求值抛 `TypeError`，Python 3.11 + cerberus 1.3.8；
  即使绕过校验，`Scheme._get_conserved_moments` 要求守恒矩必须**逐符号**出现在 `M·feq` 的行里，程序化构造的 D3Q19 原始矩基无法满足）。
  → 因此本单的 ref 侧数字只覆盖**分解几何/描述符**（这恰是锚点“cartesian-mpi 做法”的对象），物理对照用**解析解**承担独立第三臂。
  生态摩擦：`friction: mpi4py unbuildable on loongarch64`（无 apt 源、无 manylinux wheel）。
- **未做**：余数分配分解的**已实现补丁**（本单只做 XVAL 与取舍建议，未改 `domain_decomposition.py`）；ND 分解；真实多卡 NCCL/多进程 MPI 实测；图形化（matplotlib 图未生成，本报告以 CSV 数值表承担对照曲线，故 `figures: []`）。

**工件**：`artifacts/xval_cartesian_mpi/summary.json`（全参数+整套结果）、`table_decomposition.csv`（A/B 臂）、`table_equivalence.csv`（C 臂）。
复现：`PYTHONPATH=tools/xval_cartesian_mpi/shims:/tmp/pylbm_src python tools/xval_cartesian_mpi/run_xval_cartesian_mpi.py --n 24 --steps 100`。