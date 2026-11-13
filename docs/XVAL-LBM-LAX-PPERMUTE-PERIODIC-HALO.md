# XVAL-LBM-LAX-PPERMUTE-PERIODIC-HALO —— 与 vivsim `stream_cross_devices` 的交叉对标

* 仓：TensorLBM ｜ 单号：m18112（返修轮 1，原单 m17868）｜ 锚点：`XVAL-LBM-LAX-PPERMUTE-PERIODIC-HALO`
* 对标 ref（OSS）：`haimingz/vivsim` @ `fb84b12f02e18f3b9dfdf738d6a1a318d81539f9`（2026-06-28），
  `vivsim/multidevice.py::stream_cross_devices`（39 行，MIT，Copyright (c) 2024 Haiming Zhu）
* 判据：对标误差 < 15%（owner 精度标准）；等价性臂另设更严判据「与单域解逐位一致」。
* 工件（全部程序化生成、零手抄）：`artifacts/xval_ppermute_periodic_halo/{RESULTS.md, summary.json, kpi.csv, fig1..fig3}`

---

## ① 问题 + ref 出处

**问题**：TensorLBM 的多设备 halo 走的是「ghost 平面 + pack/unpack staging」
（`src/tensorlbm/domain_decomposition.py::HaloExchange`：每 rank 分配
`(n_cross, ny, nx)` 收发缓冲、逐方向 pack → 通信 → unpack）。OSS vivsim 用另一条路：

1. 局部 streaming **先做整块自回卷**（`shift_*`/`roll`，块边界是本地自环的错误值）；
2. 随后**一次 `jax.lax.ppermute`**，把各 rank 的"自回卷面"搬到邻居的**同名面**：
   自回卷之后 rank r 的 plane 0 恰好装着它自己的最后一个 plane、plane −1 恰好
   装着它自己的第一个 plane —— 正是邻居需要的内容，于是**不需要任何 staging 缓冲**；
3. device 维周期性由取模对列表 `[(i, (i+1) % n)]` 直接给出（`multidevice.py:27-28`）。

**ref 出处（逐行）**：`vivsim/multidevice.py:27-36`（`device_pairs` / `device_pairs_reverse`
/ 四条 `f.at[...].set(lax.ppermute(...))`）；配套约定来自 `vivsim/lbm/basic.py`
（`shift_x_pos/shift_y_pos/streaming`，拉式周期）与 `vivsim/lbm/lattice.py`
（`right_dirs=[1,5,8]`、`left_dirs=[3,7,6]`、`up_dirs=[2,5,6]`、`down_dirs=[4,7,8]`）。
OSS 源码在本机 `/tmp/vivsim_src`（clone 自 upstream，commit 见上），核读而非文本迁移：
本仓实现按 TensorLBM 的 `(Q, nz, ny, nx)` 布局与 `torch.roll` 约定**重写**，方向分组由
lattice C 表**程序生成**（`axis_pull_groups`），未复制 OSS 任何代码行。

**开工三查（生态病如实记）**：`/api/v1/tensor/knowledge/search`、`/api/v1/knowledge/search`、
`/api/v1/tensor/negative-knowledge/search` 均返回空/降级 —— 知识检索引擎 `127.0.0.1:8021`
`Connection refused`（`engine_offline`），本单**无 KB 可站**、无负经验命中；
以仓内既有 `docs/surveys/lettuce-xlb-lessons.md` TOP-3（XLB `distribute.py` 的 ppermute
选择性 halo）与本单 ref 源码为对标依据。记入生态摩擦。

---

## ② 方法与实现

| 件 | 路径 | 作用 |
|---|---|---|
| 核心实现 | `src/tensorlbm/parallel/ppermute_halo.py` | `device_pairs`/`device_pairs_reverse`/`sender_index`（fail-closed 置换校验）/`ppermute`（`out[dst]=in[src]`，torch+numpy，一次性 stack-gather 保证同时性）/`axis_pull_groups`/`tensor_component_index`/`PpermutePeriodicHalo`（wrap-then-permute，任意空间轴）/`oss_stream_cross_devices`（2D 原式移植，含 `swap_x_branch` 对照臂）/`oss_stream_cross_devices_jax`（真 `lax.ppermute` 通路，本沙箱不可用） |
| 集成面（L3） | `src/tensorlbm/parallel/sharded_executor.py` | `ShardedPpermuteRunner`：用**仓库真求解器** `tensorlbm.lbm_step.LBMStepExecutor`（53 个求解器共用的逐步执行器）逐 shard 推进，shard 间只多一次 `PpermutePeriodicHalo.apply`；`gather/gather_macro/stats` |
| 运行入口（L2 entry） | `tools/xval_ppermute_halo/run_xval.py` | A1–A5 五臂 + 图 + `RESULTS.md/kpi.csv/summary.json` 全程序化生成 |
| 测试 | `tests/test_parallel_ppermute_halo.py` | 27 条断言（不变量级：置换性、逐位等价、fail-closed、OSS 分支核账） |
| CLI 自检 | `python -m tensorlbm.parallel.ppermute_halo` | 打印取模对/置换表/方向分组并做冒烟断言 |

**关键正确性论证（为什么交换的只是"自回卷面"）**：TensorLBM 的 streaming 是拉式
`out[q][i] = f[q][i − c_q]`（`torch.roll`）。沿空间轴 `a`：
`c_a>0` 的方向在 `plane 0` 处取到的是**本块自己的最后一个平面**（自环错值），
正确的值恰是 rank−1 的最后一个平面 —— 而 rank−1 自回卷后的 `plane 0` 就是它自己的最后平面；
`c_a<0` 同理在 `plane −1` 与 rank+1 交换。
**轴↔分量映射**不是约定俗成而是从 `LBMStepExecutor._stream_preallocated` 的实际 roll 公式反推：
`(sx,sy,sz)=shifts[q]` 后 `roll(shifts=(sz,sy,sx))`，故 tensor 空间轴 `axis` 对应 C 的第
`ncomp−1−axis` 列（`tensor_component_index`；测试 `test_axis_component_mapping_matches_executor_roll`
把它钉死，防止后续布局改动悄悄失效）。

---

## ③ 参数表（无量纲数完整）

| 参数 | 符号 | 值 | 说明 |
|---|---|---|---|
| 网格 | L³ | 16³（主臂）/ 12³、24³（敏感性） | D3Q19，全周期 |
| 初场 | u₁(z) | `U0 sin(2πz/L)` | 剪切波（NS 精确解：`U0 e^{−νk²t}`） |
| 幅值 | U0 | 0.01 | 格子单位 |
| Mach | Ma = U0/c_s | 1.73×10⁻² | 不可压区 |
| 松弛时间 | τ | 0.6（主臂）/0.55、0.8（敏感性） | BGK |
| 运动黏度 | ν = (τ−½)/3 | 3.333×10⁻² | 格子单位 |
| 波数 | k = 2π/L | 0.3927 | 无量纲 |
| 衰减率 | νk² | 5.140×10⁻³ /step | 解析解指数 |
| 波长雷诺数 | Re_λ = U0/(νk) | 0.764 | 粘性主导（解析解适用域） |
| 步数 | N_t | 100（主臂）/60（24³） | 首末幅值比 e^{−0.514} = 0.598 |
| rank 数 | N | 1（ref）/2、4、8 | 沿 z 等分 |
| 通信选择性 | — | 10 面/步（5+5） | D3Q19 沿 z 穿越面 `|c_z|=1` |

---

## ④ 对照表（ref vs got）

**A1 单域(NP=1) vs sharded —— 判据「逐位一致」**

| N | 步 | max\|Δf\| | bit-identical | ρ rel L2 | u_y rel L2 |
|---|---|---|---|---|---|
| 2 | 20 | 0.000e+00 | ✔ | 0.0 | 0.0 |
| 4 | 20 | 0.000e+00 | ✔ | 0.0 | 0.0 |
| 8 | 20 | 0.000e+00 | ✔ | 0.0 | 0.0 |

**A3 与解析解（判据 <15%）**：L=16³、τ=0.6、≤100 步 →
**|rel err| = 1.875% ✔ PASS**（逐站误差见表 `RESULTS.md`，全程 −0.0%…−1.9%，均远小于 15%）。

**A2 OSS 原式 2D D2Q9 分支核账（本仓对 OSS 做法的独立复核）**

| split | N | 变体 | max\|Δf\| | rel L2 |
|---|---|---|---|---|
| y | 2 / 4 | OSS 原样 | 0.000e+00 | 0.0 |
| x | 2 | OSS 原样 | 0.000e+00 | 0.0 |
| x | 4 | OSS 原样 | **9.397e-01** | **3.03e-01** |
| x | 4 | perm 互换 | 0.000e+00 | 0.0 |

⇒ **OSS `dir=='x'` 分支的两个 perm 与拉式 streaming 约定互为镜像**（`right` 面 0 用了
`device_pairs_reverse`、`left` 面 −1 用了 `device_pairs`）；`dir=='y'` 分支正确。
N=2 时 `pairs` 与 `pairs_reverse` 恰好是同一条置换（`[(0,1),(1,0)]`），因此该缺陷
**只在 N≥3 显形** —— 这也是它能在 OSS 里长期存活的解释。OSS 里该函数**全仓无调用点**
（`grep -rn stream_cross_devices` 仅命中定义行），属未接线特性；本仓移植前先把它修正，
并由 `test_oss_x_branch_as_written_diverges_and_swap_fixes_it` 固化为可复现判据。

**A4 通信成本（N=4，L=16³，10 步平均）**

| 方案 | 平面/步/rank | 字节/步/rank | 张量算子/步 | staging 缓冲 | 进程内墙钟 s/步 |
|---|---|---|---|---|---|
| ppermute（选择性、零 staging） | 10 | 10 240 | 18 | 0 | 0.0116 |
| 朴素全 Q staging | 38 | 38 912 | 304 | 19 | 0.0085 |

线量比 = **3.80×**（naive/ppermute）；扩展开销 = 18 vs 304 个张量算子；staging 内存 = 0 vs 19 个平面缓冲。
**诚实记录**：单进程模拟 mesh 下 ppermute 的墙钟**更慢**（0.73×）——`stack`-gather 复制整块平面，
而朴素臂是在视图上原地 `copy_`；线量优势只有在真多卡/多节点（NCCL/互连受限）才兑现，
本沙箱无多卡、无 jax/mpi4py（见 ⑥），故**不对墙钟收益下结论**（WITHHELD）。

---

## ⑤ 收敛性 / 参数敏感性（两点以上）

| L（步数） | k | |rel err| vs 解析 | <15% |
|---|---|---|---|---|
| 12（100） | 0.5236 | 4.197% | ✔ |
| 16（100） | 0.3927 | 1.875% | ✔ |
| 24（60） | 0.2618 | 0.626% | ✔ |

误差随网格加密单调下降（≈ O(k²)：k 缩小 1.33× → 误差降 3.0×），是**离散截断**而非 halo 误差
（halo 臂 A1 逐位为 0）。τ 敏感性：τ=0.55 → 1.613%、0.6 → 1.875%、0.8 → 2.087%（均 PASS；
τ 越大衰减越快、同一步数下相对误差略升，量级与 BGK 的 k² 截断一致）。

---

## ⑥ 失败模式与工件 + 差异清单 + 移植取舍

### 失败模式
1. **换错邻居**：`pairs`/`pairs_reverse` 互为镜像（OSS x 分支即此病）；N=2 时无法暴露。
   防御：逐位等价测试覆盖 N=2/3/4 × 三根轴；`sender_index` 对非置换 perm fail-closed。
2. **轴↔分量错配**：仓库布局 `(Q,nz,ny,nx)` 而 C 列序 `(cx,cy,cz)`，若按列号当轴号会静默换错面。
   防御：`tensor_component_index` + 与 `_stream_preallocated` 对齐的回归测试。
3. **时序错配**：本 runner 在 `step()` 之后交换（post-step）。周期域上等价（collide 逐点局部），
   但**若有 post-streaming 边界条件**（Zou-He 等）必须挪到 BC 之前；故
   `ShardedPpermuteRunner` 对 `boundary_fn is not None` **fail-closed 拒绝**。
4. **不可分块**：轴长度不被 N 整除 → 直接 `ValueError`（不做静默截断）。
5. **jax 真通路 WITHHELD**：loongarch64 沙箱无 jax wheel（`import jax` 失败），
   `oss_stream_cross_devices_jax` 只保留「真 `lax.ppermute`」实现并带 guard，
   由 `test_jax_path_is_withheld_without_jax` 固化该声明；本单全部数值证据来自
   **语义等价**的 torch 模拟实现（同时性由一次性 stack-gather 保证）。

### 本仓 vs OSS 差异清单（做法 / 参数 / 边界条件）

| 维度 | vivsim（OSS） | 本仓（TensorLBM 移植） |
|---|---|---|
| 后端原语 | `jax.lax.ppermute`（真跨设备集合置换，jit/vmap 内） | torch 语义等价模拟（stack-gather，单进程多 shard）；jax 通路保留但 WITHHELD |
| 维度 | 2D D2Q9，`(9, NX, NY)` | 3D D3Q19（+D3Q27/D2Q9 表由 C 生成），`(Q, nz, ny, nx)` |
| 轴覆盖 | `dir ∈ {'x','y'}`，一次一轴 | 任意空间轴 `axis ∈ {0,1,2}`，方向分量按布局映射程序生成 |
| 方向分组 | 硬编码 `up/right/left/down_dirs` | `axis_pull_groups(C, component)` 由 lattice C 符号生成，无手抄 |
| perm 方向 | y 分支正确、x 分支镜像（N≥3 发散） | 统一按拉式约定（低面取 rank−1、高面取 rank+1），x/y 对称 |
| 边界条件 | 只用周期（函数内无 BC 处理） | 周期；带 BC 情形显式拒绝（见失败模式 3） |
| 接线状态 | `stream_cross_devices` **全仓无调用点**（未接线特性） | 有入口（`tools/xval_ppermute_halo/run_xval.py`）+ 集成面（`ShardedPpermuteRunner` 驱动真 `LBMStepExecutor`）+ 27 条测试 |
| 记账 | 无 | `planes_per_exchange` / `wire_bytes_per_step` / `tensor_ops` / staging 计数 |

### 移植取舍
1. **不引入 JAX 后端**（与仓内 `docs/surveys/lettuce-xlb-lessons.md` §4-3 的既定方针一致）：
   只借「wrap-then-permute + 取模对」的**做法语义**，实现留在 torch/Triton/NCCL 栈。
2. **保留 wrap-then-permute 而非改写成 ghost 平面**：这条路把"周期性"和"邻居归属"都压进同一条
   取模置换，省掉 pack/unpack 与 staging 内存（0 vs 19 面缓冲）；代价是必须在
   streaming **之后**交换（时序约束已写进 API 的 fail-closed）。
3. **选择性交换保留**（10 面/步，而非全 Q）：线量 3.8× 优于朴素法，与 OSS 只在穿越方向置换的精神一致。
4. **不追 OSS 的 2D 单轴接口**：本仓 3D 为主，故把「一次一轴」泛化为任意轴，并保留 2D 原式臂
   （`oss_stream_cross_devices`）专门用于复核 OSS 本身。
5. **修 OSS 的 x 分支再移植**：不把已知的镜像缺陷带进生产路径；差异与判据都留在测试里。

### 证据映射（L2/L3/L4）
* **L2** impl：`src/tensorlbm/parallel/{__init__,ppermute_halo,sharded_executor}.py`（非占位，含 fail-closed 分支）；
  entry：`tools/xval_ppermute_halo/run_xval.py`（`--quick` 冒烟 / 全量）；run_artifacts：
  `artifacts/xval_ppermute_periodic_halo/{RESULTS.md,summary.json,kpi.csv,fig1..fig3}`。
* **L3** integration：`ShardedPpermuteRunner` 调用仓内真求解器 `LBMStepExecutor.step`
  （53 个求解器共用路径），`tests/test_parallel_ppermute_halo.py::test_sharded_executor_bit_exact_against_monolithic` 证明与单域逐位一致。
* **L4** benchmark：A1（ref=单域，逐位）、A2（ref=全域 2D stream）、A3（ref=解析 `e^{−νk²t}`，|err|=1.875% < 15% ✔）、A4（成本 ref/got 对照）；
  repro：`PYTHONPATH=src python tools/xval_ppermute_halo/run_xval.py --out artifacts/xval_ppermute_periodic_halo`（wall ≈ 60–105 s，单机 CPU，无需集群）。

### 生态摩擦（如实记）
1. 知识检索引擎 `:8021` 不可达（`engine_offline`），开工三查全部降级 —— KB 复用与负经验防重蹈本单无法执行。
2. loongarch64 沙箱无 `jax`、无 `mpi4py` wheel ⇒ 真 `lax.ppermute` 跨设备通路与真实 NCCL 多卡计时均 WITHHELD（非本仓缺陷）。
3. 本仓 `d3q19.C` 的列注释（±x/±y/±z）与 tensor 轴标签（`nz,ny,nx`）对读者易误导（本单用 test 固化映射），建议后续在 `lbm_step` 注释里点明 `roll(shifts=(sz,sy,sx))` 的实际取列顺序。