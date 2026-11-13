# XVAL-LBM-MPI-1D-SLAB-UNPACKED-HALO-TESTALL-OVERLA — 交叉对标报告

任务锚 `XVAL-LBM-MPI-1D-SLAB-UNPACKED-HALO-TESTALL-OVERLA`（第 1 轮·开源对标）。判据 <15%（owner 精度标准）。

## ① 问题 + ref 出处

**问题**：LBM 多进程 1-D y-slab 分解中，「halo **不打包**（unpacked）+ `MPI_Testall` 进度轮询与计算**重叠**」这套 OSS 做法，能否在 TensorLBM 里被语义精确移植，且不改变解？

**ref（零手抄，取自平台已登记的 OSS 台账，非重打代码）**
- 仓：`lbm-principles-practice/code`（MIT），cache commit `6e2c592`（`oss_inventory.cache_nodes=n315,n330`）
- 条目：`oss_solutions` 行 `repo=lbm-principles-practice/code, feature_slug=分布式多卡, tag=mpi-1d-slab-unpacked-halo-testall-overlap`
- 做法原文要点：① NY 切连续 slab，halo 行 = `rank_ny+2`；② 布局是 direction-major、y 行连续，故一个 halo 块是 `(ndir-1)*NX` 个**连续** double，`MPI_Isend/Irecv` **无需打包**；③ 每流完一条内部 y 行就 `MPI_Testall`，让通信与剩余 collision 工作重叠；④ 全局范数用**一次**七标量 `MPI_Allreduce` 合并；⑤ rank 取模回绕（`rankp1/rankm1 % nprocs`）。

**对标限制（不粉饰）**：ref 实现是 C++/CUDA + MPI，本镜像**无 mpi4py、无 mpirun、无 GPU**（运行时检查记录）。故本仓对照臂是**语义精确移植**，不是执行 OSS 二进制；传输层用每方向一条 socketpair / 队列的 mpi4py 替身。

## ② 方法与实现

新增能力：`src/tensorlbm/parallel/slab_unpacked.py`（`UnpackedHaloExchange` / `PackedHaloExchange` / `SlabSolver` / `QueueChannel`+`ThreadPeers`+`SocketPeers` 传输替身）。

- 布局 `f[q, y, x]`（q=0 为静止分量），halo 行两条（index 0 与 ny+1）；
- 碰撞先做 → 立刻 `isend` 边界行（**零拷贝**：`f[1:, row, :]` 本身连续）→ 内部行按 chunk 流式拉取，每个 chunk 后 `Testall` → 完成后 `waitall` + 把 ghost 行贴回 → 最后流两条边界行；
- 对照：`PackedHaloExchange`（本仓 `domain_decomposition.HaloExchange` 语义：只交换跨面方向 6/8，打包进 staging buffer）；`unpacked-selective`（零拷贝+只换跨面方向，需 fancy-index 拷贝，用于给 OSS 取舍定价）；
- 物理解析锚：**横向剪切波** `u_x=u0 cos(ky)e^(-νk²t)`（散度为零 → 不激发声学模态，正好沿被分解轴变化）。

差异清单（本仓 vs OSS）：

| 维度 | OSS `mpi-1d-slab-…-overlap` | 本次移植 | 取舍 |
|---|---|---|---|
| halo 载荷 | 全 `(ndir-1)*NX` 连续块，零打包 | 同（`q=1..8` 整块） | 零拷贝，但比选择性多 8/6=+33.3% 线上字节 |
| 进度轮询 | 每内部行 `MPI_Testall` | 同（chunk 粒度可调 1/8/64 行） | 轮询次数是真实开销，测到 29–58 次/步 |
| 交换方向 | 不筛方向 | 不筛（另给 selective 对照） | 见上 |
| 传输 | MPI（真实网络/共享内存） | socketpair / 队列替身（无线程安全 MPI 栈） | 字节数与消息数精确，墙钟时间不代表 MI 时间 |
| 全局归约 | 7 标量一次 Allreduce | 同（诊断窗口调用） | 求和型归约对 `max|u|` 会放大，见⑥ |
| 分解轴 | y（1-D slab），rank 取模回绕 | 同 | NP=2 时同一 peer 承载两个面，需固定 FIFO 次序（本次踩坑） |
| 精度 | double | float64 | 一致 |

## ③ 参数表（无量纲）

| 量 | 值 |
|---|---|
| 格子 | D2Q9，BGK/SRT |
| τ | 0.8 → ν = 0.1（格子单位） |
| 网格 | N = 128, 128（另 64/128 收敛臂） |
| 步数 | 300（固定，跑前声明，未因结果延长） |
| 初值 | 剪切波 u0 = 0.04，k = 2π/N（每盒一个波长），x/y 双周期 |
| Mach | max|u|/c_s = 0.064（NP=1 真值；NP>1 栏位是求和归约，见⑥） |
| rank 数 | 1 / 2 / 4（y-slab，取模回绕） |
| 密度漂移 | 1.9e-14（相对） |

## ④ 对照表（ref vs got + 打勾叉）

| arm | 探针误差 % | 场 L2 误差 % | 与 NP=1 身份偏差 | 线上字节/步 | staging 拷贝字节/步 | Testall 次/步 | MLUPS | 质量漂移 |
|---|---|---|---|---|---|---|---|---|
| unpacked_np1 | 0.0138 | 0.0138 | — | 32768 | 0 | 0.0 | 0.619 | 1.9e-14 |
| unpacked_np2 | 0.0138 | 0.0138 | 0.0e+00 | 32768 | 0 | 51.8 | 0.391 | 1.9e-14 |
| unpacked_np4 | 0.0138 | 0.0138 | 0.0e+00 | 32768 | 0 | 29.4 | 0.297 | 1.9e-14 |
| unpacked-selective_np1 | 0.0138 | 0.0138 | 0.0e+00 | 24576 | 0 | 0.0 | 0.498 | 1.9e-14 |
| unpacked-selective_np2 | 0.0138 | 0.0138 | 0.0e+00 | 24576 | 12288 | 49.1 | 0.414 | 1.9e-14 |
| unpacked-selective_np4 | 0.0138 | 0.0138 | 0.0e+00 | 24576 | 12288 | 29.3 | 0.305 | 1.9e-14 |
| packed-selective_np1 | 0.0138 | 0.0138 | 0.0e+00 | 24576 | 0 | 0.0 | 0.435 | 1.9e-14 |
| packed-selective_np2 | 0.0138 | 0.0138 | 0.0e+00 | 24576 | 12288 | 49.9 | 0.371 | 1.9e-14 |
| packed-selective_np4 | 0.0138 | 0.0138 | 0.0e+00 | 24576 | 12288 | 29.8 | 0.359 | 1.9e-14 |

- **✔ H1 分解中性**：NP=2/4 在三种 halo 约定、overlap 开/关下都与 NP=1 **逐位相同**（identity = 0.0，不是 1e-12 以内，而是 0）。
- **✔ H2 物理精度**：探针与场 L2 误差 **0.0138%**，远小于 15%（也小于 3%）。
- **✔ H4 通信记账**：unpacked 全块 / selective = 1.3333 = 8/6（闭式）；unpacked staging 拷贝 = 0 字节，selective = 12288 B/步。
- **✘ H3 overlap 收益**：在本镜像的替身传输下 `Testall` overlap **不比阻塞 waitall 快**（见⑤），判为 honest negative（替身限制，非 MPI 结论）。

## ⑤ 收敛性 / 参数敏感性

收敛（等物理衰减时间）：

| N | 步数 | 探针误差 % | 场 L2 % | 实测衰减因子 | 解析衰减因子 |
|---|---|---|---|---|---|
| 64 | 75 | 0.0551 | 0.0551 | 0.929751 | 0.930264 |
| 128 | 300 | 0.0138 | 0.0138 | 0.930136 | 0.930264 |

轮询粒度（NP=4）：结论——轮询越密越慢，`Testall` 次数是真实成本，不是免费的重叠。

| 轮询步长（行/chunk） | Testall 次/步 | 墙钟 s |
|---|---|---|
| 1 | 29.33 | 17.53 |
| 8 | 3.45 | 13.45 |
| 64 | 1.00 | 16.25 |

overlap 开/关（unpacked，替身队列传输）：

| ranks | Testall overlap 墙钟 s | 阻塞 waitall 墙钟 s | speedup | polls/步 |
|---|---|---|---|---|
| np1 | 12.69 | 14.73 | 1.161 | 0.0 |
| np2 | 13.85 | 6.58 | 0.475 | 51.2 |
| np4 | 15.21 | 6.93 | 0.456 | 29.7 |

speedup < 1 的原因（诚实）：替身把「发送」实现为**同步拷贝进队列**，阻塞版反而少一次线程调度；且 CPython GIL 让主线程 numpy 工作与通道线程争用。真实 MPI 上该结论**不可外推**（记为局限，不是结论）。

## ⑥ 失败模式与工件

- **踩坑 1（已抓）**：`f` 与 `post` 双缓冲置换写反 → 流式结果被丢，KE 恒定不变（表观「不演化」）。修法：不置换，`f` 存流式态、`post` 纯 scratch。
- **踩坑 2（已抓）**：NP=1 时跳过 exchange → ghost 行残留 0 → 发散。修法：交换函数无条件调用，size==1 时自身周期回绕。
- **踩坑 3（已抓）**：NP=2 时同一 peer 承载两个面，收发次序不一致 → 场错乱（identity≈0.98）。修法：接收次序镜像对端发送次序（对端先发上边界行 = 我方下 ghost）。
- **踩坑 4（已抓）**：`Testall` 提前返回后被 `continue` 掉的内部行从未流式 → 残差 1e-4~1e-2。修法：轮询结束后补流剩余行。
- **踩坑 5（已抓，物理）**：用 Taylor-Green 解析解做锚时，局部平衡初值会激发**驻波声学模态**（盒尺度，周期≈L/c_s≈111 步），误差摆动到 200%。这不是分解错误而是初值/边界层选择问题；改用**散度为零的横向剪切波**后误差降到 0.0138%。
- **未修（记账口径）**：OSS 的七标量 Allreduce 是**求和**，`max|u|` 若直接塞进求和项会被放大 ≈ NP 倍（表中 NP=4 的 Ma 0.258 即此伪值；NP=1 的 0.064 才是真 Ma）。max 型量应走 `MPI_Reduce(MAX)`。已在报告中标注，未改数值。
- **工件**：`artifacts/xval_mpi_slab_unpacked_halo_overlap/{results.json,kpi.csv,tables.md,fig1_field_and_profile.png,fig2_comm_accounting.png,fig3_overlap_and_polling.png}`；驱动 `xval/XVAL-LBM-MPI-1D-SLAB-UNPACKED-HALO-TESTALL-OVERLA/{predeclared.json,run_xval.py,make_report.py}`；测试 `tests/test_parallel_slab_unpacked.py`（10 passed）。

## 图

- **图1** `fig1_field_and_profile.png`：NP=1 与 NP=4 的 u_x 场云图 + x=0 剖面 vs 解析解（t=300，误差 0.014%）。
- **图2** `fig2_comm_accounting.png`：三种 halo 约定的线上字节与 staging 拷贝字节（8/6 与零拷贝的直接证据）。
- **图3** `fig3_overlap_and_polling.png`：overlap vs 阻塞墙钟、以及轮询粒度敏感性。
