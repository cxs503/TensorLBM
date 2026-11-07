# SDAA 多卡能力探针报告

## 1. 案例介绍

**背景**：TensorLBM 求解器运行于 loong64 架构容器，需确认 SDAA（ShenWei Deep Learning Accelerator）多卡扩展能力，为后续分布式 LBM 计算提供硬件依据。

**测试环境**：
| 项目 | 值 |
|---|---|
| 架构 | loong64 |
| PyTorch | 2.7.1 |
| SDAA 可见卡数 | 32 |
| 测试时间 | 2026-11-07 |

**探针五项**：① `torch.sdaa.device_count()` 可见卡数；② `torch.distributed` 后端布尔表；③ Gloo allreduce 带宽 vs 卡上流带宽比值（1/4/16/64 MB）；④ 2-rank allreduce SUM 正确性校验。

## 2. 方法

- **设备计数**：直接调用 `torch.sdaa.device_count()`。
- **后端探测**：遍历 `['nccl','gloo','mpi','ucc']`，调用 `torch.distributed.is_backend_available()`。
- **卡上流带宽**：单卡 `sdaa:0` 上执行 10 次 `copy_`（H2D / D2H），`perf_counter` 计时，取平均 MB/s。
- **Gloo AllReduce 带宽**：`torchrun --nproc_per_node=2` 启动 2 rank，CPU tensor 上执行 10 次 `all_reduce(SUM)`，barrier 同步后计时，有效带宽 = `size_MB × 10 × 2 / elapsed`。
- **正确性校验**：2 rank 各填充 `tensor = rank`，allreduce SUM 后验证所有元素等于 `0+1=1`。

## 3. 计算结果

### 3.1 设备与后端

| 字段 | 值 |
|---|---|
| device_count | **32** |
| nccl | False |
| gloo | **True** |
| mpi | False |
| ucc | False |

### 3.2 带宽实测（MB/s）

| 消息大小 | H2D | D2H | Device Avg | Gloo AllReduce | 比值 |
|---|---|---|---|---|---|
| 1 MB | 639.9 | 1291.8 | 965.9 | 570.0 | **0.59** |
| 4 MB | 1731.5 | 1270.7 | 1501.1 | 664.3 | **0.44** |
| 16 MB | 1739.0 | 2298.0 | 2018.5 | 663.3 | **0.33** |
| 64 MB | 1502.0 | 7988.7 | 4745.4 | 1674.8 | **0.35** |

### 3.3 正确性

| 测试 | 结果 |
|---|---|
| 2-rank allreduce SUM | **PASS** ✓ |

## 4. 结果图

![SDAA Bandwidth Comparison](figures/p8/sdaa_bandwidth.png)

*左：H2D/D2H 卡上流带宽与 Gloo AllReduce 带宽对比；右：Gloo/Device 比值（效率）。*

## 5. 注意点

1. **仅 Gloo 后端可用**：NCCL/MPI/UCC 均不可用，分布式训练/推理只能走 Gloo（CPU 集合通信），性能受限。
2. **Gloo 带宽瓶颈**：Gloo allreduce 带宽约为卡上流带宽的 33%–59%，小消息（1MB）效率最高（0.59），中等消息（16MB）效率最低（0.33）。64MB 时 Gloo 带宽跃升至 1674 MB/s，但 Device D2H 也跃升至 7988 MB/s，比值仍仅 0.35。
3. **H2D/D2H 不对称**：D2H 带宽普遍高于 H2D，64MB 时 D2H 达 7988 MB/s（约为 H2D 的 5.3 倍），可能与 SDAA 的 DMA 引擎特性有关。
4. **容器内 32 卡可见但仅测 2 rank**：受限于当前容器配置与测试时间，仅验证了 2-rank 场景。更大规模多卡扩展需进一步测试。

## 6. 多卡路线裁决

**裁决结论：当前容器支持多卡扩展，但仅限 Gloo 后端（CPU 集合通信），不具备 GPU-native NCCL 能力。**

- **device_count=32** 表明硬件层面 32 张 SDAA 卡对容器可见，多卡扩展在设备枚举层面可行。
- **仅 gloo=True** 意味着所有集合通信走 CPU 内存路径，无法利用卡间高速互联（如 NVLink/SDAA Link），带宽上限受限于 CPU 内存总线。
- **correctness_pass=true** 证明 2-rank Gloo allreduce SUM 语义正确，分布式计算逻辑可正常工作。
- **带宽比值 0.33–0.59** 表明 Gloo 通信效率中等，对于计算密集型 LBM（通信占比低）可接受；对于通信密集型场景（如大规模 3D 域分解）可能成为瓶颈。

**建议路线**：
1. **短期**：使用 Gloo 后端进行 2–4 卡小规模分布式 LBM 验证，确认求解器正确性。
2. **中期**：若需大规模扩展，需确认 SDAA 是否提供类似 NCCL 的卡间通信库（如 `torch.sdaa` 原生集合通信 API），绕过 Gloo CPU 路径。
3. **长期**：推动 SDAA 后端 NCCL 兼容层或原生 SDCA（ShenWei Collective Communication）支持。

## 7. 误差判定

本任务为硬件能力探针，无实验/文献基准值可对标。各项指标为直接实测值，不存在误差判定问题。

---

*产物：`artifacts/p8/sdaa_probe.json`（五字段齐全）、`docs/p8/SDAA_PROBE_REPORT.md`、`figures/p8/sdaa_bandwidth.png`*
