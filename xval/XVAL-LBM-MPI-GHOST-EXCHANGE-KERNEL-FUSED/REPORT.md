# XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED — 第 2 轮（mission 18746）

## ① 问题设定与基准出处

**问题.** 平台原厂 `D3Q19GlooTransport.exchange_ghosts` 每步把 x 面 19 个 population 全量发
给邻居；OSS 参考（`lssfau/lbmpy` + `lbmpy_walberla`，`oss_solutions` id 113
`mpi-ghost-exchange-kernel-fused`）按 PackInfo 只发**穿越该面的方向**（D3Q19 x 面 = 5/19），
且 pack/unpack 是每个面一遍的融合 kernel。本锚点验证「方向选择性 + 融合」的 ghost 交换：
(a) 分解场与单体场**逐位相同**；(b) 对 3D Taylor–Green 解析解 rel-L2 **< 15%**（owner 带）；
(c) 交换字节按解析式降到 5/19。

**基准出处（external_anchor）**

| 量 | 出处 |
|---|---|
| 3D TGV 初值 `u=U0 sin kx cos ky cos kz`, `v=-U0 cos kx sin ky cos kz`, `p=(U0²/16)(cos2kx+cos2ky)(cos2kz+2)` | G. I. Taylor, *Phil. Mag.* 26 (1937) 674；M. E. Brachet et al., *JFM* 130 (1983) 411 |
| 线性衰减 `e^{-3νk²t}` | 模态本征值 `∇²u = -3k²u`（三轴各 `-k²`）⇒ `∂t u = ν∇²u = -3νk²u`；本报告 §①-补 给出离散核验与求解器自测 |
| ghost 交换语义/缓冲大小 | `lssfau/lbmpy` `advanced_streaming/communication.py:get_communication_slices`/`_extend_dir`（本地 checkout `/tmp/lbmpy_src`）、`lbmpy_walberla/packinfo.py` |
| 15% 精度带 | owner 2026-11 标准，等于 `predeclared.json` 里的 `owner_acceptance_pct`（**未改动**）|

**①-补 第 1 轮 honest-FAIL 的根因（本轮主发现）.** 第 1 轮（m18720）报的 24.89% 不是 ghost 交换
的缺陷：同轮 `identity_max_abs_vs_serial = 0.0`、字节比恰为 5/19。误差全在**共享的参考解**里——
3D TGV 的衰减率是 `3νk²`，上一版 `case.decay` 用了 z-不变（2D）TGV 的 `2νk²`，两者比值 1.5。
本轮先证明后修改：

* **E1 纯数学**：中心差分 `lap` 作用在 `sin(kx)cos(ky)cos(kz)` 上，与 `-k²` 之比：
  * n=16: `lap(u)/u = -0.456723`，`/(-k²) = 2.9616`（离散符号 `-12sin²(k/2)`，连续极限 3）
* **E2 求解器自测（不用参考解）**：把数值场基频模态幅度投影出来，对 `log(amp)~t` 最小二乘拟合
  出**实测衰减率**，并在两个初始幅度上重复（`U0` 与 `U0/10`）——若残差随 `U0²` 缩放，它属于
  TGV 的非线性谐波，与参考解真伪无关：

| n | U0 | Ma | 实测 λ | 2νk² | 3νk² | λ vs 2νk² | λ vs 3νk² |
|---|---|---|---|---|---|---|---|
| 16 | 0.04 | 0.0693 | 0.023981 | 0.015421 | 0.023132 | +55.5% | +3.67% |
| 16 | 0.004 | 0.0069 | 0.023981 | 0.015421 | 0.023132 | +55.5% | +3.67% |
| 16 | 0.04 | 0.0693 | 0.005871 | 0.003855 | 0.005783 | +52.3% | +1.52% |
| 16 | 0.004 | 0.0069 | 0.005835 | 0.003855 | 0.005783 | +51.3% | +0.89% |

判据（写死在 `scripts/xval_tgv_reference_audit.py`，开工前定）：接受 `3νk²` 需 `|dev|<8%` 且随 n 收缩、
`|dev vs 2νk²|>20%` 恒成立。实测 `|dev vs 2νk²|` 最小 51.3%，`|dev vs 3νk²|` 最大 3.67% ⇒
**判据满足，`2νk²` 被否证**（图 `figs/fig0_reference_audit.png`）。

## ② 方法与实现

* `packinfo.py::extend_dir` 复刻 lbmpy `_extend_dir`，`crossing_populations = set(_extend_dir(dir)) ∩ stencil`
  ⇒ D3Q19 x 面 5 个方向，**不是手写表**；`tests/test_xval_mpi_ghost_fused.py` 直接与安装的 lbmpy 函数比对。
* `kernels.py`：`pack_face` = 一次 `index_select` + `contiguous`；`unpack_face` = 一次融合索引写。每面一遍。
* `transport.py::SlabExchange`：`isend/irecv` 4 条消息；`persistent` 复用 `(19,nz,ny,nx+2)` 双 ghost 缓冲。
  四臂 = {19 通道, 5 通道} × {每步 alloc, 持久缓冲}，其外还有一条 `replicate_shipped_exchange` 对照路径。
* `case.py`：D3Q19 BGK + `torch.roll` 周期 pull 流式。本轮唯一物理改动：`decay(t)=exp(-decay_rate·t)`，
  `decay_rate = -laplacian_eigenvalue·ν = 3νk²`；2D 的 `2νk²` 保留为显式 `decay_rate_2d`（仅供审计）。
* 误差定义 `rel_L2 = ||u_num-u_ref||₂/||u_ref||₂`，`u_ref = U0 e^{-3νk²t}(sin kx cos ky cos kz, -cos kx sin ky cos kz, 0)`；
  `O(U0²)` 谐波**不**并入参考解而是计入误差。

**复现命令**

```bash
PYTHONPATH=src python scripts/xval_tgv_reference_audit.py --n 16 32 --steps 400 \
  --out xval/XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED/audit/reference_audit_n16_n32.json
PYTHONPATH=src python scripts/xval_mpi_ghost_fused.py --only n32_r2 --out /tmp/xval_scratch
python -m pytest tests/test_xval_mpi_ghost_fused.py tests/test_xval_mpi_ghost_fused_tgv_reference.py -q
```

## ③ 参数表（lattice units）

| 参数 | 值 |
|---|---|
| 格子 / 碰撞 | D3Q19 / BGK，cs²=1/3 |
| τ / ν | 0.65 / (τ-0.5)/3 = 0.05 |
| n=32 臂 | steps=200，k=0.19635，2 rank，float32，Ma=U0√3=0.0693，Re=U0/(νk)=4.07 |
| n=64 单体点 | steps=400，k=0.09817，float32，Ma=0.0693，Re=8.15，3νk²t_end=0.578（衰减到 0.561）|
| 审计点 | n=16/32，steps=400，U0∈{0.04, 0.004}，float64 |

## ④ 对照表（ref vs got，逐项判据）

**(a) 物理判据（n=32, 2 rank, 200 steps, 5 臂 + 单体）**

| arm | rel-L2 [%] | 幅值比 | <15% | <5% |
|---|---|---|---|---|
| full_alloc | 8.71 | 0.9698 | PASS | FAIL |
| full_persistent | 8.71 | 0.9698 | PASS | FAIL |
| selective_alloc | 8.71 | 0.9698 | PASS | FAIL |
| selective_persistent | 8.71 | 0.9698 | PASS | FAIL |
| *(monolithic)* | 8.71 | 0.9698 | PASS | FAIL |

**(b) 逐位一致性（阈值 0.0；数据搬运没有合法数值自由度）**

| arm | max\|arm−monolithic\| | max\|arm−selective_persistent\| | =0 |
|---|---|---|---|
| full_alloc | 0.0e+00 | 0.0e+00 | PASS |
| full_persistent | 0.0e+00 | 0.0e+00 | PASS |
| selective_alloc | 0.0e+00 | 0.0e+00 | PASS |
| selective_persistent | 0.0e+00 | 0.0e+00 | PASS |

**(c) 传输判据（解析式 n_pop·nz·ny·itemsize，每 rank 每步左右各一发）**

| arm | bytes/step | 解析式 | 与全量臂之比 | 期望 5/19 |
|---|---|---|---|---|
| full_alloc | 311296 | 77824 | 1.000000 | n/a (control) |
| full_persistent | 311296 | 77824 | 1.000000 | n/a (control) |
| selective_alloc | 81920 | 20480 | 0.263158 | PASS |
| selective_persistent | 81920 | 20480 | 0.263158 | PASS |

**(d) 预声明主配置 n=64 / 400 steps（单体；MPI 臂见 §⑥ 的队列说明）**

| 参考解 | rel-L2 [%] | 判据 <15% |
|---|---|---|
| `3νk²`（修正后，本轮） | 17.63 | FAIL |
| `2νk²`（第 1 轮的锚，被否证） | 22.79 | FAIL |

幅值比 0.9806（≈1，说明衰减率已经对），但 rel-L2 仍有 17.63% ⇒ 残差是**形状**不是幅值：线性参考缺 `O(U0²)` 谐波与
  平衡态初值的启动瞬态。该点上 15% 带**未达标（honest FAIL on this point）**，归因实验
  （`U0/10` 对照）因单次运行超出 120 s 工具上限未跑完，标 `not_run`。

## ⑤ 收敛性与参数敏感性

**(a) 衰减率的分辨率收敛（求解器自测，无参考解参与）**

| n | k | λ_fit | 3νk² | dev vs 3νk² | dev vs 2νk² |
|---|---|---|---|---|---|
| 16 | 0.39270 | 0.023981 | 0.023132 | +3.67% | +55.5% |

**(b) 幅度敏感性（U0 vs U0/10）**

| n | U0 | rel-L2 vs 2νk² | rel-L2 vs 3νk² |
|---|---|---|---|
| 16 | 0.04 | 96.87% | 31.65% |
| 16 | 0.004 | 96.86% | 31.36% |
| 16 | 0.04 | 55.96% | 5.40% |
| 16 | 0.004 | 55.10% | 2.93% |

**(c) 时间演化（n=32，同一算例，t=100…400）**：rel-L2 由 8.99% 单调降到 5.97%，幅值比 0.9824→0.9530
——早期最大，与「平衡态初值缺非平衡部分触发弱声学模态」一致，随衰减后消失。

## ⑥ 失败模式、局限与工件

* **坑（已入负知识库 id 736）**：3D TGV 用 2D 的 `2νk²` 当锚。识别信号只有在「分解 vs 单体 = 0 且
  两者都偏 25%」时才会指向参考解；只看分解层会永远空转。防复发：离散拉普拉斯本征值 + 求解器
  自测衰减率双钉（`tests/test_xval_mpi_ghost_fused_tgv_reference.py`，8 个用例全绿）。
* **n=64 的 4 配置全矩阵**（含 4 rank 与 float64 臂）已递交集群（job `cj2019`，state=`queued-after-10min`）。排队未派发，故本轮只落了 n=32 的 MPI 五臂 + n=64 单体点；
  `results.json`/`table.csv` 的 n=64 部分**未刷新**（保持第 1 轮的旧文件即错，故本轮改名为
  `results_n32_r2.json`，避免与旧数据混淆）。这是本单的未完成部分，如实标注。
* **初始化瞬态**：初值取平衡态，缺 `O(∂u)` 非平衡修正 ⇒ 早期误差抬升（n=32 时 t=100 为 8.99%）。
* **适用边界**：gloo/CPU、周期 x-ring、slab 分解；非周期面、in-place AA-pattern、通信/计算重叠、
  GPU(NCCL) 未覆盖；n 必须被 rank 数整除。
* **O(U0²) 谐波**：参考解只含线性模态，非线性谐波计入误差；§⑤(b) 给出分离办法。

**工件**：`results_n32_r2.json`、`fields_n32_r2.pt`、`audit/reference_audit_n16_n32.json`、
`audit/serial_n64_400.json`、`figs/fig0_reference_audit.png`、`figs/fig2_error_vs_analytic.png`、
`figs/fig3_transport_cost.png`、`figs/fig4_field_identity.png`、`REPORT.md`、`REFERENCE_AUDIT.md`；
源码 `src/tensorlbm/xval_mpi_ghost_fused/*`、脚本 `scripts/xval_tgv_reference_audit.py`、
`scripts/xval_mpi_ghost_fused.py`、测试 `tests/test_xval_mpi_ghost_fused{,_tgv_reference}.py`。
