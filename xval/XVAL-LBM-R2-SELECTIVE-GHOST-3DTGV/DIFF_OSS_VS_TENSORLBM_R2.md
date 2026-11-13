# 差异清单：TensorLBM 选择性 ghost exchange vs OSS（lbmpy / waLBerla）

对标对象：`lssfau/lbmpy` 2.0（wheel 位于 `/usr/lib/python3.11/site-packages/lbmpy`，
checkout `/tmp/lbmpy_src`）及其 waLBerla 侧的打包语义。
本仓：`tensorlbm.multi_gpu.D3Q19GlooTransport`（R2 新增 `selective_ghosts` /
`persistent_ghost_buffer`）与 `tensorlbm.xval_mpi_ghost_fused`。

**零手抄原则**：跨面粒子集合不是手写表，而是由
`packinfo.extend_dir` 复刻 OSS 函数
`lbmpy/advanced_streaming/communication.py::_extend_dir`（lbmpy 2.0 第 233 行：
通信方向的零分量展开为 `{-1,0,1}`，非零分量保持），
并在 `tests/test_xval_mpi_ghost_fused.py::test_crossing_populations_match_oss_extend_dir`
中对**全部 26 个通信方向**做集合相等断言。

## 1. 做法（approach）

| 维度 | OSS lbmpy / waLBerla | 本仓 TensorLBM | 影响 |
|---|---|---|---|
| 求解器形态 | 代码生成（pystencils → C 内核），`create_lb_function` 融合 collide+stream（`default_stream_collide` 系列） | 手写 PyTorch 张量算子，`collide → exchange → pull stream` 分步 | OSS 靠生成器把循环融合，本仓靠张量广播；两者都保持「拉式」读取 |
| ghost 交换粒度 | waLBerla `PackInfo`：按通信方向只打包穿越该面的 pdf（D3Q19 单面 5 个） | R1：整面 19 通道；**R2：单面 5 通道**（`crossing_right/crossing_left` 由 stencil 派生） | 线字节 5/19 = 0.2632（R2 实测精确值） |
| 面的定义 | `slice before ghost layer`（`face_index`） | 同样取 owned 平面（`f_owned[...,0]` / `[...,-1]`） | 语义一致，R2 有测试钉死 |
| 周期性 | `LBMPeriodicityHandling` / `SerialDataHandling.synchronization_function` 包装成周期环 | x 方向点对点环（rank-1 / rank+1 取模），y/z 用 `torch.roll` 自身周期 | 本仓不做全维 ghost 层，周期性由 ring + roll 承担 |
| 缓冲复用 | 生成的内核直接写预分配数组 | R1 每步 `torch.empty` 重建 padded；**R2 `persistent_ghost_buffer=True` 复用** | 分配因子与字节因子被 2×2 消融分离 |
| 校验 | 无运行时 ghost 校验（调试打印可选） | 每步 `validate_ghosts` 失败即抛（可选择只校验穿越集合） | 本仓更保守：R2 selective 模式下只校验穿越集合，否则会与「未定义的其他通道」比较 |

## 2. 参数（parameters）

| 参数 | OSS | 本仓 R2 | 说明 |
|---|---|---|---|
| 单面穿越 pdf 数 | 5（D3Q19，由 `_extend_dir` ∩ stencil 得出） | 5（同法派生，测试对拍） | 面/角方向分别为 5（面）、2（棱）、1（角） |
| 每步线字节 | 2 面 × 5 × nz × ny × itemsize | 同 | 实测 327680 B @ n=64 r2 f32（R1 整面 1245184 B） |
| 线字节比 | 5/19 = 0.2631578947368421 | 0.2631578947368421（精确） | 整数计数，无容差 |
| 时间步进 | 融合内核单趟 | collide / exchange / stream 三趟 | 本仓多一趟内存写，交换计时单列 |
| 后端 | MPI（waLBerla）/ 单机 serial | gloo 点对点（CPU），一 rank 一进程 | 本仓无 GPU 路径 |

## 3. 边界条件

| 项 | OSS | 本仓 | 备注 |
|---|---|---|---|
| x 方向 | 任意分解 + 全维 ghost 层；周期或物理边界插件 | 只支持 x 薄板（slab）分解；x 为周期环 | 非周期 x 需另写 haloing，R2 未覆盖 |
| y/z 方向 | 同样走 ghost 交换（可选） | 每 rank 全宽 + `torch.roll` 自周期 | 单 rank 内存 = nz×ny×nx_local，y/z 不分解 |
| ghost 未定义通道 | 不打包 → 不读取 | selective 模式下 ghost 面非穿越通道**未定义**，`validate_ghosts` 只查穿越集合 | 这是可读性/安全性的取舍，写在 `D3Q19GlooTransport.__init__` docstring |
| 失败模式 | 无运行时校验 | 每步校验，失败抛 `RuntimeError`；R2 测试用篡改 ghost 验证「fail closed」 | |
| 物理边界（壁面/进出口） | 插件体系（walberla boundary） | 本类只做周期环；壁面在别的类 | 对标范围仅周期 TGV |

## 4. 移植取舍（trade-off decisions）

1. **只移植打包集合，不移植内核生成器。** OSS 的收益一半来自 pystencils 融合内核
   （省一趟内存流量），本仓用 PyTorch 张量算子，融合不可得；因此 R2 只取可
   严格论证的那一半收益：**线字节 5/19**，并把 exchange 时延单列，避免把内核
   融合的收益记到交换头上。
2. **保留 19 通道默认值。** `selective_ghosts=False` 是默认，现有调用者行为逐字节
   不变（`tests/test_multi_gpu_d3q19_selective_ghosts.py` 与 R1 的
   `test_multi_gpu_d3q19_torchrun.py` 同时通过）；选择性路径是**可选加速**，
   不是语义变更。
3. **ghost 面语义变化必须显式化。** 选择性模式下 ghost 面的非穿越通道未定义，
   任何读它的下游代码都会得到垃圾。所以 (a) docstring 写明，(b) `validate_ghosts`
   只校验穿越集合，(c) 增加 fail-closed 篡改测试。
4. **周期环而非全维 ghost 层。** waLBerla 的 y/z 也走交换，本仓 y/z 全宽 + roll：
   少了两个方向的通信，也少了两个方向的并行度。这是单节点 CPU 场景的取舍，不适用
   于大规模弱扩展比对，故 R2 的扩展性结论只针对 **x 方向**面。
5. **不引入 OSS 的运行期依赖。** 只有测试对拍需要 lbmpy（`pytest.importorskip`），
   生产路径零依赖；因此 `_extend_dir` 是「复刻 + 对拍」而不是 import，避免把
   OSS 变成运行时依赖。
6. **不移植 waLBerla 的 `SerialDataHandling` 周期包装**：它解决的是「非周期分解
   边缘」问题，而本仓的 x 环天然周期；移植它只会增加一层间接。

## 5. 未覆盖（明确声明）

* 角/棱方向的通信（本仓只有 x 方向分解，只用面方向集合；26 方向的派生有测试但
  运行时只走 ±x 两个面）。
* 非周期 x 边界、壁面插件、GPU/MPI（waLBerla）后端。
* OSS 的融合内核性能收益：不做数值对标，只对标线字节与位移等价性。