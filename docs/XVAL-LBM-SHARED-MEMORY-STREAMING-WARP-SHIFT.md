# XVAL-LBM-SHARED-MEMORY-STREAMING-WARP-SHIFT

**对标对象**：`PorousMediaSimulation/openLBMPM`（clone commit `3d841891daafe7b6e9f65e51dd9bf20f2b0e1792`，2018-10-15），
shared-memory streaming with warp shift = `ShanChen2D/AccelerateGPU2D.py:1260` `calStreamingGPU`
+ `:2733` `exchangeDistrGlobalGPU`。
**本仓实现**：`src/tensorlbm/warp_shift_streaming.py`（新增），测试 `tests/test_warp_shift_streaming.py`（13 项全绿），
对标驱动 `xval/XVAL-LBM-SHARED-MEMORY-STREAMING-WARP-SHIFT/run_xval.py`（预登记 `predeclared.json`，结果 `results.json`）。
**结论**：**D2 —— 采纳规则（行瓦片共享内存暂存 + 单车道模绕 + 跨瓦片就地修补），但必须闭合两处域边界回绕；原封不动移植（D1）被否决**。

---

## ① 问题 + 参考出处

本仓流式（streaming）是整场 periodic pull（`torch.roll` 一次拷整个 `(9,ny,nx)`）。成熟 GPU 多孔介质 LBM 的
做法是：**一个 CUDA block 处理一行格点**，把这一行的一小段瓦片（宽度 = warp size = 32）的 9 个分布
放进 shared memory，在瓦片内做**单车道位移**（E/NE/SE 交到 lane+1，W/NW/SW 交到 lane−1），
瓦片边缘车道用**模绕**把"要出瓦片"的分布暂存在对侧车道，`__syncthreads()` 后再写回全局：
x 方向的半程留在同一行，y 方向的半程（N/NE/NW 与 S/SW/SE）**直接写进邻行的全局数组**；
跨瓦片的那一份再由 `exchangeDistrGlobalGPU` 在全局数组里就地搬移（西向升序、东向降序，故无需第二个缓冲）。

出处（零手抄，逐行引用，不 vendor 源码）：

| 引用 | 内容 |
|---|---|
| `AccelerateGPU2D.py:1260-1266` | kernel 签名 + `by=blockIdx.y`、`xStart=tx+bx*numberThreads` → 一块 = 一行，一块 = 一段瓦片 |
| `:1268-1276` | 9 个 `cuda.shared.array(shape=(32,), dtype=float64)` 暂存数组（瓦片宽 = warp size） |
| `:1279-1281` | 静止/纵向分布留在本车道：`distrFOut0/2/4[tx]` |
| `:1282-1305` | 单车道位移 + 边缘模绕：`[1][5][8]→tx+1`、`[3][6][7]→tx-1`；`tx==0` 把西向分布写到 lane 31、`tx==w-1` 把东向分布写到 lane 0（暂存语义） |
| `:1306` | `cuda.syncthreads()` 后写回 |
| `:1308-1330` | `fluidDistrNew[0|1|3, idx]` 同行；`[2|5|6, nx*(y+1)+x]`、`[4|7|8, nx*(y-1)+x]` 直写邻行，守卫 `by<ny-1` / `by>0` |
| `:1318-1321, :1332-1336` | **y 周期回绕的代码存在但被注释掉** → 该 kernel 本身 y 不周期 |
| `:2733-2760` | `exchangeDistrGlobalGPU`：跨瓦片搬移，西向升序 / 东向降序，守卫 `xStart<nx and xTarget<nx` → **域边缘两列永不修补** |

许可证：clone 内**无 LICENSE 文件**（`ls LICENSE*` 为空），故不 vendor 源码，只用行号引用 + 语义转写。

## ② 方法与实现

新增 `src/tensorlbm/warp_shift_streaming.py`，三个实现臂（变量只有"流式算子"）：

| 臂 | 函数 | 语义 |
|---|---|---|
| A1 as-is | `warp_shift_stream(..., x_periodic=False, y_periodic=False)` | 上游规则的**逐语句转写**（含边缘车道模绕不被修补、y 边界守卫） |
| A2 修正 | `warp_shift_stream(..., x_periodic=True, y_periodic=True)` | 同规则 + 闭合两处域回绕（移植取舍） |
| A3 参考 | `pull_stream`（`torch.roll` 整场周期 pull） | 本仓既有流式算子（控制臂） |
| 转写基准 | `oss_literal_stream` | 按 tile/lane 双重循环、按上游写出顺序与交换顺序的**逐行转写**，用来证明"向量化形式确实是同一条规则" |

向量化形式：`reshape(ny, nbx, tile)` 后 `torch.roll(±1, dims=2)`（= 瓦片内模绕），
再 `shifted[:, 1:, 0] = blk[:, :-1, tile-1]`（跨瓦片修补，等价 `exchangeDistrGlobalGPU`）；
周期模式追加 `shifted[:, 0, 0] = blk[:, -1, tile-1]`；纵向半程按行切片写入邻行。
碰撞与平衡态复用 `tensorlbm.d2q9.equilibrium/macroscopic`（float64 精确权重）。

## ③ 参数表（无量纲数完整）

| 量 | 值 | 说明 |
|---|---|---|
| 网格 | `L × L`，`L ∈ {64,128,256}` | 周期双方块，`nx % tile == 0` |
| 格子 | D2Q9，`cs² = 1/3` | `0=rest,1=E,2=N,3=W,4=S,5=NE,6=NW,7=SW,8=SE` |
| 松弛 | `τ = 0.8` | ⇒ `ν = (τ−0.5)/3 = 0.1` |
| 初场 | Taylor-Green：`u_x=u₀ sin kx cos ky`、`u_y=−u₀ cos kx sin ky`，`ρ=1−0.75u₀²(cos2kx+cos2ky)` | `k = 2π/L` |
| 振幅 | `u₀ ∈ {0.02, 0.04, 0.08}` | ⇒ `Ma = u₀√3 ∈ {0.0346, 0.0693, 0.1386}`（压缩性可忽略～中等） |
| Reynolds | `Re = u₀L/ν ∈ {12.8 … 204.8}` | 见 results 逐行 |
| 步数 | `K3` 800 步；衰减曲线 400 步；K4 10 步（至发散） | 固定时间推进，无亚松弛 |
| 瓦片宽 `tile` | 8/16/32/64/128（上游固定 32） | 性能旋钮；周期模式下不影响结果（实测 bit-exact） |
| 精度/线程 | float64，`torch.set_num_threads(1)` | 时间臂单线程，避免超订 |

## ④ 对照表（ref vs got + 判定）

**K1 语义同一性**（A1 vs 逐行转写 `oss_literal_stream`，字段 random 96×40 与 TG L=64，tile 8/16/32）

| 字段 | tile | max\|Δf\| | bit-exact | 判定 |
|---|---|---|---|---|
| random_96x40 | 8 / 16 / 32 | 0.0 | ✔ | ✔（阈值 1e-12） |
| taylor_green_L64 | 8 / 16 / 32 | 0.0 | ✔ | ✔ |

**K2 物理中性**（A2 vs 本仓 pull 参考）

| 臂 | tile | max\|Δf\| | bit-exact | 判定 |
|---|---|---|---|---|
| 单步（random & TG） | 8/16/32 | 0.0 | ✔ | ✔ |
| 耦合 200 步 BGK（TG L=64） | 16, 32 | 0.0 | ✔ | ✔ |

**K3 对解析解**（E(t)=E₀exp(−2λt)，`λ_analytic = 2νk²`，800 步）

| L | u₀ | Ma | Re | λ_ref | λ_got | 相对误差 | <15% |
|---|---|---|---|---|---|---|---|
| 64 | 0.02 | 0.0346 | 12.8 | 1.927657e-3 | 1.929709e-3 | 0.106 % | ✔ |
| 64 | 0.04 | 0.0693 | 25.6 | 1.927657e-3 | 1.930337e-3 | 0.139 % | ✔ |
| 64 | 0.08 | 0.1386 | 51.2 | 1.927657e-3 | 1.932899e-3 | 0.272 % | ✔ |
| 128 | 0.02 | 0.0346 | 25.6 | 4.819143e-4 | 4.823406e-4 | 0.088 % | ✔ |
| 128 | 0.04 | 0.0693 | 51.2 | 4.819143e-4 | 4.825554e-4 | 0.133 % | ✔ |
| 128 | 0.08 | 0.1386 | 102.4 | 4.819143e-4 | 4.834941e-4 | 0.328 % | ✔ |
| 256 | 0.02 | 0.0346 | 51.2 | 1.204786e-4 | 1.210203e-4 | 0.450 % | ✔ |
| 256 | 0.04 | 0.0693 | 102.4 | 1.204786e-4 | 1.223954e-4 | 1.591 % | ✔ |
| 256 | 0.08 | 0.1386 | 204.8 | 1.204786e-4 | 1.278669e-4 | **6.133 %** | ✔（最差） |

质量守恒：9 个算例 `|ΔM|/M ≤ 5.6e-14`。

**K4 as-is 移植的缺陷**（L=64, u₀=0.04, τ=0.8）：10 步后质量漂移 **7.34 %**（4096 → 3795.37），
**第 16 步出现非有限值**；与周期臂相比内部已差 0.2527、边界差 0.4613 ⇒ 边缘列/行携带"从未到达"的分布。

**K5 吞吐（单 CPU 线程，float64，中位数 / 5 次重复）**

| L | pull 参考 MLUPS | 最佳 warp 臂 | 相对 pull | 时间散布(max−min)/med |
|---|---|---|---|---|
| 128 | 1.65 | tile 64：1.49 | 0.90× | ≤ 0.22 |
| 256 | 1.55 | tile 8：1.53 | 0.99× | ≤ 0.11 |

⇒ **CPU 上无收益**（0.78–0.99×，全部 < 1），5 次重复的散布最高 22%，故不声称任何加速。

## ⑤ 收敛性 / 参数敏感性（≥2 点）

1. **振幅（Mach）敏感性**：同一 L 下误差随 u₀ 单调增——L=256 时 0.450 %（Ma 0.035）→ 1.591 %（Ma 0.069）
   → 6.133 %（Ma 0.139）。这是 D2Q9 BGK 的 O(Ma²) 压缩性误差，不是流式算子的误差（K2 已证算子 bit-exact）。
2. **分辨率敏感性**：固定 Ma 时误差随 L 变化（Ma 0.069：0.139 % → 0.133 % → 1.591 %），
   L=256 的抬升来自 800 步内累计的有限分辨率/有限时间拟合偏差；三条曲线全部在 15 % 内（`fig_k3_error.png`）。
3. **瓦片宽敏感性**：8/16/32/64/128 在周期模式下结果 **bit-exact 相同**（`test_tile_width_does_not_change_periodic_result`），
   只影响吞吐（K5）。

## ⑥ 失败模式与工件

* **原封不动移植（D1）必失败**：上游 kernel 只对内部一致——域边缘两列（`exchangeDistrGlobalGPU` 守卫跳过）
  与首末两行（`by<ny-1`/`by>0` 守卫）的分布永不修补 → 质量 10 步掉 7.34 %、16 步发散。上游靠其后接的
  边界条件 kernel 覆盖这些格点；本仓若照抄则无此阶段。
* **无 GPU**：`torch.cuda.is_available() == False`，上游 CUDA kernel 无法执行；OSS 臂是逐行转写而非二进制执行，
  这是本对标的**比较上限**（如实记录，不升级为"跑过 OSS"）。
* **CPU 上瓦片暂存无收益**（K5）：共享内存的意义在 GPU；张量化的 staged 拷贝在 CPU 上多一份内存流量。
* **计时受宿主负载干扰**：本机同时跑多个 mission，K5 单臂散布最高 22%，因此判据要求 `spread < 0.20` 才算通过，实际未通过。
* 工件：`xval/XVAL-LBM-SHARED-MEMORY-STREAMING-WARP-SHIFT/{predeclared.json, run_xval.py, results.json, make_figures.py,
  fig_operator.png, fig_tg_decay.png, fig_k3_error.png, fig_timing.png, DIFF_OSS_VS_TENSORLBM.md}`；
  代码 `src/tensorlbm/warp_shift_streaming.py`；测试 `tests/test_warp_shift_streaming.py`。

## 本仓 vs OSS 差异清单（做法 / 参数 / 边界条件）与移植取舍

见同目录 `xval/XVAL-LBM-SHARED-MEMORY-STREAMING-WARP-SHIFT/DIFF_OSS_VS_TENSORLBM.md`（12 条逐条对照）。

## 生态摩擦（如实记）

1. **知识库引擎离线**：`GET /api/v1/tensor/knowledge/search?q=warp` 返回 `engine_offline`（`127.0.0.1:8021` 拒连）
   → 开工三查的 capability 一查无法执行，只能用 `negative-knowledge/search`（`openLBMPM` 0 命中）+ 仓内 grep 代替。
2. **工作树无外网**：`raw.githubusercontent.com` 返回 000；OSS 源码只能复用宿主 `/tmp/openLBMPM` 的既有 clone
   （commit `3d84189` 与既有 porous 转写一致），并据此逐行引用。
3. **shell 120 s 墙**：单次 `run_shell` 超时即整组收割，10 分钟量级的对标必须 `setsid nohup` 脱离进程组才能跑完；
   本单实际 230.7 s（`results.json: wall_s_total`）。
4. **宿主超订**：默认 torch 线程数下 256² 每步 90 ms、`set_num_threads(1)` 后 46 ms——不锁线程会得出相反的吞吐结论。

## 误差判定（显式回答）

| 关键量 | 基准 | 误差 | 是否 < 3% |
|---|---|---|---|
| K1 语义同一性 max\|Δf\| | 上游逐行转写 | 0.0（bit-exact） | ✔ 是 |
| K2 算子等价性 max\|Δf\| | 本仓 pull 参考 | 0.0（bit-exact） | ✔ 是 |
| K3 衰减率 L=64/128（Ma ≤ 0.069） | 解析 `2νk²` | 0.088–0.139 % | ✔ 是 |
| K3 衰减率 L=256, Ma 0.069 | 解析 | 1.591 % | ✔ 是 |
| K3 衰减率 L=256, Ma 0.139 | 解析 | 6.133 % | ✘ 否（>3%，但 < 15% owner 带；来源是 D2Q9 压缩性误差） |
| 质量守恒（周期臂） | 初始质量 | ≤ 5.6e-14 | ✔ 是 |
| K4 as-is 质量漂移 | 初始质量 | 7.34 %（10 步） | ✘ 否（as-is 否决依据） |
| K5 吞吐 | pull 参考 | 0.78–0.99× | — CPU 无收益，不声称加速 |