# 差异清单：本仓（TensorLBM）vs openLBMPM `calStreamingGPU` / `exchangeDistrGlobalGPU`

上游 commit `3d841891daafe7b6e9f65e51dd9bf20f2b0e1792`（无 LICENSE 文件 → 不 vendor 源码，仅行号引用）。
本仓实现：`src/tensorlbm/warp_shift_streaming.py`。逐条给出差异、判定与移植取舍。

| # | 维度 | openLBMPM（OSS） | 本仓（TensorLBM） | 移植取舍 |
|---|---|---|---|---|
| D1 | 并行模型 | CUDA：`blockIdx.y` = 格点行，`blockIdx.x` = 该行的瓦片，瓦片宽 = 32 = warp size（`AccelerateGPU2D.py:1262-1266`） | 无 GPU：`reshape(ny, nbx, tile)` + `torch.roll(±1, dims=2)` 张量化，按 tile 切块得到同一组合语义 | **采纳规则，张量化实现**：无 GPU 环境下仍需"瓦片内位移 + 跨瓦片修补"的语义，用块级 roll 精确复现（K1 bit-exact） |
| D2 | 流式方向 | push/scatter：把 `f_d(node)` 推到下游格点；因此需要第二个数组，并在 `calStreaming2GPU`（`OptimizedD2Q9GPU.py:539`）整场拷回 | pull/gather：`f_new[d, y, x] = f[d, y-cy, x-cx]`，单次写新数组即可 | 两者数学等价（K2 耦合 200 步 bit-exact 相同）；**本仓保留 pull 作为参考臂**，push 形式只用于复现 OSS 规则 |
| D3 | x 方向位移 | 瓦片内**模绕**：边缘车道把"出瓦片"的分布暂存在对侧车道（`:1282-1305`），随后由 `exchangeDistrGlobalGPU` 搬走 | `torch.roll(+1/-1, dims=2)` 在每块内天然就是模绕 | 采纳：块级 roll == 上游模绕（K1 证明差为 0） |
| D4 | 跨瓦片修补 | 全局数组就地搬移：西向**升序**（`:2739-2746`）、东向**降序**（`:2747-2760`），利用暂存位避免第二个缓冲 | `shifted[:, 1:, 0] = blk[:, :-1, tile-1]`（东向）、`shifted[:, :-1, tile-1] = blk[:, 1:, 0]`（西向）——一次性向量化，顺序无关 | 采纳语义，去掉顺序技巧（张量化不需要就地交换） |
| D5 | **x 域边界** | `exchangeDistrGlobalGPU` 守卫 `xStart<nx and xTarget<nx` → **域最左/最右列永不修补**，只保留瓦片内模绕写入的"从未到达"分布（`:2745-2760`） | `x_periodic=True` 时补 `shifted[:, 0, 0] = blk[:, -1, tile-1]` / `shifted[:, -1, tile-1] = blk[:, 0, 0]` | **必须闭合**：as-is 时质量 10 步漂移 7.34 %、16 步发散（K4）；上游靠后接的边界条件 kernel 覆盖，本仓无此阶段 |
| D6 | **y 域边界** | `if (by < ny-1)` / `if (by > 0)` 守卫（`:1310-1330`），周期 y 的代码被注释掉（`:1318-1321, :1332-1336`） | `y_periodic=True` 时补 `out[d, 0, :] = staged[d, -1, :]` / `out[d, -1, :] = staged[d, 0, :]` | **必须闭合**：同上，不闭合则首末行的 N/S 分布直接消失（K4 的 edge 差 0.4613） |
| D7 | y 方向传输通路 | **不经 shared memory**：直接把瓦片结果写进邻行的全局数组（`fluidDistrNew[2|5|6, nx*(y+1)+x]`），即跨块全局写，要求新旧数组分离 | 同一思路：先做行内 x 位移得到 `staged`，再按行切片写 `out` | 采纳；但注意这正是 CPU 上多一次整场拷贝、K5 无收益的结构性原因 |
| D8 | 参数 | 瓦片宽硬编码 32（九个 `cuda.shared.array(shape=(32,))`），`nx % 32 == 0` 隐含要求 | `tile ∈ {8,16,32,64,128}` 可配，仍要求 `nx % tile == 0`（否则抛 `ValueError`） | 采纳并参数化；周期模式下 tile 对结果无影响（测试 `test_tile_width_does_not_change_periodic_result`），只影响吞吐 |
| D9 | 物理模型 | Shan-Chen 多相（原版/显式受力/彩色梯度/CSF）+ 多达 3 流体、MRT 碰撞、相互作用力 | BGK 单相（本对标只隔离流式算子），力项由 `porous`/Guo 模块另供 | **有意不移植多相**：对标对象是流式机制，多相会引入第二个变量（预登记 "variables_held_fixed"） |
| D10 | 边界条件配套 | Zou-He 恒压/恒速、对流出口、半反弹（README 列出） | 本仓另有边界模块，不在本单范围；周期臂用于物理验证 | 未移植；记为使用前提：**流式算子需与边界条件阶段配套，否则边界格点无定义** |
| D11 | 许可证 | clone 内无 LICENSE 文件（`ls LICENSE*` 空） | — | **不 vendor 源码**，只在文档/代码注释里引用 `文件:行号`；转写为独立实现（`oss_literal_stream`） |
| D12 | 精度/内存 | float64 CUDA；共享内存 9×32×8 B = 2304 B/块 | float64 torch；staged 副本 = 一整个 `(9,ny,nx)` | CPU 上 staged 副本是纯开销（K5：0.78–0.99× pull），故**只在 GPU 后端推荐该规则**，CPU 保留 pull |

## 判定（D2 采纳/否决逻辑）

* K1（语义同一性）bit-exact ⇒ 向量化形式确实是上游同一条规则（不是再推导）。
* K2（物理中性）bit-exact ⇒ 闭合两处回绕后，算子与既有周期 pull 完全等价，切换零风险。
* K3（对解析解）9 个算例最差 6.133 %（L=256, Ma=0.139），全部在 owner 15 % 带内 ⇒ 物理正确。
* K4 ⇒ **as-is（D1）否决**：不闭合边界会立刻破坏守恒并发散。
* K5 ⇒ CPU 上不采纳为默认后端（无收益，散布 22 %）；规则作为 GPU/Triton 后端的候选方案保留。

**一句话**：*规则采纳（D2），实现留在参考臂；as-is 移植（D1）否决；CPU 后端不切换。*