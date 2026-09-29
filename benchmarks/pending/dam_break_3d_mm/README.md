# 3D 溃坝 — Martin & Moyce (1952) 验证（free-surface / VOF 共性模块）

**状态：❌ 未达标（口径已修正，2026-09-29 重测）** — 修正了 run.py 两处口径 bug
后重测，误差虽有变化但仍 >50%，共性模块的自由面物理缺陷（**重力无关的伪润湿薄膜**）
是根因，未入库 `benchmarks/verified/`。

> ## 🧱 结构性障碍已固化 → `docs/free_surface_architecture_gaps.md`
>
> 本 benchmark 的缺口是**结构性质量通道耦合**（不是参数没调好），已连同障碍 A（2D SCMP
> 本构障碍）、已试过的全部失败杠杆、能力边界表一起固化到
> **[`docs/free_surface_architecture_gaps.md`](../../../docs/free_surface_architecture_gaps.md)**。
> **在按该文档 §4 引入新架构自由度（解耦质量通道 / 独立气相 EOS / 自由压力场）前，请勿重复标量调参。**
>
> 核心证据（commit `6615076`）：X（波前）与 H（残高）**互斥**——36 组 a=8 配置从不同时进 ±3%；
> H 解冻开关是 `TL_FS_LIQ_TO_IFACE`（**非** `WALL_SLIP` 单独）；要 H→0.78 需
> `WALL_SLIP+LIQ_TO_IFACE` 但抑制波前（X 钉在 1.125–1.375）；要 X→1.5954 需关
> `LIQ_TO_IFACE`+`rg≈0.983` 但 H 冻结 0.938。根因：壁面滑移/L-I 守恒交换与波前驱动
> 由**同一质量通道**控制。

## 物理问题

水柱 a×a×2a（如 32×64×32，高度 2a 沿 +y）在角部（x=0, z=0 墙），重力 -y，六面固壁，其余为气体。
无量纲：T = t·sqrt(g/a)，X = x_front/a，H = h_residual/(2a)（初始 ≈1.0）。
参考（Martin & Moyce 1952，经典数字化，取自仓库自身 2D 基准 `benchmarks/bench_fs_2d.py` 的 REF_T/REF_Z）：

| T | X_ref (M&M) | H_ref (M&M，近似) |
|---|-------------|-------------------|
| 1.0 | **1.1** | ~0.78 |
| 2.0 | **1.8** | ~0.55 |
| 3.0 | **2.7** | ~0.37 |

## 本节所做的两处口径修正（run.py）

> 历史口径有两个 bug，导致此前报告的误差不可信，且掩盖了真实信号。

1. **H 测量 bug（已修）**：旧代码
   ```python
   region = wet[:, :, 1 : a // 2 + 1]
   rows = region.any(dim=2).nonzero(as_tuple=False)   # (nz,ny) → (z,y) 对
   h = int(rows[-1, 0].item())                        # ← 取的是 max z 索引，不是 y 高度！
   ```
   `region.any(dim=2)` 把 x 压掉后得到 `(nz, ny)`，`nonzero()` 返回 `(z, y)` 对，
   `[-1,0]` 取的是**最大的 z 索引**而非 y 高度 → H 恒为伪值（旧值恰好接近 1.0 属巧合）。
   **修正**为正确取 y 高度（与 `/root/TensorLBM_feat2/_probe_dambreak.py::measure2` 的
   `h_flag` 一致）：
   ```python
   ys = region.any(dim=2).any(dim=0).nonzero(as_tuple=False)  # (ny,) → y 高度索引
   h  = int(ys[-1, 0].item()) if ys.numel() else 0
   ```
   修正后初始 H = 31/(2·16) = 0.969（水柱 32 层，最高液格 y=31），量纲正确。

2. **参考表 bug（已修）**：旧 `MM_X` 用 `T=1→1.5, T=2→2.7`（把 M&M 的 T=3 值错安到 T=2，
   且 T=1 用了无出处值）。**修正**为经典 M&M 数字化值，与 `benchmarks/bench_fs_2d.py`
   完全一致：`T=1→1.1, T=2→1.8, T=3→2.7`。检查点也从 (T=1,2) 扩到 (T=1,2,3)。
   同时给 `run()` 增加了每个采样区间的进度打印（st/T/X/H/iface/drift）。

## 重测结果（修正口径；原版模块，rho_gas=1.0，g=1e-4，tau=0.8）

运行：`run.py --a 16 --g 1e-4 --steps 1200 --interval 40 --rho_gas 1.0`（T_max=3.0，实测 321 s）
     `run.py --a 32 --g 1e-4 --steps 566  --interval 40 --rho_gas 1.0`（T_max≈1.0）

| a | T | X_sim | X_MM | X 误差 | H_sim | H_MM | mass drift | iface |
|---|----|-------|------|--------|-------|------|-----------|-------|
| 16 | 1.0 | 2.00 | 1.1 | **+81.8%** | 1.188 | 0.78 | −0.8% | 5741 |
| 16 | 2.0 | 3.06 | 1.8 | **+70.1%** | 1.188 | 0.55 | −9.2% | 13907 |
| 16 | 3.0 | 4.06 | 2.7 | **+50.5%** | 1.188 | 0.37 | −28.5% | 22545 |
| 32 | 1.0 | 1.69 | 1.1 | **+53.4%** | 1.219 | 0.78 | −1.4% | 34818 |

> a=32 仅跑到 T_max≈1.0（566 步，1172 s），其 T=2/3 检查点超出模拟范围、为外推陈旧值，
> 不计入误差（已在 result.json 标记 `extrapolated: true`，`max_err` 只统计有效点）。

- **X 仍然虚高 ~1.5–1.8 倍**：正确基准下误差 +50%~+82%，远未达 ≤3% 的达标线（修正口径后 max_err：a=16 +81.8%，a=32 +56.3%）。
- **H 不降反升**：物理上 M&M 残高应降到 ~0.78（T=1）→ 实测 H 由初始 0.97 升到 **1.19 平台**，
  符号都反了 → 说明修正后的 H 测量同样被污染（伪薄膜沿 +y 壁爬升至固定高度后平台化）。
- 网格：a=16 X(T=1)=2.00 vs a=32 X(T=1)=1.69，**无明显收敛**，误差≈网格无关 → 非离散误差而是模型缺陷。
- 质量漂移随 T 单调增大（T=3 时 −28.5%），interface 格数 1311→22545（17×）。

## 根因（本轮确认）：重力无关的伪润湿薄膜

- 波前推进是**与重力无关的伪润湿薄膜**：解析标定 `X−1 = 0.042·T·√(a/g)` 逐点吻合
  （每步 ~0.042 格）；`g=0` 与 `g=1e-4` 轨迹几乎完全重合。
- X(T) 强烈依赖 g（`a=16`：g=5e-5→X(1)≈2.45，g=1e-4→2.00，g=2e-4→1.75），
  **违反 M&M 自相似性**（T、X 均已对 g/a 无量纲化，理论上应重合）。
- 伪薄膜同时沿 x 和对壁（+y）蔓延 → 既把 X 推到虚高（interface 雾污染波前测量），
  又把 H 抬到 1.19 的伪平台。
- 结论：**口径修正不改变物理**；误差仍 >50% 是共性模块自由面/界面处理缺陷所致（非测量问题）。

## 运行方式（复现）

```
cd /root/TensorLBM_feat2
OMP_NUM_THREADS=12 PYTHONPATH=src \
  python benchmarks/pending/dam_break_3d_mm/run.py --a 16 --g 1e-4 --steps 1200 --interval 40 --rho_gas 1.0
# 注意：T = steps·sqrt(g/a)，到达 T=1/2/3 所需步数 ∝ √a。
#   a=16,g=1e-4: 400/800/1200 步 ≈ T=1/2/3；a=32 需 566/1131/1697 步。
```

## 相关文件

- 修正后结果 JSON：`outputs_remeasure/a16_g1e-04_rg1.0_result.json`（含 series/checks）
- 修正探针（多口径前端/高度）：`/root/TensorLBM_feat2/_probe_dambreak.py`
- 原始矩阵数据：`/root/TensorLBM_feat2/_matrix/*.log`、`_probe_a*_g*.json`

## 共性模块缺口（详见 /tmp/dambreak_gap.md）

1. **interface↔interface 质量交换非保守** → 正质量源；去掉后 interface 稳定、漂移 ~1%。
2. **ABB 重建 population 参与质量交换** → 对 rho_gas 病态敏感。
3. **interface 计数爆炸**（to_iface 阈值过低）→ 雾状 interface 层污染波前（X 虚高）。
4. `init_fill_rectangular` 无 z 向壁面（角部水柱需自行补墙）。
5. **伪润湿薄膜**（本轮新确认）：重力无关、逐点可标定 `ΔX=0.042·√(a/g)`/步，
   是 X 虚高与 H 伪平台的直接来源。

## 判定

- 波前误差 ≤3% 且 ≥2 档网格收敛：**不满足**（修正口径后误差仍 +50%~+82%，且网格不收敛）。
- 未写入 `benchmarks/verified/`；本目录保留可复现材料、修正过程与真实数据。

---

## 2026-09-29「三件套守恒化」尝试（负结果，H 冻结真因被证伪）

**假设（上一 agent）**：H 冻结源于自由面质量通道被硬置零（ABB 之后 gas 通道恒零），
拟用「三件套」解冻：① ABB 模式 `eq/eqref/sumeq`（平衡部分钉在 p_gas）+ 小 rho_gas；
② 守恒化的自由面质量通量闭包；③ 标志一致 `TL_FS_LIQ_TO_IFACE`。
目标：H(T=1)→0.78（塌缩）、X(T)→方柱参照 1.5954、质量守恒 <1%、f 有限。

**结论：三件套全部实现并测过，四个判据里只有（c）（d）成立；（a）H 完全不降，假设被证伪。**
H 冻结的真因不是 gas 通道置零，而是**后壁角柱滞留**（见下），gas 通道开或关都无法改变 H。

### ① ABB 模式 × rho_gas（a=8, g=1e-4, T=1）

| ABB 模式 | rho_gas | X(1) | H(1) | drift | iface |
|---|---|---|---|---|---|
| legacy | 1.0 | 1.125 | 0.938 | +8e-7 | 335 |
| eq | 0.975 | 1.500 | 0.938 | −1e-6 | 677 |
| eqref | 0.975 | 1.500 | 0.938 | −3e-7 | 631 |
| **sumeq** | **0.975** | **1.625** | 0.938 | −2e-6 | 818 |
| legacy | 0.95 | 2.375 | 0.938 | −1e-6 | 1412 |
| 任意 | 0.1 | 4.625 | 1.06–1.13 | ~1e-3 | ~4500 |

**X 可由 rho_gas 线性调（sumeq@0.975 → 1.625，对 1.5954 = +1.9%），但 H 在所有 20 组里
都不低于初值 0.938**；rho_gas↓ 只加快波前、把 H 抬到 1.0+（伪薄膜沿顶面爬升）。

### ② 守恒化自由面通量闭包（本轮实现，`src/tensorlbm/free_surface_lbm.py`）

`TL_FS_GAS_CHANNEL` 升级为多模式（默认不设 = legacy，逐位不变）：

- `conserv` / `paired`（**气体侧逐链路配对**）：界面格向气体排出的 `f_abb[q]-f_out[q]`
  在气体 pull 源 `x-c_q` 同量记账（与 L/I bulk debit 同型 roll），全局链路和恒为零。
- `redist` / `paired_redist`：排气量回灌 Körner redistribution 池（派给存活的 interface 邻居）。
- `1` / `naive`：旧的非守恒裸版本（保留复现）。

| a | 配置 | rho_gas | T | X | H | drift | iface | f 有限 |
|---|---|---|---|---|---|---|---|---|
| 16 | base legacy | 1.0 | 0.99 | 1.188 | 0.969 | −1.1e-6 | 1839 | ✓ |
| 16 | sumeq | 0.975 | 0.99 | 1.688 | 0.969 | +1.4e-6 | 5828 | ✓ |
| 16 | **conserv** | 1.0 | 0.99 | 1.250 | 1.000 | −7.3e-6 | 52684 | ✓ |
| 16 | **conserv+l2i** | 0.975 | 0.99 | 1.000 | 0.969 | −8.4e-6 | 88910 | ✓ |
| 16 | naive | 1.0 | 0.99 | 1.125 | 1.031 | +3.8e-3 | 1551 | ✓ |
| 8 | **conserv** | 1.0 | 4.0 | 1.125 | 1.000 | −1.1e-5 | 8476 | ✓ |
| 8 | conserv | 0.90 | 2.0 | 1.000 | 0.938 | +1.9e-7 | 8765 | ✓ |

- **质量守恒（判据 c）✓**：`conserv` 在 rho_gas=0.90–1.0、T≤4 内 drift ≤ 8.5e-5（<0.01%）；
  legacy 通道关闭 ~1e-6。`naive` drift −3.4e-2（旧版确实不守恒，已由 `conserv` 修复）。
  `redist` 在排气壳层没有存活 interface 接收者时（初始 mass=0 壳层全部 to_gas）会把回灌量丢失，
  drift 实测 −0.23~−0.59 —— 该模式需配合更宽接收集，本轮未启用。
- **但（a）H 不降 +（b）X 反被拖死**：`conserv` 把排气量记到气体格 → 气体格涨到 >0.01 被
  `to_iface` 提升为 interface → 伪薄膜（iface 6173→88910，约水柱格数 15–30 倍）→ 波前冻结 X≈1.0–1.25。
  即「守恒化的代价是伪薄膜」，与 naive 版同源。

### ③ 标志一致 `TL_FS_LIQ_TO_IFACE`（已有，本轮复核）

a=8, rho_gas=0.95, g=1e-4：内部 LIQUID 格排到 fill 0.3–0.6 却仍挂 LIQUID 标志；`l2i` 正确把它们
降回 INTERFACE，但 **H 仍钉在 0.938**（T=2 也一样）。

### H 冻结真因（实测，与假设矛盾）

a=8, rho_gas=0.95, +l2i, T=1.1，储水区 x=1..4 / z=mid 的顶层质量图：

```
 y\x   1     2     3     4     5     6     7     8
  15  0.58  0.00  0.00  0.00  0.00  0.00  0.00  0.00   <- 只有后壁角柱 x=1 还有水
  14  0.90  0.78  0.59  0.45  0.38  0.25  0.24  0.62
  13  0.89  0.87  0.82  0.82  0.86  0.93  0.94  0.97
```

- 整个顶层 y=15 在 x≥2 已排空，**唯一残留是后壁角柱 (x=1,z=1)**（fill 0.58 ≥ 0.5）→
  `measure()` 取「最大 y」被这一格钉死 → H=0.938 恒定。
- x=1 列 y=11–14 质量仍 0.89–0.99（几乎未排）→ 壁面滞止（halfway BC 下贴壁柱过黏），
  与 gas 通道无关：**关闭/开启/守恒化 gas 通道，H 都是 0.938**。
- 附加缺陷：`build_domain` 把 mass=0 的 interface 壳层初始化成 `equilibrium3d(rho=1)`
  （而非 rho_gas）→ ABB/排气通量被人为放大（也是 `naive`/`redist` 大 drift 的来源）。

### 本轮判定

| 判据 | 结果 |
|---|---|
| （a）H(T=1) → 0.78 | **FAIL**：所有 30+ 配置 H 恒 = 初值（a=8 0.938 / a=16 0.969），方向都不对 |
| （b）X(T) → 1.5954 | **PARTIAL**：不开 gas 通道时 sumeq+rho_gas≈0.975 可达 1.625(a=8,+1.9%)/1.688(a=16,+5.8%)；开守恒通道后 X 退化到 1.0–1.25 |
| （c）质量守恒 <1% | **PASS**：`conserv` ≤8.5e-5；legacy ~1e-6；`naive` FAIL(−3.4e-2)；`redist` FAIL |
| （d）无非有限 f | **PASS**：全部 30+ 配置 fin=True |

**不写入 `benchmarks/verified/`**。三件套实现了守恒化（c 达标），但**不能使 H 塌缩**；
下一杠杆应指向 **壁面角柱滞止**（`TL_FS_WALL_MODE=halfway` 的贴壁排空）与
**壳层初始化密度**（`build_domain` 的 rho=1 壳层），而非 gas 通道。

### 本轮产物

- 代码：`src/tensorlbm/free_surface_lbm.py`（`TL_FS_GAS_CHANNEL=conserv|redist|naive`）。
- 探针：`runs/fs_pclosure_20260929/probe_fill.py`（竖直 mass/fill/flag 剖面）、
  `probe_map2.py`（顶层 band 质量图）、`probe_redist_dbg.py`（ledger 拆分）。
- 批次日志/JSON：`runs/fs_pclosure_20260929/{batchA..F}.sh`、`A_*/B_*/C_*/D_*/E16_*/F_*`。