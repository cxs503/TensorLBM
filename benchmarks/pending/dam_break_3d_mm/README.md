# 3D 溃坝 — Martin & Moyce (1952) 验证（free-surface / VOF 共性模块）

**状态：❌ 未达标（口径已修正，2026-09-29 重测）** — 修正了 run.py 两处口径 bug
后重测，误差虽有变化但仍 >50%，共性模块的自由面物理缺陷（**重力无关的伪润湿薄膜**）
是根因，未入库 `benchmarks/verified/`。

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