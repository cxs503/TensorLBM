# free_surface_dambreak_wigley — 2D NS+VOF 自由面验证面（非 verified，PARTIAL）

**状态：PARTIAL / honest**（判据 a、b、e、f PASS；判据 d honest-FAIL；判据 c 未交付）。
按仓库纪律**不进 `benchmarks/verified/`**——只有全部判据达标才可迁入。

模块：`src/tensorlbm/free_surface.py`（交错 MAC 两相 NS + 守恒 VOF + 变密度多重网格投影）。
报告：`docs/free_surface_dambreak_vof_report.md`（论文级六要素 + 判据对照表）。
平台报告：`/tensor/reports` id 12313。

## 判据现状（数字全部机读自 `runs/p351w1_n07/summary.json`）

| 判据 | 门 | 实测 | 判定 |
|---|---|---|---|
| (a) 溃坝波前 vs Martin & Moyce (1952) | 相对 RMSE ≤10 % | 7.84 % | PASS |
| (b) 质量守恒 | ≤2 % | 0.12 % | PASS |
| (c) Wigley 兴波波高 @ Fr 0.25/0.31/0.40 | ≤15 % | 未交付 | NOT DELIVERED |
| (d) 界面 L2 收敛阶（3 网格，R²≥0.95） | ≥1.5 | 0.94（R²=0.997） | HONEST FAIL |
| (e) pytest 项数含负控 | ≥15 | 28 | PASS |
| (f) 零领域语义夹具 | `metric_alpha/beta` | 已用 | PASS |

## 脚本

| 脚本 | 用途 |
|---|---|
| `run_dambreak.py` | 单工况溃坝（`--cells/--wall/--nu/--tol`），写 JSON 工件 |
| `run_convergence.py` | 界面输运三网格 L2 收敛（RK4 流映射参考解）→ 阶 + R² |
| `run_snapshots.py` | 6 时刻 `F,p` 场快照（供场量云图） |
| `analyze.py` | 判据表 + 5 张图（零手抄，工件 sha256 入 summary） |
| `sweep_dambreak.sh` / `sweep_fast.sh` | 批量工况（壁面 × 网格敏感性） |
| `publish_report.py` | 发布平台进展报告（六要素 + 图内嵌） |

```bash
python benchmarks/free_surface_dambreak_wigley/run_dambreak.py --cells 16 --wall no-slip
python benchmarks/free_surface_dambreak_wigley/run_convergence.py --sizes 32 64 128
python benchmarks/free_surface_dambreak_wigley/run_snapshots.py --cells 16 --wall no-slip
python benchmarks/free_surface_dambreak_wigley/analyze.py
```

## 为什么不是又一个宽松门

历史资产 `src/tensorlbm/free_surface_verification.py` 的溃坝门比较对象是 `X*=1+t*` 并称其为
Martin & Moyce——**不是实验数据**；`benchmarks/pending/dam_break_2d_multimodel` 已用真实
14 点表证明四模型 4/4 FAIL。本面锚点只认 `T = t√(2g/a)` + Lethe 数字化表，且不外推到 `T>2.96`。

## 下一步（明确）

1. 判据 (d)：换几何重构（PLIC / THINC-BVD）或水平集 + 重初始化；代数 VOF 的界面增厚使 L2 饱和在 O(h)。
2. 判据 (c)：在 `free_surface.py` 骨架之上做三维扩展（`B/L` 与垂向双重加密），并把 Michell 波面
   （不止 `C_W`）实现为参考面；先用 2D+T 横剖面做中间验证步。
3. surface tension：改 balanced-force CSF（Francois et al. 2006）并配毛细波色散锚点后，才可从"已实现"升为"已验证"。