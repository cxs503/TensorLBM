# aircraft_icing_naca0012_rime — NACA 0012 IRT rime 结冰生产点 + 网格收敛阶梯  ✅ VERIFIED

状态：**verified**（2026-10-06）。本案例判据族为**内部一致性与收敛性**（质量闭合、
逐位确定性、LWC 线性响应、网格收敛 + Richardson 外推），全部通过。无外部实验锚——
这正是它与 `pending/aircraft_icing_rg15_ipw2`（有 MCCS 实测冰形锚、存在模型类差距）
的定级差异。

## 物理与求解链

翼型结冰 rime 工况（IRT 生产点）：NACA 0012、弦长 0.5334 m、
V∞ = 67 m/s、AoA 4°、
LWC 0.50 g/m³、MVD 20 µm、
T∞ = -10 °C、曝光 360 s、
Re = 2.5e+06。求解链 = D3Q19 cumulant
空气动力学（c_s = 0.1）+ Eulerian 液滴相（donor2 TVD）
+ Lagrangian 质量账 + macklin rime 密度 + 一维 Messinger
表面热力学（rime 单段生产 `run_rime_icing`）。生产格弦长 128 lu、
dx = 4.1672 mm、Stokes = 0.1551。

## 判据（verdict.json 机器值，零手抄）

| 门 | 判据 | 测得 | 结果 |
|---|---|---|---|
| G1 Lagrangian 质量闭合（9 案最差） | <=1e-12 | 0.00e+00 | PASS |
| G2 Eulerian 闭合（相对） | <=1e-6 | 2.11e-08 | PASS |
| G7 逐位确定性（det 对，15 数组 sha256） | 全等 | 15/15 arrays sha256 equal | PASS |
| G9a 沉积质量 log-log 斜率 | [0.97,1.03] | 0.9857 | PASS |
| G9b 比例拟合 R² | >=0.999 | 0.999832 | PASS |
| G9c 比例偏差 | <=3 | +1.664% | PASS |
| G9d 沉积 vs LWC 单调 | strictly increasing | true | PASS |
| G10a t_max 阶梯 L1<L2<L3 | 严格单调 | 8.334, 10.418, 10.488 mm | PASS |
| G10b Richardson 误差 L3 | |err|<=3% | -0.188% | PASS |
| G10c Richardson 误差 L4 | |err|<=3% | -0.857% | PASS |
| G10d Richardson 复现 IC-A 先验 | |p-prior|,|f_ext-prior| <=1e-9 rel | p = 4.5936, f_ext = 10.50800 mm | PASS |

## 结果（9 案；`evidence/<run>`（无扩展名 JSON 文件））

| run | 格子 | steps | LWC (g/m³) | β_pk | t_max (mm) | Eulerian 沉积 (kg) | wall (s) |
|---|---|---|---|---|---|---|---|
| `b1_prod` | 320×160 | 3000 | 0.50 | 0.911886 | 8.3344 | 1.7577e-03 | 28.8 |
| `ladder_L1` | 320×160 | 3000 | 0.50 | 0.911886 | 8.3344 | 1.7577e-03 | 25.3 |
| `ladder_L2` | 640×320 | 6000 | 0.50 | 0.901062 | 10.4180 | 8.5333e-04 | 50.1 |
| `ladder_L3` | 890×445 | 8334 | 0.50 | 0.944853 | 10.4882 | 6.1442e-04 | 80.3 |
| `ladder_L4` | 1280×640 | 13340 | 0.50 | 0.950230 | 10.4180 | 4.1384e-04 | 187.2 |
| `m20_lwc025` | 320×160 | 3000 | 0.25 | 0.908972 | 4.1672 | 8.8558e-04 | 24.9 |
| `m20_lwc10` | 320×160 | 3000 | 1.00 | 0.921669 | 20.8359 | 3.4729e-03 | 25.0 |
| `det_run1` | 320×160 | 3000 | 0.50 | 0.911886 | 8.3344 | 1.7577e-03 | 24.9 |
| `det_run2` | 320×160 | 3000 | 0.50 | 0.911886 | 8.3344 | 1.7577e-03 | 25.0 |

生产点冰形：t_max = 8.3344 mm（上/下角峰 4.167/
8.334 mm，驻点 12.502 mm）、
n_ice = 15 胞、冰面积 0.092% c²。

## 披露（verdict.json::disclosures，非门）

- **G9 frozen 超线性**：冻结质量对 LWC 的 log-log 斜率 1.463、
  比例偏差 -43.85%, -15.78%, +6.68%——冻结分数随
  供水变化导致超线性，如实披露非门；沉积侧（G9a–d）全过。
- **G9 半值重标定**：frozen 偏离 ×2 期望
  +50.0%, +26.7%、
  沉积 -0.76%, -1.21%。
- **G10 L4 量化振荡**：L4 t_max = 10.41796875 mm 与 L2 逐位相同
  （单胞量化振荡，prereg 披露条款，非门）。

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/verified/aircraft_icing_naca0012_rime/run.py
# run.py 自定位 <repo>/src（无需 PYTHONPATH）；9 runs, ~472 s total on
# one RTX 5090; outputs to <case>/output/<run>/
```

## 目录

- `run.py` — 9 案生产入口（B1 生产点 + B2 阶梯 + LWC 族 + 确定性对）
- `evidence/<run>`（无扩展名 JSON 文件） — 每案机器档案（β 曲线、质量账、映射常数、
  确定性 sha256；判据数字全部出自这些文件）
- `figs/` — 6 图：beta_curve / alpha_impact / ice_overlay / history /
  convergence（B2 阶梯 + Richardson）/ g9_lwc_linearity
- `result.json` — 机器组装汇总（门值 + 披露 + 9 案行 + 溯源 md5）

## Notes

- 判据溯源：`/nfs/wangxi/runs/icing_bm_20261006/verdict.json`
  （md5 128625e189512600b54ce5af3dec385f）；prereg md5 65b3565e6c0bbf450ef50b9ab8660a65。
- 本案例为 rime 单段生产，不存流场云图档案（画廊为 β/冰形/历史/收敛图）。
- `run.py` 自定位 `<repo>/src` 的 tensorlbm（需 IC-E-D1 warmup-bins 修复，
  exp/icing-warmup-bins @ 100f78d6 (IC-E-D1 fix)；对 mono 案与 pristine 树逐位一致，A/B 40/40 数组）。
