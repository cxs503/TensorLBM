# 衰减各向同性湍流 dit_turbulence（DIT, E(k) 与 E(t) 双判据）（pending）

**状态：❌ 未达标（根因类：固定 C_s 的 Smagorinsky LES 档间不收敛——N=256 谱斜率 +17.05% 超差、衰减指数 1.693 超带，两档 slope 绝对差 0.256 / n 绝对差 0.244（Cs0.12 轮）；三次正式两档验证（2026-08-19/20）全 FAIL，属 LES 模型/参数系统性问题，非数值 bug）。**

## 物理问题

3D 周期域衰减各向同性湍流（DIT）：随机无散初始场，目标谱
E(k)∝k⁴·exp(-2(k/m0)²)（m0=3, u_rms=0.04），无外力自由衰减。双判据
（Kolmogorov 1941 / Comte-Bellot–Corrsin）：

- 谱斜率：惯性区时间平均 lnE–lnk 拟合斜率 vs **-5/3**，≤3%；
- 衰减指数：E(t)∝t^-n，n ∈ [1.2, 1.5]（Saffman 6/5 / Batchelor 10/7 带）；
- 两档网格（同物理窗）档间收敛。

分子 τ=0.501（Re_L≈65），亚格子耗散由 Smagorinsky 提供（C_s=0.12 基准档）。

## 共性模块

`solver3d.stream3d` + `turbulence.collide_smagorinsky_bgk3d` +
`d3q9(D3Q19).equilibrium3d/macroscopic3d`（bgk 路径，float32——mrt3d 变体有
已知 dtype bug，正式档避开）。**库缺口 G-DIT-1/2**：3D FFT 球壳谱、无散
初始场、偏度分析均 run.py 自实现（库内无对应入口）。

## 计算结果现状（三次正式验证合并；result.json = 第二次 Cs=0.12 正式判定）

| 验证轮 | 档 | slope | err vs -5/3 | n | 档间差 | 判定 |
|--------|-----|-------|-------------|-----|--------|------|
| 1（08-19） | N=64 / 128 | -1.646 / -1.639 | — / +1.69% | 1.610 / 1.449 | n 差 10% | FAIL（N=64 欠分辨阻塞） |
| 2（08-20，Cs=0.12） | N=128 | -1.6385711187348206 | **+1.6857%** | 1.4491344383709526 | slope 差 0.256 / n 差 0.244 | FAIL |
| 2 | N=256 | -1.3825001626134472 | **+17.05%** | 1.6932376806677203 | （同上） | FAIL |
| 3（08-20，Cs=0.13） | N=128 | -1.6828 | -0.97% | 1.4903 | slope 差 17.91% / n 差 13.84% | FAIL |
| 3 | N=256 | -1.3843 | **+16.94%** | 1.6965 | （同上） | FAIL |

- 第三轮先做 N=128 的 C_s 精扫（单调：0.12→+1.69%、0.13→-0.97%、0.14→-4.38%、
  0.15→-7.66%），选达标点 0.13 上 N=256——**N=256 对 C_s∈[0.12,0.13] 不敏感**
  （+17.05% vs +16.94%），C_s 微调不是解。
- N=256 窗口敏感性（result.json `window_sensitivity`）：任何 (k,t) 窗组合都无
  -5/3 平台（k∈[10,44]、t∈[1000,3000] 时 +0.41% 是穿越值非平台；逐快照斜率
  先平后陡非单调）。
- 复现性：N=128 两轮逐位一致（seed=42 确定性）；质量漂移 4.4e-4、IC 无散、
  S3 偏度 -0.36~-0.39（级联签名）、无 NaN——运行本身健康。

## 不达标清单（严格标准：共性模块入口 + 直接观测量 ≤3% + ≥2 档网格单调收敛）

- 共性模块入口：部分 ✓（碰撞/流播为库入口；谱分析/初始场/偏度为 run.py
  自实现，库缺口 G-DIT-1/2 已登记）。
- 直接观测量 ≤3%：✗（正式判据 N=256 谱斜率 +17.05%、Cs=0.13 轮 +16.94%；
  n=1.693/1.697 超出 [1.2,1.5] 带）。
- ≥2 档网格单调收敛：✗（无任何共同窗口使两档同时达标；档间 slope 差
  Cs0.12 轮 0.256 / Cs0.13 轮 17.91%，n 差 0.244 / 13.84%）。

## 根因/诊断

1. **N=64 欠分辨**（第一轮决定性）：LES 亚格子耗散相对级联过强，谱快速变陡、
   无 -5/3 平台、n=1.610 超带。
2. **N=256 欠耗散**（第二/三轮决定性）：分辨率提高后数值耗散下降，固定
   C_s=0.12–0.13 的亚格子耗散相对不足 → 能量在惯性区堆积、谱变平，晚期才
   二次穿越 -5/3；衰减早期偏快（n≈1.69）。
3. **N=128 的 -5/3 是数值耗散补偿的巧合**（短暂穿越后单调变陡：
   `slope_evolution_in_window` 窗内逐点斜率从 -1.464388514697226
   一路降到 -1.8462996899071036）。
4. **C_s 微调无效**（第三轮决定性）：N=256 斜率/衰减指数对 C_s∈[0.12,0.13]
   几乎不变，而 N=128 同区间明显移动——欠耗散缺口远超微调可补偿范围。
5. 综合：斜率演化随分辨率单调平移（N=64 全程偏陡、N=128 短暂居中、
   N=256 先平后陡）→ 档间不收敛是 **LES 模型参数化的系统性问题**。

## 晋级路径

- ①换耗散自适应模型（动态 Smagorinsky / WALE / 壁适应；注意 WALE C_w=3.25
  已扫过：slope -0.336、err 79.8%，欠耗散死路，不适用自由衰减 DIT）；
  ②固定 C_s 路线需**大幅**提高（≥0.18）并重扫 128/256 两档；
  ③多 IC 系综平均（统计收敛）；④更长运行（≥3 万步）看 N=256 晚期平台。
- 均为新验证轮（owner 决策）；达标即归档 PR 移 `pending/dit_turbulence` →
  `verified/`，并顺带清偿库缺口 G-DIT-1/2（谱/无散场入库）。

## 文件与复现

- `run.py` 探针驱动；`case_N{64,128,256}.json` + `data_N*.npz` = 第二轮原始档；
  `cs_scan_20260820/` = 第三轮完整产物（含 C_s 单调表与两档判定）；
  `result.json` = 第二轮合并正式判定 + 全部窗口敏感性表。
- 复现：`python run.py --n 128 --steps 15000 --cs 0.12 --kmin 10 --kmax 24
  --t-dev-mult 146 --t-end-mult 292`（N=256 用 `--n 256 --kmax 44`，需 ~15GB 显存）。

<!-- PROVENANCE
[{"v": -1.6385711187348206, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/result.json", "k": "cases[0].spectrum_fit.slope"},
 {"v": -1.3825001626134472, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/result.json", "k": "cases[1].spectrum_fit.slope"},
 {"v": 1.4491344383709526, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/result.json", "k": "cases[0].decay_fit.n"},
 {"v": 1.6932376806677203, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/result.json", "k": "cases[1].decay_fit.n"},
 {"v": 1.6857, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/result.json", "k": "verdict.slope_err_pct_per_grid.128"},
 {"v": 17.05, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/result.json", "k": "verdict.slope_err_pct_per_grid.256"},
 {"v": 3.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/result.json", "k": "verdict.tol_pct"},
 {"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/result.json", "k": "verdict.verified"},
 {"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/result.json", "k": "verdict.converged_slope"},
 {"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/result.json", "k": "verdict.converged_n"},
 {"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/result.json", "k": "verdict.n_in_physical_band"},
 {"v": 0.0004439353942871094, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/result.json", "k": "cases[0].mass_drift_rel"},
 {"v": 128, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/result.json", "k": "cases[0].n"},
 {"v": 256, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/result.json", "k": "cases[1].n"},
 {"v": -1.6828, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/cs_scan_20260820/result.json", "k": "verdict_cs013_twogrid.slope_per_grid.128"},
 {"v": -0.97, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/cs_scan_20260820/result.json", "k": "verdict_cs013_twogrid.slope_err_pct_per_grid.128"},
 {"v": 1.4903, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/cs_scan_20260820/result.json", "k": "verdict_cs013_twogrid.n_decay_per_grid.128"},
 {"v": 1.6965, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/cs_scan_20260820/result.json", "k": "verdict_cs013_twogrid.n_decay_per_grid.256"},
 {"v": -1.3843, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/cs_scan_20260820/result.json", "k": "verdict_cs013_twogrid.slope_per_grid.256"},
 {"v": 16.94, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/cs_scan_20260820/result.json", "k": "verdict_cs013_twogrid.slope_err_pct_per_grid.256"},
 {"v": 17.91, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/cs_scan_20260820/result.json", "k": "verdict_cs013_twogrid.intergrid_slope_diff_pct"},
 {"v": 13.84, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/cs_scan_20260820/result.json", "k": "verdict_cs013_twogrid.intergrid_n_diff_pct"},
 {"v": -4.38, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/cs_scan_20260820/result.json", "k": "cs_monotonicity_N128.0.14.err_pct"},
 {"v": 1.4973, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/cs_scan_20260820/result.json", "k": "cs_monotonicity_N128.0.14.n"},
 {"v": -7.66, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/cs_scan_20260820/result.json", "k": "cs_monotonicity_N128.0.15.err_pct"},
 {"v": -0.336, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/result.json", "k": "wale_cs_scan.wale_cw325.slope"},
 {"v": 79.8, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/result.json", "k": "wale_cs_scan.wale_cw325.err"},
 {"v": -1.6462383849282989, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/case_N64.json", "k": "spectrum_fit.slope"},
 {"v": 1.6102795202619933, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/case_N64.json", "k": "decay_fit.n"},
 {"v": 0.2560709561213734, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/result.json", "k": "cases[1].spectrum_fit.slope", "k2": "cases[0].spectrum_fit.slope", "op": "diff"},
 {"v": 0.2441032422967677, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/result.json", "k": "cases[1].decay_fit.n", "k2": "cases[0].decay_fit.n", "op": "diff"},
 {"v": -1.464388514697226, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/result.json", "k": "slope_evolution_in_window.128[0][1]"},
 {"v": -1.8462996899071036, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/result.json", "k": "slope_evolution_in_window.128[19][1]"}]
-->
