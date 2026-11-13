## ① 案例介绍
fill-level（Körner/Thürey）自由面 LBM 的**新胞元重建/回填**环节与 OSS `lssfau/walberla`
`src/lbm/free_surface/` 交叉对标。算例=二维锁释放溃坝，纵横比 H/a=2（Martin & Moyce 1952 定义），
全域固壁，域 8a+2 × 2a+2，g=-1e-4（Mach 0.08），τ=0.6（ν=0.0333），ρ_l/ρ_g=1.0/1.0，
转换阈值 κ=0.41，σ=0，步数 1184(a=16)/1025(a=24)，T=t√(2g/a) 覆盖至 2.963。

## ② 怎么做的
- OSS 做法逐条从源码取（零手抄）：`PdfReconstructionModel.h:33-58,62-75`（6 重建模型，判据 n·c_i≥0）、
  `PdfRefillingModel.h:40-66`（5 回填模型）、`PdfRefillingSweep.impl.h:60-110,390-407`
  （EquilibriumRefilling：邻域平均、**排除本步 G→I 邻居**、回退 ρ=1,u=0）、
  `ExcessMassDistributionModel.h:36-90`（9 模型）、`ReconstructInterfaceCellABB.h:53-64`（ABB 公式）。
- 本仓改动（最小、默认位级不变）：`free_surface_lbm_2d._init_new_2d` 增 `refill_policy`
  ∈ {repo_average(默认), walberla_equilibrium, fixed_rho} 并透传至 `koerner_step_2d`/`free_surface_step_2d`。
- 实验脚本 `experiments/xval_filllevel_refill_2d/run_xval.py`（内建 pre-declared design/threshold），
  单线程 CPU，波前判据=列高 Σ_y fill ≥ 0.5 的最右列（抗飞溅液滴）。

## ③ 计算结果
| 策略 | a | ρ_g | rel RMSE | max 逐点 | 存量漂移 | <15% |
|---|---|---|---|---|---|---|
| repo_average | 16 | 0.01 | 289.41 % | 550.41 % | +44.33 % | ❌ |
| fixed_rho | 16 | 0.01 | 63.54 % | 97.07 % | −91.18 % | ❌ |
| repo_average | 16 | 1.0 | 20.89 % | 30.18 % | +0.010 % | ❌ |
| walberla_equilibrium | 16 | 1.0 | 20.89 % | 30.18 % | +0.010 % | ❌ |
| repo_average | 24 | 1.0 | **14.57 %** | 22.80 % | +0.016 % | RMSE ✅ / max ❌ |
| walberla_equilibrium | 24 | 1.0 | **14.57 %** | 22.80 % | +0.016 % | RMSE ✅ / max ❌ |
（全部机读自 artifacts/xval_results.json，零手抄；14 点逐点对照在 JSON 内。）

## ④ 结果图
左：Z(T) 与 M&M(1952) 数字化 14 点对照（a=16/24 两分辨率，ρ_g=1）；右：逐点相对误差 + 预声明 15% 门线。

## ⑤ 注意点
1. **ρ_gas 是压到一切的主参数**：ρ_g=0.01（非压力匹配）时 ABB 重建与液相压力不平衡 →
   存量漂移 ±44%/−91%、波前顶到远壁（Z_end=8.0）。ρ_g=ρ_l 后漂移 0.01–0.016%。
   `benchmarks/bench_fs_2d.py` 的 ρ_g=0.01 应视为缺陷默认（本单未改该文件）。
2. **回填策略在本装置上零敏感性**（三组逐位相同）：G→I 漏斗事件占比极小，
   误差由界面离散分辨率主导 → 无证据时不动默认（`walberla_equilibrium` 作 opt-in 保留）。
3. 旧 `bench_fs_2d.py` 用 flag 存在性判波前且只跑 500 步（T=0.46）却对 T≤2.96 全表插值=外推；本单另立装置。

## ⑥ 误差判定（逐项）
- rel-RMSE：a=24 ρ_g=1 → **14.57 % < 15 %**（达标）；a=16 → 20.89 %（不达标）。
- 逐点最大相对误差：a=24 → **22.80 % > 15 %**（不达标）；a=16 → 30.18 %。
- 因此 **未满足"<3%"**（本单判据为 15% 亦仅单指标勉强通过）→ 总体 **honest FAIL**，
  不做外推粉饰：误差随 a 单调下降（20.9%→14.6%），需 a≈36–48 才可能进 15%，本单预算内未验证。

## 差异清单与移植取舍（摘要）
D0 气体密度：OSS=压力匹配+Δρ=6σκ 限幅 vs 本仓曾用 0.01（已量化为缺陷默认）。
D1 重建判据：OSS 界面法向 n·c_i≥0 vs 本仓"源邻居是否 GAS"(=OnlyMissing)。
D2 重建模型族 6 选 vs 本仓单一路径。D3 回填：已移植 opt-in 与 OSS 逐条对齐（默认关）。
D4 多余质量：OSS 9 模型（含法向加权/liquid 兜底） vs 本仓均分。D5 Δρ 限幅 0.1 未实现。
D6 ABB 公式代数恒等（已核）。D7 自由滑移壁-气组合方向重建未实现。
取舍：D3 语义已就绪但不改默认（零敏感性）；D1/D2/D4 需先有法向场与模型族单元级对照再动。

## 未完成（honest）
a≥32 与 τ 敏感性未跑（工具 120s/调用墙钟 + 后台进程被收割）；静水柱伪速度控制组已实现未运行；
未新增 pytest（预算耗尽）；已验证=三策略同初值单步输出逐位一致、7 次运行 nan_at_step=null。
工件：`experiments/xval_filllevel_refill_2d/`（run_xval.py + artifacts/json/csv/png）、
`docs/XVAL_LBM_FREE_SURFACE_FILLLEVEL_RECONSTRUCT_REFILL.md`。
