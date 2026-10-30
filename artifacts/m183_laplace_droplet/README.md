# m183 · laplace_droplet sweep artifacts

Mission 183 的独立复现工件（案例 `benchmarks/verified/laplace_droplet`，模型/判据未改动）。
文档：`docs/m183_laplace_droplet_sweep.md`；驱动脚本在案例目录（`sweep_m183.py` 等）。

| 文件 | 内容 |
|---|---|
| `sweep_result.json` | 主结果：逐半径 Δp/R_eq/σ_i、Laplace 拟合与判定、与 `result.json` 基线的复现对比与趋势预测、测量口径敏感性、域尺寸独立性、被排除 run 清单（程序化生成，报告中的数字全部来自此文件） |
| `mechroute_variants.json` | honest-negative：朴素 SC 离散压强张量（v1/v2/v3 三种写法）力学路线 σ 与 Laplace σ 的偏差（46%–126%，不用于判定） |
| `timing_probe.json` | 各配置实测 ms/step 与案例缺省全量串行成本（mission 25 失败诊断的定量依据） |
| `fig1_fields_profiles.png` | 2D 全场 / 3D 中心切片密度云图 + 壳平均 ρ(r) 剖面 |
| `fig2_laplace_fit.png` | Δp vs 1/R（2D/3D）过原点与自由拟合，叠加仓库 verified 基线点 |
| `fig3_reproducibility.png` | 逐半径 σ_i 复现对比与相对偏差（3% 门线） |
| `fig4_mechanical_route.png` | 力学路线（honest-negative） |
| `fig5_convergence.png` | Δp(t) 收敛历程 |
| `fig6_domain_and_cost.png` | 周期盒尺寸敏感性 + 单步成本实测 |

原始逐 run 输出（`run.json` / `history.json` / `profile.json` / `snapshot.npy`）在
`runs/m183_laplace/`（被中断的 run 归档在 `runs/m183_laplace/_aborted/`），域敏感性专项在
`runs/m183_laplace_domain/`。

复现（CPU，约 5 分钟 2D + 20 分钟 3D）：

```bash
cd benchmarks/verified/laplace_droplet
for R in 15 25 40; do python sweep_m183.py --dim 2 --R $R --threads 16 --out ../../runs/m183_laplace/2d_R$R & done
for R in 12 15 18; do python sweep_m183.py --dim 3 --R $R --threads 16 --out ../../runs/m183_laplace/3d_R$R & done
python analyze_m183.py && python mechcheck_m183.py && python plot_m183.py && python report_m183.py
python -m pytest tests/test_m183_laplace_droplet_sweep.py -q   # 在仓库根目录
```