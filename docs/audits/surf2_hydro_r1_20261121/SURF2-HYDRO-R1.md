# SURF2-HYDRO-R1 · 界面胞静水压头补偿满窗定谳（m26841 死树抢救续做）

> 仓库 TensorLBM · 分支 agent/m26863 · 锚 = NEW:SURF2-HYDRO-R1
> 前情：m26841 no_progress 死于建臂前；14 行 OPT-IN 静水压补偿 diff 抢救自 BB/tmp/SURF2-HYDRO.kernel.patch（env TL_FS_HYDRO_Z_REF 门控，默认关=位级不变），种子=BB/tmp/DE-X-SURF2-HYDRO.R1.draft.md（已应用，未重写）。
> 战役 = DE·诊断主线 SURF2 系列。本单问：**静水压悖论（界面胞气相密度系统性偏高→气穴欠深）是否是晚窗（7ms+）欠深的一环？**
> 前情定谳（勿重验）：K2 已证「能量送不进去」——收支不闭合，主导丢失通道 = 7_mass_exchange_conversion（份额 248.98%）；MMC4 已证 -22.7% 缺口集中于 7ms+ 晚窗。

## 判决（预登记，跑前写死）

- 判决点 = 实验 1.857434 D @ 10.394 ms（pc 口径，Santini case-3）
- ctrl（env 不设）须位级复现 1.435220 D @ 11.0906 ms（abs 口径）
- **命中（HIT）** = hydro 峰深 |err| < 12% 且存活 > 10.4 ms
- **证伪（FALSIFY）** = 更深但 ≤ 1 胞量化（0.125 D ≈ 实验峰的 ±6.7%）或更浅/早死

## 双臂设置（满窗）

| 项 | ctrl | hydro |
|---|---|---|
| cells / gap / speed / rs / pd / ul | 8 / 0.18 / 0.06 / 1.53 / 3.0 / 0.15 | 同左 |
| steps 上限 | 10000（实际 window_complete ≈ 24 ms pc 停） | 同左 |
| TL_FS_HYDRO_Z_REF | 不设（空） | 24.5（初水面 level = round(3×8)+0.5） |
| 机制 | 无补偿 | rho_g_cav += rho_l·gz·(z−z_ref)/cs²，仅界面胞 |

## 数据溯源

| 数据源 | 路径 | 内容 |
|---|---|---|
| ctrl 结果 | runs/ctrl/result.json + run.log | 满窗 history（depth_over_D 时序），window_complete 24.5525ms |
| hydro 结果 | runs/hydro/result.json + run.log | history 至 22.9716ms（health_limit，drift 门） |
| 实验基准 | benchmarks/water_entry/single_drop/fig5_experiment.csv | 18 测点（pc 口径） |
| 判决输出 | runs/SURF2-HYDRO-R1/verdict.json + fig_depth.png + fig_residual.png | 程序化注入（analyze.py），零手抄 |
| 入库副本 | docs/audits/surf2_hydro_r1_20261121/ | 卡 + verdict.json + figs/ + analyze.py（runs/ 被 .gitignore，原始 result.json 留工作树盘上） |

## 结果

| 量 | ctrl | hydro | 实验 |
|---|---|---|---|
| 峰深 (D) | 1.4352204748 | 1.6366277885 | 1.857434 |
| 峰时刻 (ms, abs) | 11.090571 | 14.082107 | 10.394 (pc) |
| 峰时刻 (ms, pc) | 10.543339 | 13.534875 | 10.394 |
| 峰步 | 912 | 1158 | — |
| vs 实验 | -22.73% | **-11.89%** | — |
| stop_reason | window_complete | health_limit（drift 1.15e-4 > 1e-4 @22.97ms） | — |
| 末时刻 (ms, abs) | 24.5525 | 22.9716 | — |

- **ctrl 位级复现：PASS**（1.4352204748@11.090571 vs 锚 1.435220@11.0906，Δ = 0.000e+00）
- **hydro − ctrl = +0.201407 D = +1.611 胞**（超出 1 胞量化门 0.125 D）
- **hydro 存活 > 10.4 ms：是**（峰在 14.08ms，末时刻 22.97ms）
- 晚窗（7–13ms pc）平均残差：ctrl -20.85% → hydro -12.22%（缺口收窄 8.6 个百分点）
- 判决点 10.394ms 处：ctrl -22.80% → hydro -15.09%

## 判决

**判决：HIT（命中）** —— hydro 峰深 |err| = 11.89% < 12% 且存活 22.97ms > 10.4ms；
加深 +1.611 胞 > 1 胞量化门，非量化假象。

**诚实注记（不粉饰）**：
1. hydro 臂未跑满 24ms pc 窗——22.97ms 处液相体积漂移门（1.15e-4 > 1e-4）触发 health_limit 停跑；
   峰（14.08ms）与判决点（10.394ms pc）均远早于停跑点，判决不受影响，但满窗尾部（23–24.5ms）无 hydro 数据。
2. 补偿把峰时刻从 10.54ms(pc) 推到 13.53ms(pc)——欠深收窄的同时峰被推迟 ~3ms，
   与实验峰时刻（10.394ms pc）的偏差反而增大（ctrl 偏差 +0.15ms → hydro 偏差 +3.14ms）。
   静水压头修正了"深度"，未修正"到达时机"。
3. 补偿只作用于界面胞气相密度（ABB 重建入口），不改质量交换/耗散通道；
   下一步应查 7_mass_exchange_conversion 是否被静水压头间接修正（K2 主导通道）。

## 误差判定（逐项，显式回答 <3%）

| 关键量 | 值 | 基准 | 误差 | <3%? |
|---|---|---|---|---|
| ctrl 峰深 | 1.4352204748 D | 锚 1.435220 D | +4.7e-07 D | **是**（位级） |
| ctrl 峰时刻 | 11.090571 ms | 锚 11.0906 ms | -2.9e-05 ms | **是**（位级） |
| hydro 峰深 | 1.6366277885 D | 实验 1.857434 D | -11.89% | **否**（在预登记 12% HIT 带内） |
| hydro 峰时刻 | 13.534875 ms (pc) | 实验 10.394 ms | +3.14 ms | **否**（无 3% 时刻门，如实报） |

## 实现注记

- patch 原样应用（14 行，free_surface_lbm.py L1218-1231）；另加 1 行空串门控
  （`not in (None, "")`）：共享包装器对 off 臂传空串 env，`float("")` 会崩，空串=off 语义不变。
- 派车：/root/tp-agent/cluster/sdaa_fs_probe.py 共享包装器（--tree 工作树 + env 透传），双臂并行双卡。
