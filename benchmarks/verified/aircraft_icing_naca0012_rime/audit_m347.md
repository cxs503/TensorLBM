# SDAA 扫荡审计: aircraft_icing_naca0012_rime

**班次**: m347  
**审计日期**: 2026-10-29  
**报告ID**: 264  
**判定**: ✅ PASS (11/11 gates)

## 审计范围

独立审查 `benchmarks/verified/aircraft_icing_naca0012_rime/` 案例的全部证据：
- `result.json` — 汇总门判据与9案结果
- `evidence/<run>` — 9案机器档案(JSON)
- `figs/` — 6张结果图
- `run.py` — 9案生产入口

## 审计方法

1. 逐案提取evidence JSON，核对result.json中的门判据数字
2. 生成4张matplotlib图(beta曲线/网格收敛/LWC线性/确定性)
3. 程序化注入所有数字，零手抄
4. 发布论文级报告至 POST /api/v1/tensor/reports (ID=264)

## 门判据复核

| 门 | 判据 | result.json值 | evidence复核 | 结论 |
|---|---|---|---|---|
| G1 | Lagrangian闭合≤1e-12 | 0.00e+00 | 9案均为0 | ✅ |
| G2 | Eulerian闭合≤1e-6 | 2.11e-08 | 9案均<1e-7 | ✅ |
| G7 | 逐位确定性 | 15/15 sha256等 | det_run1==det_run2 | ✅ |
| G9a | 沉积log-log∈[0.97,1.03] | 0.9857 | 由3案LWC拟合 | ✅ |
| G9b | R²≥0.999 | 0.9998 | 线性极好 | ✅ |
| G9c | 比例偏差≤3% | +1.66% | 通过 | ✅ |
| G9d | 沉积单调递增 | true | 0.886<1.758<3.473 | ✅ |
| G10a | t_max阶梯单调 | 8.33<10.42<10.49 | L1<L2<L3严格 | ✅ |
| G10b | Richardson L3误差≤3% | -0.19% | |10.488-10.508|/10.508 | ✅ |
| G10c | Richardson L4误差≤3% | -0.86% | |10.418-10.508|/10.508 | ✅ |
| G10d | Richardson复现先验 | p=4.594,f_ext=10.508 | 与prior完全一致 | ✅ |

## 关键发现

1. **VERIFIED状态成立**: 所有11项门判据PASS，数字与evidence一致
2. **无外部实验锚**: 判据族为内部一致性(Richardson+LWC线性+确定性)
3. **已披露非门项**: frozen超线性(log-log=1.463)、L4量化振荡——如实披露
4. **GPU依赖**: 需CUDA(RTX 5090)，本审计环境无GPU，基于已有evidence复核

## 未覆盖案例队列(字母序)

已覆盖15个(含本次): acoustics, aircraft_icing_naca0012_rime(本次), laplace_cac_phasefield, maneuvering_steady_turn, permeability, permeability3d, poiseuille_3d_duct, poiseuille_3d_pipe, sphere_re100, stokes_second_problem, suboff_re1000, taylor_green_3d, thermal_cavity, thermal_cavity_3d, womersley

下一批未覆盖(字母序): aircraft_icing_naca0012_rime(✅本次), cavity, cavity_3d_full, cavity_natural_convection, couette_2d, ...
