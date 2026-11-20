# DE-X-CAVSWEEP · 16 臂气穴驱动敏感度扫描 · 分析定谳
> 战役 DE · 单滴入深水池 Santini case-3 · 2026-11-20 · TensorLBM agent/m26392
> 预注册 2026-11-20（协调者太初卡 0-15 并行，cells=8 span4）· 本单=回收+判读+写卡
> CROSS 已读：T4_DIGEST（u_limit=2.5·speed 经验值未扫描）+ ABBS（ABB 动量非误差源，补偿 3.23 点）
> 自检：5/5（文件在且非空/数字可溯/双写commit/预测区间/失败如实）

## 1. 背景与判读问题
- 精度缺口定谳：ctrl 峰深 1.4352D vs 实验 1.8574D（-22.73%，恒定项=气穴欠深）；ABB 动量注入与网格（8→12 仅 0.6pp）已排除。
- 本表补 T4 自认缺口：u_limit 扫描 + 释放速度/池深/交换模式/格速/最近面因果方向。
- 判读问题：①哪个旋钮把峰深推向 1.8574D；②u_limit 0.06→0.24 单调补偿=红灯 / 平坦=纯稳定器绿灯；③组合臂交互。

## 2. 臂表（每数字标来源臂；exp_peak=1.8574D；全 cells=8；峰=history 内 max depth_over_D）
| 臂 | 变量 | stop | t_end | peak_D@t | contact | mach_max | drift_end |
|---|---|---|---|---|---|---|---|
| ctrl | base(UL.15) | window_complete | 24.552 | 1.4352D@11.091 | 45 | 0.2598 | 4.8e-6 |
| ul006 | UL 0.06 | health_limit | 9.315 | 1.2942D@8.975 | 45 | 0.1327 | 1.01e-4 |
| ul010 | UL 0.10 | window_complete | 24.552 | 1.4352D@11.091 | 45 | 0.1732 | 3.0e-5 |
| ul020 | UL 0.20 | health_limit | 14.410 | 1.4352D@11.091 | 45 | 0.3006 | 1.1e-5 |
| ul024 | UL 0.24 | health_limit | 14.410 | 1.4352D@11.091 | 45 | 0.3006 | 1.1e-5 |
| rs135 | RS 1.35 | health_limit | 18.873 | 1.3530D@10.604 | 49 | 0.2003 | 1.05e-4 |
| rs175 | RS 1.75 | health_limit | 0.036 | 0.0005D@0.024 | — | 0.1318 | 1.02e-4 |
| gap12 | gap 0.12 | health_limit | 0.012 | 0.0D@0.0 | — | 0.1141 | 2.79e-3 |
| pd20 | PD 2.0 | health_limit | 0.036 | 0.0004D@0.024 | — | 0.1152 | 1.26e-4 |
| caponly | SMe off | health_limit | 1.301 | 0.2343D@1.289 | 45 | 0.1432 | 1.03e-4 |
| simonly | cap off | health_limit | 3.174 | 0.6273D@3.162 | 45 | 0.1383 | 1.06e-4 |
| nearest | NS on | window_complete | 24.552 | 1.4352D@11.091 | 45 | 0.2598 | 4.8e-6 |
| spd005 | spd 0.05 | window_complete | 24.555 | 1.3310D@10.235 | 54 | 0.2165 | 2.1e-5 |
| spd007 | spd 0.07 | health_limit | 0.043 | 0.0005D@0.028 | — | 0.1343 | 1.04e-4 |
| ul012pd20 | UL.12+PD2.0 | health_limit | 0.036 | 0.0004D@0.024 | — | 0.1152 | 1.26e-4 |
| rs135ul020 | UL.20(重复) | health_limit | 14.410 | 1.4352D@11.091 | 45 | 0.3006 | 1.1e-5 |

## 3. 判读
- **u_limit=纯稳定器（绿灯）**：峰深在 0.10–0.24 平坦=1.4352D（ul010/ctrl/ul020/ul024 同值，平台 9.95–11.70ms），非单调补偿向 1.8574D；ul006(0.06)=1.2942D 为下界（9.3ms 死，早于实验峰时 10.394ms）。u_limit 只控存活：0.06 死 drift 门、0.20/0.24 死 Ma 门（0.3006>0.3，精确复现原始 cal8）、0.10–0.15 存活。
- **无旋钮把峰深推向 1.8574D**：最深 1.4352D（6 臂），全 ≤1.4352D；22.7% 缺口=系统性气穴欠深（T5 13–23% 偏差同源），非可调参数——与 ABBS 裁决一致（常数项=气穴欠深）。
- **因果方向**：RS↓1.35→峰深↓1.3530（较 ctrl -5.7%）且 contact 延后 49；RS↑1.75 初始化死；spd↓0.05→峰深↓1.3310（较 ctrl -7.2%）；spd↑0.07 初始化死；PD2.0/gap0.12 初始化死（drift 1.3e-4/2.8e-3）；nearest-surface 无感（与 ctrl 逐位同）。
- **交换模式=质量守恒必要条件（实证）**：caponly(SMe off) 1.301ms 死、simonly(cap off) 3.174ms 死，均触 1e-4 drift 门——两者皆不可关。
- **交互**：ul012pd20 与 pd20 逐位同（PD2.0 主导，UL 触水前无感）；rs135ul020 与 ul020 逐位同（重复性对照过，确定性成立）。

## 4. 诚实 FAIL
- 12/16 臂死 health 门（8 臂 drift>1e-4 门 + ul020/ul024/rs135ul020 死 Ma 门 + rs135 死 drift 门），全部如实入账，不复活不插值。
- ul006 峰 1.2942D 为下界（死于实验峰时前），真峰未知；numeric_pass=False：0/16 臂过 3% 门（最深 1.4352D，误差 22.73%）。

## 5. 细扫二表建议
- 16 臂证明缺口非参数可调（u_limit 平台、RS/spd 单调变浅）→ 参数细扫预期增益 <1pp，不建议；方向应转内核气穴深度物理（ABB 重构/界面动力学）。
- 若仍要参数细扫：u_limit∈{0.12,0.13,0.14}×RS∈{1.45,1.53,1.61} 共 9 臂，~32min/臂，太初卡 9 卡并行 ~35min。

## 6. 复现
- 原始数据：宿主 /tmp/cavsweep/<arm>.json（node099，16/16 齐）；本地副本 runs/cavsweep/*.json（md5 对账过）；提取/绘图 benchmarks/water_entry/single_drop/cavsweep_extract.py + cavsweep_figures.py（数据目录缺省=repo runs/cavsweep/）。
