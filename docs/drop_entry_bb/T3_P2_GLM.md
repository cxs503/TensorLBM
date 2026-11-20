# T3-P2-GLM 处方卡：自由面·容量再分配+最近面速度+扩域 存活修复
作者 T3-P2 GLM；独立提案，仅读 A1-A4 COMPACT 四卡+ROLES，未读别族卡。

## 1. 假设
因果链一句话：自由面路径在射流冲击/飞溅区部分充满胞反复 fill-empty，默认容量再分配关（free_surface_probe.py L38 capacity=False）、界面速度未投影最近面（L39 nearest=False），致界面胞质量表示误差与伪电流累积——volume_drift>1e-4 或 mach_max>0.3 或 DropTracker 丢滴（A2§3 健康表）→ 触水后 0.7~19.7 ms health_limit 早崩（A3§1 五例；1.36 ms 级崩=case3_cac12 mach_limit/case3_flux8）。
依据：A3§4 存活与 cells、速度误差均无相关→杠杆在界面处理而非分辨率/速度匹配；A4§1 溃坝同构教训 refilling EQ→AVG 改善 8.6 点；A4§3 伪电流 u_max 高文献锚 5-25×。
保留竞争假说：H2 池域反射污染（默认 span=4/pool_depth=3 偏小；case3_wide8 幸存支持）；H3 CACSim 路径马赫敏感（case3_cac12 mach_limit，A3§4）——本处方不选 CACSim。

## 2. 处方
改什么（只改调用开关与域参数，不动算法本体）：
- free_surface_probe.py#probe() 调用加 --capacity（容量再分配，L38/L242）+ --nearest-surface（最近面速度，L39/L249）双开；禁用 --simple-exchange（case3_flux8 step_limit+8.59% 反证，A3§1）。
- 域参数 --span 6.0 --pool-depth 4.0（默认 4.0/3.0，A2§1a；case3_wide8 幸存支持扩域抑反射）。
- 网格两档 --cells 8 与 --cells 12（收敛门 minimum_grid_count=2，A1§3）。
- 路径固定 free_surface_step/D3Q19（A2§4；score.py L103 按 solver 前缀取 geometric_volume_drift）。
为何能活过 10.4 ms：capacity 消部分充满胞质量表示误差→volume_drift≤1e-4；nearest 抑界面伪电流→mach_max<0.3；扩域消边界反射→DropTracker 不丢滴。case3_cal12 与 case3_wide8 已示该路径可达 24.005 ms 满窗（A3§2）。

## 3. 预测
- postcontact_healthy_ms ∈ [20.0, 24.01] ms，中心预期 24.0（window_complete 满窗，stop_ms=24，run.py L180）；下界 20.0 由 case3_matched8 无 capacity 亦存活 19.708 ms 锚定（A3§1）。
- 峰值 10.394 ms（A1§3）必被覆盖；peak_error≤3% 且 maximum_each_point_relative_error≤3%（A1§3 门）。
- 可证伪判据：①双开 --capacity --nearest-surface 后仍 health_limit 且 postcontact_healthy_ms<10.394 ms→H1 否定，转 H2/H3；②满窗存活但 peak_error>3%→存活机制对而界面动力学精度不足，加档网格做收敛研究。

## 4. 成本
最小验证跑（8 cells/D 单档先证存活；probe 退出码 0 当且仅当 window_complete 且 geometric_volume_drift≤tol，A2§2b）：
cd /root/tp-agent/work/m26271/benchmarks/water_entry/single_drop && python free_surface_probe.py --cells 8 --capacity --nearest-surface --span 6.0 --pool-depth 4.0 --out /tmp/p2_glm_cal8.json && python score.py /tmp/p2_glm_cal8.json --output /tmp/p2_glm_score8.json
第二档（收敛门）：python free_surface_probe.py --cells 12 --capacity --nearest-surface --span 6.0 --pool-depth 4.0 --out /tmp/p2_glm_cal12.json && python score.py /tmp/p2_glm_cal8.json /tmp/p2_glm_cal12.json --output /tmp/p2_glm_score.json
判读：out JSON stop_reason=window_complete、postcontact_healthy_ms≥10.394；score status=eligible_for_review 为全门通过。

## 5. 风险
- 最可能死处：capacity+nearest 打开后界面胞计数膨胀（A4§3 坑⑥ interface 胞随细化恶化）→步数耗尽 step_limit（case3_flux8 前科）或算力超时。
- 次风险：span 6/pool_depth 4 格点数约×2.3，12 cells/D 档可能超时——先 8 后 12 分步。
- nearest 若抹平射流头部速度，depth 峰值偏低、peak_error 上升——用 --planes 输出核查垂直速度场。
- H2/H3 未被本处方单跑判别：若失败需补 span 敏感性单开关跑与 CACSim 对照跑。
