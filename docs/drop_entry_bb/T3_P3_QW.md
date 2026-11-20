# T3 议事厅队 · Qwen族处方三派 P3（DE-T3-P3RR）
锚点=NEW:DE-T3-P3RR；靶子=A3 首要失效机制：自由面路径存活（window_complete 24.005ms，A3§2）但定量超差（result.json：passed_3pct_and_converged=False、converged_grid_pairs=[]、digitization_resolves_3pct_at_every_point=False）；作者=qw38（空心#3替补）。

## 1. 假设
**H1（主）时原错配**：自由面路径时间轴自释放步起算（free_surface_probe.py L111 `time_ms=1000*step*dt_s`），score.py 把接触后曲线重标到"首个过健康门样本"（score.py L57 `valid[0]`）而非协议时原"首连通液桥"（A1§3）；健康门（mach≤0.3/depth≥0/drift≤1e-4，A2§3）剔除撞击尖峰前 k 步，整条深度-时间曲线相对实验 18 点水平平移 k·dt（8 cells/D 时 dt≈0.0081ms，k≈30–100 → 0.24–0.81ms），早期增长段（斜率~0.27 D/ms）逐点误差超 3%——形状对、相位错。
- 支持：A4 坑①本库实犯时间轴口径错（m163 缺√2）；A1§3 显式规定 time_origin=首连通液桥、无后续时移；撞前速度误差仅 0.68%（wide8，A3 表）→ 撞前物理正确，误差在接触后注册。
- 反证：若峰值深度值本身 >3%，纯平移解释不了（平移保峰值）；若平移非常数（时标错），常数重标无效。
备选（保留至判别）：
- H2 时标错：dt=speed·dx/U（run.py L37）以 U=1.4m/s 映射物理时间，映射错则曲线时间拉伸、峰值时刻偏离 10.394ms；判别=|t_peak_sim−10.394|>0.3ms。
- H3 欠分辨：8–12 cells/D、界面宽 4 格，腔尖 1–2 格；A4⑤⑥残差由采样瞬态主导、界面胞膨胀；两档收敛未达（converged_grid_pairs=[]）。
- H4 重力双重计力：排除——g_eff=1.5g 时撞速应偏 ~22%，与 0.68% 矛盾（A4 坑③）。
排序依据：H1 判别最便宜（纯后处理）且有本库前科；H2/H3 需重跑。

## 2. 处方
- a) `free_surface_probe.py#probe()` L111：history 行增列 `"t_post_ms": 1000*(step-tracker.contact_step)*sc["dt_s"]`（接触前为 null）。
- b) `score.py#assess()` L57：重标基准由 `valid[0]["time_ms"]` 改为接触时刻——取 history 中 `step==run["contact_step"]` 行的 time_ms 为 t_contact，`t=[r["time_ms"]-t_contact for r in valid]`。
- c) `free_surface_probe.py#probe()` L165：平面输出目标改 `round((sample_targets_ms[i]+t_contact_ms)/(1000*dt_s))`（现为绝对时刻，落在实验时刻之前）。
- 为什么能过 3%：撞速已 0.68%（A3 表）→ 物理对；本处方只改时间注册不改求解器 → 接触后动力学不变，最陡的早期点恢复；峰值值不受平移影响。

## 3. 预测
- postcontact_healthy_ms：处方只动测量/评分 → 5 个 window_complete run 保持 **24.00–24.01 ms**（stop_ms=24，run.py L180）；health_limit 各 run 保持 0.7–19.7 ms 不变。若走证伪分支（H2 改 dt_s）→ 区间 **21.6–26.4 ms**（±10%）。
- 峰值深度误差：H1 成立时峰值值平移不变 → 峰值误差维持现值（现值未入 A3 卡，不虚构）；逐点误差降至 **≥15/18 点 <3%**。若现峰值误差已 >3% → 残差区间 **[3%, 8%]**（σ/分辨残差，We=62.1 为 O(1/We) 量级），需第二处方（σ 扫描或 16 cells/D）。
- 可证伪判据：重标后最优常数 δt 仍使 ≥4/18 点 >3%，或峰值时刻偏离 10.394ms 超 0.3ms 且曲线形状（非仅相位）不同 → H1 证伪，升级 H3/H2。

## 4. 成本
- Step 0 判别（纯后处理，零求解器成本）：对既有 8 cells/D run 的 depth 曲线与 fig5_experiment.csv 18 点做互相关，扫 δt∈[−2,+2]ms 报最优 δt 与最大逐点误差（<1 min）。
- 最小验证跑（可直跑，已核 CLI）：
  `cd benchmarks/water_entry/single_drop && python3 free_surface_probe.py --cells 8 --speed 0.04 --steps 10000 --gap 0.4 --span 4.0 --pool-depth 3.0 --release-speed 1.4 --out /tmp/de_t3_p3rr_w8.json && python3 score.py /tmp/de_t3_p3rr_w8.json --output /tmp/de_t3_p3rr_w8_score.json`
- 量级：8 cells/D ≈3000 步，CPU 10–60 min（重跑可递交集群）；Step 0 <1 min。

## 5. 风险
- 最可能死点：健康门在接触后首步即通过（δt≈0），H1 证伪，误差为结构性（H3 分辨/H2 σ）→ 本处方零收益，须升级 16 cells/D 或 σ 扫描。
- 次：per-run 深度误差基线不在工作树（result.json 仅 run 名），现峰值误差未知 → 预测只能给条件区间；执行前先从共享镜像取 per-run JSON 定基线。
- 三：重标改变评分口径，新旧分数不可直接比——须并排报双口径，防"分数变好"假象。
