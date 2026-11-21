# DE-NXT-FORCE-W3 · 力评估与边界专线设计卡（编排单，不跑仿真）

主交付：`docs/audits/de_force_w3/TRACK_FORCE_W3.md`（= `/root/tp-agent/drop_entry/BB/NEXT/TRACK_FORCE_W3.md`，机器解析格式，2 臂 + 预登记）。
设计过程与假说更新史：`docs/audits/de_force_w3/design_draft.md`（= `/root/tp-agent/drop_entry/BB/tmp/DE-NXT-FORCE-W3.draft.md`）。
证据图与数字：`artifacts/de_force_w3/fig_force_w3.png`、`artifacts/de_force_w3/numbers.json`、`artifacts/de_force_w3/make_fig.py`（数字程序化提取自 `BB/X/*.md` 定谳卡 + `benchmarks/water_entry/single_drop/fig5_experiment.csv`，零手抄）。
报告：`artifacts/de_force_w3/build_report.py` → `/tmp/report.json`，`publish_report.py` 负责登录与投递（平台报告 id=22733）。

## 为什么本波这样设计（源码级审计结论）
1. `src/tensorlbm/free_surface_lbm.py` + `benchmarks/water_entry/single_drop/free_surface_probe.py` 审计：DE 案液滴是**运动学给定轨迹**的液体团（`velocity = -release_lattice * (moving & active)`），不是 SOLID，**没有动量交换力通道**；探针 history 无 force 字段。
2. `BB/X/FORCE2-MOMENTUM_VOL.md` 判决：`force_volume.py` 从未被接线（grep 无调用点、D2Q9 二维、源码自带 "this is wrong"），两臂 2020 步 history SHA256 位级一致；且两臂预登记的"峰误差"是几何深度、不是力 ⇒ 力线两波无一次可证伪的力级证据。
3. ABB（Körner 2005 / Thürey 2007）是本案唯一界面力通道；`free_surface_lbm.py` L1719-1725 自述"ABB 按界面速度注入动量→伪流→被 u_limit 截断"，而 u_limit 已定谳为纯稳定器（不可作修法）。

## 两臂（预登记见卡）
- **FORCE3D**（诊断探针/仪器）：新增纯后处理 `src/tensorlbm/force_momentum_exchange.py`（link 动量交换 Ladd 1994 湿节点权重 + Chen 1998 控制体通量 + Möbus 2001 伽利略帧修正），探针增 `--force-out`（默认关=逐位零行为改变），以附加质量 `m_a=0.5ρV`（Landau–Lifshitz 11）解析基准证伪；交办 FORCE2 判决书遗留项 (a)(b)(c)。
- **ABBCONS**（物理处方）：界面力自洽 `u_g* = u_g − Δp_abb_prev/(ρΔt)`，打断 ABB 注入→回灌正反馈（Thürey 2007 mass-consistent 界面速度；Möbus 2001 边界力不回灌）。最小改动=ABB 段 1 个持久张量 + 速度修正；旗标关逐位不变。

## 误差判定（如实）
空穴峰值深度误差：cells=8 = 22.731%，cells=12 = 18.035%，**均 >3%（未达标）**；撞前速度误差 0.66%（<3%，达标）；满窗健康 24.005 ms。纯网格渐近地板 8.64%–14.28%（CONV12）⇒ 纯收敛到不了 3%。本单交付的是"力级可证伪性"设计，不是精度达标。
