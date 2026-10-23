# dam_break_2d_multimodel（基线审计包，非 verified）

2D 溃坝四模型（SC/SCMP/CG/FE）对 Martin & Moyce (1952) 14 点实验表的基线审计：
流场演化 + 定量比较，判决 4/4 FAIL。详见 [REPORT.md](REPORT.md)。
修复主线=MP-DAMBREAK 常设线（S1 m211 在飞）；锚点冻结见 agent_work/mp_dambreak/prereg/anchors.md。
判据数字一律机器读自 metrics_vs_MM.json / result.json，零手抄。
