# TRACK DIAG WAVE 1

自检：5/5（格式机读通过/臂≤2/预登记齐/无重开排除路线/出处可溯）

goal_met: no
rationale: 已排除网格纯收敛/参数微调/u_limit/ABB移除/rho_g缩放/空穴压亏。本波为诊断测量专线，布设两条正交测量探针（p_cav(t)轨迹与三相动量收支），只测量不修复，产数字表喂全部设计官定位缺失物理项。

## ARM CAV_PRESSURE
task_key: DE-X-DIAG1-CAV_PRESSURE
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是诊断测量设计官。任务=布设 p_cav(t) 实测探针：在气穴中心布点采样压力-时间轨迹，与 Rayleigh-Plesset 球穴坍缩解析解逐点对比。物理出处=Rayleigh(1917)/Plesset 球穴坍缩方程，气穴闭合是液滴触水后主导动力学。最小改动=只加压力场采样探针（不改内核），cells=8 满窗三元组。验证=输出 p_cav(t) 数字表+与 Rayleigh 预测的逐点误差。
prereg: 预测区间：实测 p_cav(t) 闭合前段应贴 Rayleigh 曲线（峰偏差≤10%）。证伪判据：闭合段偏差>15% 或非物理振荡（气穴模型缺项）。

## ARM MOMENTUM_BUDGET
task_key: DE-X-DIAG1-MOMENTUM_BUDGET
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是诊断测量设计官。任务=布设液滴-气-液三相动量收支探针：对触水窗口做控制体动量通量积分，逐项列出三相动量收支并检验闭合性。物理出处=三相动量守恒（液滴动能→气穴/液体/耗散），收支不闭合即指示缺失物理项。最小改动=只加动量通量积分探针（不改内核），cells=8 满窗三元组。验证=输出三相动量收支数字表+闭合性残差。
prereg: 预测区间：三相动量收支应闭合（残差≤5%）。证伪判据：残差>10% 或系统性单向亏损（缺失动量源/汇）。
