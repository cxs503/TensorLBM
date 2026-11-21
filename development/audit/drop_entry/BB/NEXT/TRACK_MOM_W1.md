# TRACK MOM WAVE 1
goal_met: no
rationale: KERN-A 待判定，KERN-B 证伪。需探索外力项与反弹回规则变体以改善界面动量交换。
自检：N/5（格式机读通过/臂≤2/预登记齐/无重开排除路线/出处可溯）

## ARM GUO_INT
task_key: DE-X-MOM1-GUO_INT
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是力学校准官。任务=在 Guo 外力项中引入密度梯度感知修正，解决界面处密度跃变导致的动量注入偏差。物理出处：Guo et al. (2003) 外力项在密度梯度大时需修正。最小改动：修改 collision.py 中 force 计算，使用中心差分或界面力模型。验证：cells=8 满窗，峰误差≤3%。
prereg: 预测 cells=8 满窗收敛，峰值误差 2.5%±0.5%。证伪：峰值误差>5% 或界面伪电流发散。

## ARM BOUNCE_H
task_key: DE-X-MOM1-BOUNCE_H
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是边界重构官。任务=实现半步反弹回规则，改善界面处动量守恒。物理出处：Half-step bounce-back 可提高边界精度。最小改动：修改 bounce_back 逻辑，使用半步速度重构。验证：cells=8 满窗，峰误差≤3%。
prereg: 预测 cells=8 满窗收敛，峰值误差 2.5%±0.5%。证伪：峰值误差>5% 或界面伪电流发散。
