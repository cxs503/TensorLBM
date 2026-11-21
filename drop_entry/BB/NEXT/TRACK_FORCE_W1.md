# TRACK FORCE WAVE 1

自检：N/5（格式机读通过/臂≤2/预登记齐/无重开排除路线/出处可溯）

goal_met: no
rationale: 已排除u_limit/网格纯收敛/参数微调/ABB移除/rho_g缩放/空穴压亏。本波聚焦力采样内核的物理修正，通过Ladd湿节点与动量交换体积法两条正交路径，验证切面单元力计算的精度上限。

## ARM LADD_WET
task_key: DE-1-FORCE1-LADD_WET
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是力评估官。任务=实现Ladd湿节点法，通过部分体积加权修正切面单元力计算。物理出处=Ladd(1994)微观模拟，针对截断单元引入体积分数权重。最小改动=修改内核力采样权重，将标准权重替换为湿节点体积分数。验证=对比解析解，观察峰误差收敛。
prereg: 预测区间：峰误差≤10%。证伪判据：误差>15%或发散。

## ARM MOMENTUM_VOL
task_key: DE-2-FORCE2-MOMENTUM_VOL
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是力评估官。任务=实现动量交换体积法，通过控制体动量通量积分计算力。物理出处=Chen(1998)LBM理论，基于动量守恒推导。最小改动=修改内核力采样体积，将点采样替换为包围体积分。验证=对比解析解，观察峰误差收敛。
prereg: 预测区间：峰误差≤10%。证伪判据：误差>15%或发散。
