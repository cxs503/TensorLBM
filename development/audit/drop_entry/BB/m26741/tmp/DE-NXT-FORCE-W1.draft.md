# TRACK FORCE WAVE 1 (Draft)

## Self-Check
N/5 (格式机读通过/臂≤2/预登记齐/无重开排除路线/出处可溯)

## ARM LADD_WET
task_key: DE-X-FORCE1-LADD_WET
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是力评估官。任务=实现Ladd湿节点法，通过部分体积加权修正切面单元力计算。物理出处=Ladd(1994)微观模拟。最小改动=修改内核力采样权重。验证=对比解析解。
prereg: 预测区间：峰误差≤10%。证伪判据：误差>15%或发散。

## ARM MOMENTUM_VOL
task_key: DE-X-FORCE2-MOMENTUM_VOL
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是力评估官。任务=实现动量交换体积法，通过控制体动量通量积分计算力。物理出处=Chen(1998)LBM理论。最小改动=修改内核力采样体积。验证=对比解析解。
prereg: 预测区间：峰误差≤10%。证伪判据：误差>15%或发散。
