# TRACK TMG WAVE 2
goal_met: no
rationale: 前序CAVSWEEP证明参数微调无法修复22.7%峰深缺口；CAC求解器标准Collide-Stream序列在界面力项应用上存在相位误差与刚性问题。本波聚焦时间推进结构：①力项半步步长修正（Guo 2008）减少数值耗散；②界面亚循环（多碰撞/单流）稳定表面张力刚性。
## ARM FORCE_HALF
task_key: DE-X-TMG2-FORCE_HALF
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是力时序官。任务=修改cac_lbm3d.py collide()，采用Guo(2008)半步步长速度修正：u*=u+0.5*F*dt，用u*计算geq/feq，流后u+=0.5*F*dt。物理出处=Guo et al. PRA 77, 056715 (2008) Eq.14-15。最小改动=仅改collide()内u/p计算与平衡分布输入，不改变stream/boundary。验证=对比基线collide()，检查u*一致性。
prereg: 预测峰深误差≤15%（vs 22.7%基线）；证伪=误差>20%或Ma>0.3早死。
## ARM SURF_SUB
task_key: DE-X-TMG2-SURF_SUB
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是界面时序官。任务=修改cac_lbm3d.py step()，在界面区（0.1<phi<0.9）执行2次collide()后1次stream()。物理出处=界面张力力项时间尺度τ_σ≪τ_advection，单步CFL超限导致界面抖动。最小改动=在step()中增加interface_mask判断，若True则collide()两次。验证=检查界面厚度守恒与质量漂移。
prereg: 预测postcontact_healthy_ms≥15ms（vs ~10ms基线）；证伪=存活<10ms或质量漂移>1e-4。
自检：5/5（格式机读通过/臂≤2/预登记齐/无重开排除路线/出处可溯）
