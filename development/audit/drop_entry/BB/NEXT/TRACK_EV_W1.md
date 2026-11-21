# TRACK EV WAVE 1
goal_met: no
rationale: Santini case 3 峰深 1.8574D@10.39ms 未达 ≤3% 门限；排除网格/参数/u_limit/ABB移除/rho_g缩放/空穴压亏后，聚焦气穴动力学两大未证物理项：ABB 气相压力重建保真度与外部势流耦合。
## ARM ABB_DIAG
task_key: DE-1-EV1-ABB_DIAG
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是诊断官。任务=在 free_surface_probe 开启 ABB 日志（abb_population_delta, exchange_liquid_credit, exchange_interface_credit），cells=8 跑满窗 24ms。物理出处=Körner 2005 ABB 重建气相压力，若重建偏差大则界面压力失衡影响气穴形态。最小改动=仅开启日志开关，不改内核。验证=记录每步 ABB 质量漂移与气穴深度时序。
prereg: 预测 ABB 净漂移 <0.1% 总液量且与深度误差无显著相关(r<0.3)；证伪=净漂移>1% 或 r>0.5。
## ARM GAS_FLOW
task_key: DE-2-EV1-GAS_FLOW
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是处方官。任务=启用 gas_flow 势流外场（potential flow of exterior sphere, r^-3 decay），cells=8 跑满窗 24ms。物理出处=液滴撞击排开外部空气，势流近似可耦合气相压力反馈至界面。最小改动=run.py 传 --gas-flow 开关。验证=对比无势流基线的气穴深度时序与峰值。
prereg: 预测深度变化 <2%（空气惯性弱）；证伪=峰值深度偏差>3% 或时序相位偏移>0.5ms。
