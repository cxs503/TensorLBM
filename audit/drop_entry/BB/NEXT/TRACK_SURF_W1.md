# TRACK SURF WAVE 1
goal_met: no
rationale: 已排除网格/参数/u_limit/ABB/rho_g/空穴路线。本波聚焦界面法向精度与静水压平衡，二者为自由面重构核心瓶颈。
## ARM CURV
task_key: DE-X-SURF1-CURV
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是界面法向修正官。任务=在VOF界面重构步，用局部最小二乘拟合替代简单梯度计算界面曲率，并基于曲率修正法向矢量。物理出处：CSF模型中伪电流正比于曲率误差。最小改动=替换法向计算内核，cells=8满窗三元组验证。
prereg: 预测峰误差≤3%，满窗≤24ms。证伪：峰误差>5%或发散。
## ARM HYDRO
task_key: DE-X-SURF1-HYDRO
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是静水压补偿官。任务=在压力重构步加入静水压补偿项 `p_recon = p_computed + rho * g * (z - z_ref)`。物理出处：数值格式常破坏静水压平衡导致漂移。最小改动=压力重构函数+1行补偿，cells=8满窗三元组验证。
prereg: 预测峰误差≤3%，满窗≤24ms。证伪：峰误差>5%或发散。
