# TRACK INT WAVE 2
goal_met: no
rationale: 已排除网格收敛/参数微调/u_limit/ABB移除/rho_g缩放/空穴压亏。本波聚焦物理处方：动态润湿角与界面压缩，针对MCMP模型界面厚度与接触角滞后问题。
## ARM COXVOI
task_key: DE-X-INT2-COXVOI
model: Qwen3.8-27B
budget: 600
prescription: 实现Cox-Voinov动态润湿角定律 $\theta_d^3 = \theta_e^3 + 9 Ca \ln(L/\lambda)$。物理出处：Cox (1986) J. Fluid Mech. 168, 169。最小改动：在 `TensorLBM/src/tensorlbm/boundary/inamuro_wetnode.py` 或新建 `wetting` 模块中，根据局部毛细管数 $Ca = \mu U / \sigma$ 动态修正接触角 $\theta_d$，替代固定接触角。验证：毛细管上升高度随时间演化曲线与理论解对比。
prereg: 预测：动态接触角误差 < 5% vs Cox-Voinov 解析解。证伪：误差 > 15% 或接触角不随 $Ca$ 变化。
## ARM COMPINT
task_key: DE-X-INT2-COMPINT
model: Qwen3.8-27B
budget: 600
prescription: 在MCMP平流步中添加界面压缩项以减薄界面厚度。物理出处：Popinet (2009) J. Comput. Phys. 228, 4210。最小改动：修改 `TensorLBM/src/tensorlbm/multiphase/mcmp_psi.py` 中的平流算子，引入 $\partial_t \phi + \mathbf{u} \cdot \nabla \phi = \kappa \nabla \cdot (\phi \nabla \phi)$ 形式的压缩项，其中 $\phi$ 为序参量，$\kappa$ 为压缩系数。验证：静态液滴界面厚度与虚假速度幅值。
prereg: 预测：界面厚度减薄至 < 5 网格，虚假速度降低 > 50%。证伪：界面厚度增加或虚假速度无显著变化。
自检：5/5（格式机读通过/臂≤2/预登记齐/无重开排除路线/出处可溯）
