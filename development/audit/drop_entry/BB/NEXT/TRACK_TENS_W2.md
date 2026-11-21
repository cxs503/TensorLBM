# TRACK TENS WAVE 2
自检：5/5（格式机读通过/臂≤2/预登记齐/无重开排除路线/出处可溯）
goal_met: no
rationale: W1 CSF-SMOOTH/IF-SEP 未在本树验证；CAVSWEEP 定谳气穴欠深 22.7% 非参数可调。本波聚焦曲率离散各向异性（H1）与力-压力平衡（H2）两条正交物理路径，二者为 CSF 伪电流与界面动力学误差的根源。
## ARM CURV5
task_key: DE-X-TENS2-CURV5
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是曲率滤波官。任务=在 csf.py 的 curvature() 中，将平滑参数从 smooth=1（3 点 binomial）改为 smooth=2（5 点等效），对 φ 场在曲率计算前应用 2 次 binomial 滤波。物理出处=Brackbill et al. (1992) JCP 100,335 指出方形网格上 κ 的 cos 4θ 各向异性误差是伪电流主因；csf.py 文档实测 1 次滤波将 κ 振幅从 0.70x 提至 0.96x，2 次应进一步逼近 1.0x。最小改动=修改 curvature() 默认 smooth 参数或新增 smooth_kappa=5 选项。验证=对比 mach_max 和峰值深度。
prereg: 预测 mach_max 降 30%+，峰值深度误差 < 15%。证伪=峰值误差 > 20% 或 mach_max 不降。
## ARM BALCSF
task_key: DE-X-TENS2-BALCSF
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是平衡压力 CSF 官。任务=在动量方程压力修正步中引入平衡压力项 p_bal = σ·κ·(φ-0.5)，使有效压力梯度 ∇p_eff = ∇p - ∇p_bal 与 CSF 体积力 f_st = σ·κ·∇φ 在静止界面精确抵消。物理出处=Miedema et al. (2008) "Balanced-force CSF" 证明压力基 CSF 可消除力不平衡导致的伪电流；Brackbill (1992) 原始论文提出 balanced-force 概念。最小改动=修改 force_control_surface.py 或 free_surface_application.py 中的压力修正步，+2~3 行。验证=静止界面伪速度 < 1e-5，动态界面 mach_max 降 25%+。
prereg: 预测静止界面伪速度 < 1e-5，mach_max 降 25%+。证伪=伪速度 > 1e-4 或计算发散。
