# TRACK TENS WAVE 1
goal_met: no
rationale: 气穴欠深 22.7% 非参数可调；CSF 曲率噪声与力不平衡是伪电流主因，可能干扰界面动力学。
## ARM CSF-SMOOTH
task_key: DE-X-TENS1-CSF-SMOOTH
model: Qwen3.8-27B 或 DeepSeek-V4-Flash-0731
budget: 600/43200/10000000
prescription: 表面张力设计官。任务=在 CSF 曲率计算后、力施加前，对 κ 场应用 3 点滑动平均滤波（κ_smooth = (κ_{i-1} + κ_i + κ_{i+1})/3）。物理出处=CSF 曲率噪声是伪电流主因（Brackbill et al. 1992）。最小改动=修改 src/core/surface_tension.py 中曲率应用段。验证=对比 mach_max 和峰值深度。
prereg: 预测 mach_max 降 20%+，峰值深度误差 < 15%。证伪=峰值误差 > 20% 或 mach_max 不降。
## ARM IF-SEP
task_key: DE-X-TENS1-IF-SEP
model: Qwen3.8-27B 或 DeepSeek-V4-Flash-0731
budget: 600/43200/10000000
prescription: 表面张力设计官。任务=将 CSF 体积力项 σκnδ_s 吸收进压力梯度项（∇p_eff = ∇p - F_st），实现界面力-压力精确平衡。物理出处=压力基 CSF 可消除力不平衡导致的伪电流（Miedema et al. 2008）。最小改动=修改动量方程压力修正步。验证=对比界面压力分布和峰值深度。
prereg: 预测界面压力振荡降低 50%+，峰值深度误差 < 15%。证伪=峰值误差 > 20% 或计算发散。
