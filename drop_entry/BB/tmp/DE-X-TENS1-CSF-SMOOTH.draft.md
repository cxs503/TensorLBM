# DE-X-TENS1-CSF-SMOOTH-R1 草稿
## 已读
- TRACK_TENS_W1.md (任务卡)
- EV1-ABB_DIAG.md (定谳卡)
- src/tensorlbm/csf_force_model.py (CSF模型)
- src/tensorlbm/vof_interface_capture.py (VOF曲率计算)
- src/tensorlbm/free_surface_lbm_2d.py (自由表面LBM核心)
- src/tensorlbm/free_surface_lbm_27.py (3D自由表面LBM)

## 已得结论
- 曲率κ在 free_surface_lbm_2d.py:542-546 计算，用于Laplace压力跳跃
- 物理出处：Brackbill et al. 1992 CSF模型指出曲率噪声是伪电流主因
- 改动位置：在κ计算后、delta_rho计算前，应用3点滑动平均滤波
- 滤波公式：κ_smooth[i] = (κ[i-1] + κ[i] + κ[i+1]) / 3
- 已修改 free_surface_lbm_2d.py 和 free_surface_lbm_27.py

## 改动摘要
- free_surface_lbm_2d.py: 在κ计算后添加3点滑动平均滤波（x和y方向）
- free_surface_lbm_27.py: 在κ计算后添加3点滑动平均滤波（x、y、z方向）

## 下一步
1. 运行对照实验（基线 vs 平滑）
2. 测量三元组并生成报告
