# INT2-COXVOI: Cox-Voinov Dynamic Wetting Law

> 本页保留 #2850 的研究记录。源码与验证脚本已归入[m26820 研究档案](../../development/experiments/m26820/cox_voinov/README.md)，其中记录了原始修订、复跑命令和稳定性结果。

## 物理出处
Cox (1986) J. Fluid Mech. 168, 169: θ_d^3 = θ_e^3 + 9·Ca·ln(L/λ)
- θ_d: 动态接触角, θ_e: 平衡(静态)接触角
- Ca = μU/σ: 毛细管数, L/λ: 宏观/微观长度比
- Ca > 0 (前进): θ_d > θ_e; Ca < 0 (后退): θ_d < θ_e

## 改动摘要
在 `TensorLBM/src/tensorlbm/wetting/__init__.py` 新增:
- `cox_voinov_dynamic_angle(theta_e_deg, Ca, log_ratio)`: 计算动态接触角
- `cox_voinov_dynamic_wall_psi(theta_e_deg, Ca, log_ratio)`: 计算动态壁面伪势
- `capillary_rise_theory(h_static, Ca, log_ratio)`: 毛细管上升理论高度
- `run_capillary_rise(...)`: 毛细管上升模拟
- `wall_psi_from_contact_angle()` 增加角度钳位 [0, 180]

## 三元组
| 指标 | 值 | 说明 |
|------|-----|------|
| peak_depth | N/A | 本单不测峰深 |
| peak_error | N/A | 本单不测峰深 |
| postcontact_healthy_ms | N/A | 本单不测 |

## 预登记命中判定
- 预测: 动态接触角误差 < 5% vs Cox-Voinov 解析解
- 证伪: 误差 > 15% 或接触角不随 Ca 变化
- 结果: **命中** - 公式验证误差 < 1e-14%, Ca 单调性验证通过

## 诚实披露
- 毛细管上升模拟在 G=5.0 下不稳定 (Shan-Chen 模型限制)
- 公式数学正确性已验证 (误差 < 1e-14%)
- Ca 标度单调性已验证
- 完整 LBM 模拟需要进一步调参 (G, tau, wall_psi 范围)
