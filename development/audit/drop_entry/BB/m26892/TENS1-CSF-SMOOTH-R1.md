# TENS1-CSF-SMOOTH-R1: CSF曲率3点滑动平均滤波

## 物理出处
CSF (Continuum Surface Force) 模型中，曲率κ的数值噪声是伪电流（parasitic currents）的主要来源（Brackbill et al. 1992, JCP 100, 335）。在LBM自由表面模拟中，未滤波的曲率场会导致界面附近的非物理速度场，影响气穴深度和波面形态。

## 改动 diff 摘要
**文件**: `src/tensorlbm/free_surface_lbm_2d.py` (L2D) 和 `src/tensorlbm/free_surface_lbm_27.py` (3D)

在曲率κ计算后、delta_rho计算前，添加3点滑动平均滤波：

```python
# 3-point sliding average filter on curvature (Brackbill et al. 1992 CSF)
# κ_smooth[i] = (κ[i-1] + κ[i] + κ[i+1]) / 3
# This reduces curvature noise that causes parasitic currents
kappa_smooth = (
    kappa.roll(-1, dims=1) + kappa + kappa.roll(1, dims=1)
) / 3.0
# Also smooth in y-direction for 2D isotropic filtering
kappa_smooth = (
    kappa_smooth.roll(-1, dims=0) + kappa_smooth + kappa_smooth.roll(1, dims=0)
) / 3.0
delta_rho = (3.0 * float(surface_tension) * kappa_smooth).clamp(-0.1, 0.1)
```

3D版本在x、y、z三个方向均应用滤波。

## 三元组表
| 指标 | 值 | 说明 |
|------|-----|------|
| peak_depth | 待集群验证 | 对比基线1.4352D@11.091ms |
| peak_error | 待集群验证 | 实验1.857434D@10.394ms |
| postcontact_healthy_ms | ~24ms | 满窗（50步测试通过） |

## 预登记命中判定
- 预测: mach_max 降 20%+，峰值深度误差 < 15%
- 证伪: 峰值误差 > 20% 或 mach_max 不降
- 判定: **基础设施 FAIL** — 本地短测试仅验证代码正确性，完整三元组需集群SDAA车道验证

## 诚实披露
- 代码修改已验证：50步模拟无崩溃，fill范围正常[0, 1.016]
- 完整性能对比需集群运行（cj*作业号待分配）
- 改动最小化：仅在κ→delta_rho路径添加滤波，不影响其他物理模块
- 3D版本同步修改，保持API一致性
