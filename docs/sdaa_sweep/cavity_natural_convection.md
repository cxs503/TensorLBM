# SDAA Sweep: cavity_natural_convection

## 案例信息

| 项目 | 值 |
|------|-----|
| 案例名 | cavity_natural_convection |
| 物理问题 | 二维方腔自然对流（左壁加热 T=1、右壁冷却 T=0、上下壁绝热） |
| 参考 | de Vahl Davis (1983) Int. J. Numer. Methods Fluids 3:249-264 |
| 参数 | Ra=10⁴, Pr=0.71, τ=0.6 |
| 参考值 | Nu_ref=2.243, u_max_ref=16.178, v_max_ref=19.617 |
| 格子模型 | D2Q9（速度）+ D2Q5（温度 advection-diffusion） |
| 碰撞 | BGK + Guo 力格式（Boussinesq 浮力） |
| 边界 | 四壁 no-slip（pre-streaming 半程反弹）；左/右等温 ABB；上/下绝热 BB |
| 库入口 | `tensorlbm.thermal.simulate_natural_convection` 及底层函数 |

## SDAA 设备证据

```json
{
  "torch.sdaa.get_device_name(0)": "/dev/tcaicard6"
}
```

- **设备类型**: sdaa
- **TRITON_INTERPRET**: not set（thermal.py / d2q9.py / solver.py 无 triton 依赖，无需绕行）

## 运行配置

| 参数 | 值 | 说明 |
|------|-----|------|
| nx | 32 | 可行较小档（工具 120s 超时限制） |
| steps | 2000 | 可行较小档 |
| device | sdaa | torch.sdaa |
| dtype | float32 | 默认 |
| 墙钟 | 30.22s | 含初始化 |

> **注**：案例文档推荐配置为 nx=64/steps=32000 和 nx=128/steps=115500。
> SDAA 在此小网格上每步耗时 ~13ms（kernel launch overhead 主导），
> nx=64/steps=10000 预估 ~520s，超出工具 120s 超时限制。
> 故采用 nx=32/steps=2000 作为可行性验证档。
> 此网格过粗，精度判据预期 FAIL（README 显示 nx=64 的 Nu 误差为 -5.94%，
> 仅 nx≥128 才达到 ≤3%）。

## 计算结果

| 网格 | Nu_grad2 | Nu 误差 | u_max_nd | u_max 误差 | v_max_nd | v_max 误差 | 墙钟 |
|------|----------|---------|----------|-----------|----------|-----------|------|
| 32² | 1.8923 | -15.63% | 18.104 | +11.90% | 20.004 | +1.97% | 29.6s |

## 判据逐条比对

| 门 | 判据 | 结果 | 详情 |
|----|------|------|------|
| library_entry | 使用 tensorlbm.thermal 入口 | ✅ PASS | from tensorlbm.thermal import (equilibrium, temperature_equilibrium, ...) |
| direct_observable | Nu/ u_max/ v_max 误差 ≤3% | ❌ FAIL | Nu: 15.63% > 3%; u_max: 11.90% > 3%; v_max: 1.97% ≤ 3% ✓ |
| monotone_convergence | |Nu 误差| 随网格细化单调递减 | ✅ PASS | 仅 1 档网格，trivially satisfied |
| reference_locked_first | 参考值预注册锁定 | ✅ PASS | de Vahl Davis 1983, Nu_ref=2.243, pre-registered 2026-10-20 |
| case_readme | README 存在且 ≥1200B | ✅ PASS | 5491 bytes |
| figures | 图件目录与 manifest 存在 | ✅ PASS | figures/ 与 figures.json 均存在 |

## 总结

| 项目 | 值 |
|------|-----|
| Overall Verdict | **FAIL** (honest-FAIL) |
| FAIL 原因 | direct_observable 门未通过：Nu 误差 15.63%、u_max 误差 11.90% 均超过 3% 阈值 |
| 根因 | nx=32 网格过粗。案例 README 显示 nx=64 的 Nu 误差为 -5.94%，仅 nx≥128 才达到 ≤3%。受工具 120s 超时限制，无法在 SDAA 上运行 nx≥64 的完整配置。 |
| SDAA 可行性 | ✅ 代码在 SDAA 上可正常运行，无 kernel 错误/OOM/依赖缺失 |
| 设备证据 | ✅ torch.sdaa.get_device_name(0) = "/dev/tcaicard6" |
| Triton 绕行 | 不需要（thermal.py 无 triton 依赖） |
| 墙钟 | 30.22s |

## 产物

- `artifacts/sdaa_cavity_natural_convection.json` — 完整 JSON 产物（设备证据、判据数值、逐门 PASS/FAIL）
- `scripts/sdaa_sweep_cavity_natural_convection.py` — 独立运行脚本（不改业务代码）

## 后续建议

若需在 SDAA 上完成完整精度验证（nx≥128），需解决以下性能瓶颈：
1. SDAA 在小网格（nx≤128）上的 kernel launch overhead 显著，每步耗时 ~13ms（与 CPU 相当）
2. nx=128/steps=115500 预估耗时 ~25min，需放宽工具超时限制或使用更大网格以摊薄 overhead
3. 或考虑使用 CUDA graph / kernel fusion 优化 SDAA 后端
