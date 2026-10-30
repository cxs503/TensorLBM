# SDAA 验证案例扫荡：taylor_green_2d（B17）

## 案例概述

| 项目 | 值 |
|------|-----|
| 案例名称 | taylor_green_2d（B17） |
| 物理问题 | 2D Taylor–Green 涡衰减（周期域 D2Q9 BGK） |
| 解析解 | 不可压 Navier–Stokes 精确解，纯粘性衰减 |
| 来源 | `benchmarks/verified/taylor_green_2d/` |
| 运行脚本 | `scripts/sdaa_sweep_taylor_green_2d.py` |
| 结果文件 | `artifacts/sdaa_taylor_green_2d.json` |

## 配置

| 参数 | 值 |
|------|-----|
| 网格 | N=128×128 |
| Re | 100 |
| U0 | 0.05 |
| dtype | float32 |
| 碰撞 | BGK（`collide_bgk`） |
| 步数 | 4900（自动按 E/E0→e⁻³ 确定） |
| 记录间隔 | 每 100 步 |
| 设备 | sdaa:0 |

## 设备证据

| 项目 | 值 |
|------|-----|
| 设备 | sdaa:0 |
| 设备名称 | /dev/tcaicard13 |
| 设备数量 | 1 |
| 总内存 | 16039018496 MiB |
| torch 版本 | 2.7.1 |
| torch_sdaa 版本 | 3.1.1a3+gite3971b9 |

## 判据与结果

| 判据 | 公式 | 模拟值 | 理论值 | 偏差 | 阈值 | 结果 |
|------|------|--------|--------|------|------|------|
| 动能衰减率偏差 | \|γ_E_sim/γ_E_theory − 1\| | 6.165950e-04 | 6.168503e-04 | −0.0414% | ≤ 1% | **PASS** |
| 速度衰减率偏差 | \|γ_vel_sim/γ_vel_theory − 1\| | 3.084028e-04 | 3.084251e-04 | −0.0072% | ≤ 1% | **PASS** |
| 指数衰减质量 | ln E vs t 线性拟合 R² | 0.9999996 | — | — | ≥ 0.999 | **PASS** |

**总体结果：PASS**

## 附加指标

| 指标 | 值 |
|------|-----|
| 质量漂移（相对） | +5.22e-05 |
| 样本数 | 49 |
| 墙钟 | 21.3 秒 |

## 绕行记录

| 项目 | 说明 |
|------|------|
| torch.compile | SDAA 上 `teco_inductor` 后端不完整（EXPAND fallback 路径索引越界），使用 eager 路径（`mode=None` → compile_step passthrough） |
| triton | triton 已安装但未设 `TRITON_INTERPRET=1`；本案例不直接依赖 triton，仅 `torch.compile` 间接使用 |
| 编译模式 | eager（SDAA torch.compile 后端不完整，使用 eager 路径） |

## 与 CPU 基准对比

| 指标 | SDAA 结果 | CPU 基准（result.json） | 差异 |
|------|-----------|------------------------|------|
| γ_E_sim | 6.165950e-04 | 6.166361e-04 | −0.0067% |
| γ_vel_sim | 3.084028e-04 | 3.084229e-04 | −0.0065% |
| err_E_pct | −0.0414% | −0.0347% | −0.0067% |
| err_vel_pct | −0.0072% | −0.0007% | −0.0065% |
| R² | 0.9999996 | 0.9999996 | 一致 |
| mass_drift_rel | +5.22e-05 | +5.23e-05 | 一致 |

SDAA 与 CPU 结果高度一致，差异在浮点舍入误差范围内。

## 复现命令

```bash
cd /root/tp-agent/work/m65
python scripts/sdaa_sweep_taylor_green_2d.py
```

## 备注

- 本脚本为独立运行脚本，未修改任何业务代码（`tensorlbm/` 或 `benchmarks/`）。
- 所有判据数值均由脚本自动计算，零手抄。
- SDAA 设备上 `torch.compile` 尚不可用（teco_inductor 后端缺陷），使用 eager 路径运行，数值结果与 CPU 一致。
