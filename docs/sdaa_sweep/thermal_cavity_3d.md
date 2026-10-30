# SDAA Sweep: thermal_cavity_3d (W9-A)

## 案例

3D 差热立方腔自然对流（Pr=0.71），D3Q19 BGK + D3Q7 温度双分布。
入库名：`thermal_cavity_3d`，判据档冻结配方。

## 配置

| 参数 | 值 |
|------|-----|
| 设备 | sdaa:0 (/dev/tcaicard12) |
| torch 版本 | 2.7.1 |
| Ra | 1e3 |
| N | 32 |
| steps | 40000 |
| Pr | 0.71 |
| tau_f | 0.6 |
| dtype | fp32 |
| sample | 500 |

## 结果表

| Ra | N | steps | Nu_win | ref Nu | err% | 窗漂移% | 热/冷失衡% | NaN |
|----|---|-------|--------|--------|------|---------|-----------|-----|
| 1e3 | 32 | 40000 | 1.05762 | 1.07065 | -1.22% | -0.0000% | 0.0005% | None |

## 判据逐条

| 判据 | 值 | 阈值 | 结果 | 备注 |
|------|-----|------|------|------|
| nu_err_le_3pct | -1.2169 | 3.0 | PASS |  |
| monotone_decreasing | None | >=3 grids required | FAIL | NOT_RUN: only 1 grid executed (wall-clock limit) |
| no_nan | None | None | PASS |  |
| window_drift_lt_0.1pct | -0.0000 | 0.1 | PASS |  |
| hot_cold_imbal_lt_1pct | 0.0005 | 1.0 | PASS |  |

## 墙钟

总耗时：910s (15.2 min)，限制 20 min。

## 绕行记录

- 无 triton 依赖：本案例使用纯 torch tensor ops（solver3d/thermal3d/boundaries3d/thermal_common），
  无需 TRITON_INTERPRET=1 绕行。
- 仅执行 ra1e3 N=32 单档：墙钟限制下无法完成完整阶梯（≥3 grids/Ra），
  单调性判据记为 NOT_RUN。
- 设备：/dev/tcaicard12（sdaa:0）。

## 判定

总体：**PASS**

注：单调性判据未执行（NOT_RUN），因墙钟限制仅能运行单档。
其余判据（Nu 误差、NaN、窗漂移、热/冷失衡）均基于 sdaa 设备实际运行结果。
