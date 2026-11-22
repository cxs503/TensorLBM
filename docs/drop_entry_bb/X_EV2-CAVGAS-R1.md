自检：5/5（文件在且非空✓ 数字可溯=cj6188+n330+runs/cavgas_r1/✓ 双写commit✓ 预测区间✓ 失败如实✓）
# DE-X-EV2-CAVGAS-R1 · 气穴气体压力只读诊断探针
锚点=NEW:DE-X-EV2-CAVGAS-R1 · 战役=DE·多线并行攻关波2 · 分支 agent/m27171 · 仓库 TensorLBM · 2026-11-23

## 1. 物理出处
气穴气体随深度压缩，压力随体积变化（绝热 p=p0(V0/V)^γ, γ=1.4 或等温 p=p0V0/V）。
参考实现 CACSim 用真实气体相（rho_g=0.00117）使气穴更深（1.5192 vs 1.4352 D）。
本探针为只读诊断：接触后每步测量气穴体积 V 与气体密度 rho_g_field，计算压力比 p/p0。

## 2. 改动 diff 摘要
- free_surface_probe.py（已存在）：_cavgas_diagnostics() 测 cav_volume/rho_g_field；probe() 内算 (V0/V)^1.4 和 V0/V
- sdaa_fs_probe.py（本单新增）：--diag-cavgas 旗标 + WE_DIAG_CAVGAS 环境变量传递
- 默认关=位级不变

## 3. 实测三元组（SDAA cj6188, n330, card=24, wall=1933s）
| 量 | 值 | 出处 |
|---|---|---|
| stop_reason | window_complete | runs/cavgas_r1/result.json |
| peak_depth | 1.4352D@11.091ms | runs/cavgas_r1/result.json |
| postcontact_healthy_ms | 24.005ms | runs/cavgas_r1/result.json |

## 4. 预登记命中判定
- 预测压力比∈[0.5,2.0]：仅 4/1975 行命中（0.2%）
- 证伪判据（压力比>5.0 或<0.1）：1932/1975 行触发（97.8%）
- stop≠window_complete：未触发（window_complete ✓）
- **判定：证伪触发**（压力比远小于 0.1，气穴体积从 8 膨胀至 2360 cells）

## 5. 诚实披露
- 压力比极低（min=0.000349）因 cav_volume 从 8 膨胀至 2360 cells，(V0/V)^1.4 极小
- rho_g_field≈1.0（模型固定 rho_gas=1.0），未体现真实气体压缩效应
- 与 CACSim 差异：CACSim 用 rho_g=0.00117 使气穴更深，本模型 rho_gas=1.0 恒定
- 结论：当前模型气穴气体处理（恒定密度）无法复现真实气体压缩效应
