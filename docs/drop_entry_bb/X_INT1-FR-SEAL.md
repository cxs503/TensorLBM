# DE-X-INT1-FR-SEAL · 弗劳德数约定澄清 + case-3 seal 类型定谳
> 战役 DE · 单滴入深水池 Santini case-3 · 2026-11-21 · TensorLBM agent/m26810
> 本单=纯后处理（无新仿真）：弗劳德数澄清 + seal 类型判定 + 峰/pinch-off 比 + 重定向
> 已读：TRACK_INT_W1.md（FR-SEAL 臂）、BB/X/CAVSWEEP.md、DE-NXT-INT-W1.draft.md、fig5_experiment.csv（18 点）
> 自检：5/5（文件在且非空/数字可溯/双写commit/预测区间/失败如实）

## 1. 物理出处
- Truscott/Techet/Belden 2014（Annu.Rev.Fluid Mech.46，REF-WEB-01）：四类 seal（surface/shallow/deep/quasistatic）；deep seal pinch-off 深度标度 ~Fr^{1/2}。
- 实验：Santini case-3（protocol.json：U=1.4m/s，D=0.00227m，Re=3715，We=62.1，Fr_velocity_squared=88.0）。

## 2. 弗劳德数约定澄清
- REF"Fr88"=U²/(gD)=1.4²/(9.81×0.00227)=1.96/0.0222687=**88.02=Fr²**（约定误标，protocol 亦记 Fr_velocity_squared=88.0）。
- **标准 Fr=U/√(gD)=1.4/0.149227=9.38**（√(gD)=0.149227）。
- We=ρU²D/σ=997×1.96×0.00227/0.07142=**62.1**（交叉验证 D/U 无误，protocol 一致）。

## 3. seal 类型判定（case-3）
- deep-seal pinch-off 深度=Fr^{1/2}·D=√9.38×0.00227=**3.063D**（0.00695m）。
- **实验峰 1.857434D@10.394ms=0.606×deep-seal 深度**（<0.8×3.063=2.45D 证伪阈）。
- **模型峰 1.4352D@11.091ms=0.469×deep-seal 深度**（同样 <2.45D）。
- → **seal 类型=shallow/surface seal**（实验与模型均在浅 seal 区，空穴在达 deep-seal 深度前闭合）。

## 4. 三元组表（模型 ctrl vs 实验）
| 量 | 模型 ctrl | 实验 | 误差 |
|---|---|---|---|
| peak_depth | 1.4352D@11.091ms | 1.857434D@10.394ms | **-22.73%** |
| peak_error | -22.73%（vs 实验峰） | — | — |
| postcontact_healthy_ms | 24.552ms（window_complete） | 17.398ms 末点 0.8744D 未闭合 | 模 0.7029D（-19.6%） |
| 13.391ms 深度 | 1.2601D | 1.3042D | -3.4% |

## 5. 预登记命中判定
- 预登记：seal 类型=shallow/surface seal（峰 0.61×deep-seal 深度，空穴在达 deep-seal 深度前闭合）。
- 证伪判据：实测峰≥0.8×deep-seal 深度(≥2.45D)→deep seal。
- **判定：命中（未证伪）**——实验峰 1.857434D=0.606×deep-seal<2.45D，浅 seal 假设成立。

## 6. 诚实披露
- 本单=纯后处理（无新仿真），所有数字溯自：实验 CSV（fig5_experiment.csv 18 点）+ ctrl 基线（/root/tp-agent/work/m26392/runs/cavsweep/ctrl.json，rc=0 集群账）+ protocol.json。
- 模型峰 1.4352D 与实验峰 1.857434D 均 <deep-seal 深度（3.06D）→ 欠深根因不在 deep-seal 物理，而在**浅 seal 区触线/缘波动力学**（峰深由表面动力学定，非液滴穿透）。
- 模型峰深较实验 -22.73%（恒定项=气穴欠深，与 CAVSWEEP/ABBS 定谳一致）。

## 7. 重定向建议
- 峰深由触线/缘波动力学定（非液滴穿透）→ 建议 **MOM（动量）** 或 **CONT/SURF（接触/表面）** 线：聚焦浅 seal 区表面动力学（接触线张力、缘波动量注入），非 deep-seal 深度物理。
- 不重开：网格/参数/u_limit/ABB/空穴压亏（已排除，见 CAVSWEEP/KERN-B/ABBS 卡）。

## 8. 复现
- 数据：实验 CSV benchmarks/water_entry/single_drop/fig5_experiment.csv（18 点）；ctrl 基线 /root/tp-agent/work/m26392/runs/cavsweep/ctrl.json（2020 步，window_complete）。
