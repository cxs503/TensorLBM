# DE-X-EV1-GAS_FLOW-R2 · 预登记（跑前锁定，跑后禁改）

> 战役 DE · 单滴入深水池 Santini case-3 · 2026-11-21 · TensorLBM agent/m26908
> 锚点=NEW:DE-X-EV1-GAS_FLOW-R2（换道重派；m26735 双臂 CACSim 数据不足以定谳，见 §6 缺口记录）
> 处方来源：BB/NEXT/TRACK_EV_W1.md ARM GAS_FLOW 段（prereg 原文照录，未改一字）

## 1. 决策与备选

- 决策：外部势流气相耦合（gas_flow，外球 r^-3 衰减势流）是否实质改变弹坑峰深/峰时刻，从而缩小 22.73% 欠深？
- 备选：A=启用 gas_flow 势流外场（本单主臂）；B=不启用（ctrl，战役锚基线）。
- 假设（处方官）：空气惯性弱 ⇒ 深度变化 <2%。

## 2. 臂设计与旗标（跑前写死）

| 臂 | 车道 | 旗标 | 说明 |
|---|---|---|---|
| EV1-GAS_FLOW-CTRL | SDAA（tools/sdaa_fs_probe_split.py，坑⑧ 禁裸跑） | 默认（gas_flow 关） | 必须逐位复现战役锚 |
| EV1-GAS_FLOW | 同上 | --gas-flow | 唯一变量=初始外场势流速度 |

共同旗标（=CAVSWEEP ctrl 锚旗标）：cells=8 gap 0.18 speed 0.06 rs 1.53 pd 3.0 ul 0.15 span 4.0 steps 10000（满窗 24.55ms）。
各 1 臂，不重跑取最好（LBM 确定性，位级复现即重复性证据）。

## 3. 预登记判据（TRACK_EV_W1 处方段原文）

- 预测：深度变化 <2%（空气惯性弱）。
- 证伪：峰值深度偏差 >3% 或时序相位偏移 >0.5ms。
- 判决映射（跑前写死）：
  - |Δpeak_depth| ≤ 2% 且 |Δt_peak| ≤ 0.5ms ⇒ **预测成立**（gas_flow 效应低于可辨阈）；
  - |Δpeak_depth| > 3% 或 |Δt_peak| > 0.5ms ⇒ **预测证伪**（gas_flow 有实质效应）；
  - 2% < |Δpeak_depth| ≤ 3% 且 |Δt_peak| ≤ 0.5ms ⇒ **中间区**（预测未成立、未证伪，如实记录）。
- Δ 口径：GAS_FLOW 臂 vs CTRL 臂（同车道同配置，唯一变量 gas_flow）。

## 4. 基线/对照与位级锚门（跑前写死）

- ctrl 必须逐位复现战役锚（CAVSWEEP ctrl，m26392/runs/cavsweep/ctrl.json）：
  - 峰深 1.4352204748231814 D @ 11.090571428571428 ms（abs）
  - stop=window_complete，t_end=24.55248214285714 ms，contact_step=45
  - mach_max（窗内）=0.2598076211353317，末步 volume_drift=4.805885612579885e-06
  - impact_speed=1.4091844081433575 m/s
- **停止规则**：CTRL 任一项不逐位 ⇒ 基础设施 FAIL，本单判 WITHHELD（不出物理判决），缺口如实记录。
- 实验基准：1.857434 D @ 10.39446 ms（post-impact，fig5_experiment.csv 18 点）。

## 5. 指标（三元组+健康量，程序化提取零手抄）

峰深 depth_over_D 极大值；峰时刻（abs 与 pc=abs−contact_step·dt）；窗口（stop_reason、t_end）；
mach_max（窗内极大）；末步 volume_drift；contact_step；impact_speed_m_s。

## 6. 缺口记录（m26735 双臂，禁重跑，只记录）

- m26735 双臂（solver=CACSim/D3Q19）均 stop_reason=mach_limit：
  - EV1-GAS_FLOW：12 行 history，止于 t=0.133768 ms，mach_max 0.3098，depth_max 0.0096 D；
  - EV1-GAS_FLOW-CTRL：14 行 history，止于 t=0.158089 ms，mach_max 0.3674，depth_max 0.0143 D。
- 远未达 10.394 ms 峰时刻 ⇒ 三元组不可提取，不足以定谳。
- 且 CACSim 路径与战役锚车道（SDAA free_surface_probe）不同，不可与锚直接对照。
- 处置：按 R2 令最小补跑 SDAA 车道双臂（本单 §2），CACSim 双臂数据只作记录不混入判决。

## 7. 混淆与局限（跑前声明）

- 唯一变量=初始外场势流速度场（gas_flow 开/关）；其余旗标逐位同锚。
- gas_flow 只改初始分布函数（equilibrium3d 的 u 分量），不改内核/碰撞/界面处理。
- 单卡确定性求解，无随机种子；重复性以 ctrl 位级锚复现为准。
- 环境：SDAA 太初卡（自动挑闲卡），镜像 lbm_test:snap_1114，commit=本树 HEAD。
- 解释局限：判决仅对 Santini case-3、cells=8、本窗长成立；不外推到其他雷诺数/网格。
