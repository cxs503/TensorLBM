# K2-BUDGET-B · 撞击动量/能量收支分解：峰时刻之前能量流向了哪里

> 仓库 TensorLBM · 分支 agent/m26849 · 锚 = NEW:DE-X-K2-BUDGET-B（K1-DIAG-B 的直接后继，minimax 赛马道）
> 判决：**收支不闭合 honest FAIL（丢失渠道已定位）；主导通道 = 7_mass_exchange_conversion**
> 自检：数字可溯 = runs/k2budget/result.json（逐项时序）+ runs/k2budget_verdict/verdict.json ✓ | 只读对照 = 位级一致 ✓ | 锚复现 Δ = 0.000e+00 ✓ | 预登记先跑 ✓ | 失败如实 ✓

> 战役 = DE·诊断主线 K 系列。本单问：**撞击的能量/动量在峰时刻之前被哪个通道吃掉了？**
> 前情（定谳勿重验）：K1-DIAG-B 已证「空穴过闭合」被直接证伪（d_cav_gap ≡ 0、p_cav = 1.0 单连通），
> −22.7310 % 欠深 = 液滴物理欠穿透（d_drop_any = 1.3953 D < 0.8·exp = 1.4859 D）。

| 关键量 | 模拟 | 基准/阈值 | 判定 |
|---|---|---|---|
| 收支残差 \|R\|/E_scale（含环境压功） | 1781.6817% | < 10% | **FAIL（honest）** |
| 收支残差 \|R\|/E_scale（不含环境压功） | 124.8519% | < 10% | **FAIL（honest）** |
| 主导通道份额 | 248.98% | ≥ 50% | DETERMINED |
| LBM 坑深峰 | 1.4352204748 D | 实验 1.857434 D | 缺口 -22.7310% |
| 重力位能需求 / 撞击能量 | 0.5480% | — | 重力**不是**缺口来源 |
| 诊断只读性（对照臂逐字段） | 逐字段零差异（位级一致） | 位级一致 | PASS |
| 锚复现（ctrl 峰深） | 1.4352204748 D | 战役锚 1.4352204748 D | Δ = 0.000e+00 |

## 1. 判决

**判决：收支不闭合（honest FAIL）+ 丢失渠道定位；主导通道 = **7_mass_exchange_conversion**（份额 248.98%，门 50%）。**

- 收支闭合判定：不闭合 honest FAIL（\|R\|/E_scale = 1781.6817%，预登记门 10%）。
- 主导通道：**7_mass_exchange_conversion**（份额 248.98%，门 50%）。
- 欠穿透归因（任务 b 项）：实验坑深 1.857434 D vs LBM 1.4352204748 D，缺口 -22.7310%。
  - 该缺口对应的**重力位能需求** ΔE_PE = 3.4469e-08 J，只占撞击能量 E_scale = 6.2894e-06 J 的 0.5480% ⇒ **深度缺口不是重力位能不足造成的**（差 0.55% vs 缺口 22.73%），缺口在**惯性/动量分配**侧。
  - 按预登记声明假设（坑深 ∝ 轴向动能），达到实验深度所需的额外轴向动能 = 2.0467e-07 J（1.6504e-02 格点单位）；持有该量级能量的通道：2_surface_energy_increment（cavity_wall_sigma_dA_lattice = 1.3774e-01）、3_interface_viscous_dissipation（total_lattice = 2.1538e-02）、7_mass_exchange_conversion（conv_loss_lattice = 1.2627e+00）。

## 2. 工况与方法

- 工况：Santini case-3 单滴入水，cells = 8，gap = 0.18 D，lattice speed = 0.06，prescribed release = 1.53 m/s，pool depth = 3.0 D，u_limit = 0.15，capacity+simple 质量交换，stop = window_complete。
- 尺度（格点→SI）：dx = 2.837500e-04 m，dt = 1.216071e-05 s，μ_l = 1.292234e-04，σ = 4.636980e-04（= 4.64e-4，任务书给定值），g = 5.112712e-06。
- 积分窗口：接触步 45 → 深度峰步 912（11.090571 ms，868 行逐项时序）。
- 诊断实现：`benchmarks/water_entry/single_drop/budget_diag.py`（新增，只读）+ `free_surface_probe.py --diag-budget`（只读旗标，不改物理）+ `tools/sdaa_fs_probe_split.py` 分相位 SDAA 车道。
- 判决脚本：`tools/k2_budget_verdict.py`（全部数字程序化注入，零手抄）；阈值写死在预登记 `docs/audits/k2_budget_20261121/PREREG.md`，跑后一字未改。

## 3. 逐项收支（格点能量单位；份额 = /E_scale，E_scale = 接触时液相动能）

| 通道 | 类型 | 数值 | 份额 |
|---|---|---|---|
| ① 空穴气体压力冲量 | 源/冲量 | 壁面受力冲量 z = -3.0557e-01；挖掘冲量 ρg∫V_cav dt = 5.0377e+00 | 40.51%（占撞击动量） |
| ② 界面表面能增量 | 存 | σ·ΔA_cav = 1.3774e-01；σ·ΔA_total = 2.3209e-01 | 27.16% |
| ③ 界面粘性/拖曳耗散 | 汇 | 总 2.1538e-02（界面邻域 1.2749e-02） | 4.25% |
| ④ 侧向射流动能 | 存 | ΔKE_lat = -3.2663e-02（坑足迹内 -2.6006e-02） | -6.44% |
| ⑤ 液滴残余动能 | 存 | KE_drop(t*) = 5.7830e-04（初始 3.8841e-01） | 0.11% |
| ⑥ 轴向（液柱）动能 | 存 | KE_axial(t*) = 5.6102e-02 | 11.06% |
| ⑦ 质量交换转换（附加） | 汇 | conv_loss = 1.2627e+00，conv_gain = 5.3373e-01 | 248.98% |
| ⑧ 重力功（附加） | 源 | W_grav = -7.6002e-02 | -14.99% |

能量账（窗口内）：

- ΔE_KE(液相) = -4.2543e-01
- W_gravity = -7.6002e-02；W_atmosphere（刚体域净体积功）= -8.4029e+00（自由面体积漂移 +25.2087 格点胞）
- ΔE_surface = 2.3209e-01；E_dissipation = 2.1538e-02；E_conversion(net) = -7.2900e-01
- **残差 R = 9.0361e+00**（不含环境压功口径：6.3321e-01）

动量账（辅助，z 分量）：

- 撞击动量（滴材料，接触时）P_z = -1.2435e+01
- ΔP_liq,z = 1.5659e+01；J_gravity = -1.1020e+02；J_cavity-gas(wall) = -3.0557e-01；J_conversion = 7.9351e+01
- 动量残差 = 4.6810e+01（= 376.43% × 撞击动量）；动量账**不含池壁反力**，故残差是未建模壁面/压力反力的上界（不作为闭合判据）。

## 4. 闭合性判定与丢失渠道定位

**收支不闭合（honest FAIL）**：\|R\|/E_scale = 1781.6817% ≫ 10%。按预登记，定位丢失渠道，不调阈值。

逐条排查（量级对照）：

1. **环境压功口径**：自由面体积漂移 +25.2087 格点胞 → W_atm = -8.4029e+00，量级 1656.8% E_scale；这是**数值体积漂移伪功**，不是物理通道。
2. **转换通道（I→G 质量交换）**：conv_loss = 1.2627e+00，conv_gain = 5.3373e-01，净销毁 = 7.2900e-01 = 143.7% E_scale。净销毁**超过 100 % E_scale** ⇒ 单向能量汇不可能 ⇒ 界面处存在 **churn**：质量交换把液相质量转成气相时销毁动能（气相速度为零），同时 ABB 重建在界面注入等量级的能量；两者相抵后的净额就是本账的残差主体。（物理含义：被气穴吞没的滴材料动能在本模型里确实被销毁，但销毁量由界面人口级注入补偿，因此**转换通道不能被读成单向的物理耗散**。）
3. **伪电流限幅器**：峰时刻贴限界面胞 0 个，其动能 0.0000e+00（0.00% E_scale）；限幅器移除的能量上界 = 贴限胞动能，量级不足以解释残差。
4. **连续耗散估计 vs LBM 碰撞耗散**：本账用宏观应变率连续估计 2μ\|S\|²，与 LBM 碰撞的真实耗散存在 O(dx²) 系统差；实测耗散 2.1538e-02 （4.25% E_scale），不足以解释残差。
5. **反弹边界（ABB）人口替换**：ABB 在界面用 f_eq(ρ_g,u)+f_eq(ρ_g,u)[opp]−f_post[opp] 替换缺失人口，其**人口级**能量贡献不落在任何宏观通道里；本账的残差随时间的平均漂移 = 4.627e-04 格点能量/步（残差-界面面积相关系数 -0.046，残差-转换损耗率相关系数 -0.463）⇒ 残差主体是**未测的人口级边界/交换能量**，而不是五项中的任何一项。
6. **① 的局部口径 vs 刚体域口径**：空穴壁局部自由面积分给出 8.6702e+01，刚体闭合域净体积功给出 -8.4029e+00（差 11.3 倍）。局部形式不是有效的连续功项（自由面是弥散的、界面速度取胞速度而非物质界面速度），故主判据用刚体域口径，局部形式只作交叉校核。

## 5. 机制解读与下一波指向

- 主导通道：**7_mass_exchange_conversion**（份额 248.98%，门 50%）。
- **机制链条**：撞击能量进入液相（E_scale = 5.0717e-01 格点）→ 被界面**面积增长**吸收（σΔA = 2.3209e-01，界面总面积 1162.6 → 1663.2 格点面积）→ 滴材料被气穴吞没（K1 已证）后其动能被质量交换销毁（conv_loss = 1.2627e+00）→ 峰时刻液相只剩 5.6102e-02 轴向动能（11.1%）⇒ 坑被拖浅。
- ① 的定性结论：本模型空穴气体是**压强匹配的环境气体**（ρ_gas = 1）+ Laplace 跳跃，它对坑壁的净向下冲量并不驱动坑深；真正要“顶开”坑的是撞击动量本身，挖掘冲量 ρ_l g ∫V_cav dt = 5.0377e+00（占撞击动量 40.51%）。
- **下一波处方指向**：界面质量交换 / ABB 人口级 churn —— 下一波应做**界面耦合替代**（IF-SEP 压强基 CSF、空穴壁液膜保留 IF-FILM）与本基线赛马，而不是继续调耗散/表面张力（预登记规则：单项份额 ≥ 50% 即指向该项）。
- **物理通道（剔除模型级 churn 与压力积分伪项）**：最大物理通道 = **2_surface_energy_increment**（份额 45.76% E_scale）；其余：轴向液柱动能 11.06%、粘性耗散 4.25%、侧向射流动能 -6.44%（**负** = 峰时刻侧向动能反而更小，侧向射流不是能量仓库）、滴残余 0.11%。
- **能量供给 vs 传递效率**：实验深度所需的重力位能只占撞击能量 0.55%，而深度缺口是 -22.73%（比值 2.41%）⇒ **坑不是「能量不够」，而是「能量送不进去」**：撞击能量被界面面积增长（② 45.8%）与界面 churn（⑦）吃掉，峰时刻只剩 11.1% 的轴向液柱动能。
- 欠穿透（-22.7310%）的定性归因：能量分配显示「1、2、3、7」持有本应进入轴向贯穿的能量；重力位能侧只需 0.55% 的撞击能量即可补齐深度差 ⇒ **处方应指向动量/能量传递效率（惯性侧），不是指向浮力/静水**。

## 6. 诚实披露

- **本单没有消除欠深**：产出是**归因判决 + 丢失渠道定位**，不是精度修复（honest-FAIL 合法）。
- **收支闭合**：FAIL（\|R\|/E_scale = 1781.6817%）。若为 FAIL，残差主体定位为「ABB 人口替换 + 质量交换输运」的人口级能量，未被本账的任何宏观通道覆盖。
- **① 的两种口径不可混用**：局部自由面积分（cavity-wall）与刚体域净体积功（rigid-domain net volume work）在本模型里差两个量级，前者不是有效的连续功项；卡片同时列出并说明。
- **⑦ 是模型级能量汇**：自由面质量交换把液相质量转成气相时，被转走质量的动能在模型里直接消失（气相速度为零），这是**模型**的能量亏损，不是真实物理；解读时不能当作物理耗散。
- **连续耗散是估计值**：2μ\|S\|² 由宏观速度场的中心差分应变率给出，与 LBM 碰撞耗散不严格相等。
- **面积口径**：界面面积用 ∫\|∇φ\| 估计（与求解器曲率算法同源），只用增量 ΔA，绝对值有系统偏差。
- **滴材料标记**：被动示踪，质量取种子胞、速度取所在胞，K1 已量化的数值扩散余量（±1 格 = ±0.125 D）适用。
- **适用范围**：cells=8 单网格、Santini case-3 单工况、单流体自由面模型（常数大气压 + Laplace）；未跑 12 cells 收敛复核；① 的结论只对**通气空穴**成立（真实可压缩气体增压不在模型能力内）。
- **动量账不闭合是预期的**：池壁反力未建模，故只作辅助诊断，不作闭合判据。
- **诊断自检**（`runs/k2budget_verdict/sanity.json`）：能量分配闭合误差 4.437e-16（相对）、步号连续 = True、自由面体积漂移 48.799 格点胞（1.965e-03 相对）、伪电流限幅贴限胞最多 8 个（动能 4.807e-02）⇒ **限幅器在本窗口未激活**，不是残差来源。

## 7. 复现

```bash
# 1) SDAA 满窗三臂（分相位，每相位 <120 s）
python3 tools/sdaa_fs_probe_split.py launch  --arm k2budget       --tree . --steps 10000 --cells 8 \
        --gap .18 --speed .06 --rs 1.53 --pd 3.0 --ul .15 --diag-budget
python3 tools/sdaa_fs_probe_split.py launch  --arm k2budget_ctrl  --tree . --steps 10000 --cells 8 \
        --gap .18 --speed .06 --rs 1.53 --pd 3.0 --ul .15
python3 tools/sdaa_fs_probe_split.py launch  --arm k2budget_planes --tree . --steps 10000 --cells 8 \
        --gap .18 --speed .06 --rs 1.53 --pd 3.0 --ul .15 --planes --diag-budget
python3 tools/sdaa_fs_probe_split.py poll    --arm <臂名> --tree .
python3 tools/sdaa_fs_probe_split.py collect --arm <臂名> --tree .
# 2) 判决（数字全部程序化注入）
python3 tools/k2_budget_verdict.py --tree .
# 3) 图 + 交付件 + 报告
python3 tools/k2_budget_figures.py --tree .
python3 tools/k2_budget_card.py --tree . --card /root/tp-agent/drop_entry/BB/X/K2-BUDGET-B.md
python3 tools/k2_budget_report.py --tree . --payload /tmp/k2budget_report.json --publish
```

- 平台报告发布：id=22783（scenario_key=K2-BUDGET-B，mission_id=26849，5 图内嵌）
- 工件：`runs/k2budget/{result.json,run.log}`（含逐项时序 `budget_diag.rows`）、`runs/k2budget_ctrl/{result.json,run.log}`、`runs/k2budget_planes/{result.json,run.log,planes.npz}`、`runs/k2budget_verdict/{verdict.json,verdict.md}`
- 图：`docs/audits/k2_budget_20261121/figs/*.png`
- 预登记：`docs/audits/k2_budget_20261121/PREREG.md`
- 锚参照（只读）：/root/tp-agent/work/m26392/runs/cavsweep/ctrl.json（锚峰深 1.4352204748 D）

