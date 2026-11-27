# DE-NXT-INT-W1 草稿（情报检索设计官 · 续跑必读禁重做）
状态：已完成（2026-11-21）。正式卡=docs/de_intel/TRACK_INT_W1.md（=BB/NEXT/TRACK_INT_W1.md，26 行，自检 5/5）。已读：BB/X 全定谳卡(ABBS/CONV12/CAVSWEEP/KERN-B/REJECT_KERN-A/REJECT_KERN-A-R2)、BB/CROSS 五张 DIGEST(T1-T5)、REF/REF-WEB-01、SEP.md、NEXT/README、NEXT/TRACK_{DIAG,GAS}_W1（防撞线）。

## 已定谳事实（不重开）
- 现峰 1.4352D@10.543ms，peak_error 22.73%（vs 实验 1.857434D@10.394ms）；0/16 臂过 3% 门。
- 排除路线：网格纯收敛/参数微调(CAVSWEEP 16臂)/u_limit/ABB 整删/ABB Option3 重建(1.3701D)/空穴压亏(KERN-B 0.5032D 早死)/rho_g 简单缩放(0.99 以下全死)。
- KERN-A-R2 定谳：界面动量交换一阶改动全部向浅 → 欠深根因不在"界面动量交换量"。
- 气穴深度系统性偏低 13~23%（T5/T4/CONV12 同源残差）。

## 他线 W1 已占（禁撞）
- DIAG：CAV-RAY(空穴压强 vs Rayleigh)、MOM-BAL(动量收支)。
- GAS：GASVISC(气相粘性)、MASSSRC(界面质量源)。
- 我=INT 证据层（编排脑不跑重仿真），产定量证据卡喂设计，不碰他线文件。

## 我的两臂（均=诊断探针/纯后处理，不碰排除路线）
### ARM1 SEAL-DIM（seal 类型量纲定谳）
- 出处：REF-WEB-01 线索3 + Truscott/Techet/Belden 2014(Annu.Rev.Fluid Mech.46)：四类 seal；deep seal pinch-off 深度标度~Fr^{1/2}；Fr=88 高 Fr。
- 任务：量纲核对 pinch-off 时刻 vs 实测峰时刻 10.394ms。纯后处理（实验 CSV 18 点 + 既有深度曲线），无新仿真。
- 关键量：Fr=U/√(gD)=1.40/√(9.81·0.00227)=88；deep seal 预期峰时刻 >> 10.4ms（量纲核对）。
- 产出：seal 类型判定 + t_peak/t_pinch 比 + 重定向建议（给 MOM 或 GAS/CONT）。
- 预登记：预测=surface/shallow seal（t_peak 远早于 deep seal 预期）；证伪：t_peak∈[0.5,2]×t_deep_seal → deep seal，"surface seal"假设被否。

### ARM2 CAV-VOL（空穴体积/闭合时点预算）
- 出处：Santini 2017(REF-WEB-01) 空穴演化；KERN-B §2 已加 cav_volume 记录（free_surface_probe.py）。
- 任务：测 V_cav(t) 与闭合时点 t_close（V_cav 回落到峰<5% 的时刻），判欠深=「没沉够深」还是「闭合过早」。
- 最小改动：启用既有 cav_volume 记录（或加 --cav-vol 旗标默认关=零行为改变），cells=8 满窗，提取 V_cav(t)+t_close。
- 验证：模型 t_close/峰 V_cav vs 实验闭合时点（CSV）。
- 预登记：预测模型 t_close 与实验差；证伪：|t_close_m−t_close_e|/t_close_e<10% → 欠深非闭合过早（转 MOM 线）。

## 收口
- 实验 CSV 已核：18 点，峰 1.857434D@10.39446ms，窗内不闭合（17.398ms 仍 0.874434D）。
- 弗劳德数已核：标准 Fr=U/√(gD)=9.4；REF-WEB-01 "Fr88"=U²/(gD)=Fr²（约定误标，已入卡）。
- 旗标已核：--silhouette 在 free_surface_probe.py L321（默认关）；cav_volume 属 KERN-B 分支未合 main，故 DROP-PEN 用 --silhouette 而非 cav_volume。
- 正式卡 26 行 ≤60，自检 5/5，两臂均诊断探针，不碰排除路线。
