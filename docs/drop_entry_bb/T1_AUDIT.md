# T1 AUDIT · DE-T1-AUD（复核官 GLM·m26286·外族独立审计；只读=REPORT+A3/A1紧缩卡，另核代码/json/集群回执）
## 一、审计结论
### 1. 数字回溯（对 m26277@583a7c0d9 runs/*.json 逐字段核验）
- n8 cac(w6/s0.03)：stop=volume_drift✓ step26=0.1581ms✓ drift=1.002e-4✓ mach=0.112✓ contact=null✓；score: post=None✓ numeric_pass=False✓ status=pending✓
- n12 cac（集群 cj5656 succeeded✓）：step41=0.1662ms✓ mach=0.3055✓ phi@mm=0.00104✓ mach_liq=0.0579✓
- 基线 n8/n12（cj5659 succeeded✓）：均 0.0649ms mach_limit✓ → 存活 2.4~2.6×✓
- 异议⑤数值：1.002e-4/0.158ms=6.3e-4 vs 9.88e-6/0.065ms=1.5e-4（4.2×）✓；commit c5a031689 存在且 diff 描述相符✓
- 唯一瑕疵（非阻塞）：REPORT 引「score.json→numeric_pass」路径不精确——顶层键为 passed_3pct_and_converged，numeric_pass 实在 runs.<name>.numeric_pass（值 False 无误）
### 2. 证伪攻击：「伪电流在气相」
- 「在气相」成立：run.py L210 phi_at_mach_max=liquid[worst]（液相指示），0.001→该点 99.9% 气相；mach_liquid_max 0.058<<0.3055
- 「界面附近」不成立：全程 φ@mm≤7e-3、多数≤1e-4，在界面带（φ0.1~0.9）之外深气区；经典伪电流峰值应在带内；速度最大点≠受力点
- 替代解释未排除（无坐标/涡量/压力记录）：H2 初始化声波/驻波（step8 出现→step17 增至 0.22→振荡，非衰减瞬态）；H3 边界伪影；H4 挤气膜（0.3055≈5.9×液相 0.052，边缘可能）
- 判定：「气相」保留；「界面附近气相→曲率滤波」降级为未证假设 H1
### 3. 异议①-⑥判定
① 成立（代码证：L287 --cells，全文件无 --n）
② 成立（以 REPORT 自述为准；处方原文不在只读清单，未独立核验）
③ 成立（run() L100-111 形参无默认值；默认在 CLI L288/289）
④ 成立（run.py 零处 side_view_measurement；free_surface_probe.py L229 才产出→双相路径 camera_error=None→score L125 numeric_pass 恒 False；「双相路径」限定准确）
⑤ 成立（见上 4.2×；门 1e-4 对 width 非不变量在 n8 兑现）
⑥ 不成立（A3 勘误补遗已解：case3_cac12 全旗标=--width 6 --gap .13 --odd-rate 1 --uniform-drop --stencil-wall；0.065 vs 1.362=旗标差异非数据错误；T1 默认参数基线与 A3#1 不可比）
### 4. 总评：approve with notes
数据全部可回溯、honest-FAIL 判定正确、异议③④⑤实锤；机制结论部分降级；未核验项：处方原文（②）、score 顶层路径笔误（已记）。
## 二、修正单（T1-E2 执行令）
- P1 锚点复现（最高优先，集群 n=12）：python -m benchmarks.water_entry.single_drop.run --cells 12 --width 6 --gap 0.13 --odd-rate 1 --uniform-drop --stencil-wall --out runs/t1_anchor_cac12.json（不带 --speed=默认 0.04，与 A3#1 一致；随后跑 score）
- R1 诊断骑缝（一行级，随 P1 生效）：run.py L196 已有 worst=np.unravel_index(mach_field.argmax())；在 L200-210 row dict 增补 "mach_max_xyz":[int(v) for v in worst] 与 "mach_band_max":float(mach_field[(liquid>0.1)&(liquid<0.9)].max())——区分带内伪电流 vs 深气声波/边界
- 禁令：暂停「曲率滤波/气相力截断」编码——机制未证（φ@mm 在带外），先 P1+R1
- 验收：P1 复现 postcontact_healthy_ms∈[1.09,1.63]ms（=1.362±20%）→ A3#1 恢复有效基线，此后杠杆单变量扫描；若仍≈0.065ms → A3#1 记 WITHHELD 上报协调者；R1 数据裁决 H1~H4
