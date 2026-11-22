# DE-X-CONT2-CONT-PIN · 腔口触线钉扎（增摩擦）· 证伪定谳
自检：5/5（文件在且非空✓ 数字可溯=cj6059/cj6060/cj6063 回执+runs/cont2 工件✓ 双写commit✓ 预测区间✓ 失败如实✓）
CROSS已读：TRACK_CONT_W2（处方卡）+ ABBS/CONV12/CAVSWEEP/KERN-B（全读）
锚点=NEW:DE-X-CONT2-CONT-PIN · 战役=DE·多线并行攻关 · 分支 agent/m27023 · 仓库 TensorLBM · 2026-11-22

## 1. 处方与物理出处
- 处方：腔口触线钉扎（增摩擦，延迟 deep-seal→更深）。出处=Truscott deep-seal：封口受触线控制，钉住腔口→抵抗封口液内涌→液滴穿透更远→更深；逆 SLIP/DCA 证伪方向（减摩擦→变浅，故增摩擦→更深）
- 最小改动：free_surface_lbm.py free_surface_step 增 contact_pin 旗标（默认关=零行为）：对 (INTERFACE)&(GAS 邻胞在上)&(uz>0) 的腔口闭合胞，ABB 重构用 uz 乘 (1-K_pin)，K_pin=0.3
- 三重门控（GAS 上+uz>0）自排除下落液滴（uz<0）与开放水面（无 GAS 上）

## 2. 改动 diff 摘要（commit b8593a413，默认关=位级零行为）
- free_surface_lbm.py +12 行：contact_pin 参数；ABB 段 pin 块（gas_above=flags.roll(-1,dims=0)==GAS；pin_mask=iface_abb&gas_above&(_uz_pc>0)；_uz_pc×(1-K_pin) 仅改 f_eq_gas 速度，不动 rho_g）
- free_surface_probe.py +12 行：--contact-pin 旗标（默认 0.0）+ result 元数据
- 零行为验证：patched-off 60 步 history 与 unpatched 逐位相同（本地 /tmp/pin_off_patched.json vs /tmp/pin_off_base.json）

## 3. 预登记（跑前写死）
- 预测 peak∈[1.50,1.70]D（加深 0.07-0.27D）；证伪 |peak-1.4352|<0.05D 或早死(<10.394ms)

## 4. 实测三元组（cells=8 同旗标，集群 n330 8 核，绝对时轴）
| 臂 | peak_depth | peak_error | postcontact_healthy_ms | stop |
|---|---|---|---|---|
| ctrl (K_pin=0) | 1.4352D@11.091ms | 22.731% | 24.552 | window_complete |
| CONT-PIN (K_pin=0.3) | 0.7548D@3.600ms（下界） | 59.362% | 3.612 | health_limit（drift 1.05e-4） |
| CONT-PIN (K_pin=0.1) | 1.4161D@10.714ms | 23.760% | 24.552 | window_complete |
- ctrl 逐位复现锚（1.4352@11.091/24.552，contact 45，impact 1.4092 m/s，mach 0.2598）
- 0.3 臂死于 step297/3.612ms：volume_drift 4.4e-5→1.05e-4 渐进累积（非突变=非实现 bug）；max geo drift 1.30e-3（与 ctrl 1.33e-3 同态）
- 0.3 臂同步步深度（0-297 步）：181 深/84 浅/32 平，接触后多数步更深（最大 +0.0595D@2.858ms）→ 瞬态方向对，但剂量破守恒
- 0.1 臂（跑前预声明随访，非预登记臂）：满窗存活，peak 1.4161D=−0.0191D vs ctrl（证伪带内）；同步步 1007 深/981 浅/32 平=无净加深

## 5. 预登记命中判定
- 证伪判据触发：0.3 臂早死（3.612ms<10.394ms）→ 预登记臂证伪，honest-FAIL
- 预测区间 [1.50,1.70]D 未命中（0.7548D 下界/0.1 臂 1.4161D，均非加深）
- 机制解读：ABB uz 项对守恒承重（ABBS「ABB 动量承重」同源）；腔口向上动量减 30%→空穴更快塌缩+质量账破→早死；稳定剂量下无净加深→触线摩擦非本模型欠深杠杆

## 6. 诚实披露
- 0.3 臂 peak 为下界（死于实验峰时前），真峰未知；numeric_pass=False（3% 门未过）
- pin mask 自 step2 起活跃（初始水面 uz>0 胞，接触前）=预接触扰动，非纯腔口门控
- 单网格无收敛对；0.1 随访为跑前预声明的剂量判别（非预登记决策），判据=满窗且 peak>1.4352D→机制真实，实测 1.4161D 未过

## 7. 复现
- 命令：python3 -m benchmarks.water_entry.single_drop.free_surface_probe --cells 8 --gap .18 --speed .06 --capacity --simple-exchange --release-speed 1.53 --span 4 --pool-depth 3 [--contact-pin 0.3|0.1] --out runs/cont2/<arm>.json
- 分析/图：cont2_contpin_analyze.py + cont2_contpin_figures.py（runs/cont2/fig_depth_time.png）
