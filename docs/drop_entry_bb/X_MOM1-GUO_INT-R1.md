# DE-X-MOM1-GUO_INT-R1 · Guo 外力项密度梯度感知修正（界面力模型）· 证伪定谳
自检：5/5（文件在且非空✓ 数字可溯=cj6309/cj6310/cj6318 rc=0+runs/guo_int 工件✓ 双写commit✓ 预测区间✓ 失败如实✓）
CROSS已读：ABBS/CONV12/CAVSWEEP/KERN-B/KERN-A-R2（全读）+ TRACK_MOM_W1 处方
锚点=NEW:DE-X-MOM1-GUO_INT-R1 · 战役=DE·多线并行攻关·界面动量交换线 波1 · 分支 agent/m27354 · 仓库 TensorLBM · 2026-11-23

## 1. 处方与物理出处
- 处方：Guo 外力项引入密度梯度感知修正，解决界面密度跃变导致的动量注入偏差（中心差分或界面力模型）
- 出处：Guo et al. (2003) 两相 LBM 外力项——密度梯度大时源项须与局部相质量一致；Körner(2005) eq.15 fill 加权
- 机制探针（本机 40 步 wrap free_surface_step）：I 胞 rho_s≈1.001 vs fill≈0.13 → 本库界面胞群体=满液相（非混合密度）
  → legacy 力 rho_s·fill·g ≈ fill·rho_L·g 已与簿记质量一致；「O(fill) 欠注入」假设在本库不成立（no-op 发现）
- 真改动=full 模式：0<fill<1 胞力全强度（群体携带满液相质量，fill 加权重复扣减亚胞体积）；fill=0 包络胞零力

## 2. 改动 diff 摘要（commit 30eafca43+cd11caac6，默认关=逐位同）
- free_surface_lbm.py +50/-14：TL_FS_GUO_INT 三态（unset=legacy 逐位同 / mass=fill·rho_L 质量一致 / full=0<fill<1 全强度）
- 半步移同步 F/(2rho_s) 保持 Guo 一致性；MRT/BGK 两源项调用点统一 _fx/_fy/_fz
- 本机验证：legacy 60 步 vs 基线 0 分歧；full 60 步 59 行异；mass 150 步 vs legacy 仅 2.4e-6D（no-op 实证）

## 3. 预登记（跑前写死，任务书原样）
- 预测 cells=8 满窗收敛，峰值误差 2.5%±0.5%；证伪：峰值误差>5% 或界面伪电流发散
- 对照=ctrl 基线 1.4352D@11.091ms/24.552ms 满窗（CAVSWEEP 锚）

## 4. 实测三元组（cells=8 同旗标 gap.18/speed.06/capacity/simple-exchange/RS1.53/span4/pool3，u_limit 0.15）
| 臂 | 集群账 | stop | peak_depth | peak_error | postcontact_healthy_ms |
|---|---|---|---|---|---|
| ctrl | cj6309 | window_complete | 1.435220D@10.543ms(绝对11.091) | 22.731% | 24.005 |
| guo_int(mass) | cj6310 | window_complete | 1.426956D@10.568ms | 23.176% | 24.005 |
| guo_int(full) | cj6318 | window_complete | 1.373574D@10.179ms | 26.050% | 24.005 |
- 三臂均 2019 步满窗、contact 45、mach_max 0.2598、vol_drift 8.55/8.94/9.84e-5（<1e-4 门）、无 NaN/伪电流发散
- ctrl 逐位复现锚：1.4352D@11.091ms 绝对时（=10.543ms 接触后），impact 0.66%

## 5. 预登记命中判定
- 预登记臂=mass 模式：23.176%>5% → **证伪触发**（honest-FAIL）；预测区间 [2.0,3.0]% 未命中
- full 臂（事后变体，非预登记）：26.050%>5% → 同证伪；方向=变浅 -0.0616D（-4.3%），峰时提前 0.364ms
- 机制解读：full 模式让界面液相以 g 而非 fill·g 加速→界面带向下动量↑，但空穴反而变浅且峰提前
  → 与 KERN-A-R2 定谳一致：界面动量交换一阶改动全部向浅；欠深根因不在 Guo 力项（MOM 线一阶路线再+1 证伪）

## 6. 诚实披露
- 预登记 2.5%±0.5% 被两臂击穿（23.18%/26.05%）；3% 门 0/2 过；numeric_pass=False（geom_drift 1.33e-3 超门，T4/CONV12 同态非本单引入）
- mass 臂 -0.0083D 差为浮点级种子混沌放大（窗口内 max|Δdepth|=0.081D），非系统效应
- full 臂为跑前机制探针后新增的事后变体，单独标注，不冒充预登记臂
- 单网格无收敛对；peak 时刻口径=score.py 观察窗内峰（与 CAVSWEEP 11.091ms 绝对时同值）

## 7. 复现
- 命令：TL_FS_GUO_INT={,mass,full} PYTHONPATH=.:src python3 -m benchmarks.water_entry.single_drop.free_surface_probe --cells 8 --gap .18 --speed .06 --capacity --simple-exchange --release-speed 1.53 --span 4 --pool-depth 3 --steps 10000 --out runs/guo_int/<arm>.json
- 分析：score.py（benchmarks/water_entry/single_drop/）；图 guo_int_figures.py → runs/guo_int/fig_depth_time.png
