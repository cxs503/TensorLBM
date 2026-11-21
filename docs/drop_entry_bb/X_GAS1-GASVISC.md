# DE-X-GAS1-GASVISC · 气相粘性修正（GAS 胞分相 τ）· 证伪定谳
自检：5/5（文件在且非空✓ 数字可溯=cj5717/cj5718 回执(rc=0)+runs/gasvisc 工件✓ 双写commit✓ 预测区间✓ 失败如实✓）
CROSS已读：ABBS/CONV12/CAVSWEEP/KERN-B/KERN-A-R2/REJECT_KERN-A（全读）；TRACK_GAS_W1 全读
锚点=NEW:DE-X-GAS1-GASVISC · 战役=DE·多线并行攻关·气相建模线 波1 · 分支 agent/m26728 · 仓库 TensorLBM · 2026-11-21

## 1. 处方与物理出处
- 处方：碰撞步按气相掩膜对 GAS 胞用 τ_g=0.5+3·(0.018·ν_l)（默认关=零行为改变）；probe 加 --gas-nu-ratio（默认 1.0=现状）
- 出处：ν_air/ν_water=1.8e-5/1e-3≈0.018（标准物性表）；任务书机制=气相被液体粘度过度阻尼~50×→腔闭合提前→峰深系统性偏低 13–23%（T5/T4/CONV12 同源残差）
## 2. 改动 diff 摘要（commit ee25ca02b，默认关=零行为改变）
- free_surface_lbm.py +15：free_surface_step 增 gas_nu_ratio（默认 1.0）；BGK 无 SGS 分支按 GAS 掩膜对气相胞用 τ_g=0.5+r·(τ_l−0.5)（r=1.0 走原标量路径，逐位不变）
- free_surface_probe.py +9：--gas-nu-ratio 旗标（默认 1.0），result 记录 gas_nu_ratio
## 3. 预登记（跑前写死，草稿 §1）
- 臂：--gas-nu-ratio 0.018；ctrl=无旗标（=1.0）；同网格 cells=8 同旗标（gap .18 speed .06 capacity simple-exchange release 1.53 span 4 pool 3，u_limit 0.15）
- 预测 peak_depth∈[1.4352,1.65]D（收回 0.42D 缺口 30–80%）；证伪：peak_depth<1.4352D 或 postcontact_healthy_ms<10.394ms（早死）→honest-FAIL
## 4. 实测三元组（cells=8 同旗标，集群 n315 8 核，绝对时轴）
| 臂 | stop | peak_depth | peak_error | postcontact_healthy_ms |
|---|---|---|---|---|
| ctrl (r=1.0) | window_complete | 1.4352D@10.543ms(绝对11.091) | 22.731% | 24.005 |
| arm (r=0.018) | window_complete | 1.4352D@10.543ms(绝对11.091) | 22.731% | 24.005 |
- Δ(arm−ctrl) peak_depth = +1.89e-9 D（张量除舍入伪迹）；ctrl 逐位复现 CONV12 锚（1.4352/22.731/24.005，contact 45，impact 1.4092 m/s 0.656%）
## 5. 预登记命中判定
- **未命中**：臂=ctrl（收回缺口 0%，非 30–80%）；证伪判据（peak<1.4352D 或早死<10.394ms）未触发（臂满窗 24.005ms）
- 判定：气相粘性修正=**no-op**，假设「气相过阻尼致欠深」被证伪（honest 负面结果入账）
## 6. 机制解读（为何 no-op）
- **关键代码事实**：free_surface_lbm.py L1093-1098 每步把 SOLID/GAS 胞 populations 先复位 f_collide 再清零（waLBerla「gas 无 populations」语义）→ GAS 胞碰撞输出每步被丢弃 → 气相粘性在本求解器中**物理上无作用通道**
- 本机 150 步 parity：ctrl vs 臂 depth 差 ~5e-9（张量除 vs 标量除舍入伪迹，非物理效应）
## 7. 诚实披露
- 单网格无收敛对；numeric_pass=False（3% 门未过，与 ctrl 同态）；几何体积漂移 1.33e-3 超 1e-4 门（T4/CONV12 同态，非本单引入）
- 本单改动仅覆盖 BGK 无 SGS 分支（probe 实际路径）；MRT/SGS 分支未接 gas_nu_ratio（超范围，未用）
## 8. 复现
- 命令：PYTHONPATH=$PWD/src python3 -m benchmarks.water_entry.single_drop.free_surface_probe --cells 8 --gap .18 --speed .06 --capacity --simple-exchange --release-speed 1.53 --span 4 --pool-depth 3 [--gas-nu-ratio 0.018] --out runs/gasvisc/<arm>.json
- 分析/图：gasvisc_analyze.py + gasvisc_figures.py + gasvisc_report.py（数据 runs/gasvisc/，图 fig_depth_time.png）
