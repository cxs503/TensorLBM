# DE-X-K1-CAVCLOS-R2 Draft（K1 空穴过闭合臂重开）

锚点=NEW:DE-X-K1-CAVCLOS-R2 · 战役=DE·组合波后继（owner 令铺开组合比较）· 仓库 TensorLBM · 分支 agent/m27075 · 2026-11-22

## 已读
- BB/X/CMB3-REGSMAG.md（combo_hydro 1.750541D@14.313161ms pc，-5.75%，满窗；时刻门 +37.70% 未过）
- BB/X/TURB2-WALE.md（wale032 1.767636D 满窗，-4.83%；峰时刻仍晚 3.43ms）
- BB/X/CONT2-CONT-RIM.md（判别=弱穿透/动量问题：模型封口晚于实验 +6.82%，峰前腔口速度未反号、环带质量未塌、腔气持续充入 → 封口不是限制因素，瓶颈在穿透动量）
- BB/NEXT/README.md（SDAA 车道用法：sdaa_fs_probe.py，0.94s/步，全窗 ~32 分）
- 前作 m26737（commit ed91ff780，已并入 main 28d1bf631）：--cav-clos-alpha 已穿参
  free_surface_probe.py（probe 参数 + CLI）与 free_surface_lbm.py（L1509：
  `_cav_clos = _kappa*(1+cav_clos_alpha) if cav_clos_alpha>0 else _kappa`，
  `to_liq = interface_mask & (fill > 1.0 + _cav_clos)`），草稿 DE-X-K1-CAVCLOS.draft.md 在树。
- 共享包装器 /root/tp-agent/cluster/sdaa_fs_probe.py 原无 --cav-clos-alpha 旗标
  （前作死因之一：只跑 step 100 烟窗 1.216ms 即报成功，预登记从未检验）。
  本单最小扩展：+1 argparse 旗标 + 1 env 透传 + 2 行 driver kw（默认 0.0=位级零行为）。

## 处方与物理出处
- 处方：free_surface_lbm.py 界面→液体转换阈值 κ→κ·(1+α)，α=0.5。
  物理=空穴壁面液膜需更厚才转液 ⇒ 延缓空穴闭合 ⇒ 回流持续供能 ⇒ 加深。
- 实现口径（继承 m26737，勿重写）：阈值抬升施加于全部 INTERFACE 胞的 I→L 转换行
  （to_liq 行）；任务书"仅空穴壁面胞"为物理意图描述，继承接线为全局 INTERFACE 施加，
  预登记即按此接线登记，本单不重写（如实披露于终卡）。
- 与 CONT2-CONT-RIM 的张力：RIM 判别说瓶颈在穿透动量而非封口 ⇒ 本臂预期可能无效
  （|Δ|<0.05D）或早死；两种结果均有信息量，如实记。

## 预登记（继承 m26737，禁改）
- 预测：peak_depth ∈ [1.50, 1.70] D（相对 ctrl 1.435220D 加深 0.07–0.27D）
- 证伪判据：|peak − 1.435220| < 0.05 D（无效）或早死（存活 < 10.394 ms pc）→ 如实记 FAIL
- 判定基准：ctrl 1.435220D@11.091ms(abs)；实验 1.857434D@10.394ms(pc)；窗 24.005ms；contact_step 45
- 三元组：peak_depth / peak_error(vs 实验) / postcontact_healthy_ms
- 附加阶梯：若 α=0.5 满窗早死，降 α（0.3 / 0.15）找稳定前沿，如实记

## 命令（铁律：--steps 10000 满窗；经共享包装器，禁裸 cluster_queue cd 容器路径）
python3 /root/tp-agent/cluster/sdaa_fs_probe.py --arm K1-CAVCLOS-R2 \
  --tree /root/tp-agent/work/m27075 --cells 8 --gap 0.18 --speed 0.06 \
  --rs 1.53 --pd 3.0 --ul 0.15 --cav-clos-alpha 0.5 --steps 10000
（包装器 driver 内 capacity=True/simple=True/span=4.0 硬编码，与任务旗标
--capacity --simple-exchange --span 4 一致；产物 runs/K1-CAVCLOS-R2/{result.json,run.log}）

## 判定力门
- result.json history 末 step 必须 ≥ 9000 步（或 stop_reason=window_complete/health_limit
  且峰时刻早于停跑点）才具判定力；烟窗数据禁写判定。
- 交付卡落共享位 /root/tp-agent/drop_entry/BB/X/K1-CAVCLOS-R2.md（task_key 尾段命名）。

## 下一步
1. 提交草稿（本文件）✓
2. 递交 α=0.5 满窗作业（SDAA 车道，~32 分/满窗）
3. 程序化分析三元组 → 对照预登记 → 终卡 + 报告
