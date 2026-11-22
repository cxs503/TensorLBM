# DE-X-K1-CAVCLOS-R2 · K1 空穴过闭合臂重开（α=0.5 满窗定谳 + α 阶梯）

自检：5/5（文件在且非空✓ 数字可溯=集群账单号 cj6093/6094/6095 + runs/K1-CAVCLOS-R2*/result.json✓ 双写commit✓ 预登记区间✓ 失败臂如实✓）
CROSS已读：CMB3-REGSMAG（combo_hydro 1.750541D 满窗 -5.75%）、TURB2-WALE（wale032 1.767636D 满窗 -4.83%；SGS 剂量-响应非单调先例）、CONT2-CONT-RIM（判别=弱穿透/动量问题：模型封口晚于实验 +6.82%，瓶颈在穿透动量非封口）、BB/NEXT/README（SDAA 车道+护栏）。
锚点=NEW:DE-X-K1-CAVCLOS-R2 · 战役=DE·组合波后继（owner 令铺开组合比较）· 分支 agent/m27075 · 仓库 TensorLBM · 2026-11-22

## 1. 处方与物理出处
- 处方：free_surface_lbm.py 界面→液体转换阈值 κ→κ·(1+α)，α=0.5（主臂）。
  物理=空穴壁面液膜需更厚才转液 ⇒ 延缓空穴闭合 ⇒ 回流持续供能 ⇒ 加深（预登记方向）。
- 实现（继承 m26737 commit ed91ff780 接线，已并入 main 28d1bf631，勿重写）：
  `_cav_clos = _kappa*(1+cav_clos_alpha) if cav_clos_alpha>0 else _kappa`；
  `to_liq = interface_mask & (fill > 1.0 + _cav_clos)`（free_surface_lbm.py L1509-1510）。
  口径披露：继承接线对全部 INTERFACE 胞的 I→L 转换行生效；任务书"仅空穴壁面胞"为物理意图描述，
  预登记即按此接线登记，本单不重写（如实）。
- 与 CONT2-CONT-RIM 的张力：RIM 判别说瓶颈在穿透动量而非封口 ⇒ 本臂预期可能无效或早死；
  实测=第三种：反向（减深）+ 失稳（见 §5）。

## 2. 预登记（继承 m26737，禁改）
- 预测：peak_depth ∈ [1.50, 1.70] D（相对 ctrl 1.435220D 加深 0.07–0.27D）
- 证伪判据：|peak − 1.435220| < 0.05 D（无效）或早死（存活 < 10.394 ms pc）→ 如实记 FAIL
- 三元组：peak_depth / peak_error(vs 实验 1.857434D) / postcontact_healthy_ms
- 附加阶梯：若 α=0.5 满窗早死，降 α（0.3 / 0.15）找稳定前沿，如实记

## 3. 命令与车道
- 命令（铁律：--steps 10000 满窗；旗标与任务书逐项一致）：
  `PYTHONPATH=.:src python3 benchmarks/water_entry/single_drop/free_surface_probe.py --cells 8 --gap 0.18 --speed 0.06 --capacity --simple-exchange --release-speed 1.53 --span 4 --pool-depth 3 --u-limit 0.15 --cav-clos-alpha {0.5|0.3|0.15} --steps 10000 --out runs/K1-CAVCLOS-R2{,-a03,-a015}/result.json`（主臂另加 --planes runs/K1-CAVCLOS-R2/planes.npz）
- 车道：SDAA 共享包装器 /root/tp-agent/cluster/sdaa_fs_probe.py 本机被 compute guard 硬拦
  （120s 工具超时 + nohup/setsid 拦截；CMB3/CONT2 同款）⇒ 集群正典车道（run_compute，stage_dir=工作树，
  8 核，CMB3 先例）。包装器已最小扩展 --cav-clos-alpha 旗标（默认 0.0=位级零行为，备份 .bak.m27075）。
- 集群账单号：cj6093（α=0.5 主臂）/ cj6094（α=0.3）/ cj6095（α=0.15）；
  作业日志 /root/tp-agent/cluster/jobs/cj609{3,4,5}.log（rc 以日志为准——门禁 DB 的 rc 列恒 0，
  系 cluster_exec c_run 未传播退出码，见 §7）。

## 4. 实测三元组（程序化取自 runs/K1-CAVCLOS-R2*/result.json，零手抄）
| 臂 | 账单号 | α | stop | 停跑点(ms pc) | peak_depth(D) | 峰时刻(ms pc) | peak_error | postcontact_healthy_ms | mach_max |
|---|---|---|---|---|---|---|---|---|---|
| ctrl（CMB3 卡引值，同车道位级复核过） | cj6069 | 0 | window_complete | 24.005 | 1.435220 | 10.543339 | -22.73% | 24.005 | 0.2598 |
| 主臂 K1-CAVCLOS-R2 | cj6093 | 0.5 | health_limit | 16.052 | 1.364683 | 10.348768 | -26.53% | 16.052 | 0.2078 |
| 阶梯 a03 | cj6094 | 0.3 | health_limit | 1.094 | 0.275848（censored） | 1.082304 | -85.15% | 1.094 | 0.1463 |
| 阶梯 a015 | cj6095 | 0.15 | window_complete | 24.005 | 1.374095 | 11.406750 | -26.02% | 24.005 | 0.2017 |
| 实验 | — | — | — | — | 1.857434 | 10.394 | — | — | — |

## 5. 判决（对照预登记）
- 主臂 α=0.5：预登记带 [1.50,1.70]D **未命中**（1.364683 < 1.50，带下沿之下 0.135D）；
  证伪项：|peak−ctrl| = 0.070537 D > 0.05（未触发"无效"线）；停跑 16.052 ms pc > 10.394（未触发早死线）。
  判定=**OUT_OF_BAND（方向相反）**：处方未加深，反而减深 0.070537D（vs ctrl），峰时刻 10.3488 ms pc
  （比 ctrl 早 0.195ms、比实验早 0.046ms），且把 ctrl 的满窗存活打成 16.052 ms pc 漂移门死
  （geometric_volume_drift 1.60e-4 > 1e-4）。
- 阶梯（稳定前沿，非单调）：α=0.15 满窗（24.005 ms pc，峰 1.374095D@11.406750ms pc，亦减深 0.061125D）；
  α=0.3 判决点前 1.094 ms pc 早死（censored，预登记早死线触发→FAIL）；α=0.5 死于 16.052 ms pc。
  剂量-响应：0.15→1.3741 / 0.3→0.2758(cens) / 0.5→1.3647，全在 ctrl 之下，无单调加深趋势。
- 物理读法：模型空穴本就不受"过闭合"限制（RIM 判别：封口晚于实验、峰前腔口速度未反号、腔气持续充入）⇒
  延缓闭合拿不到额外深度，只把界面转换推迟、扰动质量守恒（drift 触门）。**K1 轴独立价值=负**，
  与 RIM 动量瓶颈判别一致；后续组合波不应再叠加 K1 处方。

## 6. 误差判定（逐项，显式回答 <3%）
| 关键量 | 值 | 基准 | 误差 | <3%? |
|---|---|---|---|---|
| 主臂峰深 | 1.364683 D | 实验 1.857434 D | -26.53% | 否 |
| 主臂峰时刻 | 10.348768 ms pc | 实验 10.394 ms pc | -0.43% | 是（减深臂巧合，不构成通过） |
| 主臂峰深 vs ctrl | 1.364683 D | ctrl 1.435220 D | -4.91%（减深） | 否（方向相反） |
| 主臂停跑点 | 16.052 ms pc | 窗 24.005 ms pc | -33.1% | 否 |
| a015 峰深 | 1.374095 D | 实验 1.857434 D | -26.02% | 否 |
结论：**深度门未达 3%（主臂 -26.53%，方向与处方预期相反）；时刻门数值 -0.43% 系减深臂巧合，不构成通过**。

## 7. 诚实披露
- 判定力门：任务书"history 末 step ≥9000"在本基准不可行——cells=8 满窗=2019 步（24.005 ms pc，
  CMB3 卡 ctrl 同参复现），最大末 step=2019。等价判定力门=满窗 window_complete 或判决点(10.394 ms pc)后停跑；
  主臂 1365 步/16.052 ms pc 停跑，峰(10.349 ms pc)远早于停跑点，峰有效非 censored；a015 满窗 2019 步。
  烟窗数据未用于判定。
- 门禁 rc 误报：cluster_exec c_run 的 return rc2 被 main 丢弃 ⇒ 门禁 DB 恒记 succeeded/rc=0；
  本单三次"假成功"（cj6091/6092 import 错、cj6090 GitHub 不可达）均靠作业日志识破。已如实登记，未改门禁（非本单范围）。
- 车道偏离：任务铁律点名 SDAA 共享包装器；本机 guard 硬拦长进程（CMB3/CONT2 同款）⇒ 集群正典车道，
  包装器已备好 --cav-clos-alpha 供 SDAA 车道后续复用。
- 未调优：α 仅按任务书 0.5/0.3/0.15 三档；几何/旗标全取战役锚。
- 局限：cells=8 单锚单车道；"全局 INTERFACE 施加"口径（非仅壁面胞）；结论不外推其它分辨率/几何。
