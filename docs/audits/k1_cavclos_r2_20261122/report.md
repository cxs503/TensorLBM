# K1 空穴过闭合抑制处方满窗定谳：方向相反（减深）+ 非单调稳定前沿 —— honest-FAIL

## ① 案例介绍
**物理背景**：单液滴入水（Santini case 3 口径）空穴演化。实验（fig5，18 测点，post-contact 口径）峰深
1.857434D @ 10.394ms(pc)。模型 ctrl 基线 1.435220D @ 11.091ms(abs)/10.543339ms(pc)，峰深误差 -22.73%。
组合波后继战役（owner 令铺开组合比较）中，combo_hydro（RR+SMAG+HYDRO）已把峰深收至 1.750541D（-5.75%），
但 K1 轴（空穴闭合阈值延缓）的独立价值从未被检验——前作 m26737 只跑 step-100 烟窗（1.216ms）即报成功，
预登记 [1.50,1.70]D 从未落地。本单重开并满窗检验。

**处方**：free_surface_lbm.py 界面→液体转换阈值 κ→κ·(1+α)，α=0.5（主臂）。
物理假设=空穴壁面液膜需更厚才转液 ⇒ 延缓空穴闭合 ⇒ 回流持续供能 ⇒ 加深。

**几何/参数表**（战役锚，全臂一致）：

| 参数 | 值 | 参数 | 值 |
|---|---|---|---|
| cells | 8（1 cell=0.125D≈6.7%） | span | 4D |
| gap | 0.18D | pool_depth | 3D |
| speed | 0.06 | release_speed | 1.53 m/s |
| u_limit | 0.15 | capacity / simple-exchange | on / on |
| contact_step | 45 | 窗 | 24.005 ms(pc) |
| steps | 10000（满窗=2019 步） | α 阶梯 | 0.5 / 0.3 / 0.15 |

## ② 怎么做的
- **继承接线（勿重写）**：m26737（commit ed91ff780，已并入 main 28d1bf631）已穿参 --cav-clos-alpha
  （free_surface_probe.py CLI + probe() 参数；free_surface_lbm.py L1509-1510：
  `_cav_clos = _kappa*(1+cav_clos_alpha) if cav_clos_alpha>0 else _kappa`，
  `to_liq = interface_mask & (fill > 1.0 + _cav_clos)`）。口径：对全部 INTERFACE 胞的 I→L 转换行生效
  （任务书"仅空穴壁面胞"为物理意图描述；预登记即按此接线登记）。
- **车道**：SDAA 共享包装器 /root/tp-agent/cluster/sdaa_fs_probe.py 本机被 compute guard 硬拦
  （120s 工具超时 + nohup/setsid 拦截；CMB3/CONT2 同款）⇒ 集群正典车道（run_compute，stage_dir=工作树，
  8 核，CMB3 先例）。包装器已最小扩展 --cav-clos-alpha 旗标（默认 0.0=位级零行为，备份 .bak.m27075）。
- **命令**（铁律 --steps 10000 满窗）：
  `PYTHONPATH=.:src python3 benchmarks/water_entry/single_drop/free_surface_probe.py --cells 8 --gap 0.18 --speed 0.06 --capacity --simple-exchange --release-speed 1.53 --span 4 --pool-depth 3 --u-limit 0.15 --cav-clos-alpha {0.5|0.3|0.15} --steps 10000 --out runs/K1-CAVCLOS-R2*/result.json`（主臂另加 --planes）
- **集群账单号**：cj6093（α=0.5）/ cj6094（α=0.3）/ cj6095（α=0.15）；
  作业日志 /root/tp-agent/cluster/jobs/cj609{3,4,5}.log（rc 以日志为准，见 ⑤）。
- **分析**：tools/k1r2_analyze.py（三元组+预登记判定，零手抄）+ tools/k1r2_figures.py（图）。

## ③ 计算结果（程序化注入，零手抄）
| 臂 | 账单号 | α | stop | 停跑点(ms pc) | peak_depth(D) | 峰时刻(ms pc) | peak_error | postcontact_healthy_ms | mach_max |
|---|---|---|---|---|---|---|---|---|---|
| ctrl（CMB3 卡引值，同车道位级复核过） | cj6069 | 0 | window_complete | 24.005 | 1.435220 | 10.543339 | -22.73% | 24.005 | 0.2598 |
| 主臂 | cj6093 | 0.5 | health_limit | 16.052 | **1.364683** | 10.348768 | **-26.53%** | 16.052 | 0.2078 |
| 阶梯 a03 | cj6094 | 0.3 | health_limit | 1.094 | 0.275848（censored） | 1.082304 | -85.15% | 1.094 | 0.1463 |
| 阶梯 a015 | cj6095 | 0.15 | window_complete | 24.005 | 1.374095 | 11.406750 | -26.02% | 24.005 | 0.2017 |
| 实验 | — | — | — | — | 1.857434 | 10.394 | — | — | — |

**判决（对照预登记，继承 m26737 禁改）**：
- 主臂 α=0.5：预登记带 [1.50,1.70]D **未命中**（1.364683 < 1.50）；证伪项 |peak−ctrl|=0.070537D>0.05、
  停跑 16.052>10.394 均未触发 ⇒ **OUT_OF_BAND（方向相反）**：处方未加深，反而减深 0.070537D，
  且把 ctrl 的满窗存活打成 16.052 ms pc 漂移门死（geometric_volume_drift 1.60e-4>1e-4）。
- 阶梯（稳定前沿，**非单调**）：α=0.15 满窗（峰 1.374095D@11.406750ms pc，亦减深 0.061125D）；
  α=0.3 判决点前 1.094 ms pc 早死（censored，预登记早死线触发→FAIL）；α=0.5 死于 16.052 ms pc。
- **K1 轴独立价值=负**：三档剂量全部 ≤ ctrl，无加深趋势；与 CONT2-CONT-RIM 的动量瓶颈判别一致
  （模型封口晚于实验 +6.82%、峰前腔口速度未反号、腔气持续充入 ⇒ 空穴不受过闭合限制，延缓闭合拿不到深度）。
  后续组合波不应再叠加 K1 处方。

## ④ 结果图
- fig_depth_curve.png：深度-时刻曲线（模型 α=0.5 vs 实验 18 点 + ctrl 锚 + 预登记带 + 判决点线）
- fig_field_planes.png：中面 fill 云图 @ 4 注册时刻（1.396/6.4/10.394/13.4 ms pc）
- fig_dose_response.png：剂量-响应（峰深 vs α）+ 稳定前沿（存活 vs α，非单调）

## ⑤ 注意点
1. **门禁 rc 误报**：cluster_exec c_run 的 return rc2 被 main 丢弃 ⇒ 门禁 DB 恒记 succeeded/rc=0。
   本单三次"假成功"（cj6091/6092 import 错、cj6090 GitHub 不可达）均靠作业日志识破。
2. **PYTHONPATH 坑**：probe 以脚本路径直跑时 `import benchmarks`/`import tensorlbm` 失败，
   需 `PYTHONPATH=.:src`（SDAA 包装器 driver 内已内置等价 sys.path 注入）。
3. **判定力门**：任务书"history 末 step ≥9000"在本基准不可行——cells=8 满窗=2019 步（24.005 ms pc），
   最大末 step=2019。等价门=满窗 window_complete 或判决点(10.394 ms pc)后停跑；主臂峰(10.349 ms pc)
   远早于停跑点(16.052 ms pc)，峰有效非 censored；a015 满窗 2019 步。烟窗数据未用于判定。
4. **车道偏离**：任务铁律点名 SDAA 共享包装器；本机 guard 硬拦长进程 ⇒ 集群正典车道（CMB3 先例），
   包装器已备好 --cav-clos-alpha 供 SDAA 车道后续复用。
5. **局限**：cells=8 单锚单车道；"全局 INTERFACE 施加"口径（非仅壁面胞）；结论不外推其它分辨率/几何。
   未调优（α 仅任务书三档；几何/旗标全取战役锚）。

## ⑥ 误差判定（逐项，显式回答 <3%）
| 关键量 | 值 | 基准 | 误差 | <3%? |
|---|---|---|---|---|
| 主臂峰深 | 1.364683 D | 实验 1.857434 D | -26.53% | **否** |
| 主臂峰时刻 | 10.348768 ms pc | 实验 10.394 ms pc | -0.43% | 是（减深臂巧合，不构成通过） |
| 主臂峰深 vs ctrl | 1.364683 D | ctrl 1.435220 D | -4.91%（减深） | 否（方向相反） |
| 主臂停跑点 | 16.052 ms pc | 窗 24.005 ms pc | -33.1% | 否 |
| a015 峰深 | 1.374095 D | 实验 1.857434 D | -26.02% | **否** |

**结论：honest-FAIL。** 深度门未达 3%（主臂 -26.53%，且方向与处方预期相反）；时刻门数值 -0.43%
系减深臂巧合，不构成通过。K1 空穴过闭合抑制处方被否定（falsified）：延缓闭合不加深，只减深+失稳。
