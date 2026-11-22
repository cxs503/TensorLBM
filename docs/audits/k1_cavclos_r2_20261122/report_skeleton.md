# K1-CAVCLOS-R2 进展报告（骨架，数字程序化注入）

## ① 案例介绍
- 物理背景：单液滴入水（Santini case 3 口径）空穴演化。实验（fig5，18 测点，pc 口径）峰深 1.857434D @ 10.394ms(pc)。
  模型 ctrl 基线 1.435220D @ 11.091ms(abs)/10.543339ms(pc)，峰深误差 -22.73%。
  组合波后继战役（owner 令铺开组合比较）：combo_hydro（RR+SMAG+HYDRO）已收至 1.750541D（-5.75%），
  但 K1 轴（空穴闭合阈值延缓）独立价值从未检验——前作 m26737 只跑 step-100 烟窗（1.216ms）即报成功。
- 几何/参数表：cells=8（1 cell=0.125D≈6.7%）、gap=0.18D、speed=0.06、release=1.53 m/s、
  span=4D、pool_depth=3D、u_limit=0.15、capacity+simple-exchange、contact_step=45、窗 24.005ms(pc)。
- 处方：界面→液体转换阈值 κ→κ·(1+α)，α=0.5（空穴壁面液膜需更厚才转液 ⇒ 延缓闭合 ⇒ 回流供能 ⇒ 加深）。

## ② 怎么做的
- 继承 m26737 接线（--cav-clos-alpha，free_surface_probe.py + free_surface_lbm.py L1509-1510），勿重写。
- 车道：SDAA 共享包装器本机被 compute guard 硬拦（120s 工具超时 + nohup/setsid 拦截；CMB3/CONT2 同款）
  ⇒ 集群正典车道（run_compute，stage_dir=工作树，8 核，CMB3 先例）。包装器已最小扩展 --cav-clos-alpha
  （默认 0.0=位级零行为，备份 .bak.m27075）。
- 命令：PYTHONPATH=.:src python3 benchmarks/water_entry/single_drop/free_surface_probe.py
  --cells 8 --gap 0.18 --speed 0.06 --capacity --simple-exchange --release-speed 1.53 --span 4
  --pool-depth 3 --u-limit 0.15 --cav-clos-alpha 0.5 --steps 10000 --out runs/K1-CAVCLOS-R2/result.json
  --planes runs/K1-CAVCLOS-R2/planes.npz
- 集群账单号：cj6093（主臂 α=0.5）。

## ③ 计算结果
（待填：三元组表 + 阶梯表）

## ④ 结果图
- fig_depth_curve.png：深度-时刻曲线（模型 vs 实验 18 点 + ctrl 锚 + 预登记带 + 判决点线）
- fig_field_planes.png：中面 fill 云图 @ 4 注册时刻（1.396/6.4/10.394/13.4 ms pc）

## ⑤ 注意点
（待填：门禁 rc 误报、PYTHONPATH 坑、末 step≥9000 判据不可行性、全局 vs 壁面胞口径）

## ⑥ 误差判定
（待填：逐项 vs 基准，显式回答 <3%）
