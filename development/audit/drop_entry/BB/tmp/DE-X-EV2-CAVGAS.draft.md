# DE-X-EV2-CAVGAS · 气穴气体压力只读诊断探针 · 草稿

## 已读
- 任务书：DE-X-EV2-CAVGAS-R1（SDAA probe 气穴气体压力只读诊断探针，对比 CACSim 参考实现）
- 定谳前提：峰深 1.4352D@11.091ms vs 实验 1.857434D@10.394ms（-22.73%）；网格/参数/u_limit 全排除 → 气穴欠深=物理内核问题
- 物理出处：气穴气体随深度压缩，压力随体积变化（绝热 p=p0(V0/V)^γ 或等温 p=p0V0/V）；参考实现 CACSim 用真实气体相（rho_g=0.00117）使气穴更深（1.5192 vs 1.4352 D）
- 已读：CAVSWEEP.md（16 臂，u_limit=纯稳定器，无旋钮把峰深推向 1.8574D）、TRACK_EV_W1.md（ABB_DIAG + GAS_FLOW 两臂）、FORCE2-MOMENTUM_VOL.md、INT2-COXVOI.md
- 已读代码：free_surface_probe.py（420 行，probe() 主循环 + argparse 入口）

## 已得结论
- free_surface_probe.py 当前用 rho_gas=1.0（常数大气压），无真实气体相
- 任务要求：增 --diag-cavgas 旗标（默认关=位级不变），在接触后每步测量气穴体积 V 与气体密度 rho_gas_field，计算压力比 p/p0
- 预登记：预测压力比∈[0.5,2.0]（气穴压缩 2x 内）；证伪=压力比>5.0 或<0.1（物理不合理）或 stop≠window_complete
- 验证=同网格 cells=8 同旗标（--gap .18 --speed .06 --capacity --simple-exchange --release-speed 1.53 --span 4 --pool-depth 3，u_limit 0.15），对照=ctrl 基线（1.4352D@11.091/24.552ms 满窗）
- 测量三元组 peak_depth/peak_error/postcontact_healthy_ms；存活底线=满窗 ~24ms

## 下一步
1. 读 free_surface_lbm.py 中 rho_gas 处理逻辑（GAS 单元如何表示气体密度）
2. 实现 --diag-cavgas 旗标：在接触后每步测量气穴体积 V（fill<0.5 的 GAS/INTERFACE 区域）与气体密度 rho_gas_field，计算压力比 p/p0
3. 跑 cells=8 满窗验证
4. 写卡 EV2-CAVGAS-R1.md（≤55 行）+ 双写 docs/drop_entry_bb/X_EV2-CAVGAS-R1.md
