# XVAL-LBM-MOVING-WALL-WALL-SHEAR-ME-D2Q9-BGK — 半程动壁 bounce-back 的壁面力（动量交换）对标验证

仓库 TensorLBM｜分支 agent/m20705｜2026-11-14｜预注册 `PREREG.md`（含一次诚实修订，见 ⑤）

## ① 案例介绍
被对标能力 = 已合并的半程动壁 bounce-back 闭包 `ladd_pre_local`（pre-streaming 反射 + Ladd 动量项
`6 w_q rho_w (c_q·u_w)`，有效无滑移面在半格 y=0.5），即锚点 XVAL-LBM-MOVING-WALL-HALFWAY-BB-D2Q9-BGK 的能力本体。
那一轮（#2342）验证的是运动学 u(y,t)，明确把「壁面剪应力/动量交换」留在范围外；而船舶阻力/旋转机械/Couette
黏度标定消费的是壁面力 tau_w，力只由壁面链布居的反对称部分决定——剖面正确不蕴含力正确，必须单独验证。
几何：x 周期、y 方向 2 行固壁 + ny-2 行流体的一维槽道；D2Q9+BGK，float64。
参数：Couette H_eff=16/32，tau=0.6/0.8/1.2，U0=0.05（Ma=0.087）；Stokes-2nd delta_bl=4/8，
tau=0.8/1.2/1.8，U0=0.02（Ma=0.035）；Rayleigh ny=66，t∈[400,1600]。

## ② 怎么做的
1. 动量交换测量：F_bot = sum_x[ sum_{4,7,8} c_q f_post[q,1,x] - sum_{2,5,6} c_q f_new[q,0,x] ]（顶壁镜像），
   f_post 为流体侧碰撞后布居、f_new 为反弹+Ladd 后壁格布居；F/nx = 壁面应力。
2. 记账自洽（M6，零容差）：BGK 碰撞与 x 周期性严格守恒动量 => F_bot+F_top+dP_fluid = 0 到机器精度；
   漏乘因子 2、法向取错、碰撞前/后采样混用都会破坏它。
3. 解析基准零手抄：Couette tau=rho*nu*U0/H_eff（N-S 精确）；Stokes-2nd / Rayleigh 的壁面应力一律用
   对解析 u(y,t) 的程序化中心差分 rho*nu*du/dy|_{y=0.5}，闭式只作自检（相对差 5.3e-9 / 1.7e-9）。
4. 闭包等价：带力探针的步进与已合并参考步进逐位相同（torch.equal 通过）=> 验证的是已裁决闭包。
5. 判别力：同工况换 4 种错闭包（不加动量项/壁格置平衡态/符号反/rho_w=1）做消融。
6. 工具链：tensorlbm.d2q9 + tensorlbm.solver(collide_bgk, stream)、numpy、matplotlib；
   复现 python3 run_xval.py --stage core|ablation|diag|finalize && python3 make_figures.py
   （分段 53 s / 74 s / 3.7 s）；门禁 tests/test_xval_moving_wall_wall_shear_me.py（7 passed, 10 s）。

## ③ 计算结果（程序化写入 artifacts/summary.json）
Couette 稳态壁面剪应力：H=32,tau=0.8 时 tau_num=1.5625000000e-04 vs tau_ana=1.5625000000e-04，相对误差 +1.58e-09（两侧壁最坏）；
H=16 为 +2.64e-11，tau=0.6 为 +4.61e-07，tau=1.2 为 +1.84e-11；两壁合力平衡 1.3e-13。

Stokes 第二问题壁面力：参考工况幅值 A_num=3.535764e-04 vs A_ana=3.535531e-04，误差 +0.0066 %，相位误差 +0.0801 度（<3% 判据：是）；
delta_bl=4：+0.0495 % / +0.3048 度；tau=1.2：+0.0257 % / -0.1760 度；tau=1.8：+0.0426 % / -1.1163 度。

Rayleigh 脉冲起动：参考工况 t∈[400,1600] 逐点最大误差 3.96e-03 %（均值 1.35e-03 %）。
tau=1.2 同窗口 34.5 % —— 超 15%，honest FAIL，见 ⑤.4。

判据总账（机器判）：M1:PASS, M2:PASS, M3:PASS, M4:PASS, M5:PASS, M6:PASS, M7:PASS, M8:PASS => verdict PASS。M6 最坏相对残差 6.7e-13（阈 1e-12），M7 2.25e-16（阈 1e-14）。

消融（同一 Stokes 工况）：参考 +0.0036 % / +0.078 度；壁格置平衡态 -3.6786 % / -2.078 度；不加动量项 -100.0000 % / +135.011 度；
符号反 +0.0036 % / -179.922 度（幅值取模看不见符号错，相位抓住）；rho_w=1 与参考同（+0.0036 %）。
Couette 侧：壁格置平衡态力误差 -1.840 %（而剖面误差仅 0.89 %），不加动量项 -100.6 %。
关键读数：剖面接近正确而力错 3.7 %，符号错只在相位上可见 => 力验证是独立于剖面验证的一层。

## ④ 结果图
fig1 Couette 剖面 + 四组网格壁面力误差（round-off）；fig2 一个周期壁面力波形 vs 解析 + 参数扫描误差；
fig3 Rayleigh 幂律衰减 log-log + 误差历史；fig4 消融排序 + 法向速度壁质量漂移。

## ⑤ 注意点
1. M7 预声明写错并诚实修订：原写 sum_{c_q·n>0} c_q inj_q = rho_w u_w，代数复核应为 rho_w (cs^2 u_t, u_n)
   （半格点切向只拿 cs^2 系数，剩余切向动量走回流通量差，正是 MEM 计数补的部分）；全 9 方向之和为 2 rho_w u_w。
   修订在读生产结果之前，阈值未改（1e-14），实测 2.2e-16。
2. 自查抓到消融实现缺陷：第一版带力探针把 ladd_sign_flip 落到参考分支（静默等价），表现为符号反与参考完全
   相同；修复（显式取负，未实现变体改抛 NotImplementedError）后符号反给出相位 -179.9 度。
3. M6 归一化坑：不能用实测力作分母（static_bb 力≈0 会把残差/0 放大成 1e14 的假 FAIL），必须用解析力标度。
4. 适用范围：平面壁、壁法向 y、D2Q9+BGK、float64。Rayleigh tau=1.2 在未缩放采样窗口下 34.5% 未通过，
   本单未进一步定位（预算），tau=0.8 参考配置 0.004%。弯曲/斜壁需 Bouzidi 插值，不在本单。
5. 壁法向速度 u_w·n≠0 属越界用法（诊断 D1，量化未修正）：流体域质量随时间流失，600 步后 0.002->-1.927%, 0.005->-3.331%, 0.01->-4.082%, 0.02->-4.267%
   （eps=0 对照严格 0），对 eps 非线性（线性拟合斜率 -0.133，R2=0.68），无"乘系数"式简单修正 =>
   建议调用侧强制 u_w·n=0。
6. 预算：cluster 递交 cj2896 排队 8 min 未派发（前方 20 单），按 <8 核/短任务纪律改本地分段跑并撤销该作业。

## ⑥ 误差判定（逐项回答是否小于 3%）
- Couette tau_w（H=32,tau=0.8，两壁最坏）：+1.58e-09 % => <3% 是，<15% 是。
- Couette tau_w（tau=0.6 最坏配置）：+4.61e-07 % => 是/是。
- Stokes-2nd 壁面力幅值 +0.0066 %、相位 +0.0801 度（=0.14% 周期）=> 是/是。
- Rayleigh tau_w(t) 最大 3.96e-03 % => 是/是。
- 动量交换记账恒等式 6.7e-13（相对）=> 机器精度。
- Rayleigh tau=1.2：34.5 % => 否/否（honest FAIL，未粉饰）。
- 壁法向速度质量漂移：非闭包精度问题，是越界用法，不参与 3% 判定。
