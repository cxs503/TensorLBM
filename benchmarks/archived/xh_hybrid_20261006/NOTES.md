# XH 联合多相模型 Phase 0 — 执行记录（NOTES）

任务 #234 · XH 轨 · agent 执行 · 2026-10-01 启动，2026-10-06 停批。
规则遵守声明：Phase 0 全程 CPU（CUDA_VISIBLE_DEVICES 置空，OMP≤16/
进程）、零引擎改动（worktree 只读）、未推送未开 PR、同事数据零触碰、
全部产物落 /nfs/wangxi/runs/xh_hybrid_20261006/。

## 1. 时间线

- 10-01：领任务；读取 X1 归因表、W9-B 判决、M1 标定档案、库模块
  （color_gradient2d.py / cac_lbm.py / phasefield/ / allen_cahn_lbm.py）、
  库冻结基准 poiseuille_two_phase_cg（run.py+result.json+cg_reference.py）、
  控制器 ctrl_x1.py 判据约定。
- 10-02：文献锚点核验（Fakhari 2017 JCP / Mitchell 2018 AWR / Ba 2016 PRE /
  Liang PRE 97,033309 等）；三候选架构成形。
- 10-06 早：5090 侧 git fetch（github 直连断）→ 走记忆配方：本地 SOCKS
  隧道（跳板 8.215.87.176:10870）+ 反向隧道 10871 → 浅 fetch origin main
  → **worktree /nfs/wangxi/worktrees/xh_hyb @ 4ba836c9977b9607788b9f0eca0a46a703bcc0e9**
  （= PR #327 exp/c1g2-two-phase-poiseuille 合并点）。
  **披露**：fetch 为 --depth 1（浅历史）；未来完整 fetch 会自动补深；
  主检出仅执行 fetch+worktree add，无任何写操作。
- 10-06 09:25-09:34：本地暂存 /nfs/wangxi/tmp/xh_stage/ 写探针 4 件
  （xh_common.py + p1-p4）+ vendored cg_reference.py（未改一字）；
  py_compile 过；scp 上 5090。
- 10-06 09:35：冒烟（各 30-200 步）全过：四模块导入、模板自检、
  **p1 库模块 vs 克隆位一致**（stock 臂 200 步 torch.equal）。
- 10-06 09:36-09:37：全量启动。**披露**：第一次批量 nohup 因相对路径
  logs/ 竞态只启动了 p1（p2/p3/p4 秒死，零输出）；立即以绝对路径重启
  四进程，无数据损失（p1 未受影响）。
- 10-06 09:42：五份 JSON 全部落盘（p4 两臂并行各 ~300s；p1 ~7min；
  p2 ~2min；p3 ~1min；累计 CPU 墙钟 ~22min，远低于 30min/探针）。
- 10-06 上午：memo/prereg 写制；本 NOTES；md5 冻结；停批报告。

## 2. 快照（判决可复现性锚）

- worktree：/nfs/wangxi/worktrees/xh_hyb @ 4ba836c9977b9607788b9f0eca0a46a703bcc0e9
- 引擎 md5（探针 env 块机器记录于各 JSON）：
  color_gradient2d.py=350a3230c7a9578ab4353d6e95ca2db6
  cac_lbm.py=ccc671ee4738d9266967e74f1fd8924b
  （d2q9/solver/boundaries/cg_reference/xh_common 的 md5 见 JSON env 段）
- 环境：5090（192 核）、/nfs/wangxi/venvs/tensorlbm、torch 版本号在 JSON
  env.torch、fp64（p1-p3）/fp32 步进+fp64 诊断（p4）。
- 参考锁：H-R 手算例 U_t=9.997714285714286e-4（agent 独立复算 ✓）；
  M1 b2_a 归档 Eu=0.07702678080491326 与 σ_eff 阶梯（机读自
  /nfs/wangxi/runs/bm_mp_20261001/m1_cac_fix/out/{b2_a_ny100,calib_route}.json，
  P2/P4 脚本运行时直读，零手抄）。

## 3. 探针判定速览（数字全部来自 out/*.json）

- P1：9/10 门 PASS。位一致 ✓；p1a relL2=0.002109（库冻结 0.002109392211371581
  复现 6 位）；漂移 ≤9.4e-17/步/色；chem-力 Poiseuille relL2 1.49e-4/1.25e-3；
  σ 线性 2.0014；R 散布 0.3-0.5%；β 耦合 spread 30.4%（诚实 FAIL，唯一）。
- P2：迁移门 FAIL（0.469@W4 vs M1 3D 0.924，偏差 0.455）；线性 0.99999742；
  单调 ✓；2D 阶梯 0.425/0.469/0.487/0.494（W2/4/6/8）。
- P3：m0 C 1.998→1.025（塌缩复现）；m1 →1.0245（力无效）；m2 →1.084
  FAIL 且漂移 3.16e-5/步/色 + u_max 0.476（修正案路线杀死）。
- P4：r0 Eu=0.076977（归档 0.077027，|Δ|5e-5 ✓）+ 解析约定对归档 2.0e-15 ✓；
  **r1 Eu=0.07165 ≥0.04 → H_fw**；壁带 0.155 vs 中带 0.064；
  r0/r1 壁带几乎相同（0.156/0.155）→ 壁机制病与黏度比无关。

## 4. 结论与去向

推荐 XH-1（详见 memo/architecture_memo.md）；XH-2 由 P4 降级为条件
备选（先修 CAC 壁带 ≤3%）；XH-3 兜底。prereg.md 为冻结草案待批。
Phase 1 未启动（停批协议）。

## 5. 控制器批复（Phase 0 通过 → Phase 1 放行，2026-10-06T11:05Z）

- 亲验完成：五份 out/*.json 与控制器本地副本 md5 全 MATCH；M1 锚三处机读命中
  （b2_a_ny100.json:/Eu、calib_route.json:/bars、g2_poiseuille.json eq_ny64）；
  全部判决数字由控制器从 JSON 原始字段独立复算（σ_eff 散布 30.400%、线性
  2.001391、P4 r0|r1 壁带 0.15637|0.15499、P3 C_end 三臂、r1 Eu 判 H_fw 等）；
  T3 锁定手算例控制器独立复算逐位一致。核验脚本 /nfs/wangxi/tmp/ctrl_xh_p0_verify.py。
- **批复：XH-1 架构批准。prereg.md（md5 80513afc24bd28a73225a2bc47132f1d）自本行起冻结**，
  其后仅 append-only 修正案（XH-A*，注明日期/动机/数值影响）；NOTES.md 同规则仅 append。
- Phase 1 放行 GPU5。执行顺序：T1→T2（T4/T5 随臂仪器化）→压力形式升级探针
  （R5 色账本先行，1e-10/步/色门）→T3→T6；SL-5 于 28.8 GPU·h 跳闸。
  首个 result.json 必记：本 md5 + 评估器（E1/E1'/E2/E3）md5 快照 + GPU 快照。
- 批注两则：memo §6 "20-24 GPU·h" 为粗述，预算一律以 prereg §6（≈15.6，
  保守上界 ~28）为准，无门值冲突；报告中 sha 一律脚本生成，文件字节为准。

## §7 Phase-1 预跑解释修正（append-only，均在对应臂开跑前登记；XH-A 系列，无任何门值改动）

- **XH-A0（T1(a) route-B 验收读法）**：route B 下验收臂按每 W 修正输入
  σ_in(W)=σ_cal/σ̄_eff(W)（主读法：校准后每个实测 W 自带修正输入）；
  pinned σ_cal/σ̄_eff(W*) 输入的邻 W 臂作字面读法披露组。两组都跑、
  都入 result.json，门只挂主读法组（t1 脚本开跑前 docstring 即此读法）。
- **XH-A1（T3 几何与参数解释）**：prereg T3 "D∈{64,96} 等效球径；域 3:1:3"
  与预算表 0.79M/2.65M 胞唯一自洽读法 = 域 (nz,ny,nx)=(D,3D,D)、液滴半径
  a=D/8（横截面 4 倍液滴直径、中心壁距 12a；D=96 臂 a=12 恰复现 E3 锁例
  几何）。Re_t=0.05 锁（W9-B 血统），g 按臂机器反解；σ_target=0.01
  （Ca≈0.012），σ_input=σ_target/σ̄_eff(W*) 由 T1 route-B 阶梯按线性门
  外推。液滴初始 y0=ny/2−U_t·steps/2 使测量轨迹居中（窗口内壁修正漂移
  最小的测量协议选择，非门改动）。另加 1 条非门归因臂（λ=0.5, D=64,
  a=D/16, 40k 步，走 20% 返工余量）：壁限域主导则误差约减半、离散主导
  则增大——SL-4 报告的归因判别器。
- **XH-A1b（T6 臂结构解释）**："σ_rec 门同 T1(a)" 读为校准后门：先以
  T1 传递 σ_in 跑 calib_R24 测 σ_eff10（ρ 比 10 下的传递），验收臂用
  σ_target/σ_eff10；pinned T1 传递输入的 R20/R28 臂作披露组，共 6 臂
  恰合预算行。ρ100 探针仅在 ρ10 全门绿后跑（prereg"若 10 过再探 100"，
  非门）。
- **XH-A2（T5 无量纲尺度常数）**：T1 记录的 u_max/(σ/ρ·R)^{1/2} 用了
  ρ_bulk=2.0，而引擎 plain 态总密度=1（t0 f3 位一致回执）→ 该披露档
  无量纲分母差 √2；原始 u_max 与全部 T1 门不受影响。
  t1_t5_dimensionless_patch.py 机器侧重导（JSON-in/JSON-out，双尺度
  同录可审）。
- **XH-A3（T2/T3 的 T5 适用范围）**：u_max>1e-2 杀条款面向静态臂；T2 为
  驱动 Poiseuille（物理流 u_max≈1.4e-2，与库冻结臂同量）、T3 为物理
  上升流（U_t≈3.5-5.2e-4）→ 以轮廓相对诊断 / 内部均值-uy 交叉核对分离
  物理流与杂散流，机器字段 t5_scope 如实记录该读法。
- **XH-A4（T1 v1 选臂 bug 披露）**：t1_laplace.py v1 的验收池选取
  `find(calibration) or find(acc)` 在 route B 下优先取了未修正输入的标定臂
  （该复用模式只对 route A 成立），致 (a)/(c) 门挂在错误臂组上（v1 判
  FAIL 系假阴性：修正臂组全部 |dev|≤0.54%，线性比修正基底后
  ≈1.993∈[1.97,2.03]）。处置：门值零改动，仅修选取逻辑后整池确定性重跑
  （fp64 CPU 无随机源），v1 原样保留为 out/t1_laplace_v1_selectionbug.json
  供审计；canonical=out/t1_laplace.json（v2）。同批顺带 XH-A2 的 ρ 尺度
  常数在脚本内改正（2.0→1.0）。**W6 的 R 散布 2.10% 为真实物理**（宽界面
  W=6 的曲率依赖，W3/4/5 为 0.106-0.191%）：(b) 门按字面"每个 W"挂全阶梯
  {3,4,5,6}，如实判该档 FAIL；验收集 {3,4,5} 读法同 JSON 记录为披露。
  T1 总判 FAIL 条件按 prereg 仅系于 (a)（route B 后仍 >3%）。

- **XH-A5（R5 判决前登记：压力形式静驻种群可行性界）**：账本精确的
  压力修正 corr_i=w_i·Ψ/cs²（i≠0）、corr_0=(w0−1)·Ψ/cs² 使轻侧体区的
  静驻种群 f_0 = w0·ρ_min + (w0−1)(p0/cs² − ρ_min)。自动 p0=cs²·ρ_max
  下 ρ10 红侧 Ψ/cs²=9 → f_0 = 4/9 − 5 = **−4.5556**，与
  out/e2diag.json 五个关断变体（力/重染色/korteweg/fp32）step-1 的
  f_min_r=−4.555554 逐位一致 → ρ10 爆炸由平衡态本身播种，与化学力、
  重染色、korteweg 项、fp32 均无关（全部变体仍 NaN@20-100）。
  两端约束：重染色供给需 p0/cs² ≥ 0.318·ρ_max（r5diag d 格证据）；
  静驻正性需 p0/cs² ≤ 1.8·ρ_min → 窗口非空当且仅当 ρ_max/ρ_min ≲
  5.66。ρ2 窗口 [0.636,1.8]，自动 p0/cs²=2.0 略超上沿（f_0=−0.111，
  e1 实测 6000 步稳定、umax 3.0e-4、C 保持 0.5518；同池 plain 对照
  C→1.0449 = P3 坍缩签名，A/B 闭合）；ρ10 窗口 [3.18,1.8] **为空**。
  数学事实：任何零零次矩、二阶矩 ΨI 的修正必含负分量（Σcorr=0 且
  corr≥0 ⇒ corr≡0）；库 cac_lbm 的压力形式走的是另一架构（单一
  g 场，零次矩即 p/cs²，静驻 (w0−1)p/cs² 恒负），与批准冻结的 XH-1
  双色骨架不可互换，Phase-1 范围内不换架构。SL-3 计数：fix_attempt_
  count=3（基线 p0=0.1 失败；korteweg=False 无效；p0=cs²·max 修好
  ρ≤2、不修 ρ10）已超两次上限。后果按 prereg SL-3：**T6 如实
  N/A+FAIL，XH-1 降级"低密度比专用"，停批**；T3-at-ρ2 是否继续由
  控制器决定。R5 判决字段拆分（本条登记后方运行）：机器字段
  verdict_T3_precondition_met_rho2（账本 ∧ ρ2 臂稳定 ∧ 保持率≥0.9）
  与 verdict_T6_infeasible_sl3_trip 分列，ρ10 臂各门保持字面。

- **XH-A6（控制器三项裁定，T3 放行条件；2026-10-06，T3 开跑前登记）**：
  Phase 1 停批报告全部判决数经控制器亲验后下达，本条为裁定登记（无任何
  门值改动）。
  1. **fp32 T4 读法=归因披露采纳**：prereg T4 门（≤1e-10/步/色）的定义域
     =fp64 臂（prereg 原文括注）。3D fp32 膜 2.96e-8/步=权重舍入地板
     （e3diag 链：d3q19 权重表 f32 和=1+1.49e-8，双色重染色分裂翻倍；
     r5 pA_rho2 实测 2.9566e-8 逐位同量级），观测级影响（80k 步质量
     呼吸 ~0.23%）≪3% 门。据此 verdict_T3_precondition_met_rho2 机器
     字段读法登记为：字面 False（含越域 fp32 项）但物理前置成立
     （ρ2 稳定 ∧ 保持率≥0.9 ∧ fp64 账本净：t1 5.56e-17 / t2 ≤4.4e-17 /
     e3diag pform_fp64_3d ≤1.04e-16，全部机读）。ulp-exact fp32 权重表
     重跑明确否决——重测舍入恒等式无信息量，不执行。
  2. **SL-2 弃权**：T2 唯一超线臂 g10_ny64（E1 relL2 4.573%）的误差为
     分辨率衰减的幅值/形状亏缺。控制器 12 变体独立解析扫描证实：σ 通道
     不进平界面 Poiseuille 估计器（界面平行于流动、无曲率）；E1/E1'/
     控制器自有估计器三读法同判 >3%。一次 β/W* 重标定不改变判决，T2
     定案 honest FAIL。SL-2 权利登记为"保留但弃权，物理理由如上"。
  3. **T6 维持 prereg SL-3 判 N/A+FAIL；XH-1 降级"低密度比专用"；
     T3-at-ρ2 放行**（条件：fp32 归因披露行随每臂档落盘，机读源=
     e3diag/r5/t1/t2 JSON，零手抄）。

- **XH-T3 停批记录（2026-10-06，XH-A6 后 T3-at-rho2 执行完毕；全部数字机读自
  phase1/out/t3_hadroplet.json，脚本 t3_hadroplet.py v2 md5 3343afb4c32397713e1b642349f45e3e）**：
  4 门臂全稳定（NaN=0）。机器门：
  g_t3_err_le_3pct_per_arm=false
  （lam0.5_D64 err=-2.3235%、lam0.5_D96 err=-4.3139%、
  lam2.0_D64 err=-0.8125%、lam2.0_D96 err=+0.6273%）；
  分 lambda 判：lambda=0.5 FAIL（D96 破 3%）、lambda=2.0 PASS。
  g_t3_steady_le_2pct=false（唯一破者 lam0.5_D96
  steady=2.1448%）。
  g_t3_D_ladder_monotone=false
  （字面符号比较：lambda0.5=true/lambda2.0=false；
  幅值交叉读法：lambda0.5=false/lambda2.0=true——
  两读法合取同判 False：lambda0.5 误差随细化增大，lambda2.0 过零收敛）。
  g_t4_ledger_le_1e-10=false
  （字面；T4 门定义域=fp64 臂，T3 全 fp32 膜 drift_max=3.887e-08/步
  =e3diag 权重舍入地板 2x1.490e-08，
  每臂 fp32 归因披露行已随臂落盘 arms[*].t4_fp32_attribution，XH-A6 读法登记不变）。
  g_all_stable=true。归因臂（非门，lambda=0.5/D=64/a=4）：err=+49.6037%
  ——a 减半误差剧增，XH-A1 判别器判离散/界面分辨率主导（非壁限域）。
  SL-4 不触发（双 lambda 皆 >3% 不成立：lambda=2.0 双臂过门）。
  GPU 计量：本跑 1.3888 GPU·h、Phase-1 累计
  1.3975 GPU·h（SL-5 线 28.8 未触）。
  T3 总判：FAIL（lambda=0.5 分判 FAIL；honest，未调门未加臂）。
