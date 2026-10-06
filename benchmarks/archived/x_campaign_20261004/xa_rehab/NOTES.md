# XA 轨 NOTES（append-only）

## 2026-10-04 初始化
- 暂存根: /nfs/wangxi/runs/x_campaign_20261004/xa_rehab/
- 基线 worktree: /nfs/wangxi/worktrees/mp_w11 @ a84745e5（只读，import 源）
- GPU 快照 13:41: GPU5=T1 正式跑 pid 2798677 (1886MiB, 97%) 勿碰；GPU6 空闲=本轨用；GPU0-3=vLLM 同事；GPU4=python 同事；GPU7 禁用。

## GPU6 使用窗口（记入此处）
| 起止 | 任务 | 备注 |
|---|---|---|
| (未使用) | | |

## 2026-10-04 XA-1 C1-G2 晋级打包 — 完成
- 交付: archive_c1g2/{run.py, README.md, result.json, color_gradient2d.py, cg_reference.py, crosscheck_report.json}
- run.py md5=3576b36137232699b090ff4d343781a1; ruff check+format --check 过（--config mp_w11/pyproject.toml, CI 口径）
- 判据数字机器重导: crosscheck 72/72 OK（对 judgments.g2 逐位 + CLOSURE 印刷数字按其舍入位全部一致; 4/4 臂 PASS, 最差 g10_ny64 relL2=1.092%）
- vendored 引擎 md5=81203d6d6f00104f4853a6db81815fe0 / 求值器 e78c6d1eb22d628182636c0df74fb26a 与冻结值逐位同
- CPU 冒烟 600 步 ×2 臂通过（eq/g10 解析链路均执行, g10 j_first_blue=33 与正式档一致）; 未烧 GPU
- GPU6 使用: 无（本件纯打包）

## 2026-10-04 X 轨 Phase 0 冻结记录（XA-2..X1）

- XA-1 archive_c1g2/ 完成入档（result.json md5 df99f63a77e107acb034c643bc3e086e;
  run.py 3576b36137232699b090ff4d343781a1; README 47130b7413ed2a373c9da6e643c9f63d;
  crosscheck 72/72 OK; ruff CI-clean; CPU 冒烟过）。
- XA-2 xa2_m2g2_noise/prereg.md 冻结 md5 972dcd215cd11a478ca369c098c9e065
  （9 臂 2× 步复跑, R-A/R-B/R-C, ≤1.5 GPU·h, 未跑）。
- XA-3 xa3_t2_sigma/prereg.md 冻结 md5 53d1e1ab0f12115145ff3973955fbe75
  （5 条件 × P/W = 10 静态 run + 纯后处理重判, ≤3 GPU·h, 未跑）。
- XA-4 xa4_m7_servo/prereg.md 冻结 md5 a3aabc50457a890c7b41ab303e891520
  （单臂闭环伺服, CPU 16 线程 0 GPU·h, 未跑）。
- XA-5 xa5_cg_beta/prereg.md 冻结 md5 a1e9e8785d9f2a88fa3d4376921cf151
  （β 5 × (L,R) 8 = 40 run 扫描, ≤1.5 GPU·h, 未跑）。
- X1 x1_crossval/prereg.md 冻结（md5 见本目录 ctrl/prereg_md5.txt;
  消费优先设计: A-MCMP/A-SC/C-CAC/B-MCMP 纯消费冻结档零烧卡;
  A-CAC/A-CG 与 XA-3/XA-5 消费去重; 新烧卡 ≈1.5–2.0 GPU·h, 止损 4,
  任务上限 10 GPU·h, 未跑）。
- GPU6 使用窗口: 仍为零（无任何 GPU 烧卡）。全部六件处于
  停批待控制器批准状态。

## 2026-10-04 执行阶段窗口日志

- 控制器整批放行（六条件收讫）: XA-2→XA-3→XA-5→X1→XA-4, XA-4 CPU 错峰并行。
- XA-4 prereg 跑前追加 Amendment 1（φ 界 1+1e-6→1.01, 依据=冻结臂档案
  phi_max_last=1.0023946, 先于伺服臂首步落盘）: prereg.md 新 md5
  b936dd15701a11a8bfe550ffbed2c99e。
- X1 C-CG P0 CPU 探针已判: 冻结 C1 栈无剪切驱动（唯一壁 API=静止
  bounce_back_cells）→ case (c) 收缩为 CAC 行, 零 GPU;
  回执 x1_crossval/ctrl/p0_probe_cg_shear.json。
- GPU6 窗口 #1（XA-2 M2-G2 9 臂复跑）: 开窗 2026-10-04T06:09Z（快照
  ctrl/gpu6_snapshot_*.txt, 开窗前 GPU6=5MiB 空闲）, pid 2817704,
  预计 ~1h。结束时间见后续补记。

## 2026-10-05 会话中断恢复 + XA-2 判定

- 会话 10-04 14:27 被 429 配额中断; XA-2 setsid 远端进程照常跑完
  （66.6 min, 9/9 臂落盘, 15:15 末臂, 全部 run_gate pass）。
- GPU6 窗口 #1（XA-2）关闭: 06:09Z–07:15Z, 实际 1.11 GPU·h, 预算内
  （prereg ≤1.5）。窗口 #2 待控制器复核 XA-2 后开（XA-3）。
- XA-2 判定（out/xa2_judgment.json, md5 回执在档）:
  R-A 确定性 9/9 PASS（前缀尾统计 vs formal ≤1e-9）;
  R-B 不触发——err_ext(96/144/192)=0.4593/0.5506/0.4061%, 单调条款
  FAIL（144 微升维持且微增: 原 bump +0.0683pp → ext bump +0.0913pp）;
  err_ext(192)=0.4061% ≤3% 本身过; 子窗噪声尺度 0.0038–0.0331pp,
  σ 蠕变 ±0.013pp —— bump 为窗噪声的 2–14 倍 → 结构性非单调;
  R-C: M2 G2 维持 honest FAIL（翻案权在控制器, 本轨只报数）。
- XA-4 CPU 窗口开启（控制器明示并行授权）: nice 10, 16 线程,
  pid 2830165, ~10-05T02:30Z 启动; 启动断言一处修正
  （ud_target 对档案 8 位 µ_effe 舍入差 3.4e-9, 断言 1e-12→1e-8 相对,
  formal_run 原版无此断言; 修正先于任何步数落盘）。
- XA-3 放行收到（控制器 2026-10-05 终验 XA-2 19/19 过、关案）: 10 静态 run
  上限、≤3 GPU·h、OOM 降 R 回退预授权。开窗时 GPU6 被 XB 轨占用
  （pid 2832077 xb1_formal.py, 链壳 2832008, 其窗 ~6.4 GPU·h 计划）→
  按分时制式排队: 快照存 ctrl/gpu6_snapshot_pre_xa3_20261005T024316Z.txt
  （gpu6 5806 MiB/100%）, 排队看门狗 ctrl/xa3_queue_watcher.sh 已挂
  （等链壳退出 + 3 分钟连续空闲 + 快照复核后自动开窗, 弃窗即中止重排）。
- XA-3 GPU6 window OPEN (queued behind XB chain 2832008, watcher-fired):
  snapshot=xa3_t2_sigma/ctrl/gpu6_snapshot_pre_xa3_20261005T053931Z.txt (gpu6 5 MiB,
  xb_procs=0); launch xa3_run.py (CUDA_VISIBLE_DEVICES=6, OMP=16).
  Queue history: XA-3 released by controller 2026-10-05 02:5xZ; GPU6 found
  occupied by XB window (xb1_formal chain, ~6.4 GPU*h plan per xb NOTES);
  queued per time-share protocol, 3-min grace + snapshot-abort.

## 2026-10-05 控制器 GPU 排队裁定（XA-5/X1 分配 GPU5）
- XA-4 伺服已控制器终验 9/9：PROTOCOL FAIL（V1 +5.0713%>2%），无物理
  结论、全阶梯不开（ctrl_xa4.py + xa4_m7_servo/ctrl/ 输出存档）。
- GPU6 在 XA-3 退出后的下一个窗口已归 XB-2 v3（控制器 waiter2
  pid 2858028 已挂：xa3_run.py 退出 + GPU6 连续 3 分钟 <500MiB 即开
  xb2 v3；XB-1 电池 13:37 完结后 xb2_formal.py 崩溃修复为 v3 的接力）。
  XA-5/X1 勿与该窗竞争。
- **XA-5/X1 改配 GPU5**：T1 已 CLOSED（task #224，2026-10-05 终审），
  GPU5 释放（当前 5MiB/0% 亲证）。开窗制式照旧：开窗前 nvidia-smi 快照
  存 ctrl/、OMP_NUM_THREADS≤16、NOTES 记窗。GPU5 若被占则按分时排队。

## 2026-10-05 XA-3 执行窗口 #2 关闭 + 预案外故障停批（阶段 1 中断）
- GPU6 窗口 #2（XA-3）: 05:39:31Z 开（watcher 快照 ctrl/gpu6_snapshot_pre_xa3_20261005T053931Z.txt, 5MiB 空闲）→ 07:05Z 末文件落盘、"xa3 runs done”≈07:06Z 退出, ≈1.45 GPU·h, 预算 ≤3 内。启动形制=watcher setsid env CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=16 TMPDIR=/nfs/wangxi/tmp, CWD=xa3_t2_sigma/, pid 2857406。
- 10 静态 run 终态 **8/10 落盘**: 5 个变体 P 全 health_pass=True; 3 个变体 W（stp_R16/stp_R16_b8/stp_R24）成; **stp_R32_W 与 stp_R24_mu045_W 双双 OOM**（rc=1, wall 2.1s/2.2s, 无 JSON 产物; stderr 全文在 run.log）。根因（run.log OOM 报文机器字段）: 父进程 2857406 自 R32_P 峰值后缓存分配器持有 27.01 GiB 不归还驱动, 后续 W 子进程（需 ~10/~25 GiB）2 秒内分配失败。prereg §6 的 OOM 预授权回退仅覆盖“变体 P 周期域 R32 OOM→降 box 到 R24 尺寸”一形; 本形（P 成、W 败、且连坐 µ045 条件）预案外 → 按铁律停批, xa3_judge.py 未跑（其对缺失 W 文件 FileNotFoundError, 代码路径判明）。
- GPU6 接力按控制器 waiter2 计划发生: 07:15Z 观察 xb2_formal.py pid 2867897（666MiB/30%）, 本轨零干预、零竞争。
- 机对机事实档: xa3_t2_sigma/ctrl/xa3_inventory_20261005.json（10 臂存在性+md5; 由机对机提取脚本写）。P 臂 σ_eff（win=末 4 采样, bands φ>0.9/φ<0.1）: R16=0.006543204919041093 / R16_b8=0.006543277657506193 / R24=0.006497685526737044 / R32=0.006483279093117686 / R24_mu045=0.006497850880374763; 对档案 σ_rec 相对差 −0.059265/−0.059255/−0.060088/−0.060513/−0.060092（负号=H1 预期方向）。W 复刻回执: stp_R16 与 stp_R24 的 σ_rec 对档案 calib 逐位同（0.006955418942809962 / 0.0069130817299307085）; stp_R16_b8 为新量 σ_rec=0.006852921527922828（box8 带壁, 低于 box6, 无回执目标系预期）。
- 已停批报控制器, 待裁定三选: (A) GPU5 逐字重跑两条缺失 W 复刻命令（同 argv/CWD/venv, 仅 CUDA_VISIBLE_DEVICES=6→5, 估 ~26+12 min）后原版 judge; (B) 授权 judge 小改=两条 W 回执记 N/A 披露（读法 C 消费档案 σ_rec, 不吃 W 复刻; S1/S2/S3 输入不变）即跑; (C) 控制器另裁。阶段 2（XA-5, GPU5）放行同报请求。

## 2026-10-05 GPU5 窗口 #1 开（XA-5 β* 扫描; 控制器放行=停批报告裁定 A+XA-5）
- GPU5 授权依据=NOTES 2026-10-05 控制器裁定（T1 CLOSED 释放）; 开窗快照 xa5_cg_beta/ctrl/gpu5_snapshot_pre_xa5_20261005T072830Z.txt（gpu5 5 MiB）; 控制器指定序: XA-5 → 两 W 复刻重跑（GPU5 独立进程, 裁定 A）→ xa3_judge（零改动）→ 停批报告。

## 2026-10-05 GPU5 窗口 #1 关（XA-5 扫描完）+ 判定回执
- 窗口 #1（XA-5, GPU5）: 07:28:30Z 开 → 40/40 档落盘、runner ~07:47Z 退出 ≈ 0.32 GPU·h, 预算 ≤1.5 内。零 OOM、零崩溃、40/40 stable=True。
- xa5_judge.py 零改动跑讫 → out/xa5_judgment.json (md5 adba46301d0c82e72c3b3ff1c5f1567c, utc 2026-10-05T07:48:24Z): §2 锚点断言 OK; 确定性回执 (128,25) β=0.5/0.7/0.9 对 C1 档案 σ **逐位 rel_diff=0.0** 三连; 判据规则输出 = "no complete beta row (missing arms) -> report"（F_primary/F_secondary 空）。
- 根因（机对机 tally, 40 臂全查）: 健康门 health.pass=0/40——u_max_pass 0/40（u_max_final 全域 0.001279047508722276~0.010532708000281736, 全 >1e-3 门）、width_stab_pass 7/40、mass_pass 40/40、stable 40/40。**prereg 内部矛盾实证**: §2 冻结的 15 点锚表本身来自违反 §3 门的档案臂——C1 calib_sigma.json A=0.04/β=0.7/L=128/R=25 臂 u_max_final=0.0015040066198390167（档案原值, 与本扫描逐位同）即 >1e-3; 档案 calib 臂无 width 稳定字段、其健康口径仅 stable+质量。门不挪不豁免——判据如实按冻结门产出, 矛盾上交控制器裁定。
- 诊断列（非判据, stable+mass 口径 waived u_max/width 门, 机算）: F_primary_diag β=0.5/0.6/0.7/0.8/0.9 = 0.373878/0.293490/0.201064/0.203206/0.218235——即便最宽松口径也远超 D1 门 1%, 无内部极小趋势, β 维不可治 σ_eff(L,R) 迁移。仅随报告披露, 不构成判决。
- 设备差异披露行（控制器裁定 A 要求）: XA-5 扫描与下述 W 复刻重跑在 **GPU5**; XA-3 原窗与 C1/t2 存档 calib 在 **GPU6**; 同型 RTX 5090。位一致回执（determinism 三连 0.0）= 跨设备确定性成立的最强证据。

## 2026-10-05 裁定 A 执行完毕（两 W 复刻重跑）+ xa3_judge 工具 bug 停批
- W 重跑 #1 stp_R32_W: GPU5 独立进程 07:49:20Z 起（快照 xa3_t2_sigma/ctrl/gpu5_snapshot_pre_w_rerun_R32_20261005T074920Z.txt, 5MiB）→ 08:23:5xZ 落盘, ≈34.6 min。argv 逐字=裁定 A 原文, 唯一差异 CUDA_VISIBLE_DEVICES 6→5, 其余 CWD/venv/OMP=16/TMPDIR 全同。
- W 重跑 #2 stp_R24_mu045_W: 同形 08:24:40Z 起（快照 gpu5_snapshot_pre_w_rerun_R24mu045_20261005T082440Z.txt, 5MiB）→ ~08:39:3xZ 落盘, ≈15 min。
- **位级回执（机对机, 跨设备 GPU6 档案 ↔ GPU5 重跑）**: stp_R32_W σ_rec=0.0069008720049158635 与档案 calib_R32 逐位同, r_eq0=32.1024798371758 逐位同; stp_R24_mu045_W σ_rec=0.006913281043059091 与档案 calib_R24_mu045 逐位同, r_eq0=24.13630227652888 逐位同。W 回执 4/4 齐（R16/R24 原窗 + R32/R24_mu045 重跑, 全逐位同）。零 ulp 差, 无需差不追条款。
- 10/10 静态 run 文件齐套。xa3_judge.py 按裁定"一行不动"执行 → **KeyError("R24_mu045") 于 §2 断言段第 78 行, 任何输出写出前崩溃**: P_R_EQ0 字典仅含 prereg §2 锁表的三族值 {R16,R24,R32}, 而循环遍历 P_SIGMA_RE 四键。证据: calib_R24 与 calib_R24_mu045 的 r_eq0 机器值同为 24.13630227652888（µ 不进 init, 族值同一）; prereg 引文 24.136302276492046 对两档案 rel≈1.53e-12（judge 自身 1e-9 容差内, 与 R24 同转写级）。judge 从未跑通过（前任 10-05 10:27 落盘后无成功执行记录）。
- 停批报控制器: 请示最小机械修复 P_R_EQ0 增加 "R24_mu045": 24.136302276492046（=R24 族引文值, 使第四键断言得以运行而非放松任何断言; 预计以 1.53e-12 rel 通过, 与 R24 同级）后重跑 judge; 或控制器另裁。修复未批前不动。

## 2026-10-05 控制器批准 xa3_judge 一行修复（裁定 1）执行记录
- 旧 judge md5=cf8f1bc70c1f97f1f4ccabe98696fb2f（KeyError 版）; 新 judge md5=12325d1b31db7488ea2c5bdf912a5b9b。恰加一行 P_R_EQ0["R24_mu045"]=24.136302276492046（=R24 族引文值, 控制器批准原文）; 判据/读法/容差零触碰。原"一行不动"令由本裁定取代。

## 2026-10-05 xa3_judge 第二处工具笔误被 R-A 回执拦截（裁定 1 修复版执行中）
- 一行修复版（judge md5 12325d1b31db7488ea2c5bdf912a5b9b）执行 → §0 断言全过 → **R-A 回执断言失败**（judge L158, worst=0.0016869200903416592 > 1e-12; prereg §4 "先于一切" 条款正确拦截）。判据数字未产出。
- 机对机诊断（/nfs/wangxi/tmp/xa3_ra_diag.py + xa3_ra_diag2.py, 全 11 臂）: (i) ca 重算对档案 Ca_ach 11/11 逐位同（σ_rec/gamma_fit/r_eq/µ 输入链零疑）; (ii) 档案 ratio_sh ≡ D_I/D_ref_sh（档案自带分母）11/11 dev=0.00e+00; (iii) 档案 D_ref_sh ≡ (35/32)·Ca_ach·(1+9.9743·(r_eq/(box·R_nom))³) 11/11 rel=0.00e+00——**t2 判决链 S-H 修正因子用实现半径 r_eq_end**, 正典源 t2_run.py:418 d_ref_sh = d_ref_unconf*sh_multiplier(r_eq_end/h_eff), h_eff=box·R_nom（t2_run.py:248-250 注释）。judge 的 d_ref_sh(ca, 1.0/box) 用名义 1/box = 第二处转录笔误, 产生族结构化 ratio 偏差 C16 +1.65e-3~1.69e-3 / C24 +7.7e-4~7.9e-4 / C32 +4.4e-4~4.5e-4 / Wlad +7.3e-4 / Re2 +8.1e-4（judge 重算偏高）。
- 影响: 无修复则 prereg §4 "读法 A 应逐位复现" 不可达（in-band 计数碰巧同 6/11 但值非逐位）; S1/S2/S3 的 B/C 读法同吃该因子。
- 停批请示第二处修复: judge L151 与 L167 两处同串 d_ref_sh(ca, 1.0 / box) → d_ref_sh(ca, a["r_eq"] / (box * R))（读法/门/带/规则零触碰; 修复后 R-A 回执预计 11/11 逐位过, 依 (iii) 恒等式）。未批不动。

## 2026-10-05 裁定 2 执行记录（第二处 judge 修复, 控制器独立复证后批准）
- judge md5 12325d1b31db7488ea2c5bdf912a5b9b（R-A 拦截版）→ 9c40f1404eea4617b61ae071eda2914b。恰两处 d_ref_sh(ca, 1.0/box) → d_ref_sh(ca, a["r_eq"]/(box*R))（L151 R-A 回执循环 + L167 reading() 闭包）; 判据/带/S 规则零触碰。
- 披露①: R-A 回执（prereg §4 先于一切）拦截第二处转录笔误——judge 用名义 1/box, 档案冻结公约=实现半径 r_eq/(box·R_nom); 控制器独立实现（xa_rehab/ctrl/ctrl_xa3.py）与我的 diag 三重位级证据一致（ca 11/11、分母恒等式 11/11、原判官式偏差 1.6869e-3 复证）。
- 披露②: 公式因子以 t2_run.py:418 调用点公约为准（sh_multiplier(r_eq_end/h_eff), h_eff=box·R_nom）。

## 2026-10-05 XA-3 判决落盘（judge md5 9c40f1404eea4617b61ae071eda2914b 终版）
- out/xa3_judgment.json md5=df1299899f0870b98d8be1ce1a116e9b (utc 2026-10-05T08:54:53Z)。回执: R-A worst dev=0.0（11/11 逐位）; W 回执 4/4 rel_diff=0.0; P 健康 5/5 pass; G3-A 重算对档案 3/3 逐位。
- 读法: A=6/11（出带 C24c 1.0403096389463422 / C32a 1.0424065708599148 / C32b 1.0488141704648399 / C32c 1.0606572996463073 / Re2 1.0694100595035290）; B=6/11（出带全低侧 C16a 0.9121082879529139 / C16b 0.9194749… / C16c 0.9325636… / C24a 0.9599297088009194（距带下沿 7.0e-5）/ Wlad 0.9284246…）; C=6/11（仅 Wlad 0.972360385482808 异于 A, 其余≡A）。
- S1=False（in_b=6 <9 且不>in_a=6）; S2=False（a/b/c 族 max|err| B=0.08789171204708612/0.08052499039861216/0.06743613834219853 vs A=0.04240657085991484/0.048814170464839934/0.06065729964630728, 全超 A+0.002）; S3=True（b_B 与 b_A 差 <4e-17 量级, 三族全过）。
- **总判 = KILLED**: σ-失配假设被杀; T2 G1a FAIL 维持原判; 共模归因转剩余候选（CAC 迁移率项/λ_eff≠1）。读法 B 的 G2 重判三族单调也全 False。
- 诊断列: σ_eff(P)/σ_rec−1 = −0.059265/−0.059255/−0.060088/−0.060513/−0.060092（R16/R16b8/R24/R32/R24µ045, 族平坦≈−6%）; σ_W/σ_rec−1 = +0.000000 四条件全 0（带壁复刻=档案逐位）; 隐含 ΔCa/Ca ≈ +6.3% 但方向使 ratio 系统下移、低 Ca 臂翻出带低侧——修正既不够也不对症。壁维归因: box8 带壁新量 σ=0.006852921527922828 对 box6 σ_rec 仅 −1.47%, 远小于周期−带壁 −5.9% → 差异主维=壁存在本身（周期 vs 带壁）, H2 归因如 prereg §5 属披露非判据。
- 消费链（X1 §2 dedup）: A-CAC 三点=R16/R24/R32 变体 P（md5 见 sources.xa3_P）, 零新烧卡即绪。停批报控制器 C8 亲验, X1 待放行。

## 2026-10-05 GPU5 窗口 #2 开（X1 跨模型电池; 控制器 C8 22/22 全绿后放行）
- 前置: 控制器 C8 亲验 XA-3 定案 KILLED; X1 prereg 8f63a945fc86e48bec429a3b25b6e306 亲核未动。开窗快照 x1_crossval/ctrl/gpu5_snapshot_pre_x1_20261005T090352Z.txt（gpu5 5 MiB）。新烧卡=仅 B-CG 2 臂 + B-CAC 2 臂（估 ~0.5 GPU·h, 止损 4, 任务帽 10）; A-MCMP/A-SC/B-MCMP/C-CAC 纯消费, A-CAC=XA-3 P 三点 dedup（sources.xa3_P md5 链）, A-CG=XA-5 β0.7 四档 dedup, C-CG=P0 探针已收缩为 CAC 行。x1_run.py 启动形制=setsid env CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=16 TMPDIR=/nfs/wangxi/tmp。

## 2026-10-05 X1 执行中事件（B-CAC replica 移植 bug 修复+直跑; 窗口 #2 内）
- B-CG 两臂一次过: b2m2_ny64 relL2=0.003176/jb=33/frozen 1.27e-13/59.8s; b2m2_ny128 relL2=0.002135/jb=65/frozen 7.07e-12/199.2s; 无需预授权延步; b_cg_poiseuille.json 落盘。
- B-CAC 两臂首跑 rc=1: b2_poiseuille_x1.py 的 HERE/../src_patched 相对路径在 x1 目录树下不存在 → import cac_lbm ModuleNotFoundError（移植 bug, 与 f-string 同类; 脚本自身冻结断言本就要求 /nfs/wangxi/runs/bm_mp_20261001/m1_cac_fix/src_patched 绝对树）。修复=该行改绝对路径（对齐断言）, 其余零触碰。
- 修复后按 x1_run.py 原 env/argv 逐字直跑两臂（CUDA_VISIBLE_DEVICES=5 OMP=16 TMPDIR B2_MU_LAW=harmonic B2_STEPS=50000, argv "a" {100,200}; 不重跑已成 B-CG, x1_run.py 不改）; 快照 ctrl/gpu5_snapshot_pre_bcac_20261005T091440Z.txt（gpu5 5 MiB）。m1 档案 ny100 wall=151.7822277545929s 供确定性对照预算。

## 2026-10-05 X1 全 GPU 臂跑完; judge 消费块撞健康断言停批
- 新烧卡 4/4 完成: B-CG ny64 relL2=0.003176/jb=33/frozen 1.27e-13/59.8s, ny128 relL2=0.002135/jb=65/frozen 7.07e-12/199.2s（无延步触发）; B-CAC ny100 Eu=0.07702678080491326/stat 2.6e-02/fd 4.2e-17/153s, ny200 Eu=0.35548…（liq 0.36718/gas 0.34572）/stat 9.8e-02/fd 9.9e-18/155s。**NY100 确定性回执对 m1 档案逐位**: Eu 与 u_center_num=9.380025417022128e-05 rel=0.0（fp32 同 GPU 类跨窗位同）。窗口 #2 GPU·h 合计 ≈0.17（59.8+199.2+153+155 s）, 帽 10 内。
- x1_judge.py 撞断言退出（L124, 消费 XA-5 CG 档块 assert d["health"]["pass"]）: 该旗标=XA-5 冻结门复合值, 0/40（控制器已裁定 erratum 终态）。机械层=judge 转录了源轨旗标而非按 X1 §2 口径重检。
- **实质层（机对机 /nfs/wangxi/tmp/x1_cg_health.py）**: 按 X1 自身 §2 健康口径（无 NaN ✓、逐色质量 ≤1e-10/步/色 ✓ 4/4、u_max 平台（末样-首样)/均值 ≤0.5%）, 消费 CG 4 臂平台= (128,25) 0.028191 / (200,20) 0.143593 / (200,30) 0.258344 / (200,40) 0.478845（tail20 口径 0.026772/0.191917/1.123918/0.745092）——**4/4 不过**, 伪速度在 30000 步协议终点仍在演化。X1 自跑新臂（同 run_calib.py 协议 30000 步）将同形失败, 非"重跑可救"。A-CG 行的入表语义（N/A vs 带披露入表）= 表语义决策, 停批报控制器裁定; judge 未改。

## 2026-10-05 X1 judge 二过跑后检查：两处新缺陷发现（机器证据，待控制器裁定）

- 裁定(b)补丁执行后二过：x1_judge.py md5 2021cd2944ec0b0392b6d849ffbd2e8f -> 424040e0f54862c048f6439f6d37f20b（已双录于上一事件块），RC=0，
  out/x1_crossval.json md5 fbfb2a94bb7ec99d64d1576c077a3b4a、out/x1_attribution_table.md md5 b31255ebc00cd0657341c99a6f5f0473。
- 跑后逐块核对 JSON 发现两处缺陷，均为机器证据，未自行改动，停批请批：

### 缺陷 X1-J1（致命，表行缺失）：A-CAC 块路径丢 f 前缀
- x1_judge.py L99 逐字：`d = json.load(open(XA / "xa3_t2_sigma/out/xa3_selftest_{tag}_P.json"))`
  —— `{tag}` 为字面量（与 b2 out_path 同类 f 前缀笔误），三次迭代 FileNotFoundError 全落 except，
  case_a.CAC = {"note": "XA-3 output missing ..."}，Laplace|CAC 表行整体缺失，XA-3 P 消费链（prereg 去重规则 A-CAC<-XA-3 P）未交付。
- 目标文件在盘 5/5、health.pass 5/5，md5 链不变（stp_R16_P cbdd11c52212a58ae28fb510529e2ed9 / stp_R24_P d0e7a15f95cd7e505e59c992b790ae8a /
  stp_R32_P 659eb1749acde27fc7e75bee38e84b11，其余两枚同前）。
- 全文件扫描确认这是唯一残留：grep 所有 open() 带花括号路径，L141/L193 f 前缀在位，L145 无插值不需 f。
- 提议修复（一行一字类）：仅 L99 加 f 前缀 `f"xa3_t2_sigma/out/xa3_selftest_{tag}_P.json"`，别处零触碰；门/比较器/§5 禁令零触碰。

### 缺陷 X1-J2（非致命，回执容差语义）：mcmp_analytic_recompute match=False
- judge 容差逐字（L216-218）：`abs(recomp.max - arch.max) < 1e-9 and abs(recomp.mean - arch.mean) < 1e-9`——对 14~34 量级百分比值用 1e-9 绝对容差。
- 机器证据（独立脚本，5090）：
  - H64：重算 n=55 max=27.040218588220462 mean=14.248968390643062 vs 档案 n=55 max=27.040222287178040 mean=14.248968660831451
    → abs_diff_max=3.699e-06 abs_diff_mean=2.702e-07 rel_diff_max=1.368e-07；n 同 55/55。
  - H128：重算 n=110 max=33.638003557645035 mean=15.692213530408500 vs 档案 n=110 max=33.638003468513489 mean=15.692214667797089
    → abs_diff_max=8.913e-08 abs_diff_mean=1.137e-06 rel_diff_max=2.650e-09；n 同 110/110。
  - 机制钉死：u_profile_sim 序列化逐值 fp32 精确（H64 65/65、H128 129/129 struct.pack 复核），u_profile_ana_measured_ratio 为 fp64；
    档案 errs 系 m2 脚本 fp32 算术路径（除法/×100 在 fp32 张量上）所得，judge 重算为 fp64——消费数据逐位同，仅误差度量的算术精度不同，偏差恰 fp32 ε（1.19e-7）量级。
  - FD 验证过门：max_abs_fd_residual H64 2.669e-18 / H128 1.2249790483191692e-17 vs scale_ref 4.36e-6（相对 6e-13/2.8e-12，≤1e-10 门）。
- 提议：零改动纯披露——recompute_match 回执如实留 false（1e-9 绝对容差下），表值即 fp64 重算值（与档案吻合到 7 位有效数字），
  NOTS/停批报告双披露上述数字；若控制器要求 JSON 内机器落 diff 值（如 max_abs_diff 字段）可另批一次机械编辑，默认不动。

- 状态：X1 全档停在"待 L99 修复批准 + J2 处置裁定"；批准后跑一次 judge，出全档停批报告。

## 2026-10-05 X1-J1 修复执行（控制器已批，恰 L99 一行）+ X1-J2 处置
- 控制器裁定双缺陷均批：J1 一行修复（其 ctrl_xa3.py C3 亲读 σ_eff 三点与我机器现提一致、F_σ 手算 0.9208% 吻合）；J2 零改动纯披露
  （recompute_match 如实留 false=冻结 1e-9 绝对容差的字面诚实读法；收口时其将亲钉 ①fp64 重算==表值 rel≤1e-9 ②|其−档案|≤5e-6 abs 且 rel≤2e-7 ③n 全同；不加 diff 字段、不出第三处编辑）。
- J1 执行：x1_judge.py L99 恰加 f 前缀，字节级断言全过（唯一出现 1 处、恰 +1 字符、行数不变、无其他行触碰）。
  md5：424040e0f54862c048f6439f6d37f20b -> 8afc67a20643efbf1e10b77d976945c0。
- 披露①：本 f 前缀笔误为 XA 轨第四处转录类缺陷（与 b2 out_path f 前缀同类），非门/语义改动。
- 披露②：A-CAC 块的 try/except 将 FileNotFoundError 静默吞掉、降级为 "XA-3 output missing" 缺档假象——二过表因此缺 Laplace|CAC 行而 RC=0 无任何报警，
  该静默吞异常模式是本次表行缺失未被即时发现的直接原因，如实记录。

## 2026-10-05 X1 judge 三过（J1 修复后）RC=0，Laplace|CAC 行恢复，电池全档落盘
- 运行：cd x1_crossval && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 TMPDIR=/nfs/wangxi/tmp python x1_judge.py → RC=0（零 GPU，纯后处理）。
- 输出：out/x1_crossval.json md5 fbfb2a94bb7ec99d64d1576c077a3b4a -> b1c434be2c8d394bbed52cdc19f43c0b；
  out/x1_attribution_table.md b31255ebc00cd0657341c99a6f5f0473 -> 7ffa3422d81bbe2c3ca790d8de13bf4c。
- A-CAC 恢复（XA-3 P 消费链交付）：F_sigma_pct=0.9207944922423941（与控制器 ctrl_xa3.py C3 亲读值/手算 0.9208% 吻合）、
  sigma_eff_by_R 三点、u_max_tail 三点、mode="consumed XA-3 variant-P outputs (dedup per prereg §2)"。
- J2 纯披露维持：receipts.mcmp_analytic_recompute={H64:false,H128:false}（冻结 1e-9 绝对容差字面读法），表值为 fp64 重算值，
  与档案吻合 rel 1.368e-07/2.650e-09（fp32 ε 量级，机制见上一事件块）。
- prereg 8f63a945fc86e48bec429a3b25b6e306 全程零触碰；XA 轨零 push/零 PR/零 git 改动维持。
- X1 累计 GPU ≈0.17 h（本次 judge 零 GPU）；全档停批报告已发控制器，待 ctrl_x1.py 独立重算收口。
