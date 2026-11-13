-- m17884 交差核验 verdict: 对象 m17839 XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN (TensorLBM, commit eee0988a)
-- 判级 L2：真代码+可执行入口+工件可位级复现(2532 数值叶 0 差异，仅 4 个 wall_s 不同)+12 测试全绿；
--        但 (1) tensor_benchmark_cases 对该 anchor 0 行=无机器裁判→L4 不可确认；
--             (2) 新模块零调用方/零导出→未接入主求解器→L3 不成立；
--             (3) 上游 CUDA 从未编译执行，"OSS arm" 是同作者 NumPy 转写。
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm17839 XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN', 17884, 2, $json${
 "verdict_basis": "L2 已确证、L3/L4/L5 不成立。L2 依据：src/tensorlbm/fused_push_pull.py:122-258 是真实实现（pull 索引→矩→BGK 弛豫→写入另一 8 方向缓冲，上游 17 doubles/node 布局，single_buffer=9 且位级相同），入口 xval/fused_push_pull_aa_pattern/run.py 可执行，审计员在审计窗口内重跑产出与交付 result.json 的 2532 个数值叶逐项比对=0 处非计时差异（仅 4 个 wall_s 计时字段不同），tests/test_fused_push_pull_xval.py 12 passed in 6.21s。数字全部经独立重算成立：ux/uy rel.L2=0.00191946（<3% 目标 <15% gate）、nu 反演 0.16679741 vs 0.16666667=7.84e-04、质量 4095.999999999817 vs 4096=4.5e-14、AB vs fused ux rel.L2=8.35e-14、收敛阶重算=2.0814771219688715/2.3856190152805503 与记录完全一致；负控制确实能打假（nu×1.25→0.4756 FAIL、W/E 互换→0.7847 FAIL），flip_all 0.00192 的全局符号盲区已在 §6.2 自曝。L3 不成立：新模块在 src/、benchmarks/、examples/ 内零引用，src/tensorlbm/__init__.py 导出命中 0，无边界/步进/API 接线（文档亦承认 periodic-only，墙体留在 solver）。L4 不成立：基准裁判表唯一裁判缺位（见 benchmark 键），且物理锚=解析解制造解验证，上游实现从未执行。",
 "claimed_level_implicit": "XVAL 交叉对标通过（报告 16786：0.19% 速度误差、arm-to-arm 位级相同、controls 可失败）",
 "delivered_level": 2,
 "gap_to_claim": "被断言的科学数字全部为真（非虚标）：审计员重跑位级复现，ux 0.19%、nu 7.8e-4、跨臂一致、负控制 FAIL 均核实。落级差异不在算术而在三件事：①该 anchor 在 tensor_benchmark_cases 里 0 行=没有机器可裁的基准行（平台既有 65 行无一匹配，按任务榜铁律基准判据是唯一裁判）；②交付的新能力没有任何调用方/导出，未接入主求解器；③'OSS arm' 是同作者编写的 NumPy 转写，上游 CUDA 从未编译执行，位级相同只证明转写的可复现性，不证明与上游二进制一致。故授 L2，返修项为注册裁判行、接入 solver、执行上游实现。",
 "impl_files": [
  "src/tensorlbm/fused_push_pull.py:122-258 (class FusedPushPullStepper：_pull_index:97-112 / step:193-242 / single_buffer 分支:171 / persistent_doubles_per_node:183-186)",
  "src/tensorlbm/fused_push_pull.py:260-296 (taylor_green_fields 解析解，逐项对齐 ref/LBM.cu:65-84 含 O(u^2) 压强项)",
  "src/tensorlbm/fused_push_pull.py:325-356 (tg_l2_errors，沿用上游 LBM.cu:231-262 的 sqrt(sum e^2 / sum a^2) 与 rhoa-rho0 涨落归一)",
  "xval/fused_push_pull_aa_pattern/run.py:311-470 (main：4 臂 + 3 负控制 + 分辨率 + tau/u_max 敏感性 + rho 模态分析，写出 result.json)",
  "xval/fused_push_pull_aa_pattern/oss_port.py:41-58,95-169 (上游 chapter13/gpu/LBM.cu 的 NumPy 转写；审计员逐行对照 LBM.cu:181-252 的 pull 索引/矩/tauinv=2/(6nu+1)/写入另一缓冲，形式一致)",
  "xval/fused_push_pull_aa_pattern/make_report.py:1-408 (由 result.json 渲染 docs/XVAL_m17839_FUSED_PUSH_PULL_AA_PATTERN.md 与 4 张图)"
 ],
 "entry": "python xval/fused_push_pull_aa_pattern/run.py --out result.json（审计员实跑：<2 分钟完成，产出 /tmp/audit_m17839_result.json 45,603 B）",
 "run_artifacts": [
  "xval/fused_push_pull_aa_pattern/result.json (45,601 B：base/cross_arm/cross_arm_mac/storage/controls/resolution/sensitivity/rho_mode/verdict；审计重跑仅 4 个 wall_s 字段不同、0 处数值差异)",
  "xval/fused_push_pull_aa_pattern/run.log (36 行：verdict 块原样落盘)",
  "xval/fused_push_pull_aa_pattern/figures/fig1_decay_and_errors.png|fig2_resolution.png|fig3_rho_acoustic.png|fig4_cross_arm.png (非空白占比 5.11%-29.30%，审计员实测)",
  "xval/fused_push_pull_aa_pattern/ref/{LBM.cu,LBM.h,main.cu,PROVENANCE.json} (上游快照；审计员复算 sha256：0c9854fa…/0bc95be6…/ebbb847c… 与 PROVENANCE.json 全部一致)",
  "docs/board_audit/m17884/audit_numbers.json (本次审计的重算读数：grade/ladder/controls/benchmark_judge/integration/repro 差异清单)",
  "docs/board_audit/m17884/figures/{fig1_grades_vs_analytic,fig2_resolution_ladder,fig3_controls_and_repro}.png",
  "docs/board_audit/m17884/AUDIT_m17884.md + audit_m17884.py (审计报告与可重跑脚本)"
 ],
 "tests": "tests/test_fused_push_pull_xval.py（256 行，12 个测试）：tau↔nu 映射、D2Q9 方向表对齐上游编号、pull 索引 == torch.roll、equilibrium vs 上游公式、跨臂位级相等、single_buffer 位级相等、质量/动量守恒 1e-12、解析 gate <15% 且 ux <3%、nu 反演 <15%、负控制（W/E 互换）必须 >gate、in-place 逐点更新不安全。审计员实跑 12 passed in 6.21s。断言为不变量型（阈值/精确相等），非仅执行代码。",
 "integration_callers": "无（L3 不成立）。grep -rn 'fused_push_pull' 于 src/ benchmarks/ examples/ 仅命中模块自身文档字符串；src/tensorlbm/__init__.py 导出命中 0；xval 与 tests 之外的唯一使用者不存在。跨臂 AB 参考用仓内 solver.stream+collide_bgk（一致到 8.35e-14），只证明库内数值一致，不构成接入。",
 "benchmark": {
  "anchor_slug": "XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN",
  "registered_rows_for_anchor": 0,
  "platform_total_cases": 65,
  "query": "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN' -> 0 rows",
  "sole_judge_status": "ABSENT",
  "auditor_rederived_vs_predeclared_criterion": {
   "ux_rel_l2": 0.0019194569784560188,
   "uy_rel_l2": 0.001919456978453505,
   "rho_rel_l2": 0.028852906674679987,
   "nu_rel_err": 0.0007844827157505452,
   "gate_15pct": "PASS (ux/uy/rho/nu)",
   "target_3pct": "PASS",
   "mass_rel_err": 4.46e-14,
   "ab_vs_fused_ux_rel_l2": 8.35e-14,
   "resolution_order_rederived": [2.0814771219688715, 2.3856190152805503],
   "orders_match_recorded": true,
   "negative_controls": {"nu_bias_25pct": 0.47561882987367887, "mirror_x": 0.7846823607182364, "flip_all_blind_spot": 0.00191945697846134}
  },
  "auditor_magnitude_check": {"abs_rho_err_max": 9.066656714296073e-08, "abs_rho_err_rms": 4.034222862526362e-08, "analytic_rho_fluct_rms": 1.74532012588043e-06, "rel_l2_rho_at_t801": 0.023114515226777047, "abs_ux_err_max": 2.933515667992567e-06, "note": "rho_lu2 按上游约定以 O(u^2) 涨落归一，2.89% 对应绝对误差 ~9e-08，标题幅值易被误读为 3% 物理误差"},
  "resolution_gap_for_L4": "NX=128 无跨臂对照；无 tensor_benchmark_cases 行；物理锚为解析制造解，非实验/文献数据"
 },
 "repro_cmd": "cd <checkout of agent/m17839 @ eee0988a> && python xval/fused_push_pull_aa_pattern/run.py --out /tmp/audit_m17839_result.json   # 审计员实跑 <2 min，全部数值与交付 result.json 一致；python -m pytest tests/test_fused_push_pull_xval.py -q   # 12 passed in 6.21s；python xval/fused_push_pull_aa_pattern/make_report.py   # 重渲染文档与交付文档字节相同",
 "external_anchor": "lbm-principles-practice/code chapter13/gpu/{LBM.cu,LBM.h,main.cu}（MIT，Kruger et al. 2016）已快照入 ref/ 且 sha256 经审计员复算一致（0c9854fa353b62415bda2ff29fef890ece9254617990d7f94a2313537d0feb0c / 0bc95be600f9416e58b70455ef756dea2875a14077e4dcddd4addb24b2ec0920 / ebbb847c12b4d52495bef359c44bf0a1d802a6b1c2047377dc8eed9e3e5af98e）；但上游 CUDA 从未编译执行（无 GPU/nvcc），跨臂比较对象是同作者 NumPy 转写 oss_port.py，故 'arm-to-arm bit-identical' 只证明转写可复现，不证明与上游二进制一致。数值锚为闭式 Taylor-Green（制造解验证），非实验/文献基准。",
 "audit_findings": {
  "A1_blocking_L4": "tensor_benchmark_cases 该 anchor 0 行（平台 65 行）→ 无机器裁判，L4 不可确认",
  "A2_blocking_L3": "新模块零调用方零导出（__init__ 命中 0），periodic-only，未接入 solver/boundary/API",
  "A3_word_claim": "OSS arm = 作者自写 NumPy 转写，上游 CUDA 未执行；标题 'arm-to-arm bit-identical' 易被读成与上游一致（文档 §4.2/§6.1 标签正确）",
  "A4_metric_amplification": "rho rel.L2 按 O(u^2) 涨落归一，绝对误差 9.07e-08；文档未给绝对量级",
  "A5_controls_scope": "负控制注入在 OSSFusedStepper 变体（run.py:176 _variant_step），未对交付的 FusedPushPullStepper 注入故障",
  "A6_blind_spot_confirmed": "flip_all 全局符号翻转对解析 gate 与总体量比较均不可见（文档自曝，审计确认）",
  "A7_timing_outlier": "交付 torch_ab.wall_s=117.1 s vs 审计重跑 1.43 s（81×，原跑在重负载下）；无 Mlups/带宽主张，不构成缺陷"
 },
 "level_reasoning": {
  "L1": "PASS - 12 个单元/不变量测试全绿（方向表、tau-nu、pull 索引、守恒律、gate 阈值）",
  "L2": "PASS - 真代码 + 可执行入口 + 工件位级复现（2532 数值叶 0 差异）+ 测试",
  "L3": "FAIL - 零调用方/零导出/无边界接线，periodic-only；AB 交叉只证明库内数值一致",
  "L4": "FAIL - 唯一裁判（tensor_benchmark_cases 行）缺位；上游实现未执行；锚为制造解",
  "L5": "FAIL - 无可用外部裁决基准，继承 L4"
 }
}$json$::jsonb, 'board-audit');