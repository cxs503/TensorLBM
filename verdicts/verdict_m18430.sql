-- m18430 交差核验落表：m18339 XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME (TensorLBM)
-- 判级依据见 docs/AUDIT_m18430_m18339.md；基准裁判 = tensor_benchmark_cases（本锚 1 行，m18306 登记，轮 1 网格）
-- 结论：level=3（L2 实证达标；L3 包内生产模块集成、无既有求解器采纳；L4 实质内容在但交付不达标：repro_cmd 崩在作图阶段
--       => results.json/fig2-3 不生成；交付表仅 5/14 臂；注册网格 bfl_par/halfway/bgk/trt_c 未复跑；全部成果未提交）

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm18339 XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME',
  18430,
  3,
  jsonb_build_object(
    'impl_files', jsonb_build_array(
      '/root/tp-agent/work/m18339/src/tensorlbm/boundary/interpolated_bounce_back.py:1-406 (alpha 族闭包 alpha_bfl/alpha_cli/alpha_yli + OSS 壁面值项 -alpha*3w_q*rho_w*(c_q.u_w) + write_to in {fluid,solid} + post_collision_correction/K_hat；未提交，工作树 MM)',
      '/root/tp-agent/work/m18339/src/tensorlbm/boundary/ibb_channel.py:1-337 (生产侧调用方: channel_geometry SDF+固壁掩码, validate_geometry 几何闸 Cb=8/14, TRT collide->stream->壁面 时间循环, 返回剖面/解析解/L2/拟合壁位；273 行调用 interpolated_bounce_back_2d；未提交 ?? )',
      '/root/tp-agent/work/m18339/xval/XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME/run.py:1-425 (xval driver: 14 臂扫描 + judge_row + oss_closure_check + 几何闸；未提交 ??)'
    ),
    'entry', jsonb_build_array(
      'xval/XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME/run.py::main (--quick 2 臂 / 默认 14 臂)；审计实跑 --quick 成功: N=8 d=0.375 cli L2=7.388e-07%, bfl 6.953e-01%',
      'src/tensorlbm/boundary/ibb_channel.py::run_interpolated_wall_channel(N, delta, nu=0.1, force=1e-3, closure, carrier, wall_momentum, write_to) -> IBBChannelResult(profile/L2/wall_bottom/wall_top/wall_err/steps)'
    ),
    'run_artifacts', jsonb_build_array(
      'artifacts/m18339_audit/table.csv (审计侧独立复跑 14 臂, 全 PASS; 由 /tmp/audit_full 拷入)',
      'artifacts/m18339_audit/run.log (2284B, 与交付 run.log 逐字节同尺寸; 14 臂 + geometry gate + OSS closure check)',
      'artifacts/m18339_audit/partial_results.json (14 臂含逐点 profile/解析剖面)',
      'artifacts/m18339_audit/oss_closure_check.json (cli/bfl 各 abs_diff=0.0 vs 独立 NumPy 转写)',
      'artifacts/m18339_audit/fig_A1_l2_vs_delta.png, fig_A2_wall_err.png, fig_A3_profile.png (审计侧由 table.csv/partial_results.json 生成)',
      '交付侧实际只有 5 文件: xval/.../artifacts/{run.log,table.csv(仅 5/14 臂),partial_results.json(5 条),oss_closure_check.json,ablation.json} + figures/fig1_profile_vs_exact.png；results.json/evidence.json 不存在'
    ),
    'tests', jsonb_build_array(
      'tests/test_interpolated_bounce_back_alpha.py (10 test) + tests/test_ibb_channel_exactness.py (8 test) => board-audit 独立复跑 20 passed in 75.58s (2026-11-13)',
      'tests/test_ibb_channel_exactness.py::test_poiseuille_exactness_via_production_caller (参数化 cli/bfl x delta) 锁死生产调用方精确性',
      'tests/test_interpolated_bounce_back_alpha.py::test_default_matches_historical_two_branch_operator (向后兼容逐位锁)'
    ),
    'integration_callers', jsonb_build_array(
      'src/tensorlbm/boundary/ibb_channel.py:273 调用 interpolated_bounce_back_2d（包内生产模块，非一次性脚本）',
      'NEGATIVE: 仓内无既有求解器路径采纳 —— grep interpolated_bounce_back 只命中 ibb_channel/inamuro_wetnode(文档串)/tests/xval，solver2d|boundaries 无调用 => L3 只到"包内自集成"'
    ),
    'benchmark', jsonb_build_object(
      'ref', '解析 Poiseuille u=F/(2nu)(y-y0)(y1-y), y0=0.5-delta, y1=N-0.5+delta（与 pylbm tests/test_linear_bounce_back.py 同式）+ OSS 闭包公式转写',
      'got', 'cli N=8 delta=0.125..0.9: 7.3801e-07/7.3841e-07/7.3881e-07/7.3920e-07/7.3956e-07/7.4028e-07 %; N=16 d=0.375: 7.4355e-07 %; bfl d=0.25/0.375/0.5/0.625: 1.5102/0.69526/7.3920e-07/0.59399 %; N=16 d=0.25 bfl: 0.34749 %; yli d=0.125/0.25: 3.2877/3.0203 % (审计复跑 14/14 臂 PASS, 与交付 run.log 打印值逐臂吻合 14/14)',
      'tolerance_check', '<15%/臂：14/14 过（最优 7.38e-09 相对=舍入级, 最差 3.29%）; 壁位误差 cli<=1.03e-10 lattice, bfl/yli<=4.11e-2 (超预注册 1e-2 容忍, 理论预期 alpha_hat!=1)',
      'registered_row', 'tensor_benchmark_cases id 现 1 行 (m18306 登记, 轮 1 网格 N=8/16 x d=0.25/0.5/0.9 x carrier bgk/trt_c/trt_d2 x closure halfway/bfl/bfl_par, status=failed, 插值臂 18.02-43.41%) —— 轮 2 已把同族 bfl 臂修到 0.59-1.51% 但未复跑 bfl_par/halfway/bgk/trt_c/delta=0.9 插值臂 => 该行仍 failed；本审计补登轮 2 行（见 verdict_m18430_round2_benchmark.sql）',
      'verdict', 'L4 部分不达：数值对照真且全过，但网格覆盖不全 + 交付 repro 不闭合'
    ),
    'repro_cmd', jsonb_build_array(
      'cd /root/tp-agent/work/m18339 && PYTHONPATH=src python xval/XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME/run.py --quick  => 可执行, 数值复现',
      'cd /root/tp-agent/work/m18339 && PYTHONPATH=src python xval/XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME/run.py  => 14 臂数值全部复现, 但退出非零: run.py:360 向 make_figures 传 r["result"], 缺 l2_rel_pct (在 r["metrics"]) => KeyError: l2_rel_pct, results.json/SUMMARY/fig2,fig3 永不生成',
      'PATCH: scripts/audit_m18430/patch_run_figures.py (1 行: r["result"] | r["metrics"] | {closure}); 副本实跑验证 /tmp/fixchk/out/results.json + 3 图齐（见 audit 报告 §5 E1）'
    ),
    'external_anchor', jsonb_build_array(
      'pylbm/boundary.py::LinearBounceBack 族 (BFLBounceBack/CLIBounceBack/YLIBounceBack/ParBFLBounceBack) docstring 公式 alpha_hat=-1+alpha(1/2+delta), beta=1-alpha*delta, beta_hat=1-alpha/2, K_hat=1-alpha_hat —— 报告 §1 给行级出处',
      'pylbm/tests/test_linear_bounce_back.py (基准几何 y0=0.5-delta, y1=N-0.5+delta, 源项 3w_i c_ix F 碰后加, 精确性条件 Lambda+Lambda-=delta^2/2 且壁面动量 Lambda-F)',
      'lbmpy/boundaries/boundaryconditions.py::NoSlipLinearBouzidi (第二份独立实现, 三分支闭包)',
      'HONEST: OSS 包未执行 (mpi4py 缺失, 序列桩下 pylbm 自测失败) => 对标为公式级; oss_closure_check.json abs_diff=0.0 是同一公式两实现互证, 不等于 OSS 实跑数值'
    ),
    'audit_notes', '审计独立复跑: 14 臂全 PASS 且与交付 run.log 逐臂吻合 (14/14), 20/20 测试通过 => 数字非编造; 报告六段内容齐 (REPORT.md 10.2KB) 但 tensor_reports 无 mission_id=18339 行 (未发布); 虚标项: 报告 §2c/§4 称数字取自 artifacts/results.json + 工件清单列 evidence.json/fig1..3.png, 实际 results.json/evidence.json 不存在、图只有 1 张; 交付 table.csv/partial_results.json 只 5/14 臂 (mtime 早 run.log 3 分钟, 自相矛盾); 全部交付未 commit (agent/m18339 与 main 树一致, git diff 空)',
    'blocking_defects', jsonb_build_array(
      'E1 blocking: 全量 repro_cmd 崩在作图阶段 (KeyError), results.json/fig2/fig3 永不生成 —— 解释了交付缺件; 1 行补丁已给出并在副本验证',
      'E2 虚标: 报告溯源指针 results.json/evidence.json/fig2-3 三项指向不存在文件',
      'E3 证据不完整: 交付结构化表仅 5/14 臂, 9 臂只存于 run.log stdout',
      'E4 未入库: 全部产出未提交 (git diff main agent/m18339 为空)',
      'E5 基准覆盖: 注册网格 bfl_par/halfway/bgk/trt_c/delta=0.9 插值臂未复跑, carrier 命名 trt_d2 vs trt_scheme 未显式对齐',
      'E6 报告未发布 tensor_reports (本审计代发 mission_id=18430 审计报告)'
    ),
    'level_rationale', 'L2 三件套真且审计复现 (真代码+入口+工件, 14/14 臂逐位复现) => 过 L2; L3 仅"包内自集成"(ibb_channel 生产模块调用算符, 无既有求解器路径采纳, 记 partial); L4 不达 (repro_cmd 非零退出 + 注册网格覆盖不全 + 主工件 results.json 缺); 补丁+commit+补跑后可达 L4'
  ),
  'board-audit'
);

-- 轮 2 predeclared 网格补登为基准行（裁判天平缺行；此行动机见 audit 报告 §3）
INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, ref_source, metric, tolerance, repo, status, passed_at)
VALUES (
  'XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME',
  'FORCED-PLANE-POISEUILLE-ALPHA-FAMILY-ROUND2-SWEEP (x-periodic, y open; N=8 x delta=0.125/0.25/0.375/0.5/0.625/0.9 x closure cli; N=8 x delta=0.25/0.375/0.5/0.625 x closure bfl; N=8 x delta=0.125/0.25 x closure yli; N=16 x delta=0.375 cli + delta=0.25 bfl; carrier trt_scheme = Lambda+Lambda-=delta^2/2, wall momentum u_w=Lambda- F, write_to=fluid)',
  'analytic u=F/(2nu)(y-y0)(y1-y), y0=0.5-delta, y1=N-0.5+delta (same form as pylbm tests/test_linear_bounce_back.py); numbers from board-audit m18430 independent re-run of xval run.py (artifacts/m18339_audit/table.csv), 14/14 arms reproduce the delivered run.log printout',
  'l2_rel profile error vs exact: cli 7.3801e-07..7.4028e-07 % (6 deltas, N=8) and 7.4355e-07 % (N=16 d=0.375); bfl 1.5102/0.69526/7.3920e-07/0.59399 % (d=0.25/0.375/0.5/0.625, N=8) and 0.34749 % (N=16 d=0.25); yli 3.2877/3.0203 % (d=0.125/0.25); wall-root error <=1.03e-10 lattice (cli) / <=4.11e-2 (bfl,yli); round-1 same anchor: 18.02-43.41% on interpolated arms',
  '<15% per arm (owner standard) => met by all 14 arms registered here; NOT covered by round 2: closure bfl_par (K-hat not wired into production loop), halfway control, carriers bgk/trt_c, delta=0.9 interpolated arms => the m18306 round-1 row (status=failed) is not superseded',
  'TensorLBM',
  'passed',
  now()
);