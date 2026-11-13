-- m18349 交差审计落表：m18225 XVAL-LBM-PF-EOS-MAXWELL-HIGH-DENSITY-RATIO (TensorLBM)
-- 判级依据见 docs/board_audit/m18349/VERDICT_m18225.md；基准判据唯一天平 = tensor_benchmark_cases（本锚 0 行）
-- 证据全文 = docs/board_audit/m18349/evidence_m18225.json；机核复跑 = docs/board_audit/m18349/verify_m18225.out
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm18225 XVAL-LBM-PF-EOS-MAXWELL-HIGH-DENSITY-RATIO',
  18349,
  0,
  $ev${
  "impl_files": [
    "NONE — agent/m18225 相对 main 提交数 0（git rev-list --count main..HEAD = 0；HEAD=3e2bf36d 属 m17881 合并），本单未写一行仓库源码",
    "工作树唯一未跟踪物 = .tmp_kb1..5.py（12:37–12:42 的 DB/API 查询草稿，verify_m18225.out:worktree_dirty_files），非交付实现",
    "对照：锚点既有实现 benchmarks/eos_maxwell/{probe_lib.py,run.py} + tests/test_eos_maxwell_stability.py 由 m8675 引入（git log --diff-filter=A ⇒ b245924c Merge #1598），PRE-EXISTING，不计入本单"
  ],
  "entry": [
    "NONE for m18225 — 本单无入口；锚点既有入口 benchmarks/eos_maxwell/run.py:1（argparse CLI --mode/--out/--eos/--tau-list…）属 m8675"
  ],
  "run_artifacts": [
    "NONE — benchmarks/eos_maxwell/out/results.json 在 m18225 树内不存在（verify_m18225.out:run_artifacts_out_results_json_exists=false, bytes=0），且 git ls-files benchmarks/eos_maxwell/out 为空（从未入库）",
    "仓库内只有派生表 benchmarks/eos_maxwell/{RESULTS.md,CRITERION.md}（机生成自 m8675 的 5922.9 s campaign，RESULTS.md:4-8 自述 driver sha256 ae393b704946b0ce…）",
    "m18225 mission.result.evidence.head_commit=null、branch=null；result.tool_calls=80 全为只读调研"
  ],
  "tests": [],
  "integration_callers": [
    "NONE attributable to m18225 — 锚点既有调用方 xval/*, app/backend 链路由 m8675 及其后续单建立，本单未接线"
  ],
  "benchmark": "tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-PF-EOS-MAXWELL-HIGH-DENSITY-RATIO' ⇒ 0 rows；ILIKE '%MAXWELL%' OR '%EOS%' ⇒ 0 rows（verify_m18225.out）⇒ 本锚点无注册基准裁判，L4 机裁不可达；既有 RESULTS.md:165 'tiers passed 0/6' 与 §2 ref(Maxwell 10.01/99.97/995.2) vs got(10.45/293.3/collapse) 属 m8675 数据，非本单",
  "repro_cmd": "m18225 未留 repro_cmd；既有 driver CLI 可执行性审计侧实测 `cd /root/tp-agent/work/m18225 && python3 benchmarks/eos_maxwell/run.py --help` rc=0（--mode 存在），但真复现需重跑 5922.9 s campaign 并写入未入库的 out/，本单未执行任何一步",
  "external_anchor": "仅 mission.result 文本自称 lssfau/lbmpy 的 pf-eos-maxwell + Maxwell 等面积构造；oss_solutions 中 feature_slug ILIKE '%EOS-MAXWELL%' ⇒ 0 rows，仓库内无 vendored 源码/pin hash/文献条目 ⇒ 外部锚点不可溯源（调研结论未沉淀，任务③未完成）",
  "level_delta": "m18225 自报 honest-FAIL（fail_stage=budget_exhausted+forced_finish，未声明级别）→ 判 L0（无仓库文档亦无代码；仅 mission.result 文字调研）",
  "auditor_checks": [
    "本单提交数：0（git rev-list）",
    "交付物：仅 5 个未跟踪草稿脚本；out/results.json 缺失（provenance 断链）",
    "基准裁判：tensor_benchmark_cases 该 anchor 0 行（机核查询）",
    "报告：tensor_reports WHERE mission_id=18225 ⇒ 0 rows（report_owed=false 但因 succeeded 被自动放行，属报告纪律漏网）",
    "原单自述理由复核：cluster_jobs id=1834 state=queued、cmd='ls ~/oss-cache/'、mid 空 ⇒ '队列积压 308 作业未执行'属实；m18132 status=failed/no_progress/38 tool_calls/0 changed_files 亦属实",
    "审计侧独立复跑树内 provenance 测试（非本单交付）：tests/test_eos_maxwell_stability.py -k 'results_artefact_matches or criterion_artefact_matches or maxwell_construction_hits_target or consumer_anchors_exist' ⇒ 3 passed, 2 skipped（skip 原因：campaign artefact / criterion artefact not present in this tree，即 out/results.json 未入库）"
  ],
  "predeclared_criterion_status": "总账判据 <15%：既有 m8675 数据里仅 tier_ratio10_edm 的密度比误差 0.0437（4.37%）达标，但同一 run 伪流 Ma=0.115 超预注册 0.1 门 ⇒ 该 tier 按预注册复合门 honest-FAIL；ratio 100/1000 全档塌缩或误差 1.93/—（RESULTS.md:26-31,165）",
  "gate_expectation": "judge(declared L0, evidence.tests=[]) ⇒ gate_level 0 ⇒ 原单 m18225 succeeded ⇒ reopened（返修），非消账",
  "scope_limits": "本单只核 m18225 交差；锚点既有能力（m8675）的 L 级不在本单判级范围；未重跑 5922.9 s campaign（交差核验非重做）"
}$ev$::jsonb,
  'board-audit'
)
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET level = EXCLUDED.level, evidence = EXCLUDED.evidence, mission_id = EXCLUDED.mission_id;
