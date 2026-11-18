-- m15898 board-audit of m15617 GAP-LBM-INTL-BENCH
-- 交差审计：独立核验 m15617 交付物三件套 + 基准裁判 + 判级

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm15617 GAP-LBM-INTL-BENCH',
  15898,
  2,
  $json${
    "impl_files": [
      "tools/audit/m15617_intl_bench_audit.py (439 行，commit 906b6190)：扫描 TensorLBM 全仓 45 模块，按证据层级机器化分档 verified/partial/implemented/failed，生成 SQL DDL+INSERT、CSV/JSON 汇总表、matplotlib 雷达图/直方图",
      "docs/tensorlbm_maturity_m15617.sql (69 行)：tensorknowledge 库 tensorlbm_module_maturity(45 行) + tensorlbm_compete_score(10 行) 的 DDL+INSERT"
    ],
    "entry": "python3 tools/audit/m15617_intl_bench_audit.py --repo-root . → exit 0（本审计实测通过）；输出 JSON 含 citation_stats(51/51 通过)、分档汇总(verified 7/partial 21/implemented 15/failed 2)、figures 路径",
    "run_artifacts": [
      "⚠️ 部分缺失：原始 commit 906b6190 含 9 文件（6 工件+3 文本），但 merge commit cc59fc47 仅合并 3 文本文件（.md/.sql/.py）；二进制工件（compete_radar.png 160KB, level_distribution.png 28KB, module_maturity.json 2119 行, module_maturity.csv, compete_score.csv, upstream_cites.csv）未进入 main 分支",
      "✅ 文本工件在盘：docs/TENSORLBM_INTL_BENCH_AUDIT_M15617.md (14514B), docs/tensorlbm_maturity_m15617.sql (46451B), tools/audit/m15617_intl_bench_audit.py (31720B)",
      "✅ 重跑可再生：脚本 exit 0 且 figure_error=None，重跑可重新生成全部 6 个工件"
    ],
    "tests": "无 pytest 守护；脚本自身含内部断言（citation_stats 51/51、grade() 规则、figure_error=None），但无独立测试文件",
    "integration_callers": "none：脚本仅被本审计任务调用；无平台管线/CI/其他模块引用",
    "benchmark": {
      "registered_case": "ABSENT —— SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='GAP-LBM-INTL-BENCH' → 0 行；平台无此案号注册，基准裁判缺席",
      "ref": "N/A：本任务为审计/分析型 mission，非数值基准算例；无 ref/got 数值对",
      "got_vs_ref": "N/A：审计产出为模块分档表（45 模块 verified/partial/implemented/failed）与十维竞争度评分，非误差数值",
      "quality_findings": {
        "F1_artifacts_lost_in_merge": "原始 commit 906b6190 的 6 个二进制/数据工件未进入 main（merge 仅保留 3 文本文件）；重跑可再生但当前 main 树不完整",
        "F2_no_benchmark_registration": "tensor_benchmark_cases 无 GAP-LBM-INTL-BENCH 案号，平台无法机器裁定；审计结果落库在 tensorknowledge 库而非 tensorplatform 库",
        "F3_upstream_not_rerun": "四参照（OpenLB/waLBerla/Palabos/Lettuce）仅做源码存在性校验，未复跑其算例；上游 tier 上限 4 为声明性假设",
        "F4_case_reuse": "suboff_re1000 同时支撑 wall_function_les/rans_ke/ship_resistance_hull 三个 verified 模块，证据独立性弱",
        "F5_self_reported_perf": "8.6 GLUPS(单卡)/69 GLUPS(8卡) 为项目自记（源码 docstring），无第三方复现，审计正确排除其计入证据层级"
      }
    },
    "repro_cmd": "cd <TensorLBM> && python3 tools/audit/m15617_intl_bench_audit.py --repo-root . → exit 0；语法可执行，本审计实测通过；无需 GPU/特殊依赖（仅 matplotlib+glob+json）",
    "external_anchor": "四参照源码快照：OpenLB master@6c3934d(2019-12-02) /root/openlb；waLBerla HEAD /tmp/walberla；Palabos 2.3.0+dev /tmp/palabos_probe；Lettuce master@88a6559 /tmp/up/lettuce；51 条引用活体校验 51/51 通过",
    "level_rationale": "L2：真代码工具（输入=仓内源码+result.json → 计算=扫描/分档/绘图 → 输出=SQL/CSV/JSON/PNG），本审计实测 exit 0。不达 L3：无平台管线/CI/求解器集成调用者。不达 L4：(a) 非数值基准算例，无 ref/got 误差对；(b) tensor_benchmark_cases 无此案号；(c) 二进制工件在 merge 中丢失。报告六段齐全（①—⑥），但⑥误差判定针对的是各模块内部算例误差，非本审计自身误差。",
    "verdict_summary": "三件套核验：run_artifacts ⚠️ 部分缺失（6/9 工件未入 main，重跑可再生）；benchmark ref vs got N/A（审计型任务非基准算例）；repro_cmd ✅ 语法对且实测通过。基准裁判：tensor_benchmark_cases 无 GAP-LBM-INTL-BENCH 案号 ❌。报告六段 ✅ 齐全。判级 L2（工具实现级）。"
  }$json$::jsonb,
  'board-audit'
)
ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET
  level = EXCLUDED.level,
  evidence = EXCLUDED.evidence,
  mission_id = EXCLUDED.mission_id,
  gate_state = 'pending',
  gate_level = NULL,
  gate_note = NULL
RETURNING id, level, gate_state;
