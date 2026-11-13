-- m17668 交差审计落表：m17640 XVAL-LBM-CASE-SCHEMA-JSON-UNITS-CONVERTER (TensorLBM)
-- 判级依据见 docs/AUDIT_m17640_VERDICT.md；基准判据唯一天平 = tensor_benchmark_cases（本锚 0 行）
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm17640 XVAL-LBM-CASE-SCHEMA-JSON-UNITS-CONVERTER',
  17668,
  1,
  jsonb_build_object(
    'impl_files', jsonb_build_array(
      'src/tensorlbm/unit_converter.py:1 (LBMUnitConverter，PRE-EXISTING，非本单产出)',
      'src/tensorlbm/cases/base.py:41,51,55 (CaseUnits 复用 LBMUnitConverter，PRE-EXISTING)'
    ),
    'entry', jsonb_build_array(
      'src/tensorlbm/unit_converter.py:LBMUnitConverter(...) -> .tau/.nu_lb/.dx/.dt/.ma/.summary()'
    ),
    'run_artifacts', jsonb_build_array(
      'NONE：mission.result.evidence.head_commit=null, branch=null；git status 显示 2 文件 untracked(??)，无 runs/ 或 artifacts/ 产物；无 benchmark 输出 JSON/CSV'
    ),
    'tests', jsonb_build_array(
      'tests/test_xval_case_schema_json_units_converter.py (untracked，本单唯一新增代码产物)',
      'repro: python3 -m pytest tests/test_xval_case_schema_json_units_converter.py -q => 83 passed, 2 warnings in 4.73s (m17668 独立复跑, 2026-11-13)',
      '警告实体：Ma=0.1039 >= 0.1 压缩性告警（test_case_units_vs_converter_cavity）'
    ),
    'integration_callers', jsonb_build_array(
      'src/tensorlbm/general_sim.py:58,327,518 (求解器主链路持有 self.uc = LBMUnitConverter(...)，PRE-EXISTING)',
      'src/tensorlbm/cases/base.py:41 (CaseUnits 再导出)',
      'app/backend/routers/preprocess.py, benchmarks/verified/suboff_sail_resistance/run.py (PRE-EXISTING)'
    ),
    'benchmark', jsonb_build_array(
      'tensor_benchmark_cases WHERE anchor_slug=''XVAL-LBM-CASE-SCHEMA-JSON-UNITS-CONVERTER'' => 0 rows (无基准裁判)',
      'tensor_benchmark_cases WHERE anchor_slug ILIKE ''%CASE-SCHEMA%'' OR ''%UNITS-CONVERT%'' => 0 rows',
      '故 L4 不可达：无注册基准、无 ref vs got 机裁记录；报告 0% 误差为同一代数式的恒等式，非独立基准'
    ),
    'repro_cmd', jsonb_build_array(
      'cd /root/tp-agent/work/m17640 && python3 -m pytest tests/test_xval_case_schema_json_units_converter.py -q  (可执行，83 passed)',
      'NOTE: 该文件未 commit，仓库 main 上无此测试 => 对 GitHub 检出不可复现'
    ),
    'external_anchor', jsonb_build_array(
      'DECLARED: lettucecfd/lettuce UnitConversion (MIT)',
      'ACTUAL: 非真外部依赖——test 文件内自定义 class LettuceUnitConversion（test_xval_...py:44-55，注释自称 "reconstructed from lettuce source + survey"），环境 lettuce 未安装（import lettuce => ModuleNotFoundError），无 pin 版本/hash/vendored 源码片段',
      '对比量 tau=nu/cs^2+0.5 / dx=L/N / dt=dx*u_lb/u_phys 两侧为同一解析式，误差 0% 由构造成立 => 自证，不构成独立跨实现对标'
    )
  ),
  'board-audit'
);