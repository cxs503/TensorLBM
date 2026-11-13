"""m17735 审计 evidence（gate 契约格式；指针全部来自本单实测，m17716 工作树 head=4846dcdd）。"""
import hashlib
from pathlib import Path

W = Path("/root/tp-agent/work/m17716")
LEVEL = 4
A = "artifacts/xval_block_amr_equal_level_stream_correction"


def sha(rel: str) -> str:
    p = W / rel
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "MISSING"


def size(rel: str) -> int:
    p = W / rel
    return p.stat().st_size if p.is_file() else -1


evidence = {
    "impl_files": [
        f"{W}/src/tensorlbm/amr_equal_level_stream.py",
        f"{W}/src/tensorlbm/amr_cavity2d.py",
        f"{W}/benchmarks/xval_block_amr_equal_level_stream_correction/run_xval.py",
        f"{W}/benchmarks/xval_block_amr_equal_level_stream_correction/make_report.py",
        f"{W}/docs/XVAL_BLOCK_AMR_EQUAL_LEVEL_STREAM_CORRECTION.md",
    ],
    "impl_files_note": "amr_equal_level_stream.py:1-379（EqualLevelBorderStreamCorrection：blocked 掩码→链接表、prepare/restore 流式期顶替、D2Q9）；amr_cavity2d.py:185/:199-200/:260-307（coarse_shadow 配置/校验/frozen 分支流式处 prepare/restore）；run_xval.py:1-262（F-H 标定扫描+不变量+benchmark.json 写出）；sha256 见 db_write_output.txt 与 VERDICT_m17716.md §3",
    "entry": f"{W}/benchmarks/xval_block_amr_equal_level_stream_correction/run_xval.py",
    "run_artifacts": [
        f"{W}/{A}/benchmark.json",
        f"{W}/{A}/results.json",
        f"{W}/{A}/run.log",
        f"{W}/{A}/REPORT.md",
        f"{W}/{A}/SUMMARY.md",
        f"{W}/{A}/fig1_calibration.png",
        f"{W}/{A}/fig2_correction_effect.png",
        f"{W}/{A}/figures.json",
    ],
    "tests": [
        "tests/test_amr_equal_level_stream.py::test_uniform_field_is_inert",
        "tests/test_amr_equal_level_stream.py::test_prepare_restore_is_bit_exact",
        "tests/test_amr_equal_level_stream.py::test_links_point_from_blocked_to_active",
        "tests/test_amr_equal_level_stream.py::test_donor_modes_differ_and_change_the_stream_result",
        "tests/test_amr_equal_level_stream.py::test_prolongation_scales_and_restriction_inverse",
        "tests/test_amr_equal_level_stream.py::test_analytic_calibration_peaks_at_unit_scale",
        "tests/test_amr_equal_level_stream.py::test_cavity_frozen_shadow_uses_the_correction",
    ],
    "integration_callers": [
        "amr_cavity2d: CavityAMR2D.step() frozen-shadow 分支（coarse_shadow='frozen'）在 collide 后 stream 前调用 EqualLevelBorderStreamCorrection.prepare/restore 并冻结被覆盖粗行——waLBerla 做法真实调用点；指针 src/tensorlbm/amr_cavity2d.py:260-307（worktree m17716 head=4846dcdd）；集成断言 tests/test_amr_equal_level_stream.py::test_cavity_frozen_shadow_uses_the_correction",
    ],
    "benchmark": {
        "registered_case": "tensor_benchmark_cases id=64（本单前 anchor 0 行，referee row absent → 本单以独立复跑数字补注册，先例 id=47/55），status=passed",
        "got_vs_ref": {
            "rows": 12, "passed": 12,
            "worst_err_pct": 0.09837083993351455,
            "tolerance_pct": 15.0,
            "within_tolerance": True,
            "method": "本单独立复跑 repro_cmd（/tmp/xval_repro），与已提交 benchmark.json 逐行 (kpi,ref,got,pass) 相等；ref=解析 TGV/剪切场精确非平衡（analytic_fields_2d 程序生成，零手抄）",
        },
        "report_six_sections": "tensor_reports id=16654（mission_id=17716, published, markdown 4959 B）：§1 案例介绍/§2 怎么做的/§3 计算结果/§4 结果图/§5 注意点/§6 误差判定 六段齐全，§6 逐项显式回答 <15%",
    },
    "repro_cmd": "cd /root/tp-agent/work/m17716 && PYTHONPATH=src python benchmarks/xval_block_amr_equal_level_stream_correction/run_xval.py --out artifacts/xval_block_amr_equal_level_stream_correction（本单实跑 rc=0；重算 12 行与已提交工件逐行相等，fh_scale_error=0.0009837083993351455 逐位一致）",
    "external_anchor": "lssfau/walberla block-amr-equal-level-stream-correction（oss_solutions id=283；EqualLevelBorderStreamCorrection.h/TimeStep.h/CurlBasedLevelDetermination.h）；waLBerla 本体未运行（无网络/GPL 禁搬码/无构建链）→ ref 用解析场，OSS 跑数输出 WITHHELD，报告 §5 已如实标注",
    "findings": {
        "F1_referee_row_absent": "tensor_benchmark_cases 本单前 0 行 → 已按先例补注册（audited numbers，非 subject 自报）",
        "F2_frozen_dynamics_withheld": "frozen 模式整场时间推进精度未验证（早期草稿粗层额外耗散 ~2x 衰减率未归因，已从交付移除）→ 报告 §5 如实声明，不计入能力声明；不阻塞本轮声明范围（算子不变量+F-H 标定）",
        "F3_repo_conventions_wrong": "实测 F-H 因子=1.000（err 0.0984%）：本仓 amr_cavity2d._fh_rescale=0.5833 与 amr_population_transfer.rescale_nonequilibrium=2.3333 均偏离且方向相反——属仓内既有代码问题，留后续轮，非本 mission 缺陷",
    },
    "level_rationale": "L1✅（7 passed，不变量精确 0）/L2✅（真模块 379 行+入口+工件）/L3✅（CavityAMR2D frozen 分支集成+集成测试）/L4✅（完整 xval 算例 3 分辨率×流场+误差分析+确定性复现，12/12 在容差内，referee row 已注册）/L5❌（做法的动力学收益未验证）→ 裁定 L4",
    "checks_not_run": [
        "waLBerla 本体运行（无网络/GPL/无构建链，WITHHELD）",
        "frozen 模式整场时间推进精度（subject 自报 WITHHELD，本单不代验）",
        "集群 loong64 复跑（单机 <10 s，无必要）",
    ],
}
