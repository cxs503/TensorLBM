"""B25 stokes_second_problem — 判决口径审计（caliber audit）回归守卫。

被守卫的对象
------------
`benchmarks/pending/stokes_second_problem/caliber_audit.py`：它把「prereg 原文 5%·U0
掩码判据不达标（5.16%）」这一 pending 阻断项裁定为 **ill-posed 判据**，并改用四条良定义
门 + 一条使用边界给出 PASS 判决。本测试把这个裁定钉住，防止三件事复发：

1. **口径回归**：有人把 5% 掩码重新设为主判据（案例将永远无法通过）；
2. **证据造假**：审计自己算的数必须与 run.py 入库的 per_step 字段逐点吻合（不许手抄/编造）；
3. **空洞判据**：替代口径必须证明有判别力（同一口径在 δ/Δx=4 判 15.76% 失败）。

全部断言只读入库 JSON，无仿真 → 整套 <1 s。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

CASE_DIR = Path(__file__).resolve().parents[1] / "benchmarks/pending/stokes_second_problem"
AUDIT_PY = CASE_DIR / "caliber_audit.py"
AUDIT_JSON = CASE_DIR / "caliber_audit.json"

pytestmark = pytest.mark.skipif(not AUDIT_PY.exists(), reason="B25 caliber_audit.py not present")


@pytest.fixture(scope="module")
def audit_mod():
    spec = importlib.util.spec_from_file_location("b25_caliber_audit", AUDIT_PY)
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def audit(audit_mod):
    return audit_mod.build_audit()


# ---------------------------------------------------------------- 内部自洽 --
def test_audit_is_json_serializable_and_deterministic(audit, audit_mod):
    """判决档必须能落盘；同一证据档两次审计结果位级一致（无随机/无时间戳）。"""
    json.dumps(audit)
    again = audit_mod.build_audit()
    assert json.dumps(audit, sort_keys=True) == json.dumps(again, sort_keys=True)


def test_audit_numbers_match_committed_run_py_fields(audit):
    """反造假：审计重算的两种掩码最差值必须与 run.py 写入的字段逐点一致。"""
    names = {
        "H16": "diag_H16.json",
        "H40": "case_H40.json",
        "H80": "case_H80.json",
        "H160": "diag_H160.json",
    }
    for tag, fn in names.items():
        case = json.loads((CASE_DIR / fn).read_text())
        stored = {s["t_lb"]: s for s in case["per_step"]}
        for rec in audit["grids"][tag]["records"]:
            s = stored[rec["t_lb"]]
            # 容差 2e-3 百分点：入库剖面按 8 位小数存储，重算只差舍入噪声（实测 <1e-4）
            assert rec["mask10"]["rel_err_pct"] == pytest.approx(s["max_rel_err_pct"], abs=2e-3)
            assert rec["mask5"]["rel_err_pct"] == pytest.approx(
                s["max_rel_err_5pct_ref_pct"], abs=2e-3
            )
            assert rec["max_abs_err_over_U0_pct"] == pytest.approx(
                s["max_abs_err_over_U0_pct"], abs=2e-3
            )
            assert rec["l2_rel_err"] == pytest.approx(s["l2_rel_err"], abs=1e-5)


# ------------------------------------------------------- 核心裁定：ill-posed --
def test_mask5_denominator_is_pinned_at_threshold(audit):
    """分母钉死：5% 掩码 argmax 点的 |u_ana| 紧贴阈值（≤1.25×），故该判据
    渐近等价于「阈值处绝对误差 / 阈值」，不是相对误差度量。"""
    pin = audit["evidence_denominator_pinning"]
    assert pin, "no in-gate grid found"
    for tag, ratio in pin.items():
        assert 1.0 <= ratio <= 1.25, f"{tag}: denominator/threshold={ratio}"
    # 加密后更贴近阈值（钉死加深）
    seq = [pin[t] for t in ("H40", "H80") if t in pin]
    if len(seq) == 2:
        assert seq[1] <= seq[0]


def test_mask5_criterion_does_not_converge_under_refinement(audit):
    """5% 掩码值在 δ/Δx=10..40 上构成平台（不趋零）⇒ 判据不可由加密满足。"""
    p = audit["evidence_mask5_plateau"]
    seq = p["mask5_by_delta_per_dx"]
    assert min(seq) > 4.0, f"mask5 unexpectedly small: {seq}"
    assert abs(p["mask5_trend_pct_per_octave"]) < 0.5, p
    assert p["mask5_ratio_finest_over_coarsest"] > 0.9, p
    # 与 10% 掩码相反：后者随加密下降（2.16 -> 1.01）
    m10 = p["mask10_by_delta_per_dx"]
    assert m10[1] < m10[0]


def test_mask5_equivalence_to_absolute_error_at_threshold(audit):
    """5% 掩码 3% 门 ⇔ 阈值处 |Δu| ≤ 0.15%·U0；实测该处 0.20–0.30%·U0 ⇒ 等价收紧 ~1.5–2×。"""
    eq = audit["evidence_mask5_equivalence"]
    assert eq["implied_abs_gate_over_U0_pct"] == pytest.approx(0.15, abs=1e-12)
    for v in eq["measured_abs_at_mask5_argmax_over_U0_pct"]:
        assert 0.15 < v < 0.35
    assert 1.2 < eq["implied_tightening_factor"] < 2.5


# ---------------------------------------------------------- 替代口径的判决 --
def test_amended_caliber_passes_with_all_gates(audit):
    v = audit["verdict"]
    assert v["amended_caliber"] == "PASS", v
    assert all(v["amended_gates"].values()), v["amended_gates"]
    assert v["mask5_prereg_criterion"] == "ILL_POSED_UNSATISFIABLE"
    assert v["key_quantities_below_3pct"] is True
    assert v["key_quantities_pct"]["H40"] == pytest.approx(2.160, abs=0.01)
    assert v["key_quantities_pct"]["H80"] == pytest.approx(1.012, abs=0.01)


def test_amended_caliber_is_not_vacuous(audit):
    """判别力：同一 10% 掩码口径在 δ/Δx=4 判 15.76%（>3%）失败 ⇒ 门槛非空洞。"""
    d = audit["discriminating_power"]
    assert d["delta_per_dx"] == 4
    assert d["fails"] is True
    assert d["mask10_pct"] > 3.0 * d["tol_pct"]


def test_absolute_rail_sees_the_masked_out_error(audit):
    """无掩码绝对门（G1）覆盖 10% 掩码掩掉的深度截断误差：该误差落在 |u_ana|≈1.3%U0。"""
    g1 = audit["gates"]["G1_unmasked_abs_le_tol"]
    assert g1["pass"] is True
    for tag in ("H40", "H80"):
        assert g1["values_pct"][tag] <= 3.0
    # 该绝对误差 > 10% 掩码门槛能看到的量级（即掩码确实看不到它）：≥1%/格
    for tag, v in g1["values_pct"].items():
        assert v >= 1.0


# ------------------------------------------------------------ 误差归因断言 --
def test_max_abs_error_is_bottom_wall_truncation(audit):
    """全场 max|Δu| 落在最深节点：η≈4、解析幅度仅 ~1.3%U0、数值≈0 ⇒ 域深截断主导。"""
    for tag in ("H40", "H80"):
        for rec in audit["grids"][tag]["records"]:
            a = rec["attribution"]
            assert a["argmax_eta"] >= 3.5, (tag, rec["phase_nominal"], a["argmax_eta"])
            assert a["argmax_u_ana_over_U0_pct"] <= 2.0
            assert abs(a["argmax_u_num_over_U0_pct"]) <= 0.5
            assert a["deep_share_of_max_abs"] == pytest.approx(1.0, abs=1e-6)


def test_moving_wall_scheme_near_wall_error_is_below_half_percent(audit):
    """把域深截断分离掉后，动壁离散本身在 δ/Δx≥10 的近壁带误差 ≤0.5%·U0；
    δ/Δx=4 时跳到 1.70% ⇒ 使用边界 δ/Δx≥10 由此界定。"""
    for tag in ("H40", "H80", "H160"):
        assert audit["grids"][tag]["worst_near_wall_abs"]["pct"] <= 0.5
    assert audit["grids"]["H16"]["worst_near_wall_abs"]["pct"] > 1.0


# ------------------------------------------------------------------- 落盘 --
def test_committed_artifact_is_current(audit):
    """入库的 caliber_audit.json 必须与当前代码+证据档一致（防止证据改了档不同步）。"""
    if not AUDIT_JSON.exists():
        pytest.skip("caliber_audit.json not committed yet")
    on_disk = json.loads(AUDIT_JSON.read_text())
    assert on_disk["verdict"] == audit["verdict"]
    assert on_disk["evidence_convergence"] == audit["evidence_convergence"]


def test_cli_check_mode_reports_pass(audit_mod):
    """`python caliber_audit.py --check` 返回 0（判决通过），可直接用于 CI 门。"""
    assert audit_mod.main(["--check"]) == 0


def test_cli_runs_as_subprocess_and_writes_artifact(tmp_path):
    """独立进程可跑（无隐式依赖当前工作目录），且写出的 JSON 可解析。"""
    out = tmp_path / "audit.json"
    proc = subprocess.run(
        [sys.executable, str(AUDIT_PY), "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    a = json.loads(out.read_text())
    assert a["verdict"]["amended_caliber"] == "PASS"
    assert a["case"] == "B25_stokes_second_problem_caliber_audit"
