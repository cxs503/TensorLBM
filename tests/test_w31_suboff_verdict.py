"""W31 守卫测试：DTRC SUBOFF 转录裁决工件的零手抄 / 可重建 / 判据不美化。

跑法：pytest tests/test_w31_suboff_verdict.py -q
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

import w31_suboff_transcription_verdict as W  # noqa: E402

ART = ROOT / "benchmarks" / "artifacts" / "w31_suboff_transcription_verdict.json"
DOC = ROOT / "docs" / "w31-suboff-transcription.md"


@pytest.fixture(scope="module")
def art():
    assert ART.exists(), "工件缺失：%s" % ART
    return json.loads(ART.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt():
    return W.build()


def test_artifact_paths_required_by_task(art):
    assert ART.relative_to(ROOT).as_posix() == "benchmarks/artifacts/w31_suboff_transcription_verdict.json"
    assert DOC.relative_to(ROOT).as_posix() == "docs/w31-suboff-transcription.md"
    assert (ROOT / "tests" / "test_w31_suboff_verdict.py").exists()
    for sec in ("three_family_pointwise", "conflict_attribution", "lbm_same_re_vs_arbitrated_anchor"):
        assert art[sec], "工件缺少必需节 %s" % sec


def test_provenance_sha256_matches_files(art):
    assert len(art["provenance"]) >= 7
    for rec in art["provenance"]:
        b = (ROOT / rec["path"]).read_bytes()
        assert hashlib.sha256(b).hexdigest() == rec["sha256"], rec["path"]
        assert len(b) == rec["bytes"], rec["path"]
        assert rec["path"] in W.SRC.values()


def test_artifact_rebuild_is_deterministic(art, rebuilt):
    volatile = {"generated_utc", "git"}
    a = {k: v for k, v in art.items() if k not in volatile}
    b = {k: v for k, v in rebuilt.items() if k not in volatile}
    assert a == b, "工件与脚本重建结果不一致（说明有人手改过工件）"


def test_zero_hand_copy_no_anchor_literals_in_builder():
    """构造器源码里不得出现任何锚数值字面量（值只能来自文件解析）。"""
    src = (ROOT / "benchmarks" / "w31_suboff_transcription_verdict.py").read_text(encoding="utf-8")
    f1 = W.parse_f1_doc()
    f2 = W.parse_f2_paper()
    f3 = W.parse_f3_registry()
    values = []
    for row in f1["AFF-1"] + f1["AFF-8"]:
        values += ["%.2f" % row["rt_n"], "%.1f" % row["rt_n"], "%.2f" % row["v_kn"]]
    values += ["%.3f" % r["v_ms"] for r in f2] + ["%.2f" % r["rt_n"] for r in f2]
    values += ["%.2f" % r["rt_n"] for r in f3]
    values += ["%.3f" % W.constants()["Lmax_m"], "%.3f" % W.constants()["wsa_m2"],
               "%.4e" % W.constants()["nu"]]
    leaked = []
    for v in sorted(set(values)):
        if len(v) < 4:            # 过短数字（如 "10"）会与代码常量撞车，无语义
            continue
        if re.search(r"(?<![\d.])" + re.escape(v) + r"(?![\d])", src):
            leaked.append(v)
    assert leaked == [], "构造器里出现手抄的锚数值：%s" % leaked


def test_c1_gate_and_both_denominators_preserved(art):
    c1 = next(c for c in art["criteria"] if c["id"] == "C1")
    m = c1["measured"]
    gate = art["declared_params"]["gate15_pct"]
    # 原始两种分母口径都必须原样记录，不许只留好看的
    assert m["raw_max_rel_diff_pct__denom_F1_bare"] > gate      # 17.05% > 15%
    assert m["raw_max_abs_rel_diff_pct__denom_F2_reference"] < gate  # 14.58% < 15%
    assert m["post_conversion_max_rel_diff_pct"] <= gate
    assert m["post_conversion_max_rel_diff_pct"] <= 1e-9        # 换算后逐点精确一致
    assert c1["verdict"] == "PASS"
    # 族对矩阵：三对里两对在门内、一对冲突
    fp = m["family_pair_matrix"]
    assert fp["F1_vs_F3_same_config_max_abs_pct"] <= 1e-9
    assert fp["F2_vs_F1_AFF8_row_max_abs_pct"] <= gate
    assert fp["n_pairs_conflicting"] == 1 and fp["n_pairs_within_gate"] == 2


def test_c2_every_conflict_point_is_attributed(art):
    allowed = {"unit", "scale", "wetted_area", "delta_cf"}
    rows = art["conflict_attribution"]
    assert len(rows) >= 5
    assert sum(1 for r in rows if r["unattributed"]) == 0
    for r in rows:
        assert r["class"] in allowed
        assert r["class_code"] == "SCALE_REFERENCE_BODY"
        # 另外三类必须给出量化结果并显式否证
        for k in ("unit", "wetted_area", "delta_cf"):
            assert r["alternatives_rejected"][k]["verdict"].startswith("REJECTED")
            assert isinstance(r["alternatives_rejected"][k]["max_explained_pct"], float)
        # 换算后残差必须在 15% 门内（且这里应精确为 0）
        assert abs(r["post_conversion_rel_diff_pct"]) <= art["declared_params"]["gate15_pct"]
    c2 = next(c for c in art["criteria"] if c["id"] == "C2")
    assert c2["verdict"] == "PASS"


def test_c3_same_re_gate_is_honest_fail(art):
    lbm = art["lbm_same_re_vs_arbitrated_anchor"]
    formal = json.loads((ROOT / W.SRC["LBM_formal"]).read_text(encoding="utf-8"))
    assert lbm["n_rows"] == len(formal["table"]) == 5
    for row, src in zip(lbm["rows"], formal["table"]):
        assert row["Re"] == src["Re"]
        assert row["CT_lbm"] == src["CT_sim"]
        # 该分支自记的锚 C_T 与本次裁决后锚一致（说明裁决不改变 C3 结论）
        assert abs(row["CT_EFD_vs_arbitrated_rel_diff_pct"]) < 1.0
        assert row["abs_dev_pct"] > 15.0 and row["gate15_pass"] is False
        assert row["band3_pass"] is False
    assert lbm["max_abs_dev_pct"] == pytest.approx(max(r["abs_dev_pct"] for r in lbm["rows"]))
    assert lbm["answer_lt_15pct"] is False
    assert lbm["answer_lt_3pct"] is False
    assert lbm["admissible_for_anchor"] is False
    c3 = next(c for c in art["criteria"] if c["id"] == "C3")
    assert c3["verdict"] == "honest FAIL"          # 不允许被改写成 PASS
    assert c3["measured"]["answer_lt_3pct"] is False


def test_c4_doc_is_generated_from_artifact_zero_handcopy(art):
    """正文必须等于由工件注入的结果（任何手改都会被这条抓住）。"""
    assert DOC.exists()
    on_disk = DOC.read_text(encoding="utf-8")
    assert on_disk == W.render_doc(art)
    # 关键判据数字必须出现在正文里（程序化注入）
    lbm = art["lbm_same_re_vs_arbitrated_anchor"]
    assert ("%.1f%%" % lbm["max_abs_dev_pct"]) in on_disk
    assert "`scale` / SCALE_REFERENCE_BODY" in on_disk
    assert "benchmarks/artifacts/w31_suboff_transcription_verdict.json" in on_disk


def test_claim_boundary_and_anchor_grade(art):
    cb = art["claim_boundary"]
    assert cb["may_be_cited_as_published_experimental_benchmark"] is False
    assert set(cb["refused_labels"]) >= {"A", "B"}
    assert art["anchor_grade"]["grade_letter"] == "C"
    assert art["anchor_grade"]["acquisition_of_primaries"].startswith("WITHHELD")
    assert "prereg" in art and art["prereg"]["alternatives"]


def test_figures_exist():
    for name in ("w31_ratio_vs_speed.png", "w31_lbm_same_re.png"):
        assert (ROOT / "figures" / name).exists(), name