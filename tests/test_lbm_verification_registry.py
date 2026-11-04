#!/usr/bin/env python3
"""文档数字漂移闸门：``docs/LBM_VERIFICATION_TEACHING.md`` 里的数字必须可复算。

验证欠账背景（m3579）：教学模块把 Step A–E 的误差写成"0.00% / 0.34% / 4.76%"
等具体数字，但仓库里原先**没有任何机制**把这些数字与实跑结果绑在一起 ——
文档可以无限期地宣称一个已经漂走的数值。本文件是那条缺失的闸门：

1. ``test_registry_snapshot_matches_live_chain``
   注册表 ``docs/lbm_verification_registry.json`` 的实测值与**本次重算**一致
   （物理量 0.5% 相对容差；误差百分比 0.1 个百分点绝对容差）。
2. ``test_doc_table_is_generated_from_code``
   文档里由生成器注入的表格块与**本次重新渲染**逐字一致 —— 改代码不更新文档
   就会红，手改文档数字也会红。
3. ``test_registry_entries_are_all_executable_and_green``
   注册表不许出现"没有判据的声称"：每条都必须 passed=True 且误差 < 阈值。
4. ``test_withheld_claims_carry_reason``
   Step E（圆柱 Cd 网格收敛）目前**无仓内可执行证据**，注册表必须显式记
   WITHHELD 且写明原因与所需算力，不允许静默当已验证。

运行：``python -m pytest tests/test_lbm_verification_registry.py -q``
"""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent / "validation"))

import lbm_analytic_chain as chain  # noqa: E402

REGISTRY = ROOT / "docs" / "lbm_verification_registry.json"
DOC = ROOT / "docs" / "LBM_VERIFICATION_TEACHING.md"

#: 文档里生成块的界定标记（内容由 tests/validation/lbm_analytic_chain.py 渲染）
BEGIN = "<!-- BEGIN GENERATED: lbm_chain_table -->"
END = "<!-- END GENERATED: lbm_chain_table -->"

PHYSICAL_TOL_REL = 0.005     # 物理量（ν, Cf）：0.5% 相对
ERROR_PCT_TOL_ABS = 0.1      # 误差百分比口径：0.1 个百分点绝对（同一台机器上应几乎逐位复现）


def _load_registry():
    assert REGISTRY.is_file(), f"缺少文档数字注册表 {REGISTRY}"
    return json.loads(REGISTRY.read_text())


def test_registry_schema_and_coverage():
    """注册表必须覆盖 Step A–D 四步，且 schema 版本明确。"""
    reg = _load_registry()
    assert reg["schema"] == "tensorlbm.verification.registry.v1"
    keys = {m["key"] for m in reg["measurements"]}
    assert keys == {"shear_wave", "couette", "poiseuille", "wall_friction"}, keys
    withheld = reg.get("withheld_claims", [])
    assert withheld, "Step E 等无仓内证据的声称必须显式登记为 WITHHELD"


def test_registry_snapshot_matches_live_chain():
    """注册表数字 = 本次重算数字（漂移闸门）。"""
    reg = _load_registry()
    live = {m.key: m for m in chain.run_chain("auto")}
    for rec in reg["measurements"]:
        m = live[rec["key"]]
        if rec["key"] in ("shear_wave", "wall_friction"):
            drift = abs(m.measured - rec["measured"]) / max(abs(rec["measured"]), 1e-12)
            assert drift < PHYSICAL_TOL_REL, (
                f"{rec['key']}: 注册表 {rec['measured']:.8f} vs 实测 {m.measured:.8f} "
                f"漂移 {drift * 100:.3f}% ≥ {PHYSICAL_TOL_REL * 100}% —— 请重新生成注册表"
            )
        else:
            drift = abs(m.measured - rec["measured"])
            assert drift < ERROR_PCT_TOL_ABS, (
                f"{rec['key']}: 注册表误差 {rec['measured']:.6f}% vs 实测 {m.measured:.6f}%，"
                f"差 {drift:.4f} 个百分点 ≥ {ERROR_PCT_TOL_ABS}"
            )
        # 判定也必须一致
        assert m.passed == rec["passed"], f"{rec['key']} 判定从 {rec['passed']} 变 {m.passed}"


def test_registry_entries_are_all_executable_and_green():
    """每条登记的数字都必须是可执行且达标的（不许登记一个 FAIL 却当已验证）。"""
    reg = _load_registry()
    thr = reg["threshold_pct"]
    for rec in reg["measurements"]:
        assert rec["passed"] is True, f"{rec['key']} 在注册表里是 FAIL 却被收录"
        assert rec["rel_err_pct"] < thr, f"{rec['key']} 误差 {rec['rel_err_pct']}% ≥ {thr}%"
        assert rec["settings"], f"{rec['key']} 缺少可复现的实验设置"
        assert rec["settings"].get("grid") and rec["settings"].get("steps"), rec["key"]


def test_doc_table_is_generated_from_code():
    """文档里的链路表格必须与代码重新渲染的结果逐字一致。"""
    assert DOC.is_file(), f"缺少文档 {DOC}"
    text = DOC.read_text()
    assert BEGIN in text and END in text, (
        "教学文档缺少生成块标记，文档数字无法被机器校验"
    )
    in_doc = text.split(BEGIN, 1)[1].split(END, 1)[0].strip("\n")
    rows = chain.run_chain("auto")
    nc = chain.negative_control_tau_perturbation("auto")
    bb_after = chain.bounce_back_after_stream_err("auto")
    bb_before = next(m for m in rows if m.key == "couette").rel_err_pct
    expected = chain.render_markdown_table(rows, nc, bb_before, bb_after).strip("\n")
    assert in_doc.strip() == expected, (
        "文档链路表格与实跑结果不一致（文档漂移）。重新生成：\n"
        "  python tests/validation/lbm_analytic_chain.py "
        "--markdown /tmp/t.md 后替换生成块"
    )


def test_withheld_claims_carry_reason():
    """WITHHELD 条目必须写明原因与重算成本，防止"沉默地当作已验证"。"""
    reg = _load_registry()
    for w in reg["withheld_claims"]:
        assert w["status"] == "WITHHELD"
        assert len(w.get("reason", "")) > 20, w
        assert w.get("cost", ""), w
        assert w.get("required_to_close", ""), w


def test_documented_step_e_numbers_are_flagged_in_prose():
    """Step E 的网格收敛表在正文里必须带 WITHHELD 标注（防"看似已验证"）。"""
    text = DOC.read_text()
    assert "WITHHELD" in text
    # 正文里出现的 Cd 网格收敛数字必须紧邻 WITHHELD 说明
    m = re.search(r"D=200", text)
    assert m is not None, "Step E 表格应仍在文档中（带 WITHHELD 标注）"
    window = text[max(0, m.start() - 2500): m.start() + 1500]
    assert "WITHHELD" in window, "Step E 的 Cd 收敛表未标注 WITHHELD"