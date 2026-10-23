"""docs/acoustic_attenuation_nu_k_squared.md 工件契约测试。

钉住声学衰减物理注记（δ = νk² 结论 + 离散修正链 c₂/β 闭式 +
verified acoustics 指向）与 docs 索引登记，防止文档工件被删除或
关键系数被改坏。
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_NOTE = _REPO_ROOT / "docs" / "acoustic_attenuation_nu_k_squared.md"


def _note_text() -> str:
    return _NOTE.read_text(encoding="utf-8")


def test_note_exists_and_states_delta_nu_k_squared():
    text = _note_text()
    assert "delta_cont = nu * k**2" in text
    assert "nu = cs**2 * (tau - 1/2) = (tau - 1/2) / 3" in text
    assert "benchmarks/verified/acoustics" in text


def test_note_states_discrete_correction_chain():
    text = _note_text()
    assert "c2 = (2*tau - 1)**2 / 12" in text
    assert "beta = -(12*tau**2 - 12*tau + 1) / 72" in text
    # 三链闭环的实测证据数字（τ=1.0 主档）
    assert "+1.3145%" in text
    assert "+0.0812%" in text


def test_docs_index_lists_the_note():
    index = (_REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    assert "acoustic_attenuation_nu_k_squared.md" in index
