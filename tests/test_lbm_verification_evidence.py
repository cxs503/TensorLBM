#!/usr/bin/env python3
"""证据链回归：解析解验证文档 ↔ 已提交的图/账本必须彼此自洽。

`docs/lbm_analytic_verification_r1.md` 公布测量误差，并指向一张由
`scripts/verify_lbm_analytic_convergence.py` 生成的图。此前那张图只存在于工作树：
`![...](../artifacts/lbm_analytic_verification/convergence.png)` 指向一个**从未提交**
的路径，新克隆里是死链，而 CI 侧没有任何测试看文档的「证据侧」——只测了求解器数字。
本模块补上这一侧，全部检查都是静态/读文件（不跑 LBM，秒级）：

* 两篇证据文档（`lbm_analytic_verification_r1.md`、`m4645_ci_red_fix_evidence.md`）
  里的每个图片链接都必须解析到仓库内**真实存在**的文件，且不能被 `.gitignore` 吃掉；
* 每张被引用的 PNG 必须是真 PNG（magic bytes + IHDR 尺寸），且不是 1×1 占位符；
* 已提交的错误账本 `artifacts/lbm_analytic_verification/convergence.json` 必须与测试模块
  的验收阈值一致、四个 canonical 案例全部在阈值内，且两个负控都被检出
  （阈值若不具鉴别力，下面这些「通过」就是空话）。

刻意不做的事：不在此重算 LBM（那是 `scripts/verify_lbm_analytic_convergence.py` 的活，
约 3 分钟），也不对全仓 `docs/` 做链接巡检（历史欠账不属于本测试的范围，见 §局限）。
"""

from __future__ import annotations

import json
import re
import struct
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts" / "lbm_analytic_verification"
LEDGER = ART / "convergence.json"
TEST_MODULE = REPO / "tests" / "test_lbm_verification.py"

EVIDENCE_DOCS = (
    REPO / "docs" / "lbm_analytic_verification_r1.md",
    REPO / "docs" / "m4645_ci_red_fix_evidence.md",
)

# ![alt](target)  /  ![alt](target "title")
_IMAGE = re.compile(r"!\[[^\]]*\]\(\s*([^)\s]+)(?:\s+\"[^\"]*\")?\s*\)")
_CASE_NAME = re.compile(r"\"case\":\s*\"([^\"]+)\"")
_REL_TOL = re.compile(r"^REL_TOL\s*=\s*([0-9.eE+-]+)", re.MULTILINE)


def _primary_targets(md: Path) -> list[str]:
    """图片链接里的目标路径（去掉 #anchor 与 http(s) 外链）。"""
    targets = []
    for raw in _IMAGE.findall(md.read_text(encoding="utf-8")):
        if raw.startswith(("http://", "https://", "data:")):
            continue
        targets.append(raw.split("#", 1)[0])
    return targets


def _png_size(path: Path) -> tuple[int, int]:
    """从 IHDR 读宽高（不依赖 PIL）。"""
    blob = path.read_bytes()
    assert blob[:8] == b"\x89PNG\r\n\x1a\n", f"{path} 不是 PNG（magic bytes 不符）"
    assert blob[12:16] == b"IHDR", f"{path} 缺少 IHDR 块"
    return struct.unpack(">II", blob[16:24])


def _tracked(path: Path) -> bool | None:
    """该文件是否已被 git 跟踪；不在 git 工作树里时返回 None（跳过）。"""
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "--error-unmatch", str(path.relative_to(REPO))],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - 无 git 的环境
        return None
    if proc.returncode == 0:
        return True
    if "not a git repository" in proc.stderr:
        return None
    return False


@pytest.mark.parametrize("doc", EVIDENCE_DOCS, ids=lambda p: p.name)
def test_evidence_docs_have_no_dead_image_links(doc: Path) -> None:
    """文档里的每张图都必须真的存在于仓库里（否则新克隆看到的是死链）。"""
    assert doc.is_file(), f"证据文档缺失：{doc}"
    targets = _primary_targets(doc)
    assert targets, f"{doc.name} 里没有任何图片链接——图注可能被删掉了"

    missing = []
    for target in targets:
        resolved = (doc.parent / target).resolve()
        if not resolved.is_file():
            missing.append(f"{target} -> {resolved}")
    assert not missing, f"{doc.name} 存在死链：{missing}"


@pytest.mark.parametrize("doc", EVIDENCE_DOCS, ids=lambda p: p.name)
def test_evidence_figures_are_committed_files(doc: Path) -> None:
    """图不仅要存在，还要已入库：只在工作树里生成的图对读者等于不存在。"""
    for target in _primary_targets(doc):
        resolved = (doc.parent / target).resolve()
        if not resolved.is_file():
            continue  # 存在性由上一个测试负责，这里只判「是否入库」
        tracked = _tracked(resolved)
        if tracked is None:  # pragma: no cover - 非 git 工作树（sdist 等）
            pytest.skip("不在 git 工作树内，无法判断是否入库")
        assert tracked, f"{resolved.relative_to(REPO)} 未被 git 跟踪（图只在工作树里）"


def test_referenced_pngs_are_non_trivial() -> None:
    """被引用的 PNG 必须是有内容的真图（不是 1×1 占位或空文件）。"""
    checked = 0
    for doc in EVIDENCE_DOCS:
        for target in _primary_targets(doc):
            path = (doc.parent / target).resolve()
            assert path.is_file(), f"死链：{target}"
            width, height = _png_size(path)
            assert path.stat().st_size > 10_000, (
                f"{path.name} 只有 {path.stat().st_size} 字节，疑似空图"
            )
            assert width >= 600 and height >= 300, (
                f"{path.name} 尺寸 {width}x{height} 过小，不像图表"
            )
            checked += 1
    assert checked >= 2, "至少应核对两张图（解析解收敛图 + CI 门禁归档图）"


def test_error_ledger_exists_with_expected_schema() -> None:
    payload = json.loads(LEDGER.read_text(encoding="utf-8"))
    for key in ("device", "dtype", "rel_tol_pct", "canonical", "controls", "runtime_s"):
        assert key in payload, f"错误账本缺少字段 {key}"
    assert payload["rel_tol_pct"] == pytest.approx(1.0)  # 1% 验收线，以百分数计
    assert set(payload["canonical"]) == set(
        _CASE_NAME.findall(TEST_MODULE.read_text(encoding="utf-8"))
    )


def test_error_ledger_tolerance_matches_test_module() -> None:
    """账本的阈值必须等于测试模块断言的阈值，否则「已提交通过证据」是另一套口径。"""
    match = _REL_TOL.search(TEST_MODULE.read_text(encoding="utf-8"))
    assert match, "无法从 tests/test_lbm_verification.py 解析 REL_TOL"
    rel_tol = float(match.group(1))
    payload = json.loads(LEDGER.read_text(encoding="utf-8"))
    assert payload["rel_tol_pct"] == pytest.approx(rel_tol * 100.0)


def test_all_canonical_cases_pass_the_published_ledger() -> None:
    payload = json.loads(LEDGER.read_text(encoding="utf-8"))
    tol = payload["rel_tol_pct"]
    for case, entry in payload["canonical"].items():
        assert entry["err_pct"] < tol, f"{case} 误差 {entry['err_pct']:.3e}% 超过 {tol:g}% 验收线"


def test_negative_controls_are_detected_by_the_threshold() -> None:
    """阈值必须真的能拒绝错误物理：τ 偏移与关掉壁面反弹都要被 1% 判据抓住。"""
    controls = json.loads(LEDGER.read_text(encoding="utf-8"))["controls"]
    tau = controls["tau_shift"]
    assert tau["deviation_pct"] > tau["tolerance_pct"], "τ=1.05 的粘度偏移没被检出，判据形同虚设"
    nobb = controls["no_bounce_back"]
    assert nobb["couette_err_pct"] > nobb["tolerance_pct"], (
        "关掉反弹仍判通过，Couette 案例没有壁面信息"
    )
