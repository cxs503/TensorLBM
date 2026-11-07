"""Literature anchor table for the IN25-T6 external-mature-technology survey.

Design rules (owners' discipline, 2026-11-07):

* **No hand-typed bibliography.**  Every title/author/year/venue record is
  copied programmatically from the archived Crossref / arXiv payloads into
  ``data/lit_anchor_records.json`` (see
  ``scripts/in25_t6/emit_anchor_data.py``).  This module only adds the *curated*
  layer: the claim, its applicability window, and how far the claim was
  actually verified.
* **No unsourced numerical claim.**  A claim may be ``NUMERIC`` only when the
  number is readable in an open-access abstract or in the record title (both
  archived on disk); otherwise it is ``QUALITATIVE`` and cannot enter a scoring
  gate.
* **Absence is recorded.**  Anything the survey could not verify is listed in
  :data:`ABSENT_ITEMS` with a reason and is excluded from the scorer candidates
  instead of being paraphrased from memory.

The five measures mirror the mission's five-measure seat:
``large_grid`` (大网格先例), ``parallel_lbm`` (并行 LBM 成熟实践),
``amr`` (局部加密), ``les`` (LES-LBM 湍流), ``wall_model`` (壁面模型).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from importlib import resources
from typing import Any, Mapping

__all__ = [
    "MEASURES",
    "ClaimKind",
    "EvidenceSource",
    "AnchorStatus",
    "Citation",
    "Quantity",
    "LitAnchor",
    "AbsentItem",
    "LITERATURE_ANCHORS",
    "ABSENT_ITEMS",
    "AnchorTableReport",
    "load_record_store",
    "anchor_index",
    "validate_anchor_table",
]

MEASURES: Mapping[str, str] = {
    "large_grid": "大网格先例（规模/网格收敛研究）",
    "parallel_lbm": "并行 LBM 成熟实践（扩展性/域分解）",
    "amr": "局部加密（块结构 AMR/渐变网格）",
    "les": "湍流 LES-LBM（模型常数与 MRT 稳定性）",
    "wall_model": "壁面模型（近壁处理适用域与误差阶）",
}

_RECORD_STORE = ("data", "lit_anchor_records.json")


class ClaimKind(str, Enum):
    """How strong the claim is, i.e. what it may be used for."""

    NUMERIC = "numeric"          # a number quoted from an open-access source
    QUALITATIVE = "qualitative"  # a mechanism/practice statement, no number


class EvidenceSource(str, Enum):
    """Where the verification of *this claim* came from."""

    ABSTRACT_OA = "open_access_abstract"  # abstract archived on disk (arXiv / OA)
    TITLE = "record_title"                # number stated in the record title
    REGISTRY = "registry_metadata"        # bibliographic record only


class AnchorStatus(str, Enum):
    PENDING_REDTEAM = "pending_redteam"   # verified, awaiting T5 red team
    SCORER_CANDIDATE = "scorer_candidate"  # cross-consistent -> admissible anchor
    WITHHELD = "withheld"                 # verified record, claim not usable


@dataclass(frozen=True)
class Citation:
    anchor_id: str
    title: str
    authors: tuple[str, ...]
    year: int | None
    venue: str | None
    volume: str | None
    page: str | None
    id_kind: str
    endpoint: str
    retrieved_at: str

    @property
    def short(self) -> str:
        first = self.authors[0].split(",")[0] if self.authors else "anon"
        n = len(self.authors)
        etal = " et al." if n > 2 else ("" if n <= 1 else " & " + self.authors[1].split(",")[0])
        return f"{first}{etal} ({self.year})"


@dataclass(frozen=True)
class Quantity:
    """One number carried by a claim, with the place it was read from."""

    name: str
    value: float
    unit: str
    read_from: EvidenceSource


@dataclass(frozen=True)
class LitAnchor:
    anchor_id: str
    measure: str
    claim: str
    claim_kind: ClaimKind
    evidence: EvidenceSource
    quantities: tuple[Quantity, ...]
    conditions: str
    status: AnchorStatus
    cross_refs: tuple[str, ...] = ()
    caveat: str = ""
    oa_twin: str = ""   # same work, open-access version (abstract read from here)

    @property
    def citation(self) -> Citation:
        return _CITATIONS[self.anchor_id]

    @property
    def usable_for_scoring(self) -> bool:
        return self.status is AnchorStatus.SCORER_CANDIDATE


@dataclass(frozen=True)
class AbsentItem:
    item: str
    measure: str
    reason: str
    consequence: str


# --------------------------------------------------------------------------- #
# record store
# --------------------------------------------------------------------------- #
_STORE: dict[str, dict[str, Any]] | None = None


def load_record_store() -> dict[str, dict[str, Any]]:
    """Load the generated bibliographic record store (shipped in-package)."""
    global _STORE
    if _STORE is None:
        with resources.files("tensorlbm.external_survey").joinpath(*_RECORD_STORE).open(
            "r", encoding="utf-8"
        ) as fh:
            payload = json.load(fh)
        _STORE = payload["records"]
    return _STORE


def _cite(anchor_id: str) -> Citation:
    rec = load_record_store()[anchor_id]
    return Citation(
        anchor_id=anchor_id,
        title=rec["title"],
        authors=tuple(rec["authors"]),
        year=rec["year"],
        venue=rec["venue"],
        volume=rec["volume"],
        page=rec["page"],
        id_kind=rec["id_kind"],
        endpoint=rec["endpoint"],
        retrieved_at=rec["retrieved_at"],
    )


def _q(name: str, value: float, unit: str, src: EvidenceSource) -> Quantity:
    return Quantity(name=name, value=float(value), unit=unit, read_from=src)


# --------------------------------------------------------------------------- #
# curated anchor table
# --------------------------------------------------------------------------- #
LITERATURE_ANCHORS: tuple[LitAnchor, ...] = (
    # ---------------- measure 1: large-grid precedent ----------------------
    LitAnchor(
        anchor_id="2503.19674",
        measure="large_grid",
        claim=(
            "对全附体 SUBOFF（Re=1.2e7）做三阶格式 WMLES+FW-H 需要 1.03e8 量级网格；"
            "附体（围壳/舵）使远场 SPL 比裸体高约 20 dB，峰值 85.10 dB —— 即附体流是"
            "高网格量、高噪声敏感度的主矛盾，粗网格裸体结论不能外推到 AFF-8。"
        ),
        claim_kind=ClaimKind.NUMERIC,
        evidence=EvidenceSource.ABSTRACT_OA,
        quantities=(
            _q("grid_cells", 1.03e8, "cells", EvidenceSource.ABSTRACT_OA),
            _q("reynolds", 1.2e7, "dimensionless", EvidenceSource.ABSTRACT_OA),
            _q("scheme_order", 3, "order", EvidenceSource.ABSTRACT_OA),
            _q("spl_excess_appended_vs_bare", 20.0, "dB", EvidenceSource.ABSTRACT_OA),
            _q("spl_peak", 85.10, "dB", EvidenceSource.ABSTRACT_OA),
        ),
        conditions=(
            "Re=1.2e7（基于艇长）；WMLES + Ffowcs Williams-Hawkings；均匀 Cartesian 网格；"
            "压力/噪声量，非阻力验收；648 水听器 @500D"
        ),
        status=AnchorStatus.PENDING_REDTEAM,
        cross_refs=("2407.10475",),
        caveat="摘要未给阻力误差与网格收敛阶，故只能作规模/噪声先例，不能作阻力判据。",
    ),
    LitAnchor(
        anchor_id="2407.10475",
        measure="large_grid",
        claim=(
            "裸艇体 SUBOFF Re=1.2e7 的 WMLES（高阶格式）给出与时均实验更吻合的压力系数与"
            "速度脉动，优于既有文献 WMLES —— 说明 Re≈1e7 的 SUBOFF 已有独立的高保真数值"
            "参照族，但摘要未给网格量与阻力误差。"
        ),
        claim_kind=ClaimKind.NUMERIC,
        evidence=EvidenceSource.ABSTRACT_OA,
        quantities=(_q("reynolds", 1.2e7, "dimensionless", EvidenceSource.ABSTRACT_OA),),
        conditions=(
            "裸体（AFF-1）Re=1.2e7；壁面模化 LES；验证量=Cp、速度脉动谱，非总阻力；"
            "圆柱 Re=1e4 作为前置验证"
        ),
        status=AnchorStatus.PENDING_REDTEAM,
        cross_refs=("2503.19674",),
        caveat="阻力/网格收敛数字 absent（正文不可离线核验），不得据摘要声称阻力精度。",
    ),
    LitAnchor(
        anchor_id="10.1016/j.compfluid.2018.01.013",
        measure="large_grid",
        claim=(
            "附体潜艇外流（拖曳与自航两种构型）的 LES 已是成熟做法，自航构型必须处理"
            "螺旋桨-艇体干涉 —— 即外流验证不能只看拖曳工况。"
        ),
        claim_kind=ClaimKind.QUALITATIVE,
        evidence=EvidenceSource.REGISTRY,
        quantities=(),
        conditions="模型尺度 Re；贴体有限体积 LES；notional submarine（SUBOFF 族）",
        status=AnchorStatus.PENDING_REDTEAM,
        caveat="Crossref 记录不含摘要，结论仅由题录与既有领域共识支持，属 qualitative。",
    ),
    LitAnchor(
        anchor_id="10.1080/17445302.2026.2731435",
        measure="large_grid",
        claim="SUBOFF 在偏航（yaw）工况下的 LES 已发表：操纵/斜航姿态是当前外流研究前沿，阻力验收应含 0° 以外工况。",
        claim_kind=ClaimKind.QUALITATIVE,
        evidence=EvidenceSource.REGISTRY,
        quantities=(),
        conditions="SUBOFF；多偏航角；LES；阻力/流场",
        status=AnchorStatus.PENDING_REDTEAM,
        caveat="摘要 absent；不能引用其具体数值。",
    ),
    # ---------------- measure 2: parallel LBM practice ---------------------
    LitAnchor(
        anchor_id="10.1016/j.camwa.2020.01.007",
        measure="parallel_lbm",
        claim=(
            "waLBerla 的参考架构=块结构（block-structured）网格 + MPI 域分解 + CUDA 加速的"
            "多物理场框架；LBM 大规模并行的事实标准是把块结构分解、边界/幽灵层与内核生成"
            "分层实现，而不是单一全局网格。"
        ),
        claim_kind=ClaimKind.QUALITATIVE,
        evidence=EvidenceSource.REGISTRY,
        quantities=(),
        conditions="分布式内存 + GPU；块结构网格；多物理场耦合",
        status=AnchorStatus.PENDING_REDTEAM,
        caveat="摘要 absent：具体弱扩展效率数字列为 ABSENT（见 ABSENT_ITEMS A1）。",
    ),
    LitAnchor(
        anchor_id="10.1016/j.camwa.2020.03.022",
        measure="parallel_lbm",
        claim="Palabos 提供成熟的并行 LBM 实现与数据处理器（data-processor）抽象，可作为接口设计与回归对象。",
        claim_kind=ClaimKind.QUALITATIVE,
        evidence=EvidenceSource.REGISTRY,
        quantities=(),
        conditions="MPI 并行；D2Q9/D3Q19 族；多物理扩展",
        status=AnchorStatus.PENDING_REDTEAM,
    ),
    LitAnchor(
        anchor_id="10.1145/2503210.2503273",
        measure="parallel_lbm",
        claim="混合并行（MPI+GPU）LBM 已在复杂几何上做到 1e12 网格胞规模，说明‘万亿胞’量级并行是已被验证的工程上限参考。",
        claim_kind=ClaimKind.NUMERIC,
        evidence=EvidenceSource.TITLE,
        quantities=(_q("grid_cells", 1e12, "cells", EvidenceSource.TITLE),),
        conditions="复杂几何；混合 MPI+CUDA；静态块结构分解",
        status=AnchorStatus.PENDING_REDTEAM,
        cross_refs=("10.1016/j.camwa.2020.01.007", "10.1016/j.camwa.2020.03.022"),
        caveat="数值来自题录标题（a trillion cells），未含效率指标。",
    ),
    # ---------------- measure 3: local refinement --------------------------
    LitAnchor(
        anchor_id="10.1006/jcph.1998.6089",
        measure="amr",
        claim=(
            "LBM 局部加密的首个成熟方案（Filippova-Hänel）：粗/细网格界面通过分布函数插值"
            "+ 时间步比 2 的亚循环耦合，界面处必须保持动量和粘性一致。"
        ),
        claim_kind=ClaimKind.QUALITATIVE,
        evidence=EvidenceSource.REGISTRY,
        quantities=(),
        conditions="块结构嵌套网格；细化比 2；界面插值/时间外推",
        status=AnchorStatus.PENDING_REDTEAM,
        caveat="摘要 absent；机制描述为领域共识，误差阶未核验（ABSENT A2 相关）。",
    ),
    LitAnchor(
        anchor_id="10.1016/j.jcp.2012.03.015",
        measure="amr",
        claim=(
            "多域（multi-domain）LBM 加密的后续成熟做法：界面用体积/守恒形式处理并在两侧"
            "保持一致（Lagrava 等），是把 1998 类插值方案推进到工程可用的关键一步。"
        ),
        claim_kind=ClaimKind.QUALITATIVE,
        evidence=EvidenceSource.REGISTRY,
        quantities=(),
        conditions="双侧域分解；细化比 2；界面守恒条件",
        status=AnchorStatus.PENDING_REDTEAM,
    ),
    LitAnchor(
        anchor_id="10.1103/physreve.94.053311",
        measure="amr",
        claim="熵 LBM（ELBM/KBC 族）的多域加密方案已建立，可在跨音速/不可压多算例上复现 DNS/实验趋势。",
        claim_kind=ClaimKind.QUALITATIVE,
        evidence=EvidenceSource.ABSTRACT_OA,
        quantities=(),
        conditions="熵格子（ELBM）；多域加密；算例=湍流槽道/绕球/Rayleigh-Bénard/跨音速翼型",
        status=AnchorStatus.PENDING_REDTEAM,
        oa_twin="1608.06915",
    ),
    # ---------------- measure 4: LES-LBM ----------------------------------
    LitAnchor(
        anchor_id="10.1090/fic/006/12",
        measure="les",
        claim=(
            "LBM 的亚格子模型源头文献（Hou 等）：在 LBM 中引入 Smagorinsky 型涡粘，是对"
            "高 Re 流动做 LES-LBM 的标准起点；涡粘进入碰撞频率 τ 的方式决定了稳定域。"
        ),
        claim_kind=ClaimKind.QUALITATIVE,
        evidence=EvidenceSource.REGISTRY,
        quantities=(),
        conditions="Smagorinsky 涡粘；BGK/LBGK；高 Re",
        status=AnchorStatus.PENDING_REDTEAM,
        caveat="Cs 推荐区间与误差量化在本次离线核验中 absent（ABSENT A2）。",
    ),
    LitAnchor(
        anchor_id="10.1103/physreve.61.6546",
        measure="les",
        claim=(
            "MRT 碰撞算子的理论分析（Lallemand-Luo）：给出色散/耗散/各向同性与伽利略不变性"
            "的系统判据——高 Re 稳定性应通过调节松弛率谱（而不只是 τ）获得。"
        ),
        claim_kind=ClaimKind.QUALITATIVE,
        evidence=EvidenceSource.REGISTRY,
        quantities=(),
        conditions="MRT/D3Q19 族；线性稳定性分析；任意松弛率谱",
        status=AnchorStatus.PENDING_REDTEAM,
    ),
    # ---------------- measure 5: wall models ------------------------------
    LitAnchor(
        anchor_id="10.2514/3.61193",
        measure="wall_model",
        claim=(
            "Musker 显式光滑壁速度剖面：单一显式表达式覆盖粘性底层、过渡层与对数区，"
            "无需分区拼接——是壁面函数‘全域一致’分支的代表。"
        ),
        claim_kind=ClaimKind.QUALITATIVE,
        evidence=EvidenceSource.REGISTRY,
        quantities=(),
        conditions="零压力梯度光滑壁湍流边界层；y+ 从壁面到对数区",
        status=AnchorStatus.PENDING_REDTEAM,
    ),
    LitAnchor(
        anchor_id="10.1063/5.0283930",
        measure="wall_model",
        claim=(
            "壁面函数耦合的 LBM：slip-velocity bounce-back 在精度与网格收敛一致性上表现稳定，"
            "而三阶正则化边界高度依赖壁法向速度梯度重构（即使平壁也敏感）；正则化混合方案"
            "需要标定 —— 即优先选 slip-velocity 而非高阶重构。"
        ),
        claim_kind=ClaimKind.QUALITATIVE,
        evidence=EvidenceSource.ABSTRACT_OA,
        quantities=(),
        conditions="LBM + Spalart-Allmaras + 模型一致壁面函数；算例=充分发展槽道、零压梯度平板",
        status=AnchorStatus.PENDING_REDTEAM,
        cross_refs=("2601.18443", "2611.07137"),
    ),
    LitAnchor(
        anchor_id="2601.18443",
        measure="wall_model",
        claim=(
            "自适应 Cartesian 网格上的 LBM-RANS：slip-velocity bounce-back + 解析壁面函数 + "
            "负 SA 模型，可在 NACA0012 与 MD-30P30N 上给出干净的摩阻分布；但**未在 SUBOFF 上验证**。"
        ),
        claim_kind=ClaimKind.QUALITATIVE,
        evidence=EvidenceSource.ABSTRACT_OA,
        quantities=(),
        conditions="自适应 Cartesian；LBM-RANS；NACA0012/MD-30P30N；翼型类，无 SUBOFF",
        status=AnchorStatus.PENDING_REDTEAM,
        caveat="适用范围外推到艇体（强流向压力梯度+长细比）需自行验证。",
    ),
    LitAnchor(
        anchor_id="2611.07137",
        measure="wall_model",
        claim=(
            "力评估的离散陷阱：当壁面不落在格点连线中点时，经典动量交换法（CME）产生显著力误差；"
            "插值补充 bounce-back（ISBB）破坏局部质量守恒，导致非物理压力抬升 —— 二者都会污染"
            "阻力验收。需 modified MEM + 质量守恒修正的 ISBB。"
        ),
        claim_kind=ClaimKind.QUALITATIVE,
        evidence=EvidenceSource.ABSTRACT_OA,
        quantities=(),
        conditions="阶梯化 Cartesian 边界；曲壁/斜壁；动量交换测力；ISBB",
        status=AnchorStatus.PENDING_REDTEAM,
        cross_refs=("10.1063/5.0283930",),
    ),
)


ABSENT_ITEMS: tuple[AbsentItem, ...] = (
    AbsentItem(
        item="waLBerla / Palabos 在 1e4–1e5 核上的弱/强扩展效率具体百分比",
        measure="parallel_lbm",
        reason="Crossref 记录不含摘要，正文不可离线核验；本次未做全文获取",
        consequence="并行实践只引用架构级结论（块结构+幽灵层），不得引用任何效率数字",
    ),
    AbsentItem(
        item="LES-LBM 的 Smagorinsky 常数推荐区间及其在壁面约束高 Re 流动上的误差量化",
        measure="les",
        reason="本次探针未检索到开放获取的定量结论（摘要级）",
        consequence="Cs 只能沿用本仓既有取值（0.05/0.1 族）并标注为本仓默认，不作为文献锚",
    ),
    AbsentItem(
        item="OpenFOAM 对 SUBOFF 的网格收敛阶与 GCI 数值",
        measure="large_grid",
        reason="本次 arXiv/Crossref 探针未命中带数值的开放摘要",
        consequence="‘成熟代码网格收敛数字’只能用本仓归档基准反算的 p≈2 与文献实践描述对照，并如实标注",
    ),
    AbsentItem(
        item="SUBOFF AFF-8 总阻力系数 0.004 的原始面积/密度约定",
        measure="large_grid",
        reason="一手实验报告（Liu & Huang 1998 Table 14、Groves 1989）在仓内以转录入库，原文不可离线核验",
        consequence="0.004 判为 cross-inconsistent（见 suboff_anchor），不进打分器；改用 Table 14 量纲值自算 CT",
    ),
    AbsentItem(
        item="本仓在 Re=2000 上的全新三网格收敛序列（Richardson/GCI 实测）",
        measure="large_grid",
        reason=(
            "本机会话无 GPU、单次 shell 时限内无法完成时间收敛的三网格序列；集群节点 torch 缺 "
            "libflexiblas 不可用（实测 122 ms/step @L=20, 32 线程）"
        ),
        consequence="本单不给新收敛阶；网格收敛结论只引用归档 verified 基准的反算结果（标注为审计）",
    ),
)


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AnchorTableReport:
    n_anchors: int
    per_measure: Mapping[str, int]
    n_numeric: int
    n_absent: int
    n_scorer_candidates: int
    n_graders: int
    violations: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.violations


_CITATIONS: dict[str, Citation] = {a.anchor_id: _cite(a.anchor_id) for a in LITERATURE_ANCHORS}


def anchor_index() -> dict[str, LitAnchor]:
    return {a.anchor_id: a for a in LITERATURE_ANCHORS}


def validate_anchor_table(*, min_anchors: int = 5) -> AnchorTableReport:
    """Machine gate on the anchor table (structure + provenance + honesty)."""
    violations: list[str] = []
    store = load_record_store()
    per_measure: dict[str, int] = {k: 0 for k in MEASURES}
    seen: set[str] = set()
    n_numeric = n_scorer = 0

    for a in LITERATURE_ANCHORS:
        tag = f"[{a.anchor_id}]"
        if a.anchor_id in seen:
            violations.append(f"{tag} duplicate anchor id")
        seen.add(a.anchor_id)
        if a.measure not in MEASURES:
            violations.append(f"{tag} unknown measure {a.measure!r}")
        else:
            per_measure[a.measure] += 1
        if a.anchor_id not in store:
            violations.append(f"{tag} no archived bibliographic record")
        if not a.claim.strip():
            violations.append(f"{tag} empty claim")
        if not a.conditions.strip():
            violations.append(f"{tag} empty applicability conditions")
        if not a.cross_refs and a.claim_kind is ClaimKind.NUMERIC:
            violations.append(f"{tag} numeric claim without a cross-reference partner")
        for ref in a.cross_refs:
            if ref not in {x.anchor_id for x in LITERATURE_ANCHORS}:
                violations.append(f"{tag} dangling cross_ref {ref!r}")
        if a.claim_kind is ClaimKind.NUMERIC:
            n_numeric += 1
            if not a.quantities:
                violations.append(f"{tag} numeric claim carries no quantity")
            if a.evidence not in (EvidenceSource.ABSTRACT_OA, EvidenceSource.TITLE):
                violations.append(
                    f"{tag} numeric claim not readable offline "
                    f"(evidence={a.evidence.value})"
                )
            for q in a.quantities:
                if not q.unit:
                    violations.append(f"{tag} quantity {q.name} has no unit")
                if not isinstance(q.value, float) or q.value == 0.0:
                    violations.append(f"{tag} quantity {q.name} has no usable value")
        if a.status is AnchorStatus.SCORER_CANDIDATE:
            n_scorer += 1
            if a.claim_kind is not ClaimKind.NUMERIC:
                violations.append(f"{tag} scorer candidate must carry a numeric claim")
        if a.citation.retrieved_at and not a.citation.retrieved_at.endswith("Z"):
            violations.append(f"{tag} retrieval timestamp not UTC-ISO")

    if len(LITERATURE_ANCHORS) < min_anchors:
        violations.append(f"only {len(LITERATURE_ANCHORS)} anchors, need >= {min_anchors}")
    for measure, count in per_measure.items():
        if count < 2:
            violations.append(f"measure {measure!r} has {count} anchors, need >= 2")
    try:  # the scoring layer also carries the SUBOFF ladder candidates
        from tensorlbm.external_survey.suboff_anchor import scorer_candidates

        ladder_admitted = sum(
            1 for c in scorer_candidates() if str(c["verdict"]).startswith("admit_to_scorer")
        )
    except Exception:  # pragma: no cover - defensive only
        ladder_admitted = 0
    if n_scorer + ladder_admitted < 1:
        violations.append("no scorer candidate at all (anchor table and SUBOFF ladder both empty)")
    titles = [a.citation.title for a in LITERATURE_ANCHORS]
    if len(set(titles)) != len(titles):
        violations.append("duplicate titles in anchor table")

    for absent in ABSENT_ITEMS:
        if not absent.reason.strip() or not absent.consequence.strip():
            violations.append(f"absent item {absent.item[:40]!r} lacks reason/consequence")
        if absent.measure not in MEASURES:
            violations.append(f"absent item {absent.item[:40]!r} has unknown measure")

    return AnchorTableReport(
        n_anchors=len(LITERATURE_ANCHORS),
        per_measure=per_measure,
        n_numeric=n_numeric,
        n_absent=len(ABSENT_ITEMS),
        n_scorer_candidates=n_scorer,
        n_graders=len(MEASURES),
        violations=tuple(violations),
    )