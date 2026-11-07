"""Five-measure PLAYBOOK distilled from the IN25-T6 survey (absorb-then-exceed).

Each entry is written so that a weaker model can execute it without re-reading
the papers: an action, a first concrete step, the trap that historically eats
the work, an acceptance threshold, and the anchors that license it.

:func:`validate_playbook` is the machine gate: at least five entries, one per
measure, every cited anchor must exist in
:mod:`tensorlbm.external_survey.lit_anchors` or be a SUBOFF ladder candidate, and
every acceptance criterion must carry a number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from tensorlbm.external_survey.lit_anchors import MEASURES, anchor_index

__all__ = ["PlaybookEntry", "PLAYBOOK", "validate_playbook", "PlaybookReport"]

_LADDER_ANCHORS = {"SUBOFF-AFF1-CT-LADDER", "SUBOFF-AFF8-CT-LADDER"}


@dataclass(frozen=True)
class PlaybookEntry:
    measure: str
    title: str
    action: str
    first_step: str
    trap: str
    acceptance: str
    anchors: tuple[str, ...]
    owner: str = "TensorLBM"

    @property
    def has_numeric_threshold(self) -> bool:
        return bool(re.search(r"\d", self.acceptance))


PLAYBOOK: tuple[PlaybookEntry, ...] = (
    PlaybookEntry(
        measure="large_grid",
        title="外流阻力判据改用 SUBOFF Table 14 量纲阶梯，禁用‘AFF-8 CT≈0.004’",
        action=(
            "以 Liu & Huang (1998) Table 14 的 6 个拖曳速度点换算 CT(Re) 为主判据；"
            "每个算例只与最近 Re 的阶梯点比较，并同时报出湿面积/密度约定。"
        ),
        first_step=(
            "python -c \"from tensorlbm.external_survey.suboff_anchor import "
            "scorer_candidates; print(scorer_candidates()[1]['points'])\" "
            "→ 把 points 写进新 benchmark 的 reference.json（含 S_wet 区间与 ρ）"
        ),
        trap=(
            "湿面积约定未定：S_wet 由 6.00→6.35 m² 使 CT 变 5.8%；ρ 由 999→1025.9 再变 2.6%；"
            "文献常引 0.004 需要 S_wet≈5.50 m²（小于裸体）才能成立，属不可复现约定。"
        ),
        acceptance="|CT_CFD − CT_EFD(最近 Re 点)| ≤ 10%（含面积区间）；< 3% 才可称‘定量对标成功’",
        anchors=("SUBOFF-AFF8-CT-LADDER", "SUBOFF-AFF1-CT-LADDER", "2503.19674"),
    ),
    PlaybookEntry(
        measure="parallel_lbm",
        title="并行扩展照抄块结构域分解 + 幽灵层，不先优化全局网格",
        action=(
            "把分布式层实现为「块结构 + 幽灵层交换 + 内核生成」三层（waLBerla/Palabos 架构）；"
            "通信按块而不是按格，块数 ≥ 节点数×每节点 socket 数。"
        ),
        first_step=(
            "量化一次 halo 交换的有效载荷比：Q=19 分布在 Q_PAD=32 上 → 有效率 19/32=59.4%，"
            "先把该浪费测出来再谈优化"
        ),
        trap=(
            "Q_PAD 对齐带来的无效 lane 传输；小网格上线程超订（本仓实测 L=16 时 64 线程比 16 线程慢 5 倍）；"
            "块数不足导致节点空转"
        ),
        acceptance="8 卡强扩展效率 ≥ 70%（本仓已入库的 8 卡公开数 83% 作为上限参照）",
        anchors=("10.1016/j.camwa.2020.01.007", "10.1016/j.camwa.2020.03.022",
                 "10.1145/2503210.2503273"),
    ),
    PlaybookEntry(
        measure="amr",
        title="局部加密只做 2:1 块嵌套 + 体积守恒界面，先在小算例上验界面",
        action=(
            "采用 2:1 嵌套块、时间步比 2 的亚循环，界面用守恒（体积）形式而非纯插值；"
            "加密只布置在壁面/尾流压力梯度带。"
        ),
        first_step=(
            "以「充分发展槽道 + 绕球」两算例做 AMR/均匀细网格对照，记录界面质量残差与 CD 差"
        ),
        trap=(
            "插值型界面破坏局部质量守恒 → 非物理压力抬升（与 ISBB 泄漏同源，见 2611.07137）；"
            "细化比不是 2 时需额外时间插值，误差阶掉到 1"
        ),
        acceptance="同 Re 下 AMR 与均匀细网格 CD 差 < 2%，界面质量残差 < 1e-6/步",
        anchors=("10.1006/jcph.1998.6089", "10.1016/j.jcp.2012.03.015",
                 "10.1103/physreve.94.053311", "2611.07137"),
    ),
    PlaybookEntry(
        measure="les",
        title="高 Re 稳定性先换松弛率谱（MRT/熵），Cs 只做细调并标注为工程默认",
        action=(
            "稳定性预算放在 MRT 松弛率谱/熵碰撞上；Smagorinsky 只用来在 τ 上加涡粘，"
            "并对速度做 ±c 限幅防止非平衡应力发散。"
        ),
        first_step=(
            "固定 Re 扫 Cs ∈ {0.05, 0.10, 0.14, 0.16}，记录发散阈值与 CD 漂移（用同一网格、同一步数）"
        ),
        trap=(
            "Cs 与网格强耦合：加密等同改变有效 Re；MRT 参数选错会过度耗散、抑制湍流；"
            "文献级 Cs 推荐区间在本次核验中 absent，本仓 0.05/0.1 属工程默认而非文献锚"
        ),
        acceptance="Cs 扫描区间内 CD 变化 < 5% 且无发散；否则判‘Cs 未定标’，不得报定量结论",
        anchors=("10.1090/fic/006/12", "10.1103/physreve.61.6546"),
    ),
    PlaybookEntry(
        measure="wall_model",
        title="近壁优先 slip-velocity bounce-back；力评估用 modified MEM + 质量守恒校验",
        action=(
            "高 Re 壁面用 slip-velocity BB 耦合模型一致壁面函数；避免正则化/高阶重构壁法向梯度；"
            "曲壁测力改用按真实边界位置修正的动量交换，并每步校验总质量。"
        ),
        first_step=(
            "同一 Re 下跑 BB vs 正则化两版，比较沿程摩阻分布与三网格收敛一致性；同时打印总质量漂移"
        ),
        trap=(
            "壁面不在格点中点 → 经典动量交换力误差显著；ISBB 破坏局部质量守恒 → 压力漂移；"
            "三阶重构对壁法向梯度高度敏感（平壁也敏感）"
        ),
        acceptance="y+ ∈ [30,300] 时摩阻误差 < 10%，加密时摩阻单调收敛且质量漂移 < 1e-6/步",
        anchors=("10.1063/5.0283930", "2601.18443", "2611.07137", "10.2514/3.61193"),
    ),
)


@dataclass(frozen=True)
class PlaybookReport:
    n_entries: int
    measures_covered: tuple[str, ...]
    violations: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.violations


def validate_playbook(*, min_entries: int = 5) -> PlaybookReport:
    violations: list[str] = []
    idx = anchor_index()
    covered: list[str] = []

    for e in PLAYBOOK:
        tag = f"[{e.measure}]"
        if e.measure not in MEASURES:
            violations.append(f"{tag} unknown measure")
        if e.measure not in covered:
            covered.append(e.measure)
        for field_name in ("action", "first_step", "trap", "acceptance"):
            if not getattr(e, field_name).strip():
                violations.append(f"{tag} empty {field_name}")
        if not e.has_numeric_threshold:
            violations.append(f"{tag} acceptance has no numeric threshold")
        if not e.anchors:
            violations.append(f"{tag} cites no anchor")
        for a in e.anchors:
            if a not in idx and a not in _LADDER_ANCHORS:
                violations.append(f"{tag} unknown anchor {a!r}")

    if len(PLAYBOOK) < min_entries:
        violations.append(f"only {len(PLAYBOOK)} playbook entries, need >= {min_entries}")
    missing = [m for m in MEASURES if m not in covered]
    if missing:
        violations.append(f"measures without a playbook entry: {missing}")

    return PlaybookReport(
        n_entries=len(PLAYBOOK),
        measures_covered=tuple(covered),
        violations=tuple(violations),
    )