"""Benchmark anchors for the three turbulence-validation cases.

Every number used as a comparison target is declared here *once*, with its
source, so the campaign never hand-copies a reference value.  Channel-flow
anchors are not duplicated: they come from the shipped Moser-Kim-Mansour (1999)
profile files (:mod:`tensorlbm.validation.mkm1999`, SHA-256 verified), which are
the same tables used by the wall-bounded-turbulence line of this repository.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "BFS_EXPERIMENT",
    "CYLINDER_STROUHAL",
    "BenchmarkValue",
    "bfs_reference",
    "cylinder_reference",
]


@dataclass(frozen=True)
class BenchmarkValue:
    """A reference value with an explicit source and tolerance."""

    value: float
    source: str
    tolerance: float
    note: str = ""


# ---------------------------------------------------------------------------
# Backward-facing step: primary reattachment length x_R / h
# ---------------------------------------------------------------------------
# Geometry of Armaly et al. (1983): upstream channel height H1 = 5.2 mm,
# step height h = 4.9 mm, expansion ratio ER = (H1 + h)/H1 = 1.9423,
# Re = U_mean * 2 H1 / nu.

BFS_EXPERIMENT: dict[int, BenchmarkValue] = {
    100: BenchmarkValue(
        3.00,
        "Armaly, Durst, Pereira & Schönung (1983) JFM 127:473, expansion ratio 1.9423",
        0.15,
        "experiment x_R/h = 2.9-3.1; 2D computations (Kim & Moin 1985) 2.92-3.0",
    ),
    200: BenchmarkValue(
        4.90,
        "Armaly et al. (1983) JFM 127:473; 2D computation Gartling (1990) IJNMF 11:953",
        0.15,
        "agreement between 2D computation and experiment is within a few percent up to Re~400",
    ),
    389: BenchmarkValue(
        7.90,
        "Armaly et al. (1983) JFM 127:473, Re = 389 station digitised in Kim & Moin (1985)",
        0.15,
        "last Reynolds number before the experiment becomes fully three-dimensional",
    ),
}


def bfs_reference(re: int) -> BenchmarkValue:
    key = int(round(re))
    if key not in BFS_EXPERIMENT:
        raise KeyError(f"no backward-facing-step anchor for Re={re}")
    return BFS_EXPERIMENT[key]


# ---------------------------------------------------------------------------
# Circular cylinder: Strouhal number
# ---------------------------------------------------------------------------
# Two anchors per Reynolds number: the three-dimensional *experiment* (what the
# node criterion asks for) and the two-dimensional DNS of Williamson & Brown
# (1998), which is the physically correct target for a 2D solver.  Reporting
# both is what makes the 2D restriction of the solver auditable instead of
# hidden.

CYLINDER_STROUHAL: dict[int, dict[str, BenchmarkValue]] = {
    100: {
        "experiment": BenchmarkValue(
            0.164,
            "Roshko (1954) NACA TN 3169; Williamson (1996) Annu. Rev. Fluid Mech. 28:477",
            0.05,
            "2D DNS 0.1645 (Williamson & Brown 1998) - 2D and 3D agree at Re=100",
        ),
        "dns_2d": BenchmarkValue(
            0.1645,
            "Williamson & Brown (1998) Phys. Fluids 10:2039 (2D DNS)",
            0.05,
        ),
    },
    200: {
        "experiment": BenchmarkValue(
            0.185,
            "Williamson (1996) Annu. Rev. Fluid Mech. 28:477 (3D experiment)",
            0.05,
            "3D experiments and 2D DNS genuinely differ here (2D DNS 0.197)",
        ),
        "dns_2d": BenchmarkValue(
            0.197,
            "Williamson & Brown (1998) Phys. Fluids 10:2039 (2D DNS)",
            0.05,
        ),
    },
    300: {
        "experiment": BenchmarkValue(
            0.203,
            "Norberg (2003) J. Fluids Struct. 17:57; Fey, König & Eckelmann (1998)",
            0.05,
            "2D DNS 0.212; the 2D/3D gap stays at the few-percent level",
        ),
        "dns_2d": BenchmarkValue(
            0.212,
            "Williamson & Brown (1998) Phys. Fluids 10:2039 (2D DNS)",
            0.05,
        ),
    },
}


def cylinder_reference(re: int, kind: str = "experiment") -> BenchmarkValue:
    key = int(round(re))
    if key not in CYLINDER_STROUHAL:
        raise KeyError(f"no cylinder anchor for Re={re}")
    return CYLINDER_STROUHAL[key][kind]