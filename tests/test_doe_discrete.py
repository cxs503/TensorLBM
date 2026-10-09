"""Discrete-level full-factorial round-trip regression tests.

``full_factorial`` encodes each discrete level as its normalised index
``i / (n - 1)`` so that ``generate_doe``'s nearest-level back-mapping
``round(u * (n - 1))`` recovers the index exactly.  Encoding the linear
value instead silently collapsed non-uniformly spaced levels onto the
nearest uniform slot — log-spaced Re levels 50..800 triplicated at the
low end (found in the 2026-08-21 SUBOFF sweep launch).
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.doe import DoEVariable, generate_doe
from tensorlbm.scan_runner import ScanPlan, ScanVariable, git_code_sha

LOG_LEVELS = tuple(round(50.0 * (800.0 / 50.0) ** (i / 23.0), 1) for i in range(24))


def test_non_uniform_levels_round_trip_exactly() -> None:
    levels = [1.0, 2.0, 10.0, 100.0]
    plan = generate_doe([DoEVariable(name="re", levels=levels)], method="full_factorial")
    got = sorted(row["re"] for row in plan.design_matrix)
    assert got == sorted(levels)


def test_log_spaced_levels_survive_the_round_trip() -> None:
    plan = generate_doe([DoEVariable(name="re", levels=list(LOG_LEVELS))], method="full_factorial")
    got = sorted(row["re"] for row in plan.design_matrix)
    assert got == sorted(LOG_LEVELS)
    assert len(set(got)) == len(LOG_LEVELS)


def test_two_level_factorial_is_unchanged() -> None:
    plan = generate_doe([DoEVariable(name="re", levels=[100.0, 400.0])], method="full_factorial")
    got = sorted(row["re"] for row in plan.design_matrix)
    assert got == [100.0, 400.0]


def test_duplicate_levels_are_preserved() -> None:
    plan = generate_doe(
        [DoEVariable(name="re", levels=[100.0, 400.0, 100.0])], method="full_factorial"
    )
    got = sorted(row["re"] for row in plan.design_matrix)
    assert got == [100.0, 100.0, 400.0]


@pytest.mark.parametrize("n_levels", [3, 8, 24])
def test_scan_plan_points_match_declared_levels(n_levels: int) -> None:
    levels = list(np.geomspace(50.0, 800.0, n_levels))
    plan = ScanPlan.generate(
        scan_id="doe-levels-regression",
        case="cavity",
        variables=(ScanVariable(name="re", levels=levels),),
        method="full_factorial",
        n_points=n_levels,
        seed=0,
        steps=10,
        snapshot_every=5,
        code_sha=git_code_sha(),
    )
    got = sorted(p.params["re"] for p in plan.points)
    assert got == sorted(levels)
