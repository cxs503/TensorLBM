"""Regression guard for the Poiseuille wall-treatment convergence ladder.

Defect being guarded (AUDIT-RUNTEST-6, 2026-11-11): in
``benchmarks/verified/poiseuille_convergence_order/run.py`` the three declared
wall arms ``halfway`` / ``effective`` / ``bouzidi`` all set ``q_wall = 0.5`` and
only the ``bouzidi`` label reached the interpolated kernel -- which at
``q = 0.5`` is algebraically identical to half-way bounce-back -- while
``effective`` fell through to the same ``apply_halfway_bb`` call.  The published
three-arm comparison was therefore a single computation reported three times,
bit-identical to the last digit.

The invariant this test locks:

* the arms must be *distinct computations* (their L2 errors must differ), and
* each arm must show the convergence order its physics predicts
  (half-way at the mid-point: p ~ 2; mid-point BC on a genuinely sub-cell wall:
  p ~ 1; the "effective wall position" reference: p ~ 2 again; interpolated
  Bouzidi at the sub-cell wall: p >= 1.8).

Bands are declared from theory (p = 1 or p = 2 with a generous tolerance), not
from fitting the observed values.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUN_PY = REPO / "benchmarks/verified/poiseuille_convergence_order/run.py"

H_GRIDS = (6, 12)  # refinement ratio exactly 2 (H_true is identical for all arms)
STEPS = 1200
TAU = 0.8
U_MAX = 0.04


@pytest.fixture(scope="module")
def arms():
    if not RUN_PY.exists():  # pragma: no cover - benchmark moved
        pytest.skip(f"{RUN_PY} missing")
    spec = importlib.util.spec_from_file_location("_poiseuille_conv_study", RUN_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "ARMS"):  # pragma: no cover - pre-fix baseline
        pytest.fail(
            "poiseuille_convergence_order/run.py has no ARMS table: the wall arms "
            "are not parameterised (AUDIT-RUNTEST-6 defect)"
        )
    out = {}
    for arm in mod.ARMS:
        errs = [mod.run_poiseuille_case(H, TAU, U_MAX, arm, STEPS)["l2_rel"] for H in H_GRIDS]
        out[arm] = {"errs": errs, "p": math.log(errs[0] / errs[1]) / math.log(2.0)}
    return mod, out


def test_arms_are_distinct_computations(arms):
    """The three-way comparison is only meaningful if the arms differ."""
    _, res = arms
    first_grid = {k: v["errs"][0] for k, v in res.items()}
    names = sorted(first_grid)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ea, eb = first_grid[a], first_grid[b]
            rel = abs(ea - eb) / max(ea, eb)
            assert rel > 1e-3, (
                f"arms {a!r} and {b!r} are numerically identical (l2_rel {ea:.6e} vs "
                f"{eb:.6e}, rel diff {rel:.2e}) -- the wall-model switch is a no-op"
            )


def test_halfway_at_midpoint_is_second_order(arms):
    _, res = arms
    p = res["halfway"]["p"]
    assert 1.7 < p < 2.3, f"half-way BB at q=0.5 delta: expected p~2, got {p:.3f}"


def test_midpoint_bc_on_subcell_wall_is_first_order(arms):
    """The wall-position error the benchmark exists to expose: p ~ 1."""
    _, res = arms
    p = res["subcell_halfway"]["p"]
    assert 0.7 < p < 1.4, f"mid-point BC on a q=0.25 delta wall: expected p~1, got {p:.3f}"


def test_effective_wall_reference_recovers_second_order(arms):
    """Same flow as above, scored against the effective wall -> O(delta^2)."""
    _, res = arms
    p = res["effective"]["p"]
    assert 1.7 < p < 2.4, f"effective-wall reference: expected p~2, got {p:.3f}"


def test_bouzidi_subcell_wall_is_second_order(arms):
    _, res = arms
    p = res["bouzidi"]["p"]
    assert p > 1.8, f"Bouzidi interpolated BB at q=0.25 delta: expected p>=1.8, got {p:.3f}"