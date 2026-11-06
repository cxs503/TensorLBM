"""Acceptance tests for the Couette-Poiseuille mixed-driven channel tutorial.

The tutorial (``docs/benchmarks/tutorials/couette_poiseuille.md``) ships a
runnable case (``benchmarks/verified/couette_poiseuille/run.py``) whose physics
claim is the analytic superposition

    u(y) = g/(2 nu) * y * (H - y) + u_wall * y / H

for a body-force driven channel with a sliding top plate.  These tests are the
machine-checkable form of the tutorial's assertions:

* the documented gates (profile L2, off-centre peak position, moving-wall
  velocity, mean velocity, fluid-mass drift, steady state) pass on a fast grid;
* the sliding-wall term is real physics, not decoration -- the mean velocity
  must differ from the pure-Poiseuille mean by the Couette contribution;
* the ``g = 0`` branch reproduces the linear Couette profile;
* the archived ``result.json`` still satisfies the gates and the numbers quoted
  in the tutorial are the archived ones (no hand-copied drift);
* ``README.md`` and the tutorials index link the new tutorial.

Reference: Guo, Zheng & Shi (2002) *Discrete lattice effects on the forcing term
in the lattice Boltzmann method*, Phys. Rev. E 65, 046308.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUN_PY = REPO / "benchmarks" / "verified" / "couette_poiseuille" / "run.py"
RESULT_JSON = REPO / "benchmarks" / "verified" / "couette_poiseuille" / "result.json"
DOC = REPO / "docs" / "benchmarks" / "tutorials" / "couette_poiseuille.md"
DEMO = REPO / "docs" / "benchmarks" / "demos" / "couette_poiseuille_demo.py"
TUTORIAL_INDEX = REPO / "docs" / "benchmarks" / "tutorials" / "README.md"

# Fast grid used by the tests (archived case is ny=34 / 12000 steps; this one
# keeps the same physics at ~5 s).
FAST_NY = 18
FAST_STEPS = 3000


def _load_case_module():
    spec = importlib.util.spec_from_file_location("couette_poiseuille_run_test", RUN_PY)
    assert spec is not None and spec.loader is not None, f"cannot load {RUN_PY}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def case():
    return _load_case_module()


@pytest.fixture(scope="module")
def mixed(case):
    """Mixed Couette-Poiseuille run on the fast grid."""
    return case.run_case(ny=FAST_NY, steps=FAST_STEPS)


@pytest.fixture(scope="module")
def couette(case):
    """Pure-Couette control (g = 0) on the same grid."""
    return case.run_case(ny=FAST_NY, steps=FAST_STEPS, accel=0.0)


@pytest.fixture(scope="module")
def archived():
    assert RESULT_JSON.exists(), f"missing archived result {RESULT_JSON}"
    return json.loads(RESULT_JSON.read_text())


# ---------------------------------------------------------------------------
# 1. documented gates
# ---------------------------------------------------------------------------


@pytest.mark.physics
@pytest.mark.benchmark
def test_documented_gates_pass(mixed):
    """Every gate printed by the tutorial must hold on the fast grid."""
    failed = {name: c for name, c in mixed["checks"].items() if not c["ok"]}
    assert not failed, f"tutorial gates failed: {failed}"
    assert mixed["all_ok"] is True


@pytest.mark.physics
@pytest.mark.benchmark
def test_profile_l2_below_one_percent(mixed):
    """Tighter than the 2 % gate: the fast grid already reaches <1 %."""
    assert mixed["metrics"]["profile_l2"] < 0.01, mixed["metrics"]["profile_l2"]
    assert mixed["metrics"]["profile_l2"] > 0.0


@pytest.mark.physics
def test_peak_is_off_centre(mixed):
    """The Couette shear pushes the extremum above mid-channel (0.5 < y_v/H < 1)."""
    p, m = mixed["params"], mixed["metrics"]
    ratio = m["peak_y_star"] / p["H"]
    assert 0.5 < ratio <= 1.0, ratio
    # analytic vertex y_v/H = 1/2 + nu*u_wall/(g*H^2) = 3/4 for the self-similar drive
    analytic = m["peak_y_star_exact"] / p["H"]
    assert analytic == pytest.approx(0.75, abs=1e-6)
    assert ratio == pytest.approx(analytic, abs=0.05)
    # and the numerical profile is unimodal around the peak
    u = mixed["profile"]["u_num"]
    peak = int(max(range(len(u)), key=lambda i: u[i]))
    assert peak > 0 and peak < len(u) - 1, "extremum must sit inside the channel"
    assert all(u[i] < u[i + 1] for i in range(peak)), "profile not rising before the peak"
    assert all(u[i] > u[i + 1] for i in range(peak, len(u) - 1)), "profile not falling after"


@pytest.mark.physics
def test_moving_wall_velocity_is_imposed(mixed):
    """u at the top fluid node matches the analytic value (Ladd momentum transfer)."""
    m = mixed["metrics"]
    assert m["moving_wall_uc_rel_err"] < 0.03
    assert m["u_wall_node"] == pytest.approx(m["u_wall_node_exact"], rel=0.03)
    # the plate really drags: the node below the plate moves at ~u_wall already
    assert m["u_wall_node"] > 0.95 * mixed["params"]["u_wall"]


@pytest.mark.physics
def test_stationary_wall_no_slip(mixed):
    """The fixed wall node sits on the analytic value (no-slip half-way BB)."""
    m = mixed["metrics"]
    assert m["u_bottom_node"] == pytest.approx(m["u_bottom_node_exact"], rel=0.05)
    assert m["bottom_node_rel_err"] < 0.01


@pytest.mark.physics
def test_mean_velocity_superposition(mixed):
    """ubar = g H^2/(12 nu) + u_wall/2 and the Couette half is genuinely present."""
    p, m = mixed["params"], mixed["metrics"]
    poiseuille_only = p["accel"] * p["H"] ** 2 / (12.0 * p["nu"])
    assert m["ubar"] == pytest.approx(poiseuille_only + 0.5 * p["u_wall"], rel=0.02)
    # a solver that silently dropped the moving wall would report ~poiseuille_only
    assert m["ubar"] - poiseuille_only == pytest.approx(0.5 * p["u_wall"], rel=0.05)


@pytest.mark.physics
def test_mass_conserved_and_steady(mixed):
    """Closed channel: fluid mass drift is float32 round-off, profile has settled."""
    m = mixed["metrics"]
    assert m["mass_drift"] < 1e-3, m["mass_drift"]  # measured ~1e-5
    assert m["steady_change_rel"] < 1e-3, m["steady_change_rel"]


@pytest.mark.physics
def test_pure_couette_control(couette):
    """g = 0 reproduces the linear Couette profile and puts the peak at the plate."""
    p, m = couette["params"], couette["metrics"]
    assert p["accel"] == 0.0
    assert couette["all_ok"] is True
    assert m["profile_l2"] < 0.01, m["profile_l2"]
    u = couette["profile"]["u_num"]
    assert u == sorted(u), "pure Couette profile must be monotone increasing"
    assert m["peak_y_star"] == pytest.approx(p["H"] - 0.5, abs=1e-9)
    # linearity: the mid-channel velocity is half the plate velocity
    mid = u[len(u) // 2]
    assert mid == pytest.approx(0.5 * p["u_wall"], rel=0.1)


@pytest.mark.unit
def test_input_validation(case):
    with pytest.raises(ValueError):
        case.run_case(ny=4, steps=1)
    with pytest.raises(ValueError):
        case.run_case(tau=0.5, steps=1)


# ---------------------------------------------------------------------------
# 2. archived artefact + documentation wiring
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_archived_result_meets_gates(archived):
    assert archived["case"] == "couette_poiseuille"
    assert archived["all_ok"] is True
    for name, check in archived["checks"].items():
        assert check["ok"] is True, f"archived gate {name} not ok: {check}"
    assert archived["params"]["ny"] == 34 and archived["params"]["steps"] == 12000
    prof = archived["profile"]
    assert len(prof["u_num"]) == len(prof["u_exact"]) == len(prof["y_star"]) == 32


@pytest.mark.regression
def test_tutorial_quotes_archived_numbers(archived):
    """The tutorial numbers must be the archived ones (no hand-copied drift).

    Two anchors are parsed strictly, so editing either the abstract headline or
    the results table without re-running the case turns this red; the remaining
    figures only have to be present.
    """
    text = DOC.read_text()
    m, p = archived["metrics"], archived["params"]

    abstract = re.search(r"剖面 L2 相对误差 \*\*([0-9.]+e[-+]?[0-9]+)\*\*", text)
    assert abstract, "abstract headline L2 figure not found"
    assert float(abstract.group(1)) == pytest.approx(m["profile_l2"], rel=1e-3)

    table = re.search(r"\| 壁法向剖面 L2 相对误差（32 点） \| ([0-9.]+e[-+]?[0-9]+) \|", text)
    assert table, "table-1 L2 row not found"
    assert float(table.group(1)) == pytest.approx(m["profile_l2"], rel=1e-3)

    moving_wall = re.search(r"\| 动壁节点速度（数值 / 解析） \| 0\.[0-9]+ / 0\.[0-9]+ = ([0-9.]+)%", text)
    assert moving_wall, "table-1 moving-wall row not found"
    assert float(moving_wall.group(1)) == pytest.approx(100.0 * m["moving_wall_uc_rel_err"], abs=0.01)

    expected = [
        f"{m['peak_pos_rel_err']:.2%}",
        f"{m['ubar_rel_err']:.2%}",
        f"{m['mass_drift']:.1e}",
        f"{m['u_wall_node']:.4f}",
        f"{m['peak_y_star_exact']:.1f}",
        f"{p['Re']:.1f}",
        f"{p['Ma']:.3f}",
    ]
    missing = [e for e in expected if e not in text]
    assert not missing, f"tutorial does not quote {missing}"


@pytest.mark.unit
def test_tutorial_has_runnable_code_and_assertions():
    text = DOC.read_text()
    assert "```python" in text, "tutorial must contain a runnable python block"
    assert "assert" in text, "tutorial must state its assertions"
    for gate in ("profile_l2", "peak_position", "moving_wall_uc", "mean_velocity"):
        assert gate in text, f"gate {gate} not documented"
    assert DEMO.exists(), f"missing demo script {DEMO}"


@pytest.mark.unit
def test_readme_links_tutorial():
    readme = (REPO / "README.md").read_text()
    assert "docs/benchmarks/tutorials/couette_poiseuille.md" in readme
    index = TUTORIAL_INDEX.read_text()
    assert "tutorials/couette_poiseuille.md" in index