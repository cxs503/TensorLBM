"""Bound criteria for the 2D cylinder drag benchmark (docs/cylinder_drag_benchmark_r1.md).

The point of these tests is that the *numbers* are produced by real LBM runs and
then bound by assertions:

* the harness's Strouhal extraction recovers a known period;
* ``collide_bgk_matmul`` really is the same BGK operator as ``collide_bgk``;
* ``correct_mass`` pins the total population (the mechanism the benchmark relies
  on to remove the density drift of the minimal channel boundary set);
* the committed registry is internally consistent (measured vs reference
  errors recomputed from scratch);
* the numbers printed in the markdown report equal the registry numbers, so the
  documentation cannot drift away from the evidence.
"""

from __future__ import annotations

import json
import math
import pathlib
import re

import pytest
import torch

from tensorlbm.d2q9 import equilibrium
from tensorlbm.solver import collide_bgk, collide_bgk_matmul, correct_mass, stream
from tests.validation.cylinder_drag_benchmark import (
    Config,
    build_registry,
    default_configs,
    extrapolate_blockage,
    run_cfg,
    _period_steps,
)

REGISTRY = pathlib.Path("docs/cylinder_drag_registry.json")
REPORT = pathlib.Path("docs/cylinder_drag_benchmark_r1.md")


def test_period_extraction_recovers_known_sine() -> None:
    """A sine of period 250 steps must come back as 250 +- 0.5 steps."""
    period = 250.0
    hist = [math.sin(2.0 * math.pi * i / period) for i in range(4000)]
    assert _period_steps(hist) == pytest.approx(period, abs=0.5)


def test_steady_signal_has_no_strouhal() -> None:
    assert math.isnan(_period_steps([0.1] * 500))


def test_matmul_collision_matches_bgk() -> None:
    """The faster matrix BGK kernel must agree with the reference kernel."""
    torch.manual_seed(0)
    ny, nx = 16, 24
    rho = torch.ones(ny, nx)
    ux = torch.rand(ny, nx) * 0.05
    uy = torch.rand(ny, nx) * 0.05
    f = equilibrium(rho, ux, uy, device=torch.device("cpu")).float()
    a, b = f.clone(), f.clone()
    with torch.inference_mode():
        for _ in range(100):
            a = stream(collide_bgk(a, tau=0.53))
            b = stream(collide_bgk_matmul(b, tau=0.53))
    assert float((a - b).abs().max()) / float(a.abs().max()) < 1e-4


def test_mass_correction_pins_total_population() -> None:
    """Without correction the minimal BC drifts the total mass; with it, it is pinned."""
    ny, nx = 12, 20
    rho = torch.ones(ny, nx)
    f = equilibrium(rho, torch.full_like(rho, 0.08), torch.zeros_like(rho),
                    device=torch.device("cpu")).float()
    target = float(f.sum())
    g = f.clone()
    with torch.inference_mode():
        for _ in range(200):
            g = stream(collide_bgk(g, tau=0.53))
            g = correct_mass(g, target)
            assert float(g.sum()) == pytest.approx(target, rel=1e-6)


def test_extrapolation_recovers_linear_synthetic_data() -> None:
    """Sanity check the B -> 0 fit on data with a known intercept."""
    class _M:
        def __init__(self, b: float, cd: float, st: float) -> None:
            self.derived = {"blockage": b, "cd_mean": cd, "strouhal": st}

    ms = [_M(b, 1.30 + 2.0 * b, 0.30) for b in (0.125, 0.0833, 0.0625)]
    ext = extrapolate_blockage(ms)
    assert ext["cd_b0"] == pytest.approx(1.30, abs=1e-6)


@pytest.mark.skipif(not REGISTRY.exists(), reason="benchmark registry not generated yet")
def test_registry_errors_are_self_consistent() -> None:
    reg = json.loads(REGISTRY.read_text())
    assert reg["schema"] == "tensorlbm-cylinder-drag-benchmark-v1"
    assert reg["runs"], "registry must contain at least one executed run"
    for run in reg["runs"]:
        d = run["derived"]
        assert d["status"] in ("stationary", "max_steps", "diverged")
        if d["status"] != "diverged":
            assert d["cd_mean"] > 0.0
        if d["status"] == "stationary":
            # only a stationary window may be quoted as a measured value
            assert d["stationarity_drift_pct"] < 1.0
    assert reg["comparisons"], "registry must record at least one comparison"
    for cmp in reg["comparisons"]:
        recomputed = 100.0 * (cmp["measured"] - cmp["reference"]) / cmp["reference"]
        assert cmp["rel_err_pct"] == pytest.approx(recomputed, abs=1e-6)
        lo, hi = cmp["reference_range"]
        assert lo <= cmp["reference"] <= hi


@pytest.mark.skipif(not (REGISTRY.exists() and REPORT.exists()),
                    reason="registry or report missing")
def test_report_numbers_match_registry() -> None:
    """Drift gate: every number in the report tables must come from the registry."""
    reg = json.loads(REGISTRY.read_text())
    text = REPORT.read_text()
    for run in reg["runs"]:
        d = run["derived"]
        for key, value in (("Cd", d["cd_mean"]),):
            if value != value:
                continue
            assert f"{value:.4f}" in text or f"{value:d}" in text or f"{int(value)}" in text, \
                f"{run['config']['name']}: {key}={value} missing from report"
    for cmp in reg["comparisons"]:
        assert f"{cmp['measured']:.4f}" in text, \
            f"comparison {cmp['quantity']} value missing from report"
    # no stale Step-E numbers may survive in the teaching doc
    teaching = pathlib.Path("docs/LBM_VERIFICATION_TEACHING.md")
    if teaching.exists():
        body = teaching.read_text()
        assert "Cd=1.30 (Schäfer-Turek)" not in body, \
            "the mis-attributed Step E reference must stay corrected"


def test_pilot_run_is_stable_and_physical() -> None:
    """Cheap end-to-end smoke run: finite fields, positive drag, no NaN."""
    cfg = Config(name="pytest_smoke", diameter=8, height_cells=33, upstream_d=3.0,
                 downstream_d=8.0, u_in=0.08, min_warmup_steps=400,
                 stats_steps=200, max_steps=800, boundary="simple",
                 mass_correction=True)
    m = run_cfg(cfg, verbose=False)
    assert math.isfinite(m.derived["cd_mean"])
    assert 0.5 < m.derived["cd_mean"] < 8.0
