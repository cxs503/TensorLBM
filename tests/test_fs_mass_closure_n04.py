"""Contract tests for the stage-resolved free-surface mass closure (IN13-W2-N04).

The closure ledger makes four claims and every one of them is pinned here:

* the three-fill-class segmentation is an *identity* (to round-off), not an
  approximation, and a dropped channel is visible in the closure residual;
* the attribution splits *interior* bookkeeping (which must not create liquid)
  from *boundary* reservoirs (ABB gas closure, gas zeroing), so a threshold
  means something different in the two groups;
* the per-step net change is a cancellation of larger opposing channel terms --
  the churn ratio -- which is exactly why a single scalar "mass drift" cannot
  replace the ledger;
* the hot record path performs no host synchronisation.

The last two tests run the real solver on a small closed tank; they are the
regression guard for the numbers quoted in
``docs/reports/report_fs_mass_closure_in13_w2_n04.md``.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.free_surface_lbm import (
    GAS,
    free_surface_step,
    init_flags_from_fill,
    init_mass_from_fill,
)
from tensorlbm.free_surface_mass_closure import (
    CANONICAL_CHANNELS,
    EXCHANGE_CHANNEL,
    ClosureLedger,
    class_mass_identity,
    front_coefficient,
)
from tensorlbm.free_surface_mass_ledger import refill_ablation

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _tank(nz: int, ny: int, nx: int) -> torch.Tensor:
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    solid[:, :, 0] = True
    solid[:, :, -1] = True
    return solid


def _column(nz: int = 2, ny: int = 10, nx: int = 14, width: int = 4, height: int = 6):
    solid = _tank(nz, ny, nx)
    fill = torch.zeros((nz, ny, nx))
    fill[:, 1 : 1 + height, 1 : 1 + width] = 1.0
    fill[solid] = 0.0
    flags = init_flags_from_fill(fill, solid)
    mass = init_mass_from_fill(fill, flags)
    rho = torch.where(flags == GAS, torch.full_like(fill, 1.0), torch.ones_like(fill))
    zero = torch.zeros_like(fill)
    return equilibrium3d(rho, zero, zero, zero), fill, flags, mass, solid


def _closed_loop(steps: int, *, defect: str = "none", gravity: float = 8e-5):
    """Run the real solver on a closed tank and return (closure summary, ledger)."""
    f, fill, flags, mass, solid = _column()
    closure = ClosureLedger(steps + 1, device=f.device)
    closure.set_initial(float((mass.sum() + (fill * (flags != GAS)).sum())))
    for _ in range(steps):
        previous_flags = flags.clone() if defect != "none" else None
        stage_ledger: dict = {}
        f, fill, flags, mass, _ = free_surface_step(
            f,
            fill,
            flags,
            solid,
            mass=mass,
            tau=0.8,
            gy=-gravity,
            rho_gas=1.0,
            inventory_reconciliation_ledger=stage_ledger,
        )
        if previous_flags is not None:
            mass, fill = refill_ablation(previous_flags, flags, mass, fill,
                                         include_demotion=defect == "refill_off_all")
        closure.record_step(
            stage_ledger["stage_deltas"],
            stage_ledger["observed_total_liquid_inventory_delta"],
            stage_ledger["stages"]["before_collision"]["total_liquid_inventory"],
        )
    return closure.summary(), closure


# ---------------------------------------------------------------------------
# identity + attribution algebra
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_class_identity_is_exact(seed):
    """{fill<=0} + {0<fill<1} + {fill>=1} == fill to float32 round-off."""
    generator = torch.Generator().manual_seed(seed)
    # include negative undershoot and oversaturation on purpose
    fill = (torch.rand((6, 7, 9), generator=generator) * 1.6) - 0.3
    identity = class_mass_identity(fill)
    assert float(identity["residual_ulps"]) <= 1.0
    assert float(identity["segmented"]) == pytest.approx(float(identity["total"]), rel=1e-6)


def test_closure_residual_detects_a_missing_channel():
    """A channel that is not attributed shows up as a closure residual."""
    ledger = ClosureLedger(2)
    ledger.set_initial(100.0)
    step_size = 1e-3  # 84 ulp of V0=100; anything smaller is invisible by design
    full = {name: {"total_liquid_inventory": step_size} for name in CANONICAL_CHANNELS}
    ledger.record_step(full, observed_delta=len(CANONICAL_CHANNELS) * step_size)
    summary = ledger.summary()
    # float64 accumulation only: the residual is round-off, not a leak
    assert abs(summary["closure_residual_max_abs"]) < 1e-18
    assert summary["closure_residual_max_ulps"] < 1e-6

    holed = dict(full)
    dropped = holed.pop(EXCHANGE_CHANNEL)
    ledger2 = ClosureLedger(1)
    ledger2.set_initial(100.0)
    ledger2.record_step(holed, observed_delta=len(CANONICAL_CHANNELS) * step_size)
    summary2 = ledger2.summary()
    assert summary2["closure_residual_max_abs"] == pytest.approx(
        dropped["total_liquid_inventory"], rel=1e-12
    )
    assert summary2["closure_residual_max_ulps"] > 1.0


def test_group_split_and_churn_ratio():
    """Interior and boundary groups are separated and churn >> net is reported."""
    ledger = ClosureLedger(4)
    ledger.set_initial(1000.0)
    for step in range(4):
        ledger.record_step(
            {
                "after_abb": {"total_liquid_inventory": 5e-3},
                "after_mass_exchange": {"total_liquid_inventory": -5e-3 + 1e-7},
            },
            observed_delta=1e-7,
        )
    summary = ledger.summary()
    # 4 steps x |5e-3| + |5e-3-1e-7| churn against 4 steps x 1e-7 net
    assert summary["churn_ratio"] == pytest.approx(1.0e5, rel=5e-2)
    assert summary["boundary_cumulative_delta"] == pytest.approx(2e-2, rel=1e-9)
    assert summary["interior_cumulative_delta"] == pytest.approx(-2e-2 + 4e-7, rel=1e-6)
    assert summary["exchange_steps_nonzero"] == 4
    assert summary["topology_steps"] == 0
    assert summary["quiet_steps"] == 4


def test_quiet_mask_excludes_conversion_steps():
    """Quiet statistics ignore the steps on which a conversion happened."""
    ledger = ClosureLedger(3)
    ledger.set_initial(1000.0)
    ledger.record_step({"after_mass_exchange": {"total_liquid_inventory": 1e-4}}, observed_delta=1e-4)
    ledger.record_step(
        {"after_mass_exchange": {"total_liquid_inventory": 1e-4},
         "after_topology_conversion": {"total_liquid_inventory": 1.0}},
        observed_delta=1.0 + 1e-4,
    )
    ledger.record_step({"after_mass_exchange": {"total_liquid_inventory": 2e-6}}, observed_delta=2e-6)
    summary = ledger.summary()
    assert summary["quiet_steps"] == 2
    assert summary["exchange_quiet_max_abs_step_delta"] == pytest.approx(1e-4)
    assert summary["topology_steps"] == 1


def test_record_step_does_not_synchronise_with_the_host(monkeypatch):
    """No ``.item()``/``.tolist()`` in the hot path when the total is supplied."""

    def _boom(*args, **kwargs):  # pragma: no cover - the failure is the assertion
        raise AssertionError("host synchronisation inside ClosureLedger.record_step")

    ledger = ClosureLedger(2)
    ledger.record_step({"after_abb": {"total_liquid_inventory": 1.0}}, observed_delta=1.0)
    monkeypatch.setattr(torch.Tensor, "item", _boom)
    monkeypatch.setattr(torch.Tensor, "tolist", _boom)
    ledger.record_step({"after_abb": {"total_liquid_inventory": 1.0}}, observed_delta=1.0)
    assert ledger.steps_recorded == 2


def test_front_coefficient_recovers_a_known_slope():
    """The estimator used for the dam-break anchor is unbiased on a synthetic line."""
    ts = [0.2 * i for i in range(1, 25)]
    zs = [1.2016 * t + 0.05 for t in ts]
    fit = front_coefficient(ts, zs, window=(0.5, 3.0))
    assert fit["slope"] == pytest.approx(1.2016, rel=1e-9)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)
    # points outside the window are dropped, not silently averaged in
    assert fit["samples"] == len([t for t in ts if 0.5 <= t <= 3.0])


# ---------------------------------------------------------------------------
# real solver: the regression guard for the campaign numbers
# ---------------------------------------------------------------------------


def test_closed_column_closes_and_churns():
    """On a closed column the channel set closes, but the stages churn >> net.

    This is the quantitative statement of the N04 report: the net per-step drift
    is a cancellation of opposing stage terms (churn ratio >= 10), so a scalar
    drift understates what the solver is doing by an order of magnitude.
    """
    summary, _ = _closed_loop(60)
    assert summary["closure_residual_max_ulps"] <= 100.0
    assert summary["churn_ratio"] >= 10.0
    assert summary["observed_quiet_max_abs_step_delta_relative"] <= 1e-5
    assert summary["interior_channels"] and summary["boundary_channels"]
    for name in CANONICAL_CHANNELS:
        assert name in summary["channels_detail"]


def test_negative_control_shows_in_the_interior_channels():
    """The refill defect raises interior churn and leaves the closure intact."""
    clean, _ = _closed_loop(50)
    defect, _ = _closed_loop(50, defect="refill_off_all")
    assert defect["closure_residual_max_ulps"] <= 100.0
    key = "topology_max_abs_step_delta_ulps"
    assert defect["topology_steps"] > 0
    assert defect[key] >= 2.0 * max(clean[key], 1e-12)
    # and the *net* interior budget moves with it: a defect signature of
    # |clean - defect| / |clean| >= 50% on this small grid (the campaign gate
    # J5 uses the 10x per-step criterion on the resolved grids).
    signature = abs(clean["interior_cumulative_delta"] - defect["interior_cumulative_delta"])
    assert signature >= 0.5 * abs(clean["interior_cumulative_delta"])