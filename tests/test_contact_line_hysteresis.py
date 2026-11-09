"""Unit tests for the hysteretic-wall model and criterion C4 (IN100-N02)."""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.contact_line.engine import Channel, WallStepper, sessile_drop_state
from tensorlbm.contact_line.hysteresis import (
    HysteresisGap,
    HystereticWall,
    evaluate_c4,
    loop_area,
    monotonicity,
)


class _FakeStepper:
    """Minimal stand-in exposing the attributes HystereticWall touches."""

    def __init__(self, channel: Channel, rho_v: float = 0.16) -> None:
        self.channel = channel
        self.rho_v = rho_v
        self.device = torch.device(channel.device)
        self.psi_l = 1.0
        self.wall_psi: torch.Tensor | None = None

    def uniform_wall(self, fraction: float) -> torch.Tensor:
        return torch.full(self.channel.shape, float(fraction))

    def wall_fraction_field(self, field: torch.Tensor) -> torch.Tensor:
        return field

    def set_wall_psi(self, field: torch.Tensor | float) -> None:
        self.wall_psi = field if isinstance(field, torch.Tensor) else torch.tensor(float(field))


def _rho(channel: Channel, wet_columns: int, rho_l: float = 1.9) -> torch.Tensor:
    """Liquid covering ``wet_columns`` columns of the wall band, vapour elsewhere."""
    rho = torch.full(channel.shape, 0.16)
    rho[channel.wall_rows : channel.wall_rows + 4, : int(wet_columns)] = rho_l
    return rho


class TestHystereticWall:
    def test_bind_installs_base_wall(self):
        ch = Channel(nx=40, ny=24, wall_rows=2)
        wall = HystereticWall(frac_advancing=0.85, frac_receding=0.75)
        fake = _FakeStepper(ch)
        wall.bind(fake)
        assert fake.wall_psi is not None
        assert float(fake.wall_psi.mean()) == pytest.approx(0.80, abs=1e-6)
        assert wall.state.dtype == torch.bool
        assert bool(wall.state.all())  # starts in the advancing state

    def test_advancing_and_receding_detected(self):
        ch = Channel(nx=40, ny=24, wall_rows=2)
        wall = HystereticWall(frac_advancing=0.85, frac_receding=0.75, deadband=0.05)
        fake = _FakeStepper(ch)
        wall.bind(fake)
        wall.update(_rho(ch, 10), fake)  # first call only seeds the coverage
        wall.update(_rho(ch, 20), fake)  # coverage grows -> advancing
        assert bool(wall.state.all())
        assert float(fake.wall_psi.mean()) == pytest.approx(0.85, abs=1e-6)
        # Coverage shrinks only where liquid is lost (columns 5..19): those
        # columns switch to receding, the untouched ones hold their state.  This
        # per-column partial update is what makes the wall a *pinning* model.
        wall.update(_rho(ch, 5), fake)
        mean = float(fake.wall_psi.mean())
        # only the columns that lost liquid (plus their smoothed neighbours)
        # switch, so the applied wall fraction moves strictly between the two
        # adsorption states
        assert 0.75 < mean < 0.85
        assert int(wall.state.sum()) < 30

    def test_deadband_holds_state(self):
        ch = Channel(nx=40, ny=24, wall_rows=2)
        wall = HystereticWall(frac_advancing=0.85, frac_receding=0.75, deadband=1.0)
        fake = _FakeStepper(ch)
        wall.bind(fake)
        wall.update(_rho(ch, 10), fake)
        wall.update(_rho(ch, 11), fake)  # change 1.9*4 = 7.6 per column ... below 1.0*? see below
        assert wall.n_switch == 0  # state never changed because the dead band holds it

    def test_gap_and_mean_fraction(self):
        wall = HystereticWall(frac_advancing=0.9, frac_receding=0.7)
        assert wall.delta_fraction == pytest.approx(0.2, abs=1e-12)
        assert wall.mean_fraction() == pytest.approx(0.8, abs=1e-12)


class TestMonotonicity:
    def test_increasing(self):
        out = monotonicity([0.0, 1.0, 2.0, 3.0], [0.1, 2.0, 4.0, 9.0])
        assert out["spearman"] == pytest.approx(1.0)
        assert out["strictly_increasing"] is True
        assert out["passes"] is True

    def test_decreasing_fails(self):
        out = monotonicity([0.0, 1.0, 2.0, 3.0], [9.0, 4.0, 2.0, 0.1])
        assert out["passes"] is False

    def test_too_few(self):
        assert monotonicity([1.0, 2.0], [1.0, 2.0])["passes"] is False


def _gap(delta: float, ta: float, tr: float) -> HysteresisGap:
    return HysteresisGap(
        delta_fraction=delta,
        frac_advancing=0.8 + 0.5 * delta,
        frac_receding=0.8 - 0.5 * delta,
        gx=1e-5,
        capillary=0.01,
        velocity=1e-3,
        theta_a_deg=ta,
        theta_r_deg=tr,
        theta_a_unc_deg=0.5,
        theta_r_unc_deg=0.5,
    )


class TestC4:
    def test_passes_for_monotone_gap(self):
        gaps = [_gap(0.0, 110.5, 110.0), _gap(0.05, 112.0, 109.0),
                _gap(0.10, 116.0, 108.0), _gap(0.15, 121.0, 107.0)]
        verdict = evaluate_c4(gaps)
        assert verdict.passes
        assert verdict.all_positive
        assert verdict.strictly_increasing
        assert verdict.control_gap_deg == pytest.approx(0.5)

    def test_control_must_close(self):
        gaps = [_gap(0.0, 120.0, 110.0), _gap(0.05, 122.0, 109.0),
                _gap(0.10, 126.0, 108.0), _gap(0.15, 131.0, 107.0)]
        verdict = evaluate_c4(gaps, control_tolerance_deg=3.0)
        assert not verdict.passes
        assert "control" in verdict.note

    def test_negative_gap_fails(self):
        gaps = [_gap(0.0, 110.0, 110.0), _gap(0.05, 108.0, 112.0),
                _gap(0.10, 106.0, 115.0), _gap(0.15, 104.0, 118.0)]
        verdict = evaluate_c4(gaps)
        assert not verdict.passes
        assert verdict.all_positive is False


class TestLoopArea:
    def test_reversible_loop_is_empty(self):
        ca = [1e-3, 1e-2, 1e-1]
        th = [80.0, 85.0, 90.0]
        assert loop_area(ca, th, ca, th) == pytest.approx(0.0, abs=1e-9)

    def test_open_loop_has_area(self):
        ca = [1e-3, 1e-2, 1e-1]
        up = [90.0, 95.0, 100.0]
        down = [80.0, 82.0, 85.0]
        area = loop_area(ca, up, ca, down)
        assert math.isfinite(area)
        assert area > 0.0

    def test_disjoint_range(self):
        assert math.isnan(loop_area([1e-3, 1e-2], [1.0, 2.0], [1e-1, 1.0], [3.0, 4.0]))


class TestHystereticWallOnLattice:
    """Short integration test: the hysteretic wall drives a real SCMP stepper."""

    def test_wall_field_reaches_stepper(self):
        ch = Channel(nx=48, ny=32, wall_rows=2)
        stepper = WallStepper(ch, wall_fraction=0.8)
        wall = HystereticWall(frac_advancing=0.85, frac_receding=0.75)
        wall.bind(stepper)
        f, _ = sessile_drop_state(ch, 8.0)
        f = stepper.run(f, 20)
        rho = stepper.density(f)
        wall.update(rho, stepper)
        wall.update(rho, stepper)
        assert stepper.wall_psi is not None
        assert torch.isfinite(stepper.wall_psi).all()
        assert wall.n_updates >= 1