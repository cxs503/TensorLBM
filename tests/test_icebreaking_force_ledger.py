"""Tests for the SI icebreaking force ledger."""

import math

import pytest

from tensorlbm.icebreaking import IcebreakingForceLedger, WrenchSI


def test_force_decomposition_and_positive_resistance() -> None:
    ledger = IcebreakingForceLedger(
        time_s=1.25,
        fluid_on_hull=WrenchSI(fx_n=-120.0, mz_nm=4.0),
        ice_contact_on_hull=WrenchSI(fx_n=-380.0, fy_n=10.0, mz_nm=6.0),
        other_on_hull=WrenchSI(fx_n=-5.0),
    )

    assert ledger.total_on_hull.fx_n == pytest.approx(-505.0)
    assert ledger.total_on_hull.fy_n == pytest.approx(10.0)
    assert ledger.total_on_hull.mz_nm == pytest.approx(10.0)
    assert ledger.resistance_n == pytest.approx(505.0)


def test_resistance_uses_configured_forward_axis() -> None:
    ledger = IcebreakingForceLedger(
        time_s=0.0,
        fluid_on_hull=WrenchSI(fx_n=-3.0, fy_n=8.0),
        forward_axis=(0.0, 1.0, 0.0),
    )

    assert ledger.resistance_n == pytest.approx(-8.0)


def test_record_contains_separate_and_total_contributions() -> None:
    ledger = IcebreakingForceLedger(
        time_s=2.0,
        fluid_on_hull=WrenchSI(fx_n=-2.0),
        ice_contact_on_hull=WrenchSI(fx_n=-7.0),
    )

    record = ledger.as_record()

    assert record["time_s"] == pytest.approx(2.0)
    assert record["fluid_fx_n"] == pytest.approx(-2.0)
    assert record["ice_contact_fx_n"] == pytest.approx(-7.0)
    assert record["total_fx_n"] == pytest.approx(-9.0)
    assert record["resistance_n"] == pytest.approx(9.0)


def test_wrench_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        WrenchSI(fx_n=math.nan)

    with pytest.raises(ValueError, match="finite"):
        WrenchSI(mz_nm=math.inf)


@pytest.mark.parametrize("time_s", [-1.0, math.nan, math.inf])
def test_ledger_rejects_invalid_time(time_s: float) -> None:
    with pytest.raises(ValueError, match="time_s"):
        IcebreakingForceLedger(time_s=time_s)


@pytest.mark.parametrize(
    "axis",
    [
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (1.0, 1.0),
        (1.0, math.nan, 0.0),
    ],
)
def test_ledger_rejects_invalid_forward_axis(axis: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="forward_axis"):
        IcebreakingForceLedger(time_s=0.0, forward_axis=axis)  # type: ignore[arg-type]


def test_wrench_addition_and_scaling() -> None:
    result = WrenchSI(fx_n=2.0, my_nm=3.0) + WrenchSI(fx_n=-5.0, my_nm=1.0)

    assert result.fx_n == pytest.approx(-3.0)
    assert result.my_nm == pytest.approx(4.0)
    assert result.scaled(2.0).fx_n == pytest.approx(-6.0)
    with pytest.raises(ValueError, match="finite"):
        result.scaled(math.inf)


def test_surface_traction_integrates_force_and_moment_about_origin() -> None:
    from tensorlbm.icebreaking import integrate_surface_traction

    wrench = integrate_surface_traction(
        positions_m=[(1.0, 0.0, 0.0), (0.0, 2.0, 0.0)],
        tractions_pa=[(0.0, 3.0, 0.0), (4.0, 0.0, 0.0)],
        area_weights_m2=[2.0, 0.5],
        origin_m=(0.0, 0.0, 0.0),
    )

    assert wrench.fx_n == pytest.approx(2.0)
    assert wrench.fy_n == pytest.approx(6.0)
    assert wrench.fz_n == pytest.approx(0.0)
    assert wrench.mz_nm == pytest.approx(6.0 - 4.0)


def test_surface_traction_moment_changes_with_reference_origin() -> None:
    from tensorlbm.icebreaking import integrate_surface_traction

    origin_wrench = integrate_surface_traction(
        positions_m=[(1.0, 0.0, 0.0)],
        tractions_pa=[(0.0, 2.0, 0.0)],
        area_weights_m2=[3.0],
        origin_m=(0.0, 0.0, 0.0),
    )
    shifted_wrench = integrate_surface_traction(
        positions_m=[(1.0, 0.0, 0.0)],
        tractions_pa=[(0.0, 2.0, 0.0)],
        area_weights_m2=[3.0],
        origin_m=(1.0, 0.0, 0.0),
    )

    assert origin_wrench.mz_nm == pytest.approx(6.0)
    assert shifted_wrench.mz_nm == pytest.approx(0.0)


@pytest.mark.parametrize(
    "positions,tractions,areas",
    [
        ([(0.0, 0.0, 0.0)], [], [1.0]),
        ([(0.0, 0.0)], [(1.0, 0.0, 0.0)], [1.0]),
        ([(0.0, 0.0, 0.0)], [(1.0, 0.0, 0.0)], [-1.0]),
        ([(0.0, 0.0, 0.0)], [(math.nan, 0.0, 0.0)], [1.0]),
    ],
)
def test_surface_traction_rejects_invalid_samples(positions, tractions, areas) -> None:
    from tensorlbm.icebreaking import integrate_surface_traction

    with pytest.raises(ValueError):
        integrate_surface_traction(positions, tractions, areas)



def test_particle_traction_integration_returns_forces_for_dem() -> None:
    from tensorlbm.icebreaking import integrate_particle_tractions_2d

    forces = integrate_particle_tractions_2d(
        tractions_pa=[(2.0, -3.0), (-1.0, 4.0)],
        area_weights_m2=[0.5, 2.0],
    )

    assert forces == ((1.0, -1.5), (-2.0, 8.0))
    assert sum(force[0] for force in forces) == pytest.approx(-1.0)
    assert sum(force[1] for force in forces) == pytest.approx(6.5)


@pytest.mark.parametrize(
    "tractions,areas",
    [
        ([(1.0, 2.0)], []),
        ([(1.0, 2.0, 3.0)], [1.0]),
        ([(math.inf, 0.0)], [1.0]),
        ([(1.0, 0.0)], [-1.0]),
    ],
)
def test_particle_traction_integration_rejects_invalid_inputs(tractions, areas) -> None:
    from tensorlbm.icebreaking import integrate_particle_tractions_2d

    with pytest.raises(ValueError):
        integrate_particle_tractions_2d(tractions, areas)
