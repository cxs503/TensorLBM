"""Tests for the 2-D LBM/DEM coupling prototype."""

import json

import pytest
import torch

pytest.importorskip("tensordem")

from tensorlbm.ice_coupling_2d import (
    CoupledIce2D,
    CoupledIceConfig,
    bilinear_map,
    cross2,
)


def test_mapping_force_moment_and_work():
    positions = torch.tensor(
        [[0.132, 0.267], [0.349, 0.456]], dtype=torch.float64
    )
    mapping = bilinear_map(positions, (24, 24), 0.025)
    y, x = torch.meshgrid(
        torch.arange(24, dtype=torch.float64) * 0.025,
        torch.arange(24, dtype=torch.float64) * 0.025,
        indexing="ij",
    )
    grid = torch.stack((x, y), dim=-1).reshape(-1, 2)
    forces = torch.tensor([[3.0, -5.0], [-2.0, 7.0]], dtype=torch.float64)
    velocity = grid @ torch.tensor(
        [[0.2, -0.1], [0.3, 0.4]], dtype=torch.float64
    )
    spread = mapping.T @ forces

    torch.testing.assert_close(
        spread.sum(0), forces.sum(0), rtol=0, atol=1e-14
    )
    torch.testing.assert_close(
        cross2(grid, spread).sum(),
        cross2(positions, forces).sum(),
        rtol=0,
        atol=1e-14,
    )
    torch.testing.assert_close(
        (spread * velocity).sum(),
        (forces * (mapping @ velocity)).sum(),
        rtol=0,
        atol=1e-14,
    )


def test_real_coupling_and_mid_exchange_restart():
    sim = CoupledIce2D(CoupledIceConfig(duration_s=0.01, exchange_steps=3))
    for _ in range(5):
        sim.step()

    resumed = CoupledIce2D.from_snapshot(json.loads(json.dumps(sim.snapshot())))
    for _ in range(12):
        first = sim.step()
        second = resumed.step()

    torch.testing.assert_close(sim.f, resumed.f, rtol=0, atol=0)
    torch.testing.assert_close(
        sim.dem.positions, resumed.dem.positions, rtol=0, atol=0
    )
    assert first == second
    assert first["fluid_kinetic_J"] > 0
    assert max(abs(v) for v in first["momentum_residual_kg_m_s"]) < 1e-10
    assert first["mass_relative_error"] < 1e-12
    assert first["force_mapping_error_N"] < 1e-12
    assert first["moment_mapping_error_Nm"] < 1e-12
    assert first["adjoint_power_error_W"] < 1e-12


def test_dry_no_hidden_fluid_load():
    sim = CoupledIce2D(CoupledIceConfig(wet=False, duration_s=0.002))
    for _ in range(4):
        record = sim.step()

    assert record["fluid_kinetic_J"] == 0
    assert record["fluid_fy_n"] == 0
    assert record["mass_relative_error"] == 0


def test_reject_out_of_support():
    with pytest.raises(ValueError):
        bilinear_map(torch.tensor([[-0.01, 0.2]]), (24, 24), 0.025)


@pytest.mark.parametrize(
    "bad",
    ["population", "index", "substeps", "unknown_scalar", "vector", "exchange"],
)
def test_restart_rejects_invalid_state(bad):
    sim = CoupledIce2D(CoupledIceConfig(duration_s=0.002))
    sim.step()
    state = sim.snapshot()

    if bad == "population":
        state["f"][0][0][0] = -1
    elif bad == "index":
        state["step_index"] = -1
    elif bad == "substeps":
        state["substeps"] = 999
    elif bad == "unknown_scalar":
        state["scalars"]["step"] = 42
    elif bad == "vector":
        state["vectors"]["external_impulse"] = [1.0]
    else:
        state["exchange"]["ice_load"] = [[1.0, 2.0]]

    with pytest.raises(ValueError):
        CoupledIce2D.from_snapshot(state)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dx_m", 0),
        ("dx_m", True),
        ("dx_m", "0.025"),
        ("fluid_dt_s", False),
        ("duration_s", float("nan")),
        ("density_kg_m3", True),
        ("viscosity_m2_s", float("inf")),
        ("coupling_rate_s", complex(30, 0)),
        ("ice_radius_m", -1),
        ("tool_speed_m_s", "0.2"),
        ("tool_gap_m", 0),
        ("breaking_strain", True),
        ("shear_breaking_strain", None),
        ("wet", "false"),
        ("wet", 0),
        ("nx", 48.0),
        ("ny", True),
        ("exchange_steps", 1.5),
        ("ice_nx", "9"),
        ("ice_ny", False),
    ],
)
def test_config_rejects_bad_units_and_types(field, value):
    with pytest.raises(ValueError):
        CoupledIceConfig(**{field: value})


def test_config_accepts_positive_integral_real_parameters():
    assert CoupledIceConfig(density_kg_m3=1000, coupling_rate_s=30).tau > 0.5


@pytest.mark.parametrize(
    "positions,shape,dx",
    [
        (torch.tensor([[float("nan"), 0.2]]), (24, 24), 0.025),
        (torch.tensor([[float("inf"), 0.2]]), (24, 24), 0.025),
        (torch.tensor([0.1, 0.2]), (24, 24), 0.025),
        (torch.tensor([[0.1, 0.2, 0.3]]), (24, 24), 0.025),
        (torch.empty((0, 2)), (24, 24), 0.025),
        ([[0.1, 0.2]], (24, 24), 0.025),
        (torch.tensor([[1, 2]]), (24, 24), 0.025),
        (torch.tensor([[0.1, 0.2]]), (24, 24), 0),
        (torch.tensor([[0.1, 0.2]]), (24, 24), True),
        (torch.tensor([[0.1, 0.2]]), (24, 24), "0.025"),
        (torch.tensor([[0.1, 0.2]]), (24, 24), float("nan")),
        (torch.tensor([[0.1, 0.2]]), (24.0, 24), 0.025),
        (torch.tensor([[0.1, 0.2]]), (True, 24), 0.025),
        (torch.tensor([[0.1, 0.2]]), (24,), 0.025),
        (torch.tensor([[0.1, 0.2]]), None, 0.025),
        (
            torch.tensor([[1e308, 0.2]], dtype=torch.float64),
            (24, 24),
            1e-308,
        ),
    ],
)
def test_bilinear_rejects_invalid_input_before_indexing(positions, shape, dx):
    with pytest.raises(ValueError):
        bilinear_map(positions, shape, dx)
