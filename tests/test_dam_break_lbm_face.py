"""Tests for the dam-break verification face (``benchmarks/dam_break``).

Contracts locked here:

* the Ritter (1892) analytic anchor and its self-consistency,
* the metric definitions (relative L1 profile error, front position, observed
  order) used by both benchmark routes,
* the 2-D shallow-water solver (:mod:`tensorlbm.shallow_water2d`): lake-at-rest
  exactness, mass conservation, positivity, and the Ritter accuracy/second-order
  advantage that the face's gates rely on,
* the free-surface LBM slab realisation used as the node's "2-D" route: finiteness,
  the no-direct-liquid/gas-link invariant, and spanwise symmetry under free-slip,
* the artifact schema of ``benchmarks/dam_break/{2d,3d}/result.json``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from tensorlbm.shallow_water2d import (
    H_DRY,
    SWEState,
    dam_break_initial_state,
    depth_profile_x,
    mass_drift,
    simulate,
)

REPO = Path(__file__).resolve().parents[1]
DAM_BREAK = REPO / "benchmarks" / "dam_break"
sys.path.insert(0, str(DAM_BREAK))

import common  # noqa: E402


# ---------------------------------------------------------------------------
# Ritter (1892) analytic anchor
# ---------------------------------------------------------------------------
@pytest.mark.physics
def test_ritter_depth_shape_and_bounds() -> None:
    h0, g, x_dam, t = 1.0, 1.0, 1.0, 0.8
    c0 = math.sqrt(g * h0)
    # At the dam face the analytic depth is 4/9 h0 (xi = 0).
    assert common.ritter_depth(np.array([x_dam]), t, h0, g, x_dam)[0] == pytest.approx(4.0 / 9.0)
    # Behind the rarefaction head the reservoir is still at rest.
    assert common.ritter_depth(np.array([x_dam - c0 * t]), t, h0, g, x_dam)[0] == pytest.approx(h0)
    # Beyond the front the bed is dry.
    assert common.ritter_depth(np.array([x_dam + 2 * c0 * t]), t, h0, g, x_dam)[0] == 0.0
    # The profile is monotonically non-increasing in x.
    x = np.linspace(0.0, x_dam + 2 * c0 * t, 200)
    h = common.ritter_depth(x, t, h0, g, x_dam)
    assert np.all(np.diff(h) <= 1e-12)
    assert common.ritter_front(t, h0, g, x_dam) == pytest.approx(x_dam + 2 * c0 * t)
    assert common.ritter_head(t, h0, g, x_dam) == pytest.approx(x_dam - c0 * t)


@pytest.mark.physics
def test_ritter_depth_integral_matches_self_similar_volume() -> None:
    """Wetted volume over [head, front] equals h0 c0 t exactly (Ritter).

    The reservoir is semi-infinite, so the water inside the self-similar wedge
    grows at the constant rate ``h0 c0`` (the head sweeps the reservoir at
    ``c0``); this is the closed-form volume the numeric routes can be audited
    against.
    """
    h0, g, x_dam, t = 1.0, 1.0, 0.0, 0.5
    c0 = math.sqrt(g * h0)
    x = np.linspace(-c0 * t, 2 * c0 * t, 2_000_001)
    h = common.ritter_depth(x, t, h0, g, x_dam)
    volume = float(np.trapezoid(h, x))
    assert volume == pytest.approx(h0 * c0 * t, rel=2e-5)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_profile_metrics_is_zero_for_the_analytic_profile() -> None:
    h0, g, x_dam, t = 1.0, 1.0, 1.0, 0.8
    x = np.linspace(0.0, 4.0, 800)
    h_ref = common.ritter_depth(x, t, h0, g, x_dam)
    metrics = common.profile_metrics(x, h_ref, t, h0, g, x_dam)
    assert metrics["rel_l1"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["rel_l2"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["n_cells"] > 100


def test_profile_metrics_recovers_a_known_perturbation() -> None:
    h0, g, x_dam, t = 1.0, 1.0, 1.0, 0.8
    x = np.linspace(0.0, 4.0, 4000)
    h_ref = common.ritter_depth(x, t, h0, g, x_dam)
    wet = h_ref > 0.0
    perturbed = h_ref + 0.05 * h_ref
    metrics = common.profile_metrics(x, perturbed, t, h0, g, x_dam, front_window=0.0)
    assert metrics["rel_l1"] == pytest.approx(0.05, rel=0.02)
    assert np.count_nonzero(wet) > 0


def test_front_position_interpolates_and_is_bounded() -> None:
    x = np.arange(10, dtype=float)
    h = np.array([1.0, 1.0, 1.0, 1.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0])
    pos = common.front_position(x, h)
    assert 3.0 <= pos <= 5.0
    assert common.front_position(x, np.zeros_like(h)) == pytest.approx(0.0)


def test_convergence_order_from_synthetic_errors() -> None:
    second = common.convergence_order([1.0, 0.25, 0.0625], [2.0, 2.0])
    assert second["order_mean"] == pytest.approx(2.0, abs=1e-12)
    first = common.convergence_order([1.0, 0.5, 0.25], [2.0, 2.0])
    assert first["order_mean"] == pytest.approx(1.0, abs=1e-12)


def test_gates_are_predeclared_constants() -> None:
    gates = common.gates()
    assert gates["profile_rel_error_max"] == pytest.approx(0.05)
    assert gates["front_rel_error_max"] == pytest.approx(0.05)
    assert gates["mass_drift_max"] == pytest.approx(1.0e-3)
    assert gates["convergence_order_min"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# 2-D shallow-water solver (analytic-anchor route)
# ---------------------------------------------------------------------------
def _swe_dam_break(
    nx: int, order: int, t_end: float = 0.8
) -> tuple[SWEState, np.ndarray, np.ndarray]:
    state = dam_break_initial_state(6, nx, depth=1.0, column_width=1.0, dx=4.0 / nx, dy=0.2, g=1.0)
    state = simulate(state, t_end=t_end, cfl=0.4, order=order)
    x = (np.arange(nx) + 0.5) * (4.0 / nx)
    return state, x, depth_profile_x(state)


@pytest.mark.physics
def test_swe_lake_at_rest_is_exactly_preserved() -> None:
    nx, ny = 40, 4
    h = np.full((ny, nx), 2.0)
    zeros = np.zeros_like(h)
    state = SWEState(h=h, hu=zeros.copy(), hv=zeros.copy(), dx=0.1, dy=0.1, g=1.0)
    out = simulate(state, t_end=0.3, cfl=0.4, order=2)
    assert np.max(np.abs(out.h - 2.0)) <= 1e-12
    assert np.max(np.abs(out.hu)) <= 1e-12
    assert mass_drift(out) <= 1e-12


@pytest.mark.physics
def test_swe_mass_conservation_and_positivity() -> None:
    state, _, h = _swe_dam_break(100, order=2, t_end=0.6)
    assert mass_drift(state) <= 1.0e-5
    assert np.all(state.h >= 0.0)
    assert np.all(np.isfinite(state.h))
    assert float(np.min(h)) >= 0.0
    assert np.any(h < H_DRY)


@pytest.mark.physics
def test_swe_reproduces_ritter_within_the_profile_gate() -> None:
    state, x, h = _swe_dam_break(200, order=2)
    assert state.t == pytest.approx(0.8, abs=1e-12)
    metrics = common.profile_metrics(x, h, 0.8, 1.0, 1.0, 1.0)
    assert metrics["rel_l1"] <= common.PROFILE_REL_ERROR_MAX
    front = common.front_metrics(x, h, 0.8, 1.0, 1.0, 1.0)
    assert front["x_front_analytic"] == pytest.approx(2.6, abs=1e-12)
    assert 0.0 < front["x_front"] < front["x_front_analytic"] + 0.1


@pytest.mark.physics
def test_swe_second_order_beats_first_order_on_the_anchor_metric() -> None:
    _, x_first, h_first = _swe_dam_break(100, order=1)
    _, _, h_second = _swe_dam_break(100, order=2)
    err_first = common.profile_metrics(x_first, h_first, 0.8, 1.0, 1.0, 1.0)["rel_l1"]
    err_second = common.profile_metrics(x_first, h_second, 0.8, 1.0, 1.0, 1.0)["rel_l1"]
    assert err_second < err_first
    # Measured ratio e(order 2)/e(order 1) = 0.50 / 0.41 / 0.33 at nx = 100/200/400.
    assert err_second < 0.6 * err_first
    assert err_second < common.PROFILE_REL_ERROR_MAX


# ---------------------------------------------------------------------------
# Free-surface LBM slab (the node's lattice-Boltzmann route)
# ---------------------------------------------------------------------------
@pytest.mark.physics
def test_fs_lbm_slab_smoke_conserves_topology_and_symmetry() -> None:
    torch = pytest.importorskip("torch")
    from tensorlbm.d3q19 import equilibrium3d
    from tensorlbm.free_surface_lbm import (
        GAS,
        LIQUID,
        free_surface_step,
        init_fill_rectangular,
        init_flags_from_fill,
        init_mass_from_fill,
    )

    h0, nz, ny, nx = 6, 4, 12, 24
    fill, solid = init_fill_rectangular(nz, ny, nx, h0, h0, "cpu")
    flags = init_flags_from_fill(fill, solid)
    rho0 = torch.where(flags == LIQUID, torch.ones_like(fill), torch.zeros_like(fill))
    f = equilibrium3d(rho0, torch.zeros_like(fill), torch.zeros_like(fill), torch.zeros_like(fill))
    mass = init_mass_from_fill(fill, flags, 1.0)
    mass0 = float(mass.sum())
    slip = torch.zeros_like(solid)
    slip[0, :, :] = solid[0, :, :]
    slip[-1, :, :] = solid[-1, :, :]
    g = 0.08**2 / h0
    for _ in range(8):
        f, fill, flags, mass, _ = free_surface_step(
            f, fill, flags, solid, mass=mass, tau=0.52, gy=-g, rho_gas=1.0, free_slip_mask=slip
        )
    assert bool(torch.isfinite(f).all())
    assert bool(torch.isfinite(fill).all())
    assert abs(float(mass.sum()) - mass0) / mass0 < 0.2
    # The Körner envelope keeps every L/G link represented by an interface cell.
    liquid = flags == LIQUID
    gas = flags == GAS
    liquid_neighbour_gas = torch.zeros_like(liquid)
    for dim in (0, 1, 2):
        for shift in (1, -1):
            liquid_neighbour_gas |= gas & liquid.roll(shifts=shift, dims=dim)
    assert int(liquid_neighbour_gas.sum()) == 0
    # Free-slip spanwise walls keep the slab nearly planar: the spanwise spread
    # of the water-column height stays a small fraction of the column height
    # (measured 1.2% at h0=6; the Körner conversion logic is local, so exact
    # bitwise symmetry is not expected).
    column_height = fill.clamp(0.0, 1.0).sum(dim=1)
    spread = float((column_height.max(dim=0).values - column_height.min(dim=0).values).max())
    assert spread / float(column_height.max()) <= 0.05


# ---------------------------------------------------------------------------
# Artifact schema (populated by the production runs)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("face", ["2d", "3d"])
def test_result_json_schema(face: str) -> None:
    path = DAM_BREAK / face / "result.json"
    if not path.exists():  # pragma: no cover - artifacts are produced by the runs
        pytest.skip(f"{path} not generated in this checkout")
    payload = json.loads(path.read_text())
    assert payload["gates"]["profile_rel_error_max"] == pytest.approx(0.05)
    assert payload["gates"]["mass_drift_max"] == pytest.approx(1.0e-3)
    assert payload["gates"]["convergence_order_min"] == pytest.approx(1.5)
    # The 2-D face reports one verdict per route, the 3-D face a single one.
    verdicts = (
        [payload["verdict"]]
        if "P1_profile" in payload["verdict"]
        else list(payload["verdict"].values())
    )
    assert verdicts
    for verdict in verdicts:
        assert set(verdict) >= {"P1_profile", "P2_front", "P3_mass", "P4_order", "all_pass"}
        for key in ("P1_profile", "P2_front", "P3_mass", "P4_order"):
            item = verdict[key]
            assert "value" in item and "limit" in item and isinstance(item["pass"], bool)
            assert isinstance(verdict["all_pass"], bool)
    runs = payload.get("runs") or []
    if runs:
        assert len(runs) >= 3  # a three-grid refinement study
        assert all("mass_drift" in run for run in runs)
