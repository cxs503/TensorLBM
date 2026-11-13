"""Regression battery for the force-corrected NEBB wall + body-force driver.

Capability under test: :mod:`tensorlbm.boundary.nebb` — wall-node
non-equilibrium bounce-back whose wall velocity carries the ``-F/(2 rho_w)``
force correction expected by the Guo body force, plus the channel driver that
turns it into a Poiseuille solution.

Assertions are physics assertions, not smoke tests:

* lattice/BC invariants — the wall direction sets are derived from the lattice
  vectors, and NEBB writes *only* the incoming populations of one plane;
* the no-slip channel reproduces the exact parabola with the expected second
  order of accuracy (error must shrink by >= 3x per grid doubling), reaching
  < 3 % at ny = 40 (owner stretch threshold);
* a negative control: dropping the ``-F/(2 rho_w)`` wall shift leaves a
  measurably larger wall slip — i.e. the correction is load-bearing;
* integration: the repo's main solver
  (:class:`tensorlbm.gpu_core_path.GPUCorePathSolver`) can be driven with
  ``wall_bc="nebb"`` and conserves mass, while its default (``"halfway"``)
  path is unaffected.

Runtimes are kept small (ny <= 20, a few thousand steps); the full-grid
benchmark lives in ``tensorlbm.verification.nebb_bodyforce.run``.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.boundary.d2q9 import C2Q9, OPPOSITE_2Q9, W2Q9
from tensorlbm.boundary.nebb import (
    NebbChannelConfig,
    NebbPoiseuilleChannel,
    analytic_poiseuille,
    force_corrected_velocity,
    nebb_wall_plane,
    wall_density,
    wall_directions,
)


# --------------------------------------------------------------------------- #
# Lattice / boundary invariants
# --------------------------------------------------------------------------- #
def test_wall_directions_follow_lattice_vectors() -> None:
    bottom = wall_directions(C2Q9, -1)
    top = wall_directions(C2Q9, +1)
    # stream2d advects f_k along -c_k, so at the bottom wall (outward -y) the
    # populations that arrive from outside are the ones with c_y < 0.
    assert bottom[0] == (4, 7, 8)  # reconstructed (came from y < 0)
    assert bottom[1] == (3, 5, 6)  # known (came from inside the channel)
    assert top[0] == (3, 5, 6)  # reconstructed (came from y > ny-1)
    assert top[1] == (4, 7, 8)
    assert bottom[2] == (1, 2)  # tangential
    assert {int(OPPOSITE_2Q9[k]) for k in bottom[0]} == set(bottom[1])
    assert {int(OPPOSITE_2Q9[k]) for k in top[0]} == set(top[1])


def test_streaming_convention_probe() -> None:
    """Lock the roll convention the wall sets are derived from."""
    from tensorlbm.boundary.solver2d import stream2d

    ny, nx = 6, 3
    for k, lands_at in ((3, 0), (4, 2)):  # one-hot at row 1
        f = torch.zeros(9, ny, nx, dtype=torch.float64)
        f[k, 1, :] = 1.0
        out = stream2d(f)
        assert float(out[k, lands_at, 0]) == 1.0, (k, lands_at)


def test_nebb_writes_only_incoming_populations() -> None:
    ny, nx = 6, 4
    torch.manual_seed(0)
    f = torch.rand(9, ny, nx, dtype=torch.float64) + 0.1
    before = f.clone()
    rho_w = torch.ones(nx, dtype=torch.float64)
    zeros = torch.zeros(nx, dtype=torch.float64)
    out = nebb_wall_plane(f.clone(), 0, -1, rho_w, zeros, zeros)
    in_dirs, _out, _tan = wall_directions(C2Q9, -1)
    touched = [k for k in range(9) if not torch.equal(out[k, 0, :], before[k, 0, :])]
    assert sorted(touched) == sorted(in_dirs)
    # rows above the wall plane are untouched
    assert torch.equal(out[:, 1:, :], before[:, 1:, :])


# --------------------------------------------------------------------------- #
# Wall density / force correction algebra
# --------------------------------------------------------------------------- #
def test_wall_density_is_equilibrium_consistent() -> None:
    """For a field equal to a local equilibrium at rest, ``rho_w == rho``."""
    ny, nx, rho0 = 6, 3, 1.0
    shape = (9, 1, 1)
    w = W2Q9.view(shape)
    rho = torch.full((ny, nx), rho0, dtype=torch.float64)
    zero = torch.zeros_like(rho)
    ux = torch.full_like(rho, 1e-3)
    cx = C2Q9[:, 0].view(shape).to(torch.float64)
    cc = cx * ux.unsqueeze(0)
    feq = w * rho.unsqueeze(0) * (1.0 + 3.0 * cc + 4.5 * cc * cc - 1.5 * 1e-6)
    dirs = wall_directions(C2Q9, -1)
    rho_w = wall_density(feq, 0, dirs)
    assert torch.allclose(rho_w, torch.full_like(rho_w, rho0), atol=1e-12)


def test_force_correction_signs() -> None:
    rho = torch.full((2, 3), 2.0, dtype=torch.float64)
    u = torch.zeros_like(rho)
    wall = force_corrected_velocity(u, 4.0, rho, sign=-1.0)  # -F/(2 rho)
    bulk = force_corrected_velocity(u, 4.0, rho, sign=+1.0)  # +F/(2 rho)
    assert torch.allclose(wall, torch.full_like(u, -1.0))
    assert torch.allclose(bulk, torch.full_like(u, +1.0))


def test_analytic_reference_is_parabola() -> None:
    ny, gx, nu = 10, 1e-3, 0.2
    ref = analytic_poiseuille(ny, gx, nu)
    assert ref.shape == (ny,)
    assert float(ref[0]) == 0.0 and float(ref[-1]) == 0.0
    # discrete maximum sits on the integer row nearest mid-channel
    y_mid = (ny - 1) // 2
    assert float(ref.max()) == pytest.approx(gx / (2.0 * nu) * y_mid * (ny - 1 - y_mid))


# --------------------------------------------------------------------------- #
# Physics: convergence vs the closed-form parabola
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def converged_runs() -> dict[int, dict]:
    out: dict[int, dict] = {}
    for ny in (10, 20):
        cfg = NebbChannelConfig(nx=4, ny=ny, nu=0.2, gx=1e-3, steps=60_000)
        out[ny] = NebbPoiseuilleChannel(cfg).run()
    return out


def test_channel_matches_analytic_and_converges(converged_runs: dict[int, dict]) -> None:
    err10 = converged_runs[10]["err_global_vs_analytic"]
    err20 = converged_runs[20]["err_global_vs_analytic"]
    assert err20 < 0.02, f"ny=20 error {err20:.4f} above 2 %"
    assert err10 / err20 >= 3.0, (
        f"grid refinement halves the error only {err10 / err20:.2f}x "
        f"({err10:.4f} -> {err20:.4f}); second order expects ~4x"
    )


def test_steady_state_is_reached(converged_runs: dict[int, dict]) -> None:
    for ny, run in converged_runs.items():
        assert run["steady_residual"] < 1e-10, (ny, run["steady_residual"])
        assert run["steps_done"] < run["cfg"]["ny"] * 1e4
        assert run["mass_drift"] < 1e-10, (ny, run["mass_drift"])


def test_negative_control_force_correction_is_load_bearing() -> None:
    """Without the ``-Gx/(2 rho_w)`` wall shift the wall slip grows."""
    runs = {}
    for corrected in (True, False):
        cfg = NebbChannelConfig(
            nx=4, ny=20, nu=0.2, gx=1e-3, steps=60_000,
            force_corrected_wall=corrected,
        )
        runs[corrected] = NebbPoiseuilleChannel(cfg).run()
    slip_on = max(abs(v) for v in runs[True]["wall_node_ux"])
    slip_off = max(abs(v) for v in runs[False]["wall_node_ux"])
    assert slip_off > slip_on * 1.1, (
        f"wall slip with the correction {slip_on:.6g} vs without {slip_off:.6g}: "
        "the correction is not load-bearing"
    )
    assert runs[False]["err_global_vs_analytic"] > runs[True]["err_global_vs_analytic"]


# --------------------------------------------------------------------------- #
# Integration: main core-path solver accepts the NEBB walls
# --------------------------------------------------------------------------- #
def test_core_path_nebb_wall_option_is_wired() -> None:
    from tensorlbm.gpu_core_path import (
        GPUCorePathConfig,
        GPUCorePathSolver,
        channel_analytic_velocity_nebb,
    )

    with pytest.raises(ValueError):
        GPUCorePathConfig(wall_bc="nonsense").validate()

    cfg = GPUCorePathConfig(
        nx=4, ny=12, nz=2, tau=0.8, u_max_target=0.05,
        steps=1, backend="auto", wall_bc="nebb",
    )
    solver = GPUCorePathSolver(cfg)
    solver.step(2000)
    _y, prof = solver.channel_profile()
    _yy, ref = channel_analytic_velocity_nebb(cfg)
    err = float((prof - ref[1:-1]).abs().max() / ref.abs().max())
    assert err < 0.15, f"NEBB wall arm in the core path: {err:.4f} error"
    assert abs(solver.mass() - 1.0) < 1e-9
    assert "nebb" in solver.backend_note or solver.backend == "eager"


def test_core_path_default_path_unchanged() -> None:
    """``wall_bc="halfway"`` (default) keeps the pre-existing behaviour."""
    from tensorlbm.gpu_core_path import GPUCorePathConfig, GPUCorePathSolver

    assert GPUCorePathConfig().wall_bc == "halfway"
    cfg = GPUCorePathConfig(
        nx=4, ny=12, nz=2, tau=0.8, u_max_target=0.05, steps=1, backend="eager"
    )
    solver = GPUCorePathSolver(cfg)
    solver.step(2000)
    assert solver.mass() == pytest.approx(1.0, abs=1e-7)
    assert solver.backend_note == ""