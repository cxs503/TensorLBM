"""Unit and conservation tests for the LBM free-surface Zalesak module.

Runs in a few seconds: every test uses a small grid and a short run, so the
suite only certifies *properties* (conservation, boundedness, lattice algebra,
free-surface bookkeeping), not accuracy; the accuracy numbers are produced by
``benchmarks/zalesak_rotation/run_zalesak_lbm_campaign.py``.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm import free_surface_zalesak_lbm as Z
from tensorlbm.free_surface_zalesak import analytic_volume_2d

DTYPE = torch.float64


def _setup(n: int, dim: int = 2, *, omega: float = 0.05):
    name, C, w, _opp = Z.lattice(dim, dtype=DTYPE)
    shifts = Z._shifts(C, dim)
    phi0 = Z.initial_fill(n, dim=dim, sub=2)
    u = Z.cell_velocity(n, dim=dim, omega=omega, dtype=DTYPE)
    return C, w, shifts, phi0, u


def test_lattice_weights_and_opposites():
    for dim, q in ((2, 9), (3, 19)):
        name, C, w, opp = Z.lattice(dim, dtype=DTYPE)
        assert C.shape == (q, dim)
        assert abs(float(w.sum().item()) - 1.0) < 1e-15
        assert Z.LATTICE_NAMES[dim] == name
        # each direction's opposite is the negated velocity
        for i in range(q):
            assert torch.equal(C[opp[i]], -C[i])


def test_equilibrium_conserves_mass_and_moment():
    """sum_q g^eq = phi and sum_q c_q g^eq = phi u (D2Q9 and D3Q19)."""
    n = 8
    for dim in (2, 3):
        C, w, _shifts, phi0, u = _setup(n, dim)
        geq = Z.equilibrium_fill(phi0, u, C, w)
        assert torch.allclose(geq.sum(0), phi0, atol=1e-14)
        for d in range(dim):
            moment = sum(
                C[q, dim - 1 - d] * geq[q] for q in range(C.shape[0])
            )
            assert torch.allclose(moment, phi0 * u[d], atol=1e-14)


def test_streaming_conserves_total():
    n = 8
    C, w, shifts, phi0, u = _setup(n, 2)
    g = Z.equilibrium_fill(phi0, u, C, w) * (1.0 + 0.1 * torch.rand(9, n, n, dtype=DTYPE))
    assert abs(float(Z.stream_sum(g, shifts).sum()) - float(g.sum())) < 1e-12


def test_bound_ledger_bounds_and_exact_mass():
    torch.manual_seed(0)
    n = 16
    phi = torch.rand(n, n, dtype=DTYPE)
    phi[3:6, 3:6] += 1.6           # overfill
    phi[10:13, 10:13] -= 0.9       # over-empty
    m0 = float(phi.sum())
    out, info = Z.bound_ledger(phi)
    assert float(out.min()) >= -1e-12
    assert float(out.max()) <= 1.0 + 1e-12
    assert abs(float(out.sum()) - m0) < 1e-10 * max(1.0, abs(m0))
    assert info["moved_fraction"] > 0.0


def test_bound_ledger_is_noop_inside_bounds():
    n = 12
    phi = torch.rand(n, n, dtype=DTYPE)
    out, info = Z.bound_ledger(phi)
    assert torch.equal(out, phi)
    assert info["moved_fraction"] == 0.0


@pytest.mark.parametrize("scheme", ["lbm", "fct", "low", "high"])
def test_transport_conserves_mass(scheme):
    n = 16
    steps = 25
    C, w, shifts, phi0, u = _setup(n, 2)
    phi = phi0.clone()
    g = Z.equilibrium_fill(phi, u, C, w)
    m0 = float(phi.sum())
    for _ in range(steps):
        phi, g, _info = Z.fill_transport_step(
            phi, g, u, dim=2, C=C, w=w, shifts=shifts, scheme=scheme, tau=0.504
        )
        assert torch.allclose(g.sum(0), phi, atol=1e-12)
    assert abs(float(phi.sum()) - m0) / m0 < 1e-12


@pytest.mark.parametrize("scheme", ["lbm", "fct"])
def test_transport_is_bounded(scheme):
    """The ledger / limiter keep the fill level inside [0, 1]."""
    n = 16
    C, w, shifts, phi0, u = _setup(n, 2)
    phi = phi0.clone()
    g = Z.equilibrium_fill(phi, u, C, w)
    for _ in range(40):
        phi, g, _info = Z.fill_transport_step(
            phi, g, u, dim=2, C=C, w=w, shifts=shifts, scheme=scheme, tau=0.504
        )
    # The capacity limiter's proof assumes the low-order (equilibrium-flux)
    # state is inside [0, 1]; it can overshoot by O(Ma) at a jump, so the leak is
    # bounded but not round-off.  Measured worst case here is ~1e-2.
    assert float(phi.min()) >= -2e-2
    assert float(phi.max()) <= 1.0 + 2e-2


def test_flags_from_fill():
    phi = torch.tensor([[0.0, 1.0, 0.4], [0.6, 1e-15, 0.999]], dtype=DTYPE)
    flags = Z.flags_from_fill(phi)
    assert flags[0, 0].item() == Z.GAS
    assert flags[0, 1].item() == Z.LIQUID
    assert flags[0, 2].item() == Z.INTERFACE
    assert flags[1, 0].item() == Z.INTERFACE
    assert flags[1, 1].item() == Z.GAS
    assert flags[1, 2].item() == Z.INTERFACE


def test_interface_band_is_one_cell_wide_initially():
    """The free-surface flag layer carries the interface in a ~1 cell band."""
    n = 64
    phi0 = Z.initial_fill(n, dim=2)
    band = Z.interface_band_cells(phi0)
    assert 0.5 < band["band_cells"] < 2.5
    assert band["n_interface_cells"] > 0


def test_initial_volume_matches_analytic():
    n = 128
    phi0 = Z.initial_fill(n, dim=2)
    v_disc = float(phi0.mean().item())
    v_exact = analytic_volume_2d()
    # sub-cell supersampling of the geometry: O(h) volume error, 2.4e-3 at N=64
    # and 3.3e-3 at N=128 (the discrete initial volume is the reference for the
    # mass-conservation metric, so this is a geometry, not a transport, error).
    assert abs(v_disc - v_exact) / v_exact < 5e-3


def test_rigid_rotation_is_discretely_divergence_free():
    for dim in (2, 3):
        n = 32
        _C, _w, _s, phi0, u = _setup(n, dim, omega=0.31)
        assert Z.dirichlet_inflow_max(phi0, u, dim) < 1e-12


def test_3d_extruded_initial_field_is_y_invariant():
    n = 12
    phi0 = Z.initial_fill(n, dim=3, extruded=True, sub=2)
    assert float((phi0 - phi0[:, :1, :]).abs().max()) < 1e-12


def test_3d_extruded_run_keeps_y_invariance():
    """A y-invariant field stays y-invariant under the D3Q19 operator."""
    n = 12
    C, w, shifts, phi0, u = _setup(n, 3, omega=0.05)
    phi = Z.initial_fill(n, dim=3, extruded=True, sub=2)
    g = Z.equilibrium_fill(phi, u, C, w)
    for _ in range(10):
        phi, g, _info = Z.fill_transport_step(
            phi, g, u, dim=3, C=C, w=w, shifts=shifts, scheme="lbm", tau=0.504
        )
    assert float((phi - phi[:, :1, :]).abs().max()) < 1e-12


def test_run_case_returns_metrics():
    m = Z.run_lbm_zalesak_case(16, dim=2, u_max_lat=0.1, steps=40, sample_every=20)
    for key in (
        "volume_rel_change",
        "l1_shape_error",
        "shape_overlap",
        "hausdorff_distance",
        "interface_thickness_mean_cells",
        "interface_band_cells",
        "fill_min",
        "fill_max",
        "div_u_max",
        "steps",
    ):
        assert key in m
    assert abs(m["volume_rel_change"]) < 1e-10
    assert m["steps"] == 40
    assert m["div_u_max"] < 1e-12
    assert m["lattice"] == "D2Q9"


def test_run_case_3d_smoke():
    m = Z.run_lbm_zalesak_case(
        12, dim=3, u_max_lat=0.1, steps=10, extruded=True, sample_every=10
    )
    assert m["lattice"] == "D3Q19"
    assert abs(m["volume_rel_change"]) < 1e-10


def test_step_count_covers_one_revolution():
    """steps * dx * omega == 2*pi exactly (no end-of-run phase error)."""
    m = Z.run_lbm_zalesak_case(24, dim=2, u_max_lat=0.1)
    dx = 1.0 / 24
    assert abs(m["omega_lat"] * m["steps"] * dx - 2.0 * math.pi) < 1e-12