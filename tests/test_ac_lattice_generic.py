"""Acceptance tests for :mod:`tensorlbm.phasefield.ac_lattice_generic` (m19468).

Four independent claims are pinned here, each with an anchor that is *not* the
code under test:

* **T1 lattice admissibility** - the D3Q15 and D3Q19 velocity sets used by the
  scheme satisfy the isotropy conditions the CAC/HCH derivation assumes
  (``sum w = 1``, ``sum w c_a c_b = cs^2 delta_ab``, fourth-rank isotropy
  ``sum w c_x^4 = 3 cs^4``) and the discrete Laplacian weights are
  zero-sum (a constant field must have zero Laplacian).
* **T2 derivative stencils** - on a smooth periodic field the gradient is exact
  for linear data and the Laplacian converges at second order to the analytic
  value on both lattices (the D3Q15 Laplacian is a *different* 15-point stencil,
  so this is not a tautology: it is the only check that would catch a wrong
  D3Q15 coefficient set).
* **T3 D3Q19 equivalence** - the lattice-parametric path reproduces the frozen,
  verified :mod:`tensorlbm.cac_lbm` population-for-population (machine
  precision) when asked for D3Q19, i.e. the D3Q15 arm is a lattice change and
  not a physics change.
* **T4 D3Q15 integrity** - conservative-AC phase inventory is held to round-off
  and the static-droplet run stays finite with small spurious currents.

Anchor provenance for the D3Q15 Laplacian coefficients:
lbmpy 2.0 ``phasefield_allen_cahn/derivatives.py::laplacian_symbolic`` (the
``Q == 15`` branch) gives ``-32/9`` (centre), ``7/18`` (axis), ``11/72``
(corner); the D3Q19 branch gives ``-4``, ``1/3``, ``1/6``.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm import cac_lbm
from tensorlbm.phasefield.ac_lattice_generic import (
    ACPhaseFieldSim,
    init_phi_droplet,
    iso_gradient,
    iso_laplacian,
    lattice_spec,
    shifted_neighbors,
)

CS2 = 1.0 / 3.0

KW = dict(
    rho_l=1.0,
    rho_g=0.1,
    mu_l=0.1,
    mu_g=0.01,
    sigma=0.01,
    width=3.0,
    mobility=0.05,
)


@pytest.fixture(autouse=True)
def _one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _periodic_field(n: int, k: float, dtype=torch.float64) -> torch.Tensor:
    s = torch.arange(n, dtype=dtype).view(-1, 1, 1)
    y = torch.arange(n, dtype=dtype).view(1, -1, 1)
    x = torch.arange(n, dtype=dtype).view(1, 1, -1)
    return torch.sin(k * x) + torch.cos(k * y) + torch.sin(k * s)


# ---------------------------------------------------------------------------
# T1 lattice admissibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["D3Q15", "D3Q19"])
def test_lattice_moments(name):
    spec = lattice_spec(name)
    W = spec.W
    C = spec.C.to(torch.float64)
    assert abs(float(W.sum()) - 1.0) < 1e-15
    for a in range(3):
        for b in range(3):
            mom = float((W * C[:, a] * C[:, b]).sum())
            target = CS2 if a == b else 0.0
            assert abs(mom - target) < 1e-15, (a, b, mom)
    # fourth-rank isotropy: sum w c_x^4 = 3 cs^4
    assert abs(float((W * C[:, 0] ** 4).sum()) - 3.0 * CS2**2) < 1e-15
    assert abs(float((W * C[:, 0] ** 2 * C[:, 1] ** 2).sum()) - CS2**2) < 1e-15
    # the discrete Laplacian annihilates constants
    assert abs(float(spec.lap.sum())) < 1e-14
    # gradient normalisation: sum_{i!=0} 3 w_i |c_ix| = 1
    assert math.isclose(float((3.0 * W[1:] * C[1:, 0].abs()).sum()), 1.0, rel_tol=1e-14)


def test_d3q15_laplacian_coefficients_match_lbmpy():
    """Literal coefficient set lbmpy derives for the Q == 15 branch."""
    spec = lattice_spec("D3Q15")
    lap = spec.lap.tolist()
    assert lap[0] == pytest.approx(-32.0 / 9.0, rel=0, abs=0)
    axis = [lap[i] for i, c in enumerate(spec.C.tolist()) if sum(abs(v) for v in c) == 1]
    corner = [lap[i] for i, c in enumerate(spec.C.tolist()) if sum(abs(v) for v in c) == 3]
    assert axis and all(v == pytest.approx(7.0 / 18.0, abs=0) for v in axis)
    assert len(corner) == 8 and all(v == pytest.approx(11.0 / 72.0, abs=0) for v in corner)


def test_velocity_weighted_laplacian_differs_on_d3q15_but_not_d3q19():
    """lap_scheme='weights' must collapse onto 'lbmpy' for D3Q19 only."""
    lb = lattice_spec("D3Q19", lap_scheme="lbmpy")
    wt = lattice_spec("D3Q19", lap_scheme="weights")
    assert torch.allclose(lb.lap, wt.lap, atol=1e-15)
    lb15 = lattice_spec("D3Q15", lap_scheme="lbmpy")
    wt15 = lattice_spec("D3Q15", lap_scheme="weights")
    assert float((lb15.lap - wt15.lap).abs().max()) > 1e-3


# ---------------------------------------------------------------------------
# T2 derivative stencils vs analytic data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["D3Q15", "D3Q19"])
def test_gradient_exact_on_linear_field(name):
    spec = lattice_spec(name)
    n = 12
    x = torch.arange(n, dtype=torch.float64).view(1, 1, -1)
    phi = 0.25 * x + 0.5 * torch.arange(n, dtype=torch.float64).view(1, -1, 1)
    g = iso_gradient(shifted_neighbors(phi, spec), spec)
    # periodic data: compare only where the linear ramp wraps nowhere -> interior
    assert torch.allclose(g[0][:, :, 2:-2], torch.full_like(g[0][:, :, 2:-2], 0.25), atol=1e-12)
    assert torch.allclose(g[1][:, 2:-2, :], torch.full_like(g[1][:, 2:-2, :], 0.5), atol=1e-12)


@pytest.mark.parametrize("name", ["D3Q15", "D3Q19"])
def test_laplacian_second_order_convergence(name):
    """Discrete Laplacian of sin(kx) -> -k^2 sin(kx) with O(h^2) error."""
    spec = lattice_spec(name)
    errs = []
    for n in (24, 48, 96):
        # periodic data: an integer number of wavelengths across the box, so the
        # only error is the O((k h)^2) truncation of the stencil
        k = 2.0 * math.pi * 2.0 / n
        phi = _periodic_field(n, k)
        lap = iso_laplacian(phi, shifted_neighbors(phi, spec), spec)
        err = float((lap + k * k * phi).abs().max()) / (k * k)
        errs.append(err)
    order = math.log(errs[0] / errs[1]) / math.log(2.0)
    order2 = math.log(errs[1] / errs[2]) / math.log(2.0)
    assert order > 1.8 and order2 > 1.8, (errs, order, order2)
    assert errs[-1] < 1e-2


@pytest.mark.parametrize("name", ["D3Q15", "D3Q19"])
def test_laplacian_isotropy_along_oblique_direction(name):
    """Plane wave along (1,2,3): both lattices must stay second-order accurate.

    A separable field cannot separate the lattices (for phi = phi(x) the D3Q15
    and D3Q19 stencils collapse onto the same 1-D second difference), so the
    wave vector is deliberately oblique.  The D3Q15 stencil is expected to be
    slightly worse than D3Q19 here -- its fourth-order error constant differs,
    which is exactly what the ``lap_scheme`` sensitivity arm measures.
    """
    errs = []
    for n in (48, 96):
        base = 2.0 * math.pi / n
        qx, qy, qz = 2 * base, 4 * base, 6 * base
        s = torch.arange(n, dtype=torch.float64).view(-1, 1, 1)
        y = torch.arange(n, dtype=torch.float64).view(1, -1, 1)
        x = torch.arange(n, dtype=torch.float64).view(1, 1, -1)
        phi = torch.sin(qx * x + qy * y + qz * s)
        q2 = qx * qx + qy * qy + qz * qz
        spec = lattice_spec(name)
        lap = iso_laplacian(phi, shifted_neighbors(phi, spec), spec)
        errs.append(float((lap + q2 * phi).abs().max()) / (q2 * float(phi.abs().max())))
    order = math.log(errs[0] / errs[1]) / math.log(2.0)
    assert order > 1.7, (errs, order)
    assert errs[-1] < 0.05, errs


# ---------------------------------------------------------------------------
# T3 machine-level equivalence with the frozen D3Q19 module
# ---------------------------------------------------------------------------


def test_d3q19_path_matches_frozen_cac_lbm():
    shape = (24, 24, 24)
    phi0 = init_phi_droplet(shape, 6.0, 3.0, dtype=torch.float64)
    ref = cac_lbm.CACSim(shape, **KW, dtype=torch.float64)
    ref.initialize(phi0)
    gen = ACPhaseFieldSim(shape, lattice="D3Q19", **KW, dtype=torch.float64)
    gen.initialize(phi0)
    scale = float(ref.f.abs().max())
    for _ in range(10):
        ref.step()
        gen.step()
    assert float((ref.f - gen.f).abs().max()) / scale < 1e-12
    assert float((ref.g - gen.g).abs().max()) / scale < 1e-12


# ---------------------------------------------------------------------------
# T4 D3Q15 integrity: conservation + finite spurious currents
# ---------------------------------------------------------------------------


def test_d3q15_phase_inventory_conserved_and_finite():
    shape = (20, 20, 20)
    phi0 = init_phi_droplet(shape, 5.0, 3.0, dtype=torch.float64)
    sim = ACPhaseFieldSim(shape, lattice="D3Q15", **KW, dtype=torch.float64)
    sim.initialize(phi0)
    total0 = float(phi0.sum())
    for _ in range(60):
        sim.step()
    phi, rho, grad, mu_phi, u, p = sim.macro()
    assert not torch.isnan(sim.f).any()
    assert not torch.isnan(sim.g).any()
    assert abs(float(phi.sum()) - total0) / abs(total0) < 1e-12
    umax = float((u[0] ** 2 + u[1] ** 2 + u[2] ** 2).sqrt().max())
    assert umax < 5e-2, umax
    assert float(phi.min()) > -0.05 and float(phi.max()) < 1.05


@pytest.mark.parametrize("name", ["D3Q15", "D3Q19"])
def test_uniform_phase_is_stationary(name):
    """Control: phi == 1 -> mu grad(phi) == 0 -> nothing may move."""
    shape = (12, 12, 12)
    sim = ACPhaseFieldSim(shape, lattice=name, **KW, dtype=torch.float64)
    sim.initialize(torch.ones(shape, dtype=torch.float64))
    for _ in range(20):
        sim.step()
    phi, rho, grad, mu_phi, u, p = sim.macro()
    assert float(u[0].abs().max()) < 1e-12
    assert float(p.std()) < 1e-12
    assert float(phi.min()) == pytest.approx(1.0, abs=1e-12)


def test_mixed_lattice_walberla_configuration_runs_and_conserves():
    """walberla's Allen-Cahn configuration: phase field D3Q15, hydro D3Q19."""
    shape = (20, 20, 20)
    phi0 = init_phi_droplet(shape, 5.0, 3.0, dtype=torch.float64)
    sim = ACPhaseFieldSim(
        shape, lattice_phase="D3Q15", lattice_hydro="D3Q19", **KW, dtype=torch.float64
    )
    assert sim.label == "D3Q15/D3Q19"
    assert sim.f.shape[0] == 15 and sim.g.shape[0] == 19
    sim.initialize(phi0)
    total0 = float(phi0.sum())
    for _ in range(40):
        sim.step()
    phi, rho, grad, mu_phi, u, p = sim.macro()
    assert not torch.isnan(sim.f).any() and not torch.isnan(sim.g).any()
    assert abs(float(phi.sum()) - total0) / abs(total0) < 1e-12
    assert float((u[0] ** 2 + u[1] ** 2 + u[2] ** 2).sqrt().max()) < 5e-2


def test_unsupported_lattice_raises():
    with pytest.raises(ValueError):
        lattice_spec("D2Q9")