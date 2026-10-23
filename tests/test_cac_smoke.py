"""Unit + short end-to-end tests for tensorlbm.cac_lbm (CAC two-phase LBM).

Covers: shift semantics (both signs, periodic + non-wrapping), isotropic
stencil analytic identities (Eq.27/28 exact on linear/quadratic fields),
equilibrium moment identities (Sum feq = phi; geq -> (u, p) round-trip),
the 1-D stream-with-walls round trip (junk confinement + full-way
reflection schedule), two short end-to-end runs (periodic droplet, walled
layered channel), the rho-mode body-force form in _macro_fp64, and the
harmonic mu_law default.  Ported from the W9-R2/W10-M1 smoke suite; all
CPU.
"""

import inspect

import pytest
import torch

from tensorlbm import cac_lbm as m
from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.d3q19 import C

torch.manual_seed(0)


def test_shift_field_semantics():
    f = torch.arange(10, dtype=torch.float32)
    g = m._shift_field(f, 0, 3, "p")
    assert torch.equal(g, torch.roll(f, -3, 0)) and g[0].item() == 3.0
    g = m._shift_field(f, 0, -3, "p")
    assert torch.equal(g, torch.roll(f, 3, 0)) and g[9].item() == 6.0
    g = m._shift_field(f, 0, 3, "w")
    assert torch.equal(g[:-3], f[3:]) and torch.equal(g[-3:], torch.zeros(3))
    g = m._shift_field(f, 0, -3, "w")
    assert torch.equal(g[3:], f[:-3]) and torch.equal(g[:3], torch.zeros(3))


def test_iso_gradient_linear_field():
    nz, ny, nx = 6, 7, 8
    z, y, x = torch.meshgrid(
        torch.arange(nz, dtype=torch.float64),
        torch.arange(ny, dtype=torch.float64),
        torch.arange(nx, dtype=torch.float64),
        indexing="ij",
    )
    phi = 3.0 * z + 5.0 * y + 7.0 * x
    nb = m.shifted_neighbors(phi, axes="ppp")
    gx, gy, gz = m.iso_gradient_3d(nb, phi.device)
    inner = (slice(1, -1), slice(1, -1), slice(1, -1))
    assert torch.allclose(gx[inner], torch.full_like(phi[inner], 7.0), atol=1e-12)
    assert torch.allclose(gy[inner], torch.full_like(phi[inner], 5.0), atol=1e-12)
    assert torch.allclose(gz[inner], torch.full_like(phi[inner], 3.0), atol=1e-12)


def test_iso_laplacian_quadratic_field():
    nz, ny, nx = 6, 7, 8
    z, y, x = torch.meshgrid(
        torch.arange(nz, dtype=torch.float64),
        torch.arange(ny, dtype=torch.float64),
        torch.arange(nx, dtype=torch.float64),
        indexing="ij",
    )
    phi2 = 0.5 * z * z + 2.0 * y * y + 3.0 * x * x + 0.7 * z * y - 1.3 * x * z
    nb2 = m.shifted_neighbors(phi2, axes="ppp")
    lap = m.iso_laplacian_3d(phi2, nb2)
    exact = torch.full_like(phi2, 1.0 + 4.0 + 6.0)
    inner = (slice(1, -1), slice(1, -1), slice(1, -1))
    assert torch.allclose(lap[inner], exact[inner], atol=1e-12)


def test_equilibrium_moment_identities():
    wv, c3, cf = m._unit_views(4, 5, 6, torch.float64, torch.device("cpu"))
    u = tuple(t * torch.ones(4, 5, 6, dtype=torch.float64) for t in (0.03, -0.02, 0.01))
    phit = 0.3 * torch.ones(4, 5, 6, dtype=torch.float64)
    feq = m.f_equilibrium(phit, u, wv, cf)
    assert torch.allclose(feq.sum(0), phit, atol=1e-14)
    rho_t = 0.55 * torch.ones(4, 5, 6, dtype=torch.float64)
    p_t = 0.011 * torch.ones(4, 5, 6, dtype=torch.float64)
    w0 = float(m._weights(torch.float64, torch.device("cpu"))[0].item())
    geq = m.g_equilibrium(p_t, rho_t, u, wv, cf, w0)
    grad0 = tuple(torch.zeros_like(rho_t) for _ in range(3))
    F0 = tuple(torch.zeros_like(rho_t) for _ in range(3))
    u_rt, p_rt, _ = m.macroscopic_g(geq, rho_t, grad0, 0.0, F0, w0)
    dev_u = max((u_rt[a] - u[a]).abs().max().item() for a in range(3))
    assert dev_u < 1e-13
    assert (p_rt - p_t).abs().max().item() < 1e-13


def test_stream_with_walls_round_trip():
    """1-D chain along y: shape (1, N, 1), axes 'pwp', walls at y=0 and y=N-1."""
    N = 8
    wall = torch.zeros(1, N, 1, dtype=torch.bool)
    wall[0, 0, 0] = True
    wall[0, N - 1, 0] = True
    fD = torch.zeros(19, 1, N, 1, dtype=torch.float64)
    q_up, q_dn = 3, 4
    assert C[q_up].tolist() == [0, 1, 0] and C[q_dn].tolist() == [0, -1, 0]
    fD[q_up, 0, 2, 0] = 1.0  # packet A: y=2 moving +y
    fD[q_dn, 0, 5, 0] = 1.0  # packet B: y=5 moving -y

    def stream_bounce(fd):
        return bounce_back_cells_3d(m.stream_mixed(fd, axes="pwp"), wall)

    def advance(q, pos):
        pos += 1 if q == q_up else -1
        if pos == N - 1:
            q = q_dn
        elif pos == 0:
            q = q_up
        return q, pos

    qa, ya = q_up, 2
    qb, yb = q_dn, 5
    for t in range(28):
        fD = stream_bounce(fD)
        qa, ya = advance(qa, ya)
        qb, yb = advance(qb, yb)
        tot = fD[q_up].sum().item() + fD[q_dn].sum().item()
        assert abs(tot - 2.0) <= 1e-12, f"mass leak at t={t + 1}"
        nz_up = fD[q_up, 0, :, 0].nonzero().flatten().tolist()
        nz_dn = fD[q_dn, 0, :, 0].nonzero().flatten().tolist()
        want_up = sorted(v for v in (ya if qa == q_up else -1, yb if qb == q_up else -1) if v >= 0)
        want_dn = sorted(v for v in (ya if qa == q_dn else -1, yb if qb == q_dn else -1) if v >= 0)
        assert nz_up == want_up and nz_dn == want_dn, f"schedule at t={t + 1}"
        assert fD.sum().item() == 2.0, f"leak into other directions at t={t + 1}"


@pytest.mark.slow
def test_end_to_end_periodic_droplet():
    sim = m.CACSim(
        (24, 24, 24),
        rho_l=1.0,
        rho_g=0.01,
        mu_l=0.1,
        mu_g=0.1,
        sigma=0.01,
        width=4.0,
        mobility=0.05,
        axes="ppp",
        p0=0.01,
    )
    phi0 = m.init_phi_droplet((24, 24, 24), radius=6.0, width=4.0)
    sim.initialize(phi0)
    diags = m.run(sim, 300, sample_every=100, progress_every=0)
    phi0_int = diags[0]["phi_integral_fluid"]
    drift = abs(diags[-1]["phi_integral_fluid"] - phi0_int) / phi0_int
    assert diags[-1]["nan_f"] == 0 and diags[-1]["nan_g"] == 0
    assert drift < 1e-3
    assert diags[-1]["u_max_fluid"] < 5e-2
    assert -0.05 < diags[-1]["phi_min_fluid"]
    assert diags[-1]["phi_max_fluid"] < 1.05


@pytest.mark.slow
def test_end_to_end_walled_layered_channel():
    shape_ch = (8, 26, 8)
    wall_ch = torch.zeros(shape_ch, dtype=torch.bool)
    wall_ch[:, 0, :] = True
    wall_ch[:, -1, :] = True
    sim2 = m.CACSim(
        shape_ch,
        rho_l=1.0,
        rho_g=0.5,
        mu_l=0.2,
        mu_g=0.02,
        sigma=0.001,
        width=5.0,
        mobility=0.1,
        axes="pwp",
        wall_mask=wall_ch,
    )
    phi0b = m.init_phi_layers(shape_ch, width=5.0, y0=12.5)
    sim2.initialize(phi0b)
    diags2 = m.run(sim2, 120, sample_every=60, progress_every=0)
    assert diags2[-1]["nan_f"] == 0 and diags2[-1]["nan_g"] == 0
    assert diags2[-1]["u_max_fluid"] < 5e-2


def test_library_reuse_no_inline_kernels():
    """Iron rule: the module reuses library stream/bounce, no inline copies."""
    src = inspect.getsource(m)
    assert src.count("bounce_back_cells_3d") >= 3
    assert src.count("stream_d3q19_adapter") >= 2


def test_macro_fp64_honours_gravity_mode():
    shape_i = (12, 12, 12)
    sim3 = m.CACSim(
        shape_i,
        rho_l=1.0,
        rho_g=0.5,
        mu_l=0.2,
        mu_g=0.02,
        sigma=0.01,
        width=4.0,
        mobility=0.05,
        axes="ppp",
        p0=0.01,
        gravity=(0.0, -2.5e-4, 0.0),
        gravity_mode="rho",
    )
    phi0c = m.init_phi_droplet(shape_i, radius=4.0, width=4.0)
    sim3.initialize(phi0c)
    for _ in range(5):
        sim3.step()
    phi, rho, grad, mu_phi, F, u, p, u_star = m._macro_fp64(sim3)
    f64 = torch.float64
    fm64 = sim3.fluid_mask.to(f64)
    rho_fm = rho * fm64
    want = (
        mu_phi * grad[0] + sim3.gravity[0] * rho_fm,
        mu_phi * grad[1] + sim3.gravity[1] * rho_fm,
        mu_phi * grad[2] + sim3.gravity[2] * rho_fm,
    )
    assert all(torch.equal(F[a], want[a]) for a in range(3))
    old_uniform_y = mu_phi * grad[1] + sim3.gravity[1] * fm64
    assert (F[1] - old_uniform_y).abs().max().item() > 0.0


def test_mu_law_default_is_harmonic():
    _ph = torch.tensor([0.5], dtype=torch.float64)
    _v_def = m.viscosity_of_phi(_ph, 1.0, 0.5, 0.2, 0.02).item()
    _v_har = m.viscosity_of_phi(_ph, 1.0, 0.5, 0.2, 0.02, law="harmonic").item()
    _v_lin = m.viscosity_of_phi(_ph, 1.0, 0.5, 0.2, 0.02, law="linear").item()
    _v_stp = m.viscosity_of_phi(_ph, 1.0, 0.5, 0.2, 0.02, law="step").item()
    assert _v_def == _v_har and _v_har != _v_lin and _v_lin != _v_stp
