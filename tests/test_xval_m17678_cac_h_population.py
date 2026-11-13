"""Regression tests for the XVAL m17678 TCLB-parity conservative AC kernel."""

import math

import pytest
import torch

from tensorlbm import cac_lbm2d as A
from tensorlbm.phasefield import cac_h_population as B

DT = torch.float64


def test_counter_term_is_the_documented_anti_diffusion_source():
    st = B.get_stencil("d3q27")
    phi = B.sphere_phi((16, 16, 16), 5.0, 3.0)
    n_hat, _ = B.interface_normal(phi, st, min_gradient=1e-8)
    W = 3.0
    F = B.counter_term(phi, n_hat, W, st)
    tmp1 = (1.0 - 4.0 * (phi - 0.5) ** 2) / W
    ref = torch.empty_like(F)
    for q in range(27):
        cx, cy, cz = [int(v) for v in st.C[q].tolist()]
        ref[q] = st.W[q] * tmp1 * (cx * n_hat[0] + cy * n_hat[1] + cz * n_hat[2])
    assert float((F - ref).abs().max()) == 0.0
    assert float(F.sum(dim=0).abs().max()) < 1e-14  # zero net mass


def test_static_droplet_conserves_phase_and_stays_put():
    sim = B.CACHPopulation((64, 64), width=4.0, mobility=1.0 / 3.0, stencil="d2q9",
                           min_gradient=1e-8, dtype=DT)
    sim.initialize(B.circle_phi((64, 64), 12.0, 4.0))
    m0 = sim.mass()
    r0 = math.sqrt(float((sim.phi > 0.5).sum()) / math.pi)
    sim.run(50)
    assert abs(sim.mass() - m0) / m0 < 1e-12
    r1 = math.sqrt(float((sim.phi > 0.5).sum()) / math.pi)
    assert abs(r1 - r0) / r0 < 0.02


def test_arm_a_and_arm_b_agree_in_the_static_limit():
    """The two h-population kernels differ only by the d(phi u)/dt term."""
    st = B.get_stencil("d2q9")
    phi0 = B.circle_phi((48, 48), 10.0, 4.0)
    simB = B.CACHPopulation((48, 48), width=4.0, mobility=1.0 / 3.0, stencil="d2q9",
                            min_gradient=1e-8, dtype=DT)
    simB.initialize(phi0)
    simB.step()
    wv = st.W.view(-1, 1, 1).to(DT)
    cf = [st.C[:, 0].view(-1, 1, 1).to(DT), st.C[:, 1].view(-1, 1, 1).to(DT)]
    zero = (torch.zeros_like(phi0), torch.zeros_like(phi0))
    f = A.f_equilibrium(phi0, zero, wv, cf)
    tau_f = 0.5 + 3.0 * (1.0 / 3.0)
    nb = A.shifted_neighbors(phi0, axes="pp")
    grad = A.iso_gradient_2d(phi0, axes="pp", nb=nb)
    gnorm = (grad[0] ** 2 + grad[1] ** 2).sqrt().clamp(min=1e-12)
    n_hat = (grad[0] / gnorm, grad[1] / gnorm)
    lam = A.lambda_phi(phi0, 4.0)
    fsrc = A.f_source((zero[0], zero[1]), lam, n_hat, tau_f, wv, cf)
    f = f + (1.0 / tau_f) * (A.f_equilibrium(phi0, zero, wv, cf) - f) + fsrc
    out = torch.empty_like(f)
    out[0] = f[0]
    c = st.C
    for q in range(1, 9):
        out[q] = A._shift(f[q], -int(c[q, 1]), -int(c[q, 0]), "pp")
    rel = float((out.sum(0) - simB.phi).norm() / simB.phi.norm())
    assert rel < 1e-6, rel
