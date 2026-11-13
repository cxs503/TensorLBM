"""Element-level regression tests for the corotational 2-node beam
(``tensorlbm.corotational_beam``), backing the cross-validation report
docs/XVAL-LBM-COROTATIONAL-2NODE-BEAM-NEWMARK-NEWTON.md.

These are the properties the XVAL arms rely on:
  * objectivity: a rigid rotation produces no internal force,
  * g = dU/dq (energy consistency of the TL formulation),
  * K_T = dg/dq (exact tangent, symmetric),
  * the LIFE transplant deviates from dU/dq (documented defect, measured),
  * small-deflection tip deflection = PL^3/(3EI),
  * first clamped-free frequency = (1.8751..)^2 sqrt(EI/(rho A L^4)),
  * Newmark: a non-dissipative run keeps the amplitude,
  * mass conservation: total mass = rho A L.
"""

import numpy as np
import pytest

from tensorlbm.corotational_beam import (
    CorotationalBeam2D,
    euler_bernoulli_frequency,
    straight_beam,
    wrap_angle,
)

CLAMP = [(0, "x"), (0, "y"), (0, "theta")]
A_SECTION = 1.2e5      # A/I = 1.2e5  ->  h/L = 0.01 (slender)


def _beam(n=8, form="consistent", rho=1.0, bc=CLAMP):
    return straight_beam(1.0, n, E=1.0, A=A_SECTION, I=1.0, rho=rho, bc=bc, formulation=form)


def _grad_u(b, q, eps=1e-6):
    eye = np.eye(b.n_dof)
    return np.array(
        [(b.strain_energy(q + eps * eye[i]) - b.strain_energy(q - eps * eye[i])) / (2 * eps)
         for i in range(b.n_dof)]
    )


def test_wrap_angle_range():
    a = np.array([0.0, np.pi, -np.pi, 3 * np.pi / 2, 5 * np.pi])
    w = wrap_angle(a)
    assert np.all(w <= np.pi + 1e-12) and np.all(w > -np.pi - 1e-12)


def test_total_mass_and_geometry():
    b = _beam(n=4)
    assert b.n_elem == 4 and b.n_dof == 15
    assert b.total_mass == pytest.approx(1.0 * A_SECTION * 1.0, rel=1e-12)
    assert np.allclose(b.L0, 0.25)


def test_rigid_body_rotation_gives_zero_internal_force():
    """Objectivity: q = rigid rotation of the initial geometry -> g ~ 0."""
    b = _beam()
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    q = np.zeros(b.n_dof)
    for i in range(b.n_node):
        q[3 * i: 3 * i + 2] = R @ b.x0[i] - b.x0[i]
        q[3 * i + 2] = th
    assert np.abs(b.internal_forces(q)).max() < 1e-9


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_internal_force_is_gradient_of_strain_energy(seed):
    """g = dU/dq for the energy-consistent formulation (machine precision)."""
    rng = np.random.default_rng(seed)
    b = _beam()
    q = np.zeros(b.n_dof)
    q[3:] = rng.normal(scale=0.05, size=b.n_dof - 3)
    g = b.internal_forces(q)
    gn = _grad_u(b, q)
    assert np.linalg.norm(g - gn) / np.linalg.norm(g) < 1e-8


def test_tangent_is_exact_derivative_and_symmetric():
    """K_T = dg/dq (central differences) and K_T = K_T^T."""
    rng = np.random.default_rng(7)
    b = _beam()
    q = np.zeros(b.n_dof)
    q[3:] = rng.normal(scale=0.05, size=b.n_dof - 3)
    K = b.tangent_stiffness(q)
    eps = 1e-6
    eye = np.eye(b.n_dof)
    Kfd = np.column_stack(
        [
            (b.internal_forces(q + eps * eye[i]) - b.internal_forces(q - eps * eye[i])) / (2 * eps)
            for i in range(b.n_dof)
        ]
    )
    assert np.linalg.norm(K - Kfd) / np.linalg.norm(K) < 1e-8
    assert np.abs(K - K.T).max() / np.abs(K).max() < 1e-12


def test_life_transplant_is_not_a_gradient():
    """The OSS transplant uses (M1+M2)/L0, so its force vector is not dU/dq.

    Measured, not asserted as acceptable: the deviation must be non-zero and
    bounded (this is the defect quantified in the report).
    """
    rng = np.random.default_rng(3)
    b = _beam(form="life")
    q = np.zeros(b.n_dof)
    q[3:] = rng.normal(scale=0.05, size=b.n_dof - 3)
    g = b.internal_forces(q)
    gn = _grad_u(b, q)
    dev = np.linalg.norm(g - gn) / np.linalg.norm(g)
    assert 1e-6 < dev < 5e-2


def test_life_and_tl_agree_for_inextensible_small_strain():
    """At (L-L0)/L0 -> 0 the two formulations must coincide."""
    b1, b2 = _beam(form="consistent"), _beam(form="life")
    q = np.zeros(b1.n_dof)
    q[3:] = 1e-6 * np.arange(b1.n_dof - 3)
    g1, g2 = b1.internal_forces(q), b2.internal_forces(q)
    assert np.linalg.norm(g1 - g2) / np.linalg.norm(g1) < 1e-8


@pytest.mark.parametrize("form", ["consistent", "life"])
def test_small_deflection_matches_euler_bernoulli(form):
    """tip deflection = PL^3/(3EI), tip rotation = PL^2/(2EI)."""
    P = 0.004
    b = _beam(n=8, form=form)
    q, rec = b.solve_static({b.n_node - 1: (0.0, -P, 0.0)}, n_steps=1)
    assert rec.converged
    assert b.tip_deflection(q)[1] == pytest.approx(-P / 3.0, rel=2e-4)
    assert b.tip_rotation(q) == pytest.approx(-P / 2.0, rel=2e-4)


def test_first_natural_frequency():
    """omega_1 = 3.5160 * sqrt(EI/(rho A L^4)) for a clamped-free beam."""
    kl2 = euler_bernoulli_frequency(3)
    assert kl2[0] == pytest.approx(3.51601527, rel=1e-6)
    assert kl2[1] == pytest.approx(22.03449156, rel=1e-5)
    assert kl2[2] == pytest.approx(61.69721441, rel=1e-5)
    b = _beam(n=16, rho=1.0)
    w1 = b.mode_frequencies(1)[0]
    assert w1 == pytest.approx(kl2[0] * np.sqrt(1.0 / (1.0 * A_SECTION)), rel=3e-2)


def test_newmark_undamped_amplitude_preserved():
    """gamma=1/2, beta=1/4 -> no algorithmic damping: amplitude <= 1 % change."""
    b = _beam(n=8, rho=1.0)
    w1 = b.mode_frequencies(1)[0]
    T1 = 2 * np.pi / w1
    q0, _ = b.solve_static({b.n_node - 1: (0.0, -0.004, 0.0)}, n_steps=1)
    res = b.solve_newmark(dt=T1 / 200, n_steps=400, q0=q0)
    tip = res["q_hist"][:, 3 * (b.n_node - 1) + 1]
    amp = 0.5 * (tip[:40].max() - tip[:40].min())
    amp_end = 0.5 * (tip[-40:].max() - tip[-40:].min())
    assert abs(amp_end / amp - 1.0) < 0.01
    assert res["converged"]


def test_pure_moment_gives_circular_arc():
    """phi = ML/EI = pi/2 -> R = L/phi, tip = (R sin phi, R(1-cos phi))."""
    phi = np.pi / 2
    b = _beam(n=16)
    q, _ = b.solve_static({b.n_node - 1: (0.0, 0.0, phi * 1.0)}, n_steps=20)
    pos = b.node_position(q)
    R = 1.0 / phi
    assert pos[0] == pytest.approx(R * np.sin(phi), rel=1e-3)
    assert pos[1] == pytest.approx(R * (1 - np.cos(phi)), rel=1e-3)


def test_bad_input_validation():
    with pytest.raises(ValueError):
        CorotationalBeam2D([[0.0, 0.0]], E=1.0, A=1.0, I=1.0, rho=1.0)
    with pytest.raises(ValueError):
        straight_beam(1.0, 2, E=1.0, A=1.0, I=1.0, rho=1.0, formulation="timoshenko")