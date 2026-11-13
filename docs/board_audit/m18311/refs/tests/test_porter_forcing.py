"""Unit tests for the Porter (2012) explicit-forcing primitives.

The moment identities are the analytic backbone of the scheme: if the force PDF
does not satisfy them the discretisation cannot represent ``F`` in the momentum
and energy equations, and the Poiseuille/layered benchmarks drift.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.porter_forcing import (
    C,
    CS2,
    CX,
    CY,
    OPP,
    W,
    equilibrium,
    gradient_field,
    isotropic_gradient_weights,
    isotropy_moments,
    momentum,
    physical_velocity,
    porter_collide,
    porter_force_pdf,
    viscosity_weighted_ueq,
)


def _field(rho0=1.0, ux0=0.05, uy0=-0.02, shape=(6, 5)):
    rho = np.full(shape, rho0)
    ux = np.full(shape, ux0)
    uy = np.full(shape, uy0)
    return rho, ux, uy


def test_weights_and_stencil_conventions():
    assert np.isclose(W.sum(), 1.0)
    assert np.isclose(np.sum(W * CX**2), CS2)
    for i in range(9):
        assert np.array_equal(OPP[OPP[i]], i)
        assert np.allclose(C[OPP[i]], -C[i])


def test_equilibrium_moments_and_mass_conservation():
    """Zeroth moment must be exactly ``rho`` (the ``cu^2 - cs^2 u^2`` form).

    A regression guard: the ``(cu^2 - u^2)`` variant leaks mass at O(u^2) per
    step (measured 33 % over 4000 steps in the layered benchmark).
    """
    rho, ux, uy = _field()
    feq = equilibrium(rho, ux, uy)
    assert np.allclose(feq.sum(axis=0), rho, rtol=0, atol=1e-14)
    jx = np.tensordot(CX, feq, axes=(0, 0))
    jy = np.tensordot(CY, feq, axes=(0, 0))
    assert np.allclose(jx, rho * ux, atol=1e-14)
    assert np.allclose(jy, rho * uy, atol=1e-14)
    # second moment: cs^2 rho I + rho u u
    mm = np.einsum("i,ijk->ijk", CX, feq)
    xx = np.einsum("i,ijk->jk", CX * CX, feq)
    xy = np.einsum("i,ijk->jk", CX * CY, feq)
    yy = np.einsum("i,ijk->jk", CY * CY, feq)
    assert np.allclose(xx, CS2 * rho + rho * ux * ux, atol=1e-13)
    assert np.allclose(xy, rho * ux * uy, atol=1e-13)
    assert np.allclose(yy, CS2 * rho + rho * uy * uy, atol=1e-13)
    assert mm.shape == feq.shape


def test_force_pdf_moments():
    """F_i must carry exactly the force: sum F_i = 0, sum F_i c_i = F."""
    rho, ux, uy = _field()
    fx = np.full(rho.shape, 3.0e-3)
    fy = np.full(rho.shape, -2.0e-3)
    feq = equilibrium(rho, ux, uy)
    fpdf = porter_force_pdf(fx, fy, feq, rho, ux, uy)
    assert np.allclose(fpdf.sum(axis=0), 0.0, atol=1e-18)
    assert np.allclose(np.tensordot(CX, fpdf, axes=(0, 0)), fx, atol=1e-19)
    assert np.allclose(np.tensordot(CY, fpdf, axes=(0, 0)), fy, atol=1e-19)
    # second moment at rest (exact): F u + u F = 0
    feq0 = equilibrium(rho, np.zeros_like(ux), np.zeros_like(uy))
    fpdf0 = porter_force_pdf(fx, fy, feq0, rho, np.zeros_like(ux), np.zeros_like(uy))
    for cc in (CX * CX, CX * CY, CY * CY):
        assert np.allclose(np.einsum("i,ijk->jk", cc, fpdf0), 0.0, atol=1e-19)
    # moving frame: second moment is F u + u F up to the O(u^3/cs^2) lattice term
    xx = np.einsum("i,ijk->jk", CX * CX, fpdf)
    xy = np.einsum("i,ijk->jk", CX * CY, fpdf)
    yy = np.einsum("i,ijk->jk", CY * CY, fpdf)
    scale = max(np.abs(fx).max(), np.abs(fy).max()) * max(np.abs(ux).max(), np.abs(uy).max())
    assert np.allclose(xx, 2.0 * fx * ux, atol=1e-1 * scale)
    assert np.allclose(xy, fx * uy + fy * ux, atol=1e-1 * scale)
    assert np.allclose(yy, 2.0 * fy * uy, atol=1e-1 * scale)


@pytest.mark.parametrize(
    "variant,expected",
    [("doc_literal", 0.5), ("doc_seq", 1.0), ("guo", None)],
)
def test_collision_momentum_injection(variant, expected):
    """Momentum added by one collision step.

    ``doc_literal`` adds ``0.5 F`` (which is what the recovered velocity's
    ``+0.5 F`` expects), ``doc_seq`` adds ``F`` (so it over-counts by 1.5x and
    the benchmark shows a 50 % error), ``guo`` adds ``(1 - 1/(2 tau)) F``.
    """
    rho, ux, uy = _field()
    feq = equilibrium(rho, ux, uy)
    fx = np.full(rho.shape, 2.0e-4)
    fy = np.zeros(rho.shape)
    fpdf = porter_force_pdf(fx, fy, feq, rho, ux, uy)
    tau = 1.0
    f_new = porter_collide(feq, feq, fpdf, tau, variant=variant)
    jx = np.tensordot(CX, f_new - feq, axes=(0, 0))
    coef = 1.0 - 0.5 / tau if expected is None else expected
    assert np.allclose(jx, coef * fx, rtol=1e-12, atol=1e-20)
    # mass is never injected
    assert np.allclose(f_new.sum(axis=0), feq.sum(axis=0), atol=1e-18)


def test_physical_velocity_includes_half_force():
    rho, ux, uy = _field()
    jx = rho * ux
    jy = rho * uy
    fx = np.full(rho.shape, 0.01)
    fy = np.zeros(rho.shape)
    ux_out, uy_out = physical_velocity(jx, jy, fx, fy, rho)
    assert np.allclose(ux_out, ux + 0.5 * fx / rho)
    assert np.allclose(uy_out, uy)


def test_viscosity_weighted_ueq_single_and_two_component():
    rho, ux, uy = _field(rho0=2.0)
    fx = np.full(rho.shape, 0.02)
    fy = np.zeros(rho.shape)
    jx, jy = rho * ux, rho * uy
    # single component: u_eq = u + F/(2 rho)
    ueqx, ueqy = viscosity_weighted_ueq([jx], [jy], [fx], [fy], [rho], [0.8])
    assert np.allclose(ueqx, ux + 0.5 * fx / rho)
    # equal relaxation times: identical to the physical velocity of that component
    ueqx2, _ = viscosity_weighted_ueq([jx, jx], [jy, jy], [fx, fx], [fy, fy], [rho, rho], [0.8, 0.8])
    assert np.allclose(ueqx2, ux + 0.5 * fx / rho)
    # different relaxation times: rho/tau weighted closed form
    taus = [1.0, 0.6]
    rho2 = rho * 0.5
    ueqx3, _ = viscosity_weighted_ueq([jx, rho2 * ux], [jy, rho2 * uy], [fx, fx * 0.5], [fy, fy], [rho, rho2], taus)
    num = (jx + 0.5 * fx) / taus[0] + (0.5 * rho * ux + 0.25 * fx) / taus[1]
    den = rho / taus[0] + rho2 / taus[1]
    assert np.allclose(ueqx3, num / den)


def m4_norm(off, w):
    """Gradient normalisation ``sum w cx^2`` (1.0 is the force-kernel convention)."""
    return float(np.sum(w * off[:, 0] ** 2))


def test_isotropic_stencil_invariants():
    # order 2 (the classic kernel) is normalised but anisotropic at 4th order
    # order 2 is the classic D2Q9 kernel: normalised by sum w = 1, hence
    # sum w cx^2 = cs^2 = 1/3 (the inventory's 18x force-constant convention) and
    # it is already 4th-order isotropic, failing only at 6th order (A6 = 0.6)
    off2, w2 = isotropic_gradient_weights(2)
    m2 = isotropy_moments(off2, w2)
    assert np.isclose(m2["sum_w_cx2"], CS2)
    assert np.isclose(m2["A4"], 1.0)
    assert np.isclose(m2["A6"], 0.6)
    # order 4 is the same stencil rescaled by 3 to sum w cx^2 = 1 (identical ratios)
    assert np.isclose(m4_norm(off2, w2) * 3.0, 1.0)
    # order 4: reconstruction reproduces the documented reference invariants
    off4, w4 = isotropic_gradient_weights(4)
    m4 = isotropy_moments(off4, w4)
    assert np.isclose(m4["sum_w"], 5.0 / 3.0)  # documented reference value
    assert np.isclose(m4["sum_w_cx2"], 1.0)
    assert np.isclose(m4["A4"], 1.0)
    assert np.isclose(m4["A6"], 0.6)  # documented reference value
    assert np.isclose(m4["A8"], 3.0 / 7.0, rtol=1e-6)  # documented reference value
    # order 6: isotropic through 6th order
    off6, w6 = isotropic_gradient_weights(6)
    m6 = isotropy_moments(off6, w6)
    assert np.isclose(m6["sum_w_cx2"], 1.0)
    assert np.isclose(m6["A4"], 1.0)
    assert np.isclose(m6["A6"], 1.0)


@pytest.mark.parametrize("order", [2, 4, 6])
def test_gradient_exact_on_linear_field(order):
    off, w = isotropic_gradient_weights(order)
    n = 16
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    a, b = 0.37, -1.21
    field = a * xx + b * yy
    m = isotropy_moments(off, w)
    gx, gy = gradient_field(field, off, w)
    # periodic wrap perturbs the stencil-width border; the interior is exact and
    # the stencil's own normalisation sum w cx^2 multiplies the gradient
    assert np.allclose(gx[3:-3, 3:-3], a * m["sum_w_cx2"], atol=1e-12)
    assert np.allclose(gy[3:-3, 3:-3], b * m["sum_w_cx2"], atol=1e-12)


def test_forced_channel_reproduces_analytic_poiseuille():
    """End-to-end smoke test on the shipped API (small grid, ~2 s)."""
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "xval",
                                    "XVAL-LBM-PORTER-EXPLICIT-FORCING-4TH-ORDER"))
    import got_driver as G

    ny, tau, g = 18, 1.0, 5e-6
    out = G.run_channel(ny, [tau], [np.ones((ny, 1))], [g], 4000, scheme="porter", variant="doc_literal")
    y = np.arange(1, ny - 1) - 0.5
    h = ny - 2
    u_exact = G.layered_stokes(y, h, h / 2, CS2 * (tau - 0.5), CS2 * (tau - 0.5), g)
    err = np.max(np.abs(out["ux"][1:-1] - u_exact)) / np.max(np.abs(u_exact))
    assert err < 0.03, f"forced-channel profile error {err:.3e}"
    assert out["mass_drift"] < 1e-10