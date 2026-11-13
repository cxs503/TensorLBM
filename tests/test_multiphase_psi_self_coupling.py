"""Tests for the pseudopotential / self-coupling knobs of the Shan-Chen two-component force.

These are the TensorLBM-side enablers of
``XVAL-LBM-SCMP-TWO-COMPONENT-PLANAR-INTERFACE``: the OSS reference
(``lbm-principles-practice/code`` chapter 9) uses ``psi = rho0 (1-exp(-rho/rho0))``
and a per-component self interaction, while the historical kernel here was the
linear-pseudopotential pure-cross force.

The tests cover:
1. backward compatibility - ``psi_fn=None`` / ``g_self=(0,0)`` is bit-identical
   to the historical two-component force (and to passing ``psi_linear``);
2. the discrete algebra - ``F_1 = -G12 psi_1 sum w c psi_2(x-c) - g11 psi_1 sum w c psi_1(x-c)``
   against an explicit numpy reference on a random field;
3. the OSS -> library parameter mapping ``G = -g`` (a forward-gather sum equals
   minus the backward-gather sum on a symmetric stencil, so the mapping is
   exact, not asymptotic);
4. the guard rail: ``sc_form="edm"`` rejects a pseudopotential/self coupling
   instead of silently ignoring it.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tensorlbm.multiphase import (
    C,
    W,
    psi_exp,
    psi_linear,
    sc_two_component_force,
)


def _fields(n=8, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    rho1 = 0.4 + 0.5 * torch.rand(n, n, generator=g, dtype=dtype)
    rho2 = 0.4 + 0.5 * torch.rand(n, n, generator=g, dtype=dtype)
    return rho1, rho2


def _backward_sum(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """sum_i w_i c_i field(x - c_i), explicit loop (the kernel's gather)."""
    w = W.numpy().astype(float)
    c = C.numpy()
    sx = np.zeros_like(field)
    sy = np.zeros_like(field)
    for i in range(1, 9):
        shifted = np.roll(field, (c[i, 1], c[i, 0]), axis=(0, 1))  # field(x - c_i)
        sx += w[i] * c[i, 0] * shifted
        sy += w[i] * c[i, 1] * shifted
    return sx, sy


def test_default_path_is_bit_identical():
    rho1, rho2 = _fields()
    a = sc_two_component_force(rho1, rho2, -2.0)
    b = sc_two_component_force(rho1, rho2, -2.0, psi_fn=psi_linear)
    c = sc_two_component_force(rho1, rho2, -2.0, g_self=(0.0, 0.0))
    for x, y, z in zip(a, b, c):
        assert torch.equal(x, y)
        assert torch.equal(x, z)


def test_force_matches_explicit_neighbour_sum_with_potential_and_self_term():
    rho1, rho2 = _fields()
    g12, g11, g22 = -3.0, 0.7, -0.4
    out = sc_two_component_force(
        rho1, rho2, g12, psi_fn=psi_exp, g_self=(g11, g22)
    )
    p1 = psi_exp(rho1).numpy()
    p2 = psi_exp(rho2).numpy()
    s1x, s1y = _backward_sum(p1)
    s2x, s2y = _backward_sum(p2)
    fx1 = -g12 * p1 * s2x - g11 * p1 * s1x
    fy1 = -g12 * p1 * s2y - g11 * p1 * s1y
    fx2 = -g12 * p2 * s1x - g22 * p2 * s2x
    fy2 = -g12 * p2 * s1y - g22 * p2 * s2y
    np.testing.assert_allclose(out[0].numpy(), fx1, atol=1e-14)
    np.testing.assert_allclose(out[1].numpy(), fy1, atol=1e-14)
    np.testing.assert_allclose(out[2].numpy(), fx2, atol=1e-14)
    np.testing.assert_allclose(out[3].numpy(), fy2, atol=1e-14)


def test_forward_gather_equals_minus_backward_gather():
    """The exactness of the OSS(G = -g) parameter mapping.

    The OSS reference gathers the forward neighbours,
    ``sum_i w_i c_i psi(x+c_i)``.  On the symmetric D2Q9 stencil this equals
    ``-sum_i w_i c_i psi(x-c_i)`` (substitute i -> -i), which is exactly the
    TensorLBM gather with ``G = -g``.  Checking it here keeps the mapping from
    being "fixed" by an ad-hoc factor later.
    """
    rho1, rho2 = _fields(seed=3)
    p1 = psi_exp(rho1).numpy()
    bwd, _ = _backward_sum(p1)
    fwd = -bwd
    w = W.numpy().astype(float)
    c = C.numpy()
    ref = np.zeros_like(p1)
    for i in range(1, 9):
        ref += w[i] * c[i, 0] * np.roll(p1, (-c[i, 1], -c[i, 0]), axis=(0, 1))  # psi(x + c_i)
    np.testing.assert_allclose(ref, fwd, atol=1e-14)


def test_edm_rejects_potential_and_self_coupling():
    rho1, rho2 = _fields()
    with pytest.raises(ValueError, match="edm"):
        sc_two_component_force(rho1, rho2, -2.0, sc_form="edm", psi_fn=psi_exp)
    with pytest.raises(ValueError, match="edm"):
        sc_two_component_force(rho1, rho2, -2.0, sc_form="edm", g_self=(1.0, 0.0))


def test_shape_preserving_potential_guard():
    rho1, rho2 = _fields()
    with pytest.raises(ValueError, match="shape"):
        sc_two_component_force(rho1, rho2, -2.0, psi_fn=lambda r: r[:2, :2])


def test_self_coupling_only_touches_its_own_component():
    rho1, rho2 = _fields(seed=5)
    base = sc_two_component_force(rho1, rho2, -2.0, psi_fn=psi_exp)
    with_self = sc_two_component_force(rho1, rho2, -2.0, psi_fn=psi_exp, g_self=(0.5, 0.0))
    assert torch.equal(base[0], with_self[0]) is False      # component 1 changes
    assert torch.equal(base[2], with_self[2]) is True       # component 2 untouched
    assert torch.equal(base[1], with_self[1]) is False