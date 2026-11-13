"""Tests for the equilibrium-difference extrapolation BC (XVAL port of ``hietwll/LBM_Taichi``).

The rule under test is the OSS one, ``lbm_solver.py:130``:

    f_i(x_b) = f_i^eq(x_b) - f_i^eq(x_1) + f_i(x_1)   ==   f_i^eq(x_b) + f_i^neq(x_1)

so the first test pins the algebra (modular implementation == literal transcription),
the second pins that the ``order=2`` switch is not inert, and the rest pin the
behaviour that the XVAL report relies on (uniform-state preservation, wall planes
and the fact that the BC touches nothing but the boundary plane).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.boundary.equilibrium_extrapolation import (  # noqa: E402
    EquilibriumExtrapolationSpec,
    apply_cavity_extrapolation,
    apply_channel_extrapolation,
    apply_equilibrium_extrapolation,
    fluid_mask_from_walls,
    oss_equilibrium_difference,
)
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402

DTYPE = torch.float64
GATE = 0.15  # owner precision standard: >15 % deviation is not acceptable


def _random_state(ny: int = 6, nx: int = 5, seed: int = 7):
    gen = torch.Generator().manual_seed(seed)
    rho = 1.0 + 1e-3 * torch.rand((ny, nx), generator=gen, dtype=DTYPE)
    ux = 0.05 * torch.rand((ny, nx), generator=gen, dtype=DTYPE)
    uy = 0.05 * torch.rand((ny, nx), generator=gen, dtype=DTYPE)
    feq = equilibrium(rho, ux, uy)
    f = feq * (1.0 + 0.01 * torch.rand((9, ny, nx), generator=gen, dtype=DTYPE))
    return f


PLANES = {
    "dst": (slice(None), slice(1, 2), slice(None)),
    "src1": (slice(None), slice(2, 3), slice(None)),
    "src2": (slice(None), slice(3, 4), slice(None)),
}


def test_modular_equals_literal_oss_form():
    """The port must reproduce ``feq(b) - feq(1) + f(1)`` bit for bit (float64)."""
    f = _random_state()
    spec = EquilibriumExtrapolationSpec(order=1, rho_mode="copy")
    got = apply_equilibrium_extrapolation(f, PLANES["dst"], PLANES["src1"], PLANES["src2"],
                                          0.0, 0.0, spec)[:, 1, :]
    ref = oss_equilibrium_difference(f, PLANES["dst"], PLANES["src1"], 0.0, 0.0)[:, 0, :]
    assert torch.allclose(got, ref, atol=1e-14, rtol=0.0)


def test_order_two_is_not_inert():
    """``order=2`` (Guo 2002) must differ from ``order=1`` on a non-uniform state."""
    f = _random_state()
    a = apply_equilibrium_extrapolation(f, PLANES["dst"], PLANES["src1"], PLANES["src2"], 0.0, 0.0,
                                        EquilibriumExtrapolationSpec(order=1, rho_mode="copy"))[:, 1, :]
    b = apply_equilibrium_extrapolation(f, PLANES["dst"], PLANES["src1"], PLANES["src2"], 0.0, 0.0,
                                        EquilibriumExtrapolationSpec(order=2, rho_mode="extrapolate"))[:, 1, :]
    assert not torch.allclose(a, b)
    assert float((a - b).abs().max()) > 1e-6


def test_uniform_state_is_preserved():
    """A fluid moving with the wall must be untouched: ``f = feq`` is a fixed point."""
    ny, nx = 8, 4
    U = 0.03
    rho = torch.ones((ny, nx), dtype=DTYPE)
    ux = torch.full((ny, nx), U, dtype=DTYPE)
    uy = torch.zeros((ny, nx), dtype=DTYPE)
    f = equilibrium(rho, ux, uy)
    for arm in (EquilibriumExtrapolationSpec(order=1, rho_mode="copy"),
                EquilibriumExtrapolationSpec(order=2, rho_mode="extrapolate")):
        out = apply_channel_extrapolation(f, U, U, arm)
        assert torch.allclose(out, f, atol=1e-15)


def test_bc_only_touches_wall_planes():
    """No leakage into interior rows, and the wall velocity is imposed exactly."""
    ny, nx = 8, 4
    U = 0.03
    f = _random_state(ny, nx)
    out = apply_channel_extrapolation(f, U, 0.0)
    assert torch.equal(out[:, 1:-1, :], f[:, 1:-1, :])
    assert not torch.equal(out[:, 0, :], f[:, 0, :])
    _, ux, uy = macroscopic(out)
    assert abs(float(ux[0].mean()) - U) < 1e-12
    assert float(uy[0].abs().max()) < 1e-12
    assert float(ux[-1].abs().max()) < 1e-12


def test_cavity_walls_and_corner_convention():
    """All four walls carry the prescribed velocity; the side walls win at the corners."""
    n = 9
    u_lid = 0.1
    f = _random_state(n, n)
    out = apply_cavity_extrapolation(f, u_lid)
    _, ux, uy = macroscopic(out)
    # top row minus the two corners, which follow the documented side-wall rule
    assert abs(float(ux[-1, 1:-1].mean()) - u_lid) < 1e-12      # moving lid (top row)
    assert float(ux[0].abs().max()) < 1e-12                    # bottom wall at rest
    assert float(ux[1:-1, 0].abs().max()) < 1e-12              # left wall at rest
    assert float(ux[1:-1, -1].abs().max()) < 1e-12             # right wall at rest
    # documented corner rule: the side-wall prescription is applied last (u = 0)
    assert abs(float(ux[-1, 0])) < 1e-12 and abs(float(ux[-1, -1])) < 1e-12
    assert torch.equal(out[:, 1:-1, 1:-1], f[:, 1:-1, 1:-1])


def test_spec_validation():
    with pytest.raises(ValueError):
        EquilibriumExtrapolationSpec(order=1, rho_mode="extrapolate")
    with pytest.raises(ValueError):
        EquilibriumExtrapolationSpec(order=3)
    with pytest.raises(ValueError):
        EquilibriumExtrapolationSpec(order=1, rho_mode="invent")


def test_fluid_mask():
    m = fluid_mask_from_walls(6, 5, "channel")
    assert m[0].sum() == 0 and m[-1].sum() == 0 and bool(m[1:-1].all())
    m = fluid_mask_from_walls(6, 5, "cavity")
    assert m[0].sum() == 0 and m[:, 0].sum() == 0 and m[:, -1].sum() == 0
    assert bool(m[1:-1, 1:-1].all())


def test_couette_profile_short_run():
    """Physics-level smoke: the OSS rule must hold the linear Couette profile.

    ``f^neq`` is *constant* across a linear shear, so the single-node copy is exact
    here up to compressibility/transient error -- this is the falsification case of
    the XVAL report.  A short run (2 000 steps) is used to keep the unit test fast;
    the graded numbers come from ``benchmarks/xval_equilibrium_difference_extrapolation.py``.
    """
    sys.path.insert(0, str(ROOT / "benchmarks"))
    import numpy as np

    import xval_equilibrium_difference_extrapolation as bench

    ny, nx = 17, 4
    U, re_h = 0.02, 10.0
    H = float(ny - 1)
    nu = U * H / re_h
    tau = 3.0 * nu + 0.5
    zero = torch.zeros((ny, nx), dtype=DTYPE)
    f = equilibrium(torch.ones((ny, nx), dtype=DTYPE), zero, zero)
    spec = EquilibriumExtrapolationSpec(order=1, rho_mode="copy")
    for _ in range(4000):
        f_post = bench.collide(f, tau, None)
        f_post = apply_channel_extrapolation(f_post, U, 0.0, spec)
        f = bench.stream_pull(f_post, periodic_x=True)
    _, ux, _ = macroscopic(f)  # no forcing: momentum needs no half-force correction
    y = np.arange(ny, dtype=float)
    exact = U * (1.0 - y / H)
    num = ux.numpy()[:, 0]
    rel = float(np.linalg.norm(num[1:-1] - exact[1:-1]) / np.linalg.norm(exact[1:-1]))
    # 4 000 steps is ~1/3 of the viscous time H^2/nu, so a few per cent of transient
    # remains; the owner's 15 % gate is the only claim made here.
    assert rel < GATE, f"Couette profile error {rel:.2%} after 4000 steps"