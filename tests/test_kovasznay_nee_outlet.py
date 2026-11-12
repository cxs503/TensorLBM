"""Contract tests for the EXPERIMENTAL non-equilibrium-extrapolation outflow (D2Q9).

The NEE closure is *not* validated as a production outflow (it diverges on the
Kovasznay ladder — see docs/CASE-LBM-KOVASZNAY-OUTLET-BC.md).  What is tested
here is only what was actually verified: shape/argument guards, the "interior is
never touched" invariant, and finiteness for a single application on a smooth
analytic state.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.boundaries import nonequilibrium_extrapolation_outlet_2d
from tensorlbm.d2q9 import equilibrium


def _analytic_state(ny: int = 24, nx: int = 48, re: float = 40.0, u0: float = 0.03, xmax: float = 3.0):
    lam = re / 2.0 - math.sqrt(re * re / 4.0 + 4.0 * math.pi**2)
    y = torch.arange(ny, dtype=torch.float64).view(-1, 1)
    x = torch.arange(nx, dtype=torch.float64).view(1, -1)
    xp = x / nx * xmax
    yp = y / ny
    e = torch.exp(lam * xp)
    u = (u0 * (1.0 - e * torch.cos(2.0 * math.pi * yp))).expand(ny, nx).contiguous()
    v = (u0 * (lam / (2.0 * math.pi)) * e * torch.sin(2.0 * math.pi * yp)).expand(ny, nx).contiguous()
    rho = torch.ones((ny, nx), dtype=torch.float64)
    return equilibrium(rho, u, v)


def test_single_application_is_finite_and_local() -> None:
    f = _analytic_state()
    f_new = nonequilibrium_extrapolation_outlet_2d(f)
    assert torch.isfinite(f_new).all()
    # Interior columns untouched (outflow closure acts on the last plane only).
    assert torch.equal(f_new[:, :, :-1], f[:, :, :-1])
    # Only populations entering the domain (c_x < 0) are prescribed.
    from tensorlbm.d2q9 import C

    outgoing = C[:, 0] >= 0
    assert torch.equal(f_new[outgoing][:, :, -1], f[outgoing][:, :, -1])


@pytest.mark.parametrize("bad_shape", [(9, 4, 2), (8, 8, 8)])
def test_rejects_invalid_input(bad_shape) -> None:
    with pytest.raises(ValueError):
        nonequilibrium_extrapolation_outlet_2d(torch.ones(bad_shape))
