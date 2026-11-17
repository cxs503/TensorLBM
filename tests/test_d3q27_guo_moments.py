"""Physical moment checks independent of the collision implementation."""

import pytest
import torch

from tensorlbm.d3q27 import C, equilibrium27
from tensorlbm.multiphase3d_d3q27 import collide_sc_two_component_27


@pytest.mark.parametrize("tau", [0.6, 0.8, 1.0, 1.2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sgs", [None, "smagorinsky", "wale", "vreman"])
def test_guo_supplies_one_force_increment(tau, dtype, sgs):
    shape = (4, 5, 6)
    rho = torch.full(shape, 0.8, dtype=dtype)
    x = torch.arange(shape[-1], dtype=dtype)[None, None, :].expand(shape)
    # A nonuniform, moving state also exercises the quadratic source moments
    # and spatially varying effective relaxation, unlike a rest-only test.
    u = [0.01 + 0.003 * x, torch.full_like(rho, -0.02), torch.full_like(rho, 0.015)]
    f1 = equilibrium27(rho, *u)
    f2 = equilibrium27(0.6 * rho, *[v * -0.5 for v in u])
    acceleration = torch.tensor([2e-4, -3e-4, 4e-4], dtype=dtype)
    out = collide_sc_two_component_27(
        f1,
        f2,
        G_12=0.0,
        tau1=tau,
        tau2=tau + 0.1,
        gx=float(acceleration[0]),
        gy=float(acceleration[1]),
        gz=float(acceleration[2]),
        use_guo=True,
        C_s=0.15 if sgs else 0.0,
        sgs_model=sgs or "smagorinsky",
    )
    c = C.to(dtype).T[:, :, None, None, None]
    atol = 1e-7 if dtype == torch.float32 else 2e-15
    for before, after in zip((f1, f2), out):
        torch.testing.assert_close(
            after.sum(0), before.sum(0), atol=4 * torch.finfo(dtype).eps, rtol=0
        )
        delta_momentum = (c * (after - before)[None]).sum(1)
        expected = acceleration[:, None, None, None] * before.sum(0)
        torch.testing.assert_close(delta_momentum, expected, atol=atol, rtol=0)
