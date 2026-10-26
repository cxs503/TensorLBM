"""Small CPU regressions for SUBOFF post-processing runtime imports.

These test local algebra/API execution, not experimental hull-drag validation.
Keep them in the library suite so a platform-only test run is not required to
catch a missing runtime import or a stale D3Q19 symbol.
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import C, equilibrium3d
from tensorlbm.suboff_postprocess import resistance_breakdown_3d
from tensorlbm.wall_shear import wss_from_fneq_3d


def state(dtype):
    rho = torch.ones((4, 5, 6), dtype=dtype)
    zero = torch.zeros_like(rho)
    mask = torch.zeros_like(rho, dtype=torch.bool)
    mask[1:3, 2:4, 2:4] = True
    return rho, zero, mask


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_d3q19_equilibrium_has_zero_non_equilibrium_wall_shear(dtype):
    rho, zero, mask = state(dtype)
    f = equilibrium3d(rho, zero, zero, zero)
    before = f.clone()
    result = wss_from_fneq_3d(f, rho, zero, zero, zero, 0.8, mask)
    assert result.shape == rho.shape
    assert result.dtype == dtype
    assert torch.equal(result, torch.zeros_like(result))
    assert torch.equal(f, before)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_d3q19_xy_perturbation_has_expected_shear_magnitude(dtype):
    rho, zero, mask = state(dtype)
    tau, amplitude = 0.8, 1e-4
    f = equilibrium3d(rho, zero, zero, zero)
    # Four xy-diagonal populations: sum_q (cx*cy)^2 = 4. Other off-diagonal
    # moments vanish. Density and momentum perturbations cancel by symmetry.
    xy = (C[:, 0] * C[:, 1]).to(dtype=dtype).view(19, 1, 1, 1)
    f = f + amplitude * xy
    before = f.clone()
    result = wss_from_fneq_3d(f, rho, zero, zero, zero, tau, mask)
    expected = torch.full_like(rho, (1 - 0.5 / tau) * 4 * amplitude)
    expected[mask] = 0
    # The perturbation is added to equilibrium populations in the runtime dtype.
    torch.testing.assert_close(result, expected, rtol=0, atol=4 * torch.finfo(dtype).eps)
    assert torch.equal(f, before)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stationary_uniform_state_has_zero_resistance_breakdown(dtype):
    rho, zero, mask = state(dtype)
    f = equilibrium3d(rho, zero, zero, zero)
    before = f.clone()
    result = resistance_breakdown_3d(f, rho, zero, zero, zero, mask, tau=0.8, u_ref=0.08)
    assert result["area_ref_lu2"] > 0
    for key in ("CT", "Cf", "Cp", "F_total_lu", "F_viscous_lu", "F_pressure_lu"):
        assert result[key] == 0.0, key
    assert torch.equal(f, before)
