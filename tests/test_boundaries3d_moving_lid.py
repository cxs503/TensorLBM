"""Tests for tensorlbm.boundaries3d.zou_he_moving_lid_3d (D3Q19 moving lid BC).

Correctness properties (exact, algebraically enforced by the Zou/He 1997
reconstruction — numerically verified here):
  * jy ≡ 0 and jz ≡ 0 on the lid plane (normal + spanwise momentum vanish)
  * jx ≡ rho_tot * u_lid on the lid plane (x-momentum prescribed)
  * only the five unknown cy<0 directions {4, 8, 9, 16, 18} are modified
  * the BC is mass-neutral at steady state (driven cavity converges without
    systematic mass drift; regression smoke in the benchmark run.py)
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.boundaries3d import zou_he_moving_lid_3d
from tensorlbm.d3q19 import C, equilibrium3d


def _perturbed_lid_field(nz=4, ny=16, nx=16, seed=0):
    torch.manual_seed(seed)
    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32)
    u0 = torch.zeros((nz, ny, nx), dtype=torch.float32)
    f = equilibrium3d(rho0, u0, u0, u0)
    return f + 0.01 * torch.randn_like(f)


class TestZouHeMovingLid3D:
    def test_preserves_shape_and_is_finite(self) -> None:
        f = _perturbed_lid_field()
        f_out = zou_he_moving_lid_3d(f, u_lid=0.1)
        assert f_out.shape == f.shape
        assert torch.isfinite(f_out).all()

    def test_does_not_modify_input(self) -> None:
        f = _perturbed_lid_field()
        f_before = f.clone()
        zou_he_moving_lid_3d(f, u_lid=0.1)
        torch.testing.assert_close(f, f_before)

    def test_only_unknown_cy_lt_0_directions_modified(self) -> None:
        f = _perturbed_lid_field()
        f_out = zou_he_moving_lid_3d(f, u_lid=0.1)
        changed = (f_out - f).abs().sum(dim=(1, 2, 3)) > 1e-9
        unknown = {4, 8, 9, 16, 18}  # cy<0 at top wall
        assert set(torch.nonzero(changed).flatten().tolist()) == unknown

    def test_exact_momentum_prescription(self) -> None:
        """jy=0, jz=0, jx = rho_tot*u_lid on the lid plane (exact)."""
        for u_lid in (0.06, 0.1):
            f = _perturbed_lid_field(seed=3)
            fl = zou_he_moving_lid_3d(f, u_lid)[:, :, -1, :]
            rho_tot = fl.sum()
            jx = (fl * C[:, 0].view(19, 1, 1)).sum()
            jy = (fl * C[:, 1].view(19, 1, 1)).sum()
            jz = (fl * C[:, 2].view(19, 1, 1)).sum()
            assert abs(jx.item() - rho_tot.item() * u_lid) < 1e-5
            assert abs(jy.item()) < 1e-5
            assert abs(jz.item()) < 1e-5

    def test_uy_uz_not_supported(self) -> None:
        f = _perturbed_lid_field()
        with pytest.raises(NotImplementedError):
            zou_he_moving_lid_3d(f, u_lid=0.1, uy_lid=0.01)
        with pytest.raises(NotImplementedError):
            zou_he_moving_lid_3d(f, u_lid=0.1, uz_lid=0.01)

    def test_zero_lid_velocity_is_mass_neutral_on_symmetric_field(self) -> None:
        """u_lid=0 on a y-symmetric field must leave the layer mass unchanged
        (jy=0 ⇒ Σcy>0 = Σcy<0 ⇒ BC mass change = 0)."""
        from tensorlbm.boundaries3d import _D3Q19_MIRROR_Y

        f = _perturbed_lid_field(seed=5)
        # symmetrise across y so that Σcy>0 = Σcy<0 at the lid row
        f_sym = f.clone()
        f_sym[:, :, -1, :] = 0.5 * (f[:, :, -1, :] + f[:, :, -1, :][_D3Q19_MIRROR_Y, :, :])
        f_out = zou_he_moving_lid_3d(f_sym, u_lid=0.0)
        m0 = f_sym[:, :, -1, :].sum()
        m1 = f_out[:, :, -1, :].sum()
        assert abs(m1.item() - m0.item()) < 1e-5
