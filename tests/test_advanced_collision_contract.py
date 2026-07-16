"""Acceptance tests for the lattice-neutral advanced-collision contract."""

import pytest
import torch

from tensorlbm.advanced_collision_contract import (
    WITHHELD_NO_D3Q27_CM_KERNEL,
    WITHHELD_NO_D3Q27_KBC_KERNEL,
    CollisionKernelWithheldError,
    collision_capability_matrix,
    collide_advanced_3d,
)
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.d3q27 import equilibrium27


def test_matrix_reports_real_mrt_for_both_lattices() -> None:
    matrix = collision_capability_matrix()
    assert matrix["D3Q19"]["MRT"].available
    assert matrix["D3Q27"]["MRT"].available
    assert matrix["D3Q19"]["MRT"].entrypoint == "tensorlbm.solver3d.collide_mrt3d"
    assert matrix["D3Q27"]["MRT"].entrypoint == "tensorlbm.d3q27.collide_mrt27"


@pytest.mark.parametrize("lattice,q,equilibrium", [
    ("D3Q19", 19, equilibrium3d),
    ("D3Q27", 27, equilibrium27),
])
def test_common_mrt_dispatch_is_executable_and_equilibrium_fixed_point(lattice, q, equilibrium) -> None:
    rho = torch.ones((2, 3, 4))
    zero = torch.zeros_like(rho)
    f = equilibrium(rho, zero, zero, zero)
    out = collide_advanced_3d(lattice, "MRT", f, tau=0.8)
    assert out.shape == (q, 2, 3, 4)
    assert torch.allclose(out, f, atol=2e-5)


@pytest.mark.parametrize(
    ("family", "reason"),
    [("CM", WITHHELD_NO_D3Q27_CM_KERNEL), ("KBC", WITHHELD_NO_D3Q27_KBC_KERNEL)],
)
def test_d3q27_unverified_advanced_kernels_are_explicitly_withheld(family, reason) -> None:
    rho = torch.ones((1, 1, 1))
    zero = torch.zeros_like(rho)
    f = equilibrium27(rho, zero, zero, zero)
    with pytest.raises(CollisionKernelWithheldError, match=reason):
        collide_advanced_3d("D3Q27", family, f, tau=0.8)
