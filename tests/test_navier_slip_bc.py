"""Unit tests for Navier slip boundary condition.

Tests the two limiting cases:
1. Ls=0 should degenerate to bounce-back (no-slip)
2. Ls→∞ should degenerate to specular reflection (free-slip)
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorlbm.boundaries3d import bounce_back_cells_3d, free_slip_cells_3d
from tensorlbm.d3q19 import C, equilibrium3d, macroscopic3d
from tensorlbm.navier_slip_bc import navier_slip_cells_3d, slip_length_to_beta
from tensorlbm.solver3d import stream3d


def test_slip_length_to_beta():
    """Test slip length to beta conversion."""
    # Ls=0 -> beta=0 (no-slip)
    assert slip_length_to_beta(0.0) == 0.0

    # Ls=1 -> beta=0.5
    assert abs(slip_length_to_beta(1.0) - 0.5) < 1e-10

    # Ls→∞ -> beta→1 (free-slip)
    assert slip_length_to_beta(float("inf")) == 1.0

    # Ls<0 should raise error
    with pytest.raises(ValueError):
        slip_length_to_beta(-1.0)


def test_navier_slip_Ls0_equals_bounce_back():
    """Test that Ls=0 gives identical results to bounce-back."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup
    nz, ny, nx = 4, 21, 16
    rho_init = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    ux_init = torch.zeros_like(rho_init)
    uy_init = torch.zeros_like(rho_init)
    uz_init = torch.zeros_like(rho_init)

    f_bb = equilibrium3d(rho_init, ux_init, uy_init, uz_init)
    f_ns = f_bb.clone()

    # Wall masks
    bottom_mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    bottom_mask[:, 0, :] = True

    # Apply some flow
    for step in range(10):
        rho, ux, uy, uz = macroscopic3d(f_bb)
        feq = equilibrium3d(rho, ux, uy, uz)
        f_bb = f_bb - (f_bb - feq) / 1.0
        f_bb = stream3d(f_bb)
        f_bb = bounce_back_cells_3d(f_bb, bottom_mask)

        rho, ux, uy, uz = macroscopic3d(f_ns)
        feq = equilibrium3d(rho, ux, uy, uz)
        f_ns = f_ns - (f_ns - feq) / 1.0
        f_ns = stream3d(f_ns)
        f_ns = navier_slip_cells_3d(f_ns, bottom_mask, axis=1, Ls=0.0)

    # Compare
    assert torch.allclose(f_bb, f_ns, rtol=1e-5, atol=1e-6), (
        "Navier slip with Ls=0 should be identical to bounce-back"
    )


def test_navier_slip_Ls_inf_equals_free_slip():
    """Test that Ls→∞ gives identical results to specular reflection."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup
    nz, ny, nx = 4, 21, 16
    rho_init = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    ux_init = torch.zeros_like(rho_init)
    uy_init = torch.zeros_like(rho_init)
    uz_init = torch.zeros_like(rho_init)

    f_fs = equilibrium3d(rho_init, ux_init, uy_init, uz_init)
    f_ns = f_fs.clone()

    # Wall masks
    top_mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    top_mask[:, -1, :] = True

    # Apply some flow
    for step in range(10):
        rho, ux, uy, uz = macroscopic3d(f_fs)
        feq = equilibrium3d(rho, ux, uy, uz)
        f_fs = f_fs - (f_fs - feq) / 1.0
        f_fs = stream3d(f_fs)
        f_fs = free_slip_cells_3d(f_fs, top_mask, axis=1)

        rho, ux, uy, uz = macroscopic3d(f_ns)
        feq = equilibrium3d(rho, ux, uy, uz)
        f_ns = f_ns - (f_ns - feq) / 1.0
        f_ns = stream3d(f_ns)
        f_ns = navier_slip_cells_3d(f_ns, top_mask, axis=1, Ls=float("inf"))

    # Compare
    assert torch.allclose(f_fs, f_ns, rtol=1e-5, atol=1e-6), (
        "Navier slip with Ls→∞ should be identical to specular reflection"
    )


def test_navier_slip_mass_conservation():
    """Check 100-step conservation in binary64, keeping the absolute 1e-6 gate.

    At this domain's mass (1344), one float32 ULP is 1.22e-4.  Both the
    collision and boundary arithmetic introduce rounding; the original gate
    fails even without the boundary.  Test the evolution in float64; the isolated
    boundary test below separately covers float32 without collision error.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup
    nz, ny, nx = 4, 21, 16
    rho_init = torch.ones((nz, ny, nx), dtype=torch.float64, device=device)
    ux_init = torch.zeros_like(rho_init)
    uy_init = torch.zeros_like(rho_init)
    uz_init = torch.zeros_like(rho_init)

    f = equilibrium3d(rho_init, ux_init, uy_init, uz_init)

    # Wall masks
    bottom_mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    bottom_mask[:, 0, :] = True
    top_mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    top_mask[:, -1, :] = True

    # Initial mass
    mass_initial = f.sum().item()

    # Evolve the stationary equilibrium with Navier slip
    for step in range(100):
        rho, ux, uy, uz = macroscopic3d(f)
        feq = equilibrium3d(rho, ux, uy, uz)
        f = f - (f - feq) / 1.0
        f = stream3d(f)
        f = navier_slip_cells_3d(f, bottom_mask, axis=1, Ls=2.0)
        f = navier_slip_cells_3d(f, top_mask, axis=1, Ls=2.0)

    # Final mass
    mass_final = f.sum().item()

    # Check mass conservation
    mass_drift = abs(mass_final - mass_initial)
    assert mass_drift < 1e-6, f"Navier slip should conserve mass, but drift is {mass_drift}"


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("Ls", [0.0, 0.25, 2.0, float("inf")])
def test_navier_slip_local_conservation_and_momentum(device, dtype, axis, Ls):
    """Audit the boundary alone on non-equilibrium, nonuniform populations.

    Both reflection maps are permutations: local density is unchanged,
    normal momentum reverses, and tangential momentum scales by 2*beta-1.
    These moment identities are an oracle independent of the mirror tables.
    Accumulate in float64 to distinguish population rounding from reduction
    rounding.  The 8-epsilon local budget covers the two products and sum
    in the interpolation, not an empirical full-simulation drift allowance.
    """
    generator = torch.Generator().manual_seed(310)
    f = (0.01 + 0.1 * torch.rand((19, 3, 4, 5), generator=generator, dtype=dtype)).to(device)
    before = f.clone()
    mask = torch.zeros((3, 4, 5), dtype=torch.bool, device=device)
    mask[:, 0, :] = True
    mask[1, 2, ::2] = True

    result = navier_slip_cells_3d(f, mask, axis=axis, Ls=Ls)

    assert result.dtype == dtype
    assert result.device == f.device
    assert torch.equal(f, before), "Boundary must not mutate its input"
    assert torch.equal(result[:, ~mask], before[:, ~mask])

    rho_before = before.double().sum(dim=0)
    rho_after = result.double().sum(dim=0)
    budget = 8 * torch.finfo(dtype).eps * rho_before
    assert torch.all((rho_after - rho_before).abs() <= budget)
    assert abs((result.double() - before.double()).sum().item()) <= budget[mask].sum().item()

    velocities = C.to(device=device, dtype=torch.float64)
    momentum_before = torch.einsum("qzyx,qa->azyx", before.double(), velocities)
    momentum_after = torch.einsum("qzyx,qa->azyx", result.double(), velocities)
    expected = (2 * slip_length_to_beta(Ls) - 1) * momentum_before
    expected[axis] = -momentum_before[axis]
    assert torch.all((momentum_after[:, mask] - expected[:, mask]).abs() <= budget[mask])


if __name__ == "__main__":
    test_slip_length_to_beta()
    print("✓ test_slip_length_to_beta passed")

    test_navier_slip_Ls0_equals_bounce_back()
    print("✓ test_navier_slip_Ls0_equals_bounce_back passed")

    test_navier_slip_Ls_inf_equals_free_slip()
    print("✓ test_navier_slip_Ls_inf_equals_free_slip passed")

    test_navier_slip_mass_conservation()
    print("✓ test_navier_slip_mass_conservation passed")

    print("\nAll tests passed!")
