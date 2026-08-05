"""Lattice-local, auditable MRT numerical-consistency evidence.

This deliberately checks the same *kind* of contract independently for each
lattice.  It is not an accuracy benchmark or a D3Q19-vs-D3Q27 ranking.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import FunctionType

import pytest
import torch

from tensorlbm.d3q19 import C as C19
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.d3q27 import C as C27
from tensorlbm.d3q27 import collide_mrt27, equilibrium27, macroscopic27
from tensorlbm.solver3d import collide_mrt3d


CASES = (
    pytest.param("D3Q19", 19, C19, equilibrium3d, macroscopic3d, collide_mrt3d, id="d3q19"),
    pytest.param("D3Q27", 27, C27, equilibrium27, macroscopic27, collide_mrt27, id="d3q27"),
)

# Direct callable-source fingerprints, intentionally updated only after a new
# consistency audit.  They bind this evidence to the collision implementation.
SOURCE_SHA256 = {
    "D3Q19": "847e4b6d385ae9147e1a3b2e02a7de8f19fe1ff1c1ac66a8a900ac901d7f2b13",
    "D3Q27": "4b1b55bf7b2aae49857f22d261e75666765764f5eeeb37050f105a17bafc10b5",
}


def _source_sha256(function: FunctionType) -> str:
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def _case_state(q: int, equilibrium):
    """Return a fixed float32 equilibrium and a zero-mass/momentum perturbation."""
    shape = (2, 3, 4)
    generator = torch.Generator(device="cpu").manual_seed(2718)
    rho = 0.9 + 0.2 * torch.rand(shape, generator=generator, dtype=torch.float32)
    ux = -0.035 + 0.07 * torch.rand(shape, generator=generator, dtype=torch.float32)
    uy = -0.035 + 0.07 * torch.rand(shape, generator=generator, dtype=torch.float32)
    uz = -0.035 + 0.07 * torch.rand(shape, generator=generator, dtype=torch.float32)
    feq = equilibrium(rho, ux, uy, uz)

    # A non-equilibrium perturbation that is per-cell conservative: its density
    # and all three raw momentum components are zero for the supplied stencil.
    perturbation = torch.zeros_like(feq)
    perturbation[0] = 2.0e-4
    perturbation[1] = -5.0e-5
    perturbation[2] = -5.0e-5
    perturbation[3] = -5.0e-5
    perturbation[4] = -5.0e-5
    return feq, feq + perturbation


@pytest.mark.parametrize("lattice,q,directions,equilibrium,macroscopic,collision", CASES)
def test_mrt_equilibrium_fixed_point_is_finite_and_lattice_local(
    lattice, q, directions, equilibrium, macroscopic, collision
) -> None:
    feq, _ = _case_state(q, equilibrium)
    out = collision(feq, tau=0.8)

    assert feq.dtype is torch.float32
    assert out.shape == (q, 2, 3, 4)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, feq, rtol=0.0, atol=1.0e-6)


@pytest.mark.parametrize("lattice,q,directions,equilibrium,macroscopic,collision", CASES)
def test_mrt_perturbed_collision_conserves_local_density_and_momentum(
    lattice, q, directions, equilibrium, macroscopic, collision
) -> None:
    _, f = _case_state(q, equilibrium)
    rho_before, ux_before, uy_before, uz_before = macroscopic(f)
    momentum_before = torch.stack((rho_before * ux_before, rho_before * uy_before, rho_before * uz_before))

    out = collision(f, tau=0.8)
    rho_after, ux_after, uy_after, uz_after = macroscopic(out)
    momentum_after = torch.stack((rho_after * ux_after, rho_after * uy_after, rho_after * uz_after))

    assert torch.isfinite(out).all()
    torch.testing.assert_close(rho_after, rho_before, rtol=0.0, atol=1.0e-6)
    torch.testing.assert_close(momentum_after, momentum_before, rtol=0.0, atol=1.0e-6)


@pytest.mark.parametrize("lattice,q,directions,equilibrium,macroscopic,collision", CASES)
def test_mrt_repeated_collision_is_bitwise_deterministic(
    lattice, q, directions, equilibrium, macroscopic, collision
) -> None:
    _, f = _case_state(q, equilibrium)
    first = collision(f.clone(), tau=0.8)
    second = collision(f.clone(), tau=0.8)

    assert torch.isfinite(first).all()
    assert torch.equal(first, second)


@pytest.mark.parametrize("lattice,q,directions,equilibrium,macroscopic,collision", CASES)
def test_mrt_evidence_is_bound_to_the_direct_collision_source(
    lattice, q, directions, equilibrium, macroscopic, collision
) -> None:
    assert _source_sha256(collision) == SOURCE_SHA256[lattice]


def test_d3q27_mrt_documented_float32_only_matrix_limitation() -> None:
    """D3Q27's cached transform is float32; float64 populations cannot multiply it."""
    feq, _ = _case_state(27, equilibrium27)
    with pytest.raises(RuntimeError, match="same dtype"):
        collide_mrt27(feq.double(), tau=0.8)


def test_audit_document_is_present() -> None:
    document = Path(__file__).parents[1] / "docs" / "d3q19-d3q27-mrt-consistency-audit.md"
    assert document.is_file()
