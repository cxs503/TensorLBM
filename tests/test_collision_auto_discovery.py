"""Auto-discovered collision-operator correctness matrix.

Enumerates every ``(lattice, family)`` pair from
:func:`tensorlbm.advanced_collision_contract.collision_capability_matrix`
and runs the same invariant battery on each one, so a newly admitted kernel
is covered by this file with zero edits (the lettuce-style auto-discovery
pattern requested by the D2 test-matrix roadmap item).

Invariants asserted per combination:

* **Conservation** — one collision step preserves the discrete mass
  ``Σ_q f_q`` and momentum ``Σ_q c_q f_q`` exactly (only non-conserved
  modes are relaxed).
* **Fixed point** — the equilibrium is invariant under collision at any
  admissible ``tau``.
* **Stability** — a small admissible perturbation stays finite (no
  NaN/Inf) at a moderately viscous ``tau``.

Error paths (shape, ``tau`` bound, unknown names) are exercised on one
representative combination.
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.advanced_collision_contract import (
    collide_advanced_3d,
    collision_capability_matrix,
)

_EQUILIBRIA = {}
_C_CONSTANTS = {}


def _equilibrium(lattice: str, n: int, seed: int = 0) -> torch.Tensor:
    """Small (Q, n, n, n) equilibrium field with rho~1 and |u| ~ 0.05."""
    from tensorlbm.d3q19 import equilibrium3d
    from tensorlbm.d3q27 import equilibrium27

    g = torch.Generator().manual_seed(seed)
    rho = 1.0 + 0.01 * torch.randn(n, n, n, generator=g)
    u = 0.05 * torch.randn(3, n, n, n, generator=g)
    if lattice == "D3Q19":
        return equilibrium3d(rho, u[0], u[1], u[2])
    return equilibrium27(rho, u[0], u[1], u[2])


def _c(lattice: str) -> torch.Tensor:
    from tensorlbm.d3q19 import C as C19
    from tensorlbm.d3q27 import C as C27

    return C19.to(torch.float64) if lattice == "D3Q19" else C27.to(torch.float64)


def _admitted_combos() -> list[tuple[str, str]]:
    matrix = collision_capability_matrix()
    return [
        (lattice, family)
        for lattice, families in matrix.items()
        for family, capability in families.items()
        if capability.available
    ]


COMBOS = _admitted_combos()
IDS = [f"{lat}-{fam}" for lat, fam in COMBOS]


def test_every_lattice_family_pair_is_enumerated() -> None:
    """The battery must follow the capability matrix, not a hand-typed list."""
    assert COMBOS, "capability matrix advertises no executable kernels"
    lattices = {lattice for lattice, _ in COMBOS}
    assert lattices == set(collision_capability_matrix())
    # the two production lattices both carry a full seven-family surface
    counts: dict[str, int] = {}
    for lattice, _ in COMBOS:
        counts[lattice] = counts.get(lattice, 0) + 1
    assert all(count >= 7 for count in counts.values())


@pytest.mark.parametrize(("lattice", "family"), COMBOS, ids=IDS)
def test_collision_conserves_mass_and_momentum(lattice: str, family: str) -> None:
    n = 6
    feq = _equilibrium(lattice, n)
    g = torch.Generator().manual_seed(1)
    f = feq + 1.0e-3 * (torch.randn(feq.shape, generator=g) - 0.5)
    c = _c(lattice)

    f2 = collide_advanced_3d(lattice, family, f, tau=0.55)

    torch.testing.assert_close(
        f2.sum(dim=0),
        f.sum(dim=0),
        rtol=1e-5,
        atol=1e-6,
        msg=lambda m: f"{lattice}/{family} mass drift: {m}",
    )
    momentum_before = torch.einsum("qa,q...->a...", c, f.to(torch.float64))
    momentum_after = torch.einsum("qa,q...->a...", c, f2.to(torch.float64))
    torch.testing.assert_close(
        momentum_after,
        momentum_before,
        rtol=1e-4,
        atol=1e-6,
        msg=lambda m: f"{lattice}/{family} momentum drift: {m}",
    )


@pytest.mark.parametrize(("lattice", "family"), COMBOS, ids=IDS)
def test_equilibrium_is_collision_fixed_point(lattice: str, family: str) -> None:
    feq = _equilibrium(lattice, 6, seed=2)

    f2 = collide_advanced_3d(lattice, family, feq, tau=0.8)

    torch.testing.assert_close(
        f2,
        feq,
        rtol=1e-3,
        atol=2e-4,
        msg=lambda m: f"{lattice}/{family} equilibrium not a fixed point: {m}",
    )


@pytest.mark.parametrize(("lattice", "family"), COMBOS, ids=IDS)
def test_small_perturbation_stays_finite(lattice: str, family: str) -> None:
    feq = _equilibrium(lattice, 6, seed=3)
    g = torch.Generator().manual_seed(4)
    f = feq + 1.0e-3 * (torch.randn(feq.shape, generator=g) - 0.5)

    f2 = collide_advanced_3d(lattice, family, f, tau=0.6)

    assert torch.isfinite(f2).all(), f"{lattice}/{family} produced non-finite states"


def test_tau_must_exceed_half() -> None:
    f = _equilibrium("D3Q19", 4, seed=5)
    with pytest.raises(ValueError, match="tau"):
        collide_advanced_3d("D3Q19", "BGK", f, tau=0.5)
    with pytest.raises(ValueError, match="tau"):
        collide_advanced_3d("D3Q19", "BGK", f, tau=torch.full((4, 4, 4), 0.4))


def test_wrong_population_shape_is_rejected() -> None:
    f = _equilibrium("D3Q19", 4, seed=6)
    with pytest.raises(ValueError, match="populations must have shape"):
        collide_advanced_3d("D3Q27", "BGK", f, tau=0.6)
    with pytest.raises(ValueError, match="populations must have shape"):
        collide_advanced_3d("D3Q19", "BGK", f[0], tau=0.6)


def test_unknown_lattice_or_family_is_rejected() -> None:
    f = _equilibrium("D3Q19", 4, seed=7)
    with pytest.raises(ValueError):
        collide_advanced_3d("D3Q42", "BGK", f, tau=0.6)
    with pytest.raises(ValueError):
        collide_advanced_3d("D3Q19", "ELB", f, tau=0.6)
