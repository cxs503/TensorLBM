"""DE-NXT-RELX-W3: D3Q19 lattice contract of the entropic KBC collision.

Defect context (fixed in this commit)
--------------------------------------
``tensorlbm.kbc_collision._weights`` applied the D3Q27 (Qian) weight formula
to the 19-direction velocity set, so the weights totalled 26/27
(measured sum 0.9629629629629628) and ``lattice(19)`` raised the
normalisation assertion.  The entropic KBC kernel was therefore unusable
with the D3Q19 lattice of the free-surface solver, while the solver's
``collision == "kbc"`` branch imported a module that does not exist.

These tests pin the repaired contract:

* the D3Q19 weights are the canonical ones and are identical to the solver's
  ``tensorlbm.d3q19._W_VALUES``;
* the D3Q19 velocity ordering is identical to ``tensorlbm.d3q19.C`` (so the
  populations need no permutation when the kernel is wired into the solver);
* ``collide_kbc`` with ``alpha = 1`` and ``tau_bulk = tau`` reproduces the
  solver's BGK collision (the ``alpha = 1`` limit is BGK by construction);
* the entropy solve damps the ghost segment (``alpha < 1``) on a perturbed
  field -- the structural effect the DE campaign wants to test.
"""

from __future__ import annotations

import numpy as np
import torch

from tensorlbm import kbc_collision as kbc
from tensorlbm.d3q19 import C, _W_VALUES
from tensorlbm.solver3d import equilibrium3d, macroscopic3d

TAU = 0.5003877


def _field(seed: int = 0, n: int = 4) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    u = torch.tensor(rng.normal(0.0, 0.02, size=(n, n, n, 3)), dtype=torch.float32)
    f = equilibrium3d(torch.full((n, n, n), 1.0), u[..., 0], u[..., 1], u[..., 2])
    noise = torch.tensor(1e-4 * rng.normal(0.0, 1.0, size=tuple(f.shape)), dtype=torch.float32)
    return f + noise


def test_d3q19_weights_are_canonical_and_match_solver() -> None:
    lat = kbc.lattice(19)
    assert abs(lat.w.sum() - 1.0) < 1e-12
    assert np.allclose(lat.w, np.asarray(_W_VALUES, dtype=np.float64))
    # rest 1/3, face 1/18, edge 1/36
    assert abs(lat.w[0] - 1.0 / 3.0) < 1e-15
    assert np.allclose(lat.w[1:7], 1.0 / 18.0)
    assert np.allclose(lat.w[7:19], 1.0 / 36.0)


def test_d3q19_ordering_matches_solver_populations() -> None:
    lat = kbc.lattice(19)
    assert (lat.c == C.numpy()).all()


def test_d3q27_lattice_untouched() -> None:
    lat = kbc.lattice(27)
    assert abs(lat.w.sum() - 1.0) < 1e-12
    assert abs(lat.w[0] - 8.0 / 27.0) < 1e-15


def test_alpha_one_equals_bgk() -> None:
    """alpha = 1 with tau_bulk = tau is BGK by construction (KBC, PRL 106:236304)."""
    f = _field()
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    bgk = (f - (f - feq) / TAU).numpy()

    f_post, _diag = kbc.collide_kbc(
        f.numpy().astype(np.float64),
        TAU,
        alpha_mode="fixed",
        alpha_value=1.0,
        tau_bulk=TAU,
        omega_g=1.0,
    )
    scale = max(float(np.abs(bgk).max()), 1e-12)
    assert float(np.abs(f_post - bgk).max()) / scale < 1e-3


def test_entropy_solve_damps_ghost_segment() -> None:
    """The entropy condition must pick alpha < 1 (ghost damping) on a noisy field."""
    f = _field(seed=1)
    _f_post, diag = kbc.collide_kbc(
        f.numpy().astype(np.float64),
        TAU,
        alpha_mode="entropy",
        tau_bulk=TAU,
        omega_g=1.0,
    )
    assert np.nanmin(diag.alpha) < 1.0
    assert np.nanmax(diag.alpha) > 0.0
