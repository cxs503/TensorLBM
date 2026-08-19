"""Parity tests for tensorlbm.triton_fused on z-asymmetric fields.

Regression guard for the hand-typed lattice-table bug: the original
``_CX``/``_CY``/``_CZ`` tuples of :mod:`tensorlbm.triton_fused`
disagreed with the canonical :mod:`tensorlbm.d3q19` lattice on the six
diagonal directions q in {8, 10, 12, 14, 16, 18} — the lane pairs
8<->10 (cy sign), 12<->14 / 16<->18 (cz sign) each carried the other
direction's lattice velocity.  Because the wrong table is a pure lane
permutation of the correct one, the error is invisible on uniform or
mirror-symmetric fields: every pre-existing periodic test passed and
step-1 forces matched.  Any field with structure along z (or y),
however, streams from wrong neighbour cells on those six lanes and
diverges from the reference physics within a few steps (measured
pre-fix mismatch O(1e-2), post-fix at fp32 summation-order level).

Two layers of defence:

  1. ``test_lattice_tables_match_canonical_d3q19`` — the module tuples
     AND the padded int/float device tensors of ``make_lattice_tensors``
     must equal ``d3q19.C`` / ``d3q19.W`` lane by lane.
  2. ``test_triton_matches_eager_on_asymmetric_fields`` — N-step
     trajectories of ``TritonFusedSolver3D`` vs an eager PyTorch
     pull-BGK reference built directly from the canonical tables, on
     three initial fields that each break the z mirror symmetry.
"""
from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import C, W, equilibrium3d
from tensorlbm.triton_fused import (
    TritonFusedSolver3D,
    is_available,
    make_lattice_tensors,
)


@pytest.fixture(scope="module")
def dev() -> str:
    if not is_available():
        pytest.skip("Triton fused requires CUDA + triton")
    return "cuda:0"


# ---------------------------------------------------------------------------
# 1. Table-level regression — module tables must equal canonical d3q19
# ---------------------------------------------------------------------------

def test_lattice_tables_match_canonical_d3q19(dev: str) -> None:
    """_CX/_CY/_CZ/_W tuples and the padded kernel tensors follow d3q19."""
    import tensorlbm.triton_fused as tf

    for q in range(19):
        got = (tf._CX[q], tf._CY[q], tf._CZ[q])
        ref = tuple(int(v) for v in C[q].tolist())
        assert got == ref, (
            f"lattice lane q={q}: module tuple {got} != canonical d3q19.C "
            f"{ref} (sign/permutation drift in the hand-maintained tables)")

    lat = make_lattice_tensors(dev)
    cdev = C.to(dev)
    for key, col in (("cxi", 0), ("cyi", 1), ("czi", 2)):
        assert torch.equal(lat[key][:19], cdev[:, col].to(torch.int32)), (
            f"int addressing table {key} != canonical d3q19.C[:, {col}]")
    for key, col in (("cxf", 0), ("cyf", 1), ("czf", 2)):
        assert torch.equal(lat[key][:19], cdev[:, col].to(torch.float32)), (
            f"float moment table {key} != canonical d3q19.C[:, {col}]")
    assert torch.equal(lat["w"][:19], W.to(dev)), "weights != canonical d3q19.W"


# ---------------------------------------------------------------------------
# 2. Eager reference step (pull-BGK from canonical tables)
# ---------------------------------------------------------------------------

def _eager_step(f: torch.Tensor, tau: float) -> torch.Tensor:
    """One periodic pull-stream + BGK collide step in eager PyTorch.

    Mirrors the fused kernel exactly: pull ``f[q]`` from
    ``(x - cx_q, y - cy_q, z - cz_q)`` with periodic wrap, compute
    moments with the same rho floor, relax towards the D3Q19
    equilibrium.  Only the lattice constants are taken from the
    canonical ``d3q19`` module — never from the module under test.
    """
    q_len, nz, ny, nx = f.shape
    c = C.to(device=f.device, dtype=torch.float32)
    w = W.to(device=f.device, dtype=torch.float32).view(-1, 1, 1, 1)

    f_in = torch.empty_like(f)
    for q in range(q_len):
        # pull: source cell is (x - c_q); torch.roll(a, s, dim) gives
        # out[i] = a[i - s], so the shift is +c_q (same convention as the
        # bit-exact stream test in test_triton_fused.py).
        f_in[q] = torch.roll(
            f[q],
            shifts=(int(c[q, 2]), int(c[q, 1]), int(c[q, 0])),
            dims=(0, 1, 2),
        )

    rho = f_in.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-12)
    cx = c[:, 0].view(-1, 1, 1, 1)
    cy = c[:, 1].view(-1, 1, 1, 1)
    cz = c[:, 2].view(-1, 1, 1, 1)
    ux = (cx * f_in).sum(dim=0) / rho_safe
    uy = (cy * f_in).sum(dim=0) / rho_safe
    uz = (cz * f_in).sum(dim=0) / rho_safe
    usq = ux * ux + uy * uy + uz * uz
    cu = cx * ux + cy * uy + cz * uz
    feq = (rho_safe.unsqueeze(0) * w
           * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq.unsqueeze(0)))
    return f_in - (1.0 / tau) * (f_in - feq)


# ---------------------------------------------------------------------------
# 3. Initial fields that break the z mirror symmetry
# ---------------------------------------------------------------------------

def _field_shear_yz(n: int, dev: str) -> torch.Tensor:
    """Shear with transverse components: ux(z), uy(z), uz(y).

    NOTE on why the transverse components are required: a pure ux(z)
    shear keeps the lane pairs 8<->10 (needs uy != 0 to split) and
    12<->14 / 16<->18 (need uz != 0 to split) populated identically
    along the whole trajectory, and the lane-permutation bug is then
    exactly invisible (buggy output == correct output lane by lane).
    The uy/uz components here break every buggy lane pairing while the
    z/y-dependent phases keep the streaming pulls asymmetric.
    """
    ax_z = torch.arange(n, device=dev, dtype=torch.float32) / n
    ax_y = torch.arange(n, device=dev, dtype=torch.float32) / n
    ux = (0.08 * torch.sin(2.0 * torch.pi * ax_z + 0.3)).view(n, 1, 1).expand(n, n, n)
    uy = (0.06 * torch.sin(2.0 * torch.pi * ax_z + 1.1)).view(n, 1, 1).expand(n, n, n)
    uz = (0.05 * torch.sin(2.0 * torch.pi * ax_y + 0.7)).view(1, n, 1).expand(n, n, n)
    rho = torch.ones((n, n, n), device=dev)
    return equilibrium3d(rho, ux.contiguous(), uy.contiguous(), uz.contiguous())


def _field_blob_offcentre(n: int, dev: str) -> torch.Tensor:
    """Off-centre Gaussian density blob — no mirror plane in any axis."""
    axes = torch.arange(n, device=dev, dtype=torch.float32)
    Z, Y, X = torch.meshgrid(axes, axes, axes, indexing="ij")
    sigma = n / 8.0
    z0, y0, x0 = 0.31 * n, 0.43 * n, 0.62 * n
    r2 = (Z - z0) ** 2 + (Y - y0) ** 2 + (X - x0) ** 2
    rho = 1.0 + 0.4 * torch.exp(-r2 / (2.0 * sigma * sigma))
    drift = 0.02
    return equilibrium3d(rho,
                         torch.full_like(rho, drift),
                         torch.full_like(rho, -drift),
                         torch.full_like(rho, drift))


def _field_random(n: int, dev: str) -> torch.Tensor:
    """Seeded random equilibrium — breaks every lattice symmetry."""
    rho = 1.0 + 0.05 * torch.randn((n, n, n), device=dev)
    u = 0.03 * torch.randn((3, n, n, n), device=dev)
    return equilibrium3d(rho, u[0], u[1], u[2])


_FIELD_BUILDERS = {
    "shear_yz": _field_shear_yz,
    "blob_offcentre": _field_blob_offcentre,
    "random": _field_random,
}


# ---------------------------------------------------------------------------
# 4. N-step trajectory parity on asymmetric fields
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", sorted(_FIELD_BUILDERS))
def test_triton_matches_eager_on_asymmetric_fields(field: str, dev: str) -> None:
    """Triton solver must track the canonical eager reference step-for-step.

    Pre-fix (lane-permuted tables) this fails at max|diff| = 2.4e-3 ..
    9.7e-3 on every asymmetric field; post-fix it holds at fp32
    summation-order level (measured max|diff| <= 1.8e-7, rel <= 5.2e-7
    over 12 steps at n=32).
    """
    n, tau, n_steps = 32, 0.7, 12
    torch.manual_seed(20260819)
    f0 = _FIELD_BUILDERS[field](n, dev)

    solver = TritonFusedSolver3D(nz=n, ny=n, nx=n, tau=tau, device=dev)
    f_tri = f0.clone()
    for _ in range(n_steps):
        f_tri = solver.step(f_tri)

    f_ref = f0.clone()
    for _ in range(n_steps):
        f_ref = _eager_step(f_ref, tau)

    diff = (f_tri - f_ref).abs().max().item()
    scale = max(f_ref.abs().max().item(), 1e-12)
    rel = diff / scale
    assert torch.allclose(f_tri, f_ref, rtol=1e-4, atol=1e-6), (
        f"{n_steps}-step trajectory on '{field}' diverged from the canonical "
        f"eager reference: max|diff|={diff:.3e}, rel={rel:.3e} "
        f"(z-asymmetric fields expose wrong-neighbour streaming)")


@pytest.mark.parametrize("field", sorted(_FIELD_BUILDERS))
def test_asymmetric_fields_actually_break_symmetry(field: str, dev: str) -> None:
    """Meta-test: each initial field must NOT be invariant under the buggy
    lane permutation (8<->10, 12<->14, 16<->18), otherwise the parity test
    above could pass even with wrong tables."""
    n = 32
    torch.manual_seed(20260819)
    f0 = _FIELD_BUILDERS[field](n, dev)
    perm = list(range(19))
    perm[8], perm[10] = perm[10], perm[8]
    perm[12], perm[14] = perm[14], perm[12]
    perm[16], perm[18] = perm[18], perm[16]
    delta = (f0 - f0[perm]).abs().max().item()
    assert delta > 0.0, (
        f"field '{field}' is invariant under the buggy lane permutation "
        f"(max|delta|={delta:.3e}); the parity test would be vacuous for it")
