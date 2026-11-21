"""XVAL-LBM-SPARSE-SUBLIST-NEIGHBOR-LIST: operator equivalence and physics anchors.

Cross-validates the sparse-sublist neighbour-list approach of
``sorush-khajepor/listLBM`` (transplanted into TensorLBM as
``tensorlbm.sparse_sublist_neighbor_list``) against

  A1  the ``FluidNodeTable`` push kernel of the same repo (representation
      control -- both tables are built by the same row-major compaction), and
  A2  the analytic plane-Poiseuille permeability ``k = H^2/12`` (physical
      anchor).

The tests pin three things:

1. **Table equivalence.** ``build_sparse_sublist_neighbor_list`` produces the
   same ``nodes`` / ``neighbour`` / ``index`` as ``build_fluid_node_table`` on
   the same geometry (bit for bit).
2. **Operator equivalence.** The listLBM swap pair (``stream1`` + ``stream2``)
   reproduces the ``FluidNodeTable`` push kernel bit-for-bit on a random porous
   geometry (representation neutrality).
3. **Physics unchanged.** The solver reproduces the analytic plane-Poiseuille
   permeability within the predeclared 15 % band.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensorlbm.sparse_sublist_neighbor_list import (  # noqa: E402
    HALFQS,
    build_sparse_sublist_neighbor_list,
    listlbm_stream,
    listlbm_stream1,
    listlbm_stream2,
    listlbm_step,
)
from tensorlbm.porous import (  # noqa: E402
    DIRS,
    build_fluid_node_table,
    compressed_step,
    dense_step,
    macroscopic_with_forcing,
    permeability,
)
from tensorlbm.d2q9 import C, OPPOSITE  # noqa: E402

torch.set_default_dtype(torch.float64)  # the module is a float64 reference implementation

TAU = 0.8
NU = (TAU - 0.5) / 3.0


# --------------------------------------------------------------------- helpers
def random_porous(n: int = 24, seed: int = 7, p_solid: float = 0.3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    solid = torch.rand((n, n), generator=g) < p_solid
    solid[0, 0] = False  # keep at least one fluid node
    return solid


def square_grain(n: int, porosity: float) -> torch.Tensor:
    solid = torch.zeros((n, n), dtype=torch.bool)
    side = int(round(n * math.sqrt((1.0 - porosity) / 4.0)))
    off = [n // 4, 3 * n // 4]
    for y0 in off:
        for x0 in off:
            solid[y0 - side // 2 : y0 - side // 2 + side, x0 - side // 2 : x0 - side // 2 + side] = True
    return solid


def channel_mask(ny: int, nx: int, h: int) -> torch.Tensor:
    """Plane channel: ``h`` fluid rows centred in a ``ny``-tall periodic slab."""
    solid = torch.ones((ny, nx), dtype=torch.bool)
    y0 = (ny - h) // 2
    solid[y0 : y0 + h, :] = False
    return solid


# --------------------------------------------------- 1. table equivalence
def test_table_equivalence_with_fluid_node_table():
    """Both tables are built by the same row-major compaction -> bit identical."""
    for seed in (1, 5, 13):
        solid = random_porous(n=20, seed=seed, p_solid=0.3)
        t1 = build_fluid_node_table(solid)
        t2 = build_sparse_sublist_neighbor_list(solid)
        assert torch.equal(t1.nodes, t2.nodes)
        assert torch.equal(t1.neighbour, t2.neighbour)
        assert torch.equal(t1.index, t2.index)
        assert t1.n_fluid == t2.n_fluid
        assert t1.porosity == t2.porosity
        # the OSS fluidlist_t is the whole fluid range
        assert t2.sublist == (0, t2.n_fluid - 1)


def test_halfqs_are_one_from_each_opposite_pair():
    """``HALFQS`` must be one from each opposite pair (the OSS ``halfqs``)."""
    seen = set()
    for d in HALFQS:
        opp = int(OPPOSITE[d])
        assert opp not in seen, f"{d} and {opp} are both in HALFQS"
        seen.add(d)
        seen.add(opp)
    assert len(seen) == 2 * len(HALFQS)  # all 8 non-rest directions covered


# --------------------------------------------------- 2. operator equivalence
@pytest.mark.parametrize("seed", [1, 5, 13])
def test_listlbm_stream_equals_fluid_node_table_push(seed):
    """The swap pair (stream1+stream2) == the push kernel, bit for bit."""
    solid = random_porous(n=20, seed=seed, p_solid=0.3)
    t1 = build_fluid_node_table(solid)
    t2 = build_sparse_sublist_neighbor_list(solid)
    g = torch.Generator().manual_seed(seed)
    f = torch.rand((9, t1.n_fluid), generator=g)
    f[0] = f[0] * 0.0 + 4.0 / 9.0  # keep the rest slot finite

    # one full step: collide + stream (both arms share the same collision)
    from tensorlbm.porous import collide_with_guo_forcing as _c

    f_post = _c(f, TAU, 0.0, 0.0)

    # A1: FluidNodeTable push
    f_push = torch.empty_like(f_post)
    f_push[0] = f_post[0]
    for slot, d in enumerate(DIRS):
        nb = t1.neighbour[:, slot]
        ok = nb >= 0
        if bool(ok.any()):
            src = torch.nonzero(ok, as_tuple=False).squeeze(1)
            f_push[d][nb[src]] = f_post[d][src]
        wall = torch.nonzero(~ok, as_tuple=False).squeeze(1)
        if wall.numel():
            f_push[int(OPPOSITE[d])][wall] = f_post[d][wall]

    # A2: listLBM swap pair
    f_swap = listlbm_stream(f_post, t2)

    assert torch.allclose(f_swap, f_push, atol=1e-14), (
        f"swap vs push max abs diff = {(f_swap - f_push).abs().max().item():.2e}"
    )


@pytest.mark.parametrize("seed", [1, 5, 13])
def test_listlbm_step_equals_compressed_step(seed):
    """Full listLBM step (collide + swap) == FluidNodeTable step (collide + push)."""
    solid = random_porous(n=20, seed=seed, p_solid=0.3)
    t1 = build_fluid_node_table(solid)
    t2 = build_sparse_sublist_neighbor_list(solid)
    g = torch.Generator().manual_seed(seed)
    f = torch.rand((9, t1.n_fluid), generator=g)
    f[0] = f[0] * 0.0 + 4.0 / 9.0

    f1 = compressed_step(f, t1, TAU, 1e-5, 0.0)
    f2 = listlbm_step(f, t2, TAU, 1e-5, 0.0)
    assert torch.allclose(f2, f1, atol=1e-14), (
        f"listlbm vs compressed max abs diff = {(f2 - f1).abs().max().item():.2e}"
    )


# --------------------------------------------------- 3. physics
def test_plane_channel_permeability_within_15pct():
    """Plane channel: ``k = H^2/12`` within the owner 15 % band.

    ``ux`` from :func:`macroscopic_with_forcing` is the *intrinsic* velocity
    (averaged over the ``K`` fluid nodes only); the *superficial* permeability
    ``k_super = phi * H^2/12`` with ``phi = H/ny`` is the analytic reference.
    """
    H, NY, NX = 16, 32, 16
    solid = channel_mask(NY, NX, H)
    table = build_sparse_sublist_neighbor_list(solid)
    g = 1e-5
    f = torch.zeros((9, table.n_fluid))
    f[0] = 4.0 / 9.0
    f[1] = f[2] = f[3] = f[4] = 1.0 / 9.0
    f[5] = f[6] = f[7] = f[8] = 1.0 / 36.0
    for _ in range(2500):
        f = listlbm_step(f, table, TAU, g, 0.0)
    rho, ux, uy = macroscopic_with_forcing(f, g, 0.0)
    # intrinsic velocity -> superficial via porosity
    u_super = float(ux.mean()) * table.porosity
    k = permeability(u_super, NU, g)
    k_ref = (H / NY) * H * H / 12.0  # superficial slab permeability
    err = 100.0 * abs(k - k_ref) / k_ref
    assert err < 15.0, f"permeability error {err:.2f}% (k={k:.4f}, k_ref={k_ref:.4f})"


def test_mass_conservation():
    """The swap pair conserves mass (no drift over many steps)."""
    solid = random_porous(n=16, seed=7, p_solid=0.3)
    table = build_sparse_sublist_neighbor_list(solid)
    f = torch.zeros((9, table.n_fluid))
    f[0] = 4.0 / 9.0
    f[1] = f[2] = f[3] = f[4] = 1.0 / 9.0
    f[5] = f[6] = f[7] = f[8] = 1.0 / 36.0
    m0 = float(f.sum())
    for _ in range(200):
        f = listlbm_step(f, table, TAU, 0.0, 0.0)
    m1 = float(f.sum())
    assert abs(m1 - m0) / m0 < 1e-12, f"mass drift {(m1 - m0) / m0:.2e}"
