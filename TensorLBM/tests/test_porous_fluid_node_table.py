"""Tests for the compressed fluid-node neighbour table (openLBMPM transplant).

Covers, in the order of the XVAL argument:

1. table build -- compaction order, periodic wrap, solid sentinel, stencil order;
2. streaming equivalence -- compressed push vs dense pull must agree to
   float64 round-off on a random porous geometry (representation neutrality);
3. physics -- plane-channel permeability within the owner 15 % band of
   ``k = H^2/12``;
4. conservation -- mass drift of the push scheme;
5. footprint -- the compressed state is smaller than the dense state for
   porosity < 1.
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

from tensorlbm.porous import (  # noqa: E402
    DIRS,
    build_fluid_node_table,
    compressed_step,
    dense_step,
    dense_stream,
    macroscopic_with_forcing,
    intrinsic_permeability,
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


# ----------------------------------------------------------------- 1. table build
def test_compaction_order_and_inverse_map():
    solid = random_porous(n=16, seed=3)
    table = build_fluid_node_table(solid)
    fluid = ~solid
    ys, xs = fluid.nonzero(as_tuple=True)
    # row-major compaction, identical to openLBMPM optimizeFluidArray
    assert torch.equal(table.nodes, ys * 16 + xs)
    assert int(table.n_fluid) == int(fluid.sum())
    # index is the exact inverse
    assert torch.equal(table.index[ys, xs], torch.arange(table.n_fluid))
    assert torch.equal(table.index[solid], torch.full_like(table.index[solid], -1))
    assert table.porosity == pytest.approx(float(fluid.sum()) / (16 * 16))


def test_neighbour_table_matches_bruteforce_with_periodic_wrap():
    solid = random_porous(n=12, seed=11, p_solid=0.35)
    table = build_fluid_node_table(solid)
    ny = nx = 12
    for k in range(table.n_fluid):
        y, x = divmod(int(table.nodes[k]), nx)
        for slot, d in enumerate(DIRS):
            cy, cx = int(C[d, 1]), int(C[d, 0])
            yy, xx = (y + cy) % ny, (x + cx) % nx  # OSS wraps at build time
            expected = -1 if solid[yy, xx] else int(table.index[yy, xx])
            assert int(table.neighbour[k, slot]) == expected, (k, slot)
    # stencil order is E, N, W, S, NE, NW, SW, SE == d2q9 directions 1..8
    assert DIRS == (1, 2, 3, 4, 5, 6, 7, 8)


def test_table_reports_solid_sentinel_only_for_solid():
    solid = square_grain(32, 0.75)
    table = build_fluid_node_table(solid)
    assert int((table.neighbour == -1).any())
    # a solid-free lattice has no sentinel at all
    empty = build_fluid_node_table(torch.zeros((8, 8), dtype=torch.bool))
    assert int((empty.neighbour == -1).sum()) == 0
    assert empty.neighbour.shape == (64, 8)


# --------------------------------------------------- 2. representation identity
@pytest.mark.parametrize("seed", [1, 5, 13])
def test_compressed_and_dense_steps_agree_to_roundoff(seed):
    solid = random_porous(n=20, seed=seed, p_solid=0.3)
    table = build_fluid_node_table(solid)
    ny = nx = 20
    fluid = ~solid
    g = 1e-5
    f_c = torch.zeros((9, table.n_fluid))
    f_d = torch.zeros((9, ny, nx))
    # identical initial state: rest + rho=1 on both layouts
    for f in (f_c, f_d):
        f[0] = 4.0 / 9.0
        f[1] = f[2] = f[3] = f[4] = 1.0 / 9.0
        f[5] = f[6] = f[7] = f[8] = 1.0 / 36.0
    for _ in range(25):
        f_c = compressed_step(f_c, table, TAU, g, 0.0)
        f_d = dense_step(f_d, solid, TAU, g, 0.0)
    rho_c, ux_c, uy_c = macroscopic_with_forcing(f_c, g, 0.0)
    rho_d, ux_d, uy_d = macroscopic_with_forcing(f_d, g, 0.0)
    assert torch.allclose(ux_c, ux_d[fluid], atol=1e-13)
    assert torch.allclose(uy_c, uy_d[fluid], atol=1e-13)
    assert torch.allclose(rho_c, rho_d[fluid], atol=1e-13)


def test_dense_stream_matches_repo_kernel_under_mirrored_labelling():
    """On a solid-free lattice the dense pull must equal tensorlbm.d2q9.stream.

    ``d2q9.stream`` labels slots by the side the population arrives from, this
    module labels them by velocity direction, so the kernel must agree with the
    repo kernel after the mirror map ``d -> opposite(d)``.
    """
    from tensorlbm.d2q9 import stream as repo_stream

    g = torch.Generator().manual_seed(2)
    f = torch.rand((9, 12, 10), generator=g)
    f[0] = f[0] * 0.0 + 1.0  # keep the rest slot finite and identical
    mirror = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6])
    out = dense_stream(f, torch.zeros((12, 10), dtype=torch.bool))
    ref = repo_stream(f[mirror])[mirror]
    assert torch.allclose(out, ref, atol=1e-15)


def test_bounce_back_is_half_way_on_a_closed_box():
    """No forcing, uniform state on a sealed fluid pocket -> state must not drift."""
    solid = torch.ones((9, 9), dtype=torch.bool)
    solid[3:6, 3:6] = False  # 3x3 fluid pocket fully enclosed
    table = build_fluid_node_table(solid)
    f = torch.zeros((9, table.n_fluid))
    for d, w in enumerate((4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36)):
        f[d] = w
    f0 = f.clone()
    for _ in range(500):
        f = compressed_step(f, table, TAU)
    assert torch.allclose(f, f0, atol=1e-12)
    rho, ux, uy = macroscopic_with_forcing(f)
    assert float(rho.min()) == pytest.approx(1.0, abs=1e-12)
    assert float(ux.abs().max()) < 1e-12


def test_mass_is_conserved_by_the_push_scheme():
    solid = square_grain(32, 0.75)
    table = build_fluid_node_table(solid)
    f = torch.zeros((9, table.n_fluid))
    for d, w in enumerate((4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36)):
        f[d] = w
    rho0 = float(f.sum())
    for _ in range(200):
        f = compressed_step(f, table, TAU, 1e-5, 0.0)
    assert abs(float(f.sum()) - rho0) / rho0 < 1e-12


# ---------------------------------------------------------------- 3. physics
def test_channel_permeability_within_owner_15pct():
    """Plane Poiseuille driven by a body force: k = H^2/12 (half-way walls)."""
    ny, nx, H = 40, 16, 16
    solid = torch.ones((ny, nx), dtype=torch.bool)
    y0 = (ny - H) // 2
    solid[y0 : y0 + H, :] = False
    table = build_fluid_node_table(solid)
    g = 1e-5
    f = torch.zeros((9, table.n_fluid))
    for d, w in enumerate((4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36)):
        f[d] = w
    for _ in range(8000):
        f = compressed_step(f, table, TAU, g, 0.0)
    _, ux, _ = macroscopic_with_forcing(f, g, 0.0)
    u_sup = float(ux.sum()) / table.n_total
    k_super = permeability(u_sup, NU, g)
    k_intrinsic = intrinsic_permeability(k_super, table.porosity)
    k_ref = H * H / 12.0  # intrinsic parallel-plate permeability, half-way walls
    assert abs(k_intrinsic - k_ref) / k_ref < 0.15, (k_intrinsic, k_ref)
    # the superficial value must carry the porosity of the slab cell
    assert abs(k_super - table.porosity * k_ref) / (table.porosity * k_ref) < 0.15


def test_compressed_state_is_smaller_than_dense_for_porous_media():
    table = build_fluid_node_table(square_grain(64, 0.60))
    mem = table.state_bytes()
    assert mem["pdf_only_ratio"] > 1.5
    assert mem["compressed_table_bytes"] == 9 * table.n_fluid * 8


def test_opposite_mapping_used_by_push_streaming_is_an_involution():
    for d in DIRS:
        assert int(OPPOSITE[int(OPPOSITE[d])]) == d