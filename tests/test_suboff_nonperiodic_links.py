"""Regression for nonperiodic D3Q19 wall links; CPU only."""

import pytest
import torch
from tensorlbm.d3q19 import C
from tensorlbm.boundary.nonperiodic_links import halfway_fluid_solid_links_3d


def brute(solid):
    out = torch.zeros((19, *solid.shape), dtype=torch.bool)
    nz, ny, nx = solid.shape
    for i in range(1, 19):
        cx, cy, cz = map(int, C[i].tolist())
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    zz, yy, xx = z + cz, y + cy, x + cx
                    if 0 <= zz < nz and 0 <= yy < ny and 0 <= xx < nx:
                        out[i, z, y, x] = (not bool(solid[z, y, x])) and bool(solid[zz, yy, xx])
    return out


@pytest.mark.parametrize("axis,side", [(a, s) for a in range(3) for s in (0, -1)])
def test_no_wrap_and_all_directions(axis, side):
    solid = torch.zeros((4, 5, 6), dtype=torch.bool)
    idx = [2, 2, 3]
    idx[axis] = side
    solid[tuple(idx)] = True
    links, q = halfway_fluid_solid_links_3d(solid)
    assert torch.equal(links, brute(solid))
    assert not links[:, tuple(idx)[0], tuple(idx)[1], tuple(idx)[2]].any()
    assert torch.all(q == 0.5)


def test_interior_diagonal_and_input_contract():
    solid = torch.zeros((5, 5, 5), dtype=torch.bool)
    solid[2, 2, 2] = True
    links, _ = halfway_fluid_solid_links_3d(solid)
    assert torch.equal(links, brute(solid))
    assert int(links.sum()) == 18
    with pytest.raises(ValueError):
        halfway_fluid_solid_links_3d(solid.float())
