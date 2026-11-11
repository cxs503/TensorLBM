"""Tests for the interpolated (Bouzidi) bounce-back operator and the staircase-wall-offset bench.

Gate structure
--------------
1. ``test_q_half_reduces_to_halfway`` -- the operator must be *exactly* the
   production half-way swap when the sub-grid wall sits on the link midpoint
   (q = 1/2).  This is the negative control: it proves the interpolation is not
   changing physics where the two-point stencil degenerates.
2. ``test_rest_state_is_fixed_point`` -- a fluid at rest stays at rest.
3. ``test_input_contract`` -- SDF sign convention and shape are enforced (silent
   misuse would bias every wall in a geometry).
4. ``test_offset_bench_artifacts`` -- the recorded benchmark artifact must show the
   interpolated format removes the staircase offset (|delta_eff| <= 0.05 cell) and
   brings the profile error <= 3 %, while the half-way format does neither.  Skipped
   with an explicit reason when the artifact is absent (never silently passed).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tensorlbm.boundary.interpolated_bounce_back import (
    interpolated_bounce_back_2d,
    interpolated_bounce_back_3d,
)
from tensorlbm.boundaries import bounce_back_cells
from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.d2q9 import equilibrium
from tensorlbm.d3q19 import C as C3
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver import stream
from tensorlbm.solver3d import stream3d

ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/pending/staircase_wall_offset/artifacts/result.json"
)


def _solid_slab_3d(nz=8, ny=12, nx=10):
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    yy = torch.arange(ny).view(1, ny, 1).expand(nz, ny, nx).float()
    dist = torch.minimum(yy - 0.5, (ny - 1.5) - yy)
    sdf = torch.where(solid, -dist.abs().clamp(min=1e-6), dist)
    return solid, sdf, (nz, ny, nx)


def test_q_half_reduces_to_halfway():
    """q = 1/2 on every cut link -> identical to the production half-way swap."""
    torch.manual_seed(0)
    solid, sdf, shape = _solid_slab_3d()
    rng = torch.rand(shape)
    f_pre = equilibrium3d(torch.ones(shape), 0.01 * (rng - 0.5), 0.0 * rng, 0.0 * rng)
    f_st = stream3d(f_pre)
    ref = bounce_back_cells_3d(f_st.clone(), solid)
    out, rep = interpolated_bounce_back_3d(f_st.clone(), f_pre, sdf, solid, return_report=True)
    assert rep.fallback_links == 0
    assert rep.q_min == pytest.approx(0.5) and rep.q_max == pytest.approx(0.5)

    # only the cut links are written by the interpolating operator; the half-way
    # swap also overwrites non-physical interior-solid directions, so compare on
    # the cut links only.
    from tensorlbm.d3q19 import OPPOSITE as OPP3

    checked = 0
    for i in range(1, 19):
        d = int(OPP3[i])
        shift = tuple(int(v) for v in reversed(C3[i].tolist()))
        nb_solid = torch.roll(solid, shifts=tuple(-v for v in shift), dims=(0, 1, 2))
        link = (~solid) & nb_solid
        tgt = torch.roll(link, shift, dims=(0, 1, 2))
        if not bool(tgt.any()):
            continue
        assert torch.allclose(ref[d][tgt], out[d][tgt], atol=1e-7)
        checked += int(tgt.sum())
    assert checked > 0


def test_rest_state_is_fixed_point():
    """Uniform rest fluid is a fixed point of the interpolated operator."""
    solid, sdf, shape = _solid_slab_3d()
    rho = torch.ones(shape)
    zero = torch.zeros(shape)
    f = equilibrium3d(rho, zero, zero, zero)
    f_st = stream3d(f)
    out = interpolated_bounce_back_3d(f_st, f, sdf, solid)
    r, ux, uy, uz = macroscopic3d(out)
    fluid = ~solid
    # solid nodes hold only virtual (reflected) populations; the physical
    # statement is that the *fluid* stays at rest with rho = 1.
    assert float((ux[fluid].abs()).max()) < 1e-6
    assert float((uy[fluid].abs()).max()) < 1e-6
    assert float((uz[fluid].abs()).max()) < 1e-6
    assert float((r[fluid] - 1.0).abs().max()) < 1e-6


def test_input_contract():
    solid, sdf, shape = _solid_slab_3d()
    zero = torch.zeros(shape)
    f = equilibrium3d(torch.ones(shape), zero, zero, zero)

    bad = sdf.clone()
    bad[~solid] = -1.0
    with pytest.raises(ValueError, match="positive on fluid"):
        interpolated_bounce_back_3d(f, f, bad, solid)

    bad2 = sdf.clone()
    bad2[solid] = +1.0
    with pytest.raises(ValueError, match="negative on solid"):
        interpolated_bounce_back_3d(f, f, bad2, solid)

    with pytest.raises(ValueError, match="shape"):
        interpolated_bounce_back_3d(f, f, sdf[:-1], solid[:-1])


def test_q_above_half_stays_bounded():
    """A wall deeper than half a cell (q > 1/2) must keep weights bounded."""
    n = 16
    solid = torch.zeros((n, n), dtype=torch.bool)
    solid[0, :] = True
    yy = torch.arange(n).view(n, 1).expand(n, n).float()
    sdf = torch.where(solid, -(yy + 0.2).clamp(min=1e-6), yy - 0.2)
    rho = torch.ones((n, n))
    f = equilibrium(rho, 0.01 * torch.randn((n, n)), torch.zeros((n, n)))
    f_st = stream(f)
    out, rep = interpolated_bounce_back_2d(f_st, f, sdf, solid, return_report=True)
    assert rep.cut_links > 0
    assert 0.0 < rep.q_min <= rep.q_max <= 1.0
    assert torch.isfinite(out).all()


@pytest.mark.benchmark
def test_offset_bench_artifacts():
    if not ARTIFACT.exists():
        pytest.skip(f"benchmark artifact not found at {ARTIFACT}; run run_inclined.py --all")
    data = json.loads(ARTIFACT.read_text())
    rows = data["rows"]
    interp = [r for r in rows if r["wall"] == "interp"]
    half = [r for r in rows if r["wall"] == "halfway"]
    assert interp and half, "artifact must contain both wall formats"
    for r in interp:
        assert abs(r["delta_eff"]) <= 0.05, (
            f"{r['case']}: interpolated delta_eff={r['delta_eff']:+.3f} exceeds 0.05 cell"
        )
        assert r["theta_pct"] <= 3.0, (
            f"{r['case']}: interpolated profile error {r['theta_pct']:.2f}% > 3%"
        )
        assert math.isfinite(r["Q_over_Qana"])
    # the half-way format must show a *measurable* offset somewhere, otherwise the
    # benchmark would not be probing the mechanism at all
    assert any(abs(r["delta_eff"]) > 0.05 for r in half), (
        "half-way format shows no staircase offset -> benchmark does not exercise the defect"
    )