"""Physics-level regression for the sub-cell (partially saturated) mask wall.

Round 1/2 of the mask-blend work verified a wall that sits on a cell face. The
library now also offers ``partial_cell_layer`` (Noble & Torczynski 1998), whose
wall may sit *inside* a cell. These tests pin the two claims that make that layer
usable, both against a closed-form reference rather than against another solver:

1. the published NT operator (weight ``B``, post-collision reflection) reproduces
   the analytic channel whose no-slip interface is at ``y_b = 0.5 + eps_s``
   (interface offset <= 0.05 dx, profile L2 < 1 % at eps_s = 0.5);
2. the naive transplant of upstream's float mask (raw ``eps_s`` as blend weight)
   does *not* -- its interface is displaced by a measurable amount, so a
   regression that silently reverted ``partial_cell_weight`` to ``eps_s`` fails
   here with a concrete number.

They are deliberately the same arms and the same geometry convention as
``benchmarks/xval_per_step_hook_mask_blend/run_r3_fractional.py``; the full sweep
and its figures live there, these keep the claim under ``pytest``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "xval_per_step_hook_mask_blend"
sys.path.insert(0, str(BENCH))

import run_xval as R  # noqa: E402

NY, NX, TAU, EPS = 16, 8, 0.8, 0.5
OFFSET_TOL_DX = 0.05


def _arm(mode: str) -> dict:
    return R.run_fractional_wall("couette", NY, NX, TAU, EPS, mode)


def test_nt_post_places_a_sub_cell_wall_on_the_analytic_interface():
    r = _arm("nt_post")
    assert r["finite"], "run produced a non-finite population"
    # exact interface for eps_s = 0.5 is y_b = 1.0 dx, i.e. one full cell below the
    # first fluid row's centre difference; the measured d_eff must land on it
    assert abs(r["offset_err_dx"]) <= OFFSET_TOL_DX, (
        f"NT post-collision wall displaced by {r['offset_err_dx']:+.4f} dx "
        f"(tolerance {OFFSET_TOL_DX} dx)"
    )
    assert r["profile_l2_pct"] < 1.0, f"profile L2 {r['profile_l2_pct']:.4f} % >= 1 %"


def test_raw_solid_fraction_is_not_a_valid_blend_weight():
    naive = _arm("naive_eps")
    nt = _arm("nt_post")
    # the naive transplant is the failure mode the port exists to avoid
    assert abs(naive["offset_err_dx"]) >= 0.25, (
        "raw eps_s as a blend weight was expected to displace the wall by >= 0.25 dx, "
        f"measured {naive['offset_err_dx']:+.4f} dx -- if this ever passes the "
        "upstream transplant has silently changed"
    )
    assert abs(nt["offset_err_dx"]) < 0.5 * abs(naive["offset_err_dx"]), (
        f"NT ({nt['offset_err_dx']:+.4f} dx) is not at least twice as accurate as the "
        f"raw-eps_s blend ({naive['offset_err_dx']:+.4f} dx)"
    )
    assert nt["profile_l2_pct"] < naive["profile_l2_pct"]


def test_pre_collision_reflection_is_worse_than_post_collision():
    pre = _arm("nt_pre")
    post = _arm("nt_post")
    assert abs(pre["offset_err_dx"]) > abs(post["offset_err_dx"]), (
        "the pre-collision NT variant was expected to be the worse one "
        f"(pre {pre['offset_err_dx']:+.4f} dx vs post {post['offset_err_dx']:+.4f} dx); "
        "if they coincide the 'reflect=post' default is a preference, not a requirement"
    )


def test_fractional_limits_are_bitwise():
    lim = R.run_fractional_limits(tau=TAU, ny=NY, nx=NX)
    assert lim["bit_identical_eps1"], (
        f"eps_s=1 vs bounce max rel {lim['eps1_vs_bounce_max_rel']:.3e} is not bitwise"
    )
    assert lim["bit_identical_eps0"], (
        f"eps_s=0 vs identity max rel {lim['eps0_vs_identity_max_rel']:.3e} is not bitwise"
    )
    assert lim["partial_blob_mass_rel"] < 1e-12, (
        f"interior partial cell is not mass conserving: rel {lim['partial_blob_mass_rel']:.3e}"
    )
    # B(eps) must be the NT expression, not eps itself
    w = lim["weights"]
    expected = 0.5 * (TAU - 0.5) / (0.5 + (TAU - 0.5))
    assert abs(w["eps_0.5"] - expected) < 1e-12
    assert all(w[f"eps_{k}"] < float(k) for k in ("0.25", "0.5", "0.75"))


def test_nt_post_collapses_to_bounce_back_at_eps_one():
    """The post-collision NT operator now correctly collapses to bounce-back at eps_s=1."""
    r = R.run_fractional_wall("couette", NY, NX, TAU, 1.0, "nt_post")
    assert abs(r["offset_err_dx"]) <= OFFSET_TOL_DX, (
        f"eps_s=1 post-collision interface displaced by {r['offset_err_dx']:+.4f} dx"
    )
