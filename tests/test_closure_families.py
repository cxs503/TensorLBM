"""Tests for the 2x2 closure-family axes (collision x SGS) in autograd_calib.

The family experiment (2026-08-23, ``runs/b3_famil_20260823``) showed on the
HullCase that the collision axis moves windowed C_D by ~2.7% (BGK -> MRT)
while the SGS axis moves it ~0.1% (WALE sweep) — these tests pin the
*machinery*: default bitwise-identity, fp64 MRT support, gradient
correctness through every family, and family bookkeeping end to end.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.autograd_calib import (
    COLLISION_FAMILIES,
    SGS_MODELS,
    BoxCase,
    bounded_drag,
    calibrate,
    cs_power,
    evaluate,
    synthetic_targets,
)
from tensorlbm.solver3d import (
    _get_d3q19_mrt_matrices,
    collide_mrt3d,
)

# Identifiability-safe small box (see test_autograd_calib.py header note);
# fp64 by default, short window to keep the CPU suite fast.
BOX = BoxCase(nz=6, ny=8, nx=18, radius=2, u_in=0.20, steps=40, window_start=30)
RE = 48.0


def test_family_names_are_the_public_axes() -> None:
    assert COLLISION_FAMILIES == ("bgk", "mrt")
    assert SGS_MODELS == ("smagorinsky", "wale")


def test_invalid_family_raises() -> None:
    with pytest.raises(ValueError, match="collision must be one of"):
        bounded_drag(BOX, RE, cs=0.1, collision="kbc")
    with pytest.raises(ValueError, match="sgs must be one of"):
        bounded_drag(BOX, RE, cs=0.1, sgs="dns")
    one = synthetic_targets(BOX, (RE,), cs_power(0.1, -1.0, re_ref=40.0))
    with pytest.raises(ValueError, match="collision must be one of"):
        calibrate(one, BOX, collision="trt")
    with pytest.raises(ValueError, match="sgs must be one of"):
        synthetic_targets(BOX, (RE,), cs_power(0.1, -1.0, re_ref=40.0), sgs="vreman")


def test_default_is_bitwise_plain_bgk() -> None:
    """The default axes stay the historical cs=None BGK fast path, bitwise."""
    cd_default = bounded_drag(BOX, RE, cs=None)
    cd_explicit = bounded_drag(BOX, RE, cs=None, collision="bgk", sgs="smagorinsky")
    cd_again = bounded_drag(BOX, RE, cs=None)
    assert not cd_default.requires_grad
    assert float(cd_default) == float(cd_explicit) == float(cd_again)
    assert torch.isfinite(cd_default) and float(cd_default) > 0.0


def test_every_family_runs_and_has_finite_gradient() -> None:
    for collision in COLLISION_FAMILIES:
        for sgs in SGS_MODELS:
            cs = torch.tensor(0.15, dtype=BOX.dtype, requires_grad=True)
            cd = bounded_drag(BOX, RE, cs=cs, collision=collision, sgs=sgs)
            cd.backward()
            assert torch.isfinite(cd), (collision, sgs)
            assert torch.isfinite(cs.grad), (collision, sgs)
            assert float(cs.grad) != 0.0, (collision, sgs)


def test_wale_gradient_matches_finite_differences() -> None:
    cw = torch.tensor(0.30, dtype=BOX.dtype, requires_grad=True)
    cd = bounded_drag(BOX, RE, cs=cw, collision="bgk", sgs="wale")
    cd.backward()
    h = 1e-6
    up = float(bounded_drag(BOX, RE, cs=0.30 + h, collision="bgk", sgs="wale"))
    dn = float(bounded_drag(BOX, RE, cs=0.30 - h, collision="bgk", sgs="wale"))
    rel = abs(float(cw.grad) - (up - dn) / (2 * h)) / abs((up - dn) / (2 * h))
    assert rel < 1e-6


def test_mrt_floor_moves_drag_relative_to_bgk() -> None:
    """The scientific point of the collision axis (weak but nonzero on C_D)."""
    with torch.no_grad():
        cd_bgk = float(bounded_drag(BOX, RE, cs=None, collision="bgk"))
        cd_mrt = float(bounded_drag(BOX, RE, cs=None, collision="mrt"))
    assert cd_bgk != cd_mrt
    assert abs(cd_mrt - cd_bgk) / cd_bgk < 0.3  # same physics, different collision


# ---------------------------------------------------------------------------
# dtype-aware MRT matrices (the fp64 enabler)
# ---------------------------------------------------------------------------


def test_mrt_matrices_default_and_dtype() -> None:
    device = torch.device("cpu")
    m32, mi32 = _get_d3q19_mrt_matrices(device)
    assert m32.dtype == torch.float32 and mi32.dtype == torch.float32
    m64, mi64 = _get_d3q19_mrt_matrices(device, torch.float64)
    assert m64.dtype == torch.float64 and mi64.dtype == torch.float64
    assert torch.allclose(m64, m32.double())
    assert torch.allclose(mi64 @ m64, torch.eye(19, dtype=torch.float64), atol=1e-10)


def test_collide_mrt3d_accepts_float64() -> None:
    torch.manual_seed(0)
    f = torch.rand(19, 4, 4, 5, dtype=torch.float64) + 0.1
    out = collide_mrt3d(f, tau=0.7)
    assert out.dtype == torch.float64
    assert torch.isfinite(out).all()
    # invariants: mass conserved by the MRT relaxation of the density mode (s_0 = 0)
    assert torch.allclose(out.sum(dim=0), f.sum(dim=0), rtol=1e-10)


def test_sgs_mrt_kernels_accept_float64() -> None:
    from tensorlbm.turbulence import (
        collide_smagorinsky_mrt3d,
        collide_wale_mrt3d,
    )

    torch.manual_seed(0)
    f = torch.rand(19, 4, 4, 5, dtype=torch.float64) + 0.1
    for kernel, coeff in (
        (collide_smagorinsky_mrt3d, {"C_s": 0.1}),
        (collide_wale_mrt3d, {"C_w": 0.3}),
    ):
        out = kernel(f, tau=0.7, **coeff)
        assert out.dtype == torch.float64
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# family bookkeeping through calibrate/evaluate
# ---------------------------------------------------------------------------


def test_calibrate_records_family_and_evaluate_reuses_it() -> None:
    truth = cs_power(0.10, -1.0, re_ref=40.0)
    targets = synthetic_targets(BOX, (30.0, 48.0, 70.0), truth, collision="mrt", sgs="wale")
    res = calibrate(
        targets, BOX, kind="scalar", cs0=0.05, iters=8, lr=0.05, collision="mrt", sgs="wale"
    )
    assert (res.collision, res.sgs) == ("mrt", "wale")

    # evaluate() with no explicit family reuses the recorded axes: its
    # prediction must match a hand-rolled mrt+wale rollout, not bgk+smag
    ev = evaluate(res, targets, BOX)
    with torch.no_grad():
        expect_mrt_wale = float(bounded_drag(BOX, RE, cs=res.cs(RE), collision="mrt", sgs="wale"))
        wrong_family = float(
            bounded_drag(BOX, RE, cs=res.cs(RE))  # bgk + smagorinsky
        )
    got = ev[f"{RE:g}"]["pred"]
    assert math.isclose(got, expect_mrt_wale, rel_tol=1e-12)
    assert not math.isclose(got, wrong_family, rel_tol=1e-9)


def test_evaluate_explicit_family_overrides() -> None:
    truth = cs_power(0.10, -1.0, re_ref=40.0)
    targets = synthetic_targets(BOX, (48.0,), truth)
    res = calibrate(targets, BOX, kind="scalar", cs0=0.05, iters=4, lr=0.05)
    ev = evaluate(res, targets, BOX, collision="mrt")
    with torch.no_grad():
        expect = float(bounded_drag(BOX, RE, cs=res.cs(RE), collision="mrt"))
    assert math.isclose(ev[f"{RE:g}"]["pred"], expect, rel_tol=1e-12)
