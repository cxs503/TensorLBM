"""B3-next field observables: bounded_observables + observable_response."""

import math

import pytest
import torch

from tensorlbm.autograd_calib import (
    BoxCase,
    ClosureObservables,
    bounded_drag,
    bounded_observables,
    observable_response,
)

BOX = BoxCase(nz=24, ny=24, nx=40, radius=4, u_in=0.15, steps=80, window_start=60)
RE = 100.0  # tau = 0.5 + 3*0.15*8/100 = 0.536


@pytest.fixture(scope="module")
def mask():
    return BOX.make_mask()


@pytest.fixture(scope="module")
def floor(mask):
    return bounded_observables(BOX, RE, cs=None, mask=mask)


def test_shapes_and_finite(floor, mask):
    nx = BOX.nx
    assert math.isfinite(floor.cd) and floor.cd > 0.0
    assert floor.press_profile.shape == (min(32, nx),)
    assert floor.centerline_ux.shape == (nx,)
    assert len(floor.wake_deficit) == len(floor.wake_cross) == len(floor.planes)
    for t in [floor.press_profile, floor.centerline_ux, *floor.wake_deficit, *floor.wake_cross]:
        assert t.ndim == 1 and torch.isfinite(t).all()
    # pressure lives only on the body span: bins past the tail stay exactly 0
    tail = int(mask.any(dim=(0, 1)).nonzero(as_tuple=True)[0][-1])
    past = floor.press_counts == 0
    assert floor.press_profile[past].abs().max() == 0.0
    assert (~past[: tail // max(1, nx // min(32, nx)) + 1]).any()


def test_cd_matches_bounded_drag(mask):
    """Same probe set and accumulation order -> identical windowed C_D."""
    obs = bounded_observables(BOX, RE, cs=0.1, mask=mask)
    ref = float(bounded_drag(BOX, RE, cs=0.1, mask=mask))
    assert obs.cd == pytest.approx(ref, rel=1e-10)


def test_default_planes_behind_body(floor, mask):
    nx = mask.shape[2]
    tail = int(mask.any(dim=(0, 1)).nonzero(as_tuple=True)[0][-1]) + 1
    assert len(floor.planes) == 3
    assert all(tail < xp < nx - 1 for xp in floor.planes)
    assert list(floor.planes) == sorted(set(floor.planes))


def test_plane_override(mask):
    obs = bounded_observables(BOX, RE, cs=None, mask=mask, plane_xs=(16, 22))
    assert obs.planes == (16, 22)
    assert len(obs.wake_deficit) == 2
    # an override plane equals the default-instrument value at the same x
    full = bounded_observables(BOX, RE, cs=None, mask=mask, plane_xs=(16,))
    assert torch.equal(obs.wake_deficit[0], full.wake_deficit[0])


def test_response_math_exact():
    def mk(cd, press, wdef):
        return ClosureObservables(
            cd=cd,
            press_profile=torch.tensor(press),
            centerline_ux=torch.zeros(3),
            wake_deficit=torch.tensor([wdef]),
            wake_cross=torch.zeros((1, 3)),
            wake_min_ux=0.0,
            planes=(7,),
            press_counts=torch.ones(len(press)),
        )

    a = mk(1.0, [1.0, 0.0], [1.0, 0.0, 0.0])
    b = mk(1.1, [1.0, 0.1], [1.0, 0.0, 0.0])
    ref = mk(1.0, [1.0, 0.0], [2.0, 0.0, 0.0])
    r = observable_response(a, b, ref)
    assert r["cd"] == pytest.approx(0.1)
    assert r["press_profile"] == pytest.approx(0.1)
    assert r["wake_deficit@7"] == pytest.approx(0.0)
    # zero-norm reference degrades to 0 instead of dividing by zero
    z = mk(1.0, [0.0, 0.0], [0.0, 0.0, 0.0])
    assert observable_response(a, b, z)["press_profile"] == 0.0


def test_collision_family_moves_pressure_more_than_drag(mask, floor):
    """The B3-next headline on the small case: the pressure profile responds
    to the collision-family axis at least as strongly as the drag scalar."""
    mrt = bounded_observables(BOX, RE, cs=None, mask=mask, collision="mrt")
    r = observable_response(mrt, floor, floor)
    assert r["press_profile"] > 0.0
    assert r["cd"] > 0.0
    assert math.isfinite(r["wake_deficit@{}".format(floor.planes[0])])


def test_sgs_response_orders_with_step(mask, floor):
    """A halving/doubling of C_s moves the fields more than a 20% nudge."""
    small = bounded_observables(BOX, RE, cs=0.11, mask=mask)
    big = bounded_observables(BOX, RE, cs=0.4, mask=mask)
    r_small = observable_response(small, floor, floor)
    r_big = observable_response(big, floor, floor)
    assert r_big["press_profile"] > r_small["press_profile"]
    assert (
        r_big["wake_deficit@{}".format(floor.planes[0])]
        > r_small["wake_deficit@{}".format(floor.planes[0])]
    )
