"""TDD regression for analytic circle/BFL link intersections."""

import importlib.util
from pathlib import Path

import torch

_SPEC = importlib.util.spec_from_file_location(
    "benchmark_magnus_cylinder",
    Path(__file__).parents[1] / "examples" / "benchmark_magnus_cylinder.py",
)
_MAGNUS = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MAGNUS)


def test_circle_link_fraction_uses_actual_circle_intersection_not_sdf_chord():
    """The wall fraction is the ray/circle intersection even on diagonal links."""
    y = torch.tensor([[[12.0]]])
    x = torch.tensor([[[15.0]]])
    # This diagonal link from (15,12) in direction (-1,-1) enters the circle
    # centred at (10,10), R=5.  The exact first root is (14-sqrt(164))/4.
    got = _MAGNUS.circle_link_fraction(x, y, cxq=-1, cyq=-1, cx=10.0, cy=10.0, R=5.0)
    expected = torch.tensor((14.0 - 164.0 ** 0.5) / 4.0)
    assert torch.allclose(got, expected.expand_as(got), atol=1e-6)
