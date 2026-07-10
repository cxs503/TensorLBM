"""Geometry regressions for the Magnus Bouzidi boundary links."""

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


def test_exact_geometric_surface_nodes_are_solid_not_zero_fraction_fluid_links():
    """A node exactly on an analytic cylinder must not generate q≈0 BFL links.

    With ``dist < R`` the axial points at (cx±R, cy) are labelled fluid while
    their immediate interior neighbours are solid.  Their signed-distance
    interpolation gives delta=0, which is an invalid limiting BFL link and
    injects an extrapolated population at high wall speed.  The zero level set
    belongs to the rigid body, leaving only strictly positive-distance fluid
    sources for BFL.
    """
    solid, phi = _MAGNUS.build_cylinder_geometry(
        nx=21, ny=21, nz=1, R=5.0, cx=10.0, cy=10.0, device=torch.device("cpu")
    )

    assert solid[0, 10, 15]
    assert solid[0, 10, 5]
    assert not solid[0, 10, 16]
    assert torch.isclose(phi[0, 10, 15], torch.tensor(0.0))
