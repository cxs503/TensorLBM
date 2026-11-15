"""Tests for the pylbm-inspired symbolic scheme builder.

Covers: geometry mask generation, scheme initialisation, collide-stream-BC
loop, macroscopic recovery, Re-based tau computation, obstacle placement,
and round-trip serialisation.
"""

from __future__ import annotations

import torch

from tensorlbm.symbolic_scheme import GeometrySpec, SchemeSpec, SymbolicScheme


# ---------------------------------------------------------------------------
# 1. GeometrySpec: solid mask from walls (top/bottom only are solid)
# ---------------------------------------------------------------------------
def test_geometry_walls_mask():
    """Only top/bottom walls produce solid cells; left/right are BC columns."""
    geom = GeometrySpec(shape=(64, 32), walls=["top", "bottom", "left", "right"])
    solid = geom.build_solid_mask()
    assert solid.shape == (32, 64)
    # Top and bottom edges are solid
    assert solid[0, :].all()
    assert solid[-1, :].all()
    # Left and right are NOT solid (they are BC columns)
    # Left column: only top/bottom cells are solid (corners)
    # Right column: only top/bottom cells are solid (corners)
    # Interior is fluid
    assert not solid[1:-1, 1:-1].any()


# ---------------------------------------------------------------------------
# 2. GeometrySpec: circular obstacle
# ---------------------------------------------------------------------------
def test_geometry_circle_obstacle():
    """A circular obstacle marks the correct cells as solid."""
    geom = GeometrySpec(
        shape=(64, 32),
        walls=["top", "bottom"],
        obstacles=[{"type": "circle", "center": (32, 16), "radius": 5}],
    )
    solid = geom.build_solid_mask()
    # Center of circle must be solid
    assert solid[16, 32]
    # Far corners must be fluid
    assert not solid[16, 1]
    assert not solid[16, 62]


# ---------------------------------------------------------------------------
# 3. GeometrySpec: rectangular obstacle
# ---------------------------------------------------------------------------
def test_geometry_rectangle_obstacle():
    """A rectangular obstacle marks the correct cells as solid."""
    geom = GeometrySpec(
        shape=(64, 32),
        walls=["top", "bottom"],
        obstacles=[{"type": "rectangle", "center": (32, 16), "size": (10, 8)}],
    )
    solid = geom.build_solid_mask()
    # Center must be solid
    assert solid[16, 32]
    # Corner of rectangle
    assert solid[12, 27]
    # Outside rectangle
    assert not solid[16, 5]


# ---------------------------------------------------------------------------
# 4. SchemeSpec: tau computation from Re
# ---------------------------------------------------------------------------
def test_scheme_tau_from_reynolds():
    """Tau is correctly computed from Reynolds number."""
    geom = GeometrySpec(shape=(128, 64), walls=["top", "bottom"])
    scheme = SchemeSpec(reynolds=100.0, u_in=0.05, char_length=64.0)
    tau = scheme.compute_tau(geom)
    expected_tau = 3.0 * (0.05 * 64.0 / 100.0) + 0.5
    assert abs(tau - expected_tau) < 1e-10


def test_scheme_tau_explicit():
    """Explicit tau is returned unchanged."""
    geom = GeometrySpec(shape=(64, 32))
    scheme = SchemeSpec(tau=0.7)
    assert scheme.compute_tau(geom) == 0.7


# ---------------------------------------------------------------------------
# 5. SymbolicScheme: init produces valid state
# ---------------------------------------------------------------------------
def test_symbolic_init():
    """Initialisation creates f with correct shape and mass conservation."""
    scheme = SymbolicScheme(
        geometry={"shape": (64, 32), "walls": ["top", "bottom"]},
        tau=0.6,
        u_in=0.05,
    )
    scheme.init()
    assert scheme.f.shape == (9, 32, 64)
    # Total mass should be close to fluid_count * rho0
    fluid_count = (~scheme.solid).sum().item()
    total_mass = scheme.f.sum().item()
    assert abs(total_mass - fluid_count) < fluid_count * 0.1


# ---------------------------------------------------------------------------
# 6. SymbolicScheme: step runs without error (short run)
# ---------------------------------------------------------------------------
def test_symbolic_step():
    """A few steps complete without NaN."""
    scheme = SymbolicScheme(
        geometry={"shape": (64, 32), "walls": ["top", "bottom", "left", "right"]},
        tau=0.6,
        u_in=0.05,
    )
    scheme.init()
    for _ in range(10):
        scheme.step()
    rho, ux, uy = scheme.macroscopic()
    assert not torch.isnan(rho).any()
    assert not torch.isnan(ux).any()


# ---------------------------------------------------------------------------
# 7. SymbolicScheme: to_dict round-trip
# ---------------------------------------------------------------------------
def test_to_dict():
    """Serialisation captures all scheme parameters."""
    scheme = SymbolicScheme(
        geometry={"shape": (64, 32), "walls": ["top", "bottom"]},
        reynolds=200.0,
        u_in=0.03,
        char_length=32.0,
    )
    d = scheme.to_dict()
    assert d["geometry"]["shape"] == [64, 32]
    assert d["geometry"]["walls"] == ["top", "bottom"]
    assert d["scheme"]["reynolds"] == 200.0
    assert d["scheme"]["u_in"] == 0.03


# ---------------------------------------------------------------------------
# 8. SymbolicScheme: unsupported lattice raises
# ---------------------------------------------------------------------------
def test_unsupported_lattice():
    """Non-D2Q9 lattice raises ValueError."""
    try:
        SymbolicScheme(lattice="D3Q19")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# 9. SymbolicScheme: step before init raises
# ---------------------------------------------------------------------------
def test_step_before_init():
    """Calling step() before init() raises RuntimeError."""
    scheme = SymbolicScheme(tau=0.6)
    try:
        scheme.step()
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# 10. SymbolicScheme: geometry with no walls
# ---------------------------------------------------------------------------
def test_no_walls_geometry():
    """Geometry with no walls produces empty solid mask."""
    geom = GeometrySpec(shape=(32, 16), walls=[])
    solid = geom.build_solid_mask()
    assert not solid.any()


# ---------------------------------------------------------------------------
# 11. SymbolicScheme: macroscopic after init
# ---------------------------------------------------------------------------
def test_macroscopic_after_init():
    """Macroscopic variables are recoverable after init."""
    scheme = SymbolicScheme(
        geometry={"shape": (32, 16), "walls": ["top", "bottom"]},
        tau=0.6,
        u_in=0.05,
    )
    scheme.init()
    rho, ux, uy = scheme.macroscopic()
    assert rho.shape == (16, 32)
    assert ux.shape == (16, 32)
    assert uy.shape == (16, 32)
    # rho should be close to 1.0 in fluid region
    fluid_rho = rho[~scheme.solid]
    assert (fluid_rho - 1.0).abs().max().item() < 0.1


# ---------------------------------------------------------------------------
# 12. SymbolicScheme: run method
# ---------------------------------------------------------------------------
def test_run_method():
    """run() executes the specified number of steps."""
    scheme = SymbolicScheme(
        geometry={"shape": (32, 16), "walls": ["top", "bottom"]},
        tau=0.6,
        u_in=0.05,
    )
    scheme.init()
    scheme.run(5)
    # Should not raise
    rho, ux, uy = scheme.macroscopic()
    assert not torch.isnan(rho).any()
