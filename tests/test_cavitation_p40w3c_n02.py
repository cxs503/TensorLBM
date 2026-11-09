"""P40W3C-N02 additional tests: solver stability diagnostic.

These tests document the grid-resolution stability limit discovered during
mission m7603: the pressure projection becomes unstable at cells_per_chord >= 30.
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.cavitation_airfoil import (
    CavitationFlowConfig,
    MixtureCavitationSolver,
)

torch.set_num_threads(1)


def test_solver_stable_at_cpc20():
    """Solver is stable at cpc=20 (the only usable resolution for m7603)."""
    cfg = CavitationFlowConfig(
        airfoil="NACA0015",
        alpha_deg=5.0,
        sigma=0.8,
        cells_per_chord=20,
        domain_x_chords=4.0,
        domain_y_chords=2.0,
        le_x_chords=1.0,
        le_y_chords=1.0,
        n_steps=100,
        cfl=0.2,
        c_dest=0.0,
        c_prod=0.0,
    )
    solver = MixtureCavitationSolver(cfg)
    solver.u = solver.u.masked_fill(solver.damp_u > 1.5, 0.0)
    solver.v = solver.v.masked_fill(solver.damp_v > 1.5, 0.0)
    
    for _ in range(100):
        solver.step()
    
    # Pressure should stay bounded
    assert float(solver.p.abs().max()) < 5.0
    assert torch.isfinite(solver.u).all()
    assert torch.isfinite(solver.p).all()


@pytest.mark.xfail(
    reason="Known instability: pressure blows up at cpc>=30 (P40W3C-N02 m7603)",
    strict=False,
)
def test_solver_unstable_at_cpc30():
    """Solver is unstable at cpc=30 (documents the limitation)."""
    cfg = CavitationFlowConfig(
        airfoil="NACA0015",
        alpha_deg=5.0,
        sigma=0.8,
        cells_per_chord=30,
        domain_x_chords=4.0,
        domain_y_chords=2.0,
        le_x_chords=1.0,
        le_y_chords=1.0,
        n_steps=10,
        cfl=0.2,
        c_dest=0.0,
        c_prod=0.0,
    )
    solver = MixtureCavitationSolver(cfg)
    solver.u = solver.u.masked_fill(solver.damp_u > 1.5, 0.0)
    solver.v = solver.v.masked_fill(solver.damp_v > 1.5, 0.0)
    
    for _ in range(10):
        solver.step()
    
    # This will fail because pressure blows up
    assert float(solver.p.abs().max()) < 5.0


def test_isobaric_criterion_gives_small_cavity():
    """Isobaric cutoff criterion systematically underestimates cavity length."""
    cfg = CavitationFlowConfig(
        airfoil="NACA0015",
        alpha_deg=5.0,
        sigma=0.6,
        cells_per_chord=20,
        domain_x_chords=4.0,
        domain_y_chords=2.0,
        le_x_chords=1.0,
        le_y_chords=1.0,
        n_steps=280,
        cfl=0.2,
        c_dest=0.0,
        c_prod=0.0,
    )
    solver = MixtureCavitationSolver(cfg)
    solver.u = solver.u.masked_fill(solver.damp_u > 1.5, 0.0)
    solver.v = solver.v.masked_fill(solver.damp_v > 1.5, 0.0)
    
    for _ in range(280):
        solver.step()
    
    # Isobaric cutoff
    p_v = cfg.p_v
    alpha = torch.sigmoid((p_v - solver.p.double()) / (0.05 * abs(p_v)))
    
    from tensorlbm.cavitation_airfoil import cavity_metrics
    m = cavity_metrics(alpha, cfg, threshold=0.5, geometry=solver.geometry)
    
    # L_cav should be small (underestimates experiment by ~88%)
    assert m["l_cav"] < 0.2  # Experiment: 0.65
    # But there should be some cavity
    assert float((alpha > 0.5).sum()) > 0


def test_mass_conservation_at_cpc20():
    """Mass conservation is excellent at cpc=20."""
    cfg = CavitationFlowConfig(
        airfoil="NACA0015",
        alpha_deg=5.0,
        sigma=0.8,
        cells_per_chord=20,
        domain_x_chords=4.0,
        domain_y_chords=2.0,
        le_x_chords=1.0,
        le_y_chords=1.0,
        n_steps=200,
        cfl=0.2,
        c_dest=0.0,
        c_prod=0.0,
    )
    solver = MixtureCavitationSolver(cfg)
    solver.u = solver.u.masked_fill(solver.damp_u > 1.5, 0.0)
    solver.v = solver.v.masked_fill(solver.damp_v > 1.5, 0.0)
    
    for _ in range(200):
        solver.step()
    
    # Discrete divergence residual should be small
    assert solver.div_residual_max < 1e-4
    # Global balance should be excellent
    assert solver.global_balance_max < 1e-8
