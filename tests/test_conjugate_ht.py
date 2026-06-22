"""Tests for tensorlbm.conjugate_ht – conjugate heat transfer."""
from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(ny: int = 16, nx: int = 24, device: str = "cpu"):
    """Build a minimal CHTState for testing."""
    from tensorlbm.conjugate_ht import CHTState
    from tensorlbm.d2q9 import equilibrium
    from tensorlbm.thermal import equilibrium_thermal

    dev = torch.device(device)
    rho0 = torch.ones(ny, nx, device=dev)
    ux0 = torch.zeros(ny, nx, device=dev)
    uy0 = torch.zeros(ny, nx, device=dev)
    T0 = torch.zeros(ny, nx, device=dev)

    mask_solid = torch.zeros(ny, nx, dtype=torch.bool, device=dev)
    mask_solid[6:10, 8:16] = True  # embedded solid block

    f = equilibrium(rho0, ux0, uy0, device=dev)
    g = equilibrium_thermal(T0, ux0, uy0)
    T_s = T0.clone()

    return CHTState(f=f, g=g, T_s=T_s, mask_solid=mask_solid)


# ---------------------------------------------------------------------------
# CHTConfig
# ---------------------------------------------------------------------------

def test_cht_config_defaults():
    from tensorlbm.conjugate_ht import CHTConfig

    cfg = CHTConfig()
    assert cfg.tau_f > 0.5
    assert cfg.alpha_s > 0.0
    assert 0.0 <= cfg.T_cold < cfg.T_hot


def test_cht_config_custom():
    from tensorlbm.conjugate_ht import CHTConfig

    cfg = CHTConfig(tau_f=0.7, k_ratio=10.0, T_hot=2.0, T_cold=0.5)
    assert cfg.tau_f == 0.7
    assert cfg.k_ratio == 10.0


# ---------------------------------------------------------------------------
# CHTState
# ---------------------------------------------------------------------------

def test_cht_state_shapes():
    state = _make_state(ny=12, nx=16)
    assert state.f.shape == (9, 12, 16)
    assert state.g.shape == (5, 12, 16)
    assert state.T_s.shape == (12, 16)
    assert state.mask_solid.dtype == torch.bool


# ---------------------------------------------------------------------------
# cht_solid_diffusion_step
# ---------------------------------------------------------------------------

def test_solid_diffusion_step_shape():
    from tensorlbm.conjugate_ht import cht_solid_diffusion_step

    ny, nx = 12, 16
    T_s = torch.zeros(ny, nx)
    T_s[5, 8] = 1.0  # point source
    mask_solid = torch.ones(ny, nx, dtype=torch.bool)

    T_new = cht_solid_diffusion_step(T_s, mask_solid, alpha_s=0.1)
    assert T_new.shape == (ny, nx)


def test_solid_diffusion_step_only_updates_solid():
    from tensorlbm.conjugate_ht import cht_solid_diffusion_step

    ny, nx = 8, 8
    T_s = torch.ones(ny, nx)
    # Only the centre block is solid
    mask_solid = torch.zeros(ny, nx, dtype=torch.bool)
    mask_solid[2:6, 2:6] = True

    T_new = cht_solid_diffusion_step(T_s, mask_solid, alpha_s=0.1)
    # Fluid cells should be unchanged
    assert torch.allclose(T_new[~mask_solid], T_s[~mask_solid])


def test_solid_diffusion_step_heat_spreads():
    """Temperature should spread away from a hot spot in the solid."""
    from tensorlbm.conjugate_ht import cht_solid_diffusion_step

    ny, nx = 16, 16
    T_s = torch.zeros(ny, nx)
    T_s[8, 8] = 1.0
    mask_solid = torch.ones(ny, nx, dtype=torch.bool)

    T_new = cht_solid_diffusion_step(T_s, mask_solid, alpha_s=0.2)
    # Neighbours of the hot spot should have picked up heat
    assert T_new[8, 9].item() > 0.0
    assert T_new[8, 7].item() > 0.0
    assert T_new[7, 8].item() > 0.0
    assert T_new[9, 8].item() > 0.0


def test_solid_diffusion_step_q_source():
    from tensorlbm.conjugate_ht import cht_solid_diffusion_step

    ny, nx = 8, 8
    T_s = torch.zeros(ny, nx)
    mask_solid = torch.ones(ny, nx, dtype=torch.bool)
    T_new = cht_solid_diffusion_step(T_s, mask_solid, alpha_s=0.1, Q_source=0.5)
    # All solid cells should have increased temperature due to heat source
    assert (T_new[mask_solid] >= 0.5).all()


# ---------------------------------------------------------------------------
# apply_cht_interface
# ---------------------------------------------------------------------------

def test_apply_cht_interface_shape():
    from tensorlbm.conjugate_ht import apply_cht_interface

    ny, nx = 10, 10
    T_fluid = torch.zeros(ny, nx)
    T_solid = torch.ones(ny, nx)
    mask_solid = torch.zeros(ny, nx, dtype=torch.bool)
    mask_solid[4:6, 4:6] = True

    T_f_new, T_s_new = apply_cht_interface(T_fluid, T_solid, mask_solid, k_ratio=5.0)
    assert T_f_new.shape == (ny, nx)
    assert T_s_new.shape == (ny, nx)


def test_apply_cht_interface_temperature_continuity():
    """Interface fluid cells should shift towards solid temperature."""
    from tensorlbm.conjugate_ht import apply_cht_interface

    ny, nx = 12, 12
    T_fluid = torch.zeros(ny, nx)    # cold fluid
    T_solid = torch.ones(ny, nx)     # hot solid
    mask_solid = torch.zeros(ny, nx, dtype=torch.bool)
    mask_solid[5:7, 5:7] = True

    T_f_new, _ = apply_cht_interface(T_fluid, T_solid, mask_solid, k_ratio=1.0)

    # Fluid cells directly adjacent to solid block should have T > 0
    # (shifted toward solid temperature at the interface)
    interface_cells = ~mask_solid
    # Check at least one fluid cell has non-zero T
    assert T_f_new[interface_cells].max().item() > 0.0


def test_apply_cht_interface_k_ratio_effect():
    """Higher k_ratio should produce higher interface temperature on fluid side."""
    from tensorlbm.conjugate_ht import apply_cht_interface

    ny, nx = 12, 12
    T_fluid = torch.zeros(ny, nx)
    T_solid = torch.ones(ny, nx)
    mask_solid = torch.zeros(ny, nx, dtype=torch.bool)
    mask_solid[5:7, 5:7] = True

    T_f_low, _ = apply_cht_interface(T_fluid.clone(), T_solid.clone(), mask_solid, k_ratio=1.0)
    T_f_high, _ = apply_cht_interface(T_fluid.clone(), T_solid.clone(), mask_solid, k_ratio=10.0)

    # Higher conductivity ratio → solid dominates → higher interface T in fluid
    assert T_f_high.max().item() >= T_f_low.max().item()


# ---------------------------------------------------------------------------
# run_conjugate_ht_2d – smoke test
# ---------------------------------------------------------------------------

def test_run_conjugate_ht_2d_runs():
    """A very short CHT run should complete without error."""
    from tensorlbm.conjugate_ht import CHTConfig, run_conjugate_ht_2d

    state = _make_state(ny=12, nx=16)
    cfg = CHTConfig(n_steps=5, output_interval=5)
    final = run_conjugate_ht_2d(state, cfg)
    assert final.step == 5


def test_run_conjugate_ht_2d_callback_called():
    from tensorlbm.conjugate_ht import CHTConfig, run_conjugate_ht_2d

    state = _make_state(ny=12, nx=16)
    cfg = CHTConfig(n_steps=10, output_interval=5)
    calls = []
    run_conjugate_ht_2d(state, cfg, callback=lambda s: calls.append(s.step))
    assert len(calls) == 2  # steps 5 and 10


def test_run_conjugate_ht_2d_temperature_evolves():
    """Solid temperature should change after a short run with heat source."""
    from tensorlbm.conjugate_ht import CHTConfig, run_conjugate_ht_2d

    state = _make_state(ny=12, nx=16)
    T_s_init = state.T_s.clone()
    cfg = CHTConfig(n_steps=20, output_interval=20, Q_source=0.01)
    final = run_conjugate_ht_2d(state, cfg)
    # Temperature in solid should have increased due to heat source
    assert not torch.allclose(final.T_s[final.mask_solid], T_s_init[final.mask_solid])
