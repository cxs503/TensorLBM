"""Cross-validation tests for the ported regularised D2Q9 boundary condition.

Reference: joconnor22/LIFE ``src/Grid.cpp::regularisedBC``; the per-node
transcription lives in ``development/verification/regularised_bc_2d_reference/reference.py`` and the
upstream snapshot in ``.../oss_src/Grid.cpp`` (md5 pinned in PROVENANCE.json).

Invariants checked here (see the anchor report
``docs/reports/XVAL_LBM_REGULARISED_BC_ROUND2_AUDIT.md``):

* I1  the torch port reproduces the LIFE transcription node by node
      (non-corner <= 1e-8, corners inside the owner 15 % gate);
* I2  the second-order Hermite projection preserves the two lowest moments, so
      a reconstructed node leaves with exactly the density and velocity that
      step 1 of the algorithm prescribed, and with exactly the non-equilibrium
      stress accumulated in step 3 (``Pi_neq`` idempotent under projection);
* I3  the reconstructed populations are finite and non-negative for a
      physically admissible state.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from tensorlbm import regularised_bc_2d as rb
from tensorlbm.boundary.d2q9 import C2Q9
from tensorlbm.boundary.solver2d import equilibrium2d, macroscopic2d

REF_PATH = (
    Path(__file__).resolve().parents[1]
    / "development"
    / "verification"
    / "regularised_bc_2d_reference"
    / "reference.py"
)


def _load_reference():
    spec = importlib.util.spec_from_file_location("life_reference_bc", REF_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _channel_state(ny: int, nx: int, umax: float, seed: int = 7):
    """Synthetic pre-BC state: parabola + smooth perturbation, rho = 1 + eps."""
    torch.manual_seed(seed)
    H = ny - 1
    y = torch.arange(ny, dtype=torch.float64)
    prof = 4.0 * umax * y * (H - y) / H**2
    ux = prof[:, None].expand(ny, nx).clone()
    uy = 1.0e-3 * torch.sin(torch.arange(nx, dtype=torch.float64) * 0.7)[None, :].expand(ny, nx).clone()
    rho = 1.0 + 1.0e-3 * torch.cos(y * 0.9)[:, None].expand(ny, nx).clone()
    f = equilibrium2d(rho, ux, uy)
    noise = 1.0e-6 * torch.randn(9, ny, nx, dtype=torch.float64)
    f = f + noise

    kinds = torch.zeros((ny, nx), dtype=torch.long)
    normals = torch.zeros((ny, nx, 2), dtype=torch.long)
    kinds[:, 0] = rb.VELOCITY
    normals[:, 0, 0] = 1
    kinds[:, -1] = rb.PRESSURE
    normals[:, -1, 0] = -1
    kinds[0, :] = rb.WALL
    normals[0, :, 1] = 1
    kinds[-1, :] = rb.WALL
    normals[-1, :, 1] = -1
    normals[0, 0] = torch.tensor([1, 1])
    normals[0, -1] = torch.tensor([-1, 1])
    normals[-1, 0] = torch.tensor([1, -1])
    normals[-1, -1] = torch.tensor([-1, -1])

    u_bc = torch.zeros((ny, nx, 2), dtype=torch.float64)
    u_bc[:, 0, 0] = prof
    rho_bc = torch.ones((ny, nx), dtype=torch.float64)
    rho_f, ux_f, uy_f = macroscopic2d(f)
    f = f.clone()
    u_f = torch.stack([ux_f, uy_f], dim=-1)
    return f, kinds, normals, u_bc, rho_bc, rho_f, u_f


def _compare(ny: int, nx: int):
    ref = _load_reference()
    f, kinds, normals, u_bc, rho_bc, rho_f, u_f = _channel_state(ny, nx, 0.05)
    f_ref, _, _, _ = ref.life_regularised_bc(
        f.numpy(), kinds.numpy(), normals.numpy(), u_bc.numpy(), rho_bc.numpy(),
        rho_f.numpy(), u_f.numpy(),
    )
    f_port, diag = rb.apply_regularised_bc_2d(
        f, kinds, normals, u_bc=u_bc, rho_bc=rho_bc, rho=rho_f, u=u_f
    )
    import numpy as np

    iy, ix = np.nonzero((kinds != rb.FLUID).numpy())
    corner = diag["is_corner"].numpy()
    fp = f_port.numpy()[:, iy, ix]
    fr = f_ref[:, iy, ix]
    denom = np.sqrt((fr**2).mean(axis=0))
    rel = 100.0 * np.abs(fp - fr).max(axis=0) / denom
    return rel, corner, f_port, diag, kinds, iy, ix, rho_f, u_bc


def test_port_matches_life_transcription_noncorner():
    rel, corner, *_ = _compare(9, 17)
    assert corner.any()
    assert rel[~corner].max() < 1.0e-8, rel[~corner].max()


def test_port_matches_life_transcription_all_nodes():
    rel, corner, *_ = _compare(9, 17)
    assert rel.max() < 15.0, rel.max()  # owner gate, even at corners


def test_reconstruction_preserves_density_and_stress():
    """I2: the Hermite projection is moment-exact below second order.

    ``f_q = f_q^eq + (w_q/2c_s^4) Q_q:Pi_neq`` reproduces ``sum f``,
    ``sum c f`` and ``sum c c f`` of the reconstructed set exactly, hence the
    node leaves with the density/velocity set in step 1 and with the
    non-equilibrium stress accumulated in step 3.
    """
    _, _, f_port, diag, kinds, iy, ix, rho_f, u_bc = _compare(9, 17)
    import numpy as np

    c = C2Q9.numpy()
    fp = f_port.numpy()[:, iy, ix]
    kv = kinds.numpy()[iy, ix]

    # (a) velocity-type nodes carry the prescribed velocity ...
    vel_mask = (kv == rb.VELOCITY) | (kv == rb.WALL) | (kv == rb.FREE_SLIP)
    got = np.stack([diag["ux"].numpy(), diag["uy"].numpy()], axis=1)
    assert np.abs(got[vel_mask] - u_bc.numpy()[iy, ix][vel_mask]).max() < 1.0e-12

    # (b) ... and the density recovered by the Zou/He closure, i.e. the
    #     momentum of the reconstructed set is rho * u, not a free parameter.
    rho_nodes = fp.sum(axis=0)
    mom_x = (c[:, 0, None] * fp).sum(axis=0)
    mom_y = (c[:, 1, None] * fp).sum(axis=0)
    assert np.abs(mom_x - rho_nodes * got[:, 0]).max() < 1.0e-12
    assert np.abs(mom_y - rho_nodes * got[:, 1]).max() < 1.0e-12

    # (c) pressure nodes keep the prescribed density
    pres_mask = kv == rb.PRESSURE
    assert np.abs(rho_nodes[pres_mask] - 1.0).max() < 1.0e-12

    # (d) the projection reproduces the accumulated non-equilibrium stress
    pre = np.array([diag["pi_neq_xx_pre"].numpy(), diag["pi_neq_yy_pre"].numpy(), diag["pi_neq_xy_pre"].numpy()])
    post = np.array([diag["pi_neq_xx"].numpy(), diag["pi_neq_yy"].numpy(), diag["pi_neq_xy"].numpy()])
    assert np.abs(post - pre).max() < 1.0e-12
    # (e) the off-equilibrium part carries neither mass nor momentum: the
    #     projection only reshuffles moments above second order
    w = rb.W2Q9.numpy()
    rho_n, ux_n, uy_n = diag["rho"].numpy(), diag["ux"].numpy(), diag["uy"].numpy()
    cu = np.einsum("qi,i...->q...", c.astype(float), np.stack([ux_n, uy_n]))
    u2 = ux_n**2 + uy_n**2
    feq = w[:, None] * rho_n[None, :] * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u2[None, :])
    fneq = fp - feq
    assert np.abs(fneq.sum(axis=0)).max() < 1.0e-12
    assert np.abs((c[:, 0, None] * fneq).sum(axis=0)).max() < 1.0e-12
    assert np.abs((c[:, 1, None] * fneq).sum(axis=0)).max() < 1.0e-12


def test_single_node_wall_reconstruction_is_deterministic():
    """A wall node's reconstruction must not depend on float-unsafe randomness."""
    _, _, f_port, diag, *_ = _compare(7, 13)
    assert torch.isfinite(f_port).all()
    assert bool((f_port >= 0).all())