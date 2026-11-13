"""Smoke test: the LIFE epsilon-Gram correction inside the TensorLBM IBM solver.

Anchor: XVAL-LBM-IBM-EPSILON-GRAM-IMPLICIT-CORRECTION

This is the *integration* check: the correction is not exercised in isolation
but through the repo's main solver pieces

    tensorlbm.d2q9.equilibrium / macroscopic
    tensorlbm.solver.collide_bgk / stream
    tensorlbm.ibm_force_transfer.IBMStencil / body_source_2d
    tensorlbm.ibm_validation_cases.disc_mask_2d

i.e. the same path used by the physics arm of ``run_xval.py``. A short run is
enough to assert that (a) the corrected spreading is stable, (b) the deposited
force is finite and of the expected sign, and (c) the corrected and control
arms stay within the same order of magnitude.
"""
import torch

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.ibm_epsilon_gram import (
    life_correction,
    make_life_ibm_stencil,
    spread_with_correction,
)
from tensorlbm.ibm_force_transfer import (
    body_source_2d,
    build_cylinder_markers_2d,
    interior_mask_force,
)
from tensorlbm.ibm_validation_cases import disc_mask_2d
from tensorlbm.solver import collide_bgk, stream

DT = torch.float64


def _run(arm: str, steps: int = 12, nx: int = 64, ny: int = 32, d: float = 4.0):
    u = 0.1
    tau = 0.524
    c = (0.35 * nx, 0.5 * ny)
    pts = build_cylinder_markers_2d(c, 0.5 * d, spacing=0.8)
    shape = (ny, nx)
    stencil = make_life_ibm_stencil(pts, shape)
    if arm == "A1":
        eps, ds = life_correction(pts, shape)
    else:
        eps = torch.ones(pts.shape[0], dtype=DT)
        ds = torch.ones(pts.shape[0], dtype=DT)
    mask = disc_mask_2d(shape, c, 0.5 * d - 0.5)
    rho = torch.ones(shape, dtype=DT)
    f = equilibrium(rho, u * torch.ones_like(rho), torch.zeros_like(rho))
    drag = []
    for _ in range(steps):
        _, uxf, uyf = macroscopic(f)
        ug = torch.stack([uxf, uyf])
        u_int = stencil.interpolate(ug).t()
        rho_m = torclamp(stencil.interpolate(rho))
        fm = u_int.clone()
        for _ in range(2):
            u_phys = u_int + 0.5 * fm / rho_m.unsqueeze(1)
            fm = fm + (torch.zeros_like(u_int) - u_phys)
        gx, gy = spread_with_correction(stencil, fm[:, 0], fm[:, 1], eps, ds)
        fint = interior_mask_force(ug, mask, torch.zeros_like(ug))
        f = collide_bgk(f, tau)
        f = body_source_2d(f, gx + fint[0], gy + fint[1])
        f = stream(f)
        drag.append(float(-(gx + fint[0]).sum()))
    return f, drag


def torclamp(t: torch.Tensor) -> torch.Tensor:
    return torch.maximum(t, torch.tensor(1e-12, dtype=t.dtype))


def test_corrected_spread_runs_on_main_solver_and_is_stable():
    f, drag = _run("A1")
    assert bool(torch.isfinite(f).all())
    assert all(abs(x) < 1e3 for x in drag)
    assert drag[-1] > 0.0  # inflow along +x, so the body drags the fluid backwards
    # density stays physical
    rho = f.sum(dim=0)
    assert float(rho.min()) > 0.5 and float(rho.max()) < 2.0


def test_control_and_corrected_agree_in_magnitude():
    _, drag_a1 = _run("A1")
    _, drag_a2 = _run("A2")
    ratio = abs(drag_a1[-1]) / abs(drag_a2[-1])
    assert 0.2 < ratio < 5.0, ratio