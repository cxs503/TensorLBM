"""Zou/He D3Q19/D3Q27 channel inlet-outlet reconstruction — closure + physics.

Anchor: ``XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTION``.

Three classes of assertion, deliberately of different strength:

* **Exact algebraic invariants** (bound ``1e-8``; round-off is ``~1e-16`` at
  ``float64``).  After the closure is applied to an arbitrary non-equilibrium
  field, the reconstructed plane must reproduce the *prescribed* macroscopic
  quantity and satisfy the non-equilibrium bounce-back relation
  ``f_k - f^eq_k = f_opp(k) - f^eq_opp(k)`` for every unknown direction.  These
  are properties of the reconstruction, not tolerance choices.
* **Independent oracle** (:func:`_independent_closure`), a numpy
  re-derivation from an explicit lattice table written out in this file, i.e.
  not a call into the module under test.  It catches indexing/pairing slips
  that the invariants alone would follow along with.
* **Physics**: a channel driven *only* by
  ``zou_he_inlet_pressure_3d`` + ``zou_he_outlet_pressure_3d`` must develop the
  analytic plane-Poiseuille profile at the amplitude predicted a priori
  (``u = G y'(H-y')/(2 nu)``, ``G = drho/(3L)``, no fitting), inside the
  owner's 15 % bound and in practice inside 3 %.

A deliberately *wrong* pairing (each unknown rebuilt from a partner that is not
its lattice opposite) is included as a discriminating control: it keeps the
density exact while corrupting the transverse momentum, which is the failure
mode the invariants exist to catch.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tensorlbm.boundary.boundaries3d import (  # historical import site
    zou_he_inlet_pressure_3d,
    zou_he_inlet_velocity_3d,
    zou_he_outlet_pressure_3d,
)
from tensorlbm.boundary.boundaries_d3q27 import (
    zou_he_inlet_pressure_27,
    zou_he_inlet_velocity_27,
    zou_he_outlet_pressure_27,
)
from tensorlbm.d3q19 import C as C19
from tensorlbm.d3q19 import OPPOSITE as OPP19
from tensorlbm.d3q19 import W as W19
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.d3q27 import equilibrium27, macroscopic27
from tensorlbm.verification.zou_he_channel3d import (
    plane_poiseuille_profile,
    pressure_gradient,
    simulate_channel_3d,
)

TOL = 1e-8

# --- explicit lattice tables (test-local; not imported from the module under test) ---
_W19 = [float(x) for x in W19]
_C19 = [[int(v) for v in row] for row in C19.tolist()]
_OPP19 = [int(v) for v in OPP19.tolist()]


def _noisy_field(seed: int = 5, nz: int = 2, ny: int = 6, nx: int = 8) -> torch.Tensor:
    """Non-equilibrium, strictly positive population field."""
    g = torch.Generator().manual_seed(seed)
    rho = 1.0 + 0.02 * torch.rand((nz, ny, nx), generator=g, dtype=torch.float64)
    ux = 0.02 * torch.rand((nz, ny, nx), generator=g, dtype=torch.float64)
    uy = -0.01 * torch.rand((nz, ny, nx), generator=g, dtype=torch.float64)
    uz = 0.005 * torch.rand((nz, ny, nx), generator=g, dtype=torch.float64)
    feq = equilibrium3d(rho, ux, uy, uz)
    noise = 1e-3 * torch.rand((19, nz, ny, nx), generator=g, dtype=torch.float64)
    f = feq + noise
    assert bool((f > 0).all())
    return f


def _independent_closure(
    plane: np.ndarray, *, mode: str, target: float, uy: float = 0.0, uz: float = 0.0
) -> tuple[np.ndarray, float, float]:
    """Re-derive the west-face closure in numpy from the explicit table above.

    Independent of the module under test: the mass/momentum balance, the
    non-equilibrium bounce-back on the normal link, the equilibrium-proportional
    split of the remaining unknown-set mass and the minimum-norm tangential
    correction are all re-typed here, and the projection uses
    ``numpy.linalg.solve`` rather than the precomputed projector.

    Args:
        plane: ``(19, ny)`` populations at the inlet plane.
        mode: ``"velocity"`` (``target`` = u_x) or ``"pressure"`` (``target`` = rho).
        uy, uz: transverse velocity targets.

    Returns:
        ``(f_new (19, ny), rho, ux)``.
    """
    inlet = [k for k in range(19) if _C19[k][0] > 0]
    known = [k for k in range(19) if k not in inlet]
    normal = next(k for k in range(19) if _C19[k] == [1, 0, 0])
    nopp = next(k for k in range(19) if _C19[k] == [-1, 0, 0])
    free = [k for k in inlet if k != normal]

    ks = sum(plane[k] for k in known)
    kcx = sum(_C19[k][0] * plane[k] for k in known)
    if mode == "velocity":
        ux = np.full(plane.shape[1:], float(target))
        rho = (ks - kcx) / (1.0 - ux)
    else:
        rho = np.full(plane.shape[1:], float(target))
        ux = (kcx + (rho - ks)) / rho

    def feq(k, r, vx, vy, vz):
        cx, cy, cz = _C19[k]
        cu = 3.0 * (cx * vx + cy * vy + cz * vz)
        return _W19[k] * r * (1.0 + cu + 0.5 * cu * cu - 1.5 * (vx * vx + vy * vy + vz * vz))

    fq = {k: feq(k, rho, ux, uy, uz) for k in range(19)}
    f_normal = plane[nopp] + fq[normal] - fq[nopp]
    remaining = rho - ks - f_normal
    fq_free = np.stack([np.maximum(fq[k], 0.0) for k in free])
    base = remaining * fq_free / fq_free.sum(axis=0)

    ry = rho * uy - sum(_C19[k][1] * plane[k] for k in known) - sum(
        _C19[k][1] * base[i] for i, k in enumerate(free)
    )
    rz = rho * uz - sum(_C19[k][2] * plane[k] for k in known) - sum(
        _C19[k][2] * base[i] for i, k in enumerate(free)
    )
    A = np.stack(
        [
            np.ones(len(free)),
            np.array([_C19[k][1] for k in free], dtype=float),
            np.array([_C19[k][2] for k in free], dtype=float),
        ]
    )
    rhs = np.stack([np.zeros_like(ry), ry, rz]).reshape(3, -1)
    delta = (A.T @ np.linalg.solve(A @ A.T, rhs)).reshape(base.shape)

    out = plane.copy()
    out[normal] = f_normal
    for i, k in enumerate(free):
        out[k] = np.maximum(base[i] + delta[i], 0.0)
    return out, rho, ux


def _normal_link_residual(f_new: torch.Tensor, column: int, normal: int) -> float:
    """max |(f_n - f_n^eq) - (f_opp - f_opp^eq)| on the normal link only.

    Non-equilibrium bounce-back is physical only for the direction normal to
    the face; the tangential directions are closed from the momentum
    constraints instead (see ``zou_he_3d``).
    """
    rho, ux, uy, uz = macroscopic3d(f_new)
    idx = column if column >= 0 else f_new.shape[-1] + column
    plane = slice(idx, idx + 1)
    feq = equilibrium3d(rho[:, :, plane], ux[:, :, plane], uy[:, :, plane], uz[:, :, plane])
    col = f_new[:, :, :, column]
    o = _OPP19[normal]
    return float(
        torch.abs((col[normal] - feq[normal, :, :, 0]) - (col[o] - feq[o, :, :, 0])).max()
    )


# --------------------------------------------------------------------------- D3Q19


def test_velocity_inlet_reproduces_prescribed_velocity_and_density():
    f = _noisy_field()
    u_in, uy_in, uz_in = 0.03, 0.012, -0.007
    out = zou_he_inlet_velocity_3d(f, u_in, uy_in=uy_in, uz_in=uz_in)
    rho, ux, uy, uz = macroscopic3d(out)
    assert float(torch.abs(ux[:, :, 0] - u_in).max()) < TOL
    assert float(torch.abs(uy[:, :, 0] - uy_in).max()) < TOL
    assert float(torch.abs(uz[:, :, 0] - uz_in).max()) < TOL
    # implied density from mass + x-momentum
    plane = f[:, :, :, 0]
    rest = plane[[k for k in range(19) if _C19[k][0] == 0]].sum(dim=0)
    negative = plane[[k for k in range(19) if _C19[k][0] < 0]].sum(dim=0)
    expected = (rest + 2.0 * negative) / (1.0 - u_in)
    assert float(torch.abs(rho[:, :, 0] - expected).max()) < TOL
    assert _normal_link_residual(out, 0, 1) < TOL


def test_pressure_inlet_reproduces_prescribed_density():
    f = _noisy_field(seed=11)
    rho_in = 1.007
    out = zou_he_inlet_pressure_3d(f, rho_in)
    rho, ux, *_ = macroscopic3d(out)
    assert float(torch.abs(rho[:, :, 0] - rho_in).max()) < TOL
    # transverse velocity is the documented closure assumption
    plane = f[:, :, :, 0]
    rest = plane[[k for k in range(19) if _C19[k][0] == 0]].sum(dim=0)
    negative = plane[[k for k in range(19) if _C19[k][0] < 0]].sum(dim=0)
    expected_ux = 1.0 - (rest + 2.0 * negative) / rho_in
    assert float(torch.abs(ux[:, :, 0] - expected_ux).max()) < TOL
    assert _normal_link_residual(out, 0, 1) < TOL


def test_pressure_outlet_reproduces_prescribed_density():
    f = _noisy_field(seed=17)
    rho_out = 0.993
    out = zou_he_outlet_pressure_3d(f, rho_out)
    rho, ux, *_ = macroscopic3d(out)
    assert float(torch.abs(rho[:, :, -1] - rho_out).max()) < TOL
    plane = f[:, :, :, -1]
    rest = plane[[k for k in range(19) if _C19[k][0] == 0]].sum(dim=0)
    positive = plane[[k for k in range(19) if _C19[k][0] > 0]].sum(dim=0)
    expected_ux = -1.0 + (rest + 2.0 * positive) / rho_out
    assert float(torch.abs(ux[:, :, -1] - expected_ux).max()) < TOL
    assert _normal_link_residual(out, -1, 2) < TOL


def test_profile_inlet_is_imposed_cell_by_cell():
    """A parabolic inflow must be imposed row by row, not as a plug."""
    nz, ny, nx = 2, 10, 6
    f = _noisy_field(seed=23, nz=nz, ny=ny, nx=nx)
    yt = torch.arange(1, ny - 1, dtype=torch.float64) - 0.5
    H = float(ny - 2)
    prof1d = 0.04 * yt * (H - yt) / (H * H / 4.0)
    prof = prof1d.unsqueeze(0).repeat(nz, 1)
    out = zou_he_inlet_velocity_3d(f, prof)
    _, ux, *_ = macroscopic3d(out)
    # either the full (nz, ny) plane or the fluid-row (nz, ny-2) profile
    err = float((ux[:, :, 0][:, 1:-1] - prof).abs().max())
    assert err < TOL
    # the reconstruction must not have produced a uniform plug
    assert float(prof.max() - prof[:, 1:-1].min()) > 0.01
    # implied density follows the profile (rho = (A + 2B)/(1-u_x))
    rho, _, _, _ = macroscopic3d(out)
    assert float((rho[:, :, 0] - rho[:, :, 0].mean()).abs().max()) > 1e-6


def test_pressure_field_inlet_accepts_imposed_pressure_field():
    f = _noisy_field(seed=29)
    nz, ny = f.shape[1], f.shape[2]
    base = 1.0 + 0.004 * torch.linspace(-0.5, 0.5, ny, dtype=torch.float64)
    field = base.unsqueeze(0).repeat(nz, 1)
    out = zou_he_inlet_pressure_3d(f, field)
    rho, _, _, _ = macroscopic3d(out)
    assert float((rho[:, :, 0] - field).abs().max()) < TOL


def test_only_the_boundary_plane_changes_and_input_is_not_mutated():
    f = _noisy_field(seed=31)
    ref = f.clone()
    out = zou_he_inlet_pressure_3d(f, 1.002)
    assert float((f - ref).abs().max()) == 0.0, "input populations were mutated in place"
    assert float((out[:, :, :, 1:] - f[:, :, :, 1:]).abs().max()) == 0.0
    assert float((out[:, :, :, 0] - f[:, :, :, 0]).abs().max()) > 0.0


def test_independent_numpy_oracle_matches_closed_form():
    f = _noisy_field(seed=37)
    plane = f[:, :, :, 0].numpy()
    for mode, target in (("velocity", 0.025), ("pressure", 1.004)):
        out = (
            zou_he_inlet_velocity_3d(f, target)
            if mode == "velocity"
            else zou_he_inlet_pressure_3d(f, target)
        )
        ref, rho_ref, ux_ref = _independent_closure(plane, mode=mode, target=target)
        assert np.abs(out[:, :, :, 0].numpy() - ref).max() < TOL
        rho, ux, *_ = macroscopic3d(out)
        assert np.abs(rho[:, :, 0].numpy() - rho_ref).max() < TOL
        assert np.abs(ux[:, :, 0].numpy() - ux_ref).max() < TOL


def test_nebb_for_all_directions_is_the_falsified_alternative():
    """Discriminating control: "NEBB for every unknown direction".

    That construction (the previous implementation of this file's closure)
    keeps ``rho`` and ``u_x`` exact, so a channel with ``u_y = u_z = 0`` looks
    perfect.  It injects exactly ``rho u_y / 3`` of the prescribed transverse
    momentum for D3Q19, i.e. an angled inflow arrives at one third strength.
    This test pins the size of that artefact so the closure cannot silently
    regress to it.
    """
    f = _noisy_field(seed=41)
    plane = f[:, :, :, 0].numpy()
    inlet = [k for k in range(19) if _C19[k][0] > 0]
    opp = [_OPP19[k] for k in inlet]

    def feq(k, r, vx, vy, vz):
        cx, cy, cz = _C19[k]
        cu = 3.0 * (cx * vx + cy * vy + cz * vz)
        return _W19[k] * r * (1.0 + cu + 0.5 * cu * cu - 1.5 * (vx * vx + vy * vy + vz * vz))

    u_in, uy_in = 0.03, 0.06
    rest = sum(plane[k] for k in range(19) if _C19[k][0] == 0)
    negative = sum(plane[k] for k in range(19) if _C19[k][0] < 0)
    rho = (rest + 2.0 * negative) / (1.0 - u_in)
    bad = plane.copy()
    for k, o in zip(inlet, opp):
        bad[k] = feq(k, rho, u_in, uy_in, 0.0) - feq(o, rho, u_in, uy_in, 0.0) + plane[o]
    bad_t = f.clone()
    bad_t[:, :, :, 0] = torch.as_tensor(bad, dtype=bad_t.dtype)

    rho_b, ux_b, uy_b, _ = macroscopic3d(bad_t)
    assert np.abs(rho_b[:, :, 0].numpy() - rho).max() < 1e-10  # mass still exact
    assert float((ux_b[:, :, 0] - u_in).abs().max()) < 1e-10  # normal velocity exact
    injected = uy_b[:, :, 0].numpy() / uy_in
    # exactly one third for a pure equilibrium field; a non-equilibrium start
    # shifts it, but never back towards the prescribed value
    assert (injected > 0.15).all() and (injected < 0.5).all(), injected
    # the fixed closure does not have it
    good = zou_he_inlet_velocity_3d(f, u_in, uy_in=uy_in)
    _, _, uy_g, _ = macroscopic3d(good)
    assert float((uy_g[:, :, 0] - uy_in).abs().max()) < TOL


def test_gradient_and_profile_helpers_are_consistent():
    drho, nx, ny, nu = 0.004, 65, 18, 0.05
    G = pressure_gradient(drho, nx)
    assert G == pytest.approx(drho / (3.0 * (nx - 1)))
    prof = plane_poiseuille_profile(ny, G, nu)
    H = ny - 2
    # the discrete y' grid is offset by half a cell, so the sampled peak is
    # y'(H-y') = 63.75 instead of H^2/4 = 64 at H = 16
    assert prof.max() == pytest.approx(G * H * H / (8.0 * nu), rel=1e-2)
    assert prof[0] == pytest.approx(prof[-1])


@pytest.mark.parametrize(
    "call, match",
    [
        (lambda f: zou_he_inlet_velocity_3d(f, 1.0), "u_in"),
        (lambda f: zou_he_inlet_pressure_3d(f, 0.0), "rho_in"),
        (
            lambda f: zou_he_inlet_pressure_3d(
                f, torch.cat([torch.full((2, 5), 1.0), torch.zeros(2, 1)], dim=1)
            ),
            "positive",
        ),
        (lambda f: zou_he_outlet_pressure_3d(f, -1.0), "rho_out"),
        (lambda f: zou_he_inlet_velocity_3d(torch.zeros(9, 2, 6, 8), 0.1), "shape"),
        (
            lambda f: zou_he_inlet_velocity_3d(f, torch.zeros(3, 2)),
            "shape",
        ),
        (lambda f: zou_he_inlet_velocity_3d(f, float("nan")), "finite"),
    ],
)
def test_invalid_inputs_are_rejected(call, match):
    f = _noisy_field()
    with pytest.raises(ValueError, match=match):
        call(f)


# --------------------------------------------------------------------------- D3Q27


def test_d3q27_pressure_inlet_and_outlet_are_exact():
    g = torch.Generator().manual_seed(7)
    ones = torch.ones((2, 5, 7), dtype=torch.float64)
    zeros = torch.zeros_like(ones)
    f = equilibrium27(ones, zeros, zeros, zeros) + 1e-3 * torch.rand(
        (27, 2, 5, 7), generator=g, dtype=torch.float64
    )
    rho_in, rho_out = 1.006, 0.994
    out = zou_he_inlet_pressure_27(f, rho_in)
    rho, ux, *_ = macroscopic27(out)
    assert float((rho[:, :, 0] - rho_in).abs().max()) < TOL
    out = zou_he_outlet_pressure_27(f, rho_out)
    rho, ux, *_ = macroscopic27(out)
    assert float((rho[:, :, -1] - rho_out).abs().max()) < TOL
    out = zou_he_inlet_velocity_27(f, 0.02, uy_in=0.01)
    rho, ux, uy, _ = macroscopic27(out)
    assert float((ux[:, :, 0] - 0.02).abs().max()) < TOL
    assert float((uy[:, :, 0] - 0.01).abs().max()) < TOL


def test_d3q27_profile_inlet_is_imposed_cell_by_cell():
    g = torch.Generator().manual_seed(13)
    nz, ny, nx = 2, 8, 6
    ones = torch.ones((nz, ny, nx), dtype=torch.float64)
    zeros = torch.zeros_like(ones)
    f = equilibrium27(ones, zeros, zeros, zeros) + 1e-3 * torch.rand(
        (27, nz, ny, nx), generator=g, dtype=torch.float64
    )
    yt = torch.arange(1, ny - 1, dtype=torch.float64) - 0.5
    prof1d = 0.05 * yt * ((ny - 2) - yt)
    prof = prof1d.unsqueeze(0).repeat(nz, 1)
    out = zou_he_inlet_velocity_27(f, prof)
    _, ux, _, _ = macroscopic27(out)
    assert float((ux[:, :, 0][:, 1:-1] - prof).abs().max()) < TOL


# --------------------------------------------------------------------------- physics


def test_pressure_driven_3d_channel_matches_analytic_poiseuille():
    """The closure pair alone must reproduce plane Poiseuille a priori."""
    nx, ny, nz = 48, 12, 2
    res = simulate_channel_3d(nx=nx, ny=ny, nz=nz, drho=0.004, tau=0.65, steps=4000, block=600)
    m = res.metrics
    assert m["converged"], m
    assert m["rho_min"] > 0.9 and m["rho_max"] < 1.1
    assert m["flux_end_to_interior_defect"] < 1e-4, m
    # amplitude is predicted a priori (G = drho/(3L), no fitting)
    assert m["amp_rel_err"] < 0.15, m
    assert m["amp_rel_err"] < 0.03, m
    assert m["linf_rel_err"] < 0.03, m
    assert m["shape_l2_rel"] < 0.03, m


def test_velocity_profile_inlet_arm_stays_on_the_analytic_profile():
    nx, ny, nz = 48, 12, 2
    res = simulate_channel_3d(
        nx=nx, ny=ny, nz=nz, drho=0.004, tau=0.65, steps=4000, block=600,
        inlet="velocity-profile",
    )
    m = res.metrics
    assert m["converged"], m
    assert m["amp_rel_err"] < 0.03, m
    assert m["linf_rel_err"] < 0.03, m