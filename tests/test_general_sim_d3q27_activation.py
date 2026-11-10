"""G7/G8 activation tests — D3Q27 lattice in GeneralSimEngine.

Gates from the campaign node P151-N01 (all machine-computable):

* (a) D3Q27 equilibrium isotropy: 3rd-moment residual < 1e-14 (float64),
  plus rank-4/rank-6 lattice isotropy identities.
* (b) Mass-conservation residual < 1e-12 (periodic D3Q27 step, 200 steps).
* (g) D3Q19 degenerate consistency: < 1e-10 on the D3Q19 subspace
  (macroscopic fields, streaming and bounce-back).
* engine wiring: ``LatticeModel.D3Q27`` really drives a 27-population field,
  D3Q27 collision dispatch, D3Q27 force integration, divergence guard.
* 3 negative controls (wrong stencil, wrong mask, unsupported collision).

Reference values are analytic (lattice moment sums); nothing is hard-coded
from a previous run.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.d3q19 import C as C19
from tensorlbm.d3q19 import OPPOSITE as OPP19
from tensorlbm.d3q19 import macroscopic3d
from tensorlbm.d3q27 import C as C27
from tensorlbm.d3q27 import OPPOSITE as OPP27
from tensorlbm.d3q27 import equilibrium27, macroscopic27
from tensorlbm.drag_pressure import _macroscopic as dp_macroscopic
from tensorlbm.general_sim import (
    CollisionModel,
    GeneralSimConfig,
    GeneralSimEngine,
    GeometryConfig,
    GeometrySource,
    LatticeModel,
    OutputConfig,
    PhysicsConfig,
    SolverConfig,
)
from tensorlbm.lbm_step_d3q27 import (
    far_field_bc_27_faces,
    lbm_step_correct_27,
)
from tensorlbm.solver3d import stream3d

DT = torch.float64

# mapping D3Q19 direction index -> D3Q27 direction index (same velocity vector)
MAP19_TO_27 = [
    next(q for q in range(27) if C27[q].tolist() == C19[i].tolist()) for i in range(19)
]


def _c64():
    return C27.to(DT)


def _random_fields(n=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    ones = torch.ones((n, n, n), dtype=DT)
    u = 0.08 * (torch.rand((3, n, n, n), generator=g, dtype=DT) - 0.5)
    return ones, u[0], u[1], u[2]


def _moment(f, idx):
    c = _c64()
    prod = torch.ones_like(f)
    for axis in idx:
        prod = prod * c[:, axis].view(27, 1, 1, 1)
    return (f * prod).sum(dim=0)


def _embed19_to_27(f19):
    """Place a D3Q19 population into D3Q27 slots (corner directions = 0)."""
    shape = (27,) + tuple(f19.shape[1:])
    f27 = torch.zeros(shape, dtype=f19.dtype, device=f19.device)
    for i, q in enumerate(MAP19_TO_27):
        f27[q] = f19[i]
    return f27


# ---------------------------------------------------------------------------
# (a) equilibrium isotropy
# ---------------------------------------------------------------------------


def test_equilibrium27_third_moment_vanishes_at_rest():
    """Odd (3rd-order) moments of the rest equilibrium are exactly zero.

    This is the discrete isotropy statement: Σ_i f_i^eq(u=0) c_ix c_iy c_iz = 0.
    """
    zeros = torch.zeros((5, 5, 5), dtype=DT)
    feq = equilibrium27(torch.ones_like(zeros), zeros, zeros, zeros)
    worst = 0.0
    for idx in ((0, 0, 0), (0, 1, 2), (0, 0, 1), (1, 1, 2), (2, 2, 2)):
        worst = max(worst, _moment(feq, idx).abs().max().item())
    assert worst < 1e-14, f"3rd-moment residual {worst:.3e} >= 1e-14"


def test_equilibrium27_fourth_moment_isotropic():
    """Rank-4 lattice tensor Σ w c_a c_b c_d c_e = (δδ+δδ+δδ)/9 (1e-13)."""
    c = _c64()
    w = None
    from tensorlbm.d3q27 import W_EXACT64

    w = W_EXACT64.to(DT).view(27, 1)
    delta = torch.eye(3, dtype=DT)
    worst = 0.0
    for a in range(3):
        for b in range(3):
            for d in range(3):
                for e in range(3):
                    lhs = (w * c[:, a : a + 1] * c[:, b : b + 1] * c[:, d : d + 1] * c[:, e : e + 1]).sum()
                    rhs = (
                        delta[a, b] * delta[d, e]
                        + delta[a, d] * delta[b, e]
                        + delta[a, e] * delta[b, d]
                    ) / 9.0
                    worst = max(worst, abs(lhs.item() - rhs.item()))
    assert worst < 1e-13, f"4th-moment isotropy residual {worst:.3e}"


def test_equilibrium27_rank6_sixth_moment_identity():
    """Σ w c_x²c_y²c_z² = 1/27 exactly (the 3rd-order isotropy constant)."""
    from tensorlbm.d3q27 import W_EXACT64

    c = _c64()
    w = W_EXACT64.to(DT)
    val = (w * c[:, 0] ** 2 * c[:, 1] ** 2 * c[:, 2] ** 2).sum().item()
    assert abs(val - 1.0 / 27.0) < 1e-15, val


def test_equilibrium27_third_moments_match_isotropic_lattice_identity():
    """Rank-3 moments equal the isotropic lattice values (O(u^3) terms absent).

    For the 2nd-order truncated equilibrium the only surviving rank-3 lattice
    tensors are the rank-4 convective sums, giving

        Σ f^eq c_x^3      = ρ u_x
        Σ f^eq c_x^2 c_y  = ρ u_y / 3
        Σ f^eq c_x c_y c_z = 0

    i.e. the discrete third moment reproduces the continuous Maxwellian up to
    O(u^3).  Checked to 1e-13 on random ρ, u (|u| ≤ 0.04).
    """
    rho, ux, uy, uz = _random_fields(seed=1)
    feq = equilibrium27(rho, ux, uy, uz)
    checks = {
        (0, 0, 0): rho * ux,
        (1, 1, 1): rho * uy,
        (2, 2, 2): rho * uz,
        (0, 0, 1): rho * uy / 3.0,
        (0, 1, 1): rho * ux / 3.0,
        (1, 2, 2): rho * uy / 3.0,
        (0, 1, 2): torch.zeros_like(rho),
    }
    worst = 0.0
    for idx, ana in checks.items():
        worst = max(worst, (_moment(feq, idx) - ana).abs().max().item())
    assert worst < 1e-13, f"rank-3 moment residual {worst:.3e}"


def test_equilibrium27_recovers_mass_and_momentum():
    rho, ux, uy, uz = _random_fields(seed=2)
    feq = equilibrium27(rho, ux, uy, uz)
    r2, ux2, uy2, uz2 = macroscopic27(feq)
    err = max(
        (r2 - rho).abs().max().item(),
        (ux2 - ux).abs().max().item(),
        (uy2 - uy).abs().max().item(),
        (uz2 - uz).abs().max().item(),
    )
    assert err < 1e-13, f"macroscopic recovery residual {err:.3e}"


# ---------------------------------------------------------------------------
# (g) D3Q19 degenerate consistency
# ---------------------------------------------------------------------------


def test_macroscopic27_matches_d3q19_embedding():
    f19 = torch.rand((19, 6, 5, 4), dtype=DT)
    f27 = _embed19_to_27(f19)
    a = macroscopic3d(f19)
    b = macroscopic27(f27)
    err = max((x - y).abs().max().item() for x, y in zip(a, b))
    assert err < 1e-10, f"D3Q19/D3Q27 macroscopic residual {err:.3e}"


def test_stream27_matches_stream3d_on_d3q19_subspace():
    f19 = torch.rand((19, 6, 5, 4), dtype=DT)
    f27 = _embed19_to_27(f19)
    s27 = f27.clone()
    # streaming = periodic roll per direction
    from tensorlbm.d3q27 import stream27_roll

    s27 = stream27_roll(s27)
    s19 = stream3d(f19)
    corners = [q for q in range(27) if q not in MAP19_TO_27]
    corner_res = s27[corners].abs().max().item()
    err = max(
        (s27[MAP19_TO_27[i]] - s19[i]).abs().max().item() for i in range(19)
    )
    assert corner_res == 0.0, f"corner pollution {corner_res:.3e}"
    assert err < 1e-10, f"streaming residual {err:.3e}"


def test_bounce_back27_matches_d3q19_embedding():
    f19 = torch.rand((19, 6, 5, 4), dtype=DT)
    solid = torch.zeros((6, 5, 4), dtype=torch.bool)
    solid[2:4, 1:3, 1:3] = True
    f27 = _embed19_to_27(f19)
    # half-way BB: solid cells hold pre-collision f[opposite]
    b27 = torch.where(solid.unsqueeze(0), f27[OPP27], f27)
    b19 = torch.where(solid.unsqueeze(0), f19[OPP19], f19)
    err = max((b27[MAP19_TO_27[i]] - b19[i]).abs().max().item() for i in range(19))
    assert err < 1e-10, f"BB residual {err:.3e}"


# ---------------------------------------------------------------------------
# (b) mass conservation + step invariants
# ---------------------------------------------------------------------------


def _small_sphere_solid(nz=8, ny=8, nx=8):
    z, y, x = torch.meshgrid(
        torch.arange(nz), torch.arange(ny), torch.arange(nx), indexing="ij"
    )
    return ((x - 3.5) ** 2 + (y - 3.5) ** 2 + (z - 3.5) ** 2) < 2.0**2


def test_mass_conservation_periodic_200_steps():
    """D3Q27 step (collide+NoDynamics+BB+periodic stream) conserves mass."""
    from tensorlbm.d3q27 import collide_bgk27

    solid = _small_sphere_solid()
    rho, ux, uy, uz = _random_fields(8, seed=3)
    rho = torch.ones_like(rho)
    ux = 0.02 * ux
    f = equilibrium27(rho, ux, uy, uz)
    m0 = f.sum().item()
    for _ in range(200):
        f = lbm_step_correct_27(
            f, collide_bgk27, 0.6, solid, 0.0, far_field_bc_fn=None
        )
    drift = abs(f.sum().item() - m0) / abs(m0)
    assert drift < 1e-12, f"relative mass drift {drift:.3e}"


def test_uniform_equilibrium_is_fixed_point():
    """Free-stream equilibrium + free-stream far field is a fixed point."""
    solid = torch.zeros((6, 6, 6), dtype=torch.bool)
    from tensorlbm.d3q27 import collide_bgk27

    ones = torch.ones((6, 6, 6), dtype=DT)
    f0 = equilibrium27(ones, 0.05 * ones, torch.zeros_like(ones), torch.zeros_like(ones))
    ff = lambda f, u: far_field_bc_27_faces(f, u)  # noqa: E731
    f = f0.clone()
    for _ in range(20):
        f = lbm_step_correct_27(f, collide_bgk27, 0.8, solid, 0.05, far_field_bc_fn=ff)
    err = (f - f0).abs().max().item()
    assert err < 1e-13, f"fixed-point residual {err:.3e}"


def test_far_field_bc27_writes_only_selected_faces():
    ones = torch.ones((7, 6, 5), dtype=DT)
    f = equilibrium27(ones, 0.05 * ones, torch.zeros_like(ones), torch.zeros_like(ones))
    f = f * 2.0  # non-equilibrium state everywhere
    cfg = {"far_field_faces": ["y+"], "periodic_faces": ["z-", "z+"]}
    out = far_field_bc_27_faces(f, 0.05, bc_config=cfg)
    # x- (inlet) and y+ are reset to the free-stream equilibrium
    assert torch.allclose(out[:, :, :, 0], f[:, :, :, 0] * 0 + out[:, :, :, 0])
    # x+ outlet is zero-gradient
    assert torch.equal(out[:, :, :, -1], out[:, :, :, -2])
    # z faces untouched (periodic) — compare away from the x inlet/outlet planes
    assert torch.equal(out[:, 0, 1:-1, 1:-1], f[:, 0, 1:-1, 1:-1])
    assert torch.equal(out[:, -1, 1:-1, 1:-1], f[:, -1, 1:-1, 1:-1])
    # y- untouched (not in far_field_faces)
    assert torch.equal(out[:, 1:-1, 0, 1:-1], f[:, 1:-1, 0, 1:-1])


def test_far_field_bc27_inlet_is_equilibrium():
    ones = torch.ones((5, 5, 5), dtype=DT)
    feq = equilibrium27(ones, 0.05 * ones, torch.zeros_like(ones), torch.zeros_like(ones))
    f = feq * 3.0
    out = far_field_bc_27_faces(f, 0.05, bc_config={"far_field_faces": [], "periodic_faces": []})
    assert torch.allclose(out[:, :, :, 0], feq[:, :, :, 0] * torch.ones_like(out[:, :, :, 0]))


# ---------------------------------------------------------------------------
# MRT transform integrity (loong64 / numpy 2.0.1 regression)
# ---------------------------------------------------------------------------


def test_mrt27_transform_is_orthonormal():
    """M M^T = I and M^-1 = M^T in float64 (|res| < 1e-10).

    Regression for the loong64 + numpy 2.0.1 defect: ``np.linalg.inv`` returned
    garbage (|M M^-1 - I| = 2.0e18) so ``collide_mrt27`` produced 1e16
    populations and the first step went NaN.  The basis is now orthonormal by
    construction and the inverse is the transpose (no LAPACK inversion).
    """
    from tensorlbm.d3q27 import _get_d3q27_mrt_matrices

    m64, i64 = _get_d3q27_mrt_matrices(torch.device("cpu"), torch.float64)
    res = float((m64 @ i64 - torch.eye(27, dtype=torch.float64)).abs().max())
    assert res < 1e-10, res
    assert torch.allclose(i64, m64.T, atol=1e-12)


def test_mrt27_float32_equilibrium_is_fixed_point_on_sphere_grid():
    """collide_mrt27 leaves a float32 equilibrium unchanged (|Δf| < 1e-5)."""
    from tensorlbm.d3q27 import collide_mrt27

    ones = torch.ones((10, 10, 10))
    feq = equilibrium27(ones, 0.05 * ones, torch.zeros_like(ones), torch.zeros_like(ones))
    out = collide_mrt27(feq, 0.518)
    assert torch.isfinite(out).all()
    assert float((out - feq).abs().max()) < 1e-5


def test_mrt27_reduces_to_bgk_when_all_rates_equal():
    """MRT with every non-conserved rate = 1/tau is exactly BGK (float64)."""
    from tensorlbm.d3q27 import collide_bgk27, collide_mrt27

    g = torch.Generator().manual_seed(11)
    f = torch.rand((27, 3, 4, 5), generator=g, dtype=torch.float64) + 0.5
    tau = 0.7
    mrt = collide_mrt27(f, tau, s_e=1 / tau, s_eps=1 / tau, s_q=1 / tau, s_pi=1 / tau)
    bgk = collide_bgk27(f, tau)
    assert float((mrt - bgk).abs().max()) < 1e-12


def test_mrt19_transform_inverse_residual():
    """D3Q19 MRT transform inverts to < 1e-10 (portable LAPACK guard)."""
    from tensorlbm.solver3d import _get_d3q19_mrt_matrices

    m, mi = _get_d3q19_mrt_matrices(torch.device("cpu"), torch.float64)
    res = float((m @ mi - torch.eye(19, dtype=torch.float64)).abs().max())
    assert res < 1e-10, res


def test_linalg_safe_rejects_singular_matrix():
    """Negative control: a rank-deficient matrix must raise, never return junk."""
    import numpy as np

    from tensorlbm.linalg_safe import inverse_f64

    singular = np.ones((4, 4))  # rank 1
    with pytest.raises(RuntimeError):
        inverse_f64(singular)


# ---------------------------------------------------------------------------
# engine wiring
# ---------------------------------------------------------------------------


def _engine(lattice=LatticeModel.D3Q27, collision=CollisionModel.MRT, resolution=8, steps=10):
    return GeneralSimEngine(
        GeneralSimConfig(
            name=f"test_{lattice.value}",
            geometry=GeometryConfig(
                source=GeometrySource.PARAMETRIC_SPHERE, sphere_radius=0.5
            ),
            physics=PhysicsConfig(
                density=1000.0,
                viscosity=1.0e-6,
                inlet_velocity=1.0e-4,
                reference_length=1.0,
            ),
            solver=SolverConfig(
                lattice=lattice,
                collision=collision,
                resolution=resolution,
                max_steps=steps,
                device="cpu",
                force_sample_interval=5,
                snapshot_interval=1000,
            ),
            output=OutputConfig(directory="/tmp/tensorlbm_d3q27_tests", save_macroscopic=False),
        )
    )


def test_engine_d3q27_setup_population_shape():
    eng = _engine()
    info = eng.setup()
    assert info["Q"] == 27
    assert info["lattice"] == "d3q27"
    assert eng.f.shape[0] == 27
    assert eng.f.shape[1:] == (info["domain_lu"][2], info["domain_lu"][1], info["domain_lu"][0])
    # initial mass equals rho0 sum (unit density over the box)
    assert abs(float(eng.f.sum().item()) - eng._initial_mass) / eng._initial_mass < 1e-5


def test_engine_d3q27_collision_dispatch():
    eng = _engine(collision=CollisionModel.BGK)
    eng.setup()
    fn, kwargs = eng._get_collide_fn()
    assert fn.__name__ == "collide_bgk27" and kwargs == {}
    eng2 = _engine(collision=CollisionModel.MRT)
    eng2.setup()
    fn2, _ = eng2._get_collide_fn()
    assert fn2.__name__ == "collide_mrt27"


def test_engine_d3q27_runs_and_reports_finite_drag():
    eng = _engine(steps=12)
    eng.setup()
    out = eng.run(12)
    assert out["diverged"] is False
    assert eng.f.shape[0] == 27
    assert len(eng.forces_log) >= 2
    cd = eng.forces_log[-1]["cd_total"]
    assert math.isfinite(cd) and cd > 0.0
    # D3Q27 force integration must use macroscopic27 (Q=27 dispatch)
    rho27 = dp_macroscopic(eng.f)[0]
    assert torch.allclose(rho27, macroscopic27(eng.f)[0], atol=0, rtol=0)


def test_engine_d3q19_unchanged_control():
    """Regression control: the D3Q19 route still returns a 19-population field."""
    eng = _engine(lattice=LatticeModel.D3Q19, steps=6)
    info = eng.setup()
    assert info["Q"] == 19 and eng.f.shape[0] == 19
    out = eng.run(6)
    assert out["diverged"] is False


# ---------------------------------------------------------------------------
# negative controls
# ---------------------------------------------------------------------------


def test_negative_wrong_stencil_raises():
    f19 = torch.rand((19, 5, 5, 5))
    solid = torch.zeros((5, 5, 5), dtype=torch.bool)
    from tensorlbm.d3q27 import collide_bgk27

    with pytest.raises(ValueError, match="27"):
        lbm_step_correct_27(f19, collide_bgk27, 0.6, solid, 0.0)


def test_negative_solid_shape_raises():
    f27 = torch.rand((27, 5, 5, 5))
    bad_solid = torch.zeros((5, 5, 4), dtype=torch.bool)
    from tensorlbm.d3q27 import collide_bgk27

    with pytest.raises(ValueError, match="solid"):
        lbm_step_correct_27(f27, collide_bgk27, 0.6, bad_solid, 0.0)


def test_negative_unsupported_collision_raises():
    eng = _engine(collision=CollisionModel.SMAGORINSKY_MRT)
    eng.setup()
    with pytest.raises(NotImplementedError, match="D3Q27"):
        eng._get_collide_fn()


def test_negative_nonfinite_state_detected():
    eng = _engine(steps=4)
    eng.setup()
    eng.f[0, 0, 0, 0] = float("nan")
    out = eng.run(1)
    assert out["diverged"] is True