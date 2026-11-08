"""D3Q27 activation in GeneralSimEngine (gaps G7/G8) — unit and wiring tests.

Cheap, deterministic checks (no long simulations):

* lattice isotropy moments: rank-2 / rank-4 are *identical and isotropic* for
  D3Q19 and D3Q27; the sets separate only at rank 6 (corner directions);
* :class:`~tensorlbm.general_sim.GeneralSimEngine` routes the D3Q27 lattice
  through ``d3q27.equilibrium27`` / ``collide_*27`` / ``stream27`` /
  ``boundary.boundaries_d3q27`` — verified by equal-shape, bit-exact
  comparison with the manual kernel sequence;
* the D3Q27 path rejects the D3Q19-only features loudly instead of silently
  degrading;
* wet-node momentum exchange is direction-count agnostic (19/27);
* the exact shear-wave decay recovers ν on the D3Q27 path;
* the ``POST /api/v1/calculations/lbm/d3q27_validation`` endpoint returns a
  report on a tiny configuration.

Run::

    python -m pytest tests/test_general_sim_d3q27.py -q
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.boundary.boundaries_d3q27 import bounce_back_cells_27
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.d3q27 import (
    collide_bgk27,
    equilibrium27,
    macroscopic27,
    stream27,
)
from tensorlbm.d3q27_validation import (
    isotropy_lattice_difference,
    isotropy_moment_deviation,
    run_shear_wave_case,
)
from tensorlbm.general_sim import (
    CollisionModel,
    ForceMethod,
    GeneralSimConfig,
    GeneralSimEngine,
    GeometryConfig,
    GeometrySource,
    LatticeModel,
    OutputConfig,
    OutputFormat,
    PhysicsConfig,
    SolverConfig,
    WallTreatment,
)
from tensorlbm.lbm_step_correct import lbm_step_correct
from tensorlbm.momentum_exchange import momentum_exchange_wet_node

torch.set_num_threads(min(8, torch.get_num_threads()))


# ---------------------------------------------------------------------------
# Lattice algebra
# ---------------------------------------------------------------------------


def test_rank2_rank4_isotropy_holds_for_both_lattices():
    """Both lattices reproduce the isotropic rank-2 and rank-4 tensors exactly.

    This falsifies the loose statement "D3Q19 lacks 4th-order isotropy": the
    2nd and 4th moments of D3Q19 are isotropic to round-off (1e-16).
    """
    for q in (19, 27):
        iso = isotropy_moment_deviation(q)
        assert iso["weights_match_library_table"], q
        assert iso["weight_sum"] == pytest.approx(1.0, abs=1e-12)
        assert iso["second_moment_xx"] == pytest.approx(1.0 / 3.0, abs=1e-15)
        assert iso["order2"] < 1e-14, f"D3Q{q} rank-2 residual {iso['order2']}"
        assert iso["order4"] < 1e-14, f"D3Q{q} rank-4 residual {iso['order4']}"
        assert iso["isotropic_order2"] and iso["isotropic_order4"]
        # neither lattice is isotropic at rank 6
        assert iso["order6"] > 1e-2


def test_rank6_moment_separates_the_lattices():
    """D3Q19 and D3Q27 differ at rank 6 (the 8 body-diagonal directions)."""
    diff = isotropy_lattice_difference()
    assert diff["rank4_identically_equal"]
    assert diff["rank4_max_abs_diff"] < 1e-15
    assert not diff["rank6_identically_equal"]
    # 8 corners x w=1/216 contribute exactly 1/27 to a mixed rank-6 component
    assert diff["rank6_max_abs_diff"] == pytest.approx(1.0 / 27.0, rel=1e-12)


# ---------------------------------------------------------------------------
# Engine wiring
# ---------------------------------------------------------------------------


def _engine(lattice: int, resolution: int = 8, steps: int = 6, **solver_kw) -> GeneralSimEngine:
    cfg = GeneralSimConfig(
        name=f"test_d3q{lattice}",
        geometry=GeometryConfig(
            source=GeometrySource.PARAMETRIC_SPHERE,
            sphere_radius=0.5,
            sphere_center=(0.0, 0.0, 0.0),
        ),
        physics=PhysicsConfig(
            density=1000.0, viscosity=1.0e-6, inlet_velocity=1.0e-4, reference_length=1.0
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q27 if lattice == 27 else LatticeModel.D3Q19,
            collision=solver_kw.pop("collision", CollisionModel.BGK),
            wall_treatment=solver_kw.pop("wall_treatment", WallTreatment.AUTO),
            resolution=resolution,
            max_steps=steps,
            snapshot_interval=10**9,
            force_sample_interval=2,
            device="cpu",
            force_method=ForceMethod.BOTH,
            mem_variant="wet_node",
            mass_correction=False,
            **solver_kw,
        ),
        output=OutputConfig(
            directory=f"/tmp/tensorlbm_test_d3q{lattice}",
            formats=[OutputFormat.NPY],
            save_macroscopic=False,
            save_forces=True,
        ),
    )
    return GeneralSimEngine(cfg)


def test_engine_setup_routes_d3q27_kernels():
    """setup() must build a 27-direction state with the D3Q27 equilibrium."""
    eng = _engine(27)
    info = eng.setup()
    assert info["lattice"] == "d3q27"
    assert info["q"] == 27
    assert eng.is_d3q27
    assert eng.f.shape[0] == 27

    # Disturbed state must be recovered by the D3Q27 macroscopic kernel only.
    rho, ux, uy, uz = macroscopic27(eng.f)
    assert torch.allclose(rho, torch.ones_like(rho), atol=1e-5)
    u_in = eng.uc.u_lb
    fluid = ~eng.solid
    assert torch.allclose(ux[fluid], torch.full_like(ux[fluid], u_in), atol=1e-6)

    # The D3Q19 macroscopic kernel is *not* valid for a 27-direction state.
    with pytest.raises(Exception):
        macroscopic3d(eng.f)


def test_engine_d3q19_path_unchanged():
    """The D3Q19 path still produces a 19-direction state (no regression)."""
    eng = _engine(19)
    info = eng.setup()
    assert info["q"] == 19
    assert not eng.is_d3q27
    assert eng.f.shape[0] == 19


def test_stepper_d3q27_matches_manual_kernel_sequence():
    """lbm_step_correct with injected D3Q27 kernels == the manual sequence.

    Proves the shared stepper is lattice-agnostic: collide → NoDynamics →
    half-way bounce-back → stream → far-field BC, with the D3Q27 kernels.
    """
    torch.manual_seed(0)
    nz, ny, nx = 8, 8, 10
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool)
    solid[2:5, 2:6, 4:6] = True
    rho = torch.ones(nz, ny, nx)
    ux = torch.full((nz, ny, nx), 0.03)
    ux[solid] = 0.0
    z = torch.zeros_like(ux)
    f0 = equilibrium27(rho, ux, z, z)
    f0 = f0 + 0.01 * torch.randn_like(f0) * (~solid).unsqueeze(0)
    tau = 0.7

    from tensorlbm.boundary.boundaries_d3q27 import far_field_bc_27

    got = lbm_step_correct(
        f0.clone(),
        collide_bgk27,
        tau,
        solid,
        0.03,
        far_field_bc_27,
        stream_fn=stream27,
        bounce_back_fn=lambda f, m, f_pre=None: bounce_back_cells_27(f, m),
        lattice="D3Q27",
    )

    # manual sequence (same order as the stepper)
    f_pre = f0.clone()
    f = collide_bgk27(f0, tau=tau)
    f = torch.where(solid.unsqueeze(0), f_pre, f)  # NoDynamics
    f = bounce_back_cells_27(f, solid)  # half-way BB (before streaming)
    f = stream27(f)
    f = far_field_bc_27(f, 0.03)

    assert torch.equal(got, f)


def test_engine_d3q27_rejects_d3q19_only_features():
    """wall_function / Smagorinsky have no D3Q27 kernel → loud failure."""
    eng = _engine(27, wall_treatment=WallTreatment.WALL_FUNCTION)
    with pytest.raises(NotImplementedError):
        eng._check_d3q27_supported()

    eng2 = _engine(27, collision=CollisionModel.SMAGORINSKY_MRT)
    with pytest.raises(NotImplementedError):
        eng2._check_d3q27_supported()


def test_engine_d3q27_short_run_produces_finite_drag():
    """A few steps of the D3Q27 engine produce finite force samples."""
    eng = _engine(27, steps=6)
    eng.setup()
    out = eng.run(steps=6)
    assert not out["diverged"]
    assert torch.isfinite(eng.f).all()
    assert eng.forces_log
    last = eng.forces_log[-1]
    assert math.isfinite(last["cd_mem"])
    # D=8 is a very coarse staircase sphere, so the absolute Cd level is not
    # meaningful here (see run_sphere_case for the resolved benchmarks); the
    # assertion only guards against a non-physical blow-up.
    assert abs(last["cd_mem"]) < 1.0e3
    assert math.isfinite(last["cd_total"])


# ---------------------------------------------------------------------------
# Momentum exchange, both direction counts
# ---------------------------------------------------------------------------


def _brute_force_wet_node(f, solid, c):
    nq = f.shape[0]
    nz, ny, nx = solid.shape
    total = torch.zeros(3)
    for i in range(1, nq):
        di, dj, dk = (int(v) for v in c[i].tolist())
        fs = torch.roll(f[i], (-dk, -dj, -di), dims=(0, 1, 2))
        crossing = (~solid) & torch.roll(solid, (-dk, -dj, -di), dims=(0, 1, 2))
        total += 2.0 * c[i] * float(fs[crossing].sum().item())
    return total


def test_wet_node_mem_27_matches_brute_force_and_is_lattice_agnostic():
    from tensorlbm.d3q27 import C as C27

    torch.manual_seed(3)
    nz = ny = nx = 12
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    solid = ((xx - 6) ** 2 + (yy - 6) ** 2 + (zz - 6) ** 2) < 9.0
    f = _random_state(27, nz, ny, nx, solid)

    fx, fy, fz = momentum_exchange_wet_node(f, solid)
    ref = _brute_force_wet_node(f, solid, C27.to(torch.float32))
    # float32 accumulation over ~10^3 links: agree to ~1e-7 relative
    assert fx == pytest.approx(float(ref[0]), rel=1e-5, abs=1e-5)
    assert fy == pytest.approx(float(ref[1]), rel=1e-5, abs=1e-5)
    assert fz == pytest.approx(float(ref[2]), rel=1e-5, abs=1e-5)


def _random_state(nq: int, nz: int, ny: int, nx: int, solid: torch.Tensor) -> torch.Tensor:
    rho = torch.ones(nz, ny, nx)
    ux = torch.full((nz, ny, nx), 0.02)
    z = torch.zeros_like(ux)
    eq = equilibrium27(rho, ux, z, z) if nq == 27 else equilibrium3d(rho, ux, z, z)
    noise = 0.02 * torch.randn_like(eq)
    f = eq + noise * (~solid).unsqueeze(0)
    return f


def test_wet_node_mem_27_zero_on_free_stream():
    """No spurious free-stream background on the D3Q27 path."""
    nz = ny = nx = 12
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    solid = ((xx - 6) ** 2 + (yy - 6) ** 2 + (zz - 6) ** 2) < 9.0
    rho = torch.ones(nz, ny, nx)
    ux = torch.full((nz, ny, nx), 0.02)
    ux[solid] = 0.0
    z = torch.zeros_like(ux)
    f = equilibrium27(rho, ux, z, z)
    fx, fy, fz = momentum_exchange_wet_node(f, solid)
    assert abs(fx) < 1e-6
    assert abs(fy) < 1e-6
    assert abs(fz) < 1e-6


# ---------------------------------------------------------------------------
# Hydrodynamics: exact shear wave on the D3Q27 path
# ---------------------------------------------------------------------------


def test_shear_wave_recovers_viscosity():
    """Exact shear-wave decay recovers ν on both lattices within 10 %."""
    for q in (19, 27):
        res = run_shear_wave_case(
            q, n=16, nu=0.05, amplitude=0.01, wave_vector=(1.0, 0.0, 0.0), direction=(0.0, 1.0, 0.0), periods=1
        )
        assert abs(res["nu_err_pct"]) < 10.0, (q, res["nu_err_pct"])
        assert res["l2_rel_err"] < 0.05


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


def test_api_d3q27_validation_endpoint(tmp_path):
    """POST /api/v1/calculations/lbm/d3q27_validation returns a report."""
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from app.backend.routers import lbm_d3q27

    app = fastapi.FastAPI()
    app.include_router(lbm_d3q27.router, prefix="/api/v1/calculations/lbm")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/calculations/lbm/d3q27_validation",
        json={
            "re": [100.0],
            "resolution": 8,
            "steps": 12,
            "lattices": [19, 27],
            "tg_sizes": [8],
            "isotropy_n": 8,
            "figures": False,
            "out_dir": str(tmp_path / "api_report"),
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "completed"
    assert body["isotropy_moments"]["d3q27"]["isotropic_order4"] is True
    assert "Re100" in body["sphere"]
    assert body["sphere"]["Re100"]["d3q27"]["lattice"] == "D3Q27"
    assert body["verdicts"]["Re100"]["d3q27_within_15pct_of_sn"] in (True, False)
    assert (tmp_path / "api_report" / "d3q27_validation_report.json").exists()


def test_api_rejects_out_of_range_request(tmp_path):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from app.backend.routers import lbm_d3q27

    app = fastapi.FastAPI()
    app.include_router(lbm_d3q27.router, prefix="/api/v1/calculations/lbm")
    client = TestClient(app)
    resp = client.post(
        "/api/v1/calculations/lbm/d3q27_validation",
        json={"resolution": 64, "steps": 12, "out_dir": str(tmp_path)},
    )
    assert resp.status_code == 422

def test_sphere_case_config_applies_reynolds_number():
    """The sphere validation config must actually encode the requested Re.

    Regression guard: an early version pinned ``physics.viscosity`` to the
    Re=100 value, so every "Re=200" case silently ran at Re=100 (identical Cd
    in both report groups).  ν must be derived from Re.
    """
    from tensorlbm.d3q27_validation import _sphere_config

    taus = {}
    for re_v in (100.0, 200.0):
        eng = GeneralSimEngine(_sphere_config(27, re_v, 8, 2, "/tmp/tensorlbm_test_re"))
        info = eng.setup()
        assert info["Re"] == pytest.approx(re_v, rel=1e-9)
        taus[re_v] = info["tau"]
        # tau = 0.5 + 3 nu_lb with nu_lb = u_lb * D / Re
        assert info["tau"] == pytest.approx(0.5 + 3.0 * info["nu_lb"], rel=1e-9)
    assert taus[200.0] < taus[100.0]
