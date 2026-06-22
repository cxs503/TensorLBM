"""Tests for gap-closure capabilities: cumulant LBM, streamlines,
surface integrals, and inlet profiles.

These features bring TensorLBM closer to parity with PowerFlow and XFlow.
"""
from __future__ import annotations

import math

import pytest
import torch


# ---------------------------------------------------------------------------
# Cumulant LBM (D2Q9 and D3Q27)
# ---------------------------------------------------------------------------

class TestCumulantD2Q9:
    """D2Q9 cumulant collision operator unit tests."""

    def _make_uniform_f(self, ny: int = 8, nx: int = 12, u: float = 0.05) -> torch.Tensor:
        from tensorlbm.d2q9 import equilibrium

        rho = torch.ones(ny, nx)
        ux = torch.full((ny, nx), u)
        uy = torch.zeros(ny, nx)
        return equilibrium(rho, ux, uy)

    def test_output_shape(self):
        from tensorlbm.cumulant import collide_cumulant_d2q9

        f = self._make_uniform_f()
        f_out = collide_cumulant_d2q9(f, tau=0.8)
        assert f_out.shape == f.shape

    def test_mass_conservation(self):
        """Total mass (sum of all populations) must be conserved."""
        from tensorlbm.cumulant import collide_cumulant_d2q9

        f = self._make_uniform_f(ny=16, nx=32, u=0.1)
        f_out = collide_cumulant_d2q9(f, tau=0.7)
        mass_in = float(f.sum())
        mass_out = float(f_out.sum())
        assert abs(mass_in - mass_out) < 1e-4, (
            f"Mass not conserved: in={mass_in}, out={mass_out}"
        )

    def test_momentum_conservation_x(self):
        """x-momentum must be conserved during collision."""
        from tensorlbm.cumulant import collide_cumulant_d2q9
        from tensorlbm.d2q9 import macroscopic

        f = self._make_uniform_f(u=0.08)
        # Perturb slightly to get non-equilibrium distributions
        torch.manual_seed(7)
        f = f + 0.001 * torch.randn_like(f)
        f_out = collide_cumulant_d2q9(f, tau=0.75)

        rho_in, ux_in, uy_in = macroscopic(f)
        rho_out, ux_out, uy_out = macroscopic(f_out)

        jx_in = (rho_in * ux_in).sum()
        jx_out = (rho_out * ux_out).sum()
        # Momentum exactly conserved (only floating-point round-off)
        assert abs(float(jx_in - jx_out)) < 0.05

    def test_equilibrium_is_fixed_point(self):
        """Applying the cumulant operator to an equilibrium distribution gives f_eq back."""
        from tensorlbm.cumulant import collide_cumulant_d2q9
        from tensorlbm.d2q9 import equilibrium

        rho = torch.ones(8, 12) * 1.0
        ux = torch.full((8, 12), 0.05)
        uy = torch.zeros(8, 12)
        feq = equilibrium(rho, ux, uy)
        f_out = collide_cumulant_d2q9(feq, tau=0.8)
        assert torch.allclose(feq, f_out, atol=1e-5), (
            "f_eq should be a fixed point of the collision operator"
        )

    def test_omega_b_effect(self):
        """Changing omega_b should alter the trace mode of the output."""
        from tensorlbm.cumulant import collide_cumulant_d2q9

        f = self._make_uniform_f()
        f_perturbed = f + 0.01 * torch.randn_like(f)
        f1 = collide_cumulant_d2q9(f_perturbed, tau=0.8, omega_b=1.0)
        f2 = collide_cumulant_d2q9(f_perturbed, tau=0.8, omega_b=1.8)
        # Outputs should differ when omega_b differs
        assert not torch.allclose(f1, f2, atol=1e-8)

    def test_different_tau_values(self):
        """Cumulant operator should work for a range of tau values."""
        from tensorlbm.cumulant import collide_cumulant_d2q9

        for tau in [0.51, 0.6, 0.8, 1.0, 1.5, 2.0]:
            f = self._make_uniform_f()
            f_out = collide_cumulant_d2q9(f, tau=tau)
            assert not torch.isnan(f_out).any(), f"NaN detected at tau={tau}"
            assert not torch.isinf(f_out).any(), f"Inf detected at tau={tau}"


class TestCumulantD3Q27:
    """D3Q27 cumulant collision operator unit tests."""

    def _make_f3d(self, nz: int = 4, ny: int = 6, nx: int = 8, u: float = 0.05) -> torch.Tensor:
        from tensorlbm.d3q27 import equilibrium27

        rho = torch.ones(nz, ny, nx)
        ux = torch.full((nz, ny, nx), u)
        uy = torch.zeros(nz, ny, nx)
        uz = torch.zeros(nz, ny, nx)
        return equilibrium27(rho, ux, uy, uz)

    def test_output_shape(self):
        from tensorlbm.cumulant import collide_cumulant_d3q27

        f = self._make_f3d()
        f_out = collide_cumulant_d3q27(f, tau=0.8)
        assert f_out.shape == f.shape

    def test_mass_conservation(self):
        from tensorlbm.cumulant import collide_cumulant_d3q27

        f = self._make_f3d()
        f_perturbed = f + 0.001 * torch.randn_like(f)
        f_out = collide_cumulant_d3q27(f_perturbed, tau=0.75)
        mass_in = float(f_perturbed.sum())
        mass_out = float(f_out.sum())
        assert abs(mass_in - mass_out) < 1e-3

    def test_no_nan_inf(self):
        from tensorlbm.cumulant import collide_cumulant_d3q27

        f = self._make_f3d(u=0.1)
        for tau in [0.6, 1.0, 1.5]:
            f_out = collide_cumulant_d3q27(f, tau=tau)
            assert not torch.isnan(f_out).any()
            assert not torch.isinf(f_out).any()


# ---------------------------------------------------------------------------
# Streamline tracing
# ---------------------------------------------------------------------------

class TestStreamlines2D:
    """2-D streamline tracing tests."""

    def _uniform_flow(self, ny: int = 20, nx: int = 40, u: float = 0.05):
        ux = torch.full((ny, nx), u)
        uy = torch.zeros(ny, nx)
        return ux, uy

    def test_basic_tracing(self):
        from tensorlbm.streamlines import trace_streamlines_2d

        ux, uy = self._uniform_flow()
        seeds = [(5.0, 10.0), (5.0, 15.0)]
        lines = trace_streamlines_2d(ux, uy, seeds, step_size=0.5, max_steps=20)
        assert len(lines) == 2
        for sl in lines:
            assert len(sl.points) >= 1

    def test_uniform_flow_is_straight(self):
        """In a uniform x-flow all streamlines should run horizontally."""
        from tensorlbm.streamlines import trace_streamlines_2d

        ux, uy = self._uniform_flow(u=0.1)
        seeds = [(2.0, 10.0)]
        lines = trace_streamlines_2d(ux, uy, seeds, step_size=0.3, max_steps=100)
        sl = lines[0]
        assert len(sl.points) > 5
        # y-coordinates should remain near constant for pure x-flow
        ys = [p[1] for p in sl.points]
        y_span = max(ys) - min(ys)
        assert y_span < 2.0, f"Streamline deviated too much in y: span={y_span}"

    def test_seed_points_uniform(self):
        from tensorlbm.streamlines import seed_points_uniform_2d

        seeds = seed_points_uniform_2d(40, 20, n_x=4, n_y=4)
        assert len(seeds) == 16
        xs = [s[0] for s in seeds]
        ys = [s[1] for s in seeds]
        assert min(xs) >= 0.0 and max(xs) < 40.0
        assert min(ys) >= 0.0 and max(ys) < 20.0

    def test_seed_points_line(self):
        from tensorlbm.streamlines import seed_points_line_2d

        seeds = seed_points_line_2d(x_seed=5.0, ny=20, n_seeds=8)
        assert len(seeds) == 8
        assert all(s[0] == 5.0 for s in seeds)

    def test_with_scalar_field(self):
        from tensorlbm.streamlines import trace_streamlines_2d

        ux, uy = self._uniform_flow()
        speed = torch.sqrt(ux * ux + uy * uy)
        seeds = [(5.0, 10.0)]
        lines = trace_streamlines_2d(ux, uy, seeds, scalar_field=speed, max_steps=10)
        sl = lines[0]
        # Scalars should be populated
        assert len(sl.scalars) == len(sl.points)
        assert all(s > 0.0 for s in sl.scalars)

    def test_mask_stops_integration(self):
        """Streamline must stop when it hits a solid cell."""
        from tensorlbm.streamlines import trace_streamlines_2d

        ux, uy = self._uniform_flow(nx=40)
        mask = torch.zeros(20, 40, dtype=torch.bool)
        mask[:, 20] = True   # solid wall at x=20
        seeds = [(5.0, 10.0)]
        lines = trace_streamlines_2d(ux, uy, seeds, step_size=0.5, max_steps=1000, mask=mask)
        sl = lines[0]
        # All points should be upstream of x=20
        xs = [p[0] for p in sl.points]
        assert max(xs) < 22.0

    def test_bidirectional(self):
        from tensorlbm.streamlines import trace_streamlines_2d

        ux, uy = self._uniform_flow(u=0.1)
        seeds = [(20.0, 10.0)]
        lines_fwd = trace_streamlines_2d(ux, uy, seeds, max_steps=20)
        lines_bi = trace_streamlines_2d(ux, uy, seeds, max_steps=20, bidirectional=True)
        # Bidirectional trace should have more points
        assert len(lines_bi[0].points) >= len(lines_fwd[0].points)

    def test_streamlines_to_dict(self):
        from tensorlbm.streamlines import streamlines_to_dict, trace_streamlines_2d

        ux, uy = self._uniform_flow()
        seeds = [(5.0, 10.0)]
        lines = trace_streamlines_2d(ux, uy, seeds, max_steps=10)
        d = streamlines_to_dict(lines)
        assert "n_lines" in d
        assert "lines" in d
        assert d["n_lines"] == 1
        assert isinstance(d["lines"][0]["points"], list)

    def test_stagnation_terminates(self):
        """Zero-velocity field should terminate quickly."""
        from tensorlbm.streamlines import trace_streamlines_2d

        ux = torch.zeros(10, 20)
        uy = torch.zeros(10, 20)
        seeds = [(5.0, 5.0)]
        lines = trace_streamlines_2d(ux, uy, seeds, max_steps=1000)
        sl = lines[0]
        assert sl.steps < 5


class TestStreamlines3D:
    """3-D streamline tracing smoke tests."""

    def test_basic_3d_tracing(self):
        from tensorlbm.streamlines import trace_streamlines_3d

        nz, ny, nx = 8, 10, 16
        ux = torch.full((nz, ny, nx), 0.05)
        uy = torch.zeros(nz, ny, nx)
        uz = torch.zeros(nz, ny, nx)
        seeds = [(2.0, 5.0, 4.0)]
        lines = trace_streamlines_3d(ux, uy, uz, seeds, max_steps=20)
        assert len(lines) == 1
        assert len(lines[0].points) > 0

    def test_seed_points_uniform_3d(self):
        from tensorlbm.streamlines import seed_points_uniform_3d

        seeds = seed_points_uniform_3d(16, 10, 8, n_x=2, n_y=2, n_z=2)
        assert len(seeds) == 8


# ---------------------------------------------------------------------------
# Surface integrals
# ---------------------------------------------------------------------------

class TestSurfaceIntegrals:
    """Surface integral utility tests."""

    def test_mass_flow_rate_2d_uniform(self):
        """Mass flow should equal n_cells * u_in for uniform flow."""
        from tensorlbm.surface_integrals import mass_flow_rate_2d

        ny, nx = 10, 20
        ux = torch.full((ny, nx), 0.1)
        rho = torch.ones(ny, nx)
        result = mass_flow_rate_2d(ux, rho, x_plane=10)
        assert abs(result["volume_flow"] - ny * 0.1) < 1e-5
        assert result["area"] == ny
        assert abs(result["mean_velocity"] - 0.1) < 1e-5

    def test_mass_flow_with_y_range(self):
        from tensorlbm.surface_integrals import mass_flow_rate_2d

        ux = torch.full((20, 30), 0.05)
        rho = torch.ones(20, 30)
        result = mass_flow_rate_2d(ux, rho, x_plane=15, y_range=(5, 14))
        assert result["area"] == 10

    def test_pressure_drop_uniform(self):
        """For uniform density the pressure drop should be zero."""
        from tensorlbm.surface_integrals import pressure_drop

        rho = torch.ones(10, 30)
        result = pressure_drop(rho, x_upstream=5, x_downstream=25)
        assert abs(result["delta_p"]) < 1e-8

    def test_pressure_drop_gradient(self):
        """Linearly decreasing density should give positive pressure drop."""
        from tensorlbm.surface_integrals import pressure_drop

        nx = 30
        rho = torch.linspace(1.1, 0.9, nx).unsqueeze(0).expand(10, nx)
        result = pressure_drop(rho, x_upstream=2, x_downstream=27)
        assert result["delta_p"] > 0.0

    def test_area_average_2d_uniform(self):
        from tensorlbm.surface_integrals import area_average_2d

        s = torch.full((20, 30), 2.5)
        result = area_average_2d(s)
        assert abs(result["mean"] - 2.5) < 1e-5
        assert abs(result["min"] - 2.5) < 1e-5
        assert abs(result["max"] - 2.5) < 1e-5

    def test_area_average_with_mask(self):
        """Solid cells (mask=True) should be excluded from the average."""
        from tensorlbm.surface_integrals import area_average_2d

        s = torch.ones(10, 10)
        s[5, 5] = 999.0   # would skew average if included
        mask = torch.zeros(10, 10, dtype=torch.bool)
        mask[5, 5] = True
        result = area_average_2d(s, mask=mask)
        assert abs(result["max"] - 1.0) < 1e-5

    def test_surface_force_zero_for_no_solid(self):
        """With an all-fluid mask, the surface force should be zero."""
        from tensorlbm.surface_integrals import surface_force_2d
        from tensorlbm.d2q9 import equilibrium

        ny, nx = 8, 12
        rho = torch.ones(ny, nx)
        ux = torch.full((ny, nx), 0.05)
        uy = torch.zeros(ny, nx)
        f = equilibrium(rho, ux, uy)
        mask = torch.zeros(ny, nx, dtype=torch.bool)
        result = surface_force_2d(f, mask)
        assert result["fx"] == 0.0
        assert result["fy"] == 0.0

    def test_force_coefficients_cd(self):
        """CD should equal Fdrag / q where q = 0.5 * rho * U^2 * A."""
        from tensorlbm.surface_integrals import force_coefficients

        fdrag = 0.05
        rho = 1.0
        u = 0.1
        area = 10.0
        result = force_coefficients(fdrag, 0.0, 0.0, rho, u, area)
        expected_cd = fdrag / (0.5 * rho * u * u * area)
        assert abs(result["cd"] - expected_cd) < 1e-8

    def test_mass_flow_rate_3d(self):
        from tensorlbm.surface_integrals import mass_flow_rate_3d

        nz, ny, nx = 4, 8, 16
        ux = torch.full((nz, ny, nx), 0.1)
        rho = torch.ones(nz, ny, nx)
        result = mass_flow_rate_3d(ux, rho, x_plane=8)
        assert abs(result["volume_flow"] - nz * ny * 0.1) < 1e-4
        assert result["area"] == nz * ny

    def test_surface_moment_2d_zero_force(self):
        """Zero-force case should give zero moment."""
        from tensorlbm.surface_integrals import surface_moment_2d
        from tensorlbm.d2q9 import equilibrium

        ny, nx = 8, 12
        rho = torch.ones(ny, nx)
        ux = torch.full((ny, nx), 0.05)
        uy = torch.zeros(ny, nx)
        f = equilibrium(rho, ux, uy)
        mask = torch.zeros(ny, nx, dtype=torch.bool)
        result = surface_moment_2d(f, mask, pivot_x=nx / 2, pivot_y=ny / 2)
        assert result["fx"] == 0.0
        assert result["mz"] == 0.0

    def test_moment_coefficients(self):
        from tensorlbm.surface_integrals import moment_coefficients

        result = moment_coefficients(0.01, 0.0, 0.0, 1.0, 0.1, 10.0, 5.0)
        q = 0.5 * 1.0 * 0.1 * 0.1 * 10.0 * 5.0
        assert abs(result["cl_roll"] - 0.01 / q) < 1e-8


# ---------------------------------------------------------------------------
# Inlet profiles
# ---------------------------------------------------------------------------

class TestInletProfiles:
    """Inlet velocity profile generator tests."""

    def test_log_law_shape(self):
        from tensorlbm.inlet_profiles import log_law_profile

        n = 64
        p = log_law_profile(n, u_bulk=0.1, re_tau=300.0, nu=1 / 900.0)
        assert p.shape == (n,)
        # Should be symmetric (channel flow)
        assert abs(float(p[0]) - float(p[-1])) < 0.01
        # Centreline > wall
        assert float(p[n // 2]) > float(p[0])

    def test_log_law_bulk_velocity(self):
        """Mean of the log-law profile should approximately equal u_bulk."""
        from tensorlbm.inlet_profiles import log_law_profile

        u_bulk = 0.08
        p = log_law_profile(128, u_bulk, re_tau=200.0, nu=1 / 600.0)
        assert abs(float(p.mean()) - u_bulk) < 0.005

    def test_power_law_shape(self):
        from tensorlbm.inlet_profiles import power_law_profile

        p = power_law_profile(64, u_centerline=0.1, exponent=7.0)
        assert p.shape == (64,)
        # Centreline velocity should equal u_centerline
        assert abs(float(p[32]) - 0.1) < 0.005
        # Wall cell (y=0.5) has a small but non-zero velocity in discrete grid
        assert float(p[0]) < 0.08   # well below centreline value

    def test_parabolic_profile(self):
        from tensorlbm.inlet_profiles import parabolic_profile

        p = parabolic_profile(64, u_centerline=0.1)
        assert p.shape == (64,)
        assert abs(float(p[32]) - 0.1) < 0.002
        assert float(p[0]) < 0.01

    def test_blasius_profile(self):
        from tensorlbm.inlet_profiles import blasius_profile

        # Use a large delta_99 so the B.L. spans the domain
        p = blasius_profile(64, u_inf=0.1, delta_99=40.0)
        assert p.shape == (64,)
        # Free-stream region (far from wall) should be near u_inf
        assert abs(float(p[-1]) - 0.1) < 0.02
        # Cells inside the boundary layer should be below u_inf
        assert float(p[0]) < float(p[-1])

    def test_womersley_profile(self):
        from tensorlbm.inlet_profiles import womersley_profile

        p = womersley_profile(64, u_mean=0.1, wo=5.0, phase=0.0)
        assert p.shape == (64,)
        assert not torch.isnan(p).any()

    def test_synthetic_turbulence(self):
        from tensorlbm.inlet_profiles import power_law_profile, synthetic_turbulence_2d

        mean = power_law_profile(64, 0.1)
        perturbed = synthetic_turbulence_2d(mean, turbulence_intensity=0.05, seed=0)
        assert perturbed.shape == (64,)
        # Perturbation should be non-trivial
        diff = (perturbed - mean).abs()
        assert float(diff.max()) > 1e-6

    def test_apply_inlet_profile_2d(self):
        from tensorlbm.inlet_profiles import apply_inlet_profile_2d, parabolic_profile
        from tensorlbm.d2q9 import equilibrium, macroscopic

        ny, nx = 16, 24
        rho0 = torch.ones(ny, nx)
        ux0 = torch.full((ny, nx), 0.05)
        uy0 = torch.zeros(ny, nx)
        f = equilibrium(rho0, ux0, uy0)

        profile = parabolic_profile(ny, u_centerline=0.1)
        f_new = apply_inlet_profile_2d(f, profile, x_inlet=0)

        rho_new, ux_new, uy_new = macroscopic(f_new)
        # Inlet column should have the parabolic profile applied
        ux_inlet = ux_new[:, 0]
        assert abs(float(ux_inlet[ny // 2]) - 0.1) < 0.005

    def test_apply_inlet_profile_3d(self):
        from tensorlbm.inlet_profiles import apply_inlet_profile_3d
        from tensorlbm.d3q19 import equilibrium3d, macroscopic3d

        nz, ny, nx = 4, 8, 12
        rho0 = torch.ones(nz, ny, nx)
        ux0 = torch.full((nz, ny, nx), 0.05)
        uy0 = torch.zeros(nz, ny, nx)
        uz0 = torch.zeros(nz, ny, nx)
        f = equilibrium3d(rho0, ux0, uy0, uz0)

        inlet = torch.full((nz, ny), 0.1)
        f_new = apply_inlet_profile_3d(f, inlet, x_inlet=0)
        assert f_new.shape == f.shape


# ---------------------------------------------------------------------------
# API endpoint smoke tests
# ---------------------------------------------------------------------------

class TestStreamlineAPI:
    def test_streamlines_unknown_job(self, client):
        r = client.post(
            "/api/postprocess/streamlines",
            json={"job_id": "no_such_job"},
        )
        assert r.status_code == 404

    def test_streamlines_invalid_seeds(self, client):
        """n_seeds_x above max should be rejected."""
        r = client.post(
            "/api/postprocess/streamlines",
            json={"job_id": "x", "n_seeds_x": 999},
        )
        assert r.status_code in (404, 422)

    def test_openapi_contains_streamlines(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        paths = set(r.json()["paths"].keys())
        assert "/api/postprocess/streamlines" in paths


class TestSurfaceIntegralsAPI:
    def test_surface_integrals_unknown_job(self, client):
        r = client.post(
            "/api/postprocess/surface-integrals",
            json={"job_id": "no_such_job"},
        )
        assert r.status_code == 404

    def test_surface_integrals_invalid_type(self, client):
        r = client.post(
            "/api/postprocess/surface-integrals",
            json={"job_id": "x", "integral_type": "bad_type"},
        )
        # Either 404 (no job) or 422 (bad type after job check)
        assert r.status_code in (404, 422)

    def test_openapi_contains_surface_integrals(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        paths = set(r.json()["paths"].keys())
        assert "/api/postprocess/surface-integrals" in paths


class TestInletProfileAPI:
    def test_inlet_profile_parabolic(self, client):
        r = client.post(
            "/api/postprocess/inlet-profile",
            json={"profile_type": "parabolic", "n": 32, "u_ref": 0.1},
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["n"] == 32
        assert len(data["profile"]) == 32
        assert all("y" in p and "ux" in p for p in data["profile"])

    def test_inlet_profile_log_law(self, client):
        r = client.post(
            "/api/postprocess/inlet-profile",
            json={"profile_type": "log_law", "n": 64, "u_ref": 0.1, "re_tau": 200.0},
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["n"] == 64
        assert abs(data["u_bulk"] - 0.1) < 0.01

    def test_inlet_profile_invalid_type(self, client):
        r = client.post(
            "/api/postprocess/inlet-profile",
            json={"profile_type": "invalid_type"},
        )
        assert r.status_code == 422

    def test_inlet_profile_n_too_large(self, client):
        r = client.post(
            "/api/postprocess/inlet-profile",
            json={"profile_type": "parabolic", "n": 9999},
        )
        assert r.status_code == 422

    def test_inlet_profile_with_turbulence(self, client):
        r = client.post(
            "/api/postprocess/inlet-profile",
            json={
                "profile_type": "power_law",
                "n": 32,
                "u_ref": 0.1,
                "add_synthetic_turbulence": True,
                "turbulence_intensity": 0.05,
            },
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert len(data["profile"]) == 32

    def test_openapi_contains_inlet_profile(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        paths = set(r.json()["paths"].keys())
        assert "/api/postprocess/inlet-profile" in paths


class TestCumulantSolverAPI:
    def test_cumulant_cylinder_schema(self, client):
        """Endpoint should accept valid params and return a job_id."""
        r = client.post(
            "/api/solve/cumulant-cylinder-flow",
            json={
                "nx": 40, "ny": 20,
                "re": 200.0,
                "u_in": 0.05,
                "radius": 5.0,
                "n_steps": 100,
                "output_interval": 50,
                "device": "cpu",
            },
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert "job_id" in data
        assert len(data["job_id"]) > 0

    def test_cumulant_cylinder_invalid_nx(self, client):
        """Grid exceeding schema limit should be rejected."""
        r = client.post(
            "/api/solve/cumulant-cylinder-flow",
            json={"nx": 9999, "ny": 20, "re": 200.0},
        )
        assert r.status_code == 422

    def test_openapi_contains_cumulant_endpoint(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        paths = set(r.json()["paths"].keys())
        assert "/api/solve/cumulant-cylinder-flow" in paths
