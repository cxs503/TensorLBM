"""Tests for the kinetic-flux (KFVS) compressible solver.

Verification anchors are analytic:
  * smooth-flow limit of the flux == Euler flux (consistency)
  * shock tube post-shock plateau == Rankine-Hugoniot at the measured shock Mach
  * entropy production positive inside the captured shock, ~0 outside
  * steady stagnation enthalpy conserved outside the captured discontinuity
  * O-grid geometry sanity (areas, normals, wall closure)

The 2D body-fitted path is exercised for *stability only* (a slow probe); its
current limitation is documented in docs/P153W1_N05_COMPRESSIBLE_AERO.md.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.compressible.kinetic_flux import (
    GAMMA,
    KineticFluxSolver,
    ShockTube1D,
    SolverConfig,
    ackeret_cp,
    build_annulus_mesh,
    contour_loop,
    experimental_anchor_status,
    internal_energy_correction,
    karman_tsien_cp,
    kfvs_flux,
    moving_shock_ratios,
    naca0012_half_thickness,
    normal_shock_ratios,
    prandtl_glauert_delta_cp,
    radial_levels,
)


# ---------------------------------------------------------------------------
class TestKineticFlux:
    def test_smooth_limit_is_euler_flux(self):
        """F(W, W) must equal the Euler flux for arbitrary W and normal."""
        rho, u, v, p = 1.0, 0.3, -0.15, 1.0
        nx, ny = 0.6, 0.8
        F = kfvs_flux(rho, u, v, p, rho, u, v, p, np.array(nx), np.array(ny))
        E = p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)
        qn = u * nx + v * ny
        exact = [rho * qn, rho * u * qn + p * nx, rho * v * qn + p * ny,
                 (E + p) * qn]
        for f, e in zip(F, exact):
            assert abs(float(f) - e) < 1e-12 * max(1.0, abs(e))

    def test_energy_correction_monatomic(self):
        """gamma=5/3 must recover the plain Maxwellian splitting (beta=1)."""
        assert internal_energy_correction(5.0 / 3.0) == pytest.approx(1.0)
        assert internal_energy_correction(1.4) == pytest.approx(2.0)

    def test_pressure_flux_at_rest(self):
        """Fluid at rest carries only the pressure flux (no mass/energy flux)."""
        F = kfvs_flux(1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
                      np.array(1.0), np.array(0.0))
        assert abs(float(F[0])) < 1e-14          # no mass flux
        assert float(F[1]) == pytest.approx(1.0, rel=1e-12)   # momentum == p n
        assert abs(float(F[3])) < 1e-12          # no energy flux
        # oblique normal: pressure acts along the normal only
        F2 = kfvs_flux(1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
                       np.array(0.6), np.array(0.8))
        assert float(F2[1]) == pytest.approx(0.6, rel=1e-12)
        assert float(F2[2]) == pytest.approx(0.8, rel=1e-12)


class TestRankineHugoniot:
    @pytest.mark.parametrize("M1,rho,p,T", [
        (2.0, 2.666667, 4.5, 1.6875),
        (3.0, 3.857143, 10.333333, 2.679012),
        (5.0, 5.0, 29.0, 5.8),
    ])
    def test_stationary_relations(self, M1, rho, p, T):
        rh = normal_shock_ratios(M1)
        assert rh["rho_ratio"] == pytest.approx(rho, rel=1e-5)
        assert rh["p_ratio"] == pytest.approx(p, rel=1e-5)
        assert rh["T_ratio"] == pytest.approx(T, rel=1e-5)

    def test_moving_shock_relations_are_consistent(self):
        """High-Mach limit of the density ratio is (gamma+1)/(gamma-1)."""
        rh = moving_shock_ratios(200.0)
        assert rh["rho_ratio"] == pytest.approx((GAMMA + 1) / (GAMMA - 1), rel=2e-3)

    def test_moving_shock_rejects_subsonic(self):
        with pytest.raises(ValueError):
            moving_shock_ratios(0.9)


class TestShockCapturing:
    @pytest.fixture(scope="class")
    def sod(self):
        st = ShockTube1D(nx=401, order=2)
        st.run_to(0.10)
        x1, t1 = st.shock_position(), st.time
        st.run_to(0.20)
        x2, t2 = st.shock_position(), st.time
        return st, (x2 - x1) / (t2 - t1), x2

    def test_shock_speed(self, sod):
        st, speed, _ = sod
        a_pre = math.sqrt(GAMMA * 0.1 / 0.125)
        assert speed / a_pre == pytest.approx(1.656, rel=0.02)

    def test_rankine_hugoniot_error_below_2pct(self, sod):
        st, speed, xs = sod
        a_pre = math.sqrt(GAMMA * 0.1 / 0.125)
        rh = moving_shock_ratios(speed / a_pre)
        pr = st.profiles()
        i1 = int(np.searchsorted(pr["x"], xs - 0.08))
        i2 = int(np.searchsorted(pr["x"], xs - 0.05))
        p_pl = float(np.mean(pr["p"][i1:i2]))
        rho_pl = float(np.mean(pr["rho"][i1:i2]))
        assert abs(p_pl / 0.1 / rh["p_ratio"] - 1.0) < 0.02
        assert abs(rho_pl / 0.125 / rh["rho_ratio"] - 1.0) < 0.02

    def test_entropy_production_localised(self, sod):
        st, _, xs = sod
        pr = st.profiles()
        ep = st.entropy_production()
        shock_band = np.abs(pr["x"] - xs) < 0.02
        smooth_band = pr["x"] < 0.15
        assert np.max(np.abs(ep[shock_band])) > 0.1          # entropy generated
        assert np.max(np.abs(ep[smooth_band])) < 1e-3        # isentropic regions

    def test_entropy_jump_matches_exact(self, sod):
        st, speed, xs = sod
        pr = st.profiles()
        a_pre = math.sqrt(GAMMA * 0.1 / 0.125)
        rh = moving_shock_ratios(speed / a_pre)
        ds = (pr["s"][int(np.searchsorted(pr["x"], xs - 0.06))]
              - pr["s"][int(np.searchsorted(pr["x"], xs + 0.06))])
        ds_exact = math.log(rh["p_ratio"]) - GAMMA * math.log(rh["rho_ratio"])
        assert abs(ds - ds_exact) / abs(ds_exact) < 0.05

    def test_no_negative_states(self, sod):
        st, _, _ = sod
        pr = st.profiles()
        assert np.all(pr["p"] > 0.0) and np.all(pr["rho"] > 0.0)


class TestSteadyShockEnthalpy:
    def test_enthalpy_conserved_outside_shock(self):
        from tensorlbm.compressible.kinetic_flux import NormalShock1D
        sol = NormalShock1D(nx=401, mach=2.0, order=2)
        sol.run(2000, log_every=1000)
        pr = sol.profiles()
        H = pr["H"]
        dH = np.abs(H[10:-10] - H[10]) / H[10]
        # outside the ~5 captured shock cells the steady adiabatic flow must
        # conserve stagnation enthalpy
        assert np.median(dH) < 1e-4
        assert dH[60] < 3e-3

    def test_plateau_matches_rankine_hugoniot(self):
        from tensorlbm.compressible.kinetic_flux import NormalShock1D
        rhs = normal_shock_ratios(2.0)
        sol = NormalShock1D(nx=401, mach=2.0, order=2)
        sol.run(2000, log_every=1000)
        m = sol.plateau_ratios(40, 40)
        assert abs(m["p_ratio"] / rhs["p_ratio"] - 1) < 0.02
        assert abs(m["rho_ratio"] / rhs["rho_ratio"] - 1) < 0.02


class TestOgrid:
    def test_mesh_geometry(self):
        x_loop, y_loop = contour_loop(naca0012_half_thickness, n_points=64)
        mesh = build_annulus_mesh(x_loop, y_loop, radial_levels(20, 0.02, 8.0))
        assert mesh.n_cells == 64 * 19
        assert np.all(mesh.area > 0)
        # first-cell area far below the far-field cell area (wall clustering)
        assert mesh.area[:, 0].mean() < mesh.area[:, -1].mean()
        # wall normals point away from the body centroid
        cx, cy = 0.4, 0.0
        dot = (mesh.jface_nx[:, 0] * (mesh.wall_x - cx)
               + mesh.jface_ny[:, 0] * (mesh.wall_y - cy))
        assert np.mean(dot > 0) > 0.95
        # the loop closes: first and last wall points are the trailing edge
        assert abs(mesh.wall_x[0] - 1.0) < 0.02

    def test_face_vector_closure(self):
        """Sum of face vectors must vanish per cell (metric closure).

        This is a regression guard: a per-face sign flip in the normal
        orientation breaks closure and generates spurious momentum on a
        curvilinear mesh (that bug made the 2D path diverge).
        """
        x_loop, y_loop = contour_loop(naca0012_half_thickness, n_points=64)
        mesh = build_annulus_mesh(x_loop, y_loop, radial_levels(16, 0.03, 8.0))
        Sxi = mesh.iface_nx * mesh.iface_len
        Syi = mesh.iface_ny * mesh.iface_len
        Sxj = mesh.jface_nx * mesh.jface_len
        Syj = mesh.jface_ny * mesh.jface_len
        close_x = np.roll(Sxi, -1, axis=0) - Sxi + Sxj[:, 1:] - Sxj[:, :-1]
        close_y = np.roll(Syi, -1, axis=0) - Syi + Syj[:, 1:] - Syj[:, :-1]
        assert np.max(np.abs(close_x)) < 1e-10
        assert np.max(np.abs(close_y)) < 1e-10


class TestAnalyticAnchors:
    def test_ackeret_flat_plate(self):
        theta = 0.05
        cp = float(ackeret_cp(theta, 2.0))
        assert cp == pytest.approx(2.0 * theta / math.sqrt(3.0), rel=1e-12)

    def test_ackeret_rejects_subsonic(self):
        with pytest.raises(ValueError):
            ackeret_cp(0.05, 0.8)

    def test_karman_tsien_limits(self):
        cp0 = -0.5
        # zero Mach -> identity
        assert float(karman_tsien_cp(cp0, 0.0)) == pytest.approx(cp0, rel=1e-6)
        # supersonic correction magnitude grows
        assert abs(float(karman_tsien_cp(cp0, 0.6))) > abs(cp0)

    def test_prandtl_glauert_delta_cp(self):
        assert prandtl_glauert_delta_cp(2.0, 0.0) == pytest.approx(
            4.0 * math.radians(2.0), rel=1e-12)
        assert prandtl_glauert_delta_cp(2.0, 0.63) > prandtl_glauert_delta_cp(2.0, 0.0)


class TestExperimentalAnchorDiscipline:
    def test_missing_dataset_is_withheld(self):
        st = experimental_anchor_status("/nonexistent/agard.json")
        assert st["status"] == "WITHHELD"

    def test_dataset_without_provenance_is_withheld(self, tmp_path):
        import json
        p = tmp_path / "agard.json"
        p.write_text(json.dumps({"cases": {"x": [1, 2]}}), encoding="utf-8")
        assert experimental_anchor_status(str(p))["status"] == "WITHHELD"

    def test_dataset_with_provenance_is_available(self, tmp_path):
        import json
        p = tmp_path / "agard.json"
        p.write_text(json.dumps({"provenance": {"source": "AGARD-AR-138",
                                                "digitised_by": "agent",
                                                "date": "2026-11-09"},
                                 "cases": {"Ma0.63_alpha2": {}}}), encoding="utf-8")
        assert experimental_anchor_status(str(p))["status"] == "AVAILABLE"