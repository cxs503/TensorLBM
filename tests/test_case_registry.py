"""Tests for tensorlbm.cases (lettuce ExtFlow-style case registry).

Coverage:
  * registry CRUD (register / get / list / unregister, duplicate + unknown
    name rejection);
  * unit-conversion formulas match the verified benchmarks exactly;
  * the three benchmark-aligned cases (cavity / poiseuille / suboff_n128)
    run through the registry+BC-registry path and match the *direct*
    worker/benchmark call chains to 1e-6 in eager mode;
  * BC ids are unique across cases (shared process-wide registry);
  * the poiseuille case's intentional ordered BC overlap warns instead of
    raising;
  * opt-in solver_export integration registers FieldDataProductR2 assets.
"""
from __future__ import annotations

import pytest
import torch

from tensorlbm.boundary_registry import BoundaryCondition, boundary_condition_registry
from tensorlbm.boundaries3d import (
    bounce_back_cells_3d,
    far_field_bc_3d,
    zou_he_inlet_velocity_3d,
    zou_he_moving_lid_3d,
    zou_he_outlet_pressure_3d,
)
from tensorlbm.cases import (
    CaseBase,
    ExportSpec,
    get_case,
    has_case,
    list_cases,
    register_case,
    run_case,
    unregister_case,
)
from tensorlbm.d3q19 import OPPOSITE, equilibrium3d
from tensorlbm.solver3d import (
    collide_bgk3d,
    collide_mrt3d,
    correct_mass3d,
    stream3d,
)
from tensorlbm.suboff_cad import build_suboff_mask

ATOL = 1e-6


# ---------------------------------------------------------------------------
# Registry CRUD
# ---------------------------------------------------------------------------


class TestCaseRegistryCrud:
    def test_builtin_cases_registered(self):
        names = {c["name"] for c in list_cases()}
        assert {"cavity", "poiseuille", "suboff_n128"} <= names
        info = {c["name"]: c for c in list_cases()}
        assert info["cavity"]["lattice"] == "D3Q19"
        assert info["cavity"]["default_params"]["re"] == 400.0
        assert info["suboff_n128"]["default_params"]["resolution"] == 128

    def test_get_case_unknown_lists_available(self):
        with pytest.raises(KeyError, match="cavity"):
            get_case("does_not_exist")

    def test_get_case_instantiates(self):
        case = get_case("cavity", resolution=16, re=100.0)
        assert isinstance(case, CaseBase)
        assert case.resolution == (4, 16, 16)
        assert case.units.tau == pytest.approx(3.0 * 0.06 * 16 / 100.0 + 0.5)

    def test_duplicate_name_rejected(self):
        with pytest.raises(ValueError, match="already registered"):
            @register_case("cavity")
            class DuplicateCavity(CaseBase):
                name = "cavity"

                def make_resolution(self, resolution):
                    return (2, 4, 4)

                def make_units(self, re, resolution):
                    from tensorlbm.cases import CaseUnits

                    return CaseUnits.from_reference(re, 0.05, 4.0)

                def initial_pu(self):
                    nz, ny, nx = self.resolution
                    rho = torch.ones((nz, ny, nx))
                    u = torch.zeros_like(rho)
                    return rho, u, u.clone(), u.clone()

    def test_requires_name(self):
        with pytest.raises(ValueError, match="name"):
            @register_case()
            class Anonymous(CaseBase):
                name = ""

                def make_resolution(self, resolution):
                    return (2, 4, 4)

                def make_units(self, re, resolution):
                    from tensorlbm.cases import CaseUnits

                    return CaseUnits.from_reference(re, 0.05, 4.0)

                def initial_pu(self):
                    nz, ny, nx = self.resolution
                    rho = torch.ones((nz, ny, nx))
                    u = torch.zeros_like(rho)
                    return rho, u, u.clone(), u.clone()

    def test_unregister_and_has_case(self):
        assert has_case("cavity")
        cls = unregister_case("cavity")
        assert not has_case("cavity")
        with pytest.raises(KeyError):
            unregister_case("cavity")
        # restore for the other tests
        register_case("cavity")(cls)


# ---------------------------------------------------------------------------
# Unit formulas match the verified benchmarks
# ---------------------------------------------------------------------------


class TestUnitFormulas:
    def test_cavity_tau_matches_verified_run(self):
        case = get_case("cavity", resolution=96, re=400.0)
        # benchmarks/verified/cavity/3d/run.py: tau = 3*u_lid*nx/re + 0.5
        assert case.units.tau == pytest.approx(3.0 * 0.06 * 96 / 400.0 + 0.5, rel=1e-12)

    def test_suboff_tau_matches_ai4s_pilot(self):
        case = get_case("suboff_n128", resolution=128, re=420.0)
        # examples/ai4s_export.py: nu = u_in*hull_length/re, tau = 3*nu+0.5
        nu = 0.10 * (0.6 * 128) / 420.0
        assert case.units.tau == pytest.approx(3.0 * nu + 0.5, rel=1e-12)

    def test_poiseuille_tau_matches_verified_run(self):
        case = get_case("poiseuille", resolution=10, re=50.0)
        # verified run.py: nu = u_in*2*R_eff/re with R_eff = R+0.5
        nu = 0.05 * 2.0 * 10.5 / 50.0
        assert case.units.tau == pytest.approx(3.0 * nu + 0.5, rel=1e-12)

    def test_bc_ids_unique_across_cases(self):
        cavity = get_case("cavity", resolution=16, re=100.0)
        pipe = get_case("poiseuille", resolution=5, re=30.0)
        ids_cavity = {bc.id for bc in cavity.bcs}
        ids_pipe = {bc.id for bc in pipe.bcs}
        assert 0 not in ids_cavity and 0 not in ids_pipe
        assert not (ids_cavity & ids_pipe)

    def test_poiseuille_ordered_overlap_warns(self):
        with pytest.warns(UserWarning, match="overlap"):
            case = get_case("poiseuille", resolution=5, re=30.0)
        assert len(case.bcs) == 3


# ---------------------------------------------------------------------------
# Registry path vs direct worker/benchmark chains (1e-6, eager)
# ---------------------------------------------------------------------------


def _direct_cavity_chain(nx, nz, re, u_lid, steps):
    """Verbatim step chain of benchmarks/verified/cavity/3d/run.py."""
    ny = nx
    tau = 3.0 * u_lid * nx / re + 0.5
    rho0 = torch.ones((nz, ny, nx))
    u0 = torch.zeros((nz, ny, nx))
    f = equilibrium3d(rho0, u0, u0, u0)
    wall = torch.zeros((nz, ny, nx), dtype=torch.bool)
    wall[:, :, 0] = True
    wall[:, :, -1] = True
    wall[:, 0, :] = True

    def step(f):
        f_pre = f
        f = collide_mrt3d(f, tau)
        f = torch.where(wall.unsqueeze(0), f_pre[OPPOSITE], f)
        f = stream3d(f)
        return zou_he_moving_lid_3d(f, u_lid)

    for _ in range(steps):
        f = step(f)
    return f


def _direct_poiseuille_chain(r, re, u_in, steps):
    """Verbatim step chain of benchmarks/verified/poiseuille_3d_pipe/run.py."""
    ny = nz = 2 * r + 3
    nx = 6 * r
    yc = zc = r + 1
    nu = u_in * 2.0 * (r + 0.5) / re
    tau = 3.0 * nu + 0.5
    iz = torch.arange(nz, dtype=torch.float32).view(-1, 1)
    iy = torch.arange(ny, dtype=torch.float32).view(1, -1)
    d = torch.sqrt((iy - yc) ** 2 + (iz - zc) ** 2)
    fluid = d <= r
    wall = (~fluid).unsqueeze(-1).expand(nz, ny, nx).contiguous()
    r_eff = r + 0.5
    d3 = d.unsqueeze(-1)
    ux0 = torch.where(
        fluid.unsqueeze(-1),
        (2.0 * u_in) * (1.0 - d3**2 / r_eff**2),
        torch.zeros_like(d3),
    ).expand(nz, ny, nx)
    rho0 = torch.ones((nz, ny, nx))
    f = equilibrium3d(rho0, ux0.contiguous(), torch.zeros_like(rho0), torch.zeros_like(rho0))

    def step(f):
        f = collide_bgk3d(f, tau)
        f = stream3d(f)
        f = zou_he_inlet_velocity_3d(f, u_in)
        f = zou_he_outlet_pressure_3d(f, 1.0)
        return bounce_back_cells_3d(f, wall)

    for _ in range(steps):
        f = step(f)
    return f


def _direct_suboff_chain(n, re, u_in, steps, mass_every=10):
    """Verbatim step chain of examples/ai4s_export.py::run_config."""
    nx, ny, nz = n, n // 2, n // 2
    hull_length = 0.6 * n
    nu = u_in * hull_length / re
    tau = 3.0 * nu + 0.5
    solid, _stats = build_suboff_mask(
        hull_type="bare_hull",
        nx=nx, ny=ny, nz=nz,
        cx=nx * 0.35, cy=ny / 2.0, cz=nz / 2.0,
        length=hull_length,
        device="cpu",
    )
    rho0 = torch.ones((nz, ny, nx))
    ux0 = torch.full_like(rho0, u_in)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0))
    initial_mass = float(f.sum().item())

    for step in range(1, steps + 1):
        f = collide_bgk3d(f, tau)
        f = stream3d(f)
        f = far_field_bc_3d(f, u_in, obstacle_mask=solid)
        if step % mass_every == 0:
            f = correct_mass3d(f, initial_mass)
    return f


class TestRegistryMatchesDirectPaths:
    def test_cavity_matches_verified_benchmark_chain(self):
        steps = 30
        direct = _direct_cavity_chain(nx=16, nz=4, re=400.0, u_lid=0.06, steps=steps)
        result = run_case("cavity", steps=steps, resolution=16, re=400.0)
        torch.testing.assert_close(result.f, direct, rtol=0, atol=ATOL)
        assert torch.isfinite(result.f).all()

    def test_poiseuille_matches_verified_benchmark_chain(self):
        steps = 40
        direct = _direct_poiseuille_chain(r=5, re=50.0, u_in=0.05, steps=steps)
        result = run_case("poiseuille", steps=steps, resolution=5, re=50.0)
        torch.testing.assert_close(result.f, direct, rtol=0, atol=ATOL)
        assert torch.isfinite(result.f).all()

    def test_suboff_matches_ai4s_worker_chain(self):
        steps = 20
        direct = _direct_suboff_chain(n=32, re=60.0, u_in=0.05, steps=steps)
        result = run_case(
            "suboff_n128", steps=steps, resolution=32, re=60.0, u_in=0.05,
            collision="bgk",
        )
        torch.testing.assert_close(result.f, direct, rtol=0, atol=ATOL)
        assert torch.isfinite(result.f).all()


# ---------------------------------------------------------------------------
# Runner plumbing and export integration
# ---------------------------------------------------------------------------


class TestRunner:
    def test_result_fields_and_metadata(self):
        result = run_case("cavity", steps=2, resolution=12, re=100.0)
        assert result.steps == 2
        nz, ny, nx = result.case.resolution
        assert result.f.shape == (19, nz, ny, nx)
        assert result.rho.shape == (nz, ny, nx)
        assert result.case.metadata()["case"] == "cavity"
        assert result.case.metadata()["boundary_type"] == "bounce_back,moving_lid"
        assert result.elapsed_s >= 0.0

    def test_steps_validation(self):
        with pytest.raises(ValueError, match="steps"):
            run_case("cavity", steps=-1, resolution=12, re=100.0)

    def test_missing_mask_consistent_with_case_geometry(self):
        case = get_case("cavity", resolution=16, re=100.0)
        mask = case.missing_mask()
        # lid plane y+ interior (away from the wall-adjacent columns x=1
        # and x=nx-2, whose pull sources are the solid wall cells) misses
        # exactly the cy<0 directions
        assert {q for q in range(19) if mask[q, :, -1, 2:-2].any()} == {4, 8, 9, 16, 18}
        # z faces are periodic → nothing missing at their interior cells
        # (wall cells miss q=0; wall-adjacent cells miss the wall dirs)
        assert not mask[:, 0, 2:-2, 2:-2].any()
        assert not mask[:, -1, 2:-2, 2:-2].any()

    def test_export_registers_field_products(self, tmp_path):
        from tensorlbm.data.catalog import FieldDataCatalog

        catalog = FieldDataCatalog.open(tmp_path / "catalog.db")
        export = ExportSpec(
            h5_path=tmp_path / "run.h5",
            catalog=catalog,
            code_sha="a" * 40,
            snapshot_every=5,
        )
        # cavity keeps |<rho>-1| far inside the solver_export mass gate
        # (the poiseuille startup transient legitimately trips mass_tol).
        result = run_case("cavity", steps=10, resolution=12, re=100.0, export=export)
        # steps 5 (interval) and 10 (final) → two products
        assert len(result.product_ids) == 2
        found = catalog.find_assets_by_metadata("case", "cavity", kind="field_product")
        assert len(found) >= 2
        catalog.close()

    def test_export_requires_valid_code_sha(self, tmp_path):
        from tensorlbm.data.catalog import FieldDataCatalog

        catalog = FieldDataCatalog.open(tmp_path / "catalog.db")
        export = ExportSpec(
            h5_path=tmp_path / "run.h5", catalog=catalog, code_sha="not-a-sha"
        )
        with pytest.raises(ValueError, match="code_sha"):
            run_case("poiseuille", steps=1, resolution=5, re=50.0, export=export)
        catalog.close()
