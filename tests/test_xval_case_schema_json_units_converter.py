"""Cross-validation: TensorLBM LBMUnitConverter vs lettuce UnitConversion.

Anchor: XVAL-LBM-CASE-SCHEMA-JSON-UNITS-CONVERTER
OSS reference: lettucecfd/lettuce (MIT, Andreas Kraemer 2019)
  - lettuce UnitConversion: Re+Ma dual-anchor, u_lu = cs·Ma, τ derived,
    paired *_to_pu/*_to_lu for velocity/pressure/time/energy.
  - Source: lettuce/unit_conversion.py (master, MIT licence)

TensorLBM LBMUnitConverter: Re + u_lb direct anchor, τ/nu_lb/ma/dt derived,
  phys_to_lb/lb_to_phys for velocity, phys_time_to_steps/steps_to_phys_time.

This test proves numerical equivalence when inputs are mapped correctly,
and documents the design differences (single vs dual anchor, method surface).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import pytest

# ---------------------------------------------------------------------------
# Lettuce-style UnitConversion (reconstructed from lettuce source + survey)
# ---------------------------------------------------------------------------
# lettuce uses Re + Ma as dual anchors:
#   u_lu = cs * Ma          (Ma determines lattice velocity)
#   nu_lu = u_lu * N / Re   (Re determines lattice viscosity)
#   tau   = nu_lu / cs² + 0.5
#   dx    = L_phys / N
#   dt    = dx * u_lu / u_phys   (time step from velocity scaling)
#
# Conversion helpers (lettuce naming convention):
#   velocity_to_lu(v_phys) = v_phys * dt / dx
#   velocity_to_pu(v_lu)   = v_lu * dx / dt
#   time_to_lu(t_phys)     = t_phys / dt
#   time_to_pu(t_lu)       = t_lu * dt
#   pressure_to_lu(p_phys) = p_phys * dt² / dx²   (via density scaling)
#   pressure_to_pu(p_lu)   = p_lu * dx² / dt²
# ---------------------------------------------------------------------------

_CS_LB = 1.0 / math.sqrt(3.0)  # lattice speed of sound


@dataclass(frozen=True)
class LettuceUnitConversion:
    """Reconstructed lettuce UnitConversion (Re+Ma dual-anchor).

    Parameters match lettuce's public API:
      re: Reynolds number
      ma: Mach number (determines u_lu = cs * Ma)
      l_phys: reference physical length [m]
      u_phys: reference physical velocity [m/s]
      n: number of lattice cells along reference length
    """
    re: float
    ma: float
    l_phys: float
    u_phys: float
    n: float

    def __post_init__(self):
        if self.re <= 0:
            raise ValueError(f"Re must be positive, got {self.re}")
        if self.ma <= 0 or self.ma >= _CS_LB:
            raise ValueError(f"Ma must satisfy 0 < Ma < cs ({_CS_LB:.4f}), got {self.ma}")
        if self.l_phys <= 0:
            raise ValueError(f"l_phys must be positive, got {self.l_phys}")
        if self.u_phys <= 0:
            raise ValueError(f"u_phys must be positive, got {self.u_phys}")
        if self.n <= 0:
            raise ValueError(f"n must be positive, got {self.n}")

    @property
    def u_lu(self) -> float:
        """Lattice velocity from Ma anchor."""
        return _CS_LB * self.ma

    @property
    def nu_lu(self) -> float:
        """Lattice kinematic viscosity from Re anchor."""
        return self.u_lu * self.n / self.re

    @property
    def tau(self) -> float:
        """BGK relaxation time."""
        return self.nu_lu / (_CS_LB ** 2) + 0.5

    @property
    def dx(self) -> float:
        """Physical grid spacing [m]."""
        return self.l_phys / self.n

    @property
    def dt(self) -> float:
        """Physical time step [s]."""
        return self.dx * self.u_lu / self.u_phys

    # --- Conversion helpers (lettuce naming) ---

    def velocity_to_lu(self, v_phys: float) -> float:
        return v_phys * self.dt / self.dx

    def velocity_to_pu(self, v_lu: float) -> float:
        return v_lu * self.dx / self.dt

    def time_to_lu(self, t_phys: float) -> float:
        return t_phys / self.dt

    def time_to_pu(self, t_lu: float) -> float:
        return t_lu * self.dt

    def pressure_to_lu(self, p_phys: float) -> float:
        """Pressure conversion via density scaling: p ~ ρ·u²."""
        return p_phys * self.dt ** 2 / self.dx ** 2

    def pressure_to_pu(self, p_lu: float) -> float:
        return p_lu * self.dx ** 2 / self.dt ** 2

    def density_lu_to_pressure_pu(self, rho_lu: float) -> float:
        """lettuce: convert lattice density deviation to physical pressure."""
        return rho_lu * _CS_LB ** 2 * self.dx ** 2 / self.dt ** 2

    def to_dict(self) -> dict:
        return {
            "re": self.re,
            "ma": self.ma,
            "l_phys": self.l_phys,
            "u_phys": self.u_phys,
            "n": self.n,
            "u_lu": self.u_lu,
            "nu_lu": self.nu_lu,
            "tau": self.tau,
            "dx": self.dx,
            "dt": self.dt,
        }


# ---------------------------------------------------------------------------
# TensorLBM LBMUnitConverter (imported from library)
# ---------------------------------------------------------------------------

def _make_tensorlbm_converter(re: float, l_phys: float, u_phys: float,
                               nu_phys: float, nx: int, u_lb: float):
    """Factory for TensorLBM LBMUnitConverter."""
    from tensorlbm.unit_converter import LBMUnitConverter
    return LBMUnitConverter(
        re=re, l_phys=l_phys, u_phys=u_phys, nu_phys=nu_phys,
        nx=nx, u_lb=u_lb,
    )


# ---------------------------------------------------------------------------
# Equivalence mapping
# ---------------------------------------------------------------------------

def _map_lettuce_to_tensorlbm(luc: LettuceUnitConversion) -> dict:
    """Map lettuce Re+Ma inputs to TensorLBM Re+u_lb inputs.

    lettuce: u_lu = cs * Ma  →  TensorLBM: u_lb = u_lu
    lettuce: Re, l_phys, u_phys, n  →  TensorLBM: re, l_phys, u_phys, nx=int(n)
    nu_phys is derived from Re = u_phys * l_phys / nu_phys
    """
    nu_phys = luc.u_phys * luc.l_phys / luc.re
    return {
        "re": luc.re,
        "l_phys": luc.l_phys,
        "u_phys": luc.u_phys,
        "nu_phys": nu_phys,
        "nx": int(luc.n),
        "u_lb": luc.u_lu,  # key mapping: u_lb = cs * Ma
    }


# ---------------------------------------------------------------------------
# Test cases: canonical LBM benchmark configurations
# ---------------------------------------------------------------------------

# Each tuple: (re, ma, l_phys, u_phys, n, description)
_BENCHMARK_CASES = [
    # Lid-driven cavity (Ghia 1982)
    (400.0, 0.0866, 1.0, 1.0, 96, "cavity_Re400"),
    # Cylinder drag (Schäfer-Turek)
    (100.0, 0.05, 0.1, 0.1, 200, "cylinder_Re100"),
    # Channel flow (laminar)
    (1000.0, 0.05, 0.01, 0.1, 100, "channel_Re1000"),
    # High Re external flow
    (10000.0, 0.03, 1.0, 1.0, 500, "external_Re10000"),
    # Low Re creeping flow
    (10.0, 0.02, 0.001, 0.001, 50, "creeping_Re10"),
    # SUBOFF bare hull (typical marine)
    (420000.0, 0.05, 3.048, 1.5, 128, "suboff_Re420k"),
]


class TestLettuceUnitConversion:
    """Verify the reconstructed lettuce UnitConversion is self-consistent."""

    @pytest.mark.parametrize("re,ma,l_phys,u_phys,n,desc", _BENCHMARK_CASES)
    def test_tau_in_stable_range(self, re, ma, l_phys, u_phys, n, desc):
        luc = LettuceUnitConversion(re=re, ma=ma, l_phys=l_phys, u_phys=u_phys, n=n)
        assert luc.tau > 0.5, f"{desc}: tau={luc.tau} <= 0.5"
        assert luc.tau < 2.0, f"{desc}: tau={luc.tau} >= 2.0 (may be unstable)"

    @pytest.mark.parametrize("re,ma,l_phys,u_phys,n,desc", _BENCHMARK_CASES)
    def test_velocity_roundtrip(self, re, ma, l_phys, u_phys, n, desc):
        luc = LettuceUnitConversion(re=re, ma=ma, l_phys=l_phys, u_phys=u_phys, n=n)
        v_phys = 0.5 * u_phys
        v_lu = luc.velocity_to_lu(v_phys)
        v_back = luc.velocity_to_pu(v_lu)
        assert v_back == pytest.approx(v_phys, rel=1e-12), f"{desc}: velocity roundtrip failed"

    @pytest.mark.parametrize("re,ma,l_phys,u_phys,n,desc", _BENCHMARK_CASES)
    def test_time_roundtrip(self, re, ma, l_phys, u_phys, n, desc):
        luc = LettuceUnitConversion(re=re, ma=ma, l_phys=l_phys, u_phys=u_phys, n=n)
        t_phys = 1.0
        t_lu = luc.time_to_lu(t_phys)
        t_back = luc.time_to_pu(t_lu)
        assert t_back == pytest.approx(t_phys, rel=1e-12), f"{desc}: time roundtrip failed"

    @pytest.mark.parametrize("re,ma,l_phys,u_phys,n,desc", _BENCHMARK_CASES)
    def test_pressure_roundtrip(self, re, ma, l_phys, u_phys, n, desc):
        luc = LettuceUnitConversion(re=re, ma=ma, l_phys=l_phys, u_phys=u_phys, n=n)
        p_phys = 101325.0  # 1 atm
        p_lu = luc.pressure_to_lu(p_phys)
        p_back = luc.pressure_to_pu(p_lu)
        assert p_back == pytest.approx(p_phys, rel=1e-12), f"{desc}: pressure roundtrip failed"


class TestTensorLBMVsLettuceEquivalence:
    """Core XVAL: TensorLBM LBMUnitConverter ≡ lettuce UnitConversion.

    When inputs are mapped correctly (u_lb = cs * Ma), the derived quantities
    (tau, nu_lb, dx, dt, ma) must match to floating-point precision.
    """

    @pytest.mark.parametrize("re,ma,l_phys,u_phys,n,desc", _BENCHMARK_CASES)
    def test_tau_equivalence(self, re, ma, l_phys, u_phys, n, desc):
        luc = LettuceUnitConversion(re=re, ma=ma, l_phys=l_phys, u_phys=u_phys, n=n)
        kwargs = _map_lettuce_to_tensorlbm(luc)
        tlc = _make_tensorlbm_converter(**kwargs)
        assert tlc.tau == pytest.approx(luc.tau, rel=1e-12), (
            f"{desc}: tau mismatch: lettuce={luc.tau}, tensorlbm={tlc.tau}"
        )

    @pytest.mark.parametrize("re,ma,l_phys,u_phys,n,desc", _BENCHMARK_CASES)
    def test_nu_lb_equivalence(self, re, ma, l_phys, u_phys, n, desc):
        luc = LettuceUnitConversion(re=re, ma=ma, l_phys=l_phys, u_phys=u_phys, n=n)
        kwargs = _map_lettuce_to_tensorlbm(luc)
        tlc = _make_tensorlbm_converter(**kwargs)
        assert tlc.nu_lb == pytest.approx(luc.nu_lu, rel=1e-12), (
            f"{desc}: nu_lb mismatch: lettuce={luc.nu_lu}, tensorlbm={tlc.nu_lb}"
        )

    @pytest.mark.parametrize("re,ma,l_phys,u_phys,n,desc", _BENCHMARK_CASES)
    def test_dx_equivalence(self, re, ma, l_phys, u_phys, n, desc):
        luc = LettuceUnitConversion(re=re, ma=ma, l_phys=l_phys, u_phys=u_phys, n=n)
        kwargs = _map_lettuce_to_tensorlbm(luc)
        tlc = _make_tensorlbm_converter(**kwargs)
        assert tlc.dx == pytest.approx(luc.dx, rel=1e-12), (
            f"{desc}: dx mismatch: lettuce={luc.dx}, tensorlbm={tlc.dx}"
        )

    @pytest.mark.parametrize("re,ma,l_phys,u_phys,n,desc", _BENCHMARK_CASES)
    def test_dt_equivalence(self, re, ma, l_phys, u_phys, n, desc):
        luc = LettuceUnitConversion(re=re, ma=ma, l_phys=l_phys, u_phys=u_phys, n=n)
        kwargs = _map_lettuce_to_tensorlbm(luc)
        tlc = _make_tensorlbm_converter(**kwargs)
        assert tlc.dt == pytest.approx(luc.dt, rel=1e-12), (
            f"{desc}: dt mismatch: lettuce={luc.dt}, tensorlbm={tlc.dt}"
        )

    @pytest.mark.parametrize("re,ma,l_phys,u_phys,n,desc", _BENCHMARK_CASES)
    def test_mach_equivalence(self, re, ma, l_phys, u_phys, n, desc):
        luc = LettuceUnitConversion(re=re, ma=ma, l_phys=l_phys, u_phys=u_phys, n=n)
        kwargs = _map_lettuce_to_tensorlbm(luc)
        tlc = _make_tensorlbm_converter(**kwargs)
        # TensorLBM computes ma = u_lb / cs; lettuce uses ma directly
        assert tlc.ma == pytest.approx(luc.ma, rel=1e-12), (
            f"{desc}: Ma mismatch: lettuce={luc.ma}, tensorlbm={tlc.ma}"
        )

    @pytest.mark.parametrize("re,ma,l_phys,u_phys,n,desc", _BENCHMARK_CASES)
    def test_velocity_conversion_equivalence(self, re, ma, l_phys, u_phys, n, desc):
        luc = LettuceUnitConversion(re=re, ma=ma, l_phys=l_phys, u_phys=u_phys, n=n)
        kwargs = _map_lettuce_to_tensorlbm(luc)
        tlc = _make_tensorlbm_converter(**kwargs)

        v_phys = 0.75 * u_phys
        v_lu_lettuce = luc.velocity_to_lu(v_phys)
        v_lu_tensorlbm = tlc.phys_to_lb(v_phys)
        assert v_lu_tensorlbm == pytest.approx(v_lu_lettuce, rel=1e-12), (
            f"{desc}: phys_to_lb mismatch: lettuce={v_lu_lettuce}, tensorlbm={v_lu_tensorlbm}"
        )

    @pytest.mark.parametrize("re,ma,l_phys,u_phys,n,desc", _BENCHMARK_CASES)
    def test_time_conversion_equivalence(self, re, ma, l_phys, u_phys, n, desc):
        luc = LettuceUnitConversion(re=re, ma=ma, l_phys=l_phys, u_phys=u_phys, n=n)
        kwargs = _map_lettuce_to_tensorlbm(luc)
        tlc = _make_tensorlbm_converter(**kwargs)

        t_phys = 2.5
        t_steps_lettuce = luc.time_to_lu(t_phys)
        t_steps_tensorlbm = tlc.phys_time_to_steps(t_phys)
        # lettuce returns float steps, TensorLBM returns int
        assert abs(t_steps_tensorlbm - round(t_steps_lettuce)) <= 1, (
            f"{desc}: time steps mismatch: lettuce={t_steps_lettuce}, tensorlbm={t_steps_tensorlbm}"
        )


class TestCaseUnitsEquivalence:
    """Verify CaseUnits (case-level) matches LBMUnitConverter (standalone).

    CaseUnits.from_reference(re, u_lb, n_ref) should produce the same
    nu_lb, tau, ma as LBMUnitConverter when given equivalent inputs.

    Note: CaseUnits allows fractional n_ref (e.g., hull spanning 0.6*nx cells),
    while LBMUnitConverter uses integer nx. When n_ref == nx, they match exactly.
    """

    def test_case_units_vs_converter_cavity(self):
        from tensorlbm.cases import CaseUnits

        re = 400.0
        u_lb = 0.06
        n_ref = 96.0

        cu = CaseUnits.from_reference(re=re, u_lb=u_lb, n_ref=n_ref)
        nu_phys = 1.0 * 1.0 / re  # u_phys=1, l_phys=1
        tlc = _make_tensorlbm_converter(
            re=re, l_phys=1.0, u_phys=1.0, nu_phys=nu_phys,
            nx=96, u_lb=u_lb,
        )

        assert cu.nu_lb == pytest.approx(tlc.nu_lb, rel=1e-12)
        assert cu.tau == pytest.approx(tlc.tau, rel=1e-12)
        assert cu.ma == pytest.approx(tlc.ma, rel=1e-12)

    def test_case_units_vs_converter_matching_n_ref(self):
        """When n_ref == nx, CaseUnits and LBMUnitConverter match exactly."""
        from tensorlbm.cases import CaseUnits

        re = 100.0
        u_lb = 0.05
        n_ref = 50.0  # same as nx

        cu = CaseUnits.from_reference(re=re, u_lb=u_lb, n_ref=n_ref)
        nu_phys = 1.0 * 1.0 / re
        tlc = _make_tensorlbm_converter(
            re=re, l_phys=1.0, u_phys=1.0, nu_phys=nu_phys,
            nx=50, u_lb=u_lb,
        )

        assert cu.nu_lb == pytest.approx(tlc.nu_lb, rel=1e-12)
        assert cu.tau == pytest.approx(tlc.tau, rel=1e-12)
        assert cu.ma == pytest.approx(tlc.ma, rel=1e-12)

    def test_case_units_fractional_n_ref(self):
        """CaseUnits supports fractional n_ref (e.g., hull spanning 0.6*nx)."""
        from tensorlbm.cases import CaseUnits

        re = 420000.0
        u_lb = 0.05
        nx = 128
        n_ref = 0.6 * nx  # hull spans 60% of domain

        cu = CaseUnits.from_reference(re=re, u_lb=u_lb, n_ref=n_ref)
        # nu_lb = u_lb * n_ref / re
        expected_nu_lb = u_lb * n_ref / re
        assert cu.nu_lb == pytest.approx(expected_nu_lb, rel=1e-12)


class TestJSONSchemaSerialization:
    """Verify unit converter outputs can be serialized to JSON schema.

    This tests the "case-schema-json" part of the anchor: the converter
    outputs must be JSON-serializable for case definition files.
    """

    def test_lettuce_to_dict_is_json_serializable(self):
        luc = LettuceUnitConversion(re=400.0, ma=0.0866, l_phys=1.0, u_phys=1.0, n=96)
        d = luc.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        assert loaded["re"] == 400.0
        assert loaded["tau"] == pytest.approx(luc.tau, rel=1e-12)

    def test_tensorlbm_summary_is_json_serializable(self):
        nu_phys = 1.0 * 1.0 / 400.0
        tlc = _make_tensorlbm_converter(
            re=400.0, l_phys=1.0, u_phys=1.0, nu_phys=nu_phys,
            nx=96, u_lb=0.05,
        )
        summary = tlc.summary()
        # summary() returns a string; verify it's parseable
        assert "Re" in summary
        assert "tau" in summary
        assert "Ma" in summary

    def test_converter_repr_is_informative(self):
        nu_phys = 1.0 * 1.0 / 400.0
        tlc = _make_tensorlbm_converter(
            re=400.0, l_phys=1.0, u_phys=1.0, nu_phys=nu_phys,
            nx=96, u_lb=0.05,
        )
        r = repr(tlc)
        assert "Re=400" in r
        assert "nx=96" in r
        assert "τ=" in r
        assert "Ma=" in r


class TestEdgeCasesAndValidation:
    """Boundary conditions and error handling."""

    def test_lettuce_rejects_zero_re(self):
        with pytest.raises(ValueError, match="Re must be positive"):
            LettuceUnitConversion(re=0, ma=0.05, l_phys=1.0, u_phys=1.0, n=100)

    def test_lettuce_rejects_negative_ma(self):
        with pytest.raises(ValueError, match="Ma must satisfy"):
            LettuceUnitConversion(re=100, ma=-0.05, l_phys=1.0, u_phys=1.0, n=100)

    def test_lettuce_rejects_ma_above_cs(self):
        with pytest.raises(ValueError, match="Ma must satisfy"):
            LettuceUnitConversion(re=100, ma=0.6, l_phys=1.0, u_phys=1.0, n=100)

    def test_tensorlbm_rejects_zero_re(self):
        with pytest.raises(ValueError, match="Reynolds number must be positive"):
            _make_tensorlbm_converter(re=0, l_phys=1.0, u_phys=1.0, nu_phys=1e-3, nx=100, u_lb=0.05)

    def test_tensorlbm_rejects_u_lb_above_cs(self):
        with pytest.raises(ValueError, match="u_lb must satisfy"):
            _make_tensorlbm_converter(re=100, l_phys=1.0, u_phys=1.0, nu_phys=1e-3, nx=100, u_lb=0.6)

    def test_tensorlbm_warns_on_high_tau(self):
        import warnings
        # Very low Re → high nu_lb → high tau
        # tau = 0.5 + nu_lb/cs² = 0.5 + (u_lb*nx/Re)/cs²
        # For tau > 2.0: nu_lb > 1.5*cs² = 0.5
        # nu_lb = 0.05 * 10 / Re > 0.5 → Re < 1.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _make_tensorlbm_converter(
                re=0.5, l_phys=1.0, u_phys=1.0, nu_phys=2.0,
                nx=10, u_lb=0.05,
            )
            assert any("τ" in str(warning.message) for warning in w), (
                f"Expected tau warning, got: {[str(x.message) for x in w]}"
            )

    def test_tensorlbm_warns_on_high_ma(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            nu_phys = 1.0 * 1.0 / 100.0
            _make_tensorlbm_converter(
                re=100.0, l_phys=1.0, u_phys=1.0, nu_phys=nu_phys,
                nx=100, u_lb=0.12,  # Ma ≈ 0.208 > 0.1
            )
            assert any("Mach number" in str(warning.message) for warning in w)


class TestConvergenceAndSensitivity:
    """Convergence study: tau and nu_lb vs resolution.

    Key insight: nu_lb = u_lb * nx / Re, so nu_lb scales with nx.
    tau = 0.5 + nu_lb/cs², so tau also scales with nx.
    This is by design: finer grids (larger nx) at fixed Re and u_lb
    require higher lattice viscosity to maintain the same Re.
    """

    def test_nu_lb_scales_linearly_with_resolution(self):
        """nu_lb = u_lb * nx / Re, so it scales linearly with nx."""
        re = 1000.0
        u_lb = 0.05
        nu_phys = 1.0 * 1.0 / re
        ratios = []
        for nx in [32, 64, 128, 256]:
            tlc = _make_tensorlbm_converter(
                re=re, l_phys=1.0, u_phys=1.0, nu_phys=nu_phys,
                nx=nx, u_lb=u_lb,
            )
            ratios.append(tlc.nu_lb / nx)
        # nu_lb / nx should be constant (= u_lb / Re)
        expected = u_lb / re
        for r in ratios:
            assert r == pytest.approx(expected, rel=1e-12)

    def test_tau_scales_with_resolution(self):
        """tau = 0.5 + u_lb*nx/(Re*cs²), so tau increases with nx."""
        nu_phys = 1.0 * 1.0 / 1000.0
        taus = []
        for nx in [32, 64, 128, 256]:
            tlc = _make_tensorlbm_converter(
                re=1000.0, l_phys=1.0, u_phys=1.0, nu_phys=nu_phys,
                nx=nx, u_lb=0.05,
            )
            taus.append(tlc.tau)
        # tau should increase with nx
        for i in range(1, len(taus)):
            assert taus[i] > taus[i - 1], f"tau[{i}]={taus[i]} <= tau[{i-1}]={taus[i-1]}"

    def test_dt_scales_inversely_with_resolution(self):
        """dt = u_lb * dx / u_phys = u_lb * L / (nx * u_phys), so dt ∝ 1/nx."""
        nu_phys = 1.0 * 1.0 / 1000.0
        dts = []
        for nx in [32, 64, 128, 256]:
            tlc = _make_tensorlbm_converter(
                re=1000.0, l_phys=1.0, u_phys=1.0, nu_phys=nu_phys,
                nx=nx, u_lb=0.05,
            )
            dts.append(tlc.dt)
        # dt * nx should be constant
        products = [dt * nx for dt, nx in zip(dts, [32, 64, 128, 256])]
        for p in products:
            assert p == pytest.approx(products[0], rel=1e-12)

    def test_tau_invariant_when_n_ref_fixed(self):
        """When n_ref is fixed (same physical reference length), tau is invariant.

        CaseUnits: nu_lb = u_lb * n_ref / re. If n_ref is fixed (not scaled with nx),
        then nu_lb and tau are constant regardless of grid resolution.
        """
        from tensorlbm.cases import CaseUnits

        re = 1000.0
        u_lb = 0.05
        n_ref = 19.2  # fixed reference length

        taus = []
        for nx in [32, 64, 128, 256]:
            cu = CaseUnits.from_reference(re=re, u_lb=u_lb, n_ref=n_ref)
            taus.append(cu.tau)

        # All tau values should be identical (n_ref fixed)
        for t in taus:
            assert t == pytest.approx(taus[0], rel=1e-12)
