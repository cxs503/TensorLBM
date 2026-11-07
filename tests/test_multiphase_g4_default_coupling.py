"""G4 ledger criteria: the classic psi_exp default SC coupling is exactly critical.

Ledger entry (``benchmarks/gaps.md``, G4)::

    psi_exp G=−4.0 恰临界（multiphase 默认）
    Shan-Chen 1994 伪势在 G=−4.0 无旋结线（恰为临界），无法相分离
    需 G=−5.0（密度比 35.5）或 G=−4.1（密度比 2）

Each test is a machine-checkable reading of that entry, in this library's
sign convention (``sc_single_component_force`` gathers the *backward*
neighbour, so ``G > 0`` is attractive and ``G_classic = −G``):

* C1 — closed-form critical coupling: for psi_exp (rho0=1) the EOS
  ``p = ρ/3 − (G/6)(1−e^{−ρ})²`` has ``∂p/∂ρ = 0`` iff
  ``(1−u)u = 1/G`` with ``u = e^{−ρ}``; the discriminant ``1 − 4/G`` vanishes
  at **G = 4.0 exactly** (classic −4.0), where the spinodal degenerates to the
  single point ``ρ = ln 2``.  The old default therefore cannot phase-separate.
* C2 — at the new default ``G = +5.0`` the spinodal band matches the closed
  form ``ρ ∈ [−ln u₊, −ln u₋]`` and the Maxwell equal-area construction
  closes (``∫(p−p_sat)dv = 0``).
* C3 — dynamics: the new default phase-separates (bimodal density, mass
  conserved) while the old default ``G = −4.0`` stays homogeneous — the
  honest-FAIL record of the pre-fix behaviour.
* C4 — the 3D (D3Q19) collision with the new default also separates.

Sign convention: see the CONVENTION NOTE in ``tensorlbm/multiphase.py``.
"""

from __future__ import annotations

import inspect
import math

import pytest
import torch

from tensorlbm import (
    Spinodal3DConfig,
    SpinodaleConfig,
    collide_sc_single_component,
    collide_sc_single_component_27,
    collide_sc_single_component_3d,
)
from tensorlbm.d2q9 import equilibrium
from tensorlbm.multiphase import psi_exp
from tensorlbm.sc_eos_coexistence import (
    maxwell_coexistence,
    mechanical_pressure,
    spinodal_bounds,
)

# --- Ledger numbers, transcribed verbatim from benchmarks/gaps.md (entry G4) ---
LEDGER_G_CLASSIC = -4.0  # "psi_exp G=−4.0 恰临界（multiphase 默认）"
LEDGER_G_CLASSIC_RATIO_35 = -5.0  # "需 G=−5.0（密度比 35.5）"
LEDGER_G_CLASSIC_RATIO_2 = -4.1  # "或 G=−4.1（密度比 2）"

# Library-convention equivalents (G_lib = −G_classic)
G_CRIT_LIB = 4.0
G_DEFAULT_LIB = 5.0


# ---------------------------------------------------------------------------
# C1 — closed-form critical coupling (analytic verification surface)
# ---------------------------------------------------------------------------


def _dp_drho(G: float, n: int = 20001) -> tuple[torch.Tensor, torch.Tensor]:
    """(ρ, ∂p/∂ρ) on a fine uniform grid for the psi_exp mechanical EOS."""
    rho = torch.linspace(1e-4, 4.0, n, dtype=torch.float64)
    p = mechanical_pressure(rho, G, psi_exp).to(torch.float64)
    dp = torch.gradient(p, spacing=(rho,))[0]
    return rho, dp


class TestCriticalCouplingClosedForm:
    def test_discriminant_zero_at_g4(self) -> None:
        """(1−u)u = 1/G has a double root iff G = 4: the exact critical point."""
        for G in (3.9, 4.0, 4.1, 5.0):
            disc = 1.0 - 4.0 / G
            if G < 4.0:
                assert disc < 0  # no real spinodal roots
            elif G == 4.0:
                assert disc == 0.0  # double root u = 1/2 → ρ = ln 2
            else:
                assert disc > 0

    def test_no_spinodal_at_critical(self) -> None:
        """At G = +4.0 (classic −4.0) the EOS is monotone: ∂p/∂ρ ≥ 0 up to grid
        resolution, with the minimum sitting at the degenerate spinodal ρ = ln 2."""
        rho, dp = _dp_drho(G_CRIT_LIB)
        assert float(dp.min()) > -1e-9  # monotone EOS (within grid resolution)
        rho_at_min = float(rho[int(torch.argmin(dp))])
        assert abs(rho_at_min - math.log(2.0)) < 0.02
        band = spinodal_bounds(G_CRIT_LIB, psi_exp, rho_lo=1e-3, rho_hi=8.0)
        if band is not None:
            r_v, r_l = band
            # whatever the scan resolves, it must collapse onto ρ = ln 2
            assert abs(0.5 * (r_v + r_l) - math.log(2.0)) < 0.02
            assert (r_l - r_v) / math.log(2.0) < 0.02

    def test_maxwell_g4_no_coexistence(self) -> None:
        """At the critical coupling the Maxwell construction has no solution:
        the vapour/liquid branch pressure ranges do not overlap."""
        res = maxwell_coexistence(G_CRIT_LIB, psi_exp, rho_lo=1e-4, rho_hi=8.0)
        assert not res.found
        assert "no pressure overlap" in res.reason or "no spinodal" in res.reason

    def test_spinodal_band_g5_matches_closed_form(self) -> None:
        """G = +5.0: spinodal band equals [−ln u₊, −ln u₋], u± = (1±√(1−4/G))/2."""
        G = G_DEFAULT_LIB
        u_plus = (1.0 + math.sqrt(1.0 - 4.0 / G)) / 2.0
        u_minus = (1.0 - math.sqrt(1.0 - 4.0 / G)) / 2.0
        rho_sp_v = -math.log(u_plus)  # vapour-side spinodal
        rho_sp_l = -math.log(u_minus)  # liquid-side spinodal
        band = spinodal_bounds(G, psi_exp, rho_lo=1e-3, rho_hi=8.0)
        assert band is not None
        r_v, r_l = band
        assert abs(r_v - rho_sp_v) / rho_sp_v < 0.02
        assert abs(r_l - rho_sp_l) / rho_sp_l < 0.02

    def test_maxwell_g5_closes(self) -> None:
        """Maxwell equal-area construction closes at G = +5.0 (new default)."""
        res = maxwell_coexistence(G_DEFAULT_LIB, psi_exp, rho_lo=1e-4, rho_hi=8.0)
        assert res.found, res.reason
        assert res.rho_liquid is not None and res.rho_vapour is not None
        assert res.rho_liquid > res.rho_vapour
        assert res.density_ratio is not None and res.density_ratio > 10.0
        # equal-area rule: residual of ∫(p−p_sat)dv over the loop
        assert res.area_residual is not None
        assert abs(res.area_residual) < 1e-3

    def test_maxwell_g41_near_critical_small_ratio(self) -> None:
        """Ledger's 'G=−4.1 → 密度比 2' (classic): just above critical, ratio ≈ 2."""
        res = maxwell_coexistence(4.1, psi_exp, rho_lo=1e-4, rho_hi=8.0)
        assert res.found, res.reason
        assert res.density_ratio is not None
        assert 1.5 < res.density_ratio < 3.0  # ledger: 密度比 2


# ---------------------------------------------------------------------------
# C2 — default values carry the attractive (library) sign
# ---------------------------------------------------------------------------


class TestDefaultCouplingSign:
    def test_2d_default_g_attractive(self) -> None:
        g = inspect.signature(collide_sc_single_component).parameters["G"].default
        assert g == G_DEFAULT_LIB
        assert g > 0  # library convention: G > 0 = attractive

    def test_3d_default_g_attractive(self) -> None:
        g = inspect.signature(collide_sc_single_component_3d).parameters["G"].default
        assert g == G_DEFAULT_LIB and g > 0

    def test_d3q27_default_g_attractive(self) -> None:
        g = inspect.signature(collide_sc_single_component_27).parameters["G"].default
        assert g == G_DEFAULT_LIB and g > 0

    def test_spinodal_configs_default_g_attractive(self) -> None:
        assert SpinodaleConfig().G == G_DEFAULT_LIB
        assert Spinodal3DConfig().G == G_DEFAULT_LIB


# ---------------------------------------------------------------------------
# C3 — dynamics: new default separates, old default does not (honest FAIL)
# ---------------------------------------------------------------------------


def _run_2d(G: float, n: int = 32, steps: int = 500, seed: int = 0) -> dict[str, float]:
    from tensorlbm.solver import stream

    torch.manual_seed(seed)
    rho = 0.8 + 0.4 * (torch.rand((n, n)) - 0.5)
    rho = rho.clamp(min=1e-3)
    zero = torch.zeros_like(rho)
    f = equilibrium(rho, zero, zero)
    mass0 = float(rho.sum())
    for _ in range(steps):
        f = collide_sc_single_component(f, G=G, tau=1.0, psi_fn=psi_exp)
        f = stream(f)
    rho_f = f.sum(0)
    assert torch.isfinite(rho_f).all()
    return {
        "rho_max": float(rho_f.max()),
        "rho_min": float(rho_f.min()),
        "ratio": float(rho_f.max() / max(rho_f.min(), 1e-9)),
        "mass_drift": abs(float(rho_f.sum()) - mass0) / mass0,
    }


class TestDynamics:
    def test_new_default_phase_separates(self) -> None:
        """C3 positive control: the new default G = +5.0 separates into two phases."""
        out = _run_2d(G_DEFAULT_LIB)
        assert out["ratio"] > 2.0  # bimodal: liquid + vapour peaks
        assert out["rho_max"] > 1.2 and out["rho_min"] < 0.5

    def test_old_default_g_no_separation(self) -> None:
        """C3 negative control: the old default G = −4.0 (repulsive in the
        library convention) never separates — honest-FAIL record of the
        pre-fix behaviour."""
        out = _run_2d(LEDGER_G_CLASSIC)
        assert out["ratio"] < 2.0  # homogeneous: no phase separation
        assert out["mass_drift"] < 1e-4

    def test_mass_conservation_during_separation(self) -> None:
        """Conservation law: total mass is conserved while the field separates."""
        out = _run_2d(G_DEFAULT_LIB)
        assert out["mass_drift"] < 1e-4


# ---------------------------------------------------------------------------
# C4 — 3D (D3Q19) with the new default
# ---------------------------------------------------------------------------


class Test3D:
    @pytest.mark.slow
    def test_3d_default_g_phase_separates(self) -> None:
        """C4: D3Q19 SCMP with the new default separates; old default does not."""
        from tensorlbm.d3q19 import equilibrium3d
        from tensorlbm.solver3d import stream3d

        def run3d(G: float, n: int = 12, steps: int = 300) -> float:
            torch.manual_seed(0)
            rho = 0.8 + 0.4 * (torch.rand((n, n, n)) - 0.5)
            rho = rho.clamp(min=1e-3)
            z = torch.zeros_like(rho)
            f = equilibrium3d(rho, z, z, z)
            for _ in range(steps):
                f = collide_sc_single_component_3d(f, G=G, tau=1.0)
                f = stream3d(f)
            rho_f = f.sum(0)
            assert torch.isfinite(rho_f).all()
            return float(rho_f.max() / max(rho_f.min(), 1e-9))

        assert run3d(G_DEFAULT_LIB) > 2.0
        assert run3d(LEDGER_G_CLASSIC) < 2.0
