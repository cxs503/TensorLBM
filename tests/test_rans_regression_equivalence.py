"""RANS regression equivalence verification.

Verifies that the RANS common-module extraction (commit 2341767) preserves
correct behaviour while fixing known hot-path bugs.  The key principle:
**regression verification must identify whether the original implementation
itself had bugs** — original bugs are NOT treated as the "correct baseline".

Original implementation: src/tensorlbm/rans_ke.py @ 2341767^ (pre-extraction)
Common module:           src/tensorlbm/rans_common.py @ HEAD

------------------------------------------------------------------------------
Bug inventory (original rans_ke.py @ 2341767^)
------------------------------------------------------------------------------

BUG-1: collide_rans_sa — scalar averaging of per-cell nu_t  [CORRECTNESS]
  Line 821:  nu_eff = nu_lam + nu_t.mean().item()
  Line 822:  tau_eff = min(max(3.0 * nu_eff + 0.5, 0.501), 2.0)
  Line 823:  return collide_smagorinsky_mrt3d(f, tau=tau_eff, C_s=0.0)
  → Averages the per-cell eddy viscosity to a single scalar, losing ALL
    spatial variation.  With C_s=0.0, collide_smagorinsky_mrt3d degenerates
    to plain MRT with a uniform scalar tau — the turbulence model never
    engages spatially.  Also forces a GPU→CPU sync (.item()).
  Fix: common module keeps nu_t as a per-cell field throughout.

BUG-2: collide_rans_ke — per-call mask.bool() allocation  [PERFORMANCE]
  Line 394:  mask_3d = mask.bool()
  → Allocates a new bool tensor every collision step.  Not a correctness
    bug (result is identical), but a hot-path violation.
  Fix: common module requires pre-computed bool mask from caller.

BUG-3: collide_rans_ke — tau_eff clamp range differs  [MINOR]
  Line 410:  tau_eff = (3.0 * (nu_lam + nu_t) + 0.5).clamp(0.501, 3.0)
  → Clamps to [0.501, 3.0].  Common module uses _nu_t_to_tau_eff which
    clamps only min=0.5001 (no max).  For typical nu_t ∈ [0, 0.5] and
    tau=0.7, tau_eff ≤ 2.2 < 3.0, so the max clamp never triggers.
    The min clamp differs by 0.0009 — negligible for reasonable inputs.

------------------------------------------------------------------------------
Equivalence strategy
------------------------------------------------------------------------------

k-epsilon (D3Q19 MRT):
  The original collide_rans_ke MRT collision logic (lines 405–425) is
  algebraically identical to collide_rans_mrt3d in rans_common.py:
    - Same M, M_inv matrices (_get_d3q19_mrt_matrices)
    - Same s_fixed vector [0, s_e, s_eps, 0, s_q, 0, s_q, 0, s_q,
                           0,0,0,0,0, s_pi, s_pi, 1,1,1]
    - Same stress-mode override (modes 9–13) with per-cell 1/tau_eff
    - tau_eff = tau + 3*nu_t  (algebraically: 3*(nu_lam+nu_t)+0.5 = tau+3*nu_t)
  → With nu_t values that don't trigger the max clamp, outputs are allclose.

SA (D3Q19 MRT):
  The original collide_rans_sa has BUG-1 (scalar averaging).  We verify
  that the FIXED behaviour (per-cell nu_t → common collide_rans_mrt3d)
  DIFFERS from the buggy original, and that the fixed version matches
  the common module exactly.

k-omega SST (3D):
  The original KOmegaSSTSolver.step() only accepted (ux, uy) — 2D strain.
  The current version accepts (ux, uy, uz) — full 3D strain.
  collide_rans_komega_sst (3D) is entirely NEW — no original to compare.
  → Verify physical reasonableness (finite, mass conservation, per-cell nu_t).
"""
from __future__ import annotations

import inspect

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.d3q27 import equilibrium27, macroscopic27
from tensorlbm.rans_common import (
    collide_rans_3d,
    collide_rans_bgk27,
    collide_rans_bgk3d,
    collide_rans_mrt27,
    collide_rans_mrt3d,
)
from tensorlbm.solver3d import stream3d
from tensorlbm.turbulence import _nu_t_to_tau_eff

TAU = 0.7
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f3d19(nz=4, ny=6, nx=8, u_mag=0.04, seed=42):
    """Deterministic D3Q19 equilibrium distribution."""
    g = torch.Generator().manual_seed(seed)
    rho = torch.rand((nz, ny, nx), generator=g) + 0.5
    ux = torch.rand((nz, ny, nx), generator=g) * u_mag
    uy = torch.rand((nz, ny, nx), generator=g) * u_mag
    uz = torch.rand((nz, ny, nx), generator=g) * u_mag
    return equilibrium3d(rho, ux, uy, uz)


def _f3d27(nz=4, ny=6, nx=8, u_mag=0.04, seed=42):
    """Deterministic D3Q27 equilibrium distribution."""
    g = torch.Generator().manual_seed(seed)
    rho = torch.rand((nz, ny, nx), generator=g) + 0.5
    ux = torch.rand((nz, ny, nx), generator=g) * u_mag
    uy = torch.rand((nz, ny, nx), generator=g) * u_mag
    uz = torch.rand((nz, ny, nx), generator=g) * u_mag
    return equilibrium27(rho, ux, uy, uz)


def _nu_t_field_3d(nz=4, ny=6, nx=8, seed=99):
    """Deterministic non-trivial per-cell eddy viscosity field."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand((nz, ny, nx), generator=g) * 0.05


def _non_equilibrium_f3d19(nz=4, ny=6, nx=8, seed=42):
    """Non-equilibrium D3Q19 distribution (f != feq) for meaningful collision."""
    g = torch.Generator().manual_seed(seed)
    rho = torch.rand((nz, ny, nx), generator=g) + 0.5
    ux = torch.rand((nz, ny, nx), generator=g) * 0.04
    uy = torch.rand((nz, ny, nx), generator=g) * 0.04
    uz = torch.rand((nz, ny, nx), generator=g) * 0.04
    feq = equilibrium3d(rho, ux, uy, uz)
    # Add small non-equilibrium perturbation
    pert = torch.randn(19, nz, ny, nx, generator=g) * 1e-3
    return feq + pert


def _non_equilibrium_f3d27(nz=4, ny=6, nx=8, seed=42):
    """Non-equilibrium D3Q27 distribution."""
    g = torch.Generator().manual_seed(seed)
    rho = torch.rand((nz, ny, nx), generator=g) + 0.5
    ux = torch.rand((nz, ny, nx), generator=g) * 0.04
    uy = torch.rand((nz, ny, nx), generator=g) * 0.04
    uz = torch.rand((nz, ny, nx), generator=g) * 0.04
    feq = equilibrium27(rho, ux, uy, uz)
    pert = torch.randn(27, nz, ny, nx, generator=g) * 1e-3
    return feq + pert


# ===========================================================================
# Part 1: Original bug identification (source-level)
# ===========================================================================

class TestOriginalBugIdentification:
    """Document and verify the known bugs in the original implementation.

    These tests load the ORIGINAL source (pre-extraction @ 2341767^) via
    git show and assert the buggy patterns are present.  They serve as
    living documentation that the bugs existed and were fixed.
    """

    @pytest.fixture
    def original_rans_ke_source(self):
        """Load the original rans_ke.py source from git history."""
        import subprocess
        result = subprocess.run(
            ["git", "show", "2341767^:src/tensorlbm/rans_ke.py"],
            capture_output=True, text=True, check=True,
            cwd=str(__import__("pathlib").Path(__file__).resolve().parent.parent),
        )
        return result.stdout

    def test_bug1_sa_scalar_averaging_exists(self, original_rans_ke_source):
        """BUG-1: collide_rans_sa averaged nu_t to a scalar via .mean().item()."""
        assert "nu_t.mean().item()" in original_rans_ke_source, (
            "Expected nu_t.mean().item() in original collide_rans_sa"
        )

    def test_bug1_sa_delegates_to_smagorinsky_cs0(self, original_rans_ke_source):
        """BUG-1: collide_rans_sa delegated to collide_smagorinsky_mrt3d with C_s=0."""
        assert "collide_smagorinsky_mrt3d" in original_rans_ke_source
        assert "C_s=0.0" in original_rans_ke_source

    def test_bug2_ke_mask_bool_allocation_exists(self, original_rans_ke_source):
        """BUG-2: collide_rans_ke allocated mask.bool() per call."""
        assert "mask.bool()" in original_rans_ke_source

    def test_bug3_ke_tau_eff_clamp_range(self, original_rans_ke_source):
        """BUG-3: collide_rans_ke clamped tau_eff to [0.501, 3.0]."""
        assert ".clamp(0.501, 3.0)" in original_rans_ke_source

    def test_fixed_sa_no_scalar_averaging(self):
        """The fixed collide_rans_sa must NOT contain .mean().item() in code."""
        from tensorlbm.rans_ke import collide_rans_sa
        src = inspect.getsource(collide_rans_sa)
        # Check executable code (not docstring) for the buggy patterns
        # Strip docstring by checking only lines that aren't inside triple-quotes
        assert ".mean().item()" not in src
        # The fixed version must not CALL collide_smagorinsky_mrt3d (only
        # mentions it in the docstring describing the old bug)
        assert "return collide_smagorinsky_mrt3d" not in src
        assert "collide_smagorinsky_mrt3d(f" not in src

    def test_fixed_ke_no_mask_bool_allocation(self):
        """The fixed collide_rans_ke must NOT contain mask.bool()."""
        from tensorlbm.rans_ke import collide_rans_ke
        src = inspect.getsource(collide_rans_ke)
        assert ".bool()" not in src

    def test_fixed_common_no_host_sync(self):
        """Common collision functions must have no GPU→CPU sync patterns."""
        for func in [collide_rans_bgk3d, collide_rans_mrt3d,
                     collide_rans_bgk27, collide_rans_mrt27, collide_rans_3d]:
            src = inspect.getsource(func)
            for pat in (".item()", "float(", "bool(tensor"):
                assert pat not in src, f"{func.__name__}: found '{pat}'"


# ===========================================================================
# Part 2: k-epsilon MRT equivalence (correct part)
# ===========================================================================

class TestKeMrtEquivalence:
    """Verify that the original collide_rans_ke MRT collision logic is
    algebraically equivalent to the common collide_rans_mrt3d.

    The original collide_rans_ke (lines 405–425 @ 2341767^) used:
      tau_eff = (3.0 * (nu_lam + nu_t) + 0.5).clamp(0.501, 3.0)
    which expands to tau + 3*nu_t (since nu_lam = (tau-0.5)/3).

    The common module uses _nu_t_to_tau_eff(tau, nu_t) = clamp(tau+3*nu_t, 0.5001).

    For nu_t ∈ [0, 0.05] and tau=0.7, tau_eff ∈ [0.7, 0.85] — well within
    both clamp ranges, so results must be allclose.
    """

    def test_tau_eff_algebraic_equivalence(self):
        """tau + 3*nu_t == 3*(nu_lam + nu_t) + 0.5  (algebraically identical)."""
        nu_t = _nu_t_field_3d()
        nu_lam = (TAU - 0.5) / 3.0

        original_tau_eff = (3.0 * (nu_lam + nu_t) + 0.5).clamp(0.501, 3.0)
        common_tau_eff = _nu_t_to_tau_eff(TAU, nu_t)

        # For these nu_t values, neither clamp range triggers
        assert torch.allclose(original_tau_eff, common_tau_eff, atol=1e-5)

    def test_ke_mrt_matches_common_mrt(self):
        """Original collide_rans_ke MRT logic == common collide_rans_mrt3d.

        We reconstruct the original MRT collision inline (from the pre-extraction
        source) and compare against the common module's collide_rans_mrt3d.
        Both use the same f, tau, nu_t, M, M_inv, s_fixed, and stress-mode
        override (modes 9–13).
        """
        from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
        from tensorlbm.solver3d import _get_d3q19_mrt_matrices

        f = _non_equilibrium_f3d19()
        nu_t = _nu_t_field_3d()

        # --- Reconstruct original collide_rans_ke MRT collision ---
        nu_lam = (TAU - 0.5) / 3.0
        tau_eff_orig = (3.0 * (nu_lam + nu_t) + 0.5).clamp(0.501, 3.0)
        s_nu_field_orig = 1.0 / tau_eff_orig

        device = f.device
        M, M_inv = _get_d3q19_mrt_matrices(device)
        rho, ux, uy, uz = macroscopic3d(f)
        feq = equilibrium3d(rho, ux, uy, uz)
        nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
        f_flat = f.reshape(19, -1)
        feq_flat = feq.reshape(19, -1)
        s_nu_flat_orig = s_nu_field_orig.reshape(-1)
        m = M @ f_flat
        m_eq = M @ feq_flat
        dm = m - m_eq
        s_e, s_eps, s_q, s_pi = 1.19, 1.4, 1.2, 1.19
        s_fixed = torch.tensor(
            [0.0, s_e, s_eps, 0.0, s_q, 0.0, s_q, 0.0, s_q, 0, 0, 0, 0, 0,
             s_pi, s_pi, 1.0, 1.0, 1.0],
            dtype=f.dtype, device=device,
        )
        m_star_orig = m - s_fixed.unsqueeze(1) * dm
        for k in (9, 10, 11, 12, 13):
            m_star_orig[k] = m[k] - s_nu_flat_orig * dm[k]
        out_orig = (M_inv @ m_star_orig).reshape(19, nz, ny, nx)

        # --- Common module collide_rans_mrt3d ---
        out_common = collide_rans_mrt3d(f, TAU, nu_t)

        assert torch.allclose(out_orig, out_common, atol=1e-6), (
            f"max diff: {(out_orig - out_common).abs().max().item()}"
        )

    def test_ke_mrt_matches_common_via_dispatch(self):
        """collide_rans_3d('D3Q19','MRT') == collide_rans_mrt3d (same path)."""
        f = _non_equilibrium_f3d19()
        nu_t = _nu_t_field_3d()
        out_direct = collide_rans_mrt3d(f, TAU, nu_t)
        out_dispatch = collide_rans_3d("D3Q19", "MRT", f, tau=TAU, nu_t=nu_t)
        assert torch.allclose(out_direct, out_dispatch, atol=1e-7)

    def test_ke_mrt_equivalence_with_ke_solver(self):
        """End-to-end: collide_rans_ke (current) == common collide_rans_mrt3d
        when fed the same nu_t from the same KESolver state.

        Since the current collide_rans_ke delegates to collide_rans_3d, this
        verifies the delegation path produces identical results to calling
        collide_rans_mrt3d directly with the same nu_t.
        """
        from tensorlbm.rans_ke import KESolver, collide_rans_ke

        f = _non_equilibrium_f3d19()
        _, ux, uy, uz = macroscopic3d(f)

        # Run collide_rans_ke (which delegates to common)
        solver1 = KESolver(nu=0.01)
        solver1.initialize(ux, uy, uz)
        out_via_rans_ke = collide_rans_ke(f, TAU, solver1, lattice="D3Q19",
                                          collision="MRT")

        # Reproduce: same solver state, same nu_t, call common directly
        solver2 = KESolver(nu=0.01)
        solver2.initialize(ux, uy, uz)
        nu_t = solver2.step(ux, uy, uz, None)
        out_direct = collide_rans_mrt3d(f, TAU, nu_t)

        assert torch.allclose(out_via_rans_ke, out_direct, atol=1e-6), (
            f"max diff: {(out_via_rans_ke - out_direct).abs().max().item()}"
        )


# ===========================================================================
# Part 3: SA equivalence (bug-fixed original vs common)
# ===========================================================================

class TestSaEquivalence:
    """Verify that the FIXED SA collision (per-cell nu_t) matches the common
    module, and that the BUGGY original (scalar averaging) does NOT match.

    The original collide_rans_sa had BUG-1: nu_t.mean().item() scalar averaging.
    We verify:
      1. The buggy scalar-averaged output DIFFERS from the per-cell common output
         (proving the bug was real, not a no-op)
      2. The fixed per-cell SA output MATCHES the common collide_rans_mrt3d
    """

    def test_buggy_sa_scalar_differs_from_per_cell(self):
        """The original scalar-averaged SA collision must differ from the
        per-cell common collision when nu_t is spatially non-uniform.

        This proves BUG-1 was a real correctness bug, not a no-op.
        """
        from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
        from tensorlbm.solver3d import _get_d3q19_mrt_matrices
        from tensorlbm.turbulence import collide_smagorinsky_mrt3d

        f = _non_equilibrium_f3d19()
        nu_t = _nu_t_field_3d()  # spatially non-uniform

        # --- Buggy original: scalar averaging ---
        nu_lam = (TAU - 0.5) / 3.0
        nu_eff_scalar = nu_lam + nu_t.mean().item()
        tau_eff_scalar = min(max(3.0 * nu_eff_scalar + 0.5, 0.501), 2.0)
        out_buggy = collide_smagorinsky_mrt3d(f, tau=tau_eff_scalar, C_s=0.0)

        # --- Fixed: per-cell via common module ---
        out_fixed = collide_rans_mrt3d(f, TAU, nu_t)

        # They MUST differ (the bug was real)
        assert not torch.allclose(out_buggy, out_fixed, atol=1e-6), (
            "Buggy scalar-averaged SA should differ from per-cell common "
            "when nu_t is spatially non-uniform"
        )

    def test_fixed_sa_matches_common_mrt(self):
        """The fixed collide_rans_sa (current, delegates to common) produces
        the same output as calling collide_rans_mrt3d directly with the
        same per-cell nu_t from the SASolver.
        """
        from tensorlbm.rans_ke import SASolver, collide_rans_sa

        f = _non_equilibrium_f3d19()
        _, ux, uy, uz = macroscopic3d(f)
        nz, ny, nx = ux.shape
        wall_dist = torch.full((nz, ny, nx), 5.0)

        # Run collide_rans_sa (current, delegates to common)
        solver1 = SASolver(nu=0.01)
        solver1.initialize(ux, uy, uz)
        out_via_rans_sa = collide_rans_sa(
            f, TAU, solver1, wall_dist, lattice="D3Q19", collision="MRT"
        )

        # Reproduce: same solver state, same nu_t, call common directly
        solver2 = SASolver(nu=0.01)
        solver2.initialize(ux, uy, uz)
        nu_t = solver2.step(ux, uy, uz, wall_dist, None)
        out_direct = collide_rans_mrt3d(f, TAU, nu_t)

        assert torch.allclose(out_via_rans_sa, out_direct, atol=1e-6), (
            f"max diff: {(out_via_rans_sa - out_direct).abs().max().item()}"
        )

    def test_sa_uniform_nu_t_buggy_matches_fixed(self):
        """When nu_t is uniform (spatially constant), the buggy scalar
        averaging and the per-cell common produce the same result.

        This confirms the bug only manifests with spatially varying nu_t.
        """
        from tensorlbm.turbulence import collide_smagorinsky_mrt3d

        f = _non_equilibrium_f3d19()
        nu_t_uniform = torch.full(f.shape[1:], 0.03)  # uniform

        # Buggy: scalar averaging (no-op for uniform field)
        nu_lam = (TAU - 0.5) / 3.0
        nu_eff = nu_lam + nu_t_uniform.mean().item()
        tau_eff = min(max(3.0 * nu_eff + 0.5, 0.501), 2.0)
        out_buggy = collide_smagorinsky_mrt3d(f, tau=tau_eff, C_s=0.0)

        # Fixed: per-cell (uniform → same as scalar)
        out_fixed = collide_rans_mrt3d(f, TAU, nu_t_uniform)

        # With uniform nu_t and tau_eff < 2.0 (no max clamp), they match
        # tau_eff = 0.7 + 3*0.03 = 0.79, well below 2.0
        assert torch.allclose(out_buggy, out_fixed, atol=1e-5)


# ===========================================================================
# Part 4: k-omega SST (new 3D feature, no original to compare)
# ===========================================================================

class TestKOmegaSstNewFeature:
    """k-omega SST 3D collision is NEW in the common module — no original
    3D implementation existed.  Verify physical reasonableness.
    """

    @pytest.mark.parametrize(
        ("lattice", "collision"),
        [("D3Q19", "BGK"), ("D3Q19", "MRT"),
         ("D3Q27", "BGK"), ("D3Q27", "MRT")],
    )
    def test_sst_3d_finite_and_mass(self, lattice, collision):
        """SST 3D collision produces finite output with mass conservation."""
        from tensorlbm.rans_ke import KOmegaSSTSolver, collide_rans_komega_sst

        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        macro = macroscopic3d if lattice == "D3Q19" else macroscopic27
        _, ux, uy, uz = macro(f)
        nz, ny, nx = ux.shape
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        wall_dist = torch.full((nz, ny, nx), 5.0)
        solver = KOmegaSSTSolver(mask=mask, nu_lbm=0.01)
        out = collide_rans_komega_sst(
            f, solver, ux, uy, uz, tau=TAU, wall_dist=wall_dist,
            lattice=lattice, collision=collision,
        )
        assert out.shape == f.shape
        assert torch.isfinite(out).all()
        rho_in, _, _, _ = macro(f)
        rho_out, _, _, _ = macro(out)
        assert torch.allclose(rho_out, rho_in, atol=1e-4)

    def test_sst_3d_strain_rate_uses_uz(self):
        """The 3D SST strain rate must use uz (full 3D), not 2D approximation."""
        from tensorlbm.rans_ke import KOmegaSSTSolver

        nz, ny, nx = 4, 6, 8
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        solver = KOmegaSSTSolver(mask=mask, nu_lbm=0.01)

        g = torch.Generator().manual_seed(7)
        ux = torch.rand((nz, ny, nx), generator=g) * 0.04
        uy = torch.rand((nz, ny, nx), generator=g) * 0.04
        uz = torch.rand((nz, ny, nx), generator=g) * 0.04

        s_2d = solver._compute_strain_rate(ux, uy, uz=None)
        s_3d = solver._compute_strain_rate(ux, uy, uz=uz)

        # 3D strain must differ from 2D approximation when uz != 0
        assert not torch.allclose(s_2d, s_3d, atol=1e-8)

    def test_sst_nu_t_is_per_cell(self):
        """SST nu_t must be a per-cell field, not a scalar."""
        from tensorlbm.rans_ke import KOmegaSSTSolver

        nz, ny, nx = 4, 6, 8
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        solver = KOmegaSSTSolver(mask=mask, nu_lbm=0.01)
        nu_t = solver.compute_nu_t()
        assert nu_t.ndim == 3
        assert nu_t.shape == (nz, ny, nx)


# ===========================================================================
# Part 5: Combination tests — full collide→stream→boundary loop
# ===========================================================================

class TestRansCombinationLoop:
    """RANS + BGK/MRT × D3Q19/D3Q27 full collide→stream loop.

    Verifies finite + mass conservation over multiple time steps.
    """

    @pytest.mark.parametrize(
        ("lattice", "collision"),
        [("D3Q19", "BGK"), ("D3Q19", "MRT"),
         ("D3Q27", "BGK"), ("D3Q27", "MRT")],
    )
    def test_multi_step_finite_and_mass(self, lattice, collision):
        """Run 5 collide→stream steps; verify finite + mass conservation."""
        f = (_f3d19() if lattice == "D3Q19" else _f3d27()).clone()
        nu_t = _nu_t_field_3d()
        macro = macroscopic3d if lattice == "D3Q19" else macroscopic27
        stream = stream3d if lattice == "D3Q19" else _stream27

        rho0, _, _, _ = macro(f)
        mass0 = rho0.sum().item()

        for step in range(5):
            f = collide_rans_3d(lattice, collision, f, tau=TAU, nu_t=nu_t)
            f = stream(f)

        assert torch.isfinite(f).all(), f"NaN/Inf at step {step}"
        rho_final, _, _, _ = macro(f)
        mass_final = rho_final.sum().item()
        # Periodic streaming conserves mass; collision conserves mass
        assert abs(mass_final - mass0) < 1e-3, (
            f"mass drift: {mass_final - mass0}"
        )

    @pytest.mark.parametrize(
        ("lattice", "collision"),
        [("D3Q19", "BGK"), ("D3Q19", "MRT"),
         ("D3Q27", "BGK"), ("D3Q27", "MRT")],
    )
    def test_multi_step_momentum_stable(self, lattice, collision):
        """Momentum must remain bounded over multiple steps."""
        f = (_f3d19() if lattice == "D3Q19" else _f3d27()).clone()
        nu_t = _nu_t_field_3d()
        macro = macroscopic3d if lattice == "D3Q19" else macroscopic27
        stream = stream3d if lattice == "D3Q19" else _stream27

        for _ in range(5):
            f = collide_rans_3d(lattice, collision, f, tau=TAU, nu_t=nu_t)
            f = stream(f)

        _, ux, uy, uz = macro(f)
        u_max = (ux.abs().max() + uy.abs().max() + uz.abs().max()).item()
        assert u_max < 1.0, f"velocity blew up: {u_max}"

    @pytest.mark.parametrize("collision", ["BGK", "MRT"])
    def test_d3q19_ke_full_loop(self, collision):
        """Full loop with real KESolver: collide_rans_ke → stream."""
        from tensorlbm.rans_ke import KESolver, collide_rans_ke

        f = _f3d19().clone()
        _, ux, uy, uz = macroscopic3d(f)
        solver = KESolver(nu=0.01)
        solver.initialize(ux, uy, uz)

        rho0, _, _, _ = macroscopic3d(f)
        mass0 = rho0.sum().item()

        for _ in range(3):
            f = collide_rans_ke(f, TAU, solver, lattice="D3Q19",
                                collision=collision)
            f = stream3d(f)

        assert torch.isfinite(f).all()
        rho_f, _, _, _ = macroscopic3d(f)
        assert abs(rho_f.sum().item() - mass0) < 1e-3

    @pytest.mark.parametrize("collision", ["BGK", "MRT"])
    def test_d3q19_sa_full_loop(self, collision):
        """Full loop with real SASolver: collide_rans_sa → stream."""
        from tensorlbm.rans_ke import SASolver, collide_rans_sa

        f = _f3d19().clone()
        _, ux, uy, uz = macroscopic3d(f)
        nz, ny, nx = ux.shape
        wall_dist = torch.full((nz, ny, nx), 5.0)
        solver = SASolver(nu=0.01)
        solver.initialize(ux, uy, uz)

        rho0, _, _, _ = macroscopic3d(f)
        mass0 = rho0.sum().item()

        for _ in range(3):
            f = collide_rans_sa(f, TAU, solver, wall_dist,
                                lattice="D3Q19", collision=collision)
            f = stream3d(f)

        assert torch.isfinite(f).all()
        rho_f, _, _, _ = macroscopic3d(f)
        assert abs(rho_f.sum().item() - mass0) < 1e-3

    @pytest.mark.parametrize("collision", ["BGK", "MRT"])
    def test_d3q19_sst_full_loop(self, collision):
        """Full loop with real KOmegaSSTSolver: collide_rans_komega_sst → stream."""
        from tensorlbm.rans_ke import KOmegaSSTSolver, collide_rans_komega_sst

        f = _f3d19().clone()
        _, ux, uy, uz = macroscopic3d(f)
        nz, ny, nx = ux.shape
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        wall_dist = torch.full((nz, ny, nx), 5.0)
        solver = KOmegaSSTSolver(mask=mask, nu_lbm=0.01)

        rho0, _, _, _ = macroscopic3d(f)
        mass0 = rho0.sum().item()

        for _ in range(3):
            f = collide_rans_komega_sst(
                f, solver, ux, uy, uz, tau=TAU, wall_dist=wall_dist,
                lattice="D3Q19", collision=collision,
            )
            f = stream3d(f)

        assert torch.isfinite(f).all()
        rho_f, _, _, _ = macroscopic3d(f)
        assert abs(rho_f.sum().item() - mass0) < 1e-3

    @pytest.mark.parametrize("collision", ["BGK", "MRT"])
    def test_d3q27_rans_full_loop(self, collision):
        """Full loop D3Q27 with per-cell nu_t: collide_rans_3d → stream27."""
        from tensorlbm.d3q27 import stream27

        f = _f3d27().clone()
        nu_t = _nu_t_field_3d()

        rho0, _, _, _ = macroscopic27(f)
        mass0 = rho0.sum().item()

        for _ in range(3):
            f = collide_rans_3d("D3Q27", collision, f, tau=TAU, nu_t=nu_t)
            f = stream27(f)

        assert torch.isfinite(f).all()
        rho_f, _, _, _ = macroscopic27(f)
        assert abs(rho_f.sum().item() - mass0) < 1e-3


# Helper for D3Q27 streaming (imported lazily to avoid hard dependency)
def _stream27(f):
    from tensorlbm.d3q27 import stream27
    return stream27(f)
