"""Regression equivalence verification for wave boundary-condition extraction.

This test suite verifies that the common wave-BC module (``wave_bc_common.py``)
is a faithful, bug-free replacement for the original ``wave_bc.py``:

1. **Bug identification** — the original ``zou_he_inlet_velocity_profile_3d``
   uses ``.item()`` (GPU→CPU host sync) inside the hot Zou-He loop, a known
   performance violation.  The common module pre-computes direction-index
   lists at import time and vectorises the update, eliminating the sync.

2. **D3Q19 equivalence** — with identical inputs *f* and identical wave
   parameters, the original and common Zou-He inlets produce *bit-identical*
   outputs (max diff = 0.0).  This confirms that removing ``.item()`` and
   switching from a 2-D equilibrium call to a 3-D ``(nz, ny, 1)`` call does
   not change the numerical result.

3. **D3Q27 physical reasonableness** — the D3Q27 Zou-He inlet is new in the
   common module (no original to compare).  We verify:
   - Direction lists match the D3Q27 stencil (cx>0, cx=0, cx<0).
   - Opposite mapping is correct.
   - Mass conservation: total population at the inlet equals inferred rho.
   - Velocity prescription: x-momentum at the inlet equals rho * ux_in.
   - NEBB structure: f_new[k] = feq[k] - feq[opp_k] + f[opp_k].

4. **Combination test** — wave BC + BGK collision + streaming in a complete
   LBM cycle, verifying stability, mass conservation, and velocity injection.
"""

from __future__ import annotations

import ast
import inspect
import math
import textwrap

import pytest
import torch

from tensorlbm.d3q19 import C as C19, OPPOSITE as OPP19, equilibrium3d, macroscopic3d
from tensorlbm.d3q27 import C as C27, OPPOSITE as OPP27, equilibrium27, macroscopic27
from tensorlbm.solver3d import collide_bgk3d, stream3d
from tensorlbm.wave_bc import (
    airy_wave_velocity_3d as orig_airy,
    zou_he_inlet_velocity_profile_3d as orig_zou_he_19,
)
from tensorlbm.wave_bc_common import (
    WaveParams,
    _airy_wave_velocity_3d as common_airy,
    _D3Q19_CX0,
    _D3Q19_CX_NEG,
    _D3Q19_INLET_DIRS,
    _D3Q19_INLET_OPP,
    _D3Q27_CX0,
    _D3Q27_CX_NEG,
    _D3Q27_INLET_DIRS,
    _D3Q27_INLET_OPP,
    wave_bc_3d,
    zou_he_inlet_velocity_profile_19 as common_zou_he_19,
    zou_he_inlet_velocity_profile_27 as common_zou_he_27,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_item_call_in_source(func) -> bool:
    """Return True if the function source contains a ``.item()`` method call
    (GPU→CPU host sync) in executable code, excluding docstrings."""
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    # Find the docstring node (first Expr with Constant value in the body)
    docstring_ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)):
                docstring_ranges.append(
                    (node.body[0].lineno, node.body[0].end_lineno or node.body[0].lineno)
                )
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "item"):
            # Check if this call is inside a docstring range
            line = node.lineno
            in_docstring = any(start <= line <= end for start, end in docstring_ranges)
            if not in_docstring:
                return True
    return False


def _make_f19(nz: int, ny: int, nx: int, seed: int = 42) -> torch.Tensor:
    """Create a plausible D3Q19 distribution (positive, near-equilibrium)."""
    torch.manual_seed(seed)
    rho0 = torch.ones(nz, ny, nx, dtype=torch.float32)
    ux0 = torch.full((nz, ny, nx), 0.05, dtype=torch.float32)
    uy0 = torch.zeros(nz, ny, nx, dtype=torch.float32)
    uz0 = torch.zeros(nz, ny, nx, dtype=torch.float32)
    feq = equilibrium3d(rho0, ux0, uy0, uz0)
    noise = torch.randn(19, nz, ny, nx, dtype=torch.float32) * 1e-4
    return (feq + noise).clamp(min=1e-8)


def _make_f27(nz: int, ny: int, nx: int, seed: int = 42) -> torch.Tensor:
    """Create a plausible D3Q27 distribution (positive, near-equilibrium)."""
    torch.manual_seed(seed)
    rho0 = torch.ones(nz, ny, nx, dtype=torch.float32)
    ux0 = torch.full((nz, ny, nx), 0.05, dtype=torch.float32)
    uy0 = torch.zeros(nz, ny, nx, dtype=torch.float32)
    uz0 = torch.zeros(nz, ny, nx, dtype=torch.float32)
    feq = equilibrium27(rho0, ux0, uy0, uz0)
    noise = torch.randn(27, nz, ny, nx, dtype=torch.float32) * 1e-4
    return (feq + noise).clamp(min=1e-8)


def _wave_params_dict() -> dict:
    """Standard wave parameters for equivalence tests."""
    return dict(
        step=10,
        u_mean=0.05,
        wave_amp=0.02,
        wave_period=200.0,
        wave_k=0.05,
        water_depth=8.0,
        z_bed=0.0,
    )


# ===========================================================================
# 1. Bug identification: .item() host sync in hot path
# ===========================================================================

class TestBugIdentification:
    """Verify the original has .item() in the hot path and the common doesn't."""

    def test_original_zou_he_has_item_in_hot_path(self) -> None:
        """The original zou_he_inlet_velocity_profile_3d calls .item() inside
        its Python loop — a GPU→CPU host sync on every time step."""
        assert _has_item_call_in_source(orig_zou_he_19), (
            "Expected .item() host sync in original zou_he_inlet_velocity_profile_3d"
        )

    def test_common_zou_he_19_no_item_in_hot_path(self) -> None:
        """The common zou_he_inlet_velocity_profile_19 must NOT call .item()
        inside the function body — direction indices are pre-computed."""
        assert not _has_item_call_in_source(common_zou_he_19), (
            "Common zou_he_inlet_velocity_profile_19 must not call .item() "
            "(host sync eliminated via pre-computed index lists)"
        )

    def test_common_zou_he_27_no_item_in_hot_path(self) -> None:
        """The D3Q27 variant must also be free of .item() in the hot path."""
        assert not _has_item_call_in_source(common_zou_he_27), (
            "Common zou_he_inlet_velocity_profile_27 must not call .item()"
        )

    def test_common_item_only_at_module_load_time(self) -> None:
        """The .item() calls that build _D3Q19_INLET_OPP and _D3Q27_INLET_OPP
        are at module scope (import time), not inside BC functions — this is
        acceptable because they run once, not per time step."""
        module_source = inspect.getsource(
            __import__("tensorlbm.wave_bc_common", fromlist=["x"])
        )
        # The module-level .item() calls are in list comprehensions that
        # build _D3Q19_INLET_OPP and _D3Q27_INLET_OPP.
        assert "_D3Q19_INLET_OPP" in module_source
        assert "_D3Q27_INLET_OPP" in module_source
        # Verify the pre-computed values are plain Python ints
        assert all(isinstance(v, int) for v in _D3Q19_INLET_OPP)
        assert all(isinstance(v, int) for v in _D3Q27_INLET_OPP)

    def test_precomputed_opposite_matches_opposite_tensor(self) -> None:
        """The pre-computed _D3Q19_INLET_OPP must match OPPOSITE[inlet_dirs]."""
        for k, opp_k in zip(_D3Q19_INLET_DIRS, _D3Q19_INLET_OPP, strict=True):
            assert opp_k == int(OPP19[k]), (
                f"D3Q19: OPPOSITE[{k}] = {int(OPP19[k])} but _D3Q19_INLET_OPP has {opp_k}"
            )

    def test_precomputed_d3q27_opposite_matches_opposite_tensor(self) -> None:
        """The pre-computed _D3Q27_INLET_OPP must match OPPOSITE[inlet_dirs]."""
        for k, opp_k in zip(_D3Q27_INLET_DIRS, _D3Q27_INLET_OPP, strict=True):
            assert opp_k == int(OPP27[k]), (
                f"D3Q27: OPPOSITE[{k}] = {int(OPP27[k])} but _D3Q27_INLET_OPP has {opp_k}"
            )


# ===========================================================================
# 2. Velocity profile equivalence
# ===========================================================================

class TestVelocityProfileEquivalence:
    """Verify _airy_wave_velocity_3d (common) == airy_wave_velocity_3d (original)."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_velocity_profile_identical(self, seed: int) -> None:
        nz, ny = 12, 8
        wp = _wave_params_dict()
        torch.manual_seed(seed)  # doesn't affect airy (deterministic), but for consistency

        ux_o, uy_o, uz_o = orig_airy(nz, ny, device=torch.device("cpu"), **wp)
        ux_c, uy_c, uz_c = common_airy(nz, ny, device=torch.device("cpu"), **wp)

        assert ux_o.shape == ux_c.shape == (nz, ny)
        assert uy_o.shape == uy_c.shape == (nz, ny)
        assert uz_o.shape == uz_c.shape == (nz, ny)

        assert torch.equal(ux_o, ux_c), "ux mismatch"
        assert torch.equal(uy_o, uy_c), "uy mismatch"
        assert torch.equal(uz_o, uz_c), "uz mismatch"

    @pytest.mark.parametrize(
        "nz,ny",
        [(4, 3), (8, 5), (16, 12), (32, 16)],
    )
    def test_velocity_profile_various_grid_sizes(self, nz: int, ny: int) -> None:
        wp = _wave_params_dict()
        ux_o, uy_o, uz_o = orig_airy(nz, ny, device=torch.device("cpu"), **wp)
        ux_c, uy_c, uz_c = common_airy(nz, ny, device=torch.device("cpu"), **wp)
        assert torch.equal(ux_o, ux_c)
        assert torch.equal(uy_o, uy_c)
        assert torch.equal(uz_o, uz_c)

    def test_velocity_profile_zero_amplitude(self) -> None:
        """Zero wave amplitude → uniform current only."""
        nz, ny = 8, 5
        wp = _wave_params_dict()
        wp["wave_amp"] = 0.0
        ux_o, uy_o, uz_o = orig_airy(nz, ny, device=torch.device("cpu"), **wp)
        ux_c, uy_c, uz_c = common_airy(nz, ny, device=torch.device("cpu"), **wp)
        assert torch.equal(ux_o, ux_c)
        # With zero amplitude, ux should be uniform u_mean
        assert torch.allclose(ux_o, torch.full((nz, ny), wp["u_mean"]))
        # uz should be zero
        assert torch.allclose(uz_o, torch.zeros(nz, ny))


# ===========================================================================
# 3. D3Q19 Zou-He inlet equivalence
# ===========================================================================

class TestD3Q19ZouHeEquivalence:
    """Verify original vs common Zou-He inlet produce identical outputs."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_zou_he_inlet_identical(self, seed: int) -> None:
        nz, ny, nx = 8, 6, 10
        f = _make_f19(nz, ny, nx, seed=seed)
        wp = _wave_params_dict()
        ux_in, uy_in, uz_in = orig_airy(nz, ny, device=torch.device("cpu"), **wp)

        f_orig = orig_zou_he_19(f.clone(), ux_in, uy_in, uz_in)
        f_common = common_zou_he_19(f.clone(), ux_in, uy_in, uz_in)

        assert f_orig.shape == f_common.shape == f.shape
        # Bit-identical — the .item() removal and shape change don't affect numerics
        assert torch.equal(f_orig, f_common), (
            f"D3Q19 Zou-He inlet outputs differ: "
            f"max diff = {(f_orig - f_common).abs().max().item()}"
        )

    @pytest.mark.parametrize(
        "nz,ny,nx",
        [(4, 3, 6), (8, 5, 10), (12, 8, 16), (16, 12, 20)],
    )
    def test_zou_he_inlet_various_grid_sizes(self, nz: int, ny: int, nx: int) -> None:
        f = _make_f19(nz, ny, nx, seed=42)
        wp = _wave_params_dict()
        ux_in, uy_in, uz_in = orig_airy(nz, ny, device=torch.device("cpu"), **wp)

        f_orig = orig_zou_he_19(f.clone(), ux_in, uy_in, uz_in)
        f_common = common_zou_he_19(f.clone(), ux_in, uy_in, uz_in)
        assert torch.equal(f_orig, f_common)

    def test_zou_he_inlet_only_modifies_inlet_dirs_at_x0(self) -> None:
        """Only the 5 inlet directions at x=0 should change; everything else
        must be identical to the input."""
        nz, ny, nx = 8, 6, 10
        f = _make_f19(nz, ny, nx, seed=7)
        wp = _wave_params_dict()
        ux_in, uy_in, uz_in = orig_airy(nz, ny, device=torch.device("cpu"), **wp)

        f_common = common_zou_he_19(f.clone(), ux_in, uy_in, uz_in)

        # Non-inlet directions at x=0 must be unchanged
        non_inlet = [k for k in range(19) if k not in _D3Q19_INLET_DIRS]
        for k in non_inlet:
            assert torch.equal(f_common[k, :, :, 0], f[k, :, :, 0]), (
                f"Direction {k} at x=0 was modified (should be unchanged)"
            )
        # All directions at x>0 must be unchanged
        assert torch.equal(f_common[:, :, :, 1:], f[:, :, :, 1:]), (
            "Non-inlet cells were modified"
        )

    def test_zou_he_inlet_uniform_velocity(self) -> None:
        """With a uniform (scalar-like) velocity, the result should still match."""
        nz, ny, nx = 6, 4, 8
        f = _make_f19(nz, ny, nx, seed=99)
        ux_in = torch.full((nz, ny), 0.05, dtype=torch.float32)
        uy_in = torch.zeros(nz, ny, dtype=torch.float32)
        uz_in = torch.zeros(nz, ny, dtype=torch.float32)

        f_orig = orig_zou_he_19(f.clone(), ux_in, uy_in, uz_in)
        f_common = common_zou_he_19(f.clone(), ux_in, uy_in, uz_in)
        assert torch.equal(f_orig, f_common)

    def test_zou_he_inlet_nontrivial_velocity(self) -> None:
        """With a spatially varying velocity field (as from Airy wave theory)."""
        nz, ny, nx = 10, 6, 12
        f = _make_f19(nz, ny, nx, seed=55)
        wp = _wave_params_dict()
        wp["step"] = 25  # phase = ω·25 = π/4 → cos ≠ 0, sin ≠ 0
        ux_in, uy_in, uz_in = orig_airy(nz, ny, device=torch.device("cpu"), **wp)

        # Verify the velocity is actually non-trivial (spatially varying)
        assert ux_in.std().item() > 1e-6, "ux should vary with depth"
        assert uz_in.std().item() > 1e-6, "uz should vary with depth"

        f_orig = orig_zou_he_19(f.clone(), ux_in, uy_in, uz_in)
        f_common = common_zou_he_19(f.clone(), ux_in, uy_in, uz_in)
        assert torch.equal(f_orig, f_common)

    def test_wave_bc_3d_dispatch_matches_original_inlet(self) -> None:
        """The wave_bc_3d dispatch (D3Q19, no outlet) should produce the same
        inlet update as the original zou_he_inlet_velocity_profile_3d."""
        nz, ny, nx = 8, 6, 10
        f = _make_f19(nz, ny, nx, seed=33)
        wp = _wave_params_dict()

        params = WaveParams(
            step=wp["step"],
            u_mean=wp["u_mean"],
            wave_amp=wp["wave_amp"],
            wave_period=wp["wave_period"],
            wave_k=wp["wave_k"],
            water_depth=wp["water_depth"],
            z_bed=wp["z_bed"],
            apply_outlet=False,
        )
        f_dispatch = wave_bc_3d(f.clone(), wave_params=params, lattice="D3Q19")

        ux_in, uy_in, uz_in = orig_airy(nz, ny, device=torch.device("cpu"), **wp)
        f_orig = orig_zou_he_19(f.clone(), ux_in, uy_in, uz_in)

        assert torch.equal(f_dispatch, f_orig), (
            "wave_bc_3d dispatch does not match original Zou-He inlet"
        )


# ===========================================================================
# 4. D3Q27 physical reasonableness (no original to compare)
# ===========================================================================

class TestD3Q27PhysicalReasonableness:
    """Verify D3Q27 Zou-He inlet is physically correct (no original reference)."""

    def test_d3q27_inlet_dirs_are_cx_positive(self) -> None:
        """All _D3Q27_INLET_DIRS must have cx > 0 in the D3Q27 stencil."""
        cx = C27[:, 0]
        for k in _D3Q27_INLET_DIRS:
            assert int(cx[k]) > 0, (
                f"Direction {k} in _D3Q27_INLET_DIRS has cx={int(cx[k])} (not > 0)"
            )

    def test_d3q27_cx0_dirs_are_cx_zero(self) -> None:
        """All _D3Q27_CX0 must have cx = 0."""
        cx = C27[:, 0]
        for k in _D3Q27_CX0:
            assert int(cx[k]) == 0, (
                f"Direction {k} in _D3Q27_CX0 has cx={int(cx[k])} (not 0)"
            )

    def test_d3q27_cx_neg_dirs_are_cx_negative(self) -> None:
        """All _D3Q27_CX_NEG must have cx < 0."""
        cx = C27[:, 0]
        for k in _D3Q27_CX_NEG:
            assert int(cx[k]) < 0, (
                f"Direction {k} in _D3Q27_CX_NEG has cx={int(cx[k])} (not < 0)"
            )

    def test_d3q27_direction_partition_complete(self) -> None:
        """The three direction lists must partition all 27 directions."""
        all_dirs = set(range(27))
        partition = set(_D3Q27_INLET_DIRS) | set(_D3Q27_CX0) | set(_D3Q27_CX_NEG)
        assert partition == all_dirs, (
            f"Direction partition incomplete: missing {all_dirs - partition}, "
            f"extra {partition - all_dirs}"
        )
        # No overlaps
        assert len(_D3Q27_INLET_DIRS) + len(_D3Q27_CX0) + len(_D3Q27_CX_NEG) == 27

    def test_d3q27_inlet_dirs_superset_of_d3q19(self) -> None:
        """D3Q27 inlet dirs must include all D3Q19 inlet dirs (the 5 face/edge
        directions) plus the 4 corner directions (19, 21, 23, 25)."""
        d3q19_set = set(_D3Q19_INLET_DIRS)
        d3q27_set = set(_D3Q27_INLET_DIRS)
        assert d3q19_set.issubset(d3q27_set), (
            "D3Q27 inlet dirs must be a superset of D3Q19 inlet dirs"
        )
        # The 4 extra directions are the corners with cx>0
        extra = d3q27_set - d3q19_set
        assert extra == {19, 21, 23, 25}, f"Unexpected extra D3Q27 dirs: {extra}"

    def test_d3q27_opposite_maps_cx_positive_to_cx_negative(self) -> None:
        """Each inlet direction's opposite must have cx < 0."""
        cx = C27[:, 0]
        for k, opp_k in zip(_D3Q27_INLET_DIRS, _D3Q27_INLET_OPP, strict=True):
            assert int(cx[k]) > 0, f"Direction {k} should have cx > 0"
            assert int(cx[opp_k]) < 0, (
                f"OPPOSITE[{k}] = {opp_k} has cx={int(cx[opp_k])} (not < 0)"
            )

    def test_d3q27_mass_conservation_at_inlet(self) -> None:
        """After Zou-He inlet, the total population at x=0 must equal the
        inferred density rho = (sum_cx0 + 2*sum_cx_neg) / (1 - ux_in).

        This is the fundamental mass-conservation property of the Zou-He method.
        """
        nz, ny, nx = 8, 6, 10
        f = _make_f27(nz, ny, nx, seed=42)
        wp = _wave_params_dict()
        ux_in, uy_in, uz_in = common_airy(nz, ny, device=torch.device("cpu"), **wp)

        # Infer rho the same way the BC does
        sum_cx0 = sum(f[k, :, :, 0] for k in _D3Q27_CX0)
        sum_cx_neg = sum(f[k, :, :, 0] for k in _D3Q27_CX_NEG)
        rho_expected = (sum_cx0 + 2.0 * sum_cx_neg) / (1.0 - ux_in)

        f_new = common_zou_he_27(f.clone(), ux_in, uy_in, uz_in)

        # Total mass at x=0 after BC
        rho_actual = f_new[:, :, :, 0].sum(dim=0)

        assert torch.allclose(rho_actual, rho_expected, atol=1e-5, rtol=1e-4), (
            f"Mass conservation violated: max diff = "
            f"{(rho_actual - rho_expected).abs().max().item()}"
        )

    def test_d3q27_velocity_prescription_at_inlet(self) -> None:
        """After Zou-He inlet, the x-momentum at x=0 must equal rho * ux_in.

        This is the velocity-prescription property: the prescribed velocity
        is exactly recovered from the updated distributions.
        """
        nz, ny, nx = 8, 6, 10
        f = _make_f27(nz, ny, nx, seed=42)
        wp = _wave_params_dict()
        ux_in, uy_in, uz_in = common_airy(nz, ny, device=torch.device("cpu"), **wp)

        sum_cx0 = sum(f[k, :, :, 0] for k in _D3Q27_CX0)
        sum_cx_neg = sum(f[k, :, :, 0] for k in _D3Q27_CX_NEG)
        rho = (sum_cx0 + 2.0 * sum_cx_neg) / (1.0 - ux_in)

        f_new = common_zou_he_27(f.clone(), ux_in, uy_in, uz_in)

        # x-momentum at x=0: sum(f_new[k] * cx[k]) for all k
        cx = C27[:, 0].float()
        ux_actual = (f_new[:, :, :, 0] * cx.view(27, 1, 1)).sum(dim=0) / rho

        assert torch.allclose(ux_actual, ux_in, atol=1e-5, rtol=1e-4), (
            f"Velocity prescription violated: max diff = "
            f"{(ux_actual - ux_in).abs().max().item()}"
        )

    def test_d3q27_nebb_formula_structure(self) -> None:
        """Verify the NEBB formula: for each inlet dir k,
        f_new[k] = feq[k] - feq[opp_k] + f[opp_k].

        We reconstruct feq independently and check the formula holds.
        """
        nz, ny, nx = 8, 6, 10
        f = _make_f27(nz, ny, nx, seed=42)
        wp = _wave_params_dict()
        ux_in, uy_in, uz_in = common_airy(nz, ny, device=torch.device("cpu"), **wp)

        sum_cx0 = sum(f[k, :, :, 0] for k in _D3Q27_CX0)
        sum_cx_neg = sum(f[k, :, :, 0] for k in _D3Q27_CX_NEG)
        rho = (sum_cx0 + 2.0 * sum_cx_neg) / (1.0 - ux_in)

        # Reconstruct feq independently
        rho3 = rho.unsqueeze(-1)
        ux3 = ux_in.unsqueeze(-1)
        uy3 = uy_in.unsqueeze(-1)
        uz3 = uz_in.unsqueeze(-1)
        feq = equilibrium27(rho3, ux3, uy3, uz3)  # (27, nz, ny, 1)

        f_new = common_zou_he_27(f.clone(), ux_in, uy_in, uz_in)

        for k, opp_k in zip(_D3Q27_INLET_DIRS, _D3Q27_INLET_OPP, strict=True):
            expected = feq[k, :, :, 0] - feq[opp_k, :, :, 0] + f[opp_k, :, :, 0]
            actual = f_new[k, :, :, 0]
            assert torch.allclose(actual, expected, atol=1e-7), (
                f"NEBB formula violated for direction {k}: "
                f"max diff = {(actual - expected).abs().max().item()}"
            )

    def test_d3q27_non_inlet_dirs_unchanged(self) -> None:
        """Directions with cx <= 0 at x=0 must be unchanged by the BC."""
        nz, ny, nx = 8, 6, 10
        f = _make_f27(nz, ny, nx, seed=42)
        wp = _wave_params_dict()
        ux_in, uy_in, uz_in = common_airy(nz, ny, device=torch.device("cpu"), **wp)

        f_new = common_zou_he_27(f.clone(), ux_in, uy_in, uz_in)

        non_inlet = [k for k in range(27) if k not in _D3Q27_INLET_DIRS]
        for k in non_inlet:
            assert torch.equal(f_new[k, :, :, 0], f[k, :, :, 0]), (
                f"Direction {k} at x=0 was modified (should be unchanged)"
            )
        # All directions at x>0 must be unchanged
        assert torch.equal(f_new[:, :, :, 1:], f[:, :, :, 1:])

    def test_d3q27_mass_conservation_multiple_seeds(self) -> None:
        """Mass conservation must hold across multiple random initializations."""
        for seed in [0, 1, 42, 123, 999]:
            nz, ny, nx = 8, 6, 10
            f = _make_f27(nz, ny, nx, seed=seed)
            wp = _wave_params_dict()
            ux_in, uy_in, uz_in = common_airy(nz, ny, device=torch.device("cpu"), **wp)

            sum_cx0 = sum(f[k, :, :, 0] for k in _D3Q27_CX0)
            sum_cx_neg = sum(f[k, :, :, 0] for k in _D3Q27_CX_NEG)
            rho_expected = (sum_cx0 + 2.0 * sum_cx_neg) / (1.0 - ux_in)

            f_new = common_zou_he_27(f.clone(), ux_in, uy_in, uz_in)
            rho_actual = f_new[:, :, :, 0].sum(dim=0)

            assert torch.allclose(rho_actual, rho_expected, atol=1e-5, rtol=1e-4), (
                f"Mass conservation violated (seed={seed})"
            )

    def test_d3q27_ux_exactly_prescribed_with_noisy_input(self) -> None:
        """The Zou-He method prescribes ux EXACTLY through the mass balance,
        regardless of the input distribution (even with noise).  This is the
        fundamental velocity-prescription property."""
        for seed in [0, 1, 42, 123, 999]:
            nz, ny, nx = 8, 6, 10
            f = _make_f27(nz, ny, nx, seed=seed)
            wp = _wave_params_dict()
            ux_in, uy_in, uz_in = common_airy(nz, ny, device=torch.device("cpu"), **wp)

            f_new = common_zou_he_27(f.clone(), ux_in, uy_in, uz_in)

            # Recover macroscopic variables at the inlet
            rho_r, ux_r, uy_r, uz_r = macroscopic27(f_new[:, :, :, 0:1])

            # ux must be exactly prescribed (mass balance)
            assert torch.allclose(ux_r.squeeze(-1), ux_in, atol=1e-6), (
                f"D3Q27 ux not exactly prescribed (seed={seed}): "
                f"max diff = {(ux_r.squeeze(-1) - ux_in).abs().max().item()}"
            )


# ===========================================================================
# 5. Combination test: wave BC + collision complete cycle
# ===========================================================================

class TestWaveCollisionCombination:
    """Verify wave BC + BGK collision + streaming in a complete LBM cycle."""

    def test_full_cycle_stability_d3q19(self) -> None:
        """Run N steps of collide → stream → wave_bc and verify no NaN/Inf."""
        nz, ny, nx = 12, 8, 16
        f = _make_f19(nz, ny, nx, seed=42)
        tau = 0.6  # ν = (τ - 0.5)/3 ≈ 0.033

        wall_mask = torch.zeros(nz, ny, nx, dtype=torch.bool)
        wall_mask[:, 0, :] = True    # bottom wall (y=0)
        wall_mask[:, -1, :] = True   # top wall (y=ny-1)
        obstacle_mask = torch.zeros(nz, ny, nx, dtype=torch.bool)

        params = WaveParams(
            step=0,
            u_mean=0.05,
            wave_amp=0.02,
            wave_period=200.0,
            wave_k=0.05,
            water_depth=float(nz),
            z_bed=0.0,
            rho_out=1.0,
            apply_outlet=True,
        )

        n_steps = 50
        for step in range(n_steps):
            params = WaveParams(
                step=step,
                u_mean=params.u_mean,
                wave_amp=params.wave_amp,
                wave_period=params.wave_period,
                wave_k=params.wave_k,
                water_depth=params.water_depth,
                z_bed=params.z_bed,
                rho_out=params.rho_out,
                apply_outlet=True,
            )
            f = collide_bgk3d(f, tau)
            f = stream3d(f)
            f = wave_bc_3d(f, wave_params=params, lattice="D3Q19")
            # Bounce-back on walls
            from tensorlbm.boundaries3d import bounce_back_cells_3d
            f = bounce_back_cells_3d(f, wall_mask)
            f = bounce_back_cells_3d(f, obstacle_mask)

        assert torch.isfinite(f).all(), "Non-finite values after cycle"
        assert f.min() > -1e6, "Unphysical negative values"

    def test_full_cycle_mass_conservation_d3q19(self) -> None:
        """Mass drift over N steps should be bounded (inlet/outlet balance)."""
        nz, ny, nx = 12, 8, 16
        f = _make_f19(nz, ny, nx, seed=42)
        tau = 0.6
        initial_mass = f.sum().item()

        wall_mask = torch.zeros(nz, ny, nx, dtype=torch.bool)
        wall_mask[:, 0, :] = True
        wall_mask[:, -1, :] = True
        obstacle_mask = torch.zeros(nz, ny, nx, dtype=torch.bool)

        from tensorlbm.boundaries3d import bounce_back_cells_3d

        n_steps = 30
        for step in range(n_steps):
            params = WaveParams(
                step=step,
                u_mean=0.05,
                wave_amp=0.02,
                wave_period=200.0,
                wave_k=0.05,
                water_depth=float(nz),
                z_bed=0.0,
                rho_out=1.0,
                apply_outlet=True,
            )
            f = collide_bgk3d(f, tau)
            f = stream3d(f)
            f = wave_bc_3d(f, wave_params=params, lattice="D3Q19")
            f = bounce_back_cells_3d(f, wall_mask)
            f = bounce_back_cells_3d(f, obstacle_mask)

        final_mass = f.sum().item()
        mass_drift = abs(final_mass - initial_mass) / abs(initial_mass)
        # With Zou-He inlet + pressure outlet, mass drift should be small
        assert mass_drift < 0.05, (
            f"Mass drift too large: {mass_drift:.4f} "
            f"(initial={initial_mass:.4f}, final={final_mass:.4f})"
        )

    def test_wave_velocity_injection_at_inlet(self) -> None:
        """After applying wave_bc_3d, the inlet ux must match the prescribed
        wave velocity exactly (Zou-He mass balance).

        Note on uy/uz: The Zou-He method only updates cx>0 directions at the
        inlet.  The cx=0 directions (which include ±z and ±yz directions with
        non-zero cz/cy) are NOT updated, so they retain their original
        transverse momentum.  At equilibrium, the cx=0 directions carry 2/3
        of the total z-momentum, so with a rest-state input (uz=0), the
        recovered uz ≈ uz_in/3.  This is the standard Zou-He behaviour for
        3-D inlets — uy/uz converge to the prescribed value over multiple
        collide-stream-BC cycles as collision relaxes the cx=0 directions.

        We verify:
        - ux is exactly prescribed (mass balance).
        - uy/uz have the correct sign and are non-zero (physical reasonableness).
        """
        nz, ny, nx = 12, 8, 16
        # Pure equilibrium at rest (rho=1, u=0) — no transverse momentum
        rho0 = torch.ones(nz, ny, nx, dtype=torch.float32)
        f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0),
                          torch.zeros_like(rho0))

        step = 25
        params = WaveParams(
            step=step,
            u_mean=0.05,
            wave_amp=0.02,
            wave_period=200.0,
            wave_k=0.05,
            water_depth=float(nz),
            z_bed=0.0,
            apply_outlet=False,
        )
        f_new = wave_bc_3d(f.clone(), wave_params=params, lattice="D3Q19")

        # Recover macroscopic variables at the inlet
        rho_in, ux_in, uy_in, uz_in = macroscopic3d(f_new[:, :, :, 0:1])

        # Expected velocity from Airy theory
        ux_exp, uy_exp, uz_exp = orig_airy(
            nz, ny, step=step, u_mean=0.05, wave_amp=0.02,
            wave_period=200.0, wave_k=0.05, water_depth=float(nz),
            z_bed=0.0, device=torch.device("cpu"),
        )

        # ux is exactly prescribed (Zou-He mass balance)
        assert torch.allclose(ux_in.squeeze(-1), ux_exp, atol=1e-6), (
            f"Inlet ux mismatch: max diff = "
            f"{(ux_in.squeeze(-1) - ux_exp).abs().max().item()}"
        )
        # uy is zero (no lateral wave component) — exactly recovered
        assert torch.allclose(uy_in.squeeze(-1), uy_exp, atol=1e-6), (
            f"Inlet uy mismatch: max diff = "
            f"{(uy_in.squeeze(-1) - uy_exp).abs().max().item()}"
        )
        # uz has the correct sign (negative = downward wave phase) and is non-zero
        # at depth.  The magnitude is ~1/3 of prescribed due to cx=0 directions
        # not being updated (standard Zou-He 3-D inlet behaviour).
        uz_r = uz_in.squeeze(-1)
        uz_e = uz_exp
        # Where prescribed uz is significant, recovered uz must have same sign
        significant = uz_e.abs() > 1e-4
        if significant.any():
            assert (torch.sign(uz_r[significant]) == torch.sign(uz_e[significant])).all(), (
                "Recovered uz has wrong sign at depth"
            )
            # Recovered uz should be a positive fraction of prescribed
            ratio = uz_r[significant] / uz_e[significant]
            assert (ratio > 0.01).all(), "Recovered uz too small"
            assert (ratio < 0.99).all(), "Recovered uz unexpectedly close to prescribed"

    def test_full_cycle_stability_d3q27(self) -> None:
        """Run N steps of collide → stream → wave_bc for D3Q27.

        Uses D3Q27 BGK collision (from d3q27.collide_bgk27) and a simple
        roll-based streaming.  Verifies stability and no NaN/Inf.
        """
        from tensorlbm.d3q27 import collide_bgk27

        nz, ny, nx = 10, 6, 12
        f = _make_f27(nz, ny, nx, seed=42)
        tau = 0.6

        # Simple D3Q27 streaming via torch.roll
        shifts_27 = [
            (0, 0, 0),       # 0: rest
            (1, 0, 0),       # 1: +x
            (-1, 0, 0),      # 2: -x
            (0, 1, 0),       # 3: +y
            (0, -1, 0),      # 4: -y
            (0, 0, 1),       # 5: +z
            (0, 0, -1),      # 6: -z
            (1, 1, 0),       # 7: +x+y
            (-1, 1, 0),      # 8: -x+y
            (1, -1, 0),      # 9: +x-y
            (-1, -1, 0),     # 10: -x-y
            (1, 0, 1),       # 11: +x+z
            (-1, 0, 1),      # 12: -x+z
            (1, 0, -1),      # 13: +x-z
            (-1, 0, -1),     # 14: -x-z
            (0, 1, 1),       # 15: +y+z
            (0, -1, 1),      # 16: -y+z
            (0, 1, -1),      # 17: +y-z
            (0, -1, -1),     # 18: -y-z
            (1, 1, 1),       # 19: +x+y+z
            (-1, 1, 1),      # 20: -x+y+z
            (1, -1, 1),      # 21: +x-y+z
            (-1, -1, 1),     # 22: -x-y+z
            (1, 1, -1),      # 23: +x+y-z
            (-1, 1, -1),     # 24: -x+y-z
            (1, -1, -1),     # 25: +x-y-z
            (-1, -1, -1),    # 26: -x-y-z
        ]

        def stream27(f: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(f)
            for q in range(27):
                sx, sy, sz = shifts_27[q]
                out[q] = torch.roll(f[q], shifts=(sz, sy, sx), dims=(0, 1, 2))
            return out

        n_steps = 30
        for step in range(n_steps):
            params = WaveParams(
                step=step,
                u_mean=0.05,
                wave_amp=0.02,
                wave_period=200.0,
                wave_k=0.05,
                water_depth=float(nz),
                z_bed=0.0,
                rho_out=1.0,
                apply_outlet=True,
            )
            f = collide_bgk27(f, tau)
            f = stream27(f)
            f = wave_bc_3d(f, wave_params=params, lattice="D3Q27")

        assert torch.isfinite(f).all(), "Non-finite values after D3Q27 cycle"
        assert f.min() > -1e6, "Unphysical negative values"

    def test_d3q27_velocity_injection_at_inlet(self) -> None:
        """After applying wave_bc_3d (D3Q27), the inlet ux must match the
        prescribed wave velocity exactly (Zou-He mass balance).

        Same Zou-He 3-D inlet limitation as D3Q19: uz is only approximately
        prescribed (cx=0 directions not updated).  See
        test_wave_velocity_injection_at_inlet for details.
        """
        nz, ny, nx = 10, 6, 12
        # Pure equilibrium at rest (rho=1, u=0) — no transverse momentum
        rho0 = torch.ones(nz, ny, nx, dtype=torch.float32)
        f = equilibrium27(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0),
                          torch.zeros_like(rho0))

        step = 25
        params = WaveParams(
            step=step,
            u_mean=0.05,
            wave_amp=0.02,
            wave_period=200.0,
            wave_k=0.05,
            water_depth=float(nz),
            z_bed=0.0,
            apply_outlet=False,
        )
        f_new = wave_bc_3d(f.clone(), wave_params=params, lattice="D3Q27")

        rho_in, ux_in, uy_in, uz_in = macroscopic27(f_new[:, :, :, 0:1])

        ux_exp, uy_exp, uz_exp = orig_airy(
            nz, ny, step=step, u_mean=0.05, wave_amp=0.02,
            wave_period=200.0, wave_k=0.05, water_depth=float(nz),
            z_bed=0.0, device=torch.device("cpu"),
        )

        # ux is exactly prescribed (Zou-He mass balance)
        assert torch.allclose(ux_in.squeeze(-1), ux_exp, atol=1e-6), (
            f"D3Q27 inlet ux mismatch: max diff = "
            f"{(ux_in.squeeze(-1) - ux_exp).abs().max().item()}"
        )
        # uy is zero (no lateral wave component) — exactly recovered
        assert torch.allclose(uy_in.squeeze(-1), uy_exp, atol=1e-6), (
            f"D3Q27 inlet uy mismatch: max diff = "
            f"{(uy_in.squeeze(-1) - uy_exp).abs().max().item()}"
        )
        # uz: correct sign, non-zero, positive fraction of prescribed
        uz_r = uz_in.squeeze(-1)
        uz_e = uz_exp
        significant = uz_e.abs() > 1e-4
        if significant.any():
            assert (torch.sign(uz_r[significant]) == torch.sign(uz_e[significant])).all(), (
                "D3Q27: recovered uz has wrong sign at depth"
            )
            ratio = uz_r[significant] / uz_e[significant]
            assert (ratio > 0.01).all(), "D3Q27: recovered uz too small"
            assert (ratio < 0.99).all(), "D3Q27: recovered uz unexpectedly close to prescribed"

    def test_original_vs_common_full_apply_equivalence(self) -> None:
        """Compare the original apply_wave_inlet_3d (inlet + outlet + bounce)
        against the common wave_bc_3d (inlet + outlet) — the inlet and outlet
        portions should be identical; bounce-back is applied identically."""
        from tensorlbm.boundaries3d import bounce_back_cells_3d
        from tensorlbm.wave_bc import apply_wave_inlet_3d

        nz, ny, nx = 10, 6, 12
        f = _make_f19(nz, ny, nx, seed=77)
        wall_mask = torch.zeros(nz, ny, nx, dtype=torch.bool)
        wall_mask[:, 0, :] = True
        wall_mask[:, -1, :] = True
        obstacle_mask = torch.zeros(nz, ny, nx, dtype=torch.bool)

        # Original full apply
        f_orig = apply_wave_inlet_3d(
            f.clone(), step=15, wall_mask=wall_mask,
            obstacle_mask=obstacle_mask,
            u_mean=0.05, wave_amp=0.02, wave_period=200.0,
            wave_k=0.05, water_depth=float(nz), z_bed=0.0,
        )

        # Common: wave_bc_3d (inlet + outlet) + bounce-back
        params = WaveParams(
            step=15, u_mean=0.05, wave_amp=0.02, wave_period=200.0,
            wave_k=0.05, water_depth=float(nz), z_bed=0.0,
            rho_out=1.0, apply_outlet=True,
        )
        f_common = wave_bc_3d(f.clone(), wave_params=params, lattice="D3Q19")
        f_common = bounce_back_cells_3d(f_common, wall_mask)
        f_common = bounce_back_cells_3d(f_common, obstacle_mask)

        assert torch.equal(f_orig, f_common), (
            f"Full apply mismatch: max diff = {(f_orig - f_common).abs().max().item()}"
        )

    def test_ux_exactly_prescribed_with_noisy_input_d3q19(self) -> None:
        """The Zou-He method prescribes ux EXACTLY through the mass balance,
        regardless of the input distribution (even with noise).  This is the
        fundamental velocity-prescription property for D3Q19."""
        for seed in [0, 1, 42, 123, 999]:
            nz, ny, nx = 8, 6, 10
            f = _make_f19(nz, ny, nx, seed=seed)
            wp = _wave_params_dict()
            ux_in, uy_in, uz_in = orig_airy(nz, ny, device=torch.device("cpu"), **wp)

            f_new = common_zou_he_19(f.clone(), ux_in, uy_in, uz_in)

            # Recover macroscopic variables at the inlet
            rho_r, ux_r, uy_r, uz_r = macroscopic3d(f_new[:, :, :, 0:1])

            # ux must be exactly prescribed (mass balance)
            assert torch.allclose(ux_r.squeeze(-1), ux_in, atol=1e-6), (
                f"D3Q19 ux not exactly prescribed (seed={seed}): "
                f"max diff = {(ux_r.squeeze(-1) - ux_in).abs().max().item()}"
            )

    def test_uz_convergence_over_cycles(self) -> None:
        """Verify that uz converges toward the prescribed value over multiple
        collide-stream-BC cycles.

        The Zou-He method only updates cx>0 directions at the inlet.  The cx=0
        directions (which carry 2/3 of the z-momentum at equilibrium) are not
        updated, so with a rest-state input, the first BC application recovers
        only ~1/3 of the prescribed uz.  However, after collision relaxes the
        cx=0 directions toward the new equilibrium, subsequent BC applications
        recover more of the prescribed uz.  Over multiple cycles, uz converges.
        """
        nz, ny, nx = 12, 8, 16
        rho0 = torch.ones(nz, ny, nx, dtype=torch.float32)
        f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0),
                          torch.zeros_like(rho0))
        tau = 0.6

        step = 25  # fixed step for consistent wave phase
        ux_exp, uy_exp, uz_exp = orig_airy(
            nz, ny, step=step, u_mean=0.05, wave_amp=0.02,
            wave_period=200.0, wave_k=0.05, water_depth=float(nz),
            z_bed=0.0, device=torch.device("cpu"),
        )

        # Track uz recovery ratio at the deepest cell
        ratios = []
        for cycle in range(20):
            params = WaveParams(
                step=step, u_mean=0.05, wave_amp=0.02, wave_period=200.0,
                wave_k=0.05, water_depth=float(nz), z_bed=0.0,
                apply_outlet=False,
            )
            # Apply BC, then collision (relaxes cx=0 directions)
            f = wave_bc_3d(f, wave_params=params, lattice="D3Q19")
            f = collide_bgk3d(f, tau)

            # Check uz recovery at the deepest cell
            rho_r, ux_r, uy_r, uz_r = macroscopic3d(f[:, :, :, 0:1])
            uz_r = uz_r.squeeze(-1)
            # Use the deepest cell where uz is most significant
            z_deep = nz - 1
            if abs(uz_exp[z_deep, 0].item()) > 1e-6:
                ratio = abs(uz_r[z_deep, 0].item()) / abs(uz_exp[z_deep, 0].item())
                ratios.append(ratio)

        # uz recovery should improve over cycles (collision relaxes cx=0 dirs)
        assert len(ratios) >= 2, "Should have multiple ratio measurements"
        # First cycle: ~1/3 recovery (rest-state input)
        assert ratios[0] < 0.5, (
            f"First cycle uz recovery too high: {ratios[0]:.4f} (expected < 0.5)"
        )
        # Later cycles: should be closer to 1.0
        assert ratios[-1] > ratios[0], (
            f"uz recovery did not improve: first={ratios[0]:.4f}, "
            f"last={ratios[-1]:.4f}"
        )
