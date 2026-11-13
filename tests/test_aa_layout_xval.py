"""Cross-validation tests for the 2b-t/LB-t A-A layout pattern port.

Anchor ``XVAL-LBM-NO-MULTIPHASE-LAYOUT-PATTERN-ONLY``.  The reference is the
JSON emitted by ``benchmarks/xval_lbt_layout_pattern/lbt_ref_dump.cpp``, i.e.
by the *unmodified* upstream headers, so every assertion below is bound to OSS
executable evidence rather than to a re-derivation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_lbt_layout_pattern"))

from tensorlbm.aa_layout import (  # noqa: E402
    D2Q9_DX,
    D2Q9_DY,
    D2Q9_MASK,
    D2Q9_ND,
    D2Q9_OFF,
    D2Q9_W,
    SLOT_TO_TL,
    TL_TO_SLOT,
    AaLayout,
    AaPopulation,
    equilibrium_d2q9,
)

REF_FILE = ROOT / "artifacts" / "xval_lbt_layout" / "lbt_ref_layout.json"


def _ref() -> dict:
    if not REF_FILE.is_file():
        pytest.skip(f"upstream reference dump missing: {REF_FILE}")
    return json.loads(REF_FILE.read_text())


def test_lattice_constants_match_upstream():
    ref = _ref()["npop1"]
    assert ref["ND"] == D2Q9_ND == 10
    assert ref["OFF"] == D2Q9_OFF == 5
    assert np.allclose(ref["DX"], D2Q9_DX)
    assert np.allclose(ref["DY"], D2Q9_DY)
    assert np.allclose(ref["W"], D2Q9_W)
    assert [bool(b) for b in ref["MASK"]] == D2Q9_MASK.tolist()


@pytest.mark.parametrize("key,npop", [("npop1", 1), ("npop2", 2)])
def test_index_formula_bit_exact_against_oss(key, npop):
    ref = _ref()[key]
    layout = AaLayout(nx=ref["NX"], ny=ref["NY"], nz=ref["NZ"], npop=ref["NP"])
    assert layout.npop == npop
    for x, y, z, p, n, d, idx in ref["spatial_to_linear"]:
        assert layout.spatial_to_linear(x, y, z, n, d, p) == idx
        assert layout.linear_to_spatial(idx) == (x, y, z, n, d, p)
    assert layout.total_slots == ref["total_slots"]


@pytest.mark.parametrize("key", ["npop1", "npop2"])
def test_parity_accessors_bit_exact_against_oss(key):
    ref = _ref()[key]
    layout = AaLayout(nx=ref["NX"], ny=ref["NY"], nz=ref["NZ"], npop=ref["NP"])
    for x, y, p, n, d, r_even, r_odd, w_even, w_odd in ref["parity"]:
        assert layout.index_read(0, x, y, 0, n, d, p) == r_even
        assert layout.index_read(1, x, y, 0, n, d, p) == r_odd
        assert layout.index_write(0, x, y, 0, n, d, p) == w_even
        assert layout.index_write(1, x, y, 0, n, d, p) == w_odd


def test_write_then_read_opposite_parity_is_identity():
    """Upstream's own invariant (population_base_test.hpp: write<TS> -> read<!TS>).

    Reproducing it documents *why* the OSS pattern alone cannot transport mass:
    the one-cell offset present in both accessors cancels, so a collide-stream
    kernel still has to place the populations itself.
    """
    layout = AaLayout(nx=6, ny=5, npop=2)
    for x in range(6):
        for y in range(5):
            for n in (0, 1):
                for d in range(D2Q9_OFF):
                    for p in range(2):
                        assert layout.index_write(0, x, y, 0, n, d, p) == layout.index_read(1, x, y, 0, n, d, p)
                        assert layout.index_write(1, x, y, 0, n, d, p) == layout.index_read(0, x, y, 0, n, d, p)


def test_npop_stacking_has_no_crosstalk():
    pop = AaPopulation(nx=4, ny=3, npop=2)
    rng = np.random.default_rng(7)
    f = rng.normal(size=(1, 3, 4, 2, 9))
    pop.from_dense(f)
    assert np.allclose(pop.to_dense(), f)
    assert np.allclose(pop.to_dense()[..., 0, :], f[..., 0, :])
    rho, ux, uy = pop.macroscopic()
    assert rho.shape == (1, 3, 4, 2)


def test_dense_roundtrip_drops_no_masked_slot():
    pop = AaPopulation(nx=5, ny=4, npop=1)
    f = np.zeros((1, 4, 5, 1, 9))
    f[..., 0] = 1.0
    pop.from_dense(f)
    assert np.allclose(pop.to_dense(), f)
    # slot 5 is the upstream padding slot (MASK == 0): never part of the layout
    assert SLOT_TO_TL[5] == -1
    assert TL_TO_SLOT.tolist() == [0, 1, 2, 6, 7, 3, 4, 8, 9]


def test_poiseuille_matches_analytic_and_production_kernels():
    """Physics gate: analytic Poiseuille (<15 %) and storage transparency."""
    import run_xval

    ny, nx, tau, force = 9, 4, 0.8, 2e-5
    nu = (tau - 0.5) / 3.0
    aa = run_xval.run_aa(nx, ny, tau, force, 3000, sample_every=100)
    dense = run_xval.run_dense(nx, ny, tau, force, 3000, sample_every=100)
    prof = np.array(aa["profile"])
    ana = run_xval.analytic_poiseuille(ny, force, nu)
    err = abs(prof.max() - ana.max()) / ana.max()
    assert err < 0.15, f"analytic Poiseuille error {err:.2%} exceeds the 15 % gate"
    linf, _ = run_xval.profile_error(np.array(dense["profile"]), prof)
    assert linf < 1e-10, f"layout is not transparent to the production kernels: {linf:.2e}"


def test_production_simulation_class_runs_on_lbt_layout():
    """Integration: ``LBMSimulation(layout="lbt_aa")`` == ``layout="dense"``.

    A density perturbation is seeded in both storages (a rest state would be a
    vacuous comparison) and twenty steps are advanced.
    """
    import torch

    from tensorlbm import LBMSimulation

    nx, ny, tau = 16, 8, 0.8
    dense = LBMSimulation(nx=nx, ny=ny, tau=tau, layout="dense")
    aa = LBMSimulation(nx=nx, ny=ny, tau=tau, layout="lbt_aa")

    eps = 0.01 * torch.sin(torch.arange(nx, dtype=torch.float32) * 2 * torch.pi / nx)
    eps = eps.view(1, nx, 1)
    dense.f = dense.f * (1.0 + eps.to(dense.f.dtype))
    f_np = dense.f.numpy()[None, :, :, None, :].astype(np.float64)
    aa._aa.from_dense(f_np)
    aa.f = aa._aa.to_dense()[0, :, :, 0, :]

    for _ in range(20):
        dense.step()
        aa.step()

    rho_d, u_d = dense.macroscopic()
    rho_a, u_a = aa.macroscopic()
    assert float((rho_d - rho_a).abs().max()) < 1e-5
    assert float((u_d - u_a).abs().max()) < 1e-5
    assert float(u_a.abs().max()) > 1e-3, "perturbation vanished - comparison would be vacuous"
