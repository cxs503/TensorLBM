"""Consistency between prose collision claims, the kernel registry and the evidence ledger.

Anchor: LBM-P0-KBC (m23154).  The point of these tests is that a documentation
claim can never again drift away from the registered, measured implementation
without a test failing.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))

import collision_claim_registry as ccr  # noqa: E402


def test_registry_kernels_import_and_conserve():
    """R3: every AVAILABLE registry cell must exist and conserve mass/momentum."""
    matrix = ccr.registry_matrix()
    n_cells = 0
    for lattice, row in matrix.items():
        for family, cell in row.items():
            if not cell["available"]:
                continue
            n_cells += 1
            fn = ccr.resolve_entrypoint(cell["entrypoint"])
            assert callable(fn), (lattice, family, cell["entrypoint"])
            smoke = ccr.smoke_collision(lattice, cell["entrypoint"])
            assert smoke["mass_err_rel"] <= ccr.CONSERVATION_TOL, (lattice, family, smoke)
            assert smoke["momentum_err_rel"] <= ccr.CONSERVATION_TOL, (lattice, family, smoke)
    assert n_cells >= 14


def test_shear_relaxation_is_tau_controlled():
    """Every registered kernel must relax deviatoric stress by (1 - 1/tau)."""
    for lattice, row in ccr.registry_matrix().items():
        for family, cell in row.items():
            if not cell["available"]:
                continue
            smoke = ccr.smoke_collision(lattice, cell["entrypoint"], tau=0.9)
            assert abs(smoke["shear_factor"] - smoke["shear_factor_expected"]) <= ccr.SHEAR_TOL, (
                lattice,
                family,
                smoke,
            )


def test_claim_extraction_finds_kbc_and_cumulant_surfaces():
    claims = ccr.extract_claims()
    fams = {fam for c in claims for fam in c.families}
    assert {"CUMULANT", "KBC"} <= fams
    assert any(c.source.startswith("README") for c in claims)


def test_unbacked_claim_is_flagged(monkeypatch):
    """R1: a claim whose family is not graded REAL must fail the check."""
    ledger = ccr.build_ledger()
    for rec in ledger["families"].values():
        rec["grade"] = "WITHHELD"
    res = ccr.check(ledger)
    assert not res["ok"]
    assert any(f["gate"] == "R1" for f in res["fails"])


def test_stale_withholding_is_flagged(monkeypatch):
    """R2: a WITHHELD doc line for a family graded REAL must fail the check."""

    class FakeClaim:
        def __init__(self, source, line, text, families, withheld):
            self.source, self.line, self.text = source, line, text
            self.families, self.withheld = families, withheld

        def as_dict(self):
            return dict(source=self.source, line=self.line, text=self.text,
                        families=self.families, withheld=self.withheld)

    src = ccr.KERNEL_LEVEL_SOURCES[0]
    fake = [FakeClaim(src, 1, "KBC: WITHHELD_NO_D3Q19_KBC_KERNEL", ["KBC"], True)]
    ledger = ccr.build_ledger()
    for rec in ledger["families"].values():
        rec["grade"] = "REAL"
    monkeypatch.setattr(ccr, "extract_claims", lambda: fake)
    res = ccr.check(ledger)
    assert any(f["gate"] == "R2" and f["family"] == "KBC" for f in res["fails"])


def test_ledger_file_is_regenerable_and_matches_grades():
    """The committed ledger must be reproducible from the current sources."""
    if not os.path.exists(ccr.LEDGER):
        pytest.skip("ledger not generated yet")
    with open(ccr.LEDGER, encoding="utf-8") as fh:
        committed = json.load(fh)
    fresh = ccr.build_ledger()
    assert set(fresh["families"]) == set(committed["families"])
    for fam in fresh["families"]:
        assert fresh["families"][fam]["grade"] == committed["families"][fam]["grade"], fam
