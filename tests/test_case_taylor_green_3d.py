"""CASE-TAYLOR-GREEN-3D: the case artifact runs, and its exactness claim is machine-checked.

Two cheap guards on *small* grids (the production sweep lives in
``benchmarks/case_taylor_green_3d/run_case.py --all``):

* the reference is really an exact Navier-Stokes solution (spectral curl /
  Laplacian / momentum residual ~ 1e-14), while the z-dependent vortex is
  provably *not* (non-zero curl -> no closed form);
* a 200-step decay run reproduces the reference inside the case tolerance
  (8%) on every predeclared metric, and conserves mass to 1e-12.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "benchmarks" / "case_taylor_green_3d" / "run_case.py"
SRC = REPO / "src"

TOL_PCT = 8.0


def _load():
    for p in (str(SRC), str(MODULE_PATH.parent)):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("tg3d_case_run", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rc():
    torch.set_num_threads(2)
    return _load()


def test_reference_is_an_exact_navier_stokes_solution(rc):
    v = rc.verify(n=16, amp=2e-3)
    # z-invariant Taylor-Green: convective term is a pure gradient, NS residual 0
    assert v["tg"]["convective_term_is_pure_gradient"] is True
    assert v["tg"]["curl_of_convective_term_over_norm"] < 1e-10
    assert v["tg"]["laplacian_identity_rel_residual"] < 1e-10
    assert v["tg"]["ns_momentum_residual_rel"] < 1e-10
    # Beltrami / ABC: curl(u) = k u exactly, same exactness chain
    assert v["abc_beltrami_rel_residual"] < 1e-10
    assert v["abc"]["convective_term_is_pure_gradient"] is True
    assert v["abc"]["ns_momentum_residual_rel"] < 1e-10
    # classic z-dependent vortex: no closed form (this is why it is not the reference)
    assert v["tg3d"]["convective_term_is_pure_gradient"] is False
    assert v["tg3d"]["curl_of_convective_term_over_norm"] > 1e-2


@pytest.mark.parametrize("n", [16, 24])
def test_decay_matches_exact_solution(rc, n):
    r = rc.run_decay("tg", n, tau=0.6, amp=2e-3, steps=200, sample=50)
    assert r["finite"] is True
    assert abs(r["M1_energy_decay_rel_err_pct"]) < TOL_PCT, r["M1_energy_decay_rel_err_pct"]
    assert abs(r["M2_decay_exponent_rel_err_pct"]) < TOL_PCT, r["M2_decay_exponent_rel_err_pct"]
    assert r["M3_l2_field_rel_err_pct"] < TOL_PCT, r["M3_l2_field_rel_err_pct"]
    assert r["M5_mass_drift_rel"] < 1e-12
    # the energy must actually decay
    assert r["E_final"] < r["E0"]


def test_zdependent_beltrami_companion(rc):
    r = rc.run_decay("abc", 16, tau=0.6, amp=1e-2, steps=200, sample=50)
    assert r["finite"] is True
    assert abs(r["M1_energy_decay_rel_err_pct"]) < TOL_PCT
    assert r["M3_l2_field_rel_err_pct"] < TOL_PCT