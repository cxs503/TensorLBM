"""Guards for the R2 conditioning / Korteweg audit of the SC Laplace calibration.

The audit script (`xval/XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION/audit_r2/
run_audit.py`) takes ~95 s on 6 threads, so the tests here split in two:

* **analytic / cheap (always run)** - the conditioning decomposition and its
  derivatives against finite differences, the tanh-profile integral that the
  Korteweg readout relies on, and the Korteweg prefactor that the module's
  code and docstring state differently;
* **artifact regression (skipped when the artifact is absent)** - the stored
  numbers of the audit run must keep the properties the report claims: radius
  consistency of `dp*R` below 3 %, the fit within 3 % of the verified benchmark
  `sigma = 0.05611`, a *systematic* Korteweg deficit in 25-40 %, and the same
  deficit at the off-benchmark coupling `G_lib = 4.7` (i.e. the deficit is a
  truncation property, not a per-case accident).
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys_path = str(REPO / "src")
import sys

if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

AUDIT_DIR = REPO / "xval" / "XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION" / "audit_r2"
ARTIFACT = AUDIT_DIR / "artifacts" / "audit_r2.json"


def _audit_module():
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("audit_r2_run", AUDIT_DIR / "run_audit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# conditioning (analytic)
# ---------------------------------------------------------------------------
def test_closure_decomposition_reproduces_measured_dp():
    a = _audit_module()
    for (ri, ro, g) in [(1.957, 0.1596, -5.0), (1.7061, 0.21063, -4.7)]:
        c = a.conditioning(ri, ro, g)
        direct = a.p_eos(ri, g) - a.p_eos(ro, g)
        assert abs(c["dp_from_closure"] - direct) < 1e-12
        assert abs(c["A_density_term"] + c["B_interaction_term"] - direct) < 1e-12
        # attraction: the interaction term is negative and nearly cancels A
        assert c["B_interaction_term"] < 0.0
        # ... which is exactly the conditioning problem: kappa >> 1
        assert c["condition_number_kappa"] > 100.0
        assert c["dp_over_p_mean"] < 0.2


def test_conditioning_sensitivity_matches_finite_difference():
    a = _audit_module()
    ri, ro, g = 1.957, 0.1596, -5.0
    c = a.conditioning(ri, ro, g)
    h = 1e-6
    fd_in = (a.p_eos(ri + h, g) - a.p_eos(ri - h, g)) / (2 * h)
    # dp/drho_out enters with a minus sign, because Delta p = p_in - p_out
    fd_out = -(a.p_eos(ro + h, g) - a.p_eos(ro - h, g)) / (2 * h)
    assert abs(fd_in - c["ddp_drho_in"]) < 1e-5
    assert abs(fd_out - c["ddp_drho_out"]) < 1e-5
    # a 1e-3 density bias already moves sigma by several percent
    assert c["dsigma_rel_per_1e-3_rho_in"] > 0.02


def test_condition_number_grows_with_radius_of_the_jump():
    """dp shrinks with R at fixed model parameters -> kappa must grow."""
    a = _audit_module()
    small = a.conditioning(1.9718, 0.16186, -5.0)   # R ~ 12
    large = a.conditioning(1.9491, 0.15868, -5.0)   # R ~ 30
    assert large["condition_number_kappa"] > small["condition_number_kappa"]


# ---------------------------------------------------------------------------
# Korteweg readout: integrator and prefactor
# ---------------------------------------------------------------------------
def test_tanh_profile_integral_matches_analytic():
    """`Integral (d psi/dn)^2 dn` for psi = A tanh(n/(2 xi)) is 2 A^2/(3 xi).

    (d psi/dn)^2 = A^2/(4 xi^2) sech^4(n/(2 xi)) and Integral sech^4 = 4/3,
    so the integral is A^2/(4 xi^2) * (8 xi/3) = 2 A^2/(3 xi).
    """
    torch = pytest.importorskip("torch")
    A, xi = 0.35, 1.5
    n = torch.linspace(-60.0, 60.0, 240001, dtype=torch.float64)
    ps = A * torch.tanh(n / (2 * xi))
    dn = float(n[1] - n[0])
    integ = float(((ps[1:] - ps[:-1]) / dn).pow(2).sum() * dn)
    expect = 2.0 * A * A / (3.0 * xi)
    assert abs(integ - expect) / expect < 1e-4


def test_korteweg_prefactor_code_matches_discrete_derivation():
    """The module code uses |G cs^2 kappa3|; the old eq. (6) text dropped cs^2.

    Discrete derivation (sum_i w_i c_a psi(x+c) = cs^2 d_a psi + (cs^4/2)
    d_a lap psi) gives kappa3 = cs^2/2 = 1/6 and K = G_eff cs^2 kappa3, so the
    documented `|G_eff| kappa3` was low by a factor 1/cs^2 = 3.
    """
    from tensorlbm import sc_surface_tension as sst

    st = sst.build_stencil("d2q9")
    kappa3 = float(st["kappa3"])
    assert abs(kappa3 - 1.0 / 6.0) < 1e-12
    g = -5.0
    k_code = abs(g * (1.0 / 3.0) * kappa3)
    k_doc = abs(g * kappa3)
    assert abs(k_doc / k_code - 3.0) < 1e-12


# ---------------------------------------------------------------------------
# artifact regression on the audit run
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not ARTIFACT.exists(), reason="audit artifact not produced")
def test_audit_laplace_fit_reproduces_verified_benchmark():
    d = json.loads(ARTIFACT.read_text())
    fit = d["laplace_fit"]
    assert fit["n_radii"] >= 3
    assert fit["rel_dev_max"] < 0.03, fit["rel_dev_per_radius"]
    assert abs(fit["sigma_fit"] / d["sigma_reference_verified_benchmark"] - 1.0) < 0.03


@pytest.mark.skipif(not ARTIFACT.exists(), reason="audit artifact not produced")
def test_audit_conditioning_is_recorded_and_large():
    d = json.loads(ARTIFACT.read_text())
    for c in d["cases"]:
        q = c["conditioning"]
        assert q["condition_number_kappa"] > 200.0
        assert q["dp_over_p_mean"] < 0.15
        assert c["converged"] is True
        assert c["mass_drift_rel"] < 1e-9


@pytest.mark.skipif(not ARTIFACT.exists(), reason="audit artifact not produced")
def test_audit_korteweg_deficit_is_systematic_and_negative():
    """sigma_K (module prefactor) undershoots dp*R by 25-40 % at *every* case,
    including the off-benchmark coupling -- i.e. it is a truncation offset, not
    a per-case accident, and it must not be quoted as an accuracy anchor."""
    d = json.loads(ARTIFACT.read_text())
    deficits = [-x["code_prefactor_pct"] for x in d["korteweg_vs_laplace_pct"]]
    assert all(25.0 <= f <= 40.0 for f in deficits), deficits
    assert max(deficits) - min(deficits) < 10.0, deficits
    # the documented (cs^2-less) prefactor overestimates by ~2x -> never usable
    over = [x["doc_prefactor_pct"] for x in d["korteweg_vs_laplace_pct"]]
    assert all(o > 80.0 for o in over), over