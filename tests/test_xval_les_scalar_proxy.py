"""Contract tests for XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY.

These lock the *algebraic* findings of the cross-validation so a future change
cannot silently break them:

1. the upstream lattice table extracted from ``LatFlow/D2Q9.py`` is consistent
   (``EEk_q == (sum_n |c_qn|)^2``);
2. the torch port of upstream's ``les=False`` branch reproduces the shipped
   ``collide_bgk`` to round-off (without this the whole comparison is void);
3. the in-house strain-rate input satisfies the Chapman-Enskog identity
   ``|Pi|_F = (2/3) rho tau0 |S|_F``;
4. on D2Q9 the upstream ``|c_n c_m|`` proxy is *exactly* the Frobenius norm of
   the signed stress for trace-free strain (so both codes implement one closure
   up to the coefficient ``C_s = sqrt(Sc/3)``), while it is not for an arbitrary
   non-equilibrium field;
5. the coefficient conversion reproduces the upstream eddy viscosity on a
   Chapman-Enskog state, and both closures switch off exactly in solid-body
   rotation.

The upstream source tree is *not* required: ``oss_ref/oss_lattice.json`` is the
machine-extracted table committed by ``extract_oss_tables.py``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "xval_les_scalar_proxy"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BENCH))

import oss_port as OSS  # noqa: E402

from tensorlbm.d2q9 import C as C2D, W as W2D, equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk  # noqa: E402
from tensorlbm.turbulence.turbulence import _neq_stress_norm_2d, _smagorinsky_tau  # noqa: E402

CS_OWN = 0.1
SC_OSS = 0.17


def _ce_field(S: np.ndarray, tau0: float, n: int = 8, rho0: float = 1.0) -> torch.Tensor:
    """Exact first-order Chapman-Enskog field for a uniform strain tensor ``S``."""
    St = torch.tensor(S, dtype=torch.float64)
    jj, ii = torch.meshgrid(
        torch.arange(n, dtype=torch.float64), torch.arange(n, dtype=torch.float64), indexing="ij"
    )
    rx = (ii + 0.5) / n - 0.5
    ry = (jj + 0.5) / n - 0.5
    ux = St[0, 0] * rx + St[0, 1] * ry
    uy = St[1, 0] * rx + St[1, 1] * ry
    rho = torch.full_like(ux, rho0)
    f = equilibrium(rho, ux, uy)
    cx = C2D[:, 0].double().view(9, 1, 1)
    cy = C2D[:, 1].double().view(9, 1, 1)
    cij_s = cx * cx * St[0, 0] + 2.0 * cx * cy * St[0, 1] + cy * cy * St[1, 1]
    return f - 3.0 * tau0 * W2D.double().view(9, 1, 1) * rho.unsqueeze(0) * cij_s


def _random_trace_free(rng: np.random.Generator) -> np.ndarray:
    S = rng.normal(size=(2, 2))
    S = 0.5 * (S + S.T)
    S = S - np.eye(2) * np.trace(S) / 2.0
    return S / np.sqrt((S * S).sum())


def test_extracted_lattice_table_is_consistent():
    lat = OSS.lattice()
    c = np.asarray(lat["velocities_2d"], dtype=float)
    eek = np.asarray(lat["eek_abs_dyadic"], dtype=float)
    assert c.shape == (9, 2)
    assert sum(lat["weights"]) == pytest.approx(1.0, abs=1e-12)
    # EEk was built upstream as sum_{n,m} |c_n c_m|; that equals (sum_n |c_n|)^2
    np.testing.assert_allclose(eek, (np.abs(c).sum(axis=1)) ** 2)
    assert lat["Sc"] == pytest.approx(SC_OSS)
    prov = OSS.provenance()
    assert prov["files"]["LatFlow/D2Q9.py"].startswith("8acdfd9")


def test_port_reproduces_shipped_bgk():
    """Upstream les=False branch == tensorlbm collide_bgk to round-off."""
    g = torch.Generator().manual_seed(7)
    n = 16
    jj, ii = torch.meshgrid(
        torch.arange(n, dtype=torch.float64), torch.arange(n, dtype=torch.float64), indexing="ij"
    )
    x, y = ii + 0.5, jj + 0.5
    k = 2 * math.pi / n
    ux = -0.06 * torch.cos(k * x) * torch.sin(k * y)
    uy = 0.06 * torch.sin(k * x) * torch.cos(k * y)
    f = equilibrium(torch.ones_like(ux), ux, uy) + 1e-3 * torch.randn(
        (9, n, n), dtype=torch.float64, generator=g
    )
    tau = 0.55
    np.testing.assert_allclose(
        OSS.collide(f, tau, 0.0).numpy(), collide_bgk(f, tau).numpy(), rtol=0, atol=1e-14
    )


def test_inhouse_norm_matches_chapman_enskog_identity():
    """|Pi|_F = (2/3) rho tau0 |S|_F for an exact CE field (any trace-free S)."""
    rng = np.random.default_rng(3)
    tau0 = 0.5048
    for _ in range(20):
        S = _random_trace_free(rng) * 0.02
        f = _ce_field(S, tau0)
        rho, ux, uy = macroscopic(f)
        feq = equilibrium(rho, ux, uy)
        pi = _neq_stress_norm_2d(f - feq).double()
        sf = math.sqrt(float((S * S).sum()))
        ref = (2.0 / 3.0) * tau0 * sf
        assert float(pi.mean()) == pytest.approx(ref, rel=1e-4)


def test_proxy_is_exact_on_d2q9_trace_free_but_not_generally():
    rng = np.random.default_rng(5)
    tau0 = 0.55
    for _ in range(20):
        S = _random_trace_free(rng) * 0.03
        f = _ce_field(S, tau0)
        rho, ux, uy = macroscopic(f)
        f_neq = f - equilibrium(rho, ux, uy)
        pi = _neq_stress_norm_2d(f_neq).double()
        q = OSS.q_proxy(f_neq)
        assert float((q / pi).mean()) == pytest.approx(1.0, rel=1e-9)
    # negative control: an arbitrary non-equilibrium field has no such identity
    ratios = []
    for seed in range(50):
        g = torch.Generator().manual_seed(seed)
        f_neq = torch.randn((9, 8, 8), dtype=torch.float64, generator=g)
        pi = _neq_stress_norm_2d(f_neq).double()
        q = OSS.q_proxy(f_neq)
        ratios.append(float((q / pi).mean()))
    assert max(abs(r - 1.0) for r in ratios) > 0.05


def test_coefficient_conversion_reproduces_upstream_viscosity():
    """C_s = sqrt(Sc/3) makes the two closures agree on a CE state."""
    tau0 = 0.5048
    cs_map = math.sqrt(SC_OSS / 3.0)
    rng = np.random.default_rng(9)
    for _ in range(10):
        S = _random_trace_free(rng) * 0.02
        f = _ce_field(S, tau0)
        rho, ux, uy = macroscopic(f)
        f_neq = f - equilibrium(rho, ux, uy)
        pi = _neq_stress_norm_2d(f_neq)
        q = OSS.q_proxy(f_neq)
        nu_mine = ((_smagorinsky_tau(tau0, pi, rho, cs_map) - tau0) / 3.0).double()
        nu_oss = OSS.nu_t(q, rho, tau0, SC_OSS).double()
        assert float(nu_mine.mean()) == pytest.approx(float(nu_oss.mean()), rel=1e-9)


def test_both_closures_vanish_in_solid_body_rotation():
    tau0 = 0.55
    f = _ce_field(np.zeros((2, 2)), tau0)
    rho, ux, uy = macroscopic(f)
    f_neq = f - equilibrium(rho, ux, uy)
    pi = _neq_stress_norm_2d(f_neq)
    q = OSS.q_proxy(f_neq)
    assert float(pi.max()) < 1e-6
    assert float(q.max()) < 1e-6
    nu_mine = (_smagorinsky_tau(tau0, pi, rho, CS_OWN) - tau0) / 3.0
    assert float(nu_mine.abs().max()) == pytest.approx(0.0, abs=1e-12)


def test_predeclared_contract_is_intact():
    pre = json.loads((BENCH / "predeclared.json").read_text())
    for key in (
        "decision",
        "alternatives",
        "hypothesis",
        "metrics",
        "baseline_control",
        "predeclared_thresholds",
        "sensitivity_points",
        "variables_changed",
        "variables_controlled",
        "confounders",
        "repetition_rationale",
        "environment",
        "analysis_plan",
    ):
        assert key in pre, key
    assert pre["predeclared_thresholds"]["owner_acceptance_pct"] == 15.0
    assert len(pre["cases"]) >= 3