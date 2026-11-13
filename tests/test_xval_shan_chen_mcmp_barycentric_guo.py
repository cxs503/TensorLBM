"""Shan-Chen MCMP: pseudopotential option + OSS cross-validation regression.

Anchor ``XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO``.

Three things are pinned here:

1. the new ``psi=`` argument of :func:`tensorlbm.multiphase.sc_two_component_force`
   is *opt-in*: omitting it must stay bit-identical to the historical linear
   ``psi(rho) = rho`` force;
2. ``psi=`` actually enters both components of the pair force the documented way,
   ``F1 = -G_12 psi(rho1) sum_i w_i c_i psi(rho2)(x - c_i)``, checked on a
   periodic field with a non-linear ``psi`` (so a sign/argument swap cannot hide);
3. the library arm (barycentric equilibrium velocity + Guo forcing + ``psi_exp``)
   reproduces the OSS reference profile of ``lssfau/lbmpy``
   (``tests/shan_chen/test_shan_chen_two_component.py``) within the owner
   accuracy standard of 15 % relative L2.

The reference is the fetched artifact
``artifacts/xval_lbm_shan_chen_mcmp/oss_ref_shan_chen_two_component.json``
(provenance + sha256 inside).  The test is skipped when the artifact is absent
so the suite stays usable from a checkout without network access.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.multiphase import (
    C,
    W,
    collide_sc_two_component,
    psi_exp,
    sc_two_component_force,
)
from tensorlbm.solver import stream

ART = Path(__file__).resolve().parents[1] / "artifacts" / "xval_lbm_shan_chen_mcmp"
REF_FILE = ART / "oss_ref_shan_chen_two_component.json"


def test_psi_default_is_bit_identical():
    """Omitting ``psi`` must reproduce the historical linear-psi force exactly."""
    torch.manual_seed(0)
    rho1 = torch.rand(12, 10, dtype=torch.float64) + 0.2
    rho2 = torch.rand(12, 10, dtype=torch.float64) + 0.2
    base = sc_two_component_force(rho1, rho2, -2.5, gx=1e-3, gy=-2e-3)
    same = sc_two_component_force(rho1, rho2, -2.5, gx=1e-3, gy=-2e-3, psi=None)
    explicit_linear = sc_two_component_force(
        rho1, rho2, -2.5, gx=1e-3, gy=-2e-3, psi=lambda r: r
    )
    for a, b, c in zip(base, same, explicit_linear):
        assert torch.equal(a, b)
        assert torch.allclose(a, c)


def test_psi_enters_both_components_of_the_pair_force():
    """Check the documented force law against a direct stencil evaluation."""
    ny, nx = 9, 11
    torch.manual_seed(1)
    rho1 = torch.rand(ny, nx, dtype=torch.float64) + 0.2
    rho2 = torch.rand(ny, nx, dtype=torch.float64) + 0.2
    G = -1.7
    psi1, psi2 = psi_exp(rho1), psi_exp(rho2)

    # Direct O(h) evaluation: F1 = -G psi1 * sum_i w_i c_i psi2(x - c_i)
    fx = torch.zeros_like(rho1)
    fy = torch.zeros_like(rho1)
    for i in range(9):
        cx, cy = int(C[i, 0]), int(C[i, 1])
        # torch.roll(+cy) maps out[y] = psi2[y - cy], i.e. the x - c_i sample
        shifted = torch.roll(torch.roll(psi2, cy, dims=0), cx, dims=1)
        fx += float(W[i]) * cx * shifted
        fy += float(W[i]) * cy * shifted

    Fx1, Fy1, Fx2, Fy2 = sc_two_component_force(rho1, rho2, G, psi=psi_exp)
    assert torch.allclose(Fx1, -G * psi1 * fx, atol=1e-12)
    assert torch.allclose(Fy1, -G * psi1 * fy, atol=1e-12)
    # The second component uses the swapped field; with psi_exp (non-linear) this
    # is *not* simply minus the first component.
    assert not torch.allclose(Fx2, -Fx1)


@pytest.mark.skipif(not REF_FILE.exists(), reason="OSS reference artifact not fetched")
def test_xval_matches_oss_shan_chen_two_component():
    """Library barycentric + Guo + psi_exp vs the lssfau/lbmpy reference profile."""
    ref = json.loads(REF_FILE.read_text())
    n = ref["parameters"]["n"]
    steps = ref["parameters"]["steps"]

    torch.set_num_threads(1)
    x = torch.arange(n).view(1, n)
    half = n // 2
    rho_a = torch.where(x < half, 0.1, 0.9).expand(n, n).contiguous().double()
    rho_b = torch.where(x < half, 0.9, 0.1).expand(n, n).contiguous().double()
    zero = torch.zeros_like(rho_a)
    f1, f2 = equilibrium(rho_a, zero, zero), equilibrium(rho_b, zero, zero)

    for _ in range(steps):
        f1, f2 = collide_sc_two_component(
            f1, f2, G_12=-6.0, tau1=1.0, tau2=1.0,
            u_eq="mixture", force_scheme="guo", psi=psi_exp,
        )
        f1, f2 = stream(f1), stream(f2)

    got_a, _, _ = macroscopic(f1)
    got_b, _, _ = macroscopic(f2)
    for got, key in ((got_a, "ref_a"), (got_b, "ref_b")):
        reference = torch.tensor(ref[key], dtype=torch.float64)
        rel_l2 = float(torch.linalg.norm(got[0] - reference) / torch.linalg.norm(reference))
        assert rel_l2 < 0.15, f"{key}: relative L2 error {rel_l2:.4%} exceeds 15 %"
        # the residual that the library actually shows is far tighter; guard it so
        # a silent regression to another velocity coupling is caught immediately
        assert rel_l2 < 5e-3, f"{key}: relative L2 error {rel_l2:.4%} regressed"