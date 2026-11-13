"""XVAL-LBM-TRT-LAMBDA-MAGIC-TABLE-DRIVEN -- acceptance tests (mission 18452).

The OSS reference (``lbm-principles-practice/code`` @ 6e2c592, MIT) is compiled
from the vendored verbatim sources in
``xval/XVAL-LBM-TRT-LAMBDA-MAGIC-TABLE-DRIVEN/oss_src/`` inside a tmp dir, so
the assertions compare against numbers the reference binary produced itself.

Gated (matches ``predeclared.json``):
  K1  magic relation identity  <= 1e-15
  K2  discrete table identity
  K3  solver identity vs the OSS binary, shared setup  <= 1e-12
  K4  analytic accuracy of both arms  <= 15% (owner band)
  K7  the opt-in OSS u^2 equilibrium term reproduces the OSS binary at u=0.5
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
XV = os.path.join(REPO, "xval", "XVAL-LBM-TRT-LAMBDA-MAGIC-TABLE-DRIVEN")
sys.path.insert(0, os.path.join(REPO, "src"))

from tensorlbm import scalar_transport as st  # noqa: E402


def _load_harness():
    spec = importlib.util.spec_from_file_location("xval_trt", os.path.join(XV, "xval_trt_lambda.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["xval_trt"] = mod          # dataclass() needs the module registered
    spec.loader.exec_module(mod)
    return mod


pytestmark = pytest.mark.skipif(shutil.which("g++") is None, reason="g++ required to build the OSS reference")


def _build_reference(tmpdir, stem: str) -> np.ndarray:
    src = os.path.join(XV, "oss_src", stem + ".cpp")
    text = open(src).read()
    hi = text.replace("precision(10)", "precision(17)").replace(f'"{stem}/"', f'"{stem}_hi/"')
    assert hi != text
    work = os.path.join(tmpdir, stem + "_hi.cpp")
    open(work, "w").write(hi)
    exe = os.path.join(tmpdir, stem + "_hi_bin")
    subprocess.run(["g++", "-O2", "-o", exe, work], check=True)
    os.makedirs(os.path.join(tmpdir, stem + "_hi"), exist_ok=True)
    subprocess.run([exe], cwd=tmpdir, check=True, capture_output=True, timeout=600)
    d = os.path.join(tmpdir, stem + "_hi")
    files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".dat"))
    return np.vstack([[float(t) for t in open(f).read().split()] for f in files])


@pytest.fixture(scope="module")
def harness():
    return _load_harness()


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    tmpdir = str(tmp_path_factory.mktemp("oss"))
    return {stem: _build_reference(tmpdir, stem) for stem in ("gaussian_1d_magic12", "gaussian_1d_magic6")}


def test_k1_magic_relation_identity(harness):
    out = harness.k1_magic_relation()
    assert max(out["max_abs_omega_diff"].values()) <= 1e-15
    assert out["max_abs_roundtrip_lambda_diff"] <= 1e-15


def test_k2_discrete_table_identity(harness):
    t = harness.k2_tables()
    assert t["w_identity_1d"] == 1.0
    assert t["opp_map_1d"] == 1.0
    assert t["opp_map_2d_d2q9"] == 1.0
    assert t["d2q9_weights_ok"] == 1.0
    assert t["d2q9_opposite_involutive"] == 1.0
    assert t["d3q27_opposite_involutive"] == 1.0


def test_k3_solver_identity_u0(harness, ref):
    case = harness.parse_oss_1d("gaussian_1d_magic12.cpp", "magic12")
    D = (1.0 / case.omega_hand - 0.5) / 3.0   # the odd (hand-set) rate sets D
    got = harness.run_tb(case, D, case.magic)
    want = ref["gaussian_1d_magic12"]
    assert got.shape == want.shape
    assert float(np.abs(got - want).max()) <= 1e-12
    assert harness.rel_l2(got.ravel(), want.ravel()) <= 1e-12


def test_k4_analytic_accuracy_owner_band(harness, ref):
    case = harness.parse_oss_1d("gaussian_1d_magic12.cpp", "magic12")
    D = (1.0 / case.omega_hand - 0.5) / 3.0
    got = harness.run_tb(case, D, case.magic)
    exact = np.vstack([harness.exact_periodic(case, t, D) for t in range(want_shape0(got))])
    err_ref = max(harness.rel_l2(ref["gaussian_1d_magic12"][t], exact[t]) for t in range(1, exact.shape[0]))
    err_tb = max(harness.rel_l2(got[t], exact[t]) for t in range(1, exact.shape[0]))
    assert err_ref <= 0.15, err_ref
    assert err_tb <= 0.15, err_tb


def want_shape0(a):
    return a.shape[0]


def test_k7_second_order_equilibrium_matches_oss_u05(harness, ref):
    case = harness.parse_oss_1d("gaussian_1d_magic6.cpp", "magic6")
    D = (1.0 / case.derived_rate - 0.5) / 3.0
    got = harness.run_tb(case, D, case.magic, equilibrium="oss")
    want = ref["gaussian_1d_magic6"]
    assert float(np.abs(got - want).max()) <= 1e-12
    # and the default (linear) arm really does differ at u = 0.5
    lin = harness.run_tb(case, D, case.magic, equilibrium="linear")
    assert float(np.abs(lin - want).max()) > 1e-3


def test_second_order_default_is_bit_unchanged():
    torch.manual_seed(0)
    c = torch.rand(1, 6, dtype=torch.float64) + 1.0
    ux = torch.full((1, 6), 0.1, dtype=torch.float64)
    uy = torch.zeros((1, 6), dtype=torch.float64)
    cx = torch.tensor(st.CX_D2Q5, dtype=torch.float64).view(5, 1, 1)
    cy = torch.tensor(st.CY_D2Q5, dtype=torch.float64).view(5, 1, 1)
    w = torch.tensor(st.W_D2Q5, dtype=torch.float64).view(5, 1, 1)
    legacy = w * c * (1.0 + (cx * ux + cy * uy) / st.CS2_D2Q5)
    assert torch.equal(st.equilibrium_scalar_d2q5(c, ux, uy), legacy)
    assert torch.equal(st.equilibrium_scalar_d2q5(c, ux, uy, second_order=False), legacy)
    # at u = 0 the u^2 term vanishes identically
    uz = torch.zeros_like(ux)
    assert torch.equal(st.equilibrium_scalar_d2q5(c, uz, uz, second_order=True),
                       st.equilibrium_scalar_d2q5(c, uz, uz))
    # and it does change the u = 0.5 field
    u5 = torch.full((1, 6), 0.5, dtype=torch.float64)
    assert not torch.equal(st.equilibrium_scalar_d2q5(c, u5, uz, second_order=True),
                           st.equilibrium_scalar_d2q5(c, u5, uz))
    # mass (sum over populations) is preserved by construction
    assert float(st.equilibrium_scalar_d2q5(c, u5, uz, second_order=True).sum()) == pytest.approx(float(c.sum()))


def test_collide_mass_conservation_second_order():
    torch.manual_seed(1)
    g = torch.rand(5, 1, 8, dtype=torch.float64)
    c = st.scalar_field(g)
    ux = torch.full((1, 8), 0.2, dtype=torch.float64)
    uy = torch.zeros((1, 8), dtype=torch.float64)
    for so in (False, True):
        out = st.collide_scalar_trt(g, c, ux, uy, 1.0 / (1.0 / 0.7), None, second_order=so)
        assert float(out.sum()) == pytest.approx(float(g.sum()), rel=1e-14)