"""Pointwise regression gate for the D3Q19 whole step (round 3).

Runs WITHOUT lbmpy: the reference side is the frozen bundle
``xval/.../d3q19/refdata/pointwise_d3q19_v1.npz`` produced from lbmpy 2.0 +
the definition streaming operator, with per-array sha256 in the manifest.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
D3 = ROOT / "xval" / "XVAL-LBM-POINTWISE-REFERENCE-REGRESSION" / "d3q19"
sys.path.insert(0, str(D3))

import arms3d  # noqa: E402
import refcore3d as core  # noqa: E402

REF_DIR = D3 / "refdata"
NPZ = REF_DIR / "pointwise_d3q19_v1.npz"
MANIFEST = json.loads((REF_DIR / "pointwise_d3q19_v1.manifest.json").read_text())
STEPS = MANIFEST["config"]["steps"]
TAU = MANIFEST["config"]["tau"]

pytestmark = pytest.mark.skipif(not NPZ.exists(), reason="frozen D3Q19 reference bundle missing")


@pytest.fixture(scope="module")
def data():
    with np.load(NPZ) as z:
        return {k: z[k] for k in z.files}


def test_bundle_sha256():
    assert hashlib.sha256(NPZ.read_bytes()).hexdigest() == MANIFEST["sha256"]


def test_per_array_sha256(data):
    for name, digest in MANIFEST["arrays"].items():
        arr = np.ascontiguousarray(data[name], dtype=np.float64)
        assert hashlib.sha256(arr.tobytes()).hexdigest() == digest, name


@pytest.mark.parametrize("arm", ["A1_solver3d_bgk", "A2_diff3d_step", "A3_adv_bgk_d3q19"])
def test_whole_step_matches_frozen_reference(data, arm):
    traj = arms3d.run_arm(arm, data["f0"], TAU, STEPS)
    eps1 = core.compare(traj[0], data["ref_t1"])["eps_inf"]
    eps30 = core.compare(traj[-1], data["ref_t30"])["eps_inf"]
    assert eps1 <= 1e-10, f"{arm} t=1 eps_inf={eps1:.3e}"
    assert eps30 <= 1e-10, f"{arm} t=30 eps_inf={eps30:.3e}"


@pytest.mark.parametrize("arm", ["A4_mrt_uniform", "A5_mrt_native"])
def test_mrt_arms_match_own_frozen_output(data, arm):
    """MRT has no symbolic reference in the frozen bundle; pin its own output.

    ``A5`` additionally carries the live symbolic check against lbmpy's
    ``MomentBasedLbMethod`` (see d3q19/run_xval.py, criterion C2).
    """
    traj = arms3d.run_arm(arm, data["f0"], TAU, STEPS)
    eps = core.compare(traj[-1], data[f"repo_{arm}_t30"])["eps_inf"]
    assert eps <= 1e-13, f"{arm} frozen-self eps_inf={eps:.3e}"


def test_mrt_uniform_is_not_bgk(data):
    """Documented property: uniform user-visible MRT rates are not BGK."""
    a4 = arms3d.run_arm("A4_mrt_uniform", data["f0"], TAU, STEPS)[-1]
    a1 = arms3d.run_arm("A1_solver3d_bgk", data["f0"], TAU, STEPS)[-1]
    gap = core.compare(a4, a1)["eps_inf"]
    assert gap > 1e-5, gap


def test_continuous_equilibrium_is_not_the_used_convention(data):
    """Guards against silently switching the reference equilibrium convention."""
    traj = arms3d.run_arm("A1_solver3d_bgk", data["f0"], TAU, STEPS)
    eps = core.compare(traj[0], data["ref_continuous_t1"])["eps_inf"]
    assert eps > 1e-3, eps


def test_mass_and_momentum_conservation(data):
    traj = arms3d.run_arm("A1_solver3d_bgk", data["f0"], TAU, STEPS)
    prev = data["f0"]
    for f in traj:
        c = core.conservation(prev, f)
        assert c["mass_drift"] <= 1e-12
        assert c["momentum_drift"] <= 1e-12
        prev = f
