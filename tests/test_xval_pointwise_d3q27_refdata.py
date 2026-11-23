"""Pointwise regression gate for the D3Q27 whole step (round 4).

Runs WITHOUT lbmpy: the reference side is the frozen bundle
``xval/XVAL-LBM-POINTWISE-REFERENCE-REGRESSION/d3q27/refdata/pointwise_d3q27_v1.npz``
produced from lbmpy 2.0 (SRT / TRT / repo-basis MRT / Geier cumulant) plus the
definition streaming operator, with per-array sha256 in the manifest.

This is the D3Q27 leg of the lssfau/lbmpy pointwise-reference-regression
practice: any future change to ``collide_bgk27``, ``collide_trt27``,
``collide_mrt27``, ``collide_cumulant_geier_d3q27`` or ``stream27_roll`` that
perturbs the operator beyond fp64 round-off fails here.
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
D3 = ROOT / "xval" / "XVAL-LBM-POINTWISE-REFERENCE-REGRESSION" / "d3q27"
sys.path.insert(0, str(D3))

import refcore27 as core  # noqa: E402

REF_DIR = D3 / "refdata"
NPZ = REF_DIR / "pointwise_d3q27_v1.npz"
MANIFEST_PATH = REF_DIR / "pointwise_d3q27_v1.manifest.json"

GATE = 1e-10

pytestmark = pytest.mark.skipif(
    not NPZ.exists() or not MANIFEST_PATH.exists(),
    reason="frozen D3Q27 reference bundle missing",
)


@pytest.fixture(scope="module")
def bundle():
    manifest = json.loads(MANIFEST_PATH.read_text())
    with np.load(NPZ) as z:
        data = {k: z[k] for k in z.files}
    return manifest, data


def test_bundle_sha256(bundle):
    manifest, _ = bundle
    assert hashlib.sha256(NPZ.read_bytes()).hexdigest() == manifest["sha256"]


def test_per_array_sha256(bundle):
    manifest, data = bundle
    for name, digest in manifest["arrays"].items():
        arr = np.ascontiguousarray(data[name], dtype=np.float64)
        assert hashlib.sha256(arr.tobytes()).hexdigest() == digest, name


def test_manifest_declares_all_gated_arms(bundle):
    manifest, _ = bundle
    import arms27  # noqa: PLC0415

    for arm in manifest["gated_arms"]:
        assert arm in arms27.GATED_ARMS
        assert manifest["arm_to_model"][arm] in manifest["models"]


@pytest.mark.parametrize(
    "arm",
    [
        "A1_d3q27_bgk",
        "A2_d3q27_trt",
        "A3_d3q27_mrt_native",
        "A4_d3q27_cumulant_geier",
    ],
)
def test_arm_reproduces_frozen_reference(arm, bundle):
    """Replay the arm from the stored initial field and gate the worst step."""
    manifest, data = bundle
    import arms27  # noqa: PLC0415

    cfg = manifest["config"]
    keep = tuple(cfg["keep"])
    steps = int(cfg["steps"])
    tau = float(cfg["tau"])
    worst = {}
    for gi in range(len(cfg["grids"])):
        f0 = data[f"f0_g{gi}"]
        model = arms27.REFERENCE_MODEL[arm]
        got = arms27.run_arm(arm, f0, tau, steps, keep=keep)
        for t in keep:
            ref = data[f"ref_{model}_g{gi}_t{t}"]
            worst[(gi, t)] = core.compare(got[t], ref)["eps_scaled_inf"]
    bad = {k: v for k, v in worst.items() if not v < GATE}
    assert not bad, f"{arm} drifted from the frozen reference: {bad}"
