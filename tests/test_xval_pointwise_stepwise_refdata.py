"""lbmpy-free pointwise regression gate for the *full* LBM step.

Companion of ``tests/test_xval_pointwise_reference_regression.py`` (round 1:
collision only, skips without lbmpy).  Here the reference is the frozen
artefact ``xval/.../stepwise/refdata/pointwise_stepwise_d2q9_v1.npz``, produced
by ``make_refdata.py`` from an lbmpy oracle, so this gate runs **everywhere**:

    ref step  = np.roll definition streaming ∘ lbmpy collision symbols
    repo step = tensorlbm.solver.stream ∘ tensorlbm.solver.collide_*

Invariants asserted (not merely "code runs"):
  * the frozen artefact is untouched (sha256 of the npz and of every array),
  * the 30-step in-repo trajectory reproduces every frozen state pointwise,
  * mass and total momentum are conserved to fp64 noise on a periodic box,
  * every in-repo streaming kernel equals the definition in its native layout,
  * a 1e-3 relative perturbation of tau is detected (gate is falsifiable).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
STEP = REPO / "xval" / "XVAL-LBM-POINTWISE-REFERENCE-REGRESSION" / "stepwise"
for _p in (str(REPO / "src"), str(STEP)):
    sys.path.insert(0, _p)

import refcore as core  # noqa: E402
import stepwise_ref as S  # noqa: E402  (numpy/torch only at import time)

NPZ = STEP / "refdata" / "pointwise_stepwise_d2q9_v1.npz"
MANIFEST = STEP / "refdata" / "pointwise_stepwise_d2q9_v1.manifest.json"

GATE_CODE = 1e-10      # predeclared G_code_equivalence
GATE_CONS = 1e-12      # predeclared G_conservation
GATE_STREAM = 1e-12    # predeclared G_stream_crosscheck

MAN = json.loads(MANIFEST.read_text())
CASES = MAN["cases"]


# ---------------------------------------------------------------------------
# provenance / tamper lock
# ---------------------------------------------------------------------------
def test_frozen_artefact_matches_manifest():
    assert core.sha256_file(NPZ) == MAN["npz_sha256"]
    frozen = np.load(NPZ)
    assert sorted(frozen.files) == sorted(MAN["array_sha256"])
    for key, digest in MAN["array_sha256"].items():
        assert core.sha256_file_bytes(frozen[key]) == digest, key


def test_manifest_records_the_oracle_provenance():
    prov = MAN["provenance"]
    assert prov.get("lbmpy_version"), prov
    assert MAN["dtype"] == "float64"
    assert MAN["gate_eps_scaled_inf"] == GATE_CODE
    assert len(CASES) >= 6


# ---------------------------------------------------------------------------
# the gate itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=[c["arm"] for c in CASES])
def test_in_repo_step_reproduces_frozen_reference(case):
    frozen = np.load(NPZ)
    f0 = frozen[f"{case['arm']}/f0"]
    params = dict(case["params"])
    model = case["model"]
    n_steps = case["steps"]
    checks = case["checkpoints"]

    f = f0.copy()
    mass0 = float(f0.sum())
    vel = core.tensorlbm_velocities("D2Q9")
    tot0 = np.tensordot(vel, f0, axes=(0, 0)).sum(axis=(1, 2))
    worst = 0.0
    for t in range(1, n_steps + 1):
        f = S.in_repo_step(f, model=model, **params)
        if t in checks:
            eps = core.compare(f, frozen[f"{case['arm']}/f_ref_step{t}"])["eps_scaled_inf"]
            worst = max(worst, eps)
            assert eps < GATE_CODE, f"{case['arm']} step {t}: eps={eps:.3e}"

    mass_drift = abs(float(f.sum()) - mass0) / abs(mass0)
    tot = np.tensordot(vel, f, axes=(0, 0)).sum(axis=(1, 2))
    mom_drift = float(np.abs(tot - tot0).max() / abs(mass0))
    assert mass_drift < GATE_CONS, mass_drift
    assert mom_drift < GATE_CONS, mom_drift


def test_gate_is_falsifiable_by_tau_perturbation_of_1e_3():
    case = next(c for c in CASES if c["arm"] == "S2")
    frozen = np.load(NPZ)
    f0 = frozen["S2/f0"]
    ref = frozen["S2/f_ref_step30"]
    tau = case["params"]["tau"]
    good = S.in_repo_run(f0, n_steps=30, model="srt", tau=tau)
    bad = S.in_repo_run(f0, n_steps=30, model="srt", tau=tau * (1 + 1e-3))
    assert core.compare(good, ref)["eps_scaled_inf"] < GATE_CODE
    # measured 9.9e-7 for delta_tau/tau = 1e-3: four decades above the gate
    assert core.compare(bad, ref)["eps_scaled_inf"] > 100 * GATE_CODE


def test_frozen_states_are_not_degenerate():
    frozen = np.load(NPZ)
    for case in CASES:
        a = frozen[f"{case['arm']}/f_ref_step1"]
        b = frozen[f"{case['arm']}/f_ref_step30"]
        assert np.isfinite(a).all() and np.isfinite(b).all()
        assert np.abs(a - b).max() > 1e-12  # the trajectory actually moves


# ---------------------------------------------------------------------------
# streaming cross-implementation audit (definition is lbmpy-free)
# ---------------------------------------------------------------------------
def _definition(f):
    return core.definition_stream(f)


def test_solver_stream_matches_definition():
    from tensorlbm import solver

    rng = np.random.default_rng(3)
    f = 0.2 + rng.standard_normal((9, 7, 11))
    got = solver.stream(torch.from_numpy(f.copy())).numpy()
    assert core.compare(got, _definition(f))["eps_scaled_inf"] < GATE_STREAM


def test_lattice_stream_matches_definition_direction_last_layout():
    from tensorlbm import lattice

    rng = np.random.default_rng(4)
    f = 0.2 + rng.standard_normal((9, 7, 11))
    got = lattice.stream(torch.from_numpy(np.moveaxis(f, 0, -1).copy())).numpy()
    got = np.moveaxis(got, -1, 0)
    assert core.compare(got, _definition(f))["eps_scaled_inf"] < GATE_STREAM


def test_les_bptt_conv_stream_matches_definition():
    from tensorlbm import les_bptt

    rng = np.random.default_rng(5)
    f = 0.2 + rng.standard_normal((9, 7, 11))
    got = les_bptt.stream(torch.from_numpy(f[None].copy())).numpy()[0]
    assert core.compare(got, _definition(f))["eps_scaled_inf"] < GATE_STREAM


def test_diff_boundary_grad_stream_matches_definition():
    from tensorlbm import diff_boundary_grad

    rng = np.random.default_rng(6)
    f = 0.2 + rng.standard_normal((9, 7, 11))
    got = diff_boundary_grad.stream(torch.from_numpy(f.copy())).numpy()
    assert core.compare(got, _definition(f))["eps_scaled_inf"] < GATE_STREAM


def test_h_theorem_stream_matches_definition():
    import tensorlbm.h_theorem as hth
    from tensorlbm.d2q9 import C, W_EXACT

    rng = np.random.default_rng(7)
    f = 0.2 + rng.standard_normal((9, 7, 11))
    spec = hth.LatticeSpec("D2Q9", 9, 2, C.clone(), torch.tensor(np.array(W_EXACT)))
    got = hth.stream(torch.from_numpy(f.copy()), spec).numpy()
    assert core.compare(got, _definition(f))["eps_scaled_inf"] < GATE_STREAM


def test_channel_kernel_d3q19_stream_matches_definition():
    from tensorlbm import channel_kernel_numpy as ckn

    rng = np.random.default_rng(8)
    nz, ny, nx = 5, 6, 7
    f = 0.2 + rng.standard_normal((19, nz, ny, nx))
    vel = core.tensorlbm_velocities("D3Q19").astype(int)
    fdef = np.empty_like(f)
    for q in range(19):
        fdef[q] = np.roll(f[q], shift=tuple(int(v) for v in vel[q]), axis=(0, 1, 2))
    assert np.allclose(ckn.C, vel)
    got = np.asarray(ckn.stream(f), dtype=np.float64)
    assert core.compare(got, fdef)["eps_scaled_inf"] < GATE_STREAM


# ---------------------------------------------------------------------------
# guard for the refcore split (needs lbmpy; skipped otherwise)
# ---------------------------------------------------------------------------
def test_refcore_equilibrium_matches_round1_reference():
    pytest.importorskip("lbmpy")
    import lbmpy_pointwise_ref as ref

    rng = np.random.default_rng(9)
    rho = 1.0 + 0.01 * rng.standard_normal((8, 5))
    u = 0.05 * rng.standard_normal((2, 8, 5))
    a = core.equilibrium_field("D2Q9", rho, u)
    b = ref.equilibrium_field("D2Q9", rho, u)
    assert core.compare(a, b)["eps_scaled_inf"] == 0.0
    assert core.compare(core.tensorlbm_velocities("D2Q9"),
                        ref.tensorlbm_velocities("D2Q9"))["abs_diff_max"] == 0.0