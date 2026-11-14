"""Freeze the pointwise full-step reference into an lbmpy-free artefact.

Round 1's gate ``tests/test_xval_pointwise_reference_regression.py`` skips
wherever lbmpy is absent, i.e. it is dead exactly on the machines that run the
rest of the suite.  Freezing the reference arrays removes that dependency:

    make_refdata.py  (needs lbmpy)  ->  refdata/pointwise_stepwise_d2q9_v1.npz
                                    +  refdata/...manifest.json (sha256 locked)

The gate then only needs numpy + torch, and ``--check`` re-derives the arrays
and asserts that the frozen copy is reproducible (never hand-edited).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stepwise_ref as S  # noqa: E402

REFDATA = HERE / "refdata"
NPZ = REFDATA / "pointwise_stepwise_d2q9_v1.npz"
MANIFEST = REFDATA / "pointwise_stepwise_d2q9_v1.manifest.json"
CHECKPOINTS = (1, 30)


def field_for(case: dict) -> np.ndarray:
    kind = case["field"]
    if kind.startswith("shear_wave"):
        return S.shear_wave_field(64, 8, u0=2e-2)
    if kind.startswith("non-square"):
        rng = np.random.default_rng(7)
        return S.shear_wave_field(7, 11, u0=5e-2) + 1e-3 * rng.standard_normal((9, 7, 11))
    if kind.startswith("high-Ma"):
        rng = np.random.default_rng(11)
        return S.shear_wave_field(32, 32, u0=1e-1) + 1e-2 * rng.standard_normal((9, 32, 32))
    raise ValueError(kind)


def params(case: dict) -> dict:
    p = {"tau": case["tau"]}
    if case["model"] == "trt":
        p["lambda_trt"] = case.get("lambda_trt", 3.0 / 16.0)
    return p


def generate() -> tuple[dict[str, np.ndarray], list[dict]]:
    dec = json.loads((HERE / "predeclared.json").read_text())
    arrays: dict[str, np.ndarray] = {}
    cases: list[dict] = []
    for case in dec["cases"]:
        if not case["arm"].startswith("S"):
            continue
        arm = case["arm"]
        f0 = field_for(case)
        params_ = params(case)
        ref = S.StepwiseReference(lattice="D2Q9", model=case["model"], **params_)
        out = ref.run(f0, case["steps"], checkpoints=CHECKPOINTS)
        arrays[f"{arm}/f0"] = f0
        for t in CHECKPOINTS:
            arrays[f"{arm}/f_ref_step{t}"] = out["checkpoints"][t]
        cases.append({"arm": arm, "model": case["model"], "params": params_,
                      "field": case["field"], "steps": case["steps"],
                      "checkpoints": list(CHECKPOINTS),
                      "shape": list(f0.shape),
                      "keys": [f"{arm}/f0"] + [f"{arm}/f_ref_step{t}" for t in CHECKPOINTS]})
    return arrays, cases


def write(arrays: dict[str, np.ndarray], cases: list[dict]) -> dict:
    REFDATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NPZ, **arrays)
    manifest = {
        "name": "pointwise_stepwise_d2q9_v1",
        "anchor": "XVAL-LBM-POINTWISE-REFERENCE-REGRESSION",
        "round": 2,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "xval/XVAL-LBM-POINTWISE-REFERENCE-REGRESSION/stepwise/make_refdata.py",
        "oracle": "lbmpy collision symbols evaluated pointwise + np.roll definition streaming",
        "provenance": S.provenance(),
        "npz_sha256": S.sha256_file(NPZ),
        "array_sha256": {k: S.sha256_file_bytes(v) for k, v in sorted(arrays.items())},
        "dtype": "float64",
        "gate_eps_scaled_inf": 1e-10,
        "cases": cases,
        "note": ("reference arrays only; the in-repo kernels are compared against "
                 "them by tests/test_xval_pointwise_stepwise_refdata.py, which needs "
                 "no lbmpy install"),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def check() -> int:
    """Re-derive and verify the frozen artefact is byte-reproducible."""
    manifest = json.loads(MANIFEST.read_text())
    got_sha = S.sha256_file(NPZ)
    ok_sha = got_sha == manifest["npz_sha256"]
    print(f"npz sha256 match: {ok_sha}")
    arrays, _ = generate()
    frozen = np.load(NPZ)
    worst = 0.0
    for k, v in arrays.items():
        eps = float(np.abs(frozen[k] - v).max() / max(np.abs(v).max(), 1e-300))
        worst = max(worst, eps)
        print(f"  {k:28s} eps={eps:.3e}")
    print(f"worst reproduction eps={worst:.3e} (gate {manifest['gate_eps_scaled_inf']:.0e})")
    return 0 if (ok_sha and worst < manifest["gate_eps_scaled_inf"]) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="re-derive and verify frozen data")
    args = ap.parse_args()
    if args.check:
        return check()
    arrays, cases = generate()
    manifest = write(arrays, cases)
    print(f"wrote {NPZ} ({NPZ.stat().st_size} bytes, sha256 {manifest['npz_sha256'][:16]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())