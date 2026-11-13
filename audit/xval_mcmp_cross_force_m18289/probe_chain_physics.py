#!/usr/bin/env python3
"""Board-audit probe: recompute the inherited physics-arm error from the pinned
artifacts of the predecessor round (m18126 delivery, preserved by audit m18219
at docs/board_audit/m18219/refs/artifacts/, commit 6ad01d71).

This does NOT credit m18289 with run artifacts of its own: it only tests the
one number m18289's headline inherited ("physics arm 200.34%").
"""
import io
import json
import pathlib
import subprocess

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
COMMIT = "6ad01d71"
CASES = ["arm_N64_g5_exp.npz", "arm_N64_g6_exp.npz", "arm_N64_g6_linear.npz",
         "arm_N128_g6_exp.npz"]


def load(name):
    p = f"docs/board_audit/m18219/refs/artifacts/{name}"
    blob = subprocess.run(["git", "show", f"{COMMIT}:{p}"], cwd=REPO,
                          capture_output=True).stdout
    if not blob:
        return None
    return np.load(io.BytesIO(blob))


def main():
    out = {"source_commit": COMMIT, "arms": {}}
    for name in CASES:
        z = load(name)
        if z is None:
            out["arms"][name] = {"present": False}
            continue
        a = np.asarray(z["prof_a"], float)
        ref = np.asarray(z["prof_ref"], float)
        nan = int(np.isnan(a).sum())
        if nan:
            out["arms"][name] = {"present": True, "nan_entries": nan,
                                 "err_rel_l2": None}
            continue
        d = a - ref
        # definition used by the delivering code: docs/board_audit/m18219/refs/
        # run_xval.py:177-182  rel_l2 = ||got-ref|| / ||ref - mean(ref)||
        err_metric = float(np.linalg.norm(d) / np.linalg.norm(ref - ref.mean()))
        # plain normalisation, for the definition-dependence note
        err_plain = float(np.linalg.norm(d) / np.linalg.norm(ref))
        out["arms"][name] = {"present": True, "nan_entries": 0,
                             "err_rel_l2": err_metric, "err_pct": 100.0 * err_metric,
                             "err_pct_plain_norm": 100.0 * err_plain}
    errs = [v.get("err_pct") for v in out["arms"].values() if v.get("err_pct")]
    out["evaluable_arms"] = len(errs)
    out["worst_err_pct"] = max(errs) if errs else None
    out["all_arms"] = len(CASES)
    print(json.dumps(out, indent=1))
    (pathlib.Path(__file__).parent / "probe_chain_physics.json").write_text(
        json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
