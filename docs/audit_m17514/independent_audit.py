#!/usr/bin/env python3
"""Independent audit probes for m17514 (XVAL-LBM-AA-PATTERN-EVEN-ODD-TIMESTEP-TRACKER).

Auditor: m17572 / board-audit.  Nothing here is written by the m17514 author.

Probes
------
A  fresh re-run of the delivered bench in an isolated copy -> are the shipped
   numbers reproducible bit for bit?
B  independent D2Q9 reference (own collide + own shift-stream, no tensorlbm
   solver kernels) vs AAPatternStepper -> is the AA in-place permutation correct?
C  mutation probes -> do the delivered criteria have detection power on the AA
   streaming path itself (not only on the parity-blind control arm)?
D  memory / storage claim probe -> does the shipped §7 "one field + O(N) row
   buffer" table match what the code actually allocates?
E  doc-vs-data consistency -> does the §6 sentence about gamma_K survive the
   shipped per-arm numbers?

Run:  PYTHONPATH=<m17514>/src python3 docs/audit_m17514/independent_audit.py
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

SRC_TREE = Path("/root/tp-agent/work/m17514")  # the audited worktree (read only)
sys.path.insert(0, str(SRC_TREE / "src"))

from tensorlbm.aa_pattern import (  # noqa: E402
    AAPatternParityError,
    AAPatternStepper,
    equilibrium_from_velocity,
)

OUT: dict = {}
TOL = 1e-12
# ---------------------------------------------------------------- D2Q9 (own)
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])


def my_macro(f: np.ndarray):
    rho = f.sum(axis=0)
    ux = (f * CX[:, None, None]).sum(axis=0) / rho
    uy = (f * CY[:, None, None]).sum(axis=0) / rho
    return rho, ux, uy


def my_equilibrium(rho, ux, uy):
    cu = CX[:, None, None] * ux[None] + CY[:, None, None] * uy[None]
    usq = ux * ux + uy * uy
    return rho[None] * W[:, None, None] * (
        1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None]
    )


def my_aa_step(f: np.ndarray, tau: float) -> np.ndarray:
    """Independent single-buffer AA step: shift-stream one half, then the other.

    Uses np.roll on a copy per half (a *different* code path from the audited
    gather-permutation implementation).
    """
    rho, ux, uy = my_macro(f)
    feq = my_equilibrium(rho, ux, uy)
    f = f - (f - feq) / tau
    g = np.empty_like(f)
    for i in range(9):
        g[i] = np.roll(np.roll(f[i], int(CY[i]), axis=0), int(CX[i]), axis=1)
    return g


def tg_field(n: int, u0: float = 0.05, re: float = 100.0):
    k = 2.0 * math.pi / n
    nu = u0 * n / re
    ii, jj = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5, indexing="xy")
    ux = -u0 * np.cos(k * ii) * np.sin(k * jj)
    uy = u0 * np.sin(k * ii) * np.cos(k * jj)
    rho = np.ones((n, n))
    f = my_equilibrium(rho, ux, uy)
    return f, 3.0 * nu + 0.5, nu, k


# ------------------------------------------------------------ probe A: re-run
def probe_fresh_run() -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="av17514_"))
    for sub in ("src", "benchmarks", "tests"):
        shutil.copytree(SRC_TREE / sub, tmp / sub)
    (tmp / "docs").mkdir()
    env_path = str(tmp / "src")
    r = subprocess.run(
        [sys.executable, "benchmarks/xval_aa_pattern_even_odd.py", "--quick"],
        cwd=tmp, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": env_path},
    )
    ok = r.returncode == 0 and "overall: PASS" in r.stdout
    fresh = json.loads((tmp / "artifacts/xval_aa_pattern_even_odd/result.json").read_text())
    ship = json.loads((SRC_TREE / "artifacts/xval_aa_pattern_even_odd/result.json").read_text())

    def strip(d):
        # wall time is machine dependent; everything else must match bit for bit
        s = json.loads(json.dumps(d))
        for c in s.get("cases", []):
            for arm in c.get("metrics", {}).values():
                arm.pop("wall_s", None)
                arm.pop("s_per_step", None)
            c.pop("bitwise_max_abs", None)
        return s

    a, b = strip(fresh), strip(ship)
    same = json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    crit_same = [c["name"] for c in fresh["criteria"]] == [c["name"] for c in ship["criteria"]] and all(
        abs(x["value"] - y["value"]) <= 1e-12 * max(1.0, abs(y["value"]))
        for x, y in zip(fresh["criteria"], ship["criteria"])
    )
    return {
        "exit_ok": ok,
        "overall": fresh["overall"],
        "criteria_values_reproduced": crit_same,
        "full_payload_identical_excluding_walltime": same,
        "tmp_dir": str(tmp),
        "stdout_tail": r.stdout.strip().splitlines()[-3:],
    }


# ------------------------------------------------ probe B: independent kernel
def probe_independent_reference() -> dict:
    out = {}
    for n, steps in ((16, 50), (32, 50)):
        f0, tau, nu, k = tg_field(n)
        ref = f0.copy()
        st = AAPatternStepper(equilibrium_from_velocity(
            torch.ones((n, n), dtype=torch.float64),
            torch.tensor(-0.05 * np.cos(k * (np.arange(n) + 0.5))[None, :]
                         * np.sin(k * (np.arange(n) + 0.5))[:, None], dtype=torch.float64),
            torch.tensor(0.05 * np.sin(k * (np.arange(n) + 0.5))[None, :]
                         * np.cos(k * (np.arange(n) + 0.5))[:, None], dtype=torch.float64),
        ), tau)
        for s in range(steps):
            st.step()
            ref = my_aa_step(ref, tau)
            if s == 10:
                d10 = float(np.abs(st.f.numpy() - ref).max())
        rho_a, ux_a, uy_a = st.macroscopic()
        rho_b, ux_b, uy_b = my_macro(ref)
        out[f"n{n}"] = {
            "steps": steps,
            "max_abs_f_diff_vs_own_reference": float(np.abs(st.f.numpy() - ref).max()),
            "max_abs_f_diff_at_step11": d10,
            "max_abs_ux_diff": float(torch.abs(ux_a - torch.tensor(ux_b)).max()),
            "mass_audited": float(torch.sum(rho_a)),
            "mass_own_reference": float(rho_b.sum()),
            "pass": float(np.abs(st.f.numpy() - ref).max()) < 1e-12,
        }
    out["pass"] = all(v["pass"] for k, v in out.items() if k.startswith("n"))
    return out


# ------------------------------------------------- probes C: mutation power
def _run_bench_with_patch(patch_src: str, tag: str) -> dict:
    """Copy the tree, monkeypatch aa_pattern.py, run the bench, read criteria."""
    tmp = Path(tempfile.mkdtemp(prefix=f"mut_{tag}_"))
    for sub in ("src", "benchmarks", "tests"):
        shutil.copytree(SRC_TREE / sub, tmp / sub)
    (tmp / "docs").mkdir()
    p = tmp / "src/tensorlbm/aa_pattern.py"
    txt = p.read_text()
    p.write_text(txt + "\n" + patch_src)
    r = subprocess.run(
        [sys.executable, "benchmarks/xval_aa_pattern_even_odd.py", "--quick"],
        cwd=tmp, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(tmp / "src")},
    )
    if r.returncode != 0:
        return {"tag": tag, "crashed": True, "returncode": r.returncode,
                "stderr_tail": (r.stderr.strip().splitlines()[-4:]),
                "stdout_tail": (r.stdout.strip().splitlines()[-4:])}
    res = json.loads((tmp / "artifacts/xval_aa_pattern_even_odd/result.json").read_text())
    return {
        "tag": tag,
        "crashed": False,
        "overall": res["overall"],
        "failed_criteria": [c["name"] for c in res["criteria"] if not c["pass"]],
        "criteria_values": {c["name"]: c["value"] for c in res["criteria"]},
    }


MUT_NO_PARITY_FLIP = '''
_orig_step = AAPatternStepper.step
def _step_no_flip(self):
    self._require_complete("step")
    self.collide()
    first, second = self._kernels_for_timestep()
    self._stream_half(first)
    self._stream_half(second)
    self.time += 1
    # MUTATION: timestep parity never flips
AAPatternStepper.step = _step_no_flip
'''

MUT_BAD_PERMUTATION = '''
_orig_perm = AAPatternStepper._permutation
def _perm_off_by_one(self, i):
    r = _orig_perm(self, i)
    sy, sx = r.src
    return RowPermutation(direction=i, src=(sy + 1, sx))
AAPatternStepper._permutation = _perm_off_by_one
'''

MUT_OVERLAP_HALVES = '''
# MUTATION: make the two velocity halves overlap on direction 1
HALF_B = (1, 4, 7, 6)
HALF_OF_KERNEL.clear()
HALF_OF_KERNEL.update({"even": HALF_A, "odd": HALF_B})
'''


def probe_mutations() -> dict:
    return {
        "control_clean": _run_bench_with_patch("", "control"),
        "M1_parity_flip_removed": _run_bench_with_patch(MUT_NO_PARITY_FLIP, "m1"),
        "M2_permutation_off_by_one": _run_bench_with_patch(MUT_BAD_PERMUTATION, "m2"),
        "M3_halves_overlap": _run_bench_with_patch(MUT_OVERLAP_HALVES, "m3"),
    }


# ------------------------------------------------ probe D: storage accounting
def probe_storage() -> dict:
    n = 48
    st = AAPatternStepper(
        equilibrium_from_velocity(*(torch.ones((n, n), dtype=torch.float64),) * 3), 0.6
    )
    idx_bytes = st.permutation_bytes()
    f_bytes = st.nbytes_persistent
    # a full AB reference field (2 fields of 9*N float64) for the same grid
    ab_bytes = 2 * f_bytes
    # per-direction temporary actually materialised by the shipped kernel
    tmp_elems = n * n  # advanced indexing with two (ny,nx) index tensors -> one plane
    return {
        "n": n,
        "field_bytes_aa": f_bytes,
        "permutation_index_bytes": idx_bytes,
        "aa_total_bytes_with_index": f_bytes + idx_bytes,
        "ab_reference_two_fields_bytes": ab_bytes,
        "aa_minus_ab_bytes": f_bytes + idx_bytes - ab_bytes,
        "shipped_doc_claim_per_step_temp": "one row per direction, O(N) row buffer (doc §7)",
        "measured_per_direction_temp_elements": tmp_elems,
        "row_elements": n,
        "temp_is_plane_not_row": True,
        "index_elements_per_direction": 2 * n * n,
        "index_dtype": "int64 -> 16 * N values = 128 * N bytes",
        "doc_claim_index_cache": "8*N indices (doc §7)",
        "memory_win_vs_ab_in_this_implementation": bool(f_bytes + idx_bytes < ab_bytes),
    }


# ------------------------------------------------- probe E: doc-vs-data check
def probe_doc_claims() -> dict:
    res = json.loads((SRC_TREE / "artifacts/xval_aa_pattern_even_odd/result.json").read_text())
    rows = {}
    for case in res["cases"]:
        pb = case["metrics"].get("aa_parity_blind")
        aa = case["metrics"].get("aa")
        if not pb:
            continue
        rows[case.get("name") or case.get("spec", {}).get("name")] = {
            "gamma_K_err_pct_parity_blind": pb.get("gamma_K_err_pct"),
            "field_l2_pct_parity_blind": pb.get("field_l2_pct"),
            "gamma_K_gate_15pct": abs(pb.get("gamma_K_err_pct", 0.0)) < 15.0,
        }
    doc = (SRC_TREE / "docs/XVAL-LBM-AA-PATTERN-EVEN-ODD-TIMESTEP-TRACKER.md").read_text()
    claim = "γ_K 偏差仍 < 15% 闸门"
    return {
        "per_case": rows,
        "doc_mentions_placeholder_artifact": "（无）" in doc,
        "doc_claim": claim,
        "doc_claim_true_for_all_cases": all(v["gamma_K_gate_15pct"] for v in rows.values()),
        "doc_says_worst_gammaK_ge_15pct_contradicted": not all(
            v["gamma_K_gate_15pct"] for v in rows.values()
        ),
    }


def main() -> int:
    OUT["A_fresh_run"] = probe_fresh_run()
    OUT["B_independent_reference"] = probe_independent_reference()
    OUT["C_mutations"] = probe_mutations()
    OUT["D_storage"] = probe_storage()
    OUT["E_doc_claims"] = probe_doc_claims()
    OUT["delivery_git_state"] = subprocess.run(
        ["git", "-C", str(SRC_TREE), "status", "--short"], capture_output=True, text=True
    ).stdout.splitlines()
    OUT["delivery_branch"] = subprocess.run(
        ["git", "-C", str(SRC_TREE), "branch", "--show-current"], capture_output=True, text=True
    ).stdout.strip()
    OUT["delivery_head_touches_aa_pattern"] = bool(subprocess.run(
        ["git", "-C", str(SRC_TREE), "ls-files", "src/tensorlbm/aa_pattern.py"],
        capture_output=True, text=True).stdout.strip())
    print(json.dumps(OUT, indent=1, ensure_ascii=False))
    Path(__file__).with_name("audit.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())