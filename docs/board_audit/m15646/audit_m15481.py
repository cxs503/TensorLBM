#!/usr/bin/env python3
"""m15646 board-audit of m15481 (anchor GAP-PP-3-adjoint-ootc, repo TensorLBM).

Independent re-runnable checks.  The audited work tree /root/tp-agent/work/m15481
is opened READ-ONLY; the three audited source files are used through a
sha256-pinned snapshot in code_snapshot/adjoint/.

  C1 provenance     sha256(snapshot) == sha256(mission tree); tracked in which ref?
  C2 placement      pyproject package-dir vs where the files were put; name clash
                    with the pre-existing src/tensorlbm/adjoint.py module
  C3 importability  can `tensorlbm.adjoint` (authoritative tree) reach the new code?
  C4 physics smoke  forward D2Q9 runs, mass conserved, pulse decays
  C5 math core      BPTT gradient vs central finite difference of J(tau)
  C6 OOTC == BPTT   gradient agreement when T % k == 0
  C7 T % k != 0     OOTC silently integrates a different number of steps than T
  C8 memmap claim   np.lib.format.open_memmap on numpy 2.x
  C9 memory scaling peak RSS of OOTC vs step count (O(k*N) claim)
  C10 evidence      tests / run artifacts / entry point / benchmark rows / report

usage: python3 docs/board_audit/m15646/audit_m15481.py --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNAP = HERE / "code_snapshot"
SRC = Path("/root/tp-agent/work/m15481")
MISSION_BRANCH, MISSION_COMMIT = "agent/m15481", "39515bd6"
ANCHOR = "GAP-PP-3-adjoint-ootc"
FILES = ["adjoint/bptt.py", "adjoint/lbm2d.py", "adjoint/ootc.py"]
PUT = "TensorLBM/src/tensorlbm"          # where m15481 put them
AUTH = "src/tensorlbm"                   # pyproject package-dir = src

sys.path.insert(0, str(SNAP))
import numpy as np  # noqa: E402

from adjoint.lbm2d import initial_pulse, macro, step  # noqa: E402
from adjoint.bptt import bptt_gradient  # noqa: E402
from adjoint.ootc import ootc_gradient  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sh(*a, cwd=SRC) -> str:
    r = subprocess.run(list(a), cwd=str(cwd), capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def psql(sql: str) -> str:
    r = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-tA", "-F", "\t", "-c", sql],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else f"ERROR: {r.stderr.strip()[:200]}"


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 2 ** 30


# ---------------------------------------------------------------- C1 provenance
def c1() -> dict:
    out = {"snapshot_vs_mission_tree": {}, "refs": {}}
    for f in FILES:
        m, s = SRC / PUT / f, SNAP / f
        out["snapshot_vs_mission_tree"][f] = {
            "mission": str(m), "exists_in_mission_tree": m.is_file(),
            "sha256_mission": sha(m) if m.is_file() else None,
            "sha256_snapshot": sha(s), "identical": m.is_file() and sha(m) == sha(s),
            "lines": len(s.read_text().splitlines()),
        }
    rel = [f"{PUT}/{f}" for f in FILES]
    out["refs"] = {
        "mission_branch": MISSION_BRANCH,
        "commit_on_mission_branch": sh("git", "log", "--oneline", "-1", MISSION_COMMIT),
        "tracked_in_mission_branch": sh("git", "ls-tree", "-r", "--name-only", MISSION_BRANCH, "--", *rel).split(),
        "tracked_in_main": sh("git", "ls-tree", "-r", "--name-only", "main", "--", *rel).split(),
        "commit_is_ancestor_of_main": sh("git", "merge-base", "--is-ancestor", MISSION_COMMIT, "main") == ""
        and False,  # exit status handled below
        "refs_containing_commit": sh("git", "branch", "-a", "--contains", MISSION_COMMIT).split(),
        "remote_refs_containing": sh("git", "branch", "-r", "--contains", MISSION_COMMIT).split(),
        "main_head": sh("git", "rev-parse", "--short", "main"),
        "branch_off_main_by": sh("git", "rev-list", "--left-right", "--count", f"main...{MISSION_COMMIT}"),
    }
    r = subprocess.run(["git", "merge-base", "--is-ancestor", MISSION_COMMIT, "main"],
                       cwd=str(SRC), capture_output=True)
    out["refs"]["commit_is_ancestor_of_main"] = (r.returncode == 0)
    return out


# ---------------------------------------------------------------- C2 placement
def c2() -> dict:
    pj = (SRC / "pyproject.toml").read_text()
    pkgdir = [l for l in pj.splitlines() if "package-dir" in l]
    return {
        "pyproject_package_dir": pkgdir,
        "files_placed_in": PUT,
        "authoritative_tree": AUTH,
        "authoritative_tree_has_adjoint_package": (SRC / AUTH / "adjoint").exists(),
        "authoritative_tree_has_adjoint_module": (SRC / AUTH / "adjoint.py").is_file(),
        "package_init_present_in_mission_tree": (SRC / PUT / "adjoint" / "__init__.py").is_file(),
        "existing_adjoint_module_head": sh("git", "show", "main:src/tensorlbm/adjoint.py")[:400],
        "grep_callers_outside_adjoint_dir": sh(
            "grep", "-rn", "--include=*.py", "-e", "adjoint.ootc", "-e", "adjoint.bptt",
            "-e", "ootc_gradient", "-e", "bptt_gradient", ".",
        ).splitlines(),
    }


# ---------------------------------------------------------------- C3 importability
def c3() -> dict:
    code = (
        "import sys; sys.path.insert(0, %r);\n"
        "import tensorlbm, os; print('pkg_file', tensorlbm.__file__);\n"
        "import tensorlbm.adjoint as a; print('adjoint_file', getattr(a,'__file__',None));\n"
        "print('has_ootc', hasattr(a,'ootc_gradient'))\n"
    ) % str(SRC / AUTH)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(SRC))
    return {"cmd": "python -c '<import tensorlbm.adjoint from src/ tree>'",
            "rc": r.returncode, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()[-500:]}


# ---------------------------------------------------------------- C4 physics
def c4() -> dict:
    f0 = initial_pulse(32, 32, amp=0.05, sigma=6.0)
    f = f0
    series = []
    for n in range(1, 401):
        f = step(f, 0.8)
        if n % 100 == 0:
            rho, ux, uy = macro(f)
            series.append({"step": n, "mass": float(rho.sum()),
                           "umean": float(np.mean(np.hypot(ux, uy))),
                           "umax": float(np.max(np.hypot(ux, uy))),
                           "finite": bool(np.isfinite(f).all())})
    return {"grid": "32x32", "tau": 0.8, "series": series,
            "mass_rel_drift": abs(series[-1]["mass"] - series[0]["mass"]) / series[0]["mass"]}


# ---------------------------------------------------------------- C5 BPTT vs FD
def _J(tau, T, f0, ux_ref, uy_ref):
    f = f0
    for _ in range(T):
        f = step(f, tau)
    _, ux, uy = macro(f)
    return 0.5 * float(np.sum((ux - ux_ref) ** 2 + (uy - uy_ref) ** 2))


def c5() -> dict:
    T, ny, nx, tau = 200, 24, 24, 0.85
    f0 = initial_pulse(ny, nx, amp=0.05, sigma=5.0)
    g, J, (ux_ref, uy_ref), states = bptt_gradient(tau, T, f0, np.zeros((ny, nx)), np.zeros((ny, nx)))
    out = {"T": T, "grid": f"{ny}x{nx}", "tau": tau, "J": J,
           "dJ_dtau_bptt": g, "n_stored_states": len(states), "fd": {}}
    for h in (1e-4, 1e-5, 1e-6):
        Jp = _J(tau + h, T, f0, np.zeros((ny, nx)), np.zeros((ny, nx)))
        Jm = _J(tau - h, T, f0, np.zeros((ny, nx)), np.zeros((ny, nx)))
        fd = (Jp - Jm) / (2 * h)
        out["fd"][f"h={h:g}"] = {"fd": fd, "rel_err": abs(fd - g) / abs(g)}
    out["best_rel_err"] = min(v["rel_err"] for v in out["fd"].values())
    return out


# ---------------------------------------------------------------- C6 OOTC vs BPTT
def c6() -> dict:
    T, k, ny, nx, tau = 200, 50, 24, 24, 0.85
    f0 = initial_pulse(ny, nx, amp=0.05, sigma=5.0)
    z = np.zeros((ny, nx))
    g_bptt, J_bptt, _, _ = bptt_gradient(tau, T, f0, z, z)
    ck = Path(tempfile.mkdtemp()) / "ck.npy"
    t0 = time.time()
    g_oo, J_oo, (ux_t, uy_t), n_ck, path = ootc_gradient(tau, T, f0, z, z, k=k, ckpt_path=str(ck))
    dt = time.time() - t0
    return {"T": T, "k": k, "grid": f"{nx}x{nx}", "n_checkpoints": n_ck,
            "ckpt_file_bytes": os.path.getsize(path),
            "bptt": {"g": g_bptt, "J": J_bptt}, "ootc": {"g": g_oo, "J": J_oo},
            "grad_rel_diff": abs(g_oo - g_bptt) / abs(g_bptt),
            "J_abs_diff": abs(J_oo - J_bptt),
            "wall_s": dt, "gate_1e-5_passed": abs(g_oo - g_bptt) / abs(g_bptt) < 1e-5}


# ---------------------------------------------------------------- C7 T % k != 0
def c7() -> dict:
    ny = nx = 16
    tau, f0 = 0.85, initial_pulse(ny, nx, amp=0.05, sigma=4.0)
    z = np.zeros((ny, nx))
    cases = []
    for T, k in ((200, 100), (250, 100), (200, 64)):
        g_b, J_b, _, _ = bptt_gradient(tau, T, f0, z, z)
        ck = Path(tempfile.mkdtemp()) / "ck.npy"
        g_o, J_o, _, n_ck, _ = ootc_gradient(tau, T, f0, z, z, k=k, ckpt_path=str(ck))
        # number of forward steps the OOTC loop actually integrates
        steps_integrated = (T // k) * k
        g_ref, J_ref, _, _ = bptt_gradient(tau, steps_integrated, f0, z, z)
        cases.append({"T": T, "k": k, "n_checkpoints": n_ck,
                      "steps_actually_integrated": steps_integrated,
                      "J_bptt_full_T": J_b, "J_ootc": J_o, "J_bptt_at_integrated_steps": J_ref,
                      "J_mismatch_vs_full_T": abs(J_o - J_b) / max(abs(J_b), 1e-300),
                      "grad_rel_diff_vs_full_T": abs(g_o - g_b) / max(abs(g_b), 1e-300),
                      "grad_rel_diff_vs_integrated_steps": abs(g_o - g_ref) / max(abs(g_ref), 1e-300),
                      "silently_wrong": bool(T % k != 0)})
    return {"grid": f"{nx}x{nx}", "tau": tau, "cases": cases}


# ---------------------------------------------------------------- C8 memmap claim
def c8() -> dict:
    has = hasattr(np.lib.format, "open_memmap")
    p = Path(tempfile.mkdtemp()) / "m.npy"
    err = None
    try:
        mm = np.lib.format.open_memmap(str(p), mode="w+", dtype=np.float64, shape=(2, 3))
        mm[0, 0] = 1.0
        mm.flush()
        del mm
        back = np.load(p)
        ok = float(back[0, 0]) == 1.0
    except Exception as e:  # noqa: BLE001
        ok, err = False, f"{type(e).__name__}: {e}"
    return {"numpy_version": np.__version__, "has_np_lib_format_open_memmap": has,
            "roundtrip_ok": ok, "error": err,
            "mission_claim": "ootc.py 用了 np.lib.format.open_memmap（numpy 2.x 已弃用，应改 np.memmap）"}


# ---------------------------------------------------------------- C9 memory
def c9() -> dict:
    rows = []
    for T in (500, 1000, 2000):
        ny = nx = 128
        tau, f0 = 0.9, initial_pulse(ny, nx, amp=0.05, sigma=16.0)
        z = np.zeros((ny, nx))
        ck = Path(tempfile.mkdtemp()) / "ck.npy"
        r0 = rss_gib()
        t0 = time.time()
        g, J, _, n_ck, path = ootc_gradient(tau, T, f0, z, z, k=100, ckpt_path=str(ck))
        rows.append({"T": T, "k": 100, "n_checkpoints": n_ck, "rss_before_GiB": r0,
                     "rss_peak_GiB": rss_gib(), "ckpt_GB": os.path.getsize(path) / 2 ** 30,
                     "wall_s": time.time() - t0, "finite": bool(np.isfinite(g))})
    return {"grid": "128x128", "rows": rows,
            "max_ckpt_GB": max(r["ckpt_GB"] for r in rows),
            "gate_8GiB_peak_rss": max(r["rss_peak_GiB"] for r in rows) < 8.0,
            "note": "ru_maxrss is process-peak, so it is monotone in T by construction; "
                    "the checkpoint file size is the T-linear term and it is DISK, not RAM"}


# ---------------------------------------------------------------- C10 evidence
def c10() -> dict:
    tests = [p for p in (SRC / "tests").rglob("*.py") if "adjoint" in p.name.lower() or "ootc" in p.name.lower()]
    art = sh("git", "ls-tree", "-r", "--name-only", MISSION_BRANCH).splitlines()
    art = [a for a in art if ("adjoint" in a.lower() or "ootc" in a.lower()) and a.startswith(("artifacts", "runs", "figures", "docs"))]
    return {
        "mission_declared_result": psql("SELECT result->>'fail_stage' FROM tensor_dev_missions WHERE id=15481;"),
        "mission_head_commit": psql("SELECT result->>'head_commit' FROM tensor_dev_missions WHERE id=15481;"),
        "tests_in_mission_branch": [t.relative_to(SRC).as_posix() for t in tests],
        "artifacts_runs_docs_in_mission_branch": art,
        "benchmark_cases_for_anchor": psql(
            "SELECT case_name||'|'||metric||'|'||coalesce(tolerance,'') FROM tensor_benchmark_cases "
            f"WHERE anchor_slug='{ANCHOR}';"),
        "reports_for_mission": psql("SELECT id||'|'||title FROM tensor_reports WHERE mission_id=15481;"),
        "entry_points": sh("grep", "-rn", "-e", "argparse", "-e", "__main__", "--include=*.py",
                           f"{PUT}/adjoint").splitlines(),
        "doc_md_in_mission_branch": [a for a in sh("git", "ls-tree", "-r", "--name-only", MISSION_BRANCH).splitlines()
                                     if "ADJOINT" in a.upper() or "OOTC" in a.upper()],
    }


CHECKS = {"C1_provenance": c1, "C2_placement": c2, "C3_importability": c3, "C4_physics_smoke": c4,
          "C5_math_bptt_vs_fd": c5, "C6_ootc_vs_bptt": c6, "C7_T_not_divisible_k": c7,
          "C8_memmap_claim": c8, "C9_memory_scaling": c9, "C10_evidence": c10}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    only = [x for x in a.only.split(",") if x]
    out = {}
    for name, fn in CHECKS.items():
        if only and name.split("_")[0] not in only:
            continue
        t0 = time.time()
        try:
            out[name] = fn()
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": f"{type(e).__name__}: {e}"}
        out[name]["_wall_s"] = round(time.time() - t0, 2)
        print(f"[ok] {name} {out[name]['_wall_s']}s", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"anchor": ANCHOR, "audited_mission": 15481,
                                       "audit_mission": 15646, "host": os.uname().machine,
                                       "python": sys.version.split()[0], "checks": out}, indent=1))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())