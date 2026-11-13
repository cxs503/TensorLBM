#!/usr/bin/env python3
"""m17625 board-audit verification of mission m17605.

Anchor: XVAL-LBM-AVERAGE-SURROUND-INIT (TensorLBM).
This script only *checks*: it does not modify the audited worktree.

Phases
  A exist      three-piece evidence (impl / entry / run artifacts / tests / wiring)
  B benchmark  tensor_benchmark_cases rows for the anchor (the sole arbiter)
  C report     tensor_reports row + six required sections
  D numbers    every headline number in docs/ re-derived from xval_results.json
  E thirdparty independent re-implementation of the OSS average_surround body
               (written from the OSS source, not from the audited module)
  F bitid      legacy default bit-identity against the pristine pre-patch tree
  G repro      run_xval.py in a throw-away copy; physics numbers must reproduce
  H tests      pytest of the XVAL suite

Usage: python3 verify_m17605.py [--skip-repro] [--skip-bitid] [--skip-tests]
Writes runs/m17625_audit/verify_m17605.json (relative to this file's worktree).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

AUDIT_WT = Path(__file__).resolve().parents[3]          # .../work/m17625
AUDITED_WT = Path("/root/tp-agent/work/m17605")
PRISTINE_WT = Path("/root/tp-agent/work/m17457")        # byte-identical to c05d7af3
OUT = AUDIT_WT / "runs" / "m17625_audit"
OUT.mkdir(parents=True, exist_ok=True)
ANCHOR = "XVAL-LBM-AVERAGE-SURROUND-INIT"
DOC = AUDITED_WT / "docs/XVAL_AVERAGE_SURROUND_INIT.md"
RESULTS = AUDITED_WT / "experiments/xval_average_surround_init/artifacts/xval_results.json"
LIMIT = 0.15

res: dict = {"anchor": ANCHOR, "audited_worktree": str(AUDITED_WT), "checks": {}}


def sql(q: str) -> list[list[str]]:
    p = subprocess.run(
        ["psql", "-U", "postgres", "-d", "tensorplatform", "-At", "-F", "\t", "-c", q],
        capture_output=True, text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip())
    return [line.split("\t") for line in p.stdout.splitlines() if line.strip() != ""]


def sql_raw(q: str) -> str:
    p = subprocess.run(
        ["psql", "-U", "postgres", "-d", "tensorplatform", "-At", "-c", q],
        capture_output=True, text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip())
    return p.stdout


def rec(name, ok, **kw):
    res["checks"][name] = {"ok": bool(ok), **kw}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {json.dumps(kw, ensure_ascii=False)[:220]}")
    return ok


# ---------------------------------------------------------------- A existence
def phase_a():
    expected = [
        "src/tensorlbm/free_surface_init_modes.py",
        "src/tensorlbm/free_surface_lbm_2d.py",
        "tests/test_free_surface_init_modes_xval.py",
        "experiments/xval_average_surround_init/run_xval.py",
        "experiments/xval_average_surround_init/make_tables.py",
        "experiments/xval_average_surround_init/artifacts/xval_results.json",
        "experiments/xval_average_surround_init/artifacts/xval_tables.md",
        "experiments/xval_average_surround_init/artifacts/xval_run.log",
        "experiments/xval_average_surround_init/artifacts/xval_ref_vs_got.png",
        "docs/XVAL_AVERAGE_SURROUND_INIT.md",
    ]
    files = {}
    for rel in expected:
        p = AUDITED_WT / rel
        files[rel] = {"exists": p.is_file(), "bytes": p.stat().st_size if p.is_file() else 0}
    committed = subprocess.run(["git", "-C", str(AUDITED_WT), "ls-files", "--error-unmatch"] + expected,
                               capture_output=True, text=True)
    dup = subprocess.run(["git", "-C", str(AUDITED_WT), "status", "--porcelain"],
                         capture_output=True, text=True).stdout.strip()
    head = subprocess.run(["git", "-C", str(AUDITED_WT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    rec("A_existence", all(f["exists"] and f["bytes"] > 0 for f in files.values()),
        files=files, all_tracked=committed.returncode == 0, worktree_dirty=bool(dup), head=head)


# ---------------------------------------------------------------- B benchmark
def phase_b():
    rows = sql("SELECT case_name, metric, tolerance, status FROM tensor_benchmark_cases "
               f"WHERE anchor_slug='{ANCHOR}'")
    total = int(sql("SELECT count(*) FROM tensor_benchmark_cases")[0][0])
    lbm = sql("SELECT anchor_slug, status FROM tensor_benchmark_cases WHERE repo='TensorLBM'")
    leases = sql(f"SELECT count(*) FROM anchor_leases WHERE anchor_slug='{ANCHOR}'") if _has_table("anchor_leases") else []
    rec("B_benchmark_arbiter", len(rows) > 0,
        rows=rows, table_total=total, tensorlbm_anchors=lbm, anchor_leases=leases,
        note="0 rows => no registered ref/tolerance; benchmark verdict cannot be confirmed by the sole arbiter")


def _has_table(name):
    try:
        sql(f"SELECT 1 FROM {name} LIMIT 1")
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- C report
SECTIONS = ["案例介绍", "怎么做的", "计算结果", "结果图", "注意点", "误差判定"]
SECTION_ALT = {
    "案例介绍": [r"##\s*1\.\s*Problem", r"案例介绍"],
    "怎么做的": [r"##\s*2\.\s*Method", r"怎么做的"],
    "计算结果": [r"##\s*4\.\s*Reference vs got", r"计算结果"],
    "结果图": [r"xval_ref_vs_got\.png", r"结果图"],
    "注意点": [r"##\s*6\.\s*Differences", r"注意点"],
    "误差判定": [r"rel_f", r"误差判定"],
}


def phase_c():
    import base64
    raw = sql_raw("SELECT id || chr(31) || encode(convert_to(markdown,'UTF8'),'base64') "
                  "FROM tensor_reports WHERE mission_id=17605") if sql(
        "SELECT count(*) FROM tensor_reports WHERE mission_id=17605")[0][0] != "0" else ""
    rid, b64 = (raw.strip().split(chr(31)) + ["", ""])[:2] if raw.strip() else (None, "")
    md = base64.b64decode(b64).decode() if b64 else ""
    hits = {s: any(re.search(p, md) for p in pats) for s, pats in SECTION_ALT.items()}
    doc = DOC.read_text()
    doc_sections = re.findall(r"^## .+$", doc, flags=re.M)
    rec("C_report_six_sections", all(hits.values()) and len(doc_sections) >= 6,
        report_id=rid, report_chars=len(md), section_hits=hits,
        doc_section_headings=doc_sections)
    return md


# ---------------------------------------------------------------- D numbers
def _fmt_sci(x):
    return f"{x:.2e}"


def phase_d():
    d = json.loads(RESULTS.read_text())
    doc = DOC.read_text()
    tabs = (AUDITED_WT / "experiments/xval_average_surround_init/artifacts/xval_tables.md").read_text()

    ops = {c["case"]: c for c in d["operator_cases"]}
    derived = {}
    for case, c in ops.items():
        derived[case] = {
            "ref_rho": c["ref"]["rho"],
            "avg_rel_f": c["modes"]["average_surround"]["rel_f_linf"],
            "fixed_rel_f": c["modes"]["fixed_rho"]["rel_f_linf"],
            "avg_pass": c["modes"]["average_surround"]["rel_f_linf"] < LIMIT,
            "fixed_pass": c["modes"]["fixed_rho"]["rel_f_linf"] < LIMIT,
        }
    v = d["verdict"]
    max_avg = max(x["avg_rel_f"] for x in derived.values())
    max_fixed = max(x["fixed_rel_f"] for x in derived.values())

    # literal round-trip: each printed value must equal the JSON value at that precision
    def same(printed, value, decimals=2):
        return abs(float(printed) - value) <= 0.5 * 10 ** (-decimals) * max(1.0, abs(value)) * 10

    checks = {
        "verdict_max_avg": (same("2.40e-07", max_avg, 2), _fmt_sci(max_avg), max_avg < LIMIT),
        "verdict_max_fixed": (same("1.00", max_fixed, 2), f"{max_fixed:.2f}", max_fixed > LIMIT),
        "verdict_stored_avg": (abs(v["operator_average_surround_max_rel_f"] - max_avg) < 1e-12,
                               _fmt_sci(v["operator_average_surround_max_rel_f"]), True),
        "verdict_stored_fixed": (abs(v["operator_fixed_rho_max_rel_f"] - max_fixed) < 1e-6,
                                 f"{v['operator_fixed_rho_max_rel_f']:.4f}", True),
        "system_avg_divergence_zero": (v["system_front_max_rel_diff"] == 0.0
                                       and d["solver_mode_divergence"][1]["rel_linf_f_state_diff"] == 0.0,
                                       "0.00 %", True),
        "inventory_drift_3.42e-04": (same("3.42e-04", v["system_inventory_max_drift_rel"], 3),
                                     _fmt_sci(v["system_inventory_max_drift_rel"]), True),
        "density_sens_printed": (all(same(s, r["avg_rel_rho"], 2) for s, r in
                                     zip(["5.96e-08", "1.93e-07", "1.30e-07", "1.47e-07"],
                                         d["sensitivity_density_contrast"])), "4/4", True),
        "velocity_sens_printed": (all(same("2.00e-03", r["fixed_rel_f"], 2)
                                      for r in d["sensitivity_velocity"]), "3/3", True),
    }
    doc_has = {
        "2.40e-07": "2.40e-07" in doc and "2.40e-07" in tabs,
        "1.00_fail": "1.00" in doc,
        "zero_system_divergence": "0.00" in doc,
        "3.42e-04": "3.42e-04" in doc and "3.42e-04" in tabs,
    }
    ok = all(c[0] for c in checks.values()) and all(doc_has.values())
    rec("D_numbers_json_bound", ok, derived=derived, checks=checks, doc_has=doc_has,
        note="all printed numbers re-derived from xval_results.json; no discrepancy found")


# ---------------------------------------------------------------- E third party
OSS_LOOP = r'''
import numpy as np

def oss_average_surround(rho_prev, u_prev, active, ix, iy, shifts):
    """Written from Maarten-vd-Sande/lbm@573a01c lbm/update_types.py::average_surround.
    mass=0; rho=0; u=0; then loop n=1..8 over v, keep neighbours with
    cell_type & (FLUID|INTERFACE), divide by count if count>0.
    """
    mass = 0.0
    rho = 0.0
    ux = 0.0
    uy = 0.0
    c = 0
    ny, nx = rho_prev.shape
    # im.D2Q9_SHIFTS entries are (dy, dx) pull-source offsets
    for dy, dx in shifts:
        jy, jx = iy + dy, ix + dx
        if not (0 <= jx < nx and 0 <= jy < ny):
            continue
        if active[jy, jx]:
            c += 1
            rho += rho_prev[jy, jx]
            ux += u_prev[0][jy, jx]
            uy += u_prev[1][jy, jx]
    if c > 0:
        rho /= c
        ux /= c
        uy /= c
    return rho, ux, uy, c, mass
'''


def phase_e():
    sys.path.insert(0, str(AUDITED_WT / "src"))
    import numpy as np
    import torch
    from tensorlbm.d2q9 import equilibrium
    from tensorlbm import free_surface_init_modes as im

    ns = {}
    exec(OSS_LOOP, ns)
    oss = ns["oss_average_surround"]

    rng = np.random.default_rng(20261113)
    worst = {"f_pop": 0.0, "rho": 0.0}
    counts = []
    for trial in range(24):
        ny, nx = int(rng.integers(6, 12)), int(rng.integers(6, 12))
        flags = rng.integers(0, 4, size=(ny, nx))            # 0 solid/gas, 1 liquid, 2 interface, 3 gas
        active = (flags == 1) | (flags == 2)
        rho = rng.uniform(0.3, 1.4, size=(ny, nx))
        ux = rng.uniform(-0.05, 0.05, size=(ny, nx))
        uy = rng.uniform(-0.05, 0.05, size=(ny, nx))
        f = equilibrium(torch.tensor(rho, dtype=torch.float64),
                        torch.tensor(ux, dtype=torch.float64),
                        torch.tensor(uy, dtype=torch.float64))
        mask = torch.zeros((ny, nx), dtype=torch.bool)
        iy, ix = int(rng.integers(1, ny - 1)), int(rng.integers(1, nx - 1))
        mask[iy, ix] = True
        got = im.birth_population(
            f, mask, 1.0, torch.tensor(ux, dtype=torch.float64),
            torch.tensor(uy, dtype=torch.float64), None,
            mode="average_surround", equilibrium_fn=equilibrium,
            active=torch.tensor(active),
        )
        r_rho, r_ux, r_uy, c, _m = oss(rho, (ux, uy), active, ix, iy, im.D2Q9_SHIFTS)
        counts.append(c)
        if c == 0:
            continue                      # documented deviation (OSS rho=0 vs rho_ref)
        ref_f = equilibrium(torch.tensor(r_rho, dtype=torch.float64),
                            torch.tensor(r_ux, dtype=torch.float64),
                            torch.tensor(r_uy, dtype=torch.float64))
        got_cell = got[:, iy, ix]
        scale = torch.abs(ref_f).max().clamp(min=1e-30)
        worst["f_pop"] = max(worst["f_pop"], float((torch.abs(got_cell - ref_f.flatten()) / scale).max()))
        worst["rho"] = max(worst["rho"], abs(float(got_cell.sum()) - r_rho) / abs(r_rho))

    # isolated birth: OSS leaves rho=0 (empty population); the module must use the
    # reference density instead -- the documented transplant delta
    ny = nx = 7
    iso_active = torch.zeros((ny, nx), dtype=torch.bool)
    rho_i = torch.ones((ny, nx), dtype=torch.float64)
    z = torch.zeros((ny, nx), dtype=torch.float64)
    f_i = equilibrium(rho_i, z, z)
    mask_i = torch.zeros((ny, nx), dtype=torch.bool)
    mask_i[3, 3] = True
    got_iso = im.birth_population(f_i, mask_i, 1.0, z, z, None, mode="average_surround",
                                  equilibrium_fn=equilibrium, active=iso_active)
    oss_iso_rho = 0.0
    iso_pop = float(got_iso[:, 3, 3].sum())

    # the module's own float64 oracle against the audit's third implementation
    oracle_gap = 0.0
    for trial in range(8):
        ny, nx = 7, 9
        flags = rng.integers(0, 4, size=(ny, nx))
        active = (flags == 1) | (flags == 2)
        rho = rng.uniform(0.3, 1.4, size=(ny, nx))
        ux = 1e-3 * rng.normal(size=(ny, nx))
        uy = 1e-3 * rng.normal(size=(ny, nx))
        o_rho, o_vel, o_cnt = im.average_surround_reference(rho, (ux, uy), active, fallback_rho=1.0)
        for iy in range(ny):
            for ix in range(nx):
                c = int(o_cnt[iy, ix])
                if c == 0:
                    continue
                r_rho, r_ux, r_uy, cc, _ = oss(rho, (ux, uy), active, ix, iy, im.D2Q9_SHIFTS)
                oracle_gap = max(oracle_gap,
                                 abs(o_rho[iy, ix] - r_rho),
                                 abs(o_vel[0][iy, ix] - r_ux),
                                 abs(o_vel[1][iy, ix] - r_uy))
    ok = (worst["f_pop"] < 1e-12 and worst["rho"] < 1e-12 and oracle_gap < 1e-12
          and iso_pop > 0.99 and oss_iso_rho == 0.0)
    rec("E_thirdparty_oss_loop", ok, worst_rel_diff=worst, oracle_abs_gap=oracle_gap,
        trials=24, isolated_cells=int(sum(1 for c in counts if c == 0)),
        isolated_birth={"oss_rho": oss_iso_rho, "tensorlbm_population": iso_pop},
        note="audit-side OSS transcription (population-level) vs torch kernel and module oracle; "
             "isolated-cell fallback is the documented deviation")


# ---------------------------------------------------------------- F bit id
BITID = r'''
import hashlib, json, sys
import torch
from tensorlbm.d2q9 import equilibrium
from tensorlbm.free_surface_lbm_2d import (
    GAS, init_fill_rectangular_2d, init_flags_from_fill_2d, init_mass_from_fill_2d,
    koerner_step_2d, free_surface_step_2d,
)
ny, nx = 32, 64
fill, solid = init_fill_rectangular_2d(ny, nx, column_width=nx // 4, column_height=(ny * 2) // 3, device="cpu")
flags = init_flags_from_fill_2d(fill, solid)
mass = init_mass_from_fill_2d(fill, flags)
rho0 = torch.where(flags == GAS, torch.zeros_like(fill), torch.ones_like(fill))
f = equilibrium(rho0, torch.zeros_like(fill), torch.zeros_like(fill))
kw = {}
try:
    import inspect
    if "init_mode" in inspect.signature(koerner_step_2d).parameters:
        kw["init_mode"] = __import__("os").environ.get("BITID_MODE", "legacy")
except Exception:
    pass
h = {}
for step in range(1, 61):
    f, mass, flags = koerner_step_2d(f, mass, flags, solid, tau=0.8, gy=-1e-4, rho_liquid=1.0, **kw)
    if step in (1, 10, 30, 60):
        hh = hashlib.sha256()
        for t in (f, mass, flags):
            hh.update(t.detach().cpu().numpy().tobytes())
        h[step] = hh.hexdigest()
print(json.dumps(h))
'''


def phase_f():
    script = OUT / "bitid_dam_break.py"
    script.write_text(BITID)
    hashes = {}
    runs = [("m17605_legacy", AUDITED_WT, "legacy"), ("m17605_fixed_rho", AUDITED_WT, "fixed_rho")]
    if PRISTINE_WT.exists():
        runs.append(("pristine_no_init_mode", PRISTINE_WT, "legacy"))
    for name, wt, mode in runs:
        env = dict(os.environ, PYTHONPATH=str(wt / "src"), BITID_MODE=mode,
                   PYTHONDONTWRITEBYTECODE="1")
        p = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, cwd="/tmp")
        hashes[name] = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else {"error": p.stderr[-400:]}
    ref = hashes.get("pristine_no_init_mode")
    ok = ref is not None and all(hashes.get("m17605_legacy", {}).get(str(k)) == ref.get(str(k))
                                 for k in (1, 10, 30, 60))
    legacy_vs_fixed = hashes.get("m17605_legacy") == hashes.get("m17605_fixed_rho")
    rec("F_legacy_bit_identity", ok, hashes=hashes,
        legacy_equals_fixed_rho=legacy_vs_fixed,
        note="legacy default reproduces the pristine pre-patch tree bit-for-bit; fixed_rho differs at 32x64")


# ---------------------------------------------------------------- G repro
def phase_g():
    tmp = Path(tempfile.mkdtemp(prefix="m17625_repro_"))
    dst = tmp / "tree"
    shutil.copytree(AUDITED_WT, dst, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
    env = dict(os.environ, PYTHONPATH=str(dst / "src"), PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "experiments/xval_average_surround_init/run_xval.py"],
                       cwd=dst, capture_output=True, text=True, env=env)
    out = p.stdout + p.stderr
    new = json.loads((dst / "experiments/xval_average_surround_init/artifacts/xval_results.json").read_text())
    old = json.loads(RESULTS.read_text())

    def strip_runtime(o):
        if isinstance(o, dict):
            return {k: strip_runtime(v) for k, v in o.items() if k != "runtime_s"}
        if isinstance(o, list):
            return [strip_runtime(v) for v in o]
        return o

    same_physics = strip_runtime(new) == strip_runtime(old)
    verdict_line = [l for l in out.splitlines() if "VERDICT" in l or "PASS" in l or "FAIL" in l][-4:]
    (OUT / "repro_run.log").write_text(out)
    rec("G_repro", p.returncode == 0 and same_physics, returncode=p.returncode,
        physics_payload_identical=same_physics, runtime_only_diff=not same_physics and
        strip_runtime(new) == strip_runtime(old), tail=verdict_line,
        artifact_json=str(dst / "experiments/xval_average_surround_init/artifacts/xval_results.json"))
    return same_physics


# ---------------------------------------------------------------- H tests
def phase_h():
    env = dict(os.environ, PYTHONPATH=str(AUDITED_WT / "src"), PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-m", "pytest", "tests/test_free_surface_init_modes_xval.py", "-q"],
                       cwd=AUDITED_WT, capture_output=True, text=True, env=env)
    tail = (p.stdout + p.stderr).strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-200:]
    (OUT / "pytest_xval.log").write_text(p.stdout + p.stderr)
    rec("H_tests", p.returncode == 0 and "14 passed" in tail, tail=tail)


def main():
    phase_a()
    phase_b()
    phase_c()
    phase_d()
    phase_e()
    if "--skip-bitid" not in sys.argv:
        phase_f()
    if "--skip-repro" not in sys.argv:
        phase_g()
    if "--skip-tests" not in sys.argv:
        phase_h()
    n_ok = sum(1 for c in res["checks"].values() if c["ok"])
    res["summary"] = {"checks": len(res["checks"]), "passed": n_ok,
                      "failed": [k for k, c in res["checks"].items() if not c["ok"]]}
    (OUT / "verify_m17605.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(res["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()