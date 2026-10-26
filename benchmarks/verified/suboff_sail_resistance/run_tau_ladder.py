#!/usr/bin/env python3
"""τ ladder driver — uniform-grid suboff_sail_resistance (SIR-1b / m298).

Runs the five-arm ladder frozen in
``/nfs/wangxi/runs/tau_ladder_uniform_20261025/prereg.md`` (md5
d5ab172ebc24fa67df09351b3819b836) on the single-level verified case
``benchmarks/verified/suboff_sail_resistance`` and assembles a machine-written
verdict.  Nothing here re-defines a gate: G1/G2/G3 and the ladder are taken
verbatim from the frozen prereg; the τ–Re coupling analysis is added as a
clearly-marked supplement (the prereg is append-only).

Three concrete defects of the m292 driver are fixed, none of them a gate
change:

1. **interpreter** — m292 passed a torch-less interpreter, so every arm died
   with ``ModuleNotFoundError: torch``.  The interpreter is now probed for
   ``import torch`` *and* an available CUDA device before any arm starts, and
   the driver refuses to run otherwise.
2. **viscosity wiring** — ``SUBOFF_VISCOSITY_OVERRIDE`` was declared in the
   m292 report but never existed in ``run.py``; and a lattice ``ν_lb`` cannot
   be pushed through ``PhysicsConfig.viscosity`` (a physical m²/s field that
   also re-derives ``u_lb``).  The runner now accepts ``SUBOFF_TAU_TARGET`` and
   solves for the physical viscosity that reproduces ``ν_lb=(τ-0.5)/3``
   exactly; this driver re-reads the achieved τ from ``result.json`` and fails
   any arm with ``|τ_actual - τ_target| > 1e-9``.
3. **device** — the ladder launches on ``cuda:0`` inside a
   ``CUDA_VISIBLE_DEVICES=5`` sandbox (GPU5-only discipline), instead of
   naming ``cuda:5`` while the process may not see device 5 at all.

Exit code 0 means "the ladder ran and the verdict was written", *not* that the
physics passed: pass/fail lives in the verdict JSON.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# Frozen constants (prereg.md)
# --------------------------------------------------------------------------- #
STAGING = Path("/nfs/wangxi/runs/tau_ladder_uniform_20261025")
PREREG = STAGING / "prereg.md"
PREREG_MD5_FROZEN = "d5ab172ebc24fa67df09351b3819b836"

REPO = Path("/nfs/wangxi/tensor_eco/agent_work/m298/TensorLBM")
RUN_PY = REPO / "benchmarks/verified/suboff_sail_resistance/run.py"

# prereg「阶梯设计」table, verbatim
TAU_LADDER = {"A": 0.51, "B": 0.55, "C": 0.60, "D": 0.70, "E": 1.00}
ARM_ORDER = ["A", "B", "C", "D", "E"]

# prereg「固定参数」, verbatim
RESOLUTION = 80
STEPS = 12000
U_LB = 0.05
COLLISION = "mrt"
FRICTION = "standard"
P0 = "near_wall"
GPU = "5"

# CUDA_VISIBLE_DEVICES=5 leaves the child exactly one visible GPU -> cuda:0.
DEVICE = "cuda:0"

FALLBACK_PYTHONS = [
    "/nfs/wangxi/venvs/tensorlbm/bin/python",
    "/nfs/wangxi/venvs/tensor_eco/bin/python",
    "/nfs/wangxi/venvs/ci-sim/bin/python",
]

TIMEOUT_S = 2 * 3600  # prereg「停止规则」: each arm capped at 2 h wall
ANCHOR_CADENCE = 200  # 60 samples per 12000-step arm


# --------------------------------------------------------------------------- #
# Interpreter selection — defect 1
# --------------------------------------------------------------------------- #
def _probe(py: str) -> dict:
    """``import torch`` self-test; report CUDA availability."""
    code = (
        "import json,torch;"
        "print(json.dumps({'exe':__import__('sys').executable,"
        "'torch':torch.__version__,"
        "'cuda':bool(torch.cuda.is_available()),"
        "'ndev':torch.cuda.device_count()}))"
    )
    try:
        out = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=180)
        if out.returncode != 0:
            tail = (out.stderr or "").strip().splitlines()[-1:]
            return {"ok": False, "err": tail or ["?"]}
        info = json.loads(out.stdout.strip().splitlines()[-1])
        info["ok"] = bool(info.get("torch")) and bool(info.get("cuda")) and info.get("ndev", 0) >= 1
        return info
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "err": str(exc)}


def select_interpreter() -> tuple[str, list[dict]]:
    """Pick the first interpreter that imports torch and sees a CUDA device."""
    receipts = []
    for py in FALLBACK_PYTHONS:
        if not Path(py).exists():
            receipts.append({"python": py, "ok": False, "err": "missing"})
            continue
        info = _probe(py)
        info["python"] = py
        receipts.append(info)
        if info.get("ok"):
            return py, receipts
    raise SystemExit(
        "no interpreter with torch+CUDA found; receipts=" + json.dumps(receipts, indent=1)
    )


# --------------------------------------------------------------------------- #
# Physics bookkeeping
# --------------------------------------------------------------------------- #
def nu_lb_of(tau: float) -> float:
    return (tau - 0.5) / 3.0


def re_lattice(tau: float, L: int = RESOLUTION, u_lb: float = U_LB) -> float:
    """Effective Reynolds number of a ladder arm.

    ``u_lb`` and ``L`` are frozen by the benchmark, so ``ν_lb=(τ-0.5)/3`` makes
    τ itself the Reynolds knob: ``Re = u_lb·L/ν_lb``.  That is what the τ–Re
    coupling supplement quantifies.
    """
    return u_lb * float(L) / nu_lb_of(tau)


def blasius_cf(re: float) -> float:
    return 1.328 / math.sqrt(re)


def ittc1957_cf(re: float) -> float:
    if re <= 0.0:
        return float("nan")
    den = math.log10(re) - 2.0
    if abs(den) < 1e-12:
        return float("nan")
    return 0.075 / den**2


# --------------------------------------------------------------------------- #
# One arm
# --------------------------------------------------------------------------- #
def run_arm(arm: str, tau_target: float, py: str) -> dict:
    out_dir = STAGING / f"tau_{arm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    anchor_path = out_dir / "anchors.jsonl"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU  # GPU5 only
    env["SUBOFF_TAU_TARGET"] = repr(float(tau_target))
    env["SUBOFF_ANCHOR_PROBE"] = str(anchor_path)
    env["SUBOFF_ANCHOR_CADENCE"] = str(ANCHOR_CADENCE)
    env.pop("SUBOFF_VISCOSITY_OVERRIDE", None)  # retired m292 knob (never existed)

    cmd = [
        py,
        str(RUN_PY),
        "--hull-type",
        "with_sail",
        "--resolution",
        str(RESOLUTION),
        "--steps",
        str(STEPS),
        "--device",
        DEVICE,
        "--collision",
        COLLISION,
        "--friction",
        FRICTION,
        "--p0",
        P0,
        "--out",
        str(out_dir),
    ]

    print(
        f"\n{'=' * 70}\narm {arm}: tau={tau_target} nu_lb={nu_lb_of(tau_target):.6f} "
        f"Re_lattice={re_lattice(tau_target):.2f}\n{' '.join(cmd)}\n{'=' * 70}",
        flush=True,
    )

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(RUN_PY.parent),
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
        rc, timed_out, stdout, stderr = proc.returncode, False, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:

        def _txt(v):
            if v is None:
                return ""
            return v.decode("utf-8", "replace") if isinstance(v, bytes) else v

        rc, timed_out, stdout, stderr = None, True, _txt(exc.stdout), _txt(exc.stderr)
    wall = time.time() - t0

    (out_dir / "run.log").write_text(f"$ {' '.join(cmd)}\n\n{stdout}\n--- stderr ---\n{stderr}")

    row = {
        "arm": arm,
        "tau_target": float(tau_target),
        "nu_lb_target": nu_lb_of(tau_target),
        "re_lattice_target": re_lattice(tau_target),
        "returncode": rc,
        "timed_out": timed_out,
        "wall_s": round(wall, 1),
        "result_path": str(out_dir / f"result_L{RESOLUTION}.json"),
        "anchor_path": str(anchor_path),
        "status": "fail",
    }

    res_path = Path(row["result_path"])
    if not res_path.exists():
        row["reason"] = "result.json not found"
        row["stderr_tail"] = (stderr or stdout or "").strip().splitlines()[-6:]
        return row

    res = json.loads(res_path.read_text())
    row.update(_row_from_result(res))
    _apply_arm_sanity(row, tau_target)
    return row


def _row_from_result(res: dict) -> dict:
    """Map a result.json onto the arm-row fields (shared by run and rebuild)."""
    return {
        "tau_actual": res.get("tau"),
        "nu_lb_actual": res.get("nu_lb"),
        "u_lb_actual": res.get("u_lb"),
        "re_lattice_actual": res.get("effective_Re_lattice"),
        "Cd_total": res.get("Cd_total"),
        "Cd_pressure": res.get("Cd_pressure"),
        "Cd_friction": res.get("Cd_friction"),
        "Cd_total_last500": res.get("Cd_total_last500"),
        "convergence_windows": res.get("convergence_windows"),
        "finite": res.get("finite"),
        "diverged": res.get("diverged"),
        "n_steps_done": res.get("n_steps"),
        "ms_per_step": res.get("ms_per_step"),
        "elapsed_s": res.get("elapsed_s"),
        "domain_lu": res.get("domain_lu"),
        "solid_cells": res.get("solid_cells"),
        "anchor_probe": res.get("anchor_probe"),
        "tau_coupling": res.get("tau_coupling"),
    }


def _apply_arm_sanity(row: dict, tau_target: float) -> None:
    """prereg「停止规则」 sanity for one arm; sets status ok/fail."""
    problems = []
    if row.get("diverged"):
        problems.append("diverged")
    if row.get("finite") is False:
        problems.append("non_finite")
    tau_act = row.get("tau_actual")
    if tau_act is None or abs(float(tau_act) - float(tau_target)) > 1e-9:
        problems.append(f"tau_not_reached(actual={tau_act})")
    if row.get("n_steps_done") != STEPS:
        problems.append(f"steps_incomplete({row.get('n_steps_done')}/{STEPS})")
    row["sanity_problems"] = problems
    row["status"] = "ok" if not problems else "fail"


def rebuild_arm(arm: str, tau_target: float) -> dict:
    """Rebuild one arm's row from the on-disk result.json.

    Used by ``--reassemble`` so a gate-logic fix (or any post-processing change)
    can be re-applied to a finished ladder without burning another GPU hour.
    The *physics* is never recomputed here: every number comes from the
    ``result_L{res}.json`` that the simulation itself wrote.
    """
    out_dir = STAGING / f"tau_{arm}"
    row = {
        "arm": arm,
        "tau_target": float(tau_target),
        "nu_lb_target": nu_lb_of(tau_target),
        "re_lattice_target": re_lattice(tau_target),
        "result_path": str(out_dir / f"result_L{RESOLUTION}.json"),
        "anchor_path": str(out_dir / "anchors.jsonl"),
        "status": "fail",
        "reassembled": True,
    }
    res_path = Path(row["result_path"])
    if not res_path.exists():
        row["reason"] = "result.json not found"
        return row
    row["returncode"] = 0
    row["timed_out"] = False
    row["wall_s"] = None
    row.update(_row_from_result(json.loads(res_path.read_text())))
    _apply_arm_sanity(row, tau_target)
    return row


# --------------------------------------------------------------------------- #
# Anchor series reduction
# --------------------------------------------------------------------------- #
ANCHOR_KEYS = (
    "mem_fx",
    "mem_bgsub_fx",
    "E0_x",
    "E1_x",
    "E2_x",
    "NEQ_x",
    "ct",
    "e0",
    "ct_bgsub",
    "rho_fwd",
    "rho_mid",
    "rho_aft",
    "D_QW",
)


def reduce_anchors(path: str, tail: int = 12) -> dict:
    """Window means of the FA anchor family over the last ``tail`` samples."""
    p = Path(path)
    if not p.exists():
        return {"present": False, "reason": "missing"}
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if not rows:
        return {"present": False, "reason": "empty"}
    win = rows[-min(tail, len(rows)) :]
    n = len(win)
    means = {k: sum(float(r[k]) for r in win) / n for k in ANCHOR_KEYS}
    return {
        "present": True,
        "n_samples": len(rows),
        "window_samples": n,
        "window_steps": [win[0]["step"], win[-1]["step"]],
        "means": means,
        "max_abs_identity_residual": max(abs(float(r["identity_residual"])) for r in rows),
        "identity_residual_exact_zero": all(float(r["identity_residual"]) == 0.0 for r in rows),
        "sample_ms_mean": sum(float(r["sample_ms"]) for r in win) / n,
    }


# --------------------------------------------------------------------------- #
# Verdict assembly (gates verbatim from the frozen prereg)
# --------------------------------------------------------------------------- #
def build_verdict(arms: dict, anchor_red: dict, interp_receipts: list[dict], py: str) -> dict:
    cf_ref_1000 = blasius_cf(1000.0)  # prereg「参考锁定」primary reference

    def tau_of(a: str) -> float:
        r = arms[a]
        return float(r.get("tau_actual") or r["tau_target"])

    # ---- G1: Cd_total > 0 --------------------------------------------------
    g1 = {}
    for a in ARM_ORDER:
        cd = arms[a].get("Cd_total")
        g1[a] = {"Cd_total": cd, "pass": bool(cd is not None and cd > 0.0)}

    # ---- G2: |Cd/Cf_ref - 1| <= 50% ---------------------------------------
    # The prereg freezes Cf_ref at Re=1000.  Each arm sits at a different
    # effective Re (supplement §S1), so both calibers are reported: the frozen
    # one (governing) and the Re-matched one (diagnostic, not a gate).
    g2 = {}
    for a in ARM_ORDER:
        cd = arms[a].get("Cd_total")
        re_eff = re_lattice(tau_of(a))
        cf_matched = blasius_cf(re_eff)
        e_frozen = (cd / cf_ref_1000 - 1.0) if cd is not None else None
        e_matched = (cd / cf_matched - 1.0) if cd is not None else None
        g2[a] = {
            "Cd_total": cd,
            "cf_ref_frozen_Re1000": cf_ref_1000,
            "err_vs_frozen_pct": None if e_frozen is None else 100.0 * e_frozen,
            "pass_frozen": bool(e_frozen is not None and abs(e_frozen) <= 0.50),
            "re_effective": re_eff,
            "blasius_cf_at_re_effective": cf_matched,
            "ittc1957_cf_at_re_effective": ittc1957_cf(re_eff),
            "err_vs_rematched_pct": None if e_matched is None else 100.0 * e_matched,
            "in_band_rematched_50pct": bool(e_matched is not None and abs(e_matched) <= 0.50),
            "re_shift_vs_1000_pct": 100.0 * (re_eff - 1000.0) / 1000.0,
        }

    # ---- G3: Cd_total monotone in τ over >= 3 arms -----------------------
    pts = [(tau_of(a), arms[a].get("Cd_total")) for a in ARM_ORDER]
    pts = [(t, c) for t, c in pts if c is not None]
    diffs = [pts[i + 1][1] - pts[i][1] for i in range(len(pts) - 1)]
    inc = sum(1 for d in diffs if d > 0)
    dec = sum(1 for d in diffs if d < 0)
    g3 = {
        "series": [{"tau": t, "Cd_total": c} for t, c in pts],
        "diffs": diffs,
        "n_increasing": inc,
        "n_decreasing": dec,
        "pass": bool(max(inc, dec) >= 3 and (inc == 0 or dec == 0)),
        "note": "prereg G3 = Cd_total 随 τ 单调变化（至少 3 档单调）",
    }

    # ---- overall ----------------------------------------------------------
    ladder_ran = all(arms[a]["status"] == "ok" for a in ARM_ORDER)
    gates = {
        "G1_total_drag_positive": all(v["pass"] for v in g1.values()),
        # The prereg states one inequality, |Cd_total/Cf_ref - 1| <= 50%, and
        # exactly one Cd_total exists per arm, so the criterion applies to
        # every arm.  An `any()` aggregation would let a single in-band arm
        # carry four out-of-band arms to a PASS -- that is gate-loosening and
        # is not what the frozen prereg says.  Strict form, no free pass.
        "G2_magnitude_within_50pct_frozen_ref": all(v["pass_frozen"] for v in g2.values()),
        "G3_monotone_in_tau": g3["pass"],
    }
    # Diagnostic only (never enters `overall`): which arms would survive if the
    # reference were re-matched to each arm's own effective Re.
    n_in_band_rematched = sum(1 for v in g2.values() if v["in_band_rematched_50pct"])
    if not ladder_ran:
        overall = "INCOMPLETE"
    elif all(gates.values()):
        overall = "PASS"
    else:
        overall = "HONEST-FAIL"

    md5_actual = hashlib.md5(PREREG.read_bytes()).hexdigest()
    return {
        "schema": "tau-ladder-uniform-298/v1",
        "task": "m298",
        "mission": "SIR-1b",
        "prereg": {
            "path": str(PREREG),
            "md5_frozen": PREREG_MD5_FROZEN,
            "md5_actual": md5_actual,
            "md5_match": md5_actual == PREREG_MD5_FROZEN,
        },
        "interpreter": {"selected": py, "receipts": interp_receipts},
        "frozen_setup": {
            "case": "benchmarks/verified/suboff_sail_resistance",
            "hull_type": "with_sail",
            "resolution": RESOLUTION,
            "steps": STEPS,
            "u_lb": U_LB,
            "collision": COLLISION,
            "friction": FRICTION,
            "p0": P0,
            "gpu": GPU,
            "device_arg": DEVICE,
            "ladder": TAU_LADDER,
        },
        "reference": {
            "primary": "Blasius Cf = 1.328/sqrt(Re), Re=1000",
            "cf_ref_Re1000": cf_ref_1000,
            "normalization": "S = pi(2 R_lb)L + A_sail_own (lu^2)",
            "turbulent_cross_checks_not_applicable_at_Re1000": {
                "ittc1957_cf_Re1e7": ittc1957_cf(1.0e7),
                "ittc1957_cf_Re2e6": ittc1957_cf(2.0e6),
            },
        },
        "arms": {a: arms[a] for a in ARM_ORDER},
        "gates": {
            "G1_total_drag_positive": {
                "per_arm": g1,
                "all_positive": gates["G1_total_drag_positive"],
            },
            "G2_magnitude_vs_reference": {
                "per_arm": g2,
                "any_in_band_frozen": gates["G2_magnitude_within_50pct_frozen_ref"],
            },
            "G3_monotone_in_tau": g3,
        },
        "gates_summary": gates,
        "overall": overall,
        "anchor_family": {
            "caliber": (
                "Ladd link MEM on the fluid→solid crossing set, normalized by "
                "dpS = 0.5 u_lb^2 S_total_wetted (same caliber as Cd_total)"
            ),
            "per_arm": anchor_red,
        },
        "supplement_tau_re_coupling": {
            "status": "supplement (APPEND-ONLY, m298) — 非 prereg 冻结门",
            "note": (
                "prereg 冻结 Cf_ref@Re=1000，但每个臂自身的有效 Re 不同 (ν_lb=(τ-0.5)/3, "
                "u_lb 与 L 固定 ⇒ Re=u_lb·L/ν_lb)。本表量化该偏移，并把参考校到同有效 Re "
                "口径重判带内/带外；判门仍用冻结口径。"
            ),
            "statement": (
                "u_lb and L are frozen by the benchmark, so the ladder knob τ is "
                "itself the Reynolds knob: ν_lb=(τ-0.5)/3 and Re=u_lb·L/ν_lb. "
                "A τ↑ ⇒ ν↑ ⇒ Re↓ sweep is therefore not a constant-Re comparison, "
                "and the frozen Cf_ref at Re=1000 can only be applied to the arm "
                "whose effective Re is 1000."
            ),
            "per_arm": {
                a: {
                    "tau": tau_of(a),
                    "nu_lb": nu_lb_of(tau_of(a)),
                    "re_lattice": re_lattice(tau_of(a)),
                    "re_shift_vs_1000_pct": 100.0 * (re_lattice(tau_of(a)) - 1000.0) / 1000.0,
                    "blasius_cf_at_re": blasius_cf(re_lattice(tau_of(a))),
                    "ittc1957_cf_at_re": ittc1957_cf(re_lattice(tau_of(a))),
                }
                for a in ARM_ORDER
            },
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--reassemble",
        action="store_true",
        help="rebuild verdict.json from finished result_L*.json on disk; "
        "runs no simulation (physics numbers untouched, post-processing only)",
    )
    args = ap.parse_args()

    print("τ ladder — uniform grid suboff_sail_resistance (SIR-1b / m298)", flush=True)
    print(f"staging : {STAGING}", flush=True)

    md5 = hashlib.md5(PREREG.read_bytes()).hexdigest()
    print(f"prereg  : {PREREG} md5={md5}", flush=True)
    if md5 != PREREG_MD5_FROZEN:
        raise SystemExit(f"prereg md5 {md5} != frozen {PREREG_MD5_FROZEN} — refusing to run")

    arms: dict[str, dict] = {}
    anchor_red: dict[str, dict] = {}

    if args.reassemble:
        print("mode    : --reassemble (no simulation; post-processing only)", flush=True)
        py = "(reassemble: no interpreter probe)"
        receipts: list[dict] = []
        for arm in ARM_ORDER:
            row = rebuild_arm(arm, TAU_LADDER[arm])
            arms[arm] = row
            anchor_red[arm] = reduce_anchors(row["anchor_path"])
            print(
                f"[{arm}] status={row['status']} tau={row.get('tau_actual')} "
                f"Cd={row.get('Cd_total')} problems={row.get('sanity_problems')}",
                flush=True,
            )
    else:
        py, receipts = select_interpreter()
        print(f"python  : {py} (torch+CUDA self-test ok)", flush=True)
        for rec in receipts:
            print(
                f"  probe {rec.get('python')}: ok={rec.get('ok')} {rec.get('err', '')}", flush=True
            )

        for arm in ARM_ORDER:
            row = run_arm(arm, TAU_LADDER[arm], py)
            arms[arm] = row
            anchor_red[arm] = reduce_anchors(row["anchor_path"])
            print(
                f"[{arm}] status={row['status']} tau={row.get('tau_actual')} "
                f"Cd={row.get('Cd_total')} wall={row['wall_s']}s "
                f"problems={row.get('sanity_problems')}",
                flush=True,
            )
            # incremental dump so a crash never loses finished arms
            (STAGING / "arms_partial.json").write_text(
                json.dumps(
                    {"arms": arms, "anchor_family": anchor_red}, indent=2, ensure_ascii=False
                )
            )

    verdict = build_verdict(arms, anchor_red, receipts, py)
    (STAGING / "verdict.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False))
    print(f"\nverdict -> {STAGING / 'verdict.json'}", flush=True)
    print(f"overall = {verdict['overall']}  gates={verdict['gates_summary']}", flush=True)


if __name__ == "__main__":
    main()
