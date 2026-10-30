#!/usr/bin/env python
"""SDAA sweep for sphere_re200 benchmark.

Runs the sphere Re=200 drag case on the 'sdaa' device, computes all
verification criteria from the raw force history, and writes results
to artifacts/sdaa_sphere_re200.json.

This script does NOT modify any business code. It invokes the existing
run_case() from benchmarks/verified/sphere_re200/run.py with device
set to sdaa:0.

Grid/step choices: D=16 (smallest ladder tier), steps=2500 (feasible
within 20-min wall clock on sdaa; ~0.45s/step → ~1125s). The case
README recommends 12000 steps but that exceeds the wall-clock budget.
Steps >= 8000 is a steady-convergence gate; with 2500 steps this
criterion will honestly FAIL.

Wall-clock target: <= 20 minutes.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

# ── path setup: src first, then benchmark dir ────────────────────────
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO, "src")
BENCH_DIR = os.path.join(REPO, "benchmarks", "verified", "sphere_re200")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, BENCH_DIR)

# ── device setup ──────────────────────────────────────────────────────
os.environ["W8A_DEV"] = "sdaa:0"

import torch  # noqa: E402

from run import run_case  # noqa: E402

# ── constants (from README / verify.py) ──────────────────────────────
RE_LOCK = 200.0
CD_REF_LOCK = 24.0 / RE_LOCK * (1.0 + 0.15 * RE_LOCK**0.687)  # 0.8056146866904639
GATE = 0.03  # 3%

# Case parameters (D16 tier from README, reduced steps for wall-clock budget)
D = 16
STEPS = 2500  # README recommends 12000; ~0.45s/step on sdaa → ~1125s for 2500
LAT = 3.0
UP = 3.0
DOWN = 4.0
KERNEL = "sparse"
TREAT = "hard"
SAMPLE_INTERVAL = 50
TAU = 0.5 + 3.0 * 0.05 * D / RE_LOCK  # 0.512
U_LB = 0.05

ARTIFACTS_DIR = os.path.join(REPO, "artifacts")
OUT_JSON = os.path.join(ARTIFACTS_DIR, "sdaa_sphere_re200.json")


def steady_window(hist, key="cd", frac=0.2):
    """Compute steady-state window mean, previous window mean, and drift."""
    vals = [h[key] for h in hist]
    n = len(vals)
    w = max(1, int(n * frac))
    tail = vals[-w:]
    prev = vals[-2 * w : -w] if n >= 2 * w else vals[:w]
    m_tail = sum(tail) / len(tail)
    m_prev = sum(prev) / len(prev)
    drift = (m_tail - m_prev) / m_prev * 100.0
    return m_tail, m_prev, drift, w


def cv_closure(hist):
    """Compute CV momentum-budget closure if cv_mom_cd is available."""
    pairs = [(h["cd"], h["cv_mom_cd"]) for h in hist if "cv_mom_cd" in h]
    if not pairs:
        return None, None, 0
    rel = [abs(b - a) / abs(a) for a, b in pairs]
    mean_ledger = sum(a for a, _ in pairs) / len(pairs)
    mean_cv = sum(b for _, b in pairs) / len(pairs)
    window_rel = abs(mean_cv - mean_ledger) / mean_ledger
    return sum(rel) / len(rel), window_rel, len(pairs)


def main():
    t_start = time.time()

    # ── device evidence ───────────────────────────────────────────────
    sdaa_available = torch.sdaa.is_available()
    sdaa_count = torch.sdaa.device_count()
    device_name = torch.sdaa.get_device_name(0) if sdaa_count > 0 else "N/A"

    print(f"[SDAA] available={sdaa_available}, count={sdaa_count}, name={device_name}")

    if not sdaa_available:
        print("[FAIL] SDAA not available — cannot run on sdaa device.")
        result = {
            "case": "sphere_re200",
            "device": "sdaa",
            "device_evidence": {
                "sdaa_available": False,
                "sdaa_device_count": sdaa_count,
                "device_name": device_name,
            },
            "status": "FAIL",
            "note": "SDAA device not available",
            "judgments": [],
        }
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        with open(OUT_JSON, "w") as fh:
            json.dump(result, fh, indent=2)
        return

    # ── run simulation ────────────────────────────────────────────────
    tmp_out = OUT_JSON + ".raw.json"
    print(f"[RUN] D={D}, steps={STEPS}, tau={TAU}, u_lb={U_LB}, kernel={KERNEL}, treat={TREAT}")
    print(f"[RUN] lat={LAT}, up={UP}, down={DOWN}")
    print(f"[NOTE] steps reduced from 12000 (README rec) to {STEPS} for wall-clock budget")

    run_case(
        out_path=tmp_out,
        kernel=KERNEL,
        D=D,
        steps=STEPS,
        lat=LAT,
        up=UP,
        down=DOWN,
        tau=TAU,
        ulb=U_LB,
        treat=TREAT,
        sample=SAMPLE_INTERVAL,
        cv_tail=0,  # no CV tail for speed
        mem_stats=False,
    )

    wall_s = time.time() - t_start
    print(f"[DONE] wall clock = {wall_s:.1f}s")

    # ── load results ──────────────────────────────────────────────────
    with open(tmp_out) as fh:
        data = json.load(fh)

    meta = data["meta"]
    hist = data["history"]

    # ── compute verification criteria ─────────────────────────────────
    cd_window, cd_prev, drift_pct, n_window = steady_window(hist)
    err_pct = (cd_window - CD_REF_LOCK) / CD_REF_LOCK * 100.0

    tau_expect = 3.0 * meta["u_lb"] * meta["D"] / RE_LOCK + 0.5
    tau_ok = abs(meta["tau"] - tau_expect) < 1e-9
    re_ok = abs(meta["re_eff"] - RE_LOCK) < 1e-6
    steady_ok = abs(drift_pct) < 0.3 and meta["steps"] >= 8000 and tau_ok
    within_gate = abs(err_pct / 100.0) <= GATE

    cv_rel, cv_win, cv_n = cv_closure(hist)

    # ── build judgments ───────────────────────────────────────────────
    judgments = [
        {
            "criterion": "Re locked at 200",
            "value": meta["re_eff"],
            "expected": RE_LOCK,
            "tolerance": 1e-6,
            "pass": bool(re_ok),
        },
        {
            "criterion": "tau matches Re=200 law",
            "value": meta["tau"],
            "expected": tau_expect,
            "tolerance": 1e-9,
            "pass": bool(tau_ok),
        },
        {
            "criterion": "Steady convergence (|drift| < 0.3%, steps >= 8000, tau_ok)",
            "value": {
                "drift_pct": drift_pct,
                "steps": meta["steps"],
                "tau_ok": tau_ok,
            },
            "expected": {"|drift|": "< 0.3%", "steps": ">= 8000", "tau_ok": True},
            "pass": bool(steady_ok),
        },
        {
            "criterion": f"Cd error vs Schiller-Naumann (gate {GATE*100:.0f}%)",
            "value": {
                "cd_window": cd_window,
                "cd_ref": CD_REF_LOCK,
                "err_pct": err_pct,
            },
            "expected": f"|err| <= {GATE*100:.0f}%",
            "pass": bool(within_gate),
        },
        {
            "criterion": "No divergence",
            "value": meta["diverged"],
            "expected": False,
            "pass": not meta["diverged"],
        },
    ]

    if cv_n > 0:
        judgments.append({
            "criterion": "CV momentum-budget closure",
            "value": {
                "cv_rel_mean_pct": cv_rel * 100.0,
                "cv_window_rel_pct": cv_win * 100.0,
                "cv_samples": cv_n,
            },
            "expected": "CV close to ledger",
            "pass": True,  # informational
        })

    all_pass = all(j["pass"] for j in judgments)

    # ── write artifact ────────────────────────────────────────────────
    result = {
        "case": "sphere_re200",
        "device": "sdaa",
        "device_evidence": {
            "sdaa_available": sdaa_available,
            "sdaa_device_count": sdaa_count,
            "device_name": device_name,
            "torch_sdaa_get_device_name_0": device_name,
        },
        "config": {
            "D": D,
            "steps": STEPS,
            "lat": LAT,
            "up": UP,
            "down": DOWN,
            "kernel": KERNEL,
            "treat": TREAT,
            "tau": TAU,
            "u_lb": U_LB,
            "sample_interval": SAMPLE_INTERVAL,
            "steps_note": "Reduced from 12000 (README rec) to fit 20-min wall clock on sdaa",
        },
        "meta_from_run": meta,
        "reference": {
            "formula": "Cd = 24/Re * (1 + 0.15*Re^0.687)",
            "re": RE_LOCK,
            "cd_ref": CD_REF_LOCK,
            "gate_pct": GATE * 100,
        },
        "computed": {
            "cd_window": cd_window,
            "cd_prev_window": cd_prev,
            "drift_pct": drift_pct,
            "n_window": n_window,
            "err_pct": err_pct,
            "tau_expected": tau_expect,
            "tau_ok": tau_ok,
            "re_ok": re_ok,
            "steady_ok": steady_ok,
            "within_gate": within_gate,
            "cv_closure_rel_pct": cv_rel * 100.0 if cv_rel is not None else None,
            "cv_closure_window_pct": cv_win * 100.0 if cv_win is not None else None,
            "cv_samples": cv_n,
        },
        "judgments": judgments,
        "wall_clock_s": wall_s,
        "status": "PASS" if all_pass else "FAIL",
        "triton_workaround": None,  # no triton dependency in this run path
    }

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")

    # Clean up raw file
    if os.path.exists(tmp_out):
        os.remove(tmp_out)

    print(f"\n[RESULT] status={result['status']}")
    for j in judgments:
        status = "PASS" if j["pass"] else "FAIL"
        print(f"  [{status}] {j['criterion']}")
    print(f"  wall_clock = {wall_s:.1f}s")
    print(f"  artifact = {OUT_JSON}")


if __name__ == "__main__":
    main()
