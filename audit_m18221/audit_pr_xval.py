#!/usr/bin/env python3
"""Independent audit of m18128 / XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL.

Written from scratch by the board auditor (m18221).  It re-derives the three
load-bearing claims of the delivery *without* importing the delivery's own
helpers (no `xval_driver`, no `oss_facts.json`, no `probe_lib`):

  A. OSS side: the P-R kernel really exists in the mirrored source, its ψ is
     what the delivery says it is, and it has zero call sites.
  B. E1 conformance: ψ from the (independently re-parsed) CUDA kernel text vs
     TensorLBM's `psi_eos_peng_robinson`, float64 sweep, 4 reduced temperatures.
  C. E2 reference: Peng-Robinson coexistence (rho_l, rho_v, p_sat) by an
     independent equal-area Maxwell construction, compared with the delivery's
     N5 table.

Output: audit_pr_xval.json next to this file (+ human summary on stdout).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

OSS = Path("/tmp/openLBMPM")
DELIV = Path("/root/tp-agent/work/m18128/xval/XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL")
REPO = Path("/root/tp-agent/work/m18128")
OUT = Path(__file__).resolve().parent

# Yuan & Schaefer (2006) parameters, as used on both sides of the comparison
PR_A, PR_B, PR_R = 2.0 / 49.0, 2.0 / 21.0, 1.0
T_C = PR_A / (0.45723553 / 0.07779607 * PR_B)   # re-derived, not copied
CS2 = 1.0 / 3.0


# ---------------------------------------------------------------------------
# A. OSS side — my own parse of the CUDA kernel text
# ---------------------------------------------------------------------------
def decontinue(src: str) -> str:
    return re.sub(r"\\\s*\n\s*", " ", src)


def parse_kernel():
    path = OSS / "ShanChen2D" / "OptimizedD2Q9GPU.py"
    text = path.read_text()
    lines = text.splitlines()
    # locate the kernel body
    start = next(i for i, l in enumerate(lines) if l.startswith("def calFluidPotentialGPUPR"))
    body = decontinue("\n".join(lines[start : start + 40]))
    # tmpP = <expr>  (up to the newline separating the next statement)
    m_p = re.search(r"tmpP\s*=\s*(.*?)\n", body, re.S)
    p_expr = m_p.group(1).strip()
    m_psi = re.search(r"fluidPotential\[i, indices\]\s*=\s*(sqrt\(.*?\))\s*\n", body, re.S)
    psi_expr = m_psi.group(1).strip()
    return dict(file=str(path.relative_to(OSS)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                p_expr=p_expr, psi_expr=psi_expr,
                callsites_whole_repo=None, git_head=None)


def count_callsites(kernel: str) -> dict:
    hits = {}
    for f in OSS.rglob("*.py"):
        n = sum(1 for l in f.read_text(errors="ignore").splitlines()
                if kernel in l and not l.strip().startswith("def "))
        if n:
            hits[str(f.relative_to(OSS))] = n
    return hits


def oss_psi_factory(parsed):
    """Evaluate the kernel text itself: identifier map, then eval."""
    p_src = parsed["p_expr"].replace("fluidRho[i, indices]", "rho")
    psi_src = parsed["psi_expr"].replace("fluidRho[i, indices]", "rho")
    for k, v in (("constR", "R"), ("temperatureT", "T"), ("coeffA", "A"),
                 ("coeffB", "B"), ("coeffAlpha", "alpha"), ("constC0", "c0"),
                 ("constG", "G"), ("tmpP", "tmpP"), ("sqrt", "sqrt")):
        p_src = re.sub(rf"\b{k}\b", v, p_src)
        psi_src = re.sub(rf"\b{k}\b", v, psi_src)
    env = {"sqrt": math.sqrt, "__builtins__": {}}

    def psi(rho, T, c0, G, A=PR_A, B=PR_B, R=PR_R, alpha=1.0):
        e = dict(env, rho=float(rho), T=T, A=A, B=B, R=R, alpha=alpha, c0=c0, G=G)
        try:
            tmpP = eval(p_src, e)          # noqa: S307 - OSS source text
            return eval(psi_src, dict(e, tmpP=tmpP))  # noqa: S307
        except (ValueError, ZeroDivisionError):
            return float("nan")

    return psi, p_src, psi_src


# ---------------------------------------------------------------------------
# B. E1 — conformance sweep
# ---------------------------------------------------------------------------
def arm_e1(psi_oss):
    sys.path.insert(0, str(REPO / "src"))
    from tensorlbm.multiphase import psi_eos_peng_robinson
    import torch

    rho_np = np.unique(np.concatenate([np.logspace(-3.0, np.log10(9.5), 400),
                                       np.linspace(0.02, 9.5, 400)]))
    rho_t = torch.tensor(rho_np, dtype=torch.float64)
    rows = []
    for T_r in (0.55, 0.75, 0.8266, 0.95):
        T = T_r * T_C
        lbm = psi_eos_peng_robinson(rho_t, T, a=PR_A, b=PR_B, R=PR_R).numpy()
        oss3 = np.array([psi_oss(r, T, c0=CS2, G=-1.0) for r in rho_np])
        oss_c0_3 = np.array([psi_oss(r, T, c0=3.0, G=-1.0) for r in rho_np])
        ok = np.isfinite(oss3) & np.isfinite(lbm) & (oss3 > 0)
        ok_d = np.isfinite(oss_c0_3) & np.isfinite(lbm) & (oss_c0_3 > 0)
        rel = np.abs(lbm[ok] - oss3[ok]) / oss3[ok]
        ratio = lbm[ok_d] / oss_c0_3[ok_d]
        rows.append(dict(T_r=T_r, T=T, n=int(ok.sum()),
                         max_rel_dev=float(rel.max()), median=float(np.median(rel)),
                         c0_3_ratio_mean=float(ratio.mean()),
                         c0_3_ratio_std=float(ratio.std()),
                         lbm_zero_above=int((lbm == 0).sum())))
    return dict(rows=rows, worst=float(max(r["max_rel_dev"] for r in rows)),
                threshold=1e-6,
                pass_=bool(max(r["max_rel_dev"] for r in rows) <= 1e-6),
                eps=float(np.finfo(np.float64).eps))


# ---------------------------------------------------------------------------
# C. E2 reference — independent PR Maxwell equal-area construction
# ---------------------------------------------------------------------------
def p_eos(rho, T, A=PR_A, B=PR_B, R=PR_R, alpha=1.0):
    rho = np.asarray(rho, dtype=float)
    return (rho * R * T) / (1.0 - B * rho) - (A * alpha * rho * rho) / (
        1.0 + 2.0 * B * rho - B * B * rho * rho)


def spinodal(T):
    """rho where dp/drho = 0 (three real roots => two-phase region)."""
    r = np.linspace(1e-6, 1.0 / PR_B - 1e-6, 400001)
    p = p_eos(r, T)
    dp = np.diff(p)
    idx = np.where(np.sign(dp[1:]) != np.sign(dp[:-1]))[0]
    return [float(r[i + 1]) for i in idx], r, p


def dpsi_drho(r, T):
    """d p_mech / d rho for the SC-LBM mechanical pressure p = rho cs2 + (G cs2 /2) psi^2."""
    h = 1e-6 * max(r, 1e-3)
    return (p_mech(r + h, T) - p_mech(r - h, T)) / (2 * h)


def psi_pr(rho, T):
    arg = 2.0 * (rho * CS2 - p_eos(rho, T)) / CS2
    return np.sqrt(np.clip(arg, 0.0, None))


def p_mech(rho, T, G=1.0):
    """Mechanical pressure in the library's *corrected-sign* convention
    (G=+1 = attraction): p = rho cs2 - (G cs2 / 2) psi^2 == p_EOS by construction."""
    return rho * CS2 - 0.5 * G * CS2 * psi_pr(rho, T) ** 2


def maxwell_independent(T, n_scan=1200, n_int=8001):
    """Equal-area Maxwell construction, independent implementation.

    Unknown: the vapour root rho_v; then p_sat = p_EOS(rho_v) fixes the liquid
    root rho_l (the other crossing), and the equal-area condition is

        e2(rho_v) = p_sat*(1/rho_v - 1/rho_l) - int_{rho_v}^{rho_l} p(rho)/rho^2 drho = 0

    (v = 1/rho substitution of  int (p - p_sat) dv = 0).  Solved by a sign-change
    scan + Brent refinement; no delivery code is used.
    """
    from scipy.optimize import brentq

    rho_spin_lo, rho_spin_hi = spinodal_roots(T)
    rho_top = 0.9999 / PR_B

    def liquid_root(p_sat):
        fa = p_eos(rho_spin_hi, T) - p_sat
        fb = p_eos(rho_top, T) - p_sat
        if fa * fb > 0:
            raise ValueError("no liquid root")
        return brentq(lambda r: p_eos(r, T) - p_sat, rho_spin_hi, rho_top,
                      xtol=1e-14, rtol=1e-15, maxiter=200)

    def e2(rho_v):
        p_sat = p_eos(rho_v, T)
        try:
            rho_l = liquid_root(p_sat)
        except ValueError:
            return float("nan"), p_sat, float("nan")
        rho = np.linspace(rho_v, rho_l, n_int)
        integ = np.trapezoid(p_eos(rho, T) / rho**2, rho)
        return p_sat * (1.0 / rho_v - 1.0 / rho_l) - integ, p_sat, rho_l

    grid = np.linspace(rho_spin_lo * 1e-3, rho_spin_lo * 0.999, n_scan)
    vals = np.array([e2(r)[0] for r in grid])
    ok = np.isfinite(vals)
    idx = np.where(ok[:-1] & ok[1:] & (np.sign(vals[:-1]) * np.sign(vals[1:]) < 0))[0]
    if idx.size == 0:
        raise ValueError(f"no equal-area bracket at T={T}")
    i = idx[-1]
    rho_v = brentq(lambda r: e2(r)[0], grid[i], grid[i + 1], xtol=1e-14,
                   rtol=1e-15, maxiter=300)
    resid, p_sat, rho_l = e2(rho_v)
    rho_ck = np.linspace(rho_v, rho_l, 400001)
    integ = np.trapezoid(p_eos(rho_ck, T) / rho_ck**2, rho_ck)
    v_span = 1.0 / rho_v - 1.0 / rho_l
    area_res = p_sat * v_span - integ           # == 0 at solution
    return dict(T_r=T / T_C, T=T, rho_l=float(rho_l), rho_v=float(rho_v),
                ratio=float(rho_l / rho_v), p_sat=float(p_sat),
                area_residual=float(area_res),
                n_brackets=int(idx.size),
                p_mech_balance=float(p_mech(rho_l, T) - p_mech(rho_v, T)),
                rho_cross=float(rho_cross(T)))


def spinodal_roots(T):
    r = np.linspace(1e-6, 0.9999 / PR_B, 200001)
    p = p_eos(r, T)
    s = np.where(np.diff(np.sign(np.diff(p))) != 0)[0]
    xs = [float(r[i + 1]) for i in s]
    return min(xs), max(xs)


def rho_cross(T):
    from scipy.optimize import brentq
    return brentq(lambda r: p_eos(r, T) - r * CS2, 1e-6, 0.9999 / PR_B)


def main():
    out = {}
    # ---- A. OSS side ----
    parsed = parse_kernel()
    parsed["callsites_whole_repo"] = count_callsites("calFluidPotentialGPUPR")
    try:
        parsed["git_head"] = subprocess.run(
            ["git", "-C", str(OSS), "rev-parse", "HEAD"], capture_output=True,
            text=True).stdout.strip()
    except Exception:                                    # pragma: no cover
        parsed["git_head"] = None
    psi_oss, p_src, psi_src = oss_psi_factory(parsed)
    parsed["p_eval_src"] = p_src
    parsed["psi_eval_src"] = psi_src
    out["oss"] = parsed
    print("[A] OSS kernel:", parsed["file"], parsed["sha256"][:16],
          "HEAD", (parsed["git_head"] or "?")[:12],
          "callsites", parsed["callsites_whole_repo"])

    # ---- B. E1 ----
    out["e1"] = arm_e1(psi_oss)
    print("[B] E1 worst rel dev = %.3e  (threshold 1e-6 -> %s)" %
          (out["e1"]["worst"], out["e1"]["pass_"]))

    # ---- C. independent Maxwell ----
    mw = {}
    for key, tr in (("ratio10", 0.8266), ("mid", 0.75), ("highT", 0.95)):
        mw[key] = maxwell_independent(tr * T_C)
        print("[C] %-8s Tr=%.4f rho_l=%.5f rho_v=%.5f ratio=%.4f p_sat=%.5f area=%.2e" %
              (key, mw[key]["T_r"], mw[key]["rho_l"], mw[key]["rho_v"],
               mw[key]["ratio"], mw[key]["p_sat"], mw[key]["area_residual"]))
    out["maxwell_independent"] = mw

    # delivery's N5 table (parsed from the artefact, not typed by hand)
    deliv = json.loads((DELIV / "artifacts" / "results.json").read_text())
    out["maxwell_delivery"] = {k: {kk: v[kk] for kk in ("T_r", "T", "rho_liquid",
                                                       "rho_vapour", "ratio", "p_sat")}
                               for k, v in deliv["maxwell"].items()}
    cmp_ = {}
    for k in mw:
        d = out["maxwell_delivery"].get(k)
        if not d:
            continue
        cmp_[k] = {f: dict(audit=float(mw[k][f]), delivery=float(d[df]),
                           rel=float(abs(mw[k][f] - d[df]) / abs(d[df])))
                   for f, df in (("rho_l", "rho_liquid"), ("rho_v", "rho_vapour"),
                                 ("ratio", "ratio"), ("p_sat", "p_sat"))}
    out["maxwell_compare"] = cmp_
    for k, v in cmp_.items():
        print("[C] %-8s max |rel diff| vs delivery = %.2e" %
              (k, max(x["rel"] for x in v.values())))

    # ---- D. independent re-run of one production cell (bit-level check) ----
    cells = [json.loads(l) for l in open(OUT / "cells_audit.jsonl")] if (OUT / "cells_audit.jsonl").exists() else []
    committed = [json.loads(l) for l in (DELIV / "artifacts" / "cells.jsonl").read_text().splitlines() if l.strip()]
    repro = {}
    for c in cells:
        tag = c["tag"]
        src = None
        if "L48" in tag:
            src = next((x for x in committed if x["tag"] == "local_planar_L48_ratio10"), None)
        if src:
            repro[tag] = {k: dict(audit=float(c[k]), committed=float(src[k]),
                                  abs_diff=float(abs(c[k] - src[k])))
                          for k in ("rho_l", "rho_v", "ratio", "ma_max")}
            repro[tag]["error_vs_maxwell"] = dict(audit=float(c["errors"]["worst"]),
                                                 committed=float(src["errors"]["worst"]))
            repro[tag]["bit_identical"] = bool(c["ratio"] == src["ratio"] and
                                               c["errors"]["worst"] == src["errors"]["worst"])
    out["reproduction"] = repro
    for k, v in repro.items():
        print("[D] %s bit_identical=%s ratio audit=%.12f committed=%.12f" %
              (k, v["bit_identical"], v["ratio"]["audit"], v["ratio"]["committed"]))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit_pr_xval.json").write_text(json.dumps(out, indent=1))
    print("wrote", OUT / "audit_pr_xval.json")


if __name__ == "__main__":
    main()