#!/usr/bin/env python
"""XVAL driver — TensorLBM Peng-Robinson pseudopotential vs openLBMPM.

Two arms, both driven by machine-written artefacts (no hand-typed number):

E1  formulation conformance
    The OSS P-R effective-mass expression is re-`eval`-ed **from the string that
    ``oss_probe.py`` extracted out of the OSS CUDA kernel**, and compared with
    TensorLBM's ``psi_eos_peng_robinson`` on a density sweep.  The mapping
    (``c0``, sign of ``G``, ``alpha``, ``a``, ``b``, ``R``) is derived from the
    gradient expansion of the Shan-Chen force, not fitted.

E2  thermodynamic accuracy
    LBM runs (planar slab = rigorous coexistence; droplet = Laplace/spurious)
    against the analytic PR equal-area Maxwell construction, reusing the
    validated probe machinery of ``benchmarks/eos_maxwell/probe_lib.py``.

Usage
-----
    python xval_driver.py --mode pilot          # timing smoke, writes nothing final
    python xval_driver.py --mode main --steps 20000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "benchmarks" / "eos_maxwell"))
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

import probe_lib as P  # noqa: E402
from tensorlbm.multiphase import psi_eos_peng_robinson  # noqa: E402

ART = HERE / "artifacts"

# EOS constants of the PR arm (Yuan & Schaefer), same in both implementations
PR_A, PR_B, PR_R = 2.0 / 49.0, 2.0 / 21.0, 1.0
#: PR critical temperature implied by (a, b, R) — matches the library lock file
T_C = PR_A / (0.45723553 / 0.07779607 * PR_B)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # pragma: no cover
        return "?"


# ---------------------------------------------------------------------------
# E1 — formulation conformance
# ---------------------------------------------------------------------------

def oss_psi_factory(facts: dict):
    """Build ψ(rho; T, c0, G) from the OSS source text stored in oss_facts.json."""
    p_src = facts["pr_kernel"]["p_EOS_expr_mapped"]
    psi_src = facts["pr_kernel"]["psi_expr_mapped"]

    def psi(rho, T, c0, G, A=PR_A, B=PR_B, R=PR_R, alpha=1.0):
        env = dict(rho=rho, T=T, A=A, B=B, R=R, alpha=alpha, c0=c0, G=G, sqrt=np.sqrt)
        p = eval(p_src, {"__builtins__": {}}, env)  # noqa: S307  (OSS source text)
        with np.errstate(invalid="ignore"):
            return eval(psi_src, {"__builtins__": {}}, dict(env, p=p))  # noqa: S307

    return psi


def arm_e1(facts: dict) -> dict:
    oss_psi = oss_psi_factory(facts)
    rho_np = np.concatenate([
        np.logspace(-3.0, np.log10(9.5), 400),
        np.linspace(0.02, 9.5, 400),
    ])
    rho_np = np.unique(rho_np)
    rho_t = torch.tensor(rho_np, dtype=torch.float64)

    rows = []
    for T_r in (0.55, 0.75, 0.8266, 0.95):
        T = T_r * T_C
        psi_lbm = psi_eos_peng_robinson(rho_t, T, a=PR_A, b=PR_B, R=PR_R).numpy()
        # stencil-consistent normalisation: c0 = cs^2 = 1/3, G = -1 (OSS sign)
        psi_oss_stencil = oss_psi(rho_np, T, c0=1.0 / 3.0, G=-1.0)
        # OSS's own macro-pressure convention (psi^2 term 3/2 G rho^2 => c0 = 3)
        psi_oss_diag = oss_psi(rho_np, T, c0=3.0, G=-1.0)

        # domain: both formulas are real only where p_EOS < rho/3
        ok = np.isfinite(psi_oss_stencil) & np.isfinite(psi_lbm) & (psi_oss_stencil > 0)
        ok_d = np.isfinite(psi_oss_diag) & np.isfinite(psi_lbm) & (psi_oss_diag > 0)
        rel = np.abs(psi_lbm[ok] - psi_oss_stencil[ok]) / psi_oss_stencil[ok]
        ratio_d = psi_lbm[ok_d] / psi_oss_diag[ok_d]
        rows.append(dict(
            T_r=T_r, T=T, n_compared=int(ok.sum()),
            rho_min=float(rho_np[ok].min()), rho_max=float(rho_np[ok].max()),
            max_rel_dev_stencil_c0_third=float(rel.max()),
            median_rel_dev_stencil_c0_third=float(np.median(rel)),
            c0_three_ratio_mean=float(ratio_d.mean()),
            c0_three_ratio_std=float(ratio_d.std()),
            c0_three_max_rel_dev=float(np.abs(ratio_d - ratio_d.mean()).max() / ratio_d.mean()),
            psi_lbm_max=float(psi_lbm[ok].max()),
        ))

    worst = max(r["max_rel_dev_stencil_c0_third"] for r in rows)
    return dict(
        rows=rows,
        eps_machine=float(np.finfo(np.float64).eps),
        threshold=1e-6,
        verdict_pass=bool(worst <= 1e-6),
        worst_rel_dev=worst,
        note=("psi is compared where BOTH formulas are real (p_EOS < rho*cs^2); "
              "the library clamps that branch to 0 (psi=0 above rho_cross) while "
              "the OSS source calls raw sqrt (NaN)."),
        sample=dict(rho=rho_np.tolist(),
                    psi_lbm=psi_eos_peng_robinson(rho_t, 0.8266 * T_C, a=PR_A, b=PR_B,
                                                  R=PR_R).tolist(),
                    psi_oss=oss_psi(rho_np, 0.8266 * T_C, c0=1.0 / 3.0, G=-1.0).tolist()),
    )


# ---------------------------------------------------------------------------
# E2 — LBM thermodynamics vs analytic Maxwell
# ---------------------------------------------------------------------------

T_R_TIERS = {"ratio10": 0.8266, "mid": 0.75, "highT": 0.95}


def maxwell_bundle(trs) -> dict:
    out = {}
    for name, T_r in trs.items():
        m = P.maxwell_at_tr("pr", T_r, G=1.0)
        out[name] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                     for k, v in m.items()
                     if k in ("T_r", "T", "rho_liquid", "rho_vapour", "ratio", "p_sat",
                              "rho_cross", "area_residual", "found")}
    return out


def run_cell(tag, maxwell, *, geometry, L, R=20.0, steps=20000, threads=4,
             sample=100, width=4.0, T_r=None) -> dict:
    spec = P.RunSpec(tag=tag, eos="pr", T_r=T_r, L=L, R=R, G=1.0, tau=1.0,
                     forcing="edm", geometry=geometry, steps=steps, sample=sample,
                     device="cpu", width=width, threads=threads)
    t0 = time.time()
    rec = P.run_probe(spec, maxwell)
    rec.pop("hist", None)  # keep the artefact small; tail means are already in rec
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["tag"] = tag
    rec["geometry"] = geometry
    return rec


def p_mech(rho, T, G=1.0):
    """Lattice mechanical pressure of the SCMP scheme, float64.

    p = rho*cs2 - (G/2)*cs2*psi^2 with the library's corrected-sign psi; for the
    PR arm this reconstructs p_EOS by construction (locked by
    tests/test_multiphase_psi_eos.py), so it is an independent read-out of the
    EOS pressure *from the measured densities*.
    """
    r64 = torch.as_tensor(rho, dtype=torch.float64)
    psi = psi_eos_peng_robinson(r64, T, a=PR_A, b=PR_B, R=PR_R)
    return r64 / 3.0 - 0.5 * G * (1.0 / 3.0) * psi * psi


def eos_readout(rec: dict, mw: dict) -> dict:
    """E2b: recover p_sat from the two measured plateaus (must agree with each other)."""
    T = mw["T"]
    pl = float(p_mech(rec["rho_l"], T))
    pv = float(p_mech(rec["rho_v"], T))
    ps = float(mw["p_sat"])
    return dict(
        p_l=pl, p_v=pv, p_sat_ref=ps,
        p_l_err=abs(pl - ps) / ps, p_v_err=abs(pv - ps) / ps,
        p_plates_mismatch=abs(pl - pv) / ps,
    )


def droplet_laplace(recs: list[dict], mw: dict) -> dict:
    """E3: Laplace pressure from the radial profiles of the droplet runs.

    dp(R) = p(r_core) - p(r_far) from p_mech(rho_mean); Laplace requires
    dp*R = const = sigma (2D).  Fitted by least squares through the origin and
    through both points, plus the slope/intercept of the 2-point line.
    """
    pts = []
    T = mw["T"]
    for r in recs:
        prof = r.get("radial_profile") or []
        if not prof or not r.get("stable"):
            continue
        core = [b for b in prof if b["r_mid"] <= 0.35 * r["spec"]["R"]]
        far = [b for b in prof if b["r_mid"] >= 0.80 * (r["spec"]["L"] / 2.0)]
        if not core or not far:
            continue
        p_core = float(np.mean([float(p_mech(b["rho_mean"], T)) for b in core]))
        p_far = float(np.mean([float(p_mech(b["rho_mean"], T)) for b in far]))
        pts.append(dict(tag=r["tag"], R=r["spec"]["R"], p_core=p_core, p_far=p_far,
                        dp=p_core - p_far, dp_times_R=(p_core - p_far) * r["R"],
                        r_eff=r.get("r_eff"), ma_max=r.get("ma_max")))
    out = dict(points=pts)
    if len(pts) >= 2:
        Rs = np.array([p["R"] for p in pts], float)
        dP = np.array([p["dp"] for p in pts], float)
        sigma = float(np.sum(Rs * dP) / np.sum(Rs * Rs))       # through origin
        k, b = np.polyfit(Rs, dP, 1)                            # free line
        out.update(sigma_through_origin=sigma, slope=k, intercept=b,
                   intercept_rel=abs(b) / max(abs(k) * Rs.mean(), 1e-30),
                   laplace_residual=float(np.max(np.abs(dP * Rs - sigma) / (sigma + 1e-30))))
    return out


def errors_against_maxwell(rec: dict, mw: dict) -> dict:
    def rel(a, b):
        return abs(a - b) / max(abs(b), 1e-30)

    out = dict(
        rho_l_err=rel(rec["rho_l"], mw["rho_liquid"]),
        rho_v_err=rel(rec["rho_v"], mw["rho_vapour"]),
        ratio_err=rel(rec["ratio"], mw["ratio"]),
    )
    out["worst"] = max(out.values())
    out["pass_15pct"] = bool(out["worst"] < 0.15)
    return out


def summarize(results: list[dict], e1: dict, laplace: dict) -> dict:
    """Apply the preregistered decision rules to the measured cells."""
    planar = [r for r in results
              if r["geometry"] == "planar" and r.get("errors")
              and "threads2" not in r["tag"]]
    primary = next((r for r in planar if r["reference_tier"] == "ratio10"
                    and r["spec"]["L"] == max(x["spec"]["L"] for x in planar
                                      if x["reference_tier"] == "ratio10")), None)
    d1 = "A0-conformant" if e1["verdict_pass"] else "A2-non-conformant"
    d2 = ("A0-accurate" if primary and primary["errors"]["pass_15pct"]
          else "A1-conformant-but-inaccurate")
    grid = [dict(tag=r["tag"], L=r["spec"]["L"], ratio=r.get("ratio"),
                 ratio_maxwell=r["ratio_maxwell"], ratio_err=r["errors"]["ratio_err"],
                 rho_l_err=r["errors"]["rho_l_err"], rho_v_err=r["errors"]["rho_v_err"],
                 stable=r["stable"], ma_max=r.get("ma_max"), steps=r["steps_run"])
            for r in planar if r["reference_tier"] == "ratio10"]
    return dict(
        D1=d1, D2=d2,
        primary_cell=primary["tag"] if primary else None,
        primary_worst_err=primary["errors"]["worst"] if primary else None,
        grid_ratio10=grid,
        laplace_sigma=laplace.get("sigma_through_origin"),
        laplace_residual=laplace.get("laplace_residual"),
        note="D2 is applied to the finest planar grid at the ratio-10 tier (prereg §7).",
    )


PRIMARY = [
    # (tag, geometry, L, steps, threads, T_r_key)
    ("planar_L48_ratio10", "planar", 48, 20000, 4, "ratio10"),
    ("planar_L64_ratio10", "planar", 64, 20000, 4, "ratio10"),
    ("planar_L96_ratio10", "planar", 96, 20000, 4, "ratio10"),
    ("planar_L128_ratio10", "planar", 128, 20000, 4, "ratio10"),   # predeclared refinement
    ("planar_L64_mid", "planar", 64, 20000, 4, "mid"),
    ("planar_L64_highT", "planar", 64, 20000, 4, "highT"),
    ("planar_L64_ratio10_threads2", "planar", 64, 12000, 2, "ratio10"),  # determinism
    ("droplet_L96_R20_ratio10", "droplet", 96, 20000, 4, "ratio10"),
    ("droplet_L96_R12_ratio10", "droplet", 96, 20000, 4, "ratio10"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="main", choices=["pilot", "main"])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--out", default=str(ART / "results.json"))
    a = ap.parse_args()

    ART.mkdir(parents=True, exist_ok=True)
    facts = json.loads((ART / "oss_facts.json").read_text())

    e1 = arm_e1(facts)
    print(f"[E1] worst rel dev={e1['worst_rel_dev']:.3e} pass={e1['verdict_pass']}")

    bundle = maxwell_bundle(T_R_TIERS)
    for k, v in bundle.items():
        print(f"[MW] {k}: T_r={v['T_r']:.4f} rho_l={v['rho_liquid']:.5f} "
              f"rho_v={v['rho_vapour']:.6f} ratio={v['ratio']:.4f} p_sat={v['p_sat']:.5f}")

    cells = list(PRIMARY) if a.mode == "main" else PRIMARY[:1]
    results = []
    payload = dict(
        generated=datetime.now(timezone.utc).isoformat(),
        mode=a.mode,
        e1=e1,
        maxwell=bundle,
        tc_implied=T_C,
        env=dict(python=platform.python_version(), torch=torch.__version__,
                 machine=platform.machine(), git=git_rev(),
                 driver_sha256=sha256(Path(__file__)),
                 probe_lib_sha256=sha256(REPO / "benchmarks/eos_maxwell/probe_lib.py"),
                 oss_facts_sha256=sha256(ART / "oss_facts.json")),
    )
    # predeclared stopping rule: the L=128 refinement runs only if L=96 misses
    # the 15 % band (so the artefact records the refinement as a consequence).
    refinement = [c for c in cells if c[2] == 128]
    cells = [c for c in cells if c[2] != 128]
    refine_tier = None

    for tag, geom, L, steps, threads, tr_key in cells:
        steps_i = a.steps if a.mode == "main" else 2000
        mw = bundle[tr_key]
        R = 20.0 if "R20" in tag else 12.0
        rec = run_cell(tag, mw, geometry=geom, L=L, R=R, steps=steps_i,
                       threads=threads, T_r=mw["T_r"])
        rec["reference_tier"] = tr_key
        rec["errors"] = errors_against_maxwell(rec, mw) if rec.get("rho_l") else None
        e = rec.get("errors") or {}
        print(f"[E2] {tag:32s} steps={rec['steps_run']:6d} stable={rec['stable']} "
              f"ratio={rec.get('ratio'):.4g} err={e.get('worst', float('nan')):.4g} "
              f"ma={rec.get('ma_max', float('nan')):.3g} wall={rec['wall_s']}s")
        if rec.get("rho_l") and geom == "planar":
            rec["eos_readout"] = eos_readout(rec, mw)
        results.append(rec)
        payload["cells"] = results
        Path(a.out).write_text(json.dumps(payload, indent=1))  # incremental
        if L == 96 and geom == "planar" and refine_tier is None:
            refine_tier = tr_key
            if not (rec["errors"] and rec["errors"]["pass_15pct"]) and refinement:
                print("        -> L=96 misses the 15 % band: running the predeclared"
                      " L=128 refinement")
                cells = refinement + [c for c in cells if c[0] != refinement[0][0]]

    payload["cells"] = results
    payload["laplace"] = droplet_laplace(
        [r for r in results if r["geometry"] == "droplet"], bundle["ratio10"])
    payload["verdict"] = summarize(results, e1, payload["laplace"])
    Path(a.out).write_text(json.dumps(payload, indent=1))
    print("verdict:", json.dumps(payload["verdict"], indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()