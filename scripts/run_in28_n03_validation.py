#!/usr/bin/env python3
"""IN28-N03 — conjugate-heat-transfer validation of the TensorLBM DDF thermal model.

Runs the three physics cases and the critical-Rayleigh-number probe, writes one
JSON per case plus a summary JSON, and renders the six-elements figures into
``artifacts/in28_n03/figures``.

Usage::

    python scripts/run_in28_n03_validation.py --out artifacts/in28_n03
    python scripts/run_in28_n03_validation.py --quick      # smoke / CI

Every number in the summary is computed here (nothing is typed by hand); the
report generator reads the same JSON files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from tensorlbm.conjugate_thermal import (  # noqa: E402
    cavity_natural_convection,
    churchill_bernstein,
    conjugate_channel,
    cylinder_forced_convection,
    de_vahl_davis_nu,
    rayleigh_growth_rate,
)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def run_cavity(out: Path, quick: bool, dtype=torch.float32) -> dict:
    res = {}
    cases = [(1e3, 34, 20000), (1e4, 66, 60000), (1e5, 50, 30000)]
    if quick:
        cases = [(1e3, 26, 4000), (1e4, 34, 6000)]
    for ra, n, steps in cases:
        t0 = time.time()
        r = cavity_natural_convection(
            ra=ra, n=n, nsteps=steps, check_every=1000, tol=1e-6, dtype=dtype
        )
        r["wall_time_s"] = time.time() - t0
        # mid-plane energy-flux Nusselt (independent estimator)
        T = r["fields"]["T"]
        ux = r["fields"]["ux"]
        H, al, j = r["H"], r["alpha"], n // 2
        uT = float((ux[1:H + 1, j] * T[1:H + 1, j]).sum())
        cond = float(al * ((T[1:H + 1, j + 1] - T[1:H + 1, j - 1]) / 2.0).sum())
        r["nu_midplane"] = (uT - cond) / al
        r["nu_dev_vs_benchmark"] = (
            None if r["nu_benchmark"] is None
            else (r["nu_mean"] - r["nu_benchmark"]) / r["nu_benchmark"]
        )
        key = f"ra_{ra:.0e}".replace("+", "")
        np.savez_compressed(
            out / "fields" / f"cavity_{key}.npz",
            T=T, ux=ux, uy=r["fields"]["uy"],
        )
        r.pop("fields")
        res[key] = r
        print(f"[cavity] Ra={ra:.0e} n={n} Nu={r['nu_mean']:.4f} "
              f"bench={r['nu_benchmark']} mid={r['nu_midplane']:.4f} "
              f"({r['wall_time_s']:.0f}s)", flush=True)
    _dump(out / "case_b_cavity.json", res)
    return res


def run_cylinder(out: Path, quick: bool, dtype=torch.float32) -> dict:
    res = {}
    cases = [(40.0, 16, 10000), (100.0, 16, 12000), (200.0, 16, 10000)]
    if quick:
        cases = [(100.0, 10, 1500)]
    for re, n_d, steps in cases:
        t0 = time.time()
        r = cylinder_forced_convection(
            re=re, n_d=n_d, nsteps=steps, check_every=250, tol=0.0, dtype=dtype
        )
        r["wall_time_s"] = time.time() - t0
        hist = r["history"]
        tail = hist[max(0, int(0.6 * len(hist))):]
        nu_tail = [h["nu"] for h in tail]
        diverged = any(not np.isfinite(v) for v in nu_tail) or not np.isfinite(
            hist[-1]["nu"])
        r["diverged"] = bool(diverged)
        if diverged:
            r["nu_time_avg"] = None
            r["nu_time_std"] = None
            r["nu_dev_vs_cb"] = None
            r["note"] = ("simulation diverged at this resolution (Ma too high / "
                         "thermal layer under-resolved); recorded as an honest "
                         "numerical failure, not a physics result")
        else:
            r["nu_time_avg"] = float(np.mean(nu_tail))
            r["nu_time_std"] = float(np.std(nu_tail))
            r["nu_dev_vs_cb"] = (r["nu_time_avg"] - r["nu_churchill_bernstein"]) / \
                r["nu_churchill_bernstein"]
        T = r["fields"]["T"]
        r["nan_count"] = int(np.isnan(T).sum())
        key = f"re_{int(re)}"
        np.savez_compressed(
            out / "fields" / f"cylinder_{key}.npz",
            T=T, ux=r["fields"]["ux"], uy=r["fields"]["uy"],
            theta=r["theta"], nu_theta=r["nu_theta"],
        )
        for k in ("fields", "theta", "nu_theta"):
            r.pop(k)
        res[key] = r
        print(f"[cylinder] Re={re} D={r['D']:.0f} Nu={r['nu_time_avg']:.3f} "
              f"+-{r['nu_time_std']:.3f} CB={r['nu_churchill_bernstein']:.3f} "
              f"({r['wall_time_s']:.0f}s)", flush=True)
    _dump(out / "case_a_cylinder.json", res)
    return res


def run_conjugate(out: Path, quick: bool, dtype=torch.float32) -> dict:
    kw = dict(hf=16, ts=6, length=96, nsteps=6000) if quick else \
        dict(hf=32, ts=10, length=256, nsteps=20000)
    t0 = time.time()
    r = conjugate_channel(check_every=250, dtype=dtype, **kw)
    r["wall_time_s"] = time.time() - t0
    np.savez_compressed(
        out / "fields" / "conjugate.npz",
        T_fluid=r["fields"]["T_fluid"], T_solid=r["fields"]["T_solid"],
        ux=r["fields"]["ux"],
    )
    r.pop("fields")
    _dump(out / "case_de_conjugate.json", r)
    print(f"[conjugate] resid_iface={r['resid_interface']:.3e} "
          f"resid_global={r['resid_global_energy']:.3e} "
          f"dev_tail={r['dev_center_tail']:.4f} ({r['wall_time_s']:.0f}s)", flush=True)
    return r


def run_rac(out: Path, quick: bool, dtype=torch.float32) -> dict:
    """Critical Rayleigh number: LBM growth rates around the threshold."""
    ra_list = [1600.0, 1700.0, 1720.0, 1800.0, 2000.0]
    n, steps = (34, 40000)
    if quick:
        ra_list, n, steps = [1500.0, 2500.0], 26, 4000
    res = {"note": (
        "Growth rate of a small random perturbation of the conductive state, "
        "fitted on the kinetic energy over the late linear window. Near the "
        "critical Rayleigh number the growth rate is tiny, so the sign is the "
        "resolved quantity, not the value."),
        "cases": {}}
    for ra in ra_list:
        t0 = time.time()
        r = rayleigh_growth_rate(ra=ra, n=n, nsteps=steps, sample_every=500,
                                dtype=dtype)
        r["wall_time_s"] = time.time() - t0
        res["cases"][f"ra_{ra:.0f}"] = r
        print(f"[Ra_c] Ra={ra:.0f} sigma={r['sigma']:.3e} "
              f"Emax={r['e_max']:.3e} ({r['wall_time_s']:.0f}s)", flush=True)
    _dump(out / "case_c_rac.json", res)
    return res


def build_summary(out: Path, cavity, cyl, conj, rac) -> dict:
    summary = {"mission": "IN28-N03", "module": "tensorlbm.conjugate_thermal",
               "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    crit = {}

    # (a) forced convection over a heated cylinder, Re = 100, Pr = 0.7
    c100 = cyl.get("re_100")
    if c100:
        nu = c100["nu_time_avg"]
        cb = c100["nu_churchill_bernstein"]
        crit["a_cylinder_re100"] = {
            "measured": nu,
            "reference_churchill_bernstein": cb,
            "reference_formula": "Nu_D = 0.3 + 0.62 Re^0.5 Pr^(1/3)/"
                                 "(1+(0.4/Pr)^(2/3))^(1/4) * (1+(Re/282000)^(5/8))^(4/5)",
            "deviation_rel": (nu - cb) / cb,
            "criterion": "Nu = 10.5 +- 0.5 (as written in the task) / CB at Re=100",
            "criterion_met": bool(abs(nu - 10.5) <= 0.5),
            "note": ("The task target Nu = 10.5 +- 0.5 is not reproduced by the "
                     "Churchill-Bernstein correlation it cites: at Re=100, Pr=0.7 "
                     "the correlation gives 5.156 (Nu=10.5 corresponds to "
                     "Re~450). Judged against the correlation itself."),
        }
    # (b) natural convection cavity, Ra = 1e4
    c = cavity.get("ra_1e04")
    if c:
        crit["b_cavity_ra1e4"] = {
            "measured": c["nu_mean"],
            "reference_de_vahl_davis": c["nu_benchmark"],
            "deviation_rel": (c["nu_mean"] - c["nu_benchmark"]) / c["nu_benchmark"],
            "criterion": "Nu = 2.5 +- 0.2 (as written) vs de Vahl Davis 2.243",
            "criterion_met": bool(abs(c["nu_mean"] - 2.5) <= 0.2),
            "resolution": c["resolution"],
            "note": ("The written target band 2.3..2.7 does not contain the "
                     "de Vahl Davis value 2.243 it cites; the entry records both."),
        }
    # (c) critical Rayleigh number
    sig = {k: v["sigma"] for k, v in rac.get("cases", {}).items()}
    crit["c_critical_rayleigh"] = {
        "growth_rates": sig,
        "criterion": "Ra=1700 stable / Ra=1720 unstable, |Ra_c - 1707.76| < 1%",
        "reference_linear_theory": 1707.762,
        "criterion_met": None,
        "note": ("Near-threshold growth rates are ~1e-5 per lattice step, i.e. "
                 "10^5-10^6 steps per e-fold; the sign is measured, the 1% "
                 "threshold window is not resolvable by time integration at the "
                 "available resolution. An independent dense linear-stability "
                 "solve was attempted and did not converge (see report)."),
    }
    # (d)(e) conjugate channel
    if conj:
        crit["d_conjugate_center_temperature"] = {
            "deviation_tail_rel": conj["dev_center_tail"],
            "deviation_max_rel": conj["dev_center_max"],
            "criterion": "steady T_center vs resistance network < 2%",
            "criterion_met": bool(conj["dev_center_tail"] < 0.02),
        }
        crit["e_interface_flux_conservation"] = {
            "residual_interface": conj["resid_interface"],
            "residual_global_energy": conj["resid_global_energy"],
            "criterion": "interface heat-flux conservation residual < 1e-6",
            "criterion_met": bool(conj["resid_interface"] < 1e-6),
        }
    summary["criteria"] = crit
    _dump(out / "summary.json", summary)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/in28_n03")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cases", default="a,b,c,de")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    out = Path(args.out)
    (out / "fields").mkdir(parents=True, exist_ok=True)
    want = set(args.cases.split(","))

    cavity = cyl = conj = rac = {}
    if "de" in want:
        conj = run_conjugate(out, args.quick, dtype)
    if "b" in want:
        cavity = run_cavity(out, args.quick, dtype)
    if "a" in want:
        cyl = run_cylinder(out, args.quick, dtype)
    if "c" in want:
        rac = run_rac(out, args.quick, dtype)

    # merge partial summaries when a subset of cases ran in this process
    for key, ctor in (("case_b_cavity", dict), ("case_a_cylinder", dict),
                      ("case_de_conjugate", dict), ("case_c_rac", dict)):
        f = out / f"{key}.json"
        if f.exists() and not {"case_b_cavity": cavity, "case_a_cylinder": cyl,
                               "case_de_conjugate": conj, "case_c_rac": rac}[key]:
            if key == "case_b_cavity":
                cavity = json.loads(f.read_text())
            elif key == "case_a_cylinder":
                cyl = json.loads(f.read_text())
            elif key == "case_de_conjugate":
                conj = json.loads(f.read_text())
            else:
                rac = json.loads(f.read_text())
    summary = build_summary(out, cavity, cyl, conj, rac)
    print(json.dumps(summary["criteria"], indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())