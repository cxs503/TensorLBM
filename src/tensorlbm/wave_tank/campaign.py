"""Wave-tank benchmark campaign: kh x eps sweep, criteria evaluation, figures.

Usage (run from the repository root, ``PYTHONPATH=src``)::

    python -m tensorlbm.wave_tank.campaign --list
    python -m tensorlbm.wave_tank.campaign --run kh1.5_eps0.08     # one case
    python -m tensorlbm.wave_tank.campaign --run-all
    python -m tensorlbm.wave_tank.campaign --analyze               # -> results.json
    python -m tensorlbm.wave_tank.campaign --figures               # -> figures/*.png

Every run writes ``artifacts/wave_tank/<name>.npz`` (gauge series, final free
surface, config, timing); ``--analyze`` folds the finished runs into
``artifacts/wave_tank/results.json`` together with the C1-C7 criteria verdicts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass

import numpy as np

from .gauges import fit_harmonics, phase_speed_from_phase, two_probe_split
from .stokes2 import Stokes2Wave, dispersion_solve
from .tank import WaveTankConfig, run_case

ART = os.path.join("artifacts", "wave_tank")

# Still-water depth per kh.  Together with g = 6.5e-3 (lattice) this keeps
# lambda = 84-108 cells, c = 0.26-0.31 (Mach 0.06 at eps = 0.08) and a
# first-harmonic amplitude of 1.1-1.4 cells at eps = 0.08.
DEPTH = {0.7: 12.0, 1.5: 24.0, 3.0: 40.0}


@dataclass(frozen=True)
class CaseSpec:
    name: str
    kh: float
    eps: float
    absorb: bool = True
    refine: int = 1  # 1 = production grid, 2 = dx halved (criterion C5)

    def config(self) -> WaveTankConfig:
        h = DEPTH[self.kh] * self.refine
        nu = 0.008 * self.refine  # keeps the wave Reynolds number c*lambda/nu
        g = 6.5e-3 / self.refine  # keeps the celerity (same physical wave)
        kw = dict(kh=self.kh, eps=self.eps, h=h, g=g, nu=nu, absorb=self.absorb)
        if self.refine > 1:
            # the refined run is only used for the waveform comparison (C5), so
            # it gets a compact tank and a shorter window to stay affordable
            kw.update(lam_mid=0.6, probe_frac=(0.20, 0.37),
                      warm_periods=3.0, meas_periods=2.0)
        return WaveTankConfig(**kw)

    @property
    def base(self) -> str:
        """Name of the production-grid case this one is compared against."""
        if self.refine > 1:
            return self.name.replace("_dx%d" % self.refine, "")
        return self.name


CASES: list[CaseSpec] = [
    CaseSpec("kh0.7_eps0.02", 0.7, 0.02),
    CaseSpec("kh0.7_eps0.08", 0.7, 0.08),
    CaseSpec("kh1.5_eps0.02", 1.5, 0.02),
    CaseSpec("kh1.5_eps0.08", 1.5, 0.08),
    CaseSpec("kh3.0_eps0.02", 3.0, 0.02),
    CaseSpec("kh3.0_eps0.08", 3.0, 0.08),
    CaseSpec("kh1.5_eps0.08_noreflect", 1.5, 0.08, absorb=False),
    CaseSpec("kh1.5_eps0.08_dx2", 1.5, 0.08, refine=2),
]

BY_NAME = {c.name: c for c in CASES}


# ---------------------------------------------------------------------------
# one-run driver
# ---------------------------------------------------------------------------


def run_one(spec: CaseSpec, out_dir: str = ART, *, quiet: bool = True) -> str:
    os.makedirs(out_dir, exist_ok=True)
    cfg = spec.config()
    res = run_case(cfg, quiet=quiet)
    s, m = res.series, res.metrics
    path = os.path.join(out_dir, f"{spec.name}.npz")
    np.savez_compressed(
        path,
        name=spec.name,
        config=np.array(json.dumps(cfg.to_dict())),
        t=s["t"],
        probe_eta=s["probe_eta"],
        mass=s["mass"],
        probe_cols=s["probe_cols"],
        warm_steps=np.array(cfg.schedule()["warm_steps"], dtype=np.int64),
        meas_steps=np.array(cfg.schedule()["meas_steps"], dtype=np.int64),
        mass0=np.array(m["mass0"]),
        eta_final=res.snapshot["eta"],
        phi=res.snapshot["phi"],
        wall_time_s=np.array(m["wall_time_s"]),
        ms_per_step=np.array(m["ms_per_step"]),
        steps=np.array(m["steps"], dtype=np.int64),
        cells=np.array(m["ny"] * m["nx"] * m["nz"], dtype=np.int64),
    )
    print(
        "[run] %-28s %5d steps  %7.1f s  %6.1f ms/step  %dx%dx%d  cells=%d"
        % (
            spec.name,
            m["steps"],
            m["wall_time_s"],
            m["ms_per_step"],
            m["ny"],
            m["nx"],
            m["nz"],
            m["ny"] * m["nx"] * m["nz"],
        ),
        flush=True,
    )
    return path


# ---------------------------------------------------------------------------
# analysis of one saved run
# ---------------------------------------------------------------------------


def wave_of_config(cfg: dict) -> Stokes2Wave:
    """Analytic wave of a saved run, using the *effective* surface gravity."""
    g_wave = cfg["g"] * (1.0 - cfg["rho_g"] / cfg["rho_l"])
    return Stokes2Wave.from_kh(cfg["kh"], cfg["eps"], cfg["h"], g_wave)


def analyse_case(npz_path: str) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    cfg = json.loads(str(d["config"]))
    w = wave_of_config(cfg)
    g_wave = w.g
    t = d["t"].astype(float)
    eta = d["probe_eta"].astype(float)
    mass = d["mass"].astype(float)
    cols = d["probe_cols"].astype(float)
    warm = float(d["warm_steps"])
    meas = float(d["meas_steps"])
    mass0 = float(d["mass0"])
    sel = (t >= warm) & (t < warm + meas)
    tw = t[sel] - warm
    T = w.period
    k2 = dispersion_solve(2.0 * w.omega, g_wave, cfg["h"])

    fits = [fit_harmonics(tw, eta[sel, j], w.omega, n_harmonics=2) for j in range(eta.shape[1])]
    x1, x2 = float(cols[0]), float(cols[1])
    s1 = two_probe_split(fits[0].c1, fits[1].c1, x1, x2, w.k)
    s2 = two_probe_split(fits[0].c2, fits[1].c2, x1, x2, k2)

    # phase speed from the *incident* wave phases (immune to the reflection):
    # eta_I(x) = A e^{ikx} => arg(eta) grows linearly with x
    c_meas, k_meas = phase_speed_from_phase(
        fits[0].c1, fits[1].c1, x2 - x1, w.omega, k_expected=w.k
    )

    # ---- mass drift per period (linear fit over the measurement window) ----
    m = mass[sel]
    if m.size > 2:
        slope = float(np.polyfit(t[sel], m, 1)[0])
    else:
        slope = 0.0
    drift_per_period = slope * T / mass0

    a1_err = abs(s1.a_incident - w.a) / w.a
    a2_err = abs(s2.a_incident - w.eta2) / w.eta2 if w.eta2 > 0 else float("inf")
    c_err = abs(c_meas - w.celerity) / w.celerity

    return {
        "name": str(d["name"]),
        "config": cfg,
        "theory": {
            "kh": w.kh,
            "eps": w.eps,
            "k": w.k,
            "k2": k2,
            "omega": w.omega,
            "period": T,
            "wavelength": w.wavelength,
            "celerity": w.celerity,
            "a1": w.a,
            "a2": w.eta2,
            "sigma": w.sigma,
            "g_wave": g_wave,
            "amplitude_cells": w.a,
        },
        "gauge": {
            "x1": x1,
            "x2": x2,
            "dx": x2 - x1,
            "dx_over_lambda": (x2 - x1) / w.wavelength,
            "n_samples": int(tw.size),
        },
        "harmonics": {"probe1": fits[0].to_dict(), "probe2": fits[1].to_dict()},
        "reflection_omega": s1.to_dict(),
        "reflection_2omega": s2.to_dict(),
        "metrics": {
            "c_measured": c_meas,
            "k_measured": k_meas,
            "c_error_rel": c_err,
            "a1_incident": s1.a_incident,
            "a1_error_rel": a1_err,
            "a2_incident": s2.a_incident,
            "a2_error_rel": a2_err,
            "kr": s1.kr,
            "kr_2omega": s2.kr,
            "mass_drift_per_period": drift_per_period,
            "mass_drift_total": float((m[-1] - m[0]) / mass0) if m.size else 0.0,
            "wall_time_s": float(d["wall_time_s"]),
            "ms_per_step": float(d["ms_per_step"]),
            "steps": int(d["steps"]),
            "cells": int(d["cells"]),
        },
    }


def _free_decay(out_dir: str = ART) -> list[dict]:
    """Collect the free-decay dispersion measurements saved as npz."""
    import glob

    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, "freedecay_kh*.npz"))):
        d = np.load(p, allow_pickle=True)
        rows.append(json.loads(str(d["metrics"])))
    rows.sort(key=lambda r: r["kh"])
    return rows


# ---------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------

C1_TOL = 0.01  # dispersion: phase-speed error
C2A_TOL = 0.03  # first harmonic amplitude
C2B_TOL = 0.10  # second harmonic amplitude
C3_TOL = 0.05  # absorption reflection coefficient
C3_NEG = 0.30  # negative control must exceed this
C4_TOL = 0.005  # mass drift per period
C5_TOL = 0.03  # waveform change under dx/2
C7_TOL = 1200.0  # seconds per case (20 min)


def evaluate(results: dict[str, dict]) -> dict:
    """Apply criteria C1-C7 to the analysed runs."""
    base = {n: r for n, r in results.items() if "_noreflect" not in n and "_dx2" not in n}
    crit: dict[str, dict] = {}

    # C1: dispersion relation.  Primary evidence = free-decay measurement
    # (tank oscillates at its natural frequency for a known k; no wavemaker in
    # the loop).  Secondary (informational) = two-point phase speed next to the
    # relaxation zone, which is contaminated by the start-up transient.
    fd = _free_decay()
    crit["C1_dispersion"] = {
        "tol": C1_TOL,
        "basis": "free decay of a standing wave (natural frequency for a known k)",
        "values": {("kh%.1f" % d["kh"]): d["celerity_error_rel"] for d in fd},
        "periods": {("kh%.1f" % d["kh"]): [d["period_theory"], d["period_measured"]] for d in fd},
        "g_eff_over_g_measured": {("kh%.1f" % d["kh"]): d["g_eff_over_g"] for d in fd},
        "pass": bool(fd) and len(fd) >= 3 and all(d["celerity_error_rel"] < C1_TOL for d in fd),
        "n_kh": len(fd),
    }
    c1b = {n: r["metrics"]["c_error_rel"] for n, r in sorted(base.items())
           if abs(r["config"]["eps"] - 0.08) < 1e-9}
    crit["C1b_wavemaker_phase_speed"] = {
        "tol": C1_TOL,
        "values": c1b,
        "pass": all(v < C1_TOL for v in c1b.values()) if c1b else False,
        "note": "two-point phase speed 0.2-0.4 lambda from the wavemaker; "
                "start-up transient / reflected wave bias this estimate "
                "(see README section 7), reported for transparency",
    }

    # C2: harmonic amplitudes (both eps, all kh where eps=0.08 exists)
    a1 = {n: r["metrics"]["a1_error_rel"] for n, r in sorted(base.items())}
    a2 = {n: r["metrics"]["a2_error_rel"] for n, r in sorted(base.items())}
    both = [n for n in base]
    crit["C2_first_harmonic"] = {
        "tol": C2A_TOL,
        "values": a1,
        "pass": all(v < C2A_TOL for v in a1.values()) if a1 else False,
    }
    crit["C2_second_harmonic"] = {
        "tol": C2B_TOL,
        "values": a2,
        "pass": all(v < C2B_TOL for v in a2.values()) if a2 else False,
        "note": "eps=0.02 cases have sub-cell second-harmonic amplitude; see report",
        "n_cases": len(both),
    }

    # C3: absorption + negative control
    kr = {n: r["metrics"]["kr"] for n, r in sorted(base.items())}
    neg = [r["metrics"]["kr"] for n, r in results.items() if "_noreflect" in n]
    crit["C3_reflection"] = {
        "tol": C3_TOL,
        "negative_control_min": C3_NEG,
        "values": kr,
        "pass": all(v < C3_TOL for v in kr.values()) if kr else False,
        "negative_control": neg[0] if neg else None,
        "negative_control_pass": bool(neg) and neg[0] > C3_NEG,
    }

    # C4: mass conservation
    md = {n: r["metrics"]["mass_drift_per_period"] for n, r in sorted(base.items())}
    crit["C4_mass"] = {
        "tol": C4_TOL,
        "values": md,
        "pass": all(abs(v) < C4_TOL for v in md.values()) if md else False,
    }

    # C5: grid convergence (dx halved)
    c5 = None
    if "kh1.5_eps0.08_dx2" in results:
        fine = results["kh1.5_eps0.08_dx2"]["snapshot_eta"] if "snapshot_eta" in results["kh1.5_eps0.08_dx2"] else None
        c5 = _grid_convergence(results)
    crit["C5_grid_convergence"] = c5 if c5 is not None else {"pass": False, "note": "refined run missing"}

    # C7: cost per case
    wt = {n: r["metrics"]["wall_time_s"] for n, r in sorted(results.items())}
    crit["C7_cost"] = {
        "tol_s": C7_TOL,
        "values": wt,
        "pass": all(v < C7_TOL for v in wt.values()) if wt else False,
    }
    crit["all_pass"] = all(c.get("pass", False) for k, c in crit.items() if k.startswith("C"))
    return crit


def _grid_convergence(results: dict[str, dict]) -> dict:
    """C5: free-surface waveform change when dx is halved.

    The two runs end at different physical times (the refined grid needs twice
    as many steps per period), so a pointwise snapshot comparison would be
    dominated by a pure phase offset.  The waveforms are therefore compared
    after (a) removing the mean level, (b) removing the linear trend of the
    still-water level, and (c) aligning the phase by the x-shift that maximises
    the cross-correlation.  The reported ``rel_change`` is

        RMS(eta_coarse - eta_fine, phase aligned) / a1_theory.

    Both surfaces are interpolated onto x / lambda so the two grids are
    compared in the same non-dimensional coordinate.
    """
    import os as _os

    def load(name: str):
        p = _os.path.join(ART, f"{name}.npz")
        return np.load(p, allow_pickle=True) if _os.path.exists(p) else None

    coarse = load("kh1.5_eps0.08")
    fine = load("kh1.5_eps0.08_dx2")
    if coarse is None or fine is None:
        return {"pass": False, "note": "npz missing (refined run not completed)"}

    def prep(d):
        cfg = json.loads(str(d["config"]))
        w = wave_of_config(cfg)
        eta = np.asarray(d["eta_final"], dtype=float)
        x = np.arange(eta.size, dtype=float)
        n_in = int(round(cfg["lam_in"] * w.wavelength))
        n_out = int(round(cfg["lam_out"] * w.wavelength))
        sl = slice(n_in, max(n_in + 8, x.size - n_out - 1))
        return w, x[sl] / w.wavelength, eta[sl]

    wc, xc, ec = prep(coarse)
    wf, xf, ef = prep(fine)
    lo = max(xc.min(), xf.min())
    hi = min(xc.max(), xf.max())
    xs = np.linspace(lo, hi, 600)
    ec_i = np.interp(xs, xc, ec - np.polyval(np.polyfit(xc, ec, 1), xc))
    ef_i = np.interp(xs, xf, ef - np.polyval(np.polyfit(xf, ef, 1), xf))
    ec_i = ec_i - ec_i.mean()
    ef_i = ef_i - ef_i.mean()
    # phase alignment: shift the fine profile over +- 1/2 wavelength
    dx = xs[1] - xs[0]
    max_shift = int(round(0.5 / dx))
    best = None
    for sh in range(-max_shift, max_shift + 1):
        cand = np.roll(ef_i, sh)
        if sh > 0:
            cand[:sh] = np.nan
        elif sh < 0:
            cand[sh:] = np.nan
        m = ~np.isnan(cand)
        if m.sum() < 0.5 * xs.size:
            continue
        rms = float(np.sqrt(np.mean((ec_i[m] - cand[m]) ** 2)))
        if best is None or rms < best[0]:
            best = (rms, sh)
    rms, sh = best if best else (float("nan"), 0)
    amp = wc.a
    return {
        "tol": C5_TOL,
        "rms_diff_cells": rms,
        "amplitude_cells": amp,
        "rel_change": rms / amp,
        "phase_shift_cells_fine": sh * dx * wf.wavelength if wf.wavelength else 0.0,
        "pass": bool(rms / amp < C5_TOL),
        "compared_stretch_over_lambda": [float(xs.min()), float(xs.max())],
        "grids": {
            "coarse": {"h": wc.h, "lambda": wc.wavelength, "a": wc.a},
            "fine": {"h": wf.h, "lambda": wf.wavelength, "a": wf.a},
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run")
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--sdaa", action="store_true")
    ap.add_argument("--out", default=ART)
    args = ap.parse_args(argv)

    if args.list:
        for c in CASES:
            cfg = c.config()
            sch = cfg.schedule()
            geo = cfg.geometry()
            print(
                "%-28s kh=%.1f eps=%.2f h=%4.1f lambda=%6.1f ny=%3d nx=%4d "
                "steps=%5d T=%6.1f"
                % (
                    c.name,
                    c.kh,
                    c.eps,
                    cfg.h,
                    sch["wavelength"],
                    geo["ny"],
                    geo["nx"],
                    sch["warm_steps"] + sch["meas_steps"],
                    sch["period"],
                )
            )
        return 0

    if args.run:
        spec = BY_NAME[args.run]
        run_one(spec, out_dir=args.out)
        return 0

    if args.run_all:
        for c in CASES:
            run_one(c, out_dir=args.out)
        return 0

    if args.analyze:
        results = {}
        for c in CASES:
            p = os.path.join(args.out, f"{c.name}.npz")
            if os.path.exists(p):
                results[c.name] = analyse_case(p)
        crit = evaluate(results)
        payload = {"criteria": crit, "cases": results}
        path = os.path.join(args.out, "results.json")
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        print(json.dumps(crit, indent=2, sort_keys=True))
        print(f"[analyze] wrote {path}")
        return 0

    if args.figures:
        from .figures import make_figures

        make_figures(args.out)
        return 0

    if args.sdaa:
        # machine-readable acceptance record, same shape as
        # artifacts/sdaa_<benchmark>.json used by the other verified cases
        with open(os.path.join(args.out, "results.json")) as fh:
            payload = json.load(fh)
        cases, crit = payload["cases"], payload["criteria"]
        rec = {
            "benchmark": "wave_tank",
            "case_dir": "benchmarks/verified/wave_tank",
            "engine": {
                "solver": "tensorlbm.cac_lbm.CACSim (CAC phase-field, D3Q19, quasi-2D nz=3)",
                "torch_version": __import__("torch").__version__,
                "device": "cpu (float32)",
                "python": __import__("sys").version.split()[0],
            },
            "config": {
                "sweep": "kh in {0.7,1.5,3.0} x eps=ka in {0.02,0.08}",
                "gravity_lattice": 6.5e-3,
                "gravity_effective": "g*(1-rho_g/rho_l) = 0.9 g (free-surface restoring)",
                "rho_l_rho_g": [1.0, 0.1],
                "nu": 0.008,
                "interface_width": 2.0,
                "sigma": 1e-4,
                "wavemaker": "relaxation zone, 0.7 lambda, cos^2 weight, Stokes-2nd target",
                "absorber": "relaxation zone, 0.7 lambda (disabled in the negative control)",
                "gauges": "two probes at 0.20/0.37 lambda from the inlet-zone exit (dx=0.17 lambda)",
                "measurement": "integer number of periods, harmonics at omega and 2 omega, "
                               "Goda & Suzuki two-point split",
            },
            "criteria": {k: v for k, v in crit.items()},
            "verdict": "PASS" if crit.get("all_pass") else "FAIL",
            "cases": {n: {"metrics": c["metrics"], "theory": c["theory"], "gauge": c["gauge"]}
                      for n, c in cases.items()},
            "notes": [
                "honest reporting: criteria that miss their threshold are recorded as FAIL",
                "the refined-grid case (dx/2) is used only for C5 and may be absent",
                "viscous decay of 1-2 %/period is not corrected for in the amplitude errors",
            ],
        }
        path = os.path.join(args.out, "sdaa_wave_tank.json")
        with open(path, "w") as fh:
            json.dump(rec, fh, indent=2, sort_keys=True)
        print(f"[sdaa] wrote {path}")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())