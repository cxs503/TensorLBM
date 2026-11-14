"""CASE-RT-INSTABILITY-ATWOOD — single-mode Rayleigh-Taylor at A = 1/2, spike growth rate.

Case-pool executor.  It does **not** contain a solver: everything is driven
through the library capability ``tensorlbm.rayleigh_taylor`` (Boussinesq two-layer
D2Q9 flow + interface transport), exactly as the sibling case
``benchmarks/case_pool/rayleigh_taylor`` does.  What is new here:

* the Atwood number is fixed at the anchor value ``A = 1/2`` (the sibling suite
  used ``A = 1/3`` as its reference point and ``A = 1/2`` only as a sensitivity
  arm, and its legacy phase transfer diverges at A = 1/2);
* the *growth-rate estimator* is the object of study.  The sibling suite fits
  ``ln a`` inside ``a/lam in [0.01, 0.05]``, a window that starts at the initial
  amplitude ``a0 = 0.01 lam`` -- i.e. inside the startup transient of a diffuse
  interface.  Here the same run is analysed with four pre-declared windows plus
  the instantaneous rate ``d ln a / dt``, so a window artefact is distinguishable
  from a solver error, and the viscosity implied by the measured rate
  (``nu_eff``, the closed-form inverse of the equal-viscosity dispersion
  relation) is reported;
* a grid ladder at constant ``A g / lam`` (so the dimensionless problem is
  identical: same ``sigma sqrt(lam/(A g))``, same interface resolution per
  wavelength) and two sensitivity axes (viscosity, gravity).

Pre-declared analysis (fixed before any gated run was inspected)
---------------------------------------------------------------
WINDOWS (on the ``phi = 0`` interface amplitude ``a/lam``):
    legacy   [0.010, 0.050]   -- the sibling suite's window, kept for continuity
    primary  [0.020, 0.080]   -- headline estimator of this case
    late     [0.040, 0.120]
    late2    [0.060, 0.160]
The primary window is shifted up by one initial amplitude on purpose: the linear
regime of A = 1/2 spans roughly ``a/lam < 0.1`` (nonlinear saturation of the
constant-velocity fronts sets in beyond that), and the first ~1/2 e-folding after
``t = 0`` is contaminated by (i) the tanh interface relaxing onto the
Allen-Cahn equilibrium profile and (ii) the seeded eigenmode exchanging energy
with the acoustic field.  The choice was made from that argument, not from the
result.

GATE: ``|sigma_meas/sigma_visc - 1| <= 15 %`` (case-pool tolerance of this case).

Front velocities are reported as a secondary, **ungated** diagnostic: the
Boussinesq equations solved here are invariant under ``(phi, y) -> (-phi, -y)``,
so the solver cannot reproduce the bubble/spike asymmetry of a two-fluid
potential-flow model, and the 2-D value of the geometric constant ``C`` is
itself contested in this repo (C = 1 vs C = 3, see ``ref/ref_lock.py``).  Both
conventions are printed; no gate is attached.

References: see ``ref/ref_lock.py`` (provenance + ``ref_lock.json``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

torch.set_num_threads(int(os.environ.get("RT_THREADS", "4")))

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parents[2]
SRC_ROOT = REPO_ROOT / "src"
for _p in (str(SRC_ROOT), str(REPO_ROOT)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
sys.path.insert(0, str(BENCH_DIR / "ref"))

import ref_lock  # noqa: E402  (reference values, single source of truth)

from tensorlbm.d2q9 import macroscopic  # noqa: E402
from tensorlbm.rayleigh_taylor import (  # noqa: E402
    RTConfig,
    init_rt_state,
    mode_amplitude,
    mode_amplitude_zc,
    interface_zero_crossing,
    rt_step,
)

# ---------------------------------------------------------------------------
# pre-declared analysis constants
# ---------------------------------------------------------------------------
WINDOWS = {
    "legacy": (0.010, 0.050),
    "primary": (0.020, 0.080),
    "late": (0.040, 0.120),
    "late2": (0.060, 0.160),
}
PRIMARY_WINDOW = "primary"
GATE_PCT = 15.0
#: saturated-velocity window in normalised time tau = t sqrt(A g / lam)
TAU_LO, TAU_HI = 4.0, 6.0
#: hard stops (guard rails, not analysis choices)
MAX_MA = 0.85           # stop if the lattice Mach number exceeds this
MAX_A_FRAC = 0.60       # stop if the interface amplitude exceeds this fraction of lam
#: the run must reach tau = 6.5 (terminal-velocity window), so the amplitude guard
#: only trips if the interface is about to leave the box (layers = 8 -> 4 lam per side).
AC = dict(phase_scheme="allen_cahn", ac_sigma=3.0e-4, ac_width=2.0, ac_mobility=20.0)


# ---------------------------------------------------------------------------
# case definition
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CaseSpec:
    name: str
    lam: int
    g: float
    tau: float = 0.515
    tau_phi: float = 0.501
    atwood: float = ref_lock.ATWOOD
    layers: float = 8.0
    tau_end: float = 6.5
    phase_scheme: str = AC["phase_scheme"]
    ac_sigma: float = AC["ac_sigma"]
    ac_width: float = AC["ac_width"]
    ac_mobility: float = AC["ac_mobility"]
    note: str = ""

    @property
    def nu(self) -> float:
        return (self.tau - 0.5) / 3.0

    def n_steps(self) -> int:
        return int(round(self.tau_end * math.sqrt(self.lam / (self.atwood * self.g))))

    @property
    def sample_every(self) -> int:
        return max(1, self.n_steps() // 800)

    def cfg(self) -> RTConfig:
        return RTConfig(
            lam=self.lam,
            layers=self.layers,
            g=self.g,
            atwood=self.atwood,
            tau=self.tau,
            tau_phi=self.tau_phi,
            n_steps=self.n_steps(),
            sample_every=self.sample_every,
            a_lo_frac=WINDOWS["legacy"][0],
            a_hi_frac=WINDOWS["legacy"][1],
            stop_a_frac=10.0,        # never stop early: this case needs the late time
            seed_eigenmode=True,
            phase_scheme=self.phase_scheme,
            ac_sigma=self.ac_sigma,
            ac_width=self.ac_width,
            ac_mobility=self.ac_mobility,
        )


SPECS = (
    # reference point: A = 1/2, g and nu in the "well resolved, low Ma" corner
    CaseSpec("A050_l064_g2e-4", lam=64, g=2.0e-4, note="reference point"),
    # grid ladder at constant A g / lam -> identical dimensionless problem
    CaseSpec("A050_l096_g3e-4", lam=96, g=3.0e-4, note="grid ladder 1.5x"),
    CaseSpec("A050_l128_g4e-4", lam=128, g=4.0e-4, note="grid ladder 2x"),
    # sensitivity: viscosity (tau = 0.505 -> nu = 1/600, tau = 0.530 -> nu = 0.01)
    CaseSpec("A050_l064_g2e-4_tau505", lam=64, g=2.0e-4, tau=0.505, note="low viscosity"),
    CaseSpec("A050_l064_g2e-4_tau530", lam=64, g=2.0e-4, tau=0.530, note="high viscosity"),
    # sensitivity: gravity (linear driver) at fixed resolution
    CaseSpec("A050_l064_g1e-4", lam=64, g=1.0e-4, note="half gravity"),
    # control arm: legacy advected-scalar phase transport (unbounded interface)
    CaseSpec("A050_l064_g2e-4_legacy", lam=64, g=2.0e-4, phase_scheme="bgk",
             note="legacy phase scheme control"),
)


#: interface-model sensitivity arm (discriminates "finite interface thickness /
#: Allen-Cahn mobility" from "solver error" as the cause of the growth-rate
#: deficit; run through ``--specs extra``).
SPECS_EXTRA = (
    CaseSpec("A050_l064_g2e-4_w15", lam=64, g=2.0e-4, ac_width=1.5,
             note="thinner interface (xi = 1.5 cells)"),
    CaseSpec("A050_l064_g2e-4_m10", lam=64, g=2.0e-4, ac_mobility=10.0,
             note="half Allen-Cahn mobility"),
    CaseSpec("A050_l096_g3e-4_w15", lam=96, g=3.0e-4, ac_width=1.5,
             note="thinner interface on the 1.5x grid"),
)


# ---------------------------------------------------------------------------
# analysis helpers
# ---------------------------------------------------------------------------
def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least-squares ``y = a + b x``; returns ``(b, a, r2)``."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0.0:
        return float("nan"), float("nan"), float("nan")
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return b, a, r2


def window_slope(times: list[float], amps: list[float], lo: float, hi: float) -> dict:
    """Slope of ``ln a`` over the samples whose amplitude lies in ``[lo, hi]``."""
    pts = [(t, a) for t, a in zip(times, amps) if lo <= a <= hi and a > 0.0 and math.isfinite(a)]
    if len(pts) < 4:
        return {"sigma": float("nan"), "r2": float("nan"), "n": len(pts),
                "a_range": [float("nan")] * 2, "t_range": [float("nan")] * 2}
    sig, _b, r2 = linear_fit([p[0] for p in pts], [math.log(p[1]) for p in pts])
    return {"sigma": sig, "r2": r2, "n": len(pts),
            "a_range": [min(p[1] for p in pts), max(p[1] for p in pts)],
            "t_range": [min(p[0] for p in pts), max(p[0] for p in pts)]}


def instantaneous_rate(times: list[float], amps: list[float], half: int = 12) -> list[list[float]]:
    """``d ln a / dt`` on a centred window of ``2*half+1`` samples."""
    out = []
    for i in range(half, len(times) - half):
        window_t = times[i - half: i + half + 1]
        window_a = amps[i - half: i + half + 1]
        if any((not math.isfinite(a)) or a <= 0.0 for a in window_a):
            continue
        s, _b, _r2 = linear_fit(window_t, [math.log(a) for a in window_a])
        out.append([times[i], amps[i], s])
    return out


def fourier_mode(field: torch.Tensor, lam: int) -> float:
    """Amplitude of the first Fourier mode of a 2-D field (row-mean then FFT-free cos fit)."""
    prof = field.to(torch.float64).mean(dim=0)
    prof = prof - prof.mean()
    nx = prof.shape[0]
    x = torch.arange(nx, dtype=torch.float64, device=prof.device)
    basis = torch.cos(2.0 * math.pi * x / float(lam))
    return float((2.0 / nx) * torch.abs((prof * basis).sum()).item())


def velocity_in_window(taus: list[float], fronts: list[float], lo: float, hi: float) -> float:
    """Drift velocity of a front over the tau window (linear slope)."""
    pts = [(t, f) for t, f in zip(taus, fronts) if lo <= t <= hi and math.isfinite(f)]
    if len(pts) < 3:
        return float("nan")
    s, _b, _r2 = linear_fit([p[0] for p in pts], [p[1] for p in pts])
    return s


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------
def run_case(spec: CaseSpec, out_root: Path, quick: bool = False) -> dict:
    cfg = spec.cfg()
    cfg.validate()
    torch.manual_seed(0)

    n_steps = min(cfg.n_steps, 400) if quick else cfg.n_steps
    f, g, phi = init_rt_state(cfg)
    tau_unit = math.sqrt(spec.atwood * spec.g / spec.lam)
    mass0 = float(g[:, 1:-1, :].sum().item()) or 1.0

    rec = {k: [] for k in ("step", "t", "tau", "a_col", "a_zc", "h_b", "h_s",
                           "max_u", "phi_min", "phi_max", "mass_drift_pct", "a_uy")}
    t0 = time.time()
    blown_up = False
    for step in range(1, n_steps + 1):
        f, g = rt_step(f, g, cfg, phi=phi)
        phi = g.sum(dim=0)
        if step % cfg.sample_every == 0 or step == 1:
            a_col = mode_amplitude(phi, spec.lam) / spec.lam
            a_zc = mode_amplitude_zc(phi, spec.lam) / spec.lam
            zc = interface_zero_crossing(phi)
            mean = float(zc.mean())
            _rho, ux, uy = macroscopic(f)
            speed = float(torch.sqrt(ux * ux + uy * uy).max())
            # first Fourier mode of the vertical velocity on the interface row band:
            # a flow-side growth-rate estimator that does not involve the interface
            # transport model at all (decides "solver wrong" vs "interface model damps")
            band = uy[phi.shape[0] // 2 - int(round(0.5 * spec.lam)): phi.shape[0] // 2
                      + int(round(0.5 * spec.lam)) + 1, :]
            a_uy = fourier_mode(band, spec.lam)
            mass = float(g[:, 1:-1, :].sum().item())
            rec["step"].append(float(step))
            rec["t"].append(float(step))
            rec["tau"].append(step * tau_unit)
            rec["a_col"].append(a_col)
            rec["a_zc"].append(a_zc)
            rec["h_b"].append(float(zc.max()) - mean)
            rec["h_s"].append(mean - float(zc.min()))
            rec["max_u"].append(speed)
            rec["phi_min"].append(float(phi.min()))
            rec["phi_max"].append(float(phi.max()))
            rec["mass_drift_pct"].append(100.0 * (mass - mass0) / abs(mass0))
            rec["a_uy"].append(a_uy)
            if not math.isfinite(speed) or not math.isfinite(a_zc):
                blown_up = True
                break
            if speed / math.sqrt(1.0 / 3.0) > MAX_MA or a_zc > MAX_A_FRAC:
                break

    wall = time.time() - t0
    finite = bool(rec["a_zc"]) and all(math.isfinite(x) for x in rec["a_zc"][-3:])

    # ---- linear growth rate: four pre-declared windows, two estimators
    sigmas = {}
    for est, key in (("a_zc", "a_zc"), ("a_col", "a_col")):
        amps, times = rec[key], rec["t"]
        for wname, (lo, hi) in WINDOWS.items():
            sigmas[f"{est}_{wname}"] = window_slope(times, amps, lo, hi)
    sigma_primary = sigmas[f"a_zc_{PRIMARY_WINDOW}"]["sigma"]
    sigma_legacy = sigmas["a_zc_legacy"]["sigma"]

    # flow-side estimator: ln|uy mode| slope over the time span of the primary window
    t_lo, t_hi = sigmas[f"a_zc_{PRIMARY_WINDOW}"]["t_range"]
    pts = [(t, a) for t, a in zip(rec["t"], rec["a_uy"])
           if math.isfinite(a) and a > 0.0 and t_lo <= t <= t_hi]
    if len(pts) >= 4:
        sig_uy, _b, r2_uy = linear_fit([p[0] for p in pts], [math.log(p[1]) for p in pts])
    else:
        sig_uy, r2_uy = float("nan"), float("nan")

    sig_ref = ref_lock.sigma_viscous(spec.atwood, spec.g, spec.lam, spec.nu)
    sig_inv = ref_lock.sigma_inviscid(spec.atwood, spec.g, spec.lam)
    nu_eff = ref_lock.nu_eff_from_sigma(spec.atwood, spec.g, spec.lam, sigma_primary)

    # ---- secondary (ungated) nonlinear front velocities
    u_s = velocity_in_window(rec["tau"], rec["h_s"], TAU_LO, TAU_HI)
    u_b = velocity_in_window(rec["tau"], rec["h_b"], TAU_LO, TAU_HI)
    u_s_c1 = ref_lock.goncharov(spec.atwood, spec.g, spec.lam, "s", ref_lock.C_2D)
    u_b_c1 = ref_lock.goncharov(spec.atwood, spec.g, spec.lam, "b", ref_lock.C_2D)
    u_s_c3 = ref_lock.goncharov(spec.atwood, spec.g, spec.lam, "s", ref_lock.C_3D)

    def rel(x: float, ref: float) -> float:
        return 100.0 * (x / ref - 1.0) if (math.isfinite(x) and ref) else float("nan")

    case = {
        "name": spec.name,
        "note": spec.note,
        "atwood": spec.atwood,
        "lam": spec.lam,
        "g": spec.g,
        "tau": spec.tau,
        "tau_phi": spec.tau_phi,
        "nu": spec.nu,
        "k": 2.0 * math.pi / spec.lam,
        "phase_scheme": spec.phase_scheme,
        "layers": spec.layers,
        "ac_sigma": spec.ac_sigma,
        "ac_width": spec.ac_width,
        "ac_mobility": spec.ac_mobility,
        "interface_thickness_cells": spec.ac_width,
        "n_steps_planned": cfg.n_steps,
        "n_steps_run": int(rec["step"][-1]) if rec["step"] else 0,
        "tau_end": rec["tau"][-1] if rec["tau"] else float("nan"),
        "sample_every": cfg.sample_every,
        "blown_up": blown_up,
        "finite": finite,
        "wall_seconds": wall,
        # linear
        "sigma_sim": sigma_primary,
        "sigma_sim_legacy_window": sigma_legacy,
        "sigma_inviscid": sig_inv,
        "sigma_viscous": sig_ref,
        "err_vs_viscous_pct": rel(sigma_primary, sig_ref),
        "err_vs_inviscid_pct": rel(sigma_primary, sig_inv),
        "err_legacy_window_vs_viscous_pct": rel(sigma_legacy, sig_ref),
        "sigma_flow_uy": sig_uy,
        "sigma_flow_uy_r2": r2_uy,
        "err_flow_uy_vs_viscous_pct": 100.0 * (sig_uy / sig_ref - 1.0)
        if math.isfinite(sig_uy) else float("nan"),
        "nu_eff": nu_eff,
        "nu_eff_over_nu": nu_eff / spec.nu if spec.nu and math.isfinite(nu_eff) else float("nan"),
        "windows": sigmas,
        # nonlinear (ungated)
        "u_spike_sim": u_s,
        "u_bubble_sim": u_b,
        "u_spike_ref_c1": u_s_c1,
        "u_bubble_ref_c1": u_b_c1,
        "u_spike_ref_c3": u_s_c3,
        "err_u_spike_c1_pct": rel(u_s, u_s_c1),
        "err_u_spike_c3_pct": rel(u_s, u_s_c3),
        "fr_spike_sim": ref_lock.froude(spec.atwood, spec.g, spec.lam, u_s)
        if math.isfinite(u_s) else float("nan"),
        "fr_spike_ref_c1": ref_lock.froude(spec.atwood, spec.g, spec.lam, u_s_c1),
        # invariants / health
        "ma_peak": max(rec["max_u"]) / math.sqrt(1.0 / 3.0) if rec["max_u"] else float("nan"),
        "phi_min": min(rec["phi_min"]) if rec["phi_min"] else float("nan"),
        "phi_max": max(rec["phi_max"]) if rec["phi_max"] else float("nan"),
        "mass_drift_pct_max": max(abs(x) for x in rec["mass_drift_pct"])
        if rec["mass_drift_pct"] else float("nan"),
        "a_zc_peak": max(rec["a_zc"]) if rec["a_zc"] else float("nan"),
    }
    case["gate_sigma_15pct"] = bool(math.isfinite(case["err_vs_viscous_pct"])
                                    and abs(case["err_vs_viscous_pct"]) <= GATE_PCT)

    d = out_root / spec.name
    d.mkdir(parents=True, exist_ok=True)
    with (d / "history.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["step", "tau", "a_col_over_lam", "a_zc_over_lam",
                     "h_bubble_over_lam", "h_spike_over_lam", "max_speed",
                     "phi_min", "phi_max", "mass_drift_pct", "a_uy"])
        for i, s in enumerate(rec["step"]):
            wr.writerow([int(s), f"{rec['tau'][i]:.6f}", f"{rec['a_col'][i]:.6e}",
                         f"{rec['a_zc'][i]:.6e}", f"{rec['h_b'][i]:.6e}", f"{rec['h_s'][i]:.6e}",
                         f"{rec['max_u'][i]:.6e}", f"{rec['phi_min'][i]:.6e}",
                         f"{rec['phi_max'][i]:.6e}", f"{rec['mass_drift_pct'][i]:.3e}",
                         f"{rec['a_uy'][i]:.6e}"])
    (d / "instant_rate.csv").write_text(
        "t,a_zc_over_lam,sigma_inst\n" + "\n".join(
            f"{t:.1f},{a:.6e},{s:.6e}"
            for t, a, s in instantaneous_rate(rec["t"], rec["a_zc"])
        ) + "\n", encoding="utf-8")
    (d / "case.json").write_text(json.dumps(case, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return case


# ---------------------------------------------------------------------------
# suite
# ---------------------------------------------------------------------------
def aggregate(out_root: Path) -> dict:
    cases = [json.loads(f.read_text()) for f in sorted(out_root.glob("*/case.json"))]
    summary = {
        "case_code": "CASE-RT-INSTABILITY-ATWOOD",
        "title": "2-D single-mode Rayleigh-Taylor at A = 1/2: linear (spike) growth rate",
        "solver": "tensorlbm.rayleigh_taylor (Boussinesq two-layer D2Q9 + Allen-Cahn interface)",
        "anchor": "Tryggvason/Scardovelli/Zaleski (2011) spectral-DNS lineage; targets = closed forms",
        "gate": {"tolerance_pct": GATE_PCT, "window_primary": WINDOWS[PRIMARY_WINDOW],
                 "estimator": "phi=0 interface amplitude, least squares on ln a"},
        "cases": cases,
        "n_cases": len(cases),
        "n_gate_pass": sum(1 for c in cases if c.get("gate_sigma_15pct")),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    rows = ["| case | scheme | lam | g | nu | sigma_sim | sigma_visc | err vs visc | "
            "err (legacy win) | sigma_flow(uy) err | nu_eff/nu | Ma | |phi|max | gate |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cases:
        rows.append(
            f"| {c['name']} | {c['phase_scheme']} | {c['lam']} | {c['g']:.1e} | {c['nu']:.4f} | "
            f"{c['sigma_sim']:.5e} | {c['sigma_viscous']:.5e} | {c['err_vs_viscous_pct']:+.2f}% | "
            f"{c['err_legacy_window_vs_viscous_pct']:+.2f}% | "
            f"{c['err_flow_uy_vs_viscous_pct']:+.2f}% | {c['nu_eff_over_nu']:.2f} | "
            f"{c['ma_peak']:.3f} | {max(abs(c['phi_min']), abs(c['phi_max'])):.2f} | "
            f"{'PASS' if c.get('gate_sigma_15pct') else 'FAIL'} |")
    (out_root / "acceptance_table.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="smoke run (<=400 steps per case)")
    ap.add_argument("--only", nargs="*", default=None, help="subset of case names")
    ap.add_argument("--out", default=None)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--specs", choices=("main", "extra"), default="main")
    args = ap.parse_args(argv)
    out_root = Path(args.out) if args.out else BENCH_DIR / "artifacts" / "runs"
    if args.aggregate:
        s = aggregate(out_root)
        print(f"[RT-A050] aggregate: {s['n_gate_pass']}/{s['n_cases']} within {GATE_PCT}%")
        return 0
    base = SPECS if args.specs == "main" else SPECS_EXTRA
    specs = base if not args.only else tuple(x for x in base if x.name in set(args.only))
    out_root.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        c = run_case(spec, out_root, quick=args.quick)
        print(f"[RT-A050] {spec.name:30s} sigma={c['sigma_sim']:.5e} "
              f"(visc {c['sigma_viscous']:.5e}, {c['err_vs_viscous_pct']:+.2f}%) "
              f"legacy-window {c['err_legacy_window_vs_viscous_pct']:+.2f}% "
              f"flow(uy) {c['err_flow_uy_vs_viscous_pct']:+.2f}% "
              f"u_s={c['u_spike_sim']:.4e} nu_eff/nu={c['nu_eff_over_nu']:.2f} Ma={c['ma_peak']:.2f} "
              f"phi=[{c['phi_min']:.2f},{c['phi_max']:.2f}] {c['wall_seconds']:.0f}s",
              flush=True)
    s = aggregate(out_root)
    print(f"[RT-A050] {s['n_gate_pass']}/{s['n_cases']} cases within {GATE_PCT}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())