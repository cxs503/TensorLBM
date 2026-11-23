"""CASE-RT-INSTABILITY-ATWOOD -- late-time (nonlinear) front-velocity study.

Companion of ``run_case.py``.  ``run_case.py`` gates the **linear** growth rate
and reports the saturated front velocity as ``n/a``; the anchor title of this
case is "spike growth rate", so the nonlinear half of the anchor was still an
open measurement debt.  ``REPORT.md`` closed it as F4, "unmeasurable in this
configuration", with the stated cause "the interface hits the ``a/lam = 0.60``
guardrail at ``tau ~ 1.9`` while the terminal window needs a taller box".

Two independent defects were found in that conclusion (see
``probe_symmetry.py`` for the raw diagnostic that exposes the second one):

1. **stop criterion** -- ``MAX_A_FRAC = 0.60`` is not the wall.  A ``layers = 8``
   box is 4 wavelengths deep on each side, so ``a/lam = 0.6`` still leaves 3.4
   wavelengths of fluid.  The run was stopped by an amplitude *threshold* and the
   pre-declared terminal window ``tau in [4, 6]`` (``run_case.TAU_LO/TAU_HI``) was
   never entered.  Replaced here by a wall-distance criterion
   (:func:`late_analysis.late_stop_reason`).
2. **front estimator** -- ``run_case`` defines ``h_up = max(zc) - mean(zc)`` and
   ``h_down = mean(zc) - min(zc)`` where ``zc`` is a per-column interface height
   and ``mean`` is the *instantaneous* column mean.  Both ingredients fail once
   the interface rolls up: ``interface_zero_crossing`` returns the **lowest**
   crossing, so the extrema stop tracking the bubble/spike tips, and the
   instantaneous mean drifts (measured ``mean - y0 = -0.41 lam`` at ``tau =
   3.25``).  ``probe_symmetry.py`` shows the consequence: the same run gives
   ``h_up = 1.21 lam`` / ``h_down = 0.38 lam`` from that definition, while the
   physical tips are at ``0.79 lam`` / ``0.79 lam``.  The "asymmetry" is a
   measurement artefact, not physics (the equations are invariant under
   ``(phi, y) -> (-phi, -y)``).

This driver measures, with the interface *tip* definition and a **fixed**
reference height ``y0 = ny/2`` (the initial interface mean):

    bubble tip  y_b(t) = max_x ( uppermost phi = 0 crossing of column x )
    spike  tip  y_s(t) = min_x ( lowest    phi = 0 crossing of column x )
    h_bub = y_b - y0,  h_spk = y0 - y_s   (in wavelengths)

and also reports the legacy definition, so the bias stays visible instead of
being silently replaced.  A 90th-percentile variant of the tip estimator is
carried along so an isolated detached droplet cannot fake a front.

GATE (pre-declared here, before any late-time number was inspected)
------------------------------------------------------------------
``|Fr_meas / Fr_LZ - 1| <= 15 %`` on the reference case, where
``Fr = u / sqrt(A g lam)`` for the mean of the two tip velocities and
``Fr_LZ = 1/sqrt(pi C)`` with ``C = 1`` is the Layzer-Goncharov potential-flow
limit **in the Boussinesq form** (uniform inertia = the ``A -> 0`` limit of the
``1 -+ A`` factors).  The measured ``C_eff = 1/(pi Fr^2)`` is reported as the
primary number, because it is directly comparable with the ``C = 0.90`` that the
sibling case ``benchmarks/case_pool/rayleigh_taylor`` measured at ``A = 1/2``.

The finite-Atwood two-fluid asymmetry ``u_spike/u_bubble = sqrt((1+A)/(1-A)) =
1.732`` is **NOT SUPPORTED** by a Boussinesq solver (uniform inertia) -- it is
measured, not assumed, and reported with the size of the structural gap.

Nothing in ``run_case.py`` is modified; the case definition is imported from it
so the two drivers cannot drift apart.

Usage
-----
    python3 run_late.py                 # reference + grid ladder + wall control
    python3 run_late.py --specs base    # reference point only
    python3 run_late.py --quick         # 400-step smoke (no physics claim)
    python3 run_late.py --aggregate     # rebuild summary/acceptance table
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(int(os.environ.get("RT_THREADS", "4")))

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parents[2]
SRC_ROOT = REPO_ROOT / "src"
for _p in (str(SRC_ROOT), str(REPO_ROOT), str(BENCH_DIR), str(BENCH_DIR / "ref")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import late_analysis as la  # noqa: E402
import ref_lock  # noqa: E402
from run_case import CaseSpec, WINDOWS, PRIMARY_WINDOW, window_slope  # noqa: E402

from tensorlbm.d2q9 import macroscopic  # noqa: E402
from tensorlbm.rayleigh_taylor import (  # noqa: E402
    init_rt_state,
    interface_zero_crossing,
    mode_amplitude_zc,
    rt_step,
)

OUT_ROOT = BENCH_DIR / "artifacts" / "late"
#: the linear-phase samples of these long runs are cut at this dimensionless time
#: (the fitted windows themselves are the pre-declared ones of run_case.py; the
#: cut only prevents late samples that re-enter the window after the interface
#: has rolled up from contaminating the fit).
TAU_LINEAR_MAX = 2.0


# ---------------------------------------------------------------------------
# interface measurement
# ---------------------------------------------------------------------------
def topmost_crossing(phi: torch.Tensor) -> torch.Tensor:
    """Uppermost ``phi = 0`` crossing per column (NaN where there is none).

    Same linear interpolation and the same sign change (light ``->`` heavy,
    i.e. ``negative -> positive`` in ``phi``) as
    ``tensorlbm.rayleigh_taylor.interface_zero_crossing``, but it returns the
    **last** such transition of a column instead of the first.

    With ``phi > 0`` = heavy (top) fluid, every light/heavy boundary of a column
    is such a transition, so ``min`` over columns of the *first* transition is the
    tip of the descending heavy tongue (the spike) and ``max`` over columns of the
    *last* transition is the tip of the ascending light finger (the bubble).  Both
    tips are needed; ``run_case.py`` only used the first-transition array, which is
    why its ``h_up`` stops tracking the bubble tip as soon as a column contains
    more than one transition (a rolled-up interface).
    """
    neg = phi < 0.0
    cross = neg[:-1] & ~neg[1:]
    has = cross.any(dim=0)
    nrow = cross.shape[0]
    idx = (nrow - 1) - torch.flip(cross, dims=(0,)).to(torch.int64).argmax(dim=0)
    cols = torch.arange(phi.shape[1], device=phi.device)
    c0 = phi[idx, cols].to(torch.float64)
    c1 = phi[idx + 1, cols].to(torch.float64)
    out = idx.to(torch.float64) + (-c0) / (c1 - c0)
    return torch.where(has, out, torch.full_like(out, float("nan")))


def _q(t: torch.Tensor, q: float) -> float:
    """``q``-quantile of the finite entries of ``t`` (NaN when empty)."""
    v = t[~torch.isnan(t)]
    if v.numel() == 0:
        return float("nan")
    return float(torch.quantile(v.to(torch.float64), q))


def front_state(phi: torch.Tensor, y0: float, lam: int) -> dict:
    """All front measurements of one sample (lengths in wavelengths)."""
    low = interface_zero_crossing(phi)
    top = topmost_crossing(phi)
    mean = (
        _q(low, 0.5)
        if bool(torch.isnan(low).all())
        else float(torch.nan_to_num(low, nan=0.0).sum() / max(1, int((~torch.isnan(low)).sum())))
    )
    neg = phi < 0.0
    multi = int(((neg[:-1] & ~neg[1:]).sum(dim=0) > 1).sum())
    return {
        # --- tip definition (this study)
        "h_bub": (_q(top, 1.0) - y0) / lam,
        "h_spk": (y0 - _q(low, 0.0)) / lam,
        "h_bub_q90": (_q(top, 0.90) - y0) / lam,
        "h_spk_q10": (y0 - _q(low, 0.10)) / lam,
        # --- legacy definition of run_case.py (kept visible, not used)
        "h_up_zc": (_q(low, 1.0) - mean) / lam,
        "h_dn_zc": (mean - _q(low, 0.0)) / lam,
        "mean_dev": (mean - y0) / lam,
        "multi": multi,
    }


def late_specs(base: bool = False) -> tuple[CaseSpec, ...]:
    """Long runs of the case-definition suite (``tau_end = 6.5`` is already the
    value in ``run_case.CaseSpec``; here it is actually reached)."""
    ref = CaseSpec(
        "late_l064_g2e-4", lam=64, g=2.0e-4, note="reference point, terminal window reached"
    )
    ladder = CaseSpec("late_l096_g3e-4", lam=96, g=3.0e-4, note="1.5x grid at constant A g/lam")
    wall = CaseSpec(
        "late_l064_g2e-4_L10",
        lam=64,
        g=2.0e-4,
        layers=10.0,
        note="wall-confinement control (5 lam per side)",
    )
    return (ref,) if base else (ref, ladder, wall)


# ---------------------------------------------------------------------------
# one long run
# ---------------------------------------------------------------------------
def run_case(spec: CaseSpec, out_root: Path = OUT_ROOT, quick: bool = False) -> dict:
    cfg = spec.cfg()
    cfg.validate()
    torch.manual_seed(0)

    n_steps = min(cfg.n_steps, 400) if quick else cfg.n_steps
    clearance = la.wall_clearance_lam(spec.layers)
    f, g, phi = init_rt_state(cfg)
    ny = phi.shape[0]
    y0 = 0.5 * ny  # fixed reference: initial interface mean
    tau_unit = math.sqrt(spec.atwood * spec.g / spec.lam)
    mass0 = float(g[:, 1:-1, :].sum().item())
    ncell = g[:, 1:-1, :].numel()

    keys = (
        "step",
        "tau",
        "h_bub",
        "h_spk",
        "h_bub_q90",
        "h_spk_q10",
        "h_up_zc",
        "h_dn_zc",
        "mean_dev",
        "multi",
        "a_zc",
        "max_u",
        "phi_min",
        "phi_max",
        "phi_drift",
    )
    rec: dict[str, list[float]] = {k: [] for k in keys}
    t0 = time.time()
    stop_reason = None
    for step in range(1, n_steps + 1):
        f, g = rt_step(f, g, cfg, phi=phi)
        phi = g.sum(dim=0)
        if step % cfg.sample_every == 0 or step == 1:
            fs = front_state(phi, y0, spec.lam)
            a_zc = mode_amplitude_zc(phi, spec.lam) / spec.lam
            _rho, ux, uy = macroscopic(f)
            speed = float(torch.sqrt(ux * ux + uy * uy).max())
            mass = float(g[:, 1:-1, :].sum().item())
            rec["step"].append(step)
            rec["tau"].append(step * tau_unit)
            for k in (
                "h_bub",
                "h_spk",
                "h_bub_q90",
                "h_spk_q10",
                "h_up_zc",
                "h_dn_zc",
                "mean_dev",
                "multi",
            ):
                rec[k].append(fs[k])
            rec["a_zc"].append(a_zc)
            rec["max_u"].append(speed)
            rec["phi_min"].append(float(phi.min()))
            rec["phi_max"].append(float(phi.max()))
            # phi is a *non-conserved* Allen-Cahn order parameter and its domain
            # integral is ~0 by construction, so report the drift per cell in
            # units of the +-1 amplitude instead of a meaningless percentage.
            rec["phi_drift"].append(100.0 * (mass - mass0) / ncell)
            reason = la.late_stop_reason(
                max(fs["h_bub"], fs["h_spk"]), speed / math.sqrt(1.0 / 3.0), spec.layers
            )
            if reason is not None:
                stop_reason = reason
                break
    wall = time.time() - t0

    # ---- late-time terminal velocity (the anchor quantity, tip definition)
    v_b = la.front_velocity(rec["tau"], rec["h_bub"])
    v_s = la.front_velocity(rec["tau"], rec["h_spk"])
    v_b90 = la.front_velocity(rec["tau"], rec["h_bub_q90"])
    v_s10 = la.front_velocity(rec["tau"], rec["h_spk_q10"])
    v_legacy = la.front_velocity(rec["tau"], rec["h_up_zc"])

    def to_u(slope: float) -> float:
        # h is recorded in wavelengths and tau is dimensionless, so the fitted
        # slope is d(h/lam)/dtau; multiply by lam to feed the lattice formula.
        return la.lattice_velocity(slope * spec.lam, spec.atwood, spec.g, spec.lam)

    u_b, u_s = to_u(v_b["slope"]), to_u(v_s["slope"])
    u_mean = 0.5 * (u_b + u_s)
    u_robust = 0.5 * (to_u(v_b90["slope"]) + to_u(v_s10["slope"]))
    u_legacy = to_u(v_legacy["slope"])
    u_lz = la.layzer_velocity(spec.atwood, spec.g, spec.lam, la.C_2D)
    sq = math.sqrt(spec.atwood * spec.g * spec.lam)
    fr_meas, fr_lz = u_mean / sq, la.layzer_froude(spec.atwood, spec.g, spec.lam, la.C_2D)

    # sub-window stability inside the gate window (is the velocity really constant?)
    half = la.front_velocity(
        rec["tau"][: len(rec["tau"])], rec["h_bub"], la.TAU_LO, 0.5 * (la.TAU_LO + la.TAU_HI)
    )["slope"]
    half2 = la.front_velocity(rec["tau"], rec["h_bub"], 0.5 * (la.TAU_LO + la.TAU_HI), la.TAU_HI)[
        "slope"
    ]

    # ---- two-fluid reference (structural: uniform inertia cannot produce it)
    asym = la.two_fluid_asymmetry(spec.atwood)
    u_s_2f = ref_lock.goncharov(spec.atwood, spec.g, spec.lam, "s", ref_lock.C_2D)

    # ---- linear growth rate on the same (long) run: cross-check with run_case.py
    lin = [(t, a) for t, a in zip(rec["tau"], rec["a_zc"]) if t <= TAU_LINEAR_MAX]
    sigmas = {
        w: window_slope([p[0] for p in lin], [p[1] for p in lin], lo, hi)
        for w, (lo, hi) in WINDOWS.items()
    }
    # the windows are fitted against tau, so convert the slope to d ln a / dt
    # (lattice units) before comparing with sigma_viscous
    tau_factor = math.sqrt(spec.lam / (spec.atwood * spec.g))
    sigma_primary = sigmas[PRIMARY_WINDOW]["sigma"] / tau_factor
    sig_ref = ref_lock.sigma_viscous(spec.atwood, spec.g, spec.lam, spec.nu)

    asym_dev = [
        abs(a - b) / max(a, b)
        for a, b in zip(rec["h_bub"], rec["h_spk"])
        if max(a, b) > 0.0 and math.isfinite(a) and math.isfinite(b)
    ]
    case = {
        "name": spec.name,
        "note": spec.note,
        "atwood": spec.atwood,
        "lam": spec.lam,
        "g": spec.g,
        "nu": spec.nu,
        "layers": spec.layers,
        "phase_scheme": spec.phase_scheme,
        "ac_width": spec.ac_width,
        "ac_mobility": spec.ac_mobility,
        "n_steps_planned": cfg.n_steps,
        "n_steps_run": int(rec["step"][-1]),
        "tau_reached": rec["tau"][-1],
        "stop_reason": stop_reason,
        "wall_clearance_lam": clearance,
        "wall_seconds": wall,
        "sample_every": cfg.sample_every,
        # --- terminal (nonlinear) measurement, tip definition
        "window": [la.TAU_LO, la.TAU_HI],
        "front_fit_n": v_b["n"],
        "front_fit_r2_bub": v_b["r2"],
        "front_fit_r2_spk": v_s["r2"],
        "u_bubble_sim": u_b,
        "u_spike_sim": u_s,
        "u_mean_sim": u_mean,
        "u_robust_sim": u_robust,
        "fr_meas": fr_meas,
        "fr_layzer_c1": fr_lz,
        "fr_layzer_c3": la.layzer_froude(spec.atwood, spec.g, spec.lam, 3.0),
        "err_fr_vs_layzer_c1_pct": la.rel_pct(fr_meas, fr_lz),
        "c_eff": la.effective_c(spec.atwood, spec.g, spec.lam, u_mean),
        "u_layzer_c1": u_lz,
        "slope_first_half": half,
        "slope_second_half": half2,
        "slope_subwindow_drift_pct": la.rel_pct(half2, half),
        # --- legacy estimator (biased at late time; reported, not used)
        "u_legacy_estimator": u_legacy,
        "err_legacy_estimator_vs_tip_pct": la.rel_pct(u_legacy, u_mean),
        "mean_dev_max_lam": max(abs(x) for x in rec["mean_dev"]),
        "multi_columns_max": int(max(rec["multi"])),
        # --- two-fluid (structurally unreachable: uniform inertia)
        "two_fluid_asymmetry_pred": asym,
        "two_fluid_asymmetry_sim": (u_b / u_s)
        if (math.isfinite(u_s) and u_s > 0.0)
        else float("nan"),
        "u_spike_two_fluid_c1": u_s_2f,
        "err_u_spike_vs_two_fluid_pct": la.rel_pct(u_s, u_s_2f),
        "front_symmetry_max_rel_dev": max(asym_dev) if asym_dev else float("nan"),
        # --- linear cross-check
        "sigma_sim": sigma_primary,
        "sigma_units": "lattice (d ln a / dt)",
        "sigma_viscous": sig_ref,
        "err_sigma_vs_viscous_pct": la.rel_pct(sigma_primary, sig_ref),
        "windows": sigmas,
        # --- health
        "ma_peak": max(rec["max_u"]) / math.sqrt(1.0 / 3.0),
        "phi_min": min(rec["phi_min"]),
        "phi_max": max(rec["phi_max"]),
        "phi_integral_drift_pct_per_cell": max(abs(x) for x in rec["phi_drift"]),
        "front_max_lam": max(max(rec["h_bub"]), max(rec["h_spk"])),
    }
    covered = v_b["tau_range"]
    case["window_covered"] = covered
    complete = bool(
        all(math.isfinite(x) for x in covered)
        and covered[0] <= la.TAU_LO + 0.25
        and covered[1] >= la.TAU_HI - 0.25
    )
    case["window_complete"] = complete
    case["gate_terminal_15pct"] = (
        None
        if not complete
        else bool(
            math.isfinite(case["err_fr_vs_layzer_c1_pct"])
            and abs(case["err_fr_vs_layzer_c1_pct"]) <= la.GATE_PCT
        )
    )

    d = out_root / spec.name
    d.mkdir(parents=True, exist_ok=True)
    with (d / "history.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                "step",
                "tau",
                "h_bubble_over_lam",
                "h_spike_over_lam",
                "h_bubble_q90_over_lam",
                "h_spike_q10_over_lam",
                "h_up_zc_over_lam",
                "h_dn_zc_over_lam",
                "mean_dev_over_lam",
                "multi_columns",
                "a_zc_over_lam",
                "max_speed",
                "phi_min",
                "phi_max",
                "phi_integral_drift_pct_per_cell",
            ]
        )
        for i, s in enumerate(rec["step"]):
            wr.writerow(
                [int(s), f"{rec['tau'][i]:.6f}"]
                + [
                    f"{rec[k][i]:.6e}"
                    for k in (
                        "h_bub",
                        "h_spk",
                        "h_bub_q90",
                        "h_spk_q10",
                        "h_up_zc",
                        "h_dn_zc",
                        "mean_dev",
                    )
                ]
                + [
                    int(rec["multi"][i]),
                    f"{rec['a_zc'][i]:.6e}",
                    f"{rec['max_u'][i]:.6e}",
                    f"{rec['phi_min'][i]:.6e}",
                    f"{rec['phi_max'][i]:.6e}",
                    f"{rec['phi_drift'][i]:.3e}",
                ]
            )
    (d / "case.json").write_text(
        json.dumps(case, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return case


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def aggregate(out_root: Path = OUT_ROOT) -> dict:
    cases = []
    for p in sorted(out_root.glob("*/case.json")):
        cases.append(json.loads(p.read_text(encoding="utf-8")))
    summary = {"cases": cases, "gate_pct": la.GATE_PCT, "window": [la.TAU_LO, la.TAU_HI]}
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# CASE-RT-INSTABILITY-ATWOOD -- late-time (nonlinear) front velocity",
        "",
        f"Gate: |Fr_meas / Fr_LZ(C=1) - 1| <= {la.GATE_PCT:.0f} %, window "
        f"tau in [{la.TAU_LO}, {la.TAU_HI}]; Fr = u / sqrt(A g lam), "
        "Fr_LZ = 1/sqrt(pi C) (Layzer-Goncharov, Boussinesq form).",
        "Measured C_eff = 1 / (pi Fr^2) (compare C = 0.90 from the sibling case at A = 1/2).",
        "",
        "| case | steps | tau_end | stop | Ma_peak | Fr_meas | Fr_LZ | err | gate | C_eff | "
        "robust Fr | subwindow drift |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cases:
        rob = c["u_robust_sim"] / math.sqrt(c["atwood"] * c["g"] * c["lam"])
        lines.append(
            f"| {c['name']} | {c['n_steps_run']} | {c['tau_reached']:.2f} | "
            f"{c['stop_reason'] or 'completed'} | {c['ma_peak']:.3f} | "
            f"{c['fr_meas']:.4f} | {c['fr_layzer_c1']:.4f} | "
            f"{c['err_fr_vs_layzer_c1_pct']:+.2f}% | "
            f"{'INCOMPLETE' if c['gate_terminal_15pct'] is None else ('PASS' if c['gate_terminal_15pct'] else 'FAIL')} | {c['c_eff']:.3f} | "
            f"{rob:.4f} | {c['slope_subwindow_drift_pct']:+.2f}% |"
        )
    lines += [
        "",
        "Front-velocity estimator comparison (same runs, same window):",
        "",
        "| case | u_bubble (tip) | u_spike (tip) | u_legacy (mean(zc)) | legacy vs tip | "
        "mean(zc) drift [lam] | multi-valued columns | front symmetry dev |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in cases:
        lines.append(
            f"| {c['name']} | {c['u_bubble_sim']:.4e} | {c['u_spike_sim']:.4e} | "
            f"{c['u_legacy_estimator']:.4e} | {c['err_legacy_estimator_vs_tip_pct']:+.2f}% | "
            f"{c['mean_dev_max_lam']:.3f} | {c['multi_columns_max']} | "
            f"{c['front_symmetry_max_rel_dev']:.2e} |"
        )
    lines += [
        "",
        "Two-fluid (finite-density-ratio) asymmetry -- measured, not assumed:",
        "",
        "| case | u_spike/u_bubble (sim) | two-fluid prediction | err vs two-fluid spike |",
        "|---|---|---|---|",
    ]
    for c in cases:
        lines.append(
            f"| {c['name']} | {c['two_fluid_asymmetry_sim']:.4f} | "
            f"{c['two_fluid_asymmetry_pred']:.4f} | "
            f"{c['err_u_spike_vs_two_fluid_pct']:+.2f}% |"
        )
    (out_root / "acceptance_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--specs", choices=("late", "base"), default="late")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args(argv)
    out_root = Path(args.out)
    if args.aggregate:
        s = aggregate(out_root)
        print(json.dumps({c["name"]: c["gate_terminal_15pct"] for c in s["cases"]}, indent=2))
        return 0
    for spec in late_specs(base=(args.specs == "base")):
        case = run_case(spec, out_root, quick=args.quick)
        print(
            f"[{case['name']}] steps={case['n_steps_run']} tau={case['tau_reached']:.2f} "
            f"stop={case['stop_reason']} Fr={case['fr_meas']:.4f} "
            f"C_eff={case['c_eff']:.3f} err={case['err_fr_vs_layzer_c1_pct']:+.2f}% "
            f"gate={'INCOMPLETE' if case['gate_terminal_15pct'] is None else ('PASS' if case['gate_terminal_15pct'] else 'FAIL')} "
            f"wall={case['wall_seconds']:.0f}s",
            flush=True,
        )
    aggregate(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
