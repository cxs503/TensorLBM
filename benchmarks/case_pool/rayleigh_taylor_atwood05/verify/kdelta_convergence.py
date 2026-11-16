"""CASE-RT-INSTABILITY-ATWOOD — interface-thickness (k*xi) convergence of the spike growth rate.

What this probe answers
-----------------------
The accepted case report gates the measured linear growth rate against the
sharp-interface closed forms ``sigma_inv = sqrt(A g k)`` and
``sigma_visc = sqrt(A g k + nu^2 k^4) - nu k^2`` and passes at ``-10.14 %``.
The independent verifier (``verify/REPORT_VERIFY.md``, V1/V2, disposition items
1-3) asked for two things that the accepted artifact did not have:

1. a lane that separates *model-form* error (the Allen-Cahn interface has a
   finite thickness ``xi``; the reference solution is a **sharp** interface, so
   the dimensionless group ``k*xi`` is not the same in the two problems) from
   *solver* error;
2. an estimate of how much of the ``-10 %`` is estimator (window) bias rather
   than physics.

The lane is the ``--specs kdelta`` set of ``run_case.py``: a resolution ladder
that pins every dimensionless group of the linear problem *except* ``k*xi``:

    zeta = nu k^2 / sqrt(A g k) = 1.538e-2   (viscous / inviscid ratio)
    Ma ~ sqrt(A g lam)                       (lattice compressibility)
    k*xi = 2 pi ac_width / lam               <-- the axis of this study

Implementation of the pinning: ``nu = nu0 lam/64`` (so ``tau = 0.5 + 3 nu``),
``g = g0 64/lam``.  The pre-existing grid ladder of ``SPECS`` instead pins
``A g / lam``, which makes Ma grow like ``lam`` (analytic: Ma ~ sqrt(A g lam));
the second figure panel exposes that confound.

Two estimators are reported per arm, both pre-declared:
  W  the case's accepted estimator  — LS slope of ln(a) over a/lam in [0.02, 0.08];
  P  the *peak instantaneous* rate   — max of d ln a/dt over the band
     a/lam in [0.012, 0.12] (12-sample centred slope). This is the least
     window-biased read of the asymptotic linear rate that the run contains,
     and it is what the startup-amplitude ladder is used to validate.

Usage
-----
    python3 verify/kdelta_convergence.py            # table + json + figure
"""

from __future__ import annotations

import csv
import glob
import json
import math
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[1]
ART = BENCH_DIR / "artifacts" / "kdelta"
RUNS = BENCH_DIR / "artifacts" / "runs_kdelta"
MACH_RES = BENCH_DIR / "artifacts_mach_res"
FIGS = BENCH_DIR / "figures"

#: band used by the peak-instantaneous-rate estimator (on a/lam)
PEAK_BAND = (0.012, 0.12)
#: case-pool acceptance tolerance of this case
GATE_PCT = 15.0
#: the platform's general accuracy standard quoted in the mission contract
TARGET_PCT = 3.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def ls(xs, ys):
    """Least squares ``y = a + b x`` -> ``(a, b, r2)``."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0.0:
        return float("nan"), float("nan"), float("nan")
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, (1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0)


def peak_rate(path: Path) -> tuple[float, float]:
    """(sigma_peak, a/lam at the peak) over :data:`PEAK_BAND`; nan if empty."""
    lo, hi = PEAK_BAND
    best, best_a = float("nan"), float("nan")
    if not path.exists():
        return best, best_a
    for row in csv.DictReader(path.open()):
        a = float(row["a_zc_over_lam"])
        s = float(row["sigma_inst"])
        if lo <= a <= hi and math.isfinite(s) and not (s < best if math.isfinite(best) else False):
            best, best_a = s, a
    return best, best_a


def arm_record(run_dir: Path) -> dict:
    c = json.loads((run_dir / "case.json").read_text())
    k = 2.0 * math.pi / c["lam"]
    sig_p, a_p = peak_rate(run_dir / "instant_rate.csv")
    d_win = c["sigma_inviscid"] - c["sigma_sim"]
    rec = {
        "name": c["name"],
        "lam": c["lam"],
        "g": c["g"],
        "nu": c["nu"],
        "tau": c["tau"],
        "ac_mobility": c["ac_mobility"],
        "ac_width": c["ac_width"],
        "eta0_frac": c.get("eta0_frac", 0.01),
        "n_steps": c.get("n_steps"),
        # pinned dimensionless groups
        "kxi": 2.0 * math.pi * c["ac_width"] / c["lam"],
        "zeta": c["nu"] * k * k / math.sqrt(c["atwood"] * c["g"] * k),
        "ma_peak": c["ma_peak"],
        "sqrt_Ag_lam": math.sqrt(c["atwood"] * c["g"] * c["lam"]),
        # references
        "sigma_inv": c["sigma_inviscid"],
        "sigma_visc": c["sigma_viscous"],
        # estimator W (accepted window estimator)
        "sigma_W": c["sigma_sim"],
        "err_W_vs_inv_pct": c["err_vs_inviscid_pct"],
        "err_W_vs_visc_pct": c["err_vs_viscous_pct"],
        "delta_W": d_win,
        "nu_eff_over_nu": d_win / (k * k) / c["nu"],
        # estimator P (peak instantaneous rate)
        "sigma_P": sig_p,
        "a_over_lam_at_peak": a_p,
        "err_P_vs_inv_pct": 100.0 * (sig_p / c["sigma_inviscid"] - 1.0) if math.isfinite(sig_p) else float("nan"),
        "err_P_vs_visc_pct": 100.0 * (sig_p / c["sigma_viscous"] - 1.0) if math.isfinite(sig_p) else float("nan"),
        "delta_P": c["sigma_inviscid"] - sig_p,
    }
    return rec


def extrapolate(records: list[dict]) -> dict:
    """Fit ``err(kxi) = e0 + m kxi`` on both estimators; report e0 (kxi -> 0)."""
    out = {}
    for tag, key in (("W", "err_W_vs_inv_pct"), ("P", "err_P_vs_inv_pct")):
        pts = [(r["kxi"], r[key]) for r in records
               if math.isfinite(r[key]) and math.isfinite(r["kxi"])]
        if len(pts) < 3:
            out[tag] = {"n": len(pts), "note": "insufficient points (need >= 3)"}
            continue
        # the kxi=0 arm (if it is the largest lam) is excluded from the fit only
        # if lam is far outside the ladder; here every arm is a real point.
        e0, m, r2 = ls([p[0] for p in pts], [p[1] for p in pts])
        out[tag] = {"n": len(pts), "e0_pct": e0, "slope_pct_per_kxi": m, "r2": r2,
                    "kxi_min": min(p[0] for p in pts)}
        # one-parameter (proportional) fit as a falsification check: err = c*kxi
        num = sum(p[0] * p[1] for p in pts)
        den = sum(p[0] * p[0] for p in pts)
        out[tag]["proportional_c_pct_per_kxi"] = num / den if den else float("nan")
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    all_recs = [arm_record(Path(d)) for d in sorted([d for d in glob.glob(str(RUNS / "*")) if (Path(d)/"case.json").is_file()])]

    # the kdelta ladder = similarity-preserving arms (eta0 fixed at 0.01, name KD_lNNN[...])
    ladder = [r for r in all_recs if r["eta0_frac"] == 0.01
              and r["name"].startswith("KD_l") and "_a" not in r["name"]]
    ladder.sort(key=lambda r: r["lam"])
    a0_ladder = sorted([r for r in all_recs if r["name"].startswith("KD_l064_a")],
                       key=lambda r: -r["eta0_frac"])
    ref = [r for r in all_recs if r["name"] == "KD_l064"]

    ext = extrapolate(ladder)

    # --- confound of the pre-existing grid ladder (A g / lam pinned) ---------
    confound = {"accepted_ladder": [], "kdelta_ladder": []}
    for f in sorted(glob.glob(str(MACH_RES / "*/case.json"))):
        c = json.loads(Path(f).read_text())
        # rungs of the accepted ladder reproduced by probe_mach_resolution.py
        rung = {("M_l064_g2e-4"): "l064", ("L_l096_g3e-04"): "l096"}.get(c["name"])
        if rung:
            confound["accepted_ladder"].append(
                {"lam": c["lam"], "g": c["g"], "ma_peak_measured": c["ma_peak"],
                 "name": c["name"]})
    for r in ladder:
        confound["kdelta_ladder"].append({"lam": r["lam"], "g": r["g"],
                                          "ma_peak_measured": r["ma_peak"], "name": r["name"]})
    # analytic scaling of the accepted ladder (A g / lam pinned -> Ma ~ lam)
    if confound["accepted_ladder"]:
        base = confound["accepted_ladder"][0]
        for lam in [96, 128]:
            g = base["g"] * lam / base["lam"]          # A g / lam = const
            ma = base["ma_peak_measured"] * math.sqrt(
                (0.5 * g * lam) / (0.5 * base["g"] * base["lam"]))
            confound.setdefault("accepted_ladder_analytic", []).append(
                {"lam": lam, "g": g, "ma_peak_analytic": ma})

    payload = {
        "case_code": "CASE-RT-INSTABILITY-ATWOOD",
        "probe": "kxi convergence (interface thickness -> 0 at fixed zeta, Ma)",
        "pinned_groups": {"zeta": ladder[0]["zeta"] if ladder else None,
                          "sqrt_Ag_lam": ladder[0]["sqrt_Ag_lam"] if ladder else None},
        "ladder": ladder,
        "a0_ladder": a0_ladder,
        "reference": ref,
        "extrapolation_kxi_to_zero": ext,
        "confound": confound,
        "gate_pct": GATE_PCT,
        "target_pct": TARGET_PCT,
    }
    (ART / "convergence.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8")

    # --- markdown table ------------------------------------------------------
    rows = ["| arm | lam | k\\*xi | zeta | Ma_peak | sigma_W | err_W vs inv | sigma_P | "
            "err_P vs inv | nu_eff/nu (W) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in ladder:
        rows.append(
            f"| {r['name']} | {r['lam']} | {r['kxi']:.4f} | {r['zeta']:.4e} | "
            f"{r['ma_peak']:.3f} | {r['sigma_W']:.5e} | {r['err_W_vs_inv_pct']:+.2f}% | "
            f"{r['sigma_P']:.5e} | {r['err_P_vs_inv_pct']:+.2f}% | {r['nu_eff_over_nu']:.2f} |")
    rows.append("")
    rows.append("| a0 arm | eta0/lam | sigma_W | err_W vs inv | sigma_P | err_P vs inv |")
    rows.append("|---|---|---|---|---|---|")
    for r in a0_ladder:
        rows.append(f"| {r['name']} | {r['eta0_frac']:.4f} | {r['sigma_W']:.5e} | "
                    f"{r['err_W_vs_inv_pct']:+.2f}% | {r['sigma_P']:.5e} | "
                    f"{r['err_P_vs_inv_pct']:+.2f}% |")
    rows.append("")
    for tag in ("W", "P"):
        e = ext.get(tag, {})
        if "e0_pct" in e:
            rows.append(f"* estimator {tag}: err(kxi) = {e['e0_pct']:+.2f}% "
                        f"{e['slope_pct_per_kxi']:+.1f}%·kxi, R² = {e['r2']:.4f}, "
                        f"n = {e['n']}  ->  kxi→0 error = **{e['e0_pct']:+.2f}%**")
        else:
            rows.append(f"* estimator {tag}: {e.get('note', 'n/a')}")
    (ART / "convergence_table.md").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # --- figure --------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[kdelta] matplotlib unavailable ({exc}); json/table written only")
        return 0

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=150)
    ax = axes[0]
    if ladder:
        kxi = [r["kxi"] for r in ladder]
        for tag, key, colour, label in (
                ("W", "err_W_vs_inv_pct", "tab:blue", "window LS fit (accepted estimator)"),
                ("P", "err_P_vs_inv_pct", "tab:red", "peak instantaneous rate")):
            ys = [r[key] for r in ladder]
            ax.plot(kxi, ys, "o-", color=colour, label=label)
            e = ext.get(tag, {})
            if "e0_pct" in e:
                xs = [0.0] + sorted(kxi)[::-1]
                ax.plot(xs, [e["e0_pct"] + e["slope_pct_per_kxi"] * x for x in xs],
                        "--", color=colour, alpha=0.7, lw=1.0)
                ax.plot([0.0], [e["e0_pct"]], color=colour, marker="*", ms=12)
                ax.annotate(f"kξ→0: {e['e0_pct']:+.1f}%", (0.0, e["e0_pct"]),
                            textcoords="offset points", xytext=(6, -12), color=colour,
                            fontsize=8)
        ax.axhspan(-GATE_PCT, GATE_PCT, color="tab:green", alpha=0.07,
                   label=f"case gate ±{GATE_PCT:.0f}%")
        ax.axhspan(-TARGET_PCT, TARGET_PCT, color="tab:orange", alpha=0.12,
                   label=f"platform target ±{TARGET_PCT:.0f}%")
        ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel(r"interface thickness in wavelengths  $k\xi = 2\pi\,w/\lambda$")
    ax.set_ylabel(r"error of $\sigma$ vs closed form $\sqrt{Agk}$  [%]")
    ax.set_title("A = 1/2 single-mode RT: linear spike growth rate\n"
                 r"(similarity ladder: $\zeta$, Ma pinned, $k\xi$ refined)", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    al = confound["accepted_ladder"]
    kl = confound["kdelta_ladder"]
    if al:
        ax2.plot([x["lam"] for x in al], [x["ma_peak_measured"] for x in al], "s-",
                 color="tab:purple", label=r"accepted ladder ($A g/\lambda$ pinned)")
    for x in confound.get("accepted_ladder_analytic", []):
        ax2.plot([x["lam"]], [x["ma_peak_analytic"]], "s", mfc="none",
                 color="tab:purple")
    if kl:
        ax2.plot([x["lam"] for x in kl], [x["ma_peak_measured"] for x in kl], "o-",
                 color="tab:red", label=r"this ladder ($\zeta$, Ma pinned)")
    ax2.set_xlabel(r"$\lambda$  [lattice units]")
    ax2.set_ylabel(r"peak lattice Mach number  $Ma$")
    ax2.set_title("why the old grid ladder cannot isolate $k\\xi$\n"
                  "(fineness and compressibility were moved together)", fontsize=9)
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "kdelta_convergence.png", bbox_inches="tight")
    plt.close(fig)

    print(f"[kdelta] arms: {[r['name'] for r in ladder]}")
    for tag in ("W", "P"):
        e = ext.get(tag, {})
        if "e0_pct" in e:
            print(f"[kdelta] estimator {tag}: kxi->0 error {e['e0_pct']:+.2f}% "
                  f"(slope {e['slope_pct_per_kxi']:+.1f} %/kxi, R2={e['r2']:.3f}, n={e['n']})")
    print(f"[kdelta] artifacts: {ART/'convergence.json'}  figure: {FIGS/'kdelta_convergence.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())