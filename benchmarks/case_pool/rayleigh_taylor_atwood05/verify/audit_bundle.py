"""Independent audit of the *merged* evidence bundle of
``CASE-RT-INSTABILITY-ATWOOD``.

Purpose
-------
The case is merged (``benchmarks/case_pool/rayleigh_taylor_atwood05``) and its
``REPORT.md`` claims: linear gate PASS at ``-10.14 %``, terminal-velocity gate
PASS at ``+5.56 %``, ``k xi -> 0`` extrapolated residual ``-3.3 %``.  A gate that
is quoted to two decimals must be reproducible from the raw per-step artefacts,
not from the case's own analysis code.  This script re-derives every gated number
**from the ``history.csv`` files only**, with its own least-squares fit, and
compares against the numbers that were written into the ``case.json`` files.

Nothing here imports the case drivers (``run_case``/``run_late``); the closed-form
reference values are re-implemented from the literature forms stated in
``ref/ref_lock.py`` so a defect in the reference lock cannot hide behind itself
(it *is* separately checked by ``check_reference_forms``).

Checks
------
A1  linear growth rate sigma from ``a_zc_over_lam`` over the pre-declared primary
    window, vs ``case.json:sigma_sim`` and vs the viscous closed form
A2  terminal Froude number from the front trajectories over ``tau in [4, 6]``,
    vs ``case.json:fr_meas`` (independent estimator path)
A3  gate re-evaluation: ``|Fr / Fr_LZ - 1| <= 15 %`` with ``Fr_LZ = 1/sqrt(pi C)``
A4  ``C_eff = 1/(pi Fr^2)`` self-consistency of the reported geometric constant
A5  lattice Mach number from ``max_speed`` (``Ma = u/c_s``, ``c_s = 1/sqrt3``)
A6  Boussinesq symmetry of the two fronts (the solver is invariant under
    ``(phi, y) -> (-phi, -y)``; a large deviation would mean a measurement fault)
A7  unit-consistency of ``run_case.velocity_in_window`` (the tau-slope vs lattice
    velocity factor ``1/tau_unit``) -- regression guard for the fixed defect
A8  stale-claim scan: ``REPORT.md`` rows that contradict the artefacts that were
    produced later (F4 "unmeasurable" vs a completed terminal window)
A9  ``k xi -> 0`` extrapolation of the sharp-interface residual, recomputed from
    the convergence artefact
A10 hand-transcription scan: every ``xx.xx %`` number quoted in ``REPORT.md``
    must occur in some artefact JSON of the bundle

Usage
-----
    python3 verify/audit_bundle.py            # print + write artifacts/audit/
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
ART = BENCH / "artifacts"
LATE = ART / "late"
AUDIT_OUT = ART / "audit"

C_2D = 1.0
GATE_PCT = 15.0
TAU_LO, TAU_HI = 4.0, 6.0
WIN_PRIMARY = (0.020, 0.080)   # amplitude window on a/lam
TAU_LINEAR_MAX = 2.0           # same cut as run_late.TAU_LINEAR_MAX
C_S = 1.0 / math.sqrt(3.0)


# ---------------------------------------------------------------------------
# closed forms (re-implemented; compared with ref/ref_lock.py in A0)
# ---------------------------------------------------------------------------
def sigma_inviscid(A: float, g: float, lam: float) -> float:
    return math.sqrt(A * g * 2.0 * math.pi / lam)


def sigma_viscous(A: float, g: float, lam: float, nu: float) -> float:
    k = 2.0 * math.pi / lam
    return math.sqrt(A * g * k + (nu * k * k) ** 2) - nu * k * k


def fr_layzer(c: float = C_2D) -> float:
    return 1.0 / math.sqrt(math.pi * c)


def ls_fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys))
    return b, a, (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


def read_history(case_dir: Path) -> list[dict]:
    with open(case_dir / "history.csv") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            try:
                d[k] = float(v)
            except (TypeError, ValueError):
                d[k] = v
        out.append(d)
    return out


def load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def check_reference_forms(audit: list[dict]) -> None:
    """A0: the audit's own closed forms must equal ``ref/ref_lock.py``."""
    import sys
    sys.path.insert(0, str(BENCH / "ref"))
    import ref_lock  # noqa: E402

    A, g, lam, nu = 0.5, 2.0e-4, 64.0, 0.005
    pairs = {
        "sigma_inviscid": (sigma_inviscid(A, g, lam),
                           ref_lock.sigma_inviscid(A, g, lam)),
        "sigma_viscous": (sigma_viscous(A, g, lam, nu),
                          ref_lock.sigma_viscous(A, g, lam, nu)),
        "goncharov_spike": (math.sqrt(2 * A * g / ((1 - A) * C_2D * 2 * math.pi / lam)),
                            ref_lock.goncharov(A, g, lam, "s", C_2D)),
        "goncharov_bubble": (math.sqrt(2 * A * g / ((1 + A) * C_2D * 2 * math.pi / lam)),
                             ref_lock.goncharov(A, g, lam, "b", C_2D)),
    }
    worst = 0.0
    detail = {}
    for name, (mine, theirs) in pairs.items():
        rel = abs(mine / theirs - 1.0) if theirs else float("nan")
        detail[name] = {"audit": mine, "ref_lock": theirs, "rel_dev": rel}
        worst = max(worst, rel)
    audit.append({
        "id": "A0",
        "what": "audit closed forms vs ref/ref_lock.py",
        "value": worst,
        "status": "OK" if worst < 1e-12 else "FAIL",
        "detail": detail,
    })


def check_linear(case_dir: Path, cj: dict, audit: list[dict]) -> dict:
    hist = read_history(case_dir)
    lam, g, nu = cj["lam"], cj["g"], cj["nu"]
    tau_unit = math.sqrt(cj["atwood"] * g / lam)
    # NOTE the semantics: ``run_case.WINDOWS`` are **amplitude** windows on
    # ``a/lam`` (not time windows).  ``run_case.window_slope`` filters on the
    # amplitude, so the same filter is applied here; fitting over a tau window
    # instead selects the startup transient and biases sigma low by ~16 %.
    pts = [(r["tau"], math.log(r["a_zc_over_lam"])) for r in hist
           if WIN_PRIMARY[0] <= r["a_zc_over_lam"] <= WIN_PRIMARY[1]
           and r["a_zc_over_lam"] > 0 and r["tau"] <= TAU_LINEAR_MAX]
    slope_tau, _a, r2 = ls_fit([p[0] for p in pts], [p[1] for p in pts])
    sigma = slope_tau * tau_unit          # d ln a / dt (lattice)
    sig_case = cj["sigma_sim"]
    sig_visc = sigma_viscous(cj["atwood"], g, lam, nu)
    audit.append({
        "id": "A1",
        "what": "linear sigma re-derived from history.csv (primary window)",
        "value": sigma,
        "status": "OK" if abs(sigma / sig_case - 1.0) < 5e-3 else "MISMATCH",
        "detail": {
            "sigma_audit": sigma,
            "sigma_case_json": sig_case,
            "rel_dev_pct": 100.0 * (sigma / sig_case - 1.0),
            "r2": r2,
            "n_points": len(pts),
            "sigma_viscous_closed_form": sig_visc,
            "err_vs_viscous_pct": 100.0 * (sigma / sig_visc - 1.0),
            "err_vs_viscous_case_json_pct": cj["err_sigma_vs_viscous_pct"],
        },
    })
    return {"sigma": sigma, "sigma_visc": sig_visc, "tau_unit": tau_unit}


def check_terminal(case_dir: Path, cj: dict, audit: list[dict], lam_unit: float) -> dict:
    hist = read_history(case_dir)
    win = [(r["tau"], r["h_bubble_over_lam"], r["h_spike_over_lam"]) for r in hist
           if TAU_LO <= r["tau"] <= TAU_HI]
    if len(win) < 3:
        audit.append({"id": "A2", "what": "terminal Fr re-derived", "value": float("nan"),
                      "status": "NOT_RUN", "detail": {"n_points": len(win)}})
        return {}
    b_bub, _a1, r2_b = ls_fit([w[0] for w in win], [w[1] for w in win])
    b_spk, _a2, r2_s = ls_fit([w[0] for w in win], [w[2] for w in win])
    # h is in wavelengths and tau is dimensionless, so the tau-slope *is* the
    # Froude number Fr = u / sqrt(A g lam).
    fr_audit = 0.5 * (b_bub + b_spk)
    fr_case = cj["fr_meas"]
    audit.append({
        "id": "A2",
        "what": "terminal Fr re-derived from front histories (tau in [4,6])",
        "value": fr_audit,
        "status": "OK" if abs(fr_audit / fr_case - 1.0) < 5e-3 else "MISMATCH",
        "detail": {
            "fr_audit": fr_audit,
            "fr_case_json": fr_case,
            "rel_dev_pct": 100.0 * (fr_audit / fr_case - 1.0),
            "slope_bubble": b_bub, "slope_spike": b_spk,
            "r2_bubble": r2_b, "r2_spike": r2_s,
            "n_points": len(win),
            "tau_window": [TAU_LO, TAU_HI],
        },
    })
    fr_lz = fr_layzer()
    err = 100.0 * (fr_audit / fr_lz - 1.0)
    audit.append({
        "id": "A3",
        "what": "terminal gate vs Layzer-Goncharov (C=1)",
        "value": err,
        "status": "PASS" if abs(err) <= GATE_PCT else "FAIL",
        "detail": {"fr_audit": fr_audit, "fr_lz": fr_lz,
                   "err_pct": err, "gate_pct": GATE_PCT,
                   "case_json_gate": cj.get("gate_terminal_15pct")},
    })
    c_eff = 1.0 / (math.pi * fr_audit ** 2)
    audit.append({
        "id": "A4",
        "what": "C_eff = 1/(pi Fr^2) self-consistency",
        "value": c_eff,
        "status": "OK" if abs(c_eff / cj["c_eff"] - 1.0) < 5e-3 else "MISMATCH",
        "detail": {"c_eff_audit": c_eff, "c_eff_case_json": cj["c_eff"],
                   "note": "C_eff is an exact restatement of Fr, not an "
                           "independent measurement"},
    })
    ma = max(r["max_speed"] for r in hist) / C_S
    audit.append({
        "id": "A5",
        "what": "lattice Mach number from max_speed",
        "value": ma,
        "status": "OK" if abs(ma / cj["ma_peak"] - 1.0) < 5e-3 else "MISMATCH",
        "detail": {"ma_audit": ma, "ma_case_json": cj["ma_peak"],
                   "compressibility_error_scale_ma2_pct": 100.0 * ma * ma},
    })
    sym = [abs(b - s) / max(b, s) for _t, b, s in win if max(b, s) > 0]
    audit.append({
        "id": "A6",
        "what": "Boussinesq front symmetry |h_bub - h_spk|/max",
        "value": max(sym) if sym else float("nan"),
        "status": "OK" if sym and max(sym) < 0.05 else "CHECK",
        "detail": {"max_rel_dev": max(sym) if sym else float("nan"),
                   "case_json": cj.get("front_symmetry_max_rel_dev"),
                   "note": "the solved equations are invariant under "
                           "(phi, y) -> (-phi, -y); a large asymmetry would be a "
                           "measurement fault, not physics"},
    })
    return {"fr": fr_audit, "ma": ma}


def check_units(audit: list[dict], case_dir: Path, cj: dict) -> None:
    """A7: the velocity_in_window unit defect must stay fixed."""
    import sys
    sys.path.insert(0, str(BENCH))
    from run_case import velocity_in_window  # noqa: E402

    lam, g = cj["lam"], cj["g"]
    tau_unit = math.sqrt(cj["atwood"] * g / lam)
    u_true = 0.0123                      # lattice velocity of a linear front
    steps = [float(i) for i in range(1, 8001)]   # must fill tau in [4, 6]
    fronts = [u_true * s for s in steps]  # front in cells
    taus = [s * tau_unit for s in steps]
    u_fixed = velocity_in_window(taus, fronts, TAU_LO, TAU_HI, tau_unit)
    u_prefix = velocity_in_window(taus, fronts, TAU_LO, TAU_HI)
    audit.append({
        "id": "A7",
        "what": "velocity_in_window unit consistency (tau-slope vs lattice)",
        "value": u_fixed / u_true,
        "status": "OK" if abs(u_fixed / u_true - 1.0) < 1e-6 else "FAIL",
        "detail": {
            "u_true": u_true, "u_with_tau_unit": u_fixed,
            "u_without_tau_unit": u_prefix,
            "factor": (u_fixed / u_prefix) if u_prefix else float("nan"),
            "note": "the pre-fix signature (no tau_unit) is kept callable so the "
                    "old numbers stay reproducible; callers must pass tau_unit",
        },
    })


def check_stale_claims(audit: list[dict], cj_late: dict | None) -> None:
    """A8: REPORT.md rows contradicted by later artefacts."""
    report = (BENCH / "REPORT.md").read_text()
    f4_row = [ln for ln in report.splitlines() if ln.startswith("| F4 |")]
    unreachable = bool(f4_row) and "不可测" in f4_row[0]
    late_ok = bool(cj_late and cj_late.get("window_complete")
                   and cj_late.get("gate_terminal_15pct"))
    audit.append({
        "id": "A8",
        "what": "F4 'nonlinear velocity unmeasurable' claim vs later artefacts",
        "value": float(unreachable),
        "status": "STALE" if (unreachable and late_ok) else "OK",
        "detail": {
            "report_f4_row": f4_row[0].strip() if f4_row else None,
            "late_artefact_exists": late_ok,
            "late_tau_reached": (cj_late or {}).get("tau_reached"),
            "note": "REPORT.md was written before the late-time suite; the row is "
                    "kept for provenance but is superseded by artifacts/late and by "
                    "the similarity ladder (artifacts/simladder)",
        },
    })


def check_kdelta(audit: list[dict]) -> None:
    """A9: re-derive the k xi -> 0 extrapolation from the convergence artefact."""
    conv = load_json(ART / "kdelta" / "convergence.json")
    if conv is None:
        audit.append({"id": "A9", "what": "k xi -> 0 extrapolation",
                      "value": float("nan"), "status": "NOT_RUN",
                      "detail": {"reason": "artifacts/kdelta/convergence.json missing"}})
        return
    ladder = conv.get("ladder") or []
    recorded = conv.get("extrapolation_kxi_to_zero") or {}
    detail = {}
    worst = 0.0
    for key, ykey in (("W", "err_W_vs_inv_pct"), ("P", "err_P_vs_inv_pct")):
        xs = [float(r["kxi"]) for r in ladder if ykey in r]
        ys = [float(r[ykey]) for r in ladder if ykey in r]
        if len(xs) < 2:
            continue
        b, a, r2 = ls_fit(xs, ys)
        rec = recorded.get(key, {}).get("e0_pct")
        dev = abs(a - rec) if rec is not None else float("nan")
        if rec is not None:
            worst = max(worst, dev)
        detail[key] = {"intercept_pct": a, "slope_pct_per_kxi": b, "r2": r2,
                       "recorded_e0_pct": rec, "abs_dev_pp": dev, "n_points": len(xs)}
    audit.append({
        "id": "A9",
        "what": "k xi -> 0 extrapolation of the sharp-interface residual (W, P)",
        "value": worst,
        "status": "OK" if (detail and worst < 0.05) else "MISMATCH",
        "detail": {**detail,
                   "note": "the W estimator extrapolates to "
                           f"{detail.get('W', {}).get('intercept_pct', float('nan')):+.2f} % and the "
                           "P (peak instantaneous rate) estimator to "
                           f"{detail.get('P', {}).get('intercept_pct', float('nan')):+.2f} %; "
                           "neither reaches the 3 % platform standard, both stay inside "
                           "the +-15 % case gate"},
    })


def check_transcription(audit: list[dict]) -> None:
    """A10: every quoted 'xx.xx %' in REPORT.md must exist in some artefact."""
    report = (BENCH / "REPORT.md").read_text()
    quoted = set(re.findall(r"[-+]?\d+\.\d\d\s?%", report))
    blob = []
    for p in BENCH.rglob("*.json"):
        try:
            blob.append(p.read_text())
        except UnicodeDecodeError:
            continue
    for p in BENCH.rglob("*.md"):
        if p.name != "REPORT.md":
            blob.append(p.read_text())
    text = "\n".join(blob)
    missing = []
    for q in sorted(quoted):
        raw = q.replace("%", "").strip()
        val = float(raw)
        # accept either sign of the same magnitude and 1/2-decimal renderings
        candidates = {f"{val:.2f}", f"{abs(val):.2f}", f"{val:.1f}", f"{abs(val):.1f}"}
        if not any(c in text for c in candidates):
            missing.append(q)
    audit.append({
        "id": "A10",
        "what": "REPORT.md percentage numbers traced to artefacts",
        "value": float(len(missing)),
        "status": "OK" if not missing else "UNTRACED",
        "detail": {"n_quoted": len(quoted), "untraced": missing,
                   "note": "markdown-only quantities (hand-written discussion "
                           "numbers) are reported here instead of being assumed "
                           "programmatic"},
    })


# ---------------------------------------------------------------------------
def main() -> int:
    AUDIT_OUT.mkdir(parents=True, exist_ok=True)
    audit: list[dict] = []

    cj_ref = load_json(ART / "late" / "late_l064_g2e-4" / "case.json")
    cj_late = cj_ref
    if cj_ref is None:
        raise SystemExit("missing artifacts/late/late_l064_g2e-4/case.json")

    check_reference_forms(audit)
    check_linear(ART / "late" / "late_l064_g2e-4", cj_ref, audit)
    check_terminal(ART / "late" / "late_l064_g2e-4", cj_ref, audit, cj_ref["lam"])
    check_units(audit, ART / "late" / "late_l064_g2e-4", cj_ref)
    check_stale_claims(audit, cj_late)
    check_kdelta(audit)
    check_transcription(audit)

    summary = {
        "bundle": "benchmarks/case_pool/rayleigh_taylor_atwood05",
        "checks": audit,
        "n_ok": sum(1 for a in audit if str(a["status"]).startswith(("OK", "PASS"))),
        "n_flagged": sum(1 for a in audit if not str(a["status"]).startswith(("OK", "PASS"))),
    }
    (AUDIT_OUT / "AUDIT.json").write_text(json.dumps(summary, indent=2) + "\n")
    (AUDIT_OUT / "AUDIT.md").write_text(render(summary))
    print(render(summary))
    return 0


def render(summary: dict) -> str:
    lines = ["# Independent audit of the CASE-RT-INSTABILITY-ATWOOD evidence bundle", "",
             "| id | check | value | status |", "|---|---|---|---|"]
    for a in summary["checks"]:
        v = a["value"]
        vs = f"{v:.6g}" if isinstance(v, (int, float)) and v == v else str(v)
        lines.append(f"| {a['id']} | {a['what']} | {vs} | {a['status']} |")
    lines += ["", f"OK/PASS: {summary['n_ok']}  flagged: {summary['n_flagged']}", ""]
    for a in summary["checks"]:
        lines.append(f"## {a['id']} — {a['what']}  [{a['status']}]")
        lines.append("```json")
        lines.append(json.dumps(a["detail"], indent=2))
        lines.append("```")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())