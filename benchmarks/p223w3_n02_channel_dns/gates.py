#!/usr/bin/env python3
"""Machine-checkable acceptance gates for p223w3-n02 (turbulent channel).

Every number in ``case_data/gates.json`` is computed here from the case records
written by :mod:`campaign` plus the MKM/Hoyas DNS tables shipped in the repo -
nothing is transcribed by hand.

Gates (verbatim from the campaign node):

===== ==========================================================================
(a)   log-law ``U+`` vs ``(1/kappa) ln(y+) + B``, ``y+ in [30, 200]``, error < 5 %
      (kappa = 0.41, B = 5.2), evaluated at Re_tau = 180 / 395 / 590
(b)   ``u'u'+`` peak near ``y+ ~ 15`` and its magnitude within 10 % of DNS
      (MKM-1999, Re_tau = 180)
(c)   inertial-subrange slope of the streamwise spectrum within 10 % of -5/3
(d)   friction coefficient ``Cf`` within 5 % of Dean's correlation
(e)   laminar negative control: plane-Poiseuille error < 1 %
(f)   grid convergence at Re_tau = 180: spread of the independently measured
      friction velocity across three wall-normal spacings < 3 %
(g)   pytest suite >= 25 tests, all green (asserted by the test module itself)
===== ==========================================================================
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE))

import analysis as an  # noqa: E402

RESULTS = _HERE / "case_data"
#: Authoritative MKM-1999 tables shipped with the library (header + 7/8 columns,
#: normalised by u_tau and h).  ``provenance.json`` in the same directory records
#: the upstream tarball SHA-256 (turbulence.oden.utexas.edu, chandata.tar.gz).
MKM_DIR = _REPO / "src" / "tensorlbm" / "data" / "mkm1999"
DNS_FILES = {180: ("chan180.means", "chan180.reystress"),
             395: ("chan395.means", "chan395.reystress"),
             590: ("chan590.means", "chan590.reystress")}
MEAN_COLS = ("y", "y_plus", "u_plus", "dudy", "w_mean", "dwdy", "p_mean")
STRESS_COLS = ("y", "y_plus", "uu_plus", "vv_plus", "ww_plus", "uv_plus", "uw_plus", "vw_plus")

KAPPA = 0.41
B_LOG = 5.2
SPEC_BAND = (0.05, 0.30)   # k+ band fixed a priori (not tuned to the data)


def _read_mkm(path: Path, names: tuple[str, ...]) -> dict:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append([float(v) for v in line.split()])
    arr = np.asarray(rows, dtype=float)
    if arr.shape[1] != len(names):
        raise ValueError(f"{path.name}: expected {len(names)} columns, "
                         f"got {arr.shape[1]}")
    return {name: arr[:, i] for i, name in enumerate(names)}


def load_dns(re_tau: int) -> dict:
    """Mean + Reynolds-stress tables of MKM-1999 at the requested Re_tau."""
    means_file, stress_file = DNS_FILES[re_tau]
    data = _read_mkm(MKM_DIR / means_file, MEAN_COLS)
    data.update({k: v for k, v in
                 _read_mkm(MKM_DIR / stress_file, STRESS_COLS).items()
                 if k not in data})
    return data


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gate(name: str, criterion: str, passed: bool, evidence: dict, problems=(),
         withheld=()) -> dict:
    """One gate record.  ``withheld`` lists sub-claims that could not be
    evaluated (missing case / infeasible cost); they never turn a gate green."""
    if passed and not withheld:
        status = "PASS"
    elif withheld:
        status = "WITHHELD"
    else:
        status = "FAIL"
    return {"gate": name, "criterion": criterion, "passed": bool(passed) and not withheld,
            "status": status, "evidence": evidence, "problems": list(problems),
            "withheld": list(withheld)}


def load_case(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def available(case_names) -> list[str]:
    return [n for n in case_names if (RESULTS / f"{n}.json").exists()]


# ---------------------------------------------------------------------------
# individual gates
# ---------------------------------------------------------------------------

def gate_log_law(cases: list[dict], required=(180.0, 395.0, 590.0)) -> dict:
    rows, problems = {}, []
    have = {rec["geometry"]["re_tau"] for rec in cases}
    withheld = [f"Re_tau={int(r)}: no case record (see report section 5)"
                for r in required if r not in have]
    for rec in cases:
        log = rec["diagnostics"]["log_law"]
        ok = log["rms_rel_error"] < 0.05
        if not ok:
            problems.append(f"{rec['case']}: log-law RMS {log['rms_rel_error']:.2%}")
        rows[rec["case"]] = {
            "re_tau": rec["geometry"]["re_tau"],
            "y_plus_window": [30.0, 200.0],
            "n_points": log["n_points"],
            "rms_rel_error": log["rms_rel_error"],
            "max_rel_error": log["max_rel_error"],
            "kappa_fit": log["k_fit"],
            "b_fit": log["b_fit"],
            "passed": ok,
        }
    return gate("a_log_law",
                "U+ vs (1/kappa)ln(y+)+B, y+ in [30,200], RMS error < 5% "
                "(kappa=0.41, B=5.2), all three Re_tau",
                all(r["passed"] for r in rows.values()), rows, problems, withheld)


def gate_reynolds_stress(rec: dict) -> dict:
    dns = load_dns(180)
    yp_dns = dns["y_plus"]
    uu_dns = dns["uu_plus"]
    i = int(np.argmax(uu_dns))
    dns_peak = {"value": float(uu_dns[i]), "y_plus": float(yp_dns[i])}
    sim = rec["diagnostics"]["uu_peak"]
    yp = np.asarray(rec["profiles"]["y_plus"])
    uu = np.asarray(rec["profiles"]["uu_plus"])
    # quadratic (3-point) refinement of the discrete peak position
    j = int(np.argmax(uu))
    if 0 < j < uu.size - 1:
        y0, y1, y2 = yp[j - 1], yp[j], yp[j + 1]
        f0, f1, f2 = uu[j - 1], uu[j], uu[j + 1]
        denom = (f0 - 2 * f1 + f2)
        shift = 0.5 * (f0 - f2) / denom if abs(denom) > 0 else 0.0
        y_peak = y1 + shift * (0.5 * (y2 - y0))
    else:
        y_peak = sim["y_plus"]
    rel = (sim["value"] - dns_peak["value"]) / dns_peak["value"]
    pos_ok = 10.0 <= y_peak <= 20.0
    mag_ok = abs(rel) < 0.10
    return gate(
        "b_reynolds_stress",
        "u'u'+ peak located near y+ = 15 (accept 10..20) and its magnitude "
        "within 10% of MKM-1999 DNS at Re_tau = 180",
        bool(pos_ok and mag_ok),
        {
            "dns_peak": dns_peak,
            "dns_re_tau": 178.12,
            "sim_peak_discrete": sim,
            "sim_peak_refined_y_plus": float(y_peak),
            "sim_grid_dy_plus": rec["geometry"]["dy_plus"],
            "relative_magnitude_error": float(rel),
            "position_ok": bool(pos_ok),
            "magnitude_ok": bool(mag_ok),
        },
        ([f"peak magnitude error {rel:+.1%}"] if not mag_ok else [])
        + ([f"peak position y+={y_peak:.1f}"] if not pos_ok else []),
    )


def gate_spectrum(cases: list[dict]) -> dict:
    rows, problems = {}, []
    if not cases:
        return gate("c_energy_spectrum",
                    "inertial-subrange slope of the streamwise spectrum within 10% "
                    "of -5/3 (k+ in [0.05, 0.30])", False, {},
                    ["no completed turbulent case"], ["no completed turbulent case"])
    for rec in cases:
        fit = rec["spectrum"]["fit"]
        slope = fit["slope"]
        rel = abs(slope - (-5.0 / 3.0)) / (5.0 / 3.0)
        ok = rel < 0.10
        if not ok:
            problems.append(f"{rec['case']}: slope {slope:.3f} ({rel:.1%} from -5/3)")
        rows[rec["case"]] = {"slope": slope, "rel_error_vs_-5/3": rel,
                             "band_k_plus": list(SPEC_BAND), "n_bins": fit["n"],
                             "passed": ok}
    return gate("c_energy_spectrum",
                "inertial-subrange slope of the streamwise spectrum within 10% "
                "of -5/3 (k+ in [0.05, 0.30])",
                all(r["passed"] for r in rows.values()), rows, problems)


def gate_friction(cases: list[dict], required=(180.0, 395.0, 590.0)) -> dict:
    rows, problems = {}, []
    have = {rec["geometry"]["re_tau"] for rec in cases}
    withheld = [f"Re_tau={int(r)}: no case record" for r in required if r not in have]
    for rec in cases:
        mf = rec["mean_flow"]
        err = mf["cf_error_vs_dean"]
        ok = abs(err) < 0.05
        if not ok:
            problems.append(f"{rec['case']}: Cf {mf['cf']:.5f} vs Dean "
                            f"{mf['cf_dean']:.5f} ({err:+.1%})")
        rows[rec["case"]] = {
            "re_tau": rec["geometry"]["re_tau"],
            "u_bulk": mf["u_bulk"],
            "re_bulk": mf["re_bulk"],
            "cf": mf["cf"],
            "cf_dean": mf["cf_dean"],
            "rel_error": err,
            "passed": ok,
        }
    return gate("d_friction_coefficient",
                "Cf = 2(u_tau/U_b)^2 within 5% of Dean (1978) Cf = 0.073 Re_b^-0.25",
                all(r["passed"] for r in rows.values()), rows, problems, withheld)


def gate_laminar(rec: dict) -> dict:
    norm = rec["norm"]
    ok = norm["l2_rel_error"] < 0.01
    return gate("e_laminar_control",
                "force-driven plane Poiseuille vs analytic parabola, L2 error < 1%",
                ok, {"re_laminar": rec["geometry"]["re_laminar"],
                     "steps": rec["geometry"]["steps"],
                     "l2_rel_error": norm["l2_rel_error"],
                     "max_rel_error": norm["max_rel_error"],
                     "u_max_sim": norm["u_max_sim"],
                     "u_max_exact": norm["u_max_exact"]},
                [] if ok else [f"L2 {norm['l2_rel_error']:.2%}"])


def gate_grid_convergence(cases: list[dict]) -> dict:
    """Diagnostic: do the three Re_tau=180 grids agree on the friction velocity?

    The u_tau of the body-force route is imposed, so the *physics* test is the
    spread of the two independent measurements (wall-normal stress fit over the
    plateau ``0.1 <= y/h <= 0.7``).
    """
    rows = {}
    for rec in sorted(cases, key=lambda r: -r["geometry"]["dy_plus"]):
        routes = rec["mean_flow"]["u_tau_routes"]
        rows[rec["case"]] = {
            "dy_plus": rec["geometry"]["dy_plus"],
            "ny": rec["geometry"]["ny"],
            "u_tau_body_force": routes["body_force"],
            "u_tau_stress_fit": routes["stress_fit"],
            "u_tau_imposed": routes["imposed"],
            "rel_dev_from_imposed": abs(routes["stress_fit"] - routes["imposed"])
            / routes["imposed"],
        }
    fits = np.array([r["u_tau_stress_fit"] for r in rows.values()])
    spread = float(fits.max() - fits.min()) / float(fits.mean()) if fits.size else float("nan")
    ok = spread < 0.03
    withheld = []
    if len(rows) < 3:
        withheld.append(f"only {len(rows)} Re_tau=180 grid(s) completed; the gate "
                        f"needs three (see the divergence log)")
    return gate("f_grid_convergence",
                "grid convergence at Re_tau = 180: spread of the independently "
                "measured u_tau across three resolutions < 3%",
                ok, {"grids": rows, "u_tau_spread_rel": spread,
                     "imposed_spread_rel": 0.0},
                [] if ok else [f"u_tau spread {spread:.2%} over dy+ = "
                               f"{sorted(r['dy_plus'] for r in rows.values())}"],
                withheld)


def gate_wall_units_consistency(cases: list[dict]) -> dict:
    """Extra (non-normative) check: momentum balance in wall units."""
    rows, problems = {}, []
    if not cases:
        return gate("extra_momentum_balance",
                    "diagnostic: total-stress budget in wall units", False, {},
                    ["no completed turbulent case"], ["no completed turbulent case"])
    for rec in cases:
        total = np.asarray(rec["profiles"]["uu_plus"]) * 0.0  # placeholder
        y_over_h = np.asarray(rec["profiles"]["y_over_h"])
        uv = np.asarray(rec["profiles"]["uv_plus"])
        dudy = np.gradient(np.asarray(rec["profiles"]["u_plus"]),
                           np.asarray(rec["profiles"]["y_plus"]))
        total = dudy - uv
        ref = 1.0 - y_over_h
        sel = (y_over_h > 0.1) & (y_over_h < 0.7)
        err = float(np.max(np.abs(total[sel] - ref[sel])))
        okay = err < 0.15
        if not okay:
            problems.append(f"{rec['case']}: |total stress - (1-y/h)| = {err:.3f}")
        rows[rec["case"]] = {"max_abs_balance_error": err, "passed": bool(okay)}
    return gate("extra_momentum_balance",
                "diagnostic: nu dU/dy - <u'v'> = u_tau^2 (1 - y/h) within 0.15 in "
                "wall units over 0.1 <= y/h <= 0.7",
                all(r["passed"] for r in rows.values()), rows, problems)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def evaluate(case_names=("R180_dy4", "R180_dy5", "R180_dy6", "R395_dy8", "R590_dy12"),
             laminar_name: str = "LAM_poiseuille") -> dict:
    case_names = available(case_names)
    cases = [load_case(n) for n in case_names]
    by_re = {}
    for rec in cases:
        by_re.setdefault(rec["geometry"]["re_tau"], []).append(rec)
    primary = [r for r in by_re.get(180.0, []) if r["geometry"]["dy_plus"] == 4.0] \
        or by_re.get(180.0, [])
    lam = load_case(laminar_name)

    stress_gate = (gate_reynolds_stress(primary[0]) if primary else
                   gate("b_reynolds_stress",
                        "u'u'+ peak located near y+ = 15 and its magnitude within "
                        "10% of MKM-1999 DNS at Re_tau = 180", False,
                        {"reason": "no completed Re_tau=180 case"},
                        ["no completed Re_tau=180 case"],
                        ["Re_tau=180: every run diverged (see case_data/*.failed.json)"]))
    gates = [
        gate_log_law(cases),
        stress_gate,
        gate_spectrum(cases),
        gate_friction(cases),
        gate_laminar(lam),
        gate_grid_convergence(by_re.get(180.0, [])),
        gate_wall_units_consistency(cases),
    ]
    normative = [g for g in gates if not g["gate"].startswith("extra_")]
    summary = {
        "gates": gates,
        "normative_passed": sum(g["passed"] for g in normative),
        "normative_total": len(normative),
        "all_normative_passed": all(g["passed"] for g in normative),
        "dns_reference": {str(k): {"files": list(DNS_FILES[k]),
                                   "sha256": [sha256(MKM_DIR / f)
                                              for f in DNS_FILES[k]],
                                   "citation": "Moser, Kim & Mansour, Phys. "
                                               "Fluids 11(4) 943 (1999)"}
                          for k in sorted(DNS_FILES)},
        "provenance": {
            "kernel": "src/tensorlbm/turbulence/les_chan.c",
            "kernel_sha256": sha256(_REPO / "src/tensorlbm/turbulence/les_chan.c"),
            "analysis": "benchmarks/p223w3_n02_channel_dns/analysis.py",
            "campaign": "benchmarks/p223w3_n02_channel_dns/campaign.py",
            "notes": "numbers computed by gates.py from case_data/*.json; no hand "
                     "transcription",
        },
    }
    (RESULTS / "gates.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> int:
    summary = evaluate()
    for g in summary["gates"]:
        print(f"[{g['status']}] {g['gate']}: {g['criterion']}")
        for p in g["problems"]:
            print(f"        - {p}")
    print(f"normative gates passed: {summary['normative_passed']}/"
          f"{summary['normative_total']}")
    return 0 if summary["all_normative_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())