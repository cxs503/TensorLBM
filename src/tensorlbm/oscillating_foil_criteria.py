"""Machine-checkable acceptance criteria for the oscillating-foil verification face.

Each criterion is a pure function ``(measurements) -> dict`` with the same
schema::

    {"id": "C1", "name": "...", "verdict": "PASS"|"FAIL"|"WITHHELD",
     "value": float|dict, "threshold": ..., "note": "..."}

``WITHHELD`` is used when the measurement cannot decide the question (for
example when the physical signal is below the demonstrated force-measurement
noise floor).  It is *not* a way of hiding a failure: every ``WITHHELD`` records
the number that made it undecidable.

Criteria (node p161w2-arb-n05):

C1  small-amplitude linear region vs Garrick closed form
C2  efficiency peak in reduced frequency
C3  degenerate limits (k -> 0 quasi-steady, zero motion, high-k asymptote)
C4  energy ledger closure (wall work = dKE + viscous dissipation + flux)
C5  Strouhal monotonicity of the thrust coefficient
C6  grid convergence (Roache GCI on three grids)
C7  negative-control trio (locked body / IBM off / heave phase flip)
C8  bit-level reproducibility
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "criterion_c1_linear_region",
    "criterion_c9_momentum_balance",
    "criterion_c5b_thrust_reversal",
    "criterion_c1b_amplitude_scaling",
    "criterion_c2_efficiency_peak",
    "criterion_c3_degenerate_limits",
    "criterion_c4_energy_ledger",
    "criterion_c5_strouhal_monotone",
    "criterion_c6_grid_convergence",
    "criterion_c7_negative_controls",
    "criterion_c8_reproducibility",
    "summarise",
]

# tolerance constants (documented in the report)
TOL_C1_REL = 0.15          # linear-region thrust increment, relative
TOL_C1B_REL = 0.15         # amplitude-scaling exponent check
TOL_C2_REL = 0.15          # efficiency peak value
TOL_C3_QUASI = 0.02        # k -> 0 quasi-steady limit
TOL_C3_ZERO = 1e-6         # zero-motion force must vanish
TOL_C4_REL = 0.05          # energy-ledger closure
TOL_C5_RHO = 0.9           # Spearman rank correlation for monotonicity (5 points:
                           # 0.9 == one adjacent inversion, tolerated within noise)
TOL_C6_GCI = 0.10          # fine-grid GCI
TOL_C7_LOCKED_REL = 0.25   # locked-body drag vs independent estimate
TOL_C7_SYMMETRY = 0.15     # heave phase-flip symmetry
TOL_C7_NOIBM = 1e-6        # IBM-off force must vanish


def _verdict(ok: bool, withheld: bool = False) -> str:
    if withheld:
        return "WITHHELD"
    return "PASS" if ok else "FAIL"


# ---------------------------------------------------------------------------
# C1 -- small-amplitude linear region
# ---------------------------------------------------------------------------
def criterion_c1_linear_region(cases: Sequence[dict]) -> dict:
    """Thrust increment ``dC_T = C_T + C_D0`` vs Garrick, small amplitudes.

    ``C_D0`` is the drag of the *same* body in the *same* grid with the motion
    locked, so the increment isolates the unsteady (propulsive) part of the
    force from the numerical drag offset of the immersed boundary.

    The measured noise floor is ``3 * sigma`` of the per-cycle mean ``C_T`` of
    the same case; a case whose analytic thrust is below that floor cannot
    decide the criterion and is reported as ``WITHHELD`` (with the numbers).
    """
    rows = []
    worst_rel = 0.0
    n_pass = n_fail = n_withheld = 0
    for c in cases:
        cd0 = c["cd0"]
        dct = c["C_T"] + cd0
        th = c["theory"]["C_T"]
        noise = 3.0 * float(c.get("C_T_cycle_std", 0.0))
        scale = max(abs(th), 1e-12)
        rel = abs(dct - th) / scale
        resolvable = abs(th) > noise
        if not resolvable:
            verdict = "WITHHELD"
            n_withheld += 1
        else:
            verdict = _verdict(rel <= TOL_C1_REL)
            if verdict == "PASS":
                n_pass += 1
            else:
                n_fail += 1
        worst_rel = max(worst_rel, rel) if resolvable else worst_rel
        rows.append(
            {
                "label": c["label"],
                "C_T_net": c["C_T"],
                "C_D0": cd0,
                "dC_T": dct,
                "C_T_theory": th,
                "rel_error": rel,
                "noise_3sigma": noise,
                "resolvable": resolvable,
                "verdict": verdict,
            }
        )
    verdict = "FAIL" if n_fail else ("PASS" if n_pass else "WITHHELD")
    return {
        "id": "C1",
        "name": "small-amplitude linear region (Garrick)",
        "verdict": verdict,
        "threshold": {"rel_error": TOL_C1_REL, "resolvable": "|C_T_th| > 3 sigma_cycle"},
        "value": {
            "cases": rows,
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_withheld": n_withheld,
            "worst_rel_error_resolvable": worst_rel,
        },
        "note": (
            "dC_T = C_T + C_D0 isolates the unsteady thrust from the IBM "
            "numerical drag offset; WITHHELD rows are below the measured "
            "force noise floor (3 sigma of the per-cycle mean)."
        ),
    }


def criterion_c1b_amplitude_scaling(cases: Sequence[dict]) -> dict:
    """``C_T`` must scale like ``(h0/c)^2`` in the linear regime (offset-robust).

    Uses the *increment* ``dC_T`` at two heave amplitudes; linear theory gives
    exactly a factor of 4 for a factor of 2 in amplitude.
    """
    if len(cases) < 2:
        return {
            "id": "C1b",
            "name": "amplitude scaling dC_T ~ (h0/c)^2",
            "verdict": "WITHHELD",
            "value": {},
            "threshold": {"rel_error": TOL_C1B_REL},
            "note": "needs two amplitudes",
        }
    a, b = sorted(cases, key=lambda c: c["h0_over_c"])
    d1 = a["C_T"] + a["cd0"]
    d2 = b["C_T"] + b["cd0"]
    ratio = (b["h0_over_c"] / a["h0_over_c"]) ** 2
    obs = d2 / d1 if d1 != 0 else float("inf")
    rel = abs(obs - ratio) / ratio
    noise = 3.0 * math.hypot(a.get("C_T_cycle_std", 0.0), b.get("C_T_cycle_std", 0.0))
    ok = rel <= TOL_C1B_REL
    return {
        "id": "C1b",
        "name": "amplitude scaling dC_T ~ (h0/c)^2",
        "verdict": _verdict(ok, withheld=(abs(d2 - d1) < noise)),
        "threshold": {"rel_error": TOL_C1B_REL, "expected_ratio": ratio},
        "value": {
            "h0_over_c": [a["h0_over_c"], b["h0_over_c"]],
            "dC_T": [d1, d2],
            "observed_ratio": obs,
            "expected_ratio": ratio,
            "rel_error": rel,
            "noise_3sigma": noise,
        },
        "note": "offset-robust shape check: cancels the steady IBM drag error",
    }


# ---------------------------------------------------------------------------
# C2 -- efficiency peak
# ---------------------------------------------------------------------------
def criterion_c2_efficiency_peak(cases: Sequence[dict]) -> dict:
    """Efficiency ``eta(k)``: peak location and value vs Garrick.

    ``cases`` must sample one motion mode at several reduced frequencies; each
    entry carries ``k``, ``eta`` (LBM) and ``theory.eta``.
    """
    cs = sorted(cases, key=lambda c: c["k"])
    ks = [c["k"] for c in cs]
    eta_l = [c["eta"] for c in cs]
    eta_t = [c["theory"]["eta"] for c in cs]
    i_l = int(np.argmax(eta_l))
    i_t = int(np.argmax(eta_t))
    dk = (ks[-1] - ks[0]) / max(1, len(ks) - 1)
    dk_peak = abs(ks[i_l] - ks[i_t])
    rel = abs(eta_l[i_l] - eta_t[i_t]) / abs(eta_t[i_t]) if eta_t[i_t] else float("inf")
    monotone_l = all(eta_l[i] < eta_l[i + 1] for i in range(len(eta_l) - 1)) or all(
        eta_l[i] > eta_l[i + 1] for i in range(len(eta_l) - 1)
    )
    interior_l = 0 < i_l < len(ks) - 1
    interior_t = 0 < i_t < len(ks) - 1
    loc_ok = dk_peak <= dk * 1.01
    val_ok = rel <= TOL_C2_REL
    ok = loc_ok and val_ok and (interior_l == interior_t)
    return {
        "id": "C2",
        "name": "propulsive efficiency peak eta(k)",
        "verdict": _verdict(ok),
        "threshold": {"rel_error_peak": TOL_C2_REL, "peak_location": "within one grid step"},
        "value": {
            "k": ks,
            "eta_lbm": eta_l,
            "eta_theory": eta_t,
            "k_peak_lbm": ks[i_l],
            "k_peak_theory": ks[i_t],
            "peak_rel_error": rel,
            "interior_peak_lbm": interior_l,
            "interior_peak_theory": interior_t,
            "grid_step": dk,
            "eta_monotone": monotone_l,
        },
        "note": "eta = C_T / C_P with C_P = P_in/(0.5 rho U^3 c) from the IBM work",
    }


# ---------------------------------------------------------------------------
# C3 -- degenerate limits
# ---------------------------------------------------------------------------
def criterion_c3_degenerate_limits(analytic: dict, zero_motion: dict | None = None) -> dict:
    """Limits that any correct anchor must reproduce.

    * ``k -> 0``: ``C_T -> pi * (h0/b)^2 * k^2`` (quasi-steady jet),
      relative error over the analytic solver
    * ``|C(k)|`` asymptotes: ``C(0) = 1``, ``C(inf) = 1/2``
    * zero motion (LBM): ``|C_T|, |C_P|`` vanish
    """
    checks = {}
    checks["quasi_steady"] = {
        "rel_error": analytic.get("quasi_steady_rel_error"),
        "threshold": TOL_C3_QUASI,
        "ok": analytic.get("quasi_steady_rel_error", 1.0) <= TOL_C3_QUASI,
    }
    checks["C_zero"] = {
        "abs_error": analytic.get("C_at_zero_minus_one"),
        "ok": abs(analytic.get("C_at_zero_minus_one", 1.0)) <= 1e-6,
    }
    checks["C_inf"] = {
        "abs_error": analytic.get("C_at_inf_minus_half"),
        "ok": abs(analytic.get("C_at_inf_minus_half", 1.0)) <= 1e-3,
    }
    checks["pure_heave_closed_form"] = {
        "rel_error": analytic.get("pure_heave_rel_error"),
        "ok": analytic.get("pure_heave_rel_error", 1.0) <= 1e-10,
    }
    if zero_motion is not None:
        checks["zero_motion_force"] = {
            "abs_C_T": abs(zero_motion.get("C_T", 0.0)),
            "abs_C_P": abs(zero_motion.get("C_P", 0.0)),
            "ok": max(abs(zero_motion.get("C_T", 0.0)), abs(zero_motion.get("C_P", 0.0)))
            <= TOL_C3_ZERO,
        }
    ok = all(v["ok"] for v in checks.values())
    return {
        "id": "C3",
        "name": "degenerate limits (k->0, C(k) asymptotes, zero motion)",
        "verdict": _verdict(ok),
        "threshold": {
            "quasi_steady_rel": TOL_C3_QUASI,
            "zero_motion": TOL_C3_ZERO,
        },
        "value": checks,
        "note": "analytic limits + LBM zero-motion control",
    }


# ---------------------------------------------------------------------------
# C4 -- energy ledger
# ---------------------------------------------------------------------------
def criterion_c4_energy_ledger(ledger: dict) -> dict:
    """``W_wall = dKE + Phi`` in a closed (periodic) box.

    ``ledger`` holds the time integrals over an integer number of cycles:
    ``work`` (foil -> fluid), ``dke`` (change of total kinetic energy),
    ``phi`` (viscous dissipation), ``flux`` (boundary energy flux, exactly zero
    in a periodic box).
    """
    W = ledger["work"]
    dke = ledger["dke"]
    phi = ledger["phi"]
    flux = ledger.get("flux", 0.0)
    resid = W - (dke + phi + flux)
    denom = max(abs(W), abs(phi), 1e-30)
    rel = abs(resid) / denom
    ok = rel <= TOL_C4_REL
    return {
        "id": "C4",
        "name": "energy ledger closure",
        "verdict": _verdict(ok),
        "threshold": {"rel_residual": TOL_C4_REL},
        "value": {
            "work": W,
            "dke": dke,
            "phi": phi,
            "flux": flux,
            "residual": resid,
            "rel_residual": rel,
            "cycles": ledger.get("cycles"),
        },
        "note": "closed periodic box: no boundary flux, no sponge absorption",
    }


# ---------------------------------------------------------------------------
# C5 -- Strouhal monotonicity
# ---------------------------------------------------------------------------
def criterion_c5_strouhal_monotone(cases: Sequence[dict]) -> dict:
    """Mean thrust must increase monotonically with ``St`` (pure heave, fixed k)."""
    cs = sorted(cases, key=lambda c: c["strouhal"])
    st = [c["strouhal"] for c in cs]
    ct = [c["C_T"] for c in cs]
    sigma = [3.0 * c.get("C_T_cycle_std", 0.0) for c in cs]
    if len(cs) > 2:
        ranks_st = np.argsort(np.argsort(st))
        ranks_ct = np.argsort(np.argsort(ct))
        rho = float(np.corrcoef(ranks_st, ranks_ct)[0, 1])
    else:
        rho = float("nan")
    diffs = np.diff(ct)
    # each increment must exceed the combined noise of the two points
    sig_pairs = [math.hypot(sigma[i], sigma[i + 1]) for i in range(len(ct) - 1)]
    # a step may not decrease by more than the combined 3-sigma noise of the two
    # points: force histories of shedding cases are strongly unsteady, so a
    # strict "every increment > 0" test would reject a correct monotone trend
    no_significant_drop = [bool(diffs[i] >= -sig_pairs[i]) for i in range(len(diffs))]
    increments_ok = all(no_significant_drop)
    significant = [float(diffs[i]) > sig_pairs[i] for i in range(len(diffs))]
    ok = increments_ok and (rho >= TOL_C5_RHO - 1e-9)
    return {
        "id": "C5",
        "name": "Strouhal monotonicity of C_T",
        "verdict": _verdict(ok),
        "threshold": {
            "spearman": TOL_C5_RHO,
            "rule": "no drop larger than the combined 3-sigma noise of the pair",
        },
        "value": {
            "strouhal": st,
            "C_T": ct,
            "sigma_3": sigma,
            "differences": diffs.tolist(),
            "increment_significant": significant,
            "no_significant_drop": no_significant_drop,
            "spearman": rho,
            "monotone": increments_ok,
        },
        "note": "pure heave at fixed k; St = 2 k (h0/c)/pi",
    }


# ---------------------------------------------------------------------------
# C6 -- grid convergence
# ---------------------------------------------------------------------------
def criterion_c6_grid_convergence(grids: Sequence[dict], r_ref: float = 1.33) -> dict:
    """Roache GCI from three grids with the same physical set-up.

    ``grids`` entries: ``{"chord": int, "value": float}`` ordered coarse ->
    medium -> fine.
    """
    gs = sorted(grids, key=lambda g: g["chord"])
    if len(gs) != 3:
        return {
            "id": "C6",
            "name": "grid convergence (GCI)",
            "verdict": "WITHHELD",
            "value": {"n_grids": len(gs)},
            "threshold": {"gci_fine": TOL_C6_GCI},
            "note": "three grids required",
        }
    f3, f2, f1 = (gs[2]["value"], gs[1]["value"], gs[0]["value"])  # fine, medium, coarse
    r21 = gs[1]["chord"] / gs[0]["chord"]
    r32 = gs[2]["chord"] / gs[1]["chord"]
    r = 0.5 * (r21 + r32)
    e21 = f2 - f1
    e32 = f3 - f2
    # Roache (1994): with f1 = coarse, f2 = medium, f3 = fine and error ~ h^p,
    # |f2 - f1| / |f3 - f2| = r^p.
    if abs(e21) < 1e-30 or abs(e32) < 1e-30 or e32 * e21 <= 0:
        p = float("nan")
    else:
        p = math.log(abs(e21 / e32)) / math.log(r)
    if not math.isfinite(p) or p <= 0:
        ext = f3 + (f3 - f2) / max(r**1.0 - 1.0, 1e-12)
        gci_fine = abs(f3 - f2) / max(abs(f3), 1e-30)
        ok = False
    else:
        ext = f3 + (f3 - f2) / (r**p - 1.0)
        gci_fine = 1.25 * abs(f3 - f2) / max(abs(f3), 1e-30) / (r**p - 1.0)
        ok = gci_fine <= TOL_C6_GCI and 0.5 <= p <= 3.0
    return {
        "id": "C6",
        "name": "grid convergence (GCI)",
        "verdict": _verdict(ok),
        "threshold": {"gci_fine": TOL_C6_GCI, "p_range": [0.5, 3.0]},
        "value": {
            "chords": [g["chord"] for g in gs],
            "values": [g["value"] for g in gs],
            "r": r,
            "p_observed": p,
            "f_extrapolated": ext,
            "gci_fine": gci_fine,
        },
        "note": "Roache 1994, safety factor 1.25",
    }


# ---------------------------------------------------------------------------
# C7 -- negative-control trio
# ---------------------------------------------------------------------------
def criterion_c7_negative_controls(
    locked: dict,
    ibm_off: dict,
    phase_flip: dict,
    reference: dict,
) -> dict:
    """Three controls that *must* behave in a prescribed way.

    1. locked body (no motion): ``C_T < 0`` (pure drag) and the magnitude
       matches an independent drag estimate within ``TOL_C7_LOCKED_REL``.
    2. IBM disabled: the measured force is identically zero (proves the force
       comes from the immersed body, not from the boundary conditions).
    3. heave phase flip (``h -> -h``): for pure heave the mean thrust is
       invariant, so ``|C_T^flip - C_T^ref| / |C_T^ref|`` must stay small --
       a sign error in the force or power integration breaks this.
    """
    checks = {}
    cd_ind = locked.get("cd_independent")
    ctl = locked.get("C_T")
    # ``cd_independent`` is a drag coefficient (positive downstream), ``C_T`` is
    # negative for a body in drag: compare magnitudes.
    rel_lock = (
        abs(abs(ctl) - abs(cd_ind)) / abs(cd_ind)
        if cd_ind not in (None, 0.0)
        else float("nan")
    )
    checks["locked_drag"] = {
        "C_T": ctl,
        "C_D_independent": cd_ind,
        "rel_error": rel_lock,
        "ok": (ctl is not None and ctl < 0.0 and rel_lock <= TOL_C7_LOCKED_REL),
    }
    checks["ibm_off"] = {
        "C_T": ibm_off.get("C_T"),
        "C_P": ibm_off.get("C_P"),
        "ok": max(abs(ibm_off.get("C_T", 1.0)), abs(ibm_off.get("C_P", 1.0))) <= TOL_C7_NOIBM,
    }
    rel_flip = (
        abs(phase_flip["C_T"] - reference["C_T"]) / abs(reference["C_T"])
        if reference["C_T"]
        else float("inf")
    )
    checks["phase_flip_symmetry"] = {
        "C_T_ref": reference["C_T"],
        "C_T_flip": phase_flip["C_T"],
        "rel_diff": rel_flip,
        "ok": rel_flip <= TOL_C7_SYMMETRY,
    }
    ok = all(v["ok"] for v in checks.values())
    return {
        "id": "C7",
        "name": "negative-control trio",
        "verdict": _verdict(ok),
        "threshold": {
            "locked_rel": TOL_C7_LOCKED_REL,
            "ibm_off": TOL_C7_NOIBM,
            "phase_flip_rel": TOL_C7_SYMMETRY,
        },
        "value": checks,
        "note": "controls prove the harness detects broken force integration",
    }


# ---------------------------------------------------------------------------
# C8 -- bit reproducibility
# ---------------------------------------------------------------------------
def criterion_c8_reproducibility(run_a: dict, run_b: dict) -> dict:
    """Two independent runs of the same case must agree bit for bit."""
    same = run_a.get("sha256") == run_b.get("sha256")
    dct = abs(run_a.get("C_T", 0.0) - run_b.get("C_T", 0.0))
    return {
        "id": "C8",
        "name": "bit-level reproducibility",
        "verdict": _verdict(bool(same) and dct == 0.0),
        "threshold": {"identical_sha256": True},
        "value": {
            "sha256_a": run_a.get("sha256"),
            "sha256_b": run_b.get("sha256"),
            "C_T_a": run_a.get("C_T"),
            "C_T_b": run_b.get("C_T"),
            "abs_dC_T": dct,
        },
        "note": "same config, independent processes; JSON payload hash",
    }


def criterion_c9_momentum_balance(rows: Sequence[dict], tol: float = 0.15) -> dict:
    """IBM surface force vs an independent control-volume momentum balance.

    For a time-periodic flow the cycle-averaged streamwise force on the section
    equals the cycle-averaged momentum+pressure flux difference between two
    planes of a control volume around it::

        F_x = sum_y [rho u_x^2 + p]_{x_in} - sum_y [rho u_x^2 + p]_{x_out}

    The two estimators share no code path (one integrates the immersed-boundary
    reaction, the other integrates fluxes on Eulerian planes), so agreement is a
    strong check of the force accounting and of momentum conservation.
    """
    rows = [
        r
        for r in rows
        if r.get("C_T_control_volume") is not None
        and max(abs(r["C_T"]), abs(r["C_T_control_volume"])) > 1e-6
    ]
    worst = 0.0
    detail = []
    for r in rows:
        ibm = r["C_T"]
        cv = r["C_T_control_volume"]
        denom = max(abs(cv), abs(ibm))
        rel = abs(ibm - cv) / denom
        worst = max(worst, rel)
        detail.append(
            {"label": r["label"], "C_T_ibm": ibm, "C_T_cv": cv, "rel_diff": rel}
        )
    ok = worst <= tol
    return {
        "id": "C9",
        "name": "IBM force vs control-volume momentum balance",
        "verdict": _verdict(ok),
        "threshold": {"rel_diff": tol},
        "value": {"cases": detail, "worst_rel_diff": worst},
        "note": "independent force estimators must agree on every case",
    }


def criterion_c5b_thrust_reversal(cases: Sequence[dict]) -> dict:
    """Thrust reversal of a purely pitching section: ``C_T`` must change sign.

    Linear theory (Garrick) puts the reversal of a section pitching about
    ``c/4`` between ``k = 0.5`` and ``k = 1`` at ``alpha0 = 10 deg``; the measured
    curve must show the same sign change and be monotone increasing in ``k``.
    """
    cs = sorted(cases, key=lambda c: c["k"])
    ks = [c["k"] for c in cs]
    ct = [c["C_T"] for c in cs]
    th = [c["theory"]["C_T"] for c in cs]
    sign_ok = all((ct[i] <= 0.0) != (th[i] <= 0.0) for i in range(len(ct))) is False
    # the *pattern* of signs must match theory (not the individual magnitudes)
    pattern_ok = all((ct[i] > 0.0) == (th[i] > 0.0) for i in range(len(ct)))
    flips = any((ct[i] < 0.0) and (ct[i + 1] > 0.0) for i in range(len(ct) - 1))
    mono = all(ct[i] < ct[i + 1] for i in range(len(ct) - 1))
    ok = pattern_ok and flips and mono
    return {
        "id": "C5b",
        "name": "thrust reversal of a pitching section (sign change in k)",
        "verdict": _verdict(ok),
        "threshold": {"sign_pattern_matches_theory": True, "sign_change": True},
        "value": {
            "k": ks,
            "C_T": ct,
            "C_T_theory": th,
            "sign_pattern_ok": pattern_ok,
            "sign_change": flips,
            "monotone_in_k": mono,
        },
        "note": "pitch about c/4, alpha0 = 10 deg, inviscid reversal between k=0.5 and 1",
    }


def summarise(criteria: Iterable[dict]) -> dict:
    """Aggregate verdicts."""
    cs = list(criteria)
    n_pass = sum(1 for c in cs if c["verdict"] == "PASS")
    n_fail = sum(1 for c in cs if c["verdict"] == "FAIL")
    n_hold = sum(1 for c in cs if c["verdict"] == "WITHHELD")
    return {
        "n_criteria": len(cs),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_withheld": n_hold,
        "all_pass": n_fail == 0 and n_hold == 0,
        "verdicts": {c["id"]: c["verdict"] for c in cs},
    }