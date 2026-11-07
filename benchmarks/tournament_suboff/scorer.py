"""Version-locked physical-anchor scorer for the SUBOFF tournament.

Scoring contract (frozen in prereg, md5-receipted):

* ``err = |Cd / Cf_ref - 1|``  (relative error vs the physical anchor)
* ``score = 100 * max(0, 1 - err / 0.15)``
  — 100 at zero error, 0 at 15% error (the platform's unacceptable-error
    threshold), linear in between.
* ``PASS`` iff ``err <= 0.03`` (3%, the verified-benchmark gate G1 level).
* Non-finite, negative, or zero drag -> ``status='invalid'``, score 0.
* Diverged / NaN / OOM seats -> ``status='diverged'`` (or 'error'), score 0.

The scorer is a pure function of (Cd, Cf_ref, status): no model review, no
subjective terms.  Red-team self-tests (known-solution cases) MUST pass
before any seat is scored; the scorer version string is stamped into every
result record for audit.
"""

from __future__ import annotations

import math

from . import SCORER_VERSION
from .anchors import anchor_for

PASS_ERR = 0.03
ZERO_SCORE_ERR = 0.15


def score_seat(cd: float, re: float, status: str = "ok") -> dict:
    """Score one seat.  Returns a dict with all audit fields.

    Parameters
    ----------
    cd : total drag coefficient (wetted-area normalised)
    re : Reynolds number of the seat (selects the anchor)
    status : 'ok' | 'diverged' | 'error' | 'control'
    """
    out: dict = {
        "scorer_version": SCORER_VERSION,
        "re": re,
        "status": status,
        "cd": cd,
    }
    if status in ("diverged", "error"):
        out.update(
            anchor_name=None,
            cf_ref=None,
            err_pct=None,
            score=0.0,
            passed=False,
        )
        return out
    if status == "control":
        # documented negative control (e.g. degenerate force phase):
        # recorded, never a champion candidate
        anchor_name, cf_ref = anchor_for(re)
        out.update(
            anchor_name=anchor_name,
            cf_ref=cf_ref,
            err_pct=None,
            score=0.0,
            passed=False,
            control=True,
        )
        return out
    if not (isinstance(cd, (int, float)) and math.isfinite(cd)):
        out.update(anchor_name=None, cf_ref=None, err_pct=None, score=0.0, passed=False,
                   invalid="non-finite Cd")
        return out
    if cd <= 0.0:
        out.update(anchor_name=None, cf_ref=None, err_pct=None, score=0.0, passed=False,
                   invalid="non-positive Cd (negative-resistance pathology)")
        return out
    anchor_name, cf_ref = anchor_for(re)
    err = abs(cd / cf_ref - 1.0)
    score = 100.0 * max(0.0, 1.0 - err / ZERO_SCORE_ERR)
    out.update(
        anchor_name=anchor_name,
        cf_ref=cf_ref,
        err_pct=100.0 * err,
        score=score,
        passed=err <= PASS_ERR,
    )
    return out


def red_team_self_test() -> dict:
    """Known-solution red-team checks for the scorer (T5 discipline)."""
    c: dict[str, bool] = {}
    # (a) exact anchor -> score 100, PASS
    a = score_seat(0.041995047327036086, 1000.0)
    c["exact_anchor_score100"] = a["score"] == 100.0 and a["passed"]
    # (b) 15% error -> score 0
    b = score_seat(0.041995047327036086 * 1.15, 1000.0)
    c["err15pct_score0"] = abs(b["score"]) < 1e-12 and not b["passed"]
    # (c) 3% boundary -> PASS; 3.01% -> FAIL
    c["pass_boundary"] = score_seat(0.041995047327036086 * 1.03, 1000.0)["passed"]
    c["fail_just_above"] = not score_seat(0.041995047327036086 * 1.0301, 1000.0)["passed"]
    # (d) negative drag -> invalid, score 0
    c["negative_cd"] = score_seat(-0.01, 1000.0)["score"] == 0.0
    # (e) NaN -> invalid
    c["nan_cd"] = score_seat(float("nan"), 1000.0)["score"] == 0.0
    # (f) diverged -> score 0
    c["diverged"] = score_seat(0.04, 1000.0, status="diverged")["score"] == 0.0
    # (g) monotone: larger error -> smaller score
    errs = [0.0, 0.01, 0.03, 0.06, 0.10, 0.149]
    sc = [score_seat(0.041995047327036086 * (1 + e), 1000.0)["score"] for e in errs]
    c["monotone_decreasing"] = all(x > y for x, y in zip(sc, sc[1:]))
    # (h) ITTC regime selected at Re=1e5
    c["ittc_regime"] = score_seat(0.008333333333333333, 1.0e5)["anchor_name"] == "ittc1957"
    # (i) 20% error (worse than platform threshold) -> score 0
    c["err20pct_score0"] = score_seat(0.041995047327036086 * 1.20, 1000.0)["score"] == 0.0
    return c


def all_red_team_pass() -> bool:
    return all(red_team_self_test().values())


if __name__ == "__main__":
    import json

    print(json.dumps(red_team_self_test(), indent=1))
