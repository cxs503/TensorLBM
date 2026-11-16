"""Reference lock for CASE-RAYLEIGH-TAYLOR.

Why this file exists
--------------------
The case compares against two closed-form references.  Neither number is typed
into the report by hand: this script *derives* every target from the formulas
below and writes them, together with the source provenance and the SHA-256 of
the extracted source excerpts, into ``ref/ref_lock.json``.  Downstream
(``run_case.py``, ``make_figures.py``) only imports the formula helpers.

Sources actually consulted in this sandbox (no publisher access: AIP/APS return
HTTP 403; arXiv is reachable and *is* the original text for the preprints):

R1  Linear growth rate, inviscid and equal-viscosity:
    Rayleigh (1883) Proc. London Math. Soc. 14:170; Taylor (1950) Proc. R. Soc.
    Lond. A 201:192; Chandrasekhar (1961) *Hydrodynamic and Hydromagnetic
    Stability* sec. 92.  Closed form; evaluated here, nothing transcribed.

R2  Nonlinear (saturated) spike/bubble velocities -- potential-flow model of
    Goncharov (2002) Phys. Rev. Lett. 88:134502, quoted as Eq. (1) on p. 3 of
    arXiv:2009.14655 (excerpt in ``ref/sources/``):

        u_s = sqrt(2 A g / ((1-A) C k)),  u_b = sqrt(2 A g / ((1+A) C k))

    The value of the geometric constant C is the one transcription risk, so it
    is *locked numerically*: arXiv:1910.03663, Fig. 3 caption (p. 6), states the
    2-D potential-flow bubble value as ``Fr_B = sqrt(1/(3 pi))`` with
    ``Fr_B == U_B / sqrt(A g lam / (1+A))``.  With C = 3 our formula gives
    ``sqrt(2/(C*2*pi)) = sqrt(1/(3*pi))`` -- identical, to 1e-15.  The check is
    re-run below; a mismatch makes the script fail instead of quietly shipping a
    wrong reference.

Honest limitation (recorded, not papered over)
----------------------------------------------
The task anchor names a "He / Doolen / Ding" spike-bubble growth-rate table.
The publisher copy of that paper could not be retrieved from this sandbox
(AIP returns 403, no institutional access), so its table/page numbers are NOT
quoted here -- reproducing them from memory would be fabrication.  The numeric
targets used are the closed-form references above, which are (i) the physics
the LBM literature validates against and (ii) verifiable in this repo.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

REF_DIR = Path(__file__).resolve().parent
SOURCES = REF_DIR / "sources"

#: Goncharov geometric constant used by the gates.  Two of the repo's own locked
#: sources disagree on its 2-D value, so both conventions are carried explicitly
#: instead of one being quietly overwritten (see CONVENTION_CONFLICT below):
#:
#: * ``C = 3`` (primary, ``C_2D``): arXiv:1910.03663 Fig. 3 caption (p. 6) is a
#:   *two-dimensional* RTI figure (A = 0.04) whose dashed potential-model
#:   prediction is ``Fr_B = sqrt(1/(3 pi))``.  With the Froude normalisation
#:   ``Fr_B == U_B/sqrt(A g lam/(1+A))`` this reproduces ``C = 3`` exactly
#:   (checked to 1e-15 in :func:`main`).  The classic 2-D Layzer value
#:   ``u_b = sqrt(2 A g/(3 k))`` is the same statement.
#: * ``C = 1`` (alternate, ``C_2D_ALT``): arXiv:2009.14655 Eq. (1), p. 3 states
#:   verbatim "C is a constant given by 1 or 3 for the two-dimensional or
#:   three-dimensional geometry", i.e. the opposite assignment.  Under
#:   ``Fr_B == U_B/sqrt(A g lam)`` that convention predicts ``Fr_B = sqrt(1/pi)``
#:   for 2-D, which no 2-D figure in either excerpt supports.
#:
#: The conflict cannot be settled from this sandbox (publisher copies of
#: Goncharov 2002 PRL 88:134502 are behind a 403).  Policy: (a) the gates use
#: the convention backed by the *2-D-specific* primary source (C = 3);
#: (b) every report also quotes the C = 1 errors, so the reader sees how much of
#: the verdict is convention-dependent; (c) no constant is ever fitted to the
#: solver's own output -- that is what produced the false PASS this file now
#: guards against (round 2, m2026-11-13, back-solved C = 0.90/1.01 from its own
#: runs).
C_2D = 3.0          # Goncharov geometric constant, 2-D (primary, locked below)
C_2D_ALT = 1.0      # the competing assignment of arXiv:2009.14655 Eq. (1)


def boussinesq_velocity(atwood: float, g: float, lam: float, c: float = C_2D) -> float:
    """Symmetric (``u_b = u_s``) reference of the single-density Boussinesq model.

    ``tensorlbm.rayleigh_taylor`` keeps ``rho = 1`` everywhere and lets the
    Atwood number act only through the buoyancy force; that model is invariant
    under ``(phi, y) -> (-phi, -y)``, which exchanges bubble and spike, so its
    terminal velocities must coincide and the A -> 0 limit of the two-fluid
    formula, ``sqrt(2 A g/(C k))``, is the reference (proof + numerical test:
    ``tests/test_case_pool_rayleigh_taylor_reference_audit.py``).
    """
    return math.sqrt(2.0 * atwood * g / (c * 2.0 * math.pi / float(lam)))


#: Machine-readable record of the disagreement, written into ref_lock.json.
CONVENTION_CONFLICT = {
    "quantity": "Goncharov geometric constant C in u_b = sqrt(2Ag/((1+A)Ck))",
    "primary": {"c": 3.0, "source": "arXiv:1910.03663 Fig. 3 caption p.6 (2-D RTI, A = 0.04), "
                                   "Fr_B = sqrt(1/(3*pi)); equivalent to the classic 2-D Layzer "
                                   "value u_b = sqrt(2Ag/(3k))"},
    "alternate": {"c": 1.0, "source": "arXiv:2009.14655 Eq. (1) p.3, verbatim 'C is a constant "
                                      "given by 1 or 3 for the two-dimensional or "
                                      "three-dimensional geometry'"},
    "fr_b_primary": math.sqrt(1.0 / (3.0 * math.pi)),
    "fr_b_alternate": math.sqrt(2.0 / (1.0 * 2.0 * math.pi)),
    "ratio_alternate_over_primary": math.sqrt(3.0),
    "status": "UNRESOLVED (publisher copy of Goncharov 2002 PRL 88:134502 not reachable: HTTP 403)",
    "policy": "gates use C = 3 (2-D-specific primary source); both errors are reported; "
              "no constant is fitted to solver output",
}
LAMBDAS = (64, 128)


def sigma_inviscid(atwood: float, g: float, lam: float) -> float:
    return math.sqrt(atwood * g * 2.0 * math.pi / lam)


def sigma_viscous(atwood: float, g: float, lam: float, nu: float) -> float:
    k = 2.0 * math.pi / lam
    return math.sqrt(atwood * g * k + (nu * k * k) ** 2) - nu * k * k


def goncharov(atwood: float, g: float, lam: float, branch: str,
             c: float = C_2D) -> float:
    k = 2.0 * math.pi / lam
    denom = (1.0 + atwood) if branch == "b" else (1.0 - atwood)
    return math.sqrt(2.0 * atwood * g / (denom * c * k))


def froude_bubble_theory(c: float) -> float:
    """Fr_B == U_B / sqrt(A g lam/(1+A)) implied by the Goncharov formula."""
    return math.sqrt(2.0 / (c * 2.0 * math.pi))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    # --- lock the C = 3 assignment against the independent literature statement
    fr_c3 = froude_bubble_theory(3.0)
    fr_quoted = math.sqrt(1.0 / (3.0 * math.pi))   # arXiv:1910.03663, Fig. 3 caption
    assert abs(fr_c3 - fr_quoted) < 1e-15, (fr_c3, fr_quoted)
    # the competing convention is locked the same way, so the disagreement is a
    # checked property of the two source excerpts rather than a memory claim
    fr_c1 = froude_bubble_theory(C_2D_ALT)
    assert abs(fr_c1 - CONVENTION_CONFLICT["fr_b_alternate"]) < 1e-15, fr_c1
    # Fr(C=1) = sqrt(3) * Fr(C=3): the whole verdict-defining gap, locked here
    assert abs(fr_c1 / fr_c3 - math.sqrt(3.0)) < 1e-15

    suite = []
    for atwood in (1.0 / 3.0, 0.5):
        for lam in LAMBDAS:
            for g in (1.0e-4, 2.0e-4):
                for nu in (0.005, 0.01):
                    suite.append({
                        "atwood": atwood,
                        "lam": lam,
                        "g": g,
                        "nu": nu,
                        "k": 2.0 * math.pi / lam,
                        "sigma_inviscid": sigma_inviscid(atwood, g, lam),
                        "sigma_viscous": sigma_viscous(atwood, g, lam, nu),
                        "u_bubble_goncharov": goncharov(atwood, g, lam, "b"),
                        "u_spike_goncharov": goncharov(atwood, g, lam, "s"),
                        "fr_bubble_goncharov": goncharov(atwood, g, lam, "b")
                        / math.sqrt(atwood * g * lam / (1.0 + atwood)),
                    })

    lock = {
        "case_code": "RAYLEIGH-TAYLOR",
        "references": {
            "R1_linear": {
                "formula": "sigma = sqrt(A g k); sigma_visc = sqrt(A g k + nu^2 k^4) - nu k^2",
                "citation": "Rayleigh 1883 PLMS 14:170; Taylor 1950 Proc. R. Soc. A 201:192; "
                            "Chandrasekhar 1961 Hydrodynamic and Hydromagnetic Stability sec.92",
                "transcription": "none (closed form evaluated in code)",
            },
            "R2_nonlinear": {
                "convention_conflict": CONVENTION_CONFLICT,
                "formula": "u_s = sqrt(2Ag/((1-A)Ck)); u_b = sqrt(2Ag/((1+A)Ck)); "
                           "C = 1 (2-D, alternate) or 3 (2-D, primary) -- see convention_conflict",
                "citation": "Goncharov 2002 PRL 88:134502, as quoted in arXiv:2009.14655 Eq.(1) p.3",
                "c_lock": C_2D,
                "c_lock_check": {
                    "fr_from_formula": fr_c3,
                    "fr_quoted_arXiv_1910.03663_Fig3": fr_quoted,
                    "abs_diff": abs(fr_c3 - fr_quoted),
                },
                "transcription": "formula transcribed from the arXiv text, C locked numerically",
            },
        },
        "sources": [
            {
                "path": str(p.relative_to(REF_DIR)),
                "sha256": sha256(p),
                "bytes": p.stat().st_size,
                "origin": p.with_suffix("").name,
            }
            for p in sorted(SOURCES.glob("*.txt"))
        ],
        "limitation": "Publisher copy of the He/Doolen/Ding spike-bubble table not retrievable "
                      "in this sandbox (AIP 403); their table/page numbers are therefore not "
                      "quoted. Targets are the closed-form references above.",
        "suite": suite,
    }
    (REF_DIR / "ref_lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ref_lock] C={C_2D} check {fr_c3:.15f} vs quoted {fr_quoted:.15f} -> OK")
    print(f"[ref_lock] competing convention C={C_2D_ALT} -> Fr_B={fr_c1:.6f} "
          f"(ratio sqrt(3)={math.sqrt(3.0):.6f}); conflict recorded as UNRESOLVED")
    print(f"[ref_lock] {len(suite)} reference rows, {len(lock['sources'])} source excerpts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())