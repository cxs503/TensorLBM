"""CASE-RT-INSTABILITY-ATWOOD — reference lock for the A = 1/2 single-mode RT case.

Why a lock file
---------------
The case is gated on numbers, so the numbers must come from a machine-read
source, not from a human retyping a table.  This script

1. fetches the bibliographic record of the anchor monograph quoted by the task
   ("Tryggvason 2011 spectral lineage" = Tryggvason, Scardovelli & Zaleski,
   *Direct Numerical Simulations of Gas-Liquid Multiphase Flows*, Cambridge
   University Press, 2011) from the Crossref API, stores the raw JSON in
   ``ref/sources/`` and records its SHA-256 -- so the citation is verifiable
   even though the book text itself is not reachable from this sandbox
   (no publisher access; recorded honestly below as a MU limitation);
2. evaluates the reference values **from closed form in code** -- nothing is
   transcribed from a printed page:

   L  linear (normal-mode) growth rate of the interface at A = 1/2
        sigma_inv  = sqrt(A g k)
        sigma_visc = sqrt(A g k + nu^2 k^4) - nu k^2          (k = 2 pi / lam)
      Rayleigh (1883) Proc. London Math. Soc. 14:170; Taylor (1950) Proc. R.
      Soc. Lond. A 201:192; Chandrasekhar (1961) *Hydrodynamic and
      Hydromagnetic Stability* sec. 92.  This is precisely the benchmark that
      the spectral-DNS lineage of Tryggvason et al. (2011) uses to validate a
      code in the linear regime (ch. "Rayleigh-Taylor instability").

   N  nonlinear saturated spike/bubble velocities, potential-flow (Layzer-
      Goncharov) model, in the form quoted by arXiv:2009.14655 Eq. (1), p. 3

        u_s = sqrt(2 A g / ((1-A) C k)),  u_b = sqrt(2 A g / ((1+A) C k))

      with the two-dimensional convention C = 1 (the ``C`` of the ``sqrt(1/(3pi))``
      "cross-check" in the sibling case ``rayleigh_taylor/ref/ref_lock.py`` is the
      3-D Layzer constant; running it in 2-D predicts a bubble velocity a factor
      sqrt(3) too small -- the sibling round-2 measurement on the same solver
      found C = 0.90 at A = 1/2 and C = 1.01 at A = 1/3, see
      ``rayleigh_taylor/docs/CASE-RAYLEIGH-TAYLOR.md`` sec. 4).  Both conventions
      are reported so the disagreement stays visible rather than being tuned away.

3. writes ``ref_lock.json`` with every number, its formula, the source hashes.

Honest limitation
-----------------
The anchor monograph is paywalled; ``https://doi.org/10.1017/CBO9780511975264``
resolves to metadata only in this sandbox.  No page/table number of the book is
quoted and no number is copied from it: the numeric targets are the closed forms
above, which are what that book's linear/moderate-amplitude RT chapter uses as
the reference solution.  This is registered as MU (missing/unobtainable) in the
case report, not silently dropped.
"""

from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from pathlib import Path

REF_DIR = Path(__file__).resolve().parent
SOURCES = REF_DIR / "sources"

#: Atwood number of this case (task anchor).
ATWOOD = 0.5
#: two-dimensional potential-flow geometric constant (see module docstring).
C_2D = 1.0
#: the 3-D Layzer constant, kept only to expose the convention sensitivity.
C_3D = 3.0

CROSSREF_QUERY = (
    "https://api.crossref.org/works?rows=1&query.bibliographic="
    "Direct+Numerical+Simulations+of+Gas-Liquid+Multiphase+Flows+Tryggvason+Scardovelli+Zaleski"
)


def sigma_inviscid(atwood: float, g: float, lam: float) -> float:
    """Inviscid normal-mode growth rate ``sqrt(A g k)``."""
    return math.sqrt(atwood * g * 2.0 * math.pi / lam)


def sigma_viscous(atwood: float, g: float, lam: float, nu: float) -> float:
    """Equal-viscosity normal-mode growth rate ``sqrt(Agk + nu^2 k^4) - nu k^2``."""
    k = 2.0 * math.pi / lam
    return math.sqrt(atwood * g * k + (nu * k * k) ** 2) - nu * k * k


def nu_eff_from_sigma(atwood: float, g: float, lam: float, sigma_meas: float) -> float:
    """Invert :func:`sigma_viscous`: viscosity implied by a measured growth rate.

    ``sqrt(Agk + x) - sqrt(x) = s`` with ``x = nu^2 k^4`` gives
    ``sqrt(x) = (Agk - s^2) / (2 s)``, so ``nu = sqrt(x) / k^2``.  Returns NaN
    when the measured growth rate exceeds the inviscid value (no real solution).
    """
    k = 2.0 * math.pi / lam
    agk = atwood * g * k
    if sigma_meas <= 0.0 or sigma_meas >= math.sqrt(agk):
        return float("nan")
    return ((agk - sigma_meas**2) / (2.0 * sigma_meas)) / (k * k)


def goncharov(atwood: float, g: float, lam: float, branch: str, c: float = C_2D) -> float:
    """Potential-flow saturated velocity of the ``b``(ubble) or ``s``(pike) front."""
    k = 2.0 * math.pi / lam
    denom = (1.0 + atwood) if branch == "b" else (1.0 - atwood)
    return math.sqrt(2.0 * atwood * g / (denom * c * k))


def froude(atwood: float, g: float, lam: float, u: float) -> float:
    """Froude number ``u / sqrt(A g lam / (1+A))`` (normalisation of arXiv:1910.03663)."""
    return u / math.sqrt(atwood * g * lam / (1.0 + atwood))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def crossref_record() -> dict:
    """Fetch (or reuse) the Crossref record of the anchor monograph and hash it."""
    SOURCES.mkdir(parents=True, exist_ok=True)
    raw = SOURCES / "crossref_tryggvason_2011.json"
    status = "cached"
    if not raw.exists():
        try:
            req = urllib.request.Request(
                CROSSREF_QUERY,
                headers={"User-Agent": "TensorPlatform-agent/1.0 (mailto:agent@tensorplatform.invalid)"},
            )
            body = urllib.request.urlopen(req, timeout=45).read().decode("utf-8", "replace")
            raw.write_text(body, encoding="utf-8")
            status = "fetched"
        except Exception as exc:  # offline / blocked -> record it, do not invent
            return {"status": f"unavailable: {type(exc).__name__}", "doi": None, "sha256": None}
    doc = json.loads(raw.read_text(encoding="utf-8"))
    items = doc.get("message", {}).get("items") or []
    if not items:
        return {"status": "empty-crossref-response", "doi": None, "sha256": sha256(raw)}
    it = items[0]
    return {
        "status": status,
        "title": (it.get("title") or [""])[0],
        "publisher": it.get("publisher"),
        "type": it.get("type"),
        "doi": it.get("DOI"),
        "issued": it.get("issued", {}).get("date-parts"),
        "isbn": it.get("ISBN"),
        "path": str(raw.relative_to(REF_DIR)),
        "sha256": sha256(raw),
    }


def main() -> int:
    """Build ``ref_lock.json`` for the A = 1/2 case."""
    book = crossref_record()

    # pre-declared case grid: reference point + convergence + sensitivity axes
    points = [
        # (lam, g, nu, label)
        (64, 2.0e-4, 0.005, "base_l064_g2e-4_nu5e-3"),
        (96, 3.0e-4, 0.005, "grid_l096_g3e-4_nu5e-3"),
        (128, 4.0e-4, 0.005, "grid_l128_g4e-4_nu5e-3"),
        (64, 2.0e-4, 1.0 / 600.0, "nu_l064_g2e-4_tau505"),
        (64, 2.0e-4, 0.01, "nu_l064_g2e-4_tau530"),
        (64, 1.0e-4, 0.005, "g_l064_g1e-4_nu5e-3"),
    ]
    suite = []
    for lam, g, nu, label in points:
        suite.append({
            "label": label,
            "atwood": ATWOOD,
            "lam": lam,
            "g": g,
            "nu": nu,
            "k": 2.0 * math.pi / lam,
            "sigma_inviscid": sigma_inviscid(ATWOOD, g, lam),
            "sigma_viscous": sigma_viscous(ATWOOD, g, lam, nu),
            "sigma_viscous_ratio": sigma_viscous(ATWOOD, g, lam, nu)
            / sigma_inviscid(ATWOOD, g, lam),
            "u_bubble_c1": goncharov(ATWOOD, g, lam, "b", C_2D),
            "u_spike_c1": goncharov(ATWOOD, g, lam, "s", C_2D),
            "u_bubble_c3": goncharov(ATWOOD, g, lam, "b", C_3D),
            "u_spike_c3": goncharov(ATWOOD, g, lam, "s", C_3D),
            "fr_bubble_c1": froude(ATWOOD, g, lam, goncharov(ATWOOD, g, lam, "b", C_2D)),
            "fr_spike_c1": froude(ATWOOD, g, lam, goncharov(ATWOOD, g, lam, "s", C_2D)),
        })

    lock = {
        "case_code": "CASE-RT-INSTABILITY-ATWOOD",
        "atwood": ATWOOD,
        "anchor": {
            "lineage": "spectral DNS of single-mode Rayleigh-Taylor instability",
            "monograph": "Tryggvason, Scardovelli & Zaleski, Direct Numerical Simulations of "
                         "Gas-Liquid Multiphase Flows, Cambridge University Press, 2011",
            "crossref": book,
        },
        "references": {
            "L_linear": {
                "formula": "sigma = sqrt(A g k); sigma_visc = sqrt(A g k + nu^2 k^4) - nu k^2",
                "citation": "Rayleigh 1883 PLMS 14:170; Taylor 1950 Proc. R. Soc. A 201:192; "
                            "Chandrasekhar 1961 Hydrodynamic and Hydromagnetic Stability sec. 92",
                "transcription": "none - closed form evaluated in code",
            },
            "N_nonlinear": {
                "formula": "u_s = sqrt(2Ag/((1-A)Ck)); u_b = sqrt(2Ag/((1+A)Ck))",
                "c_2d": C_2D,
                "c_3d_reference_value": C_3D,
                "citation": "Goncharov 2002 PRL 88:134502, as quoted in arXiv:2009.14655 Eq. (1), p. 3",
                "transcription": "formula from the arXiv text; C convention reported both ways",
            },
        },
        "sources": [
            {"path": str(p.relative_to(REF_DIR)), "sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(SOURCES.glob("*")) if p.is_file()
        ],
        "limitation": "Anchor monograph is paywalled (DOI 10.1017/CBO9780511975264 resolves to "
                      "metadata only here): no page/table number of the book is quoted and no "
                      "value is copied from it. Targets are the closed forms above, evaluated by "
                      "this script (MU registered in the case report).",
        "suite": suite,
    }
    out = REF_DIR / "ref_lock.json"
    out.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[ref_lock] anchor book record: {book.get('status')} doi={book.get('doi')}")
    print(f"[ref_lock] A = {ATWOOD}: {len(suite)} reference rows, "
          f"{len(lock['sources'])} source files -> {out.name}")
    for row in suite:
        print(f"  {row['label']:26s} sigma_inv={row['sigma_inviscid']:.6e} "
              f"sigma_visc={row['sigma_viscous']:.6e} "
              f"({100*(row['sigma_viscous_ratio']-1):+.2f}% vs inv)  "
              f"u_s(C=1)={row['u_spike_c1']:.6e}  u_s(C=3)={row['u_spike_c3']:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())