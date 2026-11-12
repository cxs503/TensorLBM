#!/usr/bin/env python3
"""Locked reference convention for CASE-LBM-SQUARE-CYL-NORBERG (square cylinder, Re=100).

Why this file exists
====================
Audit `m16183` (see ``audit_m16183.md`` at the repo root) found that the case
archive carried **two different reference constants** for the same physical
measurement::

    run.py            REF_CD = 1.48     REF_ST = 0.147
    verify.py         REF_CD = 1.4785   REF_ST = 0.1456
    verify_traces.py  REF_CD = 1.4785   REF_ST = 0.1456

so the *same* measured ``Cd``/``St`` produced two different error percentages
depending on which script was asked, and a "PASS" was therefore not a
well-defined statement.  The same audit also locked the force-column
definition issue: the ``wet`` column differs from ``pair``/``mem`` by +8.5 % in
``Cd``, which is larger than the 3 % acceptance bar, so the acceptance metric
must name the column it is scored on.

This module is the single source of truth.  ``run.py`` (producer),
``verify.py`` and ``verify_traces.py`` (independent verifiers) all import from
here; nothing else in the case directory may define ``REF_CD``/``REF_ST``.
``tests/test_square_cylinder_case.py`` enforces that structurally.

Reference basis (why 2D numerics and not Okajima's experiment)
--------------------------------------------------------------
The anchor is named after Norberg, but Norberg's square-cylinder St series is
**experimental** (3D wake).  A 2D solver must not be scored against 3D
experiment; the case therefore scores against the low-blockage 2D free-stream
numerical cluster, cross-checked over 14 table entries in
``REFERENCE_AUDIT.md``:

    Cd : 1.44 - 1.52, median 1.488, mean 1.485  -> centre  1.48
    St : 0.145 - 0.149 (0.159 is an outlier)    -> centre  0.147

Sources: arXiv 2411.03124v4 Tab.1, 2309.09197 Tab.2, 2308.08085v3 Tab.7/8,
2404.12123, 2405.12834.  Explicitly *not* applicable: Okajima 1982 experiment
(St 0.143-0.145, no Cd), Sohankar 1997 channel (high blockage, Cd~2.05),
Sharma & Eswaran 2004 (small domain, Cd=1.57).
"""

from __future__ import annotations

import json
from pathlib import Path

# --- locked reference values (the ONLY definition in the repository) ---------
REF_CD = 1.48
REF_ST = 0.147
CD_BAND = (1.44, 1.52)
ST_BAND = (0.145, 0.149)

# The superseded constants that v1 of verify.py/verify_traces.py used.  They are
# kept here for ONE purpose: to identify the provenance of stored error columns
# already written into the archive (case_D64.json, 2026-09-29), so that legacy
# numbers can be *declared* rather than silently rewritten.  Nothing is scored
# against them.
SUPERSEDED_REF = {"cd": 1.4785, "st": 0.1456}

# Acceptance bar for the main gate, in percent.
TOL_PCT = 3.0
# Stored err_pct values must reproduce the recomputed ones to this many
# percentage points (guards against silent reference-convention drift).
STORED_ERR_AGREEMENT_PP = 0.01

# --- locked force-column policy --------------------------------------------
# Ladd momentum-exchange columns.  ``pair`` and ``mem`` close to 0.0 % of each
# other (they are the same estimator sampled at different points of the step);
# ``wet`` is a diagnostic variant that is +8.5 % high in Cd at D=64 and would
# on its own FAIL the 3 % bar.  The acceptance metric is therefore scored on
# ``pair`` when the archive carries the triple-column schema, and on the single
# recorded summary value when it carries the summary schema.
PRIMARY_COLUMN = "pair"
SECONDARY_COLUMN = "mem"
DIAGNOSTIC_COLUMNS = ("wet",)
CLOSURE_COLUMN_PAIR_PCT = 0.5  # |pair-mem| closure tolerance, percent

# --- case geometry (shared by producer and verifiers) ----------------------
DOMAIN_D = 40.0  # square domain side in cylinder diameters -> blockage 2.5 %
SPONGE_D = 10.0  # downstream sponge width in diameters
ST_SEED = 0.14  # inlet seeding Strouhal (trigger only, does not lock the mode)

# Replaced 2026-09-29 archive values (summary schema, mem-equivalent column).
# After a fresh re-run the new numbers must reproduce these to
# REPRO_TOL_PCT before they may replace them.
OLD_ARCHIVE: dict[int, tuple[float, float]] = {
    32: (1.4816, 0.1482),
    48: (1.4844, 0.1480),
}
REPRO_TOL_PCT = 0.5

REPRO_CMD = (
    "cd benchmarks/verified/square_cylinder && "
    "python run.py --D 32 48 --device cpu --compile-mode eager --out runs/sq_local"
)


def convention_for(
    cd: float,
    st: float,
    stored_cd_pct: float | None,
    stored_st_pct: float | None,
) -> str | None:
    """Name the reference convention a stored error pair was computed under.

    Returns ``"locked"`` (this module's ``REF_CD``/``REF_ST``), ``"superseded"``
    (``SUPERSEDED_REF``, what v1 of ``verify.py`` used), or ``None`` when
    neither reproduces the stored pair within ``STORED_ERR_AGREEMENT_PP``.
    Callers must treat ``None`` as a hard failure: an error percentage whose
    convention cannot be named is not auditable.
    """

    def matches(ref_cd: float, ref_st: float) -> bool:
        ok = True
        if stored_cd_pct is not None:
            ok = ok and abs((cd - ref_cd) / ref_cd * 100.0 - stored_cd_pct) <= STORED_ERR_AGREEMENT_PP
        if stored_st_pct is not None:
            ok = ok and abs((st - ref_st) / ref_st * 100.0 - stored_st_pct) <= STORED_ERR_AGREEMENT_PP
        return ok

    if stored_cd_pct is None and stored_st_pct is None:
        return None
    if matches(REF_CD, REF_ST):
        return "locked"
    if matches(SUPERSEDED_REF["cd"], SUPERSEDED_REF["st"]):
        return "superseded"
    return None


def rel_err_pct(got: float, ref: float) -> float:
    """Signed relative error in percent (``ref`` must be non-zero)."""
    return (got - ref) / ref * 100.0


def within_band(value: float, band: tuple[float, float]) -> bool:
    lo, hi = band
    return lo <= value <= hi


def normalize_case(path: str | Path) -> dict:
    """Load one ``case_D*.json`` and flatten both archive schemas.

    Two schemas legitimately exist in this case directory:

    ``triple``
        ``force_columns`` + ``wet``/``pair``/``mem`` sub-dicts (D=64 archive).
    ``summary``
        a single ``cd``/``st`` pair plus ``cd_ref``/``st_ref`` (D=32/D=48
        archives produced by ``run.py``).

    Audit m16183 found that ``verify.py`` crashed on the ``summary`` schema
    (``IndexError`` on ``cases[ds[-1]]["pair"]``), which made the whole
    "verified" claim non-machine-checkable on the current tree.  Every consumer
    must go through this function instead of indexing raw JSON.
    """
    d = json.loads(Path(path).read_text())
    out: dict = {
        "path": str(path),
        "D": int(d["D"]),
        "nx": d.get("nx"),
        "ny": d.get("ny"),
        "steps": d.get("steps"),
        "raw": d,
    }
    if "force_columns" in d:
        out["schema"] = "triple"
        fc = d["force_columns"]
        out["n_solid_cells"] = fc.get("n_solid_cells")
        # two spellings exist in the archives: surface_links / n_links_wet
        out["surface_links"] = fc.get("surface_links", fc.get("n_links_wet"))
        out["columns"] = {k: {"cd": d[k]["cd"], "st": d[k]["st"]} for k in ("wet", "pair", "mem") if k in d}
        out["primary"] = PRIMARY_COLUMN
        out["primary_cd"] = out["columns"][PRIMARY_COLUMN]["cd"]
        out["primary_st"] = out["columns"][PRIMARY_COLUMN]["st"]
        out["closure_pair_vs_mem_pct"] = d.get("closure_pair_vs_mem_cd_pct")
        out["stored_err_pct"] = {
            "cd": (d.get(PRIMARY_COLUMN) or {}).get("err_cd_pct"),
            "st": (d.get(PRIMARY_COLUMN) or {}).get("err_st_pct"),
        }
        out["ab_wet_vs_mem_pct"] = (d.get("ab_diff") or {}).get("cd_pct_of_mem")
    elif "cd" in d:
        out["schema"] = "summary"
        out["columns"] = {"summary": {"cd": d["cd"], "st": d.get("st")}}
        out["primary"] = "summary"
        out["primary_cd"] = d["cd"]
        out["primary_st"] = d.get("st")
        out["stored_ref"] = {"cd": d.get("cd_ref"), "st": d.get("st_ref")}
        out["stored_err_pct"] = {"cd": d.get("err_pct"), "st": d.get("st_err_pct")}
    else:
        raise ValueError(f"{path}: unrecognised case schema (no force_columns, no cd)")
    return out


def load_archive(stage: str | Path) -> dict[int, dict]:
    """Map ``D -> normalized record`` for every ``case_D*.json`` in ``stage``."""
    cases: dict[int, dict] = {}
    for p in sorted(Path(stage).glob("case_D*.json")):
        rec = normalize_case(p)
        cases[rec["D"]] = rec
    return cases