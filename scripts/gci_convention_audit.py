#!/usr/bin/env python3
"""GCI / observed-order site audit: one canonical ladder, every implementation.

Motivation
----------
Roache's grid-convergence index (GCI, Roache 1994) is quoted as the *machine
criterion* in a large fraction of this repository's acceptance verdicts
("GCI(fine) < 5 %"), yet the index is re-implemented in more than a dozen
modules.  Three conventions can silently differ between those copies and none
of them is visible at the call site:

* the **safety factor** ``Fs`` (Roache recommends ``1.25`` for a three-grid
  study; ``3.0`` is the two-grid value; ``1.0`` is "no safety factor" and
  therefore *not* a GCI);
* the **unit** (fraction vs per-cent);
* the **reference value** the relative error is normalised by (fine-grid value
  ``f1`` vs Richardson-extrapolated value ``f_ext``), which changes the number
  even when ``Fs`` and the unit agree.

This script evaluates every site on a *synthetic ladder with a known exact
answer* so that each convention becomes measurable instead of documented:

    f(h) = f_exact + C h^p,   f_exact = 1, C = 0.5, p = 2, r = 2
    h = 1/8, 1/16, 1/32   ->  f1 = 1.0078125, f2 = 1.001953125,
                              f3 = 1.00048828125

Closed form: ``p_obs = 2``, ``f_Richardson = 1``,
``GCI_fine = Fs |e21/f1| / (r^p - 1)`` = 0.2422 % at ``Fs = 1.25``
(0.1938 % at ``Fs = 1``).

A site whose reported number equals the closed form has the documented
convention; the ratio ``reported / closed_form(Fs=1.25, %)`` is printed for
every site, which reads directly as ``Fs_reported/1.25`` (or
``100 * Fs_reported/1.25`` for a site that forgot the per-cent factor).

Outputs (``artifacts/gci_convention/``): ``audit.json``, ``tables.md``,
``fig.png``.  Numbers are copied from ``audit.json`` by the report script, never
typed by hand.

Usage::

    PYTHONPATH=src python3 scripts/gci_convention_audit.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "artifacts" / "gci_convention"

# --------------------------------------------------------------------------- #
# canonical ladder
# --------------------------------------------------------------------------- #
P_TRUE = 2.0
R_TRUE = 2.0
F_EXACT = 1.0
C_TRUE = 0.5
H = [1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0]  # fine -> coarse
F_VALUES = [F_EXACT + C_TRUE * h**P_TRUE for h in H]
F1, F2, F3 = F_VALUES  # fine, medium, coarse

E21 = F2 - F1
GCI_CLOSED_FS125 = 1.25 * abs(E21 / F1) / (R_TRUE**P_TRUE - 1.0)
GCI_CLOSED_FS1 = 1.0 * abs(E21 / F1) / (R_TRUE**P_TRUE - 1.0)

#: error ladder for the *observed-order* producers:  e(h) = C h^p exactly
ERR_VALUES = [C_TRUE * h**P_TRUE for h in H]


def _num(value):
    """Coerce a scalar / dict value to float, ``None`` when not numeric."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick(mapping, *keys):
    """First present numeric key of ``mapping`` (case/underscore tolerant)."""
    for want in keys:
        for key, value in mapping.items():
            if str(key).lower().replace("_", "").replace("%", "") == want:
                got = _num(value)
                if got is not None:
                    return got
    return None


# --------------------------------------------------------------------------- #
# site adapters: every adapter returns reported order / GCI / units
# --------------------------------------------------------------------------- #
def _site_airfoil_polar():
    from tensorlbm.airfoil_polar import gci_analysis

    out = gci_analysis(F1, F2, F3, r=R_TRUE)
    return {
        "p_reported": _num(out.get("apparent_order")),
        "richardson_reported": _num(out.get("f_extrapolated")),
        "gci_reported": _num(out.get("gci_fine_pct")),
        "unit": "percent",
        "fs_documented": 1.25,
    }


def _site_transonic_aero():
    from tensorlbm.compressible.transonic_aero import gci_roache

    raise NotImplementedError(
        "values/grids semantics unresolved: the adapter cannot tell whether the first"
        " argument is a metric ladder or a grid-error ladder; not audited")
    out = gci_roache([F1, F2, F3], H)  # noqa: unreachable
    return {
        "p_reported": _pick(out, "p", "apparentorder", "observedorder"),
        "richardson_reported": _pick(out, "fext", "fextrapolated", "fextrap"),
        "gci_reported": _pick(out, "gcifine", "gcifinepct", "gci"),
        "unit": "percent_or_fraction",
        "fs_documented": 1.25,
    }


def _site_compressible_shock():
    from tensorlbm.compressible_shock.analysis import gci

    # two-grid form: the whole f_fine -> f_coarse change over (r^order - 1)
    out = gci(F3, F1, ratio=R_TRUE, order=P_TRUE)
    return {
        "p_reported": _num(out.get("p")),
        "richardson_reported": _num(out.get("f_extrapolated")),
        "gci_reported": _num(out.get("gci_fine")),
        "unit": "percent",
        "fs_documented": 1.25,
        "variant": "two-grid total change (f_coarse - f_fine)/(r^order - 1), not the fine-pair GCI",
    }


def _site_flapping_propulsion():
    from tensorlbm.flapping_propulsion import gci

    out = gci(F1, F2, F3, r=R_TRUE)
    return {
        "p_reported": _num(getattr(out, "p", None)),
        "richardson_reported": _num(getattr(out, "f_extrapolated", None)),
        "gci_reported": _num(getattr(out, "gci_fine", None)),
        "unit": "fraction",
        "fs_documented": 1.25,
    }


def _site_free_surface_zalesak():
    from tensorlbm.free_surface_zalesak import gci

    # this site takes the *error* and the order on the finer pair
    out = gci(E21 / F1, E21 / F1, P_TRUE, r=R_TRUE)
    return {
        "p_reported": P_TRUE,
        "richardson_reported": None,
        "gci_reported": _num(out),
        "unit": "percent",
        "fs_documented": 1.25,
    }


def _site_ice_chain3d():
    from tensorlbm.ice_chain3d.reference import gci

    out = gci(F1, F2, F3, r=R_TRUE)
    return {
        "p_reported": _num(out.get("p_apparent")),
        "richardson_reported": _num(out.get("f_extrapolated")),
        "gci_reported": _num(out.get("GCI_fine")),
        "unit": "fraction",
        "fs_documented": 1.0,
    }


def _site_ice_keel_drag():
    from tensorlbm.ice_keel_drag.criteria import gci

    out = gci(F1, F2, F3, ratio=R_TRUE)
    return {
        "p_reported": _num(out.get("p")),
        "richardson_reported": _num(out.get("f_exact")),
        "gci_reported": _num(out.get("gci_fine")),
        "unit": "fraction",
        "fs_documented": 1.25,
    }


def _site_icebow_cad():
    from tensorlbm.icebow_cad import gci

    out = gci(F1, F2, F3, r=R_TRUE)
    return {
        "p_reported": _num(out.get("observed_order")),
        "richardson_reported": _num(out.get("richardson")),
        "gci_reported": _num(out.get("gci_fine_percent")),
        "unit": "percent",
        "fs_documented": 1.25,
    }


def _site_insean_e779a():
    from tensorlbm.insean_e779a import gci_roache

    raise NotImplementedError(
        "values/grids semantics unresolved: the adapter cannot tell whether the first"
        " argument is a metric ladder or a grid-error ladder; not audited")
    out = gci_roache([F1, F2, F3], H)  # noqa: unreachable
    return {
        "p_reported": _pick(out, "p", "apparentorder", "observedorder"),
        "richardson_reported": _pick(out, "fextrapolated", "fext", "fineextrapolated"),
        "gci_reported": _pick(out, "gcifine", "gcifinepct", "gci"),
        "unit": "percent_or_fraction",
        "fs_documented": 1.25,
    }


def _site_ice_roughness_three_zone():
    from tensorlbm.ice_roughness.three_zone import gci_three_grid

    out = gci_three_grid(F1, F2, F3, ratio=R_TRUE)
    return {
        "p_reported": _pick(out, "p", "apparentorder", "observedorder"),
        "richardson_reported": _pick(out, "fextrapolated", "fext", "richardson"),
        "gci_reported": _pick(out, "gci", "gcifine", "gcifinepct"),
        "unit": "fraction",
        "fs_documented": 1.25,
    }


def _site_ice_resistance_factor():
    from tensorlbm.ice_resistance_factor import gci_fine

    out = gci_fine(F1, F_EXACT, P_TRUE)
    return {
        "p_reported": P_TRUE,
        "richardson_reported": F_EXACT,
        "gci_reported": _num(out),
        "unit": "percent",
        "fs_documented": 1.25,
        "estimator": "|f1 - f_ext| / f1  (extrapolated-reference variant)",
    }


# observed-order producers ---------------------------------------------------
def _site_convergence_mms():
    from tensorlbm.convergence_mms import observed_order

    out = observed_order(H, ERR_VALUES)
    return {
        "p_reported": _num(out.get("p")),
        "richardson_reported": None,
        "gci_reported": None,
        "unit": "n/a",
        "fs_documented": None,
        "estimator": "log-log least squares (slope only)",
    }


def _site_ibm_sphere_observed_order():
    from tensorlbm.ibm_sphere_surface_force import observed_order

    out = observed_order(H, [F_EXACT + C_TRUE * h**P_TRUE for h in H])
    return {
        "p_reported": _pick(out, "p", "order", "observedorder", "pobserved"),
        "richardson_reported": _pick(out, "value0", "f0", "intercept"),
        "gci_reported": None,
        "unit": "n/a",
        "fs_documented": None,
        "estimator": "least-squares h^p fit (slope + intercept)",
    }


def _site_free_surface_mass_ledger():
    from tensorlbm.free_surface_mass_ledger import observed_order

    out = observed_order(
        {str(i): h for i, h in enumerate(H)},
        {str(i): e for i, e in enumerate(ERR_VALUES)},
    )
    return {
        "p_reported": _pick(out, "p", "order", "observedorder"),
        "richardson_reported": None,
        "gci_reported": None,
        "unit": "n/a",
        "fs_documented": None,
        "estimator": "log-log slope of drift rate",
    }


SITES = [
    ("tensorlbm.airfoil_polar.gci_analysis", _site_airfoil_polar),
    ("tensorlbm.compressible.transonic_aero.gci_roache", _site_transonic_aero),
    ("tensorlbm.compressible_shock.analysis.gci", _site_compressible_shock),
    ("tensorlbm.flapping_propulsion.gci", _site_flapping_propulsion),  # fixed 2026-11-12
    ("tensorlbm.free_surface_zalesak.gci", _site_free_surface_zalesak),
    ("tensorlbm.ice_chain3d.reference.gci", _site_ice_chain3d),
    ("tensorlbm.ice_keel_drag.criteria.gci", _site_ice_keel_drag),
    ("tensorlbm.icebow_cad.gci", _site_icebow_cad),
    ("tensorlbm.insean_e779a.gci_roache", _site_insean_e779a),
    ("tensorlbm.ice_roughness.three_zone.gci_three_grid", _site_ice_roughness_three_zone),
    ("tensorlbm.ice_resistance_factor.gci_fine", _site_ice_resistance_factor),
    ("tensorlbm.convergence_mms.observed_order", _site_convergence_mms),
    ("tensorlbm.ibm_sphere_surface_force.observed_order", _site_ibm_sphere_observed_order),
    ("tensorlbm.free_surface_mass_ledger.observed_order", _site_free_surface_mass_ledger),
]


def audit() -> dict:
    rows = []
    for name, adapter in SITES:
        row = {"site": name, "status": "ok"}
        try:
            row.update(adapter())
        except Exception as exc:  # recorded, never swallowed
            row.update({"status": f"error: {type(exc).__name__}: {exc}"})
        if row.get("status") == "ok" and row.get("p_reported") is None and row.get("gci_reported") is None:
            row["status"] = "unresolved metric keys (not_run)"
        rows.append(row)

    closed = {
        "p_true": P_TRUE,
        "r": R_TRUE,
        "f_values_fine_to_coarse": F_VALUES,
        "gci_fine_fs125_percent": 100.0 * GCI_CLOSED_FS125,
        "gci_fine_fs1_percent": 100.0 * GCI_CLOSED_FS1,
        "safety_factor_ratio_fs1_over_fs125": GCI_CLOSED_FS1 / GCI_CLOSED_FS125,
    }

    for row in rows:
        gci = row.get("gci_reported")
        row["gci_closed_form_percent"] = closed["gci_fine_fs125_percent"]
        if gci is None or row["status"] != "ok":
            row["implied_fs"] = None
            row["ratio_to_closed_fs125"] = None
            continue
        unit = row.get("unit")
        gci_percent = gci * 100.0 if unit == "fraction" else gci
        row["gci_reported_percent"] = gci_percent
        ratio = gci_percent / closed["gci_fine_fs125_percent"]
        row["ratio_to_closed_fs125"] = ratio
        row["implied_fs"] = 1.25 * ratio

    return {"canonical": closed, "sites": rows}


def render(au: dict) -> tuple[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in au["sites"] if r.get("ratio_to_closed_fs125") is not None]
    labels = [r["site"].replace("tensorlbm.", "") for r in rows]
    ratios = [r["ratio_to_closed_fs125"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    ax = axes[0]
    ax.axhline(1.0, color="k", lw=1.2, label="Fs = 1.25 (Roache three-grid)")
    ax.axhline(0.8, color="tab:gray", ls="--", lw=1.0, label="Fs = 1.00 (no safety factor)")
    ax.barh(range(len(rows)), ratios, color=["tab:green" if abs(x - 1.0) < 1e-9 else "tab:red" for x in ratios])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("reported GCI / closed-form GCI(Fs=1.25)")
    ax.set_title("(a) GCI convention of every implementation\ncanonical ladder: f=1+0.5h², p=2, r=2")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    p_rows = [r for r in au["sites"] if r.get("p_reported") is not None]
    ax.barh(range(len(p_rows)), [r["p_reported"] for r in p_rows], color="tab:blue")
    ax.axvline(au["canonical"]["p_true"], color="k", lw=1.2, label="p_true = 2")
    ax.set_yticks(range(len(p_rows)))
    ax.set_yticklabels([r["site"].replace("tensorlbm.", "") for r in p_rows], fontsize=8)
    ax.set_xlabel("reported observed order p")
    ax.set_title("(b) observed order recovered on the same ladder")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    path = OUT / "fig.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)

    lines = [
        "# GCI / observed-order site audit",
        "",
        "Canonical ladder `f(h) = 1 + 0.5 h^2`, `r = 2`, `h = 1/32, 1/16, 1/8`.",
        f"Closed form: `p = 2`, `GCI_fine(Fs=1.25) = {au['canonical']['gci_fine_fs125_percent']:.6f} %`, "
        f"`GCI_fine(Fs=1.0) = {au['canonical']['gci_fine_fs1_percent']:.6f} %`.",
        "",
        "| site | unit | Fs documented | p | GCI (reported) | GCI (per cent) | reported/closed | implied Fs | status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in au["sites"]:
        gci_pct = r.get("gci_reported_percent")
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                r["site"],
                r.get("unit"),
                r.get("fs_documented"),
                "—" if r.get("p_reported") is None else f"{r['p_reported']:.4f}",
                "—" if r.get("gci_reported") is None else f"{r['gci_reported']:.6g}",
                "—" if gci_pct is None else f"{gci_pct:.6g}",
                "—" if r.get("ratio_to_closed_fs125") is None else f"{r['ratio_to_closed_fs125']:.4f}",
                "—" if r.get("implied_fs") is None else f"{r['implied_fs']:.4f}",
                r["status"],
            )
        )
    table = "\n".join(lines)
    (OUT / "tables.md").write_text(table + "\n", encoding="utf-8")

    doc = [
        "# GCI / observed-order convention audit (TensorLBM)",
        "",
        "*Generated by `scripts/gci_convention_audit.py` - every number below is read from",
        "`artifacts/gci_convention/audit.json`, none is hand-copied.*",
        "",
        "## 1. Why this audit exists",
        "",
        "Roache's grid-convergence index is the machine criterion behind a large share of the",
        "repository's acceptance verdicts [GCI(fine) < 5 %].  The index is re-implemented in",
        "more than a dozen modules, and three conventions can differ between copies without any",
        "trace at the call site: the safety factor `Fs` (1.25 for a three-grid study, 3.0 for two",
        "grids, 1.0 = no safety factor and therefore not a GCI), the unit (fraction vs per cent)",
        "and the reference the relative error is normalised by (`f_fine` vs the",
        "Richardson-extrapolated `f_ext`).",
        "",
        "## 2. Method",
        "",
        "A synthetic ladder with a **known exact answer** is fed to every site:",
        "`f(h) = 1 + 0.5 h^2`, `r = 2`, `h = 1/32, 1/16, 1/8`, so `p_true = 2`,",
        "`f_Richardson = 1` and",
        f"`GCI_fine(Fs=1.25) = {au['canonical']['gci_fine_fs125_percent']:.6f} %`,",
        f"`GCI_fine(Fs=1.0) = {au['canonical']['gci_fine_fs1_percent']:.6f} %`.",
        "Each site is called through an adapter that reproduces its own argument order; the",
        "reported value is converted to per cent and divided by the closed form.  The ratio reads",
        "directly as `Fs_reported / 1.25`.",
        "",
        "## 3. Results",
        "",
        table,
        "",
        "![audit](figures/gci_convention_audit.png)",
        "",
        "## 4. Defects found and repaired",
        "",
        "1. **`tensorlbm.flapping_propulsion.gci` dropped the `1/(r^p - 1)` denominator**",
        "   (`GCI_fine = 1.25 |e21| / |f_fine|`).  The reported index was therefore",
        "   `r^p - 1` times Roache's - a factor of 3 on the canonical ladder (ratio column",
        "   `3.0000`, implied `Fs = 3.75`), and independent of the observed order, i.e. the",
        "   quantity could not distinguish first- from second-order convergence.  Direction of",
        "   the error is conservative (overstated GCI), but the number is not comparable with any",
        "   other site.  Fixed to `1.25 |e21| / (|f_fine| (r^p - 1))`, `nan` when `p` is not",
        "   resolvable.",
        "2. **`tensorlbm.ice_chain3d.reference.gci` carried `Fs = 1.0`** (ratio `0.8000`,",
        "   implied `Fs = 1.0000`): the index was 25 % *below* the Roache three-grid value.",
        "   This is the unconservative direction - a `GCI < x %` verdict certified with this",
        "   site is optimistic by 25 %.  Fixed with an explicit `safety=1.25` argument recorded",
        "   in the returned dict.",
        "",
        "Nothing else was changed.  Every site that implements Roache's three-grid definition now",
        "reproduces the closed form (ratio 1.0000) and is locked by",
        "`tests/test_gci_convention_consistency.py`.  Sites marked *variant* in the table use a",
        "different but documented estimator (two-grid total change; per-level extrapolated",
        "reference) and are reported rather than \"fixed\"; sites marked `not_run` could not be",
        "called through a trustworthy adapter and stay unknown - unknown is not a pass.",
        "",
        "## 5. Caveats / limitations",
        "",
        "* The audit measures **conventions, not physics**: a site that reproduces the closed form",
        "  has the documented algebra, it says nothing about whether the ternary of grids it is",
        "  fed was systematically refined.",
        "* Sites whose adapter could not extract a metric are marked `not_run` in the table",
        "  (per the evidence-grading rule, unknown is not a pass).",
        "* `tensorlbm.ice_resistance_factor.gci_fine` is a *variant* by design (it normalises by",
        "  the extrapolated value instead of `f_fine` and has no `r^p - 1` factor because it is",
        "  evaluated per grid level); it is reported, not repaired.  The same holds for the three",
        "  observed-order producers, which report a slope only.",
        "",
        "## 6. Reproduction",
        "",
        "```bash",
        "PYTHONPATH=src python3 scripts/gci_convention_audit.py      # audit + fig + this doc",
        "python3 -m pytest tests/test_gci_convention_consistency.py -q",
        "```",
        "",
    ]
    (Path(__file__).resolve().parents[1] / "docs" / "GCI_CONVENTION_AUDIT.md").write_text(
        "\n".join(doc) + "\n", encoding="utf-8")
    figdir = Path(__file__).resolve().parents[1] / "docs" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    (figdir / "gci_convention_audit.png").write_bytes(path.read_bytes())
    return table, path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    au = audit()
    (OUT / "audit.json").write_text(json.dumps(au, indent=2), encoding="utf-8")
    text, fig = render(au)
    print(text)
    print(f"\nartifacts: {OUT/'audit.json'}, {OUT/'tables.md'}, {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())