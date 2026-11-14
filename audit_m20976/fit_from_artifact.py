"""m20976 audit: the observed-order fit / Richardson that R2 predeclared but did
not report. Done on the five ladder maxima of the delivered artefact (the per-T
series that ``analyze()`` would need are not in the tree), using the delivered
``fit_order`` / ``richardson`` implementations unchanged so the numbers are
comparable with the design's own tolerances.

Also evaluates H1 (E_inf < 3 %) and the monotone-decrease precondition.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "repro"))
sys.path.insert(0, str(HERE.parent / "src"))

import resolution_study as R  # noqa: E402

ART = json.loads((HERE / "repro" / "r2_cj3032_maxima.json").read_text())
rows = [r for r in ART["rows"]
        if r["label"].startswith("tensorlbm_koerner") and r["tau"] == 1.0]
rows.sort(key=lambda r: r["a"])
out = {"ladder_used": [(r["a"], r["Z_cell_err_pct"], r["Z_sub05_err_pct"]) for r in rows],
       "fits": {}, "richardson_3grid": {}, "h1": {}}

for metric, key in (("Z_cell", "Z_cell_err_pct"), ("Z_sub0.05", "Z_sub05_err_pct")):
    samples = [(r["a"], r[key]) for r in rows]
    fit = R.fit_order(samples)
    out["fits"][metric] = fit
    # three-grid Richardson on the finest triple, as predeclared (32, 48, 64)
    try:  # delivered implementation needs per-T points -> not in the artefact
        rich = R.richardson({r["a"]: {"vs_mm": {metric: {"max_abs_rel_err_pct": r[key]}}}
                             for r in rows}, metric, "mm")
        out["richardson_3grid"][metric] = rich
    except Exception as e:  # noqa: BLE001
        out["richardson_3grid"][metric] = {"status": "NOT_COMPUTABLE_FROM_ARTEFACT",
                                           "why": f"{type(e).__name__}: {e}"}
        # maxima-level attempt with the observed order from the fit, grid pair 48/64
        p_obs = fit.get("p")
        if p_obs:
            e48 = rows[-2][key]
            e64 = rows[-1][key]
            r_ratio = rows[-1]["a"] / rows[-2]["a"]
            e_inf = e64 - (e64 - e48) / (r_ratio ** p_obs - 1.0)
            out["richardson_3grid"][metric]["maxima_level_attempt"] = {
                "p_observed": p_obs, "E48_pct": e48, "E64_pct": e64,
                "r": r_ratio, "E_inf_extrapolated_pct": round(e_inf, 3),
                "note": "invalid by construction: error is non-monotone, observed order p "
                        "is not resolved by the fit"}
    out["h1"][metric] = {
        "E_inf_below_3pct": (fit.get("E_inf_pct") is not None and fit["E_inf_pct"] < 3.0),
        "monotone_falling": fit.get("monotone_falling"),
        "verdict": ("H1 falsified: no admissible (E_inf,p) -> error is non-monotone" if not fit.get("ok") else ("H1 not falsified" if (fit.get("monotone_falling")
                                          and fit.get("E_inf_pct") is not None
                                          and fit["E_inf_pct"] < 3.0)
                    else "H1 falsified"),
    }

print(json.dumps(out, indent=1, default=str))
(HERE / "evidence").mkdir(parents=True, exist_ok=True)
(HERE / "evidence" / "fit_and_richardson.json").write_text(json.dumps(out, indent=1, default=str))