#!/usr/bin/env python3
"""Independent adversarial probes for the XVAL-LBM-CHAPMAN-ENSKOG-AUTO verdict (m17741).

Three questions the shipped evidence does not answer by itself:

P1 detection power  -- if the *measurement* and the *derivation* are given
                       mismatched parameters, does the 15 % gate actually fire?
                       (a gate that never fails is decoration)
P2 non-hardcoding    -- is ``nu`` really produced by the symbolic moment algebra,
                       or is it the textbook formula re-typed?  Probe: change the
                       weights of the stencil and see whether ``nu`` moves with
                       ``cs2`` (a re-typed formula keyed on tau would not).
P3 provenance drift  -- the shipped result.json stamps ``git_head=e53d994d``
                       while the artefact is committed in 15a9b393.  Re-derive
                       every reference from the module at HEAD and compare with
                       the shipped ``ref`` column (bit-level).

Run: PYTHONPATH=src python3 docs/audit_m17741/probe_m17741.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import os
REPO = Path(os.environ.get("TARGET_REPO", "/root/tp-agent/work/m17672"))  # audited worktree
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "xval_chapman_enskog_auto"))

import tensorlbm.chapman_enskog as ce  # noqa: E402
import run_xval as rx  # noqa: E402

out: dict = {}

# ---------------------------------------------------------------- P1 ----------------------------------
tau_sim, tau_pred = 0.8, 1.2
meas = rx.measure_shear_2d(32, 32, tau_sim)
d_right = ce.derive_equivalent_equation(ce.D2Q9, ce.bgk(tau_sim))
d_wrong = ce.derive_equivalent_equation(ce.D2Q9, ce.bgk(tau_pred))
err_right = abs(meas["nu"] - d_right.nu_value) / d_right.nu_value * 100.0
err_wrong = abs(meas["nu"] - d_wrong.nu_value) / d_wrong.nu_value * 100.0
out["P1_detection_power"] = {
    "measurement": {"tau_used": tau_sim, "nu_measured": meas["nu"], "r2": meas["r2"]},
    "prediction_matched": {"tau": tau_sim, "nu_pred": d_right.nu_value, "err_pct": err_right,
                           "gate_15pct": "PASS" if err_right < 15 else "FAIL"},
    "prediction_mismatched": {"tau": tau_pred, "nu_pred": d_wrong.nu_value, "err_pct": err_wrong,
                              "gate_15pct": "PASS" if err_wrong < 15 else "FAIL"},
    "conclusion": "gate has detection power" if (err_right < 15 <= err_wrong) else "GATE IS BLIND",
}

# ---------------------------------------------------------------- P2 ----------------------------------
# D2Q9 with a deliberately anisotropic weight perturbation: the fourth moment is no
# longer isotropic, so a genuine symbolic derivation must change nu (and flag isotropy).
d_ref = ce.derive_equivalent_equation(ce.D2Q9, ce.bgk(0.8))
from fractions import Fraction

w = [Fraction(x) for x in ce.D2Q9.w]
w[1] = w[1] * Fraction(11, 10)  # east weight only, renormalised below
tot = sum(w)
w = [x / tot for x in w]
stencil_mod = ce.Stencil(name="D2Q9-wmod", c=ce.D2Q9.c, w=tuple(w))
try:
    d_mod = ce.derive_equivalent_equation(stencil_mod, ce.bgk(0.8))
    out["P2_non_hardcoding"] = {
        "baseline": {"cs2": d_ref.cs2_value, "nu": d_ref.nu_value,
                     "fourth_order_isotropic": bool(d_ref.isotropy.get("fourth_order_isotropic"))},
        "weight_perturbed": {"cs2": d_mod.cs2_value, "nu": d_mod.nu_value,
                             "fourth_order_isotropic": bool(d_mod.isotropy.get("fourth_order_isotropic"))},
        "nu_moved": abs(d_mod.nu_value - d_ref.nu_value) > 1e-12,
    }
except Exception as exc:  # noqa: BLE001 - the module is allowed to refuse
    out["P2_non_hardcoding"] = {"baseline_nu": d_ref.nu_value, "perturbed_raised": f"{type(exc).__name__}: {exc}"}

# ---------------------------------------------------------------- P3 ----------------------------------
shipped = json.loads((REPO / "benchmarks/xval_chapman_enskog_auto" / "result.json").read_text())
der = shipped["derivations"]
rebuilt = {
    "D2Q9-BGK-tau0.8": ce.derive_equivalent_equation(ce.D2Q9, ce.bgk(0.8)),
    "D2Q9-MRT-tau0.8": ce.derive_equivalent_equation(ce.D2Q9, ce.d2q9_mrt(0.8)),
    "D2Q9-MRT-tau0.8-s_e1.64": ce.derive_equivalent_equation(ce.D2Q9, ce.d2q9_mrt(0.8, s_e=1.64)),
    "D3Q19-BGK-tau0.8": ce.derive_equivalent_equation(ce.D3Q19, ce.bgk(0.8, 19)),
    "D3Q27-BGK-tau0.8": ce.derive_equivalent_equation(ce.D3Q27, ce.bgk(0.8, 27)),
}
drift = {}
for key, d in rebuilt.items():
    s = der[key]
    drift[key] = {
        "nu_shipped": s["nu"], "nu_head": d.nu_value,
        "nu_rel_diff": abs(s["nu"] - d.nu_value) / max(abs(d.nu_value), 1e-30),
        "zeta_shipped": s["zeta"], "zeta_head": d.zeta_value,
    }
out["P3_provenance"] = {
    "shipped_git_head": shipped["provenance"]["git_head"],
    "shipped_module_sha256": shipped["provenance"]["chapman_enskog_sha256"],
    "rebuilt_vs_shipped": drift,
    "max_nu_rel_diff": max(v["nu_rel_diff"] for v in drift.values()),
}

print(json.dumps(out, indent=2, default=float))
(Path(__file__).resolve().parent / "probe_m17741.json").write_text(json.dumps(out, indent=2, default=float))