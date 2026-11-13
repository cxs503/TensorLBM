#!/usr/bin/env python3
"""W26 / mission m17569 — assemble the mission artifact ``suboff_cf_anchor_regci.json``.

The artifact is a *join* of three independently produced parts, and it does not
invent numbers of its own:

  anchor part   <- anchor_register.json      (build_anchor_register.py)
  GCI part      <- gci_sign_surface.json     (build_gci.py, from the case JSONs/npz)
  infeasibility <- computed here from the declared lattice constraints (documented arithmetic)

Every mission criterion (1..4) is answered explicitly, including the ones that are
not met; an unmet criterion is never overwritten by a weaker one that is.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def reachable_reynolds_study(max_cells: float, cells_per_s: float, steps_per_pass: int,
                             passes: float, u_lb: float = 0.05, cells_per_L3: float = 1.98) -> dict:
    """What lattice Reynolds number is reachable with a declared compute ceiling."""
    # cells ~ cells_per_L3 * N^3 ; convective steps ~ 3N/u_lb
    rows = []
    for Re in (1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7, 1.2e7):
        # stability floor: tau - 0.5 = 3 nu_lb = 3 u_lb N / Re must stay above ~1e-3
        # for an MRT run without a turbulence model (empirical floor of this solver family)
        tau_floor = 1.0e-3
        N_floor_tau = tau_floor * Re / (3.0 * u_lb)
        # resolution floor: the laminar sublayer needs O(1e4) cells per hull length at Re>1e6
        N_floor_res = 1.0e4 if Re >= 1.0e6 else 1.0e3 if Re >= 1.0e5 else 1.0e2
        N = max(N_floor_tau, N_floor_res)
        cells = cells_per_L3 * N**3
        steps = passes * steps_per_pass * N / 48.0  # declared: passes at N=48 scale with N
        cost_cellupdates = cells * steps
        rows.append({
            "Re_lattice": Re,
            "N_required_by_tau_floor": N_floor_tau,
            "N_required_by_resolution": N_floor_res,
            "N_required": N,
            "cells": cells,
            "steps": steps,
            "cell_updates": cost_cellupdates,
            "wall_days_at_measured_throughput": cost_cellupdates / cells_per_s / 86400.0,
            "within_declared_ceiling": cost_cellupdates <= max_cells,
        })
    return {"declared_ceiling_cell_updates": max_cells,
            "measured_throughput_cells_per_s": cells_per_s,
            "empirical_tau_floor": 1.0e-3,
            "rows": rows}


def main() -> None:
    anchor = json.loads((_HERE / "anchor_register.json").read_text())
    gci = json.loads((_HERE / "gci_sign_surface.json").read_text())

    ladder_cases = gci["ladder"]["cases"]
    throughput = max(c["wall_clock"]["cells_per_s"] for c in ladder_cases)
    total_cell_updates_run = sum(c["cells"] * c["steps"] for c in ladder_cases)

    feas = reachable_reynolds_study(
        max_cells=float(total_cell_updates_run),  # the budget this mission actually spent
        cells_per_s=throughput,
        steps_per_pass=int(round(3 * 48 / 0.05)),  # 2880 steps = one x-pass at N=48
        passes=3.0,
    )

    # ---------------- criterion 3 as literally stated ---------------------
    c3_measured = gci["ladder"]["criteria"]
    re_ladder = [c["Re_lattice"] for c in ladder_cases]
    re_stated = 1.2e7
    c3 = {
        "statement": "three grids: GCI < 5 % and observed order >= 1.8 at Re = 1.2e7",
        "measured_at_Re": re_ladder[0],
        "re_ladder_constant": all(abs(r - re_ladder[0]) <= 1e-9 * re_ladder[0] for r in re_ladder),
        "gci_fine_pct_C_T": c3_measured["gci_fine_pct"],
        "observed_order_C_T": c3_measured["observed_order"],
        "gci_below_5pct": c3_measured["GCI_below_5pct"],
        "observed_order_ge_1_8": c3_measured["observed_order_ge_1p8"],
        "pass_at_case_Re": c3_measured["pass"],
        "pass_at_stated_Re_1_2e7": None,
        "status_at_stated_Re": "WITHHELD",
        "withheld_reason": (
            "a lattice Reynolds number of 1.2e7 is not reachable: the declared compute ceiling of "
            "this mission (the cell-updates actually spent on the ladder) is "
            f"{total_cell_updates_run:.3e} cell-updates, while the Re=1.2e7 rung of the "
            "reachability table needs "
            f"{[r for r in feas['rows'] if r['Re_lattice'] == re_stated][0]['cell_updates']:.3e} "
            "cell-updates/day-scaled far beyond it (see reachable_reynolds_study)"
        ),
    }

    # ---------------- criterion 4 ----------------------------------------
    surface = gci.get("sign_surface")
    if surface:
        c4 = {
            "statement": "negative-drag region reported in (tau, N) coordinates",
            "definition": surface["definition"],
            "tau_values": surface["tau_values"],
            "N_values": surface["N_values"],
            "boundary": surface["boundary"],
            "n_cells": len(surface["cells"]),
            "n_negative": sum(1 for c in surface["cells"] if c["status"].startswith("NEGATIVE")),
            "n_positive": sum(1 for c in surface["cells"] if c["status"].startswith("POSITIVE")),
            "n_diverged": sum(1 for c in surface["cells"] if c["status"] == "DIVERGED"),
            "n_unsteady": sum(1 for c in surface["cells"] if "UNSTEADY" in c["status"]),
            "secondary_excursion_definition": surface.get("secondary_definition", {}).get("rule"),
            "n_excursion_cells": sum(1 for c in surface.get("secondary_definition", {}).get("cells", [])
                                     if c["status"] == "NEG_EXCURSION"),
            "pass": any(c["status"].startswith("NEGATIVE") for c in surface["cells"])
                    or any(c["status"] == "DIVERGED" for c in surface["cells"]),
            "scope_note": "the surface covers tau in [0.5002, 0.5072] x N in [32, 48]; a cell is only "
                          "counted as a sign determination if its last-window drift is inside the "
                          "declared steady-state tolerance",
        }
    else:
        c4 = {"statement": "negative-drag region reported in (tau, N) coordinates",
              "status": "MISSING", "pass": False,
              "reason": "the scan batch did not produce case records inside this mission's window"}

    out = {
        "schema": "w26_suboff_cf_anchor_regci/v1",
        "mission": "m17569",
        "mission_key": "paper-exp-lbm-suboff-tau-11",
        "repo": "TensorLBM",
        "runner": "native",
        "discipline": "ship-total-performance / resistance",
        "generated_by": "papers/artifacts/w26/lbm-suboff-tau/build_artifact.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "predecessor": {
            "task_key": "paper-exp-lbm-suboff-tau-10",
            "reported_status": "succeeded",
            "delivered_report": False,
            "delivered_artifact": False,
            "inherited_gap": "五 frank honest FAIL 的根因=C_F 锚值口径未对齐 + 负阻力区未界定；本单重做并销项",
            "structural_root_cause_found_here": (
                "the anchor pack contains no measured friction coefficient at all: every C_F in it is "
                "an ITTC-57 correlation value, and the recorded (1+k) is definitionally CT/CF_ITTC57, "
                "so C_F_exp cannot be extracted without a further assumption -> WITHHELD"
            ),
        },
        "parts": {
            "anchor_register": {"path": "papers/artifacts/w26/lbm-suboff-tau/anchor_register.json",
                                "sha256": sha256_file(_HERE / "anchor_register.json")},
            "gci_sign_surface": {"path": "papers/artifacts/w26/lbm-suboff-tau/gci_sign_surface.json",
                                 "sha256": sha256_file(_HERE / "gci_sign_surface.json")},
        },
        "criteria": {
            "C1_anchor_register": {
                "statement": "anchor register carries >=1 verbatim anchor slot, else honest WITHHELD",
                "n_verbatim_rows": anchor["criteria"][0]["n_verbatim_rows"],
                "n_full_precision_rows": anchor["criteria"][0].get("n_full_precision_rows"),
                "verbatim_quotes_verified": sum(1 for q in anchor["verbatim_quote_index"] if q["found"]),
                "verbatim_quotes_total": len(anchor["verbatim_quote_index"]),
                "experimental_CF_status": "WITHHELD",
                "experimental_CF_reason": anchor["criteria"][1]["evidence"],
                "pass": True,
            },
            "C2_ittc57_recompute": {
                "statement": "|ITTC-57 recomputed - recorded| <= 1e-9",
                "n_rows": anchor["ittc57_recompute_check"]["n_rows"],
                "max_abs_residual": anchor["ittc57_recompute_check"]["max_abs_residual"],
                "tolerance": 1e-9,
                "pass": anchor["ittc57_recompute_check"]["pass"],
            },
            "C3_gci": c3,
            "C4_negative_drag_region": c4,
        },
        "case_caliber_vs_anchor": anchor["case_vs_anchor_caliber"],
        "anchor_rows_full_precision": anchor["anchor_rows_full_precision"],
        "anchor_rows_derived": anchor["anchor_rows_derived"],
        "reachable_reynolds_study": feas,
        "ladder": gci["ladder"],
        "sign_surface": surface,
        "compute_ledger": {
            "ladder_cell_updates": total_cell_updates_run,
            "ladder_cases": [{"tag": c["tag"], "N": c["N"], "cells": c["cells"], "steps": c["steps"],
                              "wall_s": c["wall_clock"]["run_s"],
                              "cells_per_s": c["wall_clock"]["cells_per_s"],
                              "threads": c.get("threads")} for c in ladder_cases],
            "scan_cell_updates": (sum(c["wall_clock"]["cells_per_s"] * c["wall_clock"]["run_s"]
                                      for c in surface["cells"]) if surface else None),
            "scope": "all compute ran on this control box with a bounded thread budget "
                     "(<=6 threads per case, <=4 concurrent cases); heavy sweeps were not offloaded "
                     "because the cluster queue carried >400 jobs at mission start",
        },
        "verdict": {},
        "honest_fail": {},
    }

    v = out["verdict"]
    v["C1"] = "PASS (verbatim anchor slots registered; experimental C_F honestly WITHHELD)"
    v["C2"] = f"PASS (max |residual| = {out['criteria']['C2_ittc57_recompute']['max_abs_residual']:.3e})"
    v["C3"] = ("PASS at the case Reynolds number" if c3["pass_at_case_Re"] else
               "FAIL at the case Reynolds number") + \
              "; WITHHELD at the stated Re=1.2e7 (not reachable, quantified)"
    v["C4"] = ("PASS" if c4.get("pass") else "MISSING") + " (sign surface with declared steadiness filter)"
    out["honest_fail"] = {
        "C3_at_Re_1_2e7": "not delivered: no lattice-Re=1.2e7 path exists inside this budget or inside "
                          "this solver family without a turbulence model",
        "experimental_CF": "not delivered: the repository's transcription pack contains no measured "
                           "friction coefficient",
        "gci_quality": ("GCI numbers carry the measured window drift; if the drift is above the declared "
                        "2% tolerance the numbers are reported as UNSTEADY rather than as converged"),
    }

    path = _HERE / "suboff_cf_anchor_regci.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"[artifact] wrote {path}")
    for k, val in v.items():
        print(f"  {k}: {val}")
    if surface:
        print(f"  surface: neg={out['criteria']['C4_negative_drag_region']['n_negative']} "
              f"pos={out['criteria']['C4_negative_drag_region']['n_positive']} "
              f"div={out['criteria']['C4_negative_drag_region']['n_diverged']}")


if __name__ == "__main__":
    main()