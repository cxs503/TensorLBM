"""Re-grade the *recorded* round-2 terminal velocities against both constant conventions.

Input is the machine-written ``artifacts/round2/acceptance.json`` (from mission
m18081/m2026-11-13); no number is re-typed and no simulation is re-run.  The
round-2 file itself is left untouched -- this script only re-reads it and
reports what the same measurements mean under ``C = 3`` (primary, 2-D-specific
literature statement) and ``C = 1`` (arXiv:2009.14655 Eq. 1).

Why: round 2 gated on ``C = 1``, which was back-solved from these very runs
(quoted: "C = 0.90 (A = 1/2) and C = 1.01 (A = 1/3)").  A reference fitted to
the answer cannot falsify the solver, so the PASS it produced is void.
"""
from __future__ import annotations

import json, math, sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
for _p in (str(BENCH), str(BENCH / "ref"), str(BENCH.parents[2] / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import run_case as rc  # noqa: E402

OUT = BENCH / "artifacts" / "round3"
SRC = BENCH / "artifacts" / "round2" / "acceptance.json"


def main() -> int:
    data = json.loads(SRC.read_text())
    cases = data["cases"] if isinstance(data, dict) else data
    rows = []
    for c in cases:
        a, g, lam = c["A"], c["g"], c["lam"]
        ref3 = rc.boussinesq_velocity(a, g, lam)
        ref1 = rc.boussinesq_velocity_alt(a, g, lam)
        ub = c.get("u_b", float("nan"))
        # u_s == u_b by the (phi, y) -> (-phi, -y) symmetry of the single-density
        # Boussinesq model -- a theorem, not evidence (see the pytest file)
        rows.append({
            "case": c["name"], "scheme": c.get("scheme", "bgk"),
            "atwood": a, "lam": lam, "g": g,
            "u_measured": ub, "u_spike_measured": c.get("u_s"),
            "u_b_over_u_s": (ub / c["u_s"]) if c.get("u_s") else float("nan"),
            "u_ref_C3": ref3, "u_ref_C1": ref1,
            "err_C3_pct": 100.0 * (ub / ref3 - 1.0),
            "err_C1_pct": 100.0 * (ub / ref1 - 1.0),
            "pass_C3": abs(ub / ref3 - 1.0) <= rc.GATE_PCT / 100.0,
            "pass_C1": abs(ub / ref1 - 1.0) <= rc.GATE_PCT / 100.0,
            "sigma_err_pct_round2": c.get("sigma_err"),
            "phi_max": c.get("phi_max"),
            "source": str(SRC.relative_to(BENCH.parents[2])),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"C_primary": rc.C_2D, "C_alt": rc.C_2D_ALT, "gate_pct": rc.GATE_PCT,
               "convention_conflict": rc.CONVENTION_CONFLICT, "rows": rows}
    (OUT / "regrade_round2.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    lines = ["| case | scheme | A | lam | u_measured | ref(C=3) | err(C=3) | pass | ref(C=1) | err(C=1) | pass | u_b/u_s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['case']} | {r['scheme']} | {r['atwood']:.3f} | {r['lam']} | "
                     f"{r['u_measured']:.5f} | {r['u_ref_C3']:.5f} | {r['err_C3_pct']:+.1f}% | "
                     f"{'PASS' if r['pass_C3'] else 'FAIL'} | {r['u_ref_C1']:.5f} | "
                     f"{r['err_C1_pct']:+.1f}% | {'PASS' if r['pass_C1'] else 'FAIL'} | "
                     f"{r['u_b_over_u_s']:.4f} |")
    md = "\n".join(lines)
    (OUT / "regrade_round2.md").write_text(
        "# Round-2 recorded velocities re-graded (both conventions)\n\n" + md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
