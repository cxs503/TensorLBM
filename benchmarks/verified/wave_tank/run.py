"""Benchmark driver: phase-field LBM wave tank (Stokes-2nd generation/absorption).

Run from the repository root::

    PYTHONPATH=src python benchmarks/verified/wave_tank/run.py --list
    PYTHONPATH=src python benchmarks/verified/wave_tank/run.py --run kh1.5_eps0.08
    PYTHONPATH=src python benchmarks/verified/wave_tank/run.py --run-all
    PYTHONPATH=src python benchmarks/verified/wave_tank/run.py --analyze
    PYTHONPATH=src python benchmarks/verified/wave_tank/run.py --figures
    PYTHONPATH=src python benchmarks/verified/wave_tank/run.py --markdown

Products land in ``artifacts/wave_tank/`` (per-case ``.npz``, ``results.json``,
``figures/*.png``).  ``--markdown`` re-renders the numeric tables of
``README.md`` from ``results.json`` (no hand-copied numbers).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from tensorlbm.wave_tank.campaign import ART, CASES, analyse_case, evaluate, main as campaign_main


def render_markdown(out_dir: str = ART) -> str:
    """Render the results tables of the benchmark README from results.json."""
    with open(os.path.join(out_dir, "results.json")) as fh:
        payload = json.load(fh)
    cases, crit = payload["cases"], payload["criteria"]
    lines: list[str] = []
    lines.append("| case | kh | eps | c theory | c measured | err % | a1 incident | err % "
                 "| a2 incident | err % | KR % | mass drift %/T | steps | wall s |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in sorted(cases):
        r = cases[name]
        th, m = r["theory"], r["metrics"]
        lines.append(
            "| %s | %.1f | %.2f | %.4f | %.4f | %.2f | %.4f | %.2f | %.4f | %.2f | %.2f | %+.3f | %d | %.0f |"
            % (
                name, th["kh"], th["eps"], th["celerity"], m["c_measured"],
                100 * m["c_error_rel"], m["a1_incident"], 100 * m["a1_error_rel"],
                m["a2_incident"], 100 * m["a2_error_rel"], 100 * m["kr"],
                100 * m["mass_drift_per_period"], m["steps"], m["wall_time_s"],
            )
        )
    lines.append("")
    lines.append("| criterion | pass | detail |")
    lines.append("|---|---|---|")
    for key in sorted(crit):
        if key == "all_pass":
            continue
        c = crit[key]
        detail = c.get("note", "")
        if "tol" in c:
            detail = (detail + " tol=%s" % c["tol"]).strip()
        if "negative_control" in c and c["negative_control"] is not None:
            detail += " neg_ctrl_KR=%.3f" % c["negative_control"]
        lines.append("| %s | %s | %s |" % (key, "PASS" if c.get("pass") else "FAIL", detail))
    lines.append("| ALL | %s | |" % ("PASS" if crit.get("all_pass") else "FAIL"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run")
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--out", default=ART)
    a = ap.parse_args(argv)
    if a.markdown:
        print(render_markdown(a.out))
        return 0
    fwd = []
    for flag in ("list", "run", "run_all", "analyze", "figures"):
        v = getattr(a, flag)
        if v:
            fwd.append("--" + flag.replace("_", "-"))
            if flag == "run":
                fwd.append(v)
    fwd += ["--out", a.out]
    return campaign_main(fwd)


if __name__ == "__main__":
    raise SystemExit(main())