"""Round-3 re-grade of CASE-RAYLEIGH-TAYLOR: both constant conventions, no re-simulation.

Every published number is recomputed here from the raw ``history.csv`` /
``case.json`` of each run (round-1 legacy BGK, round-2 Allen-Cahn, round-3
resolution sweep).  Nothing is transcribed by hand, and the same function is
called by ``tests/test_case_pool_rayleigh_taylor_reference_audit.py`` so the
published table and the test cannot drift apart.

Two references are reported for every row:

* ``primary`` -- ``sqrt(2Ag/(C k))`` with ``C = 3`` (2-D-specific literature
  statement, ``ref/ref_lock.py``);
* ``alt``     -- the same with ``C = 1`` (arXiv:2009.14655 Eq. 1 convention).

The gate criterion (<10 %) is evaluated against ``primary``; the ``alt`` column
exists so the reader can see how much of the verdict is convention-dependent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
for _p in (str(BENCH_DIR), str(BENCH_DIR / "ref"), str(BENCH_DIR.parents[2] / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_case as rc  # noqa: E402

OUT = BENCH_DIR / "artifacts" / "round3"
SEARCH_ROOTS = (
    OUT / "runs",
    BENCH_DIR / "artifacts" / "runs",
    BENCH_DIR.parents[2] / "runs" / "m18081",
    BENCH_DIR.parents[2] / "runs" / "m18081" / "suite_r1",
    BENCH_DIR.parents[2] / "runs" / "m18081" / "suite_r2",
)


def _slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else float("nan")


def find_case_dir(name: str, source_root: str | None = None) -> Path | None:
    roots = [Path(source_root)] if source_root else list(SEARCH_ROOTS)
    for root in roots:
        if root is None or not Path(root).exists():
            continue
        cand = Path(root) / name
        if (cand / "case.json").exists():
            return cand
        for sub in Path(root).glob("suite*"):
            cand = sub / name
            if (cand / "case.json").exists():
                return cand
        hits = [p for p in Path(root).rglob(f"{name}/case.json")]
        if hits:
            return hits[0].parent
    return None


def evaluate(case_dir: Path) -> dict:
    case = json.loads((case_dir / "case.json").read_text())
    spec = case["spec"]
    rows = list(csv.DictReader((case_dir / "history.csv").open()))
    hb = "h_bubble" if "h_bubble" in rows[0] else "h_bubble_col"
    hs = "h_spike" if "h_spike" in rows[0] else "h_spike_col"
    win = [r for r in rows if rc.TAU_LO <= float(r["tau"]) <= rc.TAU_HI]
    atwood, g, lam = spec["atwood"], spec["g"], spec["lam"]
    u_ref = rc.boussinesq_velocity(atwood, g, lam)
    u_ref_alt = rc.boussinesq_velocity_alt(atwood, g, lam)
    out = {
        "case": case["name"],
        "source": str(case_dir),
        "scheme": spec.get("phase_scheme", "bgk"),
        "atwood": atwood,
        "lam": lam,
        "g": g,
        "ac_width": spec.get("ac_width"),
        "n_window": len(win),
        "u_ref_primary": u_ref,
        "u_ref_alt": u_ref_alt,
        "C_primary": rc.C_2D,
        "C_alt": rc.C_2D_ALT,
        "sigma_sim": case.get("sigma_sim", float("nan")),
        "sigma_err_pct": case.get("sigma_rel_error_vs_viscous_pct", float("nan")),
        "phi_max_peak": (case.get("phi_range_peak") or [None, None])[1],
        "ma_peak": case.get("ma_peak"),
    }
    if len(win) < 5:
        out.update(u_front=float("nan"), u_front_zc=float("nan"),
                   err_primary_pct=float("nan"), err_alt_pct=float("nan"),
                   fr_meas=float("nan"), pass_primary=False, pass_alt=False)
        return out
    ts = [float(r["tau"]) for r in win]
    unit = math.sqrt(atwood * g / lam)
    u = _slope(ts, [float(r[hb]) for r in win]) * unit
    uz = _slope(ts, [float(r["h_bubble_zc"]) for r in win]) * unit if "h_bubble_zc" in rows[0] else float("nan")
    fr = u / math.sqrt(atwood * g * lam)
    out.update(
        u_front=u, u_front_zc=uz, fr_meas=fr,
        err_primary_pct=100.0 * (u / u_ref - 1.0),
        err_alt_pct=100.0 * (u / u_ref_alt - 1.0),
        pass_primary=abs(u / u_ref - 1.0) <= rc.GATE_PCT / 100.0,
        pass_alt=abs(u / u_ref_alt - 1.0) <= rc.GATE_PCT / 100.0,
    )
    return out


def collect() -> list[dict]:
    rows, seen = [], set()
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for cj in sorted(root.rglob("case.json")):
            d = cj.parent
            key = (json.loads(cj.read_text())["name"], str(d))
            if key in seen:
                continue
            seen.add(key)
            rows.append(evaluate(d))
    return rows


def _fmt(v, spec="{:.5f}"):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else spec.format(v)


def write_markdown(rows: list[dict]) -> str:
    lines = [
        "# CASE-RAYLEIGH-TAYLOR round 3 — re-grade against the literature-locked reference",
        "",
        f"Primary convention: C = {rc.C_2D} (2-D-specific source, `ref/ref_lock.py`); "
        f"alternate: C = {rc.C_2D_ALT} (arXiv:2009.14655 Eq. 1). "
        f"Gate: |error| <= {rc.GATE_PCT:.0f} % on the primary column.",
        "",
        "| case | scheme | A | lam | w | u_front | ref(C=3) | err(C=3) | pass | ref(C=1) | err(C=1) | sigma err | phi_max |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['case']} | {r['scheme']} | {r['atwood']:.3f} | {r['lam']} | "
            f"{_fmt(r['ac_width'], '{:.1f}')} | {_fmt(r['u_front'])} | {_fmt(r['u_ref_primary'])} | "
            f"{_fmt(r['err_primary_pct'], '{:+.1f}%')} | {'PASS' if r['pass_primary'] else 'FAIL'} | "
            f"{_fmt(r['u_ref_alt'])} | {_fmt(r['err_alt_pct'], '{:+.1f}%')} | "
            f"{_fmt(r['sigma_err_pct'], '{:+.1f}%')} | {_fmt(r['phi_max_peak'], '{:.2f}')} |")
    n = len(rows)
    np_ = sum(1 for r in rows if r["pass_primary"])
    na_ = sum(1 for r in rows if r["pass_alt"])
    lines += ["", f"Primary-convention passes: {np_}/{n}. Alternate-convention passes: {na_}/{n}.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json-only", action="store_true")
    args = ap.parse_args(argv)
    rows = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_code": "RAYLEIGH-TAYLOR",
        "C_primary": rc.C_2D,
        "C_alt": rc.C_2D_ALT,
        "convention_conflict": rc.CONVENTION_CONFLICT,
        "tau_window": [rc.TAU_LO, rc.TAU_HI],
        "gate_pct": rc.GATE_PCT,
        "n_rows": len(rows),
        "n_pass_primary": sum(1 for r in rows if r["pass_primary"]),
        "n_pass_alt": sum(1 for r in rows if r["pass_alt"]),
        "rows": rows,
    }
    (OUT / "regrade.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    if not args.json_only:
        (OUT / "regrade.md").write_text(write_markdown(rows) + "\n")
        print(write_markdown(rows))
    print(f"[round3] {len(rows)} rows -> {OUT}/regrade.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
