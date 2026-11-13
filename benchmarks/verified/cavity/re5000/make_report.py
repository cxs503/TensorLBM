"""Produce the CASE-CAVITY-RE5000 progress report: figures + markdown + report JSON.

Reads whatever the runners left in ``out_local/`` (never invents a number), draws the
Ghia comparison figures, and writes

* ``figures/cavity_re5000_*.png``
* ``docs/cavity-re5000-report-r2.md``   (six-section paper-style report, numbers injected)
* ``/tmp/report.json``                  (payload for POST /api/v1/tensor/reports)

Usage: python benchmarks/verified/cavity/re5000/make_report.py [--post]
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
REF = HERE / "reference" / "ghia_re5000_reference.json"
OUT = HERE / "out_local"
FIG = REPO / "figures"
DOC = REPO / "docs" / "cavity-re5000-report-r2.md"
TOL = 8.0


def load_runs() -> dict:
    runs = {}
    for f in sorted(glob.glob(str(OUT / "*.json"))):
        d = json.loads(Path(f).read_text())
        for key, g in (d.get("grids") or {}).items():
            g = dict(g)
            g["_file"] = os.path.basename(f)
            g["_cmp"] = (d.get("comparison") or {}).get(key)
            runs[f"{key}({os.path.basename(f)})"] = g
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", action="store_true")
    args = ap.parse_args()

    ref = json.loads(REF.read_text())
    u_ref = np.array(ref["u_re5000"], dtype=float)
    v_ref = np.array(ref["v_re5000"], dtype=float)
    runs = load_runs()
    FIG.mkdir(exist_ok=True)

    # ---------------- figure 1: centreline profiles --------------------------
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.2))
    ax[0].plot(u_ref[:, 1], u_ref[:, 0], "o", ms=4.5, mfc="none", label="Ghia 1982 (Re=5000)")
    ax[1].plot(v_ref[:, 1], v_ref[:, 0], "o", ms=4.5, mfc="none", label="Ghia 1982 (Re=5000)")
    for name, g in runs.items():
        if g.get("status") != "ok":
            continue
        ax[0].plot(g["u_centerline"], g["coord"], "-", lw=1.2, label=f"TensorLBM {name}")
        ax[1].plot(g["v_centerline"], g["coord"], "-", lw=1.2, label=f"TensorLBM {name}")
    ax[0].set_xlabel("u / U_lid"); ax[0].set_ylabel("y / L")
    ax[0].set_title("u along x/L = 0.5 (Ghia Table I)")
    ax[1].set_xlabel("v / U_lid"); ax[1].set_ylabel("x / L")
    ax[1].set_title("v along y/L = 0.5 (Ghia Table II)")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.tight_layout()
    f1 = FIG / "cavity_re5000_centerline.png"
    fig.savefig(f1, dpi=140)

    # ---------------- figure 2: error table as a scatter --------------------
    fig2, ax2 = plt.subplots(figsize=(6.4, 3.6))
    rows = []
    for name, g in runs.items():
        cmp_ = g.get("_cmp") or {}
        if "u_rows" not in cmp_:
            continue
        ue = np.array([abs(r[3]) for r in cmp_["u_rows"]])
        ve = np.array([abs(r[3]) for r in cmp_["v_rows"]])
        ax2.plot(u_ref[:, 0], ue, "o-", ms=3, lw=1, label=f"|Δu| {name}")
        ax2.plot(v_ref[:, 0], ve, "s--", ms=3, lw=1, label=f"|Δv| {name}")
        rows.append((name, cmp_))
    ax2.axhline(TOL, color="k", ls=":", lw=1, label=f"tolerance {TOL:.0f}% U_lid")
    ax2.set_xlabel("y/L (u rows) or x/L (v rows)"); ax2.set_ylabel("|error| [% U_lid]")
    ax2.set_title("pointwise error vs Ghia (Re=5000)"); ax2.grid(alpha=0.3); ax2.legend(fontsize=7)
    fig2.tight_layout()
    f2 = FIG / "cavity_re5000_error.png"
    fig2.savefig(f2, dpi=140)

    # ---------------- numbers ----------------------------------------------
    def table_md(cmp_: dict) -> str:
        lines = ["| y/L | u ref | u got | Δu %% | x/L | v ref | v got | Δv %% |",
                 "|---|---|---|---|---|---|---|---|"]
        for (yu, ur, ug, ue), (xv, vr, vg, ve) in zip(cmp_["u_rows"], cmp_["v_rows"]):
            lines.append(f"| {yu:.4f} | {ur:+.5f} | {ug:+.5f} | {ue:+.2f} | "
                         f"{xv:.4f} | {vr:+.5f} | {vg:+.5f} | {ve:+.2f} |")
        return "\n".join(lines)

    grid_summary = []
    for name, g in sorted(runs.items()):
        c = g.get("_cmp") or {}
        grid_summary.append(
            f"| {name} | {g['n']} | {g.get('tau', float('nan')):.5f} | {g['steps']} | "
            f"{'n/a' if g.get('final_resid') is None else f'{g[chr(102)+chr(105)+chr(110)+chr(97)+chr(108)+chr(95)+chr(114)+chr(101)+chr(115)+chr(105)+chr(100)]:.2e}'} | "
            f"{g['status']} | "
            f"{'n/a' if 'max_abs_err_pct' not in c else f'{c[chr(109)+chr(97)+chr(120)+chr(95)+chr(97)+chr(98)+chr(115)+chr(95)+chr(101)+chr(114)+chr(114)+chr(95)+chr(112)+chr(99)+chr(116)]:.2f}'} | "
            f"{'n/a' if 'pass' not in c else ('PASS' if c['pass'] else 'FAIL')} |")
    grid_md = "\n".join(["| run | n | tau | steps | final resid | status | max abs err [%] | vs 8% |",
                         "|---|---|---|---|---|---|---|---|"] + grid_summary) if grid_summary else \
        "| (no finished run yet) | | | | | | | |"

    best = max(rows, key=lambda r: r[1]["max_abs_err_pct"]) if rows else None
    err_block = ""
    if rows:
        for name, c in rows:
            err_block += f"\n### {name}: {c['max_abs_err_pct']:.2f} % of U_lid " \
                         f"(rmse {c['rmse_pct']:.2f} %)\n\n" + table_md(c) + "\n"
        verdict = ("**< 3 %?** No. The finest grid available in this mission's budget still "
                   "misses the 8 % case tolerance on at least one tabulated row, so the anchor is "
                   "**not** verified; see the honest-FAIL section."
                   if not all(c["pass"] for _, c in rows) else
                   "**< 3 %?** Check the per-row table: max / rmse are quoted above; the anchor's "
                   "own gate is 8 % of U_lid, and the 3 % question is answered row by row.")
    else:
        verdict = "**< 3 %?** Cannot be answered: no run has written a scored comparison yet."

    md = f"""# CASE-CAVITY-RE5000 (TensorLBM) — round 2: reference admitted, scoring harness live

Rerun of the anchor `CASE-CAVITY-RE5000` (lid-driven square cavity, Re = 5000, tolerance
8 % of the lid speed vs Ghia 1982). Round 1 (`docs/cavity-re5000-report.md`) ended
`WITHHELD`: no machine-readable copy of the paper's tables and no converged solution.

## ① Case, physics and parameters

2-D lid-driven square cavity, `L = 1` in lattice units, lid speed `U_lid = 0.1`
(`Ma = U/cs = 0.173`, compressibility error `O(Ma^2) ~ 3 %`), `Re = U_lid L / nu = 5000`.

| quantity | value |
|---|---|
| lattice / collision | D2Q9 / regularised BGK (`collide_rlbm`, Latt & Chopard 2006) |
| boundary | V3: pre-streaming half-way bounce-back on 3 walls + Zou/He moving lid |
| tau (equal-Re recipe `3 U L / Re + 0.5`) | n=128 → 0.50768, n=160 → 0.50960, n=384 → 0.52300 |
| grids used here | {", ".join(sorted(str(g["n"]) for g in runs.values())) or "none"} |
| reference | Ghia, Ghia & Shin, *J. Comput. Phys.* **48** (1982) 387-411, Table I (u, x/L=0.5) and Table II (v, y/L=0.5), Re = 5000 column, pp. 398-399 |
| tolerance (pre-registered) | pointwise `|ref - got| <= 8 % U_lid` on the 17 tabulated rows of each table |

## ② How it was done

1. **Reference admission (the blocking item of round 1) — now solved and auditable.**
   `benchmarks/verified/cavity/re5000/build_reference.py` takes three machine-read sources:
   * the paper's own OCR text layer (pp. 398-399), dumped mechanically with `pypdf`
     (`reference/ghia1982_tables_I_II.ocr.txt`, PDF sha256 `{ref["provenance"]["primary_witness"]["pdf_sha256"][:16]}…`);
   * two independent digitised copies of the same tables (`copy_tableI_u_gist.txt`,
     `copy_tableII_v_pinn.txt`) plus a third low-Re MATLAB copy;
   and gates the output on:
   * **witness gate** — every one of the {ref["provenance"]["verification"]["table_I_u"]["witness"]["cells_checked"] * 2} tabulated values must appear in the paper's OCR text
     ({ref["provenance"]["verification"]["table_I_u"]["witness"]["exact"] + ref["provenance"]["verification"]["table_II_v"]["witness"]["exact"]} exact digit matches,
     {len(ref["provenance"]["verification"]["table_I_u"]["witness"]["one_glyph_diff"]) + len(ref["provenance"]["verification"]["table_II_v"]["witness"]["one_glyph_diff"])} one-glyph OCR differences,
     {len(ref["provenance"]["verification"]["table_I_u"]["witness"]["untraceable"]) + len(ref["provenance"]["verification"]["table_II_v"]["witness"]["untraceable"])} untraceable);
   * **internal cross-check gate** — the Re = 100/400/1000 columns of the same copies against
     the library's own `GHIA_RE100/400/1000` tables (max |Δ| = 0 / 1.5e-1 / 5e-4, the two
     non-zero entries being documented errata);
   * **target-column gate** — the Re = 5000 column itself must contain no OCR ambiguity and no
     gross bracket violation.
   Output: `reference/ghia_re5000_reference.json`, sha256 `{ref["sha256"][:16]}…`, 17 + 17 rows,
   carrying source URLs, the PDF sha256 and the full gate log. **Zero hand transcription.**
2. **Solver** — the same code path already validated at Re = 100/400/1000
   (`benchmarks/verified/cavity/run.py`), driven by `re5000/run.py` with the reference supplied
   through `--ref-json`, so a missing/again-unavailable reference can only ever yield
   `WITHHELD`, never a silent pass.
3. **Budget** — `run.py --n … --steps …`; every run writes `result.json` with the full centreline
   profiles so the numbers below are re-derivable without re-running.

## ③ Results (programmatically injected from the run artefacts)

{grid_md}

{err_block}

## ④ Figures

* `figures/cavity_re5000_centerline.png` — u(y) at x/L=0.5 and v(x) at y/L=0.5 against the
  Ghia symbols.
* `figures/cavity_re5000_error.png` — pointwise |error| per tabulated row with the 8 % tolerance
  line.

## ⑤ Notes, defects and scope

* **Errata found while admitting the reference** (all outside the scored Re = 5000 column,
  reported rather than silently repaired — see `provenance.errata` in the JSON):
  * Table I, Re = 3200, y = 0.4531: OCR reads `-0.86636` while the bracketing rows are
    `-0.04272 … -0.24427` — a lost decimal point; quarantined, column not scored.
  * Table I, Re = 10000, y = 0.5: OCR token `0.03 111` has lost the sign; magnitude exported,
    sign quarantined, column not scored.
  * Table II, Re = 400, x = 0.9063 — the digitised copy itself flags this cell as
    "probably wrong".
  * **Library defect (not fixed here, needs its own mission):** `GHIA_RE400` stores
    `v(0.1563) = 0.28003` where the paper says `0.28124` (the 0.28003 is the Re = 10000 value at
    x = 0.2266), swaps x = 0.2266/0.2344, and repeats `-0.38598` at x = 0.8047 and x = 0.9063.
    The cross-check in `build_reference.py` prints these discrepancies.
* **Scope**: Re = 5000 at these grids is a *stress* configuration for LBM (tau ≈ 0.51, wall
  boundary layers thinner than a lattice spacing); the numbers above are therefore a progress
  report on the harness, not a claim that the anchor is verified.
* **Cluster**: the 384² production run is submitted (`cj1412`) and still queued at the time of
  writing; its artefacts are **not** part of this report.

## ⑥ Error verdict

{verdict}

Honest status of the anchor after this round: reference admission **solved** (machine-verified,
reproducible), scoring harness **live and exercised on real numbers**, converged fine-grid
solution **still pending** — so `CASE-CAVITY-RE5000` stays **open / not verified**.
"""

    DOC.write_text(md, encoding="utf-8")
    print(f"[ok] {DOC}")

    # ---------------- report payload ---------------------------------------
    def b64(p: Path) -> str:
        return base64.b64encode(p.read_bytes()).decode()

    figs = [{"name": "centerline", "caption": "u along x/L=0.5 and v along y/L=0.5 vs Ghia (1982) Re=5000",
             "png_base64": b64(f1)},
            {"name": "error", "caption": "pointwise |error| [% U_lid] vs Ghia Re=5000, tolerance 8 %",
             "png_base64": b64(f2)}]
    payload = {
        "title": "CASE-CAVITY-RE5000: Ghia 1982 Re=5000 reference admitted (machine-verified) + scoring harness exercised",
        "summary": "Round-2 progress on the lid-driven cavity Re=5000 anchor: the blocking reference gap "
                   "is closed with a zero-hand-transcription, gate-verified Ghia Table I/II Re=5000 "
                   "dataset (OCR witness + cross-copy + library cross-check), the run harness now scores "
                   "against it, and real coarse-grid numbers are reported honestly (anchor still open).",
        "scenario_key": "CASE-CAVITY-RE5000",
        "mission_id": 17072,
        "tags": ["TensorLBM", "cavity", "Re5000", "Ghia1982", "verification", "reference-admission"],
        "markdown": md,
        "figures": figs,
    }
    Path("/tmp/report.json").write_text(json.dumps(payload))
    print("[ok] /tmp/report.json", len(json.dumps(payload)) // 1024, "KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())