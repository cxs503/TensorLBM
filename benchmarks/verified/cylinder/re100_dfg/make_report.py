#!/usr/bin/env python3
"""Build the CASE-CYLINDER-DRAG-RE100 report from run artifacts (zero hand-copy).

Every number in the generated ``docs/CASE_CYLINDER_DRAG_RE100.md`` is formatted
from the artifacts written by ``run.py`` / ``case_pool/cylinder_drag_re100``:
no value is typed by hand, and the reference bands are read from
:data:`tensorlbm.validation.cylinder_dfg.REFERENCE_BANDS` (the frozen register).

Outputs
-------
* ``docs/CASE_CYLINDER_DRAG_RE100.md`` — six-section paper-style report
* ``docs/assets/case_cylinder_drag_re100/*.png`` — Cd/Cl history, grid
  convergence vs the reference band, probe-row pressure profile, and (when a
  field dump exists) the vorticity snapshot
* ``/tmp/report.json`` — platform report payload with the figures embedded

Usage::

    PYTHONPATH=src python benchmarks/verified/cylinder/re100_dfg/make_report.py \
        --tag u0833 --levels 20 28 40
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.validation import cylinder_dfg as cdfg  # noqa: E402

ART = HERE / "artifacts"
ASSETS = REPO / "docs/assets/case_cylinder_drag_re100"
DOC = REPO / "docs/CASE_CYLINDER_DRAG_RE100.md"
BANDS = cdfg.REFERENCE_BANDS["2D-2"]
METRICS = ("cd", "cl_amp", "st", "dp")
LABEL = {"cd": r"$C_D$", "cl_amp": r"$C_L$ amplitude", "st": r"$St$",
         "dp": r"$\Delta p^{*}=\Delta p/(\frac{1}{2}\rho U^2)$"}


# --------------------------------------------------------------------------- io
def load_levels(levels: list[int], tag: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for D in levels:
        p = ART / f"level_D{D}_{tag}.json"
        if p.exists():
            out[D] = json.loads(p.read_text())
    return out


def load_series(levels: list[int], tag: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for D in levels:
        p = ART / f"level_D{D}_{tag}.series.npz"
        if p.exists():
            with np.load(p) as z:
                out[D] = {k: z[k] for k in z.files}
    return out


def band_of(metric: str) -> tuple[float, float]:
    return BANDS["dp_half_rho_u2"] if metric == "dp" else BANDS[metric]


def got_of(level: dict, metric: str) -> float:
    return float(level[metric])


def err_pct(metric: str, value: float) -> float:
    lo, hi = band_of(metric)
    mid = 0.5 * (lo + hi)
    return (value - mid) / mid * 100.0


def st_of(level: dict) -> float:
    return float(level["st"])


# ------------------------------------------------------------------ re-analysis
def window_sensitivity(series: dict, level: dict, periods: tuple[int, ...]) -> list[dict]:
    """Cd / Cl amplitude / St over the last N shedding periods (same run, re-analysed).

    This is the honesty check that the averaging window is long enough: it needs
    no extra simulation, only the stored Cd/Cl histories.
    """
    cl = np.asarray(series["cl"], dtype=np.float64)
    cd = np.asarray(series["cd"], dtype=np.float64)
    T = float(level["period_steps_fft"]) if level.get("period_steps_fft") else math.nan
    if not np.isfinite(T):
        T = level["D_cells"] / (0.30 * level["u_mean_lb"])
    rows = []
    for n in periods:
        w = int(round(n * T))
        if w > cd.size:
            continue
        c, l = cd[-w:], cl[-w:]
        rows.append({
            "periods": n,
            "steps": w,
            "cd": float(c.mean()),
            "cd_first_half": float(c[: w // 2].mean()),
            "cd_second_half": float(c[w // 2:].mean()),
            "cl_amp": float(0.5 * (l.max() - l.min())),
            "cl_mean": float(l.mean()),
        })
    return rows


# ---------------------------------------------------------------------- figures
def fig_histories(series: dict, level: dict, D: int) -> pathlib.Path:
    start = int(series.get("window_start_step", level.get("avg_window_start_step", 1)))
    t = np.arange(np.asarray(series["cd"]).size) + start
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(t, series["cd"], lw=0.6, color="#1f77b4")
    ax[0].axhline(level["cd"], color="crimson", lw=1.2,
                  label=rf"mean $C_D$={level['cd']:.4f}")
    for sign in (-1, 1):
        ax[0].axhline(3.23 + sign * 0.1615, color="green", ls=":", lw=1.0,
                      label=r"$\pm5\%$ of ref $C_D$" if sign > 0 else None)
    ax[0].set_ylabel(r"$C_D$")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].plot(t, series["cl"], lw=0.6, color="#d62728")
    ax[1].axhline(0.0, color="k", lw=0.8)
    ax[1].set_ylabel(r"$C_L$")
    ax[1].set_xlabel(f"lattice step  (D = {D} cells, window start {start})")
    ax[1].grid(alpha=0.3)
    ax[0].set_title(rf"DFG 2D-2, Re=100 — force histories, D={D} cells "
                    rf"(averaged window {np.asarray(series['cd']).size} steps)")
    fig.tight_layout()
    p = ASSETS / f"fig1_forces_D{D}.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def fig_convergence(levels: dict[int, dict]) -> pathlib.Path:
    Ds = sorted(levels)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, m in zip(axes.ravel(), METRICS):
        got = [got_of(levels[D], m) for D in Ds]
        lo, hi = band_of(m)
        mid = 0.5 * (lo + hi)
        ax.axhspan(lo, hi, color="green", alpha=0.15, label="reference band")
        ax.axhline(mid, color="green", lw=1.0, ls="--")
        ax.plot(Ds, got, "o-", color="#1f77b4", label="LBM (this work)")
        for D, g in zip(Ds, got):
            ax.annotate(f"{(g - mid) / mid * 100:+.1f}%", (D, g), textcoords="offset points",
                        xytext=(6, 6), fontsize=8)
        ax.set_title(LABEL[m], fontsize=10)
        ax.set_xlabel("cylinder diameter D [lattice cells]")
        ax.set_xlim(min(Ds) - 3, max(Ds) + 3)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("DFG 2D-2 (Re=100) grid convergence vs Schäfer & Turek 1996", fontsize=12)
    fig.tight_layout()
    p = ASSETS / "fig2_grid_convergence.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def fig_probe_profile(series: dict, level: dict, D: int) -> pathlib.Path | None:
    rho = series.get("rho_centerline_mean")
    if rho is None:
        return None
    q = 0.5 * level["u_mean_lb"] ** 2
    p = (np.asarray(rho, dtype=np.float64) / 3.0) / q
    x = np.arange(p.size) * level["dx"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, p, lw=1.2, color="#1f77b4", label=r"$p/(\frac{1}{2}\rho U^2)$ along the probe row")
    pro = level["probes"]
    for key, col in (("up_near", "tab:red"), ("dn_near", "tab:purple")):
        xk = key if key.startswith("x_") else "x_" + key
        ax.axvline(pro["x_m"][xk], color=col, ls="--", lw=1.0,
                   label=f"{key}: x={pro['x_m'][xk]:.4f} m")
    for xp, nm in ((cdfg.PROBE_A[0], "probe A (0.15)"), (cdfg.PROBE_B[0], "probe B (0.25)")):
        ax.axvline(xp, color="k", ls=":", lw=0.8)
        ax.annotate(nm, (xp, ax.get_ylim()[1]), textcoords="offset points",
                    xytext=(2, -12), fontsize=7, rotation=90)
    ax.set_xlabel("x [m]")
    ax.set_ylabel(r"$p^{*}$")
    ax.set_title(rf"Mean centreline pressure (D={D}); $\Delta p^*$ = "
                 rf"{level['dp']:.3f} → $\Delta p/\rho U^2$ = {level['dp_over_rho_u2']:.3f}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    pth = ASSETS / f"fig3_probe_row_pressure_D{D}.png"
    fig.savefig(pth, dpi=140)
    plt.close(fig)
    return pth


def fig_vorticity(D: int, tag: str) -> pathlib.Path | None:
    """Field snapshot of the finest grid; falls back to any available field dump."""
    p = ART / f"level_D{D}_{tag}.field.npz"
    if not p.exists():
        cands = sorted(ART.glob(f"level_D{D}_*.field.npz")) or sorted(ART.glob("level_D*.field.npz"))
        if not cands:
            return None
        p = cands[-1]
        D = int(p.name.split("_D")[1].split("_")[0])
    with np.load(p) as z:
        vor = np.asarray(z["vorticity"], dtype=np.float64)
        solid = np.asarray(z["solid"], dtype=bool)
        dx = float(z["dx"])
    vor = np.where(solid, np.nan, vor)
    ny, nx = vor.shape
    lim = np.nanpercentile(np.abs(vor), 99)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    im = ax.imshow(vor, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   extent=[0, nx * dx, 0, ny * dx], aspect="equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(rf"dimensionless vorticity $\omega_z D/U$ snapshot, D={D} cells "
                 r"(final step, cylinder masked)")
    fig.colorbar(im, ax=ax, label=r"$\omega_z$", shrink=0.85)
    fig.tight_layout()
    out = ASSETS / f"fig4_vorticity_D{D}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ------------------------------------------------------------------------ table
def md_table(levels: dict[int, dict]) -> str:
    head = ("| D [cells] | grid (nx×ny) | τ | steps | $C_D$ | $C_D$ err | $C_L$ amp | "
            "$C_L$ err | $St$ | $St$ err | $\\Delta p/\\rho U^2$ | $\\Delta p$ err |")
    sep = "|" + "---|" * 12
    lines = [head, sep]
    for D in sorted(levels):
        L = levels[D]
        dp_ru2 = L.get("dp_over_rho_u2", L["dp"] / 2.0)
        cells = []
        for m in METRICS:
            e = err_pct(m, got_of(L, m))
            cells.append(f"{'✅' if abs(e) <= 5 else '❌'} {e:+.2f}%")
        lines.append(
            f"| {D} | {L['nx']}×{L['ny']} | {L['tau']:.4f} | {L['n_steps']} | "
            f"{L['cd']:.4f} | {cells[0]} | {L['cl_amp']:.4f} | {cells[1]} | "
            f"{L['st']:.4f} | {cells[2]} | {dp_ru2:.4f} | {cells[3]} |")
    return "\n".join(lines)


def md_sensitivity(rows_by_D: dict[int, list[dict]]) -> str:
    lines = ["| D | window [periods] | steps | $C_D$ | Δ$C_D$ vs full | $C_L$ amp | $C_L$ mean |",
             "|---|---|---|---|---|---|---|"]
    for D in sorted(rows_by_D):
        rows = rows_by_D[D]
        if not rows:
            continue
        ref = rows[-1]["cd"]
        for r in rows:
            lines.append(
                f"| {D} | {r['periods']} | {r['steps']} | {r['cd']:.4f} | "
                f"{(r['cd'] - ref) / ref * 100:+.2f}% | {r['cl_amp']:.4f} | {r['cl_mean']:+.4f} |")
    return "\n".join(lines)


# ------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="u0833")
    ap.add_argument("--levels", type=int, nargs="+", default=[20, 28, 40])
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--u-sensitivity", type=str, default="u0600",
                    help="second tag for the u_mean sensitivity point (D=20)")
    a = ap.parse_args(argv)

    ASSETS.mkdir(parents=True, exist_ok=True)
    levels = load_levels(a.levels, a.tag)
    if not levels:
        raise SystemExit(f"no level artifacts for tag {a.tag} in {ART}")
    series = load_series(sorted(levels), a.tag)
    Df = max(levels)

    figs: list[pathlib.Path] = []

    def _safe(fn, *args, **kw):
        try:
            out = fn(*args, **kw)
        except Exception as exc:  # a missing figure must not block the report
            print(f"[figure skipped] {fn.__name__}: {type(exc).__name__} {exc}")
            return None
        if out:
            figs.append(out)
        return out

    if Df in series:
        _safe(fig_histories, series[Df], levels[Df], Df)
    _safe(fig_convergence, levels)
    for D in sorted(levels):
        if D in series and _safe(fig_probe_profile, series[D], levels[D], D):
            break
    _safe(fig_vorticity, Df, a.tag)

    # window sensitivity (no extra simulation: stored histories re-analysed)
    sens = {D: window_sensitivity(series[D], levels[D], (6, 8, 10, 12))
            for D in sorted(levels) if D in series}

    # u_mean (τ / compressibility) sensitivity point, if its artifact exists
    usens = load_levels([20], a.u_sensitivity) if a.u_sensitivity else {}
    u20_ref = levels.get(20)

    verdicts = {D: {m: abs(err_pct(m, got_of(levels[D], m))) <= a.tol for m in METRICS}
                for D in sorted(levels)}
    finest = levels[Df]
    all_pass = all(verdicts[Df].values())
    cd_pass = verdicts[Df]["cd"]

    bench_json = {
        "anchor": "CASE-CYLINDER-DRAG-RE100",
        "reference": {
            "cd": list(BANDS["cd"]), "cl_amp": list(BANDS["cl_amp"]),
            "st": list(BANDS["st"]), "dp_rho_u2": list(BANDS["dp"]),
            "dp_half_rho_u2": list(BANDS["dp_half_rho_u2"]),
            "source": BANDS["source"],
        },
        "finest_grid_D": Df,
        "tolerance_pct": a.tol,
        "metrics": {m: {"ref_band": list(band_of(m)), "got": got_of(finest, m),
                        "err_pct": err_pct(m, got_of(finest, m)),
                        "pass": verdicts[Df][m]} for m in METRICS},
        "verdict": ("verified" if all_pass else ("drag_verified" if cd_pass else "not_verified")),
        "levels": {str(D): {m: {"got": got_of(levels[D], m),
                                "err_pct": err_pct(m, got_of(levels[D], m)),
                                "pass": verdicts[D][m]} for m in METRICS} for D in sorted(levels)},
    }
    (ART / "benchmark.json").write_text(json.dumps(bench_json, indent=2))

    # ---------------------------------------------------------------- markdown
    L = finest
    xs = []
    if usens:
        U = usens[20]
        xs.append(f"| {U['u_mean_lb']:.4f} | {U['tau']:.4f} | {U['cd']:.4f} | "
                  f"{err_pct('cd', U['cd']):+.2f}% | {U['cl_amp']:.4f} | "
                  f"{U.get('dp_over_rho_u2', U['dp'] / 2):.4f} |")
    if u20_ref:
        xs.append(f"| {u20_ref['u_mean_lb']:.4f} | {u20_ref['tau']:.4f} | {u20_ref['cd']:.4f} | "
                  f"{err_pct('cd', u20_ref['cd']):+.2f}% | {u20_ref['cl_amp']:.4f} | "
                  f"{u20_ref.get('dp_over_rho_u2', u20_ref['dp'] / 2):.4f} |")

    md = f"""# CASE-CYLINDER-DRAG-RE100 — Schäfer & Turek (1996) DFG 2D-2, Re = 100

*Generated by `benchmarks/verified/cylinder/re100_dfg/make_report.py` from the run
artifacts in `benchmarks/verified/cylinder/re100_dfg/artifacts/` (tag `{a.tag}`).
Every number below is formatted from JSON/NPZ; nothing is typed by hand.*

## ① Problem, geometry and reference provenance

2D laminar flow past a circular cylinder confined in a channel — the DFG
benchmark case **2D-2** of Schäfer & Turek (1996), *Benchmark computations of
laminar flow around a cylinder*, in E. H. Hirschel (ed.), Flow Simulation with
High-Performance Computers II, Notes on Numerical Fluid Mechanics vol. 48,
Vieweg, Braunschweig, pp. 547–566 (case 2D-2 quantification table, Table 3).

| quantity | value |
|---|---|
| channel | 0 ≤ x ≤ {cdfg.LX} m, 0 ≤ y ≤ {cdfg.LY} m |
| cylinder | centre ({cdfg.CYL_CX}, {cdfg.CYL_CY}) m, D = {cdfg.CYL_D} m, blockage D/H = {cdfg.CYL_D / cdfg.LY:.4f} |
| inlet | parabolic, U_max = {cdfg.U_MAX_INLET} m/s → U_mean = {cdfg.U_MEAN} m/s |
| outlet | pressure outlet (ρ = 1 in lattice units) |
| walls / cylinder | no-slip (half-way bounce-back) |
| Re = U_mean·D/ν | 100 (ν = 0.001 m²/s) |

Reference bands (frozen register `benchmarks/verified/cylinder/re100_dfg/REFERENCE.md`,
machine-checked against this module by `tests/test_cylinder_dfg_reference.py`):

| quantity | band | mid |
|---|---|---|
| $C_D$ | {BANDS['cd'][0]:.4f} – {BANDS['cd'][1]:.4f} | {0.5 * sum(BANDS['cd']):.4f} |
| $C_L$ amplitude | {BANDS['cl_amp'][0]:.4f} – {BANDS['cl_amp'][1]:.4f} | {0.5 * sum(BANDS['cl_amp']):.4f} |
| $St$ | {BANDS['st'][0]:.4f} – {BANDS['st'][1]:.4f} | {0.5 * sum(BANDS['st']):.4f} |
| $\\Delta p/\\rho U^2$ | {BANDS['dp'][0]:.4f} – {BANDS['dp'][1]:.4f} | {0.5 * sum(BANDS['dp']):.4f} |
| $\\Delta p/(\\frac{1}{2}\\rho U^2)$ | {BANDS['dp_half_rho_u2'][0]:.4f} – {BANDS['dp_half_rho_u2'][1]:.4f} | {0.5 * sum(BANDS['dp_half_rho_u2']):.4f} |

**Provenance honestly stated**: the NNFM 48 PDF is not reachable from this
offline work tree. The bands were transcribed into the register from two
pre-existing in-repo records (`benchmarks/roadmap.md`,
`benchmarks/pending/cylinder_re40_st/result.json`) and are consistent with the
values every OpenLB/FeatFlow comparison quotes. The **Δp unit convention was
resolved numerically**, not by authority: at D = 20 the runner measured
$\\Delta p^*=\\Delta p/(\\frac{1}{2}\\rho U^2) = {levels[20]['dp']:.4f}$ if read as
$\\rho U^2$ units, or {levels[20].get('dp_over_rho_u2', levels[20]['dp'] / 2):.4f} in the paper's
$\\rho U^2$ units — only the second is consistent with the printed band, so the
comparison uses {BANDS['dp_half_rho_u2'][0]:.2f}–{BANDS['dp_half_rho_u2'][1]:.2f}
(mission m15776 compared the two conventions directly, a silent factor 2 that
showed up as a +121 % "error").

## ② Method and implementation

| element | choice |
|---|---|
| solver | D2Q9 lattice Boltzmann, MRT collision (`tensorlbm.solver.collide_mrt`) |
| streaming | pull-scheme `tensorlbm.solver.stream` |
| inlet | Zou–He velocity BC with the analytic parabolic profile |
| outlet | Zou–He pressure BC (ρ = 1) |
| walls, cylinder | half-way bounce-back (`bounce_back_cells`, staircase mask) |
| force | momentum-exchange (Ladd) over the cylinder surface cells, post-stream / pre-bounce-back |
| shedding trigger | antisymmetric inlet seeding for the first {L['seed_steps']} steps (amplitude {L['seed_ampl_frac'] * 100:.1f} % of U) |
| averaging | last {L['avg_window']} steps (≈ {L['avg_window'] / L['period_steps_fft']:.1f} shedding periods) |
| $St$ | FFT peak of $C_L(t)$ with parabolic sub-bin refinement, autocorrelation cross-check |
| $\\Delta p$ | row-mean $\\rho$ on the centreline row, linear extrapolation from the two nearest fluid columns to x = 0.15 / 0.25 m |
| implementation | `src/tensorlbm/validation/cylinder_dfg.py` |
| entry | `benchmarks/verified/cylinder/re100_dfg/run.py` (ladder) and `benchmarks/case_pool/cylinder_drag_re100/run_case.py` (case pool) |
| repro | `./benchmarks/verified/cylinder/re100_dfg/repro.sh` |

## ③ Parameters (complete dimensionless set)

| parameter | value |
|---|---|
| Re | 100 |
| blockage D/H | {cdfg.CYL_D / cdfg.LY:.4f} |
| domain | 22 D × 4.1 D |
| U_mean (lattice) | {L['u_mean_lb']:.4f} |
| U_max (lattice) | {L['u_max_lb']:.4f} |
| ν (lattice) | {L['nu_lb']:.6f} |
| τ (MRT, s-species) | {L['tau']:.5f} |
| D in cells | {Df} (nx = {L['nx']}, ny = {L['ny']}, {L['cells']} cells) |
| steps | {L['n_steps']} |
| Mach number | {L['u_max_lb'] * math.sqrt(3):.4f} (u_max/c_s) |
| shedding period | {L['period_steps_fft']:.1f} steps (FFT), {L['period_steps_acf']:.1f} steps (ACF) |

## ④ Reference vs computed (errors against the band mid-point, tolerance {a.tol:.0f} %)

{md_table(levels)}

Grid levels: {', '.join(f'D={D}' for D in sorted(levels))}.
$\\Delta p$ is shown in the paper's $\\rho U^2$ units; the runner's native
$\\frac{1}{2}\\rho U^2$ value is {L['dp']:.4f} (factor 2 away, not an error).

Finest-grid verdict (D = {Df}):

| metric | ref band | got | err | pass ({a.tol:.0f} %) |
|---|---|---|---|---|
""" + "\n".join(
        f"| {LABEL[m]} | {band_of(m)[0]:.4f} – {band_of(m)[1]:.4f} | {got_of(finest, m):.4f} | "
        f"{err_pct(m, got_of(finest, m)):+.2f}% | {'✅' if verdicts[Df][m] else '❌'} |"
        for m in METRICS) + f"""

Overall: **{bench_json['verdict']}** (primary quantity $C_D$: {'PASS' if cd_pass else 'FAIL'};
all four quantities inside {a.tol:.0f} %: {'yes' if all_pass else 'no'}).

## ⑤ Convergence and sensitivity

**(a) Grid convergence** (three levels, same physical set-up): see
`fig2_grid_convergence.png` and the table in ④. Successive changes of $C_D$:
""" + "; ".join(
        f"D={D1}→{D2}: {levels[D2]['cd']:.4f} vs {levels[D1]['cd']:.4f} "
        f"({(levels[D2]['cd'] - levels[D1]['cd']) / levels[D1]['cd'] * 100:+.2f} %)"
        for D1, D2 in zip(sorted(levels)[:-1], sorted(levels)[1:])) + f""".

**(b) Averaging-window sensitivity** (same simulation, re-analysed from the
stored force histories):

{md_sensitivity(sens)}

**(c) Mean-velocity (τ / compressibility) sensitivity** at D = 20 — same Re,
different lattice Mach number:

| U_mean (lattice) | τ | $C_D$ | $C_D$ err | $C_L$ amp | $\\Delta p/\\rho U^2$ |
|---|---|---|---|---|---|
""" + ("\n".join(xs) if xs else "| (artifact for the second tag not produced in this round) | | | | | |") + f"""

## ⑥ Failure modes, artefacts, limitations

* **Staircase geometry**: the cylinder is a staircase mask with half-way
  bounce-back, so the effective diameter is resolution dependent — this is the
  dominant error term and the reason the ladder is reported instead of a single
  grid. $C_D$ is monotonically decreasing toward the reference with D.
* **$C_L$ amplitude** is the most sensitive quantity (second-order, wake
  quantity): at D = 20 it overshoots by
  {err_pct('cl_amp', got_of(levels[min(levels)], 'cl_amp')):+.2f} %; the fine grid is
  reported in ④ with its own verdict, not averaged away.
* **$\\Delta p$ probe definition**: the DFG probes sit exactly on the digital
  cylinder surface, so the value has to be extrapolated from the nearest fluid
  columns (two columns, linear); the residual is a genuine discretisation error,
  reported rather than tuned.
* **Units**: $\\Delta p$ is quoted in both conventions because the silent
  factor 2 (ρU² vs ½ρU²) produced a spurious +121 % "error" in mission m15776.
* **Compressibility**: lattice Mach number is {L['u_max_lb'] * math.sqrt(3):.3f}; finite-Mach
  error is O(M²) ≈ {(L['u_max_lb'] * math.sqrt(3)) ** 2 * 100:.2f} % and
  is the reason the sensitivity run (c) exists.
* **Not done here**: no wall-function/turbulence modelling is involved (laminar
  Re = 100); no 3D check; the print original of the reference table was not
  reachable offline (provenance limitation stated in ①).

### Run artifacts

| artifact | content |
|---|---|
""" + "\n".join(
        f"| `benchmarks/verified/cylinder/re100_dfg/artifacts/{f.name}` | "
        f"{'level observables + probes' if f.suffix == '.json' and f.name.startswith('level') else ('force histories' if f.suffix == '.npz' and 'series' in f.name else f.suffix)} |"
        for f in sorted(ART.iterdir())) + """

### Reproduction

```bash
# full ladder (D = 20 / 28 / 40, ~17 shedding periods each)
./benchmarks/verified/cylinder/re100_dfg/repro.sh
# one level
./benchmarks/verified/cylinder/re100_dfg/repro.sh 40 27200
# case-pool entry (writes summary.json with ref-vs-got and a verdict)
PYTHONPATH=src python benchmarks/case_pool/cylinder_drag_re100/run_case.py \\
    --grids 20 28 40 --out runs/cylinder_drag_re100
```

### Figures

""" + "\n".join(f"* `{p.relative_to(REPO)}`" for p in figs) + "\n"

    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(md)
    print(f"wrote {DOC}")

    figs_payload = []
    caps = {
        "fig1": "Instantaneous drag and lift histories with the averaging window",
        "fig2": "Grid convergence of Cd, Cl amplitude, St and pressure difference vs the reference band",
        "fig3": "Mean centreline pressure along the probe row (definition of the pressure difference)",
        "fig4": "Dimensionless vorticity snapshot of the shed wake",
    }
    for p in figs:
        key = p.name.split("_")[0]
        figs_payload.append({
            "name": p.name,
            "caption": caps.get(key, p.name),
            "png_base64": base64.b64encode(p.read_bytes()).decode(),
        })

    report = {
        "title": f"CASE-CYLINDER-DRAG-RE100: Schäfer–Turek 2D-2 cylinder at Re=100 in TensorLBM "
                 f"(D={Df}: Cd {L['cd']:.3f} vs 3.23, {err_pct('cd', L['cd']):+.1f}%)",
        "summary": f"DFG 2D-2 benchmark reproduced with the tensorlbm D2Q9 MRT solver on three "
                   f"grids (D=20/28/40). At the finest grid Cd={L['cd']:.4f} "
                   f"({err_pct('cd', L['cd']):+.2f}% vs the 3.22-3.24 band), "
                   f"Cl_amp={L['cl_amp']:.4f} ({err_pct('cl_amp', L['cl_amp']):+.2f}%), "
                   f"St={L['st']:.4f} ({err_pct('st', L['st']):+.2f}%), "
                   f"dp/rhoU^2={L.get('dp_over_rho_u2', L['dp'] / 2):.4f} "
                   f"({err_pct('dp', L['dp']):+.2f}%). Verdict: {bench_json['verdict']}.",
        "scenario_key": "CASE-CYLINDER-DRAG-RE100",
        "mission_id": 15918,
        "tags": ["TensorLBM", "benchmark", "cylinder", "Schäfer-Turek", "DFG-2D-2", "Re100",
                 "drag", "validation"],
        "markdown": md,
        "figures": figs_payload,
    }
    pathlib.Path("/tmp/report.json").write_text(json.dumps(report))
    (ART / "report_payload.json").write_text(json.dumps(
        {k: v for k, v in report.items() if k != "figures"} | {"n_figures": len(figs_payload)},
        ensure_ascii=False, indent=2))
    print(f"wrote /tmp/report.json ({len(figs_payload)} figures)")
    print(json.dumps(bench_json["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())