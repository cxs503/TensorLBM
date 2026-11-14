"""Turbulent-channel gate reproduction + normalisation audit (T3 slot 4).

The ``channel_turb_re180`` slot of the T3 validation matrix is an honest FAIL on
the DNS ``u+`` gate (best level 5.33% > 3%).  This module does three things that
the existing case analysis (``mkm_gates_numbers.json`` / ``gates_verified.json``)
does not:

1. **Reproduces the recorded gate number from its own raw evidence.**  The
   recorded value is re-derived from the frozen L3 statistics plus the frozen
   Moser-Kim-Mansour table with an independently written interpolation and RMS,
   and the delta against the recorded ``rms_y30_100_pct`` is a machine-checked
   criterion (``|delta| <= 0.01`` percentage points).  If the pipeline changes,
   this breaks instead of drifting silently.
2. **Declares the metric convention and shows its sensitivity.**  The recorded
   5.33% turns out to be a *relative* RMS taken in the **target** friction-velocity
   normalisation (``u_tau = sqrt(target_wall_shear)``), while the simulation's
   measured ``u_tau`` is 7.64% low.  ``u+`` is conventionally normalised by the
   measured friction velocity, so the same field gives a materially larger error
   under that definition.  Both numbers are reported; the matrix entry must not
   be read as a convention-free "5.33%".
3. **Momentum-budget linearity diagnostic.**  For a fully developed plane
   channel the resolved total shear ``nu*dU/dy - <u'v'>`` must be linear in y
   and vanish at the centreline.  The residual is reported per node range: the
   near-wall derivative estimate is what degrades, not the outer balance, so this
   is a diagnostic of where the statistics are trustworthy, not a verdict.

Nothing here upgrades the slot: ``channel_turb_re180`` remains failed.  Output is
frozen next to the case with sha256 provenance for every input.

Usage::

    python benchmarks/validation_matrix/channel_budget.py            # print + write
    python benchmarks/validation_matrix/channel_budget.py --check    # verify frozen output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = REPO_ROOT / "benchmarks" / "pending" / "channel_turb_3d"
L3_RESULT = CASE_DIR / "runs_3d" / "L3_result.json"
DNS_CSV = CASE_DIR / "dns_ref" / "mkm180.csv"
LADDER = CASE_DIR / "validation_report.json"
OUT = CASE_DIR / "gate_reproduction.json"

WINDOW_Y_PLUS = (30.0, 100.0)
REPRO_TOL_PP = 0.01  # percentage points


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dns(path: Path = DNS_CSV) -> tuple[list[float], list[float]]:
    """Return (y_plus, u_plus) from the frozen Moser-Kim-Mansour table."""
    yp: list[float] = []
    up: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("y,"):
            continue
        parts = [p for p in line.replace(",", " ").split() if p]
        if len(parts) < 3:
            continue
        yp.append(float(parts[1]))
        up.append(float(parts[2]))
    return yp, up


def interp(x: float, xs: list[float], ys: list[float]) -> float:
    """Linear interpolation with clamped ends (the table is monotone in y+)."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    w = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] * (1.0 - w) + ys[hi] * w


def uplus_error(
    u: list[float],
    *,
    nu: float,
    u_tau: float,
    wall_offset: float,
    dns_yp: list[float],
    dns_up: list[float],
    window: tuple[float, float] = WINDOW_Y_PLUS,
    mode: str = "relative",
) -> dict:
    """u+ error vs DNS over the window for one (u_tau, wall-origin) convention.

    ``mode="relative"`` is the convention that reproduces the recorded case
    number: ``sqrt(mean(((u+_sim - u+_dns) / u+_dns)^2)) * 100``.
    ``mode="absolute"`` is the absolute RMS normalised by the mean DNS u+.
    """
    rows = []
    for i, ui in enumerate(u):
        y_plus = (i - wall_offset) * u_tau / nu
        if window[0] <= y_plus <= window[1]:
            rows.append((y_plus, ui / u_tau))
    if len(rows) < 3:
        return {"n_points": len(rows), "rms_pct": float("nan")}

    refs = [interp(y, dns_yp, dns_up) for y, _ in rows]
    diffs = [u_plus - ref for (_, u_plus), ref in zip(rows, refs)]
    ref_mean = sum(refs) / len(refs)
    if mode == "relative":
        rms_rel = math.sqrt(sum((d / ref) ** 2 for d, ref in zip(diffs, refs)) / len(diffs))
        rms_u_plus = rms_rel * ref_mean
        rms_pct = 100.0 * rms_rel
    else:
        rms_u_plus = math.sqrt(sum(d * d for d in diffs) / len(diffs))
        rms_pct = 100.0 * rms_u_plus / ref_mean
    return {
        "mode": mode,
        "n_points": len(rows),
        "rms_u_plus": rms_u_plus,
        "rms_pct": rms_pct,
        "y_plus_range": [rows[0][0], rows[-1][0]],
    }


def budget_linearity(l3: dict) -> dict:
    """Resolved total-shear linearity over declared node ranges (diagnostic)."""
    stats = l3["statistics"]
    u = [float(v) for v in stats["mean_velocity_profile"]]
    uv = [float(v) for v in stats["reynolds_stress_profiles_uu_vv_ww_uv"][3]]
    nu = float(l3["derived"]["nu"])
    ny = int(l3["configuration"]["ny"])
    if not (len(u) == len(uv) == ny):
        raise ValueError(f"profile length mismatch: {len(u)}/{len(uv)} vs ny={ny}")

    tau = {i: nu * 0.5 * (u[i + 1] - u[i - 1]) - uv[i] for i in range(1, ny - 1)}
    centre = (ny - 1) / 2.0
    out = {}
    for label, lo, hi in (("full[1,ny-2]", 1, ny - 2),
                          ("outer[4,ny-5]", 4, ny - 5),
                          ("core[10,ny-11]", 10, ny - 11)):
        xs = list(range(lo, hi + 1))
        ys = [tau[i] for i in xs]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        a = my - b * mx
        resid = max(abs(y - (a + b * x)) for x, y in zip(xs, ys))
        out[label] = {
            "nodes": [lo, hi],
            "tau_w_fit": a,
            "u_tau_fit": math.sqrt(a) if a > 0 else float("nan"),
            "zero_crossing_node": -a / b if b else float("nan"),
            "centre_node": centre,
            "max_abs_residual": resid,
            "residual_pct_of_tau_w": abs(100.0 * resid / a) if a > 0 else float("nan"),
        }
    return {
        "definition": "tau_total(y) = nu*dU/dy - <u'v'> must be linear and vanish at the centreline",
        "ranges": out,
        "verdict": (
            "diagnostic only: the outer/core balance closes to a few percent, the full-range "
            "residual is dominated by one-sided near-wall derivative estimates, so this range "
            "must not be used as a mesh-quality gate"
        ),
    }


def build() -> dict:
    l3 = json.loads(L3_RESULT.read_text(encoding="utf-8"))
    ladder = json.loads(LADDER.read_text(encoding="utf-8"))
    dns_yp, dns_up = load_dns()
    stats = l3["statistics"]
    nu = float(l3["derived"]["nu"])
    u = [float(v) for v in stats["mean_velocity_profile"]]
    u_tau_measured = float(stats["stationarity_friction_velocity"])
    u_tau_target = math.sqrt(float(l3["derived"]["target_wall_shear"]))

    l3_row = next((g for g in ladder["grid_levels"] if g.get("level") == "L3"), None)
    if l3_row is None:
        raise ValueError("validation_report.json has no L3 grid level")
    recorded_rms = float(l3_row["rms_y30_100_pct"])

    # The recorded gate convention: relative RMS, wall origin 0.5, TARGET u_tau.
    reproduced = uplus_error(u, nu=nu, u_tau=u_tau_target, wall_offset=0.5,
                             dns_yp=dns_yp, dns_up=dns_up, mode="relative")
    delta_pp = abs(reproduced["rms_pct"] - recorded_rms)
    reproduction_ok = bool(delta_pp <= REPRO_TOL_PP)

    conventions = {}
    for label, off, ut in (
        ("rel_offset0.5_target_u_tau", 0.5, u_tau_target),
        ("rel_offset0.5_measured_u_tau", 0.5, u_tau_measured),
        ("rel_offset0.1_measured_u_tau", 0.1, u_tau_measured),
        ("abs_offset0.5_target_u_tau", 0.5, u_tau_target),
        ("abs_offset0.5_measured_u_tau", 0.5, u_tau_measured),
    ):
        mode = "absolute" if label.startswith("abs") else "relative"
        conventions[label] = {
            "wall_offset": off,
            "u_tau": ut,
            **uplus_error(u, nu=nu, u_tau=ut, wall_offset=off,
                          dns_yp=dns_yp, dns_up=dns_up, mode=mode),
        }

    ratio = conventions["rel_offset0.5_measured_u_tau"]["rms_pct"] / conventions["rel_offset0.5_target_u_tau"]["rms_pct"]
    friction_gap = 100.0 * (u_tau_measured / u_tau_target - 1.0)
    return {
        "schema": "tensorlbm-t3-channel-gate-reproduction-v1",
        "case": "channel_turb_re180",
        "slot": "channel",
        "scope": (
            "Independent re-derivation of the recorded T3 gate number from the frozen L3 statistics "
            "and the frozen MKM-1999 table, plus the normalisation sensitivity of that number. "
            "The slot verdict is unchanged: honest FAIL on the 3% DNS u+ gate."
        ),
        "inputs": {
            "l3_result": {"path": str(L3_RESULT.relative_to(REPO_ROOT)),
                          "sha256": sha256_file(L3_RESULT)},
            "ladder": {"path": str(LADDER.relative_to(REPO_ROOT)),
                       "sha256": sha256_file(LADDER)},
            "dns_table": {
                "path": str(DNS_CSV.relative_to(REPO_ROOT)),
                "sha256": sha256_file(DNS_CSV),
                "citation": "Moser, Kim & Mansour, Phys. Fluids 11, 943 (1999), Re_tau=178.12",
            },
        },
        "reproduction": {
            "criterion": f"|recomputed - recorded| <= {REPRO_TOL_PP} percentage points",
            "recorded_rms_pct": recorded_rms,
            "recomputed_rms_pct": reproduced["rms_pct"],
            "delta_pp": delta_pp,
            "n_points": reproduced["n_points"],
            "y_plus_range": reproduced["y_plus_range"],
            "convention": "relative RMS, wall origin 0.5 node, u_tau = sqrt(target_wall_shear)",
            "pass": reproduction_ok,
        },
        "friction": {
            "u_tau_measured": u_tau_measured,
            "u_tau_target": u_tau_target,
            "gap_pct": friction_gap,
        },
        "uplus_conventions": conventions,
        "normalisation_sensitivity": {
            "ratio_measured_over_target": ratio,
            "note": (
                "u+ is conventionally normalised by the *measured* friction velocity. The recorded "
                "gate number uses the target value; under the measured value the same frozen field "
                f"gives {conventions['rel_offset0.5_measured_u_tau']['rms_pct']:.2f}%, i.e. "
                f"x{ratio:.2f} the recorded number. The matrix must declare this convention."
            ),
        },
        "budget_diagnostic": budget_linearity(l3),
        "verdict": {
            "reproduction_pass": reproduction_ok,
            "dns_gate_pass": False,
            "dns_gate_reason": (
                f"recorded best-level RMS {recorded_rms:.2f}% > 3% gate, and under the conventional "
                f"measured-u_tau normalisation it is "
                f"{conventions['rel_offset0.5_measured_u_tau']['rms_pct']:.2f}%. "
                "The matrix keeps this slot failed."
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare against the frozen output")
    args = parser.parse_args(argv)

    report = build()
    if args.check:
        if not OUT.exists():
            print(f"MISSING {OUT.relative_to(REPO_ROOT)}; run without --check")
            return 1
        frozen = json.loads(OUT.read_text(encoding="utf-8"))
        keys = ("reproduction", "friction", "uplus_conventions",
                "normalisation_sensitivity", "budget_diagnostic", "verdict")
        drift = [k for k in keys if frozen.get(k) != report.get(k)]
        print(f"{'DRIFT' if drift else 'CLEAN'} {OUT.relative_to(REPO_ROOT)} drift={drift}")
        return 1 if drift else 0

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    r = report["reproduction"]
    print(f"gate reproduction: recorded {r['recorded_rms_pct']:.3f}% vs recomputed "
          f"{r['recomputed_rms_pct']:.3f}% (delta {r['delta_pp']:.4f} pp, pass={r['pass']}, "
          f"n={r['n_points']}, y+=[{r['y_plus_range'][0]:.1f},{r['y_plus_range'][1]:.1f}])")
    f = report["friction"]
    print(f"friction: u_tau measured {f['u_tau_measured']:.6f} vs target {f['u_tau_target']:.6f} "
          f"({f['gap_pct']:+.2f}%)")
    for k, v in report["uplus_conventions"].items():
        print(f"  {k:<30} rms={v['rms_pct']:.3f}% n={v['n_points']}")
    s = report["normalisation_sensitivity"]
    print(f"normalisation: measured/target = x{s['ratio_measured_over_target']:.2f}")
    for label, blk in report["budget_diagnostic"]["ranges"].items():
        print(f"  budget {label:<14} residual={blk['residual_pct_of_tau_w']:6.2f}% "
              f"u_tau_fit={blk['u_tau_fit']:.5f}")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())