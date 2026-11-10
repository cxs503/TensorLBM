#!/usr/bin/env python3
"""High-Reynolds-number wall-model closure tier (criteria a, b).

The LES benchmark (:mod:`channel_wmles`) cannot reach the Reynolds numbers at
which the criterion band ``y+ = 30...300`` lies inside the log layer: the
D3Q19 wall-modelled channel of this repository stops converging around
``Re_tau ~ 200`` (see ``stability_map.py``), while the band top (``y+ = 300``)
is only a tenth of the half-height at ``Re_tau = 3000``.  This script therefore
measures the *closure* -- the wall model's own statement about the near-wall
layer -- across the Reynolds-number range that the criterion cares about, and
grades it against the two reference families that exist without a cluster-scale
DNS:

* **DNS (validation)** -- the ``Re_tau = 180`` Moser-Kim-Mansour table shipped
  with the repository, after it passes the integrity gate in
  :mod:`tensorlbm.wall_model_les.reference_gate`.
* **The law of the wall (verification)** -- the modelled profile's own log-layer
  slope and intercept in ``y+ = 30...300``, which must come out as the
  ``(kappa, B)`` the closure was built from.  A closure that does not reproduce
  its own asymptote is broken; this is the check that separates an
  implementation defect from a modelling limit.

Everything is written to ``artifacts/wall_model_les/``:

``high_re.json``            Reynolds sweep, DNS validation, peak table, verdicts
``fig_high_re_profile.png`` ``u+`` profiles, wall model vs DNS
``fig_high_re_stress.png``  ``-<u'v'>+`` and production, closure vs DNS
``fig_peak_drift.png``      peak abscissa vs ``Re_tau``: closure drift vs DNS

Usage
-----
    python benchmarks/wall_model_les/high_re.py --out artifacts/wall_model_les
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.wall_model_les.budget import (  # noqa: E402
    DNS_REFERENCES,
    load_dns_channel_profile,
    log_region_error,
    relative_error,
)
from tensorlbm.wall_model_les.closure_channel import (  # noqa: E402
    CLOSURES,
    log_layer_fit,
    solve_channel_closure,
    uv_monotonicity,
)
from tensorlbm.wall_model_les.models import (  # noqa: E402
    EquilibriumWallModel,
    NonEquilibriumWallModel,
)
from tensorlbm.wall_model_les.reference_gate import (  # noqa: E402
    validate_channel_dns_table,
)

BAND = (30.0, 300.0)
#: Fractions of the half-height at which the log law is still a good reading.
WALL_FRACTION_VALID = 0.2
RE_SWEEP = (180.0, 395.0, 590.0, 1000.0, 2000.0, 10000.0)
PREDECLARED = {
    "a_log_layer_profile": 0.03,
    "b_reynolds_stress_peak": 0.05,
    "kappa": 0.41,
    "b_intercept": 5.2,
}


def _reference_tables() -> dict:
    """Run the integrity gate over every shipped channel DNS table."""

    out = {}
    for key, entry in DNS_REFERENCES.items():
        gate = validate_channel_dns_table(entry["path"], re_tau=entry["re_tau"])
        out[key] = {**entry, "path": str(entry["path"]), "gate": gate.as_dict()}
    return out


def _wall_model_recovery(re_tau: float, u_tau_ref: float, dns_u_plus, *, nu=None) -> dict:
    """``u_tau`` recovered by each wall model from one exchange sample (DNS velocity)."""

    nu = (1.0 / re_tau) if nu is None else nu
    h = 1.0
    models = {
        "equilibrium_spalding": EquilibriumWallModel(law="spalding"),
        "equilibrium_log": EquilibriumWallModel(law="log"),
        "nonequilibrium_ode": NonEquilibriumWallModel(),
    }
    table = {}
    for y_plus_ex in (30.0, 50.0, 100.0):
        if y_plus_ex >= float(np.max(dns_u_plus[0])) * 0.9:
            continue
        u_sample = float(np.interp(y_plus_ex, dns_u_plus[0], dns_u_plus[1]))
        y_lat = y_plus_ex * nu / u_tau_ref
        row = {"y_plus_exchange": y_plus_ex, "u_sample_plus": u_sample}
        for label, model in models.items():
            res = model.skin_friction(u_sample, y_lat, nu, source=0.0)
            ut = float(res.u_tau.reshape(-1)[0])
            row[label] = {
                "u_tau_plus": ut,
                "rel_error": relative_error(ut, u_tau_ref),
            }
        table[f"y_plus_{int(y_plus_ex)}"] = row
    return table


def run(out_dir: Path) -> dict:
    tables = _reference_tables()
    accepted = {k: v for k, v in tables.items() if v["gate"]["verdict"] == "PASS"}
    dns_key = "mkm180" if "mkm180" in accepted else None
    dns = load_dns_channel_profile(DNS_REFERENCES["mkm180"]["path"]) if dns_key else None

    record: dict = {
        "predeclared": PREDECLARED,
        "band": list(BAND),
        "re_sweep": list(RE_SWEEP),
        "reference_tables": tables,
        "reference_tables_accepted": sorted(accepted),
        "reference_tables_rejected": sorted(set(tables) - set(accepted)),
    }

    # ------------------------------------------------------------------
    # 1. Reynolds sweep of the modelled channel
    # ------------------------------------------------------------------
    sweep: dict = {}
    solutions: dict = {}
    for closure in CLOSURES:
        rows = {}
        for re_tau in RE_SWEEP:
            sol = solve_channel_closure(re_tau, closure=closure)
            solutions[(closure, re_tau)] = sol
            band_hi = min(BAND[1], WALL_FRACTION_VALID * re_tau)
            fit = log_layer_fit(sol.y_plus, sol.u_plus, band=BAND, h_plus=re_tau)
            rows[f"{re_tau:.0f}"] = {
                **sol.as_dict(),
                "log_layer_fit": fit,
                "valid_band": [BAND[0], band_hi],
                "band_available": bool(band_hi > BAND[0]),
                "uv_monotonicity": uv_monotonicity(sol.y_plus, sol.uv_plus),
            }
        sweep[closure] = rows
    record["reynolds_sweep"] = sweep

    # ------------------------------------------------------------------
    # 2. Validation against the DNS tier + verification of the asymptote
    # ------------------------------------------------------------------
    validation: dict = {}
    if dns is not None:
        y_dns, u_dns = dns.y_plus, dns.u_plus
        dns_band_hi = min(BAND[1], float(np.max(y_dns)))
        u_at_anchor = float(np.interp(30.0, y_dns, u_dns))
        for closure in CLOSURES:
            sol = solutions[(closure, dns.re_tau)]
            u_model = np.interp(y_dns, sol.y_plus, sol.u_plus)
            # the ODE model is anchored at the first criterion-band station, as a
            # wall model anchored on one exchange sample would be
            shift = u_at_anchor - float(np.interp(30.0, sol.y_plus, sol.u_plus))
            validation[closure] = {
                "band": [BAND[0], dns_band_hi],
                "log_layer_error_absolute": log_region_error(
                    y_dns, u_model, u_dns, band=(BAND[0], dns_band_hi)
                ),
                "log_layer_error_anchored": log_region_error(
                    y_dns, u_model + shift, u_dns, band=(BAND[0], dns_band_hi)
                ),
                "anchor_shift_plus": shift,
                "uv_peak_dns_y_plus": dns.uv_peak_y_plus,
                "uv_peak_model_y_plus": sol.uv_peak_y_plus,
                "uv_peak_rel_error": relative_error(sol.uv_peak_y_plus, dns.uv_peak_y_plus),
                "uv_peak_value_dns": float(np.max(dns.uv_plus)),
                "uv_peak_value_model": float(np.max(sol.uv_plus)),
                "production_peak_dns_y_plus": dns.production_peak_y_plus,
                "production_peak_model_y_plus": sol.production_peak_y_plus,
                "production_peak_rel_error": relative_error(
                    sol.production_peak_y_plus, dns.production_peak_y_plus
                ),
            }
        record["wall_model_recovery"] = _wall_model_recovery(
            dns.re_tau, 1.0, (y_dns, u_dns), nu=1.0 / dns.re_tau
        )
    record["dns_validation"] = validation

    # verification: the closure must reproduce its own (kappa, B) at high Re,
    # where y+ = 30...300 really is inside the log layer
    verification = {}
    for closure in CLOSURES:
        sol = solutions[(closure, RE_SWEEP[-1])]
        fit = log_layer_fit(sol.y_plus, sol.u_plus, band=BAND, h_plus=RE_SWEEP[-1], wall_fraction=1.0)
        verification[closure] = {
            "re_tau": RE_SWEEP[-1],
            "kappa_fit": fit["kappa"],
            "b_fit": fit["b"],
            "kappa_rel_error_vs_declared": relative_error(fit["kappa"], PREDECLARED["kappa"]),
            "b_abs_error": abs(fit["b"] - PREDECLARED["b_intercept"]),
            "fit_max_rel_error": fit["max_rel_error"],
        }
    record["log_layer_verification"] = verification

    # Reynolds-number invariance of the stress peak inside the closure family
    invariance = {}
    for closure in CLOSURES:
        peaks = [solutions[(closure, re)]["uv_peak_y_plus"] if False else solutions[(closure, re)].uv_peak_y_plus
                 for re in RE_SWEEP]
        invariance[closure] = {
            "uv_peak_y_plus": [float(p) for p in peaks],
            "rel_range": float((max(peaks) - min(peaks)) / min(peaks)),
        }
    if dns is not None:
        invariance["dns_mkm180"] = {
            "uv_peak_y_plus": [float(dns.uv_peak_y_plus)],
            "note": (
                "the DNS peak abscissa is Reynolds-number invariant (y+ ~ 30 for "
                "Re_tau = 180...5200 in the literature); only one vetted table "
                "exists in this worktree, so the invariance itself is not "
                "measurable here and is quoted, not measured"
            ),
        }
    record["stress_peak_invariance"] = invariance

    record["verdicts"] = _verdicts(record)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "high_re.json").write_text(json.dumps(record, indent=2, default=float))
    _figures(record, solutions, dns, out_dir)
    return record


def _verdicts(record: dict) -> dict:
    val = record["dns_validation"]
    thr_a = record["predeclared"]["a_log_layer_profile"]
    thr_b = record["predeclared"]["b_reynolds_stress_peak"]
    out = {}
    if val:
        best = min(val.items(), key=lambda kv: kv[1]["log_layer_error_absolute"]["max_rel"])
        out["a_log_layer_profile"] = {
            "threshold": thr_a,
            "verdict": "PASS" if best[1]["log_layer_error_absolute"]["max_rel"] <= thr_a else "FAIL",
            "basis": (
                "closure profile vs the vetted DNS over y+ = {:.0f}...{:.0f} "
                "(the whole criterion band that the Re_tau = {} table covers)"
            ).format(
                record["band"][0], best[1]["band"][1], record["reference_tables"]["mkm180"]["re_tau"]
            ),
            "best_closure": best[0],
            "max_rel_error": best[1]["log_layer_error_absolute"]["max_rel"],
        }
        worst = max(val.items(), key=lambda kv: kv[1]["uv_peak_rel_error"])
        out["b_reynolds_stress_peak"] = {
            "threshold": thr_b,
            "verdict": "PASS" if worst[1]["uv_peak_rel_error"] <= thr_b else "FAIL",
            "basis": "closure -<u'v'>+ peak abscissa vs DNS (best closure reported; all closures listed)",
            "best_closure": min(val.items(), key=lambda kv: kv[1]["uv_peak_rel_error"])[0],
            "best_rel_error": min(v["uv_peak_rel_error"] for v in val.values()),
            "rel_errors": {k: v["uv_peak_rel_error"] for k, v in val.items()},
        }
    return out


def _figures(record: dict, solutions: dict, dns, out_dir: Path) -> None:
    colors = {"mixing_length": "#1f77b4", "log_layer": "#ff7f0e"}
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    if dns is not None:
        ax.semilogx(dns.y_plus, dns.u_plus, "ko", ms=3.5, label="DNS MKM Re_tau=180")
    for closure in CLOSURES:
        sol = solutions[(closure, 10000.0)]
        ax.semilogx(sol.y_plus, sol.u_plus, "-", color=colors[closure], lw=1.6,
                    label=f"{closure} (Re_tau=1e4)")
        ax.semilogx([], [], " ", label="")
    y = np.logspace(0.0, 4.0, 200)
    ax.semilogx(y, np.log(y) / 0.41 + 5.2, "k--", lw=1.0, label="log law (0.41, 5.2)")
    ax.axvspan(30, 300, color="grey", alpha=0.12, label="criterion band y+=30-300")
    ax.set_xlabel("$y^+$")
    ax.set_ylabel("$u^+$")
    ax.set_title("Wall-model closure vs log law and DNS")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_high_re_profile.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ax = axes[0]
    if dns is not None:
        ax.plot(dns.y_plus, dns.uv_plus, "ko", ms=3.0, label="DNS $-\\langle u'v'\\rangle^+$ (MKM-180)")
        ax.plot(dns.y_plus, dns.production_plus, "ks", ms=3.0, mfc="none",
                label="DNS production $P^+$")
    for closure in CLOSURES:
        sol = solutions[(closure, 180.0)]
        ax.plot(sol.y_plus, sol.uv_plus, "-", color=colors[closure], lw=1.4,
                label=f"{closure} $-\\langle u'v'\\rangle^+$")
        ax.plot(sol.y_plus, sol.production_plus, "--", color=colors[closure], lw=1.0,
                label=f"{closure} $P^+$")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("$y^+$")
    ax.set_ylabel("wall units")
    ax.set_title("Stress and production, Re_tau = 180")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for closure in CLOSURES:
        peaks = record["stress_peak_invariance"][closure]["uv_peak_y_plus"]
        ax.semilogx(record["re_sweep"], peaks, "o-", color=colors[closure], label=f"{closure} peak")
        prods = [record["reynolds_sweep"][closure][f"{re:.0f}"]["production_peak_y_plus"] for re in record["re_sweep"]]
        ax.semilogx(record["re_sweep"], prods, "s--", color=colors[closure], lw=0.9, alpha=0.6,
                    label=f"{closure} production peak")
    if dns is not None:
        ax.axhline(dns.uv_peak_y_plus, color="k", ls=":", lw=1.2, label="DNS $-\\langle u'v'\\rangle^+$ peak")
        ax.axhline(dns.production_peak_y_plus, color="k", ls="-.", lw=1.0, label="DNS $P^+$ peak")
    ax.set_xlabel("$Re_\\tau$")
    ax.set_ylabel("peak abscissa  $y^+$")
    ax.set_title("Peak position vs Reynolds number")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_high_re_stress.png", dpi=150)
    plt.close(fig)

    # reference integrity: the shippped tables as they arrive, before any use
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    for key, entry in record["reference_tables"].items():
        data = np.genfromtxt(entry["path"], delimiter=",", names=True)
        ok = entry["gate"]["verdict"] == "PASS"
        ax.semilogx(data["y_plus"], data["u_plus"], "-" if ok else "--",
                    color="#2ca02c" if ok else "#d62728", lw=1.6,
                    label=f"{key} ({entry['gate']['verdict']})")
    y = np.logspace(0.0, np.log10(400.0), 200)
    ax.semilogx(y, np.log(y) / 0.41 + 5.2, "k:", lw=1.0, label="log law")
    ax.set_xlabel("$y^+$")
    ax.set_ylabel("$u^+$")
    ax.set_title("Shipped DNS tables, integrity gate")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    for key, entry in record["reference_tables"].items():
        data = np.genfromtxt(entry["path"], delimiter=",", names=True)
        if "uv_plus" not in (data.dtype.names or ()):
            continue
        ok = entry["gate"]["verdict"] == "PASS"
        ax.plot(data["y_plus"], data["uv_plus"], "-" if ok else "--",
                color="#2ca02c" if ok else "#d62728", lw=1.4, label=f"{key} stored correlation")
    ax.axhline(1.0, color="k", ls=":", lw=1.0, label="physical bound |uv+| <= 1")
    ax.set_xlabel("$y^+$")
    ax.set_ylabel("stored $uv^+$")
    ax.set_title("Stored shear-stress column")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_reference_gate.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/wall_model_les")
    args = parser.parse_args()
    record = run(Path(args.out))
    print(json.dumps(record["verdicts"], indent=2, default=float))
    print(json.dumps(record["log_layer_verification"], indent=2, default=float))


if __name__ == "__main__":
    main()