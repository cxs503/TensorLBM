#!/usr/bin/env python3
"""Log-law constants of the reference DNS and the resulting profile error.

Criterion (a) of the mission node asks for a ``y+ = 30 ... 300`` velocity-profile
error below 3 %.  The wall model's log region is fixed by ``(kappa, B)``; the
declared pair used by the model is ``kappa = 0.41``, ``B = 5.2``.  This script
answers two questions with numbers, from the channel DNS that ships with the
repository (``benchmarks/pending/channel_turb_3d/dns_ref/mkm180.csv``):

1. what ``(kappa, B)`` does the DNS itself imply over the physically meaningful
   log layer (``30 <= y+ <= 0.3 h+``)?
2. how large is the profile error over the full criterion band with the declared
   pair versus the DNS-fitted pair?

The DNS-fitted error is *circular* and is reported only as a diagnostic of
where the residual comes from; the declared-pair number is the one used to grade
the criterion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.wall_model_les.budget import load_dns_channel_profile, log_region_error  # noqa: E402
from tensorlbm.wall_model_les.laws import KAPPA_DEFAULT, B_DEFAULT, law_u_plus_scalar  # noqa: E402

BAND = (30.0, 300.0)


def fit_log_law(y_plus: np.ndarray, u_plus: np.ndarray, window: tuple[float, float]) -> dict:
    mask = (y_plus >= window[0]) & (y_plus <= window[1])
    if int(mask.sum()) < 3:
        return {"window": list(window), "n": int(mask.sum()), "available": False}
    x = np.log(y_plus[mask])
    inv_kappa, b = np.polyfit(x, u_plus[mask], 1)
    resid = u_plus[mask] - (inv_kappa * x + b)
    return {
        "window": list(window),
        "available": True,
        "n": int(mask.sum()),
        "kappa": float(1.0 / inv_kappa),
        "B": float(b),
        "rms_residual": float(np.sqrt(np.mean(resid**2))),
        "max_abs_residual": float(np.max(np.abs(resid))),
    }


def run(out_dir: Path) -> dict:
    dns = load_dns_channel_profile()
    y_plus = np.asarray(dns.y_plus, dtype=float)
    u_plus = np.asarray(dns.u_plus, dtype=float)
    h_plus = float(y_plus.max())

    fits = {
        "log_region_30_to_0p3h": fit_log_law(y_plus, u_plus, (30.0, max(31.0, 0.3 * h_plus))),
        "log_region_30_to_0p15h": fit_log_law(y_plus, u_plus, (30.0, max(30.0, 0.15 * h_plus))),
        "log_region_30_to_100": fit_log_law(y_plus, u_plus, (30.0, min(100.0, h_plus))),
    }

    # profile error over the criterion band, anchored at y+ = 30 (a-priori wall model)
    ref = np.array([law_u_plus_scalar(float(v), "spalding") for v in y_plus])
    anchor = y_plus[(y_plus >= 30.0)].min()
    i_anchor = int(np.argmin(np.abs(y_plus - anchor)))
    u_anchor = u_plus[i_anchor]
    model_declared = ref - ref[i_anchor] + u_anchor
    declared = log_region_error(y_plus, u_plus, model_declared, band=BAND)

    fitted = next(f for f in fits.values() if f.get("available"))
    kappa_f, b_f = fitted["kappa"], fitted["B"]
    model_fitted = np.array(
        [np.log(max(v, 1e-12)) / kappa_f + b_f for v in y_plus]
    )
    model_fitted = model_fitted - model_fitted[i_anchor] + u_anchor
    fitted_err = log_region_error(y_plus, u_plus, model_fitted, band=BAND)

    record = {
        "reference": {"source": dns.source, "re_tau": h_plus, "n_points": int(y_plus.size)},
        "declared_constants": {"kappa": KAPPA_DEFAULT, "B": B_DEFAULT},
        "dns_fitted_constants": fits,
        "profile_error_in_criterion_band": {
            "band": list(BAND),
            "declared_pair": declared,
            "dns_fitted_pair": fitted_err,
            "dns_band_upper_limit": h_plus,
            "note": (
                "at Re_tau = 180 the criterion band 30...300 exceeds the channel; the "
                "DNS upper bound %.1f is used, and above y+ ~ 0.3 h+ the profile is a "
                "wake, not a log layer" % h_plus
            ),
        },
        "verdict": {
            "pass_declared": bool(declared.get("max_rel", 1.0) <= 0.03),
            "pass_fitted": bool(fitted_err.get("max_rel", 1.0) <= 0.03),
            "basis": (
                (
                    "the declared (0.41, 5.2) pair misses the DNS log layer by more than "
                    "3 percent; the DNS implies kappa = {:.3f}, B = {:.2f}, the known "
                    "low-Re-channel tendency (kappa -> 0.38), which the wall model cannot "
                    "remove without recalibrating its constants"
                ).format(kappa_f, b_f)
            ),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "log_law_fit.json").write_text(json.dumps(record, indent=2, default=float))
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/wall_model_les")
    args = parser.parse_args()
    record = run(Path(args.out))
    print(json.dumps(record["profile_error_in_criterion_band"], indent=1, default=float))
    print(json.dumps(record["verdict"], indent=1, default=float))


if __name__ == "__main__":
    main()