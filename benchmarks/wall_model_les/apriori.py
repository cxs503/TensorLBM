#!/usr/bin/env python3
"""A-priori wall-model benchmarks against channel DNS (criteria a and b).

What is measured
----------------
``anchor``      friction-velocity recovery: given the DNS velocity at a single
                exchange height (``y+ = 30`` and ``y+ = 50``), the wall model
                must return ``u_tau/u_tau_DNS = 1``.  This is the standard
                a-priori test of a wall model in WMLES.
``profile``     log-layer accuracy: the profile predicted by the wall model,
                anchored on the DNS velocity at ``y+ = 30``, is compared with
                the DNS ``u+`` over ``y+ = 30...300`` (criterion a, <= 3 %).
``stress peak`` the Reynolds shear stress implied by the wall-model closure
                (``-<u'v'>+ = nu_t+ dU+/dy+``) is compared with the DNS profile
                reconstructed from the exact channel stress balance; the peak
                abscissa must agree within 5 % (criterion b).

Reference data
--------------
``benchmarks/pending/channel_turb_3d/dns_ref/mkm180.csv`` (Moser, Kim & Mansour
1999, ``Re_tau = 180``), which ships with this repository.  Nothing is typed by
hand: the Reynolds stress comes from ``1 - y+/h+ - dU+/dy+``.

Usage
-----
    python benchmarks/wall_model_les/apriori.py --out artifacts/wall_model_les
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.wall_model_les import (  # noqa: E402
    EquilibriumWallModel,
    NonEquilibriumWallModel,
)
from tensorlbm.wall_model_les.budget import (  # noqa: E402
    load_dns_channel_profile,
    log_region_error,
    peak_location,
    relative_error,
)
from tensorlbm.wall_model_les.laws import law_u_plus_scalar  # noqa: E402

BAND = (30.0, 300.0)
LOG_LAW_KAPPA = 0.41
LOG_LAW_B = 5.2


def _profile_from_model_profile(
    model: NonEquilibriumWallModel, *, u_tau: float, nu: float, y_plus_max: float
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the ODE wall-model profile on a wall-unit grid."""
    # y_exchange in lattice units for a given u_tau and nu: y = y+ nu / u_tau
    y_exchange = y_plus_max * nu / u_tau
    prof = model.profile(u_tau, y_exchange, nu, 0.0)
    faces = model.layer_faces(y_exchange).numpy()
    return faces * u_tau / nu, np.asarray(prof[0].detach().cpu().numpy())


def run(out_dir: Path) -> dict:
    dns = load_dns_channel_profile()
    re_tau = dns.re_tau
    u_tau_dns = 1.0  # wall units
    nu = 1.0 / re_tau  # so that h+ = Re_tau with h = 1 (dimensionless channel)

    ue_model = EquilibriumWallModel(kappa=LOG_LAW_KAPPA, b=LOG_LAW_B, law="spalding")
    ue_log = EquilibriumWallModel(kappa=LOG_LAW_KAPPA, b=LOG_LAW_B, law="log")
    neq = NonEquilibriumWallModel()

    result: dict = {"dns": dns.as_dict(), "re_tau": re_tau}

    # ------------------------------------------------------------------
    # (a) friction velocity recovered from a single exchange sample
    # ------------------------------------------------------------------
    anchors = {}
    for y_plus_ex in (30.0, 50.0, 100.0):
        u_sample = float(np.interp(y_plus_ex, dns.y_plus, dns.u_plus))
        y_lat_units = y_plus_ex * nu / u_tau_dns
        row = {"y_plus_exchange": y_plus_ex, "u_sample_plus": u_sample}
        for label, model in (("equilibrium_spalding", ue_model), ("equilibrium_log", ue_log)):
            res = model.skin_friction(u_sample, y_lat_units, nu, source=0.0)
            row[label] = {
                "u_tau_plus": float(res.u_tau[0]),
                "rel_error": relative_error(float(res.u_tau[0]), u_tau_dns),
            }
        res_neq = neq.skin_friction(u_sample, y_lat_units, nu, source=0.0)
        row["nonequilibrium_s0"] = {
            "u_tau_plus": float(res_neq.u_tau.reshape(-1)[0]),
            "rel_error": relative_error(float(res_neq.u_tau.reshape(-1)[0]), u_tau_dns),
        }
        anchors[f"y_plus_{int(y_plus_ex)}"] = row
    result["anchor_recovery"] = anchors

    # ------------------------------------------------------------------
    # (b) log-layer profile accuracy (criterion a)
    # ------------------------------------------------------------------
    y_anchor = 30.0
    u_anchor = float(np.interp(y_anchor, dns.y_plus, dns.u_plus))
    y_anchor_lat = y_anchor * nu / u_tau_dns

    profiles = {}
    # 1) equilibrium (law-of-the-wall anchored at the exchange point)
    res = ue_model.skin_friction(u_anchor, y_anchor_lat, nu, source=0.0)
    u_tau_eq = float(res.u_tau[0])
    y_grid = dns.y_plus
    u_plus_eq = np.array(
        [law_u_plus_scalar(y * u_tau_eq / u_tau_dns, law="spalding", kappa=LOG_LAW_KAPPA, b=LOG_LAW_B)
         for y in y_grid]
    )
    profiles["equilibrium"] = log_region_error(y_grid, u_plus_eq, dns.u_plus, band=BAND)
    profiles["equilibrium"]["u_tau_plus"] = u_tau_eq

    # 2) pure log law with the same anchoring (shows the Spalding gain)
    res_log = ue_log.skin_friction(u_anchor, y_anchor_lat, nu, source=0.0)
    u_tau_log = float(res_log.u_tau[0])
    u_plus_log = np.array(
        [law_u_plus_scalar(y * u_tau_log / u_tau_dns, law="log", kappa=LOG_LAW_KAPPA, b=LOG_LAW_B)
         for y in y_grid]
    )
    profiles["log_law_only"] = log_region_error(y_grid, u_plus_log, dns.u_plus, band=BAND)
    profiles["log_law_only"]["u_tau_plus"] = u_tau_log

    # 3) non-equilibrium ODE model (mixing length, S = 0)
    y_plus_model, u_plus_model = _profile_from_model_profile(
        neq, u_tau=u_tau_dns, nu=nu, y_plus_max=float(dns.y_plus.max())
    )
    # anchor the modelled profile on the DNS value at the exchange height
    shift = u_anchor - float(np.interp(y_anchor, y_plus_model, u_plus_model))
    profiles["nonequilibrium_s0_shape_only"] = log_region_error(
        dns.y_plus, np.interp(dns.y_plus, y_plus_model, u_plus_model + shift), dns.u_plus, band=BAND
    )
    profiles["nonequilibrium_s0_shape_only"]["anchor_shift_plus"] = shift
    # The predictive limb of the same model: no anchor shift, i.e. the profile
    # that the wall model would supply with its own friction velocity.  The
    # anchored number above is a *shape* diagnostic (the intercept error is
    # removed by construction) and must not be read as a profile prediction --
    # that is exactly the reading error corrected in mission m8872.
    profiles["nonequilibrium_s0_predictive"] = log_region_error(
        dns.y_plus, np.interp(dns.y_plus, y_plus_model, u_plus_model), dns.u_plus, band=BAND
    )
    result["log_layer_profiles"] = profiles

    # ------------------------------------------------------------------
    # (c) Reynolds-stress peak from the wall-model closure (criterion b)
    # ------------------------------------------------------------------
    du_dy_model = np.gradient(u_plus_model, y_plus_model)
    # nu_t+ from the mixing length used by the ODE model
    nu_t_plus = neq.kappa * y_plus_model * (
        1.0 - np.exp(-y_plus_model / neq.van_driest_a_plus)
    )
    # -<u'v'>+ = nu_t+ dU+/dy+  (the closure has to supply the whole stress
    uv_model_plus = nu_t_plus * du_dy_model
    mask = y_plus_model >= 1.0
    uv_peak_model = peak_location(y_plus_model[mask], uv_model_plus[mask])
    uv_peak_dns = dns.uv_peak_y_plus
    # The profile above was integrated only up to the last DNS sample, so its
    # maximum sits on the truncation boundary and is NOT a peak (the previous
    # version reported that boundary value, 1.009 at y+ = 178.1, as the
    # modelled peak).  The closure is therefore re-solved across the whole
    # channel half-height, where the stress peak is an interior maximum.
    from tensorlbm.wall_model_les.closure_channel import solve_channel_closure

    closure = solve_channel_closure(dns.re_tau, closure="mixing_length")
    result["stress_peak"] = {
        "dns_y_plus": uv_peak_dns,
        "dns_value": float(np.max(dns.uv_plus)),
        "wall_model_y_plus": closure.uv_peak_y_plus,
        "relative_error": relative_error(closure.uv_peak_y_plus, uv_peak_dns),
        "wall_model_value": float(np.max(closure.uv_plus[closure.y_plus >= 1.0])),
        "production_dns_y_plus": dns.production_peak_y_plus,
        "production_model_y_plus": closure.production_peak_y_plus,
        "production_peak_relative_error": relative_error(
            closure.production_peak_y_plus, dns.production_peak_y_plus
        ),
        "profile_sampled_at_dy_plus": float(y_plus_model[1] - y_plus_model[0]),
        "legacy_truncated_profile_peak": {
            "y_plus": uv_peak_model,
            "value": float(np.max(uv_model_plus[mask])),
            "note": "boundary maximum of a profile truncated at the last DNS sample; superseded",
        },
        "solver": "closure_channel.solve_channel_closure (mixing_length, full channel)",
    }

    # ------------------------------------------------------------------
    # pressure-gradient / non-equilibrium sensitivity (cost of neglecting S)
    # ------------------------------------------------------------------
    u_sample = float(np.interp(50.0, dns.y_plus, dns.u_plus))
    y_sample = 50.0 * nu / u_tau_dns
    sensitivity = {}
    for beta in (-0.02, -0.01, 0.0, 0.01, 0.02):
        # S is expressed in units of u_tau^3 / nu  (the only available scale)
        source = beta * u_tau_dns**3 / nu
        res_neq = neq.skin_friction(u_sample, y_sample, nu, source=source)
        res_eq = ue_model.skin_friction(u_sample, y_sample, nu, source=0.0)
        sensitivity[f"S_beta_{beta:+.2f}"] = {
            "source_over_u_tau3_per_nu": beta,
            "u_tau_nonequilibrium": float(res_neq.u_tau.reshape(-1)[0]),
            "u_tau_equilibrium": float(res_eq.u_tau[0]),
            "relative_change": float(
                res_neq.u_tau.reshape(-1)[0] / res_eq.u_tau[0] - 1.0
            ),
        }
    result["non_equilibrium_sensitivity"] = sensitivity

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "apriori.json").write_text(json.dumps(result, indent=2, default=float))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/wall_model_les")
    args = parser.parse_args()
    result = run(Path(args.out))
    print(json.dumps({k: v for k, v in result.items() if k != "dns"}, indent=2, default=float))


if __name__ == "__main__":
    main()