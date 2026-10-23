"""IC-E Phase 1 B3: RG-15 IPW2 cases 3.1/3.2/3.3 official conditions,
glaze multishot production (run_glaze_icing, shots=5, 1200 s, steps=3000,
warmup=6000) -- the accept-era production contract (G13 bitwise target).

Also B4: case 3.3 + official IPW 7-bin DSD (mvd_bins), same multishot
protocol.  --b4 flag selects the bins7 arm only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from _b_common import (  # noqa: E402
    RUNROOT,
    beta_md5,
    build_cfg,
    curve_stats,
    install_rg15,
)

# R4 lock: IPW p.21 official 7-bin DSD (rg15_official_conditions.txt;
# machine-confirmed by icing_mvd_official_20260917/demo_run.log line
# "mvd_bins: n=7 d[um]=[7.3, 12.5, 18.7, 26.8, 36.8, 51.2, 81.3]
#  f=[0.050, 0.100, 0.200, 0.300, 0.200, 0.100, 0.050] mvd_eff=29.94 um")
OFFICIAL_BINS7 = (
    (7.3e-6, 0.05),
    (12.5e-6, 0.10),
    (18.7e-6, 0.20),
    (26.8e-6, 0.30),
    (36.8e-6, 0.20),
    (51.2e-6, 0.10),
    (81.3e-6, 0.05),
)

SHOTS, T_EXP, STEPS, WARMUP = 5, 1200.0, 3000, 6000
TAGS = {
    "3.1_glaze": "b3_c31_glaze",
    "3.2_mixed": "b3_c32_mixed",
    "3.3_rime": "b3_c33_rime",
}

# G3 needs the per-bin eulerian audits; run_glaze_icing does not surface
# res["bins"], so record it from the last shot via a zero-src-change
# subclass installed on the module before run_glaze_icing (it resolves
# RimeIcingSimulation from module globals at call time).
_LAST_BINS = {"bins": None}


def _install_recorder():
    import _b_common

    base = _b_common.ai.RimeIcingSimulation

    if getattr(base, "_is_ice_bm_recorder", False):
        return

    class SimRecorder(base):
        _is_ice_bm_recorder = True

        def run(self):
            r = super().run()
            bins = r.get("bins")
            if bins is None:
                _LAST_BINS["bins"] = None
            else:
                e = bins.get("eulerian") or {}
                _LAST_BINS["bins"] = {
                    "diameters_m": list(e.get("diameters", [])),
                    "fractions": list(e.get("fractions", [])),
                    "per_bin_audit": [dict(a) for a in e.get("audit", [])],
                }
            return r

    _b_common.ai.RimeIcingSimulation = SimRecorder


def run_one(case: str, tag: str, sub: str, **overrides) -> None:
    import _b_common

    cfg = build_cfg(case, T_EXP, STEPS, WARMUP, **overrides)
    dt_shot = T_EXP / SHOTS
    accel = dt_shot / (STEPS * cfg.dt_phys)
    log_lines = []

    def log(msg, _l=log_lines):
        print(msg, flush=True)
        _l.append(str(msg))

    t0 = time.perf_counter()
    g = _b_common.ai.run_glaze_icing(cfg, shots=SHOTS, log=log)
    wall_s = time.perf_counter() - t0

    out = RUNROOT / sub
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{tag}_masks.npz",
        airfoil=g["airfoil"],
        solid=g["solid"],
        ice_only=g["ice_only"],
        m_w=g["m_w"],
        s_grid=g["s_grid"],
        stag=np.array(g["stag"]),
    )
    panels = {k: np.asarray(v) for k, v in g["panels"].items()}
    np.savez_compressed(out / f"{tag}_panels.npz", **panels)
    bc = g["beta_curve"]
    np.savez_compressed(
        out / f"{tag}_beta.npz",
        s_over_c=bc["s_over_c"],
        beta=bc["beta"],
        n_cells=bc["n_cells"],
    )
    # prereg §4 guards
    assert wall_s < 3600.0, f"{tag}: wall {wall_s:.0f}s"
    for k in ("m_w", "ice_only"):
        a = np.asarray(g[k], dtype=np.float64)
        assert np.isfinite(a).all(), f"{tag}: NaN in {k}"
    assert float(g["audit"]["frozen"]) > 0.0, f"{tag}: zero frozen mass"
    assert int(np.asarray(g["ice_only"]).sum()) > 0, f"{tag}: zero ice cells"

    blob = {
        "tag": tag,
        "case": case,
        "shots": SHOTS,
        "t_exposure": T_EXP,
        "steps": STEPS,
        "warmup": WARMUP,
        "git": "archive rerun base: exp/icing-warmup-bins @ 100f78d6 (IC-E-D1 fix)",
        "accel_per_shot": accel,
        "dt_shot_s": dt_shot,
        "wall_s": wall_s,
        "module_mapping_report": cfg.mapping_report(),
        "audit": g["audit"],
        "metrics": g["metrics"],
        "shot_reports": g["shot_reports"],
        "beta_pk": float(bc["beta"].max()) if len(bc["beta"]) else None,
        "beta_curve_stats": curve_stats(bc, cfg.chord_phys),
        "beta_curve_md5": beta_md5(bc),
        "n_beta_shots": SHOTS,
        "le_diameter_eff": cfg.le_diameter_eff,
        "peak_panel": {
            "s_over_c": None,
            "n_f": None,
            "t_s_c": None,
            "regime": None,
        },
        "bins": _LAST_BINS["bins"],
        "log": log_lines,
        "gpu": torch.cuda.get_device_name(0),
    }
    # peak panel (accept format)
    p = g["panels"]
    i_pk = int(np.argmax(p["m_imp_kg_s"]))
    blob["peak_panel"] = {
        "s_over_c": float(p["s_over_c"][i_pk]),
        "n_f": float(p["n_f"][i_pk]),
        "t_s_c": float(p["t_s_c"][i_pk]),
        "regime": str(p["regime"][i_pk]),
    }
    with open(out / f"{tag}_result.json", "w") as fh:
        json.dump(blob, fh, indent=2, default=float)
    print(
        f"[B3] {tag}: wall={wall_s:.1f}s beta_pk={blob['beta_pk']} "
        f"frozen={g['audit']['frozen']:.3e} "
        f"closure={g['audit']['closure_error']:.2e} "
        f"n_f_stag={g['metrics'].get('n_f_stag')}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b4", action="store_true", help="run the bins7 arm only")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "B3/B4 need cuda (CUDA_VISIBLE_DEVICES=6)"
    install_rg15()
    _install_recorder()
    if args.b4:
        run_one("3.3_rime", "b4_bins7", "b4", mvd_bins=OFFICIAL_BINS7)
        return
    for case, tag in TAGS.items():
        run_one(case, tag, "b3")


if __name__ == "__main__":
    main()
