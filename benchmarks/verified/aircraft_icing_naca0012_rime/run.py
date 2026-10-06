"""IC-E Phase 1 B1+B2: NACA 0012 IRT rime production point, convergence
ladder, LWC linearity arm, determinism pair (IC-A protocol, prereg §0).

Archive edition: outputs to ./output/<run>/ beside this file;
tensorlbm from the installed package (needs the IC-E-D1
warmup-bins fix, branch exp/icing-warmup-bins).

Runs (name, nx, ny, steps, warmup, lwc):
  b1_prod / ladder_L1  320x160  3000  6000  5e-4   (production point)
  ladder_L2            640x320  6000 12000  5e-4
  ladder_L3            890x445  8334 16600  5e-4
  ladder_L4           1280x640 13340 26600  5e-4
  m20_lwc025           320x160  3000  6000  2.5e-4 (G9 low arm)
  m20_lwc10            320x160  3000  6000  1.0e-3 (G9 high arm; 0.5 = b1_prod)
  det_run1/det_run2    320x160  3000  6000  5e-4   (G7 pair)

Zero gate-value literals here; results JSON only (verify.py owns gates).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
RUNROOT = _HERE / "output"
_SRC = _HERE.parents[2] / "src"  # <repo>/src beside benchmarks/
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DET_ARRAYS = (
    "airfoil",
    "solid",
    "m_w",
    "impact_mass",
    "s_grid",
    "beta_grid",
    "stag",
    "hist_step",
    "hist_t",
    "hist_cd",
    "hist_cl",
    "hist_ice",
    "alpha_e",
    "impact_mass_e",
    "beta_e_grid",
)


def npz_array_hashes(npz_path) -> list:
    """IC-A determinism protocol: per-array sha256 (b_common verbatim)."""
    import hashlib

    z = np.load(npz_path)
    out = []
    for k in DET_ARRAYS:
        a = z[k]
        out.append(
            {
                "key": k,
                "shape": list(a.shape),
                "dtype": str(a.dtype),
                "sha256_first8": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:8],
            }
        )
    return out


import numpy as np  # noqa: E402
import torch  # noqa: E402

from tensorlbm.aircraft_icing import (  # noqa: E402
    IcingConfig,
    run_rime_icing,
    save_icing_artifacts,
)

RUNS = [
    ("b1_prod", 320, 160, 3000, 6000, 5.0e-4),
    ("ladder_L1", 320, 160, 3000, 6000, 5.0e-4),
    ("ladder_L2", 640, 320, 6000, 12000, 5.0e-4),
    ("ladder_L3", 890, 445, 8334, 16600, 5.0e-4),
    ("ladder_L4", 1280, 640, 13340, 26600, 5.0e-4),
    ("m20_lwc025", 320, 160, 3000, 6000, 2.5e-4),
    ("m20_lwc10", 320, 160, 3000, 6000, 1.0e-3),
    ("det_run1", 320, 160, 3000, 6000, 5.0e-4),
    ("det_run2", 320, 160, 3000, 6000, 5.0e-4),
]


def stag_thickness_mm(result, cfg) -> float:
    ice = result["ice_only"]
    x_le = int(result["metrics"]["x_le"])
    return float(ice[:, x_le].sum()) * cfg.dx_phys * 1e3


def main() -> None:
    assert torch.cuda.is_available(), "B1/B2 need cuda (CUDA_VISIBLE_DEVICES=6)"
    base = RUNROOT
    base.mkdir(parents=True, exist_ok=True)
    for name, nx, ny, steps, warmup, lwc in RUNS:
        cfg = IcingConfig(
            nx=nx,
            ny=ny,
            chord_frac=0.4,
            u_in=0.05,
            aoa_deg=4.0,
            steps=steps,
            warmup_steps=warmup,
            chord_phys=0.5334,
            v_inf=67.0,
            lwc=lwc,
            mvd=20.0e-6,
            t_static_c=-10.0,
            t_exposure=360.0,
            rime_density_mode="macklin",
            droplet_phase="eulerian",
            eulerian_scheme="donor2",
            collision="cumulant",
            c_s=0.1,
            re_lu_target=2.5e6,
            beta_window_mode="clean",
            drag_law="stokes",
            device="cuda",
            seed=0,
            log_every=250,
        )
        out_dir = base / name
        out_dir.mkdir(parents=True, exist_ok=True)
        log_lines = []

        def log(msg, _l=log_lines):
            print(msg, flush=True)
            _l.append(str(msg))

        t0 = time.time()
        r = run_rime_icing(cfg, log=log)
        wall_s = time.time() - t0

        # prereg §4 guards (NaN / zero collection / wall cap)
        assert wall_s < 3600.0, f"{name}: wall {wall_s:.0f}s"
        for k in ("m_w", "impact_mass"):
            a = np.asarray(r[k], dtype=np.float64)
            assert np.isfinite(a).all(), f"{name}: NaN in {k}"
        dep = float(r["eulerian"]["audit"]["deposited"])
        assert dep > 0.0, f"{name}: zero collection"
        assert int(np.asarray(r["ice_only"]).sum()) > 0, f"{name}: zero ice"

        files = save_icing_artifacts(r, str(out_dir))
        e, m, aud = r["eulerian"], r["metrics"], r["audit"]
        beta_e = e["beta"]
        res = {
            "name": name,
            "wall_time_s": wall_s,
            "config": {
                "nx": cfg.nx,
                "ny": cfg.ny,
                "chord_frac": cfg.chord_frac,
                "u_in": cfg.u_in,
                "steps": cfg.steps,
                "warmup_steps": cfg.warmup_steps,
                "aoa_deg": cfg.aoa_deg,
                "t_static_c": cfg.t_static_c,
                "lwc": cfg.lwc,
                "mvd": cfg.mvd,
                "t_exposure": cfg.t_exposure,
                "seed": cfg.seed,
                "droplet_phase": cfg.droplet_phase,
                "eulerian_scheme": cfg.eulerian_scheme,
                "collision": cfg.collision,
                "c_s": cfg.c_s,
                "re_lu_target": cfg.re_lu_target,
                "rime_density_mode": cfg.rime_density_mode,
                "beta_window_mode": cfg.beta_window_mode,
                "drag_law": cfg.drag_law,
                "device": "cuda",
            },
            "mapping_selected": {
                "chord_lu": cfg.chord_lu,
                "dx_phys_m": cfg.dx_phys,
                "dt_phys_s": cfg.dt_phys,
                "lwc_accel_k": cfg.lwc_accel,
                "t_equiv_s": cfg.t_equiv,
                "rho_rime_eff": cfg.rho_rime_eff,
                "stokes": cfg.stokes,
                "re_lu": cfg.re_lu,
                "beta_cap_window": cfg.beta_cap_window,
            },
            "beta_e_pk": float(beta_e["beta"].max()) if len(beta_e["beta"]) else None,
            "beta_e_curve": {
                "s_over_c": beta_e["s_over_c"].tolist(),
                "beta": beta_e["beta"].tolist(),
            },
            "audit_lagr": {
                k: aud[k]
                for k in (
                    "seeded",
                    "frozen",
                    "exited",
                    "trapped",
                    "airborne",
                    "pending",
                    "pending_fluid",
                    "pending_solid",
                    "closure_error",
                )
            },
            "audit_euler": e["audit"],
            "euler_deposited_kg": e["audit"]["deposited"],
            "n_ice_cells": m["n_ice_cells"],
            "ice_area_pct_chord2": m["ice_area_pct_chord2"],
            "t_max_mm": 1e3 * max(m.get("upper_horn_m", 0.0), m.get("lower_horn_m", 0.0)),
            "upper_horn_mm": 1e3 * m.get("upper_horn_m", 0.0),
            "lower_horn_mm": 1e3 * m.get("lower_horn_m", 0.0),
            "t_stag_mm": stag_thickness_mm(r, cfg),
            "cd0": r["cd0"],
            "cl0": r["cl0"],
            "cd_end": r["cd_end"],
            "cl_end": r["cl_end"],
            "cd_drift_pct": r["cd_drift_pct"],
            "artifacts": files,
            "array_sha256": npz_array_hashes(out_dir / "result.npz"),
            "torch_version": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
        }
        with open(out_dir / "results.json", "w") as fh:
            json.dump(res, fh, indent=2, default=float)
        print(
            f"[B1B2] {name} DONE wall={wall_s:.1f}s "
            f"beta_pk={res['beta_e_pk']} dep={dep:.4e} "
            f"t_max={res['t_max_mm']}mm",
            flush=True,
        )


if __name__ == "__main__":
    main()
