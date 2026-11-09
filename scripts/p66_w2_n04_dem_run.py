"""Long DEM run for P66-W2-N04 (run in the background, writes JSON)."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorlbm.ice_fullship.dem_field import DEMConfig, DEMField  # noqa: E402

OUT = os.path.join("runs", "p66_w2_n04")
os.makedirs(OUT, exist_ok=True)


def main(n_steps: int = 40000, warmup: int = 12000, tag: str = "base", width: float = 100.0):
    t0 = time.time()
    cfg = DEMConfig(n_steps=n_steps, channel_width=width)
    dem = DEMField(cfg)
    t_init = time.time() - t0
    R = []
    R_fore = []
    R_aft = []
    ncont = []
    vmax = []
    hot = []
    t_run = time.time()
    for k in range(n_steps):
        dem.step()
        imp = dem._hull_impulse
        if imp is not None:
            idx, F, fn, ft, vt, tmag, nvec = imp
            cnt = dem._counted[idx]
            tot = float(-F[0].sum())
            if k >= warmup:
                R.append(tot)
                R_aft.append(float(-F[0][~cnt].sum()))
                R_fore.append(float(-F[0][cnt].sum()))
                ncont.append(int(idx.size))
        if k >= warmup and k % 200 == 0:
            sp = np.hypot(dem.vx, dem.vy)
            vmax.append(float(sp.max()))
            hot.append(float((sp > 1.05 * cfg.ship_speed).mean()))
    diag = dem.diagnostics()
    audit = dem.energy_audit()
    res = {
        "tag": tag,
        "n_steps": n_steps,
        "warmup_steps": warmup,
        "physical_time_simulated_s": n_steps * cfg.dt,
        "wall_time_init_s": t_init,
        "wall_time_run_s": time.time() - t_run,
        "n_floes": dem.n,
        "channel_width": width,
        "R_ice_mean": float(np.mean(R)) if R else 0.0,
        "R_ice_std": float(np.std(R)) if R else 0.0,
        "R_ice_sem": float(np.std(R) / max(1, np.sqrt(len(R)))) if R else 0.0,
        "R_fore_mean": float(np.mean(R_fore)) if R_fore else 0.0,
        "R_aft_mean": float(np.mean(R_aft)) if R_aft else 0.0,
        "n_contacts_mean": float(np.mean(ncont)) if ncont else 0.0,
        "vmax_mean": float(np.mean(vmax)) if vmax else 0.0,
        "hot_fraction_mean": float(np.mean(hot)) if hot else 0.0,
        "energy_audit": audit,
        "diagnostics": {k: v for k, v in diag.items() if isinstance(v, (int, float))},
    }
    with open(os.path.join(OUT, f"dem_{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in res.items() if not isinstance(v, dict)}, indent=2), flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    tag = sys.argv[2] if len(sys.argv) > 2 else "base"
    width = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
    main(n_steps=n, tag=tag, width=width)