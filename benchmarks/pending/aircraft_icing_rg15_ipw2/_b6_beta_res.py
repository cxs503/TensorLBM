"""IC-E Phase 1 B6: RG-15 3.3 beta resolution ladder nx={320,480,640},
St fixed (exp12_refine.py protocol).  Disclosure tier (D3), no gate.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from _b_common import RUNROOT, build_cfg, curve_stats, install_rg15  # noqa: E402

CASE, STEPS, WARMUP, DT_SHOT = "3.3_rime", 3000, 6000, 240.0


def main() -> None:
    assert torch.cuda.is_available(), "B6 needs cuda (CUDA_VISIBLE_DEVICES=6)"
    install_rg15()
    import _b_common

    out = RUNROOT / "b6"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for nx, ny in [(320, 160), (480, 240), (640, 320)]:
        cfg = build_cfg(CASE, t_exposure=DT_SHOT, steps=STEPS, warmup=WARMUP, nx=nx, ny=ny)
        accel = DT_SHOT / (STEPS * cfg.dt_phys)
        shot_cfg = replace(
            cfg,
            freeze_in_run=False,
            beta_window_mode="trailing",
            beta_window_frac=0.0,
            accel_override=accel,
            droplet_warmup=True,
            surface_arc_sign_fix=True,
        )
        t0 = time.perf_counter()
        res = _b_common.ai.RimeIcingSimulation(shot_cfg, log=lambda *a: None).run()
        wall = time.perf_counter() - t0
        assert wall < 3600.0, f"nx={nx}: wall {wall:.0f}s"
        be = res["eulerian"]["beta"]
        assert float(res["eulerian"]["impact_mass"].sum()) > 0.0
        st = curve_stats(be, cfg.chord_phys)
        af = res["airfoil"]
        ys, xs = np.nonzero(af)
        x_le = int(xs.min())
        le_rows = np.nonzero(af[:, x_le : x_le + 3].any(axis=1))[0]
        np.savez_compressed(
            RUNROOT / "b6" / f"b6_nx{nx}_beta.npz",
            s_over_c=be["s_over_c"],
            beta=be["beta"],
            n_cells=be["n_cells"],
        )
        rows.append(
            {
                "nx": nx,
                "ny": ny,
                "chord_lu": cfg.chord_lu,
                "dx_mm": cfg.dx_phys * 1e3,
                "tau_d_lu": cfg.tau_d_lu,
                "stokes": cfg.stokes,
                "le_span_cells": len(le_rows),
                "n_airfoil_cells": int(af.sum()),
                "stats": st,
                "wall_s": wall,
                "le_diameter_cells": 2.0 * len(le_rows) / 2.0,
            }
        )
        print(
            f"[B6] nx={nx}: dx={cfg.dx_phys * 1e3:.4f}mm "
            f"St={cfg.stokes:.3f} beta_pk={st['beta_max']:.5f} "
            f"beta_stag={st['beta_stag']:.5f} wall={wall:.0f}s",
            flush=True,
        )
    with open(RUNROOT / "b6" / "b6_beta_res.json", "w") as fh:
        json.dump({"case": CASE, "rows": rows}, fh, indent=2, default=float)
    print("[B6] done")


if __name__ == "__main__":
    main()
