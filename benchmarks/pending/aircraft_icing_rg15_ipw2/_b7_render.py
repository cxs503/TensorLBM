"""IC-E Phase 1 B7: render pass -- final-geometry flow fields for the glaze
cases (b3) and the bins7 arm (b4).  Non-verdict product (gallery only):
measure-only RimeIcingSimulation with solid = airfoil | final ice mask,
short warmup, then export macroscopic fields from sim state.

CPU (OMP<=16, prereg §4).  Fields: ux/uy/rho (midplane slice of the 3-D
lattice), alpha_e (droplet volume fraction), masks.
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
from _b_common import RUNROOT, build_cfg, install_rg15  # noqa: E402

WARMUP_SHORT, STEPS_SHORT = 3000, 200
CASES = [
    ("3.1_glaze", "b3_c31_glaze", "b3"),
    ("3.2_mixed", "b3_c32_mixed", "b3"),
    ("3.3_rime", "b3_c33_rime", "b3"),
    ("3.3_rime", "b4_bins7", "b4"),
]


def main() -> None:
    torch.set_num_threads(16)
    install_rg15()
    import _b_common

    from tensorlbm.d3q19 import macroscopic3d

    rows = []
    for case, tag, sub in CASES:
        masks_p = RUNROOT / sub / f"{tag}_masks.npz"
        if not masks_p.exists():
            # B4 blocked by library defect (NOTES defect registry):
            # no final-geometry masks -> nothing to render, skip with note.
            rows.append({"tag": tag, "skipped": "no masks (B4 library-blocked)"})
            print(f"[B7] {tag}: SKIPPED (no masks: B4 library-blocked)", flush=True)
            continue
        cfg = build_cfg(case, 240.0, STEPS_SHORT, WARMUP_SHORT, device="cpu")
        accel = 240.0 / (STEPS_SHORT * cfg.dt_phys)
        shot_cfg = replace(
            cfg,
            freeze_in_run=False,
            beta_window_mode="trailing",
            beta_window_frac=0.0,
            accel_override=accel,
            droplet_warmup=True,
            surface_arc_sign_fix=True,
            device="cpu",
            log_every=1000,
        )
        sim = _b_common.ai.RimeIcingSimulation(shot_cfg, log=lambda *a: None)
        ice = np.load(masks_p)["ice_only"]
        sim.solid = sim.airfoil | torch.from_numpy(ice).to(sim.dev)
        t0 = time.perf_counter()
        sim.run()
        wall = time.perf_counter() - t0
        rho3, ux3, uy3, _ = macroscopic3d(sim.f)

        # 2-D lattice embedded in 3-D flow: f is (Q, nz=1, ny, nx), so the
        # macroscopic fields are (1, ny, nx) -- drop the singleton nz axis
        # (first attempt sliced [:, :, k] which returned a single-x column;
        # accident archived in probe/b7_first_launch_slice_fix.log lineage)
        def _flat(t):
            a = t.cpu().numpy()
            return a[0] if a.ndim == 3 and a.shape[0] == 1 else a

        ux, uy, rho = _flat(ux3), _flat(uy3), _flat(rho3)
        alpha = sim.alpha.cpu().numpy()
        if alpha.ndim == 3:
            alpha = alpha.sum(axis=0)
        out = RUNROOT / "b7"
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / f"{tag}_fields.npz",
            ux=ux,
            uy=uy,
            rho=rho,
            alpha_e=alpha,
            airfoil=sim.airfoil.cpu().numpy(),
            solid=sim.solid.cpu().numpy(),
            ice_only=ice,
        )
        rows.append(
            {
                "tag": tag,
                "wall_s_cpu": wall,
                "ux_shape": list(ux.shape),
                "alpha_shape": list(alpha.shape),
            }
        )
        print(
            f"[B7] {tag}: cpu wall={wall:.0f}s ux{ux.shape} "
            f"|u|max={float(np.hypot(ux, uy).max()):.4f}",
            flush=True,
        )
    with open(out / "b7_render.json", "w") as fh:
        json.dump({"cases": rows}, fh, indent=2, default=float)
    print("[B7] done")


if __name__ == "__main__":
    main()
