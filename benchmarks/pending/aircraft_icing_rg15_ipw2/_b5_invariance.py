"""IC-E Phase 1 B5: shot/accel invariance grid, mirror of
icing_d3_20260917/exp11_invariance.py (same 8 cells, same shot-1
measure-only semantics).  Gate family (steps=3000, dt_shot varies) +
disclosure family (dt_shot=240, steps varies) + production cell.
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
from _b_common import RUNROOT, beta_md5, build_cfg, curve_stats, install_rg15  # noqa: E402

CASE = "3.3_rime"
WARMUP = 6000
GRID = [
    (3000, 240.0, "s3000_d240"),  # production cell (G8 reference)
    (3000, 120.0, "s3000_d120"),  # gate family
    (3000, 300.0, "s3000_d300"),
    (3000, 600.0, "s3000_d600"),
    (3000, 1200.0, "s3000_d1200"),
    (800, 240.0, "s800_d240"),  # disclosure family
    (1500, 240.0, "s1500_d240"),
    (6000, 240.0, "s6000_d240"),
]


def main() -> None:
    assert torch.cuda.is_available(), "B5 needs cuda (CUDA_VISIBLE_DEVICES=6)"
    install_rg15()
    out = RUNROOT / "b5"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for steps, dt_shot, tag in GRID:
        cfg = build_cfg(CASE, t_exposure=dt_shot, steps=steps, warmup=WARMUP)
        accel = dt_shot / (steps * cfg.dt_phys)
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
        res = _b_common_mod().RimeIcingSimulation(shot_cfg, log=lambda *a: None).run()
        wall = time.perf_counter() - t0
        # prereg §4 guards
        assert wall < 3600.0, f"{tag}: wall {wall:.0f}s"
        be = res["eulerian"]["beta"]
        assert float(res["eulerian"]["impact_mass"].sum()) > 0.0, f"{tag}: zero collection"
        np.savez_compressed(
            out / f"b5_{tag}_beta.npz",
            s_over_c=be["s_over_c"],
            beta=be["beta"],
            n_cells=be["n_cells"],
            impact_total_kg=np.array([float(res["eulerian"]["impact_mass"].sum())]),
        )
        st = curve_stats(be, cfg.chord_phys)
        row = {
            "tag": tag,
            "steps": steps,
            "dt_shot_s": dt_shot,
            "accel": accel,
            "lwc_eff": shot_cfg.lwc_eff,
            "wall_s": wall,
            "md5": beta_md5(be),
            "stats": st,
            "impacted_kg_per_phys_s": float(res["eulerian"]["impact_mass"].sum() / dt_shot),
            "impact_total_kg": float(res["eulerian"]["impact_mass"].sum()),
        }
        rows.append(row)
        print(
            f"[B5] {tag}: steps={steps} dts={dt_shot:.0f} "
            f"accel={accel:.4e} beta_pk={st['beta_max']:.5f} "
            f"md5={row['md5'][:10]} wall={wall:.0f}s",
            flush=True,
        )
    with open(out / "b5_invariance.json", "w") as fh:
        json.dump({"case": CASE, "grid": rows}, fh, indent=2, default=float)
    print("[B5] done")


def _b_common_mod():
    import _b_common

    return _b_common.ai


if __name__ == "__main__":
    main()
