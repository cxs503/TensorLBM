"""Produce the field-snapshot case used by the report's void-fraction cloud map.

Kept separate from :mod:`tensorlbm.microbubble_dr.campaign` because it is a
*visualisation* run at a smaller grid (64x65, 4000 steps) whose only job is to
show the resolved bubble, the wall layer and the streamwise velocity field; the
quantified table comes from the production campaign.  The distinction is stated
in the report so no number from this run is mixed into the criteria.

Usage: ``PYTHONPATH=src python scripts/microbubble_dr_visual.py [out.json]``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tensorlbm.microbubble_dr.campaign import build_config
from tensorlbm.microbubble_dr.channel import MicrobubbleChannel2D

DEFAULT = "artifacts/microbubble_dr_in67_n01/visual_case.json"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0] if argv else DEFAULT)
    cfg = build_config("single_d10", nx=64, ny=65, n_steps=4000)
    run = MicrobubbleChannel2D(cfg).run()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(run))
    print(f"{out}: alpha={run['alpha_mean']:.4f} tau_w={run['tau_w']:.6f} "
          f"N={run['n_bubbles_placed']} R={run['bubble_radius']:.2f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())