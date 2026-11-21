"""Campaign runner: YAML-driven XML template pipeline, REAL solver arm.

Anchor: XVAL-LBM-YML-DRIVEN-XML-TEMPLATE-PIPELINE

Runs the campaign case ``cases/xval/poiseuille_zou_he_2d.yaml`` through the
full pipeline (YAML -> schema validation -> XML template -> real D2Q9 BGK +
Zou-He channel solver -> verification metrics -> verdict) and writes all
artifacts under this directory.

Usage (from the repo root):
    PYTHONPATH=src python3 runs/m25066_pipeline_real/run_campaign.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.xval.pipeline import run_pipeline  # noqa: E402


def main() -> int:
    out_dir = Path(__file__).resolve().parent
    t0 = time.time()
    result = run_pipeline(
        yaml_path=REPO_ROOT / "cases" / "xval" / "poiseuille_zou_he_2d.yaml",
        output_dir=out_dir,
    )
    wall = time.time() - t0
    print(f"wall={wall:.1f}s")
    print(result.summary_line())
    print("solver:", result.details.get("solver"))
    print("steps:", result.details.get("steps"), "converged:", result.details.get("converged"))
    print(
        "g_measured:", result.details.get("g_measured"),
        "g_nominal:", result.details.get("g_nominal"),
        "dev:", result.details.get("g_rel_dev"),
    )
    print("mass_defect:", result.details.get("mass_defect"))
    print("metrics:", {k: round(v, 6) for k, v in result.metrics.items()})
    print("errors:", result.errors)

    # Machine-readable summary for the report.
    summary = {
        "anchor": result.anchor,
        "passed": result.passed,
        "metrics": result.metrics,
        "thresholds": result.thresholds,
        "details": {k: v for k, v in result.details.items() if k != "profiles"},
        "wall_seconds": wall,
        "errors": result.errors,
    }
    with open(out_dir / "campaign_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
