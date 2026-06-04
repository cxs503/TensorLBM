from __future__ import annotations

import tensorlbm.api as stable_api
import tensorlbm.experimental as experimental_api


def test_stable_api_exports_core_symbols() -> None:
    assert hasattr(stable_api, "equilibrium")
    assert hasattr(stable_api, "macroscopic")
    assert hasattr(stable_api, "collide_bgk")
    assert hasattr(stable_api, "flow_step_image_path")


def test_experimental_api_exports_ai_symbols() -> None:
    assert hasattr(experimental_api, "run_ai_les_pipeline")
    assert hasattr(experimental_api, "FlowFieldTransformer")

