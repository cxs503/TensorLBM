"""Performance baselines for direct PyTorch LBM hot paths."""

from .contracts import BenchmarkSpec, PerformanceArtifact, artifact_to_dict
from .d3q19_mrt import (
    assert_hot_path_uses_direct_torch_kernels,
    build_d3q19_mrt_initial_state,
    run_d3q19_mrt_benchmark,
)

__all__ = [
    "BenchmarkSpec",
    "PerformanceArtifact",
    "artifact_to_dict",
    "assert_hot_path_uses_direct_torch_kernels",
    "build_d3q19_mrt_initial_state",
    "run_d3q19_mrt_benchmark",
]
