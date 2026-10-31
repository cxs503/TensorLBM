"""ITTC Ice Pool Test Benchmarking Framework.

This module implements the ITTC-recommended benchmarking procedure for
ice navigation performance prediction. It supports automated comparison
across multiple ship types against ITTC reference data.

References:
    - ITTC 7.5-04-01-01.1: Ice Resistance - Model Test Method
    - ITTC 7.5-02-04-01.1: Ice Model Testing Procedures
    - Lindqvist (1989): A direct method for ice resistance prediction
    - Riska et al. (1997): Ice resistance of merchant vessels
"""

from .ship_definitions import (
    SHIP_DATABASE,
    ShipDefinition,
    get_ship_by_type,
    list_ship_types,
)
from .ice_resistance_model import (
    ITTCIceResistanceModel,
    IceProperties,
    IceResistanceResult,
    compute_ice_resistance,
)
from .benchmark_runner import (
    ITTCBenchmarkRunner,
    FullBenchmarkReport,
    BenchmarkResult,
    run_full_benchmark,
)
from .report_generator import (
    ITTCReportGenerator,
    generate_comparison_report,
)

__all__ = [
    "SHIP_DATABASE",
    "ShipDefinition",
    "get_ship_by_type",
    "list_ship_types",
    "ITTCIceResistanceModel",
    "IceProperties",
    "IceResistanceResult",
    "compute_ice_resistance",
    "ITTCBenchmarkRunner",
    "FullBenchmarkReport",
    "BenchmarkResult",
    "run_full_benchmark",
    "ITTCReportGenerator",
    "generate_comparison_report",
]
