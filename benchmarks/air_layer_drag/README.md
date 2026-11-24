# Air-layer drag comparison

The [campaign report](../../docs/IN30_N03_AIR_LAYER_DR.md) documents the
simulation, its scope and the withheld literature-comparison verdict. The
[reference points](reference/air_lubrication_reference_points.json) preserve the
historical inputs and their provenance limits.

The [campaign driver](../../scripts/air_layer_drag_campaign.py) and
[literature checker](../../scripts/air_layer_drag_literature_check.py) use the
same reference. The checker expects `artifacts/air_layer_drag/results.json`
from a completed simulation.
