"""Model-dir-as-build-and-CI-unit framework.

Ported from CFD-GO/TCLB's practice (see ``docs/XVAL-LBM-MODEL-DIR-AS-BUILD-AND-CI-UNIT.md``):
a model *is* a directory holding its own manifest, its own case script and its
own build/CI hooks; the build targets, the CI matrix and the Makefile fragment
are all **generated** from the discovered directories, so adding a model touches
exactly one directory.

Public API::

    from tensorlbm.model_units import discover, build, test, ci_matrix

    units = discover("models/lbm")
    for unit in units:
        build(unit)
        test(unit)
"""

from __future__ import annotations

from .build import build, doctor, test
from .manifest import MANIFEST_NAME, ManifestError, ModelUnit, load_unit, parse_manifest
from .registry import (
    DEFAULT_ROOT,
    ci_matrix,
    ci_units,
    discover,
    makefile_targets,
    manifest_paths,
)

__all__ = [
    "DEFAULT_ROOT",
    "MANIFEST_NAME",
    "ManifestError",
    "ModelUnit",
    "build",
    "ci_matrix",
    "ci_units",
    "discover",
    "doctor",
    "load_unit",
    "makefile_targets",
    "manifest_paths",
    "parse_manifest",
    "test",
]
