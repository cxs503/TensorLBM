"""Cold-path data products gated by declared runtime evidence."""

from .contracts import DatasetManifest, DatasetSampleRef, FieldProduct
from .field_r2 import FieldDataProductR2

__all__ = ["DatasetManifest", "DatasetSampleRef", "FieldDataProductR2", "FieldProduct"]
