"""Cold-path data products gated by declared runtime evidence."""

from .contracts import DatasetManifest, DatasetSampleRef, FieldProduct
from .field_dataset_r2 import FieldDatasetR2, FieldSampleRefR2
from .field_r2 import FieldDataProductR2
from .solver_export import (
    load_product,
    load_product_arrays,
    register_product,
    save_fields_hdf5,
)

__all__ = [
    "DatasetManifest",
    "DatasetSampleRef",
    "FieldDataProductR2",
    "FieldDatasetR2",
    "FieldProduct",
    "FieldSampleRefR2",
    "load_product",
    "load_product_arrays",
    "register_product",
    "save_fields_hdf5",
]

from .catalog import FieldDataCatalog, AssetRecord, LineageRecord, QualityCheck, MetadataRecord
from .quality import check_field_product, validate_field_product

__all__ += [
    "FieldDataCatalog", "AssetRecord", "LineageRecord", "QualityCheck", "MetadataRecord",
    "check_field_product", "validate_field_product",
]
