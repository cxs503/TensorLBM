"""Tests for the field-data catalog and quality checks (clean-room impl)."""

import numpy as np
import pytest

from tensorlbm.data.catalog import (
    AssetRecord,
    FieldDataCatalog,
    LineageRecord,
    QualityCheck,
)
from tensorlbm.data.quality import check_field_product


@pytest.fixture
def catalog(tmp_path):
    cat = FieldDataCatalog.open(tmp_path / "catalog.db")
    yield cat
    cat.close()


def _asset(asset_id="p1", **kw):
    base = dict(
        asset_id=asset_id,
        name="u-velocity",
        kind="field_product",
        field_name="ux",
        units="lu",
        shape="(64,48,48)",
        dtype="float32",
    )
    base.update(kw)
    return AssetRecord(**base)


def test_register_and_get(catalog):
    catalog.register_asset(_asset())
    rec = catalog.get_asset("p1")
    assert rec is not None
    assert rec.name == "u-velocity"
    assert rec.field_name == "ux"


def test_list_filter(catalog):
    catalog.register_asset(_asset("p1"))
    catalog.register_asset(_asset("p2", name="pressure", field_name="p"))
    catalog.register_asset(_asset("p3", kind="dataset", field_name=None))
    fields = catalog.list_assets(kind="field_product")
    assert len(fields) == 2
    by_name = catalog.list_assets(field_name="p")
    assert len(by_name) == 1 and by_name[0].asset_id == "p2"


def test_metadata_roundtrip(catalog):
    catalog.register_asset(_asset())
    catalog.add_metadata("p1", "solver", "octree-shell", source="auto")
    meta = catalog.get_metadata("p1")
    assert len(meta) == 1 and meta[0].key == "solver"
    catalog.delete_metadata("p1", "solver")
    assert catalog.get_metadata("p1") == []


def test_lineage_and_upstream(catalog):
    catalog.register_asset(_asset("run1", kind="run"))
    catalog.register_asset(_asset("prod1", source_run_id="run1"))
    catalog.register_asset(_asset("ds1", kind="dataset"))
    catalog.add_lineage(LineageRecord("run1", "prod1", relation_type="derived_from"))
    catalog.add_lineage(LineageRecord("prod1", "ds1", relation_type="split_of"))
    upstream = catalog.upstream("ds1")
    assert upstream == ["prod1", "run1"]


def test_quality_score_updates_asset(catalog):
    catalog.register_asset(_asset())
    score = catalog.record_quality(
        "p1",
        [
            QualityCheck("finiteness", True, "ok"),
            QualityCheck("shape", True, "ok"),
            QualityCheck("mass", False, "drift"),
        ],
    )
    assert score == 67
    rec = catalog.get_asset("p1")
    assert rec.quality_score == 67
    reports = catalog.get_quality_reports("p1")
    assert len(reports) == 1 and reports[0]["status"] == "warning"


def test_archive(catalog):
    catalog.register_asset(_asset())
    catalog.archive_asset("p1")
    assert catalog.get_asset("p1").status == "archived"
    assert catalog.list_assets() == []  # default filters active


def test_invalid_asset_rejected(catalog):
    with pytest.raises(ValueError):
        catalog.register_asset(_asset(kind="bogus"))


def test_quality_checks():
    arr = np.zeros((4, 4, 4), dtype=np.float32)
    prod = type(
        "P",
        (),
        {
            "field_name": "ux",
            "shape": (4, 4, 4),
            "dtype": "float32",
            "units": "lu",
        },
    )()
    # finiteness + shape pass on a clean field
    result = check_field_product(prod, arr, mass_field=False)
    assert result.passed and result.overall_score == 100

    bad = arr.copy()
    bad[0, 0, 0] = np.nan
    result2 = check_field_product(prod, bad)
    assert not result2.passed
    assert any(not c.passed and c.check_name == "finiteness" for c in result2.checks)

    # mass conservation passes on a density field at rho=1
    rho_ok = np.ones((4, 4, 4), dtype=np.float32)
    result3 = check_field_product(prod, rho_ok, mass_field=True)
    assert result3.passed

    # mass conservation flags a drifted density field
    rho = np.full((4, 4, 4), 1.05, dtype=np.float32)
    result4 = check_field_product(prod, rho, mass_field=True)
    assert any(not c.passed and c.check_name == "mass_conservation" for c in result4.checks)
