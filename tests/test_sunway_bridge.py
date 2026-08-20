"""Tests for the SWLBM (Sunway) data bridge: force-history CSV → R2 product.

The CSV fixture mirrors the real SWLBM emitter observed on the psn002
cluster (``force_history_*.csv`` under
``/home/export/online3/swbxyh/hydro/swlbm/suboff_geshan_ok/test/result/``):
``step,F_x,F_y,F_z,C_D,C_L,C_S,Re,C_F_ITTC,wet_nodes`` with one row per
force-sampling phase, ~100 lattice steps apart.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tensorlbm.data.catalog import FieldDataCatalog
from tensorlbm.data.solver_export import load_product, load_product_arrays
from tensorlbm.data.sunway_bridge import (
    SWLBM_BRIDGE_SCHEMA,
    SWLBM_FORCE_COLUMNS,
    convert_swlbm_csv,
    convert_swlbm_field_dump,
    parse_swlbm_force_csv,
)

_CODE_SHA = "a" * 40

_HEADER = ",".join(SWLBM_FORCE_COLUMNS)

# Real-format sample rows (values in the style of the psn002 output).
_ROWS = [
    "0,8.477844650e+04,0.000000000e+00,0.000000000e+00,134.929104362,0.000000000e+00,0.000000000e+00,2.500000e+04,0.013043214,17166474",
    "100,9.242587689e+03,5.300327191e+01,-1.096616499e-03,14.710036930,8.435733730e-02,-1.745319573e-06,2.500000e+04,0.013043214,17166474",
    "200,9.101122453e+03,5.411200454e+01,-2.204191344e-03,14.484532207,8.607610221e-02,-3.509319573e-06,2.500000e+04,0.013043214,17166474",
    "300,8.998001221e+03,5.512099887e+01,-3.311177822e-03,14.320511933,8.768110331e-02,-5.280319573e-06,2.500000e+04,0.013043214,17166474",
]


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    path = tmp_path / "force_history_sample.csv"
    path.write_text(_HEADER + "\n" + "\n".join(_ROWS) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parse_swlbm_force_csv(csv_path: Path) -> None:
    history = parse_swlbm_force_csv(csv_path)
    assert history.rows == 4
    assert history.steps.tolist() == [0, 100, 200, 300]
    assert history.force_xyz.shape == (4, 3)
    assert history.coefficients.shape == (4, 3)
    assert history.wet_nodes.tolist() == [17166474] * 4
    assert history.force_xyz[0, 0] == pytest.approx(8.477844650e04)
    assert history.coefficients[-1, 0] == pytest.approx(14.320511933)
    assert len(history.csv_sha256) == 64


def test_parse_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("step,F_x\n0,1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        parse_swlbm_force_csv(path)


def test_parse_rejects_non_finite_values(csv_path: Path) -> None:
    text = csv_path.read_text().replace("9.101122453e+03", "nan", 1)
    csv_path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        parse_swlbm_force_csv(csv_path)


def test_parse_rejects_empty_data(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text(_HEADER + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no data rows"):
        parse_swlbm_force_csv(path)


def test_parse_tolerates_trailing_blank_line(csv_path: Path) -> None:
    csv_path.write_text(csv_path.read_text() + "\n\n", encoding="utf-8")
    assert parse_swlbm_force_csv(csv_path).rows == 4


# ---------------------------------------------------------------------------
# Conversion + registration
# ---------------------------------------------------------------------------

def _metadata(**overrides) -> dict[str, object]:
    base: dict[str, object] = {
        "run_id": "swlbm-suboff-geshan",
        "case": "suboff_aff8",
        "code_sha": _CODE_SHA,
        "lattice": "D3Q19",
        "queue": "q_sw_share",
        "sunway_host": "psn002",
        "re": 25000.0,
    }
    base.update(overrides)
    return base


def test_convert_swlbm_csv_round_trip(csv_path: Path, tmp_path: Path) -> None:
    catalog = FieldDataCatalog.open(tmp_path / "catalog.db")
    product_id = convert_swlbm_csv(catalog, csv_path, _metadata())

    assert product_id == "swlbm-suboff-geshan:forces"
    asset = catalog.get_asset(product_id)
    assert asset is not None
    assert asset.kind == "field_product"
    assert "sunway_bridge" in asset.tags

    product = load_product(catalog, product_id)
    arrays = load_product_arrays(product)
    assert set(arrays) == {"force_lattice", "force_coefficients", "wet_nodes"}

    expected_force = np.array(
        [row.split(",")[1:4] for row in _ROWS], dtype=np.float64
    ).astype(np.float32)
    np.testing.assert_allclose(arrays["force_lattice"], expected_force, rtol=1e-6)
    assert arrays["force_coefficients"].shape == (4, 3)
    assert arrays["wet_nodes"].dtype == np.int32
    assert np.isfinite(arrays["force_lattice"]).all()


def test_convert_records_lineage_quality_and_metrics(
    csv_path: Path, tmp_path: Path,
) -> None:
    catalog = FieldDataCatalog.open(tmp_path / "catalog.db")
    product_id = convert_swlbm_csv(catalog, csv_path, _metadata())

    lineage = catalog.get_lineage(product_id)
    assert any(
        SWLBM_BRIDGE_SCHEMA in record.transformation for record in lineage
    )
    assert catalog.get_quality_reports(product_id), "bridge must record quality checks"

    product = load_product(catalog, product_id)
    metric_ids = {metric.metric_id for metric in product.run_manifest.metrics}
    assert {"rows", "cd_last", "cd_mean_tail", "cf_ittc", "wet_nodes_last"} <= metric_ids
    metrics = {m.metric_id: m.value for m in product.run_manifest.metrics}
    assert metrics["cd_last"] == pytest.approx(14.320511933, rel=1e-6)
    assert product.run_manifest.verify_metric_evidence()


def test_convert_binds_source_csv_artifact(csv_path: Path, tmp_path: Path) -> None:
    catalog = FieldDataCatalog.open(tmp_path / "catalog.db")
    product_id = convert_swlbm_csv(catalog, csv_path, _metadata())
    product = load_product(catalog, product_id)
    assert product.source_artifact_id == "swlbm:source_csv"
    artifact = next(
        a for a in product.run_manifest.artifacts
        if a.artifact_id == "swlbm:source_csv"
    )
    assert artifact.media_type == "text/csv"
    assert artifact.verify_integrity()
    bridge = dict(product.lineage)["sunway_bridge"]
    assert bridge["schema"] == SWLBM_BRIDGE_SCHEMA
    assert bridge["rows"] == 4


def test_convert_rejects_duplicate_run_id(csv_path: Path, tmp_path: Path) -> None:
    catalog = FieldDataCatalog.open(tmp_path / "catalog.db")
    convert_swlbm_csv(catalog, csv_path, _metadata())
    with pytest.raises(ValueError, match="already registered"):
        convert_swlbm_csv(catalog, csv_path, _metadata())


def test_convert_requires_core_metadata(csv_path: Path, tmp_path: Path) -> None:
    catalog = FieldDataCatalog.open(tmp_path / "catalog.db")
    with pytest.raises(ValueError, match="code_sha"):
        convert_swlbm_csv(
            catalog, csv_path, {"run_id": "x", "case": "y"},
        )


def test_field_dump_stub_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="sunway_data_bridge"):
        convert_swlbm_field_dump(tmp_path / "dump.bin")
