# YAML-Driven XML Template Pipeline for LBM Cross-Validation

## Overview

The YAML-driven XML template pipeline provides a declarative approach to defining
and running LBM cross-validation cases. Instead of hardcoding case parameters in
Python, users write YAML configuration files that are parsed, validated, converted
to XML templates, and executed through the TensorLBM solver infrastructure.

## Pipeline Stages

```
YAML Case Definition
        ↓
  Schema Validation
        ↓
  XML Template Generation
        ↓
  Solver Execution
        ↓
  Verification Metrics
        ↓
  Verdict & Report
```

## YAML Schema

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Must be "1.0" |
| `anchor` | string | Unique case identifier (e.g., "XVAL-LBM-POISEUILLE-2D") |
| `title` | string | Human-readable case title |
| `geometry` | object | Geometry definition |
| `physics` | object | Physics model configuration |
| `solver` | object | Solver runtime settings |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `description` | string | Case description |
| `category` | string | "analytical", "experimental", or "benchmark" |
| `boundary` | object | Boundary condition specification |
| `reference` | object | Analytical/experimental reference |
| `acceptance` | object | Acceptance criteria with thresholds |
| `resolution_scan` | object | Resolution convergence scan spec |
| `tags` | list | Case tags for filtering |
| `external_anchor` | string | Literature citation |

### Example YAML

```yaml
schema_version: "1.0"
anchor: "XVAL-LBM-POISEUILLE-2D"
title: "2D Plane Poiseuille Flow"
description: "Validation against analytical Poiseuille solution"
category: "analytical"

geometry:
  type: "channel"
  dimensions: "2D"
  nx: 64
  ny: 32
  lx: 4.0
  ly: 1.0

physics:
  model: "bgk"
  lattice: "D2Q9"
  reynolds: 100.0
  density: 1.0

boundary:
  walls:
    - location: "top"
      type: "bounce_back"
    - location: "bottom"
      type: "bounce_back"
  inlet:
    type: "pressure"
    value: 1.0
  outlet:
    type: "pressure"
    value: 0.9

solver:
  max_steps: 5000
  convergence_threshold: 1.0e-6
  output_interval: 100
  seed: 42
  precision: "float64"

reference:
  type: "analytical"
  citation: "Schlichting & Gersten, Boundary-Layer Theory, §III.2.1"
  solution_fn: "poiseuille_2d"

acceptance:
  metrics:
    eps_inf: 0.02
    eps_l2: 0.02
  stopping_rule: "Both error norms must be below threshold"

tags: ["poiseuille", "analytical", "2d"]
external_anchor: "Schlichting & Gersten (2017)"
```

## XML Template Generation

The pipeline generates XML templates from YAML definitions. Supported template types:

| Model | Lattice | Template |
|-------|---------|----------|
| bgk | D2Q9 | 2D channel/cavity template |
| bgk | D3Q19 | 3D template |
| mrt | D3Q19 | MRT 3D template |

Custom templates can be registered via `XMLTemplateRegistry.register()`.

## Python API

### Loading a YAML Case

```python
from tensorlbm.xval.pipeline import load_yaml_case

case = load_yaml_case("cases/poiseuille.yaml")
print(case.anchor)  # "XVAL-LBM-POISEUILLE-2D"
print(case.geometry.nx)  # 64
```

### Running the Pipeline

```python
from pathlib import Path
from tensorlbm.xval.pipeline import run_pipeline

result = run_pipeline(
    yaml_path=Path("cases/poiseuille.yaml"),
    output_dir=Path("runs/poiseuille"),
)
print(result.summary_line())  # "[PASS] XVAL-LBM-POISEUILLE-2D: eps_inf=0.001234, eps_l2=0.000567"
```

### Custom Solver Integration

```python
from tensorlbm.xval.pipeline import YamlXmlPipeline

def my_solver(case):
    # Run actual LBM simulation
    return {"metrics": {"eps_inf": 0.001, "eps_l2": 0.0005}}

pipeline = YamlXmlPipeline(
    output_dir=Path("runs/my_case"),
    solver_fn=my_solver,
)
result = pipeline.run_from_yaml(Path("cases/my_case.yaml"))
```

### Custom XML Templates

```python
import xml.etree.ElementTree as ET
from tensorlbm.xval.pipeline import XMLTemplateRegistry, generate_xml_template

registry = XMLTemplateRegistry()

def my_template(case):
    root = ET.Element("custom_lbm_case")
    ET.SubElement(root, "anchor").text = case.anchor
    return root

registry.register("custom_model", "D2Q9", my_template)

# Use custom registry
from tensorlbm.xval.pipeline import run_pipeline
result = run_pipeline(yaml_path, registry=registry)
```

## Output Artifacts

The pipeline generates:

1. **XML template**: `{anchor}.xml` - Generated case definition
2. **Result JSON**: `artifacts/{anchor}_result.json` - Full pipeline result
3. **Verdict JSON**: `artifacts/{anchor}_verdict.json` - PASS/FAIL verdict with metrics

## Design Rationale

This pipeline follows the pattern established in je-santos/MPLBM-UT's
yml-driven-xml-template-pipeline approach:

- **YAML** provides human-readable, version-controllable case definitions
- **XML templates** are compatible with existing LBM solver input formats
- **Extensible registry** allows new physics models and boundary conditions
- **Declarative acceptance criteria** separate test logic from case definition

## References

- je-santos/MPLBM-UT: yml-driven-xml-template-pipeline pattern
- ASME V&V 20-2009: Standard for Verification and Validation in CFD
- Schlichting & Gersten (2017): Boundary-Layer Theory, 8th ed.
