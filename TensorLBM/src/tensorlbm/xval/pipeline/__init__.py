"""YAML-driven XML template pipeline for LBM cross-validation.

This module implements a declarative pipeline where validation cases are
defined in YAML, converted to XML templates, and executed through the
TensorLBM solver infrastructure.

Pipeline stages
---------------
1. **YAML ingestion**: Parse case definition YAML files with schema validation.
2. **XML generation**: Produce LBM case XML templates from YAML parameters.
3. **Execution**: Run the solver with generated XML input.
4. **Verification**: Compare results against analytical/experimental references.
5. **Reporting**: Generate JSON verdicts and human-readable reports.

Design rationale
----------------
* YAML provides human-readable, version-controllable case definitions.
* XML templates are compatible with existing LBM solver input formats.
* The pipeline is extensible: new physics models and boundary conditions
  can be added by extending the YAML schema and XML template registry.

References
----------
* je-santos/MPLBM-UT: yml-driven-xml-template-pipeline pattern
* ASME V&V 20-2009: Standard for Verification and Validation in CFD
"""

from .yaml_schema import XValYAMLCase, load_yaml_case, validate_yaml_case
from .xml_template import generate_xml_template, XMLTemplateRegistry
from .runner import YamlXmlPipeline, run_pipeline

__all__ = [
    "XValYAMLCase",
    "load_yaml_case",
    "validate_yaml_case",
    "generate_xml_template",
    "XMLTemplateRegistry",
    "YamlXmlPipeline",
    "run_pipeline",
]
