"""XML template generation from YAML case definitions.

Converts parsed YAML case definitions into XML templates compatible with
TensorLBM solver input format. Supports multiple template types for
different physics models and boundary condition configurations.

Template registry
-----------------
Templates are registered by (physics_model, lattice) key and can be
extended by adding new template functions to the registry.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .yaml_schema import XValYAMLCase

# Type alias for template generator functions
TemplateFn = Callable[[XValYAMLCase], ET.Element]


class XMLTemplateRegistry:
    """Registry of XML template generators keyed by (model, lattice)."""

    def __init__(self):
        self._templates: Dict[Tuple[str, str], TemplateFn] = {}

    def register(self, model: str, lattice: str, fn: TemplateFn) -> None:
        """Register a template generator for a (model, lattice) pair."""
        key = (model.lower(), lattice.upper())
        self._templates[key] = fn

    def get(self, model: str, lattice: str) -> Optional[TemplateFn]:
        """Get the template generator for a (model, lattice) pair."""
        return self._templates.get((model.lower(), lattice.upper()))

    def list_templates(self) -> list:
        """List all registered template keys."""
        return list(self._templates.keys())

    def has_template(self, model: str, lattice: str) -> bool:
        """Check if a template exists for the given key."""
        return (model.lower(), lattice.upper()) in self._templates


# ---------------------------------------------------------------------------
# Built-in template generators
# ---------------------------------------------------------------------------

def _bgk_d2q9_template(case: XValYAMLCase) -> ET.Element:
    """Generate XML template for BGK D2Q9 cases."""
    root = ET.Element("lbm_case")
    root.set("version", "1.0")
    root.set("generator", "yaml-xml-pipeline")

    # Metadata
    meta = ET.SubElement(root, "metadata")
    ET.SubElement(meta, "anchor").text = case.anchor
    ET.SubElement(meta, "title").text = case.title
    ET.SubElement(meta, "category").text = case.category
    ET.SubElement(meta, "description").text = case.description

    # Geometry
    geo = ET.SubElement(root, "geometry")
    geo.set("type", case.geometry.type)
    geo.set("dimensions", case.geometry.dimensions)
    ET.SubElement(geo, "resolution").text = f"{case.geometry.nx}x{case.geometry.ny}"
    ET.SubElement(geo, "domain").text = f"{case.geometry.lx}x{case.geometry.ly}"

    # Physics
    phys = ET.SubElement(root, "physics")
    phys.set("model", case.physics.model)
    phys.set("lattice", case.physics.lattice)
    if case.physics.reynolds is not None:
        ET.SubElement(phys, "reynolds").text = str(case.physics.reynolds)
    if case.physics.viscosity is not None:
        ET.SubElement(phys, "viscosity").text = str(case.physics.viscosity)
    ET.SubElement(phys, "density").text = str(case.physics.density)

    # Boundary conditions
    bc = ET.SubElement(root, "boundary_conditions")
    if case.boundary:
        for wall in case.boundary.walls:
            w = ET.SubElement(bc, "wall")
            for k, v in wall.items():
                w.set(k, v)
        if case.boundary.inlet:
            inlet = ET.SubElement(bc, "inlet")
            for k, v in case.boundary.inlet.items():
                inlet.set(k, str(v))
        if case.boundary.outlet:
            outlet = ET.SubElement(bc, "outlet")
            for k, v in case.boundary.outlet.items():
                outlet.set(k, str(v))

    # Solver
    solv = ET.SubElement(root, "solver")
    solv.set("max_steps", str(case.solver.max_steps))
    solv.set("convergence_threshold", str(case.solver.convergence_threshold))
    solv.set("output_interval", str(case.solver.output_interval))
    solv.set("seed", str(case.solver.seed))
    solv.set("precision", case.solver.precision)

    # Acceptance criteria
    if case.acceptance and case.acceptance.metrics:
        acc = ET.SubElement(root, "acceptance")
        for metric, threshold in case.acceptance.metrics.items():
            m = ET.SubElement(acc, "metric")
            m.set("name", metric)
            m.set("threshold", str(threshold))

    # Reference
    if case.reference:
        ref = ET.SubElement(root, "reference")
        ref.set("type", case.reference.type)
        if case.reference.citation:
            ET.SubElement(ref, "citation").text = case.reference.citation
        if case.reference.doi:
            ET.SubElement(ref, "doi").text = case.reference.doi

    return root


def _bgk_d3q19_template(case: XValYAMLCase) -> ET.Element:
    """Generate XML template for BGK D3Q19 cases."""
    root = ET.Element("lbm_case")
    root.set("version", "1.0")
    root.set("generator", "yaml-xml-pipeline")

    meta = ET.SubElement(root, "metadata")
    ET.SubElement(meta, "anchor").text = case.anchor
    ET.SubElement(meta, "title").text = case.title
    ET.SubElement(meta, "category").text = case.category

    geo = ET.SubElement(root, "geometry")
    geo.set("type", case.geometry.type)
    geo.set("dimensions", case.geometry.dimensions)
    ET.SubElement(geo, "resolution").text = f"{case.geometry.nx}x{case.geometry.ny}x{case.geometry.nz}"
    ET.SubElement(geo, "domain").text = f"{case.geometry.lx}x{case.geometry.ly}x{case.geometry.lz}"

    phys = ET.SubElement(root, "physics")
    phys.set("model", case.physics.model)
    phys.set("lattice", case.physics.lattice)
    if case.physics.reynolds is not None:
        ET.SubElement(phys, "reynolds").text = str(case.physics.reynolds)
    if case.physics.viscosity is not None:
        ET.SubElement(phys, "viscosity").text = str(case.physics.viscosity)

    solv = ET.SubElement(root, "solver")
    solv.set("max_steps", str(case.solver.max_steps))
    solv.set("convergence_threshold", str(case.solver.convergence_threshold))
    solv.set("output_interval", str(case.solver.output_interval))
    solv.set("seed", str(case.solver.seed))
    solv.set("precision", case.solver.precision)

    if case.acceptance and case.acceptance.metrics:
        acc = ET.SubElement(root, "acceptance")
        for metric, threshold in case.acceptance.metrics.items():
            m = ET.SubElement(acc, "metric")
            m.set("name", metric)
            m.set("threshold", str(threshold))

    return root


def _mrt_d3q19_template(case: XValYAMLCase) -> ET.Element:
    """Generate XML template for MRT D3Q19 cases."""
    root = ET.Element("lbm_case")
    root.set("version", "1.0")
    root.set("generator", "yaml-xml-pipeline")
    root.set("collision", "mrt")

    meta = ET.SubElement(root, "metadata")
    ET.SubElement(meta, "anchor").text = case.anchor
    ET.SubElement(meta, "title").text = case.title

    geo = ET.SubElement(root, "geometry")
    geo.set("type", case.geometry.type)
    geo.set("dimensions", case.geometry.dimensions)
    ET.SubElement(geo, "resolution").text = f"{case.geometry.nx}x{case.geometry.ny}x{case.geometry.nz}"

    phys = ET.SubElement(root, "physics")
    phys.set("model", case.physics.model)
    phys.set("lattice", case.physics.lattice)
    if case.physics.reynolds is not None:
        ET.SubElement(phys, "reynolds").text = str(case.physics.reynolds)

    solv = ET.SubElement(root, "solver")
    solv.set("max_steps", str(case.solver.max_steps))
    solv.set("collision", "mrt")
    solv.set("precision", case.solver.precision)

    return root


# ---------------------------------------------------------------------------
# Registry population
# ---------------------------------------------------------------------------

_default_registry = XMLTemplateRegistry()
_default_registry.register("bgk", "D2Q9", _bgk_d2q9_template)
_default_registry.register("bgk", "D3Q19", _bgk_d3q19_template)
_default_registry.register("mrt", "D3Q19", _mrt_d3q19_template)


def get_default_registry() -> XMLTemplateRegistry:
    """Get the default template registry with built-in templates."""
    return _default_registry


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_xml_template(
    case: XValYAMLCase,
    registry: Optional[XMLTemplateRegistry] = None,
) -> ET.Element:
    """Generate an XML template from a YAML case definition.

    Parameters
    ----------
    case : XValYAMLCase
        Parsed YAML case definition.
    registry : XMLTemplateRegistry, optional
        Template registry. Uses default registry if not provided.

    Returns
    -------
    xml.etree.ElementTree.Element
        Root element of the generated XML template.

    Raises
    ------
    ValueError
        If no template is registered for the case's (model, lattice) pair.
    """
    if registry is None:
        registry = get_default_registry()

    template_fn = registry.get(case.physics.model, case.physics.lattice)
    if template_fn is None:
        raise ValueError(
            f"No XML template registered for model='{case.physics.model}', "
            f"lattice='{case.physics.lattice}'. "
            f"Available: {registry.list_templates()}"
        )

    return template_fn(case)


def xml_to_string(root: ET.Element, indent: str = "  ") -> str:
    """Convert an XML element tree to a formatted string."""
    rough = ET.tostring(root, encoding="unicode")
    # Simple pretty-print
    import xml.dom.minidom
    dom = xml.dom.minidom.parseString(rough)
    return dom.toprettyxml(indent=indent)


def save_xml_template(
    case: XValYAMLCase,
    output_path: Path,
    registry: Optional[XMLTemplateRegistry] = None,
) -> Path:
    """Generate and save an XML template to file.

    Parameters
    ----------
    case : XValYAMLCase
        Parsed YAML case definition.
    output_path : Path
        Output file path for the XML template.
    registry : XMLTemplateRegistry, optional
        Template registry.

    Returns
    -------
    Path
        The output file path.
    """
    root = generate_xml_template(case, registry)
    xml_str = xml_to_string(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(xml_str)
    return output_path
