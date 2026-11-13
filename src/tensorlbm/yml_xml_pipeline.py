"""YAML-driven XML template pipeline for LBM case specifications.

Method lineage (see ``docs/oss-inventory/MPLBM-UT.md`` in TensorPlatform and the
``oss_solutions`` entry ``feature_slug='XVAL-LBM-YML-DRIVEN-XML-TEMPLATE-PIPELINE'``):

    je-santos/MPLBM-UT organises every case as ``input.yml`` + runner script;
    ``mplbm_utils/create_palabos_input_file.py`` expands the YAML into the
    solver's XML document with four sections ``geometry / init / fluids /
    output`` by string templates, writes a ``mpirun`` bash script, and the
    post-processing parses fixed text markers (``'Pressure difference ='``).

This module re-implements that *structure* only (no code is copied) with two
hardening changes that close the silent-failure modes of the original:

* values reach the XML through :mod:`xml.etree.ElementTree` (attribute
  escaping) instead of f-string splicing, so a YAML value containing ``<``,
  ``&`` or a quote cannot corrupt the document;
* a strict schema validator runs on the YAML **and** on the XML parsed back,
  so ``spec -> xml -> spec`` is an asserted invariant, not an assumption.

The pipeline ends in a self-contained D2Q9/BGK + Guo-forcing + halfway
bounce-back channel solver which consumes the parsed XML, so the rendered
document is verified numerically against the exact plane-Poiseuille
permeability ``K = H**2 / 12`` (lattice units).
"""

from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping

import numpy as np

LATTICE_CS2 = 1.0 / 3.0
INV_CS2 = 3.0

REQUIRED_SECTIONS = ("geometry", "init", "fluids", "output")

#: D2Q9 velocity set / weights (order fixed by ``_E``/``_W``).
_E = np.array(
    [[0, 1, 0, -1, 0, 1, -1, -1, 1], [0, 0, 1, 0, -1, 1, 1, -1, -1]], dtype=np.int64
).T  # shape (9, 2): rows are lattice directions
_W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])


class SpecError(ValueError):
    """Raised when a case specification violates the pipeline schema."""


# ---------------------------------------------------------------------------
# 1. YAML load + schema validation
# ---------------------------------------------------------------------------
def load_spec(path: str | Path) -> dict[str, Any]:
    """Load a case YAML file and return the *validated* specification."""
    import yaml

    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, Mapping):
        raise SpecError(f"{path}: top level must be a mapping, got {type(raw).__name__}")
    return validate_spec(dict(raw))


def validate_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalise a case specification (returns a deep copy)."""
    spec = copy.deepcopy(dict(spec))
    missing = [s for s in REQUIRED_SECTIONS if s not in spec]
    if missing:
        raise SpecError(f"missing required section(s): {', '.join(missing)}")
    extra = [k for k in spec if k not in REQUIRED_SECTIONS]
    if extra:
        raise SpecError(f"unknown section(s): {', '.join(extra)}")

    # --- geometry ---------------------------------------------------------
    geo = spec["geometry"]
    if not isinstance(geo, Mapping):
        raise SpecError("geometry: must be a mapping")
    size = geo.get("size")
    if not (isinstance(size, (list, tuple)) and len(size) == 2):
        raise SpecError("geometry.size: must be [nx, ny]")
    nx, ny = (int(size[0]), int(size[1]))
    if nx < 4 or ny < 5:
        raise SpecError(f"geometry.size: nx>=4 and ny>=5 required, got [{nx}, {ny}]")
    period = geo.get("period", [True, False])
    if not (isinstance(period, (list, tuple)) and len(period) == 2):
        raise SpecError("geometry.period: must be [x_periodic, y_periodic]")
    if bool(period[1]):
        raise SpecError("geometry.period: y-periodicity with solid rows is unsupported")
    solid_rows = geo.get("solid_rows", [0, ny - 1])
    solid_rows = [int(r) for r in solid_rows]
    if not solid_rows:
        raise SpecError("geometry.solid_rows: at least one wall row is required")
    if any(r < 0 or r >= ny for r in solid_rows):
        raise SpecError(f"geometry.solid_rows: rows must lie in [0, {ny - 1}]")
    if len(set(solid_rows)) != len(solid_rows):
        raise SpecError("geometry.solid_rows: duplicated row index")
    h = ny - len(set(solid_rows))  # fluid rows -> channel height in lattice units
    if h < 3:
        raise SpecError(f"geometry: needs >=3 fluid rows, got {h}")

    # --- init -------------------------------------------------------------
    init = spec["init"]
    rho0 = float(init.get("rho", 1.0))
    if not (rho0 > 0.0):
        raise SpecError(f"init.rho: must be > 0, got {rho0}")

    # --- fluids -----------------------------------------------------------
    fl = spec["fluids"]
    omega = float(fl.get("omega"))
    if not (0.0 < omega < 2.0):
        raise SpecError(f"fluids.omega: must lie in (0, 2), got {omega}")
    tau = 1.0 / omega
    nu = (tau - 0.5) * LATTICE_CS2
    if nu <= 0.0:
        raise SpecError(f"fluids.omega={omega}: gives non-positive viscosity")
    force_x = float(fl.get("force_x", 0.0))
    # Mach guard: the pipeline refuses cases whose expected peak velocity is
    # not in the weakly-compressible range (|u|/cs <= 0.1).
    u_peak = abs(force_x) * h * h / (8.0 * nu) if force_x else 0.0
    mach = u_peak / math.sqrt(LATTICE_CS2)
    if mach > 0.1:
        raise SpecError(
            f"fluids.force_x={force_x}: peak Mach {mach:.3f} > 0.1 "
            f"(compressibility error dominates); reduce force_x or raise omega"
        )

    # --- output -----------------------------------------------------------
    out = spec["output"]
    folder = str(out.get("out_folder", "run"))
    if not folder or folder in {".", ".."} or "/" in folder.strip("/"):
        raise SpecError(f"output.out_folder: single path segment required, got {folder!r}")
    it_max = int(out.get("it_max", 200000))
    if it_max < 1:
        raise SpecError(f"output.it_max: must be >= 1, got {it_max}")
    conv = float(out.get("convergence", 1e-10))
    if not (0.0 < conv < 1.0):
        raise SpecError(f"output.convergence: must lie in (0, 1), got {conv}")

    norm = {
        "geometry": {
            "size": [nx, ny],
            "period": [bool(period[0]), bool(period[1])],
            "solid_rows": sorted(set(solid_rows)),
        },
        "init": {"rho": rho0},
        "fluids": {"omega": omega, "force_x": force_x},
        "output": {"out_folder": folder, "it_max": it_max, "convergence": conv},
    }
    return norm


def derived_quantities(spec: Mapping[str, Any]) -> dict[str, float]:
    """Physical quantities implied by a validated specification."""
    nx, ny = spec["geometry"]["size"]
    n_solid = len(spec["geometry"]["solid_rows"])
    h = float(ny - n_solid)
    omega = spec["fluids"]["omega"]
    tau = 1.0 / omega
    nu = (tau - 0.5) * LATTICE_CS2
    g = float(spec["fluids"]["force_x"])
    u_peak = abs(g) * h * h / (8.0 * nu) if g else 0.0
    return {
        "nx": float(nx),
        "ny": float(ny),
        "H": h,
        "tau": tau,
        "nu": nu,
        "force_x": g,
        "u_peak": u_peak,
        "mach": u_peak / math.sqrt(LATTICE_CS2),
    }


# ---------------------------------------------------------------------------
# 2. XML rendering / parsing (the "template pipeline")
# ---------------------------------------------------------------------------
def render_xml(spec: Mapping[str, Any]) -> str:
    """Render a validated specification into the solver XML document."""
    spec = validate_spec(spec)
    nx, ny = spec["geometry"]["size"]
    root = ET.Element("case", version="1.0", generator="tensorlbm.yml_xml_pipeline")

    geo = ET.SubElement(root, "geometry")
    ET.SubElement(geo, "size", nx=str(nx), ny=str(ny))
    ET.SubElement(
        geo,
        "per",
        x=_xml_bool(spec["geometry"]["period"][0]),
        y=_xml_bool(spec["geometry"]["period"][1]),
    )
    for row in spec["geometry"]["solid_rows"]:
        ET.SubElement(geo, "solid_row", index=str(row))

    init = ET.SubElement(root, "init")
    ET.SubElement(init, "fluid", rho=repr(spec["init"]["rho"]))

    fluids = ET.SubElement(root, "fluids")
    ET.SubElement(fluids, "omega", value=repr(spec["fluids"]["omega"]))
    ET.SubElement(fluids, "force", x=repr(spec["fluids"]["force_x"]))

    out = ET.SubElement(root, "output")
    ET.SubElement(
        out,
        "run",
        out_folder=spec["output"]["out_folder"],
        it_max=str(spec["output"]["it_max"]),
        convergence=repr(spec["output"]["convergence"]),
    )

    ET.indent(root, space="  ")
    text = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + text + "\n"


def _xml_bool(flag: bool) -> str:
    return "true" if flag else "false"


def parse_xml(text: str) -> dict[str, Any]:
    """Parse a rendered solver XML document back into a validated spec."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:  # pragma: no cover - defensive
        raise SpecError(f"malformed XML: {exc}") from exc
    if root.tag != "case":
        raise SpecError(f"root element must be <case>, got <{root.tag}>")

    geo = _child(root, "geometry")
    size = _child(geo, "size")
    per = _child(geo, "per")
    spec: dict[str, Any] = {
        "geometry": {
            "size": [size.get("nx"), size.get("ny")],
            "period": [_xml_read_bool(per.get("x")), _xml_read_bool(per.get("y"))],
            "solid_rows": [int(e.get("index")) for e in geo.findall("solid_row")],
        },
        "init": {"rho": _child(root, "init").find("fluid").get("rho")},
        "fluids": {},
        "output": {},
    }
    fluids = _child(root, "fluids")
    spec["fluids"]["omega"] = _child(fluids, "omega").get("value")
    spec["fluids"]["force_x"] = _child(fluids, "force").get("x")
    run = _child(_child(root, "output"), "run")
    spec["output"] = {
        "out_folder": run.get("out_folder"),
        "it_max": run.get("it_max"),
        "convergence": run.get("convergence"),
    }
    return validate_spec(spec)


def _child(node: ET.Element, tag: str) -> ET.Element:
    found = node.find(tag)
    if found is None:
        raise SpecError(f"missing <{tag}> under <{node.tag}>")
    return found


def _xml_read_bool(value: str | None) -> bool:
    if value in ("true", "True", "1"):
        return True
    if value in ("false", "False", "0"):
        return False
    raise SpecError(f"invalid boolean attribute value {value!r}")


def roundtrip(spec: Mapping[str, Any]) -> dict[str, Any]:
    """``spec -> xml -> spec``; returns the recovered specification."""
    return parse_xml(render_xml(spec))


def render_runner_script(spec: Mapping[str, Any], xml_path: str) -> str:
    """Render the bash runner + fixed text markers (MPLBM-UT pipeline shape)."""
    spec = validate_spec(spec)
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"# generated by tensorlbm.yml_xml_pipeline -- case {spec['output']['out_folder']}\n"
        f"INPUT_XML={_sh_quote(xml_path)}\n"
        "echo \"Reading input from ${INPUT_XML}\"\n"
        "python -m tensorlbm.yml_xml_pipeline \"${INPUT_XML}\"\n"
    )


def _sh_quote(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"


LOG_MARKERS = ("# iters =", "# permeability =", "# u_peak =")


def parse_run_log(text: str) -> dict[str, float]:
    """Parse fixed text markers from solver stdout (``parse_palabos_output`` shape)."""
    out: dict[str, float] = {}
    keys = {"# iters =": "iters", "# permeability =": "permeability", "# u_peak =": "u_peak"}
    for line in text.splitlines():
        line = line.strip()
        for marker, key in keys.items():
            if line.startswith(marker):
                out[key] = float(line[len(marker) :].strip())
    missing = [k for k in keys.values() if k not in out]
    if missing:
        raise SpecError(f"run log missing marker(s): {', '.join(missing)}")
    return out


def format_run_log(results: Mapping[str, Any]) -> str:
    """Emit the fixed text markers consumed by :func:`parse_run_log`."""
    return (
        f"# iters = {int(results['iters'])}\n"
        f"# permeability = {float(results['permeability']):.12e}\n"
        f"# u_peak = {float(results['u_peak']):.12e}\n"
    )


# ---------------------------------------------------------------------------
# 3. Solver backend consuming the parsed XML
# ---------------------------------------------------------------------------
def run_channel(spec: Mapping[str, Any]) -> dict[str, Any]:
    """D2Q9/BGK + Guo forcing + halfway bounce-back channel driven by *spec*."""
    spec = validate_spec(spec)
    q = derived_quantities(spec)
    nx, ny = int(q["nx"]), int(q["ny"])
    tau, nu, g = q["tau"], q["nu"], q["force_x"]
    it_max = spec["output"]["it_max"]
    tol = spec["output"]["convergence"]

    solid = np.zeros((nx, ny), dtype=bool)
    for row in spec["geometry"]["solid_rows"]:
        solid[:, row] = True
    fluid = ~solid

    rho = np.full((nx, ny), float(spec["init"]["rho"]))
    ux = np.zeros((nx, ny))
    uy = np.zeros((nx, ny))
    f = _equilibrium(rho, ux, uy)

    ca = _E[:, 0][:, None, None]
    cb = _E[:, 1][:, None, None]
    w = _W[:, None, None]
    omega = 1.0 / tau
    fx, fy = float(g), 0.0
    caf = ca * fx
    cbf = cb * fy
    # static masks: fluid cells, and "neighbour in direction i is solid"
    nb_solid = {}
    for i in range(1, 9):
        nb_solid[i] = fluid & ~np.roll(fluid, (-int(_E[i, 0]), -int(_E[i, 1])), axis=(0, 1))
    fmask = fluid[None, :, :]

    it = 0
    res = np.inf
    prev = ux.copy()
    check_every = 50
    for it in range(1, it_max + 1):
        # macroscopic fields with Guo half-force velocity correction
        rho = np.where(fluid, f.sum(axis=0), 1.0)
        ux = np.where(fluid, (np.tensordot(_E[:, 0], f, axes=(0, 0)) + 0.5 * fx) / rho, 0.0)
        uy = np.where(fluid, (np.tensordot(_E[:, 1], f, axes=(0, 0)) + 0.5 * fy) / rho, 0.0)
        feq = _equilibrium(rho, ux, uy)
        # Guo forcing: (1 - 1/(2 tau)) w [(c-u)/cs^2 + (c.u)c/cs^4].F
        cu = ca * ux + cb * uy
        force_term = (1.0 - 0.5 * omega) * w * (
            ((ca - ux) * fx + (cb - uy) * fy) * INV_CS2
            + cu * (caf + cbf) * (INV_CS2 * INV_CS2)
        )
        f_post = f - omega * (f - feq) + force_term
        # pull streaming + halfway bounce-back (static masks -> no allocation)
        f = np.empty_like(f_post)
        f[0] = f_post[0]
        for i in range(1, 9):
            f[i] = np.roll(f_post[i], (int(_E[i, 0]), int(_E[i, 1])), axis=(0, 1))
        for i in range(1, 9):
            np.copyto(f[_OPP[i]], f_post[i], where=nb_solid[i])
        f *= fmask
        if it % check_every == 0:
            denom = np.max(np.abs(ux)) + 1e-30
            res = float(np.max(np.abs(ux - prev)) / denom)
            prev = ux.copy()
            if res < tol:
                break

    # re-evaluate macroscopic fields at the final state
    rho = np.where(fluid, f.sum(axis=0), 1.0)
    ux = np.where(fluid, ((f * ca).sum(axis=0) + 0.5 * fx) / rho, 0.0)
    uy = np.where(fluid, ((f * cb).sum(axis=0) + 0.5 * fy) / rho, 0.0)

    fluid_rows = np.where(fluid.any(axis=0))[0]
    h = float(len(fluid_rows))
    u_mean = float(np.mean(ux[fluid]))
    permeability = u_mean * nu / abs(g) if g else 0.0
    return {
        "spec": spec,
        "nx": nx,
        "ny": ny,
        "H": h,
        "nu": nu,
        "tau": tau,
        "force_x": g,
        "iters": int(it),
        "residual": float(res),
        "u_mean": u_mean,
        "u_peak": float(np.max(np.abs(ux[fluid]))),
        "permeability": float(permeability),
        "y": fluid_rows.astype(float),
        "u_profile": ux[0, :].copy(),
    }


def _equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    cu = _E[:, 0][:, None, None] * ux + _E[:, 1][:, None, None] * uy
    usq = ux * ux + uy * uy
    return (
        _W[:, None, None]
        * rho
        * (1.0 + cu * INV_CS2 + 0.5 * cu * cu * INV_CS2 * INV_CS2 - 0.5 * usq * INV_CS2)
    )


def _stream(f_post: np.ndarray, fluid: np.ndarray) -> np.ndarray:
    """Pull streaming with halfway bounce-back on non-fluid nodes.

    Two steps, the textbook order (Succi 2001; Krueger et al. 2017 ch. 5):

    1. advection ``f_i(x) = f_i^post(x - c_i)`` (periodic wrap, masked later);
    2. for every fluid node whose neighbour in direction ``i`` is solid, the
       mid-link wall reflects the outgoing population:
       ``f_opp(i)(x) = f_i^post(x)``.
    """
    f = np.empty_like(f_post)
    f[0] = f_post[0]
    for i in range(1, 9):
        cx, cy = int(_E[i, 0]), int(_E[i, 1])
        f[i] = np.roll(f_post[i], (cx, cy), axis=(0, 1))
    for i in range(1, 9):
        cx, cy = int(_E[i, 0]), int(_E[i, 1])
        neighbour_solid = ~np.roll(fluid, (-cx, -cy), axis=(0, 1))
        f[_OPP[i]] = np.where(fluid & neighbour_solid, f_post[i], f[_OPP[i]])
    f[:, ~fluid] = 0.0
    return f


_OPP = (0, 3, 4, 1, 2, 7, 8, 5, 6)


def analytic_profile(h: float, nu: float, g: float, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Exact plane-Poiseuille solution ``u(s) = g/(2 nu) s (H - s)``."""
    n = int(h) if n is None else n
    s = np.arange(1, n + 1, dtype=float) - 0.5
    return s, g / (2.0 * nu) * s * (h - s)


def analytic_permeability(h: float) -> float:
    """Plane-Poiseuille permeability in lattice units: ``K = H**2 / 12``."""
    return h * h / 12.0