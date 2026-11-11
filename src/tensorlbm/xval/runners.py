"""Tool adapters for XVAL-LBM-TAYLORGREEN.

Each adapter consumes the *same* :class:`~tensorlbm.xval.case_spec.TG2DSpec` and
emits the *same* artefacts, so that the comparison table is built from files
rather than from per-tool bespoke code:

    k_history_<tool>.csv     step,kinetic_energy,umax,phase_deg,peak_x_frac,peak_y_frac
    field_<tool>_<step>.csv  i,j,ux,uy
    run_info_<tool>.json     tool name, version, dtype, wall time, status

Adapters
--------
``run_tensorlbm``  in-house solver (``tensorlbm.d2q9`` + ``tensorlbm.solver``)
``run_palabos``    independent OSS lattice-Boltzmann library (gitlab/unigespc)
``run_openfoam``   independent OSS finite-volume solver (``icoFoam``)

A tool that is not installed returns ``{"available": False, ...}`` with a
machine-readable ``friction`` record instead of raising, so the campaign can
degrade the criteria explicitly (owner's cross-validation protocol allows a
single-reference fallback as long as the degradation is recorded).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import metrics as M
from .case_spec import TG2DSpec

HIST_HEADER = "step,kinetic_energy,umax,phase_deg,peak_x_frac,peak_y_frac"

_PALABOS_SRC = Path(__file__).resolve().parents[3] / "benchmarks" / "xval_taylor_green" / "palabos"


# ------------------------------------------------------------------ helpers
def _write_history(path: Path, rows: list[tuple[float, float, float, float, float, float]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(HIST_HEADER + "\n")
        for r in rows:
            fh.write(",".join(repr(float(x)) for x in r) + "\n")


def load_history(path: str | Path) -> dict[str, list[float]]:
    """Load a ``k_history_*.csv`` written by any tool adapter."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {name: np.atleast_1d(data[name]).astype(float).tolist() for name in data.dtype.names}


def load_field(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a ``field_*.csv`` (``i,j,ux,uy``) into two (n, n) arrays."""
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    n = int(max(raw[:, 0].max(), raw[:, 1].max())) + 1
    ux = np.zeros((n, n))
    uy = np.zeros((n, n))
    idx_i = raw[:, 0].astype(int)
    idx_j = raw[:, 1].astype(int)
    ux[idx_j, idx_i] = raw[:, 2]
    uy[idx_j, idx_i] = raw[:, 3]
    return ux, uy


# ------------------------------------------------- in-house TensorLBM tool
def run_tensorlbm(
    spec: TG2DSpec,
    out_dir: str | Path,
    *,
    dtype: str = "float64",
    threads: int = 8,
) -> dict[str, Any]:
    """Run the case with the in-house D2Q9 BGK solver (real simulation)."""
    import torch

    from ..d2q9 import equilibrium, macroscopic
    from ..solver import collide_bgk, stream

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(threads)))
    dt = torch.float64 if dtype == "float64" else torch.float32
    n = spec.n

    jj, ii = torch.meshgrid(
        torch.arange(n, dtype=dt), torch.arange(n, dtype=dt), indexing="ij"
    )
    x = ii + 0.5
    y = jj + 0.5
    ux0 = spec.boost - spec.u0 * torch.cos(spec.k * x) * torch.sin(spec.k * y)
    uy0 = spec.u0 * torch.sin(spec.k * x) * torch.cos(spec.k * y)
    rho = torch.ones_like(ux0)
    f = equilibrium(rho, ux0, uy0)

    tau = spec.tau
    rows: list[tuple[float, float, float, float, float, float]] = []
    field_err: dict[int, float] = {}

    def sample(step: int) -> None:
        _, ux, uy = macroscopic(f)
        ux_np = ux.detach().numpy()
        uy_np = uy.detach().numpy()
        k_energy = float(0.5 * (ux_np**2 + uy_np**2).mean())
        umax = float(np.sqrt((ux_np**2 + uy_np**2).max()))
        phase = M.mode_phase_deg(spec, ux_np)
        px, py = M.peak_position_fraction(spec, ux_np, uy_np)
        rows.append((float(step), k_energy, umax, phase, px, py))
        if step in spec.field_steps:
            np.savetxt(
                out / f"field_tensorlbm_{step}.csv",
                np.column_stack(
                    [
                        np.tile(np.arange(n), n),
                        np.repeat(np.arange(n), n),
                        ux_np.ravel(),
                        uy_np.ravel(),
                    ]
                ),
                delimiter=",",
                header="i,j,ux,uy",
                comments="",
                fmt=["%d", "%d", "%.17g", "%.17g"],
            )
            ex, ey = M.exact_velocity(spec, step)
            field_err[step] = M.field_l2_error_pct(ux_np, uy_np, ex, ey)

    wall0 = time.time()
    with torch.no_grad():
        sample(0)
        for step in range(1, spec.steps + 1):
            f = collide_bgk(stream(f), tau)
            if step % spec.sample_every == 0:
                sample(step)
    wall = time.time() - wall0

    _write_history(out / "k_history_tensorlbm.csv", rows)
    info = {
        "tool": "tensorlbm",
        "available": True,
        "engine": "in-house D2Q9 BGK (torch)",
        "dtype": dtype,
        "device": "cpu",
        "threads": threads,
        "steps": spec.steps,
        "wall_sec": wall,
        "torch_version": torch.__version__,
    }
    (out / "run_info_tensorlbm.json").write_text(json.dumps(info, indent=2))
    return {
        "tool": "tensorlbm",
        "available": True,
        "run_info": info,
        "history": load_history(out / "k_history_tensorlbm.csv"),
        "field_l2_pct": {int(k): v for k, v in field_err.items()},
        "wall_sec": wall,
    }


# ------------------------------------------------------ palabos (OSS) tool
def palabos_binary() -> Path | None:
    for cand in (
        Path(os.environ.get("XVAL_PALABOS_BIN", "")) if os.environ.get("XVAL_PALABOS_BIN") else None,
        _PALABOS_SRC / "bin" / "tg2d_palabos",
    ):
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            return Path(cand)
    return None


def run_palabos(spec: TG2DSpec, out_dir: str | Path, *, timeout: int = 1800) -> dict[str, Any]:
    """Run the case with palabos (independent OSS codebase, D2Q9 BGK)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    binary = palabos_binary()
    if binary is None:
        return {
            "tool": "palabos",
            "available": False,
            "friction": {
                "reason": "palabos reference binary not built",
                "remedy": "benchmarks/xval_taylor_green/palabos/build.sh (PLB_ROOT/PLB_BUILD)",
            },
        }
    argv = [str(binary)] + spec.palabos_argv(str(out))
    wall0 = time.time()
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    wall = time.time() - wall0
    (out / "palabos_stdout.log").write_text((proc.stdout or "") + (proc.stderr or ""))
    if proc.returncode != 0:
        raise RuntimeError(f"palabos run failed rc={proc.returncode}: {proc.stderr[-2000:]}")
    field_err: dict[int, float] = {}
    for step in spec.field_steps:
        fpath = out / f"field_palabos_{step}.csv"
        if fpath.is_file():
            ux, uy = load_field(fpath)
            ex, ey = M.exact_velocity(spec, step)
            field_err[step] = M.field_l2_error_pct(ux, uy, ex, ey)
    info = json.loads((out / "run_info_palabos.json").read_text())
    info.update({"available": True, "wall_sec": wall, "binary": str(binary)})
    return {
        "tool": "palabos",
        "available": True,
        "run_info": info,
        "history": load_history(out / "k_history_palabos.csv"),
        "field_l2_pct": field_err,
        "wall_sec": wall,
    }


# ---------------------------------------------------- OpenFOAM icoFoam tool
def ico_foam_binary() -> str | None:
    cand = os.environ.get("XVAL_ICOFOAM", "")
    if cand and Path(cand).is_file():
        return cand
    return shutil.which("icoFoam")


def _foam_header(cls: str, obj: str, location: str) -> str:
    return (
        "FoamFile\n{\n"
        f"    version 2.0;\n    format ascii;\n    class {cls};\n"
        f'    location "{location}";\n    object {obj};\n}}\n'
    )


def write_openfoam_case(spec: TG2DSpec, case_dir: str | Path) -> Path:
    """Write a 2-D cyclic icoFoam case for the Taylor-Green vortex.

    Mesh: N x N x 1 hexahedra, cyclic in x and y, ``empty`` front/back (2-D).
    ``0/U``/``0/p`` are written as explicit non-uniform lists holding the same
    analytic initial condition (and the exact pressure) used by the lattice
    tools, so no tool-specific initialisation utility is involved.
    ``0/p`` carries the kinematic pressure p/rho of the TG solution.
    """
    case = Path(case_dir)
    for sub in ("0", "constant", "system"):
        (case / sub).mkdir(parents=True, exist_ok=True)
    n = spec.n
    L = spec.L

    (case / "system" / "blockMeshDict").write_text(
        _foam_header("dictionary", "blockMeshDict", "system")
        + f"""
convertToMeters 1;

vertices
(
    (0 0 0) ({L} 0 0) ({L} {L} 0) (0 {L} 0)
    (0 0 1) ({L} 0 1) ({L} {L} 1) (0 {L} 1)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({n} {n} 1) simpleGrading (1 1 1)
);

edges
();

boundary
(
    xMin {{ type cyclic; neighbourPatch xMax; faces (); }}
    xMax {{ type cyclic; neighbourPatch xMin; faces (); }}
    yMin {{ type cyclic; neighbourPatch yMax; faces (); }}
    yMax {{ type cyclic; neighbourPatch yMin; faces (); }}
    frontAndBack {{ type empty; faces (); }}
);

mergePatchPairs
();
"""
    )

    jj, ii = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    x = ii + 0.5
    y = jj + 0.5
    ux = spec.boost - spec.u0 * np.cos(spec.k * x) * np.sin(spec.k * y)
    uy = spec.u0 * np.sin(spec.k * x) * np.cos(spec.k * y)
    # exact kinematic pressure of the 2D Taylor-Green vortex (t = 0)
    p = -0.25 * spec.u0**2 * (np.cos(2 * spec.k * x) + np.cos(2 * spec.k * y))

    def vec_list(a: np.ndarray) -> str:
        return "\n".join(f"({v:.17g} {w:.17g} 0)" for v, w in zip(a.ravel(order="C"), uy.ravel(order="C")))

    u_body = "\n".join(
        f"({v:.17g} {w:.17g} 0)"
        for v, w in zip(ux.ravel(order="C"), uy.ravel(order="C"))
    )
    p_body = "\n".join(f"{v:.17g}" for v in p.ravel(order="C"))
    n_cells = n * n
    (case / "0" / "U").write_text(
        _foam_header("volVectorField", "U", "0")
        + f"""
dimensions [0 1 -1 0 0 0 0];
internalField nonuniform List<vector>
{n_cells}
(
{u_body}
)
;
boundaryField
{{
    xMin {{ type cyclic; }}
    xMax {{ type cyclic; }}
    yMin {{ type cyclic; }}
    yMax {{ type cyclic; }}
    frontAndBack {{ type empty; }}
}}
"""
    )
    (case / "0" / "p").write_text(
        _foam_header("volScalarField", "p", "0")
        + f"""
dimensions [0 2 -2 0 0 0 0];
internalField nonuniform List<scalar>
{n_cells}
(
{p_body}
)
;
boundaryField
{{
    xMin {{ type cyclic; }}
    xMax {{ type cyclic; }}
    yMin {{ type cyclic; }}
    yMax {{ type cyclic; }}
    frontAndBack {{ type empty; }}
}}
"""
    )
    del vec_list

    (case / "constant" / "transportProperties").write_text(
        _foam_header("dictionary", "transportProperties", "constant")
        + f"\nnu              [0 2 -1 0 0 0 0] {spec.nu:.17g};\n"
    )
    (case / "constant" / "momentumTransport").write_text(
        _foam_header("dictionary", "momentumTransport", "constant")
        + "\nsimulationType  laminar;\n"
    )
    (case / "system" / "controlDict").write_text(
        _foam_header("dictionary", "controlDict", "system")
        + f"""
application     icoFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {spec.steps};
deltaT          1;
writeControl    timeStep;
writeInterval   {spec.sample_every};
purgeWrite      0;
writeFormat     ascii;
writePrecision  17;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable false;
"""
    )
    (case / "system" / "fvSchemes").write_text(
        _foam_header("dictionary", "fvSchemes", "system")
        + """
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }
divSchemes      { default none; div(phi,U) Gauss linear; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
"""
    )
    (case / "system" / "fvSolution").write_text(
        _foam_header("dictionary", "fvSolution", "system")
        + """
solvers
{
    p
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-12;
        relTol          0;
    }
    pFinal { $p; relTol 0; }
    U
    {
        solver          PBiCG;
        preconditioner  DILU;
        tolerance       1e-12;
        relTol          0;
    }
}
PISO
{
    nCorrectors     3;
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
}
"""
    )
    return case


def _parse_foam_vector_field(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse an ASCII OpenFOAM ``volVectorField``/``volScalarField`` file."""
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    start = None
    for idx, ln in enumerate(lines):
        if ln.startswith("internalField"):
            start = idx
            break
    if start is None:
        raise ValueError(f"no internalField in {path}")
    if "nonuniform" not in lines[start]:
        raise ValueError(f"uniform internalField unsupported in {path}")
    # skip the count line then the '(' line
    idx = start + 2
    values: list[tuple[float, float]] = []
    while idx < len(lines) and lines[idx] not in (")", ""):
        token = lines[idx]
        if token.startswith("("):
            parts = token.strip("()").split()
            if len(parts) >= 2:
                values.append((float(parts[0]), float(parts[1])))
        idx += 1
    arr = np.asarray(values, dtype=float)
    return arr[:, 0], arr[:, 1]


def run_openfoam(spec: TG2DSpec, out_dir: str | Path, *, timeout: int = 3600) -> dict[str, Any]:
    """Run the case with ``icoFoam`` (independent OSS finite-volume solver)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    binary = ico_foam_binary()
    if binary is None:
        return {
            "tool": "openfoam",
            "available": False,
            "friction": {
                "reason": "icoFoam not found on PATH (XVAL_ICOFOAM unset) -- "
                "no OpenFOAM package for this OS image",
                "remedy": "build OpenFOAM from source (see docs/XVAL_TAYLORGREEN.md) "
                "or export XVAL_ICOFOAM=/path/to/icoFoam",
            },
        }
    case = write_openfoam_case(spec, out / "case")
    env = dict(os.environ)
    wall0 = time.time()
    logs: dict[str, str] = {}
    for cmd in (["blockMesh"], ["checkMesh"], ["icoFoam"]):
        exe = shutil.which(cmd[0]) if cmd[0] != "icoFoam" else binary
        if exe is None:
            logs[cmd[0]] = "unavailable"
            continue
        proc = subprocess.run(
            [exe], cwd=str(case), capture_output=True, text=True, timeout=timeout, env=env
        )
        logs[cmd[0]] = proc.stdout[-4000:] + proc.stderr[-2000:]
        if cmd[0] == "icoFoam" and proc.returncode != 0:
            raise RuntimeError("icoFoam failed:\n" + logs["icoFoam"][-2000:])
    wall = time.time() - wall0
    (out / "openfoam_stdout.log").write_text(json.dumps(logs, indent=1)[:200000])

    # sample every written time directory with the same metric code
    rows: list[tuple[float, float, float, float, float, float]] = []
    field_err: dict[int, float] = {}
    times = sorted(
        int(p.name) for p in case.iterdir() if p.is_dir() and p.name.isdigit()
    )
    for t in times:
        ux_flat, uy_flat = _parse_foam_vector_field(case / str(t) / "U")
        n = spec.n
        ux = ux_flat.reshape(n, n)
        uy = uy_flat.reshape(n, n)
        k_energy = float(0.5 * (ux**2 + uy**2).mean())
        umax = float(np.sqrt((ux**2 + uy**2).max()))
        rows.append(
            (
                float(t),
                k_energy,
                umax,
                M.mode_phase_deg(spec, ux),
                *M.peak_position_fraction(spec, ux, uy),
            )
        )
        if t in spec.field_steps:
            ex, ey = M.exact_velocity(spec, t)
            field_err[t] = M.field_l2_error_pct(ux, uy, ex, ey)
    _write_history(out / "k_history_openfoam.csv", rows)
    for step in spec.field_steps:
        if (case / str(step) / "U").is_file():
            ux_flat, uy_flat = _parse_foam_vector_field(case / str(step) / "U")
            n = spec.n
            np.savetxt(
                out / f"field_openfoam_{step}.csv",
                np.column_stack(
                    [
                        np.tile(np.arange(n), n),
                        np.repeat(np.arange(n), n),
                        ux_flat,
                        uy_flat,
                    ]
                ),
                delimiter=",",
                header="i,j,ux,uy",
                comments="",
                fmt=["%d", "%d", "%.17g", "%.17g"],
            )
    info = {
        "tool": "openfoam",
        "available": True,
        "engine": "icoFoam (PISO, Euler ddt, Gauss linear)",
        "binary": binary,
        "nu": spec.nu,
        "steps": spec.steps,
        "written_times": times,
        "wall_sec": wall,
    }
    (out / "run_info_openfoam.json").write_text(json.dumps(info, indent=2))
    return {
        "tool": "openfoam",
        "available": True,
        "run_info": info,
        "history": load_history(out / "k_history_openfoam.csv"),
        "field_l2_pct": field_err,
        "wall_sec": wall,
    }