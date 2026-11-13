"""Reference arm: compile and run the *unmodified* OSS Shan-Chen code.

Reference implementation
------------------------
``lbm-principles-practice/code``, ``chapter9/shanchen.cpp`` (MIT; Krüger,
Kusumaatmaja, Kuzmin, Shardt, Silva, Viggen, *The Lattice Boltzmann Method:
Principles and Practice*, Springer 2016).  Snapshot:
``ref/shanchen.cpp`` (sha256 ``216029bb...537bfd``).

The two-component planar-interface case is the ``nfluids = 2`` branch of that
file (``initialisation()``: ``y < ny/2`` -> (rho_A, rho_B) = (0.1, 0.9), else
(0.9, 0.1); total density 1).  The file ships with ``nfluids = 1``, therefore
every arm here is produced by a **parameter-only** textual patch of the
``const`` declarations (see :data:`PARAM_PATTERNS`).  No statement inside a
function is touched; the patch is recorded as a unified diff and the sha256 of
each generated source is stored in the arm manifest so the reference numbers
are reproducible from the upstream file.

What is measured (zero hand-writing): the ``data/`` files written by the OSS
binary itself (``density_comp0_*``, ``density_comp1_*``, ``press_*``,
``velx_*``, ``vely_*``).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REF_SRC = HERE / "ref" / "shanchen.cpp"
BUILD = HERE / "build"

#: Upstream constants that this module is allowed to patch (declaration line
#: ``const <type> <name> = <value>;``).  Values are written verbatim, so a
#: float stays a float and an int stays an int.
PARAM_PATTERNS = {
    name: re.compile(rf"(const (?:int|double) {name} = )[^;]+(;)")
    for name in (
        "nx", "ny", "nsteps", "noutput", "nfluids", "tauA", "tauB",
        "gA", "gB", "gAB", "rho0", "rhol", "rhog", "radius",
    )
}


def _fmt(name: str, value: Any) -> str:
    return str(int(value)) if name in {"nx", "ny", "nsteps", "noutput", "nfluids"} else repr(float(value))


def make_variant(label: str, params: dict[str, Any]) -> dict[str, Any]:
    """Write ``build/<label>.cpp`` = upstream source + parameter-only patch.

    Returns a manifest dict with the sha256 of the upstream file, of the
    generated file and the unified diff between them.
    """
    import difflib

    BUILD.mkdir(parents=True, exist_ok=True)
    src = REF_SRC.read_text()
    out = src
    applied: dict[str, tuple[str, str]] = {}
    for name, value in params.items():
        if name not in PARAM_PATTERNS:
            raise KeyError(f"{name!r} is not a patchable upstream constant")
        pat = PARAM_PATTERNS[name]
        found = pat.findall(out)
        if len(found) != 1:
            raise RuntimeError(f"{name}: expected exactly 1 declaration, found {len(found)}")
        old = pat.search(out).group(0)
        new = pat.sub(lambda m: m.group(1) + _fmt(name, value) + m.group(2), out, count=1)
        applied[name] = (old.strip(), new.strip())
        out = new
    cpp = BUILD / f"{label}.cpp"
    cpp.write_text(out)
    diff = "".join(
        difflib.unified_diff(
            src.splitlines(True), out.splitlines(True),
            f"ref/shanchen.cpp", f"build/{label}.cpp", n=0,
        )
    )
    manifest = {
        "label": label,
        "params": params,
        "applied": applied,
        "sha256_upstream": hashlib.sha256(src.encode()).hexdigest(),
        "sha256_variant": hashlib.sha256(out.encode()).hexdigest(),
        "diff": diff,
        "source": str(cpp.relative_to(HERE)),
    }
    return manifest


def build(label: str, manifest: dict[str, Any], opt: str = "-O2") -> Path:
    """Compile ``build/<label>.cpp`` into ``build/<label>`` (g++)."""
    exe = BUILD / label
    cmd = ["g++", opt, "-o", str(exe), str(BUILD / f"{label}.cpp")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"g++ failed for {label}:\n{res.stderr}")
    manifest["compile"] = " ".join(cmd)
    manifest["compiler"] = subprocess.run(
        ["g++", "--version"], capture_output=True, text=True
    ).stdout.splitlines()[0]
    return exe


_FIELD_KEYS = ("density_comp0", "density_comp1", "press", "velx", "vely")


def run(label: str, manifest: dict[str, Any], keep: bool = False) -> dict[str, Any]:
    """Run the compiled OSS binary and parse the fields it writes.

    The binary writes ``data/<field>_<step:05d>.dat`` every ``noutput`` steps;
    the last complete set is parsed into ``float64`` arrays of shape (ny, nx).
    """
    exe = build(label, manifest)
    work = BUILD / f"run_{label}"
    if work.exists():
        shutil.rmtree(work)
    (work / "data").mkdir(parents=True)
    shutil.copy(exe, work / exe.name)
    t0 = time.time()
    res = subprocess.run([f"./{exe.name}"], cwd=work, capture_output=True, text=True)
    wall = time.time() - t0
    log = res.stdout
    if res.returncode != 0:
        raise RuntimeError(f"{label}: OSS binary returned {res.returncode}\n{res.stderr[-2000:]}")
    steps = sorted(int(p.stem.split("_")[-1]) for p in (work / "data").glob("density_comp0_*.dat"))
    if not steps:
        raise RuntimeError(f"{label}: OSS binary wrote no data files")
    last = steps[-1]
    fields = {}
    for key in _FIELD_KEYS:
        path = work / "data" / f"{key}_{last:05d}.dat"
        if path.exists():
            fields[key] = np.loadtxt(path)
    # time series of the diagnostic the binary prints for the droplet branch is
    # meaningless for the planar branch, so only a finiteness summary is kept
    finite = bool(np.all(np.isfinite(fields.get("density_comp0", np.array([np.nan])))))
    return {
        "label": label,
        "step_last": last,
        "steps_written": steps,
        "fields": fields,
        "finite": finite,
        "wall_seconds": wall,
        "log_tail": log.strip().splitlines()[-3:],
        "params": manifest.get("params"),
        "manifest": manifest,
    }


def arm(label: str, params: dict[str, Any], keep_fields: bool = True) -> dict[str, Any]:
    """Convenience: variant -> build -> run, returning the parsed result."""
    man = make_variant(label, params)
    res = run(label, man)
    if not keep_fields:
        res.pop("fields", None)
    return res


def save(res: dict[str, Any], outdir: Path) -> Path:
    """Persist one arm as JSON (arrays as lists) under ``outdir``."""
    outdir.mkdir(parents=True, exist_ok=True)
    payload = dict(res)
    fields = payload.pop("fields", {})
    payload["fields_shape"] = {k: list(v.shape) for k, v in fields.items()}
    payload["fields"] = {k: np.asarray(v).tolist() for k, v in fields.items()}
    path = outdir / f"oss_{res['label']}.json"
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    return path


def load(outdir: Path, label: str) -> dict[str, Any]:
    with open(outdir / f"oss_{label}.json") as fh:
        res = json.load(fh)
    res["fields"] = {k: np.asarray(v, dtype=float) for k, v in res["fields"].items()}
    return res


if __name__ == "__main__":  # tiny smoke test of the harness itself
    r = arm("smoke", {"nfluids": 2, "nx": 16, "ny": 16, "nsteps": 20, "noutput": 10})
    print(json.dumps({k: r[k] for k in ("label", "step_last", "finite", "wall_seconds")}, indent=1))
    print({k: v.shape for k, v in r["fields"].items()})