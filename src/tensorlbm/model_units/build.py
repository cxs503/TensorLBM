"""Build and test one model unit, in isolation.

In TCLB a model directory is compiled into its own binary (``CLB/<model>``) with
its own ``OPT`` flags, and CI runs *that binary's* regression test.  The Python
analogue has two obligations, and this module implements exactly those:

``build(unit)``
    Validate the manifest, import **only** the unit's kernel module in a fresh
    interpreter, record the import closure (a model must not drag the whole
    library in) and a deterministic equilibrium probe.  The JSON it writes is
    the unit's build artifact -- the equivalent of ``CLB/<model>`` existing.

``test(unit)``
    Run the unit's own ``case.py`` against its declared analytic/derived
    reference and judge the declared KPI against the declared threshold,
    writing a result artifact.  An honest failure is reported as ``failed`` and
    never retried with a loosened threshold.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

from .manifest import ModelUnit
from .registry import DEFAULT_ROOT, discover

ARTIFACT_ROOT = Path("artifacts") / "model_units"
RUN_ROOT = Path("runs") / "model_units"


def _probe(unit: ModelUnit, timeout: int = 300) -> dict[str, Any]:
    """Import the kernel module alone in a subprocess and probe it."""
    kernel = unit.kernel_source
    code = textwrap.dedent(
        f"""
        import hashlib, json, sys
        import torch

        mod = __import__({kernel!r}, fromlist=["probe"])
        closure = sorted(
            m for m in sys.modules if m.split(".")[0] == "tensorlbm"
        )
        probe = {{"module": {kernel!r}, "closure": closure, "n_modules": len(closure)}}
        torch.manual_seed(0)
        if hasattr(mod, "equilibrium3d"):
            rho = torch.ones(3, 4, 5, dtype=torch.float64)
            feq = mod.equilibrium3d(rho, rho * 0.01, rho * -0.02, rho * 0.03)
            probe["equilibrium"] = "equilibrium3d"
        elif hasattr(mod, "equilibrium"):
            rho = torch.ones(4, 5, dtype=torch.float64)
            feq = mod.equilibrium(rho, rho * 0.01, rho * -0.02)
            probe["equilibrium"] = "equilibrium"
        else:
            raise SystemExit("kernel exposes no equilibrium entry point")
        payload = feq.detach().double().cpu().numpy().round(12).tobytes()
        probe["feq_checksum"] = hashlib.sha256(payload).hexdigest()
        probe["feq_sum"] = float(feq.detach().double().sum())
        probe["finite"] = bool(torch.isfinite(feq).all())
        print(json.dumps(probe))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"unit {unit.unit_id}: kernel import probe failed\n"
            f"stdout={proc.stdout[-2000:]}\nstderr={proc.stderr[-4000:]}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def library_surface() -> int:
    """Number of modules shipped by the ``tensorlbm`` package (on disk).

    Used as the denominator of the isolation ratio: a model build should import
    a handful of modules out of the whole library, not the whole library.  This
    is deliberately a static count -- ``import tensorlbm`` is lazy, so a runtime
    closure of the bare package would understate the surface.
    """
    import tensorlbm

    root = Path(tensorlbm.__file__).resolve().parent
    return sum(1 for _ in root.rglob("*.py"))


def _monolithic_closure() -> int:
    """Backwards-compatible name: the package surface (module count on disk)."""
    return library_surface()


def build(unit: ModelUnit, out_root: Path | str = ARTIFACT_ROOT) -> dict[str, Any]:
    """Build one unit: validation + isolated kernel import + probe artifact."""
    t0 = time.perf_counter()
    probe = _probe(unit)
    mono = _monolithic_closure()
    record: dict[str, Any] = {
        "kind": "model_unit_build",
        "unit": unit.to_dict(),
        "kernel_import": probe,
        "library_surface_modules": mono,
        "monolithic_tensorlbm_modules": mono,
        "isolation_ratio": (probe["n_modules"] / mono) if mono else None,
        "features": list(unit.manifest.get("features", [])),
        "build_seconds": round(time.perf_counter() - t0, 3),
        "python": sys.version.split()[0],
    }
    out = Path(out_root) / f"{unit.name}.build.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record["artifact"] = str(out)
    return record


def _load_case(unit: ModelUnit):
    spec = importlib.util.spec_from_file_location(
        f"tensorlbm_model_unit_{unit.name}", unit.case_path
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load case script {unit.case_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entry = str(unit.manifest["case"]["entry"])
    if not hasattr(module, entry):
        raise AttributeError(f"{unit.case_path}: missing entry point {entry!r}")
    return getattr(module, entry)


def test(unit: ModelUnit, out_root: Path | str = RUN_ROOT) -> dict[str, Any]:
    """Run one unit's case and judge its declared KPI against its threshold."""
    entry = _load_case(unit)
    acc = unit.acceptance
    t0 = time.perf_counter()
    outcome = entry(dict(unit.params))
    seconds = time.perf_counter() - t0

    kpis = {k: float(v) for k, v in dict(outcome.get("kpis", {})).items()}
    kpi_name = str(acc["kpi"])
    if kpi_name not in kpis:
        raise KeyError(f"unit {unit.unit_id}: case did not report declared KPI {kpi_name!r}")
    threshold = float(acc["threshold"])
    value = kpis[kpi_name]
    status = "passed" if value <= threshold else "failed"
    record: dict[str, Any] = {
        "kind": "model_unit_test",
        "unit": unit.to_dict(),
        "kpi": kpi_name,
        "value": value,
        "threshold": threshold,
        "status": status,
        "kpis": kpis,
        "metrics": outcome.get("metrics", {}),
        "curves": outcome.get("curves", {}),
        "seconds": round(seconds, 3),
    }
    out = Path(out_root) / unit.name / "result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record["artifact"] = str(out)
    return record


def doctor(root: Path | str = DEFAULT_ROOT, repo_root: Path | None = None) -> dict[str, Any]:
    """Consistency gates for the unit registry (the CI ``plan`` job's checks)."""
    from .registry import _filesystem_manifests, _git_tracked_manifests, manifest_paths

    root_path = Path(root)
    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    fs = {p.resolve() for p in _filesystem_manifests(root_path)}
    git = set()
    for path in _git_tracked_manifests(repo):
        try:
            path.resolve().relative_to(root_path.resolve())
        except ValueError:
            continue
        git.add(path.resolve())
    union = set(manifest_paths(root_path, repo_root=repo))
    units = discover(root_path, repo_root=repo)

    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, advisory: bool = False) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail, "advisory": advisory})

    add(
        "discovery_union_complete",
        union == (fs | git),
        f"filesystem={len(fs)} git={len(git)} union={len(union)}",
    )
    # Advisory only: TCLB's discovery union exists so a not-yet-committed model
    # still builds locally; the hard gate is that discovery sees every manifest.
    add(
        "filesystem_only_manifests",
        True,
        f"{len(fs - git)} manifest(s) not in the git index (advisory)",
        advisory=True,
    )
    add("all_units_valid", len(units) == len(union), f"units={len(units)} manifests={len(union)}")
    add(
        "every_ci_unit_has_case",
        all(u.case_path.is_file() for u in units if u.ci),
        f"ci_units={sum(1 for u in units if u.ci)}",
    )
    workflow = Path(".github") / "workflows" / "model-units.yml"
    if workflow.is_file():
        text = workflow.read_text(encoding="utf-8")
        names = [u.name for u in units]
        hand_listed = [n for n in names if n in text]
        add("workflow_matrix_not_hand_written", not hand_listed, f"hand listed: {hand_listed}")
        add(
            "workflow_uses_generated_matrix",
            "fromJSON(needs.plan.outputs.matrix)" in text,
            "strategy.matrix must come from the plan job",
        )
    else:  # pragma: no cover - workflow always present in this repo
        add("workflow_present", False, str(workflow))

    return {
        "kind": "model_unit_doctor",
        "root": str(root_path),
        "units": [u.unit_id for u in units],
        "checks": checks,
        "ok": all(c["ok"] for c in checks if not c["advisory"]),
    }
