"""HTTP and import regressions for job isolation, paths, and study summaries."""

from __future__ import annotations

import ast
import base64
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

import pytest
from backend.routers import jobs, postprocess
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture()
def contract_client():
    app = FastAPI()
    app.include_router(jobs.router, prefix="/api/jobs")
    app.include_router(postprocess.router, prefix="/api/postprocess")
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def make_job(job_manager, monkeypatch, tmp_path):
    monkeypatch.setattr(job_manager, "_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(job_manager, "_jobs", {})

    def create(job_id="job", *, result=None, status="completed", group="study"):
        job = job_manager.Job(
            job_id, job_id, "test", {"study": {"group": group, "design_point": {"re": 100}}}
        )
        job.status = job_manager.JobStatus(status)
        job.result = result or {}
        job_manager._jobs[job_id] = job
        return job

    return create


@pytest.mark.parametrize("kind", ["files", "images"])
@pytest.mark.parametrize("attack", ["parent", "prefix-sibling", "absolute", "symlink"])
def test_job_download_rejects_escape(contract_client, make_job, tmp_path, kind, attack):
    job = make_job()
    sibling = tmp_path / "job-other"
    sibling.mkdir()
    secret = sibling / "secret.png"
    secret.write_bytes(b"must not escape")
    paths = {
        "parent": "../job-other/secret.png",
        "prefix-sibling": "../job-other/secret.png",
        "absolute": str(secret),
        "symlink": "linked.png",
    }
    if attack == "parent":
        (tmp_path / "secret.png").write_bytes(b"must not escape")
        paths[attack] = "../secret.png"
    if attack == "symlink":
        (job.output_dir / "linked.png").symlink_to(secret)
    response = contract_client.get(f"/api/jobs/{job.job_id}/{kind}/{quote(paths[attack], safe='')}")
    assert response.status_code == 403, response.text
    assert "must not escape" not in response.text


@pytest.mark.parametrize("kind", ["files", "images"])
def test_job_download_preserves_success_and_not_found(contract_client, make_job, kind):
    job = make_job()
    (job.output_dir / "inside.png").write_bytes(b"owned image")
    response = contract_client.get(f"/api/jobs/job/{kind}/inside.png")
    assert response.status_code == 200
    if kind == "files":
        assert response.content == b"owned image"
    else:
        assert base64.b64decode(response.json()["data"].split(",")[1]) == b"owned image"
    assert contract_client.get(f"/api/jobs/job/{kind}/missing.png").status_code == 404
    assert contract_client.get(f"/api/jobs/missing/{kind}/inside.png").status_code == 404


def test_safe_path_allows_nested_and_internal_symlinks(tmp_path):
    root = tmp_path / "root"
    (root / "nested").mkdir(parents=True)
    target = root / "nested" / "data"
    target.write_text("owned")
    (root / "link").symlink_to(target)
    assert jobs._safe_job_path(root, "nested/data") == target
    assert jobs._safe_job_path(root, str(target)) == target
    assert jobs._safe_job_path(root, "link") == target
    assert jobs._safe_job_path(root, "nested/../nested/data") == target


CHECKPOINT_ROUTES = ["field-data", "field-data-3d", "export", "export-vtk"]


@pytest.mark.parametrize("route", CHECKPOINT_ROUTES)
@pytest.mark.parametrize(
    "attack",
    [
        "prefix-sibling",
        "absolute",
        "directory-symlink",
        "file-symlink",
        "latest-symlink",
        "metadata-symlink",
    ],
)
def test_checkpoint_rejects_escape_before_loading(
    contract_client, make_job, tmp_path, monkeypatch, route, attack
):
    import tensorlbm

    job = make_job()
    sibling = tmp_path / "job-other"
    sibling.mkdir()
    secret = sibling / "checkpoint_f.pt"
    secret.write_bytes(b"not a tensor: loading this is a failure")
    checkpoint = "../job-other"
    if attack == "absolute":
        checkpoint = str(sibling)
    elif attack == "directory-symlink":
        (job.output_dir / "linked").symlink_to(sibling, target_is_directory=True)
        checkpoint = "linked"
    elif attack in {"file-symlink", "latest-symlink", "metadata-symlink"}:
        directory = job.output_dir / "checkpoint"
        directory.mkdir()
        if attack == "metadata-symlink":
            (directory / "checkpoint_f.pt").write_bytes(b"owned")
            (directory / "checkpoint_meta.json").symlink_to(secret)
        else:
            (directory / "checkpoint_f.pt").symlink_to(secret)
        checkpoint = "latest" if attack == "latest-symlink" else "checkpoint"

    loaded = []

    def forbidden_load(*args, **kwargs):
        loaded.append(args)
        raise AssertionError("Untrusted checkpoint was loaded")

    monkeypatch.setattr(tensorlbm, "load_checkpoint", forbidden_load)
    response = contract_client.get(
        f"/api/postprocess/{route}/job", params={"checkpoint": checkpoint}
    )
    assert response.status_code == 403, response.text
    assert loaded == []


@pytest.mark.parametrize("route", CHECKPOINT_ROUTES)
def test_checkpoint_missing_remains_404(contract_client, make_job, route):
    make_job()
    response = contract_client.get(
        f"/api/postprocess/{route}/job", params={"checkpoint": "missing"}
    )
    assert response.status_code == 404, response.text


def test_study_reads_real_results_without_expanding_polling(contract_client, make_job):
    make_job("a", result={"cd": 0.8, "series": [0.6, 0.8]})
    make_job("b", result={"cd": 0.4})
    make_job("other", result={"cd": -100}, group="another-study")
    response = contract_client.get("/api/postprocess/study-compare/study")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["n_total"] == data["n_completed"] == 2
    assert data["metric_summary"]["cd"]["min"] == 0.4
    assert data["metric_summary"]["cd"]["mean"] == pytest.approx(0.6)
    a = next(row for row in data["jobs"] if row["job_id"] == "a")
    assert a["metrics"]["series_final"] == 0.8
    assert a["metrics"]["series_mean"] == pytest.approx(0.7)
    listing = contract_client.get("/api/jobs/").json()
    assert all(row["result"] == {} for row in listing["jobs"])
    assert jobs.job_manager.get_job("a").to_dict(include_result=False)["result"] == {}


def test_study_best_job_matches_sparse_metric(contract_client, make_job):
    # list_jobs sorts newest first: the first completed row has no cd.
    make_job("winner", result={"cd": 0.4})
    make_job("loser", result={"cd": 0.8})
    make_job("no-cd", result={"re": 100})
    make_job("running", result={"cd": -1}, status="running")
    response = contract_client.get("/api/postprocess/study-compare/study")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["n_total"] == 4
    assert data["n_completed"] == 3
    assert data["metric_summary"]["cd"]["best_job_id"] == "winner"
    assert data["metric_summary"]["cd"]["min"] == 0.4


def test_study_tolerates_job_removed_after_listing(contract_client, make_job, monkeypatch):
    make_job()
    monkeypatch.setattr(jobs.job_manager, "get_job", lambda job_id: None)
    response = contract_client.get("/api/postprocess/study-compare/study")
    assert response.status_code == 404


def test_required_routers_share_application_job_manager(app, job_manager):
    from backend import main

    assert main.app is app
    assert main.job_manager is job_manager
    for name in ("jobs", "postprocess", "solver", "orchestration"):
        module = main._router_imports[name]
        assert module.job_manager is job_manager


@pytest.mark.parametrize("namespace", ["backend", "app.backend"])
def test_entrypoint_keeps_router_namespace_consistent(namespace):
    root = Path(__file__).resolve().parents[2]
    script = f"""
import importlib
main = importlib.import_module('{namespace}.main')
for name in ('jobs', 'postprocess', 'solver', 'orchestration'):
    router = main._router_imports[name]
    assert router.job_manager is main.job_manager, name
"""
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            (str(root), str(root / "app"), os.environ.get("PYTHONPATH", ""))
        ),
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_platform_tests_use_only_served_backend_namespace():
    # Keep independent module instances from returning via future test imports.
    violations = []
    for path in Path(__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            if any(name == "app.backend" or name.startswith("app.backend.") for name in names):
                violations.append(f"{path.name}:{node.lineno}")
    assert not violations, violations


@pytest.mark.parametrize("router,required", [("jobs", True), ("agent", False)])
def test_router_import_failure_policy(router, required):
    root = Path(__file__).resolve().parents[2]
    script = f"""
import importlib
import builtins
original_builtin = builtins.__import__
def failing_builtin(name, *args, **kwargs):
    if name == 'backend.routers.{router}':
        raise RuntimeError('injected router import failure')
    return original_builtin(name, *args, **kwargs)
builtins.__import__ = failing_builtin
original = importlib.import_module
def failing_import(name, package=None):
    if name in ('.routers.{router}', 'backend.routers.{router}'):
        raise RuntimeError('injected router import failure')
    return original(name, package)
importlib.import_module = failing_import
try:
    import backend.main as main
except RuntimeError as exc:
    assert {required!r}, str(exc)
    assert str(exc) == 'injected router import failure'
else:
    assert not {required!r}, 'Required router import failure was swallowed'
    assert '{router}' not in main._router_imports
    assert 'jobs' in main._router_imports
"""
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join((str(root / "app"), os.environ.get("PYTHONPATH", ""))),
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("route", CHECKPOINT_ROUTES)
@pytest.mark.parametrize("checkpoint", ["latest", "checkpoint", "checkpoint/checkpoint_f.pt"])
def test_owned_checkpoint_still_loads(contract_client, make_job, route, checkpoint):
    import torch

    from tensorlbm.checkpoint import save_checkpoint
    from tensorlbm.d2q9 import equilibrium
    from tensorlbm.d3q19 import equilibrium3d

    job = make_job()
    shape = (4, 4, 4) if route == "field-data-3d" else (4, 4)
    rho, zero = torch.ones(shape), torch.zeros(shape)
    f = equilibrium3d(rho, zero, zero, zero) if len(shape) == 3 else equilibrium(rho, zero, zero)
    save_checkpoint(f, 7, job.output_dir / "checkpoint")
    response = contract_client.get(
        f"/api/postprocess/{route}/job", params={"checkpoint": checkpoint}
    )
    assert response.status_code == 200, response.text
    if route.startswith("field-data"):
        assert response.json()["step"] == 7
    else:
        assert response.content


def test_safe_path_rejects_invalid_path_and_symlink_loop(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "loop").symlink_to(root / "loop")
    assert jobs._safe_job_path(root, "bad\x00name") is None
    assert jobs._safe_job_path(root, "loop") is None
