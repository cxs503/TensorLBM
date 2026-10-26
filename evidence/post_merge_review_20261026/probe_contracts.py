"""Isolated, dependency-free probes for three post-merge review findings.

Run from the repository root:
    python evidence/post_merge_review_20261026/probe_contracts.py

No API server, solver or real job is started. Files and job registries used by
these probes are temporary. The source code under review is not modified.
This documents current behavior; it is not an acceptance test that approves it.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main():
    scratch = REPO / "artifacts/runs/post-merge-review"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="contract-probe-", dir=scratch) as temp:
        root = Path(temp)
        # job_manager is stdlib-only. Import exactly the two namespace forms
        # used in app/tests and the app startup. Both modules remain isolated
        # to this short-lived Python process and use this temporary output root.
        os.environ["TENSORLBM_OUTPUT_ROOT"] = str(root / "outputs")
        sys.path.insert(0, str(REPO))
        sys.path.insert(0, str(REPO / "app"))
        served = importlib.import_module("backend.job_manager")
        alternate = importlib.import_module("app.backend.job_manager")
        job = served.Job("probe-job", "review probe", "test", {"study": {"group": "probe"}})
        job.status = served.JobStatus.COMPLETED
        job.result = {"drag_coefficient": 0.8, "series": [0.7, 0.8]}
        with served._jobs_lock:
            served._jobs[job.job_id] = job
        try:
            namespace = {
                "same_source_file": Path(served.__file__).resolve()
                == Path(alternate.__file__).resolve(),
                "same_module_object": served is alternate,
                "same_registry_object": served._jobs is alternate._jobs,
                "same_job_class": served.Job is alternate.Job,
                "served_namespace_finds_job": served.get_job(job.job_id) is job,
                "alternate_namespace_finds_job": alternate.get_job(job.job_id) is not None,
            }
            listing = served.list_jobs()
            summary_contract = {
                "direct_job_result": job.result,
                "list_jobs_result": listing[0]["result"],
                "list_jobs_study_group": listing[0]["config"]["study"]["group"],
                "note": "study_compare currently reads metrics from list_jobs_result, which is empty",
            }
        finally:
            with served._jobs_lock:
                served._jobs.pop(job.job_id, None)

        # Extract the real, pure containment function from its AST without
        # importing FastAPI or running router initialization.
        source_path = REPO / "app/backend/routers/jobs.py"
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_safe_job_path"
        )
        namespace_for_function = {"Path": Path}
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"),
            namespace_for_function,
        )
        check_path = namespace_for_function["_safe_job_path"]
        job_root = root / "job-1"
        sibling = root / "job-10"
        unrelated = root / "unrelated"
        for directory in (job_root, sibling, unrelated):
            directory.mkdir()
            (directory / "probe.txt").write_text("temporary review fixture\n")
        sibling_target = check_path(job_root, "../job-10/probe.txt")
        containment = {
            "source": "app/backend/routers/jobs.py",
            "function_line": function.lineno,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "normal_child_accepted": check_path(job_root, "probe.txt") is not None,
            "unrelated_parent_escape_rejected": check_path(job_root, "../unrelated/probe.txt")
            is None,
            "same_prefix_sibling_accepted": sibling_target is not None,
            "sibling_actually_within_root": (sibling / "probe.txt")
            .resolve()
            .is_relative_to(job_root.resolve()),
            "scope": "local pure-function probe only; no HTTP request or real file disclosure",
        }
        print(
            json.dumps(
                {
                    "source_revision": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
                    ).strip(),
                    "python": sys.version.split()[0],
                    "namespace_identity": namespace,
                    "job_result_contract": summary_contract,
                    "path_containment": containment,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
