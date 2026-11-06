#!/usr/bin/env python3
"""Collect the GitHub Actions evidence behind the 2026-11-06 main-CI triage.

The triage question is narrow and must not be answered from memory: were the
red jobs on main *failing tests* or *cancelled before a runner picked them up*?
This script answers it from the Actions API and writes ``summary.json`` so every
number in ``README.md`` is produced by a program, never copied by hand.

Usage::

    python evidence/ci_triage_20261106/collect.py            # writes summary.json
    python evidence/ci_triage_20261106/collect.py --offline  # re-render README tables only

The public API is used unauthenticated (60 requests/hour); set GITHUB_TOKEN to
raise the limit.  Network failures are reported as such instead of guessed.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = "cxs503/TensorLBM"
WORKFLOW = "ci.yml"
TRIAGE_SHA = "1ef76442af63bf68bd00b449a72ef1440823fc06"
OUT = Path(__file__).resolve().parent / "summary.json"


def api(path: str) -> dict:
    url = path if path.startswith("https://") else f"https://api.github.com{path}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "tensorlbm-ci-triage",
            **(
                {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"}
                if os.environ.get("GITHUB_TOKEN")
                else {}
            ),
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def seconds_between(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return (
        datetime.strptime(end, fmt).replace(tzinfo=timezone.utc)
        - datetime.strptime(start, fmt).replace(tzinfo=timezone.utc)
    ).total_seconds()


def job_record(job: dict) -> dict:
    steps = job.get("steps") or []
    return {
        "name": job["name"],
        "status": job["status"],
        "conclusion": job["conclusion"],
        "labels": job.get("labels"),
        "runner_name": job.get("runner_name"),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "lifetime_seconds": seconds_between(job.get("started_at"), job.get("completed_at")),
        "steps_recorded": len(steps),
        "steps_conclusions": sorted(
            {step["conclusion"] for step in steps if step.get("conclusion")}
        ),
    }


def run_record(run: dict) -> dict:
    return {
        "id": run["id"],
        "head_sha": run["head_sha"],
        "event": run["event"],
        "status": run["status"],
        "conclusion": run["conclusion"],
        "created_at": run["created_at"],
        "run_started_at": run.get("run_started_at"),
        "updated_at": run["updated_at"],
        "lifetime_seconds": seconds_between(run.get("run_started_at"), run["updated_at"]),
    }


def collect() -> dict:
    runs = api(f"/repos/{REPO}/actions/workflows/{WORKFLOW}/runs?per_page=40")["workflow_runs"]
    triage_run = next(
        run
        for run in runs
        if run["head_sha"] == TRIAGE_SHA and run["event"] == "push" and run["status"] == "completed"
    )
    jobs = api(f"/repos/{REPO}/actions/runs/{triage_run['id']}/jobs?per_page=100")["jobs"]
    conclusions: dict[str, int] = {}
    for run in runs:
        key = f"{run['status']}/{run['conclusion']}"
        conclusions[key] = conclusions.get(key, 0) + 1
    return {
        "collected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repository": REPO,
        "workflow": WORKFLOW,
        "triage_sha": TRIAGE_SHA,
        "triage_run": run_record(triage_run),
        "triage_jobs": [job_record(job) for job in jobs],
        "job_conclusions": sorted({job["conclusion"] for job in jobs}),
        "max_job_lifetime_seconds": max(
            (record["lifetime_seconds"] or 0.0 for record in map(job_record, jobs)), default=0.0
        ),
        "failures_reported_by_any_job": [
            record["name"] for record in map(job_record, jobs) if record["conclusion"] == "failure"
        ],
        "recent_runs": [run_record(run) for run in runs],
        "recent_run_conclusions": conclusions,
    }


def render(data: dict) -> str:
    run = data["triage_run"]
    lines = [
        "| job | conclusion | runner | lifetime (s) | steps recorded |",
        "| --- | --- | --- | --- | --- |",
    ]
    for job in data["triage_jobs"]:
        lines.append(
            f"| {job['name']} | {job['conclusion']} | {job['runner_name'] or '—'} | "
            f"{job['lifetime_seconds']:.0f} | {job['steps_recorded']} |"
        )
    lines += [
        "",
        f"run {run['id']} (event `{run['event']}`, sha `{run['head_sha'][:8]}`) "
        f"lived {run['lifetime_seconds']:.0f} s; "
        f"jobs that reported a failure: {data['failures_reported_by_any_job'] or 'none'}.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline", action="store_true", help="re-render from the existing summary.json"
    )
    args = parser.parse_args()
    if args.offline:
        data = json.loads(OUT.read_text(encoding="utf-8"))
    else:
        try:
            data = collect()
        except (urllib.error.URLError, StopIteration) as error:
            print(f"[collect] GitHub API unavailable: {error}")
            return 1
        OUT.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(render(data))
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
