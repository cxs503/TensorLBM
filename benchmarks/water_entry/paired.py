"""Assess a driven entry run against its matched quiescent control."""

import argparse
import csv
import json
from pathlib import Path


def read_run(path):
    with (path / "interface_history.csv").open() as fp:
        history = list(csv.DictReader(fp))
    return json.loads((path / "run_metadata.json").read_text()), history


def assess(driven_directory, control_directory):
    driven, dh = read_run(driven_directory)
    control, ch = read_run(control_directory)
    ignored = {
        "g",
        "drop_speed",
        "require_contact",
        "run_name",
        "output_root",
        "overwrite",
        "output_interval",
    }
    diffs = sorted(
        k
        for k in set(driven["config"]) | set(control["config"])
        if k not in ignored and driven["config"].get(k) != control["config"].get(k)
    )
    nsteps = driven["config"]["n_steps"]
    interval = driven["config"]["track_interval"]
    expected = list(range(0, nsteps + 1, interval))
    if expected[-1] != nsteps:
        expected.append(nsteps)
    event = driven["impact"]["first_bridge_interval_open_closed"]
    bridge_indices = [i for i, row in enumerate(dh) if row["state"] == "first_bridge"]
    event_consistent = False
    if len(bridge_indices) == 1 and bridge_indices[0] > 0:
        i = bridge_indices[0]
        event_consistent = (
            event == [int(dh[i - 1]["step"]), int(dh[i]["step"])]
            and all(r["state"] == "separated" for r in dh[:i])
            and all(r["state"] == "after_first_bridge" for r in dh[i + 1 :])
        )
    checks = {
        "same_configuration_except_forcing_and_output": not diffs,
        "matched_healthy_duration": all(
            r["health"]["passed"] and r["health"]["completed_steps"] == nsteps
            for r in (driven, control)
        ),
        "both_histories_complete": all(
            [int(row["step"]) for row in h] == expected for h in (dh, ch)
        ),
        "both_initially_separated": all(h and h[0]["state"] == "separated" for h in (dh, ch)),
        "control_has_zero_drive": control["config"]["g"] == control["config"]["drop_speed"] == 0,
        "driven_has_input": driven["config"]["g"] != 0 or driven["config"]["drop_speed"] != 0,
        "driven_bridge_observed": event_consistent,
        "control_remains_separated": control["impact"]["first_bridge_interval_open_closed"] is None
        and bool(ch)
        and all(r["state"] == "separated" for r in ch),
    }
    return {
        "schema_version": 1,
        "scope": "resolved liquid bridge versus quiescent control over sampled time window",
        "passed": all(checks.values()),
        "checks": checks,
        "mismatched_keys": diffs,
        "driven_bridge": event,
        "control_bridge": control["impact"]["first_bridge_interval_open_closed"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("driven", type=Path)
    parser.add_argument("control", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = assess(args.driven, args.control)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
