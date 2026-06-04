# TensorLBM Observability Notes

## Job lifecycle visibility (platform)

- Job status transitions are emitted through WebSocket `/ws` as `job_update`.
- `GET /api/jobs/{id}` returns current status and result payload.
- `GET /api/jobs/{id}/logs` returns line-oriented runtime logs.

## Output schema expectations

- `run_metadata.json`: normalized run config + derived values + diagnostics.
- `*.csv`: structured time-series diagnostics (`forces.csv`, benchmark-specific outputs).
- `flow_step_XXXXXX.png`: canonical step image artifact.
- `checkpoint_XXXXXX.pt`: restartable field checkpoint.

## Failure triage sequence

1. Check `/api/jobs/{id}` status and `error` field.
2. Inspect `/api/jobs/{id}/logs` for first exception point.
3. Inspect `run_metadata.json` and nearest checkpoint/step image.

