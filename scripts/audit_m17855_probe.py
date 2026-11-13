"""Board-audit probe for m17855 (anchor XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN, mission m17790).

Runs entirely against the *audited* worktree (/root/tp-agent/work/m17790) via an
explicit sys.path insert, so it reproduces exactly the code the m17790 mission
committed (only reachable through ``TensorLBM/src``, see ``cli`` stage).

Stages (kept separate because the guard run takes ~75 s):
  python3 scripts/audit_m17855_probe.py trace    # NaN onset at shipped defaults
  python3 scripts/audit_m17855_probe.py guard    # registry.run at defaults -> ExampleResult
  python3 scripts/audit_m17855_probe.py cli      # documented CLI, no PYTHONPATH
  python3 scripts/audit_m17855_probe.py figure    # fig from artifacts probe JSON

Outputs: artifacts/board_audit_m17855/probe.json (+ fig_defaults_divergence.png)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

AUDITED = Path("/root/tp-agent/work/m17790")
SRC = AUDITED / "TensorLBM" / "src"
OUT = Path(__file__).resolve().parents[1] / "artifacts" / "board_audit_m17855"
PROBE = OUT / "probe.json"


def _load() -> dict:
    return json.loads(PROBE.read_text()) if PROBE.exists() else {}


def _save(d: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PROBE.write_text(json.dumps(d, indent=2, sort_keys=True))


def _imports():
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import torch  # noqa: F401
    from tensorlbm.verification.lid_driven_cavity import CavityConfig, CavitySolver
    from tensorlbm.examples import registry
    return CavityConfig, CavitySolver, registry


def stage_trace() -> None:
    import torch
    CavityConfig, CavitySolver, _ = _imports()
    cfg = CavityConfig(n=65, re=100.0, u_lid=0.1, max_steps=500, use_rlbm=True, device="cpu")
    solver = CavitySolver(cfg)
    trace, onset = [], None
    for step in range(1, 121):
        solver.step()
        if step % 5 == 0:
            rho, ux, _uy = solver._macroscopic(solver.f)
            nan = bool(torch.isnan(rho).any())
            umax = None if nan else float(ux.abs().max())
            trace.append({"step": step, "max_abs_ux": umax, "nan": nan})
            if nan and onset is None:
                onset = step
    d = _load()
    d["trace_defaults"] = {
        "config": {"n": 65, "re": 100.0, "u_lid": 0.1, "tau": round(cfg.tau, 6),
                   "use_rlbm": True, "dtype": "torch.float64"},
        "nan_onset_step": onset,
        "max_abs_ux_exceeds_lid_by_step": next(
            (t["step"] for t in trace if t["max_abs_ux"] and t["max_abs_ux"] > 0.1), None),
        "trace": trace,
    }
    _save(d)
    print(json.dumps({k: d["trace_defaults"][k] for k in
                      ("config", "nan_onset_step", "max_abs_ux_exceeds_lid_by_step")}, indent=2))


def stage_guard() -> None:
    _, _, registry = _imports()
    res = registry.run("lid_driven_cavity", n=65, re=100, max_steps=500,
                      output_root="/tmp/m17855_audit_defaults", overwrite=True)
    d = _load()
    d["defaults_registry_run"] = {
        "params": {"n": 65, "re": 100, "max_steps": 500},
        "success": res.success, "error": res.error, "elapsed_s": res.elapsed_s,
        "metrics": res.metrics, "artifacts": res.artifacts,
    }
    _save(d)
    print(json.dumps(d["defaults_registry_run"], indent=2))


def stage_smoke() -> None:
    _, _, registry = _imports()
    res = registry.run("lid_driven_cavity", n=17, re=10, u_lid=0.05, max_steps=50,
                       output_root="/tmp/m17855_audit_smoke", overwrite=True)
    d = _load()
    d["ci_smoke_run"] = {"params": {"n": 17, "re": 10, "max_steps": 50},
                         "success": res.success, "error": res.error,
                         "metrics": res.metrics, "artifacts": res.artifacts}
    _save(d)
    print(json.dumps(d["ci_smoke_run"], indent=2))


def stage_cli() -> None:
    proc = subprocess.run([sys.executable, "-m", "tensorlbm.examples.runner", "--list"],
                          cwd=str(AUDITED), capture_output=True, text=True)
    d = _load()
    d["documented_cli"] = {"cmd": "python3 -m tensorlbm.examples.runner --list",
                           "cwd": str(AUDITED), "returncode": proc.returncode,
                           "stderr_tail": proc.stderr.strip().splitlines()[-1:] or []}
    _save(d)
    print(json.dumps(d["documented_cli"], indent=2))


def stage_figure() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = _load()
    tr = d["trace_defaults"]["trace"]
    steps = [t["step"] for t in tr]
    umax = [t["max_abs_ux"] for t in tr]
    onset = d["trace_defaults"]["nan_onset_step"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=140)
    valid = [(s, u) for s, u in zip(steps, umax) if u is not None]
    ax.semilogy([s for s, _ in valid], [u for _, u in valid], "o-", ms=3, lw=1.2,
                label="max|u_x| (example defaults)")
    ax.axhline(0.1, color="tab:green", ls="--", lw=1.0, label="lid speed u_lid = 0.1")
    ax.axvline(onset, color="tab:red", ls=":", lw=1.2, label=f"NaN onset (step {onset})")
    ax.set_xlabel("LBM step")
    ax.set_ylabel("max|u_x| (lattice units)")
    ax.set_title("m17790 example at shipped defaults (n=65, Re=100, tau=0.692)\n"
                 "diverges: Ghia metrics become NaN; success=True anyway", fontsize=8.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = OUT / "fig_defaults_divergence.png"
    fig.savefig(p)
    print(p)


if __name__ == "__main__":
    stage = sys.argv[1]
    {"trace": stage_trace, "guard": stage_guard, "smoke": stage_smoke,
     "cli": stage_cli, "figure": stage_figure}[stage]()