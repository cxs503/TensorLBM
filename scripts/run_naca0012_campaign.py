"""Full campaign driver for the NACA0012 polar (single cluster job).

Order of work, with fail-fast guards so that a physics-level bug is caught
before hours of compute are spent:

1. validation battery (force symmetry, lift sign, bounce-back comparison);
2. main polar sweep (three grids x six incidences at the main Reynolds number);
3. Reynolds support study;
4. test-suite run;
5. summary of what was produced.

Everything is written under ``artifacts/naca0012_polar`` so the job artefacts
can be fetched with ``run_cluster --action fetch``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

OUT = Path("artifacts/naca0012_polar")
THREADS = int(os.environ.get("TP_THREADS", "14"))


def _run_validation() -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _validate_cluster import probe  # noqa: WPS433

    res = {}
    res["a0"] = probe(0.0, nsteps=2000)
    res["a3"] = probe(3.0, nsteps=2000)
    res["a3_potential"] = probe(3.0, nsteps=2000, init="potential")
    res["a3_staircase"] = probe(3.0, nsteps=2000, wall="staircase")
    cl0 = res["a0"][-1][1]
    cl3 = res["a3"][-1][1]
    cd3 = res["a3"][-1][2]
    ok = abs(cl0) < 0.02 and cl3 > 0.0 and cd3 > 0.0
    res["_verdict"] = {"cl_alpha0": cl0, "cl_alpha3": cl3, "cd_alpha3": cd3,
                       "passed": bool(ok)}
    (OUT / "validation.json").write_text(json.dumps(
        {k: (v if k == "_verdict" else [[int(s), float(a), float(b)] for s, a, b in v])
         for k, v in res.items()}, indent=1))
    return res["_verdict"]


def _run_sweep(cases_path: str, workers: int, tag: str) -> None:
    cmd = [sys.executable, "scripts/run_polar_sweep.py", "--cases-file", cases_path,
           "--workers", str(workers), "--threads", str(THREADS),
           "--out", str(OUT)]
    print(f"[campaign] {tag}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=False)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("[campaign] stage 1: validation battery", flush=True)
    verdict = _run_validation()
    print(f"[campaign] validation verdict: {verdict}", flush=True)
    if not verdict["passed"]:
        print("[campaign] ABORT: physics sanity check failed", flush=True)
        return 2

    n_cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    workers = max(2, min(8, n_cores // max(THREADS, 1)))
    print(f"[campaign] stage 2: main sweep with {workers} workers x {THREADS} threads",
          flush=True)
    _run_sweep("artifacts/naca0012_polar/cases/main.json", workers, "main")
    print(f"[campaign] stage 3: Reynolds study  (t={time.time() - t0:.0f}s)", flush=True)
    _run_sweep("artifacts/naca0012_polar/cases/re.json", workers, "re")
    print(f"[campaign] stage 4: tests (t={time.time() - t0:.0f}s)", flush=True)
    with open(OUT / "pytest.log", "w") as fh:
        subprocess.run([sys.executable, "-m", "pytest", "tests/test_naca0012_aero.py",
                        "-q", "-p", "no:cacheprovider"], stdout=fh,
                       stderr=subprocess.STDOUT, check=False)
    print(f"[campaign] done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())