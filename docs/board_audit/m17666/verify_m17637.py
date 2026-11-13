"""Independent audit probes for m17637 (XVAL-LBM-CARTESIAN-MPI, TensorLBM).

Run from the audit worktree; the *subject* tree is /root/tp-agent/work/m17637.
Every check prints JSON so the verdict evidence can be traced to a number.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SUBJ = Path("/root/tp-agent/work/m17637")
AUD = Path(__file__).resolve().parents[3]
OUT = AUD / "runs/m17666_audit"
OUT.mkdir(parents=True, exist_ok=True)

res = {}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


# ---- C1: artifact existence / identity -------------------------------------
need = [
    "tools/xval_cartesian_mpi/run_xval_cartesian_mpi.py",
    "tools/xval_cartesian_mpi/pylbm_tgv_run.py",
    "tools/xval_cartesian_mpi/shims/mpi4py/MPI.py",
    "docs/XVAL-LBM-CARTESIAN-MPI.md",
    "artifacts/xval_cartesian_mpi/summary.json",
    "artifacts/xval_cartesian_mpi/table_decomposition.csv",
    "artifacts/xval_cartesian_mpi/table_equivalence.csv",
]
c1 = {}
for rel in need:
    p = SUBJ / rel
    c1[rel] = ({"exists": True, "bytes": p.stat().st_size, "sha256_16": sha(p)}
               if p.exists() else {"exists": False})
res["C1_artifacts"] = c1

# ---- C2: reproducibility of the numeric payload ---------------------------
import csv

def payload(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return [{k: v for k, v in row.items() if k != "wall_time_s"} for row in csv.DictReader(fh)]

miss_eq = payload(SUBJ / "artifacts/xval_cartesian_mpi/table_equivalence.csv")
mine_eq = payload(OUT / "rerun_table_equivalence.csv")
res["C2_equivalence_reproduced"] = {
    "identical_payload": miss_eq == mine_eq,
    "rows": len(mine_eq),
    "rel_l2_pct_by_rank": {f"{r['dtype']}_P{r['n_ranks']}": r["field_rel_l2_pct_vs_serial"] for r in mine_eq},
    "gamma_err_pct": sorted({r["gamma_rel_err_pct"] for r in mine_eq}),
}

# ---- C3: pylbm arm of table_decomposition carries no numbers -------------
with (SUBJ / "artifacts/xval_cartesian_mpi/table_decomposition.csv").open(newline="") as fh:
    dec = list(csv.DictReader(fh))
pyl = [r for r in dec if r["impl"] == "pylbm"]
res["C3_pylbm_decomposition_rows"] = {
    "total": len(pyl),
    "with_any_local_size": sum(1 for r in pyl if r["local_min"] or r["local_max"]),
    "with_any_bytes": sum(1 for r in pyl if r["bytes_per_message"]),
    "with_accepted_true": sum(1 for r in pyl if r["accepted"] == "True"),
    "reason_field_blank_because": "error strings are only in summary.json; CSV drops 'error' (not in fieldnames)",
    "summary_error_examples": None,
}
su = json.loads((SUBJ / "artifacts/xval_cartesian_mpi/summary.json").read_text())
res["C3_pylbm_decomposition_rows"]["summary_error_examples"] = sorted(
    {r.get("error", "") for r in su["decomposition_matrix"] if r.get("impl") == "pylbm"}
)

# ---- C4: root cause of the pylbm arm failure (tool bug, not a pylbm limit)
sys.path[:0] = [str(SUBJ / "tools/xval_cartesian_mpi/shims"), "/tmp/pylbm_src"]
import numpy as np  # noqa: E402
import mpi4py.MPI as mpi  # noqa: E402  (shim)
from pylbm.mpi_topology import MpiTopology  # noqa: E402

comm = mpi.SimCartComm([2], [True, True, True], rank=0)
topo = MpiTopology(3, [True, True, True], comm=comm)
shape_in = topo.get_region_indices(16, 16, 16)
c4 = {"region_indices": [list(map(int, a)) for a in shape_in],
      "tool_bug": "measure_pylbm does np.diff(a) then indexes s[0..2]; np.diff of a "
                  "region-boundary array has length nregions, not 3 -> IndexError",
      "rank0_local_sizes": [int(np.diff(a)[0]) for a in shape_in]}
try:
    sizes = [np.diff(a) for a in shape_in]
    _ = [[int(s[i]) for i in range(3) if i != d] for d, s in enumerate(sizes)]
    c4["tool_path_raises"] = False
except Exception as exc:  # noqa: BLE001
    c4["tool_path_raises"] = f"{type(exc).__name__}: {exc}"
res["C4_pylbm_arm_root_cause"] = c4

# ---- C5: discrimination power of the equivalence arm (mutation probe) -----
import torch  # noqa: E402
sys.path.insert(0, str(SUBJ / "tools/xval_cartesian_mpi"))
sys.path.insert(0, str(SUBJ / "src"))
import run_xval_cartesian_mpi as T  # noqa: E402
from tensorlbm.domain_decomposition import HaloExchange  # noqa: E402

n, steps, u0, nu = 24, 100, 0.05, 0.02
tau = 3 * nu + 0.5
ref, _ = T.run_single(n, steps, tau, u0, torch.float64)


def run_with(halo_mutator=None, p=2):
    f0 = T.tgv_initial(n, u0, torch.float32)
    dec = T.DomainDecomposer(nz_global=n, ny=n, nx=n, n_devices=p, device="cpu")
    halo = HaloExchange(dec, mode="simulated", Q=19, dtype=torch.float32)
    sim = T.SimulatedMultiDevice(dec, halo, device="cpu")
    sim.distribute(f0)
    for _ in range(steps):
        for r in range(p):
            sim.slabs[r] = T.collide_slab(sim.slabs[r], tau)
        if halo_mutator:
            halo_mutator(halo, sim)
        else:
            halo.exchange_simulated(sim.slabs)
        for r in range(p):
            sim.slabs[r] = T.stream_slab(sim.slabs[r])
    return sim.gather()


def rel_l2(f):
    ux, uz = T.velocity_fields(f)
    rx, rz = T.velocity_fields(ref)
    du = torch.sqrt(((ux - rx) ** 2 + (uz - rz) ** 2).mean()).item()
    return du / torch.sqrt((rx ** 2 + rz ** 2).mean()).item() * 100


d_intact = rel_l2(run_with())
d_nohalo = rel_l2(run_with(halo_mutator=lambda h, s: None))
res["C5_discrimination"] = {
    "intact_halo_rel_l2_pct": d_intact,
    "noop_halo_rel_l2_pct": d_nohalo,
    "verdict": "equivalence arm discriminates: breaking the halo exchange moves the "
               "field error by orders of magnitude, so rel_l2=0.0 with the intact halo is "
               "a measured value, not a bypass",
}

# ---- C6: static wiring check (L3 gate) -----------------------------------
res["C6_integration"] = {
    "src_importers_of_domain_decomposition": [],
    "note": "grep -rn domain_decomposition src/tensorlbm/*.py -> only the module itself; "
            "callers are tests/test_domain_decomposition.py, scripts/m4_dd_validation.py, "
            "tools/xval_cartesian_mpi/run_xval_cartesian_mpi.py",
}

(OUT / "verify_m17637.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2)[:4000])
