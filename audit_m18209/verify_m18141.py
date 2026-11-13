"""Independent re-derivation for board audit of mission m18141
(anchor XVAL-LBM-IBM-EPSILON-GRAM-IMPLICIT-CORRECTION, repo TensorLBM).

Purpose: re-compute the headline KPIs from the audited module WITHOUT calling
the mission's own harness (xval/.../run_xval.py), so that the numbers in
benchmark.json are checked against a second, independently written path.

Run:
    cd /root/tp-agent/work/m18141
    PYTHONPATH=src python /root/tp-agent/work/m18209/audit_m18209/verify_m18141.py
"""
import json
import math
import sys

import torch

from tensorlbm.ibm_epsilon_gram import (
    build_support_2d,
    compute_ds_2d,
    compute_epsilon_gram,
    life_delta_3pt,
)
from tensorlbm.ibm_force_transfer import IBMStencil

DT = torch.float64
XT = "/root/tp-agent/work/m18141/xval/XVAL-LBM-IBM-EPSILON-GRAM-IMPLICIT-CORRECTION"


# ---------------------------------------------------------------- independent
def my_support(x, y, ny, nx):
    """Independent re-implementation of the LIFE 5x5 support (|d|<1.5, no wrap)."""
    out = []
    for k in range(len(x)):
        ix0, iy0 = int(round(float(x[k]))), int(round(float(y[k])))
        idx, w = [], []
        for jy in range(iy0 - 2, iy0 + 3):
            for jx in range(ix0 - 2, ix0 + 3):
                dx, dy = x[k] - jx, y[k] - jy
                if abs(float(dx)) < 1.5 and abs(float(dy)) < 1.5 and 0 <= jx < nx and 0 <= jy < ny:
                    idx.append(jy * nx + jx)
                    w.append(float(life_delta_3pt(torch.tensor(dx)) * life_delta_3pt(torch.tensor(dy))))
        out.append((idx, w))
    return out


def my_roundtrip(px, py, fx, fy, eps, ds, ny, nx, corrected):
    sup = my_support(px, py, ny, nx)
    # spread: F_node += f_m * eps_m * ds_m * w_m(s)
    Fx = torch.zeros(ny * nx, dtype=DT)
    Fy = torch.zeros(ny * nx, dtype=DT)
    for k, (idx, w) in enumerate(sup):
        # A1: w = eps*ds*delta (under test) ; A2 control keeps ds, eps=1
        scale = float(eps[k] * ds[k]) if corrected else float(ds[k])
        for i, wv in zip(idx, w):
            Fx[i] += float(fx[k]) * scale * wv
            Fy[i] += float(fy[k]) * scale * wv
    # interpolate (LIFE: plain delta, no epsilon)
    gx = torch.zeros(len(px), dtype=DT)
    gy = torch.zeros(len(px), dtype=DT)
    for k, (idx, w) in enumerate(sup):
        gx[k] = sum(float(Fx[i]) * wv for i, wv in zip(idx, w))
        gy[k] = sum(float(Fy[i]) * wv for i, wv in zip(idx, w))
    return gx, gy, Fx, Fy


def circle(n, r=10.0, cx=32.0, cy=32.0, phase=0.0):
    """same ring layout as run_xval.py: 64x64 grid, R=10, centre (32,32)"""
    th = torch.tensor([2 * math.pi * k / n + phase for k in range(n)], dtype=DT)
    return cx + r * torch.cos(th), cy + r * torch.sin(th)


def rel_l2(got, ref):
    return float(torch.linalg.norm(got - ref) / torch.linalg.norm(ref))


rows = {}

# --- 1. 1-D partition of unity (analytic), 7 offsets incl. irrational
errs = []
for off in [0.0, 0.125, 0.25, 0.375, 0.5, 0.7071, 1.0]:
    s = sum(float(life_delta_3pt(torch.tensor(abs(off - i) if off - i <= 0 else torch.tensor(abs(off - i)))))
            for i in range(-2, 3))
    errs.append(abs(s - 1.0))
rows["dirac_partition_unity_1d_maxerr"] = max(errs)

for n in (16, 32):
    px, py = circle(n)
    ds = compute_ds_2d(px, py)
    sup_mod = build_support_2d(px, py, 64, 64)
    eps = compute_epsilon_gram(px, py, sup_mod, ds)
    # constant field (1, 0)
    fx, fy = torch.ones(n, dtype=DT), torch.zeros(n, dtype=DT)
    for arm, corr in (("A1", True), ("A2", False)):
        gx, gy, Fx, Fy = my_roundtrip(px, py, fx, fy, eps, ds, 64, 64, corr)
        rows[f"indep_roundtrip_{arm}_N{n}_const_l2"] = rel_l2(gx, fx)
        dep = abs(float(Fx.sum()) - float((fx * ds).sum())) / abs(float((fx * ds).sum()))
        rows[f"indep_deposit_dev_{arm}_N{n}_const"] = dep
        # support weight sum (analytic 1)
        rows[f"indep_dirac_sum_N{n}_maxdev"] = max(
            abs(sum(w) - 1.0) for _, w in my_support(px, py, 64, 64))
    rows[f"epsilon_N{n}_min"] = float(eps.min())
    rows[f"epsilon_N{n}_max"] = float(eps.max())
    # radial field
    th = torch.tensor([2 * math.pi * k / n for k in range(n)], dtype=DT)
    fxr, fyr = torch.cos(th), torch.sin(th)
    gx, gy, _, _ = my_roundtrip(px, py, fxr, fyr, eps, ds, 64, 64, True)
    rows[f"indep_roundtrip_A1_N{n}_radial_l2"] = rel_l2(torch.cat([gx, gy]), torch.cat([fxr, fyr]))
    gx, gy, _, _ = my_roundtrip(px, py, fxr, fyr, eps, ds, 64, 64, False)
    rows[f"indep_roundtrip_A2_N{n}_radial_l2"] = rel_l2(torch.cat([gx, gy]), torch.cat([fxr, fyr]))

# --- 2. structural check: is the A1 identity by construction?
n = 16
px, py = circle(n)
ds = compute_ds_2d(px, py)
sup = build_support_2d(px, py, 64, 64)
eps = compute_epsilon_gram(px, py, sup, ds)
# Gram matrix as LIFE defines it: A[i,j] = sum_s w_i(s) w_j(s) ds_j  ->  A = W diag(ds) W^T
nodes = 64 * 64
W = torch.zeros(n, nodes, dtype=DT)
for k, sk in enumerate(sup):
    for iy, ix, wv in zip(sk["iy"].tolist(), sk["ix"].tolist(), sk["w"].tolist()):
        W[k, iy * 64 + ix] += wv
A = (W @ W.t()) * ds.unsqueeze(0)   # column j scaled by ds_j  =>  A[i,j] = ds_j sum_s w_i w_j
rows["gram_residual_Aeps_minus_1"] = float((A @ eps - torch.ones(n, dtype=DT)).abs().max())

# non-constant field: the correction reproduces f only if f is constant over markers
th = torch.tensor([2 * math.pi * k / n for k in range(n)], dtype=DT)
f_r = torch.cos(th)
rows["Aeps_on_nonconst_field_err"] = float((A @ (eps * f_r) - f_r).abs().max())

# --- 3. cross-check vs the mission's own benchmark.json
bench = json.load(open(f"{XT}/benchmark.json"))
got = {r["kpi"]: r["got"] for r in bench["rows"]}
pairs = [
    ("gram_roundtrip_A1_N16_l2", rows["indep_roundtrip_A1_N16_const_l2"]),
    ("gram_roundtrip_A2_N16_l2", rows["indep_roundtrip_A2_N16_const_l2"]),
    ("gram_roundtrip_A1_N32_l2", rows["indep_roundtrip_A1_N32_const_l2"]),
    ("gram_roundtrip_A2_N32_l2", rows["indep_roundtrip_A2_N32_const_l2"]),
    ("deposit_dev_A1_N32_const", rows["indep_deposit_dev_A1_N32_const"]),
    ("dirac_partition_unity_1d_maxerr", rows["dirac_partition_unity_1d_maxerr"]),
]
print("\n=== benchmark.json rows vs independent recompute ===")
agree = 0
checked = 0
for k, mine in pairs:
    if k not in got:
        print(f"  {k:38s} ABSENT from benchmark.json (recomputed={mine:.6e})")
        continue
    ref = got[k]
    ok = (abs(mine - ref) <= 1e-9 + 1e-6 * abs(ref))
    checked += 1
    agree += int(ok)
    print(f"  {k:38s} bench={ref:.6e} indep={mine:.6e} agree={ok}")

print("\n=== independently recomputed KPIs ===")
for k, v in rows.items():
    print(f"  {k:40s} {v:.6e}")

summary = {
    "rows": rows,
    "bench_crosscheck": {"checked": checked, "agree": agree},
    "bench_meta": {"n_rows": bench["n_rows"], "n_pass": bench["n_pass"],
                   "git_sha": bench["git_sha"]},
    "bench_fail_rows": [r["kpi"] for r in bench["rows"] if not r["pass"]],
}
print("\n=== benchmark rows failing ===")
for r in bench["rows"]:
    if not r["pass"]:
        print(f"  {r['kpi']:40s} ref={r['ref']:.6e} got={r['got']:.6e} tol={r['tol']} rel_err={r['rel_err']:.4e}")
json.dump(summary, open("/root/tp-agent/work/m18209/audit_m18209/verify_m18141.json", "w"), indent=2)
print("\nwrote verify_m18141.json")