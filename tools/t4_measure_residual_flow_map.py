#!/usr/bin/env python3
"""Measure the residual-over-prior flow map on the T4 medium corpus (m18630).

Reproduces the 120-epoch one-step numbers quoted in
``docs/T4_FLOW_MAP_RESIDUAL_m18630.md`` against an existing
``artifacts/t4_physics_ai_loop_medium/dataset`` (build it with
``python bench_t4_medium.py --stage produce``).  Kept separate from the bench so
the parameterisation comparison (residual vs direct at equal budget) is
re-runnable without regenerating the LBM corpus.

Usage: python tools/t4_measure_residual_flow_map.py [--epochs 120] [--direct]
"""
import sys, time, json
sys.path.insert(0, "/root/tp-agent/work/m18630/src")
import torch

torch.set_num_threads(8)
from tensorlbm.ai.flow_field_loop import load_dataset, viscous_prior_step, _rel_l2
from tensorlbm.ai.fno import FNO2d, FNO2dArch

torch.manual_seed(0)
DS = "/root/tp-agent/work/m18630/artifacts/t4_physics_ai_loop_medium/dataset"
data = load_dataset(DS)
runs = data["runs"]
NU_REF = 1e-2
DT = 200
HOLD = {75.0}


def pairs(run):
    ux, uy = run["ux"], run["uy"]
    u0 = run["u0"]
    xs, ys, bases = [], [], []
    for i in range(ux.shape[0] - 1):
        x = torch.stack([ux[i] / u0, uy[i] / u0, torch.full_like(ux[i], run["nu"] / NU_REF)])
        y = torch.stack([ux[i + 1] / u0, uy[i + 1] / u0])
        b = torch.stack(viscous_prior_step(ux[i], uy[i], run["nu"], DT)) / u0
        xs.append(x); ys.append(y); bases.append(b)
    return torch.stack(xs), torch.stack(ys), torch.stack(bases)


tr_x, tr_y, tr_b = [], [], []
ho = {}
for rid, run in sorted(runs.items()):
    x, y, b = pairs(run)
    if round(run["re"], 6) in HOLD:
        ho[rid] = (x, y, b)
    else:
        tr_x.append(x); tr_y.append(y); tr_b.append(b)
X = torch.cat(tr_x); Y = torch.cat(tr_y); B = torch.cat(tr_b)
print("train pairs", tuple(X.shape), "holdout runs", list(ho), flush=True)


def errs(model, residual):
    out = {}
    for rid, (x, y, b) in ho.items():
        with torch.no_grad():
            pred = model(x)
        if residual:
            pred = pred + b
        n = x.shape[0]
        out[rid] = (round(sum(_rel_l2(pred[i], y[i]) for i in range(n)) / n, 4),
                    round(sum(_rel_l2(b[i], y[i]) for i in range(n)) / n, 4),
                    round(sum(_rel_l2(x[i][:2], y[i]) for i in range(n)) / n, 4))
    return out


def train(residual, epochs, width, layers, modes, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    T = (Y - B) if residual else Y
    arch = FNO2dArch(in_channels=3, out_channels=2, width=width, n_layers=layers,
                     modes_x=modes, modes_y=modes, mlp_hidden=128)
    model = FNO2d(arch)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = X.shape[0]
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 16):
            idx = perm[i:i + 16]
            loss = torch.mean((model(X[idx]) - T[idx]) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * idx.numel()
        sched.step()
        if ep == 1 or ep % max(1, epochs // 6) == 0:
            print(f"  ep{ep:4d} mse {tot/n:.4e} ({time.perf_counter()-t0:.0f}s)", flush=True)
    return model


print(f"reference mse: zero-output {float((Y**2).mean()):.4f} "
      f"identity {float(((Y - X[:, :2])**2).mean()):.4f} prior {float(((Y - B)**2).mean()):.4f}", flush=True)

res = {}
m = train(True, 120, 16, 2, 8)
res["C_residual_120ep"] = errs(m, True)
print("C", res["C_residual_120ep"], flush=True)
m = train(False, 120, 16, 2, 8)
res["B_direct_120ep"] = errs(m, False)
print("B", res["B_direct_120ep"], flush=True)
print(json.dumps(res, indent=1))