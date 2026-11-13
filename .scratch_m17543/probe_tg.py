import math, sys, os, json
sys.path.insert(0, "/root/tp-agent/work/m17543/src")
sys.path.insert(0, "/root/tp-agent/work/m17543/benchmarks/case_taylor_green_3d")
import torch
torch.set_num_threads(4)
import run_case as rc
from tensorlbm.d3q27 import collide_bgk27, equilibrium27, macroscopic27, C, W_EXACT64
from tensorlbm.lbm_step_d3q27 import lbm_step_correct_27

cs2 = 1.0 / 3.0


def stress_init(n, amp, tau_l, Kmode):
    """f = feq + f^(1), f^(1) = K * w_i * Q_iab S_ab  (Chapman-Enskog consistent IC)."""
    import numpy as np
    u = rc.init_fields("tg", n, amp)
    rho = torch.ones_like(u[0])
    f = equilibrium27(rho, *u)
    un = [t.numpy().astype(np.float64) for t in u]
    S = [[0.5 * (rc._grad(un[a], n)[b] + rc._grad(un[b], n)[a]) for b in range(3)] for a in range(3)]
    S = torch.from_numpy(np.stack([[np.asarray(S[a][b]) for b in range(3)] for a in range(3)]))
    e = C.to(torch.float64)
    Q = torch.einsum('ia,ib->iab', e, e) - cs2 * torch.eye(3, dtype=torch.float64)
    Sab = torch.einsum('iab,ab...->i...', Q, S)
    K = -rho * (tau_l - (0.5 if Kmode == "half" else 0.0)) / cs2
    f = f + (K * W_EXACT64[:, None, None, None] * Sab)
    return f


def run(n, tau, amp, steps, sample, mode):
    nu = (tau - 0.5) / 3.0
    k2 = rc.k2_of("tg", n)
    u0 = rc.init_fields("tg", n, amp)
    f = equilibrium27(torch.ones_like(u0[0]), *u0) if mode == "eq" else stress_init(n, amp, tau, mode)
    rho0, ux0, uy0, uz0 = macroscopic27(f)
    print(f"  [{mode}] IC velocity rel err = {rc.l2_rel((ux0,uy0,uz0), u0):.3e}")
    solid = torch.zeros(u0[0].shape, dtype=torch.bool)
    hist = []
    with torch.no_grad():
        for step in range(1, steps + 1):
            f = lbm_step_correct_27(f, collide_bgk27, tau, solid, 0.0, far_field_bc_fn=None)
            if step % sample == 0:
                rho, ux, uy, uz = macroscopic27(f)
                e = rc.kinetic_energy(ux, uy, uz)
                ex = rc.exact_fields("tg", n, amp, nu, step)
                ee = rc.kinetic_energy(*ex)
                hist.append({"t": step, "E_rel_err_pct": 100 * (e - ee) / ee, "l2": rc.l2_rel((ux, uy, uz), ex)})
    return hist, nu, k2


if __name__ == "__main__":
    n, tau, amp, steps, sample = 24, 0.6, 2e-3, 200, 5
    for mode in ("eq", "half", "full"):
        hist, nu, k2 = run(n, tau, amp, steps, sample, mode)
        lam = 2 * nu * k2
        ts = [h["t"] for h in hist]
        y = [math.log(1.0 + h["E_rel_err_pct"] / 100.0) for h in hist]
        i0 = len(ts) // 2
        tm = sum(ts[i0:]) / len(ts[i0:]); ym = sum(y[i0:]) / len(y[i0:])
        slope = sum((t - tm) * (v - ym) for t, v in zip(ts[i0:], y[i0:])) / sum((t - tm) ** 2 for t in ts[i0:])
        print(f"mode={mode} delta_lambda_asym={-slope/lam*100:+.4f}%  first: {[(h['t'], round(h['E_rel_err_pct'],4)) for h in hist[:6]]}")
        print(f"      last: {[(h['t'], round(h['E_rel_err_pct'],4)) for h in hist[-3:]]}  l2={hist[-1]['l2']:.4e}")