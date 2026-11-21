import sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_per_step_hook_mask_blend")); sys.path.insert(0, str(ROOT / "src"))
import run_xval as R
import tensorlbm.step_hooks as SH
from tensorlbm.solver import collide_bgk, stream
torch.set_num_threads(4); DT = torch.float64
U = 0.02

def build(ny, nx, tau, eps, variant):
    y_b = 0.5 + eps; y_t = ny - 1.5; H = y_t - y_b
    steps = R.steps_for_couette(H, (tau - .5) / 3)
    solid = torch.zeros(ny, nx, dtype=DT); solid[0, :] = 1.
    part = torch.zeros(ny, nx, dtype=DT); part[1, :] = float(eps)
    top = torch.zeros(ny, nx, dtype=DT); top[-1, :] = 1.
    f0 = R.rest_field(ny, nx); tb = SH.tables_for(f0)
    B = SH.partial_cell_weight(part, tau)
    opp = tb.opposite
    w = tb.weights; wsh = [9, 1, 1]
    cu = SH._velocity_kernel((U, 0.0), tb, f0).reshape(wsh)
    mom = 6.0 * w.view(*wsh) * cu
    # fixed layers: solid row0 (bounce), top (bounce)
    fixed = [SH.stationary_bounce_layer("s0", solid), SH.stationary_bounce_layer("t", top)]

    def op(state, t):
        f = state["f"]
        fstr = stream(f)
        fpost = collide_bgk(fstr, tau)
        fpost = SH.compose_masked_boundaries(fpost, fstr, fixed, tables=tb, state=state)
        # partial row 1, hand-written variants (row slice only)
        rp_pre = fstr[opp][:, 1:2, :]
        rp_post = fpost[opp][:, 1:2, :]
        fp = fpost[:, 1:2, :]
        b = B[1:2].unsqueeze(0)  # (1,1,nx)
        if variant == "post_B_momB":      out = (1 - b) * fp + b * (rp_post + mom)
        elif variant == "post_B_mom1":    out = (1 - b) * fp + b * rp_post + mom
        elif variant == "pre_B_mom1":     out = (1 - b) * fp + b * rp_pre + mom
        elif variant == "pre_B_momB":     out = (1 - b) * fp + b * (rp_pre + mom)
        elif variant == "post_B_mom_eps": out = (1 - b) * fp + b * rp_post + (b / (B[1:2].unsqueeze(0) + 1e-30)) * 0 + mom * (part[1:2].unsqueeze(0))
        fpost = fpost.clone(); fpost[:, 1:2, :] = out
        state["f"] = fpost; state["t"] = t
        return state
    return steps, f0, op, y_b, y_t, H

def run(ny, tau, eps, variant):
    steps, f0, op, y_b, y_t, H = build(ny, 8, tau, eps, variant)
    with torch.no_grad():
        st, _ = SH.solve_with_hooks({"f": f0.clone()}, steps, operator=op)
    rho, ux, _ = R.macroscopic(st["f"]); u = ux.numpy()[:, 0]
    ys = np.arange(2, ny - 1, dtype=float); uu = u[2:ny - 1]
    ref = U * (y_t - ys) / H
    A = np.polyfit(ys, uu, 1); u_at = A[1] + A[0] * y_b
    return 100 * np.linalg.norm(uu - ref) / np.linalg.norm(ref), u_at / U

for variant in ("post_B_momB", "post_B_mom1", "pre_B_mom1", "pre_B_momB", "post_B_mom_eps"):
    row = []
    for eps in (0.25, 0.5, 0.75, 1.0):
        l2, uw = run(24, 0.8, eps, variant)
        row.append(f"eps{eps:.2f}: L2={l2:6.2f}% u@y_b/U={uw:+.3f}")
    print(f"{variant:16s} " + " | ".join(row))