"""Probe: sub-cell wall position recovered by naive / NT(pre) / NT(post) masks."""
import sys, math
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tensorlbm import step_hooks as SH
from tensorlbm.d2q9 import equilibrium, macroscopic

DT = torch.float64
torch.set_num_threads(4)


def rest(ny, nx):
    return equilibrium(torch.ones(ny, nx, dtype=DT), torch.zeros(ny, nx, dtype=DT), torch.zeros(ny, nx, dtype=DT))


def masks(ny, nx, eps):
    solid = torch.zeros(ny, nx, dtype=DT); solid[0, :] = 1.0
    part = torch.zeros(ny, nx, dtype=DT); part[1, :] = float(eps)
    lid = torch.zeros(ny, nx, dtype=DT); lid[-1, :] = 1.0
    return solid, part, lid


def run(ny=16, nx=8, tau=0.8, eps=0.5, mode="nt_post", u=0.02, steps=None):
    nu = (tau - 0.5) / 3.0
    y_b = 0.5 + eps
    y_t = ny - 1.5
    H = y_t - y_b
    steps = steps or int(12.0 * H * H / (math.pi**2 * nu)) + 200
    solid, part, lid = masks(ny, nx, eps)
    if mode == "naive_eps":
        players = [SH.stationary_bounce_layer("solid0", solid),
                   SH.BoundaryLayer(name="part1", mask=part, kind="bounce")]
    else:
        players = [SH.stationary_bounce_layer("solid0", solid),
                   SH.partial_cell_layer("part1", part, tau, reflect=mode.split("_")[1])]
    players.append(SH.moving_bounce_layer("lid", lid, (u, 0.0)))
    op = SH.masked_step(tau=tau, layers=players)
    with torch.no_grad():
        st, _ = SH.solve_with_hooks({"f": rest(ny, nx)}, steps, operator=op)
        _, ux, _ = macroscopic(st["f"])
    rows = np.arange(2, ny - 1, dtype=float)
    got = ux.numpy()[2:-1, nx // 2]
    exact = u * (rows - y_b) / H
    slope, icept = np.polyfit(rows, got, 1)
    y0 = -icept / slope
    return dict(mode=mode, eps=eps, y_b=y_b, d_eff=y0, off=y0 - y_b,
                l2=100 * np.linalg.norm(got - exact) / np.linalg.norm(exact))


for eps in (0.25, 0.5, 0.75, 1.0):
    for mode in ("naive_eps", "nt_pre", "nt_post"):
        r = run(eps=eps, mode=mode)
        print(f"eps={eps:4.2f} {mode:10s} y_b={r['y_b']:.3f} d_eff={r['d_eff']:+.4f} off={r['off']:+.4f} L2={r['l2']:.4f}%")