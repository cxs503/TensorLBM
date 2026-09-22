"""CV momentum-budget instrument — copied verbatim from W5-B diag/bfl_worker.py
(cv_box_force_exact), authored in the W5-B track. Pure measurement (torch
sums over exact link sets); no collide/stream/equilibrium/bounce/far_field
definitions. Disclosed in NOTES.md as instrument code.
"""

from __future__ import annotations

import torch


def cv_box_force_exact(f_pre_stream, solid, margin=6):
    """Exact control-volume momentum budget of the box around the body."""
    from tensorlbm.d3q19 import C as C19

    dev = f_pre_stream.device
    zs, ys, xs = torch.nonzero(solid, as_tuple=True)
    m = margin
    nz, ny, nx = solid.shape
    z0 = max(int(zs.min()) - m, 2)
    z1 = min(int(zs.max()) + m, nz - 3)
    y0 = max(int(ys.min()) - m, 2)
    y1 = min(int(ys.max()) + m, ny - 3)
    x0 = max(int(xs.min()) - m, 2)
    x1 = min(int(xs.max()) + m, nx - 3)
    inbox = torch.zeros_like(solid)
    inbox[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] = True
    c = C19.to(dev)
    T = torch.zeros(3, device=dev, dtype=torch.float64)
    for i in range(1, 19):
        di, dj, dk = int(c[i][0]), int(c[i][1]), int(c[i][2])
        nb_in = torch.roll(inbox, (-dk, -dj, -di), dims=(0, 1, 2))
        leaving = inbox & ~nb_in
        entering = (~inbox) & nb_in
        if not (leaving.any() or entering.any()):
            continue
        s_out = float(f_pre_stream[i][leaving].sum().item())
        s_in = float(f_pre_stream[i][entering].sum().item())
        T[0] += (s_in - s_out) * di
        T[1] += (s_in - s_out) * dj
        T[2] += (s_in - s_out) * dk
    return [float(T[0]), float(T[1]), float(T[2])], (x0, x1, y0, y1, z0, z1)
