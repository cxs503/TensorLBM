"""Track the first resolved liquid bridge in a two-dimensional tank."""

import numpy as np


def _component(mask, seed):
    """Four-neighbour flood fill using horizontal runs, without extra dependencies."""
    out = np.zeros_like(mask)
    stack = [seed]
    ny, nx = mask.shape
    while stack:
        y, x = stack.pop()
        if out[y, x] or not mask[y, x]:
            continue
        left = right = x
        while left and mask[y, left - 1]:
            left -= 1
        while right + 1 < nx and mask[y, right + 1]:
            right += 1
        out[y, left : right + 1] = True
        for row in (y - 1, y + 1):
            if 0 <= row < ny:
                valid = mask[row, left : right + 1] & ~out[row, left : right + 1]
                starts = np.flatnonzero(np.diff(valid.astype(np.int8), prepend=0) == 1)
                stack.extend((row, left + int(k)) for k in starts)
    return out


class ImpactTracker:
    """Follow the initially detached liquid by overlap. y increases downwards.

    Touching requires a four-connected path of positive phase to the deep-pool
    seed. More than half the previous drop footprint must still be identifiable.
    Losing that identity is terminal. Contact times are sampling brackets.
    """

    def __init__(self, solid, drop_seed, pool_seed):
        self.solid = np.asarray(solid, dtype=bool).copy()
        self.drop_seed = drop_seed
        self.pool_seed = pool_seed
        self.previous = None
        self.last_step = None
        self.contact = None
        self.terminal_state = None

    def observe(self, phi, step):
        field = np.asarray(phi)
        if self.last_step is not None and step <= self.last_step:
            raise ValueError("observation steps must increase")
        if field.shape != self.solid.shape or not np.isfinite(field[~self.solid]).all():
            raise ValueError("expected a finite fluid phase field matching the tank")
        liquid = (field > 0) & ~self.solid
        pool = _component(liquid, self.pool_seed)
        metrics = dict(
            step=step,
            state=self.terminal_state or "tracking_lost",
            drop_centroid_y=None,
            centerline_gap=None,
            pool_centerline_y=None,
            drop_area_cells=None,
        )
        # The top of the pool-connected component remains defined after contact.
        # It then includes the joined drop and is not the bottom of a crater.
        x = self.drop_seed[1]
        top = np.flatnonzero(pool[:, x])
        if top.size and top[0] > 0:
            p = top[0]
            col = field[:, x]
            if not self.solid[p - 1, x] and col[p - 1] <= 0:
                metrics["pool_centerline_y"] = float(p - col[p] / (col[p] - col[p - 1]))
        # After a first bridge the original drop has lost its separate identity.
        if self.contact is not None:
            metrics["state"] = "after_first_bridge"
        elif self.terminal_state is None and pool.any():
            if self.previous is None:
                drop = _component(liquid, self.drop_seed)
                if drop[self.pool_seed]:
                    metrics["state"] = "initially_connected"
                    self.terminal_state = metrics["state"]
            else:
                overlap = self.previous & liquid
                if np.count_nonzero(overlap & pool) > self.previous.sum() / 2:
                    self.contact = [self.last_step, step]
                    metrics["state"] = "first_bridge"
                    drop = None
                else:
                    # A transient detached satellite cannot stand in for the original drop.
                    ys, xs = np.nonzero(overlap & ~pool)
                    if ys.size:
                        mid = np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2)
                        drop = _component(liquid, (ys[mid], xs[mid]))
                        if np.count_nonzero(drop & self.previous) <= self.previous.sum() / 2:
                            drop = None
                    else:
                        drop = None
            if drop is not None and drop.any() and self.terminal_state is None:
                self.previous = drop
                ys, _ = np.nonzero(drop)
                metrics.update(
                    state="separated",
                    drop_centroid_y=float(ys.mean()),
                    drop_area_cells=int(ys.size),
                )
                bottom = np.flatnonzero(drop[:, x])
                if bottom.size and metrics["pool_centerline_y"] is not None and bottom[-1] < p:
                    d = bottom[-1]
                    if not self.solid[d + 1, x] and col[d + 1] <= 0:
                        yd = d + col[d] / (col[d] - col[d + 1])
                        metrics["centerline_gap"] = float(metrics["pool_centerline_y"] - yd)
        if metrics["state"] == "tracking_lost":
            self.terminal_state = metrics["state"]
        self.last_step = step
        return metrics
