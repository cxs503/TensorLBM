"""Lattice and model constants."""

import torch


class D2Q9:
    """Standard D2Q9 discrete velocity set and weights."""

    c = torch.tensor(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
            [1, 1],
            [-1, 1],
            [-1, -1],
            [1, -1],
        ],
        dtype=torch.int64,
    )
    w = torch.tensor([4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4, dtype=torch.float32)
