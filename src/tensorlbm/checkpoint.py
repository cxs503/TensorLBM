"""Simulation checkpoint: save and load LBM distribution tensors.

Allows long simulations to be interrupted and resumed by persisting the
full distribution tensor **f** and the step counter to a compressed NumPy
archive (``.npz``).

Example
-------
.. code-block:: python

    from tensorlbm.checkpoint import save_checkpoint, load_checkpoint

    # Save every 1000 steps
    save_checkpoint(run_dir / "checkpoint.npz", f, step=1000)

    # Resume
    f, start_step = load_checkpoint(run_dir / "checkpoint.npz", device)
"""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: Path | str, f: torch.Tensor, step: int) -> None:
    """Save the distribution tensor *f* and *step* to a ``.npz`` file.

    Args:
        path: Destination file path (created or overwritten).
        f: Distribution tensor of any shape.
        step: Current simulation step index.
    """
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        f=f.detach().cpu().numpy(),
        step=np.array(step, dtype=np.int64),
    )


def load_checkpoint(
    path: Path | str,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, int]:
    """Load a checkpoint saved by :func:`save_checkpoint`.

    Args:
        path: Source ``.npz`` file path.
        device: Target device for the loaded tensor.

    Returns:
        Tuple ``(f, step)`` where *f* is a float32 tensor on *device* and
        *step* is the saved step index.
    """
    import numpy as np

    path = Path(path)
    data = np.load(str(path))
    f = torch.tensor(data["f"], dtype=torch.float32, device=device)
    step = int(data["step"])
    return f, step


__all__ = ["save_checkpoint", "load_checkpoint"]
