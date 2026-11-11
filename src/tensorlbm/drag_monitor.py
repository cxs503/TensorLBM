"""Running-average drag convergence monitor — re-exported from yplus_guide.

Kept as a stable import path: ``from tensorlbm.drag_monitor import DragMonitor``
resolves to :class:`tensorlbm.yplus_guide.DragMonitor` (the y+-aware
implementation).
"""

from .yplus_guide import DragMonitor

__all__ = ["DragMonitor"]
