"""Compatibility import for the historical round-1 study.

The original implementation is now shipped by tensorlbm.verification.
This path remains because the frozen study and external checkout users may
still import two_phase_only from this directory. Use PYTHONPATH=src.
"""

import sys

from tensorlbm.verification import two_phase_only as _canonical


def __getattr__(name):
    return getattr(_canonical, name)


def __dir__():
    return dir(_canonical)


__all__ = [name for name in dir(_canonical) if not name.startswith("_")]
sys.modules[__name__] = _canonical
