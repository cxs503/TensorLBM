"""Compatibility import for the historical D3Q27 collision module."""

import sys

from tensorlbm.verification.d3q27_spherical.collision import d3q27

sys.modules[__name__] = d3q27
