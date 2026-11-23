"""Compatibility import for the historical D3Q27 spherical-force module."""

import sys

from tensorlbm.verification.d3q27_spherical.boundary import spherical_force

sys.modules[__name__] = spherical_force
