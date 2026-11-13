"""D2Q9 direction sets of the vendored OSS, copied verbatim from upstream.

Source: ``haimingz/vivsim`` ``vivsim/lbm/lattice.py`` (MIT) at commit
``fb84b12f02e18f3b9dfdf738d6a1a318d81539f9``::

    right_dirs=[1, 5, 8]  left_dirs=[3, 7, 6]
    up_dirs=[2, 5, 6]     down_dirs=[4, 7, 8]     opp_dirs=[0, 3, 4, 1, 2, 7, 8, 5, 6]

The numbering is identical to ``tensorlbm.d2q9`` (0 rest, 1 E, 2 N, 3 W, 4 S,
5 NE, 6 NW, 7 SW, 8 SE), so the transferred population subsets can be compared
one-to-one between the two codes.

``multigrid.py`` imports only these four tuples; they are the entire surface of
``vivsim/lbm/basic.py`` that the vendored module needs.  Values are plain
``numpy.int32`` arrays, which is exactly what upstream builds before wrapping
them with ``jnp.asarray``.
"""

from __future__ import annotations

import numpy as np

LEFT_DIRS = np.array([3, 7, 6], dtype=np.int32)
RIGHT_DIRS = np.array([1, 5, 8], dtype=np.int32)
UP_DIRS = np.array([2, 5, 6], dtype=np.int32)
DOWN_DIRS = np.array([4, 7, 8], dtype=np.int32)
OPP_DIRS = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)