"""D2Q9 lattice constants for 2D LBM."""

import torch

# D2Q9 lattice velocities (9 directions)
C2Q9 = torch.tensor([
    [ 0,  0],  # 0: rest
    [ 1,  0],  # 1: east
    [-1,  0],  # 2: west
    [ 0,  1],  # 3: north
    [ 0, -1],  # 4: south
    [ 1,  1],  # 5: NE
    [-1,  1],  # 6: NW
    [ 1, -1],  # 7: SE
    [-1, -1],  # 8: SW
], dtype=torch.int64)

# D2Q9 weights
W2Q9 = torch.tensor([
    4.0 / 9.0,   # 0
    1.0 / 9.0,   # 1-4
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,  # 5-8
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
], dtype=torch.float64)

# Opposite direction indices for D2Q9
OPPOSITE_2Q9 = torch.tensor([
    0,  # 0 -> 0
    2,  # 1 -> 2
    1,  # 2 -> 1
    4,  # 3 -> 4
    3,  # 4 -> 3
    8,  # 5 -> 8
    7,  # 6 -> 7
    6,  # 7 -> 6
    5,  # 8 -> 5
], dtype=torch.int64)

# Speed of sound squared for D2Q9
CS2_2Q9 = 1.0 / 3.0
