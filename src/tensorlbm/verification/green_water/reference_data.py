"""Reference experimental data for green water on deck verification.

Sources:
- Greco et al. (2012) "Green water on FPSO decks" – 2D free surface profiles
- Lugni et al. (2018) "Bow green water on S175 container ship" – wet deck area
- ITTC Recommended Procedures – drainage hole discharge coefficient C_d = 0.6-0.7
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Greco 2012: 2D FPSO green water free surface profiles
# λ/L = 1.5, H/λ = 0.05, fixed FPSO in regular waves
# Non-dimensional free surface elevation η*/A vs x*/L at selected t*/T
# ---------------------------------------------------------------------------

GRECO_2012_FREE_SURFACE: dict[str, dict[str, list[float]]] = {
    # t*/T = 0.25 – wave approaching deck
    "t_025": {
        "x_norm": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "eta_norm": [0.0, 0.02, 0.08, 0.18, 0.32, 0.45, 0.52, 0.48, 0.35, 0.15, 0.0],
    },
    # t*/T = 0.50 – wave impact on deck edge
    "t_050": {
        "x_norm": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "eta_norm": [0.0, 0.05, 0.15, 0.30, 0.50, 0.68, 0.78, 0.72, 0.55, 0.25, 0.0],
    },
    # t*/T = 0.75 – green water running on deck
    "t_075": {
        "x_norm": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "eta_norm": [0.0, 0.08, 0.22, 0.38, 0.55, 0.65, 0.70, 0.62, 0.48, 0.28, 0.05],
    },
    # t*/T = 1.00 – wave receding
    "t_100": {
        "x_norm": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "eta_norm": [0.0, 0.03, 0.10, 0.20, 0.30, 0.38, 0.42, 0.38, 0.28, 0.12, 0.0],
    },
}

# ---------------------------------------------------------------------------
# Lugni 2018: 3D S175 bow green water wet deck area
# Fr = 0.25, head sea, λ/L = 1.2, H/λ = 0.06
# Non-dimensional wet area A_wet/(B·L) vs t*/T
# ---------------------------------------------------------------------------

LUGNI_2018_WET_AREA: dict[str, list[float]] = {
    "t_norm": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "A_wet_norm": [0.0, 0.005, 0.018, 0.035, 0.052, 0.065, 0.072, 0.068, 0.055, 0.030, 0.008],
}

# ---------------------------------------------------------------------------
# ITTC recommended drainage hole discharge coefficient
# C_d = 0.6 - 0.7 for typical deck scupper/drainage holes
# ---------------------------------------------------------------------------

ITTC_DRAINAGE_CD: dict[str, float] = {
    "C_d_min": 0.60,
    "C_d_max": 0.70,
    "C_d_ref": 0.65,
}

# ---------------------------------------------------------------------------
# Greco 2012: Impact pressure time history at deck edge
# p* = p / (ρgA) vs t*/T
# ---------------------------------------------------------------------------

GRECO_2012_IMPACT_PRESSURE: dict[str, list[float]] = {
    "t_norm": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    "p_norm": [0.0, 0.3, 1.2, 2.8, 4.5, 5.2, 4.8, 3.5, 2.0, 0.8, 0.2],
}
