"""Service helpers shared by CAD routers."""
from __future__ import annotations

import base64
import io
from typing import Any


def figure_to_png_data_url(fig: Any, *, dpi: int = 100) -> str:
    """Serialize a matplotlib figure into a PNG data URL."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()
