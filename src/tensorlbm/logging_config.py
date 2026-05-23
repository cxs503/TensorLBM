"""Logging configuration helpers for TensorLBM."""
from __future__ import annotations

import logging

logger = logging.getLogger("tensorlbm")


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root tensorlbm logger with a sensible console handler."""
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)


__all__ = ["logger", "configure_logging"]
