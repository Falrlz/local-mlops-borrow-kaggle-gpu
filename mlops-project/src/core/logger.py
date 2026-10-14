"""
Logging utility.
"""

import logging

from rich.logging import RichHandler


def get_logger(name: str = "mlops", level: int = logging.INFO) -> logging.Logger:
    """Get or configure a logger with Rich console handler."""

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_level=True,
            show_path=False,
            markup=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False

    return logger
