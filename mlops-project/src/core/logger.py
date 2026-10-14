"""
Logging utility and a progress bar for batch loops.
"""

import logging

from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


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


def progress_bar() -> Progress:
    """
    A batch progress bar that disappears when the loop ends.

    Tasks need a `status` field (e.g. the running loss), shown after the bar.
    """

    return Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[status]}"),
        TimeElapsedColumn(),
        TextColumn("<"),
        TimeRemainingColumn(),
        transient=True,
    )
