"""
utils/logger.py
===============
Configures the root logger for the Vision-Auth pipeline.
Call `setup_logging()` once at application entry point.
"""

import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """
    Configure the root logger with a console handler (and optionally a
    rotating file handler).

    Parameters
    ----------
    level    : Logging level string — DEBUG | INFO | WARNING | ERROR.
    log_file : Optional path to a .log file. Rotated at 5 MB, 3 backups.
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt = "%H:%M:%S"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]

    if log_file is not None:
        from logging.handlers import RotatingFileHandler
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                log_file,
                maxBytes=5 * 1024 * 1024,  # 5 MB
                backupCount=3,
                encoding="utf-8",
            )
        )

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=date_fmt,
        handlers=handlers,
    )
