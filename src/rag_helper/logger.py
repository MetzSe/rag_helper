"""
Logging utility for the rag_helper package.

Provides a standardized logging configuration with both console and file handlers,
and automatic silencing of verbose third-party loggers.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Constants for default configuration
LOG_FORMAT = "%(asctime)s [%(levelname)s] - %(name)s: %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"
LOG_FILE_FORMAT = "%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s"
SILENT_LOGGERS = {
    "sentence_transformers.SentenceTransformer": "WARNING",
    "httpx": "WARNING",
}


def setup_logging(
    log_level: str = "INFO",
    log_base_dir: Optional[str] = None,
    log_subdirs: bool = True,
    log_subdir_format: str = "%Y/%m/%d",
    force: bool = False,
) -> None:
    """
    Setup logging configuration for the application.

    Args:
        log_level (str): Logging level (DEBUG, INFO, etc.). Defaults to "INFO".
        log_base_dir (Optional[str]): Base directory for log files. Defaults to project root / logs.
        log_subdirs (bool): Whether to organize logs into subdirectories. Defaults to True.
        log_subdir_format (str): Strftime format for subdirectories. Defaults to "%Y/%m/%d".
        force (bool): If True, overwrite existing logging configuration. Defaults to False.
    """
    # Basic console logging
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))

    level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        handlers=[handler],
        force=force,
    )

    # Determine log directory
    if log_base_dir:
        base = Path(log_base_dir)
    else:
        # Default to /logs relative to the project root
        base = Path(__file__).resolve().parent.parent.parent / "logs"

    if log_subdirs:
        base = base / datetime.now().strftime(log_subdir_format)

    # Ensure the directory exists
    try:
        base.mkdir(parents=True, exist_ok=True)
        log_file = base / f"app_{datetime.now().strftime('%Y%m%d%H%M%S_%f')}.log"
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(LOG_FILE_FORMAT, datefmt=LOG_DATE_FORMAT)
        )

        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        logging.getLogger(__name__).info(f"Logging file saved at {log_file}")
    except Exception as e:
        print(f"Warning: Could not setup file logging: {e}", file=sys.stderr)

    # Configure silent loggers
    for logger_name, silence_level in SILENT_LOGGERS.items():
        logging.getLogger(logger_name).setLevel(getattr(logging, silence_level))


# Automatically setup logging with defaults when imported
setup_logging()
