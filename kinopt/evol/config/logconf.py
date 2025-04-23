import logging
import os
import re
from datetime import datetime
from logging.handlers import RotatingFileHandler

from kinopt.evol.config.constants import LOG_DIR
from kinopt.evol.utils.iodata import format_duration

# Color mapping for console output
LOG_COLORS = {
    "DEBUG": "\033[92m",    # Green
    "INFO": "\033[94m",     # Blue
    "WARNING": "\033[93m",  # Yellow
    "ERROR": "\033[91m",    # Red
    "CRITICAL": "\033[95m", # Magenta
    "ELAPSED": "\033[96m",  # Cyan (right-aligned clock)
    "ENDC": "\033[0m",      # Reset
}

class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, width=200):
        super().__init__(fmt, datefmt)
        self.start_time = datetime.now()
        self.width = width

    def format(self, record):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        elapsed_str = f"{LOG_COLORS['ELAPSED']}⏱ {format_duration(elapsed)}{LOG_COLORS['ENDC']}"

        # Compose colored parts
        color = LOG_COLORS.get(record.levelname, LOG_COLORS["INFO"])
        time_str = f"{LOG_COLORS['DEBUG']}{self.formatTime(record)}{LOG_COLORS['ENDC']}"
        name_str = f"{LOG_COLORS['WARNING']}{record.name}{LOG_COLORS['ENDC']}"
        level_str = f"{color}{record.levelname}{LOG_COLORS['ENDC']}"
        msg_str = f"{color}{record.getMessage()}{LOG_COLORS['ENDC']}"

        raw_msg = f"{time_str} - {name_str} - {level_str} - {msg_str}"
        no_ansi_len = len(self.remove_ansi(raw_msg))
        padding = max(0, self.width - no_ansi_len)
        return f"{raw_msg}{' ' * padding}{elapsed_str}"

    @staticmethod
    def remove_ansi(s):
        ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        return ansi_escape.sub('', s)

def setup_logger(
    name="phoskintime",
    log_file=None,
    level=logging.DEBUG,
    log_dir=LOG_DIR,
    rotate=True,
    max_bytes=2 * 1024 * 1024,
    backup_count=5
):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    if rotate:
        file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    else:
        file_handler = logging.FileHandler(log_file)

    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler()
    stream_format = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(stream_format)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    return logger